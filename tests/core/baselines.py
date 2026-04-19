"""Baseline KV Cache eviction methods for comparison.

This module provides implementations of baseline KV cache compression methods
for comparison with TTKV's tiered compression approach:

- H2O: Heavy-Hitter Oracle - Evicts tokens based on cumulative attention scores
- ScissorHands: Attention-based eviction with windowing

These implementations are simplified versions used for benchmarking TTKV's
performance against standard binary eviction strategies.

Note: Binary methods (keep/drop) lose information permanently, while TTKV's
tiered compression preserves information through progressive compression.
"""

import sys
sys.path.insert(0, '../src')

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np

from ttkv import CacheConfig


class H2OCache:
    """Heavy-Hitter Oracle cache eviction method.

    Keeps the top-k tokens based on cumulative attention scores.
    Tokens with lowest accumulated attention are evicted first.

    This is a binary eviction method: tokens are either kept at full
    precision or completely removed.

    Args:
        config: Cache configuration
        max_cache_size: Maximum number of tokens to retain
    """

    def __init__(self, config: CacheConfig, max_cache_size: int = 2048):
        self.config = config
        self.max_cache_size = max_cache_size
        self.clear()

    def clear(self):
        """Clear all cached data."""
        self.k_cache = []
        self.v_cache = []
        self.positions = []
        self.accumulated_attention = []
        self.total_tokens = 0

    def add(self, k: torch.Tensor, v: torch.Tensor,
            retention: Optional[torch.Tensor] = None,
            positions: Optional[torch.Tensor] = None):
        """Add KV pairs to the cache.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            retention: Optional retention scores for attention tracking
            positions: Optional position indices
        """
        self.k_cache.append(k)
        self.v_cache.append(v)

        if positions is not None:
            self.positions.append(positions)
        else:
            seq_len = k.size(2)
            start_pos = self.total_tokens
            pos = torch.arange(start_pos, start_pos + seq_len).unsqueeze(0)
            self.positions.append(pos)

        if retention is not None:
            self.accumulated_attention.append(retention.clone())
        else:
            seq_len = k.size(2)
            attn = torch.ones(1, seq_len) * 0.5
            self.accumulated_attention.append(attn)

        self.total_tokens += k.size(2)

    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cache compressed by evicting low-attention tokens.

        Returns:
            Tuple of (compressed_keys, compressed_values, compressed_positions)
        """
        if not self.k_cache:
            return (torch.empty(1, 12, 0, 64),
                    torch.empty(1, 12, 0, 64),
                    torch.empty(1, 0, dtype=torch.long))

        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        pos_all = torch.cat(self.positions, dim=1)
        attn_all = torch.cat(self.accumulated_attention, dim=1)

        batch_size, num_heads, total_len, head_dim = k_all.shape

        if total_len <= self.max_cache_size:
            return k_all, v_all, pos_all

        # Keep top-k by attention
        _, indices = torch.topk(attn_all[0], self.max_cache_size, dim=0)
        indices = indices.sort()[0]

        k_comp = k_all[:, :, indices, :]
        v_comp = v_all[:, :, indices, :]
        pos_comp = pos_all[:, indices]

        return k_comp, v_comp, pos_comp

    def get_stats(self) -> Dict:
        """Get compression statistics.

        Returns:
            Dictionary with total_tokens, compressed_tokens, compression_ratio
        """
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}

        total_len = sum(k.size(2) for k in self.k_cache)
        compressed_len = min(total_len, self.max_cache_size)

        return {
            'total_tokens': total_len,
            'compressed_tokens': compressed_len,
            'compression_ratio': total_len / max(compressed_len, 1)
        }


class ScissorHandsCache:
    """ScissorHands cache eviction method.

    Evicts tokens based on a combination of recent attention and position.

    Args:
        config: Cache configuration
        max_cache_size: Maximum number of tokens to retain
        recency_weight: Weight for recency vs attention (0-1)
    """

    def __init__(self, config: CacheConfig, max_cache_size: int = 2048,
                 recency_weight: float = 0.3):
        self.config = config
        self.max_cache_size = max_cache_size
        self.recency_weight = recency_weight
        self.clear()

    def clear(self):
        """Clear all cached data."""
        self.k_cache = []
        self.v_cache = []
        self.positions = []
        self.accumulated_attention = []
        self.total_tokens = 0

    def add(self, k: torch.Tensor, v: torch.Tensor,
            retention: Optional[torch.Tensor] = None,
            positions: Optional[torch.Tensor] = None):
        """Add KV pairs to the cache.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            retention: Optional retention scores for attention tracking
            positions: Optional position indices
        """
        self.k_cache.append(k)
        self.v_cache.append(v)

        if positions is not None:
            self.positions.append(positions)
        else:
            seq_len = k.size(2)
            start_pos = self.total_tokens
            pos = torch.arange(start_pos, start_pos + seq_len).unsqueeze(0)
            self.positions.append(pos)

        if retention is not None:
            self.accumulated_attention.append(retention.clone())
        else:
            seq_len = k.size(2)
            attn = torch.ones(1, seq_len) * 0.5
            self.accumulated_attention.append(attn)

        self.total_tokens += k.size(2)

    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cache compressed by evicting low-importance tokens.

        Returns:
            Tuple of (compressed_keys, compressed_values, compressed_positions)
        """
        if not self.k_cache:
            return (torch.empty(1, 12, 0, 64),
                    torch.empty(1, 12, 0, 64),
                    torch.empty(1, 0, dtype=torch.long))

        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        pos_all = torch.cat(self.positions, dim=1)
        attn_all = torch.cat(self.accumulated_attention, dim=1)

        batch_size, num_heads, total_len, head_dim = k_all.shape

        if total_len <= self.max_cache_size:
            return k_all, v_all, pos_all

        # Combined score: attention + recency
        normalized_pos = pos_all.float() / total_len
        recency_score = normalized_pos
        combined_score = (1 - self.recency_weight) * attn_all + \
                         self.recency_weight * recency_score

        _, indices = torch.topk(combined_score[0], self.max_cache_size, dim=0)
        indices = indices.sort()[0]

        k_comp = k_all[:, :, indices, :]
        v_comp = v_all[:, :, indices, :]
        pos_comp = pos_all[:, indices]

        return k_comp, v_comp, pos_comp

    def get_stats(self) -> Dict:
        """Get compression statistics.

        Returns:
            Dictionary with total_tokens, compressed_tokens, compression_ratio
        """
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}

        total_len = sum(k.size(2) for k in self.k_cache)
        compressed_len = min(total_len, self.max_cache_size)

        return {
            'total_tokens': total_len,
            'compressed_tokens': compressed_len,
            'compression_ratio': total_len / max(compressed_len, 1)
        }


def run_comparison():
    """Run comparison between baseline methods and TTKV.

    Generates a comparison table showing compression ratios and
    the key insight that tiered compression preserves information
    while binary methods don't.
    """
    print("=" * 80)
    print("KV CACHE COMPARISON: Tiered vs Binary Eviction")
    print("=" * 80)

    seq_len = 4096
    tau = 0.9

    config = CacheConfig(
        tier0_size=256,
        tier1_size=2048,
        tier1_compression=4,
        tier2_compression=16,
        tau_threshold=tau,
    )

    batch_size = 1
    num_heads = 12
    head_dim = 64

    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0)

    from ttkv import compute_type_prior_retention
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    retention = compute_type_prior_retention(token_ids)

    # Baseline
    baseline_tokens = seq_len

    # H2O
    h2o = H2OCache(config, max_cache_size=2048)
    h2o.add(k, v, retention, positions)
    h2o_stats = h2o.get_stats()

    # ScissorHands
    scissor = ScissorHandsCache(config, max_cache_size=2048)
    scissor.add(k, v, retention, positions)
    scissor_stats = scissor.get_stats()

    # Tiered
    from ttkv import TieredKVCache
    tiered = TieredKVCache(config)
    tiered.add(k, v, retention, positions)
    _ = tiered.get_compressed_cache()
    tiered_stats = tiered.get_stats()

    print(f"\nTest: {seq_len} tokens, τ={tau}")
    print()
    print(f"{'Method':<30} | {'Total':>6} | {'Kept':>6} | {'Ratio':>6} | Type")
    print("-" * 70)
    print(f"{'Baseline (no compression)':<30} | {baseline_tokens:>6} | {baseline_tokens:>6} | {'1.00x':>6} | none")
    print(f"{'H2O':<30} | {h2o_stats['total_tokens']:>6} | {h2o_stats['compressed_tokens']:>6} | {h2o_stats['compression_ratio']:>5.2f}x | binary")
    print(f"{'ScissorHands':<30} | {scissor_stats['total_tokens']:>6} | {scissor_stats['compressed_tokens']:>6} | {scissor_stats['compression_ratio']:>5.2f}x | binary")
    print(f"{'Tiered (Ours)':<30} | {tiered_stats['total_tokens']:>6} | {tiered_stats['compressed_tokens']:>6} | {tiered_stats['compression_ratio']:>5.2f}x | tiered")
    print()
    print("Key Insight:")
    print("- Binary methods (H2O, ScissorHands): Tokens kept OR dropped permanently")
    print("- Tiered method: Tokens compressed progressively (Tier 0→1→2)")
    print("- Tiered preserves information longer via structural floor")
    print()


if __name__ == "__main__":
    run_comparison()
