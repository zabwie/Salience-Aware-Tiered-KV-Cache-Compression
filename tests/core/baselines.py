"""
Baseline KV Cache Eviction Methods for Comparison

H2O: Heavy Hitter Oracle - keeps tokens with highest accumulated attention
ScissorHands: Keeps tokens with highest attention in recent window

Both use BINARY eviction (keep/drop), unlike tiered compression.
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
    """
    H2O: Heavy Hitter Oracle
    Keeps top-k tokens by accumulated attention score.
    Binary eviction - tokens are either kept or permanently dropped.
    """
    
    def __init__(self, config: CacheConfig, max_cache_size: int = 2048):
        self.config = config
        self.max_cache_size = max_cache_size
        self.clear()
    
    def clear(self):
        self.k_cache = []
        self.v_cache = []
        self.positions = []
        self.accumulated_attention = []  # Track attention per position
        self.total_tokens = 0
    
    def add(self, k: torch.Tensor, v: torch.Tensor, 
            retention: Optional[torch.Tensor] = None,
            positions: Optional[torch.Tensor] = None):
        """Add to cache and update accumulated attention scores."""
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
    
    def add_with_attention_pattern(self, k: torch.Tensor, v: torch.Tensor,
                                   attention_to_needle: torch.Tensor,
                                   positions: Optional[torch.Tensor] = None):
        """Add with explicit attention pattern (for testing)."""
        self.k_cache.append(k)
        self.v_cache.append(v)
        
        if positions is not None:
            self.positions.append(positions)
        else:
            seq_len = k.size(2)
            start_pos = self.total_tokens
            pos = torch.arange(start_pos, start_pos + seq_len).unsqueeze(0)
            self.positions.append(pos)
        
        self.accumulated_attention.append(attention_to_needle)
        self.total_tokens += k.size(2)
    
    def _update_accumulated_attention(self, new_attention: torch.Tensor):
        """Update accumulated attention with EMA decay."""
        ema_decay = 0.95
        for i, attn in enumerate(self.accumulated_attention):
            self.accumulated_attention[i] = ema_decay * attn + (1 - ema_decay) * new_attention
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current cache (before eviction)."""
        if not self.k_cache:
            return None, None, None
        
        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        pos_all = torch.cat(self.positions, dim=1)
        
        return k_all, v_all, pos_all
    
    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evict low-attention tokens, keep top-k by accumulated attention.
        Binary eviction - tokens are kept or dropped, no compression.
        """
        if not self.k_cache:
            return None, None, None
        
        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        pos_all = torch.cat(self.positions, dim=1)
        attn_all = torch.cat(self.accumulated_attention, dim=1)
        
        batch_size, num_heads, total_len, head_dim = k_all.shape
        device = k_all.device
        
        # If under budget, return all
        if total_len <= self.max_cache_size:
            return k_all, v_all, pos_all
        
        # Binary eviction: keep top-k by attention score
        _, top_indices = torch.topk(attn_all[0], self.max_cache_size, dim=0)
        top_indices = top_indices.sort()[0]  # Sort to maintain order
        
        k_kept = k_all[:, :, top_indices, :]
        v_kept = v_all[:, :, top_indices, :]
        pos_kept = pos_all[:, top_indices]
        
        return k_kept, v_kept, pos_kept
    
    def get_stats(self) -> Dict:
        """Return compression stats."""
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}
        
        k_comp, _, _ = self.get_compressed_cache()
        compressed_len = k_comp.size(2) if k_comp is not None else 0
        
        return {
            'total_tokens': self.total_tokens,
            'compressed_tokens': compressed_len,
            'compression_ratio': self.total_tokens / max(compressed_len, 1),
            'method': 'H2O (binary eviction)'
        }


class ScissorHandsCache:
    """
    ScissorHands: Attention-based eviction with recent window focus.
    Keeps tokens that receive highest attention from recent query positions.
    Binary eviction - tokens outside the attention window are dropped.
    """
    
    def __init__(self, config: CacheConfig, max_cache_size: int = 2048, 
                 attention_window: int = 256):
        self.config = config
        self.max_cache_size = max_cache_size
        self.attention_window = attention_window
        self.clear()
    
    def clear(self):
        self.k_cache = []
        self.v_cache = []
        self.positions = []
        self.total_tokens = 0
    
    def add(self, k: torch.Tensor, v: torch.Tensor,
            retention: Optional[torch.Tensor] = None,
            positions: Optional[torch.Tensor] = None):
        """Add to cache."""
        self.k_cache.append(k)
        self.v_cache.append(v)
        
        if positions is not None:
            self.positions.append(positions)
        else:
            seq_len = k.size(2)
            start_pos = self.total_tokens
            pos = torch.arange(start_pos, start_pos + seq_len).unsqueeze(0)
            self.positions.append(pos)
        
        self.total_tokens += k.size(2)
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current cache."""
        if not self.k_cache:
            return None, None, None
        
        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        pos_all = torch.cat(self.positions, dim=1)
        
        return k_all, v_all, pos_all
    
    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.k_cache:
            return None, None, None
        
        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        pos_all = torch.cat(self.positions, dim=1)
        
        batch_size, num_heads, total_len, head_dim = k_all.shape
        device = k_all.device
        
        if total_len <= self.max_cache_size:
            return k_all, v_all, pos_all
        
        # ScissorHands: only use queries from recent window
        recent_start = max(0, total_len - self.attention_window)
        recent_positions = torch.arange(recent_start, total_len, device=device)
        
        # CRITICAL FIX: ScissorHands computes attention FROM recent queries TO ALL positions
        # But tokens far away get near-zero attention due to softmax normalization
        # over the full sequence. The slow-burn problem is real.
        attention_scores = torch.zeros(batch_size, total_len, device=device)
        
        for q_pos in recent_positions:
            query = k_all[0, 0, q_pos, :]
            # Attention over FULL sequence (not just window)
            scores = torch.matmul(query.unsqueeze(0), k_all[0, 0, :, :].T) / (head_dim ** 0.5)
            scores = F.softmax(scores, dim=-1)
            attention_scores += scores.squeeze(0)
        
        attention_scores = attention_scores / len(recent_positions)
        
        # ATTENTION WINDOW LIMITATION: The slow-burn problem occurs because
        # softmax over long contexts dilutes attention to distant tokens.
        # A token at position 0 gets attention_score ≈ 1/15000 ≈ 0.000067
        # which is negligible compared to nearby tokens.
        
        # SLOW-BURN PROBLEM: Needle at position 0 gets attention_score = 0
        # because it's outside the recent window. It will be evicted.
        # Note: For the slow-burn test to show the problem, the needle must have
        # distinctive enough KV vectors that attention would attend to it.
        # But with random filler tokens, it might still get some attention.
        
        _, top_indices = torch.topk(attention_scores[0], self.max_cache_size, dim=0)
        top_indices = top_indices.sort()[0]
        
        k_kept = k_all[:, :, top_indices, :]
        v_kept = v_all[:, :, top_indices, :]
        pos_kept = pos_all[:, top_indices]
        
        return k_kept, v_kept, pos_kept
    
    def get_stats(self) -> Dict:
        """Return compression stats."""
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}
        
        k_comp, _, _ = self.get_compressed_cache()
        compressed_len = k_comp.size(2) if k_comp is not None else 0
        
        return {
            'total_tokens': self.total_tokens,
            'compressed_tokens': compressed_len,
            'compression_ratio': self.total_tokens / max(compressed_len, 1),
            'method': 'ScissorHands (binary eviction)'
        }


def run_comparison(seq_len: int = 4096, tau: float = 0.9):
    """
    Compare all three methods on same sequence.
    """
    print("=" * 80)
    print("KV CACHE COMPARISON: Tiered vs Binary Eviction")
    print("=" * 80)
    print(f"\nTest: {seq_len} tokens, τ={tau}")
    print()
    
    from ttkv import TieredKVCache
    from ttkv import compute_type_prior_retention
    
    # Generate test data
    batch_size = 1
    num_heads = 12
    head_dim = 64
    
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(seq_len).unsqueeze(0)
    token_ids = torch.randint(0, 50000, (batch_size, seq_len))
    retention = compute_type_prior_retention(token_ids)
    
    # Test each method
    results = []
    
    # 1. Baseline (no compression)
    config_baseline = CacheConfig(tau_threshold=0.0)
    cache_baseline = TieredKVCache(config_baseline)
    cache_baseline.add(k, v, retention, positions)
    stats_baseline = cache_baseline.get_stats()
    results.append({
        'method': 'Baseline (no compression)',
        'tokens': stats_baseline['total_tokens'],
        'kept': stats_baseline['compressed_tokens'],
        'ratio': stats_baseline['compression_ratio'],
        'type': 'none'
    })
    
    # 2. H2O (binary eviction)
    h2o_cache = H2OCache(config_baseline, max_cache_size=2048)
    h2o_cache.add(k, v, retention, positions)
    stats_h2o = h2o_cache.get_stats()
    results.append({
        'method': 'H2O',
        'tokens': stats_h2o['total_tokens'],
        'kept': stats_h2o['compressed_tokens'],
        'ratio': stats_h2o['compression_ratio'],
        'type': 'binary'
    })
    
    # 3. ScissorHands (binary eviction)
    sh_cache = ScissorHandsCache(config_baseline, max_cache_size=2048)
    sh_cache.add(k, v, retention, positions)
    stats_sh = sh_cache.get_stats()
    results.append({
        'method': 'ScissorHands',
        'tokens': stats_sh['total_tokens'],
        'kept': stats_sh['compressed_tokens'],
        'ratio': stats_sh['compression_ratio'],
        'type': 'binary'
    })
    
    # 4. Tiered (our method)
    config_tiered = CacheConfig(tau_threshold=tau)
    tiered_cache = TieredKVCache(config_tiered)
    tiered_cache.add(k, v, retention, positions)
    stats_tiered = tiered_cache.get_stats()
    results.append({
        'method': 'Tiered (Ours)',
        'tokens': stats_tiered['total_tokens'],
        'kept': stats_tiered['compressed_tokens'],
        'ratio': stats_tiered['compression_ratio'],
        'type': 'tiered'
    })
    
    # Print comparison table
    print(f"{'Method':<20} | {'Total':>8} | {'Kept':>8} | {'Ratio':>8} | {'Type':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<20} | {r['tokens']:>8} | {r['kept']:>8} | {r['ratio']:>7.2f}x | {r['type']:<15}")
    
    print("\n" + "=" * 80)
    print("Key Insight:")
    print("  - Binary methods (H2O, ScissorHands): Tokens kept OR dropped permanently")
    print("  - Tiered method: Tokens compressed progressively (Tier 0→1→2)")
    print("  - Tiered preserves information longer via structural floor")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_comparison(seq_len=4096)
