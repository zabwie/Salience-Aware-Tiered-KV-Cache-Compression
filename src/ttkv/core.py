import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CacheConfig:
    """Configuration for tiered KV cache compression.

    Args:
        hidden_dim: Hidden dimension of the model (default: 768)
        num_heads: Number of attention heads (default: 12)
        head_dim: Dimension per attention head (default: 64)
        tier0_size: Size of uncompressed tier (recent tokens) (default: 256)
        tier1_size: Size of tier 1 cache (middle tier) (default: 2048)
        tier1_compression: Compression ratio for tier 1 (default: 4)
        tier2_compression: Compression ratio for tier 2 (default: 16)
        salience_hidden: Hidden dimension for salience scorer (default: 256)
        type_priors: Dictionary mapping token types to retention priorities
        tau_threshold: Threshold for protected tokens (default: 0.8)
    """
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    tier0_size: int = 256
    tier1_size: int = 2048
    tier1_compression: int = 4
    tier2_compression: int = 16
    salience_hidden: int = 256
    type_priors: Dict[str, float] = field(default_factory=dict)
    tau_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not self.type_priors:
            self.type_priors = {
                'NAMED_ENTITY': 1.0,
                'NUMERIC': 1.0,
                'CONTENT_WORD': 0.7,
                'FUNCTION_WORD': 0.1,
                'PUNCTUATION': 0.0,
                'OTHER': 0.5,
            }


class SalienceScorer(nn.Module):
    """Neural network for scoring token salience based on hidden states.

    Args:
        hidden_dim: Dimension of input hidden states
        salience_hidden: Dimension of hidden layer
    """

    def __init__(self, hidden_dim: int = 768, salience_hidden: int = 256) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, salience_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(salience_hidden, salience_hidden // 2),
            nn.ReLU(),
            nn.Linear(salience_hidden // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute salience scores from hidden states.

        Args:
            hidden_states: Tensor of shape [..., hidden_dim]

        Returns:
            Salience scores of shape [...]
        """
        return self.net(hidden_states).squeeze(-1)


class RetentionScheduler(nn.Module):
    """Combines salience scores with type priors for retention scheduling.

    Uses a learnable parameter to balance between salience and type-based retention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha: nn.Parameter = nn.Parameter(torch.tensor(0.5))

    def forward(self, salience_scores: torch.Tensor, type_priors: torch.Tensor) -> torch.Tensor:
        """Compute retention scores by combining salience and type priors.

        Args:
            salience_scores: Token salience scores
            type_priors: Type-based retention priorities

        Returns:
            Combined retention scores
        """
        alpha: torch.Tensor = torch.sigmoid(self.alpha)
        salience_norm: torch.Tensor = torch.sigmoid(salience_scores)
        return alpha * salience_norm + (1 - alpha) * type_priors


class TieredKVCache:
    """Three-tier KV cache with progressive compression.

    Implements a tiered compression strategy:
    - Tier 0: Recent tokens (uncompressed)
    - Tier 1: Middle tokens (4:1 compression)
    - Tier 2: Old tokens (16:1 compression)

    Args:
        config: CacheConfig instance with compression parameters
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config: CacheConfig = config
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.retention_scores: List[torch.Tensor] = []
        self.positions: List[torch.Tensor] = []
        self.total_tokens: int = 0
        self.clear()

    def clear(self) -> None:
        """Clear all cached data."""
        self.k_cache = []
        self.v_cache = []
        self.retention_scores = []
        self.positions = []
        self.total_tokens = 0

    def add(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
            positions: torch.Tensor) -> None:
        """Add new key-value pairs to the cache.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            retention: Retention scores of shape [batch, seq]
            positions: Position indices of shape [batch, seq]
        """
        self.k_cache.append(k)
        self.v_cache.append(v)
        self.retention_scores.append(retention)
        self.positions.append(positions)
        self.total_tokens += k.size(2)

    def _extract_and_stack(self, tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract and stack tensors with padding for variable lengths.

        Args:
            tensor_list: List of tensors to stack

        Returns:
            Stacked tensor with padding, or None if list is empty
        """
        if not tensor_list:
            return None

        if tensor_list[0].dim() == 1:
            # 1D position tensors: [len] -> need to pad to same length
            max_len: int = max(t.shape[0] for t in tensor_list)
            padded: List[torch.Tensor] = []
            for t in tensor_list:
                pad_len: int = max_len - t.shape[0]
                if pad_len > 0:
                    t = torch.cat([t, torch.zeros(pad_len, device=t.device, dtype=t.dtype)], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0) if padded else None
        elif tensor_list[0].dim() == 3:
            # 3D KV tensors: [heads, len, head_dim]
            max_len = max(t.shape[1] for t in tensor_list)
            padded = []
            for t in tensor_list:
                pad_len = max_len - t.shape[1]
                if pad_len > 0:
                    pad_shape: Tuple[int, int, int] = (t.shape[0], pad_len, t.shape[2])
                    t = torch.cat([t, torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0) if padded else None
        else:
            return None

    def get_compressed_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the compressed KV cache with three-tier compression.

        Returns:
            Tuple of (compressed_keys, compressed_values, compressed_positions)
        """
        if not self.k_cache:
            return None, None, None

        k_all: torch.Tensor = torch.cat(self.k_cache, dim=2)
        v_all: torch.Tensor = torch.cat(self.v_cache, dim=2)
        retention_all: torch.Tensor = torch.cat(self.retention_scores, dim=1)
        positions_all: torch.Tensor = torch.cat(self.positions, dim=1)

        batch_size: int
        num_heads: int
        total_len: int
        head_dim: int
        batch_size, num_heads, total_len, head_dim = k_all.shape
        device: torch.device = k_all.device
        tau: float = self.config.tau_threshold

        protected_mask: torch.Tensor = retention_all > tau
        unprotected_mask: torch.Tensor = ~protected_mask

        k_tiers: List[torch.Tensor] = []
        v_tiers: List[torch.Tensor] = []
        pos_tiers: List[torch.Tensor] = []

        # Tier: Protected tokens (high retention)
        if protected_mask.any():
            k_prot_list: List[torch.Tensor] = []
            v_prot_list: List[torch.Tensor] = []
            pos_prot_list: List[torch.Tensor] = []
            for b in range(batch_size):
                mask: torch.Tensor = protected_mask[b]
                if mask.any():
                    k_prot_list.append(k_all[b, :, mask, :])
                    v_prot_list.append(v_all[b, :, mask, :])
                    pos_prot_list.append(positions_all[b, mask])

            if k_prot_list:
                k_tiers.append(self._extract_and_stack(k_prot_list))
                v_tiers.append(self._extract_and_stack(v_prot_list))
                pos_tiers.append(self._extract_and_stack(pos_prot_list))

        # Tier 0: Recent tokens (uncompressed)
        recent_mask: torch.Tensor = unprotected_mask.clone()
        for b in range(batch_size):
            recent_mask[b] = unprotected_mask[b] & (torch.arange(total_len, device=device) < self.config.tier0_size)

        if recent_mask.any():
            k_rec_list: List[torch.Tensor] = []
            v_rec_list: List[torch.Tensor] = []
            pos_rec_list: List[torch.Tensor] = []
            for b in range(batch_size):
                mask = recent_mask[b]
                if mask.any():
                    k_rec_list.append(k_all[b, :, mask, :])
                    v_rec_list.append(v_all[b, :, mask, :])
                    pos_rec_list.append(positions_all[b, mask])

            if k_rec_list:
                k_tiers.append(self._extract_and_stack(k_rec_list))
                v_tiers.append(self._extract_and_stack(v_rec_list))
                pos_tiers.append(self._extract_and_stack(pos_rec_list))

        # Tier 1: Middle tokens (4:1 compression)
        middle_mask: torch.Tensor = unprotected_mask.clone()
        for b in range(batch_size):
            idx: torch.Tensor = torch.arange(total_len, device=device)
            middle_mask[b] = unprotected_mask[b] & (idx >= self.config.tier0_size) & (idx < self.config.tier1_size)

        k_mid_list: List[torch.Tensor] = []
        v_mid_list: List[torch.Tensor] = []
        pos_mid_list: List[torch.Tensor] = []

        if middle_mask.any():
            for b in range(batch_size):
                mask = middle_mask[b]
                if mask.any():
                    k_mid_list.append(k_all[b, :, mask, :])
                    v_mid_list.append(v_all[b, :, mask, :])
                    pos_mid_list.append(positions_all[b, mask])

            if k_mid_list:
                k_mid: torch.Tensor = self._extract_and_stack(k_mid_list)
                v_mid: torch.Tensor = self._extract_and_stack(v_mid_list)
                pos_mid: torch.Tensor = self._extract_and_stack(pos_mid_list)

                ret_mid_list: List[torch.Tensor] = []
                for b in range(batch_size):
                    mask = middle_mask[b]
                    if mask.any():
                        ret_mid_list.append(retention_all[b, mask])
                ret_mid: torch.Tensor = self._extract_and_stack(ret_mid_list)

                k_comp: torch.Tensor
                v_comp: torch.Tensor
                pos_comp: torch.Tensor
                k_comp, v_comp, pos_comp = self._compress(
                    k_mid, v_mid, ret_mid, pos_mid, self.config.tier1_compression
                )
                k_tiers.append(k_comp)
                v_tiers.append(v_comp)
                pos_tiers.append(pos_comp)

        # Tier 2: Old tokens (16:1 compression)
        old_mask: torch.Tensor = unprotected_mask.clone()
        for b in range(batch_size):
            old_mask[b] = unprotected_mask[b] & (torch.arange(total_len, device=device) >= self.config.tier1_size)

        k_old_list: List[torch.Tensor] = []
        v_old_list: List[torch.Tensor] = []
        pos_old_list: List[torch.Tensor] = []

        if old_mask.any():
            for b in range(batch_size):
                mask = old_mask[b]
                if mask.any():
                    k_old_list.append(k_all[b, :, mask, :])
                    v_old_list.append(v_all[b, :, mask, :])
                    pos_old_list.append(positions_all[b, mask])

        if k_old_list:
            k_old: torch.Tensor = self._extract_and_stack(k_old_list)
            v_old: torch.Tensor = self._extract_and_stack(v_old_list)
            pos_old: torch.Tensor = self._extract_and_stack(pos_old_list)

            ret_old_list: List[torch.Tensor] = []
            for b in range(batch_size):
                mask = old_mask[b]
                if mask.any():
                    ret_old_list.append(retention_all[b, mask])
            ret_old: torch.Tensor = self._extract_and_stack(ret_old_list)

            k_comp, v_comp, pos_comp = self._compress(
                k_old, v_old, ret_old, pos_old, self.config.tier2_compression
            )
            k_tiers.append(k_comp)
            v_tiers.append(v_comp)
            pos_tiers.append(pos_comp)

        if k_tiers:
            return torch.cat(k_tiers, dim=2), torch.cat(v_tiers, dim=2), torch.cat(pos_tiers, dim=1)
        else:
            return (torch.empty(batch_size, num_heads, 0, head_dim, device=device),
                    torch.empty(batch_size, num_heads, 0, head_dim, device=device),
                    torch.empty(batch_size, 0, device=device, dtype=torch.long))

    def _compress(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
                  positions: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using weighted pooling based on retention scores.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            retention: Retention scores of shape [batch, seq]
            positions: Position indices of shape [batch, seq]
            ratio: Compression ratio

        Returns:
            Tuple of (compressed_keys, compressed_values, compressed_positions)
        """
        batch_size: int
        num_heads: int
        seq_len: int
        head_dim: int
        batch_size, num_heads, seq_len, head_dim = k.shape
        device: torch.device = k.device

        if seq_len == 0:
            return k, v, positions

        compressed_len: int = (seq_len + ratio - 1) // ratio
        k_out: List[torch.Tensor] = []
        v_out: List[torch.Tensor] = []
        pos_out: List[torch.Tensor] = []

        for b in range(batch_size):
            k_batch: List[torch.Tensor] = []
            v_batch: List[torch.Tensor] = []
            pos_batch: List[torch.Tensor] = []
            for i in range(compressed_len):
                start: int = i * ratio
                end: int = min((i + 1) * ratio, seq_len)

                k_chunk: torch.Tensor = k[b, :, start:end, :]
                v_chunk: torch.Tensor = v[b, :, start:end, :]
                ret_chunk: torch.Tensor = retention[b, start:end]
                pos_chunk: torch.Tensor = positions[b, start:end]

                weights: torch.Tensor = F.softmax(ret_chunk, dim=0).unsqueeze(0).unsqueeze(-1)
                k_pooled: torch.Tensor = (k_chunk * weights).sum(dim=1)
                v_pooled: torch.Tensor = (v_chunk * weights).sum(dim=1)
                pos_pooled: torch.Tensor = (pos_chunk.float() * weights.squeeze()).sum().long()

                k_batch.append(k_pooled)
                v_batch.append(v_pooled)
                pos_batch.append(pos_pooled)

            k_out.append(torch.stack(k_batch, dim=1))
            v_out.append(torch.stack(v_batch, dim=1))
            pos_out.append(torch.stack(pos_batch))

        return torch.stack(k_out, dim=0), torch.stack(v_out, dim=0), torch.stack(pos_out, dim=0)

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics.

        Returns:
            Dictionary with total_tokens, compressed_tokens, and compression_ratio
        """
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}

        k_comp: Optional[torch.Tensor]
        k_comp, _, _ = self.get_compressed_cache()
        compressed_len: int = k_comp.size(2) if k_comp is not None else 0

        return {
            'total_tokens': self.total_tokens,
            'compressed_tokens': compressed_len,
            'compression_ratio': self.total_tokens / max(compressed_len, 1)
        }
