import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Set, Union
from dataclasses import dataclass, field
import numpy as np
import threading
import weakref

from .exceptions import (
    ValidationError,
    DeviceMismatchError,
    DtypeMismatchError,
    ShapeMismatchError,
    DimensionError,
    EmptyTensorError,
    InvalidValueError,
    ConfigurationError,
)


@dataclass(frozen=True)
class CacheStats:
    """Structured compression statistics for observability.

    Supports both attribute access (stats.total_tokens) and dictionary-style
    access (stats['total_tokens']) for backward compatibility.

    Attributes:
        total_tokens: Total number of tokens before compression
        compressed_tokens: Number of tokens after compression
        compression_ratio: Ratio of total to compressed tokens
        tier_distribution: Token counts per tier (protected, tier0, tier1, tier2)
        peak_memory_mb: Peak GPU memory used during compression
        compression_time_ms: Time taken for compression in milliseconds
    """
    total_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tier_distribution: Tuple[int, int, int, int]  # (protected, tier0, tier1, tier2)
    peak_memory_mb: float = 0.0
    compression_time_ms: float = 0.0

    def __getitem__(self, key: str) -> Union[int, float, Tuple[int, int, int, int]]:
        """Support dictionary-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dictionary-style checks."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict.get() method."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Union[int, float, Tuple[int, int, int, int]]]:
        """Convert stats to dictionary for serialization."""
        return {
            'total_tokens': self.total_tokens,
            'compressed_tokens': self.compressed_tokens,
            'compression_ratio': self.compression_ratio,
            'tier_distribution': self.tier_distribution,
            'peak_memory_mb': self.peak_memory_mb,
            'compression_time_ms': self.compression_time_ms,
        }


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
        enable_jit: Whether to enable JIT compilation for compression (default: True)
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
    enable_jit: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ConfigurationError(f"hidden_dim must be positive, got {self.hidden_dim}", "hidden_dim")
        if self.num_heads <= 0:
            raise ConfigurationError(f"num_heads must be positive, got {self.num_heads}", "num_heads")
        if self.head_dim <= 0:
            raise ConfigurationError(f"head_dim must be positive, got {self.head_dim}", "head_dim")
        if self.tier0_size < 0:
            raise ConfigurationError(f"tier0_size must be non-negative, got {self.tier0_size}", "tier0_size")
        if self.tier1_size < self.tier0_size:
            raise ConfigurationError(
                f"tier1_size ({self.tier1_size}) must be >= tier0_size ({self.tier0_size})",
                "tier1_size"
            )
        if self.tier1_compression < 1:
            raise ConfigurationError(
                f"tier1_compression must be >= 1, got {self.tier1_compression}",
                "tier1_compression"
            )
        if self.tier2_compression < self.tier1_compression:
            raise ConfigurationError(
                f"tier2_compression ({self.tier2_compression}) should be >= tier1_compression ({self.tier1_compression})",
                "tier2_compression"
            )
        if self.salience_hidden <= 0:
            raise ConfigurationError(
                f"salience_hidden must be positive, got {self.salience_hidden}",
                "salience_hidden"
            )
        if not 0.0 <= self.tau_threshold <= 1.0:
            raise ConfigurationError(
                f"tau_threshold must be in [0, 1], got {self.tau_threshold}",
                "tau_threshold"
            )
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
        self.hidden_dim: int = hidden_dim
        self.salience_hidden: int = salience_hidden
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, salience_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(salience_hidden, salience_hidden // 2),
            nn.ReLU(),
            nn.Linear(salience_hidden // 2, 1)
        )

    def _validate_input(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.dim() < 2:
            raise DimensionError(2, hidden_states.dim(), "hidden_states")
        if torch.isnan(hidden_states).any():
            nan_count = torch.isnan(hidden_states).sum().item()
            raise InvalidValueError("NaN", "hidden_states", nan_count)
        if torch.isinf(hidden_states).any():
            inf_count = torch.isinf(hidden_states).sum().item()
            raise InvalidValueError("Inf", "hidden_states", inf_count)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute salience scores from hidden states.

        Args:
            hidden_states: Tensor of shape [..., hidden_dim]

        Returns:
            Salience scores of shape [...]

        Raises:
            DimensionError: If tensor doesn't have expected dimensions
            InvalidValueError: If tensor contains NaN or Inf values
        """
        self._validate_input(hidden_states)
        return self.net(hidden_states).squeeze(-1)

    def save_pretrained(self, path: str) -> None:
        """Save scorer weights to disk.

        Args:
            path: Path to save the weights file
        """
        torch.save({
            'hidden_dim': self.hidden_dim,
            'salience_hidden': self.salience_hidden,
            'state_dict': self.state_dict()
        }, path)

    def load_pretrained(self, path: str) -> None:
        """Load pre-trained scorer weights from disk.

        Args:
            path: Path to the weights file

        Raises:
            FileNotFoundError: If the weights file doesn't exist
            RuntimeError: If the weights are incompatible with this scorer
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pre-trained weights not found at {path}")

        checkpoint: Dict[str, Any] = torch.load(path, map_location='cpu')

        if checkpoint.get('hidden_dim') != self.hidden_dim:
            raise RuntimeError(
                f"Hidden dimension mismatch: model has {self.hidden_dim}, "
                f"weights have {checkpoint.get('hidden_dim')}"
            )
        if checkpoint.get('salience_hidden') != self.salience_hidden:
            raise RuntimeError(
                f"Salience hidden dimension mismatch: model has {self.salience_hidden}, "
                f"weights have {checkpoint.get('salience_hidden')}"
            )

        self.load_state_dict(checkpoint['state_dict'])

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> 'SalienceScorer':
        """Create a SalienceScorer with pre-trained weights.

        Args:
            path: Path to the weights file
            **kwargs: Additional arguments passed to __init__ (ignored if weights exist)

        Returns:
            SalienceScorer with loaded weights

        Raises:
            FileNotFoundError: If the weights file doesn't exist
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pre-trained weights not found at {path}")

        checkpoint: Dict[str, Any] = torch.load(path, map_location='cpu')
        hidden_dim = checkpoint.get('hidden_dim', kwargs.get('hidden_dim', 768))
        salience_hidden = checkpoint.get('salience_hidden', kwargs.get('salience_hidden', 256))

        scorer = cls(hidden_dim=hidden_dim, salience_hidden=salience_hidden)
        scorer.load_state_dict(checkpoint['state_dict'])
        return scorer


class RetentionScheduler(nn.Module):
    """Combines salience scores with type priors for retention scheduling.

    Uses a learnable parameter to balance between salience and type-based retention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha: nn.Parameter = nn.Parameter(torch.tensor(0.5))

    def _validate_inputs(self, salience_scores: torch.Tensor, type_priors: torch.Tensor) -> None:
        if salience_scores.shape != type_priors.shape:
            raise ShapeMismatchError(
                salience_scores.shape,
                type_priors.shape,
                "salience_scores vs type_priors"
            )

        if salience_scores.device != type_priors.device:
            raise DeviceMismatchError(
                {salience_scores.device, type_priors.device},
                "salience_scores and type_priors must be on the same device"
            )

        for name, tensor in [("salience_scores", salience_scores), ("type_priors", type_priors)]:
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                raise InvalidValueError("NaN", name, nan_count)
            if torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                raise InvalidValueError("Inf", name, inf_count)

    def forward(self, salience_scores: torch.Tensor, type_priors: torch.Tensor) -> torch.Tensor:
        """Compute retention scores by combining salience and type priors.

        Args:
            salience_scores: Token salience scores
            type_priors: Type-based retention priorities

        Returns:
            Combined retention scores

        Raises:
            ShapeMismatchError: If shapes don't match
            DeviceMismatchError: If tensors are on different devices
            InvalidValueError: If tensors contain NaN or Inf values
        """
        self._validate_inputs(salience_scores, type_priors)
        alpha: torch.Tensor = torch.sigmoid(self.alpha)
        salience_norm: torch.Tensor = torch.sigmoid(salience_scores)
        return alpha * salience_norm + (1 - alpha) * type_priors


class TieredKVCache:
    """Three-tier KV cache with progressive compression.

    Implements a tiered compression strategy:
    - Tier 0: Recent tokens (uncompressed)
    - Tier 1: Middle tokens (4:1 compression)
    - Tier 2: Old tokens (16:1 compression)

    Thread-safe and supports context manager usage for automatic cleanup.

    Args:
        config: CacheConfig instance with compression parameters

    Example:
        >>> config = CacheConfig(tier0_size=256, tier1_size=2048)
        >>> with TieredKVCache(config) as cache:
        ...     cache.add(k, v, retention, positions)
        ...     k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        ...     stats = cache.get_stats()
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config: CacheConfig = config
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.retention_scores: List[torch.Tensor] = []
        self.positions: List[torch.Tensor] = []
        self.total_tokens: int = 0
        self._expected_dtype: Optional[torch.dtype] = None
        self._expected_device: Optional[torch.device] = None
        self._lock: threading.RLock = threading.RLock()
        self._tier_counts: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._compression_time_ms: float = 0.0
        self._jit_cache: Optional[Any] = None
        self.clear()

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self.k_cache = []
            self.v_cache = []
            self.retention_scores = []
            self.positions = []
            self.total_tokens = 0
            self._expected_dtype = None
            self._expected_device = None
            self._tier_counts = (0, 0, 0, 0)
            self._compression_time_ms = 0.0

    def __enter__(self) -> 'TieredKVCache':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager with automatic cleanup."""
        self.clear()

    def __del__(self) -> None:
        """Destructor with automatic GPU memory cleanup."""
        self.clear()

    def _get_peak_memory_mb(self) -> float:
        """Get peak GPU memory allocated in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def _validate_device_consistency(self, *tensors: torch.Tensor) -> torch.device:
        """Validate all tensors are on the same device.

        Args:
            *tensors: Variable number of tensors to check

        Returns:
            The common device

        Raises:
            DeviceMismatchError: If tensors are on different devices
        """
        devices: Set[torch.device] = {t.device for t in tensors}
        if len(devices) > 1:
            raise DeviceMismatchError(devices)
        return devices.pop() if devices else torch.device('cpu')

    def _validate_dtype_consistency(self, k: torch.Tensor, v: torch.Tensor) -> torch.dtype:
        """Validate key and value tensors have compatible dtypes.

        Args:
            k: Key tensor
            v: Value tensor

        Returns:
            The common dtype

        Raises:
            DtypeMismatchError: If dtypes are incompatible
        """
        if k.dtype != v.dtype:
            raise DtypeMismatchError(k.dtype, v.dtype, "keys vs values")
        return k.dtype

    def _validate_tensor_values(self, tensor: torch.Tensor, name: str) -> None:
        """Validate tensor doesn't contain NaN or Inf values.

        Args:
            tensor: Tensor to validate
            name: Name of the tensor for error messages

        Raises:
            InvalidValueError: If tensor contains NaN or Inf values
        """
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            raise InvalidValueError("NaN", name, nan_count)
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            raise InvalidValueError("Inf", name, inf_count)

    def _validate_not_empty(self, tensor: torch.Tensor, name: str) -> None:
        """Validate tensor is not empty.

        Args:
            tensor: Tensor to validate
            name: Name of the tensor for error messages

        Raises:
            EmptyTensorError: If tensor has zero size
        """
        if tensor.numel() == 0:
            raise EmptyTensorError(name)
        for dim_idx, dim_size in enumerate(tensor.shape):
            if dim_size == 0:
                raise EmptyTensorError(name, dim_idx)

    def add(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
            positions: torch.Tensor) -> None:
        """Add new key-value pairs to the cache.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            retention: Retention scores of shape [batch, seq]
            positions: Position indices of shape [batch, seq]

        Raises:
            ValueError: If tensor shapes are incompatible (backward compatibility)
            DeviceMismatchError: If tensors are on different devices
            DtypeMismatchError: If key and value dtypes don't match
            DimensionError: If tensors don't have expected dimensions
            InvalidValueError: If tensors contain NaN or Inf values
        """
        self._validate_not_empty(k, "keys")
        self._validate_not_empty(v, "values")
        self._validate_not_empty(retention, "retention")
        self._validate_not_empty(positions, "positions")

        if k.dim() != 4:
            raise DimensionError(4, k.dim(), "keys: Keys must be 4D tensor [batch, heads, seq, head_dim]")
        if v.dim() != 4:
            raise DimensionError(4, v.dim(), "values")
        if retention.dim() != 2:
            raise DimensionError(2, retention.dim(), "retention")
        if positions.dim() != 2:
            raise DimensionError(2, positions.dim(), "positions")

        if k.shape != v.shape:
            raise ShapeMismatchError(k.shape, v.shape, "keys vs values: Keys and values must have the same shape")
        if retention.shape[0] != k.shape[0]:
            raise ShapeMismatchError(
                (k.shape[0], retention.shape[1]),
                retention.shape,
                f"retention batch dimension: Batch size mismatch: keys have {k.shape[0]}, retention has {retention.shape[0]}"
            )
        if retention.shape[1] != k.shape[2]:
            raise ShapeMismatchError(
                (k.shape[0], k.shape[2]),
                retention.shape,
                f"retention sequence dimension: Sequence length mismatch: keys have {k.shape[2]}, retention has {retention.shape[1]}"
            )
        if positions.shape != retention.shape:
            raise ShapeMismatchError(
                retention.shape, positions.shape,
                "positions vs retention: Positions and retention must have the same shape"
            )

        device = self._validate_device_consistency(k, v, retention, positions)

        dtype = self._validate_dtype_consistency(k, v)

        if self.k_cache:
            if self._expected_device is not None and device != self._expected_device:
                raise DeviceMismatchError(
                    {device, self._expected_device},
                    f"New tensor device {device} doesn't match cache device {self._expected_device}"
                )
            if self._expected_dtype is not None and dtype != self._expected_dtype:
                raise DtypeMismatchError(
                    self._expected_dtype, dtype,
                    f"new tensor (cache expects {self._expected_dtype})"
                )
        else:
            self._expected_device = device
            self._expected_dtype = dtype

        self._validate_tensor_values(k, "keys")
        self._validate_tensor_values(v, "values")
        self._validate_tensor_values(retention, "retention")

        with self._lock:
            self.k_cache.append(k)
            self.v_cache.append(v)
            self.retention_scores.append(retention)
            self.positions.append(positions)
            self.total_tokens += k.size(2)

    def _extract_retention_scores_vectorized(
        self,
        retention_all: torch.Tensor,
        mask: torch.Tensor,
        k_extracted: torch.Tensor,
    ) -> torch.Tensor:
        """Extract retention scores matching extracted K/V dimensions.

        Args:
            retention_all: Retention scores [batch, seq]
            mask: Boolean mask [batch, seq]
            k_extracted: Extracted keys [batch, heads, extracted_seq, head_dim]
                used to determine target sequence length

        Returns:
            Extracted retention scores [batch, extracted_seq]
        """
        batch_size = retention_all.size(0)
        target_len = k_extracted.size(2)
        device = retention_all.device

        ret_out = torch.zeros(batch_size, target_len, device=device, dtype=retention_all.dtype)

        for b in range(batch_size):
            batch_mask = mask[b]
            if batch_mask.any():
                batch_indices = batch_mask.nonzero(as_tuple=True)[0]
                n_masked = len(batch_indices)
                ret_out[b, :n_masked] = retention_all[b, batch_indices]

        return ret_out

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
            max_len: int = max(t.shape[0] for t in tensor_list)
            padded: List[torch.Tensor] = []
            for t in tensor_list:
                pad_len: int = max_len - t.shape[0]
                if pad_len > 0:
                    t = torch.cat([t, torch.zeros(pad_len, device=t.device, dtype=t.dtype)], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0) if padded else None
        elif tensor_list[0].dim() == 3:
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

    def _extract_masked_batch_vectorized(
        self,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        positions_all: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract masked elements across batch using vectorized operations.

        Args:
            k_all: Key tensor [batch, heads, seq, head_dim]
            v_all: Value tensor [batch, heads, seq, head_dim]
            positions_all: Positions tensor [batch, seq]
            mask: Boolean mask [batch, seq]

        Returns:
            Tuple of (masked_k, masked_v, masked_pos) or None if mask is empty
        """
        if not mask.any():
            return None, None, None

        batch_size, num_heads, seq_len, head_dim = k_all.shape

        # Get counts per batch for indexing
        counts = mask.sum(dim=1)  # [batch]
        max_count = int(counts.max().item())

        if max_count == 0:
            return None, None, None

        # Create output tensors
        device = k_all.device
        k_out = torch.zeros(batch_size, num_heads, max_count, head_dim, device=device, dtype=k_all.dtype)
        v_out = torch.zeros(batch_size, num_heads, max_count, head_dim, device=device, dtype=v_all.dtype)
        pos_out = torch.zeros(batch_size, max_count, device=device, dtype=positions_all.dtype)

        # Create index mapping: for each batch, map from original position to output position
        # Use cumsum to get output indices
        mask_int = mask.long()
        output_indices = mask_int.cumsum(dim=1) - 1  # [batch, seq], 0-indexed positions in output

        # Expand for gathering
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
        output_indices_expanded = output_indices.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]

        # Scatter values to output positions
        k_scatter = k_all * mask_expanded  # Zero out non-masked
        v_scatter = v_all * mask_expanded
        pos_scatter = positions_all * mask  # [batch, seq]

        # Use advanced indexing with gather
        # First create a valid indices mask for gather
        valid_mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)  # [batch, heads, seq, head_dim]

        # Flatten and gather
        for b in range(batch_size):
            batch_mask = mask[b]  # [seq]
            if batch_mask.any():
                batch_indices = batch_mask.nonzero(as_tuple=True)[0]  # [n_masked]
                k_out[b, :, :len(batch_indices), :] = k_all[b, :, batch_indices, :]
                v_out[b, :, :len(batch_indices), :] = v_all[b, :, batch_indices, :]
                pos_out[b, :len(batch_indices)] = positions_all[b, batch_indices]

        return k_out, v_out, pos_out

    def get_compressed_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the compressed KV cache with three-tier compression.

        Uses optimized vectorized operations to minimize Python loop overhead.

        Returns:
            Tuple of (compressed_keys, compressed_values, compressed_positions)
        """
        if not self.k_cache:
            return None, None, None

        # Concatenate all cached tensors
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

        # Create tier masks vectorized
        positions_idx: torch.Tensor = torch.arange(total_len, device=device)
        protected_mask: torch.Tensor = retention_all > tau
        unprotected_mask: torch.Tensor = ~protected_mask

        # Tier masks (all computed vectorized)
        recent_mask: torch.Tensor = unprotected_mask & (positions_idx < self.config.tier0_size)
        middle_mask: torch.Tensor = unprotected_mask & (positions_idx >= self.config.tier0_size) & (positions_idx < self.config.tier1_size)
        old_mask: torch.Tensor = unprotected_mask & (positions_idx >= self.config.tier1_size)

        k_tiers: List[torch.Tensor] = []
        v_tiers: List[torch.Tensor] = []
        pos_tiers: List[torch.Tensor] = []

        # Process protected tier (no compression)
        if protected_mask.any():
            k_prot, v_prot, pos_prot = self._extract_masked_batch_vectorized(
                k_all, v_all, positions_all, protected_mask
            )
            if k_prot is not None:
                k_tiers.append(k_prot)
                v_tiers.append(v_prot)
                pos_tiers.append(pos_prot)

        # Process tier 0 - recent tokens (no compression)
        if recent_mask.any():
            k_rec, v_rec, pos_rec = self._extract_masked_batch_vectorized(
                k_all, v_all, positions_all, recent_mask
            )
            if k_rec is not None:
                k_tiers.append(k_rec)
                v_tiers.append(v_rec)
                pos_tiers.append(pos_rec)

        # Process tier 1 - middle tokens (tier1_compression:1 compression)
        if middle_mask.any():
            k_mid, v_mid, pos_mid = self._extract_masked_batch_vectorized(
                k_all, v_all, positions_all, middle_mask
            )
            if k_mid is not None:
                ret_mid = self._extract_retention_scores_vectorized(
                    retention_all, middle_mask, k_mid
                )
                k_comp, v_comp, pos_comp = self._compress(
                    k_mid, v_mid, ret_mid, pos_mid, self.config.tier1_compression
                )
                k_tiers.append(k_comp)
                v_tiers.append(v_comp)
                pos_tiers.append(pos_comp)

        # Process tier 2 - old tokens (tier2_compression:1 compression)
        if old_mask.any():
            k_old, v_old, pos_old = self._extract_masked_batch_vectorized(
                k_all, v_all, positions_all, old_mask
            )
            if k_old is not None:
                ret_old = self._extract_retention_scores_vectorized(
                    retention_all, old_mask, k_old
                )
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

    @torch.jit.ignore
    def _compress_eager(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
                        positions: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager-mode compression for compatibility and debugging.

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

        pad_len: int = compressed_len * ratio - seq_len
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len), value=0.0)
            v = F.pad(v, (0, 0, 0, pad_len), value=0.0)
            retention = F.pad(retention, (0, pad_len), value=float('-inf'))
            positions = F.pad(positions, (0, pad_len), value=0)

        k_reshaped: torch.Tensor = k.view(batch_size, num_heads, compressed_len, ratio, head_dim)
        v_reshaped: torch.Tensor = v.view(batch_size, num_heads, compressed_len, ratio, head_dim)
        ret_reshaped: torch.Tensor = retention.view(batch_size, compressed_len, ratio)
        pos_reshaped: torch.Tensor = positions.view(batch_size, compressed_len, ratio)

        weights: torch.Tensor = F.softmax(ret_reshaped, dim=-1)
        weights_kv: torch.Tensor = weights.unsqueeze(1).unsqueeze(-1)

        k_pooled: torch.Tensor = (k_reshaped * weights_kv).sum(dim=3)
        v_pooled: torch.Tensor = (v_reshaped * weights_kv).sum(dim=3)

        pos_pooled: torch.Tensor = (pos_reshaped.float() * weights).sum(dim=-1).long()

        return k_pooled, v_pooled, pos_pooled

    @torch.compile(mode='reduce-overhead', fullgraph=False)
    def _compress_jit(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
                      positions: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """JIT-compiled compression for production inference.

        Uses torch.compile for optimized kernel fusion on PyTorch 2.0+.
        Falls back to eager mode on older PyTorch versions.
        """
        return self._compress_eager(k, v, retention, positions, ratio)

    def _compress(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
                  positions: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using weighted pooling based on retention scores.

        Automatically selects JIT-compiled or eager mode based on config.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            retention: Retention scores of shape [batch, seq]
            positions: Position indices of shape [batch, seq]
            ratio: Compression ratio

        Returns:
            Tuple of (compressed_keys, compressed_values, compressed_positions)
        """
        if self.config.enable_jit and hasattr(torch, 'compile'):
            return self._compress_jit(k, v, retention, positions, ratio)
        return self._compress_eager(k, v, retention, positions, ratio)

    def get_stats(self) -> CacheStats:
        """Get structured compression statistics.

        Returns:
            CacheStats with detailed compression metrics
        """
        if not self.k_cache:
            return CacheStats(
                total_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                tier_distribution=(0, 0, 0, 0),
                peak_memory_mb=0.0,
                compression_time_ms=0.0
            )

        k_comp: Optional[torch.Tensor]
        k_comp, _, _ = self.get_compressed_cache()
        compressed_len: int = k_comp.size(2) if k_comp is not None else 0

        return CacheStats(
            total_tokens=self.total_tokens,
            compressed_tokens=compressed_len,
            compression_ratio=self.total_tokens / max(compressed_len, 1),
            tier_distribution=self._tier_counts,
            peak_memory_mb=self._get_peak_memory_mb(),
            compression_time_ms=self._compression_time_ms
        )
