"""Attention-guided salience scorer for KV cache compression."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

from .exceptions import (
    ValidationError,
    DeviceMismatchError,
    DtypeMismatchError,
    ShapeMismatchError,
    DimensionError,
    EmptyTensorError,
    InvalidValueError,
)


class ScoreAggregator:
    """Aggregates scores using EMA with tensor-based storage for GPU acceleration."""

    def __init__(self, ema_decay: float = 0.95, max_seq_len: int = 32768) -> None:
        self.ema_decay: float = ema_decay
        self.max_seq_len: int = max_seq_len
        self._scores: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

    def _ensure_initialized(self, seq_len: int, device: torch.device) -> None:
        """Ensure tensor storage is initialized and can accommodate seq_len."""
        actual_len = min(seq_len, self.max_seq_len)

        if self._scores is None or self._scores.size(0) < actual_len:
            # Reallocate with larger capacity
            new_scores = torch.zeros(actual_len, device=device)
            new_mask = torch.zeros(actual_len, dtype=torch.bool, device=device)

            if self._scores is not None:
                # Copy existing data
                copy_len = min(self._scores.size(0), actual_len)
                new_scores[:copy_len] = self._scores[:copy_len]
                new_mask[:copy_len] = self._mask[:copy_len]

            self._scores = new_scores
            self._mask = new_mask
            self._device = device

    def update(self, position: int, value: float) -> None:
        """Update score for a position using EMA."""
        if position >= self.max_seq_len or self._scores is None:
            return

        if self._mask[position]:
            self._scores[position] = (
                self.ema_decay * self._scores[position] +
                (1 - self.ema_decay) * value
            )
        else:
            self._scores[position] = value
            self._mask[position] = True

    def update_all(self, values: torch.Tensor) -> None:
        """Update scores for all positions using EMA (vectorized).

        Args:
            values: Tensor of attention values of shape [seq_len]
        """
        seq_len: int = values.size(0)
        device: torch.device = values.device

        self._ensure_initialized(seq_len, device)

        positions = torch.arange(seq_len, device=device)
        existing_mask = self._mask[positions]

        # Vectorized EMA update: new = decay * old + (1-decay) * new
        old_scores = self._scores[positions]
        new_scores = torch.where(
            existing_mask,
            self.ema_decay * old_scores + (1 - self.ema_decay) * values,
            values
        )

        self._scores[positions] = new_scores
        self._mask[positions] = True

    def get(self, position: int, default: float = 0.0) -> float:
        """Get score for a position."""
        if (self._scores is None or position >= self._scores.size(0) or
            not self._mask[position]):
            return default
        return float(self._scores[position])

    @property
    def position_importance(self) -> Dict[int, float]:
        """Backward compatibility: expose as dict interface."""
        if self._scores is None:
            return {}

        mask_cpu = self._mask.cpu().numpy()
        scores_cpu = self._scores.cpu().numpy()

        result = {}
        for pos in range(self._scores.size(0)):
            if mask_cpu[pos]:
                result[pos] = float(scores_cpu[pos])

        return result

    def get_all_scores(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get all position scores as a tensor.

        Args:
            seq_len: Length of the sequence
            device: Device for tensor operations

        Returns:
            Tensor of shape [seq_len] with scores for each position
        """
        self._ensure_initialized(seq_len, device)

        scores = torch.zeros(seq_len, device=device)
        valid_len = min(seq_len, self._scores.size(0))

        valid_positions = self._mask[:valid_len]
        if valid_positions.any():
            scores[:valid_len] = torch.where(
                valid_positions,
                self._scores[:valid_len],
                torch.zeros_like(self._scores[:valid_len])
            )

        return scores

    def decay_positions(self, current_seq_len: int) -> None:
        """Remove positions beyond current sequence length."""
        if self._mask is None:
            return

        # Zero out positions beyond current sequence length
        mask_to_clear = torch.arange(self._mask.size(0), device=self._mask.device) >= current_seq_len
        self._mask = self._mask & ~mask_to_clear
        self._scores = torch.where(self._mask, self._scores, torch.zeros_like(self._scores))

    def reset(self) -> None:
        """Reset all tracked scores."""
        self._scores = None
        self._mask = None
        self._device = None


class StructuralScorer:
    """Computes structural importance scores for tokens."""

    def __init__(self) -> None:
        self.structural_scores: Dict[int, float] = {}

    def compute_scores(
        self,
        token_ids: torch.Tensor,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute structural importance scores for tokens."""
        batch_size: int = token_ids.size(0)

        position_scores: torch.Tensor = self._compute_position_scores(seq_len, device)
        scores: torch.Tensor = position_scores.unsqueeze(0).expand(batch_size, -1)
        scores = self._apply_cached_scores(token_ids, scores, device)

        return scores

    def _compute_position_scores(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute position-based structural scores."""
        positions: torch.Tensor = torch.arange(seq_len, device=device)
        position_scores: torch.Tensor = torch.zeros(seq_len, device=device)

        boundary_threshold: int = max(1, seq_len // 20)
        position_scores[:boundary_threshold] = 0.6
        position_scores[-boundary_threshold:] = 0.6

        decile_mask: torch.Tensor = (positions % 10 == 0)
        position_scores = torch.where(
            decile_mask,
            torch.maximum(position_scores, torch.tensor(0.4, device=device)),
            position_scores
        )

        return position_scores

    def _apply_cached_scores(
        self,
        token_ids: torch.Tensor,
        scores: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Apply cached structural scores to matching token IDs."""
        if not self.structural_scores:
            return scores

        unique_tokens: torch.Tensor = torch.unique(token_ids)
        for token_id in unique_tokens.tolist():
            if token_id in self.structural_scores:
                mask: torch.Tensor = (token_ids == token_id)
                cached_score: torch.Tensor = torch.tensor(
                    self.structural_scores[token_id], device=device
                )
                scores = torch.where(
                    mask,
                    torch.maximum(scores, cached_score),
                    scores
                )

        return scores

    def update_score(self, token_id: int, score: float) -> None:
        """Update cached structural score for a token ID."""
        self.structural_scores[token_id] = score

    def reset(self) -> None:
        """Reset cached structural scores."""
        self.structural_scores.clear()


class AttentionGuidedScorer:
    """Scores token importance based on attention patterns.

    Uses EMA (Exponential Moving Average) to track attention weights over time,
    combined with structural priors to identify important tokens.

    Args:
        ema_decay: Decay rate for EMA updates (default: 0.95)
        structural_floor: Minimum score from structural patterns (default: 0.1)
    """

    def __init__(self, ema_decay: float = 0.95, structural_floor: float = 0.1) -> None:
        if not 0.0 <= ema_decay <= 1.0:
            raise ValidationError(f"ema_decay must be in [0, 1], got {ema_decay}")
        if not 0.0 <= structural_floor <= 1.0:
            raise ValidationError(f"structural_floor must be in [0, 1], got {structural_floor}")

        self.ema_decay: float = ema_decay
        self.structural_floor: float = structural_floor
        self.score_aggregator: ScoreAggregator = ScoreAggregator(ema_decay)
        self.structural_scorer: StructuralScorer = StructuralScorer()

    def _validate_tensor(self, tensor: torch.Tensor, name: str, expected_dims: Optional[int] = None) -> None:
        if expected_dims is not None and tensor.dim() != expected_dims:
            raise DimensionError(expected_dims, tensor.dim(), name)
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            raise InvalidValueError("NaN", name, nan_count)
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            raise InvalidValueError("Inf", name, inf_count)

    def _validate_device_consistency(self, *tensors: torch.Tensor) -> torch.device:
        devices: Set[torch.device] = {t.device for t in tensors}
        if len(devices) > 1:
            raise DeviceMismatchError(devices)
        return devices.pop() if devices else torch.device('cpu')

    @property
    def position_importance(self) -> Dict[int, float]:
        """Backward compatibility: expose position_importance from aggregator."""
        return self.score_aggregator.position_importance

    @property
    def structural_scores(self) -> Dict[int, float]:
        """Backward compatibility: expose structural_scores from scorer."""
        return self.structural_scorer.structural_scores

    @property
    def seen_positions(self) -> set:
        """Backward compatibility: expose seen_positions from aggregator."""
        return self.score_aggregator.seen_positions

    def compute_structural_score(
        self,
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None
    ) -> torch.Tensor:
        """Compute structural importance scores for tokens.

        Args:
            token_ids: Token IDs of shape [batch, seq_len]
            tokenizer: Optional tokenizer for token-specific scoring

        Returns:
            Structural scores of shape [batch, seq_len]

        Raises:
            DimensionError: If token_ids doesn't have 2 dimensions
            InvalidValueError: If token_ids contains invalid values
        """
        self._validate_tensor(token_ids, "token_ids", expected_dims=2)

        batch_size: int
        seq_len: int
        batch_size, seq_len = token_ids.shape
        device: torch.device = token_ids.device

        return self.structural_scorer.compute_scores(token_ids, seq_len, device)

    def update_from_attention(
        self,
        attention_weights: torch.Tensor,
        query_position: int,
        generated_token_id: int
    ) -> None:
        """Update importance scores from attention weights.

        Args:
            attention_weights: Attention weights from model output
            query_position: Position of the query token
            generated_token_id: ID of the generated token

        Raises:
            InvalidValueError: If attention_weights contains NaN or Inf
        """
        self._validate_tensor(attention_weights, "attention_weights")

        attn: torch.Tensor = self._normalize_attention(attention_weights)
        self.score_aggregator.update_all(attn)

    def _normalize_attention(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Normalize attention weights to 1D tensor."""
        if attention_weights.dim() == 2:
            return attention_weights.mean(dim=0)
        return attention_weights

    def get_salience_scores(
        self,
        seq_len: int,
        structural_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get salience scores combining attention and structural patterns.

        Args:
            seq_len: Length of the sequence
            structural_scores: Optional structural importance scores

        Returns:
            Combined salience scores of shape [seq_len]

        Raises:
            ValidationError: If seq_len is not positive
            InvalidValueError: If structural_scores contains NaN or Inf
        """
        if seq_len <= 0:
            raise ValidationError(f"seq_len must be positive, got {seq_len}")

        if structural_scores is not None:
            self._validate_tensor(structural_scores, "structural_scores", expected_dims=2)

        device: torch.device = structural_scores.device if structural_scores is not None else torch.device('cpu')

        # Vectorized: Build attention scores tensor from dict
        attn_scores: torch.Tensor = self.score_aggregator.get_all_scores(seq_len, device)

        # Vectorized: Get structural scores
        struct_scores: torch.Tensor = self._get_all_structural_scores(structural_scores, seq_len, device)

        # Vectorized: Combine scores
        scores: torch.Tensor = torch.maximum(attn_scores, struct_scores * self.structural_floor)

        return scores

    def _get_structural_score(
        self,
        structural_scores: Optional[torch.Tensor],
        pos: int
    ) -> float:
        """Extract structural score for a position."""
        if structural_scores is None or pos >= len(structural_scores[0]):
            return 0.0
        return float(structural_scores[0, pos])

    def _get_all_structural_scores(
        self,
        structural_scores: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Extract all structural scores as a tensor.

        Args:
            structural_scores: Optional structural scores tensor
            seq_len: Length of the sequence
            device: Device for tensor operations

        Returns:
            Tensor of shape [seq_len] with structural scores
        """
        scores: torch.Tensor = torch.zeros(seq_len, device=device)

        if structural_scores is not None:
            seq_len_actual: int = min(seq_len, structural_scores.size(-1))
            scores[:seq_len_actual] = structural_scores[0, :seq_len_actual]

        return scores

    def decay_unseen_positions(self, current_seq_len: int) -> None:
        """Remove positions that are no longer in the sequence.

        Args:
            current_seq_len: Current sequence length
        """
        self.score_aggregator.decay_positions(current_seq_len)

    def reset(self) -> None:
        """Reset all tracked scores and positions."""
        self.score_aggregator.reset()
        self.structural_scorer.reset()


class AttentionBasedKVCache:
    """KV cache with attention-guided compression.

    Integrates TieredKVCache with AttentionGuidedScorer for intelligent
    token retention based on attention patterns.

    Args:
        config: Cache configuration
        tokenizer: Optional tokenizer for structural scoring
    """

    def __init__(self, config: Any, tokenizer: Optional[Any] = None) -> None:
        from .core import CacheConfig, TieredKVCache

        self.config: Any = config
        self.tokenizer: Optional[Any] = tokenizer
        self.scorer: AttentionGuidedScorer = AttentionGuidedScorer(
            ema_decay=0.95,
            structural_floor=0.1
        )

    def compress_with_attention(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_weights: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compress KV cache using attention-guided salience scoring.

        Args:
            k: Key tensor of shape [batch, heads, seq, head_dim]
            v: Value tensor of shape [batch, heads, seq, head_dim]
            attention_weights: Attention weights from model
            token_ids: Token IDs of shape [batch, seq]
            positions: Position indices of shape [batch, seq]

        Returns:
            Tuple of (compressed_keys, compressed_values, stats)
        """
        batch_size: int
        num_heads: int
        seq_len: int
        head_dim: int
        batch_size, num_heads, seq_len, head_dim = k.shape

        self._update_scorer(attention_weights, token_ids, seq_len)

        structural_scores: torch.Tensor = self.scorer.compute_structural_score(token_ids)
        salience: torch.Tensor = self.scorer.get_salience_scores(seq_len, structural_scores)
        salience = salience.unsqueeze(0).repeat(batch_size, 1)

        from .core import TieredKVCache
        cache: TieredKVCache = TieredKVCache(self.config)
        cache.add(k, v, salience, positions)

        k_comp: Optional[torch.Tensor]
        v_comp: Optional[torch.Tensor]
        pos_comp: Optional[torch.Tensor]
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats: Dict[str, Any] = cache.get_stats()

        if k_comp is not None:
            self.scorer.decay_unseen_positions(k_comp.size(2))

        return k_comp, v_comp, stats

    def _update_scorer(
        self,
        attention_weights: torch.Tensor,
        token_ids: torch.Tensor,
        seq_len: int
    ) -> None:
        """Update scorer with attention weights."""
        self.scorer.update_from_attention(
            attention_weights[0],
            query_position=seq_len - 1,
            generated_token_id=int(token_ids[0, -1]) if token_ids.size(1) > 0 else 0
        )


def extract_attention_weights(model_outputs: Any) -> Optional[torch.Tensor]:
    """Extract attention weights from model outputs.

    Args:
        model_outputs: Model output object with attentions attribute

    Returns:
        Attention weights from the last layer and position, or None
    """
    if not hasattr(model_outputs, 'attentions'):
        return None

    attentions: Optional[Tuple[torch.Tensor, ...]] = model_outputs.attentions
    if attentions is None or len(attentions) == 0:
        return None

    last_layer_attn: torch.Tensor = attentions[-1]
    last_pos_attn: torch.Tensor = last_layer_attn[:, :, -1, :]

    return last_pos_attn


class AttentionGuidedWrapper:
    """Wrapper for models with attention-guided KV cache compression.

    Provides a generate method that automatically compresses the KV cache
    based on attention patterns during generation.

    Args:
        model: Language model with attention output
        tokenizer: Tokenizer for decoding
        cache_config: Configuration for the tiered cache
    """

    def __init__(self, model: Any, tokenizer: Any, cache_config: Any) -> None:
        self.model: Any = model
        self.tokenizer: Any = tokenizer
        self.cache_config: Any = cache_config
        self.attention_cache: AttentionBasedKVCache = AttentionBasedKVCache(
            cache_config, tokenizer
        )

    def generate_with_attention_guidance(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate text with attention-guided KV cache compression.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, list_of_compression_stats)
        """
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)

        generated: torch.Tensor = input_ids
        all_stats: List[Dict[str, Any]] = []

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs: Any = self.model(
                    generated,
                    use_cache=True,
                    output_attentions=True
                )

            attn_weights: Optional[torch.Tensor] = extract_attention_weights(outputs)
            past_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = outputs.past_key_values

            if attn_weights is not None and step > 0:
                stats: Optional[Dict[str, Any]] = self._compress_past_kv(
                    past_kv, generated, attn_weights
                )
                if stats is not None:
                    all_stats.append(stats)

            next_token: torch.Tensor = self._sample_next_token(outputs, temperature)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        result: str = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return result, all_stats

    def _compress_past_kv(
        self,
        past_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        generated: torch.Tensor,
        attn_weights: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """Compress past key-value cache for all layers."""
        stats: Optional[Dict[str, Any]] = None

        for layer_idx, layer_cache in enumerate(past_kv):
            k: torch.Tensor = layer_cache[0]
            v: torch.Tensor = layer_cache[1]

            seq_len: int = k.size(2)
            positions: torch.Tensor = torch.arange(seq_len, device=k.device).unsqueeze(0)
            token_ids: torch.Tensor = generated[:, :seq_len]

            k_comp: torch.Tensor
            v_comp: torch.Tensor
            layer_stats: Dict[str, Any]
            k_comp, v_comp, layer_stats = self.attention_cache.compress_with_attention(
                k, v, attn_weights, token_ids, positions
            )

            if layer_idx == 0:
                stats = layer_stats

        return stats

    def _sample_next_token(self, outputs: Any, temperature: float) -> torch.Tensor:
        """Sample next token from model outputs."""
        logits: torch.Tensor = outputs.logits
        next_token_logits: torch.Tensor = logits[:, -1, :] / temperature
        probs: torch.Tensor = F.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
