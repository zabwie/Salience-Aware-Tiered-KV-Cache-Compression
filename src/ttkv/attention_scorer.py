"""Attention-guided salience scorer for KV cache compression."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class AttentionGuidedScorer:
    """Scores token importance based on attention patterns.

    Uses EMA (Exponential Moving Average) to track attention weights over time,
    combined with structural priors to identify important tokens.

    Args:
        ema_decay: Decay rate for EMA updates (default: 0.95)
        structural_floor: Minimum score from structural patterns (default: 0.1)
    """

    def __init__(self, ema_decay: float = 0.95, structural_floor: float = 0.1) -> None:
        self.ema_decay: float = ema_decay
        self.structural_floor: float = structural_floor
        self.position_importance: Dict[int, float] = {}
        self.structural_scores: Dict[int, float] = {}
        self.seen_positions: set = set()

    def compute_structural_score(self, token_ids: torch.Tensor,
                                  tokenizer: Optional[Any] = None) -> torch.Tensor:
        """Compute structural importance scores for tokens.

        Args:
            token_ids: Token IDs of shape [batch, seq_len]
            tokenizer: Optional tokenizer for token-specific scoring

        Returns:
            Structural scores of shape [batch, seq_len]
        """
        batch_size: int
        seq_len: int
        batch_size, seq_len = token_ids.shape
        scores: torch.Tensor = torch.zeros(batch_size, seq_len)

        for b in range(batch_size):
            for pos in range(seq_len):
                token_id: int = int(token_ids[b, pos])

                if token_id in self.structural_scores:
                    scores[b, pos] = self.structural_scores[token_id]
                    continue

                score: float = 0.0
                if pos < seq_len // 20 or pos > seq_len * 19 // 20:
                    score = max(score, 0.6)
                if pos % 10 == 0:
                    score = max(score, 0.4)

                self.structural_scores[token_id] = score
                scores[b, pos] = score

        return scores

    def update_from_attention(self, attention_weights: torch.Tensor,
                               query_position: int,
                               generated_token_id: int) -> None:
        """Update importance scores from attention weights.

        Args:
            attention_weights: Attention weights from model output
            query_position: Position of the query token
            generated_token_id: ID of the generated token
        """
        attn: torch.Tensor
        if attention_weights.dim() == 2:
            attn = attention_weights.mean(dim=0)
        else:
            attn = attention_weights

        seq_len: int = attn.size(0)

        for pos in range(seq_len):
            attn_val: float = float(attn[pos])
            if pos in self.position_importance:
                self.position_importance[pos] = (
                    self.ema_decay * self.position_importance[pos] +
                    (1 - self.ema_decay) * attn_val
                )
            else:
                self.position_importance[pos] = attn_val
            self.seen_positions.add(pos)

    def get_salience_scores(self, seq_len: int,
                            structural_scores: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
        """Get salience scores combining attention and structural patterns.

        Args:
            seq_len: Length of the sequence
            structural_scores: Optional structural importance scores

        Returns:
            Combined salience scores of shape [seq_len]
        """
        scores: torch.Tensor = torch.zeros(seq_len)

        for pos in range(seq_len):
            attn_score: float = self.position_importance.get(pos, 0.0)
            struct_score: float = 0.0
            if structural_scores is not None and pos < len(structural_scores[0]):
                struct_score = float(structural_scores[0, pos])
            scores[pos] = max(attn_score, struct_score * self.structural_floor)

        return scores

    def decay_unseen_positions(self, current_seq_len: int) -> None:
        """Remove positions that are no longer in the sequence.

        Args:
            current_seq_len: Current sequence length
        """
        positions_to_decay: List[int] = []
        for pos in list(self.position_importance.keys()):
            if pos >= current_seq_len:
                positions_to_decay.append(pos)

        for pos in positions_to_decay:
            del self.position_importance[pos]
            self.seen_positions.discard(pos)

    def reset(self) -> None:
        """Reset all tracked scores and positions."""
        self.position_importance.clear()
        self.structural_scores.clear()
        self.seen_positions.clear()


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

    def compress_with_attention(self, k: torch.Tensor, v: torch.Tensor,
                                 attention_weights: torch.Tensor,
                                 token_ids: torch.Tensor,
                                 positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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

        self.scorer.update_from_attention(
            attention_weights[0],
            query_position=seq_len - 1,
            generated_token_id=int(token_ids[0, -1]) if token_ids.size(1) > 0 else 0
        )

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
        self.attention_cache: AttentionBasedKVCache = AttentionBasedKVCache(cache_config, tokenizer)

    def generate_with_attention_guidance(self, input_ids: torch.Tensor,
                                         max_new_tokens: int = 50,
                                         temperature: float = 0.7) -> Tuple[str, List[Dict[str, Any]]]:
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
                compressed_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for layer_idx, layer_cache in enumerate(past_kv):
                    k: torch.Tensor = layer_cache[0]
                    v: torch.Tensor = layer_cache[1]

                    seq_len: int = k.size(2)
                    positions: torch.Tensor = torch.arange(seq_len, device=k.device).unsqueeze(0)
                    token_ids: torch.Tensor = generated[:, :seq_len]

                    k_comp: torch.Tensor
                    v_comp: torch.Tensor
                    stats: Dict[str, Any]
                    k_comp, v_comp, stats = self.attention_cache.compress_with_attention(
                        k, v, attn_weights, token_ids, positions
                    )

                    compressed_past.append((k_comp, v_comp))

                    if layer_idx == 0:
                        all_stats.append(stats)

            logits: torch.Tensor = outputs.logits
            next_token_logits: torch.Tensor = logits[:, -1, :] / temperature
            probs: torch.Tensor = F.softmax(next_token_logits, dim=-1)
            next_token: torch.Tensor = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        result: str = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return result, all_stats
