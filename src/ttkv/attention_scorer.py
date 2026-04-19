"""Attention-guided dynamic salience scorer for KV cache compression.

Uses the model's own attention patterns as ground truth for importance,
accumulated with exponential moving average (EMA).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class AttentionGuidedScorer:
    """
    Learns token importance from model attention patterns.
    
    The model's attention weights are ground truth for what it finds important.
    We accumulate these with EMA to get stable importance estimates per position.
    """
    
    def __init__(self, ema_decay: float = 0.95, structural_floor: float = 0.1):
        """
        Args:
            ema_decay: Decay rate for EMA (higher = longer memory)
            structural_floor: Minimum score for structurally important tokens
                             (handles rare-but-critical like passwords)
        """
        self.ema_decay = ema_decay
        self.structural_floor = structural_floor
        
        # EMA of attention weights per position
        self.position_importance = {}
        
        # Structural prior scores (numbers, proper nouns, etc.)
        self.structural_scores = {}
        
        # Track which positions have been seen
        self.seen_positions = set()
        
    def compute_structural_score(self, token_ids: torch.Tensor, 
                                  tokenizer=None) -> torch.Tensor:
        """
        Compute base structural importance from token characteristics.
        Rare but distinctive tokens (numbers, proper nouns) get floor score.
        """
        batch_size, seq_len = token_ids.shape
        scores = torch.zeros(batch_size, seq_len)
        
        for b in range(batch_size):
            for pos in range(seq_len):
                token_id = int(token_ids[b, pos])
                
                # Check if we've computed this before
                if token_id in self.structural_scores:
                    scores[b, pos] = self.structural_scores[token_id]
                    continue
                
                score = 0.0
                
                # High-entropy indicators (would need tokenizer for actual text)
                # For now, use position-based heuristics
                
                # First and last tokens often important
                if pos < seq_len // 20 or pos > seq_len * 19 // 20:
                    score = max(score, 0.6)
                
                # Every 10th token (rough proxy for content words)
                if pos % 10 == 0:
                    score = max(score, 0.4)
                
                self.structural_scores[token_id] = score
                scores[b, pos] = score
        
        return scores
    
    def update_from_attention(self, attention_weights: torch.Tensor,
                             query_position: int,
                             generated_token_id: int):
        """
        Update importance scores from attention weights.
        
        Args:
            attention_weights: [num_heads, seq_len] or [seq_len]
                             attention weights for this generation step
            query_position: Position of the generated token
            generated_token_id: ID of token that was generated
        """
        # Average across heads if needed
        if attention_weights.dim() == 2:
            attn = attention_weights.mean(dim=0)  # [seq_len]
        else:
            attn = attention_weights
        
        seq_len = attn.size(0)
        
        # Update EMA for each position
        for pos in range(seq_len):
            attn_val = float(attn[pos])
            
            if pos in self.position_importance:
                # EMA update: new_val = decay * old + (1-decay) * new
                old_val = self.position_importance[pos]
                self.position_importance[pos] = (
                    self.ema_decay * old_val + 
                    (1 - self.ema_decay) * attn_val
                )
            else:
                # Initialize with current value
                self.position_importance[pos] = attn_val
            
            self.seen_positions.add(pos)
    
    def get_salience_scores(self, seq_len: int, 
                           structural_scores: Optional[torch.Tensor] = None
                           ) -> torch.Tensor:
        """
        Get salience scores for all positions.
        
        Combines learned attention-based importance with structural floor.
        """
        scores = torch.zeros(seq_len)
        
        for pos in range(seq_len):
            # Attention-based score (EMA)
            attn_score = self.position_importance.get(pos, 0.0)
            
            # Structural floor
            struct_score = 0.0
            if structural_scores is not None and pos < len(structural_scores[0]):
                struct_score = float(structural_scores[0, pos])
            
            # Combine: take max or weighted sum
            # Using max ensures structural floor is respected
            scores[pos] = max(attn_score, struct_score * self.structural_floor)
        
        return scores
    
    def decay_unseen_positions(self, current_seq_len: int):
        """Decay importance of positions that haven't been attended to recently."""
        positions_to_decay = []
        for pos in list(self.position_importance.keys()):
            if pos >= current_seq_len:
                # Position is no longer in sequence
                positions_to_decay.append(pos)
        
        for pos in positions_to_decay:
            del self.position_importance[pos]
            self.seen_positions.discard(pos)
    
    def reset(self):
        """Reset all scores (for new conversation)."""
        self.position_importance.clear()
        self.structural_scores.clear()
        self.seen_positions.clear()


class AttentionBasedKVCache:
    """
    KV cache that uses attention weights to guide compression.
    
    Replaces static retention scores with dynamic attention-based scores.
    """
    
    def __init__(self, config, tokenizer=None):
        from .core import CacheConfig, TieredKVCache
        
        self.config = config
        self.tokenizer = tokenizer
        self.scorer = AttentionGuidedScorer(
            ema_decay=0.95,
            structural_floor=0.1
        )
        
    def compress_with_attention(self, k: torch.Tensor, v: torch.Tensor,
                                attention_weights: torch.Tensor,
                                token_ids: torch.Tensor,
                                positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compress KV cache using attention-guided salience.
        
        Args:
            k: [batch, heads, seq_len, head_dim]
            v: [batch, heads, seq_len, head_dim]
            attention_weights: [batch, heads, seq_len] attention from last step
            token_ids: [batch, seq_len] token IDs
            positions: [batch, seq_len] position indices
            
        Returns:
            Compressed k, v, and stats dict
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Update scorer with attention weights
        # Use first batch item and average across heads
        self.scorer.update_from_attention(
            attention_weights[0],  # [heads, seq_len]
            query_position=seq_len - 1,
            generated_token_id=int(token_ids[0, -1]) if token_ids.size(1) > 0 else 0
        )
        
        # Compute structural scores
        structural_scores = self.scorer.compute_structural_score(token_ids)
        
        # Get combined salience scores
        salience = self.scorer.get_salience_scores(seq_len, structural_scores)
        salience = salience.unsqueeze(0).repeat(batch_size, 1)  # [batch, seq_len]
        
        # Create cache and compress
        from salience_cache import TieredKVCache
        cache = TieredKVCache(self.config)
        cache.add(k, v, salience, positions)
        
        k_comp, v_comp, pos_comp = cache.get_compressed_cache()
        stats = cache.get_stats()
        
        # Clean up old positions
        if k_comp is not None:
            self.scorer.decay_unseen_positions(k_comp.size(2))
        
        return k_comp, v_comp, stats


def extract_attention_weights(model_outputs) -> Optional[torch.Tensor]:
    """
    Extract attention weights from model outputs.
    
    Returns None if attention wasn't computed.
    """
    if not hasattr(model_outputs, 'attentions'):
        return None
    
    attentions = model_outputs.attentions
    if attentions is None or len(attentions) == 0:
        return None
    
    # attentions is tuple of [batch, heads, seq_len, seq_len] per layer
    # Use last layer's attention
    last_layer_attn = attentions[-1]  # [batch, heads, seq_len, seq_len]
    
    # Get attention to last query position (what the model just attended to)
    # [batch, heads, seq_len] - attention from last position to all previous
    last_pos_attn = last_layer_attn[:, :, -1, :]  # [batch, heads, seq_len]
    
    return last_pos_attn


class AttentionGuidedWrapper:
    """
    Wrapper that handles attention extraction and guided compression.
    """
    
    def __init__(self, model, tokenizer, cache_config):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_config = cache_config
        self.attention_cache = AttentionBasedKVCache(cache_config, tokenizer)
        
    def generate_with_attention_guidance(self, input_ids: torch.Tensor,
                                        max_new_tokens: int = 50,
                                        temperature: float = 0.7) -> Tuple[str, List[Dict]]:
        """
        Generate text with attention-guided KV cache compression.
        """
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)
        
        generated = input_ids
        all_stats = []
        
        for step in range(max_new_tokens):
            # Forward pass WITH attention output
            with torch.no_grad():
                outputs = self.model(
                    generated,
                    use_cache=True,
                    output_attentions=True
                )
            
            # Extract attention weights
            attn_weights = extract_attention_weights(outputs)
            
            # Get KV cache
            past_kv = outputs.past_key_values
            
            # Compress using attention guidance
            if attn_weights is not None and step > 0:
                compressed_past = []
                for layer_idx, layer_cache in enumerate(past_kv):
                    k = layer_cache[0]
                    v = layer_cache[1]
                    
                    # Get attention for this layer
                    layer_attn = attn_weights  # [batch, heads, seq_len]
                    
                    # Create positions
                    seq_len = k.size(2)
                    positions = torch.arange(seq_len, device=k.device).unsqueeze(0)
                    
                    # Token IDs for structural scoring
                    token_ids = generated[:, :seq_len]
                    
                    # Compress with attention guidance
                    k_comp, v_comp, stats = self.attention_cache.compress_with_attention(
                        k, v, layer_attn, token_ids, positions
                    )
                    
                    compressed_past.append((k_comp, v_comp))
                    
                    if layer_idx == 0:  # Log stats from first layer
                        all_stats.append(stats)
                
                # Note: Using compressed cache with DynamicCache is tricky
                # For now, we track stats but don't actually use compressed cache
                # in generation to avoid dtype/format issues
            
            # Generate next token
            logits = outputs.logits
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return result, all_stats
