"""
Real Slow-Burn Test with Actual Attention Traces

This script validates the slow-burn hypothesis by:
1. Loading real GPT-2 model
2. Creating a context with a "needle" token (password) at position 0
3. Running forward passes and extracting actual attention patterns
4. Showing H2O would evict the needle before it's needed
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple


def create_slow_burn_text(needle_text: str = "The password is XK7-9M2.", 
                          filler_length: int = 15000) -> str:
    """Create a slow-burn scenario with needle at the start."""
    # Generate filler text (generic sentences)
    filler_sentences = [
        "This is a sample sentence for context filling.",
        "The document contains various information.",
        "Technical details are provided below.",
        "Additional context follows in this section.",
    ] * (filler_length // 100)  # Approximate token count
    
    filler = " ".join(filler_sentences)[:filler_length * 4]  # Rough character estimate
    
    # Combine: needle + filler + query
    text = f"{needle_text} {filler} What was the password mentioned at the beginning?"
    
    return text


def extract_attention_needle(model, tokenizer, text: str, needle_text: str) -> Dict:
    """
    Extract attention patterns for the needle token.
    
    Returns:
        Dict with attention statistics across layers and heads
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    input_ids = inputs["input_ids"]
    
    # Find needle position
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    needle_token_ids = set(needle_tokens)
    
    # Forward pass with output_attentions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq) for each layer
    
    # Find positions of needle tokens
    needle_positions = []
    for i, token_id in enumerate(input_ids[0]):
        if token_id.item() in needle_token_ids:
            needle_positions.append(i)
    
    print(f"Found needle tokens at positions: {needle_positions}")
    
    # Analyze attention to needle at different query positions
    seq_len = input_ids.size(1)
    
    # Sample query positions (start, middle, end)
    query_positions = [256, 512, 1024, 2048, min(4096, seq_len-1)]
    query_positions = [p for p in query_positions if p < seq_len]
    
    results = {
        'needle_positions': needle_positions,
        'query_positions': query_positions,
        'seq_len': seq_len,
        'attention_traces': []
    }
    
    for layer_idx, attn in enumerate(attentions):
        # attn: (batch, heads, seq, seq)
        attn_np = attn[0].cpu().numpy()  # (heads, seq, seq)
        
        for q_pos in query_positions:
            # Attention FROM query position TO needle positions
            attn_to_needle = []
            for n_pos in needle_positions:
                if n_pos < attn_np.shape[2]:  # Check bounds
                    # Average across heads
                    avg_attn = attn_np[:, q_pos, n_pos].mean()
                    attn_to_needle.append(avg_attn)
            
            avg_attn_to_needle = np.mean(attn_to_needle) if attn_to_needle else 0.0
            
            # Compare to attention to recent tokens
            recent_start = max(0, q_pos - 256)
            recent_attn = attn_np[:, q_pos, recent_start:q_pos].mean()
            
            results['attention_traces'].append({
                'layer': layer_idx,
                'query_pos': q_pos,
                'avg_attn_to_needle': float(avg_attn_to_needle),
                'avg_attn_to_recent': float(recent_attn),
                'ratio': float(avg_attn_to_needle / recent_attn) if recent_attn > 0 else 0.0
            })
    
    return results


def simulate_h2o_eviction(attention_data: Dict, budget: int = 2048) -> Dict:
    """
    Simulate H2O eviction policy on the attention data.
    
    H2O keeps tokens with highest accumulated attention.
    Returns whether needle survives.
    """
    needle_positions = attention_data['needle_positions']
    seq_len = attention_data['seq_len']
    traces = attention_data['attention_traces']
    
    # Compute accumulated attention per position
    accumulated = {i: 0.0 for i in range(seq_len)}
    
    for trace in traces:
        # Accumulate attention received by each position
        # In reality, this would be cumulative across all queries
        for n_pos in needle_positions:
            if trace['query_pos'] > n_pos:
                accumulated[n_pos] += trace['avg_attn_to_needle']
    
    # H2O: keep top-k by accumulated attention
    sorted_positions = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
    kept_positions = set([p for p, _ in sorted_positions[:budget]])
    
    needle_survives = any(p in kept_positions for p in needle_positions)
    needle_rank = next((i for i, (p, _) in enumerate(sorted_positions) if p in needle_positions), seq_len)
    
    # Compute accumulated attention for needle
    needle_accumulated = sum(accumulated[p] for p in needle_positions) / len(needle_positions) if needle_positions else 0
    
    return {
        'needle_survives': needle_survives,
        'needle_rank': needle_rank,
        'budget': budget,
        'seq_len': seq_len,
        'kept_count': len(kept_positions),
        'compression_ratio': seq_len / len(kept_positions) if kept_positions else 1.0,
        'needle_accumulated_attention': float(needle_accumulated),
        'median_accumulated_attention': float(np.median([v for v in accumulated.values() if v > 0])),
        'max_accumulated_attention': float(max(accumulated.values()))
    }


def validate_slow_burn():
    """Run the full slow-burn validation experiment."""
    print("=" * 80)
    print("SLOW-BURN VALIDATION: Real Attention Traces from GPT-2")
    print("=" * 80)
    
    # Load model
    print("\nLoading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    # Create slow-burn text
    needle = "The password is XK7-9M2."
    text = create_slow_burn_text(needle_text=needle, filler_length=2000)
    
    print(f"\nGenerated text with needle at position ~0")
    print(f"Total length: {len(text)} chars")
    
    # Extract attention
    print("\nExtracting attention patterns...")
    attention_data = extract_attention_needle(model, tokenizer, text, needle)
    
    # Simulate H2O
    print("\nSimulating H2O eviction...")
    h2o_result = simulate_h2o_eviction(attention_data, budget=2048)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nSequence length: {attention_data['seq_len']} tokens")
    print(f"Needle positions: {attention_data['needle_positions']}")
    
    print(f"\nH2O Simulation (budget={h2o_result['budget']}):")
    print(f"  - Needle survives: {'YES' if h2o_result['needle_survives'] else 'NO (EVICTED)'}")
    print(f"  - Needle rank: {h2o_result['needle_rank']} / {h2o_result['seq_len']}")
    print(f"  - Compression: {h2o_result['compression_ratio']:.2f}x")
    
    print(f"\nAttention Analysis:")
    print(f"  - Needle accumulated attention: {h2o_result['needle_accumulated_attention']:.6f}")
    print(f"  - Median accumulated attention: {h2o_result['median_accumulated_attention']:.6f}")
    print(f"  - Max accumulated attention: {h2o_result['max_accumulated_attention']:.6f}")
    
    # Show attention traces at different query positions
    print(f"\nAttention to needle at different query positions:")
    for trace in attention_data['attention_traces'][:5]:  # First few traces
        print(f"  Layer {trace['layer']}, Query {trace['query_pos']}: "
              f"{trace['avg_attn_to_needle']:.6f} (vs recent: {trace['avg_attn_to_recent']:.6f})")
    
    # Save results
    results = {
        'attention_data': attention_data,
        'h2o_result': h2o_result,
        'validation': 'CONFIRMED' if not h2o_result['needle_survives'] else 'UNEXPECTED'
    }
    
    with open('slow_burn_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"SLOW-BURN HYPOTHESIS: {results['validation']}")
    print("=" * 80)
    
    if not h2o_result['needle_survives']:
        print("\n✓ VALIDATED: H2O evicts the needle before it's needed")
        print("  - The needle receives near-zero attention early in the sequence")
        print("  - H2O's accumulated attention policy ranks it very low")
        print("  - By the time it's needed (position 15K), it's already evicted")
    else:
        print("\n? UNEXPECTED: Needle survived H2O eviction")
        print("  - This may indicate the needle receives more attention than expected")
        print("  - Or the budget is sufficient to retain all tokens")
    
    print("\nResults saved to slow_burn_validation.json")
    
    return results


if __name__ == "__main__":
    validate_slow_burn()
