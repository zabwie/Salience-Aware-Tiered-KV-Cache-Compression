# FFN Component Profiling Learnings

## TinyLlama FFN Structure (SwiGLU)

### Component Shapes (per layer)
- **gate_proj**: (5632, 2048) - 11,534,336 params, 22.00 MB FP16
- **up_proj**: (5632, 2048) - 11,534,336 params, 22.00 MB FP16  
- **down_proj**: (2048, 5632) - 11,534,336 params, 22.00 MB FP16

All three projections have equal parameter counts but different shapes:
- gate_proj and up_proj expand from hidden_dim (2048) to intermediate_dim (5632)
- down_proj projects back from intermediate_dim to hidden_dim

### Timing Breakdown (per forward pass)
- **gate_proj**: 26.2% of FFN time (4.55 ms)
- **up_proj**: 17.6% of FFN time (3.05 ms)
- **down_proj**: 56.2% of FFN time (9.74 ms)

**Key insight**: down_proj dominates timing at 56% - this is the final projection back to hidden dimension.

### Memory Breakdown
- **Total FFN weight memory**: 1.523 GB (22 layers × 3 projections × 22 MB)
- **Total FFN activation memory**: 6.748 GB
- **Weights**: 18.4% of FFN memory
- **Activations**: 81.6% of FFN memory

## Critical Finding: FFN is ACTIVATION-BOUND

The FFN memory is dominated by activations (81.6%), not weights (18.4%).

### Implications for Optimization

**Weight quantization (4-bit) impact:**
- Weight savings: 1.142 GB (75% of 1.523 GB)
- Extra context tokens unlocked: ~86 tokens
- Extra context at seq_len=512: only 0.2x (negligible)

**Conclusion**: Quantization won't help much for FFN memory. Need activation compression instead.

### Activation Memory Per Token
- 12,870 KB per token across all 22 layers
- This scales linearly with sequence length
- At seq_len=512: 6.6 GB just for FFN activations

## Recommendations

1. **Skip FFN weight quantization** - minimal memory benefit for context extension
2. **Focus on activation compression** - this is where the memory is
3. **Consider activation checkpointing** - trades compute for memory
4. **Investigate sparse attention** - KV cache is likely the real bottleneck

## Methodology Notes

- Used forward hooks on each Linear layer within LlamaMLP
- Measured timing with time.perf_counter() in pre/post hooks
- Captured activation memory by tracking input/output tensor sizes
- Profiled with batch=1, seq_len=512, 10 forward passes
