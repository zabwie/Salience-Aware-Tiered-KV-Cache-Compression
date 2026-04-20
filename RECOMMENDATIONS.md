# TTKV Development Recommendations

This document outlines recommended improvements for production deployment and future enhancements.

## Immediate Improvements (Non-blocking)

### 1. Add Trained Scorer Weights
**Priority: High**

**Current State:** The `SalienceScorer` uses randomly initialized weights.

**Recommendation:** Provide pre-trained scorer weights aligned with the paper's ablation studies.

```python
# Load pre-trained weights
scorer = SalienceScorer(hidden_dim=768, salience_hidden=256)
scorer.load_state_dict(torch.load('ttkv_scorer_gpt2.pt'))
```

**Benefits:**
- Better retention of named entities and numeric values
- Higher compression ratios with maintained quality
- Consistent performance across different domains

**Implementation:**
1. Train scorer on diverse corpora with labeled named entities
2. Export weights for GPT-2 and other supported models
3. Include weights as package data

### 2. Vectorize Compression Loops
**Priority: Medium**

**Current State:** The `_compress` method uses Python loops over batches and chunks.

**Recommendation:** Replace nested loops with vectorized PyTorch operations.

```python
# Current approach (loops)
for b in range(batch_size):
    for i in range(compressed_len):
        # ... compression logic

# Recommended approach (vectorized)
# Use scatter/gather or tensor operations
```

**Benefits:**
- 5-10x speedup on large batches
- Better GPU utilization
- Reduced Python overhead

**Benchmarks to Measure:**
- Compression time for 16K tokens: Target <10ms
- Memory overhead: Keep minimal

### 3. Add Input Validation
**Priority: Medium**

**Current State:** Limited validation of input shapes.

**Recommendation:** Add explicit shape checking with helpful error messages.

```python
def add(self, k: torch.Tensor, v: torch.Tensor,
        retention: torch.Tensor, positions: torch.Tensor) -> None:
    # Validate shapes
    if k.shape != v.shape:
        raise ValueError(f"Keys {k.shape} and values {v.shape} must match")
    if retention.shape[0] != k.shape[0]:
        raise ValueError(f"Batch size mismatch: k={k.shape[0]}, retention={retention.shape[0]}")
    if retention.shape[1] != k.shape[2]:
        raise ValueError(f"Sequence length mismatch: k={k.shape[2]}, retention={retention.shape[1]}")
```

**Benefits:**
- Fail fast with clear error messages
- Easier debugging for users
- Prevents silent corruption

---

## Future Improvements

### 1. GPU Optimization
**Priority: High**

**Goal:** Optimize CUDA kernels for production deployment.

**Approaches:**
- **Custom CUDA kernels:** For weighted pooling in `_compress`
- **Kernel fusion:** Combine retention score calculation with compression
- **FlashAttention integration:** Leverage memory-efficient attention patterns
- **Streams:** Overlap compression with generation

**Target Performance:**
- Compression overhead: <5% of inference time
- Support for 32K+ contexts on consumer GPUs (RTX 3060 12GB)

### 2. Quantization Support
**Priority: High**

**Goal:** Add 4-bit and 8-bit KV cache compression.

**Benefits:**
- 4x additional memory reduction
- Minimal quality loss with proper scaling
- Compatible with existing compression tiers

**Implementation:**
```python
class QuantizedKVCache:
    def __init__(self, config: CacheConfig, quantization: str = 'int8'):
        self.quantization = quantization  # 'int8', 'int4', or 'fp16'
    
    def add(self, k: torch.Tensor, v: torch.Tensor, ...):
        # Quantize before storing
        k_quant = self._quantize(k)
        v_quant = self._quantize(v)
        # ... rest of compression
```

### 3. Streaming Compression
**Priority: Medium**

**Goal:** Real-time adaptive compression during generation.

**Current State:** Compression happens at `get_compressed_cache()` calls.

**Recommended State:** Background compression with adaptive thresholds.

```python
class StreamingKVCache(TieredKVCache):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.compression_threshold = config.tier1_size
        self.background_compression = True
    
    def add(self, ...):
        super().add(...)
        if self.total_tokens > self.compression_threshold:
            self._compress_background()
```

**Benefits:**
- Smoother latency (no pause for compression)
- Adaptive compression ratios based on memory pressure
- Better for streaming applications

### 4. Additional Baselines
**Priority: Low**

**Goal:** Include H2O and ScissorHands implementations in package.

**Current State:** Baselines are in `tests/core/baselines.py` for testing.

**Recommended State:** Move to `ttkv.baselines` submodule.

```python
from ttkv.baselines import H2OCache, ScissorHandsCache

# Use for comparison
h2o = H2OCache(config, max_cache_size=2048)
tiered = TieredKVCache(config)

# Run comparison experiments
```

**Benefits:**
- Users can reproduce paper benchmarks
- Easy A/B testing in production
- Standardized evaluation framework

### 5. Memory Profiling Tools
**Priority: Low**

**Goal:** Built-in memory profiling for optimization.

```python
from ttkv.utils import MemoryProfiler

with MemoryProfiler() as profiler:
    cache = TieredKVCache(config)
    cache.add(k, v, retention, positions)

print(profiler.get_report())
# Shows peak memory, tier distribution, compression overhead
```

### 6. Model-Specific Optimizations
**Priority: Medium**

**Goal:** Pre-configured settings for popular models.

```python
from ttkv.presets import get_preset

# GPT-2 optimized config
config = get_preset('gpt2')

# LLaMA optimized config
config = get_preset('llama-7b')

# Custom model
config = get_preset('custom', hidden_dim=4096, num_heads=32)
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas for contributors:
1. **CUDA kernels** for compression (High impact)
2. **Quantization** research (High impact)
3. **Pre-trained scorers** for different domains (Medium impact)
4. **Streaming compression** implementation (Medium impact)
5. **Benchmarks** for new models (Low impact, high visibility)

---

## References

- [Paper](paper/main.tex) - Full technical details
- [API Reference](README.md#api-reference) - Current implementation
- [Benchmarks](tests/core/benchmark_comprehensive.py) - Performance metrics
- [Integration Tests](tests/integration/) - Usage examples
