# Technical Assessment: llama.cpp KV Cache API & Custom Compression Integration

## Executive Summary

**Feasibility: MEDIUM-HIGH Complexity**

Integrating custom KV cache compression (like the existing Python tiered compression logic) into llama.cpp is technically possible but requires significant engineering effort. The current API provides limited direct access to KV cache entries, and implementing tiered compression with retention scores would require modifications to llama.cpp's core KV cache management system.

---

## 1. KV Cache Read/Write API Exposure

### Current State: LIMITED DIRECT ACCESS

llama.cpp does **NOT** expose direct read/write access to individual KV cache entries through its public C API. The KV cache is managed internally through the `llama_kv_cache` class hierarchy.

### Available Public API Functions:

```c
// Clear operations
LLAMA_API void llama_kv_self_clear(struct llama_context * ctx);

// Sequence manipulation
LLAMA_API bool llama_kv_self_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
LLAMA_API void llama_kv_self_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
LLAMA_API void llama_kv_self_seq_keep(llama_context * ctx, llama_seq_id seq_id);
LLAMA_API void llama_kv_self_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
LLAMA_API void llama_kv_self_seq_div(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);

// Update/apply pending operations
LLAMA_API void llama_kv_self_update(struct llama_context * ctx);

// State persistence (save/load)
LLAMA_API bool llama_state_seq_save_file(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const char * tokens, size_t n_token);
LLAMA_API bool llama_state_seq_load_file(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, size_t * n_token_out, llama_token * tokens_out, size_t n_token_max, size_t * n_loaded_tokens_out);
```

### Key Limitations:

1. **No direct tensor access**: Cannot read/write individual K/V tensor values
2. **No cell-level API**: Cannot access individual KV cells directly
3. **View API removed**: The `llama_kv_cache_view` debugging API was deprecated and removed in PR #13653 (May 2025)
4. **Internal data structures**: KV cache data is stored in backend-specific buffers (CPU, CUDA, Metal) with opaque access

### Internal Architecture:

The KV cache is implemented in:
- `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp` - Main cache implementation
- `src/llama-kv-cells.h` - Cell metadata management
- Uses `ggml_tensor` for actual K/V storage with backend-specific buffers

```cpp
class llama_kv_cache : public llama_memory_i {
    // Internal storage - not directly accessible
    std::vector<kv_layer> layers;
    std::vector<llama_kv_cells> v_cells;
    // ...
};
```

---

## 2. KV Cache Modification Between Inference Steps

### Current Capabilities: CONSTRAINED BUT POSSIBLE

llama.cpp allows certain modifications between inference steps, but with significant constraints:

### Supported Operations:

| Operation | Function | Use Case |
|-----------|----------|----------|
| Remove tokens | `llama_kv_self_seq_rm()` | Eviction, sliding window |
| Copy sequences | `llama_kv_self_seq_cp()` | Forking sequences |
| Keep only one sequence | `llama_kv_self_seq_keep()` | Sequence isolation |
| Shift positions | `llama_kv_self_seq_add()` | Position adjustment |
| Divide positions | `llama_kv_self_seq_div()` | Windowed attention |
| Clear all | `llama_kv_self_clear()` | Full reset |

### Workflow for Modifications:

```cpp
// 1. Perform modifications
llama_kv_self_seq_rm(ctx, seq_id, p0, p1);  // Remove tokens in range
llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta);  // Shift positions

// 2. Apply updates (includes defragmentation if needed)
llama_kv_self_update(ctx);

// 3. Continue with next inference step
llama_decode(ctx, batch);
```

### Critical Limitations for Custom Compression:

1. **No value modification**: Cannot modify the actual K/V tensor values
2. **No selective compression**: Cannot compress specific tokens while keeping others uncompressed
3. **No retention score support**: No built-in mechanism for salience-based retention
4. **All-or-nothing quantization**: Current quantization applies uniformly to entire cache

---

## 3. Custom KV Cache Compression Integration Complexity

### Assessment: COMPLEX (Requires Core Modifications)

### Current Built-in Support:

llama.cpp already supports various KV cache quantization types:

```cpp
// Supported cache types (as of latest)
// f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
// Recent additions: turbo3_0, turbo4_0 (TurboQuant)
```

Configuration via CLI:
```bash
./llama-server -m model.gguf --cache-type-k q4_0 --cache-type-v q4_0
```

### Integration Requirements for Tiered Compression:

To implement the existing Python tiered compression logic (full retention, 4:1, 16:1 compression based on retention scores), the following modifications would be needed:

#### Phase 1: GGML Type System Extension

**Files to modify:**
- `ggml/include/ggml.h` - Add new GGML types for compressed representations
- `ggml/src/ggml.c` - Implement quantize/dequantize operations
- `ggml/src/ggml-backend.c` - Backend support

**Required additions:**
```cpp
enum ggml_type {
    // ... existing types ...
    GGML_TYPE_KV_TIER1,  // 4:1 compression
    GGML_TYPE_KV_TIER2,  // 16:1 compression
};
```

#### Phase 2: KV Cache Structure Modifications

**Files to modify:**
- `src/llama-kv-cells.h` - Add retention score tracking
- `src/llama-kv-cache.h` - Add tiered storage support
- `src/llama-kv-cache.cpp` - Implement tiered compression logic

**Required additions:**
```cpp
class llama_kv_cells {
    // Add retention scores per cell
    std::vector<float> retention_scores;
    
    // Add compression tier tracking
    std::vector<uint8_t> compression_tier;
};
```

#### Phase 3: Compression/Decompression Logic

**New files needed:**
- `src/llama-kv-compress.h` - Compression interface
- `src/llama-kv-compress.cpp` - Implementation

**Core functionality:**
- Weighted pooling based on retention scores
- Multi-tier storage management
- Decompression for attention computation

#### Phase 4: Attention Graph Modifications

**Files to modify:**
- `src/llama-graph.cpp` - Modify attention computation to handle compressed KV
- `src/llama-context.cpp` - Integration with context management

**Challenge:** Attention mechanism must transparently handle mixed compressed/uncompressed KV entries

#### Phase 5: GPU Kernel Development (Optional but Recommended)

**New files needed:**
- `ggml/src/ggml-cuda/ggml-cuda-kv-compress.cu` - CUDA kernels
- `ggml/src/ggml-metal/ggml-metal-kv-compress.m` - Metal kernels

### Complexity Factors:

| Aspect | Complexity | Notes |
|--------|------------|-------|
| CPU implementation | Medium | Straightforward C++ implementation |
| CUDA kernels | High | Requires GPU optimization expertise |
| Metal kernels | High | macOS/iOS support |
| Attention integration | High | Must maintain correctness |
| State serialization | Medium | Save/load compressed state |
| Multi-sequence support | High | Complex with compression |

---

## 4. Python/PyTorch to C++ Porting Requirements

### Assessment: SIGNIFICANT EFFORT REQUIRED

### Existing Python Implementation Analysis:

From `salience_cache.py`:

```python
class TieredKVCache:
    def __init__(self, config: CacheConfig):
        self.k_cache = []  # List of tensors
        self.v_cache = []  # List of tensors
        self.retention_scores = []  # List of tensors
        
    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tiered compression logic:
        # - Tier 0: Full retention (recent tokens)
        # - Tier 1: 4:1 compression
        # - Tier 2: 16:1 compression
        # Uses retention scores for weighted pooling
```

### Porting Requirements:

#### 1. Neural Network Components (SalienceScorer, RetentionScheduler)

**Current Python:**
```python
class SalienceScorer(nn.Module):
    def __init__(self, hidden_dim: int = 768, salience_hidden: int = 256):
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, salience_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(salience_hidden, salience_hidden // 2),
            nn.ReLU(),
            nn.Linear(salience_hidden // 2, 1)
        )
```

**C++ Implementation:**
- Use GGML for tensor operations
- Implement forward pass manually
- ~200-300 lines of C++ code
- Requires GGML graph construction

#### 2. Tensor Operations

**Python (PyTorch):**
```python
k_all = torch.cat(self.k_cache, dim=2)
weights = F.softmax(ret_chunk, dim=0).unsqueeze(0).unsqueeze(-1)
k_pooled = (k_chunk * weights).sum(dim=1)
```

**C++ (GGML):**
```cpp
// Equivalent GGML operations
ggml_tensor * k_all = ggml_concat(ctx, k_cache_list, 2);
ggml_tensor * weights = ggml_softmax(ctx, ret_chunk);
ggml_tensor * k_pooled = ggml_sum_rows(ctx, ggml_mul(ctx, k_chunk, weights));
```

#### 3. Compression Algorithm

**Key operations to port:**
- Weighted pooling with softmax weights
- Chunk-based compression (4:1, 16:1 ratios)
- Position tracking through compression
- Batch handling

#### 4. Integration Points

**Files requiring modification:**
1. `src/llama-context.cpp` - Hook into decode loop
2. `src/llama-kv-cache.cpp` - Add compression triggers
3. `src/llama-graph.cpp` - Handle compressed KV in attention
4. `include/llama.h` - Public API extensions

### Estimated Porting Effort:

| Component | Lines of Code | Complexity | Time Estimate |
|-----------|---------------|------------|---------------|
| Core compression logic | ~500 | Medium | 1-2 weeks |
| GGML tensor operations | ~300 | Medium | 1 week |
| KV cache integration | ~400 | High | 2-3 weeks |
| Attention graph mods | ~300 | High | 2 weeks |
| CUDA kernels | ~800 | Very High | 3-4 weeks |
| Testing & debugging | - | - | 2-3 weeks |
| **Total** | **~2300** | | **3-4 months** |

---

## 5. Alternative Approaches

### Option 1: External Compression (Recommended for Prototyping)

**Approach:** Keep compression logic in Python, use llama.cpp for inference only

**Workflow:**
1. Run inference step in llama.cpp
2. Export KV cache to Python (via state save/load)
3. Apply compression in Python
4. Import compressed cache back to llama.cpp

**Pros:**
- Faster development
- No modifications to llama.cpp
- Easier experimentation

**Cons:**
- High overhead from serialization
- Not suitable for production
- Complex synchronization

### Option 2: Hybrid Approach

**Approach:** Implement compression as a llama.cpp "adapter" or "plugin"

**Implementation:**
- Create wrapper around llama.cpp context
- Intercept KV cache updates
- Apply compression before storage

**Pros:**
- Cleaner separation of concerns
- Easier to maintain
- Can be optional feature

**Cons:**
- Still requires core API extensions
- May have performance overhead

### Option 3: Full Native Integration (Recommended for Production)

**Approach:** Full implementation within llama.cpp codebase

**Implementation:**
- Add compression types to GGML
- Modify KV cache for tiered storage
- Implement GPU kernels
- Full test coverage

**Pros:**
- Best performance
- Clean integration
- Production-ready

**Cons:**
- Significant development effort
- Requires C++/CUDA expertise
- Ongoing maintenance burden

---

## 6. Recommendations

### Short Term (1-2 months):

1. **Prototype with external approach**
   - Validate compression algorithm effectiveness
   - Measure overhead of Python/C++ boundary crossing
   - Determine if native integration is justified

2. **Design native API**
   - Define clean interface for compression
   - Plan GGML type extensions
   - Design GPU kernel architecture

### Medium Term (3-6 months):

1. **Implement CPU-only version**
   - Port compression logic to C++
   - Integrate with KV cache
   - Add comprehensive tests

2. **Performance optimization**
   - Profile and optimize hot paths
   - Implement batch processing
   - Memory layout optimization

### Long Term (6+ months):

1. **GPU kernel development**
   - CUDA implementation
   - Metal implementation (Apple Silicon)
   - Performance tuning

2. **Production hardening**
   - Edge case handling
   - Error recovery
   - Documentation

---

## 7. Conclusion

### Feasibility: ✅ POSSIBLE

Integrating custom KV cache compression into llama.cpp is technically feasible but requires substantial engineering effort:

- **Minimum viable product:** 2-3 months (CPU only)
- **Production-ready implementation:** 4-6 months (with GPU support)
- **Maintenance overhead:** Ongoing (tracking llama.cpp updates)

### Key Challenges:

1. **API limitations** - No direct KV cache access; requires core modifications
2. **Multi-backend support** - Must support CPU, CUDA, Metal
3. **Attention integration** - Complex to handle mixed compressed/uncompressed KV
4. **State management** - Save/load compressed state

### Success Factors:

1. Strong C++ and CUDA/Metal expertise
2. Deep understanding of llama.cpp internals
3. Comprehensive testing strategy
4. Clear performance benchmarks

### Final Recommendation:

**Proceed with native integration** if:
- Compression provides significant memory savings (>50%)
- Performance overhead is acceptable (<10% slowdown)
- Long-term maintenance commitment exists

**Use external approach** if:
- Quick prototyping needed
- Compression effectiveness is unproven
- Limited development resources

---

## References

1. llama.cpp KV cache source: https://github.com/ggml-org/llama.cpp/tree/master/src
2. llama.cpp public API: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
3. GGML tensor operations: https://github.com/ggml-org/ggml
4. TurboQuant implementation: https://github.com/ggml-org/llama.cpp/pull/21307
5. KV cache refactoring PRs: #13660, #13706, #13653
