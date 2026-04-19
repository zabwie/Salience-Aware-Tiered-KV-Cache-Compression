# Staged Optimizations: Memory + Speed

## Philosophy
**Verify before proceeding.** Each stage must prove it works before we touch the next one. No parallel development, no "we'll debug it later." If a stage fails verification, we stop and fix it.

**Time estimates are optimistic.** Budget 3-5x for realistic planning. Surprises happen.

---

## Stage 0: Profiling (Foundation)

**Goal:** Understand where time and memory actually go before optimizing.

### Why This First
- 20 minutes of profiling saves days of optimizing the wrong thing
- Bottleneck may not be where we expect
- Changes stage ordering if profiling reveals different bottlenecks

### Implementation
```bash
python tests/profile_baseline.py --model tinyllama --seq-len 512
```

**Measure:**
- [ ] Per-layer time breakdown (attention vs FFN vs other)
- [ ] Per-head attention weight distribution
- [ ] Memory by component (weights, KV cache, activations, gradients)
- [ ] SDPA kernel behavior (fused vs unfused heads)

### Decision Gate
**Review profiling data and decide:**
- [ ] Is attention head skipping actually the bottleneck? (If FFN dominates, skip Stage 1)
- [ ] Can we skip head computation, or only mask outputs? (SDPA fusion matters)
- [ ] Is residual caching worth it vs other memory optimizations?

**If profiling shows different bottlenecks → Revise stage ordering**

---

## Stage 1: Attention Head Skipping (Faster Inference)

**Goal:** Skip attention heads with near-zero contribution at each generation step.

### Why This First
- Builds on existing `attention_guided_scorer.py`
- Lower risk than residual caching (computation only, no memory changes)
- Easier to verify (timing benchmarks are unambiguous)
### Implementation

**⚠️ Implementation Reality Check:**
Standard Transformers use fused SDPA kernels (FlashAttention, etc.) where heads are computed together. "Skipping" heads may mean:
- **Option A:** Mask head outputs to zero (easy, but doesn't skip computation)
- **Option B:** Use unfused attention path (slower baseline, but real skipping possible)
- **Option C:** Custom CUDA kernel (high effort, real speedup)

Profiling (Stage 0) determines which path is viable.

1. **Instrumentation** (10 min)
   - Add per-head attention statistics to `update_from_attention()`
   - Track which heads have <threshold contribution
   - Log without skipping first (measure baseline)

2. **Skipping Logic** (30-60 min — depends on SDPA path)
   - Add `head_threshold` parameter (default 0.05)
   - **If unfused path available:** Skip head computation entirely
   - **If fused SDPA:** Mask attention weights, accept minimal compute savings
   - Document which path taken

3. **Verification Gate** (15 min)
   ```bash
   python tests/head_skip_verify.py --model tinyllama --seq-len 512
   ```
   **Must pass ALL:**
   - [ ] Needle retrieval: 100% (regression test)
   - [ ] Perplexity delta: <2% vs baseline
   - [ ] Speedup: >10% on 512-token generation
   - [ ] Skipped heads: >20% on average

### Verification Failure = STOP
If any check fails:
- Debug the specific failure
- Adjust threshold
- Re-verify
- **DO NOT proceed to Stage 2 until this passes**

---

## Stage 2: Residual Stream Caching (Memory Savings)

**Goal:** Cache residual activations selectively based on salience scores.

### Prerequisites (MUST be true)
- [ ] Stage 1 fully verified and passing
- [ ] Head skipping working in production
- [ ] No open bugs from Stage 1
### Implementation

**⚠️ High Risk Warning:**
Residual stream is the information backbone. Compressing it mid-forward-pass requires accurate reconstruction for downstream layers. Getting `exact outputs` is harder than the time estimate suggests.

1. **Design** (30 min)
   - Decide tiered caching: full / 2:1 compressed / dropped
   - Hook into transformer forward pass
   - Save residual after each layer
   - **Critical:** Plan reconstruction path — how do downstream layers get full residuals?

2. **Core Logic** (60-120 min — higher than estimated)
   - Reuse salience scores from `AttentionGuidedScorer`
   - Tier 0: Recent tokens (full precision)
   - Tier 1: High salience (mean pool 2:1)
   - Tier 2: Low salience (drop or store sparse)
   - **Reconstruction:** Expand compressed residuals before next layer
   - **Validation:** Check reconstruction error at each layer

3. **Verification Gate** (60 min — includes debugging)
   ```bash
   python tests/residual_cache_verify.py --model tinyllama --layers 12
   ```
   **Must pass ALL:**
   - [ ] End-to-end generation matches baseline (exact outputs)
   - [ ] Memory usage: <70% of baseline at 2048 tokens
   - [ ] No quality regression on needle test
   - [ ] Speed: Not slower than baseline (caching overhead acceptable)

### Verification Failure = STOP
If any check fails:
- Debug the specific failure
- Check cache reconstruction logic
- Verify tier boundaries
- **DO NOT combine with other optimizations until this passes alone**

---

## Stage 3: Combined System (Optional)

**Only if Stages 1 AND 2 pass independently.**

### Implementation
- Enable both optimizations together
- Measure combined effect

### Verification Gate
```bash
python tests/combined_verify.py --model mistral-7b --ctx 4096
```
**Must pass:**
- [ ] Memory: Stage 2 savings maintained
- [ ] Speed: Stage 1 speedup maintained
- [ ] Quality: No worse than individual stages
- [ ] Integration: No conflicts between systems

---

## Verification Checklist (Per Stage)

| Stage | Primary Metric | Regression Test | Acceptance |
|-------|---------------|----------------|------------|
| 1 | Speedup % | Needle 100% | >10% speedup, <2% ppl |
| 2 | Memory % | Exact output match | <70% memory, no slowdown |
| 3 | Combined | All above | Maintains both gains |

---

## Anti-Patterns (Forbidden)

- ❌ Implementing Stage 2 while Stage 1 is buggy
- ❌ "We'll fix it in integration"
- ❌ Skipping verification "because it should work"
- ❌ Parallel development of stages
- ❌ Combining optimizations before individual verification

---

## Success Criteria

**Stage 1 Success:**
- Head skipping verified on TinyLlama
- Needle test passes
- >10% speedup demonstrated

**Stage 2 Success:**
- Residual caching verified independently
- Memory reduction proven
- No quality regression

**Overall Success:**
- Both systems work independently
- Combined system works
- Ready for Mistral-7B testing

---

## Current Status

- [ ] Stage 1: Not started
- [ ] Stage 2: Blocked on Stage 1
- [ ] Stage 3: Blocked on Stages 1+2

**Next Action:** Begin Stage 1 instrumentation.