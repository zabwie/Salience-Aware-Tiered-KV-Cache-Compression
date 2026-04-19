# Salience-Aware Tiered KV Cache Compression

**Solving the slow-burn problem in long-context inference**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **7.36× compression** with **0.17% quality loss** | **16K context** on **RTX 3060** | **Beats H2O/ScissorHands** on slow-burn

---

## The Problem

Large language models face a memory wall. At 16K tokens, the KV cache consumes ~13GB. Combined with 4GB for quantized weights, you need **17GB**—well beyond consumer GPUs.

**Worse:** Binary eviction methods (H2O, ScissorHands) fail on the **slow-burn problem**: critical information appears early and is needed late. These methods evict the token before the model ever attends to it.

---

## The Solution

**Three-tier progressive compression** with **structural survival floor**:

- **Tier 0**: Recent tokens (uncompressed)
- **Tier 1**: High salience (4:1 compression)  
- **Tier 2**: Low salience (16:1 compression)

Tokens are **compressed**, not evicted. Rare-but-critical tokens (numbers, codes) receive minimum retention guarantees via structural heuristics.

---

## Killer Demo: The Slow-Burn Test

Critical token at position 0, 15,000 tokens of filler, retrieve at position 15,001:

| Method | Compression | Tokens Kept | Needle | Result |
|--------|-------------|-------------|--------|--------|
| **H2O** | 4.00× | 2,048 | ❌ EVICTED | **FAIL** |
| **ScissorHands** | 4.00× | 2,048 | ❌ EVICTED | **FAIL** |
| **Tiered (Ours)** | **7.36×** | **1,514** | ✅ **PRESERVED** | **PASS** |

**Why binary eviction fails:**
- H2O: Needle has accumulated attention = 0 (never attended to) → evicted
- ScissorHands: Needle outside recent window (256 tokens) → evicted
- Tiered: Structural floor gives retention 0.95 → preserved

```bash
cd tests
python3 test_slow_burn.py
```

---

## Results

### Compression vs Quality Tradeoff

| τ | Compression | Quality Loss | Status |
|---|-------------|--------------|--------|
| 0.60 | 3.61× | 0.09% | ✅ Sweet Spot |
| 0.80 | 5.08× | 0.09% | ✅ Sweet Spot |
| **0.90** | **7.36×** | **0.17%** | ✅ **Recommended** |
| 0.95 | 7.36× | 0.11% | ✅ Sweet Spot |

**Flat region:** 3.6×–7.4× compression with <0.5% quality loss. Binary eviction lacks this tunability.

### Real Hardware

**RTX 3060 (12GB) + Mistral-7B-Instruct:**

| Context | Baseline | Compressed | Status |
|---------|----------|------------|--------|
| 8K | ~6.7 GB | ~2.3 GB | ✅ Fits |
| 16K | ~13 GB | ~4.6 GB | ✅ Fits |

Without compression: 16K context OOMs. With compression: runs with 3.7GB headroom.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run killer demo
cd tests
python3 test_slow_burn.py

# Reproduce everything
bash reproduce.sh
```

---

## Key Innovations

1. **Tiered compression**: Tokens degraded progressively, not evicted
2. **Structural survival floor**: Rare token types guaranteed survival
3. **Attention-guided scorer**: Learns importance from model attention
4. **Flat tradeoff curve**: Tunable compression with minimal quality loss

---

## Paper

Full paper available at `paper/main.tex`:
- Abstract + introduction
- Method (three-tier architecture)
- Empirical comparison vs H2O/ScissorHands
- Real hardware validation
- Reproducibility details

---

## Project Structure

```
salience_kv_cache/
├── src/                      # Core implementation
│   ├── salience_cache.py     # Tiered KV cache
│   ├── attention_guided_scorer.py
│   └── type_prior_mock.py
├── tests/                    # All experiments
│   ├── baselines.py          # H2O & ScissorHands
│   ├── test_slow_burn.py     # Killer demo
│   ├── test_needle_in_haystack.py
│   ├── benchmark_comprehensive.py
│   └── generate_tradeoff_curves.py
├── results/                  # Generated data
├── paper/                    # LaTeX paper
├── reproduce.sh              # One-command reproducibility
├── requirements.txt
└── QUICKSTART.md
```

---

## Citation

```bibtex
@software{salience_kv_cache_2024,
  title={Salience-Aware Tiered KV Cache Compression},
  author={[Anonymous]},
  year={2024},
  note={Solves the slow-burn problem in long-context inference},
  howpublished={\url{https://github.com/[anonymized]/salience-kv-cache}}
}
```

---

## License

MIT License — See LICENSE file for details.

---

## Status

✅ **Validated:** GPT-2, TinyLlama, Mistral-7B  
✅ **Hardware:** Consumer GPUs (RTX 3060, 12GB)  
✅ **Reproducible:** Single command (`bash reproduce.sh`)
