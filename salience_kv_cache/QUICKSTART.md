# Quick Start Guide

Run the killer demo in 30 seconds:

```bash
# Clone and enter
cd salience_kv_cache

# Install dependencies
pip install -r requirements.txt

# Run the slow burn test (the killer demo)
cd tests
python3 test_slow_burn.py
```

**What you'll see:**
- H2O: FAIL (evicts critical token before attended)
- ScissorHands: FAIL (token outside attention window)
- Tiered (Ours): PASS (structural floor preserves it)

**Expected output:**
```
SLOW BURN TEST: Critical Token at Position 0, Needed at Position 15K
Method            | Kept   | Ratio  | Preserved | Result
------------------|--------|--------|-----------|--------
H2O               | 2,048  | 7.32x  | NO        | FAIL (evicted)
ScissorHands      | 2,048  | 7.32x  | NO        | FAIL (outside window)
Tiered (Ours)     | 1,514  | 9.91x  | YES       | PASS
```

## Full Reproducibility

Run everything (takes ~5 minutes):

```bash
bash reproduce.sh
```

This executes:
1. Baseline comparison
2. Slow burn test
3. Needle-in-haystack test
4. Comprehensive benchmark
5. Tradeoff curve generation

## Key Results

- **7.36× compression** at 0.17% quality loss (τ=0.9)
- **16K context** on RTX 3060 (12GB)
- **Beats H2O** by 1.8× compression with better quality
- **Killer demo** exposes fundamental flaw in binary eviction

## File Structure

```
salience_kv_cache/
├── src/                    # Core implementation
├── tests/                  # All experiments
├── results/                # Generated data
├── paper/                  # LaTeX paper
├── reproduce.sh            # One-command reproducibility
└── requirements.txt        # Dependencies
```

## Citation

```bibtex
@software{salience_kv_cache_2024,
  title={Salience-Aware Tiered KV Cache Compression},
  author={[Anonymized]},
  year={2024},
  note={Solves the slow-burn problem in long-context inference}
}
```
