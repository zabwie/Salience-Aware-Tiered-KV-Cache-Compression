#!/bin/bash
# Reproducibility script for Salience KV Cache
# Run this to reproduce all paper results

set -e  # Exit on error

echo "=========================================="
echo "Salience KV Cache - Reproducibility Script"
echo "=========================================="
echo ""

# Check Python version
python3 --version || (echo "Python 3 required" && exit 1)

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Run all tests
cd tests

echo ""
echo "=========================================="
echo "1. Baseline Comparison"
echo "=========================================="
python3 baselines.py

echo ""
echo "=========================================="
echo "2. Slow Burn Test (Killer Demo)"
echo "=========================================="
python3 test_slow_burn.py

echo ""
echo "=========================================="
echo "3. Needle-in-Haystack Test"
echo "=========================================="
python3 test_needle_in_haystack.py

echo ""
echo "=========================================="
echo "4. Comprehensive Benchmark"
echo "=========================================="
python3 benchmark_comprehensive.py

echo ""
echo "=========================================="
echo "5. Generate Tradeoff Curves"
echo "=========================================="
python3 generate_tradeoff_curves.py

echo ""
echo "=========================================="
echo "Reproducibility Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/comprehensive_benchmark.json"
echo "  - results/tradeoff_data_8192.json"
echo "  - results/plot_tradeoff_8192.py"
echo ""
echo "Paper available at: paper/main.tex"
