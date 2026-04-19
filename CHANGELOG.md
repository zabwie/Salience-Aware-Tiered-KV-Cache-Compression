# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-19

### Changed
- **BREAKING**: Released as stable production version (was 0.1.0-beta)
- Fixed GitHub repository URLs in `pyproject.toml` (was incorrectly pointing to 3tkv)
- Enhanced type hints throughout the codebase:
  - `core.py`: Added comprehensive type annotations to all functions and classes
  - `attention_scorer.py`: Full type coverage including complex tensor operations
  - `type_prior.py`: Complete type annotations for classification functions

### Added
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing
  - Runs on Python 3.9, 3.10, 3.11, and 3.12
  - Unit tests, integration tests, and benchmarks
  - Coverage reporting with Codecov integration
- **Type Checking**: mypy configuration in `pyproject.toml`
  - Strict mode for core functionality
  - Proper overrides for test modules
- **Integration Tests**: New test suite in `tests/integration/`
  - End-to-end model integration tests
  - Multi-batch compression tests
  - Memory efficiency tests at 32K sequences
  - Edge case handling (empty cache, clearing, etc.)
- **Documentation**: 
  - `CONTRIBUTING.md` with development guidelines
  - `CHANGELOG.md` with version history

### Fixed
- Fixed incorrect GitHub URLs in package metadata (3tkv → ttkv)
- All source files now have complete type annotations
- Added py.typed marker for PEP 561 compliance

## [0.1.0] - 2024-04-18

### Added
- Initial beta release
- Three-tier KV cache compression (0: uncompressed, 1: 4:1, 2: 16:1)
- Attention-guided salience scoring
- Type-based retention priors
- Support for PyTorch 2.0+ and transformers 4.30+
- Python 3.9+ compatibility
- Comprehensive benchmark suite
- Academic paper with ablation studies
- Reproducible experiments

### Features
- **TieredKVCache**: Core three-tier compression with structural survival floor
- **AttentionGuidedScorer**: EMA-based attention tracking
- **MockTypePriorClassifier**: Rule-based token classification
- **Compression Ratios**: Up to 16:1 on distant tokens
- **Quality**: <0.2% perplexity increase at 7.36x compression on GPT-2

## Future Roadmap

### [1.1.0] - Planned
- Additional compression algorithms (learned pooling, vector quantization)
- Support for more model architectures (LLaMA, Mistral, etc.)
- CUDA kernels for faster compression
- Streaming KV cache support

### [1.2.0] - Planned
- KV cache quantization (8-bit, 4-bit)
- Dynamic compression ratio selection
- Memory profiling tools
- Integration with HuggingFace PEFT

## Security

- No known security issues in current release
- All dependencies are pinned to minimum secure versions
- Regular security audits via GitHub Dependabot

## Compatibility

| Version | Python | PyTorch | Transformers | Status |
|---------|--------|---------|------------|--------|
| 1.0.0   | 3.9-3.12 | 2.0+ | 4.30+ | ✅ Stable |
| 0.1.0   | 3.9-3.12 | 2.0+ | 4.30+ | ⚠️ Beta |
