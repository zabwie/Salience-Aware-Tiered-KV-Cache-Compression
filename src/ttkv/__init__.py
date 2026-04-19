"""
TTKV - Three Tiered Key Value Cache Compression

A PyTorch library for efficient KV cache compression in transformer models.
Implements three-tier progressive compression with structural survival floor
to prevent loss of critical tokens.

Key Features:
- Three-tier compression: uncompressed → 4:1 → 16:1
- Structural survival floor for rare-but-critical tokens
- Attention-guided salience scoring
- Compatible with HuggingFace Transformers

Example:
    >>> from ttkv import TieredKVCache, CacheConfig
    >>> config = CacheConfig(tier0_size=256, tau_threshold=0.9)
    >>> cache = TieredKVCache(config)
"""

__version__ = "0.1.0"
__author__ = "Zabdiel Pérez Muñiz"
__license__ = "MIT"

from .core import (
    CacheConfig,
    SalienceScorer,
    RetentionScheduler,
    TieredKVCache,
)

from .attention_scorer import (
    AttentionGuidedScorer,
    AttentionBasedKVCache,
    AttentionGuidedWrapper,
    extract_attention_weights,
)

from .type_prior import (
    MockTypePriorClassifier,
    create_mock_retention,
    compute_type_prior_retention,
)

__all__ = [
    # Core components
    "CacheConfig",
    "SalienceScorer",
    "RetentionScheduler",
    "TieredKVCache",
    # Attention-guided components
    "AttentionGuidedScorer",
    "AttentionBasedKVCache",
    "AttentionGuidedWrapper",
    "extract_attention_weights",
    # Type prior components
    "MockTypePriorClassifier",
    "create_mock_retention",
    "compute_type_prior_retention",
]