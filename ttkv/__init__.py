"""TTKV - Three Tiered Key Value Cache Compression"""

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
    "CacheConfig",
    "SalienceScorer",
    "RetentionScheduler",
    "TieredKVCache",
    "AttentionGuidedScorer",
    "AttentionBasedKVCache",
    "AttentionGuidedWrapper",
    "extract_attention_weights",
    "MockTypePriorClassifier",
    "create_mock_retention",
    "compute_type_prior_retention",
]
