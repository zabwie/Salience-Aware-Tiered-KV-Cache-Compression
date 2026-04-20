"""TTKV - Three Tiered Key Value Cache Compression"""

__version__ = "1.0.0"
__author__ = "Zabdiel Pérez Muñiz"
__license__ = "MIT"

from .core import (
    CacheConfig,
    CacheStats,
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

from .exceptions import (
    TTKVError,
    ValidationError,
    DeviceMismatchError,
    DtypeMismatchError,
    ShapeMismatchError,
    DimensionError,
    EmptyTensorError,
    InvalidValueError,
    ConfigurationError,
    CacheError,
    CompressionError,
)

__all__ = [
    # Core components
    "CacheConfig",
    "CacheStats",
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
    # Exceptions
    "TTKVError",
    "ValidationError",
    "DeviceMismatchError",
    "DtypeMismatchError",
    "ShapeMismatchError",
    "DimensionError",
    "EmptyTensorError",
    "InvalidValueError",
    "ConfigurationError",
    "CacheError",
    "CompressionError",
]
