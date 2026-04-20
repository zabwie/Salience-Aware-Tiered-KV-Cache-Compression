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

from .ner_classifier import (
    NERClassifier,
    TokenClassifier,
)

from .trainer import (
    AttentionDataset,
    SalienceTrainer,
    train_on_gpt2,
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
    "CacheConfig",
    "CacheStats",
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
    "NERClassifier",
    "TokenClassifier",
    "AttentionDataset",
    "SalienceTrainer",
    "train_on_gpt2",
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
