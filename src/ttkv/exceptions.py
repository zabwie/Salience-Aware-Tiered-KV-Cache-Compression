"""Custom exceptions for TTKV.

This module defines a comprehensive exception hierarchy for TTKV,
providing detailed error messages for various failure modes.
"""

from typing import Optional, Set, Any
import torch


class TTKVError(Exception):
    """Base exception for TTKV.

    All TTKV-specific exceptions inherit from this class.
    """
    pass


class ValidationError(TTKVError, ValueError):
    """Input validation failed.

    Raised when input tensors, configuration, or parameters fail validation.
    Inherits from ValueError for backward compatibility.
    """

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        self.field = field
        self.value = value
        super().__init__(message)


class DeviceMismatchError(TTKVError):
    """Tensors on different devices.

    Raised when operations require tensors to be on the same device
    but they are on different devices (e.g., CPU vs CUDA).
    """

    def __init__(self, devices: Set[torch.device], message: Optional[str] = None):
        self.devices = devices
        if message is None:
            device_list = ', '.join(str(d) for d in devices)
            message = f"Tensors on different devices: {{{device_list}}}"
        super().__init__(message)


class DtypeMismatchError(TTKVError):
    """Tensors have incompatible dtypes.

    Raised when operations require tensors to have compatible dtypes
    but they have different or incompatible types.
    """

    def __init__(self, expected_dtype: torch.dtype, actual_dtype: torch.dtype,
                 tensor_name: Optional[str] = None):
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype
        self.tensor_name = tensor_name

        name_str = f" for '{tensor_name}'" if tensor_name else ""
        message = f"Dtype mismatch{name_str}: expected {expected_dtype}, got {actual_dtype}"
        super().__init__(message)


class ShapeMismatchError(TTKVError, ValueError):
    """Tensor shapes are incompatible.

    Raised when tensor shapes don't match expected dimensions
    or are incompatible for the requested operation.
    Inherits from ValueError for backward compatibility.
    """

    def __init__(self, expected_shape: tuple, actual_shape: tuple,
                 tensor_name: Optional[str] = None):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.tensor_name = tensor_name

        name_str = f" for '{tensor_name}'" if tensor_name else ""
        message = f"Shape mismatch{name_str}: expected {expected_shape}, got {actual_shape}"
        super().__init__(message)


class DimensionError(TTKVError, ValueError):
    """Tensor has incorrect number of dimensions.

    Raised when a tensor doesn't have the expected number of dimensions.
    Inherits from ValueError for backward compatibility.
    """

    def __init__(self, expected_dims: int, actual_dims: int,
                 tensor_name: Optional[str] = None):
        self.expected_dims = expected_dims
        self.actual_dims = actual_dims
        self.tensor_name = tensor_name

        name_str = f" for '{tensor_name}'" if tensor_name else ""
        message = f"Dimension mismatch{name_str}: expected {expected_dims}D tensor, got {actual_dims}D"
        super().__init__(message)


class EmptyTensorError(TTKVError):
    """Tensor is empty or has zero size.

    Raised when an operation encounters an empty tensor where
    non-empty data is required.
    """

    def __init__(self, tensor_name: Optional[str] = None, dimension: Optional[int] = None):
        self.tensor_name = tensor_name
        self.dimension = dimension

        name_str = f" '{tensor_name}'" if tensor_name else ""
        dim_str = f" at dimension {dimension}" if dimension is not None else ""
        message = f"Empty tensor{name_str}{dim_str}"
        super().__init__(message)


class InvalidValueError(TTKVError):
    """Tensor contains invalid values.

    Raised when tensors contain NaN, Inf, or other invalid values
    that would cause numerical instability.
    """

    def __init__(self, value_type: str, tensor_name: Optional[str] = None,
                 count: Optional[int] = None):
        self.value_type = value_type
        self.tensor_name = tensor_name
        self.count = count

        name_str = f" in '{tensor_name}'" if tensor_name else ""
        count_str = f" ({count} occurrences)" if count is not None else ""
        message = f"Tensor{name_str} contains {value_type} values{count_str}"
        super().__init__(message)


class ConfigurationError(TTKVError):
    """Invalid configuration.

    Raised when CacheConfig or other configuration objects
    have invalid or incompatible settings.
    """

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        field_str = f" for field '{field}'" if field else ""
        super().__init__(f"Configuration error{field_str}: {message}")


class CacheError(TTKVError):
    """Cache operation failed.

    Raised when cache operations fail, such as adding to a full cache
    or retrieving from an empty cache.
    """
    pass


class CompressionError(TTKVError):
    """Compression operation failed.

    Raised when KV cache compression fails due to invalid inputs
    or numerical issues.
    """
    pass


__all__ = [
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
