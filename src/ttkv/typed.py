"""Runtime type checking with beartype for TTKV.

This module provides optional runtime type checking using beartype
for enhanced debugging and development. Falls back to no-op if beartype
is not installed.
"""

from typing import Optional, TYPE_CHECKING, Callable, Any
import functools
import warnings

# Try to import beartype, but don't fail if it's not available
try:
    from beartype import beartype
    from beartype.typing import (
        Tuple as BTuple,
        Dict as BDict,
        List as BList,
        Optional as BOptional,
    )
    from jaxtyping import Float, Int, jaxtyped
    BEARTYPE_AVAILABLE = True
except ImportError:
    BEARTYPE_AVAILABLE = False
    beartype = None


# Type alias for type hints that work with or without beartype
if BEARTYPE_AVAILABLE:
    # Beartype-enhanced types with shape annotations
    Tensor4D = Float[torch.Tensor, "batch heads seq head_dim"]
    Tensor2D = Float[torch.Tensor, "batch seq"]
    IntTensor2D = Int[torch.Tensor, "batch seq"]
else:
    # Fallback to standard types
    from typing import TypeVar
    Tensor4D = TypeVar('Tensor4D', bound='torch.Tensor')
    Tensor2D = TypeVar('Tensor2D', bound='torch.Tensor')
    IntTensor2D = TypeVar('IntTensor2D', bound='torch.Tensor')


def _noop_decorator(func: Callable) -> Callable:
    """No-op decorator for when beartype is not available."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper


def maybe_beartype(func: Callable) -> Callable:
    """Apply beartype decorator if available, otherwise no-op.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with runtime type checking if beartype is available
    """
    if BEARTYPE_AVAILABLE and beartype is not None:
        try:
            return beartype(func)
        except Exception as e:
            warnings.warn(f"Failed to apply beartype: {e}")
            return func
    return func


def check_beartype_available() -> bool:
    """Check if beartype is available for runtime type checking.

    Returns:
        True if beartype is installed and functional
    """
    return BEARTYPE_AVAILABLE


class TypeCheckingDisabled:
    """Context manager to temporarily disable beartype checking."""

    def __init__(self) -> None:
        self._original_beartype = None

    def __enter__(self) -> 'TypeCheckingDisabled':
        """Enter context manager, disabling type checking."""
        if BEARTYPE_AVAILABLE:
            global beartype
            self._original_beartype = beartype
            beartype = lambda f: f  # type: ignore
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, restoring type checking."""
        if BEARTYPE_AVAILABLE and self._original_beartype is not None:
            global beartype
            beartype = self._original_beartype


def warn_if_no_beartype() -> None:
    """Warn user if beartype is not installed."""
    if not BEARTYPE_AVAILABLE:
        warnings.warn(
            "beartype is not installed. Install with: pip install beartype jaxtyping\n"
            "For development, runtime type checking is highly recommended.",
            UserWarning,
            stacklevel=2
        )