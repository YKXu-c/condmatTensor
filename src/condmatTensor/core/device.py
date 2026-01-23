"""Device management utilities for GPU acceleration.

This module provides utilities for managing device selection (CPU/GPU) across
the condmatTensor library. It allows users to easily enable GPU acceleration
where available while maintaining CPU as the default.

LEVEL 1 utility module.
"""

from typing import Optional, Union

import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get device with automatic CUDA detection and CPU fallback.

    This is the primary utility for device selection across the library.
    When CUDA is requested but not available, it falls back to CPU with a warning.

    Args:
        device: Device specification. Can be:
            - None (default): Returns CPU device
            - 'cuda' or 'cpu': String device specification
            - torch.device: Direct torch.device object

    Returns:
        torch.device object, either 'cuda' or 'cpu'

    Examples:
        >>> from condmatTensor.core import get_device
        >>> # Get CPU device (default)
        >>> device = get_device()
        >>> # Get CUDA device if available
        >>> device = get_device("cuda")
        >>> # Get device with automatic detection
        >>> device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    """
    if device is None:
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        return torch.device("cpu")

    if device not in ("cuda", "cpu"):
        raise ValueError(f"Invalid device: {device}. Use 'cuda', 'cpu', or torch.device")

    return torch.device(device)


def get_default_device() -> torch.device:
    """Get the default device (CPU for this library).

    The condmatTensor library uses CPU as the default to ensure consistent
    behavior across all environments. GPU acceleration is opt-in via the
    device parameter in relevant functions.

    Returns:
        torch.device('cpu')

    Examples:
        >>> from condmatTensor.core import get_default_device
        >>> device = get_default_device()
        >>> assert device.type == "cpu"
    """
    return torch.device("cpu")


def is_cuda_available() -> bool:
    """Check if CUDA is available on the system.

    Returns:
        True if CUDA is available, False otherwise

    Examples:
        >>> from condmatTensor.core import is_cuda_available
        >>> if is_cuda_available():
        ...     print("GPU acceleration available")
    """
    return torch.cuda.is_available()


__all__ = ["get_device", "get_default_device", "is_cuda_available"]
