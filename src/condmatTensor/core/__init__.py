"""Core module: BaseTensor and utilities."""

from condmatTensor.core.base import BaseTensor
from condmatTensor.core.device import get_device, get_default_device, is_cuda_available
from condmatTensor.core.types import OrbitalMetadata

__all__ = ["BaseTensor", "get_device", "get_default_device", "is_cuda_available", "OrbitalMetadata"]
