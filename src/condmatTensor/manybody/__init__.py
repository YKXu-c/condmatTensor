"""Many-body physics module for DMFT and other methods.

This module provides tools for many-body calculations including:
- Bare Green's function G₀ computation
- Self-energy Σ initialization and manipulation
- Spectral function A(ω) calculation
- Matsubara frequency generation

LEVEL 4 of the 10-level architecture.
"""

from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    BareGreensFunction,
    SelfEnergy,
    SpectralFunction,
)

__all__ = [
    "generate_matsubara_frequencies",
    "BareGreensFunction",
    "SelfEnergy",
    "SpectralFunction",
]
