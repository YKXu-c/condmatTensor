"""Many-body physics module for DMFT and other methods.

This module provides tools for many-body calculations including:
- Bare Green's function G₀ computation
- Self-energy Σ initialization and manipulation
- Spectral function A(ω) calculation
- Matsubara frequency generation
- Local magnetic models: Kondo lattice, spin-fermion, H = H₀ + J@S

LEVEL 4 of the 10-level architecture.
"""

from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    BareGreensFunction,
    SelfEnergy,
    SpectralFunction,
)

from condmatTensor.manybody.magnetic import (
    LocalMagneticModel,
    KondoLatticeSolver,
    SpinFermionModel,
    pauli_matrices,
)

__all__ = [
    "generate_matsubara_frequencies",
    "BareGreensFunction",
    "SelfEnergy",
    "SpectralFunction",
    "LocalMagneticModel",
    "KondoLatticeSolver",
    "SpinFermionModel",
    "pauli_matrices",
]
