"""Many-body physics module for DMFT and other methods.

This module provides tools for many-body calculations including:
- Bare Green's function G₀ computation
- Self-energy Σ initialization and manipulation
- Spectral function A(ω) calculation
- Matsubara frequency generation
- Analytic continuation methods (Pade, Bethe lattice, MaxEnt)
- Local magnetic models: Kondo lattice, spin-fermion, H = H₀ + J@S
- Impurity solvers for DMFT (IPT, ED, NRG, etc.)
- DMFT self-consistency loop

LEVEL 4 of the 10-level architecture.
"""

from condmatTensor.manybody.preprocessing import (
    generate_matsubara_frequencies,
    calculate_dos_range,
    BareGreensFunction,
    SelfEnergy,
    SpectralFunction,
)

from condmatTensor.manybody.analytic_continuation import (
    AnalyticContinuationMethod,
    SimpleContinuation,
    PadeContinuation,
    BetheLatticeContinuation,
    MaxEntContinuation,
    create_continuation_method,
)

from condmatTensor.manybody.magnetic import (
    LocalMagneticModel,
    KondoLatticeSolver,
    SpinFermionModel,
    pauli_matrices,
)

# Impurity solvers (ABC + implementations)
from condmatTensor.manybody.impSolvers import (
    ImpuritySolverABC,
    IPTSolver,
    # ED, NRG, CTQMC, etc. to be added as implemented
)

from condmatTensor.manybody.dmft import (
    SingleSiteDMFTLoop,
    MixingMethod,
    LinearMixing,
)

__all__ = [
    # Preprocessing
    "generate_matsubara_frequencies",
    "calculate_dos_range",
    "BareGreensFunction",
    "SelfEnergy",
    "SpectralFunction",
    # Analytic continuation
    "AnalyticContinuationMethod",
    "SimpleContinuation",
    "PadeContinuation",
    "BetheLatticeContinuation",
    "MaxEntContinuation",
    "create_continuation_method",
    # Magnetic models
    "LocalMagneticModel",
    "KondoLatticeSolver",
    "SpinFermionModel",
    "pauli_matrices",
    # Impurity solvers
    "ImpuritySolverABC",
    "IPTSolver",
    # DMFT
    "SingleSiteDMFTLoop",
    "MixingMethod",
    "LinearMixing",
]
