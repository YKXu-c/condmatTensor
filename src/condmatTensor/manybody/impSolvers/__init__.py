"""Impurity solvers for DMFT.

This subdirectory contains all impurity solver implementations.
All solvers inherit from ImpuritySolverABC to ensure consistent interface.

Available solvers:
- IPT: Iterated Perturbation Theory (second-order)
- ED: Exact Diagonalization (future)
- NRG: Numerical Renormalization Group (future)
- CTQMC: Continuous-Time Quantum Monte Carlo (future)
- CNN_CI: CNN-selected Configuration Interaction (future)

The ABC pattern enables:
1. Polymorphism: DMFT loop works with any impurity solver
2. Type Safety: ABC ensures all solvers implement required methods
3. Extensibility: New solvers can be added without modifying DMFT loop
4. Clear Interface: Well-defined contract between DMFT loop and solvers

Usage Example:
    >>> from condmatTensor.manybody.impSolvers import IPTSolver, ImpuritySolverABC
    >>> solver = IPTSolver(beta=10.0, n_max=100)
    >>> assert isinstance(solver, ImpuritySolverABC)  # Type checking
    >>> Sigma = solver.solve(G_input)

References:
    - "Dynamical mean-field theory" - Georges et al., Rev. Mod. Phys. 68, 13 (1996)
    - TRIQS 3.3.1 Tutorial: "A first DMFT calculation"
    - Haule, Rutgers lecture notes: "Perturbation theory" (2017)
"""

from .base import ImpuritySolverABC
from .ipt import IPTSolver

__all__ = [
    "ImpuritySolverABC",
    "IPTSolver",
]
