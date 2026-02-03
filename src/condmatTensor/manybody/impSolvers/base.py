"""Abstract base class for impurity solvers in DMFT.

All impurity solvers must inherit from ImpuritySolverABC and implement
the required methods. This ensures a consistent interface for the DMFT
self-consistency loop.

Supported impurity solvers:
- IPT: Iterated Perturbation Theory (second-order)
- ED: Exact Diagonalization (future)
- NRG: Numerical Renormalization Group (future)
- CTQMC: Continuous-Time Quantum Monte Carlo (future)
- CNN-CI: CNN-selected Configuration Interaction (future)

References:
    - "Dynamical mean-field theory" - Georges et al., Rev. Mod. Phys. 68, 13 (1996)
    - TRIQS 3.3.1 Documentation - DMFT tutorials
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from condmatTensor.core import BaseTensor


class ImpuritySolverABC(ABC):
    """Abstract base class for DMFT impurity solvers.

    All impurity solvers must implement the solve() method which takes
    a Weiss field G₀ and returns the self-energy Σ.

    The impurity problem in DMFT:
        G₀^(-1)(iω) = G_loc^(-1)(iω) + Σ(iω)
        Σ_new(iω) = 𝒢[G₀]

    where 𝒢 represents the impurity solver mapping.
    """

    @abstractmethod
    def solve(
        self,
        G_input: "BaseTensor",
        **kwargs,
    ) -> "BaseTensor":
        """Solve the impurity problem.

        Given the Weiss field G₀, compute the self-energy Σ.

        **Flexible API Design:**
        Different solvers may need different inputs:
        - IPT: Accepts G(k,iω), extracts G_loc internally
        - ED: May need G_loc directly
        - NRG: May need hybridization function

        The solver should handle input flexibility internally.

        Args:
            G_input: Green's function from DMFT. Can be:
                - G(k,iω): k-dependent, labels=['iwn', 'k', 'orb_i', 'orb_j']
                - G_loc(iω): local only, labels=['iwn', 'orb_i', 'orb_j']
            **kwargs: Solver-specific parameters (e.g., U values, iteration params)

        Returns:
            BaseTensor with self-energy Σ(iωₙ), labels=['iwn', 'orb_i', 'orb_j']
            Shape: (n_omega, n_orb, n_orb)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Impurity solvers must implement solve()")

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """Return the name of this solver.

        Used for logging and output identification.

        Returns:
            Solver name string (e.g., 'IPT', 'ED', 'NRG', 'CTQMC')
        """
        raise NotImplementedError("Impurity solvers must implement solver_name")

    @property
    @abstractmethod
    def supported_orbitals(self) -> int:
        """Return maximum number of orbitals supported.

        Returns:
            -1 for unlimited (supports any number of orbitals)
            0 for single-orbital only
            n for multi-orbital up to n orbitals
        """
        raise NotImplementedError("Impurity solvers must implement supported_orbitals")
