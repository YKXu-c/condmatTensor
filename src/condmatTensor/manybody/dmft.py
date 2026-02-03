"""DMFT (Dynamical Mean Field Theory) self-consistency loop.

Implements the DMFT self-consistency condition for lattice models.
The loop alternates between:
1. Lattice problem: G(k, iω) = [iω + μ - H(k) - Σ(iω)]^(-1)
2. Impurity problem: Σ(iω) = 𝒢[G₀] where G₀^(-1) = G_loc^(-1) + Σ

Supports multi-orbital systems using OrbitalMetadata to identify
localized orbitals (e.g., f-orbitals) that receive DMFT treatment.

The impurity solver interface is polymorphic via ImpuritySolverABC,
enabling use of IPT, ED, NRG, CTQMC, etc. without code changes.

References:
    - "Dynamical mean-field theory of strongly correlated fermion systems"
      A. Georges et al., Rev. Mod. Phys. 68, 13 (1996)
    - "A First Course in Dynamical Mean-Field Theory" - Kollar
    - TRIQS 3.3.1 Documentation - DMFT tutorials
"""

from typing import Optional, Dict, Union, List
import torch
import math


class MixingMethod:
    """Base class for mixing self-energy updates.

    Different mixing strategies can improve convergence:
    - Linear mixing: Σ = (1-α)·Σ_old + α·Σ_new
    - Anderson mixing: Adaptive mixing using history
    - Bayesian mixing: Optimized mixing via Bayesian optimization
    """

    def mix(
        self,
        Sigma_old: torch.Tensor,
        Sigma_new: torch.Tensor,
        iteration: int,
        history: list,
    ) -> torch.Tensor:
        """Mix old and new self-energies.

        Args:
            Sigma_old: Previous iteration self-energy
            Sigma_new: Newly computed self-energy
            iteration: Current iteration number
            history: List of previous Σ values for adaptive methods

        Returns:
            Mixed self-energy for next iteration
        """
        raise NotImplementedError("MixingMethod subclasses must implement mix()")


class LinearMixing(MixingMethod):
    """Simple linear mixing: Σ = (1-α)·Σ_old + α·Σ_new"""

    def __init__(self, alpha: float = 0.5):
        """Initialize linear mixing.

        Args:
            alpha: Mixing parameter (0 < alpha <= 1)
                   Small alpha = more stable but slower
                   Large alpha = faster but may oscillate
        """
        self.alpha = alpha

    def mix(
        self,
        Sigma_old: torch.Tensor,
        Sigma_new: torch.Tensor,
        iteration: int,
        history: list,
    ) -> torch.Tensor:
        """Apply linear mixing.

        Σ_mixed = (1-α)·Σ_old + α·Σ_new
        """
        return (1 - self.alpha) * Sigma_old + self.alpha * Sigma_new


class SingleSiteDMFTLoop:
    """Single-site DMFT self-consistency loop.

    Solves the DMFT equations for a lattice model with local interactions.
    The self-consistency condition is:

        G_loc^(-1)(iω) = G₀^(-1)(iω) - Σ(iω)

    where G_loc is the local Green's function of the lattice, G₀ is the
    Weiss field (impurity bath), and Σ is the self-energy.

    **DMFT Algorithm (7 steps):**
        1. Start with guess Σ(iωₙ) = 0 (non-interacting)
        2. Compute lattice G(k,iω) = [iω+μ-H(k)-Σ(iω)]^(-1)
        3. Extract local G_loc(iω) = (1/N_k) Σ_k G(k,iω)
        4. Compute Weiss field: G₀^(-1) = G_loc^(-1) + Σ
        5. Solve impurity problem: Σ_new = solver.solve(G₀)
        6. Mix: Σ = (1-α)·Σ_old + α·Σ_new
        7. Check convergence: |ΔΣ|/|Σ| < tol

    Attributes:
        Hk: k-space Hamiltonian
        omega: Matsubara frequency grid
        solver: Impurity solver (MUST inherit from ImpuritySolverABC)
        mu: Chemical potential
        mixing: Mixing parameter or MixingMethod instance
        verbose: Print progress information
    """

    def __init__(
        self,
        Hk: "BaseTensor",
        omega: torch.Tensor,
        solver: "ImpuritySolverABC",
        mu: float = 0.0,
        mixing: Union[float, MixingMethod] = 0.5,
        verbose: bool = True,
    ) -> None:
        """Initialize DMFT loop.

        Args:
            Hk: k-space Hamiltonian with labels=['k', 'orb_i', 'orb_j']
            omega: Matsubara frequency grid (fermionic)
            solver: Impurity solver instance (MUST inherit from ImpuritySolverABC)
                    Can be IPTSolver, EDSolver, NRGSolver, etc.
            mu: Chemical potential (default: 0.0)
            mixing: Mixing parameter for Σ updates (0 < mixing <= 1)
                    Can be float (for LinearMixing) or MixingMethod instance
            verbose: Print iteration progress

        Raises:
            TypeError: If solver does not inherit from ImpuritySolverABC
        """
        from .impSolvers.base import ImpuritySolverABC

        if not isinstance(solver, ImpuritySolverABC):
            raise TypeError(
                f"solver must be an instance of ImpuritySolverABC, "
                f"got {type(solver).__name__}. "
                f"Available solvers: IPTSolver, (more to come)"
            )

        # Import BaseTensor for type checking
        from condmatTensor.core import BaseTensor

        self.Hk = Hk
        self.omega = omega
        self.solver = solver
        self.mu = mu
        self.verbose = verbose

        # Set up mixing method
        if isinstance(mixing, float):
            self.mixing_method = LinearMixing(mixing)
        else:
            self.mixing_method = mixing

        # Result storage
        self._Sigma: Optional["BaseTensor"] = None
        self._G_loc: Optional["BaseTensor"] = None
        self._G0: Optional["BaseTensor"] = None
        self._n_iterations: int = 0
        self._convergence_history: Dict[str, list] = {
            "Sigma_diff": [],
            "Sigma_norm": [],
        }

    def run(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        Sigma_init: Optional["BaseTensor"] = None,
    ) -> "BaseTensor":
        """Run DMFT self-consistency loop.

        Implements the 7-step DMFT algorithm.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance (|ΔΣ| / |Σ|)
            Sigma_init: Initial self-energy guess (default: zero)

        Returns:
            BaseTensor with converged self-energy Σ(iω)
            Labels: ['iwn', 'orb_i', 'orb_j']
        """
        from condmatTensor.core import BaseTensor

        # Initialize Σ
        if Sigma_init is not None:
            Sigma = Sigma_init.tensor.clone()
        else:
            # Initialize to zero (non-interacting limit)
            n_omega = len(self.omega)
            n_orb = self.Hk.shape[-1]
            Sigma = torch.zeros(
                (n_omega, n_orb, n_orb),
                dtype=torch.complex128,
                device=self.Hk.tensor.device,
            )

        # Extract Hk tensor and ensure proper shape
        Hk_tensor = self.Hk.tensor
        k_idx = self.Hk.labels.index('k')
        N_k = Hk_tensor.shape[k_idx]
        n_orb = Hk_tensor.shape[-1]

        # Permute Hk to (N_k, n_orb, n_orb) if needed
        if k_idx != 0:
            perm = [k_idx] + [i for i in range(len(self.Hk.labels)) if i != k_idx]
            Hk_tensor = Hk_tensor.permute(perm)

        # Main DMFT loop
        for iteration in range(max_iter):
            # Step 2: Compute lattice Green's function G(k, iω)
            G_kw = self._compute_lattice_greens_function(Hk_tensor, Sigma)

            # Step 3: Extract local G_loc
            G_loc_tensor = self._extract_local_greens_function_tensor(G_kw)
            G_loc = BaseTensor(
                tensor=G_loc_tensor,
                labels=['iwn', 'orb_i', 'orb_j'],
                orbital_names=self.Hk.orbital_names,
                orbital_metadatas=self.Hk.orbital_metadatas,
            )

            # Step 4: Compute Weiss field G₀
            G0_tensor = self._compute_weiss_field(G_loc_tensor, Sigma)
            G0 = BaseTensor(
                tensor=G0_tensor,
                labels=['iwn', 'orb_i', 'orb_j'],
                orbital_names=self.Hk.orbital_names,
                orbital_metadatas=self.Hk.orbital_metadatas,
            )

            # Step 5: Solve impurity problem
            Sigma_new_base = self.solver.solve(G0)
            Sigma_new = Sigma_new_base.tensor

            # Step 6: Mix
            Sigma_old = Sigma.clone()
            Sigma = self.mixing_method.mix(Sigma_old, Sigma_new, iteration, self._convergence_history["Sigma_diff"])

            # Step 7: Check convergence
            diff = self._compute_l2_diff(Sigma, Sigma_old)

            # Store convergence data
            self._convergence_history["Sigma_diff"].append(diff)
            self._convergence_history["Sigma_norm"].append(torch.norm(Sigma).item())

            if self.verbose:
                print(f"Iteration {iteration + 1}: |ΔΣ|/|Σ| = {diff:.6e}")

            if diff < tol:
                if self.verbose:
                    print(f"Converged in {iteration + 1} iterations")
                break

        # Store final results
        self._n_iterations = iteration + 1

        self._Sigma = BaseTensor(
            tensor=Sigma,
            labels=['iwn', 'orb_i', 'orb_j'],
            orbital_names=self.Hk.orbital_names,
            orbital_metadatas=self.Hk.orbital_metadatas,
        )

        self._G_loc = G_loc
        self._G0 = G0

        return self._Sigma

    def _compute_lattice_greens_function(
        self,
        Hk_tensor: torch.Tensor,
        Sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lattice Green's function G(k, iω).

        G(k, iωₙ) = [iωₙ + μ - H(k) - Σ(iωₙ)]^(-1)

        Args:
            Hk_tensor: Hamiltonian in k-space, shape (N_k, n_orb, n_orb)
            Sigma: Self-energy, shape (n_omega, n_orb, n_orb)

        Returns:
            G_kw: Green's function, shape (n_omega, N_k, n_orb, n_orb)
        """
        n_omega = len(self.omega)
        N_k = Hk_tensor.shape[0]
        n_orb = Hk_tensor.shape[-1]

        G_kw = torch.zeros(
            (n_omega, N_k, n_orb, n_orb),
            dtype=torch.complex128,
            device=Hk_tensor.device,
        )

        # For each Matsubara frequency, compute G(k, iωₙ)
        for i, iwn_val in enumerate(self.omega):
            # Build (iωₙ + μ - Hₖ - Σ) for all k-points
            # Hk_tensor: (N_k, n_orb, n_orb)
            # Sigma[i]: (n_orb, n_orb)
            inv_matrix = iwn_val + self.mu - Hk_tensor - Sigma[i]  # (N_k, n_orb, n_orb)

            # Invert at each k-point
            G_kw[i] = torch.linalg.inv(inv_matrix)

        return G_kw

    def _extract_local_greens_function_tensor(
        self,
        G_kw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract local Green's function by k-summation.

        G_loc(iωₙ) = (1/N_k) Σ_k G(k, iωₙ)

        Args:
            G_kw: Green's function, shape (n_omega, N_k, n_orb, n_orb)

        Returns:
            G_loc tensor, shape (n_omega, n_orb, n_orb)
        """
        # Average over k-points (dim=1)
        return torch.mean(G_kw, dim=1)

    def _compute_weiss_field(
        self,
        G_loc: torch.Tensor,
        Sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Weiss field G₀ using inverted Dyson equation.

        From Dyson: G^(-1)(iω) = G₀^(-1)(iω) - Σ(iω)
        Therefore: G₀^(-1)(iω) = G^(-1)(iω) + Σ(iω)

        Implementation:
            G₀_inv[i] = torch.linalg.inv(G_loc[i]) + Sigma[i]
            G₀[i] = torch.linalg.inv(G₀_inv[i])

        Args:
            G_loc: Local Green's function, shape (n_omega, n_orb, n_orb)
            Sigma: Self-energy, shape (n_omega, n_orb, n_orb)

        Returns:
            Weiss field G₀ tensor, shape (n_omega, n_orb, n_orb)
        """
        n_omega = G_loc.shape[0]
        n_orb = G_loc.shape[-1]

        G0 = torch.zeros_like(G_loc)

        for i in range(n_omega):
            # G_loc^(-1) + Σ
            G_loc_inv = torch.linalg.inv(G_loc[i])
            G0_inv = G_loc_inv + Sigma[i]

            # G₀ = (G_loc^(-1) + Σ)^(-1)
            G0[i] = torch.linalg.inv(G0_inv)

        return G0

    def _compute_l2_diff(
        self,
        Sigma_new: torch.Tensor,
        Sigma_old: torch.Tensor,
    ) -> float:
        """Compute L2 norm of self-energy change.

        L2 diff: ||Σ_new - Σ_old|| / ||Σ_old||

        Args:
            Sigma_new: New self-energy
            Sigma_old: Old self-energy

        Returns:
            L2 relative difference
        """
        diff = torch.norm(Sigma_new - Sigma_old)
        norm = torch.norm(Sigma_old)
        return (diff / (norm + 1e-10)).item()

    @property
    def Sigma(self) -> Optional["BaseTensor"]:
        """Final converged self-energy."""
        return self._Sigma

    @property
    def G_loc(self) -> Optional["BaseTensor"]:
        """Final local Green's function."""
        return self._G_loc

    @property
    def G0(self) -> Optional["BaseTensor"]:
        """Final Weiss field."""
        return self._G0

    @property
    def n_iterations(self) -> int:
        """Number of iterations taken to converge."""
        return self._n_iterations

    def get_convergence_history(self) -> Dict[str, list]:
        """Get convergence history for plotting/analysis.

        Returns:
            Dictionary with:
                - 'Sigma_diff': L2 difference at each iteration
                - 'Sigma_norm': L2 norm of Σ at each iteration
        """
        return self._convergence_history
