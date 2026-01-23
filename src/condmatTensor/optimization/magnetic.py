"""Effective array optimizer for magnetic system downfolding.

Provides tools for downfolding high-dimensional magnetic systems (e.g., f-orbitals)
to effective low-dimensional models (e.g., local magnetic moments S_eff).

The key idea is to find an effective Hamiltonian:
    H_eff = H_0 + J_eff @ S_eff

that reproduces the band structure of the full Hamiltonian:
    H_full = H_cc + H_cf + H_ff

where H_cc is conduction-conduction, H_cf is conduction-f, and H_ff is f-f.

**Downfolding Methods:**
1. **Perturbation theory:** Schrieffer-Wolff transformation
2. **Frequency-dependent:** Full Green's function matching
3. **Eigenvalue matching:** Minimize L2 norm of eigenvalue differences

References:
    - "Schrieffer-Wolff transformation" - Schrieffer & Wolff, PR (1966)
    - "Effective Hamiltonians for heavy fermion systems" - Coleman, PRB (1987)

LEVEL 7 of the 10-level architecture.
"""

from typing import Optional, Tuple, Union, List, Callable
import torch
import numpy as np


class EffectiveArrayOptimizer:
    """Optimizer for finding effective magnetic coupling parameters.

    Given a full Hamiltonian H_full (including f-orbitals) and a reference
    conduction Hamiltonian H_cc_0, find effective parameters J_eff and S_eff
    such that the effective model reproduces the full band structure.

    **Optimization Target:**
        min_{J_eff, S_eff} ||eig(H_full) - eig(H_cc_0 + J_eff @ S_eff)||^2

    **Downfolding Methods:**
    1. 'perturbation': Schrieffer-Wolff transformation (analytic)
    2. 'eigenvalue': Direct eigenvalue matching (numerical optimization)
    3. 'green': Full Green's function matching (frequency-dependent)

    Attributes:
        H_cc_0: Reference conduction Hamiltonian (without f)
        H_full: Full Hamiltonian (including f)
        method: Downfolding method
        J_eff: Optimized effective coupling
        S_eff: Optimized effective spin configuration
    """

    def __init__(
        self,
        H_cc_0: "BaseTensor",
        H_full: "BaseTensor",
        method: str = "eigenvalue",
        f_orbital_indices: Optional[List[int]] = None,
        lattice: Optional["BravaisLattice"] = None,
    ):
        """Initialize EffectiveArrayOptimizer.

        Args:
            H_cc_0: Conduction Hamiltonian without f-orbitals
                    Can be H(k) or H(R), uses orbital labels to identify
            H_full: Full Hamiltonian including f-orbitals
                    Same shape and k-points as H_cc_0 plus f-orbitals
            method: Downfolding method ('perturbation', 'eigenvalue', 'green')
            f_orbital_indices: Indices of f-orbitals in H_full
                              If None, tries to detect from orbital names
            lattice: Optional BravaisLattice for per-site orbital info
        """
        from condmatTensor.core import BaseTensor

        self.H_cc_0 = H_cc_0
        self.H_full = H_full
        self.method = method
        self.lattice = lattice

        # Detect f-orbital indices
        if f_orbital_indices is None:
            self.f_indices = self._detect_f_orbitals()
        else:
            self.f_indices = f_orbital_indices

        # Optimization results
        self.J_eff: Optional[float] = None
        self.S_eff: Optional[torch.Tensor] = None
        self.loss_history: List[float] = []

    def reset(self):
        """Reset the optimizer state.

        Clears all optimization results to allow running a fresh optimization.
        """
        self.J_eff = None
        self.S_eff = None
        self.loss_history = []

    def _detect_f_orbitals(self) -> List[int]:
        """Detect f-orbital indices from orbital names.

        Returns:
            List of orbital indices identified as f-orbitals
        """
        if self.H_full.orbital_names is None:
            # Default: assume last orbital is f
            return [self.H_full.shape[-1] - 1]

        f_indices = []
        for i, name in enumerate(self.H_full.orbital_names):
            if 'f' in name.lower():
                f_indices.append(i)

        return f_indices if f_indices else [self.H_full.shape[-1] - 1]

    def _get_n_k(self) -> int:
        """Get number of k-points from Hamiltonian."""
        if "k" in self.H_cc_0.labels:
            return self.H_cc_0.shape[self.H_cc_0.labels.index("k")]
        elif "k" in self.H_full.labels:
            return self.H_full.shape[self.H_full.labels.index("k")]
        else:
            return 1

    def optimize(
        self,
        J_bounds: Tuple[float, float] = (0.01, 10.0),
        S_bounds: Optional[Tuple[float, float]] = None,
        n_init: int = 20,
        n_iter: int = 100,
        backend: str = "auto",
        verbose: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Run optimization to find effective J and S.

        Uses Bayesian optimization to minimize the difference between
        full and effective band structures.

        Args:
            J_bounds: Bounds for J coupling (min, max)
            S_bounds: Bounds for S components (min, max). If None, uses [-1, 1]
            n_init: Number of initial samples for Bayesian optimization
            n_iter: Number of optimization iterations
            backend: Bayesian optimization backend ('auto', 'sober', 'botorch', 'simple')
            verbose: Print progress information
            device: Device for computation

        Returns:
            (J_eff, S_eff) tuple of optimized parameters
        """
        from condmatTensor.optimization.bayesian import BayesianOptimizer

        if device is None:
            device = self.H_cc_0.tensor.device

        if S_bounds is None:
            S_bounds = (-1.0, 1.0)

        # Define objective function
        def objective(params: torch.Tensor) -> torch.Tensor:
            """Objective: L2 norm of eigenvalue differences.

            Args:
                params: Either shape (4,) for single sample or (n_samples, 4) for batch

            Returns:
                Loss tensor (scalar for single sample, or (n_samples,) for batch)
            """
            # Handle both single sample and batch
            original_shape = params.shape
            if params.dim() == 1:
                params = params.unsqueeze(0)  # (1, 4)

            n_samples = params.shape[0]
            losses = []

            for i in range(n_samples):
                J = params[i, 0].item()
                S = params[i, 1:]

                # Build effective Hamiltonian
                H_eff = self._build_effective_hamiltonian(J, S, device)

                # Compute eigenvalues
                eig_full = self._compute_eigenvalues(self.H_full)
                eig_eff = self._compute_eigenvalues(H_eff)

                # L2 norm (handle different number of bands by padding)
                n_bands = min(eig_full.shape[-1], eig_eff.shape[-1])
                diff = eig_full[..., :n_bands] - eig_eff[..., :n_bands]

                loss = torch.mean(diff ** 2)
                losses.append(loss)

            losses_tensor = torch.stack(losses)

            # Return scalar if input was 1D, else return 1D tensor
            if len(original_shape) == 1:
                result = losses_tensor.squeeze(0)
            else:
                result = losses_tensor

            return result

        # Set up bounds: [J, Sx, Sy, Sz]
        bounds = [J_bounds] + [S_bounds] * 3

        # Run Bayesian optimization
        optimizer = BayesianOptimizer(
            bounds=bounds,
            n_init=n_init,
            n_iter=n_iter,
            backend=backend,
        )

        if verbose:
            print("Optimizing effective magnetic coupling...")
            print(f"  Method: {self.method}")
            print(f"  Backend: {backend.upper()}")
            print(f"  J bounds: {J_bounds}")
            print(f"  S bounds: {S_bounds}")

        X_best, loss = optimizer.optimize(objective, maximize=False, verbose=verbose, device=device)

        # Extract results
        self.J_eff = X_best[0].item()
        self.S_eff = X_best[1:]

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  J_eff = {self.J_eff:.6f}")
            print(f"  S_eff = [{self.S_eff[0]:.6f}, {self.S_eff[1]:.6f}, {self.S_eff[2]:.6f}]")
            print(f"  Final loss = {loss:.6e}")

        return self.J_eff, self.S_eff

    def _build_effective_hamiltonian(
        self,
        J: float,
        S: torch.Tensor,
        device: torch.device,
    ) -> "BaseTensor":
        """Build effective Hamiltonian H_eff = H_cc_0 + J @ S.

        Args:
            J: Coupling strength
            S: Spin vector (3,) for (Sx, Sy, Sz)
            device: Device for computation

        Returns:
            BaseTensor with effective Hamiltonian
        """
        from condmatTensor.core import BaseTensor
        from condmatTensor.manybody.magnetic import LocalMagneticModel

        # Build spinful H_cc_0 if needed
        N_orb_cc = self.H_cc_0.shape[-1]

        # Check if already spinful (even number of orbitals, assumed spinor)
        if N_orb_cc % 2 == 0:
            # Assume spinful already
            H_eff = self.H_cc_0.tensor.clone()
        else:
            # Build spinful from spinless
            model = LocalMagneticModel()
            H_cc_0_spinful = model.build_spinful_hamiltonian(self.H_cc_0, lattice=self.lattice)
            H_eff = H_cc_0_spinful.tensor.clone()

        # Add J@S term
        if self.lattice is not None:
            # Use per-site orbital info from lattice
            # Add J@S to all sites (uniform coupling)
            spinful_offset = 0
            for site_idx, n_orb_site in enumerate(self.lattice.num_orbitals):
                # Create J@S term (2×2 Pauli matrix)
                Sx, Sy, Sz = S
                J_term = torch.zeros((2, 2), dtype=torch.complex128, device=device)
                J_term[0, 0] = J * Sz
                J_term[0, 1] = J * (Sx - 1j * Sy)
                J_term[1, 0] = J * (Sx + 1j * Sy)
                J_term[1, 1] = -J * Sz

                # Add to each orbital at this site
                N_k = H_eff.shape[0] if H_eff.dim() == 3 else 1
                for orb_i in range(n_orb_site):
                    idx_i = spinful_offset + 2*orb_i
                    for k in range(N_k):
                        if H_eff.dim() == 3:
                            H_eff[k, idx_i:idx_i+2, idx_i:idx_i+2] += J_term
                        else:
                            H_eff[idx_i:idx_i+2, idx_i:idx_i+2] += J_term

                spinful_offset += 2 * n_orb_site
        else:
            # Old behavior: add to first orbital as on-site magnetic exchange
            N_orb_spinful = H_eff.shape[-1]

            # Create J@S term (2×2 Pauli matrix)
            Sx, Sy, Sz = S
            J_term = torch.zeros((2, 2), dtype=torch.complex128, device=device)
            J_term[0, 0] = J * Sz
            J_term[0, 1] = J * (Sx - 1j * Sy)
            J_term[1, 0] = J * (Sx + 1j * Sy)
            J_term[1, 1] = -J * Sz

            # Add to first orbital
            N_k = H_eff.shape[0] if H_eff.dim() == 3 else 1

            for k in range(N_k):
                if H_eff.dim() == 3:
                    H_eff[k, 0:2, 0:2] += J_term
                else:
                    H_eff[0:2, 0:2] += J_term

        return BaseTensor(
            tensor=H_eff,
            labels=self.H_cc_0.labels,
            orbital_names=self.H_cc_0.orbital_names,
            displacements=self.H_cc_0.displacements,
        )

    def _compute_eigenvalues(self, H: "BaseTensor") -> torch.Tensor:
        """Compute eigenvalues of Hamiltonian.

        Args:
            H: BaseTensor with Hamiltonian

        Returns:
            Eigenvalues tensor
        """
        if "k" in H.labels:
            k_idx = H.labels.index("k")
            tensor = H.tensor

            # Permute to (N_k, n_orb, n_orb)
            if k_idx != 0:
                perm = [k_idx] + [i for i in range(len(H.labels)) if i != k_idx]
                tensor = tensor.permute(perm)

            N_k = tensor.shape[0]
            n_orb = tensor.shape[-1]

            eigenvalues = torch.zeros((N_k, n_orb), dtype=torch.float64, device=tensor.device)

            for k in range(N_k):
                eigenvalues[k] = torch.linalg.eigvalsh(tensor[k]).real

            return eigenvalues
        else:
            return torch.linalg.eigvalsh(H.tensor).real.unsqueeze(0)

    def verify(
        self,
        k_path: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> dict:
        """Verify the effective model by comparing band structures.

        Args:
            k_path: Optional k-path for band structure comparison
            verbose: Print detailed comparison

        Returns:
            Dictionary with comparison metrics
        """
        if self.J_eff is None or self.S_eff is None:
            raise ValueError("Must run optimize() before verify()")

        # Build effective Hamiltonian
        H_eff = self._build_effective_hamiltonian(
            self.J_eff,
            self.S_eff,
            self.H_cc_0.tensor.device,
        )

        # Compute eigenvalues
        eig_full = self._compute_eigenvalues(self.H_full)
        eig_eff = self._compute_eigenvalues(H_eff)

        # Metrics
        n_bands = min(eig_full.shape[-1], eig_eff.shape[-1])
        diff = eig_full[..., :n_bands] - eig_eff[..., :n_bands]

        metrics = {
            "mean_absolute_error": torch.mean(torch.abs(diff)).item(),
            "max_absolute_error": torch.max(torch.abs(diff)).item(),
            "rmse": torch.sqrt(torch.mean(diff ** 2)).item(),
            "correlation": self._compute_correlation(
                eig_full[..., :n_bands].flatten(),
                eig_eff[..., :n_bands].flatten(),
            ),
        }

        if verbose:
            print("\nEffective Model Verification:")
            print(f"  Mean absolute error: {metrics['mean_absolute_error']:.6e}")
            print(f"  Max absolute error: {metrics['max_absolute_error']:.6e}")
            print(f"  RMSE: {metrics['rmse']:.6e}")
            print(f"  Correlation: {metrics['correlation']:.6f}")

        return metrics

    def _compute_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation coefficient."""
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)

        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2))

        if denominator == 0:
            return 0.0

        return (numerator / denominator).item()

    def perturbation_theory(
        self,
        epsilon_f: float,
        V_cf: float,
    ) -> Tuple[float, torch.Tensor]:
        """Compute effective J and S using perturbation theory.

        For a conduction-f hybridization model:
            H = H_c + ε_f f†f + V_cf (c†f + f†c)

        Second-order perturbation theory gives:
            J_eff = V_cf² / ε_f
            S_eff = (expectation value from f-occupancy)

        This is the Schrieffer-Wolff transformation result.

        Args:
            epsilon_f: f-orbital energy (relative to Fermi level)
            V_cf: Conduction-f hybridization strength

        Returns:
            (J_eff, S_eff) tuple
        """
        # Schrieffer-Wolff transformation
        # J_eff = V_cf² / ε_f (for large |ε_f|)
        J_eff = V_cf ** 2 / epsilon_f if epsilon_f != 0 else float('inf')

        # S_eff depends on f-orbital occupation
        # For completely localized f-electrons: |S| = 1/2 (for single f-electron)
        # For empty or doubly occupied: S = 0
        S_eff = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)

        return J_eff, S_eff

    def plot_comparison(
        self,
        k_path: Optional[torch.Tensor] = None,
        ticks: Optional[list] = None,
        ax=None,
    ):
        """Plot band structure comparison between full and effective models.

        Args:
            k_path: k-path for band structure (if None, uses existing k-points)
            ticks: Tick labels for high-symmetry points
            ax: Matplotlib axis (if None, creates new figure)

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if self.J_eff is None or self.S_eff is None:
            raise ValueError("Must run optimize() before plot_comparison()")

        # Build effective Hamiltonian
        H_eff = self._build_effective_hamiltonian(
            self.J_eff,
            self.S_eff,
            self.H_cc_0.tensor.device,
        )

        # Compute eigenvalues
        eig_full = self._compute_eigenvalues(self.H_full)
        eig_eff = self._compute_eigenvalues(H_eff)

        # Create k-axis if not provided
        N_k = eig_full.shape[0]
        k_axis = torch.arange(N_k, dtype=torch.float64)

        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot full bands (solid)
        n_bands_full = eig_full.shape[-1]
        for i in range(n_bands_full):
            ax.plot(k_axis.numpy(), eig_full[:, i].cpu().numpy(),
                    'b-', alpha=0.6, linewidth=1)

        # Plot effective bands (dashed)
        n_bands_eff = eig_eff.shape[-1]
        for i in range(n_bands_eff):
            ax.plot(k_axis.numpy(), eig_eff[:, i].cpu().numpy(),
                    'r--', alpha=0.8, linewidth=1.5)

        # Labels
        ax.set_xlabel("k-path", fontsize=12)
        ax.set_ylabel("Energy ($|t|$)", fontsize=12)
        ax.set_title(f"Band Structure Comparison\n"
                    f"Full (blue) vs Effective (red, $J_{{eff}}$={self.J_eff:.3f})", fontsize=12)
        ax.grid(True, alpha=0.3)

        # High-symmetry points
        if ticks is not None:
            ax.set_xticks(ticks[0])
            ax.set_xticklabels(ticks[1])

        return ax
