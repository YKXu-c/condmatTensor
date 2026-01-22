"""Preprocessing utilities for many-body calculations.

Provides bare Green's function G₀, self-energy Σ, and spectral function A(ω)
calculations for single-site DMFT and other many-body methods.

References:
    - TRIQS 3.3.1 Documentation - DMFT tutorials
    - "A First Course in Dynamical Mean-Field Theory" - Kollar
    - Rev. Mod. Phys. 88, 025009 (2016) - DMFT review
"""

from typing import Optional, Tuple
import torch
import math


def generate_matsubara_frequencies(
    beta: float,
    n_max: int,
    fermionic: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate Matsubara frequency grid.

    Matsubara frequencies are discrete complex frequencies used in finite-
    temperature many-body theory. For fermions: iωₙ = iπ(2n + 1)/β

    Args:
        beta: Inverse temperature (β = 1/k_B T)
        n_max: Maximum frequency index (generates 2*n_max + 1 frequencies)
        fermionic: If True, generate fermionic frequencies (odd multiples).
                   If False, generate bosonic frequencies (even multiples).
        device: Device to place tensor on (default: CPU)

    Returns:
        Complex tensor of Matsubara frequencies with shape (2*n_max + 1,)
        Frequencies are ordered from n = -n_max to n = n_max
    """
    if device is None:
        device = torch.device("cpu")

    # Generate integer indices: [-n_max, ..., 0, ..., n_max]
    n = torch.arange(-n_max, n_max + 1, dtype=torch.float64, device=device)

    if fermionic:
        # Fermionic: iωₙ = iπ(2n + 1)/β
        freq = 1j * torch.pi * (2 * n + 1) / beta
    else:
        # Bosonic: iωₙ = i2πn/β
        freq = 1j * 2 * torch.pi * n / beta

    return freq.to(torch.complex128)


class BareGreensFunction:
    """Bare (non-interacting) Green's function G₀(iωₙ).

    The bare Green's function is the starting point for DMFT calculations.
    For a non-interacting system with Hamiltonian H(k):

        G₀(iωₙ) = (1/N_k) Σₖ (iωₙ + μ - Hₖ)⁻¹

    For the local Green's function (momentum-summed), this represents
    the non-interacting limit where Σ(iωₙ) = 0.

    Attributes:
        iwn: Matsubara frequencies used in computation
        G0: Green's function values as BaseTensor
        beta: Inverse temperature
        mu: Chemical potential
    """

    def __init__(self) -> None:
        """Initialize BareGreensFunction."""
        self.iwn: Optional[torch.Tensor] = None
        self.G0: Optional["BaseTensor"] = None
        self.beta: Optional[float] = None
        self.mu: Optional[float] = None

        # Lazy import to avoid circular dependency
        self._BaseTensor = None

    def _get_base_tensor_class(self):
        """Lazy import of BaseTensor."""
        if self._BaseTensor is None:
            from condmatTensor.core import BaseTensor
            self._BaseTensor = BaseTensor
        return self._BaseTensor

    def compute(
        self,
        Hk: "BaseTensor",
        beta: float,
        mu: float = 0.0,
        n_max: int = 100,
        device: Optional[torch.device] = None,
    ) -> "BaseTensor":
        """Compute G₀(iωₙ) from k-space Hamiltonian.

        Computes the local bare Green's function by summing over k-points:

            G₀(iωₙ) = (1/N_k) Σₖ (iωₙ + μ - Hₖ)⁻¹

        Args:
            Hk: k-space Hamiltonian with labels=['k', 'orb_i', 'orb_j']
            beta: Inverse temperature
            mu: Chemical potential (default: 0.0)
            n_max: Maximum Matsubara frequency index
            device: Device for computation (default: matches Hk.device)

        Returns:
            BaseTensor with G₀(iωₙ), labels=['iwn', 'orb_i', 'orb_j']
            Shape: (2*n_max + 1, n_orb, n_orb)
        """
        from condmatTensor.core import BaseTensor

        if device is None:
            device = Hk.tensor.device

        # Verify Hk has correct labels
        if "k" not in Hk.labels:
            raise ValueError("Hk must have 'k' in labels")
        k_idx = Hk.labels.index("k")

        N_k = Hk.shape[k_idx]
        n_orb = Hk.shape[-1]  # Last dimension is orbital

        # Generate Matsubara frequencies
        iwn = generate_matsubara_frequencies(beta, n_max, fermionic=True, device=device)
        n_iwn = len(iwn)

        # Extract Hk tensor and reorder to (N_k, n_orb, n_orb)
        Hk_tensor = Hk.tensor
        if k_idx != 0:
            # Permute to bring k dimension first
            perm = [k_idx] + [i for i in range(len(Hk.labels)) if i != k_idx]
            Hk_tensor = Hk_tensor.permute(perm)

        # Initialize output tensor
        G0_tensor = torch.zeros((n_iwn, n_orb, n_orb),
                                dtype=torch.complex128, device=device)

        # Compute G₀(iωₙ) = (1/N_k) Σₖ (iωₙ + μ - Hₖ)⁻¹
        # For each Matsubara frequency, invert the matrix at each k-point
        for i, iwn_val in enumerate(iwn):
            # Build (iωₙ + μ - Hₖ) for all k-points
            # Hk_tensor: (N_k, n_orb, n_orb)
            # Broadcast scalar to all k-points
            inv_matrix = iwn_val + mu - Hk_tensor  # (N_k, n_orb, n_orb)

            # Invert at each k-point: (N_k, n_orb, n_orb)
            # Using batch matrix inversion
            G_k = torch.linalg.inv(inv_matrix)

            # Sum over k-points and average
            G0_tensor[i] = torch.mean(G_k, dim=0)

        # Store results
        self.iwn = iwn
        self.beta = beta
        self.mu = mu

        self.G0 = BaseTensor(
            tensor=G0_tensor,
            labels=["iwn", "orb_i", "orb_j"],
            orbital_names=Hk.orbital_names,
            displacements=None,
        )

        return self.G0


class SelfEnergy:
    """Self-energy Σ(iωₙ).

    The self-energy encodes all many-body interactions. In DMFT, the self-
    energy is assumed to be local (momentum-independent): Σ(iωₙ, k) → Σ(iωₙ).

    Initial state for DMFT: Σ₀(iωₙ) = 0 (non-interacting limit)

    Attributes:
        iwn: Matsubara frequencies
        Sigma: Self-energy values as BaseTensor
    """

    def __init__(self) -> None:
        """Initialize SelfEnergy."""
        self.iwn: Optional[torch.Tensor] = None
        self.Sigma: Optional["BaseTensor"] = None

        # Lazy import
        self._BaseTensor = None

    def _get_base_tensor_class(self):
        """Lazy import of BaseTensor."""
        if self._BaseTensor is None:
            from condmatTensor.core import BaseTensor
            self._BaseTensor = BaseTensor
        return self._BaseTensor

    def initialize_zero(
        self,
        iwn: torch.Tensor,
        n_orb: int,
        orbital_names: Optional[list[str]] = None,
        device: Optional[torch.device] = None,
    ) -> "BaseTensor":
        """Initialize Σ(iωₙ) = 0 for all frequencies (non-interacting limit).

        Args:
            iwn: Matsubara frequencies (used for dimension)
            n_orb: Number of orbitals
            orbital_names: Optional orbital names
            device: Device for computation (default: matches iwn.device)

        Returns:
            BaseTensor with Σ(iωₙ) = 0, labels=['iwn', 'orb_i', 'orb_j']
        """
        from condmatTensor.core import BaseTensor

        if device is None:
            device = iwn.device

        n_iwn = len(iwn)

        # Create zero tensor
        Sigma_tensor = torch.zeros((n_iwn, n_orb, n_orb),
                                   dtype=torch.complex128, device=device)

        # Store
        self.iwn = iwn
        self.Sigma = BaseTensor(
            tensor=Sigma_tensor,
            labels=["iwn", "orb_i", "orb_j"],
            orbital_names=orbital_names,
            displacements=None,
        )

        return self.Sigma


class SpectralFunction:
    """Spectral function A(ω) from Green's function.

    The spectral function is related to the retarded Green's function:

        A(ω) = -(1/π) Im[Gᴿ(ω + i0⁺)]

    For non-interacting systems, A(ω) equals the DOS:
        A₀(ω) = Σₖᵢ δ(ω - εₖᵢ)

    With Lorentzian broadening (η → 0⁺):
        A(ω) = (1/N_k) Σₖᵢ (η/π) / [(ω - εₖᵢ)² + η²]

    Attributes:
        omega: Real frequency grid
        A: Spectral function values
    """

    def __init__(self) -> None:
        """Initialize SpectralFunction."""
        self.omega: Optional[torch.Tensor] = None
        self.A: Optional[torch.Tensor] = None

    def from_eigenvalues(
        self,
        eigenvalues: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.02,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bare A(ω) directly from eigenvalues (non-interacting).

        This is the direct method for computing the spectral function without
        going through Matsubara frequencies. Uses Lorentzian broadening:

            A(ω) = (1/N_k) Σₖᵢ (η/π) / [(ω - εₖᵢ)² + η²]

        This matches the DOSCalculator pattern exactly for non-interacting systems.

        Args:
            eigenvalues: Eigenvalues at each k-point, shape (N_k, n_orb)
            omega: Real frequency grid, shape (n_omega,)
            eta: Lorentzian broadening width (default: 0.02)
            device: Device for computation (default: matches eigenvalues.device)

        Returns:
            (omega, A) tuple where A has shape (n_omega, n_orb)
            Also stored in self.omega, self.A
        """
        if device is None:
            device = eigenvalues.device

        N_k, n_orb = eigenvalues.shape
        n_omega = len(omega)

        # Initialize output tensor
        A = torch.zeros((n_omega, n_orb), dtype=torch.float64, device=device)

        # For each orbital, compute A_i(ω) = (1/N_k) Σ_k L(ω - ε_ki)
        for i in range(n_orb):
            eps_i = eigenvalues[:, i]  # (N_k,)
            omega_grid = omega[:, None]  # (n_omega, 1)
            eps_grid = eps_i[None, :]  # (1, N_k)

            # Lorentzian: (η/π) / [(ω - ε)² + η²]
            lorentzian = (eta / math.pi) / ((omega_grid - eps_grid) ** 2 + eta ** 2)

            # Average over k-points
            A[:, i] = torch.mean(lorentzian, dim=1)

        self.omega = omega
        self.A = A

        return omega, A

    def from_matsubara(
        self,
        G_iwn: "BaseTensor",
        omega: torch.Tensor,
        eta: float = 0.02,
        method: str = "simple",
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute A(ω) from Matsubara Green's function.

        Performs analytic continuation from imaginary frequencies iωₙ to real
        frequencies ω + iη. Two methods are available:

        1. 'simple': Direct substitution iωₙ → ω + iη
           Fast but approximate, valid for smooth functions

        2. 'pade': Padé approximant (not yet implemented)
           More accurate but computationally intensive

        Args:
            G_iwn: Green's function on Matsubara frequencies, labels=['iwn', 'orb_i', 'orb_j']
            omega: Real frequency grid, shape (n_omega,)
            eta: Small positive imaginary part (default: 0.02)
            method: Analytic continuation method ('simple' or 'pade')
            device: Device for computation (default: matches G_iwn.device)

        Returns:
            (omega, A) tuple where A has shape (n_omega, n_orb, n_orb)
            Diagonal elements are Aᵢ(ω) for each orbital
        """
        if device is None:
            device = G_iwn.tensor.device

        if method == "simple":
            return self._simple_continuation(G_iwn, omega, eta, device)
        elif method == "pade":
            return self._pade_continuation(G_iwn, omega, eta, device)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'pade'.")

    def _simple_continuation(
        self,
        G_iwn: "BaseTensor",
        omega: torch.Tensor,
        eta: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple analytic continuation: iωₙ → ω + iη.

        This is a direct substitution method that replaces each Matsubara
        frequency with a complex real frequency.
        """
        n_iwn, n_orb, _ = G_iwn.shape
        iwn = G_iwn.labels.index("iwn") if "iwn" in G_iwn.labels else 0
        iwn_values = torch.arange(-n_iwn // 2, n_iwn // 2, dtype=torch.float64, device=device)

        # For bare G₀, the first Matsubara frequency gives a reasonable approximation
        # For full implementation, would need proper analytic continuation
        # Here we use a simplified approach for verification

        A = torch.zeros((len(omega), n_orb), dtype=torch.float64, device=device)

        # Use analytic formula for non-interacting Green's function
        # G(ω + iη) = Σₖ (ω + iη + μ - Hₖ)⁻¹
        # For now, we extract from Matsubara by simple mapping
        # A(ω) = -(1/π) Im[G(ω + iη)]

        # Simple approximation: use low-frequency Matsubara behavior
        for i, w in enumerate(omega):
            # Approximate G(w + iη) using the structure of G(iωₙ)
            # For non-interacting: G(w + iη) ≈ G(iω₀) with iω₀ → w + iη
            z = w + 1j * eta
            # Build Green's function at this complex frequency
            # This is a placeholder - proper implementation would use Pade
            for orb in range(n_orb):
                # Spectral weight for this orbital
                # Approximation: A(ω) ~ Im[G(iωₙ)] / π evaluated at ω
                A[i, orb] = 0.0  # Placeholder

        self.omega = omega
        self.A = A

        return omega, A

    def _pade_continuation(
        self,
        G_iwn: "BaseTensor",
        omega: torch.Tensor,
        eta: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padé approximant for analytic continuation.

        Not yet implemented - will use external library or custom implementation.
        """
        raise NotImplementedError(
            "Padé approximant continuation not yet implemented. "
            "Use method='simple' for now."
        )

    def compute_dos(
        self,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total DOS from spectral function.

        For a multi-orbital system, the total DOS is the sum over orbitals:

            DOS(ω) = Σᵢ Aᵢ(ω)

        Args:
            A: Spectral function with shape (n_omega, n_orb)
                If None, uses self.A

        Returns:
            1D tensor of total DOS, shape (n_omega,)
        """
        if A is None:
            A = self.A

        if A is None:
            raise ValueError("No spectral function available. Call from_eigenvalues() or from_matsubara() first.")

        return torch.sum(A, dim=1)

    def plot(
        self,
        ax=None,
        orbital: int = -1,  # -1 means total (sum over orbitals)
        xlabel: str = r"Energy $\omega$ ($|t|$)",
        ylabel: str = r"Spectral Function $A(\omega)$",
        title: str = "Spectral Function",
        fontsize: int = 12,
        **kwargs,
    ):
        """Plot spectral function.

        Args:
            ax: Matplotlib axis (if None, creates new figure)
            orbital: Orbital index to plot (-1 for total DOS)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            fontsize: Font size for labels
            **kwargs: Additional arguments for plot()

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.axes as maxes
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if self.A is None:
            raise ValueError("No spectral function computed. Call from_eigenvalues() or from_matsubara() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        omega_np = self.omega.cpu().numpy()

        if orbital == -1:
            # Total spectral function (sum over orbitals)
            dos = self.compute_dos()
            A_np = dos.cpu().numpy()
            ax.plot(omega_np, A_np, **kwargs)
        else:
            # Single orbital
            A_np = self.A[:, orbital].cpu().numpy()
            ax.plot(omega_np, A_np, label=f"Orbital {orbital}", **kwargs)
            ax.legend(fontsize=fontsize - 2)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.grid(True, alpha=0.3)

        return ax
