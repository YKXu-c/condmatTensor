"""Preprocessing utilities for many-body calculations.

Provides bare Green's function G₀, self-energy Σ, and spectral function A(ω)
calculations for single-site DMFT and other many-body methods.

References:
    - TRIQS 3.3.1 Documentation - DMFT tutorials
    - "A First Course in Dynamical Mean-Field Theory" - Kollar
    - Rev. Mod. Phys. 88, 025009 (2016) - DMFT review
"""

from typing import Optional, Tuple, Union, List
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
    temperature many-body theory.

    **Indexing Scheme**:
    - Frequencies are indexed from n = -n_max to n = +n_max (NOT from 0!)
    - For n_max=3: indices are [-3, -2, -1, 0, 1, 2, 3]
    - This symmetric indexing is physically meaningful for fermionic sums

    **Fermionic Frequencies**:
        iωₙ = iπ(2n + 1)/β

    **Note**: The "zero" index (n=0) does NOT mean zero frequency!
    - n=0 gives: iω₀ = iπ/β
    - True zero frequency would require half-integer n

    **For plotting**: Use torch.arange(len(omega)) for x-axis labels.
    The actual Matsubara frequencies are omega[label_index].

    Args:
        beta: Inverse temperature (β = 1/k_B T)
        n_max: Maximum frequency index (generates 2*n_max + 1 frequencies)
        fermionic: If True, generate fermionic frequencies (odd multiples).
                   If False, generate bosonic frequencies (even multiples).
        device: Device to place tensor on (default: CPU)

    Returns:
        Complex tensor of Matsubara frequencies with shape (2*n_max + 1,)
        Frequencies are ordered from n = -n_max to n = n_max
        Example for n_max=2: [iω₋₂, iω₋₁, iω₀, iω₊₁, iω₊₂]
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


def calculate_dos_range(
    evals_min: float,
    evals_max: float,
    sigma_shift: float,
    U_max: float,
    margin: float = 2.0,
) -> Tuple[float, float]:
    """Auto-calculate DOS range for interacting systems.

    For interacting systems with Hubbard U, the DOS can extend significantly
    beyond the non-interacting band range due to:
    1. Self-energy shifts: Re[Σ] can push bands up or down
    2. Hubbard bands: Can appear at ±U/2 from the Fermi level

    This function computes a safe energy range for DOS calculations.

    Args:
        evals_min: Minimum eigenvalue from band structure (units of |t|)
        evals_max: Maximum eigenvalue from band structure (units of |t|)
        sigma_shift: Maximum |Re[Σ]| from self-energy (units of |t|)
        U_max: Maximum Hubbard U value (units of |t|)
        margin: Additional safety margin (default: 2.0)

    Returns:
        (omega_min, omega_max) - Energy range for DOS calculation

    Example:
        >>> evals_min, evals_max = -3.0, 5.0  # Kagome-F bands
        >>> sigma_shift = 1.5  # From self-energy
        >>> U_max = 4.0  # Hubbard U on f-orbital
        >>> omega_min, omega_max = calculate_dos_range(
        ...     evals_min, evals_max, sigma_shift, U_max
        ... )
        >>> print(f"DOS range: [{omega_min:.1f}, {omega_max:.1f}]")
        DOS range: [-12.5, 14.5]
    """
    # Account for band structure, self-energy shift, and Hubbard bands
    # Hubbard bands can appear at approximately ±U/2 from renormalized bands
    width = (evals_max - evals_min) + 2 * sigma_shift + U_max
    center = (evals_min + evals_max) / 2

    omega_min = center - width / 2 - margin
    omega_max = center + width / 2 + margin

    return omega_min, omega_max


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
        tolerance: Numerical tolerance for Pade algorithm (< 1e-6)
    """

    def __init__(self, tolerance: float = 1e-12) -> None:
        """Initialize SpectralFunction.

        Args:
            tolerance: Numerical tolerance for Pade comparisons (default: 1e-12).
                      Must be < 1e-6 for double precision accuracy.
        """
        if tolerance >= 1e-6:
            raise ValueError(f"tolerance must be < 1e-6, got {tolerance}")
        self.omega: Optional[torch.Tensor] = None
        self.A: Optional[torch.Tensor] = None
        self.beta: Optional[float] = None  # Inverse temperature
        self.tolerance = tolerance

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
        beta: Optional[float] = None,
        n_min: int = 0,
        n_max: Optional[int] = None,
        **method_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute A(ω) from Matsubara Green's function.

        Performs analytic continuation from imaginary frequencies iωₙ to real
        frequencies ω + iη using modular continuation methods.

        Available methods:
        1. 'simple': Direct substitution iωₙ → ω + iη
           Fast but approximate, valid for smooth functions only

        2. 'pade': Padé approximant (Vidberg-Serene continued fraction)
           More accurate but computationally intensive

        3. 'bethe': Bethe lattice analytical solution
           Semi-elliptical DOS, useful for testing

        4. 'maxent': Maximum entropy method (NOT YET IMPLEMENTED)
           Most robust for noisy data

        Args:
            G_iwn: Green's function on Matsubara frequencies, labels=['iwn', 'orb_i', 'orb_j']
            omega: Real frequency grid, shape (n_omega,)
            eta: Small positive imaginary part (default: 0.02)
            method: Analytic continuation method ('simple', 'pade', 'bethe', 'maxent')
            device: Device for computation (default: matches G_iwn.device)
            beta: Inverse temperature (required for Pade continuation)
            n_min: Minimum Matsubara index for Pade (default: 0)
            n_max: Maximum Matsubara index for Pade (default: N//2)
            **method_kwargs: Method-specific parameters:
                - For 'bethe': z (coordination number), t (hopping), lattice
                - For 'maxent': alpha, default_model, etc.

        Returns:
            (omega, A) tuple where A has shape (n_omega, n_orb)
            Diagonal elements are Aᵢ(ω) for each orbital

        Raises:
            ValueError: If method is unknown
            NotImplementedError: If method is not yet implemented (e.g., 'maxent')
        """
        if device is None:
            device = G_iwn.tensor.device

        # Store beta for Pade continuation
        if beta is not None:
            self.beta = beta
        elif self.beta is None and method == "pade":
            raise ValueError("beta must be provided for Pade continuation")

        # Use the new modular framework
        from condmatTensor.manybody.analytic_continuation import (
            create_continuation_method,
        )

        continuation_method = create_continuation_method(method)

        # Prepare method-specific parameters
        method_params = {
            'eta': eta,
            'beta': beta,
            'n_min': n_min,
            'n_max': n_max,
            **method_kwargs,
        }

        # Call the continuation method
        A = continuation_method.continue_to_real_axis(
            G_iwn.tensor, omega, **method_params
        )

        self.omega = omega
        self.A = A

        return omega, A

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
        n_min: int = 0,
        n_max: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padé approximant for analytic continuation.

        Uses continued fraction representation following the Vidberg-Serene algorithm.
        This provides a more accurate analytic continuation from Matsubara frequencies
        to real frequencies than the simple substitution method.

        The Padé approximant constructs a rational function C_M(z) = A_M(z) / B_M(z)
        that matches G(iωₙ) at M selected Matsubara frequencies and is evaluated
        at z = ω + iη to obtain the spectral function.

        Args:
            G_iwn: Green's function on Matsubara frequencies, labels=['iwn', 'orb_i', 'orb_j']
            omega: Real frequency grid for output, shape (n_omega,)
            eta: Imaginary shift (iωₙ → ω + iη)
            device: Device for computation
            n_min: Minimum Matsubara index to use (default: 0)
            n_max: Maximum Matsubara index (default: N//2)

        Returns:
            (omega, A) tuple where A has shape (n_omega, n_orb)
            Diagonal elements are Aᵢ(ω) for each orbital

        Reference:
            Vidberg H.J. and Serene J.W., J. Low Temp. Phys. 29, 179 (1977)
            Beach K.S., Gooding R.J., Marsiglio F., Phys. Rev. B 61, 5147 (2000)
        """
        n_iwn, n_orb, _ = G_iwn.shape
        if n_max is None:
            n_max = n_iwn // 2

        # Generate Matsubara frequencies for the selected range
        # We need the actual frequency values, not just indices
        # Use symmetric selection around zero for best Padé stability
        n_vals = torch.arange(n_iwn, device=device) - n_iwn // 2

        # Select frequencies: prefer low frequencies where signal is strongest
        # Start from n_vals=0 (first positive frequency)
        idx_start = (n_iwn // 2) + n_min  # Convert to actual index

        # OPTIMIZED FREQUENCY SELECTION:
        # Instead of blindly using n_max frequencies, select based on |G| magnitude
        # Higher frequencies have exponentially smaller |G| and contribute noise

        # Get the first n_max candidate frequencies
        idx_end_candidate = min(idx_start + n_max, n_iwn)

        # Use first orbital as reference (all orbitals should have similar decay)
        G_candidate = G_iwn.tensor[idx_start:idx_end_candidate, 0, 0]

        # Compute |G| for candidate frequencies
        G_mag = torch.abs(G_candidate)

        # Adaptive selection: use frequencies where |G| > threshold
        # Threshold: relative (0.1% of max) OR absolute (self.tolerance)
        # This automatically excludes noisy high-frequency data
        relative_threshold = 1e-3 * G_mag.max().item()
        threshold = max(relative_threshold, self.tolerance)

        # Find indices where |G| > threshold
        significant_mask = G_mag > threshold
        n_significant = significant_mask.sum().item()

        # Cap at a reasonable maximum to avoid numerical instability
        # Empirically, 20-30 frequencies is optimal for most cases
        n_optimal = min(n_significant, 30)

        # Ensure we use at least a minimum number of frequencies
        n_optimal = max(n_optimal, 8)

        idx_end = idx_start + n_optimal

        # Handle boundary conditions
        if idx_end > n_iwn:
            idx_end = n_iwn
            idx_start = max(0, idx_end - n_optimal)

        # Complex frequencies for Padé construction
        n_vals_selected = n_vals[idx_start:idx_end]
        wn = (2 * n_vals_selected + 1) * math.pi / self.beta  # Fermionic Matsubara frequencies

        # Complex Matsubara frequencies
        z_iwn = 1j * wn

        # Extract diagonal elements (orbital-diagonal approximation)
        A = torch.zeros((len(omega), n_orb), dtype=torch.float64, device=device)

        for orb in range(n_orb):
            # Get G_ii(iωₙ) for this orbital
            G_diag = G_iwn.tensor[idx_start:idx_end, orb, orb]

            # Compute Pade approximant for this orbital
            A[:, orb] = self._pade_continued_fraction(
                G_diag, z_iwn, omega, eta, device
            )

        self.omega = omega
        self.A = A

        return omega, A

    def _pade_continued_fraction(
        self,
        G_iwn: torch.Tensor,
        z_iwn: torch.Tensor,
        omega: torch.Tensor,
        eta: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute Pade approximant using continued fractions.

        Uses the Vidberg-Serene algorithm which constructs a continued fraction
        representation of the Pade approximant:

        C_M(z) = a₀ / (1 + a₁(z-z₀) / (1 + a₂(z-z₁) / (...)))

        where the coefficients aᵢ are determined from the values G(iωₙ).

        Args:
            G_iwn: Green's function values at Matsubara frequencies (1D)
            z_iwn: Complex Matsubara frequencies iωₙ
            omega: Real frequency grid
            eta: Imaginary shift
            device: Computation device

        Returns:
            Spectral function A(ω) = -(1/π)Im[G(ω + iη)]
        """
        N = len(G_iwn)
        n_omega = len(omega)

        # Flatten G_iwn to 1D for g-table construction (extract diagonal)
        G_iwn_1d = G_iwn[:, 0]  # Shape (N,)

        # 1. Build g-table using continued fraction recursion (Vidberg-Serene)
        # g[i, j] represents the j-th element in the i-th column
        g = torch.zeros((N, N), dtype=torch.complex128, device=device)

        # First column: original G values
        g[0, :] = G_iwn_1d

        # Build the g-table recursively
        # g[i, j] = (g[i-1, i-1] - g[i-1, j]) / ((z_j - z_{i-1}) * g[i-1, j])
        for i in range(1, N):
            for j in range(i, N):
                numerator = g[i-1, i-1] - g[i-1, j]
                denominator = (z_iwn[j] - z_iwn[i-1]) * g[i-1, j]
                if torch.abs(denominator) > self.tolerance:
                    g[i, j] = numerator / denominator
                else:
                    g[i, j] = 0.0

        # 2. Extract continued fraction coefficients from diagonal
        # a₀ = g[0,0], a₁ = g[1,1], a₂ = g[2,2], ...
        a = torch.zeros(N, dtype=torch.complex128, device=device)
        for i in range(N):
            a[i] = g[i, i]

        # 3. Evaluate continued fraction at each real frequency
        A = torch.zeros(n_omega, dtype=torch.float64, device=device)

        for i_omega, w_val in enumerate(omega):
            z_val = w_val + 1j * eta

            # Evaluate continued fraction from bottom up
            # C(z) = a₀ / (1 + a₁(z-z₀) / (1 + a₂(z-z₁) / (1 + ...)))
            result = 0j
            for i in range(N-1, 0, -1):
                result = a[i] * (z_val - z_iwn[i-1]) / (1.0 + result)

            G_pade = a[0] / (1.0 + result)

            # Spectral function: A(ω) = -(1/π)Im[G(ω + iη)]
            A[i_omega] = -1.0 / math.pi * G_pade.imag

        return A

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
