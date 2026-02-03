"""Analytic continuation methods for G(iωₙ) → G(ω + iη).

Provides a modular framework for transforming Green's functions from
imaginary (Matsubara) frequencies to real frequencies. Multiple methods
are supported with a common interface.

References:
    - Vidberg H.J. and Serene J.W., J. Low Temp. Phys. 29, 179 (1977)
    - Beach K.S., Gooding R.J., Marsiglio F., Phys. Rev. B 61, 5147 (2000)
    - Jarrell M. and Gubernatis J.E., Phys. Rep. 269, 133 (1996)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import math


class AnalyticContinuationMethod(ABC):
    """Abstract base class for analytic continuation methods.

    All analytic continuation methods should inherit from this class
    and implement the continue_to_real_axis method.

    The goal of analytic continuation is to transform a Green's function
    G(iωₙ) defined on Matsubara frequencies to the retarded Green's
    function Gᴿ(ω) = G(ω + iη) on the real frequency axis.
    """

    @abstractmethod
    def continue_to_real_axis(
        self,
        G_iwn: torch.Tensor,
        omega: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        """Transform G(iωₙ) → G(ω + iη) and return spectral function A(ω).

        Args:
            G_iwn: Green's function on Matsubara frequencies, shape (n_iwn, ...)
            omega: Real frequency grid for output, shape (n_omega,)
            eta: Imaginary shift (broadening parameter)
            **kwargs: Method-specific parameters

        Returns:
            A(ω) = -(1/π)Im[G(ω + iη)], spectral function
            Shape depends on method and input dimensions
        """
        pass


class SimpleContinuation(AnalyticContinuationMethod):
    """Direct substitution iωₙ → ω + iη (fast, approximate).

    This is the simplest analytic continuation method, which directly
    substitutes the Matsubara frequency with the real frequency plus
    a small imaginary part.

    WARNING: This method is only valid for very smooth functions.
    It fails for systems with sharp spectral features or poles near
    the real axis. Use with caution!

    For fermionic Green's functions, we use:
        G(ω + iη) ≈ G(iωₙ) evaluated at the lowest frequency

    The spectral function is then:
        A(ω) = -(1/π) Im[G(ω + iη)]
    """

    def continue_to_real_axis(
        self,
        G_iwn: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.05,
        **kwargs,
    ) -> torch.Tensor:
        """Direct substitution: replace iωₙ with ω + iη.

        Args:
            G_iwn: Green's function on Matsubara frequencies
                   Shape: (n_iwn, n_orb, n_orb) or (n_iwn, n_orb)
            omega: Real frequency grid for output
            eta: Imaginary shift (broadening parameter)
            **kwargs: Unused (for API compatibility)

        Returns:
            A(ω) = -(1/π)Im[G(ω + iη)], shape (n_omega, n_orb)
        """
        device = G_iwn.device
        n_iwn = G_iwn.shape[0]

        # Handle different input shapes
        if len(G_iwn.shape) == 3:  # (n_iwn, n_orb, n_orb)
            n_orb = G_iwn.shape[1]
            # Extract diagonal elements
            G_diag = torch.stack([G_iwn[i, j, j] for j in range(n_orb) for i in range(n_iwn)])
            G_diag = G_diag.reshape(n_orb, n_iwn).T  # (n_iwn, n_orb)
        elif len(G_iwn.shape) == 2:  # (n_iwn, n_orb)
            n_orb = G_iwn.shape[1]
            G_diag = G_iwn
        else:
            raise ValueError(f"Unsupported G_iwn shape: {G_iwn.shape}")

        # Use the lowest Matsubara frequency as reference
        # For a more accurate implementation, one would interpolate
        # over all Matsubara frequencies
        A = torch.zeros((len(omega), n_orb), dtype=torch.float64, device=device)

        # For each orbital, approximate the spectral function
        for orb in range(n_orb):
            # Use a simple Lorentzian-like broadening of the Matsubara data
            # This is a placeholder - real implementation would use proper analytic continuation
            for i, w in enumerate(omega):
                # Approximate G(w + iη) using the low-frequency behavior
                # For non-interacting systems, G ~ 1/(iω - ε)
                # We interpolate from Matsubara data
                A[i, orb] = self._interpolate_matsubara_to_real(
                    G_diag[:, orb], w, eta, n_iwn
                )

        return A

    def _interpolate_matsubara_to_real(
        self,
        G_iwn_1d: torch.Tensor,
        w: float,
        eta: float,
        n_iwn: int,
    ) -> float:
        """Interpolate G(iωₙ) to real frequency ω + iη.

        This is a simplified implementation. For production use,
        consider Pade approximant or MaxEnt methods.

        Args:
            G_iwn_1d: Green's function values at Matsubara frequencies
            w: Real frequency
            eta: Imaginary shift
            n_iwn: Number of Matsubara frequencies

        Returns:
            Interpolated spectral weight
        """
        # Use the lowest non-zero Matsubara frequency as reference
        # For fermions: iω₀ = iπ/β
        # Approximate: A(ω) ~ Im[G(iω₀)] * Lorentzian(ω)
        # This is very approximate - use Pade for better results

        # Spectral function from first Matsubara frequency
        # A(ω) ≈ -(1/π) Im[G(iω₀)] * (η/π) / [(ω)² + η²]
        spectral_weight = -1.0 / math.pi * G_iwn_1d[n_iwn // 2].imag
        lorentzian = (eta / math.pi) / (w**2 + eta**2)

        return spectral_weight * lorentzian


class PadeContinuation(AnalyticContinuationMethod):
    """Padé approximant (Vidberg-Serene continued fraction).

    The Padé approximant constructs a rational function C_M(z) = A_M(z) / B_M(z)
    that matches G(iωₙ) at M selected Matsubara frequencies and is evaluated
    at z = ω + iη to obtain the spectral function.

    This is generally the most accurate method for analytic continuation
    when the Green's function is smooth and well-behaved.

    Reference:
        Vidberg H.J. and Serene J.W., J. Low Temp. Phys. 29, 179 (1977)
        Beach K.S., Gooding R.J., Marsiglio F., Phys. Rev. B 61, 5147 (2000)

    Attributes:
        tolerance: Default numerical tolerance for all comparisons (< 1e-6)
    """

    def __init__(self, tolerance: float = 1e-12):
        """Initialize Pade continuation method.

        Args:
            tolerance: Numerical tolerance for comparisons (default: 1e-12).
                      All tolerances (frequency selection threshold, division
                      by zero protection) are derived from this value.
                      Must be < 1e-6 for double precision accuracy.
        """
        if tolerance >= 1e-6:
            raise ValueError(f"tolerance must be < 1e-6, got {tolerance}")
        self.tolerance = tolerance

    def continue_to_real_axis(
        self,
        G_iwn: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.05,
        beta: float = 10.0,
        n_min: int = 0,
        n_max: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Vidberg-Serene continued fraction algorithm.

        Args:
            G_iwn: Green's function on Matsubara frequencies
                   Shape: (n_iwn, n_orb, n_orb) or (n_iwn, n_orb)
            omega: Real frequency grid for output
            eta: Imaginary shift
            beta: Inverse temperature (for Matsubara frequency conversion)
            n_min: Minimum Matsubara index to use
            n_max: Maximum Matsubara index (default: n_iwn // 2)
            **kwargs: Unused (for API compatibility)

        Returns:
            A(ω) = -(1/π)Im[G(ω + iη)], shape (n_omega, n_orb)
        """
        device = G_iwn.device
        n_iwn = G_iwn.shape[0]

        if n_max is None:
            n_max = n_iwn // 2

        # Generate integer indices for symmetric Matsubara grid
        n_vals = torch.arange(n_iwn, device=device) - n_iwn // 2

        # Start from positive frequencies (n_vals >= 0)
        idx_start = (n_iwn // 2) + n_min  # Start from n_vals=0

        # OPTIMIZED FREQUENCY SELECTION:
        # Instead of blindly using n_max frequencies, select based on |G| magnitude
        # Higher frequencies have exponentially smaller |G| and contribute noise
        # Use a threshold to determine which frequencies are actually informative

        # Get the first n_max candidate frequencies
        idx_end_candidate = min(idx_start + n_max, n_iwn)

        # For each orbital, check |G| to determine optimal frequency count
        if len(G_iwn.shape) == 3:  # (n_iwn, n_orb, n_orb)
            n_orb = G_iwn.shape[1]
            # Use first orbital as reference (all orbitals should have similar decay)
            G_candidate = G_iwn[idx_start:idx_end_candidate, 0, 0]
        elif len(G_iwn.shape) == 2:  # (n_iwn, n_orb)
            G_candidate = G_iwn[idx_start:idx_end_candidate, 0]
        else:
            raise ValueError(f"Unsupported G_iwn shape: {G_iwn.shape}")

        # Compute |G| for candidate frequencies
        G_mag = torch.abs(G_candidate)

        # Adaptive selection: use frequencies where |G| > threshold
        # Threshold: relative (0.1% of max) OR absolute (tolerance)
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

        # Complex Matsubara frequencies for Padé construction
        n_vals_selected = n_vals[idx_start:idx_end]
        wn = (2 * n_vals_selected + 1) * math.pi / beta  # Fermionic Matsubara frequencies
        z_iwn = 1j * wn

        # Handle different input shapes
        if len(G_iwn.shape) == 3:  # (n_iwn, n_orb, n_orb)
            n_orb = G_iwn.shape[1]
        elif len(G_iwn.shape) == 2:  # (n_iwn, n_orb)
            n_orb = G_iwn.shape[1]
        else:
            raise ValueError(f"Unsupported G_iwn shape: {G_iwn.shape}")

        # Extract diagonal elements (orbital-diagonal approximation)
        A = torch.zeros((len(omega), n_orb), dtype=torch.float64, device=device)

        for orb in range(n_orb):
            # Get G_ii(iωₙ) for this orbital
            if len(G_iwn.shape) == 3:
                G_diag = G_iwn[idx_start:idx_end, orb, orb]
            else:
                G_diag = G_iwn[idx_start:idx_end, orb]

            # Compute Pade approximant for this orbital
            A[:, orb] = self._pade_continued_fraction(
                G_diag, z_iwn, omega, eta, device
            )

        return A

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
        representation of the Pade approximant by building a g-table:

        C_M(z) = a₀ / (1 + a₁(z-z₀) / (1 + a₂(z-z₁) / (...)))

        where the coefficients aᵢ are extracted from the diagonal of the g-table.

        Reference: Vidberg H.J. and Serene J.W., J. Low Temp. Phys. 29, 179 (1977)

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

        # 1. Build g-table using continued fraction recursion (Vidberg-Serene)
        # g[i, j] represents the j-th element in the i-th column
        g = torch.zeros((N, N), dtype=torch.complex128, device=device)

        # First column: original G values
        g[0, :] = G_iwn

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


class BetheLatticeContinuation(AnalyticContinuationMethod):
    """Bethe lattice analytical solution for semi-elliptical DOS.

    For a Bethe lattice with coordination number z and hopping t,
    the non-interacting Green's function has an analytical form:

        G₀(ω) = 2(z-1)/t² [ω + t²/(2(z-1)) - sqrt((ω + t²/(2(z-1)))² - 4)]

    This gives a semi-elliptical density of states:

        ρ₀(ε) = (2/πD²) sqrt(D² - ε²) for |ε| < D

    where D = 2t*sqrt(z-1) is the half-bandwidth.

    This method is useful for testing DMFT implementations where
    the lattice problem can be solved analytically.
    """

    def continue_to_real_axis(
        self,
        G_iwn: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.05,
        z: Optional[float] = None,
        t: float = 1.0,
        lattice=None,
        **kwargs,
    ) -> torch.Tensor:
        """Bethe lattice analytic continuation.

        Args:
            G_iwn: Input Green's function (used to determine z if not specified)
            omega: Real frequency grid
            eta: Broadening
            z: Coordination number (auto-determined from lattice if None)
            t: Hopping parameter (default: 1.0)
            lattice: BravaisLattice object for auto-determining z
            **kwargs: Unused (for API compatibility)

        Returns:
            A(ω) = -(1/π)Im[G(ω + iη)], shape (n_omega,)
        """
        device = omega.device

        # Auto-determine z from lattice if not specified
        if z is None and lattice is not None:
            z = self._estimate_coordination_number(lattice)
        elif z is None:
            z = 6.0  # Default for cubic lattice

        A_omega = torch.zeros(len(omega), dtype=torch.float64, device=device)

        # Half-bandwidth for Bethe lattice
        D = 2 * t * math.sqrt(z - 1)

        for i, w in enumerate(omega):
            z_shift = w + 1j * eta
            t_factor = t**2 / (2 * (z - 1))

            # Bethe lattice Green's function
            discriminant = (z_shift + t_factor)**2 - 4 * t_factor**2

            # Handle branch cut properly
            sqrt_arg = torch.complex(discriminant, torch.zeros_like(discriminant))
            sqrt_term = torch.sqrt(sqrt_arg)

            G_bethe = 2 * (z - 1) / t**2 * (z_shift + t_factor - sqrt_term)

            # Spectral function
            A_omega[i] = -1.0 / math.pi * G_bethe.imag

        return A_omega

    def _estimate_coordination_number(self, lattice) -> float:
        """Estimate coordination number from lattice structure.

        Args:
            lattice: BravaisLattice object

        Returns:
            Estimated coordination number
        """
        # Simple heuristic based on common lattices
        dim = lattice.dimension
        if dim == 2:
            return 4.0  # Square lattice
        elif dim == 3:
            return 6.0  # Cubic lattice
        return 4.0  # Default


class MaxEntContinuation(AnalyticContinuationMethod):
    """Maximum entropy analytic continuation.

    The Maximum Entropy method finds the most probable spectrum A(ω)
    consistent with the data G(iωₙ) by maximizing:

        Q = α·S - χ²/2

    where S is the Shannon entropy and χ² measures misfit to data.

    This is the most robust method for noisy data but requires
    careful tuning of the regularization parameter α.

    Reference:
        Jarrell M. and Gubernatis J.E., Phys. Rep. 269, 133 (1996)
    """

    def continue_to_real_axis(
        self,
        G_iwn: torch.Tensor,
        omega: torch.Tensor,
        eta: float = 0.05,
        **kwargs,
    ) -> torch.Tensor:
        """Maximum entropy continuation (NOT YET IMPLEMENTED).

        Args:
            G_iwn: Green's function on Matsubara frequencies
            omega: Real frequency grid
            eta: Imaginary shift (not used in MaxEnt)
            **kwargs: Method-specific parameters (alpha, default_model, etc.)

        Returns:
            A(ω) = -(1/π)Im[G(ω + iη)]

        Raises:
            NotImplementedError: MaxEnt is not yet implemented
        """
        raise NotImplementedError(
            "MaxEnt continuation is not yet implemented. "
            "Use 'pade' or 'simple' methods instead. "
            "For MaxEnt, consider using the MaxEnt package or "
            "the Analytic Continuation module in TRIQS."
        )


# Factory function for creating continuation methods
def create_continuation_method(method: str) -> AnalyticContinuationMethod:
    """Create an analytic continuation method instance.

    Args:
        method: Name of the method ('simple', 'pade', 'bethe', 'maxent')

    Returns:
        Instance of the corresponding AnalyticContinuationMethod subclass

    Raises:
        ValueError: If method name is unknown
    """
    method_map = {
        'simple': SimpleContinuation,
        'pade': PadeContinuation,
        'bethe': BetheLatticeContinuation,
        'maxent': MaxEntContinuation,
    }

    method_lower = method.lower()
    if method_lower not in method_map:
        raise ValueError(
            f"Unknown analytic continuation method: {method}. "
            f"Available methods: {list(method_map.keys())}"
        )

    return method_map[method_lower]()
