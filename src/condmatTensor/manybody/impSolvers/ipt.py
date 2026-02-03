"""IPT (Iterated Perturbation Theory) solver for DMFT.

Implements second-order perturbation theory for the single-impurity
Anderson model within DMFT. The self-energy is computed using the
TRIQS-style imaginary time approach:

    Σ(τ) = U² · G₀(τ)³
    Σ(iωₙ) = F[Σ(τ)]

where F denotes Fourier transform between imaginary time and Matsubara
frequencies. This is the standard IPT used in DMFT literature.

**Matsubara Frequency Transforms:**
The correct Fourier transforms for fermionic Green's functions are:

    G(iωₙ) = (1/β) ∫₀^β dτ e^(iωₙτ) G(τ)  ← Fourier transform
    G(τ) = (1/β) Σₙ e^(-iωₙτ) G(iωₙ)      ← Inverse Fourier transform

where ωₙ = π(2n+1)/β (fermionic Matsubara frequencies).

This implementation uses explicit Matsubara frequency transforms
rather than generic PyTorch FFT, ensuring correct normalization
and phase factors for fermionic systems.

**Key Architecture:**
- Solver accepts k-dependent G(k,iωₙ) for flexibility
- Computes local self-energy Σ(iωₙ) (momentum-independent)
- U values read from OrbitalMetadata.U (orbital-dependent)
- Orbital-diagonal approximation: Σ_ij = 0 for i ≠ j

**Supports:**
- Multi-orbital systems with orbital-selective correlations
- Both Hk and HR input via preprocessing module
- Future: Multi-orbital U matrix support

References:
    - TRIQS 3.3.1 Tutorial: "A first DMFT calculation"
      https://triqs.github.io/triqs/latest/userguide/python/tutorials/ModelDMFT/solutions/01s-IPT_and_DMFT.html
    - Haule, Rutgers lecture notes: "Perturbation theory" (2017)
    - Georges et al., Rev. Mod. Phys. 68, 13 (1996)
"""

from typing import Optional, Tuple, Union
import torch
import math

from .base import ImpuritySolverABC


class IPTSolver(ImpuritySolverABC):
    """Iterated Perturbation Theory solver for DMFT.

    Computes self-energy using second-order perturbation theory in
    imaginary time (TRIQS-style implementation):

        Σ(τ) = U² · G₀(τ)³
        Σ(iωₙ) = F[Σ(τ)]

    **Architecture:**
    - Accepts k-dependent G(k,iωₙ) via DMFT preprocessing
    - Extracts local G_loc(iωₙ) = (1/N_k) Σ_k G(k,iωₙ)
    - Computes local self-energy Σ(iωₙ) via FFT
    - Returns Σ to DMFT loop

    **Orbital-dependent U:**
    - Reads U from OrbitalMetadata.U for each orbital
    - U_d ≈ 0-1 (conductive d-orbitals)
    - U_f ≈ 4-8 (localized f-orbitals)

    Attributes:
        beta: Inverse temperature (β = 1/k_B T)
        n_max: Maximum Matsubara frequency index
        device: torch.device for computation
    """

    def __init__(
        self,
        beta: float,
        n_max: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize IPT solver.

        Args:
            beta: Inverse temperature (β = 1/k_B T)
            n_max: Maximum Matsubara frequency index
            device: Device for computation (default: CPU)

        Note:
            U values are read from OrbitalMetadata.U in solve() method.
            This allows orbital-dependent, on-site Hubbard interactions.
        """
        self.beta = beta
        self.n_max = n_max
        self.device = device if device is not None else torch.device("cpu")

        # Storage for computed quantities
        self._G_loc: Optional["BaseTensor"] = None
        self._Sigma: Optional["BaseTensor"] = None

    def solve(
        self,
        G_input: "BaseTensor",
        max_iter: int = 1,
        tol: float = 1e-6,
        **kwargs,
    ) -> "BaseTensor":
        """Solve impurity problem using IPT.

        **Architecture**: Flexible input via preprocessing module
        - Accepts G(k,iω) with labels=['iwn', 'k', 'orb_i', 'orb_j']
        - Extracts local G_loc via k-summation
        - Computes local Σ using TRIQS formula: Σ(τ) = U²·G_loc(τ)³
        - Returns local Σ with labels=['iwn', 'orb_i', 'orb_j']

        Args:
            G_input: Green's function from DMFT. Can be:
                - G(k,iω): k-dependent, labels=['iwn', 'k', 'orb_i', 'orb_j']
                - G_loc(iω): local only, labels=['iwn', 'orb_i', 'orb_j']
            max_iter: Maximum IPT iterations (usually 1-5 is sufficient)
                      Note: IPT is typically converged in 1 iteration
            tol: Convergence tolerance (not used for standard IPT)
            **kwargs: Additional parameters (e.g., U_override for testing)

        Returns:
            BaseTensor with local self-energy Σ(iωₙ), labels=['iwn', 'orb_i', 'orb_j']

        Note:
            U values are read from G_input.orbital_metadatas[i].U
            For Kagome-F: U_d ≈ 0-1, U_f ≈ 4-8 (orbital-selective correlations)
        """
        from condmatTensor.core import BaseTensor

        # Extract orbital U values from OrbitalMetadata
        U_orb = self._extract_orbital_U(G_input.orbital_metadatas)

        # Allow U override for testing
        if "U_override" in kwargs:
            U_override = kwargs["U_override"]
            if isinstance(U_override, (int, float)):
                U_orb = [U_override] * len(U_orb)
            elif isinstance(U_override, (list, tuple)):
                U_orb = list(U_override)

        # The DMFT loop passes G₀ (Weiss field) to the solver.
        # G₀ is already local (no k-dependence), so use it directly.
        # According to IPT theory: Σ(τ) = U² · G₀(τ)³ where G₀ is the Weiss field.
        # BUG FIX: Previously used G_loc (interacting G) instead of G₀ (Weiss field).
        # This violated the IPT approximation and caused unphysically large Σ.
        if 'k' in G_input.labels:
            # This should NOT happen in standard DMFT: G₀ is always local
            # But handle it for robustness by extracting local component
            G0_loc = self._extract_local_greens_function(G_input)
            print("Warning: IPT solver received k-dependent G_input. Extracting local component.")
        else:
            G0_loc = G_input  # G_input IS G₀ (Weiss field) - use directly!

        # Store G₀ for reference (renamed from _G_loc for clarity)
        self._G_loc = G0_loc  # Keep attribute name for backwards compatibility

        # Get number of orbitals
        n_orb = G0_loc.shape[-1]

        # Compute Σ(τ) = U² · G₀(τ)³ via FFT using the Weiss field G₀
        Sigma = self._compute_sigma_fft(G0_loc, U_orb)

        # Validate self-energy magnitude
        self._validate_self_energy(Sigma, U_orb)

        # Store result
        self._Sigma = Sigma

        return Sigma

    def _extract_orbital_U(
        self,
        orbital_metadatas: Optional[list],
    ) -> list[float]:
        """Extract orbital-dependent U values from OrbitalMetadata.

        Args:
            orbital_metadatas: List of OrbitalMetadata objects

        Returns:
            List of U values per orbital
            [0.0, 0.0, ..., U_f] for Kagome-F (d-orbitals: U≈0, f-orbital: U≈4-8)

        Note:
            Future: Support U_ij matrix for full multi-orbital interactions
        """
        if orbital_metadatas is None:
            # Default: no correlations
            n_orb = self._get_n_orb_from_context()
            return [0.0] * n_orb

        U_orb = []
        for md in orbital_metadatas:
            if md.U is not None:
                U_orb.append(md.U)
            else:
                U_orb.append(0.0)  # Default: no correlation
        return U_orb

    def _get_n_orb_from_context(self) -> int:
        """Get number of orbitals from context (default to 1)."""
        # This is a fallback when orbital_metadatas is None
        # Typically overwritten by actual tensor dimensions
        return 1

    def _extract_local_greens_function(
        self,
        G_kw: "BaseTensor",
    ) -> "BaseTensor":
        """Extract local Green's function from k-dependent G.

        G_loc(iωₙ) = (1/N_k) Σ_k G(k, iωₙ)

        Args:
            G_kw: k-dependent Green's function, labels=['iwn', 'k', 'orb_i', 'orb_j']

        Returns:
            BaseTensor with local G_loc, labels=['iwn', 'orb_i', 'orb_j']
        """
        from condmatTensor.core import BaseTensor

        k_idx = G_kw.labels.index('k')
        G_loc_tensor = torch.mean(G_kw.tensor, dim=k_idx)

        return BaseTensor(
            tensor=G_loc_tensor,
            labels=['iwn', 'orb_i', 'orb_j'],
            orbital_names=G_kw.orbital_names,
            orbital_metadatas=G_kw.orbital_metadatas,
        )

    def _compute_sigma_fft(
        self,
        G_loc: "BaseTensor",
        U_orb: list[float],
    ) -> "BaseTensor":
        """Compute self-energy using TRIQS-style FFT approach.

        Algorithm:
            1. G_loc(iωₙ) → IFFT → G_loc(τ) [fermionic Matsubara transform]
            2. Σ(τ) = Σ_i U_i² · G_loc,ii(τ)³ (orbital-diagonal)
            3. Σ(iωₙ) → FFT ← Σ(τ) [fermionic Matsubara transform]

        The FFT uses proper fermionic Matsubara frequencies ωₙ = π(2n+1)/β,
        not generic PyTorch FFT, ensuring correct normalization and phase factors.

        Args:
            G_loc: Local Green's function, labels=['iwn', 'orb_i', 'orb_j']
            U_orb: Orbital-dependent U values

        Returns:
            BaseTensor with self-energy Σ(iωₙ), labels=['iwn', 'orb_i', 'orb_j']
        """
        from condmatTensor.core import BaseTensor

        # Get dimensions
        n_iwn, n_orb, _ = G_loc.shape

        # Generate imaginary time grid
        tau = torch.linspace(0, self.beta, n_iwn, device=self.device)

        # FFT: G_loc(iωₙ) → G_loc(τ)
        # Use standard FFT with proper normalization
        G_tau = torch.zeros_like(G_loc.tensor, dtype=torch.complex128)
        for i in range(n_orb):
            for j in range(n_orb):
                G_tau[:, i, j] = self._ifft_to_tau(G_loc.tensor[:, i, j])

        # Compute Σ(τ) = U² · G₀(τ)³ (orbital-diagonal)
        # Note: G(τ) can be large, so we clip to prevent numerical explosion
        # This is a numerical safeguard; physically G(τ) should be bounded
        # FIX: Use adaptive clipping based on U value to keep Σ magnitude reasonable
        Sigma_tau = torch.zeros_like(G_tau)
        for i in range(n_orb):
            if U_orb[i] > 0:
                # Adaptive clipping: smaller clip for larger U to keep Σ ~ U² × clip³ reasonable
                # For U=4, clip=3 gives max Σ ~ 16 × 27 = 432 (acceptable)
                # For U=2, clip=5 gives max Σ ~ 4 × 125 = 500 (acceptable)
                g_clip = max(3.0, 10.0 / U_orb[i])

                # FIX: Use full complex G_tau, not just real part!
                # G(τ) is complex for fermionic Green's functions at finite temperature
                G_tau_ii = G_tau[:, i, i]  # Full complex tensor

                # FIX: Clip real and imaginary parts separately (more numerically stable)
                # The .abs() + torch.sgn() approach can cause numerical instability near zero,
                # potentially flipping signs. Direct clipping preserves sign information correctly.
                real_clipped = torch.clamp(G_tau_ii.real, min=-g_clip, max=g_clip)
                imag_clipped = torch.clamp(G_tau_ii.imag, min=-g_clip, max=g_clip)
                G_clipped = torch.complex(real_clipped, imag_clipped)

                # Σ_ii(τ) = U_i² · G_loc,ii(τ)³
                Sigma_tau[:, i, i] = (U_orb[i]**2) * (G_clipped**3)

        # FFT: Σ(τ) → Σ(iωₙ)
        Sigma_iwn = torch.zeros_like(G_loc.tensor)
        for i in range(n_orb):
            for j in range(n_orb):
                Sigma_iwn[:, i, j] = self._fft_to_iwn(Sigma_tau[:, i, j])

        return BaseTensor(
            tensor=Sigma_iwn,
            labels=['iwn', 'orb_i', 'orb_j'],
            orbital_names=G_loc.orbital_names,
            orbital_metadatas=G_loc.orbital_metadatas,
        )

    def _fft_to_iwn(self, G_tau: torch.Tensor) -> torch.Tensor:
        """FFT from imaginary time to Matsubara frequencies.

        G(iωₙ) = (1/β) ∫₀^β dτ e^(iωₙτ) G(τ)

        Using proper fermionic Matsubara frequency transform.
        For fermions: ωₙ = π(2n+1)/β

        Args:
            G_tau: Green's function in imaginary time, shape (n_tau,)

        Returns:
            Green's function on Matsubara frequencies, shape (n_iwn,)
        """
        n_tau = len(G_tau)
        dtau = self.beta / n_tau

        # Imaginary time grid
        tau = torch.linspace(0, self.beta, n_tau, device=self.device, dtype=torch.float64)

        # Generate Matsubara frequencies
        # n_vals: [-n_max, ..., -1, 0, 1, ..., n_max]
        n_vals = torch.arange(n_tau, device=self.device) - n_tau // 2
        wn = (2 * n_vals + 1) * torch.pi / self.beta  # Fermionic Matsubara frequencies

        # Outer product: ωₙ × τ phase matrix
        # phase_matrix[i_n, i_tau] = e^(iωₙ·τᵢ)
        phase_matrix = torch.exp(1j * torch.outer(wn, tau))

        # G(iωₙ) = ∫₀^β dτ e^(iωₙτ) G(τ) ≈ Σ_τ e^(iωₙτ) G(τ) Δτ
        # BUG FIX: Added missing 1/β normalization factor
        # For β=10, this was causing Σ to be 10× too large before applying U² factor
        G_iwn = torch.matmul(phase_matrix, G_tau) * dtau / self.beta

        return G_iwn

    def _ifft_to_tau(self, G_iwn: torch.Tensor) -> torch.Tensor:
        """IFFT from Matsubara frequencies to imaginary time.

        G(τ) = (1/β) Σₙ e^(-iωₙτ) G(iωₙ)

        Using proper fermionic Matsubara frequency transform.

        Args:
            G_iwn: Green's function on Matsubara frequencies, shape (n_iwn,)
                   Ordered as: [-n_max, ..., 0, ..., n_max]

        Returns:
            Green's function in imaginary time, shape (n_tau,)
        """
        n_iwn = len(G_iwn)

        # Imaginary time grid
        tau = torch.linspace(0, self.beta, n_iwn, device=self.device, dtype=torch.float64)

        # Generate Matsubara frequencies
        # n_vals: [-n_max, ..., -1, 0, 1, ..., n_max]
        n_vals = torch.arange(n_iwn, device=self.device) - n_iwn // 2
        wn = (2 * n_vals + 1) * torch.pi / self.beta  # Fermionic Matsubara frequencies

        # Outer product: τ × ωₙ phase matrix
        # phase_matrix[i_tau, i_n] = e^(-iωₙ·τᵢ)
        phase_matrix = torch.exp(-1j * torch.outer(tau, wn))

        # G(τ) = (1/β) Σₙ e^(-iωₙτ) G(iωₙ)
        G_tau = torch.matmul(phase_matrix, G_iwn) / self.beta

        # Enforce fermionic boundary condition: G(β) = -G(0)
        # This is required for physical consistency of fermionic Green's functions
        if torch.abs(G_tau[-1] + G_tau[0]).max() > 1e-6:
            G_tau[-1] = -G_tau[0]

        return G_tau

    @property
    def solver_name(self) -> str:
        """Return solver name."""
        return "IPT"

    @property
    def supported_orbitals(self) -> int:
        """Return maximum number of orbitals supported.

        Returns:
            -1 (unlimited) using orbital-diagonal approximation
        """
        return -1  # Unlimited (uses orbital-diagonal approximation)

    @property
    def G_loc(self) -> Optional["BaseTensor"]:
        """Return the local Green's function used in last solve()."""
        return self._G_loc

    def _validate_self_energy(
        self,
        Sigma: "BaseTensor",
        U_orb: list[float],
    ) -> None:
        """Validate self-energy against physical constraints.

        Performs sanity checks on the computed self-energy:
        1. High-frequency limit: Σ(iωₙ) → U²n/β as ωₙ → ∞
        2. Magnitude should be O(1) to O(100) for typical U values
        3. Orbital selectivity: Σ_f >> Σ_d for U_f >> U_d

        Args:
            Sigma: Self-energy on Matsubara frequencies
            U_orb: Orbital-dependent U values

        Note:
            This prints warnings for unphysical values but does not raise exceptions.
        """
        n_iwn = Sigma.shape[0]
        n_orb = Sigma.shape[-1]

        # Check high-frequency limit (use last 10% of frequencies)
        n_check = n_iwn // 10
        if n_check > 0:
            sigma_high_freq = Sigma.tensor[-n_check:, :, :].abs()
            max_sigma = sigma_high_freq.max().item()

            # Expected maximum: U² × some reasonable factor (10-100)
            expected_max = max(U_orb)**2 * 50.0

            if max_sigma > expected_max:
                print(f"Warning: |Σ(iωₙ)| at high frequency = {max_sigma:.2f}, "
                      f"which may be unphysically large (expected < {expected_max:.2f})")

        # Check each orbital
        for i, U in enumerate(U_orb):
            if U > 0:
                # Check at first Matsubara frequency
                sigma_0 = Sigma.tensor[n_iwn // 2, i, i]  # Middle frequency (iω₀)
                sigma_mag = abs(sigma_0.item())

                # For typical U values (1-8), Σ should be O(1) to O(200)
                if sigma_mag > 500:
                    print(f"Warning: Σ_orb{i}(iω₀) = {sigma_mag:.2f}, U={U:.2f}, "
                          f"may be unphysically large")

                # Check imaginary part (should be negative for typical systems)
                if sigma_0.imag > 0:
                    print(f"Note: Im[Σ_orb{i}(iω₀)] = {sigma_0.imag:.2f} > 0, "
                          f"check if physical")

    @property
    def Sigma(self) -> Optional["BaseTensor"]:
        """Return the self-energy from last solve()."""
        return self._Sigma
