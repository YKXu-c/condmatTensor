"""Local magnetic models: Kondo lattice, spin-fermion, H = H₀ + J@S.

This module implements models where electrons couple to local magnetic moments
via exchange interaction. The general Hamiltonian is:

    H = H₀ + J * Σ_i S_i · σ_i

where H₀ is the non-interacting tight-binding Hamiltonian, J is the exchange
coupling strength, S_i are local moments (classical 3-vectors or quantum spins),
and σ_i are Pauli matrices acting on electron spin.

**Spinor Convention:**
This module uses the spinor approach where spin is encoded in the orbital
index rather than a separate dimension:
- Orbital ordering: [orb_0_up, orb_0_down, orb_1_up, orb_1_down, ...]
- Hamiltonian labels: ['k', 'orb_i', 'orb_j'] where orbitals include spin
- Pauli matrices are embedded as 2×2 blocks in the hopping matrix

**Benefits of spinor approach:**
1. Natural for spin-orbit coupling (mixes ↑↓ states)
2. Compatible with Wannier90 spinor format
3. Easy to extend to spin-1, pseudospin, etc.
4. Magnetic exchange is simple on-site term

References:
    - "Two-Impurity Kondo Physics" - R. Rhyno, UCSD (2017)
    - "The Kondo Lattice Model" - M. Lavagna, hal-01896229 (1998)
    - "Ferromagnetic Kondo-lattice Model" - W. Nolting, PSSb 250 (2003)
    - "Kondo s-d Model" - arXiv:2303.18235 (2023)

LEVEL 4 of the 10-level architecture.
"""

from typing import Optional, Tuple, Union, List
import torch
import math


class LocalMagneticModel:
    """H = H₀ + J@S model solver using spinor approach.

    This class implements local magnetic moment models where electrons couple
    to classical or quantum local moments via exchange interaction.

    **Spinor Convention:**
    Each orbital becomes a spinor: [orb_0_up, orb_0_down, orb_1_up, orb_1_down, ...]
    Hamiltonian shape: (N_k, 2*N_orb, 2*N_orb) with labels=['k', 'orb_i', 'orb_j']

    **Pauli Matrices (embedded in orbital space):**
        σx = [[0, 1],     σy = [[0, -i],    σz = [[1, 0],
             [1, 0]]           [i,  0]]          [0, -1]]

    **Magnetic Exchange Term:**
        H_J@S[i] = J * (Sx*σx + Sy*σy + Sz*σz)

        In the 2×2 spinor block for orbital i:
        [[J*Sz,      J*(Sx - i*Sy)],
         [J*(Sx + i*Sy),  -J*Sz    ]]

    Attributes:
        H0: Non-interacting spinful Hamiltonian
        J: Exchange coupling strength
        S: Local spin configuration (N_sites, 3) for (Sx, Sy, Sz)
        Sigma: Self-energy from magnetic interactions
    """

    def __init__(
        self,
        H0: Optional["BaseTensor"] = None,
        J: float = 1.0,
        S_init: Optional[torch.Tensor] = None,
    ):
        """Initialize LocalMagneticModel.

        Args:
            H0: Non-interacting spinful Hamiltonian with shape (N_k, 2*N_orb, 2*N_orb)
                Labels should be ['k', 'orb_i', 'orb_j']
            J: Exchange coupling strength
            S_init: Initial spin configuration, shape (N_sites, 3) for (Sx, Sy, Sz)
                    If None, initialized to random unit vectors
        """
        from condmatTensor.core import BaseTensor

        self.H0 = H0
        self.J = J

        # Initialize spin configuration
        if S_init is not None:
            self.S = S_init
        else:
            self.S = None

        # Lazy import for SelfEnergy
        self._SelfEnergy = None

    def _get_self_energy_class(self):
        """Lazy import of SelfEnergy."""
        if self._SelfEnergy is None:
            from condmatTensor.manybody.preprocessing import SelfEnergy
            self._SelfEnergy = SelfEnergy
        return self._SelfEnergy()

    def build_spinful_hamiltonian(
        self,
        H0_spinless: "BaseTensor",
        soc_tensor: Optional["BaseTensor"] = None,
        lattice: Optional["BravaisLattice"] = None,
        device: Optional[torch.device] = None,
    ) -> "BaseTensor":
        """Build spinful Hamiltonian from spinless Hamiltonian.

        Converts H0 (N_orb × N_orb) to H0_spinful (2*N_orb × 2*N_orb)
        using spinor ordering: [orb_0_up, orb_0_down, orb_1_up, orb_1_down, ...]

        **Without SOC:** Diagonal in spin space
            H_spinful = diag(H0, H0)  # Same H0 for ↑ and ↓

        **With SOC:** Off-diagonal spin terms from soc_tensor
            H_spinful = [[H0,      H_SOC],
                         [H_SOC†,  H0    ]]

        Args:
            H0_spinless: BaseTensor with shape (N_k, N_orb, N_orb)
                         Labels should contain 'k' dimension
            soc_tensor: Optional spin-orbit coupling tensor (N_k, N_orb, N_orb)
                        Provides off-diagonal spin terms
            lattice: Optional BravaisLattice for per-site orbital info.
                     If None, assumes uniform spin doubling (2× per orbital).
            device: Device for computation (default: matches H0_spinless.device)

        Returns:
            BaseTensor with shape (N_k, 2*N_orb, 2*N_orb)
            Labels: ['k', 'orb_i', 'orb_j'] where orbitals include spin
        """
        from condmatTensor.core import BaseTensor

        if device is None:
            device = H0_spinless.tensor.device

        # Verify H0_spinless has k dimension
        if "k" not in H0_spinless.labels:
            raise ValueError("H0_spinless must have 'k' in labels")
        k_idx = H0_spinless.labels.index("k")

        # Get dimensions
        tensor = H0_spinless.tensor
        N_k = tensor.shape[k_idx]
        N_orb = H0_spinless.shape[-1]

        # Permute to (N_k, N_orb, N_orb) if needed
        if k_idx != 0:
            perm = [k_idx] + [i for i in range(len(H0_spinless.labels)) if i != k_idx]
            tensor = tensor.permute(perm)

        # Calculate spinful dimensions
        if lattice is None:
            # Old behavior: uniform 2× spin doubling for all orbitals
            N_spinful = 2 * N_orb
            offsets = None
        else:
            # New behavior: per-site spin doubling based on lattice.num_orbitals
            offsets = lattice.orbital_offsets()
            N_spinful = sum(2 * n for n in lattice.num_orbitals)

        # Initialize spinful Hamiltonian
        H_spinful = torch.zeros(
            (N_k, N_spinful, N_spinful),
            dtype=torch.complex128,
            device=device,
        )

        if lattice is None:
            # Fill diagonal blocks (spin-conserving)
            # For each orbital pair (i, j):
            #   H_spinful[:, 2*i, 2*j] = H0[:, i, j]     (↑↑)
            #   H_spinful[:, 2*i+1, 2*j+1] = H0[:, i, j] (↓↓)
            for i in range(N_orb):
                for j in range(N_orb):
                    H_spinful[:, 2*i, 2*j] = tensor[:, i, j]           # ↑↑
                    H_spinful[:, 2*i+1, 2*j+1] = tensor[:, i, j]       # ↓↓
        else:
            # Apply spin doubling per-site
            # CRITICAL FIX: Must copy ALL hopping (both intra-site AND inter-site)
            #
            # Reference: "Kondo lattice s-d model J exchange coupling"
            # - Hopping exists between different lattice sites (e.g., Kagome A↔B, A↔C, B↔C)
            # - Spinful Hamiltonian must preserve all k-dependent hopping terms
            # - Sources:
            #   * "Kondo breakdown in multi-orbital Anderson lattices" - Eickhoff et al., SciPost Phys. 17, 069 (2024)
            #     https://scipost.org/SciPostPhys.17.3.069/pdf
            #   * "Bilayer Kondo lattice models" - arXiv:2005.11342 (2020)
            #     https://arxiv.org/pdf/2005.11342
            #   * "Orbital-selective orthogonal metal transition" - Phys. Rev. B 86, 115113 (2012)
            #     https://link.aps.org/doi/10.1103/PhysRevB.86.115113

            # Compute cumulative spinful offsets for each site
            spinful_offsets = [0]
            for n_orb_site in lattice.num_orbitals:
                spinful_offsets.append(spinful_offsets[-1] + 2 * n_orb_site)

            # Iterate over ALL pairs of sites (including inter-site hopping)
            for src_site_idx, n_src_orb in enumerate(lattice.num_orbitals):
                src_offset = offsets[src_site_idx]
                src_spinful_offset = spinful_offsets[src_site_idx]

                for dst_site_idx, n_dst_orb in enumerate(lattice.num_orbitals):
                    dst_offset = offsets[dst_site_idx]
                    dst_spinful_offset = spinful_offsets[dst_site_idx]

                    # Copy ALL hopping from src_site orbitals to dst_site orbitals
                    # This includes both intra-site (src == dst) and inter-site (src != dst) hopping
                    for i in range(n_src_orb):
                        for j in range(n_dst_orb):
                            # Spin-conserving hopping: ↑↑ and ↓↓
                            H_spinful[:, dst_spinful_offset + 2*i, src_spinful_offset + 2*j] = tensor[:, dst_offset + i, src_offset + j]      # ↑↑
                            H_spinful[:, dst_spinful_offset + 2*i + 1, src_spinful_offset + 2*j + 1] = tensor[:, dst_offset + i, src_offset + j]  # ↓↓

        # Add SOC if provided (off-diagonal spin terms)
        if soc_tensor is not None:
            soc = soc_tensor.tensor
            if soc.dim() == 2:
                soc = soc.unsqueeze(0).expand(N_k, -1, -1)

            # SOC adds mixing between ↑ and ↓
            if lattice is None:
                H_spinful[:, :N_orb, N_orb:] = soc        # ↑↓
                H_spinful[:, N_orb:, :N_orb] = torch.conj(soc).mT  # ↓↑
            else:
                # Per-site SOC (future enhancement)
                pass

        # Create orbital names with spin suffixes
        orbital_names = None
        orbital_metadatas = None
        if H0_spinless.orbital_names is not None:
            orbital_names = []
            orbital_metadatas = []
            if lattice is None:
                for name in H0_spinless.orbital_names:
                    orbital_names.append(f"{name}_up")
                    orbital_names.append(f"{name}_down")
                # Create orbital metadatas with spin if source has metadatas
                if H0_spinless.orbital_metadatas is not None:
                    from condmatTensor.core.types import OrbitalMetadata
                    for md in H0_spinless.orbital_metadatas:
                        md_up = OrbitalMetadata(
                            site=md.site, orb=md.orb, spin='up',
                            local=md.local, U=md.U
                        )
                        md_down = OrbitalMetadata(
                            site=md.site, orb=md.orb, spin='down',
                            local=md.local, U=md.U
                        )
                        orbital_metadatas.extend([md_up, md_down])
            else:
                # Per-site orbital naming
                for site_idx, n_orb_site in enumerate(lattice.num_orbitals):
                    for i in range(n_orb_site):
                        base_name = H0_spinless.orbital_names[offsets[site_idx] + i]
                        orbital_names.append(f"{base_name}_up")
                        orbital_names.append(f"{base_name}_down")
                # Create orbital metadatas with spin if source has metadatas
                if H0_spinless.orbital_metadatas is not None:
                    from condmatTensor.core.types import OrbitalMetadata
                    for md in H0_spinless.orbital_metadatas:
                        md_up = OrbitalMetadata(
                            site=md.site, orb=md.orb, spin='up',
                            local=md.local, U=md.U
                        )
                        md_down = OrbitalMetadata(
                            site=md.site, orb=md.orb, spin='down',
                            local=md.local, U=md.U
                        )
                        orbital_metadatas.extend([md_up, md_down])

        return BaseTensor(
            tensor=H_spinful,
            labels=["k", "orb_i", "orb_j"],
            orbital_names=orbital_names,
            orbital_metadatas=orbital_metadatas if orbital_metadatas else None,
            displacements=None,
        )

    def add_magnetic_exchange(
        self,
        Hk: "BaseTensor",
        S_config: torch.Tensor,
        J: Optional[float] = None,
        lattice: Optional["BravaisLattice"] = None,
    ) -> "BaseTensor":
        """Add J@S term: J * (Sx*σx + Sy*σy + Sz*σz) to Hamiltonian.

        The magnetic exchange term couples local moments to electron spin via
        Pauli matrices. For each site i with local moment S_i = (Sx, Sy, Sz):

            H_J@S[i] = J * (Sx*σx + Sy*σy + Sz*σz)

        In the 2×2 spinor block for orbital i:
            [[J*Sz,      J*(Sx - i*Sy)],
             [J*(Sx + i*Sy),  -J*Sz    ]]

        Args:
            Hk: BaseTensor with shape (N_k, 2*N_orb, 2*N_orb)
                Labels should be ['k', 'orb_i', 'orb_j']
            S_config: Local spin configuration, shape (N_sites, 3) for (Sx, Sy, Sz)
                      Can be shorter than N_orb (applied to first N_sites sites)
            J: Coupling strength (uses self.J if None)
            lattice: Optional BravaisLattice for per-site orbital info.
                     If None, assumes uniform 2 spin channels per site.

        Returns:
            BaseTensor with J@S added to on-site terms
            Shape: (N_k, 2*N_orb, 2*N_orb)
        """
        from condmatTensor.core import BaseTensor

        if J is None:
            J = self.J

        # Verify Hk has correct shape
        if "k" not in Hk.labels:
            raise ValueError("Hk must have 'k' in labels")

        N_k = Hk.shape[Hk.labels.index("k")]
        N_spinful = Hk.shape[-1]

        # Ensure S_config has correct shape
        if S_config.dim() == 1:
            S_config = S_config.unsqueeze(0)

        # Clone Hamiltonian
        H_total = Hk.tensor.clone()

        if lattice is None:
            # Old behavior: assume uniform 2 spin channels (stride 2)
            N_orb = N_spinful // 2
            N_sites = min(S_config.shape[0], N_orb)

            for i in range(N_sites):
                Sx, Sy, Sz = S_config[i]

                # On-site J@S contribution (2×2 spinor block)
                J_term = torch.zeros((2, 2), dtype=torch.complex128, device=Hk.tensor.device)
                J_term[0, 0] = J * Sz                    # ↑↑: +J*Sz
                J_term[0, 1] = J * (Sx - 1j * Sy)         # ↑↓: J*(Sx - i*Sy)
                J_term[1, 0] = J * (Sx + 1j * Sy)         # ↓↑: J*(Sx + i*Sy)
                J_term[1, 1] = -J * Sz                   # ↓↓: -J*Sz

                # Add to on-site diagonal elements for all k-points
                H_total[:, 2*i:2*i+2, 2*i:2*i+2] += J_term.unsqueeze(0)
        else:
            # New behavior: use site offsets from lattice
            offsets = lattice.orbital_offsets()
            N_sites = min(S_config.shape[0], lattice.num_sites)
            spinful_offset = 0

            for site_idx in range(N_sites):
                Sx, Sy, Sz = S_config[site_idx]
                n_orb_site = lattice.num_orbitals[site_idx]
                n_spinful_site = 2 * n_orb_site

                # On-site J@S contribution (2×2 spinor block)
                J_term = torch.zeros((2, 2), dtype=torch.complex128, device=Hk.tensor.device)
                J_term[0, 0] = J * Sz                    # ↑↑: +J*Sz
                J_term[0, 1] = J * (Sx - 1j * Sy)         # ↑↓: J*(Sx - i*Sy)
                J_term[1, 0] = J * (Sx + 1j * Sy)         # ↓↑: J*(Sx + i*Sy)
                J_term[1, 1] = -J * Sz                   # ↓↓: -J*Sz

                # Add to each orbital at this site
                for orb_i in range(n_orb_site):
                    idx_i = spinful_offset + 2*orb_i
                    for orb_j in range(n_orb_site):
                        idx_j = spinful_offset + 2*orb_j
                        if orb_i == orb_j:
                            # On-diagonal orbital: add full J@S Pauli matrix
                            H_total[:, idx_i:idx_i+2, idx_j:idx_j+2] += J_term.unsqueeze(0)
                        # Off-diagonal orbital terms remain zero

                spinful_offset += n_spinful_site

        return BaseTensor(
            tensor=H_total,
            labels=Hk.labels,
            orbital_names=Hk.orbital_names,
            displacements=Hk.displacements,
        )

    def add_effective_magnetic_field(
        self,
        Hk: "BaseTensor",
        B_field: Union[torch.Tensor, Tuple[float, float, float]],
        g_factor: float = 2.0,
        mu_B: float = 1.0,
        lattice: Optional["BravaisLattice"] = None,
    ) -> "BaseTensor":
        """Add external magnetic field via Zeeman coupling.

        Simulates external magnetic field by adding Zeeman term to ALL sites:
            H_B = μ_B * g * (Bx*σx + By*σy + Bz*σz)

        This is equivalent to adding the same effective magnetic field to each site:
            S_eff = (g*μ_B/J) * B  (if reusing magnetic exchange infrastructure)

        **Physical Basis:** Zeeman coupling
            H_B = μ_B * g * (B · σ)
                = μ_B * g * (Bx*σx + By*σy + Bz*σz)

        Args:
            Hk: BaseTensor with shape (N_k, 2*N_orb, 2*N_orb)
            B_field: Magnetic field vector, either tensor (Bx, By, Bz) or tuple
            g_factor: Landé g-factor (default: 2.0 for electron)
            mu_B: Bohr magneton in energy units (default: 1.0)
            lattice: Optional BravaisLattice for per-site orbital info.
                     If None, assumes uniform 2 spin channels per site.

        Returns:
            BaseTensor with Zeeman term added

        Example:
            >>> # B-field in z-direction
            >>> Hk_B = model.add_effective_magnetic_field(Hk, B_field=(0, 0, 0.5))
        """
        from condmatTensor.core import BaseTensor

        if isinstance(B_field, (tuple, list)):
            B_field = torch.tensor(B_field, dtype=torch.float64)

        N_k = Hk.shape[Hk.labels.index("k")]
        N_spinful = Hk.shape[-1]

        H_total = Hk.tensor.clone()

        # Zeeman coupling strength
        alpha = g_factor * mu_B
        Bx, By, Bz = B_field

        # Zeeman term as 2×2 Pauli matrix
        H_zeeman = torch.zeros((2, 2), dtype=torch.complex128, device=Hk.tensor.device)
        H_zeeman[0, 0] = alpha * Bz                     # ↑↑: +α*Bz
        H_zeeman[0, 1] = alpha * (Bx - 1j * By)          # ↑↓: α*(Bx - i*By)
        H_zeeman[1, 0] = alpha * (Bx + 1j * By)          # ↓↑: α*(Bx + i*By)
        H_zeeman[1, 1] = -alpha * Bz                    # ↓↓: -α*Bz

        if lattice is None:
            # Old behavior: assume uniform 2 spin channels (stride 2)
            N_orb = N_spinful // 2
            for i in range(N_orb):
                H_total[:, 2*i:2*i+2, 2*i:2*i+2] += H_zeeman.unsqueeze(0)
        else:
            # New behavior: use site offsets from lattice
            spinful_offset = 0
            for site_idx, n_orb_site in enumerate(lattice.num_orbitals):
                n_spinful_site = 2 * n_orb_site
                for orb_i in range(n_orb_site):
                    idx_i = spinful_offset + 2*orb_i
                    H_total[:, idx_i:idx_i+2, idx_i:idx_i+2] += H_zeeman.unsqueeze(0)
                spinful_offset += n_spinful_site

        return BaseTensor(
            tensor=H_total,
            labels=Hk.labels,
            orbital_names=Hk.orbital_names,
            displacements=Hk.displacements,
        )

    def add_effective_field_via_S_eff(
        self,
        Hk: "BaseTensor",
        S_eff: torch.Tensor,
        J: float,
    ) -> "BaseTensor":
        """Add magnetic field by reusing J@S infrastructure.

        Alternative method to add magnetic field by treating it as a uniform
        local moment on all sites:
            H_B = J * S_eff · σ  (same S_eff for all sites)

        This is useful if you want to treat B-field on equal footing with
        local magnetic moments.

        Args:
            Hk: BaseTensor with shape (N_k, 2*N_orb, 2*N_orb)
            S_eff: Effective spin vector (3,) for (Sx_eff, Sy_eff, Sz_eff)
            J: Coupling strength

        Returns:
            BaseTensor with uniform magnetic field added
        """
        N_orb = Hk.shape[-1] // 2

        # Create uniform S config for all sites
        S_uniform = S_eff.unsqueeze(0).repeat(N_orb, 1)

        return self.add_magnetic_exchange(Hk, S_uniform, J)

    def compute_green_function(
        self,
        Hk: "BaseTensor",
        omega_n: torch.Tensor,
        mu: float = 0.0,
    ) -> "BaseTensor":
        """Compute Green's function G(k, iωₙ) = (iωₙ + μ - Hₖ)⁻¹.

        Args:
            Hk: k-space Hamiltonian with labels=['k', 'orb_i', 'orb_j']
            omega_n: Matsubara frequencies (N_omega,)
            mu: Chemical potential

        Returns:
            BaseTensor with G(k, iωₙ), labels=['iwn', 'k', 'orb_i', 'orb_j']
            Shape: (N_omega, N_k, n_orb, n_orb)
        """
        from condmatTensor.core import BaseTensor

        # Verify Hk has correct labels
        if "k" not in Hk.labels:
            raise ValueError("Hk must have 'k' in labels")
        k_idx = Hk.labels.index("k")

        N_k = Hk.shape[k_idx]
        n_orb = Hk.shape[-1]
        n_omega = len(omega_n)

        # Extract Hk tensor and reorder to (N_k, n_orb, n_orb)
        Hk_tensor = Hk.tensor
        if k_idx != 0:
            perm = [k_idx] + [i for i in range(len(Hk.labels)) if i != k_idx]
            Hk_tensor = Hk_tensor.permute(perm)

        # Initialize output tensor
        G_tensor = torch.zeros(
            (n_omega, N_k, n_orb, n_orb),
            dtype=torch.complex128,
            device=Hk.tensor.device,
        )

        # Compute G(iωₙ, k) = (iωₙ + μ - Hₖ)⁻¹
        for i, iwn_val in enumerate(omega_n):
            # Build (iωₙ + μ - Hₖ) for all k-points
            inv_matrix = iwn_val + mu - Hk_tensor  # (N_k, n_orb, n_orb)

            # Invert at each k-point
            G_k = torch.linalg.inv(inv_matrix)  # (N_k, n_orb, n_orb)

            G_tensor[i] = G_k

        return BaseTensor(
            tensor=G_tensor,
            labels=["iwn", "k", "orb_i", "orb_j"],
            orbital_names=Hk.orbital_names,
            displacements=None,
        )

    def self_consistency_loop(
        self,
        beta: float,
        mixing: float = 0.5,
        tol: float = 1e-6,
        max_iter: int = 100,
        n_max: int = 100,
        mu: float = 0.0,
        lattice: Optional["BravaisLattice"] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """Solve for self-consistent S configuration.

        Similar to DMFT loop:
        1. Start with guess for S
        2. Build H(S) = H0 + J@S
        3. Compute G(k,ω) = [iω - H(S)]^(-1)
        4. Extract local Green's function G_loc(i,ω)
        5. Update S from expectation value: S = ⟨c† σ c⟩
        6. Mix: S_new = (1-mixing)*S_old + mixing*S_new
        7. Check convergence: |S_new - S_old| < tol

        **Mean-field approximation:**
        The spin expectation is computed from the local Green's function:
            S_i = (1/β) Σ_ω Tr[σ * G_loc(i, ω)]

        where G_loc(i, ω) is the 2×2 spinor block for orbital i.

        Args:
            beta: Inverse temperature (β = 1/k_B T)
            mixing: Mixing parameter for S update (0 < mixing <= 1)
            tol: Convergence tolerance
            max_iter: Maximum iterations
            n_max: Maximum Matsubara frequency index
            mu: Chemical potential
            lattice: Optional BravaisLattice for per-site orbital info.
                     If None, assumes uniform 2 spin channels per site.
            verbose: Print progress information

        Returns:
            (S_final, n_iterations) tuple
        """
        from condmatTensor.manybody.preprocessing import generate_matsubara_frequencies

        if self.H0 is None:
            raise ValueError("H0 must be set before running self-consistency loop")

        # Initialize S if not set
        if lattice is None:
            N_orb = self.H0.shape[-1] // 2
            if self.S is None:
                # Random unit vectors
                S = torch.randn((N_orb, 3), dtype=torch.float64, device=self.H0.tensor.device)
                S = S / torch.norm(S, dim=1, keepdim=True)
            else:
                S = self.S.clone()
        else:
            N_sites = lattice.num_sites
            if self.S is None:
                # Random unit vectors
                S = torch.randn((N_sites, 3), dtype=torch.float64, device=self.H0.tensor.device)
                S = S / torch.norm(S, dim=1, keepdim=True)
            else:
                S = self.S.clone()

        # Generate Matsubara frequencies
        iwn = generate_matsubara_frequencies(beta, n_max, fermionic=True,
                                             device=self.H0.tensor.device)

        # Self-consistency loop
        for iteration in range(max_iter):
            # Build Hamiltonian with current S
            Hk_S = self.add_magnetic_exchange(self.H0, S, self.J, lattice=lattice)

            # Compute Green's function
            G_kw = self.compute_green_function(Hk_S, iwn, mu)

            # Extract local G for each site and compute S
            if lattice is None:
                # Old behavior: uniform stride 2
                S_new = torch.zeros_like(S)
                for i in range(N_orb):
                    # G_loc(i, iωₙ): shape (N_omega, 2, 2)
                    G_loc = G_kw.tensor[:, 0, 2*i:2*i+2, 2*i:2*i+2]

                    # Compute spin expectation: S = (1/β) Σ_ω Tr[σ * G_loc(i,ω)]
                    # Using Matsubara sum: (1/β) Σ_ω → (1/(2π)) ∫ d(iω)
                    # For now, use simple average over frequencies

                    # Sx = (1/2) * Tr[σx * G]
                    S_new[i, 0] = 0.5 * torch.mean(G_loc[:, 0, 1] + G_loc[:, 1, 0])

                    # Sy = (1/2) * Tr[σy * G] = (i/2) * Tr[σy * G]
                    S_new[i, 1] = 0.5j * torch.mean(G_loc[:, 0, 1] - G_loc[:, 1, 0])

                    # Sz = (1/2) * Tr[σz * G]
                    S_new[i, 2] = 0.5 * torch.mean(G_loc[:, 0, 0] - G_loc[:, 1, 1])
            else:
                # New behavior: use site offsets from lattice
                offsets = lattice.orbital_offsets()
                spinful_offset = 0
                S_new = torch.zeros_like(S)

                for site_idx in range(lattice.num_sites):
                    n_orb_site = lattice.num_orbitals[site_idx]
                    n_spinful_site = 2 * n_orb_site

                    # Extract local G for this site
                    G_loc = G_kw.tensor[:, 0, spinful_offset:spinful_offset+n_spinful_site,
                                          spinful_offset:spinful_offset+n_spinful_site]

                    # For each orbital at this site, compute spin expectation
                    for orb_i in range(n_orb_site):
                        idx_i = 2*orb_i
                        G_loc_orb = G_loc[:, idx_i:idx_i+2, idx_i:idx_i+2]

                        # Sx = (1/2) * Tr[σx * G]
                        S_new[site_idx, 0] += 0.5 * torch.mean(G_loc_orb[:, 0, 1] + G_loc_orb[:, 1, 0])

                        # Sy = (1/2) * Tr[σy * G] = (i/2) * Tr[σy * G]
                        S_new[site_idx, 1] += 0.5j * torch.mean(G_loc_orb[:, 0, 1] - G_loc_orb[:, 1, 0])

                        # Sz = (1/2) * Tr[σz * G]
                        S_new[site_idx, 2] += 0.5 * torch.mean(G_loc_orb[:, 0, 0] - G_loc_orb[:, 1, 1])

                    # Average over orbitals at this site
                    S_new[site_idx] /= n_orb_site

                    spinful_offset += n_spinful_site

            # Take real part (expectation values are real)
            S_new = S_new.real

            # Mix with old S
            S_old = S.clone()
            S = (1 - mixing) * S_old + mixing * S_new

            # Check convergence
            diff = torch.norm(S - S_old)
            if verbose:
                print(f"Iteration {iteration + 1}: |ΔS| = {diff:.6e}")

            if diff < tol:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations")
                break

        self.S = S
        return S, iteration + 1


class KondoLatticeSolver(LocalMagneticModel):
    """Specialized solver for Kondo lattice models.

    The Kondo lattice model describes conduction electrons coupled to
    localized f-electron moments via exchange interaction:

        H = Σ_k ε_k c†_kσ c_kσ + J Σ_i S_i · s_i

    where s_i = (1/2) c†_iσ σ_{σσ'} c_iσ' is the electron spin density.

    This class provides methods specific to Kondo lattice physics:
    - Kondo temperature estimation
    - RKKY interaction calculation
    - Heavy fermion band structure

    References:
        - "The Kondo Lattice Model" - M. Lavigna, Nuovo Cimento (1998)
        - "Heavy Fermions" - P. Coleman, Handbook of Magnetism (2007)
    """

    def __init__(
        self,
        H0: Optional["BaseTensor"] = None,
        J: float = 1.0,
        S_init: Optional[torch.Tensor] = None,
    ):
        """Initialize KondoLatticeSolver.

        Args:
            H0: Conduction electron Hamiltonian
            J: Kondo coupling strength (typically J < 0 for antiferromagnetic)
            S_init: Initial f-moment configuration
        """
        super().__init__(H0, J, S_init)

    def estimate_kondo_temperature(
        self,
        density_of_states: float,
        J: Optional[float] = None,
    ) -> float:
        """Estimate Kondo temperature using mean-field theory.

        T_K ~ D * exp(-1/(|J|*ρ))

        where D is the bandwidth and ρ is the DOS at Fermi level.

        Args:
            density_of_states: DOS at Fermi level
            J: Kondo coupling (uses self.J if None)

        Returns:
            Estimated Kondo temperature (in same units as bandwidth)
        """
        if J is None:
            J = abs(self.J)

        # Mean-field Kondo temperature
        # For proper estimate, need bandwidth D and DOS ρ
        # T_K = D * exp(-1/(J*ρ))
        # This is a simplified formula

        return torch.exp(torch.tensor(-1.0 / (J * density_of_states), dtype=torch.float64)).item()

    def compute_rkky_interaction(
        self,
        Hk: "BaseTensor",
        q_vector: torch.Tensor,
    ) -> float:
        """Compute RKKY interaction between local moments.

        The RKKY interaction is mediated by conduction electrons:
            J_RKKY(r) ∝ J^2 * χ(r)

        where χ(r) is the static spin susceptibility.

        Args:
            Hk: Conduction electron Hamiltonian
            q_vector: Wavevector for susceptibility

        Returns:
            RKKY coupling strength at this q-vector
        """
        # Static susceptibility χ(q) at ω=0
        # χ(q, ω=0) = Σ_k (f(ε_k) - f(ε_{k+q})) / (ε_{k+q} - ε_k)

        # For now, return placeholder
        # Full implementation requires diagonalization and Fermi function
        return 0.0


class SpinFermionModel(LocalMagneticModel):
    """General spin-fermion coupling model.

    More general than Kondo lattice, allowing:
    - Multiple fermion species
    - Non-local spin-spin interactions
    - Time-dependent spin dynamics

    H = H_0 + H_int
    H_int = Σ_{ij} J_ij · S_i · s_j
    """

    def __init__(
        self,
        H0: Optional["BaseTensor"] = None,
        J_tensor: Optional[torch.Tensor] = None,
        S_init: Optional[torch.Tensor] = None,
    ):
        """Initialize SpinFermionModel.

        Args:
            H0: Fermion Hamiltonian
            J_tensor: Coupling tensor J_ij (can be non-local)
            S_init: Initial spin configuration
        """
        super().__init__(H0, J=1.0, S_init=S_init)
        self.J_tensor = J_tensor


def pauli_matrices(device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Pauli matrices as complex tensors.

    Args:
        device: Device to place tensors on

    Returns:
        (σx, σy, σz) tuple of 2×2 complex matrices
    """
    if device is None:
        device = torch.device("cpu")

    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)

    return sigma_x, sigma_y, sigma_z
