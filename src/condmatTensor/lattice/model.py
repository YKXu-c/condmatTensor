"""Lattice model classes: BravaisLattice for crystal structure."""

from typing import List, Optional, Tuple, Union
import torch


class TightBindingModel:
    """
    General tight-binding model builder.

    Reads symbolic hopping terms and constructs H(R) and H(k).
    Each hopping term is: (orb_i, orb_j, displacement, hopping_value)
    where displacement is in units of lattice vectors.

    Supports both integer orbital indices and string orbital labels:
        (0, 1, torch.tensor([0, 0]), 1.0)  # using indices
        ("A", "B", torch.tensor([0, 0]), 1.0)  # using labels

    Attributes:
        lattice: BravaisLattice object
        hoppings: List of (orb_i, orb_j, displacement, value) tuples
        orbital_labels: List of orbital names (e.g., ['A', 'B', 'C'])
    """

    def __init__(
        self,
        lattice: "BravaisLattice",
        orbital_labels: Optional[List[str]] = None,
        hoppings: Optional[List[Tuple]] = None,
    ) -> None:
        """
        Initialize TightBindingModel.

        Args:
            lattice: BravaisLattice object
            orbital_labels: List of orbital names (e.g., ['A', 'B', 'C'] for Kagome)
                          If None, uses default ['orb_0', 'orb_1', ...]
            hoppings: List of hopping tuples (orb_i, orb_j, displacement, value)
        """
        self.lattice = lattice
        if orbital_labels is None:
            self.orbital_labels = [f"orb_{i}" for i in range(lattice.total_orbitals)]
        else:
            if len(orbital_labels) != lattice.total_orbitals:
                raise ValueError(
                    f"Number of orbital labels ({len(orbital_labels)}) must match "
                    f"total orbitals ({lattice.total_orbitals})"
                )
            self.orbital_labels = orbital_labels

        # Create label to index mapping
        self._label_to_idx = {label: i for i, label in enumerate(self.orbital_labels)}

        self.hoppings = hoppings if hoppings is not None else []

    def _resolve_orbital(self, orb: Union[int, str]) -> int:
        """Convert orbital label or index to integer index."""
        if isinstance(orb, int):
            return orb
        elif isinstance(orb, str):
            if orb not in self._label_to_idx:
                raise ValueError(f"Unknown orbital label: {orb}. Known labels: {self.orbital_labels}")
            return self._label_to_idx[orb]
        else:
            raise TypeError(f"Orbital must be int or str, got {type(orb)}")

    def add_hopping(
        self,
        orb_i: Union[int, str],
        orb_j: Union[int, str],
        displacement: Union[torch.Tensor, List[float], Tuple[float, ...]],
        value: float = 1.0,
        add_hermitian: bool = True,
    ) -> None:
        """
        Add a hopping term.

        Args:
            orb_i: Source orbital index or label (e.g., 0 or 'A')
            orb_j: Target orbital index or label (e.g., 1 or 'B')
            displacement: Displacement vector (fractional coordinates)
            value: Hopping amplitude
            add_hermitian: If True, automatically add Hermitian conjugate
        """
        # Get lattice dtype for consistency
        lattice_dtype = self.lattice.cell_vectors.dtype

        if isinstance(displacement, (list, tuple)):
            displacement = torch.tensor(displacement, dtype=lattice_dtype)
        displacement = displacement.clone().detach().to(dtype=lattice_dtype)

        # Convert to integer indices
        orb_i_idx = self._resolve_orbital(orb_i)
        orb_j_idx = self._resolve_orbital(orb_j)

        self.hoppings.append((orb_i_idx, orb_j_idx, displacement, value))

        if add_hermitian and orb_i_idx != orb_j_idx:
            # Hermitian conjugate: swap orbitals, negate displacement, conjugate value
            conj_value = torch.conj(value) if isinstance(value, torch.Tensor) else value
            self.hoppings.append((orb_j_idx, orb_i_idx, -displacement, conj_value))

    def build_HR(self) -> "BaseTensor":
        """
        Build real-space Hamiltonian H(R) from hopping terms.

        Returns:
            BaseTensor with labels=['R', 'orb_i', 'orb_j'] and orbital_names set
        """
        from condmatTensor.core import BaseTensor

        # Get unique displacements
        disp_tuples = list(set(tuple(hop[2].tolist()) for hop in self.hoppings))
        disp_tuples.sort()  # Sort for consistency

        # Use lattice dtype for consistency
        lattice_dtype = self.lattice.cell_vectors.dtype
        displacements = torch.stack([torch.tensor(d, dtype=lattice_dtype) for d in disp_tuples])

        n_R = len(displacements)
        n_orb = self.lattice.total_orbitals

        # Build Hamiltonian tensor with complex128
        H_R = torch.zeros((n_R, n_orb, n_orb), dtype=torch.complex128)

        # Create displacement to index mapping
        disp_to_idx = {tuple(d.tolist()): i for i, d in enumerate(displacements)}

        for orb_i, orb_j, disp, value in self.hoppings:
            R_idx = disp_to_idx[tuple(disp.tolist())]
            H_R[R_idx, orb_i, orb_j] += value

        # Convert displacements to Cartesian coordinates
        disp_cart = displacements @ self.lattice.cell_vectors

        return BaseTensor(
            tensor=H_R,
            labels=["R", "orb_i", "orb_j"],
            orbital_names=self.orbital_labels,
            displacements=disp_cart,
        )

    def build_Hk(self, k_path: torch.Tensor) -> "BaseTensor":
        """
        Build k-space Hamiltonian H(k) directly from hopping terms.

        H(k) = Σ_R H(R) * exp(i*k*R)

        Args:
            k_path: K-points in fractional coordinates, shape (N_k, dim)

        Returns:
            BaseTensor with labels=['k', 'orb_i', 'orb_j'] and orbital_names set
        """
        from condmatTensor.core import BaseTensor

        N_k = len(k_path)
        n_orb = self.lattice.total_orbitals

        Hk = torch.zeros((N_k, n_orb, n_orb), dtype=torch.complex128)

        # Convert k-path to Cartesian once
        k_cart = k_path @ self.lattice.reciprocal_vectors().T  # (N_k, dim)

        # For each hopping term, add contribution to H(k)
        for orb_i, orb_j, disp_frac, value in self.hoppings:
            # Convert displacement to Cartesian
            disp_cart = disp_frac @ self.lattice.cell_vectors

            # Compute phase factor: exp(i * k @ R)
            # k_cart: (N_k, dim), disp_cart: (dim,)
            phases = torch.exp(1j * torch.matmul(k_cart, disp_cart))  # (N_k,)

            # Add contribution to all k-points
            Hk[:, orb_i, orb_j] += value * phases

        return BaseTensor(
            tensor=Hk,
            labels=["k", "orb_i", "orb_j"],
            orbital_names=self.orbital_labels,
            displacements=None,  # k-space has no displacements
        )


class BravaisLattice:
    """
    Bravais lattice with multiple sites per unit cell.

    Represents a periodic crystal structure with lattice vectors
    and basis positions for atoms/sites within the unit cell.

    Attributes:
        cell_vectors: Lattice vectors, shape (dim, dim)
            Each row is a lattice vector in Cartesian coordinates
        basis_positions: List of basis atom positions, each (dim,)
            Positions are in fractional coordinates (relative to lattice vectors)
        num_orbitals: Number of orbitals per site
        dim: Spatial dimension (2 for 2D, 3 for 3D)
    """

    def __init__(
        self,
        cell_vectors: torch.Tensor,
        basis_positions: List[torch.Tensor],
        num_orbitals: int = 1,
    ) -> None:
        """
        Initialize BravaisLattice.

        Args:
            cell_vectors: Lattice vectors, shape (dim, dim)
            basis_positions: List of basis atom positions in fractional coords
            num_orbitals: Number of orbitals per site
        """
        self.cell_vectors = cell_vectors
        self.basis_positions = basis_positions
        self.num_orbitals = num_orbitals
        self.dim = cell_vectors.shape[0]

        self.num_sites = len(basis_positions)

    @property
    def num_basis(self) -> int:
        """Number of basis sites in unit cell."""
        return self.num_sites

    @property
    def total_orbitals(self) -> int:
        """Total number of orbitals in unit cell."""
        return self.num_sites * self.num_orbitals

    def reciprocal_vectors(self) -> torch.Tensor:
        """
        Compute reciprocal lattice vectors.

        For 2D: b_i = 2π * ε_ij * a_j / |a_1 × a_2|
        For 3D: b_i = 2π * ε_ijk * a_j × a_k / (a_1 · (a_2 × a_3))

        Returns:
            Reciprocal lattice vectors, shape (dim, dim)
        """
        a = self.cell_vectors

        if self.dim == 2:
            # 2D case
            det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
            b = torch.zeros_like(a)
            b[0, 0] = 2 * torch.pi * a[1, 1] / det
            b[0, 1] = -2 * torch.pi * a[1, 0] / det
            b[1, 0] = -2 * torch.pi * a[0, 1] / det
            b[1, 1] = 2 * torch.pi * a[0, 0] / det
            return b

        else:  # dim == 3
            # 3D case: b_i = 2π * (a_j × a_k) / (a_i · (a_j × a_k))
            volume = torch.dot(a[0], torch.cross(a[1], a[2]))
            b = torch.zeros_like(a)
            b[0] = 2 * torch.pi * torch.cross(a[1], a[2]) / volume
            b[1] = 2 * torch.pi * torch.cross(a[2], a[0]) / volume
            b[2] = 2 * torch.pi * torch.cross(a[0], a[1]) / volume
            return b

    def high_symmetry_points(self) -> dict[str, torch.Tensor]:
        """
        Return high-symmetry points in fractional coordinates.

        For triangular lattice with basis vectors a1=(1/2, √3/2), a2=(1, 0):
            Γ = (0, 0)
            K = (1/3, 1/√3) - Dirac point where bands touch
            M = (1/2, 0)      - midpoint of Brillouin zone edge

        At K point: Kagome lattice has Dirac cone with eigenvalues (-2, 1, 1)

        Returns:
            Dictionary mapping point names to fractional coordinates
        """
        import math
        if self.dim == 2:
            return {
                "G": torch.tensor([0.0, 0.0]),
                "K": torch.tensor([1.0 / 3.0, 1.0 / math.sqrt(3)]),
                "M": torch.tensor([0.5, 0.0]),
            }
        return {}

    def __repr__(self) -> str:
        return (
            f"BravaisLattice(dim={self.dim}, num_sites={self.num_sites}, "
            f"num_orbitals={self.num_orbitals})"
        )
