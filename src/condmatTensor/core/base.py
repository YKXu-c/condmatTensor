"""BaseTensor class: unified tensor representation for physics objects."""

from typing import Optional, List, Union, Dict, Any
import torch


class BaseTensor:
    """
    Unified tensor class for condensed matter physics objects.

    All physics objects (Hamiltonian, Green's function, Self-energy) use
    this single class with semantic labels for each dimension.

    Attributes:
        tensor: Underlying PyTorch tensor data
        labels: Semantic labels for each dimension (e.g., ['k', 'orb_i', 'orb_j'])
        orbital_names: Physical names of orbitals (e.g., ['px', 'py', 'pz'])
        orbital_metadatas: Structured orbital metadata (OrbitalMetadata list)
        displacements: Real-space displacements for H(R) tensors, shape (N_R, dim)
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        labels: List[str],
        orbital_names: Optional[List[str]] = None,
        orbital_metadatas: Optional[List["OrbitalMetadataLike"]] = None,
        displacements: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize BaseTensor.

        Args:
            tensor: Underlying tensor data
            labels: Semantic labels for each dimension
            orbital_names: Physical names of orbitals (e.g., ['px', 'py'])
            orbital_metadatas: Structured orbital metadata. Can be:
                - List of OrbitalMetadata objects
                - List of dictionaries with metadata fields
                - List of strings (will be parsed)
                If provided, takes precedence over orbital_names.
            displacements: Real-space displacements for H(R) tensors, shape (N_R, dim)
        """
        if len(labels) != tensor.ndim:
            raise ValueError(
                f"Number of labels ({len(labels)}) must match tensor.ndim ({tensor.ndim})"
            )

        self.tensor = tensor
        self.labels = labels
        self.displacements = displacements

        # Handle orbital_metadatas (takes precedence over orbital_names)
        if orbital_metadatas is not None:
            self._orbital_metadatas = self._normalize_metadatas(orbital_metadatas)
            self.orbital_names = [md.to_string() for md in self._orbital_metadatas]
        elif orbital_names is not None:
            self.orbital_names = orbital_names
            self._orbital_metadatas = None
        else:
            self.orbital_names = None
            self._orbital_metadatas = None

    def to_k_space(self, k: torch.Tensor) -> "BaseTensor":
        """
        Fourier transform from real-space (R) to momentum-space (k).

        H_{αβ}(k) = Σ_R H_{αβ}(R) · exp(i·k·R)

        Args:
            k: K-points in Cartesian coordinates, shape (N_k, dim)

        Returns:
            BaseTensor with R dimension replaced by k

        Raises:
            ValueError: If displacements is None or labels don't contain 'R'
        """
        if self.displacements is None:
            raise ValueError("Cannot Fourier transform: displacements is None")

        if "R" not in self.labels:
            raise ValueError("Cannot Fourier transform: 'R' not in labels")

        # Find R dimension index
        r_idx = self.labels.index("R")
        N_k = k.shape[0]

        # Compute phase factors: exp(i * k @ R.T)
        # k: (N_k, dim), displacements: (N_R, dim)
        # Match dtype with tensor
        k_dtype = k.dtype if k.is_complex() else torch.complex128
        phases = torch.exp(1j * torch.matmul(k, self.displacements.T)).to(dtype=k_dtype)  # (N_k, N_R)

        # Build einsum equation based on R position
        # Generate subscripts: e.g., "Rab" for labels=['R', 'a', 'b']
        subscripts = list("abcdefghijklmnopqrstuvwxyz")[:self.ndim]
        r_sub = subscripts[r_idx]

        # Build input and output subscripts
        input_sub = "".join(subscripts)
        # Phase subscripts: kR (N_k, N_R)
        phase_sub = "k" + r_sub
        # Output: replace R with k
        output_sub = input_sub.replace(r_sub, "k")

        # Perform Fourier transform
        einsum_str = f"{input_sub},{phase_sub}->{output_sub}"
        fourier_tensor = torch.einsum(einsum_str, self.tensor, phases)

        # Create new labels (replace 'R' with 'k')
        new_labels = [label if label != "R" else "k" for label in self.labels]

        # Reorder dimensions so k comes first (convention)
        k_idx = new_labels.index("k")
        if k_idx != 0:
            perm = [k_idx] + [i for i in range(len(new_labels)) if i != k_idx]
            fourier_tensor = fourier_tensor.permute(perm)
            new_labels = [new_labels[i] for i in perm]

        return BaseTensor(
            tensor=fourier_tensor,
            labels=new_labels,
            orbital_names=self.orbital_names,
            orbital_metadatas=self._orbital_metadatas,
            displacements=None,  # k-space tensors don't have displacements
        )

    def to(self, device: torch.device) -> "BaseTensor":
        """Move tensor to device (CPU/GPU)."""
        new_tensor = self.tensor.to(device)
        new_displacements = self.displacements.to(device) if self.displacements is not None else None
        return BaseTensor(
            tensor=new_tensor,
            labels=self.labels,
            orbital_names=self.orbital_names,
            orbital_metadatas=self._orbital_metadatas,
            displacements=new_displacements,
        )

    @property
    def shape(self) -> torch.Size:
        """Return tensor shape."""
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.tensor.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Return tensor dtype."""
        return self.tensor.dtype

    def __repr__(self) -> str:
        return f"BaseTensor(shape={self.shape}, labels={self.labels}, dtype={self.dtype})"

    def _normalize_metadatas(
        self, metadatas: List["OrbitalMetadataLike"]
    ) -> List["OrbitalMetadata"]:
        """Normalize orbital metadata inputs to OrbitalMetadata objects.

        Args:
            metadatas: List of OrbitalMetadata, dict, or string

        Returns:
            List of OrbitalMetadata objects
        """
        from .types import OrbitalMetadata

        normalized = []
        for md in metadatas:
            if isinstance(md, OrbitalMetadata):
                normalized.append(md)
            elif isinstance(md, dict):
                normalized.append(OrbitalMetadata.from_dict(md))
            elif isinstance(md, str):
                normalized.append(OrbitalMetadata.from_string(md))
            else:
                raise TypeError(
                    f"orbital_metadatas must be list of OrbitalMetadata, dict, or str, "
                    f"got {type(md)}"
                )
        return normalized

    @property
    def orbital_metadatas(self) -> Optional[List["OrbitalMetadata"]]:
        """Return orbital metadata list.

        Lazily generates from orbital_names if not explicitly set.

        Returns:
            List of OrbitalMetadata objects, or None if orbital_names is None
        """
        if self._orbital_metadatas is not None:
            return self._orbital_metadatas

        # Lazy generation from orbital_names
        if self.orbital_names is not None:
            from .types import OrbitalMetadata
            self._orbital_metadatas = [
                OrbitalMetadata.from_string(name) for name in self.orbital_names
            ]
            return self._orbital_metadatas

        return None

    def get_f_orbitals(self) -> List[int]:
        """Get indices of f-orbitals.

        Returns:
            List of orbital indices identified as f-orbitals

        Examples:
            >>> H = BaseTensor(...)
            >>> f_indices = H.get_f_orbitals()
            >>> print(f"F-orbitals at: {f_indices}")
        """
        if self.orbital_metadatas is None:
            return []
        return [i for i, md in enumerate(self.orbital_metadatas) if md.is_f_orbital()]

    def get_orbitals_by_site(self, site: str) -> List[int]:
        """Get indices of orbitals at a specific site.

        Args:
            site: Site identifier (e.g., 'Ce1', 'atom1')

        Returns:
            List of orbital indices at this site

        Examples:
            >>> H = BaseTensor(...)
            >>> ce_orbitals = H.get_orbitals_by_site('Ce1')
        """
        if self.orbital_metadatas is None:
            return []
        return [i for i, md in enumerate(self.orbital_metadatas) if md.site == site]

    def get_spinful_orbitals(self) -> List[int]:
        """Get indices of spinful orbitals.

        Returns:
            List of orbital indices that have spin information

        Examples:
            >>> H = BaseTensor(...)
            >>> spinful_indices = H.get_spinful_orbitals()
        """
        if self.orbital_metadatas is None:
            return []
        return [i for i, md in enumerate(self.orbital_metadatas) if md.is_spinful()]

    def is_spinful_system(self) -> bool:
        """Check if system is spinful.

        A system is considered spinful if all orbitals have spin information.

        Returns:
            True if the system is spinful, False otherwise

        Examples:
            >>> H = BaseTensor(...)
            >>> if H.is_spinful_system():
            ...     print("System has spin")
        """
        metadatas = self.orbital_metadatas
        if metadatas is None or len(metadatas) == 0:
            return False
        return all(md.is_spinful() for md in metadatas)

    def get_localized_orbitals(self) -> List[int]:
        """Get indices of localized orbitals.

        Returns:
            List of orbital indices identified as localized

        Examples:
            >>> H = BaseTensor(...)
            >>> local_indices = H.get_localized_orbitals()
        """
        if self.orbital_metadatas is None:
            return []
        return [i for i, md in enumerate(self.orbital_metadatas) if md.is_localized()]
