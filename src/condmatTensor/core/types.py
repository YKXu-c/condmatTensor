"""Type definitions for condmatTensor core module."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class OrbitalMetadata:
    """
    Structured metadata for a single orbital.

    This class provides a richer way to describe orbitals beyond simple string
    labels. It enables querying orbitals by properties (e.g., f-orbitals,
    localized orbitals, spinful systems) for many-body physics applications
    like DMFT and heavy fermion systems.

    Attributes:
        site: Site identifier (e.g., 'Ce1', 'atom1', or None)
        orb: Orbital type (e.g., 's', 'px', 'dxy', 'f', etc.)
        spin: Spin projection ('up', 'down', or None for spinless)
        local: Localized (True) vs conductive (False), None if unspecified
        U: Hubbard U parameter (None if not applicable)
        name: Optional display name override

    Examples:
        >>> md = OrbitalMetadata(site='Ce1', orb='f', spin='up', local=True, U=7.0)
        >>> md.to_string()
        'Ce1-f-spin_up-local-U7.0'

        >>> md2 = OrbitalMetadata.from_string('La-dxy-spin_up')
        >>> md2.site, md2.orb, md2.spin
        ('La', 'dxy', 'up')
    """
    site: Optional[str] = None
    orb: Optional[str] = None
    spin: Optional[str] = None  # 'up', 'down', or None
    local: Optional[bool] = None
    U: Optional[float] = None
    name: Optional[str] = None

    def to_string(self) -> str:
        """Convert to string format (e.g., 'Ce1-dxy-spin_up-local-U7.0')."""
        parts = []
        if self.site:
            parts.append(self.site)
        if self.orb:
            parts.append(self.orb)
        if self.spin:
            parts.append(f"spin_{self.spin}")
        if self.local is not None:
            parts.append("local" if self.local else "conductive")
        if self.U is not None:
            parts.append(f"U{self.U}")
        return "-".join(parts) if parts else "unknown"

    def is_f_orbital(self) -> bool:
        """Check if this is an f-orbital."""
        return self.orb is not None and 'f' in self.orb.lower()

    def is_spinful(self) -> bool:
        """Check if this orbital has spin information."""
        return self.spin is not None

    def is_localized(self) -> bool:
        """Check if this orbital is localized."""
        return self.local if self.local is not None else False

    @classmethod
    def from_string(cls, s: str) -> 'OrbitalMetadata':
        """Parse orbital metadata from string format.

        Supports parsing strings like:
        - 'Ce1-f-spin_up-local-U7.0'
        - 'La-dxy-spin_up'
        - 'px'
        - 'up'

        Args:
            s: String representation of orbital metadata

        Returns:
            OrbitalMetadata object with parsed fields

        Examples:
            >>> md = OrbitalMetadata.from_string('Ce1-f-spin_up-local-U7.0')
            >>> md.site, md.orb, md.spin, md.local, md.U
            ('Ce1', 'f', 'up', True, 7.0)
        """
        metadata = cls()
        parts = s.split('-')
        for part in parts:
            part_lower = part.lower()
            if 'spin_up' in part_lower or part == 'up':
                metadata.spin = 'up'
            elif 'spin_down' in part_lower or part == 'down':
                metadata.spin = 'down'
            elif part_lower == 'local':
                metadata.local = True
            elif part_lower == 'conductive':
                metadata.local = False
            elif part_lower.startswith('u'):
                try:
                    metadata.U = float(part[1:])
                except ValueError:
                    pass
            elif part_lower in ['s', 'px', 'py', 'pz', 'dx2-y2', 'dxy', 'dxz', 'dyz', 'dz2', 'f']:
                metadata.orb = part_lower
            elif metadata.site is None:
                metadata.site = part
        return metadata

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with all metadata fields
        """
        return {
            'site': self.site,
            'orb': self.orb,
            'spin': self.spin,
            'local': self.local,
            'U': self.U,
            'name': self.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OrbitalMetadata':
        """Create OrbitalMetadata from dictionary.

        Args:
            d: Dictionary with orbital metadata fields

        Returns:
            OrbitalMetadata object

        Examples:
            >>> d = {'site': 'Ce1', 'orb': 'f', 'spin': 'up', 'local': True, 'U': 7.0}
            >>> md = OrbitalMetadata.from_dict(d)
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Type alias for orbital metadata input
OrbitalMetadataLike = Union[OrbitalMetadata, Dict[str, Any], str]
