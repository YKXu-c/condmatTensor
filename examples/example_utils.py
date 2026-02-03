"""Shared utilities for condmatTensor examples.

This module provides common functions used across all example scripts to reduce
code duplication and ensure consistent behavior.

Key utilities:
- Path setup: Automatic src/ path configuration
- Device management: Consistent device selection with logging
- Lattice builders: Pre-configured Kagome lattice constructors
- Hopping model builders: Standard hopping patterns
- Plotting helpers: Consistent figure setup and saving
"""

from pathlib import Path
import sys
import math
import torch
import matplotlib.pyplot as plt

from condmatTensor.core import get_device
from condmatTensor.lattice import BravaisLattice, HoppingModel
from condmatTensor.analysis.plotting_style import (
    DEFAULT_FIGURE_SIZES,
    DEFAULT_COLORS,
)

# =============================================================================
# Path Setup (auto-run on import)
# =============================================================================


def setup_project_path() -> None:
    """Add src/ directory to Python path for imports.

    This function runs automatically when example_utils is imported,
    eliminating the need for manual sys.path.insert() calls in examples.
    """
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# Auto-run path setup on module import
setup_project_path()

# =============================================================================
# Device Management
# =============================================================================


def get_example_device(description: str = "") -> torch.device:
    """Get device for computation with optional description logging.

    Args:
        description: Optional description to print with device info

    Returns:
        torch.device: 'cuda' if available, else 'cpu'

    Example:
        >>> device = get_example_device("for diagonalization")
        Using device: cuda - for diagonalization
    """
    device = get_device()
    if description:
        print(f"Using device: {device} - {description}")
    return device

# =============================================================================
# Lattice Builders
# =============================================================================


def build_kagome_lattice(t: float = -1.0) -> BravaisLattice:
    """Build spinless Kagome lattice.

    The Kagome lattice has 3 sites per unit cell arranged in a triangle.
    Each site has 1 s-orbital.

    Args:
        t: Hopping parameter (for reference, default -1)

    Returns:
        BravaisLattice with 3 sites, 1 orbital each

    Reference:
        D. L. Bergman et al., Phys. Rev. B 76, 094417 (2007)
    """
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
    ]

    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[1, 1, 1],
    )


def build_kagome_spinful_lattice() -> BravaisLattice:
    """Build spinful Kagome lattice.

    The Kagome lattice has 3 sites per unit cell, each with spin ↑ and ↓.
    Total of 6 spinor orbitals (following spinor convention).

    Returns:
        BravaisLattice with 3 sites, 2 orbitals (spin up/down) each

    Spinor Convention:
        Each site has [orb_up, orb_down] ordering.
        Total orbitals: [A_up, A_down, B_up, B_down, C_up, C_down]
    """
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
    ]

    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2, 2, 2],
    )


def build_kagome_f_lattice(t: float = -1.0) -> BravaisLattice:
    """Build Kagome-F lattice with 4 sites per unit cell.

    Extended Kagome lattice with additional f-orbital site at (1/3, 1/3).
    Used for studying f-d hybridization and Kondo physics.

    Args:
        t: Hopping parameter (for reference, default -1)

    Returns:
        BravaisLattice with 4 sites (A, B, C, F), 1 orbital each
    """
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),      # A site
        torch.tensor([0.5, 0.0]),      # B site
        torch.tensor([0.25, sqrt3 / 4]),  # C site
        torch.tensor([1/3, 1/3]),      # F site (f-orbital)
    ]

    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[1, 1, 1, 1],
    )


def build_kagome_f_spinful_lattice() -> BravaisLattice:
    """Build spinful Kagome-F lattice.

    Extended Kagome lattice with f-orbital site, including spin degrees.
    Total of 8 spinor orbitals (4 sites × 2 spin).

    Returns:
        BravaisLattice with 4 sites, 2 orbitals (spin) each

    Spinor Convention:
        Total orbitals: [A_up, A_down, B_up, B_down, C_up, C_down, F_up, F_down]
    """
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),      # A site
        torch.tensor([0.5, 0.0]),      # B site
        torch.tensor([0.25, sqrt3 / 4]),  # C site
        torch.tensor([1/3, 1/3]),      # F site (f-orbital)
    ]

    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2, 2, 2, 2],
    )

# =============================================================================
# Hopping Model Builders
# =============================================================================


def build_kagome_model(lattice: BravaisLattice, t: float = -1.0) -> HoppingModel:
    """Build Kagome hopping model with standard nearest-neighbor hopping.

    Uses string orbital labels for clarity. Includes all 12 hopping terms
    for the 3-site Kagome lattice (intra-cell + inter-cell).

    Args:
        lattice: BravaisLattice object (should have 3 sites)
        t: Hopping parameter (default -1)

    Returns:
        HoppingModel with Kagome nearest-neighbor hopping pattern

    Hopping pattern (spinless, use string labels 'A', 'B', 'C'):
        - A <-> B: [0,0] and [-1,0]
        - A <-> C: [0,0] and [0,-1]
        - B <-> C: [0,0] and [1,-1]
    """
    model = HoppingModel(lattice, orbital_labels=['A', 'B', 'C'])

    # A <-> B hopping (intra-cell)
    model.add_hopping('A', 'B', [0, 0], t, add_hermitian=False)
    model.add_hopping('B', 'A', [0, 0], t, add_hermitian=False)

    # A <-> B hopping (inter-cell)
    model.add_hopping('A', 'B', [-1, 0], t, add_hermitian=False)
    model.add_hopping('B', 'A', [1, 0], t, add_hermitian=False)

    # A <-> C hopping (intra-cell)
    model.add_hopping('A', 'C', [0, 0], t, add_hermitian=False)
    model.add_hopping('C', 'A', [0, 0], t, add_hermitian=False)

    # A <-> C hopping (inter-cell)
    model.add_hopping('A', 'C', [0, -1], t, add_hermitian=False)
    model.add_hopping('C', 'A', [0, 1], t, add_hermitian=False)

    # B <-> C hopping (intra-cell)
    model.add_hopping('B', 'C', [0, 0], t, add_hermitian=False)
    model.add_hopping('C', 'B', [0, 0], t, add_hermitian=False)

    # B <-> C hopping (inter-cell)
    model.add_hopping('B', 'C', [1, -1], t, add_hermitian=False)
    model.add_hopping('C', 'B', [-1, 1], t, add_hermitian=False)

    return model


def build_kagome_spinful_model(lattice: BravaisLattice, t: float = -1.0) -> HoppingModel:
    """Build spinful Kagome hopping model.

    Spin-conserving nearest-neighbor hopping for spinful Kagome lattice.
    Each orbital has spin-up and spin-down variants.

    Args:
        lattice: BravaisLattice object (should have 3 sites with 2 orbitals each)
        t: Hopping parameter (default -1)

    Returns:
        HoppingModel with spinful Kagome hopping pattern

    Orbital labels: [A_up, A_down, B_up, B_down, C_up, C_down]
    """
    orbital_labels = [
        "A_up", "A_down",
        "B_up", "B_down",
        "C_up", "C_down",
    ]
    model = HoppingModel(lattice, orbital_labels=orbital_labels)

    # A <-> B hopping (spin-up)
    model.add_hopping("A_up", "B_up", [0, 0], t, add_hermitian=False)
    model.add_hopping("B_up", "A_up", [0, 0], t, add_hermitian=False)
    model.add_hopping("A_up", "B_up", [-1, 0], t, add_hermitian=False)
    model.add_hopping("B_up", "A_up", [1, 0], t, add_hermitian=False)

    # A <-> B hopping (spin-down)
    model.add_hopping("A_down", "B_down", [0, 0], t, add_hermitian=False)
    model.add_hopping("B_down", "A_down", [0, 0], t, add_hermitian=False)
    model.add_hopping("A_down", "B_down", [-1, 0], t, add_hermitian=False)
    model.add_hopping("B_down", "A_down", [1, 0], t, add_hermitian=False)

    # A <-> C hopping (spin-up)
    model.add_hopping("A_up", "C_up", [0, 0], t, add_hermitian=False)
    model.add_hopping("C_up", "A_up", [0, 0], t, add_hermitian=False)
    model.add_hopping("A_up", "C_up", [0, -1], t, add_hermitian=False)
    model.add_hopping("C_up", "A_up", [0, 1], t, add_hermitian=False)

    # A <-> C hopping (spin-down)
    model.add_hopping("A_down", "C_down", [0, 0], t, add_hermitian=False)
    model.add_hopping("C_down", "A_down", [0, 0], t, add_hermitian=False)
    model.add_hopping("A_down", "C_down", [0, -1], t, add_hermitian=False)
    model.add_hopping("C_down", "A_down", [0, 1], t, add_hermitian=False)

    # B <-> C hopping (spin-up)
    model.add_hopping("B_up", "C_up", [0, 0], t, add_hermitian=False)
    model.add_hopping("C_up", "B_up", [0, 0], t, add_hermitian=False)
    model.add_hopping("B_up", "C_up", [1, -1], t, add_hermitian=False)
    model.add_hopping("C_up", "B_up", [-1, 1], t, add_hermitian=False)

    # B <-> C hopping (spin-down)
    model.add_hopping("B_down", "C_down", [0, 0], t, add_hermitian=False)
    model.add_hopping("C_down", "B_down", [0, 0], t, add_hermitian=False)
    model.add_hopping("B_down", "C_down", [1, -1], t, add_hermitian=False)
    model.add_hopping("C_down", "B_down", [-1, 1], t, add_hermitian=False)

    return model


def build_kagome_f_model(
    lattice: BravaisLattice,
    t: float = -1.0,
    tf: float = -0.3,
    fd_hybridization: float = 0.5
) -> HoppingModel:
    """Build Kagome-F hopping model.

    Extended Kagome lattice with f-orbital site including f-d hybridization.

    Args:
        lattice: BravaisLattice object (should have 4 sites)
        t: d-d hopping parameter (default -1)
        tf: f-f hopping parameter (default -0.3)
        fd_hybridization: f-d hybridization strength (default 0.5)

    Returns:
        HoppingModel with Kagome-F hopping pattern
    """
    model = HoppingModel(lattice, orbital_labels=['A', 'B', 'C', 'F'])

    # Kagome d-d hopping (A, B, C)
    # A <-> B
    model.add_hopping('A', 'B', [0, 0], t, add_hermitian=False)
    model.add_hopping('B', 'A', [0, 0], t, add_hermitian=False)
    model.add_hopping('A', 'B', [-1, 0], t, add_hermitian=False)
    model.add_hopping('B', 'A', [1, 0], t, add_hermitian=False)

    # A <-> C
    model.add_hopping('A', 'C', [0, 0], t, add_hermitian=False)
    model.add_hopping('C', 'A', [0, 0], t, add_hermitian=False)
    model.add_hopping('A', 'C', [0, -1], t, add_hermitian=False)
    model.add_hopping('C', 'A', [0, 1], t, add_hermitian=False)

    # B <-> C
    model.add_hopping('B', 'C', [0, 0], t, add_hermitian=False)
    model.add_hopping('C', 'B', [0, 0], t, add_hermitian=False)
    model.add_hopping('B', 'C', [1, -1], t, add_hermitian=False)
    model.add_hopping('C', 'B', [-1, 1], t, add_hermitian=False)

    # f-f hopping (localized, typically weaker)
    model.add_hopping('F', 'F', [0, 0], tf, add_hermitian=False)
    model.add_hopping('F', 'F', [1, 0], tf, add_hermitian=False)
    model.add_hopping('F', 'F', [0, 1], tf, add_hermitian=False)

    # f-d hybridization (F connects to A, B, C)
    # F <-> A
    model.add_hopping('F', 'A', [0, 0], fd_hybridization, add_hermitian=False)
    model.add_hopping('A', 'F', [0, 0], fd_hybridization, add_hermitian=False)

    # F <-> B
    model.add_hopping('F', 'B', [0, 0], fd_hybridization, add_hermitian=False)
    model.add_hopping('B', 'F', [0, 0], fd_hybridization, add_hermitian=False)

    # F <-> C
    model.add_hopping('F', 'C', [0, 0], fd_hybridization, add_hermitian=False)
    model.add_hopping('C', 'F', [0, 0], fd_hybridization, add_hermitian=False)

    return model

# =============================================================================
# Plotting Helpers
# =============================================================================


def setup_example_figure(
    plot_type: str = 'single',
    **kwargs
) -> tuple:
    """Set up matplotlib figure with standardized sizing.

    Args:
        plot_type: Type of plot ('single', 'dual', 'triple', 'band_dos', etc.)
        **kwargs: Additional arguments (e.g., figsize overrides)

    Returns:
        tuple: (fig, ax) for single panel, (fig, axes) for multi-panel

    Example:
        >>> fig, ax = setup_example_figure('single')
        >>> fig, axes = setup_example_figure('dual')
    """
    figsize = DEFAULT_FIGURE_SIZES.get(plot_type, DEFAULT_FIGURE_SIZES['single'])
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']

    n_panels = {'dual': 2, 'triple': 3, 'band_dos': 2}.get(plot_type, 1)

    if n_panels == 1:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        return fig, axes


def save_example_figure(
    fig,
    filename: str,
    dpi: int = 150,
    tight: bool = True
) -> None:
    """Save figure with standard settings.

    Args:
        fig: matplotlib Figure object
        filename: Output filename
        dpi: Resolution (default 150)
        tight: Whether to apply tight_layout (default True)
    """
    if tight:
        plt.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")
