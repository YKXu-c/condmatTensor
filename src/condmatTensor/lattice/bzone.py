"""Brillouin zone utilities: k-mesh and k-path generation."""

from typing import List, Optional
import torch


def generate_kmesh(lattice, nk: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate uniform k-mesh in fractional coordinates.

    Creates a uniform grid of k-points covering the Brillouin zone.

    Args:
        lattice: BravaisLattice object
        nk: Number of k-points along each dimension
        device: Device to place tensor on (default: CPU)

    Returns:
        K-points in fractional coordinates, shape (nk^dim, dim)
    """
    dim = lattice.dim

    # Create 1D grids for each dimension
    grids = torch.meshgrid(
        *[torch.linspace(0, 1, nk, device=device) for _ in range(dim)],
        indexing="ij"
    )

    # Stack and reshape to (nk^dim, dim)
    k_frac = torch.stack(grids, dim=-1).reshape(-1, dim)

    return k_frac


def generate_k_path(
    lattice,
    points: List[str],
    n_per_segment: int,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, List[tuple[int, str]]]:
    """
    Generate k-point path along high-symmetry lines.

    Args:
        lattice: BravaisLattice object
        points: List of high-symmetry point labels (e.g., ['G', 'K', 'M', 'G'])
        n_per_segment: Number of k-points per segment (excluding endpoints)
        device: Device to place tensor on

    Returns:
        k_path: K-points in fractional coordinates, shape (n_total, dim)
        ticks: List of (index, label) for plot markers
    """
    sym_points = lattice.high_symmetry_points()

    # Build path segments
    path_segments = []
    ticks = [(0, points[0])]

    for i in range(len(points) - 1):
        start_label, end_label = points[i], points[i + 1]
        start = sym_points[start_label]
        end = sym_points[end_label]

        # Create linear interpolation between points
        t = torch.linspace(0, 1, n_per_segment, device=device)
        segment = start[None, :] + t[:, None] * (end - start)[None, :]

        if i > 0:  # Skip duplicate endpoint
            segment = segment[1:]

        path_segments.append(segment)
        ticks.append((ticks[-1][0] + len(segment), end_label))

    k_path = torch.cat(path_segments, dim=0)

    return k_path, ticks


def k_frac_to_cart(k_frac: torch.Tensor, lattice) -> torch.Tensor:
    """
    Convert k-points from fractional to Cartesian coordinates.

    k_cart = k_frac @ b where b are reciprocal lattice vectors

    Args:
        k_frac: K-points in fractional coordinates, shape (N_k, dim)
        lattice: BravaisLattice object

    Returns:
        K-points in Cartesian coordinates, shape (N_k, dim)
    """
    b = lattice.reciprocal_vectors()
    return torch.matmul(k_frac, b.T)
