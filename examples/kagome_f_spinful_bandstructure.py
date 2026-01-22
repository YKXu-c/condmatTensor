#!/usr/bin/env python3
"""
Kagome-F Lattice with Spinful Orbitals Example

This example demonstrates the Kagome-F lattice with central f-orbital, all with
explicit spin degrees of freedom. The model consists of:
- 3 Kagome sites (s-orbitals) × 2 spin = 6 orbitals
- 1 f-orbital site × 2 spin = 2 orbitals
- Total: 8 spinor orbitals

**Model Hamiltonian:**
    H = H_Kagome + H_f + H_hyb [+ H_mag]

Where:
    H_Kagome: Kagome-Kagome hopping (t)
    H_f: f-orbital on-site energy (ε_f)
    H_hyb: Kagome-f hybridization (t_f)
    H_mag: Optional local magnetic moments on f-orbitals

**Expected Results:**
- 8 bands total (4 sites × 2 spin)
- f-orbital hybridization with Kagome bands
- Tunable via t_f (hybridization) and ε_f (f-energy)

Reference:
    - Heavy fermion systems: Hewson, "The Kondo Problem" (1993)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import matplotlib.pyplot as plt

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, TightBindingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure
from condmatTensor.manybody import LocalMagneticModel


def build_kagome_f_spinful_lattice(
    t: float = -1.0,
    t_f: float = -0.5,
    epsilon_f: float = 0.0,
) -> tuple[BravaisLattice, TightBindingModel]:
    """
    Build Kagome-F lattice with spinful orbitals.

    Args:
        t: Kagome-Kagome hopping
        t_f: Kagome-f hybridization
        epsilon_f: f-orbital on-site energy

    Returns:
        (lattice, tb_model) tuple
    """
    import math

    # Triangular lattice vectors
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    # 4 sites per unit cell: 3 Kagome sites + 1 f-orbital at center
    basis_positions = [
        torch.tensor([0.0, 0.0]),      # Site A (Kagome)
        torch.tensor([0.5, 0.0]),      # Site B (Kagome)
        torch.tensor([0.25, sqrt3 / 4]),  # Site C (Kagome)
        torch.tensor([1/3, 1/3]),      # Site f (center of triangle)
    ]

    # Note: BravaisLattice doesn't take orbital_labels - set in TightBindingModel
    # For spinful case: 4 sites × 2 spin (up/down) = 8 orbitals total
    # num_orbitals is now a list: [2, 2, 2, 2] for 4 sites, each with 2 spin orbitals
    lattice = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2, 2, 2, 2],  # 4 sites, each with 2 spin orbitals
    )

    # Orbital labels with spin: 4 sites × 2 spin = 8 orbitals
    orbital_labels = [
        "A_up", "A_down",
        "B_up", "B_down",
        "C_up", "C_down",
        "f_up", "f_down",
    ]
    tb_model = TightBindingModel(lattice, orbital_labels=orbital_labels)

    # Kagome-Kagome hopping (spin-conserving)
    # Must match the spinless case in kagome_with_f_bandstructure.py and kagome_bandstructure.py
    # Using add_hermitian=False to match the reference implementation

    # A <-> B hopping
    # Intra-cell: A(0,0) -> B(0,0)
    tb_model.add_hopping("A_up", "B_up", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("B_up", "A_up", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("A_down", "B_down", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("B_down", "A_down", [0, 0], t, add_hermitian=False)

    # Inter-cell: A(-1,0) -> B(0,0) [equivalent to A(0,0) -> B(1,0)]
    tb_model.add_hopping("A_up", "B_up", [-1, 0], t, add_hermitian=False)
    tb_model.add_hopping("B_up", "A_up", [1, 0], t, add_hermitian=False)
    tb_model.add_hopping("A_down", "B_down", [-1, 0], t, add_hermitian=False)
    tb_model.add_hopping("B_down", "A_down", [1, 0], t, add_hermitian=False)

    # A <-> C hopping
    # Intra-cell: A(0,0) -> C(0,0)
    tb_model.add_hopping("A_up", "C_up", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("C_up", "A_up", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("A_down", "C_down", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("C_down", "A_down", [0, 0], t, add_hermitian=False)

    # Inter-cell: A(0,-1) -> C(0,0) [equivalent to A(0,0) -> C(0,1)]
    tb_model.add_hopping("A_up", "C_up", [0, -1], t, add_hermitian=False)
    tb_model.add_hopping("C_up", "A_up", [0, 1], t, add_hermitian=False)
    tb_model.add_hopping("A_down", "C_down", [0, -1], t, add_hermitian=False)
    tb_model.add_hopping("C_down", "A_down", [0, 1], t, add_hermitian=False)

    # B <-> C hopping
    # Intra-cell: B(0,0) -> C(0,0)
    tb_model.add_hopping("B_up", "C_up", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("C_up", "B_up", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("B_down", "C_down", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("C_down", "B_down", [0, 0], t, add_hermitian=False)

    # Inter-cell: B(1,-1) -> C(0,0) [equivalent to B(0,0) -> C(-1,1)]
    tb_model.add_hopping("B_up", "C_up", [1, -1], t, add_hermitian=False)
    tb_model.add_hopping("C_up", "B_up", [-1, 1], t, add_hermitian=False)
    tb_model.add_hopping("B_down", "C_down", [1, -1], t, add_hermitian=False)
    tb_model.add_hopping("C_down", "B_down", [-1, 1], t, add_hermitian=False)

    # Kagome-f hopping (spin-conserving)
    # f-orbital connects to all 3 Kagome sites in the same unit cell
    tb_model.add_hopping("A_up", "f_up", [0, 0], t_f)
    tb_model.add_hopping("A_down", "f_down", [0, 0], t_f)
    tb_model.add_hopping("B_up", "f_up", [0, 0], t_f)
    tb_model.add_hopping("B_down", "f_down", [0, 0], t_f)
    tb_model.add_hopping("C_up", "f_up", [0, 0], t_f)
    tb_model.add_hopping("C_down", "f_down", [0, 0], t_f)

    # f-orbital on-site energy
    tb_model.add_hopping("f_up", "f_up", [0, 0], epsilon_f)
    tb_model.add_hopping("f_down", "f_down", [0, 0], epsilon_f)

    return lattice, tb_model


def plot_comparison(tf_values: list, epsilon_f: float = 0.0):
    """Plot band structure for different t_f values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, t_f in enumerate(tf_values):
        lattice, tb_model = build_kagome_f_spinful_lattice(t_f=t_f, epsilon_f=epsilon_f)
        k_path, ticks = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=50)

        Hk = tb_model.build_Hk(k_path)
        eigenvalues, _ = diagonalize(Hk.tensor)

        bs = BandStructure()
        bs.compute(eigenvalues, k_path, ticks)
        bs.plot(ax=axes[idx], title=f"t_f = {t_f:.2f}")

    plt.tight_layout()
    plt.savefig("kagome_f_spinful_tf_sweep.png", dpi=150)
    print("Comparison plot saved to kagome_f_spinful_tf_sweep.png")


def main():
    """Main function to run Kagome-F spinful band structure calculation."""
    print("=" * 70)
    print("Kagome-F Lattice with Spinful Orbitals")
    print("=" * 70)

    # Build lattice and Hamiltonian
    t = -1.0
    t_f = -0.5
    epsilon_f = 0.0

    lattice, tb_model = build_kagome_f_spinful_lattice(t, t_f, epsilon_f)

    print(f"\nLattice: Total: {lattice.total_orbitals} orbitals ({lattice.num_sites} sites)")
    print(f"  Orbitals per site: {lattice.num_orbitals}")
    print(f"  Orbital labels: {tb_model.orbital_labels}")

    # Generate k-path
    k_path, ticks = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=50)

    # Build k-space Hamiltonian
    Hk = tb_model.build_Hk(k_path)

    print(f"\nHamiltonian shape: {Hk.shape}")
    print(f"Labels: {Hk.labels}")

    # Diagonalize
    eigenvalues, eigenvectors = diagonalize(Hk.tensor)

    print(f"\nEigenvalues shape: {eigenvalues.shape}")
    print(f"Energy range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

    # Plot band structure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bs = BandStructure()
    bs.compute(eigenvalues, k_path, ticks)
    bs.plot(ax=axes[0], title=f"Kagome-F Lattice with Spin (8 bands, t_f={t_f})")

    # Color by f-orbital character
    # Compute f-orbital weight in each eigenstate
    f_weight = torch.zeros_like(eigenvalues)
    f_indices = [6, 7]  # f_up, f_down indices
    for k in range(eigenvalues.shape[0]):
        for band in range(eigenvalues.shape[1]):
            # Sum of squared amplitudes on f-orbitals
            f_weight[k, band] = torch.sum(torch.abs(eigenvectors[k, f_indices, band]) ** 2)

    # Plot with f-orbital weight
    for band in range(eigenvalues.shape[1]):
        axes[1].scatter(
            torch.arange(eigenvalues.shape[0]),
            eigenvalues[:, band],
            c=f_weight[:, band],
            cmap='viridis',
            s=10,
            vmin=0,
            vmax=1,
        )

    axes[1].set_xlabel("k-path index", fontsize=12)
    axes[1].set_ylabel("Energy ($|t|$)", fontsize=12)
    axes[1].set_title("Band Structure (colored by f-orbital weight)", fontsize=12)
    plt.colorbar(axes[1].collections[0], ax=axes[1], label="f-orbital weight")

    # Add high-symmetry point markers
    if ticks is not None:
        tick_indices = [t[0] for t in ticks]
        tick_labels = [t[1] for t in ticks]
        axes[1].set_xticks(tick_indices)
        axes[1].set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig("kagome_f_spinful_bandstructure.png", dpi=150)
    print("\nPlot saved to kagome_f_spinful_bandstructure.png")

    # Test with local magnetic moments on f-orbital
    print("\n" + "=" * 70)
    print("Testing local magnetic moment on f-orbital")
    print("=" * 70)

    # H = H0 + J * S_f · σ_f
    J = -0.3
    S_f = torch.tensor([0.0, 0.0, 1.0])  # Local moment pointing up (z-direction)

    model = LocalMagneticModel(H0=Hk, J=J)
    Hk_mag = model.add_magnetic_exchange(Hk, S_config=S_f.unsqueeze(0), J=J)

    eigenvalues_mag, _ = diagonalize(Hk_mag.tensor)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bs_nomag = BandStructure()
    bs_nomag.compute(eigenvalues, k_path, ticks)
    bs_nomag.plot(ax=axes[0], title="Without magnetic moment")

    bs_mag = BandStructure()
    bs_mag.compute(eigenvalues_mag, k_path, ticks)
    bs_mag.plot(ax=axes[1], title=f"With local moment S=({S_f[0]:.1f}, {S_f[1]:.1f}, {S_f[2]:.1f}), J={J}")

    plt.tight_layout()
    plt.savefig("kagome_f_spinful_with_magnetic_moment.png", dpi=150)
    print("\nPlot saved to kagome_f_spinful_with_magnetic_moment.png")

    # Plot t_f dependence
    print("\n" + "=" * 70)
    print("Plotting t_f dependence...")
    print("=" * 70)

    tf_values = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5]
    plot_comparison(tf_values, epsilon_f=0.0)

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
