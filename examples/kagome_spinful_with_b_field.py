#!/usr/bin/env python3
"""
Spinful Kagome Lattice in External Magnetic Field

This example demonstrates the Zeeman effect on a spinful Kagome lattice.
An external magnetic field couples to electron spins via the Zeeman term:

    H_B = μ_B * g * (Bx*σx + By*σy + Bz*σz)

**Expected Results:**
- Without B-field: Spin-degenerate bands (6 bands in 3 pairs)
- With B-field: Zeeman splitting (each band splits into ↑ and ↓)
- Splitting magnitude: ΔE = g * μ_B * |B|

**Physical Effect:**
The magnetic field breaks spin degeneracy by adding an energy difference
between spin-up and spin-down electrons. This is observable in:
- Quantum oscillations
- Cyclotron resonance
- Spin polarization measurements

Reference:
    - "Zeeman effect in solids": Ashcroft & Mermin, Ch. 10
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import matplotlib.pyplot as plt

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure
from condmatTensor.manybody import LocalMagneticModel


def build_kagome_spinful_lattice(t: float = -1.0) -> tuple[BravaisLattice, HoppingModel]:
    """Build Kagome lattice with spinful s-orbitals."""
    import math

    # Triangular lattice vectors
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    # 3 sites per unit cell
    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
    ]

    # Note: BravaisLattice doesn't take orbital_labels - set in HoppingModel
    # For spinful case: 3 sites × 2 spin (up/down) = 6 orbitals total
    # num_orbitals is now a list: [2, 2, 2] for 3 sites, each with 2 spin orbitals
    lattice = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2, 2, 2],  # 3 sites, each with 2 spin orbitals
    )

    # Spinful orbital labels: 3 sites × 2 spin = 6 orbitals
    orbital_labels = [
        "A_up", "A_down",
        "B_up", "B_down",
        "C_up", "C_down",
    ]
    tb_model = HoppingModel(lattice, orbital_labels=orbital_labels)

    # Nearest-neighbor hopping (spin-conserving)
    # Must match the spinless case in kagome_bandstructure.py
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

    return lattice, tb_model


def main():
    """Main function to demonstrate Zeeman effect."""
    print("=" * 70)
    print("Spinful Kagome Lattice in External Magnetic Field")
    print("=" * 70)

    # Build lattice and Hamiltonian
    t = -1.0
    lattice, tb_model = build_kagome_spinful_lattice(t)

    print(f"\nLattice: Total: {lattice.total_orbitals} orbitals ({lattice.num_sites} sites)")
    print(f"  Orbitals per site: {lattice.num_orbitals}")

    # Generate k-path
    k_path, ticks = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=50)

    # Build k-space Hamiltonian without B-field
    Hk0 = tb_model.build_Hk(k_path)

    # Diagonalize without B-field
    eig0, _ = diagonalize(Hk0.tensor)

    print(f"\nWithout B-field:")
    print(f"  Energy range: [{eig0.min():.4f}, {eig0.max():.4f}]")

    # Test different magnetic field strengths
    B_values = [0.0, 0.1, 0.3, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, Bz in enumerate(B_values):
        # Add magnetic field in z-direction
        model = LocalMagneticModel(H0=Hk0)
        Hk_B = model.add_effective_magnetic_field(Hk0, B_field=(0, 0, Bz))

        eig_B, _ = diagonalize(Hk_B.tensor)

        # Plot bands
        bs = BandStructure()
        bs.compute(eig_B, k_path, ticks)
        bs.plot(ax=axes[idx], title=f"B_z = {Bz}")

        # Print Zeeman splitting estimate
        if Bz > 0:
            # Estimate splitting from band pairs
            n_pairs = eig_B.shape[1] // 2
            splittings = []
            for i in range(n_pairs):
                # Mean energy difference between band 2i and 2i+1
                splitting = (eig_B[:, 2*i].mean() - eig_B[:, 2*i+1].mean()).abs()
                splittings.append(splitting.item())

            avg_splitting = torch.tensor(splittings).mean().item()
            print(f"  B_z = {Bz}: Average Zeeman splitting = {avg_splitting:.4f}")

    plt.tight_layout()
    plt.savefig("kagome_spinful_zeeman_sweep.png", dpi=150)
    print("\nPlot saved to kagome_spinful_zeeman_sweep.png")

    # Detailed comparison: B = 0 vs B = 0.3
    print("\n" + "=" * 70)
    print("Detailed comparison: B = 0 vs B = (0, 0, 0.3)")
    print("=" * 70)

    Bz = 0.3
    model = LocalMagneticModel(H0=Hk0)
    Hk_B = model.add_effective_magnetic_field(Hk0, B_field=(0, 0, Bz))
    eig_B, _ = diagonalize(Hk_B.tensor)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bs0 = BandStructure()
    bs0.compute(eig0, k_path, ticks)
    bs0.plot(ax=axes[0], title="Without B-field (spin degenerate)")

    bsB = BandStructure()
    bsB.compute(eig_B, k_path, ticks)
    bsB.plot(ax=axes[1], title=f"With B-field B_z = {Bz}")

    plt.tight_layout()
    plt.savefig("kagome_spinful_zeeman_comparison.png", dpi=150)
    print("\nPlot saved to kagome_spinful_zeeman_comparison.png")

    # Analyze Zeeman splitting at Gamma point
    print("\nZeeman splitting analysis at Gamma point:")
    print("  Band   |   E(B=0)   |   E(B>0)   |   Splitting")
    print("  " + "-" * 50)

    # Find Gamma point (k = 0)
    k_idx = 0  # Assuming first k-point is Gamma

    for i in range(min(6, eig0.shape[1])):
        e0 = eig0[k_idx, i].item()
        eB = eig_B[k_idx, i].item()
        splitting = abs(eB - e0)
        print(f"  {i:2d}     |   {e0:7.4f}   |   {eB:7.4f}   |   {splitting:7.4f}")

    # Test in-plane magnetic field
    print("\n" + "=" * 70)
    print("Testing in-plane magnetic field B = (0.3, 0, 0)")
    print("=" * 70)

    Hk_Bx = model.add_effective_magnetic_field(Hk0, B_field=(0.3, 0, 0))
    eig_Bx, _ = diagonalize(Hk_Bx.tensor)

    Hk_By = model.add_effective_magnetic_field(Hk0, B_field=(0, 0.3, 0))
    eig_By, _ = diagonalize(Hk_By.tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bsBx = BandStructure()
    bsBx.compute(eig_Bx, k_path, ticks)
    bsBx.plot(ax=axes[0], title="B_x = 0.3")

    bsBy = BandStructure()
    bsBy.compute(eig_By, k_path, ticks)
    bsBy.plot(ax=axes[1], title="B_y = 0.3")

    bsBz = BandStructure()
    bsBz.compute(eig_B, k_path, ticks)
    bsBz.plot(ax=axes[2], title="B_z = 0.3")

    plt.tight_layout()
    plt.savefig("kagome_spinful_zeeman_directions.png", dpi=150)
    print("\nPlot saved to kagome_spinful_zeeman_directions.png")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
