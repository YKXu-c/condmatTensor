#!/usr/bin/env python3
"""
Spinful Kagome Lattice Band Structure Example

This example demonstrates the spinful Kagome lattice with explicit spin degrees
of freedom. The Kagome lattice has 3 sites per unit cell, each with s-orbital
↑ and ↓, giving 6 spinor orbitals total.

**Spinor Convention:**
- Each orbital becomes a spinor: [orb_0_up, orb_0_down, orb_1_up, orb_1_down, ...]
- Hamiltonian labels: ['k', 'orb_i', 'orb_j'] where orbitals include spin
- Without spin-orbit coupling or magnetic field: bands are spin-degenerate

**Expected Results:**
- 6 bands (3 orbitals × 2 spin)
- Spin degeneracy without SOC or B-field
- Flat band at E = -2|t|
- Dirac points at K point

Reference:
    - D. L. Bergman et al., Phys. Rev. B 76, 094417 (2007)
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
from condmatTensor.analysis import BandStructure, DOSCalculator
from condmatTensor.manybody import LocalMagneticModel


def build_kagome_spinful_lattice(t: float = -1.0) -> BravaisLattice:
    """
    Build Kagome lattice with spinful s-orbitals.

    Returns:
        BravaisLattice object with 3 sites, each with spin ↑ and ↓
    """
    import math

    # Triangular lattice vectors
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    # 3 sites per unit cell (forming a triangle)
    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
    ]

    # Note: BravaisLattice doesn't take orbital_labels - set in TightBindingModel
    # For spinful case: 3 sites × 2 spin (up/down) = 6 orbitals total
    # num_orbitals is now a list: [2, 2, 2] for 3 sites, each with 2 spin orbitals
    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[2, 2, 2],  # 3 sites, each with 2 spin orbitals
    )


def build_kagome_spinful_hamiltonian(lattice: BravaisLattice, t: float = -1.0) -> TightBindingModel:
    """
    Build spinful tight-binding Hamiltonian for Kagome lattice.

    Nearest-neighbor hopping (spin-conserving):
        - A <-> B, B <-> C, C <-> A with displacement [0, 0]
        - Plus periodic boundary connections

    Args:
        lattice: BravaisLattice object
        t: Hopping parameter (default -1)

    Returns:
        TightBindingModel with spinful orbitals (A_up, A_down, B_up, B_down, C_up, C_down)
    """
    # Spinful orbital labels: 3 sites × 2 spin = 6 orbitals
    orbital_labels = [
        "A_up", "A_down",
        "B_up", "B_down",
        "C_up", "C_down",
    ]

    tb_model = TightBindingModel(lattice, orbital_labels=orbital_labels)

    # Nearest-neighbor hopping (spin-conserving)
    # Must match the spinless case in kagome_bandstructure.py

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

    return tb_model


def main():
    """Main function to run spinful Kagome band structure calculation."""
    print("=" * 70)
    print("Spinful Kagome Lattice Band Structure")
    print("=" * 70)

    # Build lattice and Hamiltonian
    t = -1.0
    lattice = build_kagome_spinful_lattice(t)
    tb_model = build_kagome_spinful_hamiltonian(lattice, t)

    print(f"\nLattice: Total: {lattice.total_orbitals} orbitals ({lattice.num_sites} sites)")
    print(f"  Orbitals per site: {lattice.num_orbitals}")
    print(f"  Orbital labels: {tb_model.orbital_labels}")

    # Generate k-path: Γ -> K -> M -> Γ
    k_path, ticks = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=50)

    # Build k-space Hamiltonian
    Hk = tb_model.build_Hk(k_path)

    print(f"\nHamiltonian shape: {Hk.shape}")
    print(f"Labels: {Hk.labels}")

    # Diagonalize (extract tensor from BaseTensor)
    eigenvalues, eigenvectors = diagonalize(Hk.tensor)

    print(f"\nEigenvalues shape: {eigenvalues.shape}")
    print(f"Energy range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

    # Check for flat band at E = -2|t|
    expected_flat_band_energy = -2 * abs(t)
    flat_band_idx = torch.argmin(torch.abs(eigenvalues.mean(dim=0) - expected_flat_band_energy))
    print(f"\nFlat band check:")
    print(f"  Expected flat band energy: {expected_flat_band_energy:.4f}")
    print(f"  Closest band mean energy: {eigenvalues.mean(dim=0)[flat_band_idx]:.4f}")
    print(f"  Band variance (flatness): {eigenvalues[:, flat_band_idx].var():.6e}")

    # Plot band structure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bs = BandStructure()
    bs.compute(eigenvalues, k_path, ticks)
    bs.plot(ax=axes[0], title="Spinful Kagome Lattice (6 bands)")

    # Compute DOS
    dos_calc = DOSCalculator()
    omega = torch.linspace(-4, 2, 500)
    dos_calc.from_eigenvalues(eigenvalues, omega, eta=0.02)
    dos = dos_calc.rho

    axes[1].plot(omega.numpy(), dos.numpy(), 'b-', linewidth=2)
    axes[1].set_xlabel("Energy ($|t|$)", fontsize=12)
    axes[1].set_ylabel("DOS", fontsize=12)
    axes[1].set_title("Density of States", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(expected_flat_band_energy, color='r', linestyle='--',
                   label=f'Flat band at E={expected_flat_band_energy:.1f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("kagome_spinful_bandstructure.png", dpi=150)
    print("\nPlot saved to kagome_spinful_bandstructure.png")

    # Verify spin degeneracy (bands should come in pairs)
    print("\nSpin degeneracy check:")
    eig_sorted = torch.sort(eigenvalues, dim=1).values
    for i in range(0, eigenvalues.shape[1], 2):
        if i + 1 < eigenvalues.shape[1]:
            diff = (eig_sorted[:, i] - eig_sorted[:, i + 1]).abs().max()
            print(f"  Bands {i//2} pair (spin up/down): max difference = {diff:.6e}")

    # Test LocalMagneticModel: Build spinful Hamiltonian from spinless
    print("\n" + "=" * 70)
    print("Testing LocalMagneticModel: spinless -> spinful conversion")
    print("=" * 70)

    # Build spinless version (num_orbitals=1 for spinless)
    import math
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors_spinless = torch.stack([a1, a2])
    basis_positions_spinless = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
    ]
    lattice_spinless = BravaisLattice(
        cell_vectors=cell_vectors_spinless,
        basis_positions=basis_positions_spinless,
        num_orbitals=[1, 1, 1],  # Spinless: 1 orbital per site
    )
    tb_spinless = TightBindingModel(lattice_spinless, orbital_labels=["A", "B", "C"])

    # Add hoppings (same as before but without spin)
    tb_spinless.add_hopping("A", "B", [0, 0], t)
    tb_spinless.add_hopping("B", "C", [0, 0], t)
    tb_spinless.add_hopping("C", "A", [0, 0], t)
    tb_spinless.add_hopping("A", "B", [0, -1], t)
    tb_spinless.add_hopping("B", "C", [-1, 0], t)
    tb_spinless.add_hopping("C", "A", [0, -1], t)

    Hk_spinless = tb_spinless.build_Hk(k_path)

    # Convert to spinful using LocalMagneticModel
    model = LocalMagneticModel()
    Hk_spinful = model.build_spinful_hamiltonian(Hk_spinless)

    print(f"Spinless Hamiltonian shape: {Hk_spinless.shape}")
    print(f"Spinful Hamiltonian shape: {Hk_spinful.shape}")

    # Diagonalize and compare
    eig_spinless, _ = diagonalize(Hk_spinless.tensor)
    eig_spinful, _ = diagonalize(Hk_spinful.tensor)

    print("\nSpinless -> spinful verification:")
    print(f"  Spinless bands: {eig_spinless.shape[1]}")
    print(f"  Spinful bands: {eig_spinful.shape[1]} (should be 2x)")

    # Each spinless band should appear twice in spinful
    for i in range(eig_spinless.shape[1]):
        # Find closest matches in spinful
        e = eig_spinless[:, i].mean()
        distances = torch.abs(eig_spinful.mean(dim=0) - e)
        closest = torch.topk(distances, k=2, largest=False).indices
        print(f"  Spinless band {i}: matched to spinful bands {closest.tolist()}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
