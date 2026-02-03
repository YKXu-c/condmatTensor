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
    D. L. Bergman et al., Phys. Rev. B 76, 094417 (2007)
"""

# example_utils handles path setup automatically
from example_utils import (
    get_example_device,
    build_kagome_spinful_lattice,
    build_kagome_spinful_model,
    setup_example_figure,
    save_example_figure,
)

import torch
import matplotlib.pyplot as plt

from condmatTensor.lattice import generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure, DOSCalculator
from condmatTensor.manybody import LocalMagneticModel


def main():
    """Main function to run spinful Kagome band structure calculation."""
    print("=" * 70)
    print("Spinful Kagome Lattice Band Structure")
    print("=" * 70)

    # Get device
    device = get_example_device("for diagonalization")

    # Parameters
    t = -1.0

    # Build lattice and Hamiltonian using utility functions
    lattice = build_kagome_spinful_lattice()
    tb_model = build_kagome_spinful_model(lattice, t)

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

    # Plot band structure using utility function
    fig, axes = setup_example_figure('dual')

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

    save_example_figure(fig, "kagome_spinful_bandstructure.png")

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

    # Build spinless version using utility function
    from example_utils import build_kagome_lattice, build_kagome_model
    from condmatTensor.lattice import BravaisLattice

    lattice_spinless = build_kagome_lattice(t)
    tb_spinless = build_kagome_model(lattice_spinless, t)

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
