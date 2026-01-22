#!/usr/bin/env python3
"""
Effective Array Optimizer for Kagome-F Lattice

This example demonstrates the EffectiveArrayOptimizer which finds an
effective local magnetic moment model that reproduces the full Kagome-F
band structure.

**Goal:**
Find parameters J_eff and S_eff such that:
    H_eff = H_cc + J_eff @ S_eff

reproduces the band structure of:
    H_full = H_cc + H_cf + H_ff

**Method:**
Bayesian optimization to minimize the L2 norm of eigenvalue differences:
    min_{J, S} ||eig(H_full) - eig(H_eff)||^2

**Expected Results:**
- Optimized J_eff and S_eff
- Band structure comparison showing good agreement
- Verification metrics (MAE, RMSE, correlation)

Reference:
    - "Effective Hamiltonians for heavy fermion systems" - Coleman, PRB (1987)
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
from condmatTensor.optimization import EffectiveArrayOptimizer


def build_kagome_lattice(t: float = -1.0) -> BravaisLattice:
    """Build pure Kagome lattice (conduction electrons only)."""
    import math

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
        num_orbitals=[1, 1, 1],  # 3 sites, each with 1 orbital
    )


def build_kagome_hamiltonian(lattice: BravaisLattice, t: float = -1.0) -> BaseTensor:
    """Build Kagome tight-binding Hamiltonian."""
    tb_model = TightBindingModel(lattice)

    # Nearest-neighbor hopping (must match kagome_bandstructure.py)
    # Using add_hermitian=False to match the reference implementation
    # A <-> B hopping
    tb_model.add_hopping(0, 1, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(1, 0, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(0, 1, [-1, 0], t, add_hermitian=False)
    tb_model.add_hopping(1, 0, [1, 0], t, add_hermitian=False)
    # A <-> C hopping
    tb_model.add_hopping(0, 2, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(2, 0, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(0, 2, [0, -1], t, add_hermitian=False)
    tb_model.add_hopping(2, 0, [0, 1], t, add_hermitian=False)
    # B <-> C hopping
    tb_model.add_hopping(1, 2, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(2, 1, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(1, 2, [1, -1], t, add_hermitian=False)
    tb_model.add_hopping(2, 1, [-1, 1], t, add_hermitian=False)

    return tb_model.build_Hk(torch.tensor([[0.0, 0.0]]))


def build_kagome_f_system(
    t: float = -1.0,
    t_f: float = -0.5,
    epsilon_f: float = 0.0,
) -> tuple[BravaisLattice, BaseTensor]:
    """Build Kagome-F system with f-orbital.

    Returns:
        (lattice, H_full) where H_full includes Kagome and f-orbitals
    """
    import math

    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    # 4 sites: 3 Kagome + 1 f-orbital
    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
        torch.tensor([1/3, 1/3]),  # f-orbital at center
    ]

    # Note: BravaisLattice doesn't take orbital_labels - set in TightBindingModel
    orbital_labels = ["A", "B", "C", "f"]
    lattice = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[1, 1, 1, 1],  # 4 sites, each with 1 orbital
    )
    tb_model = TightBindingModel(lattice, orbital_labels=orbital_labels)

    # Kagome-Kagome hopping (must match kagome_with_f_bandstructure.py)
    # Using add_hermitian=False to match the reference implementation
    # A <-> B hopping
    tb_model.add_hopping("A", "B", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("B", "A", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("A", "B", [-1, 0], t, add_hermitian=False)
    tb_model.add_hopping("B", "A", [1, 0], t, add_hermitian=False)
    # A <-> C hopping
    tb_model.add_hopping("A", "C", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("C", "A", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("A", "C", [0, -1], t, add_hermitian=False)
    tb_model.add_hopping("C", "A", [0, 1], t, add_hermitian=False)
    # B <-> C hopping
    tb_model.add_hopping("B", "C", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("C", "B", [0, 0], t, add_hermitian=False)
    tb_model.add_hopping("B", "C", [1, -1], t, add_hermitian=False)
    tb_model.add_hopping("C", "B", [-1, 1], t, add_hermitian=False)

    # Kagome-f hopping
    tb_model.add_hopping("A", "f", [0, 0], t_f)
    tb_model.add_hopping("B", "f", [0, 0], t_f)
    tb_model.add_hopping("C", "f", [0, 0], t_f)

    # f-orbital on-site energy
    tb_model.add_hopping("f", "f", [0, 0], epsilon_f)

    # Build on a test k-path
    k_path, _ = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=30)
    H_full = tb_model.build_Hk(k_path)

    return lattice, H_full


def main():
    """Main function to run effective array optimization."""
    print("=" * 70)
    print("Effective Array Optimizer for Kagome-F Lattice")
    print("=" * 70)

    # System parameters
    t = -1.0
    t_f = -0.3
    epsilon_f = 0.5  # f-orbital above Fermi level

    print(f"\nSystem parameters:")
    print(f"  t (Kagome-Kagome) = {t}")
    print(f"  t_f (Kagome-f) = {t_f}")
    print(f"  ε_f (f-orbital energy) = {epsilon_f}")

    # Build Kagome-only Hamiltonian (reference conduction)
    lattice_cc = build_kagome_lattice(t)
    k_path_cc, _ = generate_k_path(lattice_cc, ["G", "K", "M", "G"], n_per_segment=30)

    # Build spinful H_cc_0
    from condmatTensor.manybody import LocalMagneticModel

    H_cc_spinless = build_kagome_hamiltonian(lattice_cc, t)
    # Rebuild on k_path with correct hopping
    tb_cc = TightBindingModel(lattice_cc)
    # A <-> B hopping
    tb_cc.add_hopping(0, 1, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(1, 0, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(0, 1, [-1, 0], t, add_hermitian=False)
    tb_cc.add_hopping(1, 0, [1, 0], t, add_hermitian=False)
    # A <-> C hopping
    tb_cc.add_hopping(0, 2, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(2, 0, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(0, 2, [0, -1], t, add_hermitian=False)
    tb_cc.add_hopping(2, 0, [0, 1], t, add_hermitian=False)
    # B <-> C hopping
    tb_cc.add_hopping(1, 2, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(2, 1, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(1, 2, [1, -1], t, add_hermitian=False)
    tb_cc.add_hopping(2, 1, [-1, 1], t, add_hermitian=False)
    H_cc_0 = tb_cc.build_Hk(k_path_cc)

    # Build full Kagome-F Hamiltonian
    lattice_full, H_full = build_kagome_f_system(t, t_f, epsilon_f)

    print(f"\nHamiltonian dimensions:")
    print(f"  H_cc_0: {H_cc_0.shape}")
    print(f"  H_full: {H_full.shape}")

    # Note: For this example, we need to build spinful versions
    # Build spinful H_cc_0
    model = LocalMagneticModel()
    H_cc_0_spinful = model.build_spinful_hamiltonian(H_cc_0)

    # For H_full, we also need spinful
    # Let's rebuild with spinful orbitals
    import math
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
        torch.tensor([1/3, 1/3]),
    ]

    orbital_labels_full = ["A", "B", "C", "f"]
    lattice_full_spinless = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[1, 1, 1, 1],  # 4 sites, each with 1 orbital
    )
    tb_full = TightBindingModel(lattice_full_spinless, orbital_labels=orbital_labels_full)
    # Kagome-Kagome hopping (must match kagome_with_f_bandstructure.py)
    # Using add_hermitian=False to match the reference implementation
    # A <-> B hopping
    tb_full.add_hopping("A", "B", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("B", "A", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("A", "B", [-1, 0], t, add_hermitian=False)
    tb_full.add_hopping("B", "A", [1, 0], t, add_hermitian=False)
    # A <-> C hopping
    tb_full.add_hopping("A", "C", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("C", "A", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("A", "C", [0, -1], t, add_hermitian=False)
    tb_full.add_hopping("C", "A", [0, 1], t, add_hermitian=False)
    # B <-> C hopping
    tb_full.add_hopping("B", "C", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("C", "B", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("B", "C", [1, -1], t, add_hermitian=False)
    tb_full.add_hopping("C", "B", [-1, 1], t, add_hermitian=False)
    # Kagome-f hopping
    tb_full.add_hopping("A", "f", [0, 0], t_f)
    tb_full.add_hopping("B", "f", [0, 0], t_f)
    tb_full.add_hopping("C", "f", [0, 0], t_f)
    # f-orbital on-site
    tb_full.add_hopping("f", "f", [0, 0], epsilon_f)

    H_full_spinless = tb_full.build_Hk(k_path_cc)
    H_full_spinful = model.build_spinful_hamiltonian(H_full_spinless)

    print(f"\nSpinful Hamiltonians:")
    print(f"  H_cc_0_spinful: {H_cc_0_spinful.shape}")
    print(f"  H_full_spinful: {H_full_spinful.shape}")

    # Set up optimizer
    print("\n" + "=" * 70)
    print("Setting up EffectiveArrayOptimizer")
    print("=" * 70)

    optimizer = EffectiveArrayOptimizer(
        H_cc_0=H_cc_0_spinful,
        H_full=H_full_spinful,
        method="eigenvalue",
    )

    print(f"F-orbital indices: {optimizer.f_indices}")

    # Run optimization (fewer iterations for demonstration)
    print("\nRunning optimization...")
    J_eff, S_eff = optimizer.optimize(
        J_bounds=(0.01, 2.0),
        n_init=10,
        n_iter=30,
        verbose=True,
    )

    # Verify the effective model
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    metrics = optimizer.verify(verbose=True)

    # Plot comparison
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Eigenvalues comparison
    eig_full = optimizer._compute_eigenvalues(optimizer.H_full)
    H_eff = optimizer._build_effective_hamiltonian(J_eff, S_eff, optimizer.H_cc_0.tensor.device)
    eig_eff = optimizer._compute_eigenvalues(H_eff)

    # Plot full system
    k_axis = torch.arange(eig_full.shape[0], dtype=torch.float64)
    for i in range(eig_full.shape[1]):
        axes[0].plot(k_axis.numpy(), eig_full[:, i].cpu().numpy(), 'b-', alpha=0.6)

    axes[0].set_xlabel("k-path index", fontsize=12)
    axes[0].set_ylabel("Energy ($|t|$)", fontsize=12)
    axes[0].set_title("Full Kagome-F System", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Plot effective model
    for i in range(eig_eff.shape[1]):
        axes[1].plot(k_axis.numpy(), eig_eff[:, i].cpu().numpy(), 'r-', alpha=0.8)

    axes[1].set_xlabel("k-path index", fontsize=12)
    axes[1].set_ylabel("Energy ($|t|$)", fontsize=12)
    axes[1].set_title(f"Effective Model ($J_{{eff}}$={J_eff:.3f})", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kagome_f_effective_array_comparison.png", dpi=150)
    print("\nPlot saved to kagome_f_effective_array_comparison.png")

    # Comparison plot using built-in method
    fig, ax = plt.subplots(figsize=(10, 6))
    optimizer.plot_comparison(ax=ax)
    plt.savefig("kagome_f_effective_array_bands.png", dpi=150)
    print("Band comparison saved to kagome_f_effective_array_bands.png")

    # Perturbation theory estimate
    print("\n" + "=" * 70)
    print("Perturbation Theory Estimate")
    print("=" * 70)

    J_pert, S_pert = optimizer.perturbation_theory(epsilon_f, t_f)
    print(f"  J_eff (perturbation theory) = {J_pert:.6f}")
    print(f"  S_eff (perturbation theory) = {S_pert.tolist()}")
    print(f"  J_eff (optimized) = {J_eff:.6f}")
    print(f"  S_eff (optimized) = {S_eff.tolist()}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
