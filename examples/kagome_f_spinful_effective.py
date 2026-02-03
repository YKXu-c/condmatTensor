#!/usr/bin/env python3
"""
Effective Array Optimizer for Kagome-F Spinful Lattice

This example demonstrates the proper way to use EffectiveArrayOptimizer with
spinful systems. The key is to use the **spinless convention** for lattice
construction, then explicitly convert to spinful using build_spinful_hamiltonian()
with the lattice parameter.

**Proper Usage Pattern:**
1. Build spinless models (num_orbitals without spin): [1, 1, 1] for Kagome
2. Convert to spinful with lattice parameter for correct orbital offsets
3. Pass lattice to optimizer for per-site orbital handling

Reference:
    - "Effective Hamiltonians for heavy fermion systems" - Coleman, PRB (1987)
"""

# example_utils handles path setup automatically
from example_utils import (
    get_example_device,
    build_kagome_lattice,
    build_kagome_f_lattice,
    setup_example_figure,
    save_example_figure,
)

import argparse
import torch
import time
import matplotlib.pyplot as plt

from condmatTensor.core import get_device
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path
from condmatTensor.optimization import EffectiveArrayOptimizer
from condmatTensor.manybody.magnetic import LocalMagneticModel


def build_kagome_hamiltonian_spinless(
    lattice: BravaisLattice,
    k_path: torch.Tensor,
    t: float = -1.0,
) -> HoppingModel:
    """Build Kagome tight-binding Hamiltonian (SPINLESS)."""
    from example_utils import build_kagome_model
    tb_model = build_kagome_model(lattice, t)
    return tb_model


def build_kagome_f_hamiltonian_spinless(
    lattice: BravaisLattice,
    k_path: torch.Tensor,
    t: float = -1.0,
    t_f: float = -0.5,
    epsilon_f: float = 0.0,
) -> HoppingModel:
    """Build Kagome-F tight-binding Hamiltonian (SPINLESS)."""
    from example_utils import build_kagome_f_model
    tb_model = build_kagome_f_model(lattice, t=t, tf=t_f, fd_hybridization=t_f)
    tb_model.add_hopping("F", "F", [0, 0], epsilon_f, add_hermitian=False)
    return tb_model


def plot_comparison(
    optimizer: EffectiveArrayOptimizer,
    save_path: str = "kagome_f_spinful_effective_comparison.png"
):
    """Plot band structure comparison between full and effective models."""
    fig, axes = setup_example_figure('dual')

    eig_full = optimizer._compute_eigenvalues(optimizer.H_full)
    H_eff = optimizer._build_effective_hamiltonian(
        optimizer.J_eff, optimizer.S_eff, optimizer.H_cc_0.tensor.device
    )
    eig_eff = optimizer._compute_eigenvalues(H_eff)

    k_axis = torch.arange(eig_full.shape[0], dtype=torch.float64)

    for i in range(eig_full.shape[1]):
        axes[0].plot(k_axis.numpy(), eig_full[:, i].cpu().numpy(),
                    'b-', alpha=0.6, linewidth=1)

    axes[0].set_xlabel("k-path index", fontsize=12)
    axes[0].set_ylabel("Energy ($|t|$)", fontsize=12)
    axes[0].set_title("Full Kagome-F Spinful System", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    for i in range(eig_eff.shape[1]):
        axes[1].plot(k_axis.numpy(), eig_eff[:, i].cpu().numpy(),
                    'r-', alpha=0.8, linewidth=1.5)

    axes[1].set_xlabel("k-path index", fontsize=12)
    axes[1].set_ylabel("Energy ($|t|$)", fontsize=12)
    axes[1].set_title(f"Effective Model ($J_{{eff}}$={optimizer.J_eff:.3f})", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    save_example_figure(fig, save_path)


def plot_overlaid_bands(
    optimizer: EffectiveArrayOptimizer,
    save_path: str = "kagome_f_spinful_effective_overlaid.png"
):
    """Plot overlaid band structures for direct comparison."""
    fig, ax = setup_example_figure('single')

    eig_full = optimizer._compute_eigenvalues(optimizer.H_full)
    H_eff = optimizer._build_effective_hamiltonian(
        optimizer.J_eff, optimizer.S_eff, optimizer.H_cc_0.tensor.device
    )
    eig_eff = optimizer._compute_eigenvalues(H_eff)

    k_axis = torch.arange(eig_full.shape[0], dtype=torch.float64)

    for i in range(eig_full.shape[1]):
        ax.plot(k_axis.numpy(), eig_full[:, i].cpu().numpy(),
               'b-', alpha=0.3, linewidth=1, label='Full' if i == 0 else '')

    for i in range(eig_eff.shape[1]):
        ax.plot(k_axis.numpy(), eig_eff[:, i].cpu().numpy(),
               'r--', alpha=0.8, linewidth=1.5, label='Effective' if i == 0 else '')

    ax.set_xlabel("k-path index", fontsize=12)
    ax.set_ylabel("Energy ($|t|$)", fontsize=12)
    ax.set_title(f"Band Structure Overlaid Comparison\n$J_{{eff}}$={optimizer.J_eff:.3f}, "
                f"$S_{{eff}}$=[{optimizer.S_eff[0]:.2f}, {optimizer.S_eff[1]:.2f}, {optimizer.S_eff[2]:.2f}]",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_example_figure(fig, save_path)


def main():
    """Main function to run effective array optimization for spinful systems."""
    parser = argparse.ArgumentParser(
        description="Effective Array Optimizer for Kagome-F Spinful Lattice"
    )
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "sober", "botorch", "simple"],
                        help="Bayesian optimization backend (default: auto)")
    parser.add_argument("--n-init", type=int, default=10,
                        help="Number of initial samples (default: 10)")
    parser.add_argument("--n-iter", type=int, default=30,
                        help="Number of optimization iterations (default: 30)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run on (default: auto)")

    args = parser.parse_args()

    print("=" * 70)
    print("Effective Array Optimizer for Kagome-F Spinful Lattice")
    print("=" * 70)

    # System parameters
    t = -1.0
    t_f = -0.3
    epsilon_f = 0.5

    print(f"\nSystem parameters:")
    print(f"  t (Kagome-Kagome) = {t}")
    print(f"  t_f (Kagome-f) = {t_f}")
    print(f"  ε_f (f-orbital energy) = {epsilon_f}")

    # Step 1: Build SPINLESS lattices
    print("\n" + "=" * 70)
    print("Step 1: Build SPINLESS lattices")
    print("=" * 70)

    # Build spinless lattices (num_orbitals=[1,1,1])
    import math
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    basis_kagome = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
    ]

    basis_kagome_f = basis_kagome + [torch.tensor([1/3, 1/3])]

    lattice_cc = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_kagome,
        num_orbitals=[1, 1, 1],
    )

    lattice_full = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_kagome_f,
        num_orbitals=[1, 1, 1, 1],
    )

    print(f"  Kagome lattice: num_orbitals = {lattice_cc.num_orbitals} (spinless)")
    print(f"  Kagome-F lattice: num_orbitals = {lattice_full.num_orbitals} (spinless)")

    # Step 2: Build k-path
    print("\n" + "=" * 70)
    print("Step 2: Build k-path")
    print("=" * 70)

    k_path, k_labels = generate_k_path(lattice_cc, ["G", "K", "M", "G"], n_per_segment=30)
    print(f"  k-path shape: {k_path.shape}")
    print(f"  High-symmetry points: {k_labels}")

    # Step 3: Build SPINLESS Hamiltonians
    print("\n" + "=" * 70)
    print("Step 3: Build SPINLESS Hamiltonians")
    print("=" * 70)

    tb_cc = build_kagome_hamiltonian_spinless(lattice_cc, k_path, t)
    tb_full = build_kagome_f_hamiltonian_spinless(lattice_full, k_path, t, t_f, epsilon_f)

    H_cc_spinless = tb_cc.build_Hk(k_path)
    H_full_spinless = tb_full.build_Hk(k_path)

    print(f"  H_cc_spinless shape: {H_cc_spinless.shape}")
    print(f"  H_full_spinless shape: {H_full_spinless.shape}")

    # Step 4: Convert to SPINFUL
    print("\n" + "=" * 70)
    print("Step 4: Convert to SPINFUL with lattice parameter")
    print("=" * 70)

    model = LocalMagneticModel()
    H_cc_0 = model.build_spinful_hamiltonian(H_cc_spinless, lattice=lattice_cc)
    H_full = model.build_spinful_hamiltonian(H_full_spinless, lattice=lattice_full)

    print(f"  H_cc_0 (spinful) shape: {H_cc_0.shape}")
    print(f"  H_full (spinful) shape: {H_full.shape}")

    # Verify spinful dimensions
    N_k = k_path.shape[0]
    assert H_cc_0.shape == (N_k, 6, 6), f"Expected H_cc_0 shape (N_k, 6, 6), got {H_cc_0.shape}"
    assert H_full.shape == (N_k, 8, 8), f"Expected H_full shape (N_k, 8, 8), got {H_full.shape}"
    print("  Spinful dimension checks passed!")

    # Step 5: Set up optimizer
    print("\n" + "=" * 70)
    print("Step 5: Set up EffectiveArrayOptimizer with lattice")
    print("=" * 70)

    optimizer = EffectiveArrayOptimizer(
        H_cc_0=H_cc_0,
        H_full=H_full,
        method="eigenvalue",
        lattice=lattice_cc,
    )

    print(f"  F-orbital indices: {optimizer.f_indices}")

    # Step 6: Run optimization
    print("\n" + "=" * 70)
    print("Step 6: Run Bayesian optimization")
    print("=" * 70)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = get_device(args.device)
    print(f"  Using device: {device}")

    start_time = time.time()
    J_eff, S_eff = optimizer.optimize(
        J_bounds=(0.01, 2.0),
        n_init=args.n_init,
        n_iter=args.n_iter,
        backend=args.backend,
        verbose=True,
        device=device,
    )
    elapsed_time = time.time() - start_time

    print(f"\n  Optimization completed in {elapsed_time:.2f}s")

    # Step 7: Verify
    print("\n" + "=" * 70)
    print("Step 7: Verify the effective model")
    print("=" * 70)

    metrics = optimizer.verify(verbose=True)

    # Step 8: Plot results
    print("\n" + "=" * 70)
    print("Step 8: Generate plots")
    print("=" * 70)

    plot_comparison(optimizer)
    plot_overlaid_bands(optimizer)

    # Step 9: Validation
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)

    expected_f_indices = [6, 7]
    assert optimizer.f_indices == expected_f_indices
    print("  F-orbital detection: PASS")
    assert metrics['rmse'] < 1.0
    print("  RMSE < 1.0: PASS")
    assert 0.0 < J_eff < 2.0
    print("  J_eff in (0, 2): PASS")

    print("\n  All validation checks passed!")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\nEffective parameters:")
    print(f"  J_eff = {J_eff:.6f}")
    print(f"  S_eff = [{S_eff[0]:.6f}, {S_eff[1]:.6f}, {S_eff[2]:.6f}]")

    print(f"\nAccuracy metrics:")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  Correlation = {metrics['correlation']:.6f}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)

    return optimizer, metrics


if __name__ == "__main__":
    main()
