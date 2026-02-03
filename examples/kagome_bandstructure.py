#!/usr/bin/env python3
"""
Kagome Lattice Band Structure Example

The Kagome lattice is a triangular lattice with corner-sharing triangles.
It has 3 sites per unit cell and features a flat band due to destructive
interference on the triangle loops.

Lattice structure:
    - Triangular Bravais lattice with 3-site basis
    - Sites form corner-sharing triangles
    - Nearest-neighbor hopping t = -1 (convention)

Reference:
    - Flat band in Kagome lattice: D. L. Bergman et al., Phys. Rev. B 76, 094417 (2007)
"""

# example_utils handles path setup automatically
from example_utils import (
    get_example_device,
    build_kagome_lattice,
    build_kagome_model,
    setup_example_figure,
    save_example_figure,
)

import torch
import matplotlib.pyplot as plt

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import generate_k_path, generate_kmesh
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure, DOSCalculator
from condmatTensor.manybody import SpectralFunction


def build_kagome_hamiltonian_k_space(lattice, k_path: torch.Tensor, t: float = -1.0) -> torch.Tensor:
    """Build k-space Hamiltonian for Kagome lattice (analytic).

    The Kagome lattice Hamiltonian has the form:
    H(k) = -t * [[0, 1 + e^{-ik·a1}, 1 + e^{-ik·a2}],
                 [1 + e^{ik·a1}, 0, 1 + e^{ik·(a1-a2)}],
                 [1 + e^{ik·a2}, 1 + e^{ik·(a2-a1)}, 0]]

    This gives:
    - Flat band at E = -2t
    - Dirac points at K where bands touch (E = t)

    Args:
        lattice: BravaisLattice object
        k_path: K-points in fractional coordinates, shape (N_k, 2)
        t: Hopping parameter (positive t gives flat band at -2t)

    Returns:
        Hamiltonian in k-space, shape (N_k, 3, 3)
    """
    a1 = lattice.cell_vectors[0]
    a2 = lattice.cell_vectors[1]

    N_k = len(k_path)
    Hk = torch.zeros((N_k, 3, 3), dtype=torch.complex128)

    for i, k_frac in enumerate(k_path):
        k_cart = k_frac @ lattice.reciprocal_vectors().T

        # Phase factors for nearest-neighbor hopping
        g1 = torch.exp(1j * torch.dot(k_cart, a1))
        g2 = torch.exp(1j * torch.dot(k_cart, a2))

        # Build Hamiltonian (nearest-neighbor only)
        H = torch.zeros((3, 3), dtype=torch.complex128)

        H[0, 1] = 1 + torch.conj(g1)
        H[0, 2] = 1 + torch.conj(g2)
        H[1, 0] = 1 + g1
        H[1, 2] = 1 + g1 * torch.conj(g2)
        H[2, 0] = 1 + g2
        H[2, 1] = 1 + torch.conj(g1) * g2

        Hk[i] = -t * H

    return Hk


def main():
    """Main function: compare two methods for Kagome lattice band structure."""
    print("=" * 70)
    print("Kagome Lattice Band Structure: Method Comparison")
    print("=" * 70)

    # Device selection
    device = get_example_device("for diagonalization")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Parameters
    t = -1.0
    n_per_segment = 100

    # Build lattice using utility function
    print("\n1. Building Kagome lattice...")
    lattice = build_kagome_lattice(t)
    print(f"   Lattice: {lattice}")
    print(f"   Cell vectors:\n   {lattice.cell_vectors}")

    # Generate k-path along high-symmetry lines: Γ-K-M-Γ
    print("\n2. Generating k-path (Γ-K-M-Γ)...")
    k_path, ticks = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment)
    print(f"   Number of k-points: {len(k_path)}")

    # ============================================================
    # METHOD 1: Direct k-space construction (analytic formula)
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 1: Direct k-space construction")
    print("=" * 70)

    print("\n3a. Building k-space Hamiltonian directly...")
    Hk_direct = build_kagome_hamiltonian_k_space(lattice, k_path, t=t)
    print(f"   Hamiltonian: {Hk_direct.shape}")

    print("\n4a. Diagonalizing...")
    eigenvalues_direct, _ = diagonalize(Hk_direct, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_direct.shape}")

    print("\n   Band ranges (Method 1):")
    for i in range(3):
        emin = eigenvalues_direct[:, i].min().item()
        emax = eigenvalues_direct[:, i].max().item()
        print(f"   Band {i}: [{emin:.4f}, {emax:.4f}]")

    # ============================================================
    # METHOD 2: TightBindingModel with k-space build_Hk
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 2: TightBindingModel.build_Hk (same hoppings as Method 1)")
    print("=" * 70)

    print("\n3b. Creating HoppingModel with Kagome hoppings...")
    tb_model = build_kagome_model(lattice, t)
    print(f"   Number of hopping terms: {len(tb_model.hoppings)}")
    print(f"   Orbital labels: {tb_model.orbital_labels}")

    print("\n4b. Building H(k) directly from hopping terms...")
    Hk_tb = tb_model.build_Hk(k_path)
    print(f"   H(k): {Hk_tb}")
    print(f"   Labels: {Hk_tb.labels}")

    print("\n5b. Diagonalizing...")
    eigenvalues_tb, _ = diagonalize(Hk_tb.tensor, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_tb.shape}")

    print("\n   Band ranges (Method 2):")
    for i in range(3):
        emin = eigenvalues_tb[:, i].min().item()
        emax = eigenvalues_tb[:, i].max().item()
        print(f"   Band {i}: [{emin:.4f}, {emax:.4f}]")

    # ============================================================
    # METHOD 3: Real-space H(R) + Fourier transform
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 3: H(R) -> Fourier transform (same hoppings as Method 2)")
    print("=" * 70)

    print("\n3c. Building H(R) from hopping terms...")
    H_R = tb_model.build_HR()
    print(f"   H(R): {H_R}")
    print(f"   Labels: {H_R.labels}")
    print(f"   Orbital names: {H_R.orbital_names}")
    print(f"   Number of R-vectors: {H_R.shape[0]}")

    print("\n4c. Fourier transform H(R) -> H(k)...")
    k_cart = k_path @ lattice.reciprocal_vectors().T
    Hk_fourier = H_R.to_k_space(k_cart)
    print(f"   H(k): {Hk_fourier}")
    print(f"   Labels: {Hk_fourier.labels}")

    print("\n5c. Diagonalizing...")
    eigenvalues_fourier, _ = diagonalize(Hk_fourier.tensor, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_fourier.shape}")

    print("\n   Band ranges (Method 3):")
    for i in range(3):
        emin = eigenvalues_fourier[:, i].min().item()
        emax = eigenvalues_fourier[:, i].max().item()
        print(f"   Band {i}: [{emin:.4f}, {emax:.4f}]")

    # ============================================================
    # COMPARISON: Check that all methods agree
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Method Agreement")
    print("=" * 70)

    diff_direct_tb = torch.max(torch.abs(eigenvalues_direct - eigenvalues_tb)).item()
    diff_direct_fourier = torch.max(torch.abs(eigenvalues_direct - eigenvalues_fourier)).item()
    diff_tb_fourier = torch.max(torch.abs(eigenvalues_tb - eigenvalues_fourier)).item()

    print(f"\n   Max difference (Direct vs HoppingModel):  {diff_direct_tb:.2e}")
    print(f"   Max difference (Direct vs Fourier):            {diff_direct_fourier:.2e}")
    print(f"   Max difference (HoppingModel vs Fourier): {diff_tb_fourier:.2e}")

    if diff_direct_tb < 1e-6 and diff_direct_fourier < 1e-6:
        print("\n   ✓ All methods agree! (difference < 1e-6)")
    else:
        print("\n   ✗ Methods disagree! Check implementation.")

    # ============================================================
    # PLOTTING
    # ============================================================
    print("\n" + "=" * 70)
    print("Creating comparison plot...")
    print("=" * 70)

    fig, axes = setup_example_figure('comparison_3')

    bs1 = BandStructure()
    bs1.compute(eigenvalues_direct, k_path, ticks)
    bs1.plot(ax=axes[0], ylabel="Energy ($|t|$)", title="Method 1: Direct k-space")

    bs2 = BandStructure()
    bs2.compute(eigenvalues_tb, k_path, ticks)
    bs2.plot(ax=axes[1], ylabel="Energy ($|t|$)", title="Method 2: HoppingModel.build_Hk")

    bs3 = BandStructure()
    bs3.compute(eigenvalues_fourier, k_path, ticks)
    bs3.plot(ax=axes[2], ylabel="Energy ($|t|$)", title="Method 3: H(R) → Fourier")

    for ax in axes:
        ax.text(0.05, 0.95, f"Flat band at -2",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_example_figure(fig, "kagome_bandstructure_comparison.png")

    # ============================================================
    # DOS CALCULATION: Using k-mesh
    # ============================================================
    print("\n" + "=" * 70)
    print("DOS Calculation: Using uniform k-mesh")
    print("=" * 70)

    print("\n6a. Generating uniform k-mesh for DOS...")
    k_mesh = generate_kmesh(lattice, nk=50)
    print(f"   Number of k-points: {len(k_mesh)}")

    print("\n7a. Building H(k) on k-mesh...")
    Hk_mesh = tb_model.build_Hk(k_mesh)
    print(f"   H(k) mesh: {Hk_mesh.tensor.shape}")

    print("\n8a. Diagonalizing on k-mesh...")
    eigenvalues_mesh, _ = diagonalize(Hk_mesh.tensor, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_mesh.shape}")

    print("\n9a. Computing DOS with Lorentzian broadening...")
    omega = torch.linspace(-4, 4, 500, dtype=torch.float64)
    dos = DOSCalculator()
    omega_vals, rho_vals = dos.from_eigenvalues(eigenvalues_mesh, omega, eta=0.05)
    print(f"   DOS grid: {omega_vals.shape}")
    print(f"   DOS range: [{rho_vals.min():.4f}, {rho_vals.max():.4f}]")

    print("\n10a. Creating DOS plot...")
    fig, ax = setup_example_figure('single')
    dos.plot(ax=ax, title="Kagome Lattice DOS (50x50 k-mesh)")
    ax.axvline(x=-2, color='red', linestyle='--', alpha=0.7, label='Flat band (E = -2)')
    ax.legend(fontsize=10)
    save_example_figure(fig, "kagome_dos.png")

    # ============================================================
    # Bare Spectral Function A(ω) from Eigenvalues
    # ============================================================
    print("\n" + "=" * 70)
    print("Bare Spectral Function A(ω) (Non-Interacting)")
    print("=" * 70)

    print("\n11. Computing bare spectral function A(ω) from eigenvalues...")
    spec_func = SpectralFunction()
    omega_spec, A = spec_func.from_eigenvalues(eigenvalues_mesh, omega, eta=0.05)
    print(f"   A(ω) shape: {A.shape}")
    print(f"   A(ω) range: [{A.min():.4f}, {A.max():.4f}]")

    dos_from_A = spec_func.compute_dos(A)
    print(f"   DOS from A(ω) shape: {dos_from_A.shape}")
    print(f"   DOS from A(ω) range: [{dos_from_A.min():.4f}, {dos_from_A.max():.4f}]")

    diff_dos = torch.max(torch.abs(dos_from_A - rho_vals)).item()
    print(f"\n   Max difference between DOS from A(ω) and DOS from eigenvalues: {diff_dos:.2e}")

    if diff_dos < 1e-10:
        print("   ✓ A(ω) matches DOS exactly for non-interacting case!")

    print("\n12. Creating A(ω) plot (orbital-resolved)...")
    fig, axes = setup_example_figure('comparison_3')

    ax1 = axes[0]
    colors = plt.cm.tab10(torch.linspace(0, 1, 3))
    for i in range(3):
        ax1.plot(omega.cpu().numpy(), A[:, i].cpu().numpy(),
                label=f"Orbital {i}", color=colors[i])
    ax1.set_xlabel(r"Energy $\omega$ ($|t|$)", fontsize=12)
    ax1.set_ylabel(r"Spectral Function $A(\omega)$", fontsize=12)
    ax1.set_title("Orbital-Resolved A(ω)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(omega.cpu().numpy(), dos_from_A.cpu().numpy(),
             color='black', linewidth=1.5)
    ax2.fill_between(omega.cpu().numpy(), 0, dos_from_A.cpu().numpy(),
                     alpha=0.3, color='gray')
    ax2.set_xlabel(r"Energy $\omega$ ($|t|$)", fontsize=12)
    ax2.set_ylabel(r"Spectral Function $A(\omega)$", fontsize=12)
    ax2.set_title("Total A(ω) (Sum over Orbitals)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(omega.cpu().numpy(), dos_from_A.cpu().numpy(),
             label='A(ω) total', color='blue', linewidth=1.5)
    ax3.plot(omega.cpu().numpy(), rho_vals.cpu().numpy(),
             label='DOS from eigenvalues', color='red',
             linewidth=1.5, linestyle='--')
    ax3.set_xlabel(r"Energy $\omega$ ($|t|$)", fontsize=12)
    ax3.set_ylabel(r"Density of States", fontsize=12)
    ax3.set_title("A(ω) vs DOS Comparison", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    save_example_figure(fig, "kagome_spectral_function.png")

    # ============================================================
    # Band Structure + DOS Combined Plot
    # ============================================================
    print("\n" + "=" * 70)
    print("Band Structure + DOS Combined Plot")
    print("=" * 70)

    print("\n10b. Creating combined plot (bands + DOS with shared energy axis)...")
    bs_combined = BandStructure()
    bs_combined.compute(eigenvalues_tb, k_path, ticks)

    ax_bands, ax_dos = bs_combined.plot_with_dos(
        eigenvalues_mesh=eigenvalues_mesh,
        omega=omega,
        eta=0.05,
        title="Kagome Lattice: Band Structure + DOS"
    )

    ax_bands.text(0.05, 0.95, f"Flat band at E = -2",
                  transform=ax_bands.transAxes, fontsize=9,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_dos.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    fig = ax_bands.figure
    save_example_figure(fig, "kagome_bandstructure_with_dos.png")

    print("\n" + "=" * 70)
    print("Done! All plots saved.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
