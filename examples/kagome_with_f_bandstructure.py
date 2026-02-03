#!/usr/bin/env python3
"""
Kagome Lattice with Central f-Orbital (Kagome-F) Band Structure Example

The Kagome-F lattice is a Kagome lattice with an additional atom at the center
of each triangle. This creates a 4-site per unit cell system.

Lattice structure:
    - Triangular Bravais lattice with 4-site basis
    - 3 Kagome sites forming corner-sharing triangles
    - 1 central f-orbital site at the center of each triangle
    - Nearest-neighbor hopping t between Kagome sites
    - Hopping t_f between Kagome sites and central f-orbital
    - Optional on-site energy ε_f for the f-orbital

Physical relevance:
    - Model for heavy-fermion materials with localized f-electrons
    - Kagome metals with additional orbital degrees of freedom
    - Topological flat bands with orbital hybridization

Reference:
    - Kagome lattice with f-electron models for heavy fermion systems
    - Flat band physics in multi-orbital Kagome systems
"""

# example_utils handles path setup automatically
from example_utils import (
    get_example_device,
    build_kagome_lattice,
    build_kagome_model,
    build_kagome_f_lattice,
    build_kagome_f_model,
    setup_example_figure,
    save_example_figure,
)

import torch
import matplotlib.pyplot as plt

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path, generate_kmesh
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure, DOSCalculator
from condmatTensor.manybody import SpectralFunction


def build_kagome_f_hamiltonian_k_space(
    lattice: BravaisLattice,
    k_path: torch.Tensor,
    t: float = -1.0,
    t_f: float = -0.5,
    epsilon_f: float = 0.0,
) -> torch.Tensor:
    """Build k-space Hamiltonian for Kagome-F lattice.

    The Hamiltonian has the form:
    H(k) = [[H_Kagome(k),     H_Kf(k)   ],
            [H_Kf(k)^†,       epsilon_f ]]

    Args:
        lattice: BravaisLattice object
        k_path: K-points in fractional coordinates
        t: Hopping parameter between Kagome sites
        t_f: Hopping parameter between Kagome sites and f-orbital
        epsilon_f: On-site energy for f-orbital

    Returns:
        Hamiltonian in k-space, shape (N_k, 4, 4)
    """
    a1 = lattice.cell_vectors[0]
    a2 = lattice.cell_vectors[1]

    N_k = len(k_path)
    Hk = torch.zeros((N_k, 4, 4), dtype=torch.complex128)

    for i, k_frac in enumerate(k_path):
        k_cart = k_frac @ lattice.reciprocal_vectors().T

        # Phase factors for nearest-neighbor hopping
        g1 = torch.exp(1j * torch.dot(k_cart, a1))
        g2 = torch.exp(1j * torch.dot(k_cart, a2))

        # Build 4x4 Hamiltonian
        H = torch.zeros((4, 4), dtype=torch.complex128)

        # Kagome-Kagome hopping (same as pure Kagome)
        H[0, 1] = 1 + torch.conj(g1)
        H[0, 2] = 1 + torch.conj(g2)
        H[1, 0] = 1 + g1
        H[1, 2] = 1 + g1 * torch.conj(g2)
        H[2, 0] = 1 + g2
        H[2, 1] = 1 + torch.conj(g1) * g2

        # Kagome-f-orbital hopping
        H[0, 3] = t_f / t
        H[1, 3] = t_f / t
        H[2, 3] = t_f / t
        H[3, 0] = torch.conj(H[0, 3])
        H[3, 1] = torch.conj(H[1, 3])
        H[3, 2] = torch.conj(H[2, 3])
        H[3, 3] = epsilon_f / t if t != 0 else 0

        Hk[i] = -t * H

    return Hk


def main():
    """Main function: Kagome-F lattice band structure with parameter scan."""
    print("=" * 70)
    print("Kagome-F Lattice Band Structure (Kagome with Central f-Orbital)")
    print("=" * 70)

    # Parameters
    t = -1.0
    t_f = -0.5
    epsilon_f = 0.0
    n_per_segment = 100

    # Build lattice using utility function
    print("\n1. Building Kagome-F lattice...")
    lattice = build_kagome_f_lattice(t)
    print(f"   Lattice: {lattice}")
    print(f"   Total orbitals per unit cell: {lattice.total_orbitals}")
    print(f"   Basis positions:")
    for i, pos in enumerate(lattice.basis_positions):
        print(f"      Site {i}: {pos.tolist()}")

    # Generate k-path along high-symmetry lines: Γ-K-M-Γ
    print("\n2. Generating k-path (Γ-K-M-Γ)...")
    k_path, ticks = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment)
    print(f"   Number of k-points: {len(k_path)}")

    # ============================================================
    # METHOD 1: Direct k-space construction
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 1: Direct k-space construction")
    print("=" * 70)

    print("\n3a. Building k-space Hamiltonian directly...")
    print(f"   Parameters: t={t}, t_f={t_f}, ε_f={epsilon_f}")
    Hk_direct = build_kagome_f_hamiltonian_k_space(lattice, k_path, t=t, t_f=t_f, epsilon_f=epsilon_f)
    print(f"   Hamiltonian: {Hk_direct.shape}")

    print("\n4a. Diagonalizing...")
    eigenvalues_direct, _ = diagonalize(Hk_direct, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_direct.shape}")

    print("\n   Band ranges (Method 1):")
    for i in range(4):
        emin = eigenvalues_direct[:, i].min().item()
        emax = eigenvalues_direct[:, i].max().item()
        print(f"   Band {i}: [{emin:.4f}, {emax:.4f}]")

    # ============================================================
    # METHOD 2: TightBindingModel with build_Hk
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 2: TightBindingModel.build_Hk")
    print("=" * 70)

    print("\n3b. Creating HoppingModel...")
    tb_model = build_kagome_f_model(lattice, t=t, tf=t_f, fd_hybridization=t_f)
    # Add f-orbital on-site energy
    tb_model.add_hopping("F", "F", [0, 0], epsilon_f, add_hermitian=False)
    print(f"   Number of hopping terms: {len(tb_model.hoppings)}")
    print(f"   Orbital labels: {tb_model.orbital_labels}")

    print("\n4b. Building H(k) from hopping terms...")
    Hk_tb = tb_model.build_Hk(k_path)
    print(f"   H(k): {Hk_tb}")

    print("\n5b. Diagonalizing...")
    eigenvalues_tb, _ = diagonalize(Hk_tb.tensor, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_tb.shape}")

    print("\n   Band ranges (Method 2):")
    for i in range(4):
        emin = eigenvalues_tb[:, i].min().item()
        emax = eigenvalues_tb[:, i].max().item()
        print(f"   Band {i}: [{emin:.4f}, {emax:.4f}]")

    # ============================================================
    # METHOD 3: H(R) -> Fourier transform
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 3: H(R) -> Fourier transform")
    print("=" * 70)

    print("\n3c. Building H(R) from hopping terms...")
    H_R = tb_model.build_HR()
    print(f"   H(R): {H_R}")
    print(f"   Number of R-vectors: {H_R.shape[0]}")

    print("\n4c. Fourier transform H(R) -> H(k)...")
    k_cart = k_path @ lattice.reciprocal_vectors().T
    Hk_fourier = H_R.to_k_space(k_cart)
    print(f"   H(k): {Hk_fourier}")

    print("\n5c. Diagonalizing...")
    eigenvalues_fourier, _ = diagonalize(Hk_fourier.tensor, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_fourier.shape}")

    print("\n   Band ranges (Method 3):")
    for i in range(4):
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
    # PLOTTING: Three panel comparison
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
        ax.text(0.05, 0.95, f"t={t}, t_f={t_f}\nε_f={epsilon_f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_example_figure(fig, "kagome_with_f_bandstructure_comparison.png")

    # ============================================================
    # PARAMETER SCAN: Effect of f-orbital coupling
    # ============================================================
    print("\n" + "=" * 70)
    print("PARAMETER SCAN: Effect of f-orbital coupling (t_f)")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    t_f_values = [0.0, -0.3, -0.5, -1.0]

    for idx, t_f_val in enumerate(t_f_values):
        print(f"\n   Computing for t_f = {t_f_val}...")
        Hk_scan = build_kagome_f_hamiltonian_k_space(lattice, k_path, t=t, t_f=t_f_val, epsilon_f=epsilon_f)
        eig_scan, _ = diagonalize(Hk_scan, hermitian=True)

        bs = BandStructure()
        bs.compute(eig_scan, k_path, ticks)
        bs.plot(ax=axes[idx], ylabel="Energy ($|t|$)", title=f"t_f = {t_f_val}")

        print(f"   Band ranges for t_f = {t_f_val}:")
        for i in range(4):
            emin = eig_scan[:, i].min().item()
            emax = eig_scan[:, i].max().item()
            print(f"      Band {i}: [{emin:.4f}, {emax:.4f}]")

        bandwidth = (eig_scan[:, :].max(dim=0)[0] - eig_scan[:, :].min(dim=0)[0])
        flat_band_idx = bandwidth.argmin().item()
        print(f"      Flattest band: {flat_band_idx} (width = {bandwidth[flat_band_idx]:.4f})")

    save_example_figure(fig, "kagome_with_f_tf_scan.png")

    # ============================================================
    # COMPARISON: Pure Kagome vs Kagome-F
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Pure Kagome vs Kagome-F")
    print("=" * 70)

    fig, axes = setup_example_figure('dual')

    # Pure Kagome (3 bands) - using utility function
    print("\n   Computing pure Kagome bands...")
    kagome_lattice = build_kagome_lattice(t)
    kagome_tb = build_kagome_model(kagome_lattice, t)
    Hk_kagome = kagome_tb.build_Hk(k_path)
    eig_kagome, _ = diagonalize(Hk_kagome.tensor, hermitian=True)

    bs_kagome = BandStructure()
    bs_kagome.compute(eig_kagome, k_path, ticks)
    bs_kagome.plot(ax=axes[0], ylabel="Energy ($|t|$)", title="Pure Kagome (3 bands)")

    # Kagome-F (4 bands)
    bs_f = BandStructure()
    bs_f.compute(eigenvalues_direct, k_path, ticks)
    bs_f.plot(ax=axes[1], ylabel="Energy ($|t|$)", title=f"Kagome-F (4 bands, t_f={t_f})")

    axes[0].text(0.05, 0.95, "Flat band at -2",
                transform=axes[0].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    axes[1].text(0.05, 0.95, f"4 bands\nf-orbital hybridized",
                transform=axes[1].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    save_example_figure(fig, "kagome_vs_kagome_f.png")

    # ============================================================
    # DOS CALCULATION: Using k-mesh
    # ============================================================
    print("\n" + "=" * 70)
    print("DOS Calculation: Using uniform k-mesh")
    print("=" * 70)

    print("\n6. Generating uniform k-mesh for DOS...")
    k_mesh = generate_kmesh(lattice, nk=50)
    print(f"   Number of k-points: {len(k_mesh)}")

    print("\n7. Building H(k) on k-mesh...")
    Hk_mesh = tb_model.build_Hk(k_mesh)
    print(f"   H(k) mesh: {Hk_mesh.tensor.shape}")

    print("\n8. Diagonalizing on k-mesh...")
    eigenvalues_mesh, _ = diagonalize(Hk_mesh.tensor, hermitian=True)
    print(f"   Eigenvalues: {eigenvalues_mesh.shape}")

    print("\n9. Computing DOS with Lorentzian broadening...")
    omega = torch.linspace(-4, 4, 500, dtype=torch.float64)
    dos = DOSCalculator()
    omega_vals, rho_vals = dos.from_eigenvalues(eigenvalues_mesh, omega, eta=0.05)
    print(f"   DOS grid: {omega_vals.shape}")
    print(f"   DOS range: [{rho_vals.min():.4f}, {rho_vals.max():.4f}]")

    print("\n10. Creating DOS plot...")
    fig, ax = setup_example_figure('single')
    dos.plot(ax=ax, title=f"Kagome-F Lattice DOS (t={t}, t_f={t_f})")
    ax.text(0.95, 0.95, f"t={t}\nt_f={t_f}\nε_f={epsilon_f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    save_example_figure(fig, "kagome_f_dos.png")

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

    print("\n12. Creating A(ω) plot (orbital-resolved for Kagome-F)...")
    fig, axes = setup_example_figure('comparison_3')

    ax1 = axes[0]
    colors = plt.cm.tab10(torch.linspace(0, 1, 4))
    orbital_labels = ["A", "B", "C", "f"]
    for i in range(4):
        ax1.plot(omega.cpu().numpy(), A[:, i].cpu().numpy(),
                label=f"Orbital {orbital_labels[i]}", color=colors[i])
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
    ax3.plot(omega.cpu().numpy(), A[:, 3].cpu().numpy(),
             label='f-orbital', color='red', linewidth=2)
    ax3.fill_between(omega.cpu().numpy(), 0, A[:, 3].cpu().numpy(),
                     alpha=0.3, color='red')
    kagome_A = torch.sum(A[:, :3], dim=1)
    ax3.plot(omega.cpu().numpy(), kagome_A.cpu().numpy(),
             label='Kagome sites (sum)', color='blue',
             linewidth=1.5, linestyle='--')
    ax3.set_xlabel(r"Energy $\omega$ ($|t|$)", fontsize=12)
    ax3.set_ylabel(r"Spectral Function $A(\omega)$", fontsize=12)
    ax3.set_title("f-Orbital vs Kagome Sites", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    save_example_figure(fig, "kagome_f_spectral_function.png")

    # ============================================================
    # Band Structure + DOS Combined Plot
    # ============================================================
    print("\n" + "=" * 70)
    print("Band Structure + DOS Combined Plot")
    print("=" * 70)

    print("\n11. Creating combined plot (bands + DOS with shared energy axis)...")
    bs_combined = BandStructure()
    bs_combined.compute(eigenvalues_tb, k_path, ticks)

    ax_bands, ax_dos = bs_combined.plot_with_dos(
        eigenvalues_mesh=eigenvalues_mesh,
        omega=omega,
        eta=0.05,
        title=f"Kagome-F Lattice: Band Structure + DOS (t={t}, t_f={t_f})"
    )

    ax_bands.text(0.05, 0.95, f"t={t}\nt_f={t_f}\nε_f={epsilon_f}",
                  transform=ax_bands.transAxes, fontsize=9,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_dos.text(0.95, 0.95, f"4 bands\nf-orbital\nhybridized",
                transform=ax_dos.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    fig = ax_bands.figure
    save_example_figure(fig, "kagome_with_f_bandstructure_with_dos.png")

    print("\n" + "=" * 70)
    print("Done! All plots saved.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
