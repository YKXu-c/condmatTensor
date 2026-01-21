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

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import matplotlib.pyplot as plt

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, TightBindingModel, generate_k_path, generate_kmesh
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure, DOSCalculator


def build_kagome_lattice(t: float = -1.0) -> BravaisLattice:
    """
    Build Kagome lattice.

    The Kagome lattice has a triangular Bravais lattice with 3 sites per unit cell.
    Sites are at fractional positions:
        - r1 = (0, 0)
        - r2 = (1/2, 0)
        - r3 = (1/4, sqrt(3)/4)

    Args:
        t: Hopping parameter (default -1)

    Returns:
        BravaisLattice object
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

    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=1,
    )


def build_kagome_hamiltonian_real_space(lattice: BravaisLattice, t: float = -1.0) -> BaseTensor:
    """
    Build real-space tight-binding Hamiltonian for Kagome lattice.

    Nearest-neighbor connections:
        - Site 0 connects to Site 1 (same unit cell)
        - Site 0 connects to Site 2 (same unit cell)
        - Site 1 connects to Site 2 (same unit cell)
        - Plus periodic boundary connections to neighboring cells

    Args:
        lattice: BravaisLattice object
        t: Hopping parameter

    Returns:
        BaseTensor with real-space Hamiltonian, labels=['R', 'orb_i', 'orb_j']
    """
    import math

    # Define hopping terms: (orb_i, orb_j, displacement)
    # Displacement is in units of lattice vectors
    hoppings = []

    # Intra-cell hopping (R = 0)
    hoppings.append((0, 1, torch.tensor([0, 0])))  # Site 0 -> 1
    hoppings.append((1, 0, torch.tensor([0, 0])))  # Site 1 -> 0
    hoppings.append((0, 2, torch.tensor([0, 0])))  # Site 0 -> 2
    hoppings.append((2, 0, torch.tensor([0, 0])))  # Site 2 -> 0
    hoppings.append((1, 2, torch.tensor([0, 0])))  # Site 1 -> 2
    hoppings.append((2, 1, torch.tensor([0, 0])))  # Site 2 -> 1

    # Inter-cell hopping (periodic boundary)
    # Site 0 connects to Site 1 in cell (0, -1)
    hoppings.append((0, 1, torch.tensor([0, -1])))
    hoppings.append((1, 0, torch.tensor([0, 1])))

    # Site 1 connects to Site 2 in cell (-1, 0)
    hoppings.append((1, 2, torch.tensor([-1, 0])))
    hoppings.append((2, 1, torch.tensor([1, 0])))

    # Site 2 connects to Site 0 in cell (0, -1)
    hoppings.append((2, 0, torch.tensor([0, -1])))
    hoppings.append((0, 2, torch.tensor([0, 1])))

    # Get unique displacements
    displacements = torch.stack(list(set(tuple(hop[2].tolist()) for hop in hoppings)))
    displacements = torch.tensor(displacements)

    n_R = len(displacements)
    n_orb = lattice.total_orbitals  # 3 for Kagome

    # Build Hamiltonian tensor H[R_ij, orb_i, orb_j]
    H_R = torch.zeros((n_R, n_orb, n_orb), dtype=torch.complex128)

    for orb_i, orb_j, disp in hoppings:
        # Find index of this displacement
        for idx, d in enumerate(displacements):
            if torch.allclose(d, disp):
                R_idx = idx
                break
        H_R[R_idx, orb_i, orb_j] = t

    # Convert to lattice vector coordinates
    disp_cart = displacements @ lattice.cell_vectors

    return BaseTensor(
        tensor=H_R,
        labels=["R", "orb_i", "orb_j"],
        orbital_names=None,
        displacements=disp_cart,
    )


def build_kagome_hamiltonian_k_space(lattice: BravaisLattice, k_path: torch.Tensor, t: float = -1.0) -> torch.Tensor:
    """
    Build k-space Hamiltonian for Kagome lattice (analytic).

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
        # Convert to Cartesian k-space coordinates
        k_cart = k_frac @ lattice.reciprocal_vectors().T

        # Compute phase factors for nearest-neighbor hopping
        # The basis vectors in the triangular lattice are not orthogonal
        g1 = torch.exp(1j * torch.dot(k_cart, a1))
        g2 = torch.exp(1j * torch.dot(k_cart, a2))

        # Build Hamiltonian (nearest-neighbor only)
        # H_ij = -t * Σ_<ij> exp(ik·δ_ij)
        H = torch.zeros((3, 3), dtype=torch.complex128)

        # Site 0 (origin) connects to:
        H[0, 1] = 1 + torch.conj(g1)  # Site 1 in same cell and cell (-1, 0)
        H[0, 2] = 1 + torch.conj(g2)  # Site 2 in same cell and cell (0, -1)

        # Site 1 connects to:
        H[1, 0] = 1 + g1  # Hermitian conjugate
        H[1, 2] = 1 + g1 * torch.conj(g2)  # Site 2 in same cell and cell (-1, 1)

        # Site 2 connects to:
        H[2, 0] = 1 + g2  # Hermitian conjugate
        H[2, 1] = 1 + torch.conj(g1) * g2  # Hermitian conjugate

        Hk[i] = -t * H

    return Hk


def main():
    """Main function: compare two methods for Kagome lattice band structure."""
    print("=" * 70)
    print("Kagome Lattice Band Structure: Method Comparison")
    print("=" * 70)

    # Parameters
    t = -1.0  # Hopping parameter (effective hopping = -t)
    # Note: The analytic formula H = -t * M means:
    #   - If we want effective hopping -1 (flat band at -2), we set t = 1
    #   - If we want effective hopping +1 (flat band at +2), we set t = -1
    # Here we use t = -1 for consistency with the Kagome lattice standard result
    n_per_segment = 100

    # Build lattice
    print("\n1. Building Kagome lattice...")
    lattice = build_kagome_lattice()
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
    #    Using SAME hopping definition as Method 1
    # ============================================================
    print("\n" + "=" * 70)
    print("METHOD 2: TightBindingModel.build_Hk (same hoppings as Method 1)")
    print("=" * 70)

    print("\n3b. Creating TightBindingModel with Kagome hoppings...")
    # Use the analytic hopping structure directly in k-space form
    tb_model = TightBindingModel(lattice, orbital_labels=["A", "B", "C"])

    # Use add_hermitian=False to match analytic structure exactly
    # IMPORTANT: Use hopping value = -t to match the analytic formula H = -t * M
    hop_val = -t  # This is the actual hopping amplitude in the Hamiltonian

    # A -> B: 1 + exp(-ik·a1)
    tb_model.add_hopping("A", "B", [0, 0], hop_val, add_hermitian=False)      # intra: 1
    tb_model.add_hopping("B", "A", [0, 0], hop_val, add_hermitian=False)      # intra: 1
    tb_model.add_hopping("A", "B", [-1, 0], hop_val, add_hermitian=False)     # inter: exp(-ik·a1)
    tb_model.add_hopping("B", "A", [1, 0], hop_val, add_hermitian=False)      # inter: exp(+ik·a1)

    # A -> C: 1 + exp(-ik·a2)
    tb_model.add_hopping("A", "C", [0, 0], hop_val, add_hermitian=False)      # intra: 1
    tb_model.add_hopping("C", "A", [0, 0], hop_val, add_hermitian=False)      # intra: 1
    tb_model.add_hopping("A", "C", [0, -1], hop_val, add_hermitian=False)     # inter: exp(-ik·a2)
    tb_model.add_hopping("C", "A", [0, 1], hop_val, add_hermitian=False)      # inter: exp(+ik·a2)

    # B -> C: 1 + exp(ik·(a1-a2))
    tb_model.add_hopping("B", "C", [0, 0], hop_val, add_hermitian=False)      # intra: 1
    tb_model.add_hopping("C", "B", [0, 0], hop_val, add_hermitian=False)      # intra: 1
    tb_model.add_hopping("B", "C", [1, -1], hop_val, add_hermitian=False)     # inter: exp(ik·(a1-a2))
    tb_model.add_hopping("C", "B", [-1, 1], hop_val, add_hermitian=False)     # inter: exp(-ik·(a1-a2))

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
    #    Using SAME hopping model
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
    # Convert k-path to Cartesian for Fourier transform
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

    print(f"\n   Max difference (Direct vs TightBindingModel):  {diff_direct_tb:.2e}")
    print(f"   Max difference (Direct vs Fourier):            {diff_direct_fourier:.2e}")
    print(f"   Max difference (TightBindingModel vs Fourier): {diff_tb_fourier:.2e}")

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Method 1: Direct k-space
    bs1 = BandStructure()
    bs1.compute(eigenvalues_direct, k_path, ticks)
    bs1.plot(ax=axes[0], ylabel="Energy ($|t|$)", title="Method 1: Direct k-space")

    # Method 2: TightBindingModel
    bs2 = BandStructure()
    bs2.compute(eigenvalues_tb, k_path, ticks)
    bs2.plot(ax=axes[1], ylabel="Energy ($|t|$)", title="Method 2: TightBindingModel.build_Hk")

    # Method 3: Fourier from H(R)
    bs3 = BandStructure()
    bs3.compute(eigenvalues_fourier, k_path, ticks)
    bs3.plot(ax=axes[2], ylabel="Energy ($|t|$)", title="Method 3: H(R) → Fourier")

    # Add flat band annotation
    for ax in axes:
        ax.text(0.05, 0.95, f"Flat band at -2",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("kagome_bandstructure_comparison.png", dpi=150)
    print(f"   Saved: kagome_bandstructure_comparison.png")

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
    fig, ax = plt.subplots(figsize=(7, 5))
    dos.plot(ax=ax, title="Kagome Lattice DOS (50x50 k-mesh)")

    # Add flat band annotation
    ax.axvline(x=-2, color='red', linestyle='--', alpha=0.7, label='Flat band (E = -2)')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("kagome_dos.png", dpi=150)
    print(f"   Saved: kagome_dos.png")

    print("\n" + "=" * 70)
    print("Done! All plots saved.")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
