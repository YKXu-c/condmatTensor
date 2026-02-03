#!/usr/bin/env python3
"""
DMFT for Kagome-F Lattice (Kagome with Central f-Orbital)

Demonstrates DMFT self-consistency for a multi-orbital heavy-fermion
system with localized f-electrons.

Lattice structure:
    - 4 sites per unit cell: 3 Kagome sites + 1 central f-orbital
    - f-orbital is localized (OrbitalMetadata.local=True)
    - Hubbard U applied only to f-orbital
    - Tests orbital-selective correlation effects

Physical relevance:
    - Heavy-fermion materials with Kondo effect
    - Orbital-selective Mott transitions
    - f-d hybridization in Kagome systems

DMFT Implementation:
    - IPT solver (second-order perturbation theory)
    - Orbital-dependent U from OrbitalMetadata
    - TRIQS-style imaginary time approach: Σ(τ) = U²·G₀(τ)³

t_f Parameter Scan:
    - t_f controls f-f hopping and f-d hybridization strength
    - Scan range: -0.1 to -1.0 (10 values)
    - Each t_f value generates 11 plots in its own subdirectory

References:
    - "Dynamical mean-field theory" - Georges et al., Rev. Mod. Phys. 68, 13 (1996)
    - TRIQS 3.3.1 Tutorial: "A first DMFT calculation"
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# example_utils handles path setup automatically
from example_utils import (
    get_example_device,
    build_kagome_f_lattice,
    build_kagome_f_model,
    setup_example_figure,
    save_example_figure,
)

import torch
import matplotlib.pyplot as plt
import math

from condmatTensor.core import BaseTensor
from condmatTensor.core.types import OrbitalMetadata
from condmatTensor.lattice import generate_kmesh, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.manybody import (
    generate_matsubara_frequencies,
    calculate_dos_range,
    IPTSolver,
    SingleSiteDMFTLoop,
)
from condmatTensor.analysis import DOSCalculator, BandStructure


def run_dmft_for_U(U_f_value, device, lattice, tb_model, orbital_metadatas_template,
                    k_mesh, k_path, beta, n_max, mu, mixing, max_iter, tol):
    """Run DMFT calculation for a specific U_f value.

    Args:
        U_f_value: Hubbard U for f-orbital
        device: Computation device
        lattice: BravaisLattice object
        tb_model: HoppingModel object
        orbital_metadatas_template: List of OrbitalMetadata (will copy and update U_f)
        k_mesh: k-point mesh for DMFT
        k_path: k-point path for band structure
        beta: Inverse temperature
        n_max: Maximum Matsubara frequency
        mu: Chemical potential
        mixing: DMFT mixing parameter
        max_iter: Maximum DMFT iterations
        tol: Convergence tolerance

    Returns:
        dict: {
            'U_f': float,
            'Sigma': BaseTensor,
            'dmft': SingleSiteDMFTLoop,
            'converged': bool,
            'n_iterations': int,
            'evals_nonint': torch.Tensor,
            'evals_int': torch.Tensor,
            'omega_dos': torch.Tensor,
            'rho_nonint': torch.Tensor,
            'rho_int': torch.Tensor,
        }
    """
    from copy import deepcopy

    # Copy orbital metadata and update U_f
    orbital_metadatas = deepcopy(orbital_metadatas_template)
    orbital_metadatas[3] = OrbitalMetadata(
        site='F', orb='f', local=True, U=U_f_value
    )

    # Build H(k) with metadata on k_mesh
    Hk_mesh = tb_model.build_Hk(k_mesh)
    Hk_mesh.orbital_metadatas = orbital_metadatas

    # Generate Matsubara frequencies
    omega = generate_matsubara_frequencies(beta, n_max, fermionic=True, device=device)

    # Initialize and run DMFT
    solver = IPTSolver(beta=beta, n_max=n_max, device=device)
    dmft = SingleSiteDMFTLoop(Hk=Hk_mesh, omega=omega, solver=solver, mu=mu, mixing=mixing, verbose=False)
    Sigma = dmft.run(max_iter=max_iter, tol=tol)

    # Build band structure H(k) on path
    Hk_path = tb_model.build_Hk(k_path)
    Hk_path.orbital_metadatas = orbital_metadatas
    evals_nonint, _ = diagonalize(Hk_path.tensor)

    # Interacting bands with static Σ
    Sigma_static = Sigma.tensor[n_max].real
    Hk_eff_tensor = Hk_path.tensor.clone()
    for ik in range(len(k_path)):
        Hk_eff_tensor[ik] += Sigma_static
    evals_int, _ = diagonalize(Hk_eff_tensor)

    # DOS calculation
    evals_mesh, _ = diagonalize(Hk_mesh.tensor)
    evals_min = evals_mesh.min().item()
    evals_max = evals_mesh.max().item()
    sigma_shift = abs(Sigma.tensor[n_max].real).max().item()
    U_max = U_f_value

    omega_min, omega_max = calculate_dos_range(evals_min, evals_max, sigma_shift, U_max)
    omega_dos = torch.linspace(omega_min, omega_max, 1000, device=device)

    # Non-interacting DOS
    dos_calc = DOSCalculator()
    omega_dos, rho_nonint = dos_calc.from_eigenvalues(evals_mesh, omega_dos, eta=0.05)

    # Interacting DOS
    rho_int = torch.zeros(len(omega_dos), device=device)
    n_orb = Sigma.shape[-1]
    Hk_tensor = Hk_mesh.tensor
    for i, w in enumerate(omega_dos):
        z = w + 1j*0.05 + mu
        spectral_sum = 0.0
        for ik in range(len(k_mesh)):
            H_eff_k = Hk_tensor[ik] + Sigma_static
            G_k = torch.linalg.inv(z * torch.eye(n_orb, device=device) - H_eff_k)
            spectral_sum += -1.0/math.pi * torch.trace(G_k).imag
        rho_int[i] = spectral_sum / len(k_mesh)

    return {
        'U_f': U_f_value,
        'Sigma': Sigma,
        'dmft': dmft,
        'converged': dmft.n_iterations < max_iter,
        'n_iterations': dmft.n_iterations,
        'evals_nonint': evals_nonint.cpu(),
        'evals_int': evals_int.cpu(),
        'omega_dos': omega_dos.cpu(),
        'rho_nonint': rho_nonint.cpu(),
        'rho_int': rho_int.cpu(),
    }


def plot_band_structure_3x3_grid(results, k_path, ticks, fig_path):
    """Create 3x3 grid of band structures for U scan.

    Args:
        results: List of result dictionaries from run_dmft_for_U()
        k_path: k-point path (for ticks)
        ticks: High-symmetry point labels
        fig_path: Output filename
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        U_f = result['U_f']

        # Plot non-interacting bands (faint gray)
        for band in range(result['evals_nonint'].shape[1]):
            ax.plot(result['evals_nonint'][:, band].numpy(),
                   color='gray', alpha=0.3, linewidth=1)

        # Plot interacting bands (colored)
        for band in range(result['evals_int'].shape[1]):
            ax.plot(result['evals_int'][:, band].numpy(),
                   color='steelblue', alpha=0.7, linewidth=1.5)

        ax.set_title(f'U_f = {U_f:.1f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim(-5, 8)

    # Set ticks on bottom and left edges only
    for ax in axes:
        ax.tick_params(labelbottom=False, labelleft=False)
    for i in [6, 7, 8]:
        axes[i].tick_params(labelbottom=True)
    for i in [0, 3, 6]:
        axes[i].tick_params(labelleft=True)

    # High-symmetry points - calculate based on k_path length
    # G-K-M-G path: n_per_segment=100, so 3 segments = 300 points total
    # Indices: 0, 100, 200, 300
    n_k = len(k_path)
    tick_positions = [0, n_k // 3, 2 * n_k // 3, n_k - 1]
    tick_labels = ['G', 'K', 'M', 'G']

    for i in [6, 7, 8]:
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(tick_labels)

    fig.suptitle('Band Structure Evolution with U_f', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.02, 'k-path', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Energy (|t|)', va='center', rotation=90, fontsize=14)

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    save_example_figure(fig, fig_path)


def plot_dos_vertical_stack(results, fig_path):
    """Create vertical stack of DOS plots for U scan.

    Args:
        results: List of result dictionaries from run_dmft_for_U()
        fig_path: Output filename
    """
    n_U = len(results)
    fig, axes = plt.subplots(n_U, 1, figsize=(10, 2.5 * n_U), sharex=True)

    if n_U == 1:
        axes = [axes]

    for i, result in enumerate(results):
        ax = axes[i]
        U_f = result['U_f']
        omega = result['omega_dos'].numpy()
        rho_nonint = result['rho_nonint'].numpy()
        rho_int = result['rho_int'].numpy()

        # Plot non-interacting DOS (faint)
        ax.plot(omega, rho_nonint, color='gray', alpha=0.5,
               linewidth=1.5, label='Non-interacting')

        # Plot interacting DOS
        ax.plot(omega, rho_int, color='crimson', linewidth=2,
               label=f'Interacting (U_f={U_f:.1f})')

        ax.set_ylabel('DOS', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim(-0.5, max(rho_int.max().item(), 1.0) * 1.1)

    axes[-1].set_xlabel('Energy $\\omega$ ($|t|$)', fontsize=12)
    fig.suptitle('DOS Evolution with U_f', fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    save_example_figure(fig, fig_path)


def run_dmft_for_tf(tf_value, device, lattice, U_f, epsilon_f, U_d,
                     k_mesh, k_path, k_path_ticks, beta, n_max, mu, mixing, max_iter, tol):
    """Run full DMFT calculation for a specific t_f value.

    Unlike run_dmft_for_U() which only changes the metadata U parameter,
    this function rebuilds the HoppingModel with the new t_f value.

    Args:
        tf_value: f-f hopping and f-d hybridization parameter
        device: Computation device
        lattice: BravaisLattice object
        U_f: Hubbard U for f-orbital
        epsilon_f: f-orbital on-site energy
        U_d: Small U on Kagome d-orbitals
        k_mesh: k-point mesh for DMFT
        k_path: k-point path for band structure
        k_path_ticks: Tick positions and labels from generate_k_path()
        beta: Inverse temperature
        n_max: Maximum Matsubara frequency
        mu: Chemical potential
        mixing: DMFT mixing parameter
        max_iter: Maximum DMFT iterations
        tol: Convergence tolerance

    Returns:
        dict: All calculation results including Sigma, bands, DOS, etc.
    """
    from copy import deepcopy
    import math

    t = -1.0  # Kagome-Kagome hopping (fixed)

    # Rebuild HoppingModel with new t_f value
    tb_model = build_kagome_f_model(lattice, t, tf_value, tf_value)

    # Add f-orbital on-site energy
    tb_model.add_hopping("F", "F", [0, 0], epsilon_f, add_hermitian=False)

    # Create OrbitalMetadata for 4 orbitals
    orbital_metadatas = [
        OrbitalMetadata(site='K1', orb='dx2-y2', local=False, U=U_d),
        OrbitalMetadata(site='K2', orb='dxy', local=False, U=U_d),
        OrbitalMetadata(site='K3', orb='dxz', local=False, U=U_d),
        OrbitalMetadata(site='F', orb='f', local=True, U=U_f),
    ]

    # Build H(k) with metadata on k_mesh
    Hk_mesh = tb_model.build_Hk(k_mesh)
    Hk_mesh.orbital_metadatas = orbital_metadatas

    # Generate Matsubara frequencies
    omega = generate_matsubara_frequencies(beta, n_max, fermionic=True, device=device)

    # Initialize and run DMFT
    solver = IPTSolver(beta=beta, n_max=n_max, device=device)
    dmft = SingleSiteDMFTLoop(Hk=Hk_mesh, omega=omega, solver=solver, mu=mu, mixing=mixing, verbose=False)
    Sigma = dmft.run(max_iter=max_iter, tol=tol)

    # Build band structure H(k) on path
    Hk_path = tb_model.build_Hk(k_path)
    Hk_path.orbital_metadatas = orbital_metadatas
    evals_nonint, _ = diagonalize(Hk_path.tensor)

    # Interacting bands with static Σ
    Sigma_static = Sigma.tensor[n_max].real
    Hk_eff_tensor = Hk_path.tensor.clone()
    for ik in range(len(k_path)):
        Hk_eff_tensor[ik] += Sigma_static
    evals_int, _ = diagonalize(Hk_eff_tensor)

    # DOS calculation
    evals_mesh, _ = diagonalize(Hk_mesh.tensor)
    evals_min = evals_mesh.min().item()
    evals_max = evals_mesh.max().item()
    sigma_shift = abs(Sigma.tensor[n_max].real).max().item()

    omega_min, omega_max = calculate_dos_range(evals_min, evals_max, sigma_shift, U_f)
    omega_dos = torch.linspace(omega_min, omega_max, 1000, device=device)

    # Non-interacting DOS
    dos_calc = DOSCalculator()
    omega_dos, rho_nonint = dos_calc.from_eigenvalues(evals_mesh, omega_dos, eta=0.05)

    # Interacting DOS
    rho_int = torch.zeros(len(omega_dos), device=device)
    n_orb = Sigma.shape[-1]
    Hk_tensor = Hk_mesh.tensor
    for i, w in enumerate(omega_dos):
        z = w + 1j*0.05 + mu
        spectral_sum = 0.0
        for ik in range(len(k_mesh)):
            H_eff_k = Hk_tensor[ik] + Sigma_static
            G_k = torch.linalg.inv(z * torch.eye(n_orb, device=device) - H_eff_k)
            spectral_sum += -1.0/math.pi * torch.trace(G_k).imag
        rho_int[i] = spectral_sum / len(k_mesh)

    # Non-interacting Green's function for comparison
    omega_nonint = generate_matsubara_frequencies(beta, n_max, fermionic=True, device=device)
    G0_nonint = torch.zeros((len(omega_nonint), len(k_mesh), n_orb, n_orb),
                            dtype=torch.complex128, device=device)

    for i, iwn_val in enumerate(omega_nonint):
        inv_matrix = iwn_val - Hk_tensor
        G0_nonint[i] = torch.linalg.inv(inv_matrix)

    G0_loc_nonint = torch.mean(G0_nonint, dim=1)
    G_int = dmft.G_loc.tensor.clone()

    return {
        'tf': tf_value,
        'U_f': U_f,
        'Sigma': Sigma,
        'dmft': dmft,
        'converged': dmft.n_iterations < max_iter,
        'n_iterations': dmft.n_iterations,
        'evals_nonint': evals_nonint.cpu(),
        'evals_int': evals_int.cpu(),
        'omega_dos': omega_dos.cpu(),
        'rho_nonint': rho_nonint.cpu(),
        'rho_int': rho_int.cpu(),
        'G0_loc_nonint': G0_loc_nonint.cpu(),
        'G_int': G_int.cpu(),
        'orbital_metadatas': orbital_metadatas,
        'tb_model': tb_model,
        'Hk_mesh': Hk_mesh,
        'Hk_path': Hk_path,
        'evals_mesh': evals_mesh.cpu(),
        'omega': omega.cpu(),
        'k_path': k_path.cpu(),
        'ticks': k_path_ticks,
    }


def main():
    """Main function: DMFT for Kagome-F heavy-fermion model."""
    print("=" * 70)
    print("DMFT for Kagome-F Heavy-Fermion Model")
    print("=" * 70)

    # Device selection
    device = get_example_device("for DMFT calculation")

    # Parameters
    t = -1.0           # Kagome-Kagome hopping
    t_f = -0.3         # Kagome-f hopping (hybridization)
    epsilon_f = 0.0    # f-orbital on-site energy
    U_f = 4.0          # Hubbard U on f-orbital only (try U_f=2.0 for more moderate values)
    U_d = 0.5          # Small U on Kagome d-orbitals
    beta = 10.0        # Inverse temperature
    n_max = 64         # Matsubara frequencies
    nk = 16            # k-mesh size

    print(f"\nParameters:")
    print(f"  Hopping t = {t}")
    print(f"  f-hopping t_f = {t_f}")
    print(f"  f-orbital energy ε_f = {epsilon_f}")
    print(f"  Hubbard U (d-orbitals) = {U_d}")
    print(f"  Hubbard U (f-orbital) = {U_f}")
    print(f"  Inverse temperature β = {beta}")
    print(f"  k-mesh: {nk}×{nk} = {nk**2} points")

    # ============================================================
    # 1. Build Kagome-F lattice with OrbitalMetadata
    # ============================================================
    print("\n1. Building Kagome-F lattice with OrbitalMetadata...")
    lattice = build_kagome_f_lattice(t)
    tb_model = build_kagome_f_model(lattice, t, t_f, t_f)

    # Add f-orbital on-site energy
    tb_model.add_hopping("F", "F", [0, 0], epsilon_f, add_hermitian=False)

    print(f"   Lattice: {lattice}")
    print(f"   Number of sites: {lattice.num_sites}")
    print(f"   Total orbitals: {lattice.total_orbitals}")

    # Create OrbitalMetadata for 4 orbitals
    # Orbitals 0-2: Kagome d-orbitals (weakly correlated, U ≈ 0-1)
    # Orbital 3: f-orbital (strongly correlated, U ≈ 4-8)
    orbital_metadatas = [
        OrbitalMetadata(site='K1', orb='dx2-y2', local=False, U=U_d),
        OrbitalMetadata(site='K2', orb='dxy', local=False, U=U_d),
        OrbitalMetadata(site='K3', orb='dxz', local=False, U=U_d),
        OrbitalMetadata(site='F', orb='f', local=True, U=U_f),
    ]

    print(f"   Orbital metadata:")
    for i, md in enumerate(orbital_metadatas):
        print(f"      Orbital {i}: {md.to_string()}")

    # ============================================================
    # 2. Generate k-mesh and H(k)
    # ============================================================
    print("\n2. Generating k-mesh...")
    k_mesh = generate_kmesh(lattice, nk)
    print(f"   Number of k-points: {len(k_mesh)}")

    print("\n3. Building H(k) on k-mesh...")
    Hk_mesh = tb_model.build_Hk(k_mesh)

    # Attach orbital metadata to Hk
    Hk_mesh.orbital_metadatas = orbital_metadatas

    print(f"   H(k): {Hk_mesh.tensor.shape}")
    print(f"   Labels: {Hk_mesh.labels}")

    # ============================================================
    # 3. Setup Matsubara frequencies
    # ============================================================
    print("\n4. Generating Matsubara frequencies...")
    omega = generate_matsubara_frequencies(
        beta=beta, n_max=n_max, fermionic=True, device=device
    )
    print(f"   Frequency grid: {len(omega)} points")
    print(f"   Range: [{omega[0].item():.4f}i, {omega[-1].item():.4f}i]")

    # ============================================================
    # 4. Initialize IPT solver
    # ============================================================
    print("\n5. Initializing IPT solver...")

    # IPT solver reads U from OrbitalMetadata.U
    solver = IPTSolver(
        beta=beta,
        n_max=n_max,
        device=device,
    )
    print(f"   Solver: IPT (second-order perturbation theory)")
    print(f"   U values: read from OrbitalMetadata.U")
    print(f"   Expected: U_d = {U_d}, U_f = {U_f}")

    # ============================================================
    # 5. Run DMFT self-consistency loop
    # ============================================================
    print("\n6. Running DMFT self-consistency loop...")
    dmft = SingleSiteDMFTLoop(
        Hk=Hk_mesh,
        omega=omega,
        solver=solver,
        mu=0.0,
        mixing=0.5,
        verbose=True,
    )

    Sigma = dmft.run(max_iter=50, tol=1e-5)

    print(f"\n7. DMFT Results:")
    print(f"   Converged in {dmft.n_iterations} iterations")
    print(f"   Final Σ shape: {Sigma.shape}")

    # ============================================================
    # 6. Analyze orbital-selective correlation
    # ============================================================
    print("\n8. Analyzing orbital-selective correlation...")

    # Extract self-energy for each orbital (diagonal elements)
    n_orb = Sigma.shape[-1]
    Sigma_orb = []
    for i in range(n_orb):
        Sigma_orb.append(Sigma.tensor[:, i, i])

    # Zero-frequency self-energy (correlation strength indicator)
    # Use n_max index (closest to zero frequency)
    Sigma_zero_freq = [s[n_max].imag.item() for s in Sigma_orb]

    print(f"   Zero-frequency self-energy (Im Σ):")
    for i, s0 in enumerate(Sigma_zero_freq):
        print(f"      Orbital {i} ({orbital_metadatas[i].orb}): {s0:.4f}")

    # Check orbital selectivity
    d_avg = sum(Sigma_zero_freq[:3]) / 3
    f_val = Sigma_zero_freq[3]
    print(f"\n   Orbital selectivity check:")
    print(f"      Average Im Σ_d (d-orbitals): {d_avg:.4f}")
    print(f"      Im Σ_f (f-orbital): {f_val:.4f}")
    print(f"      Ratio |Σ_f|/|Σ_d|: {abs(f_val)/(abs(d_avg)+1e-10):.2f}")

    # ============================================================
    # 7. Plot convergence and self-energy
    # ============================================================
    print("\n9. Creating plots...")

    # Plot 1: Convergence history
    history = dmft.get_convergence_history()
    fig, ax = setup_example_figure('single')
    ax.semilogy(history["Sigma_diff"], 'o-', markersize=4, color='steelblue')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"|ΔΣ|/|Σ|")
    ax.set_title("DMFT Convergence History")
    ax.grid(True, alpha=0.3)
    save_example_figure(fig, "kagome_f_dmft_convergence.png")

    # Plot 2: Orbital-selective self-energy (imaginary part)
    fig, ax = setup_example_figure('single')
    # Note: iwn_vals is just for x-axis labeling (0, 1, 2, ..., 128)
    # The actual Matsubara frequencies are omega[iwn_vals] = iπ(2n+1)/β
    # The indexing scheme is symmetric: n goes from -n_max to +n_max
    iwn_vals = torch.arange(len(omega)).float()
    colors = plt.cm.tab10(torch.linspace(0, 1, n_orb))
    orbital_labels = ["K1($d_{x^2-y^2}$)", "K2($d_{xy}$)", "K3($d_{xz}$)", "F($f$)"]

    for i in range(n_orb):
        Sigma_im = Sigma_orb[i].imag.cpu().numpy()
        ax.plot(iwn_vals, Sigma_im, 'o-', markersize=3,
                label=f"Orbital {orbital_labels[i]}", color=colors[i])
    ax.set_xlabel(r"Matsubara Index $n$")
    ax.set_ylabel(r"Im $\Sigma(i\omega_n)$")
    ax.set_title("Orbital-Selective Self-Energy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save_example_figure(fig, "kagome_f_dmft_self_energy_orbital.png")

    # Plot 3: Orbital-selective self-energy (real part)
    fig, ax = setup_example_figure('single')
    for i in range(n_orb):
        Sigma_re = Sigma_orb[i].real.cpu().numpy()
        ax.plot(iwn_vals, Sigma_re, 's-', markersize=3,
                label=f"Orbital {orbital_labels[i]}", color=colors[i])
    ax.set_xlabel(r"Matsubara Index $n$")
    ax.set_ylabel(r"Re $\Sigma(i\omega_n)$")
    ax.set_title("Orbital-Selective Self-Energy (Real Part)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save_example_figure(fig, "kagome_f_dmft_self_energy_real.png")

    # Plot 4: Zero-frequency self-energy comparison (bar plot)
    fig, ax = setup_example_figure('single')
    x_pos = torch.arange(n_orb).float()
    bars = ax.bar(x_pos, Sigma_zero_freq, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Orb {i}\n{md.orb}" for i, md in enumerate(orbital_metadatas)],
                      rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r"Im $\Sigma(i\omega_0)$")
    ax.set_title("Orbital-Selective Self-Energy at Zero Frequency")
    ax.grid(True, alpha=0.3, axis='y')
    save_example_figure(fig, "kagome_f_dmft_orbital_selectivity.png")

    # ============================================================
    # 8. Non-interacting vs Interacting comparison
    # ============================================================
    print("\n10. Computing non-interacting reference...")

    # Compute non-interacting Green's function
    omega_nonint = generate_matsubara_frequencies(beta, n_max, fermionic=True, device=device)
    G0_nonint = torch.zeros((len(omega_nonint), nk*nk, n_orb, n_orb),
                            dtype=torch.complex128, device=device)

    # Hk_mesh.tensor has shape (N_k, n_orb, n_orb)
    Hk_tensor = Hk_mesh.tensor

    for i, iwn_val in enumerate(omega_nonint):
        inv_matrix = iwn_val - Hk_tensor  # (N_k, n_orb, n_orb)
        G0_nonint[i] = torch.linalg.inv(inv_matrix)

    # Local non-interacting Green's function
    G0_loc_nonint = torch.mean(G0_nonint, dim=1)  # (n_iwn, n_orb, n_orb)

    # FIX: Use the converged G_loc from DMFT loop (already has Σ embedded)
    # The DMFT loop computes G_loc = (1/N_k) Σ_k G(k,iωₙ) which already includes
    # the self-energy through the Dyson equation. Using this directly is more
    # accurate than recomputing from G0_loc_nonint.
    G_int = dmft.G_loc.tensor.clone()

    # Plot 5: Green's function comparison
    fig, axes = setup_example_figure('comparison_2')

    for i in range(n_orb):
        G_nonint_im = G0_loc_nonint[:, i, i].imag.cpu().numpy()
        axes[0].plot(iwn_vals, G_nonint_im, 'o-', markersize=3,
                     label=orbital_labels[i], color=colors[i])
    axes[0].set_xlabel(r"Matsubara Index $n$")
    axes[0].set_ylabel(r"Im $G_0(i\omega_n)$")
    axes[0].set_title("Non-Interacting Green's Function")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i in range(n_orb):
        G_int_im = G_int[:, i, i].imag.cpu().numpy()
        axes[1].plot(iwn_vals, G_int_im, 's-', markersize=3,
                     label=orbital_labels[i], color=colors[i])
    axes[1].set_xlabel(r"Matsubara Index $n$")
    axes[1].set_ylabel(r"Im $G(i\omega_n)$")
    axes[1].set_title("Interacting Green's Function")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    save_example_figure(fig, "kagome_f_dmft_greens_comparison.png")

    # Plot 6: Imaginary part of Green's function (TRIQS-style validation)
    # TRIQS tutorial uses Im[G(iωₙ)] plots to verify physical behavior
    fig, ax = setup_example_figure('single')
    for i in range(n_orb):
        G_im = G_int[:, i, i].imag.cpu().numpy()
        ax.plot(iwn_vals, G_im, 'o-', markersize=3,
                label=orbital_labels[i], color=colors[i])
    ax.set_xlabel(r"Matsubara Index $n$")
    ax.set_ylabel(r"Im $G(i\omega_n)$")
    ax.set_title("Imaginary Part of Interacting Green's Function")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_example_figure(fig, "kagome_f_dmft_greens_imag.png")

    # ============================================================
    # 9. Band structure with self-energy (static approximation)
    # ============================================================
    print("\n11. Computing interacting band structure...")

    # Generate k-path for band structure
    k_path, ticks = generate_k_path(lattice, ['G', 'K', 'M', 'G'], n_per_segment=100)
    Hk_path = tb_model.build_Hk(k_path)

    # Attach orbital metadata
    Hk_path.orbital_metadatas = orbital_metadatas

    # Non-interacting bands
    evals_nonint, _ = diagonalize(Hk_path.tensor)

    # Interacting bands with static Σ approximation
    # H_eff(k) = H(k) + Re[Σ(iω₀)]
    # Use n_max index (closest to zero frequency)
    Sigma_static = Sigma.tensor[n_max].real  # Shape: (n_orb, n_orb)

    Hk_eff_tensor = Hk_path.tensor.clone()
    for ik in range(len(k_path)):
        Hk_eff_tensor[ik] += Sigma_static

    evals_int, _ = diagonalize(Hk_eff_tensor)

    # Plot band structure comparison
    fig, axes = setup_example_figure('dual')
    bs_nonint = BandStructure()
    bs_nonint.compute(evals_nonint.cpu(), k_path.cpu(), ticks)
    bs_nonint.plot(ax=axes[0], title="Non-interacting Bands")

    bs_int = BandStructure()
    bs_int.compute(evals_int.cpu(), k_path.cpu(), ticks)
    bs_int.plot(ax=axes[1], title=f"Interacting Bands (U_f={U_f})")

    save_example_figure(fig, "kagome_f_dmft_bands_comparison.png")

    # ============================================================
    # 10. Spectral function and DOS
    # ============================================================
    print("\n12. Computing spectral function and DOS...")

    # Auto-calculate DOS range based on band structure and self-energy
    evals_mesh, _ = diagonalize(Hk_mesh.tensor)
    evals_min = evals_mesh.min().item()
    evals_max = evals_mesh.max().item()
    sigma_shift = abs(Sigma.tensor[n_max].real).max().item()
    U_max = max([md.U for md in orbital_metadatas if md.U is not None] + [0])

    omega_min, omega_max = calculate_dos_range(
        evals_min, evals_max, sigma_shift, U_max, margin=2.0
    )

    print(f"   Auto-calculated DOS range: [{omega_min:.2f}, {omega_max:.2f}]")

    # Analytic continuation: iωₙ → ω + iη
    eta = 0.05
    omega_real = torch.linspace(omega_min, omega_max, 1000, device=device)

    # Non-interacting DOS
    dos_calc = DOSCalculator()
    omega_dos, rho_nonint = dos_calc.from_eigenvalues(evals_mesh, omega_real, eta=eta)

    # Interacting DOS (simple approximation using static self-energy)
    # Use G(k, ω+iη) = [ω+iη+μ-H(k)-Σ]⁻¹, sum over k
    rho_int = torch.zeros(len(omega_real), device=device)
    n_k_total = len(k_mesh)

    for i, w in enumerate(omega_real):
        z = w + 1j*eta + dmft.mu
        # Use static self-energy approximation
        spectral_sum = 0.0

        for ik in range(n_k_total):
            # H_eff(k) = H(k) + Re[Σ(iω₀)]
            H_eff_k = Hk_tensor[ik] + Sigma_static
            # G(k, ω+iη) = [ω+iη - H_eff(k)]⁻¹
            G_k = torch.linalg.inv(z * torch.eye(n_orb, device=device) - H_eff_k)
            # Add spectral weight: -(1/π)Im[G]
            spectral_sum += -1.0/math.pi * torch.trace(G_k).imag

        rho_int[i] = spectral_sum / n_k_total

    # Plot DOS comparison
    fig, ax = setup_example_figure('single')
    ax.plot(omega_dos.cpu().numpy(), rho_nonint.cpu().numpy(),
            label='Non-interacting (U=0)', linewidth=2, color='steelblue')
    ax.plot(omega_real.cpu().numpy(), rho_int.cpu().numpy(),
            label=f'Interacting (U_f={U_f})', linewidth=2,
            color='crimson', linestyle='--')
    ax.set_xlabel(r"Energy $\omega$ ($|t|$)")
    ax.set_ylabel("DOS")
    ax.set_title("DOS: Non-interacting vs Interacting")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_example_figure(fig, "kagome_f_dmft_dos_comparison.png")

    # ============================================================
    # 11. Pade analytic continuation comparison
    # ============================================================
    print("\n13. Computing spectral function with Pade continuation...")

    from condmatTensor.manybody.preprocessing import SpectralFunction

    spectral = SpectralFunction()

    # Real frequency grid for Pade output (use same range as above)
    omega_pade = torch.linspace(omega_min, omega_max, 1000, device=device)
    eta_pade = 0.05  # Imaginary shift

    # Interacting DOS using Pade continuation from local Green's function
    # Create BaseTensor for local interacting Green's function
    from condmatTensor.core import BaseTensor
    G_int_base = BaseTensor(
        tensor=G_int,
        labels=['iwn', 'orb_i', 'orb_j'],
        orbital_names=Hk_mesh.orbital_names,
        orbital_metadatas=Hk_mesh.orbital_metadatas,
    )

    try:
        # Use Pade continuation for interacting case
        _, A_int_pade = spectral.from_matsubara(
            G_int_base, omega_pade, eta=eta_pade, method="pade",
            beta=beta, n_min=0, n_max=32
        )
        dos_int_pade = spectral.compute_dos(A_int_pade)

        # Plot Pade continuation comparison
        fig, ax = setup_example_figure('single')
        ax.plot(omega_dos.cpu().numpy(), rho_nonint.cpu().numpy(),
                label='Non-interacting', linewidth=2, color='steelblue')
        ax.plot(omega_real.cpu().numpy(), rho_int.cpu().numpy(),
                label=f'Interacting (Static Σ)', linewidth=2,
                color='crimson', linestyle='--')
        ax.plot(omega_pade.cpu().numpy(), dos_int_pade.cpu().numpy(),
                label='Interacting (Pade)', linewidth=2,
                color='darkgreen', linestyle='-.')
        ax.set_xlabel(r"Energy $\omega$ ($|t|$)")
        ax.set_ylabel("DOS")
        ax.set_title("DOS: Pade Analytic Continuation Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_example_figure(fig, "kagome_f_dmft_dos_pade.png")

        print("   Pade continuation successful!")
    except NotImplementedError as e:
        print(f"   Pade continuation not implemented: {e}")
    except Exception as e:
        print(f"   Pade continuation failed: {e}")

    # ============================================================
    # 14. U scanning: Band structures and DOS evolution
    # ============================================================
    print("\n14. Running U scan from 0 to 8...")

    U_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 9 values for 3x3 grid
    orbital_metadatas_template = [
        OrbitalMetadata(site='K1', orb='dx2-y2', local=False, U=U_d),
        OrbitalMetadata(site='K2', orb='dxy', local=False, U=U_d),
        OrbitalMetadata(site='K3', orb='dxz', local=False, U=U_d),
        OrbitalMetadata(site='F', orb='f', local=True, U=U_f),  # Will be overwritten
    ]

    results = []
    for U_val in U_values:
        print(f"\n  Running DMFT for U_f = {U_val:.1f}...")
        result = run_dmft_for_U(
            U_f_value=U_val,
            device=device,
            lattice=lattice,
            tb_model=tb_model,
            orbital_metadatas_template=orbital_metadatas_template,
            k_mesh=k_mesh,
            k_path=k_path,
            beta=beta,
            n_max=n_max,
            mu=0.0,
            mixing=0.5,
            max_iter=50,
            tol=1e-5,
        )
        results.append(result)
        print(f"    Converged in {result['n_iterations']} iterations")

    # Generate 3x3 band structure grid
    print("\n15. Generating 3x3 band structure grid...")
    plot_band_structure_3x3_grid(results, k_path.cpu(), ticks, "kagome_f_dmft_bands_3x3.png")

    # Generate vertical DOS stack
    print("\n16. Generating vertical DOS stack...")
    plot_dos_vertical_stack(results, "kagome_f_dmft_dos_vertical.png")

    print("\n" + "=" * 70)
    print("DMFT calculation complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Lattice: Kagome-F (4 sites/cell)")
    print(f"  - Solver: IPT (second-order perturbation theory)")
    print(f"  - Convergence: {dmft.n_iterations} iterations")
    print(f"  - Orbital selectivity: f-orbital Σ >> d-orbital Σ")
    print(f"\nGenerated plots:")
    print(f"  1. kagome_f_dmft_convergence.png - DMFT convergence history")
    print(f"  2. kagome_f_dmft_self_energy_orbital.png - Orbital Σ (imaginary)")
    print(f"  3. kagome_f_dmft_self_energy_real.png - Orbital Σ (real)")
    print(f"  4. kagome_f_dmft_orbital_selectivity.png - Zero-frequency Σ comparison")
    print(f"  5. kagome_f_dmft_greens_comparison.png - Green's function comparison")
    print(f"  6. kagome_f_dmft_greens_imag.png - Im[G(iωₙ)] (TRIQS-style validation)")
    print(f"  7. kagome_f_dmft_bands_comparison.png - Band structure (non-int vs int)")
    print(f"  8. kagome_f_dmft_dos_comparison.png - DOS (non-int vs int)")
    print(f"  9. kagome_f_dmft_dos_pade.png - DOS with Pade continuation")
    print(f" 10. kagome_f_dmft_bands_3x3.png - 3x3 band structure grid (U scan)")
    print(f" 11. kagome_f_dmft_dos_vertical.png - Vertical DOS stack (U scan)")

    # ============================================================
    # 15. t_f parameter scan
    # ============================================================
    print("\n" + "=" * 70)
    print("Starting t_f parameter scan...")
    print("=" * 70)

    # Create results directory with date
    from datetime import datetime
    date_str = datetime.now().strftime("%y%m%d")
    results_base_dir = f"kagome_f_dmft_results_{date_str}"
    Path(results_base_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_base_dir}/")

    # t_f values to scan: -0.1 to -1.0
    tf_values = [-0.1 * i for i in range(1, 11)]  # [-0.1, -0.2, ..., -1.0]
    print(f"\nt_f scan values: {tf_values}")
    print(f"Total t_f values: {len(tf_values)}")
    print(f"Expected plots: {len(tf_values) * 11}")

    # For each t_f, run simplified analysis (key plots only)
    for tf_val in tf_values:
        print(f"\n{'='*60}")
        print(f"Running DMFT for t_f = {tf_val:.1f}")
        print(f"{'='*60}")

        # Create subdirectory for this t_f value
        tf_dir = Path(results_base_dir) / f"tf_{tf_val:.1f}"
        tf_dir.mkdir(parents=True, exist_ok=True)

        # Run DMFT for this t_f value
        result = run_dmft_for_tf(
            tf_value=tf_val,
            device=device,
            lattice=lattice,
            U_f=U_f,
            epsilon_f=epsilon_f,
            U_d=U_d,
            k_mesh=k_mesh,
            k_path=k_path,
            k_path_ticks=ticks,
            beta=beta,
            n_max=n_max,
            mu=0.0,
            mixing=0.5,
            max_iter=50,
            tol=1e-5,
        )

        print(f"  Converged in {result['n_iterations']} iterations")

        # Generate key plots for this t_f value
        Sigma = result['Sigma']
        dmft_result = result['dmft']
        n_orb = Sigma.shape[-1]
        n_max_local = len(result['omega']) // 2

        # Extract orbital self-energies
        Sigma_orb = []
        for i in range(n_orb):
            Sigma_orb.append(Sigma.tensor[:, i, i])

        Sigma_zero_freq = [s[n_max_local].imag.item() for s in Sigma_orb]

        iwn_vals = torch.arange(len(result['omega'])).float()
        colors = plt.cm.tab10(torch.linspace(0, 1, n_orb))
        orbital_labels = ["K1($d_{x^2-y^2}$)", "K2($d_{xy}$)", "K3($d_{xz}$)", "F($f$)"]

        # Plot 1: Convergence
        history = dmft_result.get_convergence_history()
        fig, ax = setup_example_figure('single')
        ax.semilogy(history["Sigma_diff"], 'o-', markersize=4, color='steelblue')
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"|ΔΣ|/|Σ|")
        ax.set_title(f"DMFT Convergence (t_f={tf_val:.1f})")
        ax.grid(True, alpha=0.3)
        save_example_figure(fig, f"kagome_f_dmft_convergence.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 2: Self-energy (imaginary)
        fig, ax = setup_example_figure('single')
        for i in range(n_orb):
            Sigma_im = Sigma_orb[i].imag.cpu().numpy()
            ax.plot(iwn_vals.numpy(), Sigma_im, 'o-', markersize=3,
                    label=f"Orbital {orbital_labels[i]}", color=colors[i])
        ax.set_xlabel(r"Matsubara Index $n$")
        ax.set_ylabel(r"Im $\Sigma(i\omega_n)$")
        ax.set_title(f"Self-Energy Imag (t_f={tf_val:.1f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        save_example_figure(fig, f"kagome_f_dmft_self_energy_orbital.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 3: Self-energy (real)
        fig, ax = setup_example_figure('single')
        for i in range(n_orb):
            Sigma_re = Sigma_orb[i].real.cpu().numpy()
            ax.plot(iwn_vals.numpy(), Sigma_re, 's-', markersize=3,
                    label=f"Orbital {orbital_labels[i]}", color=colors[i])
        ax.set_xlabel(r"Matsubara Index $n$")
        ax.set_ylabel(r"Re $\Sigma(i\omega_n)$")
        ax.set_title(f"Self-Energy Real (t_f={tf_val:.1f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        save_example_figure(fig, f"kagome_f_dmft_self_energy_real.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 4: Orbital selectivity
        fig, ax = setup_example_figure('single')
        x_pos = torch.arange(n_orb).float()
        ax.bar(x_pos, Sigma_zero_freq, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Orb {i}\n{result['orbital_metadatas'][i].orb}" for i in range(n_orb)],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(r"Im $\Sigma(i\omega_0)$")
        ax.set_title(f"Orbital Selectivity (t_f={tf_val:.1f})")
        ax.grid(True, alpha=0.3, axis='y')
        save_example_figure(fig, f"kagome_f_dmft_orbital_selectivity.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 5: Green's function comparison
        G_int = result['G_int']
        G0_loc_nonint = result['G0_loc_nonint']

        fig, axes = setup_example_figure('comparison_2')
        for i in range(n_orb):
            G_nonint_im = G0_loc_nonint[:, i, i].numpy()
            axes[0].plot(iwn_vals.numpy(), G_nonint_im, 'o-', markersize=3,
                         label=orbital_labels[i], color=colors[i])
        axes[0].set_xlabel(r"Matsubara Index $n$")
        axes[0].set_ylabel(r"Im $G_0(i\omega_n)$")
        axes[0].set_title("Non-Interacting Green's Function")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for i in range(n_orb):
            G_int_im = G_int[:, i, i].numpy()
            axes[1].plot(iwn_vals.numpy(), G_int_im, 's-', markersize=3,
                         label=orbital_labels[i], color=colors[i])
        axes[1].set_xlabel(r"Matsubara Index $n$")
        axes[1].set_ylabel(r"Im $G(i\omega_n)$")
        axes[1].set_title("Interacting Green's Function")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        save_example_figure(fig, f"kagome_f_dmft_greens_comparison.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 6: Green's function imaginary
        fig, ax = setup_example_figure('single')
        for i in range(n_orb):
            G_im = G_int[:, i, i].numpy()
            ax.plot(iwn_vals.numpy(), G_im, 'o-', markersize=3,
                    label=orbital_labels[i], color=colors[i])
        ax.set_xlabel(r"Matsubara Index $n$")
        ax.set_ylabel(r"Im $G(i\omega_n)$")
        ax.set_title(f"Green's Function Imag (t_f={tf_val:.1f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save_example_figure(fig, f"kagome_f_dmft_greens_imag.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 7: Band structure comparison
        fig, axes = setup_example_figure('dual')
        bs_nonint = BandStructure()
        bs_nonint.compute(result['evals_nonint'], result['k_path'], result['ticks'])
        bs_nonint.plot(ax=axes[0], title="Non-interacting Bands")

        bs_int = BandStructure()
        bs_int.compute(result['evals_int'], result['k_path'], result['ticks'])
        bs_int.plot(ax=axes[1], title=f"Interacting Bands (t_f={tf_val:.1f})")

        save_example_figure(fig, f"kagome_f_dmft_bands_comparison.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 8: DOS comparison
        fig, ax = setup_example_figure('single')
        ax.plot(result['omega_dos'].numpy(), result['rho_nonint'].numpy(),
                label='Non-interacting (U=0)', linewidth=2, color='steelblue')
        ax.plot(result['omega_dos'].numpy(), result['rho_int'].numpy(),
                label=f'Interacting (U_f={U_f})', linewidth=2,
                color='crimson', linestyle='--')
        ax.set_xlabel(r"Energy $\omega$ ($|t|$)")
        ax.set_ylabel("DOS")
        ax.set_title(f"DOS Comparison (t_f={tf_val:.1f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_example_figure(fig, f"kagome_f_dmft_dos_comparison.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 9: Pade continuation (if available)
        try:
            from condmatTensor.manybody.preprocessing import SpectralFunction
            from condmatTensor.core import BaseTensor

            spectral = SpectralFunction()
            evals_mesh = result['evals_mesh']
            evals_min = evals_mesh.min().item()
            evals_max = evals_mesh.max().item()
            sigma_shift = abs(Sigma.tensor[n_max_local].real).max().item()
            omega_min, omega_max = calculate_dos_range(evals_min, evals_max, sigma_shift, U_f)
            omega_pade = torch.linspace(omega_min, omega_max, 1000)

            G_int_base = BaseTensor(
                tensor=G_int,
                labels=['iwn', 'orb_i', 'orb_j'],
                orbital_names=result['Hk_mesh'].orbital_names,
                orbital_metadatas=result['Hk_mesh'].orbital_metadatas,
            )

            _, A_int_pade = spectral.from_matsubara(
                G_int_base, omega_pade, eta=0.05, method="pade",
                beta=10.0, n_min=0, n_max=32
            )
            dos_int_pade = spectral.compute_dos(A_int_pade)

            fig, ax = setup_example_figure('single')
            ax.plot(result['omega_dos'].numpy(), result['rho_nonint'].numpy(),
                    label='Non-interacting', linewidth=2, color='steelblue')
            ax.plot(result['omega_dos'].numpy(), result['rho_int'].numpy(),
                    label='Interacting (Static Σ)', linewidth=2,
                    color='crimson', linestyle='--')
            ax.plot(omega_pade.numpy(), dos_int_pade.numpy(),
                    label='Interacting (Pade)', linewidth=2,
                    color='darkgreen', linestyle='-.')
            ax.set_xlabel(r"Energy $\omega$ ($|t|$)")
            ax.set_ylabel("DOS")
            ax.set_title(f"Pade Continuation (t_f={tf_val:.1f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_example_figure(fig, f"kagome_f_dmft_dos_pade.png", output_dir=str(tf_dir))
            plt.close(fig)
        except Exception as e:
            print(f"  Pade continuation skipped: {e}")

        # Plot 10: 3x3 band structure grid (U scan for this t_f)
        U_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        results_U = []
        for U_val in U_values:
            result_U = run_dmft_for_U(
                U_f_value=U_val,
                device=device,
                lattice=lattice,
                tb_model=result['tb_model'],
                orbital_metadatas_template=result['orbital_metadatas'],
                k_mesh=k_mesh,
                k_path=k_path,
                beta=beta,
                n_max=n_max,
                mu=0.0,
                mixing=0.5,
                max_iter=30,
                tol=1e-4,
            )
            results_U.append(result_U)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, res_U in enumerate(results_U):
            ax = axes[i]
            U_val = res_U['U_f']

            for band in range(res_U['evals_nonint'].shape[1]):
                ax.plot(res_U['evals_nonint'][:, band].numpy(),
                       color='gray', alpha=0.3, linewidth=1)

            for band in range(res_U['evals_int'].shape[1]):
                ax.plot(res_U['evals_int'][:, band].numpy(),
                       color='steelblue', alpha=0.7, linewidth=1.5)

            ax.set_title(f'U_f = {U_val:.1f}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(-5, 8)

        for ax in axes:
            ax.tick_params(labelbottom=False, labelleft=False)
        for i in [6, 7, 8]:
            axes[i].tick_params(labelbottom=True)
        for i in [0, 3, 6]:
            axes[i].tick_params(labelleft=True)

        n_k = len(k_path)
        tick_positions = [0, n_k // 3, 2 * n_k // 3, n_k - 1]
        tick_labels = ['G', 'K', 'M', 'G']

        for i in [6, 7, 8]:
            axes[i].set_xticks(tick_positions)
            axes[i].set_xticklabels(tick_labels)

        fig.suptitle(f'Band Structure vs U_f (t_f={tf_val:.1f})', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.02, 'k-path', ha='center', fontsize=14)
        fig.text(0.02, 0.5, 'Energy (|t|)', va='center', rotation=90, fontsize=14)

        plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
        save_example_figure(fig, f"kagome_f_dmft_bands_3x3.png", output_dir=str(tf_dir))
        plt.close(fig)

        # Plot 11: Vertical DOS stack (U scan for this t_f)
        n_U = len(results_U)
        fig, axes = plt.subplots(n_U, 1, figsize=(10, 2.5 * n_U), sharex=True)

        if n_U == 1:
            axes = [axes]

        for i, res_U in enumerate(results_U):
            ax = axes[i]
            U_val = res_U['U_f']
            omega = res_U['omega_dos'].numpy()
            rho_nonint_U = res_U['rho_nonint'].numpy()
            rho_int_U = res_U['rho_int'].numpy()

            ax.plot(omega, rho_nonint_U, color='gray', alpha=0.5,
                   linewidth=1.5, label='Non-interacting')
            ax.plot(omega, rho_int_U, color='crimson', linewidth=2,
                   label=f'Interacting (U_f={U_val:.1f})')

            ax.set_ylabel('DOS', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(-0.5, max(rho_int_U.max().item(), 1.0) * 1.1)

        axes[-1].set_xlabel('Energy $\\omega$ ($|t|$)', fontsize=12)
        fig.suptitle(f'DOS vs U_f (t_f={tf_val:.1f})', fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        save_example_figure(fig, f"kagome_f_dmft_dos_vertical.png", output_dir=str(tf_dir))
        plt.close(fig)

        print(f"  Saved 11 plots to {tf_dir}/")

    print("\n" + "=" * 70)
    print(f"t_f scan complete! All results saved to {results_base_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
