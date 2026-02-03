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

**Bayesian Optimization Backends:**
- auto: Automatically detect best available backend (SOBER > BoTorch > Simple)
- sober: Sequential Optimization using Ensemble of Regressors (preferred)
- botorch: Gaussian Process with Expected Improvement
- simple: Thompson sampling fallback

Reference:
    - "Effective Hamiltonians for heavy fermion systems" - Coleman, PRB (1987)
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
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

import argparse
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import DOSCalculator
from condmatTensor.optimization import EffectiveArrayOptimizer


def build_kagome_f_system(
    t: float = -1.0,
    t_f: float = -0.5,
    epsilon_f: float = 0.0,
) -> tuple[BravaisLattice, HoppingModel, torch.Tensor]:
    """Build Kagome-F system with f-orbital.

    Returns:
        (lattice, tb_model, H_full) where H_full includes Kagome and f-orbitals
    """
    lattice = build_kagome_f_lattice(t)
    tb_model = build_kagome_f_model(lattice, t=t, tf=t_f, fd_hybridization=t_f)
    tb_model.add_hopping("F", "F", [0, 0], epsilon_f, add_hermitian=False)

    k_path, _ = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=30)
    H_full = tb_model.build_Hk(k_path)

    return lattice, tb_model, H_full


def compare_backends(
    optimizer: EffectiveArrayOptimizer,
    backends: list = ["auto", "botorch", "simple"],
    n_init: int = 10,
    n_iter: int = 30,
    seed: int = 42,
    trace_optimization: bool = False,
    device: str = "cpu",
) -> dict:
    """Run optimization with multiple backends and compare results."""
    results = {}

    original_device = optimizer.H_cc_0.tensor.device
    target_device = torch.device(device)

    optimizer.H_cc_0 = optimizer.H_cc_0.to(target_device)
    optimizer.H_full = optimizer.H_full.to(target_device)

    print(f"\nDevice: {device.upper()}")
    if torch.cuda.is_available() and device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    for backend_name in backends:
        print(f"\n{'=' * 70}")
        print(f"Testing {backend_name.upper()} backend on {device.upper()}")
        print('=' * 70)

        optimizer.reset()

        try:
            start_time = time.time()

            J_eff, S_eff = optimizer.optimize(
                J_bounds=(0.01, 2.0),
                n_init=n_init,
                n_iter=n_iter,
                backend=backend_name,
                verbose=True,
                device=target_device,
            )

            elapsed_time = time.time() - start_time
            metrics = optimizer.verify(verbose=False)

            results[backend_name] = {
                'J_eff': J_eff,
                'S_eff': S_eff.cpu().numpy() if isinstance(S_eff, torch.Tensor) else S_eff,
                'time': elapsed_time,
                'rmse': metrics['rmse'],
                'mae': metrics['mean_absolute_error'],
                'correlation': metrics['correlation'],
                'device': device,
            }

            print(f"\n{backend_name.upper()} results:")
            print(f"  J_eff = {J_eff:.6f}")
            print(f"  S_eff = [{S_eff[0]:.6f}, {S_eff[1]:.6f}, {S_eff[2]:.6f}]")
            print(f"  Time = {elapsed_time:.2f}s")
            print(f"  RMSE = {metrics['rmse']:.6f}")
            print(f"  Correlation = {metrics['correlation']:.6f}")

        except (ImportError, ValueError, RuntimeError) as e:
            print(f"\n{backend_name.upper()} backend failed: {e}")
            print(f"  Skipping...")

    optimizer.H_cc_0 = optimizer.H_cc_0.to(original_device)
    optimizer.H_full = optimizer.H_full.to(original_device)

    return results


def compare_backends_cpu_gpu(
    optimizer: EffectiveArrayOptimizer,
    backends: list = ["simple", "botorch"],
    n_init: int = 10,
    n_iter: int = 30,
    seed: int = 42,
) -> dict:
    """Compare all backends on both CPU and GPU."""
    all_results = {}

    print("\n" + "=" * 70)
    print("CPU BENCHMARK")
    print("=" * 70)
    all_results['cpu'] = compare_backends(
        optimizer, backends=backends, n_init=n_init, n_iter=n_iter,
        seed=seed, trace_optimization=False, device="cpu",
    )

    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU BENCHMARK")
        print("=" * 70)
        all_results['gpu'] = compare_backends(
            optimizer, backends=backends, n_init=n_init, n_iter=n_iter,
            seed=seed, trace_optimization=False, device="cuda",
        )
    else:
        all_results['gpu'] = {}

    return all_results


def plot_backend_comparison(results: dict, save_path: str = "kagome_f_backend_comparison.png"):
    """Create 4-panel comparison plot for different backends."""
    if not results:
        return

    fig, axes = setup_example_figure('comparison_2x3')
    if isinstance(axes, plt.Axes):
        axes = np.array([[fig.add_subplot(3, 2, i)] for i in range(1, 7)])

    backend_names = list(results.keys())
    n_backends = len(backend_names)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    backend_colors = [colors[i % len(colors)] for i in range(n_backends)]

    # Panel 1: J_eff comparison
    ax = axes[0, 0]
    J_values = [results[b]['J_eff'] for b in backend_names]
    bars = ax.bar(range(n_backends), J_values, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('$J_{eff}$', fontsize=12)
    ax.set_title('Effective Exchange Parameter', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Time comparison
    ax = axes[0, 1]
    times = [results[b]['time'] for b in backend_names]
    bars = ax.bar(range(n_backends), times, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Optimization Time', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: RMSE comparison
    ax = axes[1, 0]
    rmses = [results[b]['rmse'] for b in backend_names]
    bars = ax.bar(range(n_backends), rmses, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Correlation comparison
    ax = axes[1, 1]
    correlations = [results[b]['correlation'] for b in backend_names]
    bars = ax.bar(range(n_backends), correlations, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Eigenvalue Correlation', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Bayesian Optimization Backend Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_example_figure(fig, save_path)


def plot_cpu_gpu_comparison(all_results: dict, save_path: str = "kagome_f_cpu_gpu_comparison.png"):
    """Create comprehensive CPU vs GPU comparison plot."""
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))[:3]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'cpu': '#3498db', 'gpu': '#e74c3c'}

    for idx, backend_name in enumerate(backend_list):
        # Time comparison
        ax = axes[0, idx]
        devices, times = [], []
        if backend_name in cpu_results:
            devices.append('CPU')
            times.append(cpu_results[backend_name]['time'])
        if backend_name in gpu_results:
            devices.append('GPU')
            times.append(gpu_results[backend_name]['time'])
        if devices:
            ax.bar(devices, times, color=[colors[d.lower()] for d in devices],
                   alpha=0.7, edgecolor='black')
            ax.set_ylabel('Time (s)', fontsize=11)
            ax.set_title(f'{backend_name.upper()}: Time', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        # RMSE comparison
        ax = axes[1, idx]
        devices, rmses = [], []
        if backend_name in cpu_results:
            devices.append('CPU')
            rmses.append(cpu_results[backend_name]['rmse'])
        if backend_name in gpu_results:
            devices.append('GPU')
            rmses.append(gpu_results[backend_name]['rmse'])
        if devices:
            ax.bar(devices, rmses, color=[colors[d.lower()] for d in devices],
                   alpha=0.7, edgecolor='black')
            ax.set_ylabel('RMSE', fontsize=11)
            ax.set_title(f'{backend_name.upper()}: RMSE', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('CPU vs GPU Performance Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_example_figure(fig, save_path)


def plot_band_dos_comparison(
    all_results: dict,
    optimizer: EffectiveArrayOptimizer,
    lattice: BravaisLattice,
    k_path: torch.Tensor,
    save_path: str = "kagome_f_band_dos_comparison.png"
):
    """Plot band structures and DOS for all results."""
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))
    eig_full = optimizer._compute_eigenvalues(optimizer.H_full)

    fig, axes = plt.subplots(len(backend_list), 2, figsize=(16, 5 * len(backend_list)))
    if len(backend_list) == 1:
        axes = axes.reshape(1, -1)

    k_axis = torch.arange(eig_full.shape[0], dtype=torch.float64).numpy()

    for idx, backend_name in enumerate(backend_list):
        full_color, cpu_color, gpu_color = '#2c3e50', '#3498db', '#e74c3c'

        # Band structure plot
        ax_band = axes[idx, 0]

        for i in range(eig_full.shape[1]):
            ax_band.plot(k_axis, eig_full[:, i].cpu().numpy(),
                        color=full_color, alpha=0.3, linewidth=1)

        if backend_name in cpu_results:
            res = cpu_results[backend_name]
            J_cpu = res['J_eff']
            S_cpu = torch.tensor(res['S_eff'], dtype=torch.float64)
            H_eff_cpu = optimizer._build_effective_hamiltonian(J_cpu, S_cpu, torch.device('cpu'))
            eig_eff_cpu = optimizer._compute_eigenvalues(H_eff_cpu)

            for i in range(eig_eff_cpu.shape[1]):
                ax_band.plot(k_axis, eig_eff_cpu[:, i].cpu().numpy(),
                            color=cpu_color, alpha=0.7, linewidth=1.5, linestyle='-')

        if backend_name in gpu_results:
            res = gpu_results[backend_name]
            J_gpu = res['J_eff']
            S_gpu = torch.tensor(res['S_eff'], dtype=torch.float64)
            H_eff_gpu = optimizer._build_effective_hamiltonian(J_gpu, S_gpu, torch.device('cpu'))
            eig_eff_gpu = optimizer._compute_eigenvalues(H_eff_gpu)

            for i in range(eig_eff_gpu.shape[1]):
                ax_band.plot(k_axis, eig_eff_gpu[:, i].cpu().numpy(),
                            color=gpu_color, alpha=0.7, linewidth=1.5, linestyle='--')

        ax_band.set_ylabel("Energy ($|t|$)", fontsize=11)
        ax_band.set_title(f'{backend_name.upper()}: Band Structure Comparison', fontsize=12, fontweight='bold')
        ax_band.grid(True, alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=full_color, alpha=0.5, label='Full System'),
            Line2D([0], [0], color=cpu_color, alpha=0.8, label='CPU Result'),
            Line2D([0], [0], color=gpu_color, alpha=0.8, linestyle='--', label='GPU Result'),
        ]
        ax_band.legend(handles=legend_elements, fontsize=9, loc='upper right')

        # DOS plot
        ax_dos = axes[idx, 1]

        e_min = eig_full.min().item() - 1.0
        e_max = eig_full.max().item() + 1.0
        omega = torch.linspace(e_min, e_max, 200, dtype=torch.float64)

        dos_calc = DOSCalculator()
        omega_full, dos_full_vals = dos_calc.from_eigenvalues(eig_full, omega, eta=0.05)

        ax_dos.plot(omega_full.cpu().numpy(), dos_full_vals.cpu().numpy(),
                   color=full_color, alpha=0.5, linewidth=2, label='Full System')

        if backend_name in cpu_results:
            res = cpu_results[backend_name]
            J_cpu = res['J_eff']
            S_cpu = torch.tensor(res['S_eff'], dtype=torch.float64)
            H_eff_cpu = optimizer._build_effective_hamiltonian(J_cpu, S_cpu, torch.device('cpu'))
            eig_eff_cpu = optimizer._compute_eigenvalues(H_eff_cpu)
            omega_cpu, dos_cpu_vals = dos_calc.from_eigenvalues(eig_eff_cpu, omega, eta=0.05)

            ax_dos.plot(omega_cpu.cpu().numpy(), dos_cpu_vals.cpu().numpy(),
                       color=cpu_color, alpha=0.8, linewidth=2, label='CPU Result')

        if backend_name in gpu_results:
            res = gpu_results[backend_name]
            J_gpu = res['J_eff']
            S_gpu = torch.tensor(res['S_eff'], dtype=torch.float64)
            H_eff_gpu = optimizer._build_effective_hamiltonian(J_gpu, S_gpu, torch.device('cpu'))
            eig_eff_gpu = optimizer._compute_eigenvalues(H_eff_gpu)
            omega_gpu, dos_gpu_vals = dos_calc.from_eigenvalues(eig_eff_gpu, omega, eta=0.05)

            ax_dos.plot(omega_gpu.cpu().numpy(), dos_gpu_vals.cpu().numpy(),
                       color=gpu_color, alpha=0.8, linewidth=2, linestyle='--', label='GPU Result')

        ax_dos.set_xlabel("Energy ($|t|$)", fontsize=11)
        ax_dos.set_ylabel("DOS", fontsize=11)
        ax_dos.set_title(f'{backend_name.upper()}: Density of States', fontsize=12, fontweight='bold')
        ax_dos.grid(True, alpha=0.3)
        ax_dos.legend(fontsize=9)
        ax_dos.set_ylim(bottom=0)

    plt.suptitle('Band Structure & DOS Comparison (Backends × Devices)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_example_figure(fig, save_path)


def print_detailed_results_table(all_results: dict):
    """Print detailed results table for all results."""
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))

    print("\n" + "=" * 90)
    print("DETAILED RESULTS TABLE")
    print("=" * 90)

    print(f"{'Backend':<12} {'Device':<8} {'J_eff':<12} {'Time (s)':<12} {'RMSE':<12} {'Correlation':<14}")
    print("-" * 90)

    for backend in backend_list:
        if backend in cpu_results:
            res = cpu_results[backend]
            print(f"{backend.upper():<12} {'CPU':<8} {res['J_eff']:<12.6f} {res['time']:<12.2f} "
                  f"{res['rmse']:<12.6f} {res['correlation']:<14.6f}")
        if backend in gpu_results:
            res = gpu_results[backend]
            print(f"{backend.upper():<12} {'GPU':<8} {res['J_eff']:<12.6f} {res['time']:<12.2f} "
                  f"{res['rmse']:<12.6f} {res['correlation']:<14.6f}")

    print("=" * 90)


def plot_comprehensive_summary(all_results: dict, save_path: str = "kagome_f_comprehensive_summary.png"):
    """Create comprehensive summary plot with all metrics and devices."""
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))
    n_backends = len(backend_list)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    device_colors = {'cpu': '#3498db', 'gpu': '#e74c3c'}

    x = np.arange(n_backends)
    width = 0.35

    # Panel 1: Time comparison
    ax = axes[0, 0]
    cpu_times = [cpu_results.get(b, {}).get('time', 0) for b in backend_list]
    gpu_times = [gpu_results.get(b, {}).get('time', 0) for b in backend_list]

    ax.bar(x - width/2, cpu_times, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_times):
        ax.bar(x + width/2, gpu_times, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('(a) Optimization Time', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backend_list], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: RMSE comparison
    ax = axes[0, 1]
    cpu_rmse = [cpu_results.get(b, {}).get('rmse', 0) for b in backend_list]
    gpu_rmse = [gpu_results.get(b, {}).get('rmse', 0) for b in backend_list]

    ax.bar(x - width/2, cpu_rmse, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_rmse):
        ax.bar(x + width/2, gpu_rmse, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('(b) Accuracy (RMSE)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backend_list], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Correlation comparison
    ax = axes[0, 2]
    cpu_corr = [cpu_results.get(b, {}).get('correlation', 0) for b in backend_list]
    gpu_corr = [gpu_results.get(b, {}).get('correlation', 0) for b in backend_list]

    ax.bar(x - width/2, cpu_corr, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_corr):
        ax.bar(x + width/2, gpu_corr, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('(c) Eigenvalue Correlation', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backend_list], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: J_eff comparison
    ax = axes[1, 0]
    cpu_J = [cpu_results.get(b, {}).get('J_eff', 0) for b in backend_list]
    gpu_J = [gpu_results.get(b, {}).get('J_eff', 0) for b in backend_list]

    ax.bar(x - width/2, cpu_J, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_J):
        ax.bar(x + width/2, gpu_J, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax.set_ylabel('$J_{eff}$', fontsize=11)
    ax.set_title('(d) Effective Exchange Parameter', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backend_list], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 5: GPU Speedup
    ax = axes[1, 1]
    speedups, backend_names_speedup = [], []

    for b in backend_list:
        if b in cpu_results and b in gpu_results:
            cpu_time = cpu_results[b]['time']
            gpu_time = gpu_results[b]['time']
            if gpu_time > 0:
                speedups.append(cpu_time / gpu_time)
                backend_names_speedup.append(b.upper())

    if speedups:
        colors_map = {'SIMPLE': '#3498db', 'BOTORCH': '#e74c3c', 'SOBER': '#2ecc71'}
        colors_list = [colors_map.get(b, '#95a5a6') for b in backend_names_speedup]
        ax.bar(range(len(speedups)), speedups, color=colors_list, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(speedups)))
        ax.set_xticklabels(backend_names_speedup, fontsize=10)
        ax.set_ylabel('Speedup (CPU/GPU)', fontsize=11)
        ax.set_title('(e) GPU Speedup', fontsize=12, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')

    summary_data = []
    for b in backend_list:
        row = [
            b.upper(),
            f"{cpu_results.get(b, {}).get('time', 0):.2f}s",
            f"{gpu_results.get(b, {}).get('time', 0):.2f}s" if gpu_results.get(b, {}) else "N/A",
            f"{cpu_results.get(b, {}).get('rmse', 0):.4f}",
            f"{cpu_results.get(b, {}).get('correlation', 0):.3f}",
        ]
        summary_data.append(row)

    table = ax.table(cellText=summary_data,
                      colLabels=['Backend', 'CPU Time', 'GPU Time', 'CPU RMSE', 'CPU Corr'],
                      cellLoc='center',
                      bbox=[0, 0, 1, 1],
                      colWidths=[0.15, 0.17, 0.17, 0.17, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    ax.set_title('(f) Results Summary', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Comprehensive Bayesian Optimization Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_example_figure(fig, save_path)


def main():
    """Main function to run effective array optimization."""
    parser = argparse.ArgumentParser(
        description="Effective Array Optimizer for Kagome-F Lattice"
    )
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "sober", "botorch", "simple"],
                        help="Bayesian optimization backend (default: auto)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available backends")
    parser.add_argument("--cpu-gpu", action="store_true",
                        help="Compare all backends on both CPU and GPU")
    parser.add_argument("--n-init", type=int, default=10,
                        help="Number of initial samples (default: 10)")
    parser.add_argument("--n-iter", type=int, default=30,
                        help="Number of optimization iterations (default: 30)")

    args = parser.parse_args()

    print("=" * 70)
    if args.compare and args.cpu_gpu:
        print("Comprehensive CPU vs GPU Backend Comparison Mode")
    elif args.compare:
        print("Backend Comparison Mode")
    elif args.cpu_gpu:
        print("CPU vs GPU Comparison Mode")
    else:
        print("Effective Array Optimizer for Kagome-F Lattice")
    print("=" * 70)

    # System parameters
    t = -1.0
    t_f = -0.3
    epsilon_f = 0.5

    print(f"\nSystem parameters:")
    print(f"  t (Kagome-Kagome) = {t}")
    print(f"  t_f (Kagome-f) = {t_f}")
    print(f"  ε_f (f-orbital energy) = {epsilon_f}")

    # Build lattices and Hamiltonians
    lattice_cc = build_kagome_lattice(t)
    k_path_cc, _ = generate_k_path(lattice_cc, ["G", "K", "M", "G"], n_per_segment=30)

    # Build H_cc_0 (Kagome-only)
    from condmatTensor.manybody import LocalMagneticModel
    from condmatTensor.lattice import BravaisLattice as BravaisLatticeBase
    import math

    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2])
    a2 = torch.tensor([1.0, 0.0])
    cell_vectors = torch.stack([a1, a2])

    # Build spinful H_cc_0
    tb_cc = HoppingModel(lattice_cc)
    tb_cc.add_hopping(0, 1, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(1, 0, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(0, 1, [-1, 0], t, add_hermitian=False)
    tb_cc.add_hopping(1, 0, [1, 0], t, add_hermitian=False)
    tb_cc.add_hopping(0, 2, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(2, 0, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(0, 2, [0, -1], t, add_hermitian=False)
    tb_cc.add_hopping(2, 0, [0, 1], t, add_hermitian=False)
    tb_cc.add_hopping(1, 2, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(2, 1, [0, 0], t, add_hermitian=False)
    tb_cc.add_hopping(1, 2, [1, -1], t, add_hermitian=False)
    tb_cc.add_hopping(2, 1, [-1, 1], t, add_hermitian=False)
    H_cc_0 = tb_cc.build_Hk(k_path_cc)

    # Build full Kagome-F system
    basis_positions_full = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, sqrt3 / 4]),
        torch.tensor([1/3, 1/3]),
    ]
    lattice_full = BravaisLatticeBase(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions_full,
        num_orbitals=[1, 1, 1, 1],
    )

    tb_full = HoppingModel(lattice_full, orbital_labels=["A", "B", "C", "f"])
    tb_full.add_hopping("A", "B", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("B", "A", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("A", "B", [-1, 0], t, add_hermitian=False)
    tb_full.add_hopping("B", "A", [1, 0], t, add_hermitian=False)
    tb_full.add_hopping("A", "C", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("C", "A", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("A", "C", [0, -1], t, add_hermitian=False)
    tb_full.add_hopping("C", "A", [0, 1], t, add_hermitian=False)
    tb_full.add_hopping("B", "C", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("C", "B", [0, 0], t, add_hermitian=False)
    tb_full.add_hopping("B", "C", [1, -1], t, add_hermitian=False)
    tb_full.add_hopping("C", "B", [-1, 1], t, add_hermitian=False)
    tb_full.add_hopping("A", "f", [0, 0], t_f)
    tb_full.add_hopping("B", "f", [0, 0], t_f)
    tb_full.add_hopping("C", "f", [0, 0], t_f)
    tb_full.add_hopping("f", "f", [0, 0], epsilon_f)

    H_full_spinless = tb_full.build_Hk(k_path_cc)

    # Build spinful versions
    model = LocalMagneticModel()
    H_cc_0_spinful = model.build_spinful_hamiltonian(H_cc_0)
    H_full_spinful = model.build_spinful_hamiltonian(H_full_spinless)

    print(f"\nHamiltonian dimensions:")
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

    if args.compare and args.cpu_gpu:
        # Comprehensive CPU vs GPU comparison
        print("\nRunning comprehensive CPU vs GPU comparison...")

        all_results = compare_backends_cpu_gpu(
            optimizer,
            backends=["simple", "botorch", "sober"],
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=42,
        )

        if all_results and (all_results.get('cpu') or all_results.get('gpu')):
            plot_comprehensive_summary(all_results)
            plot_cpu_gpu_comparison(all_results)
            k_path, _ = generate_k_path(lattice_cc, ["G", "K", "M", "G"], n_per_segment=30)
            plot_band_dos_comparison(all_results, optimizer, lattice_cc, k_path)

        print_detailed_results_table(all_results)

    elif args.compare:
        # Compare all backends on CPU
        print("\nRunning backend comparison on CPU...")
        results = compare_backends(
            optimizer,
            backends=["simple", "botorch", "sober"],
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=42,
            device="cpu",
        )

        if results:
            plot_backend_comparison(results)

    elif args.cpu_gpu:
        # CPU vs GPU comparison for single backend
        backend_name = "botorch" if args.backend == "auto" else args.backend
        print(f"\nRunning {backend_name.upper()} backend on both CPU and GPU...")

        all_results = compare_backends_cpu_gpu(
            optimizer,
            backends=[backend_name],
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=42,
        )

        if all_results:
            plot_cpu_gpu_comparison(all_results)

    else:
        # Single optimization
        print(f"\nRunning optimization with {args.backend.upper()} backend...")

        J_eff, S_eff = optimizer.optimize(
            J_bounds=(0.01, 2.0),
            n_init=args.n_init,
            n_iter=args.n_iter,
            backend=args.backend,
            verbose=True,
        )

        print("\n" + "=" * 70)
        print("Verification")
        print("=" * 70)

        metrics = optimizer.verify(verbose=True)

        # Plot comparison
        print("\nGenerating comparison plot...")
        fig, axes = setup_example_figure('dual')

        eig_full = optimizer._compute_eigenvalues(optimizer.H_full)
        H_eff = optimizer._build_effective_hamiltonian(J_eff, S_eff, optimizer.H_cc_0.tensor.device)
        eig_eff = optimizer._compute_eigenvalues(H_eff)

        k_axis = torch.arange(eig_full.shape[0], dtype=torch.float64)
        for i in range(eig_full.shape[1]):
            axes[0].plot(k_axis.numpy(), eig_full[:, i].cpu().numpy(), 'b-', alpha=0.6)

        axes[0].set_xlabel("k-path index", fontsize=12)
        axes[0].set_ylabel("Energy ($|t|$)", fontsize=12)
        axes[0].set_title("Full Kagome-F System", fontsize=12)
        axes[0].grid(True, alpha=0.3)

        for i in range(eig_eff.shape[1]):
            axes[1].plot(k_axis.numpy(), eig_eff[:, i].cpu().numpy(), 'r-', alpha=0.8)

        axes[1].set_xlabel("k-path index", fontsize=12)
        axes[1].set_ylabel("Energy ($|t|$)", fontsize=12)
        axes[1].set_title(f"Effective Model ($J_{{eff}}$={J_eff:.3f})", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        save_example_figure(fig, "kagome_f_effective_array_comparison.png")

        # Using built-in comparison method
        fig, ax = setup_example_figure('single')
        optimizer.plot_comparison(ax=ax)
        save_example_figure(fig, "kagome_f_effective_array_bands.png")

        # Perturbation theory estimate
        print("\n" + "=" * 70)
        print("Perturbation Theory Estimate")
        print("=" * 70)

        J_pert, S_pert = optimizer.perturbation_theory(epsilon_f, t_f)
        print(f"  J_eff (perturbation theory) = {J_pert:.6f}")
        print(f"  S_eff (perturbation theory) = {S_pert.tolist()}")
        print(f"  J_eff (optimized) = {J_eff:.6f}")
        print(f"  S_eff (optimized) = {S_eff.tolist()}")

        # Validation
        print("\n" + "=" * 70)
        print("Validation Checks")
        print("=" * 70)

        assert metrics['rmse'] < 1.0, f"RMSE should be < 1.0"
        assert metrics['correlation'] > 0.9, f"Correlation should be > 0.9"
        assert 0.0 < J_eff < 2.0, f"J_eff should be in (0, 2)"
        print("  All validation checks passed!")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
