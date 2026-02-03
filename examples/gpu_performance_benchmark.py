#!/usr/bin/env python3
"""
GPU Performance Benchmark for condmatTensor

This script benchmarks the performance difference between CPU and GPU
for various condensed matter physics operations:

1. Hamiltonian construction (H(k) from TightBindingModel)
2. Diagonalization (eigenvalue computation)
3. DOS calculation (large k-mesh)
4. Bayesian optimization

**Expected Results:**
- GPU shows significant speedup for:
  - Large k-mesh operations (100x100+)
  - Diagonalization of large matrices
  - Batch operations
- CPU may be competitive for:
  - Small k-meshes (< 20x20)
  - Serial operations with small data

Usage:
    python gpu_performance_benchmark.py [--device-only DEVICE]

Output:
    gpu_performance_benchmark.png - 4-panel comparison plot
"""

# example_utils handles path setup automatically
from example_utils import (
    get_example_device,
    build_kagome_lattice,
    build_kagome_model,
    setup_example_figure,
    save_example_figure,
)

import time
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from condmatTensor.core import is_cuda_available, get_device
from condmatTensor.lattice import BravaisLattice, generate_kmesh, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.optimization import BayesianOptimizer


def benchmark_hamiltonian_construction(
    lattice: BravaisLattice,
    tb_model,
    nk_list: list,
    device: torch.device,
    n_trials: int = 3,
) -> list:
    """Benchmark H(k) construction for different k-mesh sizes."""
    times = []

    for nk in nk_list:
        trial_times = []
        for _ in range(n_trials):
            k_mesh = generate_kmesh(lattice, nk=nk, device=device)

            start = time.time()
            Hk = tb_model.build_Hk(k_mesh)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            trial_times.append(elapsed)

        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        times.append(avg_time)
        print(f"  nk={nk:3d}: {avg_time:.4f}s +/- {std_time:.4f}s")

    return times


def benchmark_diagonalization(
    nk_list: list,
    n_orb: int,
    device: torch.device,
    n_trials: int = 3,
) -> list:
    """Benchmark eigensolver for different matrix sizes."""
    times = []

    for nk in nk_list:
        trial_times = []
        for _ in range(n_trials):
            H = torch.randn(nk, n_orb, n_orb, dtype=torch.complex128, device=device)
            H = (H + H.conj().transpose(-1, -2)) / 2

            start = time.time()
            evals = torch.zeros(nk, n_orb, dtype=torch.float64, device=device)
            for k in range(nk):
                evals[k] = torch.linalg.eigvalsh(H[k]).real
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            trial_times.append(elapsed)

        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        times.append(avg_time)
        print(f"  nk={nk:4d}, n_orb={n_orb}: {avg_time:.4f}s +/- {std_time:.4f}s")

    return times


def benchmark_dos(
    lattice: BravaisLattice,
    tb_model,
    nk_list: list,
    device: torch.device,
    n_trials: int = 3,
) -> list:
    """Benchmark DOS calculation for different k-mesh sizes."""
    times = []

    for nk in nk_list:
        trial_times = []
        for _ in range(n_trials):
            k_mesh = generate_kmesh(lattice, nk=nk, device=device)
            Hk = tb_model.build_Hk(k_mesh)

            start = time.time()
            N_k = Hk.shape[0]
            n_orb = Hk.shape[-1]
            evals = torch.zeros(N_k, n_orb, dtype=torch.float64, device=device)
            for k in range(N_k):
                evals[k] = torch.linalg.eigvalsh(Hk.tensor[k]).real

            evals_flat = evals.flatten()
            hist = torch.histc(evals_flat, bins=200, min=-5, max=5)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            trial_times.append(elapsed)

        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        times.append(avg_time)
        print(f"  nk={nk:3d}: {avg_time:.4f}s +/- {std_time:.4f}s")

    return times


def benchmark_bayesian_optimization(
    device: torch.device,
    n_init: int = 10,
    n_iter: int = 30,
    n_trials: int = 3,
) -> float:
    """Benchmark Bayesian optimization."""
    def objective(X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        a = 1.0
        b = 100.0
        x = X[:, 0]
        y = X[:, 1]
        result = (a - x) ** 2 + b * (y - x ** 2) ** 2
        return result.squeeze() if result.numel() == 1 else result

    bounds = [(-2.0, 2.0), (-1.0, 3.0)]

    trial_times = []
    for _ in range(n_trials):
        opt = BayesianOptimizer(
            bounds=bounds,
            n_init=n_init,
            n_iter=n_iter,
            backend="simple",
            seed=42,
        )

        start = time.time()
        opt.optimize(objective, maximize=False, verbose=False, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        trial_times.append(elapsed)

    avg_time = np.mean(trial_times)
    std_time = np.std(trial_times)
    print(f"  {n_init}+{n_iter} iterations: {avg_time:.4f}s +/- {std_time:.4f}s")

    return avg_time


def run_benchmark(device_only: str = None) -> dict:
    """Run all benchmarks on CPU and GPU."""
    results = {}

    if device_only:
        devices_to_test = [torch.device(device_only)]
    else:
        devices_to_test = [torch.device("cpu")]
        if is_cuda_available():
            devices_to_test.append(torch.device("cuda"))

    # Build lattice and model
    lattice = build_kagome_lattice()
    tb_model = build_kagome_model(lattice)

    nk_list = [20, 40, 60, 80, 100]
    nk_diag_list = [100, 500, 1000, 2000]
    nk_dos_list = [20, 40, 60, 80, 100]

    for device in devices_to_test:
        print(f"\n{'=' * 70}")
        print(f"Running benchmarks on {device}")
        print('=' * 70)

        tb_model_device = tb_model.to(device) if device.type == "cuda" else tb_model

        print("\n1. Hamiltonian Construction")
        h_construction_times = benchmark_hamiltonian_construction(
            lattice, tb_model_device, nk_list, device
        )

        print("\n2. Diagonalization")
        diag_times = benchmark_diagonalization(nk_diag_list, 3, device)

        print("\n3. DOS Calculation")
        dos_times = benchmark_dos(lattice, tb_model_device, nk_dos_list, device)

        print("\n4. Bayesian Optimization")
        bo_time = benchmark_bayesian_optimization(device)

        results[str(device)] = {
            'h_construction': (nk_list, h_construction_times),
            'diagonalization': (nk_diag_list, diag_times),
            'dos': (nk_dos_list, dos_times),
            'bayesian_opt': bo_time,
        }

    return results


def plot_benchmark_results(
    results: dict,
    save_path: str = "gpu_performance_benchmark.png",
):
    """Create 4-panel comparison plot of benchmark results."""
    fig, axes = setup_example_figure('comparison_2x3')
    if isinstance(axes, plt.Axes):
        axes = np.array([[fig.add_subplot(2, 2, i)] for i in range(1, 5)]).reshape(2, 2)

    colors = {'cpu': '#e74c3c', 'cuda': '#3498db'}
    markers = {'cpu': 'o', 'cuda': 's'}

    devices = list(results.keys())

    # Panel 1: Hamiltonian construction
    ax = axes[0, 0]
    for device in devices:
        nk_list, times = results[device]['h_construction']
        ax.plot(nk_list, times, marker=markers.get(device, 'o'),
                color=colors.get(device, 'gray'), label=device.upper(),
                linewidth=2, markersize=8)
    ax.set_xlabel('k-mesh size (nk)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Hamiltonian Construction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Diagonalization
    ax = axes[0, 1]
    for device in devices:
        nk_list, times = results[device]['diagonalization']
        ax.plot(nk_list, times, marker=markers.get(device, 'o'),
                color=colors.get(device, 'gray'), label=device.upper(),
                linewidth=2, markersize=8)
    ax.set_xlabel('Number of k-points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Diagonalization (n_orb=3)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: DOS calculation
    ax = axes[1, 0]
    for device in devices:
        nk_list, times = results[device]['dos']
        ax.plot(nk_list, times, marker=markers.get(device, 'o'),
                color=colors.get(device, 'gray'), label=device.upper(),
                linewidth=2, markersize=8)
    ax.set_xlabel('k-mesh size (nk)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('DOS Calculation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: Bayesian optimization
    ax = axes[1, 1]
    if 'cpu' in results and 'cuda' in results:
        devices_bar = ['CPU', 'GPU']
        times_bar = [results['cpu']['bayesian_opt'], results['cuda']['bayesian_opt']]
        colors_bar = [colors['cpu'], colors['cuda']]
        bars = ax.bar(devices_bar, times_bar, color=colors_bar, alpha=0.7,
                      edgecolor='black', linewidth=2)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Bayesian Optimization (10+30 iterations)', fontsize=13, fontweight='bold')

        speedup = times_bar[0] / times_bar[1] if times_bar[1] > 0 else 0
        ax.text(0.5, max(times_bar) * 0.9, f'GPU Speedup: {speedup:.2f}x',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        for bar, val in zip(bars, times_bar):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=11)
    else:
        device = devices[0]
        time_val = results[device]['bayesian_opt']
        ax.bar([device.upper()], [time_val], color=colors.get(str(device), 'gray'),
               alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Bayesian Optimization (10+30 iterations)', fontsize=13, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('GPU vs CPU Performance Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_example_figure(fig, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Performance Benchmark for condmatTensor"
    )
    parser.add_argument("--device-only", type=str, choices=["cpu", "cuda"],
                        help="Run benchmark on specified device only")

    args = parser.parse_args()

    print("=" * 70)
    print("condmatTensor GPU Performance Benchmark")
    print("=" * 70)

    if args.device_only == "cuda" and not is_cuda_available():
        print("Error: CUDA requested but not available.")
        return

    if not args.device_only and not is_cuda_available():
        print("Note: CUDA not available. Running CPU-only benchmarks.")

    results = run_benchmark(device_only=args.device_only)

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)

        for key in ['h_construction', 'diagonalization', 'dos']:
            nk_list_cpu, times_cpu = results['cpu'][key]
            nk_list_cuda, times_cuda = results['cuda'][key]

            if len(times_cpu) == len(times_cuda):
                avg_speedup = np.mean([t_c / t_g for t_c, t_g in zip(times_cpu, times_cuda)])
                print(f"  {key}: {avg_speedup:.2f}x average speedup on GPU")

        bo_speedup = results['cpu']['bayesian_opt'] / results['cuda']['bayesian_opt']
        print(f"  bayesian_opt: {bo_speedup:.2f}x speedup on GPU")

    plot_benchmark_results(results)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
