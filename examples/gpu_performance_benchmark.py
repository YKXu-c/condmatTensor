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

**Requirements:**
- CUDA-capable GPU (for GPU benchmarks)
- PyTorch with CUDA support

Usage:
    python gpu_performance_benchmark.py [--device-only DEVICE]

Output:
    gpu_performance_benchmark.png - 4-panel comparison plot
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from condmatTensor.core import is_cuda_available, get_device
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_kmesh, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.optimization import BayesianOptimizer


def build_kagome_lattice(t: float = -1.0) -> BravaisLattice:
    """Build pure Kagome lattice for benchmarking."""
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
        num_orbitals=[1, 1, 1],
    )


def build_kagome_model(lattice: BravaisLattice, t: float = -1.0) -> HoppingModel:
    """Build Kagome tight-binding model."""
    tb_model = HoppingModel(lattice)

    # Nearest-neighbor hopping
    tb_model.add_hopping(0, 1, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(1, 0, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(0, 1, [-1, 0], t, add_hermitian=False)
    tb_model.add_hopping(1, 0, [1, 0], t, add_hermitian=False)
    tb_model.add_hopping(0, 2, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(2, 0, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(0, 2, [0, -1], t, add_hermitian=False)
    tb_model.add_hopping(2, 0, [0, 1], t, add_hermitian=False)
    tb_model.add_hopping(1, 2, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(2, 1, [0, 0], t, add_hermitian=False)
    tb_model.add_hopping(1, 2, [1, -1], t, add_hermitian=False)
    tb_model.add_hopping(2, 1, [-1, 1], t, add_hermitian=False)

    return tb_model


def benchmark_hamiltonian_construction(
    lattice: BravaisLattice,
    tb_model: HoppingModel,
    nk_list: list,
    device: torch.device,
    n_trials: int = 3,
) -> list:
    """Benchmark H(k) construction for different k-mesh sizes.

    Args:
        lattice: BravaisLattice for k-mesh generation
        tb_model: TightBindingModel for Hamiltonian construction
        nk_list: List of k-mesh sizes to test
        device: Device to run benchmark on
        n_trials: Number of trials for averaging

    Returns:
        List of average construction times for each k-mesh size
    """
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
    """Benchmark eigensolver for different matrix sizes.

    Args:
        nk_list: List of number of k-points (matrices to diagonalize)
        n_orb: Number of orbitals (matrix size)
        device: Device to run benchmark on
        n_trials: Number of trials for averaging

    Returns:
        List of average diagonalization times for each k-mesh size
    """
    times = []

    for nk in nk_list:
        trial_times = []
        for _ in range(n_trials):
            # Create random Hermitian matrices
            H = torch.randn(nk, n_orb, n_orb, dtype=torch.complex128, device=device)
            H = (H + H.conj().transpose(-1, -2)) / 2  # Make Hermitian

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
    tb_model: HoppingModel,
    nk_list: list,
    device: torch.device,
    n_trials: int = 3,
) -> list:
    """Benchmark DOS calculation for different k-mesh sizes.

    Args:
        lattice: BravaisLattice for k-mesh generation
        tb_model: TightBindingModel for Hamiltonian construction
        nk_list: List of k-mesh sizes to test
        device: Device to run benchmark on
        n_trials: Number of trials for averaging

    Returns:
        List of average DOS calculation times for each k-mesh size
    """
    times = []

    for nk in nk_list:
        trial_times = []
        for _ in range(n_trials):
            k_mesh = generate_kmesh(lattice, nk=nk, device=device)
            Hk = tb_model.build_Hk(k_mesh)

            start = time.time()
            # Compute eigenvalues
            N_k = Hk.shape[0]
            n_orb = Hk.shape[-1]
            evals = torch.zeros(N_k, n_orb, dtype=torch.float64, device=device)
            for k in range(N_k):
                evals[k] = torch.linalg.eigvalsh(Hk.tensor[k]).real

            # Compute DOS
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
    """Benchmark Bayesian optimization.

    Args:
        device: Device to run benchmark on
        n_init: Number of initial samples
        n_iter: Number of optimization iterations
        n_trials: Number of trials for averaging

    Returns:
        Average optimization time
    """
    # Simple 2D test function (Rosenbrock)
    def objective(X):
        """Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2"""
        # Handle both single sample (1D) and batch (2D) inputs
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
            backend="simple",  # Use simple backend for consistency
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
    """Run all benchmarks on CPU and GPU.

    Args:
        device_only: If specified, only run on this device ('cpu' or 'cuda')

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # Determine which devices to test
    if device_only:
        devices_to_test = [torch.device(device_only)]
    else:
        devices_to_test = [torch.device("cpu")]
        if is_cuda_available():
            devices_to_test.append(torch.device("cuda"))

    # Build lattice and model
    lattice = build_kagome_lattice()
    tb_model = build_kagome_model(lattice)

    # Benchmark parameters
    nk_list = [20, 40, 60, 80, 100]  # k-mesh sizes for H(k) construction
    nk_diag_list = [100, 500, 1000, 2000]  # k-points for diagonalization
    nk_dos_list = [20, 40, 60, 80, 100]  # k-mesh sizes for DOS

    for device in devices_to_test:
        print(f"\n{'=' * 70}")
        print(f"Running benchmarks on {device}")
        print('=' * 70)

        # Move model to device
        tb_model_device = tb_model.to(device) if device.type == "cuda" else tb_model

        # 1. Hamiltonian construction
        print("\n1. Hamiltonian Construction")
        h_construction_times = benchmark_hamiltonian_construction(
            lattice, tb_model_device, nk_list, device
        )

        # 2. Diagonalization
        print("\n2. Diagonalization")
        diag_times = benchmark_diagonalization(nk_diag_list, 3, device)

        # 3. DOS calculation
        print("\n3. DOS Calculation")
        dos_times = benchmark_dos(lattice, tb_model_device, nk_dos_list, device)

        # 4. Bayesian optimization
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
    """Create 4-panel comparison plot of benchmark results.

    Args:
        results: Dictionary of benchmark results from run_benchmark()
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

    # Panel 4: Bayesian optimization (bar chart)
    ax = axes[1, 1]
    if 'cpu' in results and 'cuda' in results:
        devices_bar = ['CPU', 'GPU']
        times_bar = [results['cpu']['bayesian_opt'], results['cuda']['bayesian_opt']]
        colors_bar = [colors['cpu'], colors['cuda']]
        bars = ax.bar(devices_bar, times_bar, color=colors_bar, alpha=0.7,
                      edgecolor='black', linewidth=2)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Bayesian Optimization (10+30 iterations)', fontsize=13, fontweight='bold')

        # Add speedup annotation
        speedup = times_bar[0] / times_bar[1] if times_bar[1] > 0 else 0
        ax.text(0.5, max(times_bar) * 0.9, f'GPU Speedup: {speedup:.2f}x',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add value labels on bars
        for bar, val in zip(bars, times_bar):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=11)
    else:
        # Single device
        device = devices[0]
        time_val = results[device]['bayesian_opt']
        ax.bar([device.upper()], [time_val], color=colors.get(str(device), 'gray'),
               alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Bayesian Optimization (10+30 iterations)', fontsize=13, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('GPU vs CPU Performance Comparison', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Performance Benchmark for condmatTensor"
    )
    parser.add_argument(
        "--device-only",
        type=str,
        choices=["cpu", "cuda"],
        help="Run benchmark on specified device only"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("condmatTensor GPU Performance Benchmark")
    print("=" * 70)

    # Check CUDA availability
    if args.device_only == "cuda" and not is_cuda_available():
        print("Error: CUDA requested but not available.")
        print("Install PyTorch with CUDA support: https://pytorch.org/")
        return

    if not args.device_only and not is_cuda_available():
        print("Note: CUDA not available. Running CPU-only benchmarks.")

    # Run benchmarks
    results = run_benchmark(device_only=args.device_only)

    # Plot results
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)

        # Calculate speedups
        for key in ['h_construction', 'diagonalization', 'dos']:
            nk_list_cpu, times_cpu = results['cpu'][key]
            nk_list_cuda, times_cuda = results['cuda'][key]

            if len(times_cpu) == len(times_cuda):
                avg_speedup = np.mean([t_c / t_g for t_c, t_g in zip(times_cpu, times_cuda)])
                print(f"  {key}: {avg_speedup:.2f}x average speedup on GPU")

        bo_speedup = results['cpu']['bayesian_opt'] / results['cuda']['bayesian_opt']
        print(f"  bayesian_opt: {bo_speedup:.2f}x speedup on GPU")

    # Plot results
    plot_benchmark_results(results)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
