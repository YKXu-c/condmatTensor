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

import argparse
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import copy

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, TightBindingModel, generate_k_path


class OptimizationTracker:
    """Track optimization history for analysis and convergence plotting."""

    def __init__(self):
        self.X_history: List[torch.Tensor] = []
        self.y_history: List[torch.Tensor] = []
        self.best_history: List[float] = []
        self.best_X_history: List[torch.Tensor] = []

    def record(self, X: torch.Tensor, y: torch.Tensor, best_X: torch.Tensor, best_y: float):
        """Record optimization iteration data.

        Args:
            X: All observed points
            y: All observed function values
            best_X: Best point found so far
            best_y: Best function value found so far
        """
        self.X_history.append(X.clone().detach())
        self.y_history.append(y.clone().detach())
        self.best_X_history.append(best_X.clone().detach())
        self.best_history.append(best_y)

    def get_best_trajectory(self) -> tuple[List[float], List[torch.Tensor]]:
        """Get the trajectory of best values and points."""
        return self.best_history, self.best_X_history


def plot_convergence(
    histories: Dict[str, dict],
    save_path: str = "kagome_f_convergence.png"
):
    """Plot convergence curves for different backends.

    Args:
        histories: Dictionary of backend histories with keys:
                   - 'best_history': List of best values per iteration
                   - 'backend_name': Name of the backend
        save_path: Path to save the figure
    """
    if not histories:
        print("No convergence data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for different backends
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    backend_names = list(histories.keys())

    for i, (backend_name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        best_history = history.get('best_history', [])

        if best_history:
            iterations = range(len(best_history))
            ax.plot(iterations, best_history, 'o-',
                   color=color, linewidth=2, markersize=4,
                   label=backend_name.upper())

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best RMSE', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often better for convergence

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved to {save_path}")


from condmatTensor.solvers import diagonalize
from condmatTensor.analysis import BandStructure, DOSCalculator
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


def compare_backends(
    optimizer: EffectiveArrayOptimizer,
    backends: list = ["auto", "botorch", "simple"],
    n_init: int = 10,
    n_iter: int = 30,
    seed: int = 42,
    trace_optimization: bool = False,
    device: str = "cpu",
) -> dict:
    """Run optimization with multiple backends and compare results.

    Args:
        optimizer: EffectiveArrayOptimizer instance to use for optimization
        backends: List of backend names to test
        n_init: Number of initial samples for Bayesian optimization
        n_iter: Number of optimization iterations
        seed: Random seed for reproducibility
        trace_optimization: If True, track iteration history for convergence plotting
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary containing results for each backend:
        {
            'backend_name': {
                'J_eff': float,
                'S_eff': np.ndarray,
                'time': float,
                'rmse': float,
                'mae': float,
                'correlation': float,
                'best_history': List[float],  # if trace_optimization=True
                'device': str,
            }
        }
    """
    results = {}

    # Move optimizer to target device
    original_device = optimizer.H_cc_0.tensor.device
    target_device = torch.device(device)

    # Move tensors to target device
    optimizer.H_cc_0 = optimizer.H_cc_0.to(target_device)
    optimizer.H_full = optimizer.H_full.to(target_device)

    print(f"\nDevice: {device.upper()}")
    if torch.cuda.is_available() and device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    for backend_name in backends:
        print(f"\n{'=' * 70}")
        print(f"Testing {backend_name.upper()} backend on {device.upper()}")
        print('=' * 70)

        # Reset optimizer for each run
        optimizer.reset()

        # Initialize tracker if requested
        tracker = OptimizationTracker() if trace_optimization else None

        try:
            # Time the optimization
            start_time = time.time()

            # Run optimization with specific backend
            J_eff, S_eff = optimizer.optimize(
                J_bounds=(0.01, 2.0),
                n_init=n_init,
                n_iter=n_iter,
                backend=backend_name,
                verbose=True,
                device=target_device,
            )

            elapsed_time = time.time() - start_time

            # Get verification metrics
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

            # Add best_history if tracking
            if trace_optimization and tracker is not None:
                results[backend_name]['best_history'] = tracker.best_history

            print(f"\n{backend_name.upper()} results:")
            print(f"  J_eff = {J_eff:.6f}")
            print(f"  S_eff = [{S_eff[0]:.6f}, {S_eff[1]:.6f}, {S_eff[2]:.6f}]")
            print(f"  Time = {elapsed_time:.2f}s")
            print(f"  RMSE = {metrics['rmse']:.6f}")
            print(f"  MAE = {metrics['mean_absolute_error']:.6f}")
            print(f"  Correlation = {metrics['correlation']:.6f}")

        except ImportError as e:
            print(f"\n{backend_name.upper()} backend not available: {e}")
            print(f"  Skipping...")
        except (ValueError, RuntimeError) as e:
            # Handle SOBER device compatibility issues
            if ("Automatic CPU parallelization" in str(e) or
                "GPU computations" in str(e) or
                "parallelization" in str(e)):
                print(f"\n{backend_name.upper()} backend has known device compatibility issues")
                print(f"  Error: {str(e)[:100]}...")
                print(f"  This is a known limitation of the SOBER library on certain systems")
                print(f"  Skipping...")
            else:
                import traceback
                print(f"\n{backend_name.upper()} backend failed: {e}")
                traceback.print_exc()
                print(f"  Skipping...")
        except Exception as e:
            import traceback
            print(f"\n{backend_name.upper()} backend failed: {e}")
            traceback.print_exc()
            print(f"  Skipping...")

    # Move tensors back to original device
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
    """Compare all backends on both CPU and GPU.

    Args:
        optimizer: EffectiveArrayOptimizer instance to use for optimization
        backends: List of backend names to test (default: simple, botorch; sober excluded due to device compatibility)
        n_init: Number of initial samples for Bayesian optimization
        n_iter: Number of optimization iterations
        seed: Random seed for reproducibility

    Returns:
        Nested dictionary containing results for each backend and device:
        {
            'cpu': {'backend_name': {...}, ...},
            'gpu': {'backend_name': {...}, ...},
        }
    """
    """Compare all backends on both CPU and GPU.

    Args:
        optimizer: EffectiveArrayOptimizer instance to use for optimization
        backends: List of backend names to test
        n_init: Number of initial samples for Bayesian optimization
        n_iter: Number of optimization iterations
        seed: Random seed for reproducibility

    Returns:
        Nested dictionary containing results for each backend and device:
        {
            'cpu': {'backend_name': {...}, ...},
            'gpu': {'backend_name': {...}, ...},
        }
    """
    all_results = {}

    # CPU comparison
    print("\n" + "=" * 70)
    print("CPU BENCHMARK")
    print("=" * 70)
    all_results['cpu'] = compare_backends(
        optimizer,
        backends=backends,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
        trace_optimization=False,
        device="cpu",
    )

    # GPU comparison (if available)
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU BENCHMARK")
        print("=" * 70)
        all_results['gpu'] = compare_backends(
            optimizer,
            backends=backends,
            n_init=n_init,
            n_iter=n_iter,
            seed=seed,
            trace_optimization=False,
            device="cuda",
        )
    else:
        print("\n" + "=" * 70)
        print("GPU NOT AVAILABLE")
        print("=" * 70)
        all_results['gpu'] = {}

    return all_results


def plot_backend_comparison(results: dict, save_path: str = "kagome_f_backend_comparison.png"):
    """Create 4-panel comparison plot for different backends.

    Args:
        results: Dictionary of backend results from compare_backends()
        save_path: Path to save the figure
    """
    if not results:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    backend_names = list(results.keys())
    n_backends = len(backend_names)

    # Colors for different backends
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    backend_colors = [colors[i % len(colors)] for i in range(n_backends)]

    # Panel 1: J_eff comparison (bar chart)
    ax = axes[0, 0]
    J_values = [results[b]['J_eff'] for b in backend_names]
    bars = ax.bar(range(n_backends), J_values, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('$J_{eff}$', fontsize=12)
    ax.set_title('Effective Exchange Parameter', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, J_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(J_values) * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Panel 2: Optimization time (bar chart)
    ax = axes[0, 1]
    times = [results[b]['time'] for b in backend_names]
    bars = ax.bar(range(n_backends), times, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Optimization Time', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times) * 0.05,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)

    # Panel 3: RMSE comparison (bar chart)
    ax = axes[1, 0]
    rmses = [results[b]['rmse'] for b in backend_names]
    bars = ax.bar(range(n_backends), rmses, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, rmses)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmses) * 0.05,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Panel 4: Correlation comparison (bar chart)
    ax = axes[1, 1]
    correlations = [results[b]['correlation'] for b in backend_names]
    bars = ax.bar(range(n_backends), correlations, color=backend_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in backend_names], fontsize=11)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Eigenvalue Correlation', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, correlations)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Bayesian Optimization Backend Comparison', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBackend comparison plot saved to {save_path}")


def plot_cpu_gpu_comparison(all_results: dict, save_path: str = "kagome_f_cpu_gpu_comparison.png"):
    """Create comprehensive CPU vs GPU comparison plot.

    Args:
        all_results: Nested dictionary with 'cpu' and 'gpu' keys
        save_path: Path to save the figure
    """
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        print("No CPU/GPU results to plot.")
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})

    # Get all backends that have results
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())
    if not all_backends:
        return

    # Create figure with 2x3 layout: 3 backends x 2 metrics
    n_backends = min(len(all_backends), 3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    backend_list = sorted(list(all_backends))[:n_backends]
    colors = {'cpu': '#3498db', 'gpu': '#e74c3c'}

    for idx, backend_name in enumerate(backend_list):
        # Time comparison
        ax = axes[0, idx]
        devices = []
        times = []
        if backend_name in cpu_results:
            devices.append('CPU')
            times.append(cpu_results[backend_name]['time'])
        if backend_name in gpu_results:
            devices.append('GPU')
            times.append(gpu_results[backend_name]['time'])

        if devices:
            bars = ax.bar(devices, times, color=[colors[d.lower()] for d in devices],
                         alpha=0.7, edgecolor='black')
            ax.set_ylabel('Time (s)', fontsize=11)
            ax.set_title(f'{backend_name.upper()}: Time', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            for i, (bar, val) in enumerate(zip(bars, times)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times) * 0.05,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=10)

        # RMSE comparison
        ax = axes[1, idx]
        devices = []
        rmses = []
        if backend_name in cpu_results:
            devices.append('CPU')
            rmses.append(cpu_results[backend_name]['rmse'])
        if backend_name in gpu_results:
            devices.append('GPU')
            rmses.append(gpu_results[backend_name]['rmse'])

        if devices:
            bars = ax.bar(devices, rmses, color=[colors[d.lower()] for d in devices],
                         alpha=0.7, edgecolor='black')
            ax.set_ylabel('RMSE', fontsize=11)
            ax.set_title(f'{backend_name.upper()}: RMSE', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            for i, (bar, val) in enumerate(zip(bars, rmses)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmses) * 0.05,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('CPU vs GPU Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCPU/GPU comparison plot saved to {save_path}")


def plot_band_dos_comparison(
    all_results: dict,
    optimizer: EffectiveArrayOptimizer,
    lattice: BravaisLattice,
    k_path: torch.Tensor,
    save_path: str = "kagome_f_band_dos_comparison.png"
):
    """Plot band structures and DOS for all 6 results (3 backends × 2 devices).

    Args:
        all_results: Nested dictionary with 'cpu' and 'gpu' keys
        optimizer: EffectiveArrayOptimizer instance
        lattice: BravaisLattice for the system
        k_path: k-path points for band structure
        save_path: Path to save the figure
    """
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        print("No results for band/DOS plotting.")
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))

    # Get reference eigenvalues from full system
    eig_full = optimizer._compute_eigenvalues(optimizer.H_full)

    # Create figure: 3 backends × 2 rows (band + DOS)
    fig, axes = plt.subplots(len(backend_list), 2, figsize=(16, 5 * len(backend_list)))
    if len(backend_list) == 1:
        axes = axes.reshape(1, -1)

    k_axis = torch.arange(eig_full.shape[0], dtype=torch.float64).numpy()

    for idx, backend_name in enumerate(backend_list):
        # Colors
        full_color = '#2c3e50'
        cpu_color = '#3498db'
        gpu_color = '#e74c3c'

        # ===== Band structure plot =====
        ax_band = axes[idx, 0]

        # Plot full system (reference)
        for i in range(eig_full.shape[1]):
            ax_band.plot(k_axis, eig_full[:, i].cpu().numpy(),
                        color=full_color, alpha=0.3, linewidth=1)

        # Plot CPU result if available
        if backend_name in cpu_results:
            res_cpu = cpu_results[backend_name]
            J_cpu = res_cpu['J_eff']
            S_cpu = torch.tensor(res_cpu['S_eff'], dtype=torch.float64)
            H_eff_cpu = optimizer._build_effective_hamiltonian(J_cpu, S_cpu, torch.device('cpu'))
            eig_eff_cpu = optimizer._compute_eigenvalues(H_eff_cpu)

            for i in range(eig_eff_cpu.shape[1]):
                ax_band.plot(k_axis, eig_eff_cpu[:, i].cpu().numpy(),
                            color=cpu_color, alpha=0.7, linewidth=1.5, linestyle='-')

        # Plot GPU result if available
        if backend_name in gpu_results:
            res_gpu = gpu_results[backend_name]
            J_gpu = res_gpu['J_eff']
            S_gpu = torch.tensor(res_gpu['S_eff'], dtype=torch.float64)
            H_eff_gpu = optimizer._build_effective_hamiltonian(J_gpu, S_gpu, torch.device('cpu'))
            eig_eff_gpu = optimizer._compute_eigenvalues(H_eff_gpu)

            for i in range(eig_eff_gpu.shape[1]):
                ax_band.plot(k_axis, eig_eff_gpu[:, i].cpu().numpy(),
                            color=gpu_color, alpha=0.7, linewidth=1.5, linestyle='--')

        ax_band.set_ylabel("Energy ($|t|$)", fontsize=11)
        ax_band.set_title(f'{backend_name.upper()}: Band Structure Comparison', fontsize=12, fontweight='bold')
        ax_band.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=full_color, alpha=0.5, label='Full System'),
            Line2D([0], [0], color=cpu_color, alpha=0.8, label='CPU Result'),
            Line2D([0], [0], color=gpu_color, alpha=0.8, linestyle='--', label='GPU Result'),
        ]
        ax_band.legend(handles=legend_elements, fontsize=9, loc='upper right')

        # ===== DOS plot =====
        ax_dos = axes[idx, 1]

        # Create energy grid for DOS
        e_min = eig_full.min().item() - 1.0
        e_max = eig_full.max().item() + 1.0
        omega = torch.linspace(e_min, e_max, 200, dtype=torch.float64)

        # Compute DOS for full system
        dos_calc = DOSCalculator()
        omega_full, dos_full_vals = dos_calc.from_eigenvalues(eig_full, omega, eta=0.05)

        # Plot full system DOS
        ax_dos.plot(omega_full.cpu().numpy(), dos_full_vals.cpu().numpy(),
                   color=full_color, alpha=0.5, linewidth=2, label='Full System')

        # Plot CPU DOS if available
        if backend_name in cpu_results:
            res_cpu = cpu_results[backend_name]
            J_cpu = res_cpu['J_eff']
            S_cpu = torch.tensor(res_cpu['S_eff'], dtype=torch.float64)
            H_eff_cpu = optimizer._build_effective_hamiltonian(J_cpu, S_cpu, torch.device('cpu'))
            eig_eff_cpu = optimizer._compute_eigenvalues(H_eff_cpu)
            omega_cpu, dos_cpu_vals = dos_calc.from_eigenvalues(eig_eff_cpu, omega, eta=0.05)

            ax_dos.plot(omega_cpu.cpu().numpy(), dos_cpu_vals.cpu().numpy(),
                       color=cpu_color, alpha=0.8, linewidth=2, label='CPU Result')

        # Plot GPU DOS if available
        if backend_name in gpu_results:
            res_gpu = gpu_results[backend_name]
            J_gpu = res_gpu['J_eff']
            S_gpu = torch.tensor(res_gpu['S_eff'], dtype=torch.float64)
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

    plt.suptitle('Band Structure & DOS Comparison (3 Backends × 2 Devices)',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBand/DOS comparison plot saved to {save_path}")


def plot_accuracy_speedup_heatmap(all_results: dict, save_path: str = "kagome_f_accuracy_speedup_heatmap.png"):
    """Create heatmap showing accuracy-speedup tradeoff for all backends/devices.

    Args:
        all_results: Nested dictionary with 'cpu' and 'gpu' keys
        save_path: Path to save the figure
    """
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        print("No results for heatmap.")
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ===== Plot 1: Accuracy (RMSE) comparison =====
    ax1 = axes[0]

    # Prepare data
    data_rmse = []
    row_labels = []

    for backend in backend_list:
        row_labels.append(backend.upper())
        row_data = []
        # CPU RMSE
        if backend in cpu_results:
            rmse_cpu = cpu_results[backend]['rmse']
            # Normalize: lower is better, show as percentage of max
            row_data.append(rmse_cpu)
        else:
            row_data.append(np.nan)
        # GPU RMSE
        if backend in gpu_results:
            rmse_gpu = gpu_results[backend]['rmse']
            row_data.append(rmse_gpu)
        else:
            row_data.append(np.nan)
        data_rmse.append(row_data)

    data_rmse = np.array(data_rmse)
    col_labels = ['CPU', 'GPU']  # Set based on actual columns

    # Create heatmap
    im1 = ax1.imshow(data_rmse, cmap='RdYlGn_r', aspect='auto')

    # Set ticks
    ax1.set_xticks(np.arange(data_rmse.shape[1]))
    ax1.set_yticks(np.arange(len(row_labels)))
    ax1.set_xticklabels(col_labels[:data_rmse.shape[1]], fontsize=12)
    ax1.set_yticklabels(row_labels, fontsize=12)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(data_rmse.shape[1]):
            if not np.isnan(data_rmse[i, j]):
                text = ax1.text(j, i, f'{data_rmse[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=11, fontweight='bold')
            else:
                text = ax1.text(j, i, 'N/A',
                              ha="center", va="center", color="gray", fontsize=10)

    ax1.set_title('(a) RMSE (lower is better)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Device', fontsize=12)
    ax1.set_ylabel('Backend', fontsize=12)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('RMSE', fontsize=11)

    # ===== Plot 2: Speedup heatmap =====
    ax2 = axes[1]

    # Calculate speedups relative to CPU
    data_speedup = []
    row_labels_speedup = []
    for backend in backend_list:
        if backend in cpu_results and backend in gpu_results:
            row_labels_speedup.append(backend.upper())
            cpu_time = cpu_results[backend]['time']
            gpu_time = gpu_results[backend]['time']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            data_speedup.append([speedup])

    # If no valid speedup data, show placeholder
    if not data_speedup:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No GPU data available\nfor speedup calculation',
                ha='center', va='center', fontsize=12, style='italic', color='gray')
    else:
        data_speedup = np.array(data_speedup)
        col_labels_speedup = ['GPU Speedup']

        # Create heatmap for speedup
        cmap_speedup = plt.cm.RdYlGn
        im2 = ax2.imshow(data_speedup, cmap=cmap_speedup, aspect='auto',
                         vmin=max(0.5, data_speedup.min()), vmax=max(1.5, data_speedup.max()))

        # Set ticks
        ax2.set_xticks(np.arange(data_speedup.shape[1]))
        ax2.set_yticks(np.arange(len(row_labels_speedup)))
        ax2.set_xticklabels(col_labels_speedup, fontsize=12)
        ax2.set_yticklabels(row_labels_speedup, fontsize=12)

    # Add text annotations
    for i in range(len(row_labels_speedup)):
        for j in range(data_speedup.shape[1]):
            if not np.isnan(data_speedup[i, j]):
                text = ax2.text(j, i, f'{data_speedup[i, j]:.2f}x',
                              ha="center", va="center", color="black", fontsize=11, fontweight='bold')
            else:
                text = ax2.text(j, i, 'N/A',
                              ha="center", va="center", color="gray", fontsize=10)

    ax2.set_title('(b) GPU Speedup (CPU time / GPU time)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Backend', fontsize=12)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Speedup (x)', fontsize=11)

    plt.suptitle('Accuracy vs Speedup Tradeoff Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nAccuracy-speedup heatmap saved to {save_path}")


def print_detailed_results_table(all_results: dict):
    """Print detailed results table for all 6 results.

    Args:
        all_results: Nested dictionary with 'cpu' and 'gpu' keys
    """
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        print("No results to display.")
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())

    if not all_backends:
        return

    backend_list = sorted(list(all_backends))

    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)

    # Header
    print(f"{'Backend':<12} {'Device':<8} {'J_eff':<12} {'Time (s)':<12} {'RMSE':<12} {'MAE':<12} {'Correlation':<14}")
    print("-" * 100)

    # Rows
    for backend in backend_list:
        # CPU row
        if backend in cpu_results:
            res = cpu_results[backend]
            print(f"{backend.upper():<12} {'CPU':<8} {res['J_eff']:<12.6f} {res['time']:<12.2f} "
                  f"{res['rmse']:<12.6f} {res['mae']:<12.6f} {res['correlation']:<14.6f}")

        # GPU row
        if backend in gpu_results:
            res = gpu_results[backend]
            print(f"{backend.upper():<12} {'GPU':<8} {res['J_eff']:<12.6f} {res['time']:<12.2f} "
                  f"{res['rmse']:<12.6f} {res['mae']:<12.6f} {res['correlation']:<14.6f}")

    print("=" * 100)

    # Speedup summary
    if cpu_results and gpu_results:
        print("\nGPU SPEEDUP SUMMARY")
        print("-" * 50)
        print(f"{'Backend':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<12}")
        print("-" * 50)

        for backend in backend_list:
            if backend in cpu_results and backend in gpu_results:
                cpu_time = cpu_results[backend]['time']
                gpu_time = gpu_results[backend]['time']
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"{backend.upper():<12} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<12.2f}x")

        print("-" * 50)

    # Accuracy ranking
    print("\nACCURACY RANKING (lower RMSE is better)")
    print("-" * 60)

    all_results_list = []
    for backend in backend_list:
        if backend in cpu_results:
            all_results_list.append(('CPU', backend, cpu_results[backend]['rmse'],
                                    cpu_results[backend]['correlation']))
        if backend in gpu_results:
            all_results_list.append(('GPU', backend, gpu_results[backend]['rmse'],
                                    gpu_results[backend]['correlation']))

    # Sort by RMSE (lower is better), then by correlation (higher is better)
    all_results_list.sort(key=lambda x: (x[2], -x[3]))

    for rank, (device, backend, rmse, corr) in enumerate(all_results_list, 1):
        print(f"  {rank}. {backend.upper()} ({device.upper()}): "
              f"RMSE={rmse:.6f}, Correlation={corr:.6f}")

    print("-" * 60)

    # Performance ranking (lower time is better)
    print("\nPERFORMANCE RANKING (lower time is better)")
    print("-" * 50)

    # Sort by time (lower is better)
    time_ranking = [(device, backend, res['time'])
                   for device, results in [('CPU', cpu_results), ('GPU', gpu_results)]
                   for backend, res in results.items()]
    time_ranking.sort(key=lambda x: x[2])

    for rank, (device, backend, time) in enumerate(time_ranking, 1):
        print(f"  {rank}. {backend.upper()} ({device.upper()}): {time:.2f}s")

    print("-" * 50)
    print("=" * 100)


def plot_comprehensive_summary(all_results: dict, save_path: str = "kagome_f_comprehensive_summary.png"):
    """Create comprehensive summary plot with all metrics and devices.

    Args:
        all_results: Nested dictionary with 'cpu' and 'gpu' keys
        save_path: Path to save the figure
    """
    if not all_results or (not all_results.get('cpu') and not all_results.get('gpu')):
        print("No results to plot.")
        return

    cpu_results = all_results.get('cpu', {})
    gpu_results = all_results.get('gpu', {})

    # Get all backends that have results
    all_backends = set(cpu_results.keys()) | set(gpu_results.keys())
    if not all_backends:
        return

    backend_list = sorted(list(all_backends))
    n_backends = len(backend_list)

    # Create 3x3 layout for comprehensive comparison
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Colors
    backend_colors_map = {
        'simple': '#3498db',
        'botorch': '#e74c3c',
        'sober': '#2ecc71',
    }
    device_colors = {'cpu': '#3498db', 'gpu': '#e74c3c'}

    # Organize data for plotting
    x = np.arange(n_backends)
    width = 0.35

    # Panel 1: Time comparison (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    cpu_times = [cpu_results.get(b, {}).get('time', 0) for b in backend_list]
    gpu_times = [gpu_results.get(b, {}).get('time', 0) for b in backend_list]

    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_times):
        bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('(a) Optimization Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([b.upper() for b in backend_list], fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, b in enumerate(backend_list):
        if b in cpu_results:
            y = cpu_times[i]
            ax1.text(i - width/2, y + max(cpu_times) * 0.02, f'{y:.2f}s',
                   ha='center', va='bottom', fontsize=9, rotation=0)
        if b in gpu_results:
            y = gpu_times[i]
            if y > 0:
                ax1.text(i + width/2, y + max([y for y in gpu_times if y > 0]) * 0.02, f'{y:.2f}s',
                       ha='center', va='bottom', fontsize=9, rotation=0)

    # Panel 2: RMSE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    cpu_rmse = [cpu_results.get(b, {}).get('rmse', 0) for b in backend_list]
    gpu_rmse = [gpu_results.get(b, {}).get('rmse', 0) for b in backend_list]

    bars1 = ax2.bar(x - width/2, cpu_rmse, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_rmse):
        bars2 = ax2.bar(x + width/2, gpu_rmse, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('(b) Accuracy (RMSE)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.upper() for b in backend_list], fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, b in enumerate(backend_list):
        if b in cpu_results:
            y = cpu_rmse[i]
            ax2.text(i - width/2, y + max(cpu_rmse) * 0.02, f'{y:.4f}',
                   ha='center', va='bottom', fontsize=9)
        if b in gpu_results and gpu_rmse[i] > 0:
            y = gpu_rmse[i]
            ax2.text(i + width/2, y + max([y for y in gpu_rmse if y > 0]) * 0.02, f'{y:.4f}',
                   ha='center', va='bottom', fontsize=9)

    # Panel 3: Correlation comparison
    ax3 = fig.add_subplot(gs[0, 2])
    cpu_corr = [cpu_results.get(b, {}).get('correlation', 0) for b in backend_list]
    gpu_corr = [gpu_results.get(b, {}).get('correlation', 0) for b in backend_list]

    bars1 = ax3.bar(x - width/2, cpu_corr, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_corr):
        bars2 = ax3.bar(x + width/2, gpu_corr, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax3.set_ylabel('Correlation', fontsize=12)
    ax3.set_title('(c) Eigenvalue Correlation', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.set_xticks(x)
    ax3.set_xticklabels([b.upper() for b in backend_list], fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: J_eff comparison
    ax4 = fig.add_subplot(gs[1, 0])
    cpu_J = [cpu_results.get(b, {}).get('J_eff', 0) for b in backend_list]
    gpu_J = [gpu_results.get(b, {}).get('J_eff', 0) for b in backend_list]

    bars1 = ax4.bar(x - width/2, cpu_J, width, label='CPU', color=device_colors['cpu'], alpha=0.7)
    if any(gpu_J):
        bars2 = ax4.bar(x + width/2, gpu_J, width, label='GPU', color=device_colors['gpu'], alpha=0.7)

    ax4.set_ylabel('$J_{eff}$', fontsize=12)
    ax4.set_title('(d) Effective Exchange Parameter', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([b.upper() for b in backend_list], fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    # Panel 5: GPU Speedup
    ax5 = fig.add_subplot(gs[1, 1])
    speedups = []
    backend_names_speedup = []

    for b in backend_list:
        if b in cpu_results and b in gpu_results:
            cpu_time = cpu_results[b]['time']
            gpu_time = gpu_results[b]['time']
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                speedups.append(speedup)
                backend_names_speedup.append(b.upper())

    if speedups:
        colors_speedup = [backend_colors_map.get(b.lower(), '#95a5a6') for b in backend_names_speedup]
        bars = ax5.bar(range(len(speedups)), speedups, color=colors_speedup, alpha=0.7, edgecolor='black')
        ax5.set_xticks(range(len(speedups)))
        ax5.set_xticklabels(backend_names_speedup, fontsize=11)
        ax5.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
        ax5.set_title('(e) GPU Speedup', fontsize=13, fontweight='bold')
        ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
        ax5.legend(fontsize=9)
        ax5.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, speedups)):
            label = f'{val:.2f}x' if val >= 1 else f'{val:.2f}x'
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups) * 0.02,
                    label, ha='center', va='bottom', fontsize=10)

    # Panel 6: Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_data = []
    for b in backend_list:
        row = [
            b.upper(),
            f"{cpu_results.get(b, {}).get('time', 0):.2f}s",
            f"{gpu_results.get(b, {}).get('time', 0):.2f}s" if gpu_results.get(b, {}) else "N/A",
            f"{cpu_results.get(b, {}).get('rmse', 0):.4f}",
            f"{gpu_results.get(b, {}).get('rmse', 0):.4f}" if gpu_results.get(b, {}) else "N/A",
            f"{cpu_results.get(b, {}).get('correlation', 0):.3f}",
        ]
        summary_data.append(row)

    table = ax6.table(cellText=summary_data,
                      colLabels=['Backend', 'CPU Time', 'GPU Time', 'CPU RMSE', 'GPU RMSE', 'CPU Corr'],
                      cellLoc='center',
                      bbox=[0, 0, 1, 1],
                      colWidths=[0.12, 0.14, 0.14, 0.14, 0.14, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    ax6.set_title('(f) Results Summary', fontsize=13, fontweight='bold', pad=20)

    # Panel 7: Ranking chart (bottom row, spans all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Calculate scores for ranking (lower RMSE is better, higher correlation is better)
    backend_scores = []
    for b in backend_list:
        # Combined score: normalize RMSE and correlation
        rmse_cpu = cpu_results.get(b, {}).get('rmse', 1.0)
        corr_cpu = cpu_results.get(b, {}).get('correlation', 0.0)
        score = (1.0 - rmse_cpu) + corr_cpu  # Higher is better
        backend_scores.append((score, b.upper(), 'CPU'))

        if b in gpu_results:
            rmse_gpu = gpu_results[b].get('rmse', 1.0)
            corr_gpu = gpu_results[b].get('correlation', 0.0)
            score = (1.0 - rmse_gpu) + corr_gpu
            backend_scores.append((score, b.upper(), 'GPU'))

    # Sort by score
    backend_scores.sort(reverse=True)

    # Display ranking
    y_pos = 0.8
    ax7.text(0.5, 1.0, 'Overall Ranking (by accuracy)',
            ha='center', fontsize=14, fontweight='bold')

    for idx, (score, name, device) in enumerate(backend_scores[:6]):  # Top 6
        color = device_colors[device.lower()]
        ax7.text(0.1, y_pos, f"{idx + 1}. {name} ({device.upper()}): {score:.3f}",
                fontsize=12, fontweight='bold', color=color)
        y_pos -= 0.12

    # Add note about SOBER
    if 'sober' not in all_backends:
        ax7.text(0.5, 0.3, '* SOBER backend has known device compatibility issues on this system',
                ha='center', fontsize=10, style='italic', color='gray')

    plt.suptitle('Comprehensive Bayesian Optimization Comparison\n(3 Backends × 2 Devices = 6 Results)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComprehensive summary plot saved to {save_path}")


def main():
    """Main function to run effective array optimization."""
    parser = argparse.ArgumentParser(
        description="Effective Array Optimizer for Kagome-F Lattice"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "sober", "botorch", "simple"],
        help="Bayesian optimization backend (default: auto)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available backends instead of single optimization"
    )
    parser.add_argument(
        "--cpu-gpu",
        action="store_true",
        help="Compare all backends on both CPU and GPU"
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of initial samples for Bayesian optimization (default: 10)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of optimization iterations (default: 30)"
    )
    parser.add_argument(
        "--trace-optimization",
        action="store_true",
        help="Enable optimization tracing for convergence plots"
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=None,
        help="Convergence tolerance for early stopping (default: None)"
    )

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

    # Build full Kagome-F Hamiltonian
    lattice_full, H_full = build_kagome_f_system(t, t_f, epsilon_f)

    print(f"\nHamiltonian dimensions:")
    print(f"  H_cc_0: {H_cc_0.shape}")
    print(f"  H_full: {H_full.shape}")

    # Build spinful versions
    model = LocalMagneticModel()
    H_cc_0_spinful = model.build_spinful_hamiltonian(H_cc_0)

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
        num_orbitals=[1, 1, 1, 1],
    )
    tb_full = TightBindingModel(lattice_full_spinless, orbital_labels=orbital_labels_full)
    # Kagome-Kagome hopping
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

    if args.compare and args.cpu_gpu:
        # Comprehensive CPU vs GPU comparison
        print("\nRunning comprehensive CPU vs GPU comparison...")
        print("This will test all backends on both CPU and GPU (if available)...")
        print("Please wait, this may take several minutes...\n")

        all_results = compare_backends_cpu_gpu(
            optimizer,
            backends=["simple", "botorch", "sober"],
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=42,
        )

        # Plot comprehensive summary
        if all_results and (all_results.get('cpu') or all_results.get('gpu')):
            plot_comprehensive_summary(all_results)
            plot_cpu_gpu_comparison(all_results)
            plot_accuracy_speedup_heatmap(all_results)

            # Plot band structures and DOS for all results
            k_path, _ = generate_k_path(lattice_cc, ["G", "K", "M", "G"], n_per_segment=30)
            plot_band_dos_comparison(all_results, optimizer, lattice_cc, k_path)

        # Print detailed results table
        print_detailed_results_table(all_results)

        # Validation
        print("\n" + "=" * 70)
        print("Validation Checks")
        print("=" * 70)

        for device in ['cpu', 'gpu']:
            device_results = all_results.get(device, {})
            if device_results:
                for backend_name, res in device_results.items():
                    assert res['rmse'] < 1.0, f"{device}/{backend_name}: RMSE should be < 1.0"
                    assert res['correlation'] > 0.9, f"{device}/{backend_name}: Correlation should be > 0.9"

        print("\n  All validation checks passed!")

    elif args.compare:
        # Compare all backends on CPU
        print("\nRunning backend comparison on CPU...")
        results = compare_backends(
            optimizer,
            backends=["simple", "botorch", "sober"],
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=42,
            trace_optimization=args.trace_optimization,
            device="cpu",
        )

        # Plot comparison
        if results:
            plot_backend_comparison(results)

            # Plot convergence if tracking was enabled
            if args.trace_optimization:
                histories = {
                    name: {'best_history': res.get('best_history', [])}
                    for name, res in results.items()
                    if res.get('best_history')
                }
                if histories:
                    plot_convergence(histories)

        print("\n" + "=" * 70)
        print("Backend comparison complete!")
        print("=" * 70)

        # Add assertions for validation
        print("\n" + "=" * 70)
        print("Validation Checks")
        print("=" * 70)

        for backend_name, res in results.items():
            assert res['rmse'] < 1.0, f"{backend_name}: RMSE should be < 1.0, got {res['rmse']}"
            assert res['correlation'] > 0.9, f"{backend_name}: Correlation should be > 0.9, got {res['correlation']}"
            assert 0.0 < res['J_eff'] < 2.0, f"{backend_name}: J_eff should be in (0, 2), got {res['J_eff']}"
            print(f"  {backend_name.upper()}: All validation checks passed")

        print("\n  All backend results are within expected ranges!")

    elif args.cpu_gpu:
        # CPU vs GPU comparison for single backend
        backend_name = args.backend

        # If using "auto" backend, switch to "botorch" for CPU+GPU comparison
        # to avoid SOBER's known device compatibility issues
        if backend_name == "auto":
            backend_name = "botorch"
            print(f"\nNote: Using BOTARCH backend instead of AUTO for CPU+GPU comparison")
            print(f"      (to avoid SOBER's known device compatibility issues)")

        print(f"\nRunning {backend_name.upper()} backend on both CPU and GPU...")

        backends_to_test = [backend_name]
        all_results = compare_backends_cpu_gpu(
            optimizer,
            backends=backends_to_test,
            n_init=args.n_init,
            n_iter=args.n_iter,
            seed=42,
        )

        # Plot results
        if all_results:
            plot_cpu_gpu_comparison(all_results)

        # Print summary
        print("\n" + "=" * 70)
        print(f"{backend_name.upper()} Backend: CPU vs GPU Summary")
        print("=" * 70)

        for device in ['cpu', 'gpu']:
            device_results = all_results.get(device, {})
            if device_results and backend_name in device_results:
                res = device_results[backend_name]
                print(f"\n{device.upper()}:")
                print(f"  J_eff = {res['J_eff']:.6f}")
                print(f"  Time = {res['time']:.2f}s")
                print(f"  RMSE = {res['rmse']:.6f}")
                print(f"  Correlation = {res['correlation']:.6f}")

        # Calculate speedup
        if (all_results.get('cpu', {}).get(backend_name) and
            all_results.get('gpu', {}).get(backend_name)):
            cpu_time = all_results['cpu'][backend_name]['time']
            gpu_time = all_results['gpu'][backend_name]['time']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"\n  GPU Speedup: {speedup:.2f}x")

    else:
        # Single optimization with specified backend
        print(f"\nRunning optimization with {args.backend.upper()} backend...")

        J_eff, S_eff = optimizer.optimize(
            J_bounds=(0.01, 2.0),
            n_init=args.n_init,
            n_iter=args.n_iter,
            backend=args.backend,
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

        # Validation assertions
        print("\n" + "=" * 70)
        print("Validation Checks")
        print("=" * 70)

        assert metrics['rmse'] < 1.0, f"RMSE should be < 1.0, got {metrics['rmse']}"
        assert metrics['correlation'] > 0.9, f"Correlation should be > 0.9, got {metrics['correlation']}"
        assert 0.0 < J_eff < 2.0, f"J_eff should be in (0, 2), got {J_eff}"
        print("  All validation checks passed!")

        print("\n" + "=" * 70)
        print("Example complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
