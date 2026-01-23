#!/usr/bin/env python3
"""
Bayesian Optimization for Tight-Binding Parameter Fitting

This example demonstrates using Bayesian optimization to fit tight-binding
model parameters to match a target band structure.

**Scenario:**
Given a target band structure generated with known hopping parameters,
use Bayesian optimization to recover those parameters.

**Features:**
- Generate target bands with known parameters
- Use Bayesian optimization to recover parameters
- Compare grid search vs Bayesian optimization
- Visualize optimization landscape and convergence

**Applications:**
- Model calibration from experimental data
- Effective model parameter extraction
- Automated tight-binding fitting
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

from condmatTensor.core import BaseTensor
from condmatTensor.lattice import BravaisLattice, TightBindingModel, generate_k_path
from condmatTensor.solvers import diagonalize
from condmatTensor.optimization import BayesianOptimizer


def build_kagome_lattice() -> BravaisLattice:
    """Build pure Kagome lattice."""
    import math
    sqrt3 = math.sqrt(3)
    a1 = torch.tensor([0.5, sqrt3 / 2], dtype=torch.float64)
    a2 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    cell_vectors = torch.stack([a1, a2])

    basis_positions = [
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.5, 0.0], dtype=torch.float64),
        torch.tensor([0.25, sqrt3 / 4], dtype=torch.float64),
    ]

    return BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=[1, 1, 1],
    )


def build_kagome_model(t: float) -> BaseTensor:
    """Build Kagome tight-binding model with hopping t."""
    lattice = build_kagome_lattice()
    k_path, _ = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=30)
    # Ensure k_path has the same dtype as the lattice (float64)
    k_path = k_path.to(dtype=torch.float64)

    tb_model = TightBindingModel(lattice)

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

    Hk = tb_model.build_Hk(k_path)
    return Hk


def compute_eigenvalues(Hk: BaseTensor) -> torch.Tensor:
    """Compute eigenvalues for all k-points."""
    evals = []
    for i in range(Hk.tensor.shape[0]):
        H_k = Hk.tensor[i]  # (n_orb, n_orb)
        eval = torch.linalg.eigvalsh(H_k)
        evals.append(eval)
    return torch.stack(evals)  # (n_k, n_bands)


class OptimizationTracker:
    """Track optimization history for analysis."""

    def __init__(self):
        self.X_history = []
        self.y_history = []
        self.best_history = []
        self.best_X_history = []

    def record(self, X: torch.Tensor, y: torch.Tensor, best_X: torch.Tensor, best_y: float):
        """Record optimization iteration."""
        self.X_history.append(X.clone().detach())
        self.y_history.append(y.clone().detach())
        self.best_X_history.append(best_X.clone().detach())
        self.best_history.append(best_y)


def band_structure_rmse(target_evals: torch.Tensor, trial_evals: torch.Tensor) -> float:
    """Compute RMSE between target and trial band structures."""
    # Align bands by sorting eigenvalues at each k-point
    diff = target_evals - trial_evals
    mse = torch.mean(diff ** 2).item()
    return np.sqrt(mse)


def objective_function(X: torch.Tensor, target_evals: torch.Tensor, lattice: BravaisLattice) -> torch.Tensor:
    """Objective function for Bayesian optimization.

    Args:
        X: Parameter tensor of shape (n_samples, n_params) or (n_params,)
        target_evals: Target eigenvalues of shape (n_k, n_bands)
        lattice: BravaisLattice for building models

    Returns:
        Loss tensor of shape (n_samples,) or (1,)
    """
    # Handle both 1D and 2D inputs
    if X.ndim == 1:
        X = X.unsqueeze(0)

    n_samples = X.shape[0]
    device = X.device  # Use the same device as input
    losses = []

    # Move target_evals to the same device as X for computation
    target_evals_device = target_evals.to(device)

    for i in range(n_samples):
        t = X[i, 0].item()

        # Build model with trial parameters
        Hk = build_kagome_model(t)
        trial_evals = compute_eigenvalues(Hk).to(device)

        # Compute RMSE
        rmse = band_structure_rmse(target_evals_device, trial_evals)
        losses.append(rmse)

    return torch.tensor(losses, dtype=torch.float64, device=device)


def grid_search(
    target_evals: torch.Tensor,
    lattice: BravaisLattice,
    t_range: Tuple[float, float] = (-1.5, -0.5),
    n_points: int = 20,
) -> Tuple[float, float, List]:
    """Perform grid search over hopping parameter.

    Args:
        target_evals: Target eigenvalues
        lattice: BravaisLattice
        t_range: Range of hopping parameter to search
        n_points: Number of grid points

    Returns:
        (best_t, best_rmse, rmse_values)
    """
    print("=" * 70)
    print("Grid Search")
    print("=" * 70)

    t_values = np.linspace(t_range[0], t_range[1], n_points)
    rmse_values = []

    for t in t_values:
        Hk = build_kagome_model(t)
        trial_evals = compute_eigenvalues(Hk)
        rmse = band_structure_rmse(target_evals, trial_evals)
        rmse_values.append(rmse)
        print(f"  t = {t:6.3f}, RMSE = {rmse:.6f}")

    best_idx = np.argmin(rmse_values)
    best_t = t_values[best_idx]
    best_rmse = rmse_values[best_idx]

    print(f"\nBest grid search result:")
    print(f"  t = {best_t:.6f}")
    print(f"  RMSE = {best_rmse:.6f}")

    return best_t, best_rmse, rmse_values


def bayesian_optimization_search(
    target_evals: torch.Tensor,
    lattice: BravaisLattice,
    bounds: List[Tuple[float, float]],
    backend: str = "auto",
    n_init: int = 10,
    n_iter: int = 30,
    seed: int = 42,
) -> Tuple[torch.Tensor, float, OptimizationTracker]:
    """Perform Bayesian optimization over hopping parameter.

    Args:
        target_evals: Target eigenvalues
        lattice: BravaisLattice
        bounds: Parameter bounds
        backend: Bayesian optimization backend
        n_init: Number of initial samples
        n_iter: Number of optimization iterations
        seed: Random seed

    Returns:
        (best_X, best_y, tracker)
    """
    print("=" * 70)
    print(f"Bayesian Optimization ({backend.upper()})")
    print("=" * 70)

    tracker = OptimizationTracker()

    # Create objective function closure
    def objective_fn(X):
        return objective_function(X, target_evals, lattice)

    # Create optimizer with n_init, n_iter, seed set during construction
    opt = BayesianOptimizer(bounds=bounds, backend=backend, n_init=n_init, n_iter=n_iter, seed=seed)

    # Run optimization with tracking
    # Note: We'll need to modify BayesianOptimizer to support callbacks
    # For now, just run directly
    # Use CPU device to match target_evals device
    from condmatTensor.core import get_device
    device = get_device("cpu")  # Force CPU to match target_evals
    X_best, y_best = opt.optimize(
        objective_fn,
        verbose=True,
        device=device,
    )

    print(f"\nBest Bayesian result:")
    # Handle both tensor and float return types for y_best
    y_val = y_best.item() if hasattr(y_best, 'item') else float(y_best)
    print(f"  t = {X_best[0].item():.6f}")
    print(f"  RMSE = {y_val:.6f}")

    return X_best, y_best, tracker


def plot_results(
    target_t: float,
    grid_t: float,
    grid_rmse: float,
    grid_rmse_values: List,
    bayes_t: torch.Tensor,
    bayes_rmse: float,  # Changed from torch.Tensor to float since y_best can be float
    t_range: Tuple[float, float],
    save_path: str = "bayesian_parameter_sweep_results.png",
):
    """Plot comparison of grid search and Bayesian optimization results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Loss landscape
    ax = axes[0]
    t_values = np.linspace(t_range[0], t_range[1], len(grid_rmse_values))
    ax.plot(t_values, grid_rmse_values, 'b-', linewidth=2, label='Grid Search')
    ax.axvline(target_t, color='g', linestyle='--', linewidth=2, label=f'Target (t={target_t:.3f})')
    ax.axvline(grid_t, color='r', linestyle='--', linewidth=2, label=f'Grid Best (t={grid_t:.3f})')
    ax.axvline(bayes_t.item(), color='orange', linestyle='--', linewidth=2,
               label=f'Bayesian Best (t={bayes_t.item():.3f})')
    ax.set_xlabel('Hopping Parameter t', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Parameter Fitting Loss Landscape', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Comparison bar chart
    ax = axes[1]
    methods = ['Grid Search', 'Bayesian Opt']
    errors = [grid_rmse, bayes_rmse]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Final RMSE', fontsize=12)
    ax.set_title('Final RMSE Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(errors) * 0.02,
                f'{val:.6f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")


def main():
    """Main function to run parameter fitting example."""
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for Tight-Binding Parameter Fitting"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "sober", "botorch", "simple"],
        help="Bayesian optimization backend (default: auto)"
    )
    parser.add_argument(
        "--target-t",
        type=float,
        default=-1.0,
        help="Target hopping parameter (default: -1.0)"
    )
    parser.add_argument(
        "--t-range",
        type=float,
        nargs=2,
        default=[-1.5, -0.5],
        help="Hopping parameter search range (default: -1.5 -0.5)"
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=20,
        help="Number of grid search points (default: 20)"
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of initial Bayesian samples (default: 10)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of Bayesian iterations (default: 30)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Bayesian Optimization for Tight-Binding Parameter Fitting")
    print("=" * 70)

    print(f"\nParameters:")
    print(f"  Target t = {args.target_t}")
    print(f"  Search range = [{args.t_range[0]}, {args.t_range[1]}]")
    print(f"  Grid points = {args.grid_points}")
    print(f"  Bayesian n_init = {args.n_init}")
    print(f"  Bayesian n_iter = {args.n_iter}")
    print(f"  Backend = {args.backend}")

    # Build lattice
    lattice = build_kagome_lattice()

    # Generate target band structure
    print("\n" + "=" * 70)
    print("Generating Target Band Structure")
    print("=" * 70)
    Hk_target = build_kagome_model(args.target_t)
    target_evals = compute_eigenvalues(Hk_target)
    print(f"  Target eigenvalues shape: {target_evals.shape}")
    print(f"  Flat band at E = {target_evals[0, 0].item():.3f} (expected -2.0)")
    print(f"  Top band at E = {target_evals[0, -1].item():.3f} (expected 4.0)")

    # Grid search
    grid_t, grid_rmse, grid_rmse_values = grid_search(
        target_evals, lattice,
        t_range=args.t_range,
        n_points=args.grid_points,
    )

    # Bayesian optimization
    bounds = [(args.t_range[0], args.t_range[1])]
    bayes_t, bayes_rmse, tracker = bayesian_optimization_search(
        target_evals, lattice,
        bounds=bounds,
        backend=args.backend,
        n_init=args.n_init,
        n_iter=args.n_iter,
        seed=42,
    )
    # Convert bayes_rmse to float if it's a tensor
    bayes_rmse_val = bayes_rmse.item() if hasattr(bayes_rmse, 'item') else float(bayes_rmse)

    # Plot results
    plot_results(
        target_t=args.target_t,
        grid_t=grid_t,
        grid_rmse=grid_rmse,
        grid_rmse_values=grid_rmse_values,
        bayes_t=bayes_t,
        bayes_rmse=bayes_rmse_val,
        t_range=args.t_range,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Target parameter:     t = {args.target_t:.6f}")
    print(f"Grid search result:   t = {grid_t:.6f}, error = {abs(grid_t - args.target_t):.6f}")
    print(f"Bayesian result:      t = {bayes_t.item():.6f}, error = {abs(bayes_t.item() - args.target_t):.6f}")
    print(f"\nGrid search RMSE:     {grid_rmse:.6f}")
    print(f"Bayesian RMSE:        {bayes_rmse_val:.6f}")

    # Verify results are reasonable
    tol = 0.05
    assert abs(grid_t - args.target_t) < tol, \
        f"Grid search failed to find correct parameter (error={abs(grid_t - args.target_t)})"
    assert abs(bayes_t.item() - args.target_t) < tol, \
        f"Bayesian optimization failed to find correct parameter (error={abs(bayes_t.item() - args.target_t)})"

    print("\n✅ Both methods found parameters within tolerance")
    print("=" * 70)


if __name__ == "__main__":
    main()
