"""SOBER backend for Bayesian optimization.

SOBER (Sequential Optimization using Ensemble of Regressors) is a
Bayesian optimization library that uses an ensemble of neural networks
as the surrogate model.

Reference: https://github.com/ma921/SOBER

This module provides the SOBER backend implementation for the
BayesianOptimizer class.

LEVEL 7 backend module.
"""

from typing import Callable, Optional, Tuple, List, Union
import torch
import numpy as np

from condmatTensor.optimization.bayesian.utils import latin_hypercube_sampling


class SoberBackend:
    """SOBER backend for Bayesian optimization.

    Uses the SoberWrapper class from the sober package to perform
    Bayesian optimization with an ensemble of neural networks.

    The SOBER API expects:
    - Bounds as torch.tensor with shape (2, n_dim): [[min_1, ..., min_d], [max_1, ..., max_d]]
    - Tensors with numpy mode disabled (disable_numpy_mode=True)

    Attributes:
        bounds: Parameter bounds [(min_1, max_1), ..., (min_d, max_d)]
        n_init: Number of initial random samples
        seed: Random seed for reproducibility
        X_observed: Observed parameter points
        y_observed: Observed objective values
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        n_init: int = 10,
        seed: Optional[int] = None,
    ):
        """Initialize SOBER backend.

        Args:
            bounds: List of (min, max) tuples for each parameter dimension
            n_init: Number of initial random samples
            seed: Random seed for reproducibility
        """
        self.bounds = bounds
        self.n_init = n_init
        self.seed = seed

        self.X_observed: Optional[torch.Tensor] = None
        self.y_observed: Optional[torch.Tensor] = None

        self._sober_optimizer = None
        self._bounds_tensor = None  # Shape (2, n_dim) for SOBER

    def _initialize_bounds_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert bounds to SOBER format (2, n_dim) tensor.

        SOBER expects bounds as [[min_1, ..., min_d], [max_1, ..., max_d]].

        Args:
            device: Device for the tensor

        Returns:
            Bounds tensor with shape (2, n_dim)
        """
        n_dim = len(self.bounds)
        bounds_tensor = torch.zeros((2, n_dim), dtype=torch.float64, device=device)

        for d, (min_val, max_val) in enumerate(self.bounds):
            bounds_tensor[0, d] = min_val
            bounds_tensor[1, d] = max_val

        return bounds_tensor

    def _initialize_random(
        self,
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate random initial samples using Latin Hypercube Sampling.

        Args:
            n_samples: Number of samples to generate
            device: Device to place tensor on

        Returns:
            Random samples with shape (n_samples, n_dim)
        """
        return latin_hypercube_sampling(self.bounds, n_samples, device, self.seed)

    def suggest_next(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """Suggest next point to evaluate using SOBER.

        Args:
            device: Device for computation

        Returns:
            Suggested point, shape (n_dim,)
        """
        try:
            from sober import SoberWrapper
        except ImportError:
            raise ImportError(
                "SOBER backend requested but not installed. "
                "Install from: https://github.com/ma921/SOBER/releases"
            )

        # Initialize bounds tensor in SOBER format
        if self._bounds_tensor is None:
            self._bounds_tensor = self._initialize_bounds_tensor(device)

        # Convert observed data to numpy (SOBER uses numpy internally)
        X_np = self.X_observed.cpu().numpy()
        y_np = self.y_observed.cpu().numpy().reshape(-1, 1)  # SOBER expects (n, 1)

        bounds_np = self._bounds_tensor.cpu().numpy()

        # Create SOBER optimizer with each iteration
        # disable_numpy_mode=True ensures tensors are returned
        self._sober_optimizer = SoberWrapper(
            X=X_np,
            y=y_np,
            n_init=0,  # We already have initial samples
            bounds=bounds_np,
            verbose=False,
        )

        # Get next suggestion
        # SOBER returns numpy array, convert back to torch
        X_next_np = self._sober_optimizer.suggest(n_samples=1)

        return torch.tensor(X_next_np[0], dtype=torch.float64, device=device)

    def update(self, X: torch.Tensor, y: torch.Tensor):
        """Update observed data.

        Args:
            X: New parameter points, shape (n, n_dim)
            y: New objective values, shape (n,)
        """
        if self.X_observed is None:
            self.X_observed = X
            self.y_observed = y
        else:
            self.X_observed = torch.cat([self.X_observed, X], dim=0)
            self.y_observed = torch.cat([self.y_observed, y], dim=0)


def run_sober_optimization(
    objective: Callable[[torch.Tensor], torch.Tensor],
    bounds: List[Tuple[float, float]],
    n_init: int = 10,
    n_iter: int = 50,
    maximize: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Run Bayesian optimization using SOBER backend.

    This is a convenience function that runs the full optimization loop
    using SOBER as the backend.

    Args:
        objective: Function to optimize, takes X (n, d) and returns y (n,)
        bounds: List of (min, max) tuples for each parameter dimension
        n_init: Number of initial random samples
        n_iter: Number of optimization iterations
        maximize: If True, maximize the objective. If False, minimize.
        seed: Random seed for reproducibility
        verbose: Print progress information
        device: Device for computation

    Returns:
        (X_best, y_best) tuple where:
        - X_best: Best parameters found, shape (n_dim,)
        - y_best: Best objective value found
    """
    if device is None:
        device = torch.device("cpu")

    backend = SoberBackend(bounds=bounds, n_init=n_init, seed=seed)

    # Phase 1: Initial random sampling
    if verbose:
        print(f"Phase 1: Initial random sampling ({n_init} samples)")

    X_init = backend._initialize_random(n_init, device)
    y_init = objective(X_init)

    if y_init.dim() == 0:
        y_init = y_init.unsqueeze(0)
    elif y_init.dim() > 1:
        y_init = y_init.squeeze()

    backend.update(X_init, y_init)

    # Find best initial point
    if maximize:
        best_idx = torch.argmax(backend.y_observed)
    else:
        best_idx = torch.argmin(backend.y_observed)

    X_best = backend.X_observed[best_idx].clone()
    y_best = backend.y_observed[best_idx].item()

    if verbose:
        print(f"  Initial best: {y_best:.6f}")

    # Phase 2: Bayesian optimization iterations
    if verbose:
        print(f"Phase 2: SOBER optimization ({n_iter} iterations)")

    for iteration in range(n_iter):
        # Generate next candidate point
        X_candidate = backend.suggest_next(device)
        y_candidate = objective(X_candidate)

        if y_candidate.dim() == 0:
            y_candidate = y_candidate.unsqueeze(0)
        elif y_candidate.dim() > 1:
            y_candidate = y_candidate.squeeze()

        # Update observed data
        backend.update(X_candidate.unsqueeze(0), y_candidate)

        # Update best
        if maximize:
            if y_candidate.item() > y_best:
                X_best = X_candidate.clone()
                y_best = y_candidate.item()
        else:
            if y_candidate.item() < y_best:
                X_best = X_candidate.clone()
                y_best = y_candidate.item()

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}/{n_iter}: best = {y_best:.6f}")

    if verbose:
        print(f"\nOptimization complete!")
        print(f"  Best value: {y_best:.6f}")
        print(f"  Best parameters: {X_best.cpu().numpy()}")

    return X_best, y_best


__all__ = ["SoberBackend", "run_sober_optimization"]
