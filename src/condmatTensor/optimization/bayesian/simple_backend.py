"""Simple fallback backend for Bayesian optimization.

This module provides a simple fallback implementation for Bayesian optimization
when neither SOBER nor BoTorch are available. It uses Thompson sampling with
scikit-learn's Gaussian Process or falls back to pure random search.

LEVEL 7 backend module.
"""

from typing import Callable, Optional, Tuple, List, Union
import torch
import numpy as np

from condmatTensor.optimization.bayesian.utils import latin_hypercube_sampling


class SimpleBackend:
    """Simple fallback backend for Bayesian optimization.

    Uses Thompson sampling with scikit-learn's Gaussian Process as the
    surrogate model. Falls back to pure random search if scikit-learn
    is not available.

    Attributes:
        bounds: Parameter bounds [(min_1, max_1), ..., (min_d, max_d)]
        n_init: Number of initial random samples
        seed: Random seed for reproducibility
        X_observed: Observed parameter points
        y_observed: Observed objective values
        maximize: Whether to maximize or minimize the objective
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        n_init: int = 10,
        seed: Optional[int] = None,
    ):
        """Initialize simple backend.

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
        self.maximize: bool = False  # Will be set during optimization
        self._rng: Optional[np.random.RandomState] = None  # Persistent RNG for Thompson sampling

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
        """Suggest next point to evaluate using Thompson sampling.

        Fits a simple GP using scikit-learn and samples from the posterior.
        Falls back to random search if scikit-learn is not available.

        Args:
            device: Device for computation

        Returns:
            Suggested point, shape (n_dim,)
        """
        # Try to use scikit-learn for simple GP
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

            X_np = self.X_observed.cpu().numpy()
            y_np = self.y_observed.cpu().numpy()

            # Fit GP
            kernel = C(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=self.seed)
            gp.fit(X_np, y_np)

            # Sample from posterior at random points
            n_candidates = 100
            X_random = self._initialize_random(n_candidates, device)
            X_random_np = X_random.cpu().numpy()

            posterior_mean, posterior_std = gp.predict(X_random_np, return_std=True)

            # Thompson sampling: sample from posterior
            # Use persistent RNG to get different random numbers each iteration
            if self._rng is None:
                self._rng = np.random.RandomState(self.seed)
            posterior_sample = posterior_mean + posterior_std * self._rng.randn(len(posterior_mean))

            # Select best candidate
            if self.maximize:
                best_idx = np.argmax(posterior_sample)
            else:
                best_idx = np.argmin(posterior_sample)

            return torch.tensor(X_random_np[best_idx], dtype=torch.float64, device=device)

        except ImportError:
            # Fallback to pure random search
            return self._initialize_random(1, device).squeeze(0)

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


def run_simple_optimization(
    objective: Callable[[torch.Tensor], torch.Tensor],
    bounds: List[Tuple[float, float]],
    n_init: int = 10,
    n_iter: int = 50,
    maximize: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Run Bayesian optimization using simple backend.

    This is a convenience function that runs the full optimization loop
    using the simple backend (Thompson sampling or random search).

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

    backend = SimpleBackend(bounds=bounds, n_init=n_init, seed=seed)
    backend.maximize = maximize

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

    # Phase 2: Optimization iterations
    if verbose:
        print(f"Phase 2: Simple optimization ({n_iter} iterations)")

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


__all__ = ["SimpleBackend", "run_simple_optimization"]
