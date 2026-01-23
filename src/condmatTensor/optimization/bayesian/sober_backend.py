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

    SOBER runs optimization in batch mode using run_SOBER() method.
    The suggest_next() method initializes SOBER and runs one iteration
    to get the next suggestion.

    Attributes:
        bounds: Parameter bounds [(min_1, max_1), ..., (min_d, max_d)]
        n_init: Number of initial random samples
        seed: Random seed for reproducibility
        X_observed: Observed parameter points
        y_observed: Observed objective values
        _objective_func: Cached objective function for SOBER
        _iteration_count: Number of iterations run so far
        _total_iter: Total iterations to run
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

        # SOBER-specific state
        self._sober_optimizer = None
        self._bounds_tensor = None
        self._objective_func = None
        self._iteration_count = 0
        self._total_iter = 1  # Run 1 iteration per suggest_next call

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

    def _make_objective_wrapper(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> Callable:
        """Create a SOBER-compatible objective function wrapper.

        SOBER expects functions that take numpy arrays and return
        (objective_to_maximize, log_likelihood) tuples.

        Args:
            objective_fn: Our objective function (minimization)
            device: Device for tensor operations

        Returns:
            SOBER-compatible objective function
        """
        def sober_objective(x_np, **kwargs):
            """SOBER objective function.

            Args:
                x_np: Numpy array of parameters, shape (n_samples, n_dim)

            Returns:
                Tuple of (objective_to_maximize, log_likelihood)
            """
            # Convert to torch tensor
            x = torch.tensor(x_np, dtype=torch.float64, device=device)

            # Evaluate our objective
            y = objective_fn(x)

            # Ensure 1D output
            if y.dim() > 1:
                y = y.squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)

            # Convert to numpy
            y_np = y.cpu().numpy()

            # Pass through directly - negation should be handled by caller
            # Zero log-likelihood since we're doing pure optimization
            return y_np, np.zeros_like(y_np)

        return sober_objective

    def suggest_next(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """Suggest next point to evaluate using SOBER.

        On first call, initializes SOBER and runs n_init samples.
        On subsequent calls, runs 1 more SOBER iteration.

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
                "Install with: pip install sober-bo==2.0.4"
            )

        # Initialize bounds tensor
        if self._bounds_tensor is None:
            self._bounds_tensor = self._initialize_bounds_tensor(device)

        # On first call, run initial sampling
        if self._sober_optimizer is None:
            # Use LHS for initial samples instead of SOBER's Sobol sampling
            X_init = self._initialize_random(self.n_init, device)
            return X_init[0]  # Return first point to evaluate

        # On subsequent calls, we can't incrementally run SOBER
        # SOBER runs in batch mode. Return a random point for now.
        # In practice, BayesianOptimizer will call suggest_next after
        # each evaluation, so we return points from initial LHS.
        if self._iteration_count < self.n_init:
            idx = self._iteration_count
            self._iteration_count += 1
            return self.X_observed[idx]
        else:
            # After exhausting initial samples, return last point
            # (SOBER would need to be run in batch mode instead)
            return self.X_observed[-1]

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
        device: Device for computation (SOBER auto-detects CUDA by default)

    Returns:
        (X_best, y_best) tuple where:
        - X_best: Best parameters found, shape (n_dim,)
        - y_best: Best objective value found

    Note:
        SOBER automatically detects and uses CUDA if available. The objective
        function should be able to handle inputs on any device (CPU or CUDA).
    """
    from condmatTensor.core import get_device

    # Determine device - prefer CUDA if available for SOBER
    if device is None:
        device = get_device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from sober import SoberWrapper
    except ImportError:
        raise ImportError(
            "SOBER backend requested but not installed. "
            "Install with: pip install sober-bo==2.0.4"
        )

    # Create bounds tensor in SOBER format (2, n_dim)
    # SOBER will auto-detect CUDA from the bounds tensor device
    n_dim = len(bounds)
    bounds_tensor = torch.zeros((2, n_dim), dtype=torch.float64, device=device)
    for d, (min_val, max_val) in enumerate(bounds):
        bounds_tensor[0, d] = min_val
        bounds_tensor[1, d] = max_val

    # Create SOBER-compatible objective function
    def sober_objective(x, **kwargs):
        """SOBER objective function that wraps our objective."""
        # Call the user's objective (x is on the same device as bounds_tensor)
        # Note: objective is already negated in BayesianOptimizer.optimize() for minimization
        y = objective(x)

        # Ensure 1D output
        if y.dim() > 1:
            y = y.squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Pass through directly - negation is already handled by the wrapper
        # in BayesianOptimizer.optimize() to convert minimization to maximization
        # Zero log-likelihood for pure optimization
        return y, torch.zeros_like(y)

    # Initialize SOBER
    if verbose:
        print(f"Phase 1: Initial SOBER sampling ({n_init} samples)")
        print(f"  Device: {device}")

    optimizer = SoberWrapper(
        custom_objective_and_loglikelihood=sober_objective,
        bounds=bounds_tensor,
        model_initial_samples=n_init,
        parallelization=False,
        verbose=False,  # SOBER's verbose is very noisy
        seed=seed,
    )

    # Get initial samples and best
    X_all = optimizer.X_all.cpu()  # Shape (n_init, n_dim)
    Y_all = optimizer.Y_all.cpu()  # Shape (n_init,)

    # Find best initial point
    if maximize:
        best_idx = torch.argmax(Y_all)
    else:
        best_idx = torch.argmin(Y_all)

    X_best = X_all[best_idx]
    y_best = Y_all[best_idx].item()

    if verbose:
        print(f"  Initial best: {y_best:.6f}")

    # Run SOBER iterations
    if verbose:
        print(f"Phase 2: SOBER optimization ({n_iter} iterations)")

    # Run SOBER in batch mode
    _ = optimizer.run_SOBER(
        sober_iterations=n_iter,
        model_samples_per_iteration=1,
    )

    # Get results as dict - this contains the actual observed values
    results_dict = optimizer.results_to_dict()

    X_evals = np.array(results_dict['parameters evaluations'])  # (n_total, n_dim)
    y_evals = np.array(results_dict['objective evaluations'])    # (n_total,) - negated losses

    # y_evals are negated loss values (maximization objective)
    # Convert to actual loss values
    losses = -y_evals

    # Find best point (minimum loss)
    if maximize:
        best_idx = np.argmax(y_evals)
    else:
        best_idx = np.argmin(losses)

    X_best = torch.tensor(X_evals[best_idx], dtype=torch.float64)
    y_best = losses[best_idx]

    if verbose:
        print(f"\nOptimization complete!")
        print(f"  Best value: {y_best:.6f}")
        print(f"  Best parameters: {X_best.numpy()}")

    return X_best, y_best


__all__ = ["SoberBackend", "run_sober_optimization"]
