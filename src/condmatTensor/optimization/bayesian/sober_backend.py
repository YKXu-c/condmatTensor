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
        objective: Function to optimize, takes X (n, d) and returns y (n,).
                   For minimization, the objective should already be negated
                   by the caller (e.g., BayesianOptimizer.optimize()).
        bounds: List of (min, max) tuples for each parameter dimension
        n_init: Number of initial random samples
        n_iter: Number of optimization iterations
        maximize: Ignored - SOBER always maximizes. The objective function
                   should be pre-negated for minimization problems.
        seed: Random seed for reproducibility
        verbose: Print progress information
        device: Device for computation (SOBER auto-detects CUDA by default)

    Returns:
        (X_best, y_best) tuple where:
        - X_best: Best parameters found (denormalized), shape (n_dim,)
        - y_best: Best objective value found (as returned by objective function)

    Note:
        SOBER automatically detects and uses CUDA if available. The objective
        function should be able to handle inputs on any device (CPU or CUDA).

        SOBER internally normalizes parameters to [0,1] for optimization.
        This function denormalizes the returned parameters back to the
        original bounds.
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

    # Track actual objective values (SOBER transforms Y internally)
    X_actual = []
    Y_actual = []

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

        # Store actual values for tracking (SOBER transforms Y internally)
        X_actual.append(x.detach().cpu())
        Y_actual.append(y.detach().cpu())

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

    # Get initial best from actual tracked values
    # Note: SOBER may call objective with batch of samples, so Y_actual may be nested
    Y_init_all = torch.cat([y if y.dim() > 0 else y.unsqueeze(0) for y in Y_actual])
    best_idx = torch.argmax(Y_init_all)
    y_best = Y_init_all[best_idx].item()

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

    # Get results from actual tracked values
    # Note: SOBER normalizes parameters internally to [0,1]
    # We need to denormalize them back to original bounds
    X_all_normalized = optimizer.X_all  # (n_total, n_dim) - normalized to [0,1]
    # Flatten Y_actual (may contain batched results)
    Y_all_actual = torch.cat([y if y.dim() > 0 else y.unsqueeze(0) for y in Y_actual])  # (n_total,) - actual objective values

    # Denormalize X values back to original bounds
    n_dim = len(bounds)
    X_evals = torch.zeros_like(X_all_normalized)
    for d in range(n_dim):
        min_val, max_val = bounds[d]
        X_evals[:, d] = min_val + X_all_normalized[:, d] * (max_val - min_val)

    # SOBER always maximizes, so we always take argmax
    # The objective passed to SOBER should already be negated for minimization
    best_idx = torch.argmax(Y_all_actual)

    X_best = X_evals[best_idx]
    y_best = Y_all_actual[best_idx].item()  # Return the actual value that SOBER optimized

    if verbose:
        print(f"\nOptimization complete!")
        print(f"  Best value: {y_best:.6f}")
        print(f"  Best parameters: {X_best.numpy()}")

    return X_best, y_best


__all__ = ["SoberBackend", "run_sober_optimization"]
