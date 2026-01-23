"""BoTorch backend for Bayesian optimization.

BoTorch is a library for Bayesian optimization built on PyTorch.
It uses Gaussian Process (GP) surrogate models with acquisition functions
like Expected Improvement (EI).

Reference: https://botorch.org/

This module provides the BoTorch backend implementation for the
BayesianOptimizer class.

LEVEL 7 backend module.
"""

from typing import Callable, Optional, Tuple, List, Union
import torch

from condmatTensor.optimization.bayesian.utils import latin_hypercube_sampling


class BotorchBackend:
    """BoTorch backend for Bayesian optimization.

    Uses Gaussian Process (GP) surrogate models with Expected Improvement
    (EI) as the acquisition function.

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
        """Initialize BoTorch backend.

        Args:
            bounds: List of (min, max) tuples for each parameter dimension
            n_init: Number of initial random samples
            seed: Random seed for reproducibility
        """
        try:
            try:
                from botorch import fit_gpytorch_mll_torch
                fit_gpytorch = fit_gpytorch_mll_torch
            except ImportError:
                # Older botorch versions
                from botorch import fit_gpytorch_mll as fit_gpytorch
            from botorch.acquisition import ExpectedImprovement
            from botorch.optim import optimize_acqf
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as e:
            raise ImportError(
                "BoTorch backend requested but not installed. "
                "Install with: pip install botorch gpytorch"
            ) from e

        self.bounds = bounds
        self.n_init = n_init
        self.seed = seed

        self.X_observed: Optional[torch.Tensor] = None
        self.y_observed: Optional[torch.Tensor] = None

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
        """Suggest next point to evaluate using BoTorch.

        Uses Expected Improvement (EI) as the acquisition function with
        a Gaussian Process surrogate model.

        Args:
            device: Device for computation

        Returns:
            Suggested point, shape (n_dim,)
        """
        try:
            try:
                from botorch import fit_gpytorch_mll_torch
                fit_gpytorch = fit_gpytorch_mll_torch
            except ImportError:
                # Older botorch versions
                from botorch import fit_gpytorch_mll as fit_gpytorch
        except ImportError:
            from botorch import fit_gpytorch_model as fit_gpytorch
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        # BoTorch expects specific shapes
        # X: (n, d) with n as batch dimension, d as feature dimension
        # y: (n, 1)

        X = self.X_observed.clone()
        y = self.y_observed.clone()
        # Ensure y is 2D with shape (n, 1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        elif y.dim() > 2:
            y = y.squeeze()

        # Define bounds for optimization (normalized to [0, 1]^d)
        # BoTorch expects bounds as (2, d) tensor
        bounds_normalized = torch.tensor([[0.0] * len(self.bounds), [1.0] * len(self.bounds)],
                                         dtype=torch.float64, device=device)

        # Normalize X to [0, 1]
        X_normalized = torch.zeros_like(X)
        for d, (min_val, max_val) in enumerate(self.bounds):
            X_normalized[:, d] = (X[:, d] - min_val) / (max_val - min_val)

        # Fit GP model
        gp = SingleTaskGP(X_normalized, y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch(mll)

        # Compute acquisition function
        best_value = y.max().item()
        EI = ExpectedImprovement(model=gp, best_f=best_value)

        # Optimize acquisition function
        candidate, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds_normalized,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )

        # Denormalize candidate
        candidate_original = torch.zeros_like(candidate.squeeze())
        for d, (min_val, max_val) in enumerate(self.bounds):
            candidate_original[d] = candidate[0, d] * (max_val - min_val) + min_val

        return candidate_original.to(device)

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


def run_botorch_optimization(
    objective: Callable[[torch.Tensor], torch.Tensor],
    bounds: List[Tuple[float, float]],
    n_init: int = 10,
    n_iter: int = 50,
    maximize: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Run Bayesian optimization using BoTorch backend.

    This is a convenience function that runs the full optimization loop
    using BoTorch as the backend.

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

    backend = BotorchBackend(bounds=bounds, n_init=n_init, seed=seed)

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
        print(f"Phase 2: BoTorch optimization ({n_iter} iterations)")

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


__all__ = ["BotorchBackend", "run_botorch_optimization"]
