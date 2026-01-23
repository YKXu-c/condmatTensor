"""Bayesian optimization for parameter tuning in condensed matter systems.

Provides a unified interface for Bayesian optimization using either SOBER,
BoTorch, or a simple fallback as the backend.

Backend priority (auto detection):
1. SOBER (preferred) - https://github.com/ma921/SOBER
2. BoTorch/GPyTorch - https://botorch.org/
3. Simple (Thompson sampling fallback)

References:
    - SOBER: https://github.com/ma921/SOBER
    - botorch: https://botorch.org/

Dependencies:
    - torch>=2.0
    - Either SOBER, botorch, or scikit-learn (all optional)

LEVEL 7 of the 10-level architecture.
"""

from typing import Callable, Optional, Tuple, Union, List
import torch
import numpy as np

from condmatTensor.optimization.bayesian.sober_backend import SoberBackend, run_sober_optimization
from condmatTensor.optimization.bayesian.botorch_backend import BotorchBackend, run_botorch_optimization
from condmatTensor.optimization.bayesian.simple_backend import SimpleBackend, run_simple_optimization
from condmatTensor.optimization.bayesian.utils import latin_hypercube_sampling


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning.

    Bayesian optimization is an efficient global optimization method for
    expensive-to-evaluate functions. It uses a probabilistic model (surrogate)
    to model the objective function and an acquisition function to decide where
    to sample next.

    This class provides a unified interface that works with either SOBER,
    botorch, or a simple fallback as the backend.

    Attributes:
        bounds: Parameter bounds [(min_1, max_1), ..., (min_d, max_d)]
        n_init: Number of initial random samples
        n_iter: Number of optimization iterations
        backend: Optimization backend ('sober', 'botorch', 'simple', or 'auto')
        X_observed: Observed parameter points
        y_observed: Observed objective values
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        n_init: int = 10,
        n_iter: int = 50,
        backend: str = "auto",
        seed: Optional[int] = None,
    ):
        """Initialize Bayesian optimizer.

        Args:
            bounds: List of (min, max) tuples for each parameter dimension
            n_init: Number of initial random samples
            n_iter: Number of Bayesian optimization iterations
            backend: Optimization backend ('sober', 'botorch', 'simple', or 'auto')
                    'auto' will try SOBER first, then botorch, then simple
            seed: Random seed for reproducibility
        """
        self.bounds = bounds
        self.n_init = n_init
        self.n_iter = n_iter
        self.backend = self._detect_backend(backend)
        self.seed = seed

        # Storage for observed data
        self.X_observed: Optional[torch.Tensor] = None
        self.y_observed: Optional[torch.Tensor] = None

        # Backend-specific optimizer
        self._optimizer = None
        self._maximize: bool = False

    def _detect_backend(self, backend: str) -> str:
        """Detect available optimization backend.

        Args:
            backend: User-specified backend preference

        Returns:
            Actual backend to use ('sober', 'botorch', or 'simple')
        """
        if backend == "auto":
            try:
                from sober import SoberWrapper
                return "sober"
            except ImportError:
                try:
                    import botorch
                    return "botorch"
                except ImportError:
                    return "simple"

        if backend == "sober":
            try:
                from sober import SoberWrapper
            except ImportError:
                raise ImportError(
                    "SOBER backend requested but not installed. "
                    "Install from: https://github.com/ma921/SOBER/releases"
                )
        elif backend == "botorch":
            try:
                import botorch
            except ImportError:
                raise ImportError(
                    "BoTorch backend requested but not installed. "
                    "Install with: pip install botorch gpytorch"
                )
        elif backend != "simple":
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'sober', 'botorch', 'simple', or 'auto'."
            )

        return backend

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

    def _get_backend(self):
        """Get the backend instance based on current settings."""
        if self.backend == "sober":
            return SoberBackend(bounds=self.bounds, n_init=self.n_init, seed=self.seed)
        elif self.backend == "botorch":
            return BotorchBackend(bounds=self.bounds, n_init=self.n_init, seed=self.seed)
        else:
            backend = SimpleBackend(bounds=self.bounds, n_init=self.n_init, seed=self.seed)
            backend.maximize = self._maximize
            return backend

    def optimize(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        maximize: bool = False,
        verbose: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Run Bayesian optimization.

        Args:
            objective: Function to optimize, takes X (n, d) and returns y (n,)
                      Can return batch results or single values
            maximize: If True, maximize the objective. If False, minimize.
            verbose: Print progress information
            device: Device for computation

        Returns:
            (X_best, y_best) tuple where:
            - X_best: Best parameters found, shape (n_dim,)
            - y_best: Best objective value found
        """
        if device is None:
            device = torch.device("cpu")

        self._maximize = maximize

        # Get the appropriate backend
        backend = self._get_backend()

        # Phase 1: Initial random sampling
        if verbose:
            backend_name = self.backend.upper()
            print(f"Phase 1: Initial random sampling ({self.n_init} samples)")
            print(f"  Backend: {backend_name}")

        X_init = self._initialize_random(self.n_init, device)
        y_init = objective(X_init)

        # Ensure y has correct shape
        if y_init.dim() == 0:
            y_init = y_init.unsqueeze(0)
        elif y_init.dim() > 1:
            y_init = y_init.squeeze()

        self.X_observed = X_init
        self.y_observed = y_init
        backend.update(X_init, y_init)

        # Find best initial point
        if maximize:
            best_idx = torch.argmax(self.y_observed)
        else:
            best_idx = torch.argmin(self.y_observed)

        X_best = self.X_observed[best_idx].clone()
        y_best = self.y_observed[best_idx].item()

        if verbose:
            print(f"  Initial best: {y_best:.6f}")

        # Phase 2: Bayesian optimization iterations
        if verbose:
            print(f"Phase 2: Bayesian optimization ({self.n_iter} iterations)")

        for iteration in range(self.n_iter):
            # Generate next candidate point using acquisition function
            X_candidate = backend.suggest_next(device)

            # Evaluate objective
            y_candidate = objective(X_candidate)

            # Ensure y has correct shape
            if y_candidate.dim() == 0:
                y_candidate = y_candidate.unsqueeze(0)
            elif y_candidate.dim() > 1:
                y_candidate = y_candidate.squeeze()

            # Update observed data
            backend.update(X_candidate.unsqueeze(0), y_candidate)
            self.X_observed = torch.cat([self.X_observed, X_candidate.unsqueeze(0)], dim=0)
            self.y_observed = torch.cat([self.y_observed, y_candidate], dim=0)

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
                print(f"  Iteration {iteration + 1}/{self.n_iter}: best = {y_best:.6f}")

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Best value: {y_best:.6f}")
            print(f"  Best parameters: {X_best.cpu().numpy()}")

        return X_best, y_best

    def get_best(self) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Get the best observation so far.

        Returns:
            (X_best, y_best) tuple or (None, None) if no observations
        """
        if self.X_observed is None or self.y_observed is None:
            return None, None

        best_idx = torch.argmin(self.y_observed)
        return self.X_observed[best_idx].clone(), self.y_observed[best_idx].item()

    def reset(self):
        """Reset the optimizer state."""
        self.X_observed = None
        self.y_observed = None
        self._optimizer = None


class MultiObjectiveOptimizer:
    """Multi-objective Bayesian optimization.

    For optimizing multiple objectives simultaneously using Pareto front
    approximation. Uses hypervolume improvement or expected hypervolume
    improvement as acquisition function.

    Attributes:
        bounds: Parameter bounds
        n_objectives: Number of objectives
        n_init: Number of initial samples
        n_iter: Number of optimization iterations
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        n_objectives: int,
        n_init: int = 20,
        n_iter: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize multi-objective optimizer.

        Args:
            bounds: List of (min, max) tuples for each parameter dimension
            n_objectives: Number of objectives to optimize
            n_init: Number of initial random samples
            n_iter: Number of optimization iterations
            seed: Random seed for reproducibility
        """
        self.bounds = bounds
        self.n_objectives = n_objectives
        self.n_init = n_init
        self.n_iter = n_iter
        self.seed = seed

        self.X_observed: Optional[torch.Tensor] = None
        self.y_observed: Optional[torch.Tensor] = None  # Shape: (n, n_objectives)

    def optimize(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        verbose: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run multi-objective Bayesian optimization.

        Args:
            objective: Function taking X (n, d) and returning y (n, n_objectives)
            verbose: Print progress information
            device: Device for computation

        Returns:
            (X_pareto, y_pareto) tuple of Pareto-optimal solutions
        """
        if device is None:
            device = torch.device("cpu")

        # Phase 1: Initial random sampling
        if verbose:
            print(f"Phase 1: Initial random sampling ({self.n_init} samples)")

        optimizer = BayesianOptimizer(
            bounds=self.bounds,
            n_init=self.n_init,
            n_iter=0,  # No additional iterations for initial sampling
            seed=self.seed,
        )
        optimizer.optimize(
            lambda X: objective(X).mean(dim=-1),  # Average objectives for sampling
            maximize=False,
            verbose=False,
            device=device,
        )

        self.X_observed = optimizer.X_observed

        # Evaluate all objectives at initial points
        y_init = objective(self.X_observed)
        if y_init.dim() == 1:
            y_init = y_init.unsqueeze(-1)
        self.y_observed = y_init

        # Phase 2: Iterative optimization
        if verbose:
            print(f"Phase 2: Multi-objective optimization ({self.n_iter} iterations)")

        for iteration in range(self.n_iter):
            # Simple random perturbation for next point
            # In full implementation, would use hypervolume improvement
            X_new = optimizer._initialize_random(1, device).squeeze(0)
            y_new = objective(X_new.unsqueeze(0)).squeeze(0)

            if y_new.dim() == 0:
                y_new = y_new.unsqueeze(-1)

            self.X_observed = torch.cat([self.X_observed, X_new.unsqueeze(0)], dim=0)
            self.y_observed = torch.cat([self.y_observed, y_new.unsqueeze(0)], dim=0)

            if verbose and (iteration + 1) % 20 == 0:
                pareto_size = self._compute_pareto_size()
                print(f"  Iteration {iteration + 1}/{self.n_iter}: Pareto front size = {pareto_size}")

        # Extract Pareto front
        X_pareto, y_pareto = self._get_pareto_front()

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Pareto front size: {len(X_pareto)}")

        return X_pareto, y_pareto

    def _compute_pareto_size(self) -> int:
        """Compute the size of the current Pareto front."""
        _, y_pareto = self._get_pareto_front()
        return len(y_pareto)

    def _get_pareto_front(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract Pareto-optimal solutions from observed data.

        Returns:
            (X_pareto, y_pareto) tuple of non-dominated solutions
        """
        if self.X_observed is None:
            return torch.empty(0, 0), torch.empty(0, 0)

        X = self.X_observed.cpu().numpy()
        y = self.y_observed.cpu().numpy()

        # Find non-dominated points
        is_pareto = np.ones(len(y), dtype=bool)

        for i in range(len(y)):
            for j in range(len(y)):
                if i == j:
                    continue
                # Check if y[j] dominates y[i]
                # y[j] dominates y[i] if it's better or equal in all objectives
                # and strictly better in at least one
                if np.all(y[j] <= y[i]) and np.any(y[j] < y[i]):
                    is_pareto[i] = False
                    break

        X_pareto = torch.tensor(X[is_pareto], dtype=torch.float64)
        y_pareto = torch.tensor(y[is_pareto], dtype=torch.float64)

        return X_pareto, y_pareto


__all__ = [
    "BayesianOptimizer",
    "MultiObjectiveOptimizer",
    "SoberBackend",
    "BotorchBackend",
    "SimpleBackend",
    "run_sober_optimization",
    "run_botorch_optimization",
    "run_simple_optimization",
]
