"""Shared utilities for Bayesian optimization backends.

This module contains common functions used across different Bayesian
optimization backends (SOBER, BoTorch, Simple).

LEVEL 7 utility module.
"""

from typing import List, Tuple, Optional
import torch


def latin_hypercube_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Latin Hypercube Sampling for initial Bayesian optimization points.

    Latin Hypercube Sampling (LHS) is a stratified sampling technique that
    ensures better coverage of the parameter space compared to pure random
    sampling. Each parameter dimension is divided into n_samples equal
    strata, and one sample is taken from each stratum.

    Args:
        bounds: List of (min, max) tuples for each parameter dimension
        n_samples: Number of samples to generate
        device: Device to place tensor on
        seed: Random seed for reproducibility

    Returns:
        Samples with shape (n_samples, n_dim)

    Example:
        >>> bounds = [(0.0, 1.0), (-1.0, 1.0)]
        >>> samples = latin_hypercube_sampling(bounds, n_samples=10, device=torch.device("cpu"))
        >>> print(samples.shape)
        torch.Size([10, 2])
    """
    n_dim = len(bounds)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    samples = torch.zeros((n_samples, n_dim), dtype=torch.float64, device=device)

    for d in range(n_dim):
        min_val, max_val = bounds[d]
        # Generate random permutation for stratified sampling
        perm = torch.randperm(n_samples, generator=generator)
        stratum_size = (max_val - min_val) / n_samples
        # One sample from each stratum with random position within stratum
        samples[:, d] = min_val + (perm + torch.rand(n_samples, generator=generator)) * stratum_size

    return samples


def compute_rmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """Compute Root Mean Square Error between two tensors.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value as a float
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def compute_mae(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """Compute Mean Absolute Error between two tensors.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value as a float
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()


def compute_correlation(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """Compute Pearson correlation coefficient between two tensors.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Correlation coefficient as a float
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Center the data
    y_true_centered = y_true_flat - y_true_flat.mean()
    y_pred_centered = y_pred_flat - y_pred_flat.mean()

    # Compute correlation
    numerator = (y_true_centered * y_pred_centered).sum()
    denominator = torch.sqrt((y_true_centered ** 2).sum()) * torch.sqrt((y_pred_centered ** 2).sum())

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


__all__ = [
    "latin_hypercube_sampling",
    "compute_rmse",
    "compute_mae",
    "compute_correlation",
]
