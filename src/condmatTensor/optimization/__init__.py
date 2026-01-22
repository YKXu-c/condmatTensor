"""Optimization module for parameter tuning and model reduction.

This module provides tools for:
- Bayesian optimization for parameter search
- Effective array/model downfolding for magnetic systems
- ML-based surrogate models (future)

LEVEL 7 of the 10-level architecture.
"""

from condmatTensor.optimization.bayesian import (
    BayesianOptimizer,
    MultiObjectiveOptimizer,
)

from condmatTensor.optimization.magnetic import (
    EffectiveArrayOptimizer,
)

__all__ = [
    "BayesianOptimizer",
    "MultiObjectiveOptimizer",
    "EffectiveArrayOptimizer",
]
