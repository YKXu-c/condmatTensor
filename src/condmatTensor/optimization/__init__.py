"""Optimization module for parameter tuning and model reduction.

This module provides tools for:
- Bayesian optimization for parameter search (SOBER, BoTorch, Simple)
- Effective array/model downfolding for magnetic systems
- ML-based surrogate models (future)

Backend priority (auto detection):
1. SOBER (preferred) - https://github.com/ma921/SOBER
2. BoTorch/GPyTorch - https://botorch.org/
3. Simple (Thompson sampling fallback)

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
