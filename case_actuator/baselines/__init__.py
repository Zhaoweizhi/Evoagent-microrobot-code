"""Baselines package with lazy optimizer imports."""

from .base_optimizer import BaseOptimizer, PARAM_BOUNDS, PENALTY_FITNESS

__all__ = [
    "BaseOptimizer",
    "GAOptimizer",
    "BOOptimizer",
    "PSOOptimizer",
    "PARAM_BOUNDS",
    "PENALTY_FITNESS",
]


def __getattr__(name):
    # Lazy-load heavy optimizer modules to avoid importing optional dependencies
    # unless they are explicitly requested.
    if name == "GAOptimizer":
        from .ga_optimizer import GAOptimizer

        return GAOptimizer
    if name == "BOOptimizer":
        from .bo_optimizer import BOOptimizer

        return BOOptimizer
    if name == "PSOOptimizer":
        from .pso_optimizer import PSOOptimizer

        return PSOOptimizer
    raise AttributeError(f"module 'baselines' has no attribute '{name}'")
