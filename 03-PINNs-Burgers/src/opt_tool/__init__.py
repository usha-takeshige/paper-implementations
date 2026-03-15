"""Common abstractions for hyperparameter optimization.

This module provides shared base classes and utilities used by both
the Bayesian optimization (bo) and LLM-based optimization (opt_agent) modules.
It does not depend on any other src module.

Public API
----------
HyperParameter, SearchSpace    -- define the hyperparameter search domain
ObjectiveFunction              -- abstract base class for objective functions
TrialResult                    -- result of a single hyperparameter evaluation
BaseOptimizerConfig            -- common optimizer configuration base class
BaseOptimizationResult         -- common optimization result base class
BaseOptimizer                  -- Template Method base class for 3-phase optimization
"""

from opt_tool.base import BaseOptimizationResult, BaseOptimizer, BaseOptimizerConfig
from opt_tool.objective import ObjectiveFunction
from opt_tool.result import TrialResult
from opt_tool.space import HyperParameter, SearchSpace

__all__ = [
    "HyperParameter",
    "SearchSpace",
    "ObjectiveFunction",
    "TrialResult",
    "BaseOptimizerConfig",
    "BaseOptimizationResult",
    "BaseOptimizer",
]
