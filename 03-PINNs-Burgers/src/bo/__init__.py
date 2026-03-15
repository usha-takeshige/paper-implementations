"""Bayesian optimization module for PINN hyperparameter tuning.

Provides a BoTorch-based Gaussian process Bayesian optimization loop
for tuning the neural network hyperparameters of BurgersPINNSolver.

Public API
----------
SearchSpace, HyperParameter                    -- define the hyperparameter search domain
ObjectiveFunction                              -- abstract base class for objective functions
AccuracyObjective                              -- maximize accuracy (-rel_l2_error)
AccuracySpeedObjective                         -- maximize accuracy × speed
BayesianOptimizer, BOConfig                   -- run the BO loop
BOResult, TrialResult                          -- store optimization results
ReportGenerator                                -- generate a markdown report from results
"""

from opt_tool.objective import ObjectiveFunction
from opt_tool.result import TrialResult          # re-exported for backward compatibility
from opt_tool.space import HyperParameter, SearchSpace  # re-exported for backward compatibility
from bo.objective import AccuracyObjective, AccuracySpeedObjective
from bo.optimizer import BayesianOptimizer
from bo.report import ReportGenerator
from bo.result import BOConfig, BOResult

__all__ = [
    "SearchSpace",
    "HyperParameter",
    "ObjectiveFunction",
    "AccuracyObjective",
    "AccuracySpeedObjective",
    "BayesianOptimizer",
    "BOConfig",
    "BOResult",
    "TrialResult",
    "ReportGenerator",
]
