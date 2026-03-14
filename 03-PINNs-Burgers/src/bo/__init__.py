"""Bayesian optimization module for PINN hyperparameter tuning.

Provides a BoTorch-based Gaussian process Bayesian optimization loop
for tuning the neural network hyperparameters of BurgersPINNSolver.

Public API
----------
SearchSpace, HyperParameter   -- define the hyperparameter search domain
ObjectiveFunction             -- evaluate PINN accuracy for a given config
BayesianOptimizer, BOConfig   -- run the BO loop
BOResult, TrialResult         -- store optimization results
ReportGenerator               -- generate a markdown report from results
"""

from bo.objective import ObjectiveFunction
from bo.optimizer import BayesianOptimizer
from bo.report import ReportGenerator
from bo.result import BOConfig, BOResult, TrialResult
from bo.space import HyperParameter, SearchSpace

__all__ = [
    "SearchSpace",
    "HyperParameter",
    "ObjectiveFunction",
    "BayesianOptimizer",
    "BOConfig",
    "BOResult",
    "TrialResult",
    "ReportGenerator",
]
