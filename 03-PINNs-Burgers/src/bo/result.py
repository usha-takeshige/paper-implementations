"""Result and configuration data classes for Bayesian optimization."""

from dataclasses import dataclass
from typing import Literal

from opt_tool.base import BaseOptimizationResult, BaseOptimizerConfig


@dataclass(frozen=True)
class BOConfig(BaseOptimizerConfig):
    """Configuration for Bayesian optimization.

    Holds all hyperparameters that control the BO loop: number of initial
    Sobol samples, number of GP-based iterations, acquisition function type,
    and optimizer settings for optimize_acqf.

    Inherits n_initial, n_iterations, seed from BaseOptimizerConfig.
    """

    acquisition: Literal["EI", "UCB"] = "EI"
    ucb_beta: float = 2.0
    num_restarts: int = 10
    raw_samples: int = 512


@dataclass(frozen=True)
class BOResult(BaseOptimizationResult):
    """Result of the entire Bayesian optimization run.

    Aggregates all trial results and identifies the best-performing
    hyperparameter configuration along with the BO settings used.
    Inherits trials, best_params, best_objective, best_trial_id, objective_name
    from BaseOptimizationResult.
    """

    bo_config: BOConfig
