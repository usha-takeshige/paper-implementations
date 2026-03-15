"""Result and configuration data classes for Bayesian optimization."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from opt_tool.base import BaseOptimizationResult, BaseOptimizerConfig
from opt_tool.result import TrialResult


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


class BOResult(BaseOptimizationResult, BaseModel):
    """Result of the entire Bayesian optimization run.

    Aggregates all trial results and identifies the best-performing
    hyperparameter configuration along with the BO settings used.
    Inherits from BaseOptimizationResult for unified type hierarchy.
    """

    model_config = ConfigDict(frozen=True)

    trials: list[TrialResult] = Field(
        description="全試行結果（初期サンプル + BO 反復）"
    )
    best_params: dict[str, float | int] = Field(
        description="目的関数が最大の試行のハイパーパラメータ"
    )
    best_objective: float = Field(description="最大の目的関数値")
    best_trial_id: int = Field(description="最良試行の trial_id")
    bo_config: BOConfig = Field(description="使用した BO 設定（レポート記載用）")
    objective_name: str = Field(description="使用した ObjectiveFunction の名称")
