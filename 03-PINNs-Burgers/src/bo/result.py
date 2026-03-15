"""Result and configuration data classes for Bayesian optimization."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BOConfig(BaseModel):
    """Configuration for Bayesian optimization.

    Holds all hyperparameters that control the BO loop: number of initial
    Sobol samples, number of GP-based iterations, acquisition function type,
    and optimizer settings for optimize_acqf.
    """

    model_config = ConfigDict(frozen=True)

    n_initial: int = Field(default=5, description="初期 Sobol サンプル数")
    n_iterations: int = Field(default=20, description="BO 反復回数（GP 更新サイクル数）")
    acquisition: Literal["EI", "UCB"] = Field(
        default="EI", description="獲得関数の種類"
    )
    ucb_beta: float = Field(
        default=2.0,
        description="UCB の探索係数 β（acquisition='UCB' 時のみ有効）",
    )
    num_restarts: int = Field(
        default=10,
        description="optimize_acqf の多点再スタート数",
    )
    raw_samples: int = Field(
        default=512,
        description="optimize_acqf の初期候補サンプル数",
    )
    seed: int = Field(default=42, description="乱数シード（再現性確保用）")


class TrialResult(BaseModel):
    """Result of a single hyperparameter evaluation trial.

    Stores the hyperparameter values that were evaluated, the resulting
    objective value, error metrics, and metadata about whether this trial
    came from the initial Sobol sampling phase or the GP-guided BO phase.
    """

    model_config = ConfigDict(frozen=True)

    trial_id: int = Field(description="試行番号（0 始まり）")
    params: dict[str, float | int] = Field(
        description="評価したハイパーパラメータ値（実スケール）"
    )
    objective: float = Field(
        description="目的関数値（高いほど良い）。具体的な計算式は使用する ObjectiveFunction サブクラスに依存"
    )
    rel_l2_error: float = Field(
        description="相対 L2 誤差 ‖u_pred - u_ref‖₂ / ‖u_ref‖₂"
    )
    elapsed_time: float = Field(description="学習にかかった経過時間 [秒]")
    is_initial: bool = Field(
        description="True: Sobol 初期サンプル、False: BO 提案点"
    )


class BOResult(BaseModel):
    """Result of the entire Bayesian optimization run.

    Aggregates all trial results and identifies the best-performing
    hyperparameter configuration along with the BO settings used.
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
