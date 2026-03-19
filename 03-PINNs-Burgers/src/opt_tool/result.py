"""Common trial result data class for hyperparameter optimization."""

from pydantic import BaseModel, ConfigDict, Field


class TrialResult(BaseModel):
    """Result of a single hyperparameter evaluation trial.

    Stores the hyperparameter values that were evaluated, the resulting
    objective value, error metrics, and metadata about whether this trial
    came from the initial Sobol sampling phase or the optimizer-guided phase.
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
    proposal_time: float = Field(
        default=0.0,
        description="提案にかかった経過時間 [秒]（BOは acquisition 最適化、LLMは chain.invoke）",
    )
    is_initial: bool = Field(
        description="True: Sobol 初期サンプル、False: オプティマイザ提案点"
    )
