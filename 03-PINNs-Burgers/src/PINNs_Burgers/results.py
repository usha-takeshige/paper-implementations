"""Result data classes for forward and inverse problems."""

from pydantic import BaseModel, ConfigDict, Field

from .network import PINN


class ForwardResult(BaseModel):
    """順問題の最適化結果を保持する。

    Algorithm 2 出力に対応。
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: PINN = Field(description="学習済みネットワーク θ*")
    loss_history: list[float] = Field(description="エポックごとの総損失値の履歴")


class InverseResult(BaseModel):
    """逆問題の最適化結果（θ* と ν*）を保持する。

    Algorithm 3 出力，式(11) に対応。
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: PINN = Field(description="学習済みネットワーク θ*")
    nu: float = Field(description="同定された動粘性係数 ν*（式(11) の推定値）")
    loss_history: list[float] = Field(description="エポックごとの総損失値の履歴")
