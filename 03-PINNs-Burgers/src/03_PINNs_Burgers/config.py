"""Configuration data classes for PINNs Burgers equation solver."""

from pydantic import BaseModel, ConfigDict, Field


class PDEConfig(BaseModel):
    """Burgers 方程式の問題設定値を保持する。

    式(1)(2)(3)(4)，Section 2 に対応。
    """

    model_config = ConfigDict(frozen=True)

    nu: float = Field(description="ν：動粘性係数（デフォルト 0.01/π）")
    x_min: float = Field(description="x ドメイン下限（-1.0）")
    x_max: float = Field(description="x ドメイン上限（1.0）")
    t_min: float = Field(description="t ドメイン下限（0.0）")
    t_max: float = Field(description="t ドメイン上限（1.0）")


class NetworkConfig(BaseModel):
    """ネットワーク構造に関するハイパーパラメータを保持する。

    式(5)，Section 6.1 に対応。
    活性化関数は論文で tanh に固定されているため設定項目に含めない。
    """

    model_config = ConfigDict(frozen=True)

    n_hidden_layers: int = Field(description="L：隠れ層数（デフォルト 4）")
    n_neurons: int = Field(description="N：各層のニューロン数（デフォルト 20）")


class TrainingConfig(BaseModel):
    """学習に関するハイパーパラメータを保持する。

    Algorithm 2 Step 2，Section 6.1 に対応。
    """

    model_config = ConfigDict(frozen=True)

    n_u: int = Field(description="N_u：初期・境界条件データ点数（デフォルト 100）")
    n_f: int = Field(description="N_f：コロケーション点数（デフォルト 10000）")
    lr: float = Field(description="η：学習率")
    epochs_adam: int = Field(description="E_Adam：Adam フェーズのエポック数")
    epochs_lbfgs: int = Field(
        description="E_LBFGS：L-BFGS フェーズのエポック数（逆問題では不使用）"
    )
