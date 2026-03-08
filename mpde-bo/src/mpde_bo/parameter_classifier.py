"""パラメータ重要度の分類を担うモジュール。"""

from dataclasses import dataclass, field

from botorch.models import SingleTaskGP
from torch import Tensor

from mpde_bo.importance_analyzer import ImportanceAnalyzer


@dataclass
class ClassificationConfig:
    """パラメータ分類の閾値設定。

    Attributes:
        eps_l: ARD 長さスケールの閾値 ε_ℓ。
               ``ℓ_i < eps_l`` を満たす場合に長さスケール条件を通過する。
               小さいほど「重要」と判定されにくくなる。
        eps_e: MPDE の閾値 ε_e。
               ``ê_i* > eps_e`` を満たす場合に MPDE 条件を通過する。
               大きいほど「重要」と判定されにくくなる。
    """

    eps_l: float
    eps_e: float


@dataclass
class ParameterClassification:
    """パラメータ分類の結果。

    Attributes:
        important: 重要パラメータの次元インデックス (X^⊤)。
        unimportant: 非重要パラメータの次元インデックス (X^⊥)。
    """

    important: list[int] = field(default_factory=list)
    unimportant: list[int] = field(default_factory=list)


class ParameterClassifier:
    """ARD 長さスケールと MPDE の両条件でパラメータを分類する (method.md §6 ステップ 5)。

    各パラメータ i を以下の条件で重要 / 非重要に振り分ける。

    .. code-block:: text

        ℓ_i < ε_ℓ  AND  ê_i* > ε_e  →  重要  (important)
        それ以外                       →  非重要 (unimportant)

    ``ImportanceAnalyzer`` は MPDE の数値計算を担い、
    本クラスは閾値処理による判定のみを担う (SRP)。

    Args:
        analyzer: ICE / MPDE 計算を委譲する ``ImportanceAnalyzer``。
        config: 分類閾値の設定。
    """

    def __init__(self, analyzer: ImportanceAnalyzer, config: ClassificationConfig) -> None:
        self._analyzer = analyzer
        self._config = config

    def classify(self, model: SingleTaskGP, train_X: Tensor) -> ParameterClassification:
        """各パラメータを重要 / 非重要に分類する。"""
        N = train_X.shape[-1]
        length_scales = model.covar_module.base_kernel.lengthscale.squeeze(0).detach()

        important: list[int] = []
        unimportant: list[int] = []

        for i in range(N):
            ls_i = length_scales[i].item()
            mpde_i = self._analyzer.compute_mpde(model, train_X, [i])
            if ls_i < self._config.eps_l and mpde_i > self._config.eps_e:
                important.append(i)
            else:
                unimportant.append(i)

        return ParameterClassification(important=important, unimportant=unimportant)
