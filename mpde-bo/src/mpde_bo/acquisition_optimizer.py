"""獲得関数の生成と最大化を担うモジュール。"""

from dataclasses import dataclass

import torch
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from torch import Tensor

_VALID_ACQUISITIONS = {"EI", "UCB", "PI"}


@dataclass
class AcquisitionConfig:
    """獲得関数の設定。

    Attributes:
        type: 獲得関数の種類。"EI" (デフォルト)、"UCB"、"PI" から選択。
        ucb_beta: UCB の探索・活用トレードオフパラメータ β。
        num_restarts: マルチスタート最適化の再スタート回数。
        raw_samples: 初期候補点のランダムサンプル数。
    """

    type: str = "EI"
    ucb_beta: float = 2.0
    num_restarts: int = 10
    raw_samples: int = 512

    def __post_init__(self) -> None:
        if self.type not in _VALID_ACQUISITIONS:
            raise ValueError(
                f"Unknown acquisition type '{self.type}'. "
                f"Choose from {_VALID_ACQUISITIONS}."
            )


class AcquisitionOptimizer:
    """BoTorch の獲得関数を生成し、``optimize_acqf`` で最大化する。

    重要パラメータ空間 X^⊤ 上での最適化は、``fixed_features`` に
    非重要次元のサンプル値を渡すことで実現する。これにより BoTorch の
    最適化ルーティンをそのまま活用しつつ探索空間を制限できる。

    獲得関数の対応:

    .. code-block:: text

        "EI"  → ExpectedImprovement(model, best_f)
        "UCB" → UpperConfidenceBound(model, beta)
        "PI"  → ProbabilityOfImprovement(model, best_f)

    Args:
        config: 獲得関数と最適化の設定。
        bounds: 探索空間の境界、shape ``(2, N)``。
                ``bounds[0]`` が下限、``bounds[1]`` が上限。
    """

    def __init__(self, config: AcquisitionConfig, bounds: Tensor) -> None:
        self._config = config
        self._bounds = bounds

    def maximize(
        self,
        model: SingleTaskGP,
        train_Y: Tensor,
        fixed_features: dict[int, float],
    ) -> Tensor:
        """獲得関数を最大化して次の試行点を返す。shape: (N,)"""
        acqf = self._build_acqf(model, train_Y)
        ff = fixed_features if fixed_features else None
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self._bounds,
            q=1,
            num_restarts=self._config.num_restarts,
            raw_samples=self._config.raw_samples,
            fixed_features=ff,
        )
        return candidate.squeeze(0)

    def _build_acqf(
        self, model: SingleTaskGP, train_Y: Tensor
    ) -> ExpectedImprovement | UpperConfidenceBound | ProbabilityOfImprovement:
        """設定に応じた BoTorch 獲得関数インスタンスを生成する。"""
        best_f = train_Y.max()
        if self._config.type == "EI":
            return ExpectedImprovement(model=model, best_f=best_f)
        elif self._config.type == "UCB":
            return UpperConfidenceBound(model=model, beta=self._config.ucb_beta)
        elif self._config.type == "PI":
            return ProbabilityOfImprovement(model=model, best_f=best_f)
        else:
            raise ValueError(f"Unknown acquisition type '{self._config.type}'.")
