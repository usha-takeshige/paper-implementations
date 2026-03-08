"""MPDE-BO のメインアルゴリズムループを担うモジュール。"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from botorch.models import SingleTaskGP
from torch import Tensor

from mpde_bo.acquisition_optimizer import AcquisitionOptimizer
from mpde_bo.gp_model_manager import GPModelManager
from mpde_bo.parameter_classifier import ParameterClassification, ParameterClassifier


@dataclass
class BOResult:
    """``MPDEBOOptimizer.optimize`` の返り値。

    Attributes:
        best_x: 全観測点のうち最大観測値に対応する入力点、shape ``(N,)``。
        best_y: ``best_x`` での観測値。
        train_X: 全観測点の履歴、shape ``(n0 + T, N)``。
        train_Y: 全観測値の履歴、shape ``(n0 + T, 1)``。
    """

    best_x: Tensor
    best_y: float
    train_X: Tensor
    train_Y: Tensor


class MPDEBOOptimizer:
    """MPDE-BO アルゴリズム全体のオーケストレータ (method.md §6)。

    各ループで以下を実行する。

    .. code-block:: text

        1. GP モデルを構築
        ─ for t = 1..T:
          3. ARD 長さスケールを取得
          4. MPDE を計算
          5. パラメータを分類 (重要 / 非重要)
          6. 重要次元の獲得関数を最大化
          7. 非重要次元を一様サンプリング
          8. 次の評価点 x_t を決定
          9. f(x_t) を評価し観測データを更新
         10. GP を更新
        12. 最大観測点 x* を返す

    クラス自身はループ制御のみを担い、各処理はサブコンポーネントに委譲する。

    Args:
        gp_manager: GP モデルの構築・更新を行う ``GPModelManager``。
        classifier: パラメータ分類を行う ``ParameterClassifier``。
        acq_optimizer: 獲得関数の最大化を行う ``AcquisitionOptimizer``。
        bounds: 探索空間の境界、shape ``(2, N)``。
    """

    def __init__(
        self,
        gp_manager: GPModelManager,
        classifier: ParameterClassifier,
        acq_optimizer: AcquisitionOptimizer,
        bounds: Tensor,
    ) -> None:
        self._gp_manager = gp_manager
        self._classifier = classifier
        self._acq_optimizer = acq_optimizer
        self._bounds = bounds
        self._gen = torch.Generator()

    def optimize(
        self,
        f: Callable[[Tensor], float],
        T: int,
        train_X: Tensor,
        train_Y: Tensor,
        callback: Callable[[int, SingleTaskGP, ParameterClassification], None] | None = None,
    ) -> BOResult:
        """MPDE-BO を T ステップ実行して最良観測点を返す。"""
        model = self._gp_manager.build(train_X, train_Y)
        all_X = train_X.clone()
        all_Y = train_Y.clone()

        for t in range(1, T + 1):
            cls = self._classifier.classify(model, all_X)
            fixed = self._sample_unimportant(cls.unimportant)
            x_new = self._acq_optimizer.maximize(model, all_Y, fixed)

            y_new_val = f(x_new)
            y_new = torch.tensor([[y_new_val]], dtype=all_Y.dtype)

            all_X = torch.cat([all_X, x_new.unsqueeze(0)], dim=0)
            all_Y = torch.cat([all_Y, y_new], dim=0)
            model = self._gp_manager.update(model, x_new.unsqueeze(0), y_new)

            if callback is not None:
                callback(t, model, cls)

        best_idx = int(all_Y.argmax().item())
        return BOResult(
            best_x=all_X[best_idx],
            best_y=all_Y.max().item(),
            train_X=all_X,
            train_Y=all_Y,
        )

    def _sample_unimportant(self, unimportant_dims: list[int]) -> dict[int, float]:
        """非重要次元を探索境界内の一様分布からサンプリングする。"""
        fixed: dict[int, float] = {}
        for dim in unimportant_dims:
            lo = self._bounds[0, dim].item()
            hi = self._bounds[1, dim].item()
            val = lo + (hi - lo) * torch.rand(1, generator=self._gen).item()
            fixed[dim] = val
        return fixed
