"""N90 評価指標を計算するモジュール (method.md §8)。"""

from collections.abc import Callable

import torch
from torch import Tensor

from mpde_bo.benchmark import BenchmarkFunction
from mpde_bo.optimizer import BOResult

_N_INIT = 3  # 各試行での初期観測数


class N90Evaluator:
    """N90 評価指標を計算する (method.md §8)。

    異なる μ_i を持つ ``n_functions`` 個の目的関数に対してアルゴリズムを実行し、
    最適値の ``threshold_ratio`` 以上を達成するのに必要な評価回数の
    90 パーセンタイルを返す。

    .. code-block:: text

        N90 = Percentile_90 [ min{t : f(x_t) ≥ r * f(x*)} ]_{n_functions 関数}

    ``algorithm`` には ``(f, T, train_X_init, train_Y_init) → BOResult`` の
    シグネチャを持つ任意の BO アルゴリズムを渡せるため、MPDE-BO と
    比較手法を同一インターフェースで評価できる。

    Args:
        budget: 各関数に対する最大評価回数 T (初期データを除く)。
        n_functions: 評価に使う目的関数の数。
        threshold_ratio: 達成目標の割合 r (デフォルト 0.9 = 90%)。
        n_important: ベンチマーク関数の重要パラメータ数。
        n_unimportant: ベンチマーク関数の非重要パラメータ数。
        grid_size: ベンチマーク関数の探索空間幅 M。
        generator: 再現性のための乱数生成器。
    """

    def __init__(
        self,
        budget: int = 200,
        n_functions: int = 100,
        threshold_ratio: float = 0.9,
        n_important: int = 2,
        n_unimportant: int = 8,
        grid_size: int = 100,
        generator: torch.Generator | None = None,
    ) -> None:
        self._budget = budget
        self._n_functions = n_functions
        self._threshold_ratio = threshold_ratio
        self._n_important = n_important
        self._n_unimportant = n_unimportant
        self._grid_size = grid_size
        self._gen = generator

    @property
    def budget(self) -> int:
        """最大評価回数。"""
        return self._budget

    def evaluate(self, algorithm: Callable[..., BOResult]) -> float:
        """N90 スコアを計算する。小さいほど効率の良いアルゴリズムを示す。"""
        steps_to_threshold: list[int] = []
        N = self._n_important + self._n_unimportant
        M = float(self._grid_size)

        for _ in range(self._n_functions):
            f = BenchmarkFunction(
                n_important=self._n_important,
                n_unimportant=self._n_unimportant,
                grid_size=self._grid_size,
                generator=self._gen,
            )
            train_X, train_Y = self._init_data(N, M)
            result = algorithm(f, self._budget, train_X, train_Y)
            t = self._steps_to_reach(result, f.optimal_value)
            steps_to_threshold.append(t)

        steps_tensor = torch.tensor(steps_to_threshold, dtype=torch.double)
        return torch.quantile(steps_tensor, 0.9).item()

    def _init_data(self, N: int, M: float) -> tuple[Tensor, Tensor]:
        """初期観測点を一様サンプリングで生成する。"""
        X = torch.rand(_N_INIT, N, dtype=torch.double, generator=self._gen) * M
        # 目的関数値は algorithm 呼び出し側で評価されるため、ここでは仮置き
        Y = torch.zeros(_N_INIT, 1, dtype=torch.double)
        return X, Y

    def _steps_to_reach(self, result: BOResult, optimal_value: float) -> int:
        """最適値の threshold_ratio 以上を初めて達成した評価ステップ数を返す。"""
        target = self._threshold_ratio * optimal_value
        cummax = result.train_Y.cummax(dim=0).values
        mask = (cummax >= target).squeeze(-1)
        if not mask.any():
            return self._budget
        return int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
