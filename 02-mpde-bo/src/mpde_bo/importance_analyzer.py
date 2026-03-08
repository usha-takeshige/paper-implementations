"""ICE / MPDE / APDE を計算するモジュール。"""

import torch
from botorch.models import SingleTaskGP
from torch import Tensor

_AUTO_GRID_POINTS = 50


class ImportanceAnalyzer:
    """ICE・MPDE・APDE を計算する (method.md §5)。

    BoTorch の GP モデルの事後平均を利用し、
    各パラメータが目的関数に与える影響を定量化する。
    ステートレスな設計なので、任意のモデル・データと組み合わせて使える。

    用語:
        S   : 注目するパラメータの添字集合 (``param_indices``)
        x_S : 注目パラメータの値
        x_C : 固定パラメータ (補集合) の値
        ICE : Individual Conditional Expectation — 個別の観測点を固定して
              x_S を変化させた GP 予測平均
        MPDE: Maximum Partial Dependence Effect — ICE の最大変動幅
        APDE: Average Partial Dependence Effect — 平均 ICE の変動幅
    """

    def compute_ice(
        self,
        model: SingleTaskGP,
        train_X: Tensor,
        param_indices: list[int],
        x_S_grid: Tensor | None = None,
    ) -> Tensor:
        """ICE を計算する。返り値の shape: (n, g)。"""
        n, N = train_X.shape
        if x_S_grid is None:
            x_S_grid = self._auto_grid(train_X, param_indices)
        g = x_S_grid.shape[0]

        # X_eval[i, j] = train_X[i] を x_S_grid[j] で上書きした点
        X_eval = train_X.unsqueeze(1).expand(n, g, N).clone()
        for k, dim in enumerate(param_indices):
            X_eval[:, :, dim] = x_S_grid[:, k].unsqueeze(0).expand(n, g)

        X_flat = X_eval.reshape(n * g, N)
        with torch.no_grad():
            mu = model.posterior(X_flat).mean.squeeze(-1)
        return mu.reshape(n, g)

    def compute_mpde(
        self,
        model: SingleTaskGP,
        train_X: Tensor,
        param_indices: list[int],
        x_S_grid: Tensor | None = None,
    ) -> float:
        """MPDE を計算する。e_S* = max_i [max_{x_S} μ - min_{x_S} μ]。"""
        ice = self.compute_ice(model, train_X, param_indices, x_S_grid)
        row_ranges = ice.max(dim=1).values - ice.min(dim=1).values
        return row_ranges.max().item()

    def compute_apde(
        self,
        model: SingleTaskGP,
        train_X: Tensor,
        param_indices: list[int],
        x_S_grid: Tensor | None = None,
    ) -> float:
        """APDE を計算する。ê_S = max_{x_S} mean_i μ - min_{x_S} mean_i μ。"""
        ice = self.compute_ice(model, train_X, param_indices, x_S_grid)
        pdp = ice.mean(dim=0)  # 観測点方向に平均 → shape (g,)
        return (pdp.max() - pdp.min()).item()

    def _auto_grid(self, train_X: Tensor, param_indices: list[int]) -> Tensor:
        """各次元の観測範囲内で等間隔グリッドを生成する。shape: (g, |S|)"""
        grids = []
        for dim in param_indices:
            lo = train_X[:, dim].min().item()
            hi = train_X[:, dim].max().item()
            grids.append(
                torch.linspace(lo, hi, _AUTO_GRID_POINTS, dtype=train_X.dtype)
            )
        return torch.stack(grids, dim=-1)  # (g, |S|)
