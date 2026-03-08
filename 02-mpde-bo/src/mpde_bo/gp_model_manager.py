"""GP モデルの構築・更新・推論を担うモジュール。"""

from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

_VALID_KERNELS = {"matern52", "rbf"}


@dataclass
class GPConfig:
    """GP モデルの設定。

    Attributes:
        kernel: カーネルの種類。"matern52" (デフォルト) または "rbf"。
        noise_var: 観測ノイズ分散。None のとき最尤推定で決定する。
    """

    kernel: str = "matern52"
    noise_var: float | None = None


class GPModelManager:
    """SingleTaskGP モデルの構築・更新・推論インターフェース。

    BoTorch の ``SingleTaskGP`` を ARD カーネルで構築し、
    ``fit_gpytorch_mll`` による最尤推定でハイパーパラメータを最適化する。

    内部でデータを保持しているため、``update`` は前回の観測を引き継いだ
    新モデルを返す。

    Args:
        config: GP モデルの設定。
    """

    def __init__(self, config: GPConfig) -> None:
        self._config = config
        self._train_X: Tensor | None = None
        self._train_Y: Tensor | None = None

    def build(self, train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
        """データから GP モデルを構築してフィットする。"""
        kernel = self._make_kernel(train_X.shape[-1])
        model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self._train_X = train_X
        self._train_Y = train_Y
        return model

    def update(self, model: SingleTaskGP, new_X: Tensor, new_Y: Tensor) -> SingleTaskGP:
        """新しい観測点を加えて GP モデルを再構築する。"""
        assert self._train_X is not None and self._train_Y is not None
        train_X = torch.cat([self._train_X, new_X], dim=0)
        train_Y = torch.cat([self._train_Y, new_Y], dim=0)
        return self.build(train_X, train_Y)

    def get_length_scales(self, model: SingleTaskGP) -> Tensor:
        """学習済みモデルから ARD 長さスケールを取得する。shape: (N,)"""
        return model.covar_module.base_kernel.lengthscale.squeeze(0).detach()

    def predict(self, model: SingleTaskGP, X: Tensor) -> tuple[Tensor, Tensor]:
        """任意の点における事後平均と事後分散を返す。shape: (m,), (m,)"""
        with torch.no_grad():
            posterior = model.posterior(X)
        mu = posterior.mean.squeeze(-1)
        sigma2 = posterior.variance.squeeze(-1)
        return mu, sigma2

    def _make_kernel(self, n_dims: int) -> ScaleKernel:
        """カーネル名から gpytorch カーネルを生成する。"""
        if self._config.kernel == "matern52":
            return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_dims))
        elif self._config.kernel == "rbf":
            return ScaleKernel(RBFKernel(ard_num_dims=n_dims))
        else:
            raise ValueError(
                f"Unknown kernel '{self._config.kernel}'. "
                f"Choose from {_VALID_KERNELS}."
            )
