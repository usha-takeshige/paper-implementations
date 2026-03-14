"""PDE residual computation via automatic differentiation."""

import torch

from .data import CollocationPoints
from .network import PINN


class PDEResidualComputer:
    """Algorithm 1 の全ステップを実行し，PDE 残差 f を計算する。

    Burgers 方程式 f = u_t + u * u_x - nu * u_xx を自動微分で計算する。
    式(6)，Algorithm 1 に対応。

    Note:
        u_t と u_x の計算時に create_graph=True を指定することで，
        u_xx の自動微分が可能になる。
    """

    def compute(
        self,
        model: PINN,
        collocation: CollocationPoints,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        """PDE 残差を計算する。

        Args:
            model: 学習中のネットワーク u_theta(t, x)。
            collocation: コロケーション点 (t_f, x_f)。
            nu: 動粘性係数（スカラーテンソル）。

        Returns:
            f = u_t + u * u_x - nu * u_xx，形状 (N_f, 1)。
        """
        # requires_grad=True を持つコピーを作成し，元のテンソルを変更しない
        t = collocation.t.clone().requires_grad_(True)
        x = collocation.x.clone().requires_grad_(True)

        u = model(t, x)

        # u_t，u_x：create_graph=True で計算グラフを保持し u_xx の微分を可能にする
        u_t = torch.autograd.grad(
            u,
            t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        u_x = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

        # u_xx：u_x を x で微分
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0]

        # Burgers 方程式の残差：f = u_t + u * u_x - nu * u_xx（式(6)）
        f = u_t + u * u_x - nu * u_xx
        return f
