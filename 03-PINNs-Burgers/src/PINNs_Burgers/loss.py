"""Loss function computation for PINNs."""

import torch

from .data import BoundaryData, CollocationPoints
from .network import PINN
from .residual import PDEResidualComputer


class LossFunction:
    """L_data（式(8)）と L_phys（式(9)）を計算し，総損失 L（式(7)）を集計する。

    式(7)(8)(9) に対応。
    """

    def __init__(self, residual_computer: PDEResidualComputer) -> None:
        """初期化する。

        Args:
            residual_computer: PDE 残差計算オブジェクト。
        """
        self.residual_computer = residual_computer

    def compute(
        self,
        model: PINN,
        boundary_data: BoundaryData,
        collocation: CollocationPoints,
        nu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """総損失・データ損失・物理損失を計算する。

        Args:
            model: 学習中のネットワーク u_theta(t, x)。
            boundary_data: 初期・境界条件データ点。
            collocation: コロケーション点。
            nu: 動粘性係数（スカラーテンソル）。

        Returns:
            (L_total, L_data, L_phys) のタプル。式(7)(8)(9) に対応。
        """
        # データ損失 L_data = (1/N_u) * Σ |u_theta(t_u^i, x_u^i) - u^i|^2（式(8)）
        u_pred = model(boundary_data.t, boundary_data.x)
        L_data = torch.mean((u_pred - boundary_data.u) ** 2)

        # 物理損失 L_phys = (1/N_f) * Σ |f(t_f^j, x_f^j)|^2（式(9)）
        f = self.residual_computer.compute(model, collocation, nu)
        L_phys = torch.mean(f**2)

        # 総損失 L = L_data + L_phys（式(7)）
        L_total = L_data + L_phys

        return L_total, L_data, L_phys
