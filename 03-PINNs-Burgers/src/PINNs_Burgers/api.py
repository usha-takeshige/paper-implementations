"""Public API for PINNs Burgers equation solver (Facade pattern)."""

import torch

from .config import NetworkConfig, PDEConfig, TrainingConfig
from .data import BoundaryData, CollocationPoints
from .loss import LossFunction
from .network import PINN
from .residual import PDEResidualComputer
from .results import ForwardResult, InverseResult
from .solver import ForwardSolver, InverseSolver


class BurgersPINNSolver:
    """PINNs による Burgers 方程式ソルバーのメインインターフェース。

    Algorithm 2（順問題）と Algorithm 3（逆問題）をパブリック API として公開する。
    内部クラスの組み立てを担う Facade として機能する。
    MPS（Apple Silicon）が利用可能な場合は自動的に使用する。
    """

    def __init__(
        self,
        pde_config: PDEConfig,
        network_config: NetworkConfig,
        training_config: TrainingConfig,
    ) -> None:
        """初期化する。

        Args:
            pde_config: 問題設定（ν，ドメイン範囲）。
            network_config: ネットワーク構造設定（層数，ニューロン数）。
            training_config: 学習設定（N_u，N_f，lr，エポック数）。
        """
        self.pde_config = pde_config
        self.network_config = network_config
        self.training_config = training_config
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    def solve_forward(
        self,
        data: BoundaryData,
        collocation: CollocationPoints,
    ) -> ForwardResult:
        """Algorithm 2: PINN 順問題ソルバー。

        ν を PDEConfig から固定値として使用し，θ を Adam + L-BFGS で最適化する。
        対応数式: 式(7)(8)(9)(10)。

        Args:
            data: 初期・境界条件データ点。
            collocation: コロケーション点。

        Returns:
            学習済みネットワーク θ* と損失履歴を含む ForwardResult。
        """
        model = PINN(self.network_config).to(self._device)
        residual_computer = PDEResidualComputer()
        loss_fn = LossFunction(residual_computer)
        solver = ForwardSolver(loss_fn, self.pde_config, self._device)

        loss_history = solver.train(model, data, collocation, self.training_config)

        return ForwardResult(model=model, loss_history=loss_history)

    def solve_inverse(
        self,
        observations: BoundaryData,
        collocation: CollocationPoints,
        nu_init: float,
    ) -> InverseResult:
        """Algorithm 3: PINN 逆問題ソルバー（動粘性係数の同定）。

        θ と ν を同時に Adam で最適化する。対応数式: 式(11)。

        Args:
            observations: 観測データ点（初期・境界条件を含む）。
            collocation: コロケーション点。
            nu_init: ν の初期値。

        Returns:
            学習済みネットワーク θ*，推定された ν*，損失履歴を含む InverseResult。
        """
        model = PINN(self.network_config).to(self._device)
        residual_computer = PDEResidualComputer()
        loss_fn = LossFunction(residual_computer)
        solver = InverseSolver(loss_fn, nu_init, self._device)

        loss_history = solver.train(model, observations, collocation, self.training_config)

        return InverseResult(
            model=model,
            nu=solver.nu,
            loss_history=loss_history,
        )
