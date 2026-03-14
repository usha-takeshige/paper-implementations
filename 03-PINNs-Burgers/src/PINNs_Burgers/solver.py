"""Solver classes implementing Template Method pattern for PINNs training."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .config import PDEConfig, TrainingConfig
from .data import BoundaryData, CollocationPoints
from .loss import LossFunction
from .network import PINN


class BaseSolver(ABC):
    """Algorithm 2/3 共通のトレーニングループ骨格を定義する（Template Method パターン）。

    Algorithm 2/3 の共通部分（Step 3–10）に対応。
    サブクラスは _get_trainable_params，_get_nu，_after_step をオーバーライドする。
    """

    def __init__(self, loss_fn: LossFunction) -> None:
        """初期化する。

        Args:
            loss_fn: 損失計算オブジェクト。
        """
        self.loss_fn = loss_fn

    def train(
        self,
        model: PINN,
        boundary_data: BoundaryData,
        collocation: CollocationPoints,
        config: TrainingConfig,
    ) -> list[float]:
        """Template Method：Adam によるトレーニングループを実行する。

        Args:
            model: 学習対象のネットワーク。
            boundary_data: 初期・境界条件データ点。
            collocation: コロケーション点。
            config: 学習設定。

        Returns:
            エポックごとの総損失値の履歴。
        """
        params = self._get_trainable_params(model)
        optimizer = torch.optim.Adam(params, lr=config.lr)
        loss_history: list[float] = []

        for _ in range(config.epochs_adam):
            optimizer.zero_grad()
            nu = self._get_nu()
            L_total, _, _ = self.loss_fn.compute(model, boundary_data, collocation, nu)
            L_total.backward()
            optimizer.step()
            loss_history.append(L_total.item())
            self._after_step()

        return loss_history

    @abstractmethod
    def _get_trainable_params(self, model: PINN) -> list[nn.Parameter]:
        """最適化対象のパラメータリストを返す（θ のみ or [θ, ν]）。"""
        ...

    @abstractmethod
    def _get_nu(self) -> torch.Tensor:
        """損失計算に使用する nu テンソルを返す。"""
        ...

    @abstractmethod
    def _after_step(self) -> None:
        """各ステップ後の追加処理（逆問題では ν の値を記録するなど）。"""
        ...


class ForwardSolver(BaseSolver):
    """Algorithm 2 の Phase 1（Adam）と Phase 2（L-BFGS）の二段階最適化を実装する。

    Algorithm 2 に対応。
    """

    def __init__(self, loss_fn: LossFunction, pde_config: PDEConfig) -> None:
        """初期化する。

        Args:
            loss_fn: 損失計算オブジェクト。
            pde_config: 問題設定（ν を固定値として使用）。
        """
        super().__init__(loss_fn)
        self._nu_value = torch.tensor(pde_config.nu, dtype=torch.float32)

    def train(
        self,
        model: PINN,
        boundary_data: BoundaryData,
        collocation: CollocationPoints,
        config: TrainingConfig,
    ) -> list[float]:
        """Adam + L-BFGS の二段階最適化を実行する。

        Args:
            model: 学習対象のネットワーク。
            boundary_data: 初期・境界条件データ点。
            collocation: コロケーション点。
            config: 学習設定。

        Returns:
            エポックごとの総損失値の履歴（Adam フェーズ + L-BFGS フェーズ）。
        """
        # Phase 1: Adam（Algorithm 2 Step 2 Phase 1）
        loss_history = super().train(model, boundary_data, collocation, config)

        # Phase 2: L-BFGS（Algorithm 2 Step 2 Phase 2）
        params = self._get_trainable_params(model)
        optimizer = torch.optim.LBFGS(
            params,
            max_iter=config.epochs_lbfgs,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            """L-BFGS が要求するクロージャ（損失を再計算して返す）。"""
            optimizer.zero_grad()
            nu = self._get_nu()
            L_total, _, _ = self.loss_fn.compute(
                model, boundary_data, collocation, nu
            )
            L_total.backward()
            loss_history.append(L_total.item())
            return L_total

        optimizer.step(closure)

        return loss_history

    def _get_trainable_params(self, model: PINN) -> list[nn.Parameter]:
        """θ のみを返す（ν は PDEConfig から固定値として使用）。"""
        return list(model.parameters())

    def _get_nu(self) -> torch.Tensor:
        """PDEConfig に設定された固定の nu テンソルを返す。"""
        return self._nu_value

    def _after_step(self) -> None:
        """Algorithm 2 では追加処理なし。"""
        pass


class InverseSolver(BaseSolver):
    """Algorithm 3 の θ と ν の同時最適化を実装する。

    Algorithm 3 に対応。ν を学習可能パラメータとして θ と同時に最適化する。
    """

    def __init__(self, loss_fn: LossFunction, nu_init: float) -> None:
        """初期化する。

        Args:
            loss_fn: 損失計算オブジェクト。
            nu_init: ν の初期値（学習可能パラメータとして扱う）。
        """
        super().__init__(loss_fn)
        # ν を学習可能パラメータとして宣言（式(11) 対応）
        self._nu_param = nn.Parameter(
            torch.tensor(nu_init, dtype=torch.float32)
        )
        self._nu_history: list[float] = []

    def _get_trainable_params(self, model: PINN) -> list[nn.Parameter]:
        """[θ, ν] を返す（ν を学習可能パラメータとして追加）。"""
        return list(model.parameters()) + [self._nu_param]

    def _get_nu(self) -> torch.Tensor:
        """現在の学習可能な nu パラメータを返す。"""
        return self._nu_param

    def _after_step(self) -> None:
        """各ステップ後に ν の現在値を記録する。"""
        self._nu_history.append(self._nu_param.item())

    @property
    def nu(self) -> float:
        """現在の ν の推定値を返す。"""
        return self._nu_param.item()
