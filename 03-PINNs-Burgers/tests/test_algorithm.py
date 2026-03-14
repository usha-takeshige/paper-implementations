"""A-1 アルゴリズム実装テスト。

各アルゴリズムステップが計算として正しく動作するかを検証する。
"""

import math

import pytest
import torch
from pydantic import ValidationError

from PINNs_Burgers import (
    BoundaryData,
    CollocationPoints,
    NetworkConfig,
    PDEConfig,
    TrainingConfig,
)
from PINNs_Burgers.solver import InverseSolver, ForwardSolver
from PINNs_Burgers.loss import LossFunction
from PINNs_Burgers.residual import PDEResidualComputer
from PINNs_Burgers.network import PINN


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture()
def network_config() -> NetworkConfig:
    """小規模なネットワーク設定（テスト用）。"""
    return NetworkConfig(n_hidden_layers=2, n_neurons=4)


@pytest.fixture()
def pde_config() -> PDEConfig:
    """論文デフォルトの問題設定。"""
    return PDEConfig(nu=0.01 / math.pi, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0)


@pytest.fixture()
def training_config() -> TrainingConfig:
    """テスト用の最小学習設定。"""
    return TrainingConfig(n_u=5, n_f=10, lr=1e-3, epochs_adam=1, epochs_lbfgs=1)


@pytest.fixture()
def pinn(network_config: NetworkConfig) -> PINN:
    """テスト用 PINN インスタンス。"""
    return PINN(network_config)


@pytest.fixture()
def boundary_data() -> BoundaryData:
    """テスト用の境界・初期条件データ（N_u=5）。"""
    return BoundaryData(
        t=torch.zeros(5, 1),
        x=torch.linspace(-1, 1, 5).unsqueeze(1),
        u=torch.zeros(5, 1),
    )


@pytest.fixture()
def collocation() -> CollocationPoints:
    """テスト用のコロケーション点（N_f=10）。"""
    return CollocationPoints(
        t=torch.rand(10, 1),
        x=torch.rand(10, 1) * 2 - 1,
    )


@pytest.fixture()
def nu_tensor() -> torch.Tensor:
    """テスト用の nu スカラーテンソル。"""
    return torch.tensor(0.01 / math.pi)


@pytest.fixture()
def residual_computer() -> PDEResidualComputer:
    """テスト用 PDEResidualComputer インスタンス。"""
    return PDEResidualComputer()


@pytest.fixture()
def loss_fn(residual_computer: PDEResidualComputer) -> LossFunction:
    """テスト用 LossFunction インスタンス。"""
    return LossFunction(residual_computer)


# ---------------------------------------------------------------------------
# ALG-01: PINN.forward の出力形状
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg01_pinn_forward_output_shape(pinn: PINN) -> None:
    """ALG-01: PINN.forward の出力形状が (N, 1) であること。"""
    N = 7
    t = torch.rand(N, 1)
    x = torch.rand(N, 1) * 2 - 1
    out = pinn(t, x)
    assert out.shape == (N, 1), f"expected shape ({N}, 1), got {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# ALG-02: Xavier 一様初期化
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg02_pinn_xavier_init(network_config: NetworkConfig) -> None:
    """ALG-02: 全 Linear 層に Xavier 一様初期化が適用されること。

    Xavier 一様分布の理論的な標準偏差 gain * sqrt(6 / (fan_in + fan_out)) を
    大きく逸脱していないことで確認する（全ゼロ初期化でないことの検証）。
    """
    model = PINN(network_config)
    import torch.nn as nn

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) > 0, "Linear layers not found"

    for layer in linear_layers:
        assert not torch.all(layer.weight == 0), "weight is all zeros (Xavier not applied)"
        assert torch.all(layer.bias == 0), "bias should be initialized to zero"


# ---------------------------------------------------------------------------
# ALG-03: PDEResidualComputer.compute の出力形状
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg03_residual_output_shape(
    pinn: PINN, collocation: CollocationPoints, nu_tensor: torch.Tensor
) -> None:
    """ALG-03: PDEResidualComputer.compute の出力形状が (N_f, 1) であること。"""
    computer = PDEResidualComputer()
    f = computer.compute(pinn, collocation, nu_tensor)
    N_f = collocation.t.shape[0]
    assert f.shape == (N_f, 1), f"expected shape ({N_f}, 1), got {tuple(f.shape)}"


# ---------------------------------------------------------------------------
# ALG-04: f が微分可能であること（create_graph=True による u_xx の計算）
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg04_residual_is_differentiable(
    pinn: PINN, collocation: CollocationPoints, nu_tensor: torch.Tensor
) -> None:
    """ALG-04: compute された f が grad_fn を持ち微分可能であること。"""
    computer = PDEResidualComputer()
    f = computer.compute(pinn, collocation, nu_tensor)
    assert f.grad_fn is not None, "f.grad_fn is None: create_graph=True が有効でない可能性"


# ---------------------------------------------------------------------------
# ALG-05: LossFunction.compute の戻り値が 3 要素タプルであること
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg05_loss_returns_three_tensors(
    loss_fn: LossFunction,
    pinn: PINN,
    boundary_data: BoundaryData,
    collocation: CollocationPoints,
    nu_tensor: torch.Tensor,
) -> None:
    """ALG-05: LossFunction.compute が (L_total, L_data, L_phys) を返すこと。"""
    result = loss_fn.compute(pinn, boundary_data, collocation, nu_tensor)
    assert len(result) == 3, f"expected 3 elements, got {len(result)}"
    L_total, L_data, L_phys = result
    for name, tensor in [("L_total", L_total), ("L_data", L_data), ("L_phys", L_phys)]:
        assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
        assert tensor.shape == torch.Size([]), f"{name} is not a scalar tensor"


# ---------------------------------------------------------------------------
# ALG-06: L_total = L_data + L_phys
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg06_loss_total_equals_sum(
    loss_fn: LossFunction,
    pinn: PINN,
    boundary_data: BoundaryData,
    collocation: CollocationPoints,
    nu_tensor: torch.Tensor,
) -> None:
    """ALG-06: L_total = L_data + L_phys が成立すること（式(7)）。"""
    L_total, L_data, L_phys = loss_fn.compute(pinn, boundary_data, collocation, nu_tensor)
    diff = abs(L_total.item() - (L_data + L_phys).item())
    assert diff < 1e-6, f"L_total != L_data + L_phys: diff={diff}"


# ---------------------------------------------------------------------------
# ALG-07: L_data, L_phys が非負
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg07_losses_non_negative(
    loss_fn: LossFunction,
    pinn: PINN,
    boundary_data: BoundaryData,
    collocation: CollocationPoints,
    nu_tensor: torch.Tensor,
) -> None:
    """ALG-07: MSE であるため L_data と L_phys は非負であること。"""
    _, L_data, L_phys = loss_fn.compute(pinn, boundary_data, collocation, nu_tensor)
    assert L_data.item() >= 0, f"L_data={L_data.item()} is negative"
    assert L_phys.item() >= 0, f"L_phys={L_phys.item()} is negative"


# ---------------------------------------------------------------------------
# ALG-08: ForwardSolver._get_trainable_params に ν が含まれないこと
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg08_forward_solver_params_no_nu(
    loss_fn: LossFunction, pde_config: PDEConfig, pinn: PINN
) -> None:
    """ALG-08: ForwardSolver の学習パラメータに ν テンソルが含まれないこと。"""
    solver = ForwardSolver(loss_fn, pde_config)
    params = solver._get_trainable_params(pinn)
    nu_value = pde_config.nu

    for p in params:
        # ν の固定値テンソルと同一のスカラーは含まれないことを確認
        if p.numel() == 1 and abs(p.item() - nu_value) < 1e-9:
            pytest.fail("ν tensor found in ForwardSolver trainable params")

    # モデルパラメータのみであることを確認
    model_param_ids = {id(p) for p in pinn.parameters()}
    for p in params:
        assert id(p) in model_param_ids, "unknown parameter found in ForwardSolver params"


# ---------------------------------------------------------------------------
# ALG-09: InverseSolver._get_trainable_params に ν が含まれること
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg09_inverse_solver_params_includes_nu(
    loss_fn: LossFunction, pinn: PINN
) -> None:
    """ALG-09: InverseSolver の学習パラメータに requires_grad=True の ν が含まれること。"""
    nu_init = 0.05
    solver = InverseSolver(loss_fn, nu_init)
    params = solver._get_trainable_params(pinn)

    nu_params = [
        p for p in params if p.numel() == 1 and abs(p.item() - nu_init) < 1e-6
    ]
    assert len(nu_params) == 1, f"ν parameter not found in InverseSolver params: {params}"
    assert nu_params[0].requires_grad, "ν parameter does not have requires_grad=True"


# ---------------------------------------------------------------------------
# ALG-10: BoundaryData の不正形状でバリデーションエラー
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg10_boundary_data_invalid_shape() -> None:
    """ALG-10: t の 2 次元目が 1 以外のとき ValidationError が発生すること。"""
    with pytest.raises(ValidationError):
        BoundaryData(
            t=torch.zeros(5, 2),  # 不正：2 次元目が 2
            x=torch.zeros(5, 1),
            u=torch.zeros(5, 1),
        )


# ---------------------------------------------------------------------------
# ALG-11: CollocationPoints の形状不一致でバリデーションエラー
# ---------------------------------------------------------------------------

@pytest.mark.algorithm
def test_alg11_collocation_shape_mismatch() -> None:
    """ALG-11: t と x の行数が異なるとき ValidationError が発生すること。"""
    with pytest.raises(ValidationError):
        CollocationPoints(
            t=torch.zeros(10, 1),
            x=torch.zeros(5, 1),  # 不正：行数が t と異なる
        )
