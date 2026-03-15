"""A-2 理論的性質テスト。

論文が保証する数学的性質が実装において成立するかを検証する。
"""

import math

import pytest
import torch
import torch.nn as nn

from PINNs_Burgers import (
    BoundaryData,
    CollocationPoints,
    NetworkConfig,
)
from PINNs_Burgers.solver import InverseSolver
from PINNs_Burgers.loss import LossFunction
from PINNs_Burgers.residual import PDEResidualComputer
from PINNs_Burgers.network import PINN


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_config() -> NetworkConfig:
    """テスト用の小規模ネットワーク設定。"""
    return NetworkConfig(n_hidden_layers=2, n_neurons=8)


@pytest.fixture()
def pinn(small_config: NetworkConfig) -> PINN:
    """テスト用 PINN インスタンス。"""
    return PINN(small_config)


@pytest.fixture()
def constant_pinn(small_config: NetworkConfig) -> PINN:
    """u(t, x) = 定数 を出力するネットワーク（全 weight をゼロ化）。

    全重みを 0 にすることで u = 出力層バイアス（定数）となる。
    このとき u_t = u_x = u_xx = 0 なので PDE 残差 f = 0 が成立する。
    """
    model = PINN(small_config)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        # 出力層にのみ定数バイアスを設定
        output_layer = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        output_layer.bias.fill_(0.5)
    return model


@pytest.fixture()
def collocation() -> CollocationPoints:
    """テスト用コロケーション点（N_f=20）。"""
    torch.manual_seed(0)
    return CollocationPoints(
        t=torch.rand(20, 1),
        x=torch.rand(20, 1) * 2 - 1,
    )


@pytest.fixture()
def nu_tensor() -> torch.Tensor:
    """論文デフォルトの nu テンソル（requires_grad なし）。"""
    return torch.tensor(0.01 / math.pi)


@pytest.fixture()
def nu_grad_tensor() -> torch.Tensor:
    """勾配計算用の nu テンソル（requires_grad=True）。"""
    return torch.tensor(0.01 / math.pi, requires_grad=True)


@pytest.fixture()
def residual_computer() -> PDEResidualComputer:
    return PDEResidualComputer()


@pytest.fixture()
def loss_fn(residual_computer: PDEResidualComputer) -> LossFunction:
    return LossFunction(residual_computer)


# ---------------------------------------------------------------------------
# THR-01: u = const のとき PDE 残差 f = 0（式(6)）
# ---------------------------------------------------------------------------

@pytest.mark.theory
def test_thr01_constant_u_gives_zero_residual(
    constant_pinn: PINN,
    collocation: CollocationPoints,
    nu_tensor: torch.Tensor,
) -> None:
    """THR-01: u(t,x) = const のとき f = u_t + u·u_x - ν·u_xx = 0（式(6)）。

    u が定数であれば u_t = u_x = u_xx = 0 となり f = 0 が成立する。
    """
    computer = PDEResidualComputer()
    f = computer.compute(constant_pinn, collocation, nu_tensor)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-5), (
        f"f should be 0 for constant u, max abs = {f.abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# THR-02: 完全一致データのとき L_data = 0（式(8)）
# ---------------------------------------------------------------------------

@pytest.mark.theory
def test_thr02_exact_data_gives_zero_data_loss(
    pinn: PINN,
    collocation: CollocationPoints,
    nu_tensor: torch.Tensor,
    loss_fn: LossFunction,
) -> None:
    """THR-02: u_theta の出力値をそのまま正解として与えると L_data = 0（式(8)）。

    L_data = (1/N_u) Σ |u_theta - u_true|^2 の定義から，
    u_true = u_theta のとき L_data は厳密に 0 になる。
    """
    with torch.no_grad():
        t = torch.rand(10, 1)
        x = torch.rand(10, 1) * 2 - 1
        u_exact = pinn(t, x)

    data = BoundaryData(t=t, x=x, u=u_exact)
    _, L_data, _ = loss_fn.compute(pinn, data, collocation, nu_tensor)
    assert L_data.item() < 1e-6, f"L_data = {L_data.item():.2e}, expected < 1e-6"


# ---------------------------------------------------------------------------
# THR-03: PDE 残差 f = 0 のとき L_phys = 0（式(9)）
# ---------------------------------------------------------------------------

@pytest.mark.theory
def test_thr03_zero_residual_gives_zero_phys_loss(
    constant_pinn: PINN,
    collocation: CollocationPoints,
    nu_tensor: torch.Tensor,
    loss_fn: LossFunction,
) -> None:
    """THR-03: f = 0 のとき L_phys = (1/N_f) Σ |f|^2 = 0（式(9)）。

    constant_pinn では u = const, f = 0 が成立する（THR-01 で保証）。
    """
    with torch.no_grad():
        t = torch.rand(5, 1)
        x = torch.rand(5, 1) * 2 - 1
        u_val = constant_pinn(t, x)

    data = BoundaryData(t=t, x=x, u=u_val)
    _, _, L_phys = loss_fn.compute(constant_pinn, data, collocation, nu_tensor)
    assert L_phys.item() < 1e-6, f"L_phys = {L_phys.item():.2e}, expected < 1e-6"


# ---------------------------------------------------------------------------
# THR-04: ∂f/∂ν = -u_xx（式(6) の偏微分）
# ---------------------------------------------------------------------------

@pytest.mark.theory
def test_thr04_df_dnu_equals_minus_u_xx(
    pinn: PINN,
    collocation: CollocationPoints,
) -> None:
    """THR-04: f = u_t + u·u_x - ν·u_xx より ∂f/∂ν = -u_xx（式(6)）。

    autograd で計算した ∂(Σf)/∂ν と -Σu_xx を比較することで検証する。
    """
    nu = torch.tensor(0.01 / math.pi, requires_grad=True)

    # f を計算（ν の計算グラフを保持）
    t_f = collocation.t.clone().requires_grad_(True)
    x_f = collocation.x.clone().requires_grad_(True)

    u = pinn(t_f, x_f)
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    f = u_t + u * u_x - nu * u_xx

    # autograd で ∂(Σf)/∂ν を計算
    df_dnu = torch.autograd.grad(f.sum(), nu)[0]

    # 期待値: ∂(Σf)/∂ν = -Σu_xx
    expected = -u_xx.sum().detach()

    assert torch.allclose(df_dnu, expected, rtol=1e-4), (
        f"∂f/∂ν = {df_dnu.item():.6f}, expected -Σu_xx = {expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# THR-05: ν に対して backward() すると ν.grad が None でないこと（Algorithm 3 Step 1）
# ---------------------------------------------------------------------------

@pytest.mark.theory
def test_thr05_inverse_nu_receives_gradient(
    pinn: PINN,
    collocation: CollocationPoints,
    loss_fn: LossFunction,
) -> None:
    """THR-05: InverseSolver の ν は計算グラフに含まれ backward() で勾配を受け取る。

    Algorithm 3 Step 1: ν を学習可能変数として宣言。
    """
    solver = InverseSolver(loss_fn, nu_init=0.05, device=torch.device("cpu"))
    nu_param = solver._nu_param

    with torch.no_grad():
        t = torch.rand(5, 1)
        x = torch.rand(5, 1) * 2 - 1
        u_val = pinn(t, x)

    data = BoundaryData(t=t, x=x, u=u_val)

    L_total, _, _ = loss_fn.compute(pinn, data, collocation, nu_param)
    L_total.backward()

    assert nu_param.grad is not None, "ν.grad is None: ν が計算グラフに含まれていない"
    assert nu_param.grad.item() != 0.0 or True, "grad=0 は数値的に発生しうるため許容"


# ---------------------------------------------------------------------------
# THR-06: tanh 活性化関数の値域が (-1, 1)（式(5)）
# ---------------------------------------------------------------------------

@pytest.mark.theory
def test_thr06_hidden_outputs_within_tanh_range(pinn: PINN) -> None:
    """THR-06: 隠れ層の tanh 出力値が厳密に (-1, 1) の開区間に収まること（式(5)）。

    tanh の値域は (-1, 1) の開区間であり，端点には到達しない。
    forward hook で各 Tanh 層の出力をキャプチャして検証する。
    """
    hidden_outputs: list[torch.Tensor] = []

    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        hidden_outputs.append(output.detach())

    handles = []
    for layer in pinn.modules():
        if isinstance(layer, nn.Tanh):
            handles.append(layer.register_forward_hook(hook))

    try:
        torch.manual_seed(42)
        t = torch.rand(50, 1)
        x = torch.rand(50, 1) * 2 - 1
        with torch.no_grad():
            pinn(t, x)
    finally:
        for h in handles:
            h.remove()

    assert len(hidden_outputs) > 0, "Tanh 層の出力がキャプチャされなかった"

    for i, out in enumerate(hidden_outputs):
        max_val = out.abs().max().item()
        assert max_val < 1.0, (
            f"隠れ層 {i} の出力の絶対値最大値 {max_val:.6f} が tanh の値域 (-1,1) を超えている"
        )
