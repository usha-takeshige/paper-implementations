"""振る舞いテスト：グラフ出力による確認（BHV-01〜05）。

自動判定が困難な視覚的・定性的な性質をグラフで出力し，人間が目視確認する。

実行方法:
    uv run python tests/check_behavior.py

出力先:
    tests/behavior_output/*.png
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

sys.path.insert(0, "src")
from importlib import import_module

pkg = import_module("03_PINNs_Burgers")
BoundaryData = pkg.BoundaryData
BurgersPINNSolver = pkg.BurgersPINNSolver
CollocationPoints = pkg.CollocationPoints
NetworkConfig = pkg.NetworkConfig
PDEConfig = pkg.PDEConfig
TrainingConfig = pkg.TrainingConfig

_solver_mod = import_module("03_PINNs_Burgers.solver")
_loss_mod = import_module("03_PINNs_Burgers.loss")
_residual_mod = import_module("03_PINNs_Burgers.residual")
_network_mod = import_module("03_PINNs_Burgers.network")

InverseSolver = _solver_mod.InverseSolver
LossFunction = _loss_mod.LossFunction
PDEResidualComputer = _residual_mod.PDEResidualComputer
PINN = _network_mod.PINN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "behavior_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi


# ---------------------------------------------------------------------------
# データ読み込みとサンプリング
# ---------------------------------------------------------------------------

def load_data(path: str = "data/burgers_shock.mat") -> tuple:
    """burgers_shock.mat を読み込みグリッドと参照解を返す。

    Returns:
        (x_grid, t_grid, X_mesh, T_mesh, usol)
        x_grid: (256, 1), t_grid: (100, 1), usol: (256, 100)
    """
    raw = scipy.io.loadmat(path)
    x_grid = raw["x"].astype(np.float32)   # (256, 1)
    t_grid = raw["t"].astype(np.float32)   # (100, 1)
    usol = raw["usol"].astype(np.float32)  # (256, 100)
    X_mesh, T_mesh = np.meshgrid(x_grid.ravel(), t_grid.ravel(), indexing="ij")
    return x_grid, t_grid, X_mesh, T_mesh, usol


def sample_boundary_data(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    usol: np.ndarray,
    n_u: int,
    seed: int = 0,
) -> BoundaryData:
    """初期・境界条件データを n_u 点ランダムサンプリングする。

    初期条件（t=0）・境界条件（x=±1）を全点収集してからサンプリングする。
    順問題で使用する。
    """
    rng = np.random.default_rng(seed)

    # 初期条件: u(0, x) = -sin(πx)
    t_ic = np.zeros((len(x_grid), 1), dtype=np.float32)
    x_ic = x_grid
    u_ic = usol[:, [0]]

    # 境界条件: u(t, -1) = 0, u(t, 1) = 0
    t_bc = t_grid
    x_bc_left = np.full_like(t_grid, x_grid[0, 0])
    u_bc_left = usol[[0], :].T

    x_bc_right = np.full_like(t_grid, x_grid[-1, 0])
    u_bc_right = usol[[-1], :].T

    t_all = np.vstack([t_ic, t_bc, t_bc])
    x_all = np.vstack([x_ic, x_bc_left, x_bc_right])
    u_all = np.vstack([u_ic, u_bc_left, u_bc_right])

    idx = rng.choice(len(t_all), size=min(n_u, len(t_all)), replace=False)

    return BoundaryData(
        t=torch.tensor(t_all[idx]),
        x=torch.tensor(x_all[idx]),
        u=torch.tensor(u_all[idx]),
    )


def sample_scattered_observations(
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    n_u: int,
    seed: int = 2,
) -> BoundaryData:
    """全領域から散在観測データを n_u 点ランダムサンプリングする。

    逆問題では境界・初期条件だけでなく全領域の観測値が必要。
    """
    rng = np.random.default_rng(seed)
    n_total = X_mesh.size
    idx = rng.choice(n_total, size=min(n_u, n_total), replace=False)

    x_flat = X_mesh.ravel()[idx].reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel()[idx].reshape(-1, 1).astype(np.float32)
    u_flat = usol.ravel()[idx].reshape(-1, 1).astype(np.float32)

    return BoundaryData(
        t=torch.tensor(t_flat),
        x=torch.tensor(x_flat),
        u=torch.tensor(u_flat),
    )


def sample_collocation(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    n_f: int,
    seed: int = 1,
) -> CollocationPoints:
    """コロケーション点をドメイン内にランダムサンプリングする。"""
    rng = np.random.default_rng(seed)
    x_min, x_max = float(x_grid.min()), float(x_grid.max())
    t_min, t_max = float(t_grid.min()), float(t_grid.max())
    t_f = rng.uniform(t_min, t_max, (n_f, 1)).astype(np.float32)
    x_f = rng.uniform(x_min, x_max, (n_f, 1)).astype(np.float32)
    return CollocationPoints(t=torch.tensor(t_f), x=torch.tensor(x_f))


# ---------------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------------

PDE_CFG = PDEConfig(nu=NU_TRUE, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0)
NET_CFG = NetworkConfig(n_hidden_layers=4, n_neurons=20)
TRAIN_CFG = TrainingConfig(
    n_u=100,
    n_f=10_000,
    lr=1e-3,
    epochs_adam=2_000,
    epochs_lbfgs=50,
)
INV_TRAIN_CFG = TrainingConfig(
    n_u=100,
    n_f=10_000,
    lr=1e-3,
    epochs_adam=2_000,
    epochs_lbfgs=0,  # 逆問題では L-BFGS 不使用
)


# ---------------------------------------------------------------------------
# BHV-01: 損失収束曲線
# ---------------------------------------------------------------------------

def bhv01_loss_curve(loss_history: list[float], title_suffix: str = "") -> None:
    """BHV-01: L_total の収束曲線を対数スケールでプロットする。

    確認観点:
        損失が学習終了時に初期値の 1/10 以下に収束していること。
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L_total (log scale)")
    ax.set_title(
        f"BHV-01: 損失収束曲線{title_suffix}\n"
        "確認観点: 学習終了時の L_total が初期値の 1/10 以下に収束すること"
    )
    ax.grid(True, which="both", alpha=0.3)

    if len(loss_history) > 1:
        ratio = loss_history[-1] / (loss_history[0] + 1e-12)
        ax.annotate(
            f"最終損失 / 初期損失 = {ratio:.3f}",
            xy=(len(loss_history) - 1, loss_history[-1]),
            xytext=(len(loss_history) * 0.6, max(loss_history) * 0.5),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
        )

    fname = os.path.join(OUTPUT_DIR, "bhv01_loss_curve.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-01] 保存: {fname}")


# ---------------------------------------------------------------------------
# BHV-02: 推定解と参照解のヒートマップ
# ---------------------------------------------------------------------------

def bhv02_solution_heatmap(
    model: PINN,
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """BHV-02: 推定解 u_theta と参照解 usol の等高線ヒートマップ。

    確認観点:
        衝撃波の位置・形状が参照解と概ね一致すること。
    """
    x_flat = X_mesh.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel().reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        u_pred = model(
            torch.tensor(t_flat), torch.tensor(x_flat)
        ).numpy().reshape(X_mesh.shape)

    vmin = float(usol.min())
    vmax = float(usol.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [u_pred, usol], ["推定解 u_θ", "参照解 u_ref"]):
        im = ax.pcolormesh(
            t_grid.ravel(), x_grid.ravel(), data,
            cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
        )
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(label)

    fig.suptitle(
        "BHV-02: 推定解 vs 参照解（論文 Figure 1 に対応）\n"
        "確認観点: 衝撃波の位置・形状が概ね一致すること"
    )
    fname = os.path.join(OUTPUT_DIR, "bhv02_solution_heatmap.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-02] 保存: {fname}")


# ---------------------------------------------------------------------------
# BHV-03: 点別絶対誤差ヒートマップ
# ---------------------------------------------------------------------------

def bhv03_error_heatmap(
    model: PINN,
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """BHV-03: |u_theta - u_ref| の誤差場ヒートマップ。

    確認観点:
        誤差が全体的に 0 に近い薄い色であること（衝撃波近傍を除く）。
    """
    x_flat = X_mesh.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel().reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        u_pred = model(
            torch.tensor(t_flat), torch.tensor(x_flat)
        ).numpy().reshape(X_mesh.shape)

    error = np.abs(u_pred - usol)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(
        t_grid.ravel(), x_grid.ravel(), error,
        cmap="hot_r", vmin=0, shading="auto",
    )
    fig.colorbar(im, ax=ax, label="|u_θ - u_ref|")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(
        f"BHV-03: 点別絶対誤差 |u_θ - u_ref|（最大誤差 = {error.max():.4f}）\n"
        "確認観点: 衝撃波近傍を除き誤差が 0 に近い薄い色であること"
    )
    fname = os.path.join(OUTPUT_DIR, "bhv03_error_heatmap.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-03] 保存: {fname}")


# ---------------------------------------------------------------------------
# BHV-04: 逆問題における ν の推移
# ---------------------------------------------------------------------------

def bhv04_nu_history(nu_history: list[float]) -> None:
    """BHV-04: ν の推定値がエポックごとに真値 0.01/π に近づくかを確認する。

    確認観点:
        学習終了時の ν* が真値 ± 5% 以内に収束すること。
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nu_history, linewidth=1.0, label="ν 推定値")
    ax.axhline(NU_TRUE, color="red", linestyle="--", linewidth=1.5, label=f"真値 ν* = {NU_TRUE:.5f}")
    ax.fill_between(
        range(len(nu_history)),
        NU_TRUE * 0.95, NU_TRUE * 1.05,
        alpha=0.15, color="red", label="真値 ±5%",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ν")
    ax.set_title(
        "BHV-04: 逆問題における ν の推移（Algorithm 3，式(11)）\n"
        f"確認観点: 学習終了時の ν* が真値 {NU_TRUE:.5f} の ±5% 以内に収束すること\n"
        f"最終推定値: {nu_history[-1]:.5f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = os.path.join(OUTPUT_DIR, "bhv04_nu_history.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-04] 保存: {fname}")


# ---------------------------------------------------------------------------
# BHV-05: 初期条件 u(0, x) = -sin(πx) の再現確認
# ---------------------------------------------------------------------------

def bhv05_initial_condition(
    model: PINN,
    x_grid: np.ndarray,
    usol: np.ndarray,
) -> None:
    """BHV-05: t=0 断面での推定解と初期条件 -sin(πx) を比較する。

    確認観点:
        推定曲線と参照曲線が概ね重なること。
    """
    x_vals = x_grid.ravel().reshape(-1, 1).astype(np.float32)
    t_vals = np.zeros_like(x_vals)

    with torch.no_grad():
        u_pred_ic = model(
            torch.tensor(t_vals), torch.tensor(x_vals)
        ).numpy().ravel()

    u_ref_ic = usol[:, 0]  # t=0 の参照解（= -sin(πx)）

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_grid.ravel(), u_ref_ic, "b-", linewidth=2.0, label="参照解 -sin(πx)")
    ax.plot(x_grid.ravel(), u_pred_ic, "r--", linewidth=1.5, label="推定解 u_θ(0, x)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(0, x)")
    ax.set_title(
        "BHV-05: 初期条件 u(0, x) = -sin(πx) の再現（論文 Section 5.1, 式(2)-(4)）\n"
        "確認観点: 推定曲線と参照曲線が概ね重なること"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = os.path.join(OUTPUT_DIR, "bhv05_initial_condition.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-05] 保存: {fname}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    """全振る舞いテストを実行する。"""
    print("=== 振る舞いテスト: データ読み込み ===")
    x_grid, t_grid, X_mesh, T_mesh, usol = load_data()
    print(f"  x: {x_grid.shape}, t: {t_grid.shape}, usol: {usol.shape}")

    boundary_data = sample_boundary_data(x_grid, t_grid, usol, n_u=TRAIN_CFG.n_u)
    collocation = sample_collocation(x_grid, t_grid, n_f=TRAIN_CFG.n_f)
    print(f"  BoundaryData: {boundary_data.t.shape}, CollocationPoints: {collocation.t.shape}")

    # -----------------------------------------------------------------------
    # 順問題（BHV-01〜03，BHV-05）
    # -----------------------------------------------------------------------
    print("\n=== 順問題の学習（Adam×2000 + L-BFGS×50） ===")
    forward_solver = BurgersPINNSolver(PDE_CFG, NET_CFG, TRAIN_CFG)
    forward_result = forward_solver.solve_forward(boundary_data, collocation)
    print(f"  学習完了: 損失履歴 {len(forward_result.loss_history)} ステップ")
    print(f"  最終損失: {forward_result.loss_history[-1]:.4e}")

    print("\n=== BHV-01: 損失収束曲線 ===")
    bhv01_loss_curve(forward_result.loss_history, title_suffix="（順問題）")

    print("\n=== BHV-02: 推定解 vs 参照解ヒートマップ ===")
    bhv02_solution_heatmap(forward_result.model, X_mesh, T_mesh, usol, t_grid, x_grid)

    print("\n=== BHV-03: 点別絶対誤差ヒートマップ ===")
    bhv03_error_heatmap(forward_result.model, X_mesh, T_mesh, usol, t_grid, x_grid)

    print("\n=== BHV-05: 初期条件 u(0, x) = -sin(πx) の再現確認 ===")
    bhv05_initial_condition(forward_result.model, x_grid, usol)

    # -----------------------------------------------------------------------
    # 逆問題（BHV-04）: InverseSolver を直接使用して nu_history を取得
    # 逆問題では全領域の散在観測データを使用する
    # -----------------------------------------------------------------------
    print("\n=== 逆問題の学習（Adam×2000，ν 初期値 = 0.005） ===")
    observations = sample_scattered_observations(X_mesh, T_mesh, usol, n_u=INV_TRAIN_CFG.n_u)
    print(f"  散在観測データ: {observations.t.shape[0]} 点（全領域サンプリング）")
    pinn_inv = PINN(NET_CFG)
    residual_computer = PDEResidualComputer()
    loss_fn = LossFunction(residual_computer)
    inv_solver = InverseSolver(loss_fn, nu_init=0.005)
    inv_solver.train(pinn_inv, observations, collocation, INV_TRAIN_CFG)

    print(f"  真値 ν* = {NU_TRUE:.5f}")
    print(f"  推定値 ν  = {inv_solver.nu:.5f}")
    print(f"  誤差率     = {abs(inv_solver.nu - NU_TRUE) / NU_TRUE * 100:.1f}%")

    print("\n=== BHV-04: ν の推移グラフ ===")
    bhv04_nu_history(inv_solver._nu_history)

    print(f"\n=== 完了: 全グラフを {OUTPUT_DIR}/ に保存しました ===")


if __name__ == "__main__":
    main()
