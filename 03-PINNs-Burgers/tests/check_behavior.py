"""Behavior tests: visual verification of PINNs training results (BHV-01 to BHV-05).

Outputs graphs for human inspection of qualitative properties that cannot be
verified automatically.

Usage:
    uv run python tests/check_behavior.py

Output:
    tests/behavior_output/*.png
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from PINNs_Burgers import (
    BoundaryData,
    BurgersPINNSolver,
    CollocationPoints,
    NetworkConfig,
    PDEConfig,
    TrainingConfig,
)
from PINNs_Burgers.solver import InverseSolver
from PINNs_Burgers.loss import LossFunction
from PINNs_Burgers.residual import PDEResidualComputer
from PINNs_Burgers.network import PINN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "behavior_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi


# ---------------------------------------------------------------------------
# Data loading and sampling
# ---------------------------------------------------------------------------

def load_data(path: str = "data/burgers_shock.mat") -> tuple:
    """Load burgers_shock.mat and return grid arrays and reference solution.

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
    """Randomly sample n_u points from initial and boundary conditions.

    Collects all initial condition (t=0) and boundary condition (x=+-1)
    points, then draws n_u samples. Used for the forward problem.
    """
    rng = np.random.default_rng(seed)

    # Initial condition: u(0, x) = -sin(pi*x)
    t_ic = np.zeros((len(x_grid), 1), dtype=np.float32)
    x_ic = x_grid
    u_ic = usol[:, [0]]

    # Boundary conditions: u(t, -1) = 0, u(t, 1) = 0
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
    """Randomly sample n_u scattered observation points from the full domain.

    The inverse problem requires observations from the entire domain,
    not just boundary and initial conditions.
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
    """Randomly sample n_f collocation points from the interior of the domain."""
    rng = np.random.default_rng(seed)
    x_min, x_max = float(x_grid.min()), float(x_grid.max())
    t_min, t_max = float(t_grid.min()), float(t_grid.max())
    t_f = rng.uniform(t_min, t_max, (n_f, 1)).astype(np.float32)
    x_f = rng.uniform(x_min, x_max, (n_f, 1)).astype(np.float32)
    return CollocationPoints(t=torch.tensor(t_f), x=torch.tensor(x_f))


# ---------------------------------------------------------------------------
# Common configuration
# ---------------------------------------------------------------------------

PDE_CFG = PDEConfig(nu=NU_TRUE, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0)
NET_CFG = NetworkConfig(n_hidden_layers=8, n_neurons=10)
TRAIN_CFG = TrainingConfig(
    n_u=100,
    n_f=10_000,
    lr=1e-2,
    epochs_adam=3079,
    epochs_lbfgs=50,
)
INV_TRAIN_CFG = TrainingConfig(
    n_u=100,
    n_f=10_000,
    lr=1e-3,
    epochs_adam=2_000,
    epochs_lbfgs=0,  # L-BFGS is not used for the inverse problem
)


# ---------------------------------------------------------------------------
# BHV-01: Loss convergence curve
# ---------------------------------------------------------------------------

def bhv01_loss_curve(loss_history: list[float], title_suffix: str = "") -> None:
    """BHV-01: Plot L_total convergence on a log scale.

    Check: L_total at the end of training should be less than 1/10 of the
    initial value.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L_total (log scale)")
    ax.set_title(
        f"BHV-01: Loss convergence curve{title_suffix}\n"
        "Check: final L_total should be < 1/10 of the initial value"
    )
    ax.grid(True, which="both", alpha=0.3)

    if len(loss_history) > 1:
        ratio = loss_history[-1] / (loss_history[0] + 1e-12)
        ax.annotate(
            f"final / initial = {ratio:.3f}",
            xy=(len(loss_history) - 1, loss_history[-1]),
            xytext=(len(loss_history) * 0.6, max(loss_history) * 0.5),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
        )

    fname = os.path.join(OUTPUT_DIR, "bhv01_loss_curve.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-01] saved: {fname}")


# ---------------------------------------------------------------------------
# BHV-02: Predicted solution vs reference solution heatmaps
# ---------------------------------------------------------------------------

def bhv02_solution_heatmap(
    model: PINN,
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """BHV-02: Side-by-side heatmaps of predicted u_theta and reference usol.

    Check: shock wave position and shape should visually match the reference.
    """
    device = next(model.parameters()).device
    x_flat = X_mesh.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel().reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        u_pred = model(
            torch.tensor(t_flat, device=device), torch.tensor(x_flat, device=device)
        ).cpu().numpy().reshape(X_mesh.shape)

    vmin = float(usol.min())
    vmax = float(usol.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [u_pred, usol], ["Predicted u_theta", "Reference u_ref"]):
        im = ax.pcolormesh(
            t_grid.ravel(), x_grid.ravel(), data,
            cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
        )
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(label)

    fig.suptitle(
        "BHV-02: Predicted solution vs reference solution (cf. Paper Figure 1)\n"
        "Check: shock wave position and shape should visually match"
    )
    fname = os.path.join(OUTPUT_DIR, "bhv02_solution_heatmap.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-02] saved: {fname}")


# ---------------------------------------------------------------------------
# BHV-03: Pointwise absolute error heatmap
# ---------------------------------------------------------------------------

def bhv03_error_heatmap(
    model: PINN,
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """BHV-03: Heatmap of pointwise absolute error |u_theta - u_ref|.

    Check: error should be close to 0 (light color) across most of the domain,
    except near the shock wave.
    """
    device = next(model.parameters()).device
    x_flat = X_mesh.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel().reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        u_pred = model(
            torch.tensor(t_flat, device=device), torch.tensor(x_flat, device=device)
        ).cpu().numpy().reshape(X_mesh.shape)

    error = np.abs(u_pred - usol)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(
        t_grid.ravel(), x_grid.ravel(), error,
        cmap="hot_r", vmin=0, shading="auto",
    )
    fig.colorbar(im, ax=ax, label="|u_theta - u_ref|")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(
        f"BHV-03: Pointwise absolute error |u_theta - u_ref| (max = {error.max():.4f})\n"
        "Check: error should be light (near 0) except near the shock wave"
    )
    fname = os.path.join(OUTPUT_DIR, "bhv03_error_heatmap.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-03] saved: {fname}")


# ---------------------------------------------------------------------------
# BHV-04: nu estimation trajectory (inverse problem)
# ---------------------------------------------------------------------------

def bhv04_nu_history(nu_history: list[float]) -> None:
    """BHV-04: Plot the trajectory of the estimated nu vs the true value.

    Check: the estimated nu* should converge to within +-5% of the true value
    0.01/pi by the end of training (Algorithm 3, Eq. 11).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nu_history, linewidth=1.0, label="Estimated nu")
    ax.axhline(NU_TRUE, color="red", linestyle="--", linewidth=1.5,
               label=f"True nu* = {NU_TRUE:.5f}")
    ax.fill_between(
        range(len(nu_history)),
        NU_TRUE * 0.95, NU_TRUE * 1.05,
        alpha=0.15, color="red", label="True value +/-5%",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("nu")
    ax.set_title(
        "BHV-04: nu estimation trajectory (Algorithm 3, Eq. 11)\n"
        f"Check: final nu* should converge within +-5% of true value {NU_TRUE:.5f}\n"
        f"Final estimate: {nu_history[-1]:.5f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = os.path.join(OUTPUT_DIR, "bhv04_nu_history.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-04] saved: {fname}")


# ---------------------------------------------------------------------------
# BHV-05: Initial condition u(0, x) = -sin(pi*x) reproduction
# ---------------------------------------------------------------------------

def bhv05_initial_condition(
    model: PINN,
    x_grid: np.ndarray,
    usol: np.ndarray,
) -> None:
    """BHV-05: Compare predicted u(0, x) against the reference -sin(pi*x).

    Check: the two curves should nearly overlap (Paper Section 5.1, Eq. 2-4).
    """
    device = next(model.parameters()).device
    x_vals = x_grid.ravel().reshape(-1, 1).astype(np.float32)
    t_vals = np.zeros_like(x_vals)

    with torch.no_grad():
        u_pred_ic = model(
            torch.tensor(t_vals, device=device), torch.tensor(x_vals, device=device)
        ).cpu().numpy().ravel()

    u_ref_ic = usol[:, 0]  # reference at t=0: -sin(pi*x)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_grid.ravel(), u_ref_ic, "b-", linewidth=2.0, label="Reference -sin(pi*x)")
    ax.plot(x_grid.ravel(), u_pred_ic, "r--", linewidth=1.5, label="Predicted u_theta(0, x)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(0, x)")
    ax.set_title(
        "BHV-05: Initial condition u(0,x) = -sin(pi*x) reproduction\n"
        "(Paper Section 5.1, Eq. 2-4)  Check: curves should nearly overlap"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = os.path.join(OUTPUT_DIR, "bhv05_initial_condition.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  [BHV-05] saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all behavior tests and save graphs."""
    print("=== Behavior tests: loading data ===")
    x_grid, t_grid, X_mesh, T_mesh, usol = load_data()
    print(f"  x: {x_grid.shape}, t: {t_grid.shape}, usol: {usol.shape}")

    boundary_data = sample_boundary_data(x_grid, t_grid, usol, n_u=TRAIN_CFG.n_u)
    collocation = sample_collocation(x_grid, t_grid, n_f=TRAIN_CFG.n_f)
    print(f"  BoundaryData: {boundary_data.t.shape}, CollocationPoints: {collocation.t.shape}")

    # -----------------------------------------------------------------------
    # Forward problem (BHV-01 to BHV-03, BHV-05)
    # -----------------------------------------------------------------------
    print("\n=== Forward problem training (Adam x2000 + L-BFGS x50) ===")
    forward_solver = BurgersPINNSolver(PDE_CFG, NET_CFG, TRAIN_CFG)
    forward_result = forward_solver.solve_forward(boundary_data, collocation)
    print(f"  Training done: {len(forward_result.loss_history)} steps")
    print(f"  Final loss: {forward_result.loss_history[-1]:.4e}")

    print("\n=== BHV-01: Loss convergence curve ===")
    bhv01_loss_curve(forward_result.loss_history, title_suffix=" (forward problem)")

    print("\n=== BHV-02: Predicted vs reference solution heatmaps ===")
    bhv02_solution_heatmap(forward_result.model, X_mesh, T_mesh, usol, t_grid, x_grid)

    print("\n=== BHV-03: Pointwise absolute error heatmap ===")
    bhv03_error_heatmap(forward_result.model, X_mesh, T_mesh, usol, t_grid, x_grid)

    print("\n=== BHV-05: Initial condition u(0, x) = -sin(pi*x) reproduction ===")
    bhv05_initial_condition(forward_result.model, x_grid, usol)

    # -----------------------------------------------------------------------
    # Inverse problem (BHV-04): use InverseSolver directly to access nu_history
    # Full-domain scattered observations are used for nu identification.
    # -----------------------------------------------------------------------
    print("\n=== Inverse problem training (Adam x2000, nu_init = 0.005) ===")
    observations = sample_scattered_observations(X_mesh, T_mesh, usol, n_u=INV_TRAIN_CFG.n_u)
    print(f"  Scattered observations: {observations.t.shape[0]} points (full domain)")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn_inv = PINN(NET_CFG).to(device)
    residual_computer = PDEResidualComputer()
    loss_fn = LossFunction(residual_computer)
    inv_solver = InverseSolver(loss_fn, nu_init=0.005, device=device)
    inv_solver.train(pinn_inv, observations, collocation, INV_TRAIN_CFG)

    print(f"  True   nu* = {NU_TRUE:.5f}")
    print(f"  Est.   nu  = {inv_solver.nu:.5f}")
    print(f"  Error      = {abs(inv_solver.nu - NU_TRUE) / NU_TRUE * 100:.1f}%")

    print("\n=== BHV-04: nu estimation trajectory ===")
    bhv04_nu_history(inv_solver._nu_history)

    print(f"\n=== Done: all graphs saved to {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    main()
