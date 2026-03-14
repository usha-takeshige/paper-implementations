"""Example: Forward problem -- solve Burgers equation with known viscosity.

Demonstrates how to use BurgersPINNSolver.solve_forward() to learn u(t, x)
from initial and boundary condition data.

Burgers equation (Eq. 1):
    u_t + u * u_x - nu * u_xx = 0   on [-1, 1] x [0, 1]
    u(0, x) = -sin(pi * x)
    u(t, -1) = u(t, 1) = 0

Usage:
    cd 03-PINNs-Burgers
    uv run python example/forward_problem.py

Output:
    example/output/forward_loss_curve.png
    example/output/forward_solution_heatmap.png
    example/output/forward_error_heatmap.png
"""

import math
import os
import numpy as np
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from PINNs_Burgers import (
    BoundaryData,
    BurgersPINNSolver,
    CollocationPoints,
    NetworkConfig,
    PDEConfig,
    TrainingConfig,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi  # kinematic viscosity (Paper Section 5.1)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_reference_data(path: str = "data/burgers_shock.mat") -> tuple:
    """Load reference solution from burgers_shock.mat.

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
    """Sample n_u points from initial and boundary conditions (Eq. 8).

    Collects all IC (t=0) and BC (x=+-1) points, then randomly draws n_u.
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


def sample_collocation_points(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    n_f: int,
    seed: int = 1,
) -> CollocationPoints:
    """Uniformly sample n_f collocation points from the domain interior (Eq. 9)."""
    rng = np.random.default_rng(seed)
    x_min, x_max = float(x_grid.min()), float(x_grid.max())
    t_min, t_max = float(t_grid.min()), float(t_grid.max())
    t_f = rng.uniform(t_min, t_max, (n_f, 1)).astype(np.float32)
    x_f = rng.uniform(x_min, x_max, (n_f, 1)).astype(np.float32)
    return CollocationPoints(t=torch.tensor(t_f), x=torch.tensor(x_f))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_loss_curve(loss_history: list[float]) -> None:
    """Plot training loss on a log scale."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=range(len(loss_history)), y=loss_history, ax=ax, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L_total (log scale)")
    ax.set_title(
        "Forward Problem: Training Loss Curve\n"
        f"Final loss = {loss_history[-1]:.4e} "
        f"(reduced to {loss_history[-1] / loss_history[0]:.2%} of initial)"
    )
    path = os.path.join(OUTPUT_DIR, "forward_loss_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_solution_heatmap(
    model: torch.nn.Module,
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """Plot predicted u_theta and reference usol side by side."""
    x_flat = X_mesh.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel().reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        u_pred = model(
            torch.tensor(t_flat), torch.tensor(x_flat)
        ).numpy().reshape(X_mesh.shape)

    vmin = float(usol.min())
    vmax = float(usol.max())

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, data, label in zip(
        axes,
        [u_pred, usol],
        ["Predicted u_theta(t, x)", "Reference u_ref(t, x)"],
    ):
        im = ax.pcolormesh(
            t_grid.ravel(), x_grid.ravel(), data,
            cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
        )
        fig.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(label)

    fig.suptitle(
        "Forward Problem: Predicted vs Reference Solution (cf. Paper Figure 1)\n"
        "The shock wave position and shape should match between the two panels."
    )
    path = os.path.join(OUTPUT_DIR, "forward_solution_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_heatmap(
    model: torch.nn.Module,
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """Plot pointwise absolute error |u_theta - u_ref|."""
    x_flat = X_mesh.ravel().reshape(-1, 1).astype(np.float32)
    t_flat = T_mesh.ravel().reshape(-1, 1).astype(np.float32)

    with torch.no_grad():
        u_pred = model(
            torch.tensor(t_flat), torch.tensor(x_flat)
        ).numpy().reshape(X_mesh.shape)

    error = np.abs(u_pred - usol)
    rel_l2 = np.linalg.norm(error) / (np.linalg.norm(usol) + 1e-12)

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(
        t_grid.ravel(), x_grid.ravel(), error,
        cmap="hot_r", vmin=0, shading="auto",
    )
    fig.colorbar(im, ax=ax, label="|u_theta - u_ref|")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(
        f"Forward Problem: Pointwise Absolute Error (max = {error.max():.4f}, "
        f"relative L2 = {rel_l2:.4f})\n"
        "Error should be near 0 (dark) across most of the domain."
    )
    path = os.path.join(OUTPUT_DIR, "forward_error_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run forward problem example."""
    print("=== Forward Problem Example ===")
    print(f"  nu = {NU_TRUE:.6f}  (0.01 / pi, Paper Section 5.1)\n")

    # --- Load reference data ---
    print("Loading reference data from data/burgers_shock.mat ...")
    x_grid, t_grid, X_mesh, T_mesh, usol = load_reference_data()
    print(f"  x: {x_grid.shape}, t: {t_grid.shape}, usol: {usol.shape}")

    # --- Sample training data ---
    n_u = 100   # number of boundary/IC data points (N_u)
    n_f = 10_000  # number of collocation points (N_f)

    boundary_data = sample_boundary_data(x_grid, t_grid, usol, n_u=n_u)
    collocation = sample_collocation_points(x_grid, t_grid, n_f=n_f)
    print(f"\n  BoundaryData: {boundary_data.t.shape[0]} points (N_u = {n_u})")
    print(f"  CollocationPoints: {collocation.t.shape[0]} points (N_f = {n_f})")

    # --- Configure solver ---
    pde_cfg = PDEConfig(
        nu=NU_TRUE,
        x_min=-1.0, x_max=1.0,
        t_min=0.0,  t_max=1.0,
    )
    net_cfg = NetworkConfig(
        n_hidden_layers=5,  # L: 5 hidden layers (Paper Section 6.1)
        n_neurons=20,       # N: 20 neurons per layer
    )
    train_cfg = TrainingConfig(
        n_u=n_u,
        n_f=n_f,
        lr=1e-3,
        epochs_adam=2_000,   # Adam phase (Paper Algorithm 2)
        epochs_lbfgs=50,     # L-BFGS phase (Paper Algorithm 2)
    )

    # --- Solve ---
    print("\nTraining (Adam x2000 + L-BFGS x50) ...")
    solver = BurgersPINNSolver(pde_cfg, net_cfg, train_cfg)
    result = solver.solve_forward(boundary_data, collocation)
    print(f"  Steps: {len(result.loss_history)}")
    print(f"  Final loss: {result.loss_history[-1]:.4e}")

    # --- Visualize ---
    print("\nGenerating plots ...")
    plot_loss_curve(result.loss_history)
    plot_solution_heatmap(result.model, X_mesh, T_mesh, usol, t_grid, x_grid)
    plot_error_heatmap(result.model, X_mesh, T_mesh, usol, t_grid, x_grid)

    print(f"\nDone. Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
