"""Example: Inverse problem -- identify kinematic viscosity nu from observations.

Demonstrates how to use BurgersPINNSolver.solve_inverse() to simultaneously
learn u(t, x) and estimate the unknown viscosity nu from scattered data.

Inverse problem (Eq. 11):
    Minimize L(theta, nu) = L_u + L_f over (theta, nu)
    where nu is treated as a learnable parameter alongside network weights.

Usage:
    cd 03-PINNs-Burgers
    uv run python example/inverse_problem.py

Output:
    example/output/inverse_loss_curve.png
    example/output/inverse_nu_trajectory.png
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

NU_TRUE = 0.01 / math.pi   # ground-truth kinematic viscosity (Paper Section 5.1)
NU_INIT = 0.005             # initial guess for nu (deliberately wrong)


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


def sample_scattered_observations(
    X_mesh: np.ndarray,
    T_mesh: np.ndarray,
    usol: np.ndarray,
    n_u: int,
    seed: int = 2,
) -> BoundaryData:
    """Randomly sample n_u scattered observation points from the full domain.

    The inverse problem uses observations from the entire (t, x) domain,
    unlike the forward problem which only uses IC/BC data.
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
        "Inverse Problem: Training Loss Curve (Eq. 11)\n"
        f"Final loss = {loss_history[-1]:.4e} "
        f"(reduced to {loss_history[-1] / loss_history[0]:.2%} of initial)"
    )
    path = os.path.join(OUTPUT_DIR, "inverse_loss_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_nu_trajectory(
    nu_history: list[float],
    nu_true: float,
    nu_init: float,
) -> None:
    """Plot nu estimation trajectory vs the true value (Algorithm 3, Eq. 11)."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    # Estimated nu trajectory
    sns.lineplot(
        x=range(len(nu_history)),
        y=nu_history,
        ax=ax,
        linewidth=1.5,
        label="Estimated nu",
    )

    # True value and +-5% band
    ax.axhline(nu_true, color="red", linestyle="--", linewidth=1.5,
               label=f"True nu* = {nu_true:.5f}")
    ax.fill_between(
        range(len(nu_history)),
        nu_true * 0.95, nu_true * 1.05,
        alpha=0.15, color="red", label="True value +/-5%",
    )

    # Initial guess marker
    ax.axhline(nu_init, color="gray", linestyle=":", linewidth=1.2,
               label=f"Initial guess = {nu_init:.5f}")

    nu_final = nu_history[-1]
    rel_error = abs(nu_final - nu_true) / nu_true * 100

    ax.set_xlabel("Epoch")
    ax.set_ylabel("nu")
    ax.set_title(
        "Inverse Problem: Kinematic Viscosity Estimation (Algorithm 3, Eq. 11)\n"
        f"Final nu* = {nu_final:.5f}  (true = {nu_true:.5f},  "
        f"relative error = {rel_error:.2f}%)"
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "inverse_nu_trajectory.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run inverse problem example."""
    print("=== Inverse Problem Example ===")
    print(f"  True nu*   = {NU_TRUE:.6f}  (0.01 / pi)")
    print(f"  Initial nu = {NU_INIT:.6f}  (deliberately wrong initial guess)\n")

    # --- Load reference data ---
    print("Loading reference data from data/burgers_shock.mat ...")
    x_grid, t_grid, X_mesh, T_mesh, usol = load_reference_data()
    print(f"  x: {x_grid.shape}, t: {t_grid.shape}, usol: {usol.shape}")

    # --- Sample training data ---
    # The inverse problem uses scattered observations from the full domain
    # (not just IC/BC data like the forward problem).
    n_u = 100   # number of scattered observation points (N_u)
    n_f = 10_000  # number of collocation points (N_f)

    observations = sample_scattered_observations(X_mesh, T_mesh, usol, n_u=n_u)
    collocation = sample_collocation_points(x_grid, t_grid, n_f=n_f)
    print(f"\n  Scattered observations: {observations.t.shape[0]} points (N_u = {n_u})")
    print(f"  CollocationPoints: {collocation.t.shape[0]} points (N_f = {n_f})")

    # --- Configure solver ---
    # PDEConfig.nu is ignored by solve_inverse(); nu_init is used instead.
    pde_cfg = PDEConfig(
        nu=NU_TRUE,          # ignored for the inverse problem
        x_min=-1.0, x_max=1.0,
        t_min=0.0,  t_max=1.0,
    )
    net_cfg = NetworkConfig(
        n_hidden_layers=5,   # L: 5 hidden layers (Paper Section 6.1)
        n_neurons=20,        # N: 20 neurons per layer
    )
    train_cfg = TrainingConfig(
        n_u=n_u,
        n_f=n_f,
        lr=1e-3,
        epochs_adam=2_000,   # Adam phase only (no L-BFGS for inverse, Algorithm 3)
        epochs_lbfgs=0,      # L-BFGS is not used for the inverse problem
    )

    # --- Solve ---
    print(f"\nTraining (Adam x{train_cfg.epochs_adam}, nu optimized jointly with theta) ...")
    solver = BurgersPINNSolver(pde_cfg, net_cfg, train_cfg)
    result = solver.solve_inverse(observations, collocation, nu_init=NU_INIT)

    nu_est = result.nu
    rel_error = abs(nu_est - NU_TRUE) / NU_TRUE * 100
    print(f"\n  True   nu* = {NU_TRUE:.6f}")
    print(f"  Est.   nu* = {nu_est:.6f}")
    print(f"  Relative error = {rel_error:.2f}%")
    print(f"  Final loss: {result.loss_history[-1]:.4e}")

    # --- Visualize ---
    print("\nGenerating plots ...")
    plot_loss_curve(result.loss_history)

    # Access the internal nu history via the solver internals
    # (BurgersPINNSolver wraps InverseSolver; nu_history is tracked per step)
    from importlib import import_module as _imp
    _solver_mod = _imp("PINNs_Burgers.solver")
    _loss_mod = _imp("PINNs_Burgers.loss")
    _residual_mod = _imp("PINNs_Burgers.residual")
    _network_mod = _imp("PINNs_Burgers.network")

    PINN = _network_mod.PINN
    PDEResidualComputer = _residual_mod.PDEResidualComputer
    LossFunction = _loss_mod.LossFunction
    InverseSolver = _solver_mod.InverseSolver

    # Rerun with direct InverseSolver to capture per-epoch nu trajectory.
    print("  Re-running inverse solver to capture nu trajectory ...")
    pinn_model = PINN(net_cfg)
    inv_solver = InverseSolver(LossFunction(PDEResidualComputer()), nu_init=NU_INIT)
    inv_solver.train(pinn_model, observations, collocation, train_cfg)

    plot_nu_trajectory(inv_solver._nu_history, NU_TRUE, NU_INIT)

    print(f"\nDone. Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
