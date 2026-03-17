"""PINN solution heatmap visualization.

Re-trains a BurgersPINNSolver with the best hyperparameters found by an
optimizer and plots the predicted solution against the reference solution.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from PINNs_Burgers import (
    BurgersPINNSolver,
    BoundaryData,
    CollocationPoints,
    NetworkConfig,
    PDEConfig,
    TrainingConfig,
)


def plot_best_solution_heatmap(
    best_params: dict[str, float | int],
    pde_config: PDEConfig,
    training_config: TrainingConfig,
    boundary_data: BoundaryData,
    collocation: CollocationPoints,
    x_mesh: np.ndarray,
    t_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    output_path: str,
    *,
    title_prefix: str = "",
) -> None:
    """Re-train PINN with best hyperparameters and plot predicted vs reference.

    Parameters
    ----------
    best_params:
        Hyperparameter dict with keys ``n_hidden_layers``, ``n_neurons``,
        ``lr``, and ``epochs_adam``.
    pde_config:
        PDE configuration (domain, viscosity, etc.).
    training_config:
        Training configuration used as a template; ``n_u``, ``n_f``, and
        ``epochs_lbfgs`` are taken from this object while ``lr`` and
        ``epochs_adam`` are overridden by ``best_params``.
    boundary_data:
        Pre-sampled boundary condition data.
    collocation:
        Pre-sampled collocation points.
    x_mesh:
        2-D spatial grid (output of ``np.meshgrid``), shape ``(Nx, Nt)``.
    t_mesh:
        2-D time grid (output of ``np.meshgrid``), shape ``(Nx, Nt)``.
    usol:
        Reference solution array, shape ``(Nx, Nt)``.
    t_grid:
        1-D time grid, shape ``(Nt,)``.
    x_grid:
        1-D spatial grid, shape ``(Nx,)``.
    output_path:
        Absolute path where the PNG is saved.
    title_prefix:
        Short label prepended to the chart title (e.g. ``"BO"``, ``"LLM"``,
        ``"Hybrid"``).
    """
    net_cfg = NetworkConfig(
        n_hidden_layers=int(best_params["n_hidden_layers"]),
        n_neurons=int(best_params["n_neurons"]),
    )
    train_cfg = TrainingConfig(
        n_u=training_config.n_u,
        n_f=training_config.n_f,
        lr=float(best_params["lr"]),
        epochs_adam=int(best_params["epochs_adam"]),
        epochs_lbfgs=training_config.epochs_lbfgs,
    )

    print(
        f"  Re-training with best params: n_hidden_layers={best_params['n_hidden_layers']}, "
        f"n_neurons={best_params['n_neurons']}, lr={best_params['lr']:.2e}, "
        f"epochs_adam={best_params['epochs_adam']}"
    )
    solver = BurgersPINNSolver(pde_config, net_cfg, train_cfg)
    forward_result = solver.solve_forward(boundary_data, collocation)
    model = forward_result.model

    device = next(model.parameters()).device
    x_flat = torch.tensor(x_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_flat = torch.tensor(t_mesh.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(t_flat, x_flat).cpu().numpy().reshape(x_mesh.shape)

    rel_l2 = np.linalg.norm(u_pred - usol) / (np.linalg.norm(usol) + 1e-12)
    vmin = float(usol.min())
    vmax = float(usol.max())

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, label in zip(
        axes,
        [u_pred, usol],
        ["Best PINN Prediction u_theta(t, x)", "Reference u_ref(t, x)"],
    ):
        im = ax.pcolormesh(
            t_grid.ravel(), x_grid.ravel(), data,
            cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
        )
        fig.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(label)

    prefix = f"{title_prefix} " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Best Solution: Predicted vs Reference  (rel L2 = {rel_l2:.4e})\n"
        "Left panel should match right panel closely."
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")
