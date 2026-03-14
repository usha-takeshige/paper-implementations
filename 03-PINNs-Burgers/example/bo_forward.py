"""Example: Bayesian optimization for PINN hyperparameter tuning.

Uses BoTorch-based Gaussian process Bayesian optimization to search for
the best neural network hyperparameters for BurgersPINNSolver.solve_forward().

Shares the same data pipeline as forward_problem.py.

Objective:  maximize  1 / (rel_l2_error * elapsed_time)

Search space:
    n_hidden_layers : int   [2, 8]     linear
    n_neurons       : int   [10, 100]  linear
    lr              : float [1e-4, 1e-2]  log
    epochs_adam     : int   [500, 5000]   linear

Usage:
    cd 03-PINNs-Burgers
    uv run python example/bo_forward.py

Output (example/output/):
    bo_convergence.png           -- cumulative best objective vs trial
    bo_objective_scatter.png     -- all trial objectives colored by type
    bo_parallel_coords.png       -- parallel coordinates of hyperparameters
    bo_best_solution_heatmap.png -- predicted vs reference solution heatmap
    bo_report.md                 -- markdown summary report
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from pathlib import Path

# Allow importing from sibling example scripts
sys.path.insert(0, str(Path(__file__).parent))
from forward_problem import (
    load_reference_data,
    sample_boundary_data,
    sample_collocation_points,
)

from PINNs_Burgers import NetworkConfig, PDEConfig, TrainingConfig, BurgersPINNSolver
from bo import (
    BOConfig,
    BayesianOptimizer,
    BOResult,
    HyperParameter,
    ObjectiveFunction,
    ReportGenerator,
    SearchSpace,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "bo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi  # kinematic viscosity (Paper Section 5.1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_convergence(result: BOResult) -> None:
    """Plot cumulative best objective value vs trial number."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    best_so_far: list[float] = []
    current_best = float("-inf")
    for t in result.trials:
        current_best = max(current_best, t.objective)
        best_so_far.append(current_best)

    trial_ids = list(range(len(result.trials)))
    n_initial = result.bo_config.n_initial
    ax.axvline(x=n_initial - 0.5, color="gray", linestyle="--", alpha=0.6,
               label="Sobol / BO boundary")
    sns.lineplot(x=trial_ids, y=best_so_far, ax=ax, linewidth=1.5,
                 marker="o", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    ax.set_title(
        "BO Convergence: Cumulative Best Objective vs Trial\n"
        "Curve should rise monotonically and plateau as BO converges."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bo_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_objective_scatter(result: BOResult) -> None:
    """Scatter plot of objective values for all trials, colored by type."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    initial_ids = [t.trial_id for t in result.trials if t.is_initial]
    initial_obj = [t.objective for t in result.trials if t.is_initial]
    bo_ids = [t.trial_id for t in result.trials if not t.is_initial]
    bo_obj = [t.objective for t in result.trials if not t.is_initial]

    ax.scatter(initial_ids, initial_obj, label="Sobol initial", alpha=0.8,
               marker="o", s=60)
    ax.scatter(bo_ids, bo_obj, label="BO proposal", alpha=0.8, marker="^", s=60)
    ax.axhline(y=result.best_objective, color="red", linestyle="--", alpha=0.7,
               label=f"Best = {result.best_objective:.4e}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective  1/(rel_l2_error × time)")
    ax.set_title(
        "BO Objective Values: Sobol Initial vs BO Proposals\n"
        "BO proposals should tend toward higher objective values."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bo_objective_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_parallel_coords(result: BOResult, search_space: SearchSpace) -> None:
    """Parallel coordinates plot of hyperparameter values vs objective."""
    sns.set_theme(style="whitegrid")
    param_names = [hp.name for hp in search_space.parameters]

    # Normalize param values to [0,1] for display
    rows = []
    for t in result.trials:
        x = search_space.to_tensor(t.params).squeeze(0).tolist()
        rows.append(x + [t.objective])

    cols = param_names + ["objective"]
    data = np.array(rows)

    # Normalize each column to [0, 1] for display
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-12)

    fig, axes = plt.subplots(1, len(cols) - 1, figsize=(12, 5), sharey=False)
    cmap = plt.cm.viridis  # type: ignore[attr-defined]
    obj_norm = data_norm[:, -1]

    for i in range(len(cols) - 1):
        for j, row in enumerate(data_norm):
            color = cmap(obj_norm[j])
            axes[i].plot([0, 1], [row[i], row[i + 1]], color=color, alpha=0.6,
                         linewidth=0.8)
        axes[i].set_xlim(0, 1)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels([cols[i], cols[i + 1]], rotation=15, fontsize=8)
        axes[i].set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap,  # type: ignore[attr-defined]
                                norm=plt.Normalize(vmin=data[:, -1].min(),
                                                   vmax=data[:, -1].max()))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label="Objective")
    fig.suptitle(
        "Parallel Coordinates: Hyperparameter Values vs Objective\n"
        "Brighter lines correspond to higher objective values."
    )
    path = os.path.join(OUTPUT_DIR, "bo_parallel_coords.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_solution_heatmap(
    result: BOResult,
    search_space: SearchSpace,
    pde_config: PDEConfig,
    base_training_config: TrainingConfig,
    x_mesh: np.ndarray,
    t_mesh: np.ndarray,
    usol: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> None:
    """Re-train PINN with best hyperparameters and plot predicted vs reference."""
    from forward_problem import sample_boundary_data, sample_collocation_points

    # Reload boundary and collocation data for final evaluation
    boundary_data = sample_boundary_data(
        x_grid, t_grid, usol, n_u=base_training_config.n_u
    )
    collocation = sample_collocation_points(x_grid, t_grid, n_f=base_training_config.n_f)

    best = result.best_params
    net_cfg = NetworkConfig(
        n_hidden_layers=int(best["n_hidden_layers"]),
        n_neurons=int(best["n_neurons"]),
    )
    train_cfg = TrainingConfig(
        n_u=base_training_config.n_u,
        n_f=base_training_config.n_f,
        lr=float(best["lr"]),
        epochs_adam=int(best["epochs_adam"]),
        epochs_lbfgs=base_training_config.epochs_lbfgs,
    )

    print(
        f"  Re-training with best params: n_hidden_layers={best['n_hidden_layers']}, "
        f"n_neurons={best['n_neurons']}, lr={best['lr']:.2e}, "
        f"epochs_adam={best['epochs_adam']}"
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

    fig.suptitle(
        f"BO Best Solution: Predicted vs Reference  (rel L2 = {rel_l2:.4e})\n"
        "Left panel should match right panel closely."
    )
    path = os.path.join(OUTPUT_DIR, "bo_best_solution_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Bayesian optimization for PINN hyperparameter tuning."""
    print("=== Bayesian Optimization: PINN Hyperparameter Tuning ===")
    print(f"  nu = {NU_TRUE:.6f}  (0.01 / pi)\n")

    # 1. Load reference data (shared with forward_problem.py)
    print("Loading reference data ...")
    x_grid, t_grid, X_mesh, T_mesh, usol = load_reference_data()
    print(f"  x: {x_grid.shape}, t: {t_grid.shape}, usol: {usol.shape}")

    # 2. Fixed configuration
    pde_config = PDEConfig(
        nu=NU_TRUE, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0
    )
    base_training_config = TrainingConfig(
        n_u=100, n_f=10_000, lr=1e-3, epochs_adam=2_000, epochs_lbfgs=50
    )

    # 3. Sample training data (fixed across all BO trials for fair comparison)
    boundary_data = sample_boundary_data(x_grid, t_grid, usol, n_u=100)
    collocation = sample_collocation_points(x_grid, t_grid, n_f=10_000)

    # 4. Define search space
    search_space = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])

    # 5. Define objective function
    objective = ObjectiveFunction(
        pde_config=pde_config,
        boundary_data=boundary_data,
        collocation=collocation,
        x_mesh=X_mesh,
        t_mesh=T_mesh,
        usol=usol,
        base_training_config=base_training_config,
    )

    # 6. Run Bayesian optimization
    bo_config = BOConfig(n_initial=5, n_iterations=20, acquisition="EI", seed=42)
    print(
        f"\nStarting BO: n_initial={bo_config.n_initial}, "
        f"n_iterations={bo_config.n_iterations}, "
        f"acquisition={bo_config.acquisition}\n"
    )
    optimizer = BayesianOptimizer(search_space, objective, bo_config)
    result = optimizer.optimize()

    print(f"\nBest trial: #{result.best_trial_id}")
    print(f"  Best objective: {result.best_objective:.4e}")
    print(f"  Best params:    {result.best_params}")

    # 7. Generate markdown report
    print("\nGenerating report ...")
    reporter = ReportGenerator(output_dir=OUTPUT_DIR)
    report_path = reporter.generate(result, search_space)
    print(f"  Report saved: {report_path}")

    # 8. Visualizations
    print("\nGenerating plots ...")
    plot_convergence(result)
    plot_objective_scatter(result)
    plot_parallel_coords(result, search_space)
    plot_best_solution_heatmap(
        result, search_space, pde_config, base_training_config,
        X_mesh, T_mesh, usol, t_grid, x_grid,
    )

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
