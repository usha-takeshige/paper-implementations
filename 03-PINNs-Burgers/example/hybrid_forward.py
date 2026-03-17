"""Example: Hybrid LLM + Bayesian Optimization for PINN hyperparameter tuning.

Combines LLM-guided broad exploration (Phase 1) with Bayesian Optimization
on the narrowed search space (Phase 2) to find the best PINN hyperparameters
for BurgersPINNSolver.solve_forward().

Shares the same data pipeline as bo_forward.py and opt_agent_forward.py.

Objective:  AccuracyObjective  (maximize  -rel_l2_error)

Search space:
    n_hidden_layers : int   [2, 8]     linear
    n_neurons       : int   [10, 100]  linear
    lr              : float [1e-4, 1e-2]  log
    epochs_adam     : int   [500, 5000]   linear

Usage:
    cd 03-PINNs-Burgers
    uv run python example/hybrid_forward.py

Prerequisites:
    Set GEMINI_API_KEY in a .env file.

Output (example/hybrid_output/):
    hybrid_llm_report.md             -- per-iteration LLM report (Phase 1)
    hybrid_convergence.png           -- cumulative best objective vs trial
    hybrid_objective_scatter.png     -- all trial objectives colored by phase
    hybrid_space_comparison.png      -- original vs narrowed bounds bar chart
    hybrid_best_solution_heatmap.png -- predicted vs reference solution heatmap
    hybrid_trials.csv                -- all trial results (both phases)
"""

import csv
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
from bo import AccuracyObjective, BOConfig, HyperParameter, SearchSpace
from opt_agent.config import LLMConfig, LLMIterationMeta, LLMResult
from opt_agent.report import IterationReportWriter
from opt_tool.result import TrialResult
from hybrid import HybridOptimizer, HybridResult

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hybrid_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi  # kinematic viscosity (Paper Section 5.1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_convergence(result: HybridResult) -> None:
    """Plot cumulative best objective value vs trial number for both phases."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    all_trials = result.llm_trials + result.bo_trials
    best_so_far: list[float] = []
    current_best = float("-inf")
    for t in all_trials:
        current_best = max(current_best, t.objective)
        best_so_far.append(current_best)

    trial_ids = list(range(len(all_trials)))
    n_llm = len(result.llm_trials)

    ax.axvline(x=n_llm - 0.5, color="gray", linestyle="--", alpha=0.6,
               label="Phase 1 / Phase 2 boundary")
    sns.lineplot(x=trial_ids, y=best_so_far, ax=ax, linewidth=1.5,
                 marker="o", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    ax.set_title(
        "Hybrid Convergence: Cumulative Best Objective vs Trial\n"
        "Curve should rise monotonically and plateau as BO converges."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "hybrid_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_objective_scatter(result: HybridResult) -> None:
    """Scatter plot of objective values for all trials, colored by phase."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    llm_initial = [t for t in result.llm_trials if t.is_initial]
    llm_guided = [t for t in result.llm_trials if not t.is_initial]
    bo_initial = [t for t in result.bo_trials if t.is_initial]
    bo_guided = [t for t in result.bo_trials if not t.is_initial]

    offset = 0
    if llm_initial:
        ids = [t.trial_id + offset for t in llm_initial]
        ax.scatter(ids, [t.objective for t in llm_initial],
                   label="Phase 1: Sobol initial", marker="o", s=60, alpha=0.8)
    if llm_guided:
        ids = [t.trial_id + offset for t in llm_guided]
        ax.scatter(ids, [t.objective for t in llm_guided],
                   label="Phase 1: LLM proposal", marker="^", s=60, alpha=0.8)

    offset = len(result.llm_trials)
    if bo_initial:
        ids = [t.trial_id + offset for t in bo_initial]
        ax.scatter(ids, [t.objective for t in bo_initial],
                   label="Phase 2: Sobol initial", marker="s", s=60, alpha=0.8)
    if bo_guided:
        ids = [t.trial_id + offset for t in bo_guided]
        ax.scatter(ids, [t.objective for t in bo_guided],
                   label="Phase 2: BO proposal", marker="D", s=60, alpha=0.8)

    ax.axhline(y=result.best_objective, color="red", linestyle="--", alpha=0.7,
               label=f"Best = {result.best_objective:.4e}")
    ax.axvline(x=len(result.llm_trials) - 0.5, color="gray", linestyle="--",
               alpha=0.4, label="Phase boundary")
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Objective  ({result.objective_name})")
    ax.set_title(
        "Hybrid Objective Values: Phase 1 (LLM) vs Phase 2 (BO)\n"
        "BO proposals should tend toward higher objective values."
    )
    ax.legend(fontsize=8)
    path = os.path.join(OUTPUT_DIR, "hybrid_objective_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_space_comparison(
    result: HybridResult, original_space: SearchSpace
) -> None:
    """Bar chart comparing original vs narrowed search space width per parameter."""
    sns.set_theme(style="whitegrid")
    params = original_space.parameters
    param_names = [hp.name for hp in params]
    n = len(params)

    orig_widths: list[float] = []
    narr_widths: list[float] = []
    for orig_hp, narr_hp in zip(params, result.narrowed_space.parameters):
        if orig_hp.log_scale:
            orig_widths.append(math.log(orig_hp.high) - math.log(orig_hp.low))
            narr_widths.append(math.log(narr_hp.high) - math.log(narr_hp.low))
        else:
            orig_widths.append(orig_hp.high - orig_hp.low)
            narr_widths.append(narr_hp.high - narr_hp.low)

    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, orig_widths, width, label="Original", alpha=0.8)
    ax.bar(x + width / 2, narr_widths, width, label="Narrowed", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=15, ha="right")
    ax.set_ylabel("Parameter range width\n(log-scale params: log space)")
    ax.set_title(
        "Search Space Comparison: Original vs Narrowed\n"
        "Narrowed bars should be shorter than or equal to original bars."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "hybrid_space_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_solution_heatmap(
    result: HybridResult,
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
        f"Hybrid Best Solution: Predicted vs Reference  (rel L2 = {rel_l2:.4e})\n"
        "Left panel should match right panel closely."
    )
    path = os.path.join(OUTPUT_DIR, "hybrid_best_solution_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_trials_csv(result: HybridResult) -> None:
    """Save all trial results from both phases to a CSV file.

    Columns: global_trial_id, phase, trial_id, is_initial,
             n_hidden_layers, n_neurons, lr, epochs_adam,
             objective, rel_l2_error, elapsed_time
    """
    param_names = ["n_hidden_layers", "n_neurons", "lr", "epochs_adam"]
    fieldnames = [
        "global_trial_id", "phase", "trial_id", "is_initial",
        *param_names,
        "objective", "rel_l2_error", "elapsed_time", "proposal_time",
    ]
    path = os.path.join(OUTPUT_DIR, "hybrid_trials.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        global_id = 0
        for phase, trials in [("llm", result.llm_trials), ("bo", result.bo_trials)]:
            for t in trials:
                row: dict = {
                    "global_trial_id": global_id,
                    "phase": phase,
                    "trial_id": t.trial_id,
                    "is_initial": t.is_initial,
                    "objective": t.objective,
                    "rel_l2_error": t.rel_l2_error,
                    "elapsed_time": t.elapsed_time,
                    "proposal_time": t.proposal_time,
                }
                for name in param_names:
                    row[name] = t.params.get(name, "")
                writer.writerow(row)
                global_id += 1
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run hybrid LLM + BO optimization for PINN hyperparameter tuning."""
    torch.manual_seed(42)
    print("=== Hybrid LLM + Bayesian Optimization: PINN Hyperparameter Tuning ===")
    print(f"  nu = {NU_TRUE:.6f}  (0.01 / pi)\n")

    # 1. Load reference data
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

    # 3. Sample training data
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
    objective = AccuracyObjective(
        pde_config=pde_config,
        boundary_data=boundary_data,
        collocation=collocation,
        x_mesh=X_mesh,
        t_mesh=T_mesh,
        usol=usol,
        base_training_config=base_training_config,
        device="cpu",
    )

    # 6. Run hybrid optimization
    llm_config = LLMConfig(n_initial=5, n_iterations=10, seed=42)
    bo_config = BOConfig(n_initial=5, n_iterations=10, acquisition="EI", seed=42)
    n_llm_iterations = 10
    print(
        f"\nStarting Hybrid: n_llm_initial={llm_config.n_initial}, "
        f"n_llm_iterations={n_llm_iterations}, "
        f"n_bo_initial={bo_config.n_initial}, "
        f"n_bo_iterations={bo_config.n_iterations}\n"
    )

    # Phase 1 streaming report (LLM)
    writer = IterationReportWriter(
        output_dir=OUTPUT_DIR,
        search_space=search_space,
        llm_config=llm_config,
        objective_name=objective.name,
        filename="hybrid_llm_report.md",
    )
    iteration_metas: list[LLMIterationMeta] = []

    def on_llm_iteration(
        meta: LLMIterationMeta,
        trial: TrialResult,
        all_trials: list[TrialResult],
    ) -> None:
        """Write Phase 1 Sobol results on first call, then append each LLM iteration."""
        if meta.iteration_id == 0:
            initial = [t for t in all_trials if t.is_initial]
            writer.write_initial_trials(initial)
        writer.append_iteration(meta, trial, all_trials)
        iteration_metas.append(meta)

    optimizer = HybridOptimizer(
        search_space=search_space,
        objective=objective,
        llm_config=llm_config,
        bo_config=bo_config,
        on_llm_iteration=on_llm_iteration,
        n_llm_iterations=n_llm_iterations,
        top_k_ratio=0.3,
        margin_ratio=0.1,
    )
    result = optimizer.optimize()

    # Finalize Phase 1 report using the LLM trials
    llm_trials = result.llm_trials
    best_llm = max(llm_trials, key=lambda t: t.objective)
    llm_result_for_report = LLMResult(
        trials=llm_trials,
        best_params=best_llm.params,
        best_objective=best_llm.objective,
        best_trial_id=best_llm.trial_id,
        objective_name=result.objective_name,
        llm_config=llm_config,
        iteration_metas=iteration_metas,
    )
    report_path = writer.finalize(llm_result_for_report)
    print(f"\n  Phase 1 report saved: {report_path}")

    print(f"\nBest trial: #{result.best_trial_id}")
    print(f"  Best objective: {result.best_objective:.4e}")
    print(f"  Best params:    {result.best_params}")

    # 7. Save CSV and visualizations
    print("\nSaving trial history ...")
    save_trials_csv(result)
    print("\nGenerating plots ...")
    plot_convergence(result)
    plot_objective_scatter(result)
    plot_space_comparison(result, search_space)
    plot_best_solution_heatmap(
        result, pde_config, base_training_config,
        X_mesh, T_mesh, usol, t_grid, x_grid,
    )

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
