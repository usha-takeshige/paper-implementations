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

import math
import os
import sys

import torch

from pathlib import Path

# Allow importing from sibling example scripts
sys.path.insert(0, str(Path(__file__).parent))
from forward_problem import (
    load_reference_data,
    sample_boundary_data,
    sample_collocation_points,
)

from PINNs_Burgers import PDEConfig, TrainingConfig
from bo import AccuracyObjective, BOConfig, HyperParameter, SearchSpace
from opt_agent.config import LLMConfig, LLMIterationMeta, LLMResult
from opt_agent.report import IterationReportWriter
from opt_tool.result import TrialResult
from hybrid import (
    HybridOptimizer,
    plot_convergence,
    plot_objective_scatter,
    plot_space_comparison,
    save_trials_csv,
)
from opt_viz.pinn_heatmap import plot_best_solution_heatmap

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hybrid_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi  # kinematic viscosity (Paper Section 5.1)


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
    save_trials_csv(result, os.path.join(OUTPUT_DIR, "hybrid_trials.csv"))
    print("\nGenerating plots ...")
    plot_convergence(result, os.path.join(OUTPUT_DIR, "hybrid_convergence.png"))
    plot_objective_scatter(result, os.path.join(OUTPUT_DIR, "hybrid_objective_scatter.png"))
    plot_space_comparison(
        result, search_space, os.path.join(OUTPUT_DIR, "hybrid_space_comparison.png")
    )
    plot_best_solution_heatmap(
        result.best_params,
        pde_config,
        base_training_config,
        boundary_data,
        collocation,
        X_mesh,
        T_mesh,
        usol,
        t_grid,
        x_grid,
        os.path.join(OUTPUT_DIR, "hybrid_best_solution_heatmap.png"),
        title_prefix="Hybrid",
    )

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
