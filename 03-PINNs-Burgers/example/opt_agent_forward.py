"""Example: LLM-based hyperparameter optimization for PINN training.

Uses LangChain + Gemini to propose hyperparameter configurations for
BurgersPINNSolver.solve_forward(), then evaluates each proposal.

Uses the same data pipeline and search space as bo_forward.py for a
fair comparison between Bayesian optimization and LLM-based optimization.

Requires:
    GEMINI_API_KEY=<your_api_key>  in a .env file (or environment variable)
    GEMINI_MODEL_NAME=gemini-2.0-flash  (optional, defaults to gemini-2.0-flash)

Usage:
    cd 03-PINNs-Burgers
    uv run python example/opt_agent_forward.py

Output (example/opt_agent_output/):
    opt_agent_report.md                 -- running report (updated each iteration)
    opt_agent_convergence.png           -- cumulative best objective vs trial
    opt_agent_objective_scatter.png     -- all trial objectives colored by type
    opt_agent_parallel_coords.png       -- parallel coordinates of hyperparameters
    opt_agent_best_solution_heatmap.png -- predicted vs reference solution heatmap
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
from bo import (
    AccuracyObjective,
    HyperParameter,
    SearchSpace,
    TrialResult,
)
from opt_agent import IterationReportWriter, LLMConfig, LLMIterationMeta, LLMOptimizer
from opt_viz import plot_convergence, plot_objective_scatter, plot_parallel_coords
from opt_viz.pinn_heatmap import plot_best_solution_heatmap

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "opt_agent_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi  # kinematic viscosity (Paper Section 5.1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run LLM-based hyperparameter optimization for PINN training."""
    torch.manual_seed(42)
    print("=== LLM-based Optimization: PINN Hyperparameter Tuning ===")
    print(f"  nu = {NU_TRUE:.6f}  (0.01 / pi)\n")

    # 1. Load reference data (shared with forward_problem.py)
    print("Loading reference data ...")
    x_grid, t_grid, X_mesh, T_mesh, usol = load_reference_data()
    print(f"  x: {x_grid.shape}, t: {t_grid.shape}, usol: {usol.shape}")

    # 2. Fixed configuration (same as bo_forward.py for fair comparison)
    pde_config = PDEConfig(
        nu=NU_TRUE, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0
    )
    base_training_config = TrainingConfig(
        n_u=100, n_f=10_000, lr=1e-3, epochs_adam=2_000, epochs_lbfgs=50
    )

    # 3. Sample training data (fixed across all trials for fair comparison)
    boundary_data = sample_boundary_data(x_grid, t_grid, usol, n_u=100)
    collocation = sample_collocation_points(x_grid, t_grid, n_f=10_000)

    # 4. Define search space (identical to bo_forward.py)
    search_space = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])

    # 5. Define objective function (identical to bo_forward.py)
    objective = AccuracyObjective(
        pde_config=pde_config,
        boundary_data=boundary_data,
        collocation=collocation,
        x_mesh=X_mesh,
        t_mesh=T_mesh,
        usol=usol,
        base_training_config=base_training_config,
    )

    # 6. Run LLM-based optimization with per-iteration report
    llm_config = LLMConfig(n_initial=5, n_iterations=20, seed=42)
    print(
        f"\nStarting LLM optimization: n_initial={llm_config.n_initial}, "
        f"n_iterations={llm_config.n_iterations}\n"
    )

    # Initialize running report (header + config written immediately)
    writer = IterationReportWriter(
        output_dir=OUTPUT_DIR,
        search_space=search_space,
        llm_config=llm_config,
        objective_name=objective.name,
    )

    def on_iteration(
        meta: LLMIterationMeta,
        trial: TrialResult,
        all_trials: list[TrialResult],
    ) -> None:
        """Write Phase 1 on first call, then append each LLM iteration."""
        if meta.iteration_id == 0:
            initial = [t for t in all_trials if t.is_initial]
            writer.write_initial_trials(initial)
        writer.append_iteration(meta, trial, all_trials)

    optimizer = LLMOptimizer(
        search_space, objective, llm_config, on_iteration=on_iteration
    )
    result = optimizer.optimize()

    print(f"\nBest trial: #{result.best_trial_id}")
    print(f"  Best objective: {result.best_objective:.4e}")
    print(f"  Best params:    {result.best_params}")

    # Finalize report with best-result summary and convergence table
    report_path = writer.finalize(result)
    print(f"\n  Report saved: {report_path}")

    # 8. Visualizations
    print("\nGenerating plots ...")
    plot_convergence(
        result.trials,
        llm_config.n_initial,
        os.path.join(OUTPUT_DIR, "opt_agent_convergence.png"),
        title_prefix="LLM Optimizer",
        boundary_label="Sobol / LLM boundary",
    )
    plot_objective_scatter(
        result.trials,
        result.objective_name,
        result.best_objective,
        llm_config.n_initial,
        os.path.join(OUTPUT_DIR, "opt_agent_objective_scatter.png"),
        title_prefix="LLM Optimizer",
        proposal_label="LLM proposal",
    )
    plot_parallel_coords(
        result.trials,
        search_space,
        os.path.join(OUTPUT_DIR, "opt_agent_parallel_coords.png"),
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
        os.path.join(OUTPUT_DIR, "opt_agent_best_solution_heatmap.png"),
        title_prefix="LLM",
    )

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
