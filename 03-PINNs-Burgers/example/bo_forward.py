"""Example: Bayesian optimization for PINN hyperparameter tuning.

Uses BoTorch-based Gaussian process Bayesian optimization to search for
the best neural network hyperparameters for BurgersPINNSolver.solve_forward().

Shares the same data pipeline as forward_problem.py.

Objective:  AccuracyObjective  (maximize  -rel_l2_error)
            Switch to AccuracySpeedObjective to also penalize training time.

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

import torch

from pathlib import Path

# Allow importing from sibling example scripts
sys.path.insert(0, str(Path(__file__).parent))
from forward_problem import (
    load_reference_data,
    sample_boundary_data,
    sample_collocation_points,
)

from PINNs_Burgers import PDEConfig, TrainingConfig, BurgersPINNSolver
from bo import (
    AccuracyObjective,
    BOConfig,
    BayesianOptimizer,
    HyperParameter,
    ReportGenerator,
    SearchSpace,
)
from opt_viz import plot_convergence, plot_objective_scatter, plot_parallel_coords
from opt_viz.pinn_heatmap import plot_best_solution_heatmap

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "bo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi  # kinematic viscosity (Paper Section 5.1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Bayesian optimization for PINN hyperparameter tuning."""
    torch.manual_seed(42)
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
    objective = AccuracyObjective(
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
    plot_convergence(
        result.trials,
        bo_config.n_initial,
        os.path.join(OUTPUT_DIR, "bo_convergence.png"),
        title_prefix="BO",
        boundary_label="Sobol / BO boundary",
    )
    plot_objective_scatter(
        result.trials,
        result.objective_name,
        result.best_objective,
        bo_config.n_initial,
        os.path.join(OUTPUT_DIR, "bo_objective_scatter.png"),
        title_prefix="BO",
        proposal_label="BO proposal",
    )
    plot_parallel_coords(
        result.trials,
        search_space,
        os.path.join(OUTPUT_DIR, "bo_parallel_coords.png"),
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
        os.path.join(OUTPUT_DIR, "bo_best_solution_heatmap.png"),
        title_prefix="BO",
    )

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
