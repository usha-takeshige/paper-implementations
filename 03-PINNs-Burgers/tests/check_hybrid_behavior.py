"""Behavior check script for Hybrid LLM + BO Optimizer (BHV-HYB-01 to BHV-HYB-03).

Runs the full hybrid optimization pipeline with real PINN training and LLM calls,
then outputs graphs for human review. Does NOT use pytest assertions.

Requires:
    - GEMINI_API_KEY set in .env file
    - Reference data at data/burgers_shock.mat

Usage:
    cd 03-PINNs-Burgers
    uv run python tests/check_hybrid_behavior.py
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from pathlib import Path

# Allow importing example utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "example"))
from forward_problem import (
    load_reference_data,
    sample_boundary_data,
    sample_collocation_points,
)

from bo import AccuracyObjective, BOConfig, HyperParameter, SearchSpace
from opt_agent.config import LLMConfig
from hybrid import HybridOptimizer, HybridResult

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "check_hybrid_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_TRUE = 0.01 / math.pi


# ---------------------------------------------------------------------------
# BHV-HYB-01: Cumulative best objective is monotonically non-decreasing
# ---------------------------------------------------------------------------


def check_bhv_hyb_01_convergence(result: HybridResult) -> None:
    """BHV-HYB-01: Plot cumulative best objective across all trials.

    Confirm: The curve is monotonically non-decreasing and Phase 2 maintains
    or improves upon Phase 1's best value.
    """
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

    ax.axvline(x=n_llm - 0.5, color="gray", linestyle="--", alpha=0.7,
               label="Phase 1 / Phase 2 boundary")
    sns.lineplot(x=trial_ids, y=best_so_far, ax=ax, linewidth=1.5,
                 marker="o", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    ax.set_title(
        "[BHV-HYB-01] Cumulative Best Objective vs Trial\n"
        "PASS: Curve is monotonically non-decreasing; Phase 2 meets or exceeds Phase 1 best."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bhv_hyb_01_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [BHV-HYB-01] Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-HYB-02: Narrowed space is narrower than original
# ---------------------------------------------------------------------------


def check_bhv_hyb_02_space_comparison(
    result: HybridResult, original_space: SearchSpace
) -> None:
    """BHV-HYB-02: Bar chart of original vs narrowed parameter widths.

    Confirm: All narrowed bars are equal to or shorter than original bars.
    """
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
        "[BHV-HYB-02] Search Space: Original vs Narrowed\n"
        "PASS: All narrowed bars are shorter than or equal to original bars."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bhv_hyb_02_space_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [BHV-HYB-02] Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-HYB-03: Phase 2 (BO) converges better than Phase 1 (LLM)
# ---------------------------------------------------------------------------


def check_bhv_hyb_03_phase_comparison(result: HybridResult) -> None:
    """BHV-HYB-03: Scatter plot comparing objective values between phases.

    Confirm: Phase 2 later trials tend to have higher objective values than
    the Phase 1 average.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    offset_llm = 0
    offset_bo = len(result.llm_trials)

    llm_initial = [t for t in result.llm_trials if t.is_initial]
    llm_guided = [t for t in result.llm_trials if not t.is_initial]
    bo_initial = [t for t in result.bo_trials if t.is_initial]
    bo_guided = [t for t in result.bo_trials if not t.is_initial]

    if llm_initial:
        ax.scatter([t.trial_id + offset_llm for t in llm_initial],
                   [t.objective for t in llm_initial],
                   label="Phase 1: Sobol", marker="o", s=60, alpha=0.7, color="steelblue")
    if llm_guided:
        ax.scatter([t.trial_id + offset_llm for t in llm_guided],
                   [t.objective for t in llm_guided],
                   label="Phase 1: LLM", marker="^", s=60, alpha=0.7, color="cornflowerblue")
    if bo_initial:
        ax.scatter([t.trial_id + offset_bo for t in bo_initial],
                   [t.objective for t in bo_initial],
                   label="Phase 2: Sobol", marker="s", s=60, alpha=0.7, color="darkorange")
    if bo_guided:
        ax.scatter([t.trial_id + offset_bo for t in bo_guided],
                   [t.objective for t in bo_guided],
                   label="Phase 2: BO", marker="D", s=60, alpha=0.7, color="tomato")

    llm_mean = np.mean([t.objective for t in result.llm_trials])
    ax.axhline(y=llm_mean, color="steelblue", linestyle=":", alpha=0.7,
               label=f"Phase 1 mean = {llm_mean:.4e}")
    ax.axhline(y=result.best_objective, color="red", linestyle="--", alpha=0.7,
               label=f"Best = {result.best_objective:.4e}")
    ax.axvline(x=len(result.llm_trials) - 0.5, color="gray", linestyle="--", alpha=0.4)

    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Objective  ({result.objective_name})")
    ax.set_title(
        "[BHV-HYB-03] Objective Values: Phase 1 (LLM) vs Phase 2 (BO)\n"
        "PASS: Phase 2 later trials have higher objective values than Phase 1 average."
    )
    ax.legend(fontsize=8)
    path = os.path.join(OUTPUT_DIR, "bhv_hyb_03_phase_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [BHV-HYB-03] Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run hybrid behavior checks with real PINN training and Gemini LLM."""
    torch.manual_seed(42)
    print("=== Hybrid Behavior Check (requires GEMINI_API_KEY) ===")

    # Load data
    x_grid, t_grid, X_mesh, T_mesh, usol = load_reference_data()
    pde_config_module = __import__("PINNs_Burgers")
    from PINNs_Burgers import PDEConfig, TrainingConfig

    pde_config = PDEConfig(
        nu=NU_TRUE, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0
    )
    base_training_config = TrainingConfig(
        n_u=100, n_f=10_000, lr=1e-3, epochs_adam=2_000, epochs_lbfgs=50
    )
    boundary_data = sample_boundary_data(x_grid, t_grid, usol, n_u=100)
    collocation = sample_collocation_points(x_grid, t_grid, n_f=10_000)

    search_space = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])
    objective = AccuracyObjective(
        pde_config=pde_config,
        boundary_data=boundary_data,
        collocation=collocation,
        x_mesh=X_mesh,
        t_mesh=T_mesh,
        usol=usol,
        base_training_config=base_training_config,
    )

    llm_config = LLMConfig(n_initial=5, n_iterations=10, seed=42)
    bo_config = BOConfig(n_initial=5, n_iterations=10, acquisition="EI", seed=42)

    optimizer = HybridOptimizer(
        search_space=search_space,
        objective=objective,
        llm_config=llm_config,
        bo_config=bo_config,
        n_llm_iterations=10,
        top_k_ratio=0.3,
        margin_ratio=0.1,
    )
    result = optimizer.optimize()

    print("\nGenerating behavior check graphs ...")
    check_bhv_hyb_01_convergence(result)
    check_bhv_hyb_02_space_comparison(result, search_space)
    check_bhv_hyb_03_phase_comparison(result)
    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
