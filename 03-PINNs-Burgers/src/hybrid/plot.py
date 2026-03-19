"""Hybrid optimizer visualization and CSV output utilities.

These functions are specific to :class:`~hybrid.result.HybridResult`
and cannot be expressed with the generic ``opt_viz`` functions because
the hybrid result stores Phase 1 (LLM) and Phase 2 (BO) trials in
separate lists and exposes a ``narrowed_space`` attribute.
"""

from __future__ import annotations

import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hybrid.result import HybridResult
from opt_tool.space import SearchSpace


def plot_convergence(result: HybridResult, output_path: str) -> None:
    """Plot cumulative best objective value vs trial number for both phases.

    Parameters
    ----------
    result:
        Completed hybrid optimization result.
    output_path:
        Absolute path where the PNG is saved.
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

    ax.axvline(
        x=n_llm - 0.5,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="Phase 1 / Phase 2 boundary",
    )
    sns.lineplot(x=trial_ids, y=best_so_far, ax=ax, linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    ax.set_title(
        "Hybrid Convergence: Cumulative Best Objective vs Trial\n"
        "Curve should rise monotonically and plateau as BO converges."
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_objective_scatter(result: HybridResult, output_path: str) -> None:
    """Scatter plot of objective values for all trials, colored by phase.

    Trials are split into four categories: Phase 1 Sobol initial,
    Phase 1 LLM-guided, Phase 2 Sobol initial, and Phase 2 BO-guided.

    Parameters
    ----------
    result:
        Completed hybrid optimization result.
    output_path:
        Absolute path where the PNG is saved.
    """
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

    ax.axhline(
        y=result.best_objective,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best = {result.best_objective:.4e}",
    )
    ax.axvline(x=len(result.llm_trials) - 0.5, color="gray", linestyle="--",
               alpha=0.4, label="Phase boundary")
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Objective  ({result.objective_name})")
    ax.set_title(
        "Hybrid Objective Values: Phase 1 (LLM) vs Phase 2 (BO)\n"
        "BO proposals should tend toward higher objective values."
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_space_comparison(
    result: HybridResult,
    original_space: SearchSpace,
    output_path: str,
) -> None:
    """Bar chart comparing original vs narrowed search space width per parameter.

    Parameters
    ----------
    result:
        Completed hybrid optimization result containing ``narrowed_space``.
    original_space:
        The full search space provided before Phase 1.
    output_path:
        Absolute path where the PNG is saved.
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
        "Search Space Comparison: Original vs Narrowed\n"
        "Narrowed bars should be shorter than or equal to original bars."
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_trials_csv(result: HybridResult, output_path: str) -> None:
    """Save all trial results from both phases to a CSV file.

    Columns: ``global_trial_id``, ``phase``, ``trial_id``, ``is_initial``,
    ``n_hidden_layers``, ``n_neurons``, ``lr``, ``epochs_adam``,
    ``objective``, ``rel_l2_error``, ``elapsed_time``, ``proposal_time``.

    Parameters
    ----------
    result:
        Completed hybrid optimization result.
    output_path:
        Absolute path where the CSV is saved.
    """
    param_names = ["n_hidden_layers", "n_neurons", "lr", "epochs_adam"]
    fieldnames = [
        "global_trial_id", "phase", "trial_id", "is_initial",
        *param_names,
        "objective", "rel_l2_error", "elapsed_time", "proposal_time",
    ]
    with open(output_path, "w", newline="") as f:
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
    print(f"  Saved: {output_path}")
