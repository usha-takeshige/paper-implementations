"""Generic single-phase optimization visualization utilities.

Provides plot functions that accept ``list[TrialResult]`` and metadata
directly, so they can be reused by BO, LLM, and any future optimizer without
depending on optimizer-specific result types.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


def plot_convergence(
    trials: list[TrialResult],
    n_initial: int,
    output_path: str,
    *,
    title_prefix: str = "",
    boundary_label: str = "Sobol / Proposal boundary",
) -> None:
    """Plot cumulative best objective value vs trial number.

    Parameters
    ----------
    trials:
        All trials in evaluation order.
    n_initial:
        Number of Sobol (random) initial trials; used to draw the boundary.
    output_path:
        Absolute path where the PNG is saved.
    title_prefix:
        Short label prepended to the chart title (e.g. ``"BO"``, ``"LLM"``).
    boundary_label:
        Legend label for the Sobol / proposal boundary vertical line.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    best_so_far: list[float] = []
    current_best = float("-inf")
    for t in trials:
        current_best = max(current_best, t.objective)
        best_so_far.append(current_best)

    trial_ids = list(range(len(trials)))
    ax.axvline(
        x=n_initial - 0.5,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=boundary_label,
    )
    sns.lineplot(x=trial_ids, y=best_so_far, ax=ax, linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    prefix = f"{title_prefix} " if title_prefix else ""
    ax.set_title(
        f"{prefix}Convergence: Cumulative Best Objective vs Trial\n"
        "Curve should rise monotonically and plateau as the optimizer converges."
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_objective_scatter(
    trials: list[TrialResult],
    objective_name: str,
    best_objective: float,
    n_initial: int,
    output_path: str,
    *,
    title_prefix: str = "",
    proposal_label: str = "Proposal",
) -> None:
    """Scatter plot of objective values for all trials, colored by type.

    Parameters
    ----------
    trials:
        All trials in evaluation order.
    objective_name:
        Name of the objective function (used for the y-axis label).
    best_objective:
        Best observed objective value (drawn as a horizontal reference line).
    n_initial:
        Number of Sobol initial trials; trials at index < n_initial are marked
        as ``is_initial``.
    output_path:
        Absolute path where the PNG is saved.
    title_prefix:
        Short label prepended to the chart title (e.g. ``"BO"``, ``"LLM"``).
    proposal_label:
        Legend label for non-initial (algorithm-guided) trials.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))

    initial_ids = [t.trial_id for t in trials if t.is_initial]
    initial_obj = [t.objective for t in trials if t.is_initial]
    proposal_ids = [t.trial_id for t in trials if not t.is_initial]
    proposal_obj = [t.objective for t in trials if not t.is_initial]

    ax.scatter(initial_ids, initial_obj, label="Sobol initial", alpha=0.8, marker="o", s=60)
    ax.scatter(proposal_ids, proposal_obj, label=proposal_label, alpha=0.8, marker="^", s=60)
    ax.axhline(
        y=best_objective,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best = {best_objective:.4e}",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Objective  ({objective_name})")
    prefix = f"{title_prefix} " if title_prefix else ""
    ax.set_title(
        f"{prefix}Objective Values: Sobol Initial vs {proposal_label}s\n"
        f"{proposal_label}s should tend toward higher objective values."
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_parallel_coords(
    trials: list[TrialResult],
    search_space: SearchSpace,
    output_path: str,
) -> None:
    """Parallel coordinates plot of hyperparameter values vs objective.

    Parameters
    ----------
    trials:
        All trials; each trial must have a ``params`` dict matching
        ``search_space``.
    search_space:
        The hyperparameter search space used during optimization.
    output_path:
        Absolute path where the PNG is saved.
    """
    sns.set_theme(style="whitegrid")
    param_names = [hp.name for hp in search_space.parameters]

    rows = []
    for t in trials:
        x = search_space.to_tensor(t.params).squeeze(0).tolist()
        rows.append(x + [t.objective])

    cols = param_names + ["objective"]
    data = np.array(rows)
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-12)

    fig, axes = plt.subplots(1, len(cols) - 1, figsize=(12, 5), sharey=False)
    cmap = plt.cm.viridis  # type: ignore[attr-defined]
    obj_norm = data_norm[:, -1]

    for i in range(len(cols) - 1):
        for j, row in enumerate(data_norm):
            color = cmap(obj_norm[j])
            axes[i].plot([0, 1], [row[i], row[i + 1]], color=color, alpha=0.6, linewidth=0.8)
        axes[i].set_xlim(0, 1)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels([cols[i], cols[i + 1]], rotation=15, fontsize=8)
        axes[i].set_yticks([])

    sm = plt.cm.ScalarMappable(  # type: ignore[attr-defined]
        cmap=cmap,
        norm=plt.Normalize(vmin=data[:, -1].min(), vmax=data[:, -1].max()),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label="Objective")
    fig.suptitle(
        "Parallel Coordinates: Hyperparameter Values vs Objective\n"
        "Brighter lines correspond to higher objective values."
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")
