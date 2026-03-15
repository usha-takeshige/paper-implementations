"""Common Markdown report utilities for hyperparameter optimization."""

from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


def build_search_space_table(search_space: SearchSpace) -> str:
    """Build the Markdown table rows for the search space definition.

    Parameters
    ----------
    search_space:
        Search space whose parameters are rendered as a Markdown table.

    Returns
    -------
    str
        Newline-joined table rows (without header) ready to embed in a report.
    """
    scale_label = {True: "log", False: "linear"}
    return "\n".join([
        f"| {hp.name:<17} | {hp.param_type:<5} | {hp.low:<6} | {hp.high:<6} | {scale_label[hp.log_scale]:<6} |"
        for hp in search_space.parameters
    ])


def build_convergence_table(trials: list[TrialResult]) -> str:
    """Build the Markdown table rows for the convergence history.

    Computes the cumulative maximum objective across all trials and renders
    each trial's best-so-far value as a Markdown table row.

    Parameters
    ----------
    trials:
        All trial results in evaluation order.

    Returns
    -------
    str
        Newline-joined table rows (without header) ready to embed in a report.
    """
    rows: list[tuple[int, float]] = []
    current_best = float("-inf")
    for t in trials:
        current_best = max(current_best, t.objective)
        rows.append((t.trial_id, current_best))
    return "\n".join([
        f"| {tid:<5} | {best:.4e}              |"
        for tid, best in rows
    ])


def build_best_result_section(
    best_trial: TrialResult,
    best_params: dict[str, float | int],
) -> str:
    """Build the Markdown section for the best result.

    Parameters
    ----------
    best_trial:
        The trial with the highest objective value.
    best_params:
        Hyperparameter values of the best trial.

    Returns
    -------
    str
        Markdown snippet containing the best hyperparameter table and metrics.
    """
    param_rows = "\n".join([
        f"| {name:<17} | {val}   |"
        for name, val in best_params.items()
    ])
    return (
        f"| Hyperparameter    | Value  |\n"
        f"|-------------------|--------|\n"
        f"{param_rows}\n\n"
        f"**Metrics**:\n"
        f"- Relative L2 Error: {best_trial.rel_l2_error:.4e}\n"
        f"- Elapsed Time: {best_trial.elapsed_time:.2f} s"
    )
