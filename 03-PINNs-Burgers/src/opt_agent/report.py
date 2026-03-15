"""Per-iteration Markdown report writer for LLM-based hyperparameter optimization."""

import os
from datetime import datetime, timezone

from opt_agent.config import LLMConfig, LLMIterationMeta, LLMResult
from opt_tool.report_utils import build_convergence_table, build_search_space_table
from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


class IterationReportWriter:
    """Write opt_agent_report.md incrementally, one section per LLM iteration.

    Mirrors the structure of bo.report.ReportGenerator, but streams content
    to disk after each iteration so that partial results are available even
    if the optimization run is interrupted.

    Workflow
    --------
    1. ``__init__``                      — Creates the file and writes the
       report header (timestamp, config, search space).
    2. ``write_initial_trials``          — Called after Phase 1 (Sobol).
       Appends the Phase 1 results table.
    3. ``append_iteration``              — Passed as ``on_iteration`` callback
       to ``LLMOptimizer``.  Called after each LLM-guided iteration; appends
       one section containing:

       - LLM analysis of the current search state
       - Proposed parameters and reasoning
       - Evaluation result of the proposed point

    4. ``finalize``                      — Called after ``optimize()`` returns.
       Appends the final best-result summary and full convergence table.
    """

    def __init__(
        self,
        output_dir: str,
        search_space: SearchSpace,
        llm_config: LLMConfig,
        objective_name: str,
    ) -> None:
        """Create the output file and write the report header.

        Parameters
        ----------
        output_dir:
            Directory where ``opt_agent_report.md`` is saved.
        search_space:
            Search space definition written in the header.
        llm_config:
            LLM optimization configuration written in the header.
        objective_name:
            Name of the objective function.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, "opt_agent_report.md")
        self._n_iterations = llm_config.n_iterations

        space_rows = build_search_space_table(search_space)
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        header = f"""# LLM-based Hyperparameter Optimization Report
## Burgers PINNs Hyperparameter Tuning

Started: {now}

---

## 1. Configuration

| Parameter         | Value  |
|-------------------|--------|
| objective         | {objective_name} |
| n_initial         | {llm_config.n_initial}   |
| n_iterations      | {llm_config.n_iterations}   |
| seed              | {llm_config.seed}   |
| optimizer         | LLM (Gemini via LangChain) |

### Search Space

| Hyperparameter    | Type  | Low    | High   | Scale  |
|-------------------|-------|--------|--------|--------|
{space_rows}

---

"""
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(header)

    def write_initial_trials(self, initial_trials: list[TrialResult]) -> None:
        """Append Phase 1 (Sobol) results to the report.

        Parameters
        ----------
        initial_trials:
            All Sobol initial trials produced in Phase 1.
        """
        trial_rows = "\n".join([
            f"| {t.trial_id:<5} | "
            f"{t.params.get('n_hidden_layers', '-'):<15} | "
            f"{t.params.get('n_neurons', '-'):<9} | "
            f"{float(t.params.get('lr', 0.0)):.2e} | "
            f"{t.params.get('epochs_adam', '-'):<11} | "
            f"{t.rel_l2_error:.4e}     | "
            f"{t.elapsed_time:.2f}     | "
            f"{t.objective:.4e}   |"
            for t in initial_trials
        ])
        best_initial = max(initial_trials, key=lambda t: t.objective)

        section = f"""## 2. Phase 1: Sobol Initial Exploration

| Trial | n_hidden_layers | n_neurons | lr       | epochs_adam | Rel L2 Error | Time (s) | Objective  |
|-------|-----------------|-----------|----------|-------------|--------------|----------|------------|
{trial_rows}

**Best after Phase 1** — Trial #{best_initial.trial_id}:
objective = {best_initial.objective:.4e}, rel_l2 = {best_initial.rel_l2_error:.4e}

---

## 3. Phase 2: LLM-Guided Search

"""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(section)

    def append_iteration(
        self,
        meta: LLMIterationMeta,
        trial: TrialResult,
        all_trials: list[TrialResult],
    ) -> None:
        """Append one LLM iteration section to the report.

        Intended to be passed directly as the ``on_iteration`` callback of
        ``LLMOptimizer``.  On the first call (``meta.iteration_id == 0``),
        ``write_initial_trials`` must already have been called.

        Parameters
        ----------
        meta:
            LLM metadata for this iteration (analysis, proposed_params, reasoning).
        trial:
            Evaluation result of the LLM-proposed parameters.
        all_trials:
            All trials completed so far (Phase 1 + Phase 2 up to this point).
        """
        best_so_far = max(all_trials, key=lambda t: t.objective)
        param_rows = "\n".join([
            f"| {name:<17} | {val}   |"
            for name, val in trial.params.items()
        ])

        section = f"""### Iteration {meta.iteration_id + 1} / {self._n_iterations}  (Trial #{trial.trial_id})

#### LLM Analysis of Current State

{meta.analysis_report}

#### Proposed Parameters and Reasoning

| Hyperparameter    | Proposed Value |
|-------------------|----------------|
{param_rows}

**Reasoning:**
{meta.reasoning}

#### Evaluation Result

| Metric            | Value          |
|-------------------|----------------|
| Objective         | {trial.objective:.4e}   |
| Rel L2 Error      | {trial.rel_l2_error:.4e}   |
| Elapsed Time      | {trial.elapsed_time:.2f} s   |

**Best so far** — Trial #{best_so_far.trial_id}: objective = {best_so_far.objective:.4e}

---

"""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(section)

    def finalize(self, result: LLMResult) -> str:
        """Append the final best-result summary and convergence table.

        Parameters
        ----------
        result:
            Final ``LLMResult`` returned by ``LLMOptimizer.optimize()``.

        Returns
        -------
        str
            Absolute path of the completed report file.
        """
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        best = next(t for t in result.trials if t.trial_id == result.best_trial_id)
        best_param_rows = "\n".join([
            f"| {name:<17} | {val}   |"
            for name, val in result.best_params.items()
        ])

        convergence_rows = build_convergence_table(result.trials)

        footer = f"""## 4. Final Best Result

Completed: {now}

**Trial ID**: {result.best_trial_id}
**Objective**: {result.best_objective:.4e}

| Hyperparameter    | Value  |
|-------------------|--------|
{best_param_rows}

**Metrics**:
- Relative L2 Error: {best.rel_l2_error:.4e}
- Elapsed Time: {best.elapsed_time:.2f} s

---

## 5. Convergence

Best objective per trial (cumulative max):

| Trial | Best Objective So Far |
|-------|-----------------------|
{convergence_rows}
"""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(footer)
        return self._path
