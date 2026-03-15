"""Markdown report generator for Bayesian optimization results."""

import os
from datetime import datetime, timezone

from bo.result import BOResult
from opt_tool.report_utils import build_convergence_table, build_search_space_table
from opt_tool.space import SearchSpace


class ReportGenerator:
    """Generate a markdown report from Bayesian optimization results.

    Writes a structured report to bo_report.md in the specified output
    directory, covering: BO configuration, best result, all trial details,
    and convergence (cumulative best objective per trial).
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize ReportGenerator.

        Parameters
        ----------
        output_dir:
            Directory where bo_report.md will be saved (e.g. "example/output").
        """
        self._output_dir = output_dir

    def generate(self, result: BOResult, search_space: SearchSpace) -> str:
        """Generate and save a markdown report from BOResult.

        Parameters
        ----------
        result:
            Return value of BayesianOptimizer.optimize().
        search_space:
            Search space definition included in the report header.

        Returns
        -------
        str
            Absolute path of the saved bo_report.md file.
        """
        content = self._build_report(result, search_space)
        os.makedirs(self._output_dir, exist_ok=True)
        output_path = os.path.join(self._output_dir, "bo_report.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def _build_report(self, result: BOResult, search_space: SearchSpace) -> str:
        """Build the full markdown report string."""
        cfg = result.bo_config
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Section 1: Configuration ---
        config_rows = "\n".join([
            f"| objective         | {result.objective_name} |",
            f"| n_initial         | {cfg.n_initial}   |",
            f"| n_iterations      | {cfg.n_iterations}   |",
            f"| acquisition       | {cfg.acquisition} |",
            f"| seed              | {cfg.seed}   |",
            f"| num_restarts      | {cfg.num_restarts}   |",
            f"| raw_samples       | {cfg.raw_samples}   |",
        ])

        space_rows = build_search_space_table(search_space)

        # --- Section 2: Best result ---
        best = next(t for t in result.trials if t.trial_id == result.best_trial_id)
        best_param_rows = "\n".join([
            f"| {name:<17} | {val}   |"
            for name, val in result.best_params.items()
        ])

        # --- Section 3: All trials ---
        trial_rows = []
        for t in result.trials:
            trial_type = "initial" if t.is_initial else "BO"
            row = (
                f"| {t.trial_id:<5} | {trial_type:<7} | "
                f"{t.params.get('n_hidden_layers', '-'):<8} | "
                f"{t.params.get('n_neurons', '-'):<9} | "
                f"{t.params.get('lr', 0.0):.2e} | "
                f"{t.params.get('epochs_adam', '-'):<11} | "
                f"{t.rel_l2_error:.4e}     | "
                f"{t.elapsed_time:.2f}     | "
                f"{t.objective:.4e}   |"
            )
            trial_rows.append(row)
        all_trials_str = "\n".join(trial_rows)

        # --- Section 4: Convergence ---
        convergence_rows = build_convergence_table(result.trials)

        return f"""# Bayesian Optimization Report
## Burgers PINNs Hyperparameter Tuning

Generated: {now}

---

## 1. Configuration

| Parameter         | Value  |
|-------------------|--------|
{config_rows}

### Search Space

| Hyperparameter    | Type  | Low    | High   | Scale  |
|-------------------|-------|--------|--------|--------|
{space_rows}

---

## 2. Best Result

**Trial ID**: {result.best_trial_id}
**Objective**: {result.best_objective:.4e}

| Hyperparameter    | Value  |
|-------------------|--------|
{best_param_rows}

**Metrics**:
- Relative L2 Error: {best.rel_l2_error:.4e}
- Elapsed Time: {best.elapsed_time:.2f} s

---

## 3. All Trials

| Trial | Type    | n_layers | n_neurons | lr       | epochs_adam | Rel L2 Error | Time (s) | Objective  |
|-------|---------|----------|-----------|----------|-------------|--------------|----------|------------|
{all_trials_str}

> **Type**: `initial` = Sobol initial sample, `BO` = Bayesian optimization proposal

---

## 4. Convergence

Best objective per trial (cumulative max):

| Trial | Best Objective So Far |
|-------|-----------------------|
{convergence_rows}
"""
