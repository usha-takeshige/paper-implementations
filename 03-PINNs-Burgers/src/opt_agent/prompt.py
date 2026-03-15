"""Prompt builder for LLM-guided hyperparameter optimization."""

from bo.result import TrialResult
from bo.space import SearchSpace


class PromptBuilder:
    """Build LangChain prompt strings from search space and trial history.

    All methods are static; this class acts as a namespace for prompt
    construction logic, keeping it separate from the API call logic in
    GeminiChain.
    """

    @staticmethod
    def build_system_prompt(
        search_space: SearchSpace,
        objective_name: str,
    ) -> str:
        """Build the system prompt describing the optimization task.

        Includes: task description, search space definition (name, type,
        bounds, scale), objective function description, and output format.

        Parameters
        ----------
        search_space:
            Search space definition with parameter bounds and types.
        objective_name:
            Name of the objective function being optimized.

        Returns
        -------
        str
            System prompt string.
        """
        scale_label = {True: "log", False: "linear"}
        param_lines = []
        for hp in search_space.parameters:
            param_lines.append(
                f"  - {hp.name}: type={hp.param_type}, "
                f"low={hp.low}, high={hp.high}, scale={scale_label[hp.log_scale]}"
            )
        params_str = "\n".join(param_lines)

        return f"""You are an expert machine learning hyperparameter optimizer.
Your goal is to propose the next hyperparameter configuration to maximize the objective function.

## Task
Optimize hyperparameters for a Physics-Informed Neural Network (PINN) solving Burgers' equation.
The objective function is: {objective_name}
Higher objective values are better.

## Search Space
{params_str}

## Instructions
1. Analyze the provided trial history to identify patterns and trends.
2. Propose the next hyperparameter configuration to explore.
3. Focus on regions of the search space that are likely to yield high objective values.
4. Respect the bounds of each parameter strictly.
5. For integer parameters (int type), propose integer values only.
6. For log-scale parameters, consider exploring values across multiple orders of magnitude.

## Output Format
Return a JSON object with these fields:
- analysis_report: Your analysis of the current search history (what has worked, what hasn't)
- proposed_params: A dictionary with the next hyperparameter values to evaluate
- reasoning: Your reasoning for choosing these specific parameter values
"""

    @staticmethod
    def build_human_prompt(
        trials: list[TrialResult],
        iteration_id: int,
    ) -> str:
        """Build the human message describing current trial history.

        Includes: current iteration number, all trial results as a table,
        and the current best configuration.

        Parameters
        ----------
        trials:
            All trials evaluated so far (initial + LLM-guided).
        iteration_id:
            Current LLM iteration number (0-based).

        Returns
        -------
        str
            Human prompt string.
        """
        best = max(trials, key=lambda t: t.objective)

        # Build trial table
        header = "| Trial | Type    | n_hidden_layers | n_neurons | lr       | epochs_adam | rel_l2_error | objective  |"
        sep =    "|-------|---------|-----------------|-----------|----------|-------------|--------------|------------|"
        rows = []
        for t in trials:
            trial_type = "initial" if t.is_initial else "LLM"
            lr_val = t.params.get("lr", 0.0)
            row = (
                f"| {t.trial_id:<5} | {trial_type:<7} | "
                f"{t.params.get('n_hidden_layers', '-'):<15} | "
                f"{t.params.get('n_neurons', '-'):<9} | "
                f"{float(lr_val):.2e} | "
                f"{t.params.get('epochs_adam', '-'):<11} | "
                f"{t.rel_l2_error:.4e}     | "
                f"{t.objective:.4e}   |"
            )
            rows.append(row)
        table = "\n".join([header, sep] + rows)

        return f"""## Iteration {iteration_id + 1}

### Trial History ({len(trials)} trials so far)

{table}

### Current Best
- Trial ID: {best.trial_id}
- Objective: {best.objective:.4e}
- Parameters: {best.params}
- Rel L2 Error: {best.rel_l2_error:.4e}

Please analyze the above results and propose the next hyperparameter configuration.
"""
