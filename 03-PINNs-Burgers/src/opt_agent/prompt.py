"""Prompt builders for LLM-guided hyperparameter optimization."""

from abc import ABC, abstractmethod

from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


class PromptBuilder(ABC):
    """Abstract base class for LLM prompt construction.

    Subclasses define the optimization goal communicated to the LLM via the
    system and human prompts.  Two concrete implementations are provided:

    - ``MaximizeObjectivePromptBuilder``: instructs the LLM to directly
      maximize the objective value (standalone LLM optimization).
    - ``NarrowSearchSpacePromptBuilder``: instructs the LLM to explore broadly
      and identify promising regions (Hybrid Phase 1).
    """

    @abstractmethod
    def build_system_prompt(
        self,
        search_space: SearchSpace,
        objective_name: str,
    ) -> str:
        """Build the system prompt describing the optimization task.

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
        ...

    @abstractmethod
    def build_human_prompt(
        self,
        trials: list[TrialResult],
        iteration_id: int,
    ) -> str:
        """Build the human message describing current trial history.

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
        ...

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trial_table(trials: list[TrialResult]) -> str:
        """Format trials as a Markdown table string."""
        header = (
            "| Trial | Type    | n_hidden_layers | n_neurons | lr       "
            "| epochs_adam | rel_l2_error | objective  |"
        )
        sep = (
            "|-------|---------|-----------------|-----------|----------"
            "|-------------|--------------|------------|"
        )
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
        return "\n".join([header, sep] + rows)


class MaximizeObjectivePromptBuilder(PromptBuilder):
    """Prompt builder that instructs the LLM to maximize the objective function.

    Used for standalone LLM optimization where the goal is to find
    hyperparameters that directly maximize the objective value.
    """

    def build_system_prompt(
        self,
        search_space: SearchSpace,
        objective_name: str,
    ) -> str:
        """Build system prompt focused on maximizing the objective."""
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

    def build_human_prompt(
        self,
        trials: list[TrialResult],
        iteration_id: int,
    ) -> str:
        """Build human prompt with trial history for objective maximization."""
        best = max(trials, key=lambda t: t.objective)
        table = self._build_trial_table(trials)

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


class NarrowSearchSpacePromptBuilder(PromptBuilder):
    """Prompt builder that instructs the LLM to identify promising regions.

    Used in the Hybrid optimizer Phase 1, where the LLM's role is to broadly
    explore and identify the regions of the search space most likely to
    contain the optimum.  The resulting exploration data is used to narrow
    the search space for Bayesian Optimization in Phase 2.
    """

    def build_system_prompt(
        self,
        search_space: SearchSpace,
        objective_name: str,
    ) -> str:
        """Build system prompt focused on identifying promising parameter regions."""
        scale_label = {True: "log", False: "linear"}
        param_lines = []
        for hp in search_space.parameters:
            param_lines.append(
                f"  - {hp.name}: type={hp.param_type}, "
                f"low={hp.low}, high={hp.high}, scale={scale_label[hp.log_scale]}"
            )
        params_str = "\n".join(param_lines)

        return f"""You are an expert machine learning hyperparameter optimizer.
Your goal is to identify the most promising regions of the search space through broad exploration.

## Task
Optimize hyperparameters for a Physics-Informed Neural Network (PINN) solving Burgers' equation.
The objective function is: {objective_name}
Higher objective values are better.

You are in the EXPLORATION PHASE of a two-phase optimization.
Your proposals will be used to identify promising regions of the search space.
A Bayesian optimizer will then refine the search within those narrowed regions.

## Search Space
{params_str}

## Instructions
1. Analyze the trial history to identify which regions of the search space are
   most promising (high objective values).
2. Propose the next configuration to explore an under-sampled but potentially
   promising region of the search space.
3. Aim for DIVERSITY: avoid clustering too closely around previously evaluated points.
4. If the history reveals a promising region, explore its boundaries to help
   define the range for the next optimization phase.
5. Respect the bounds of each parameter strictly.
6. For integer parameters (int type), propose integer values only.
7. For log-scale parameters, consider exploring values across multiple orders of magnitude.

## Output Format
Return a JSON object with these fields:
- analysis_report: Your analysis identifying the most promising regions discovered so far
- proposed_params: A dictionary with the next hyperparameter values to evaluate
- reasoning: Your reasoning for choosing these values to maximize search space coverage
"""

    def build_human_prompt(
        self,
        trials: list[TrialResult],
        iteration_id: int,
    ) -> str:
        """Build human prompt with trial history for search space exploration."""
        best = max(trials, key=lambda t: t.objective)
        table = self._build_trial_table(trials)

        return f"""## Iteration {iteration_id + 1}

### Trial History ({len(trials)} trials so far)

{table}

### Current Best
- Trial ID: {best.trial_id}
- Objective: {best.objective:.4e}
- Parameters: {best.params}
- Rel L2 Error: {best.rel_l2_error:.4e}

Please analyze the search history to identify the most promising regions,
then propose the next configuration to explore. Focus on coverage and identifying
the boundaries of high-performing regions of the search space.
"""
