"""Configuration and result data classes for LLM-based hyperparameter optimization."""

from dataclasses import dataclass

from opt_tool.base import BaseOptimizationResult, BaseOptimizerConfig
from opt_tool.result import TrialResult


@dataclass(frozen=True)
class LLMConfig(BaseOptimizerConfig):
    """Configuration for LLM-based hyperparameter optimization.

    Holds the parameters controlling the optimization loop: number of initial
    Sobol samples, number of LLM-guided iterations, and the random seed.
    Mirrors BOConfig fields for fair comparison.

    Inherits n_initial, n_iterations, seed from BaseOptimizerConfig.
    """


@dataclass(frozen=True)
class LLMIterationMeta:
    """Metadata for a single LLM-guided iteration.

    Stores the analysis report, proposed parameters, and reasoning produced
    by the LLM during Phase 2 of the optimization loop.
    """

    iteration_id: int
    """Iteration number (0-based) within the LLM-guided phase."""

    analysis_report: str
    """LLM-generated analysis of the current exploration state (natural language)."""

    proposed_params: dict[str, float | int]
    """Parameters proposed by the LLM (original scale)."""

    reasoning: str
    """LLM-generated reasoning for the proposed parameters (natural language)."""

    proposal_time: float
    """LLM の chain.invoke() にかかった経過時間 [秒]"""


@dataclass(frozen=True)
class LLMResult(BaseOptimizationResult):
    """Result of the entire LLM-based optimization run.

    Aggregates all trial results and identifies the best-performing
    hyperparameter configuration along with LLM reasoning metadata.
    Mirrors BOResult for fair comparison.
    Inherits trials, best_params, best_objective, best_trial_id, objective_name
    from BaseOptimizationResult.
    """

    llm_config: LLMConfig
    """LLM optimization configuration used in this run."""

    iteration_metas: list[LLMIterationMeta]
    """LLM reasoning metadata for each LLM-guided iteration."""
