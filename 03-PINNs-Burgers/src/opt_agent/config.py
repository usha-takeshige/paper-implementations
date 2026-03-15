"""Configuration and result data classes for LLM-based hyperparameter optimization."""

from dataclasses import dataclass, field

from bo.result import TrialResult


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM-based hyperparameter optimization.

    Holds the parameters controlling the optimization loop: number of initial
    Sobol samples, number of LLM-guided iterations, and the random seed.
    Mirrors BOConfig fields for fair comparison.
    """

    n_initial: int = 5
    """Number of initial Sobol quasi-random samples (same as BOConfig.n_initial)."""

    n_iterations: int = 20
    """Number of LLM-guided iterations (same as BOConfig.n_iterations)."""

    seed: int = 42
    """Random seed for Sobol sampling reproducibility."""


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


@dataclass(frozen=True)
class LLMResult:
    """Result of the entire LLM-based optimization run.

    Aggregates all trial results and identifies the best-performing
    hyperparameter configuration along with LLM reasoning metadata.
    Mirrors BOResult for fair comparison.
    """

    trials: list[TrialResult]
    """All trial results (initial Sobol + LLM-guided)."""

    best_params: dict[str, float | int]
    """Hyperparameters from the trial with the highest objective."""

    best_objective: float
    """Maximum objective value across all trials."""

    best_trial_id: int
    """trial_id of the best trial."""

    llm_config: LLMConfig
    """LLM optimization configuration used in this run."""

    objective_name: str
    """Name of the ObjectiveFunction used."""

    iteration_metas: list[LLMIterationMeta]
    """LLM reasoning metadata for each LLM-guided iteration."""
