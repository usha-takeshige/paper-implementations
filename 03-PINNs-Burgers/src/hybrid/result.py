"""Result dataclass for the Hybrid LLM + BO optimizer."""

from dataclasses import dataclass

from bo.result import BOConfig
from opt_agent.config import LLMConfig
from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


@dataclass(frozen=True)
class HybridResult:
    """Immutable result of the 2-phase hybrid LLM + BO optimization.

    Stores Phase 1 (LLM-guided) and Phase 2 (BO-guided) trials separately,
    along with the narrowed search space used in Phase 2 and the overall best.

    Unlike BOResult / LLMResult, this class does NOT inherit from
    BaseOptimizationResult because the trial results are stored in two
    separate lists (llm_trials and bo_trials) rather than a single list.

    Attributes
    ----------
    llm_trials:
        All trials evaluated during Phase 1 (Sobol initial + LLM-guided).
    bo_trials:
        All trials evaluated during Phase 2 (Sobol initial + BO-guided).
    narrowed_space:
        Search space derived from top Phase 1 trials, used in Phase 2.
    best_params:
        Hyperparameter values of the best trial across both phases.
    best_objective:
        Maximum objective value across all Phase 1 and Phase 2 trials.
    best_trial_id:
        Index of the best trial in the combined ``llm_trials + bo_trials`` list.
    objective_name:
        Name of the objective function used for optimization.
    llm_config:
        LLM configuration used in Phase 1 (original user-provided config).
    bo_config:
        Bayesian optimization configuration used in Phase 2.
    """

    llm_trials: list[TrialResult]
    bo_trials: list[TrialResult]
    narrowed_space: SearchSpace
    best_params: dict[str, float | int]
    best_objective: float
    best_trial_id: int
    objective_name: str
    llm_config: LLMConfig
    bo_config: BOConfig
