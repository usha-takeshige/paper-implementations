"""Hybrid LLM + Bayesian Optimization hyperparameter optimizer (Facade)."""

import dataclasses
import math

from bo.optimizer import BayesianOptimizer
from bo.result import BOConfig
from hybrid.result import HybridResult
from opt_agent.chain import BaseChain
from opt_agent.config import LLMConfig
from opt_agent.optimizer import LLMOptimizer
from opt_tool.objective import ObjectiveFunction
from opt_tool.result import TrialResult
from opt_tool.space import HyperParameter, SearchSpace


class HybridOptimizer:
    """2-phase hyperparameter optimizer combining LLM exploration and BO exploitation.

    Phase 1 uses LLMOptimizer to broadly explore the original search space.
    The top-performing trials from Phase 1 are used to narrow the search space.
    Phase 2 then runs BayesianOptimizer within the narrowed space for precise
    local convergence.

    This class acts as a Facade over LLMOptimizer and BayesianOptimizer and
    does NOT inherit BaseOptimizer, because HybridResult stores two separate
    trial lists (llm_trials / bo_trials) which is incompatible with the
    Template Method structure of BaseOptimizer.

    Parameters
    ----------
    search_space:
        Original search space defining hyperparameter bounds.
    objective:
        Objective function to maximize.
    llm_config:
        Configuration for the LLM optimizer (Phase 1).
    bo_config:
        Configuration for the Bayesian optimizer (Phase 2).
    chain:
        LLM chain implementation for Phase 1. If None, GeminiChain is built
        from GEMINI_API_KEY and GEMINI_MODEL_NAME environment variables
        (delegated to LLMOptimizer).
    n_llm_iterations:
        Number of LLM-guided iterations in Phase 1 (overrides
        llm_config.n_iterations using dataclasses.replace).
    top_k_ratio:
        Fraction of Phase 1 trials used to derive the narrowed search space.
        For example, 0.3 selects the top 30% by objective.
    margin_ratio:
        Margin added beyond the top-k range when narrowing bounds.
        Applied as a fraction of the original parameter range (or log-range).
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        llm_config: LLMConfig,
        bo_config: BOConfig,
        chain: BaseChain | None = None,
        n_llm_iterations: int = 10,
        top_k_ratio: float = 0.3,
        margin_ratio: float = 0.1,
    ) -> None:
        """Initialize HybridOptimizer."""
        self._search_space = search_space
        self._objective = objective
        self._llm_config = llm_config
        self._bo_config = bo_config
        self._chain = chain
        self._n_llm_iterations = n_llm_iterations
        self._top_k_ratio = top_k_ratio
        self._margin_ratio = margin_ratio

    def optimize(self) -> HybridResult:
        """Run the 2-phase hybrid optimization pipeline.

        Returns
        -------
        HybridResult
            Combined result containing Phase 1 trials, narrowed search space,
            Phase 2 trials, and the overall best hyperparameters.
        """
        # Phase 1: LLM-guided broad exploration
        print("=== Phase 1: LLM Exploration ===")
        modified_llm_config = dataclasses.replace(
            self._llm_config, n_iterations=self._n_llm_iterations
        )
        llm_optimizer = LLMOptimizer(
            search_space=self._search_space,
            objective=self._objective,
            config=modified_llm_config,
            chain=self._chain,
        )
        llm_result = llm_optimizer.optimize()
        llm_trials = llm_result.trials

        # Search space narrowing
        print("\n=== Narrowing Search Space ===")
        narrowed_space = self._narrow_search_space(llm_trials)
        self._print_narrowing(self._search_space, narrowed_space)

        # Phase 2: Bayesian Optimization on narrowed space
        print("\n=== Phase 2: Bayesian Optimization ===")
        bo_optimizer = BayesianOptimizer(
            search_space=narrowed_space,
            objective=self._objective,
            config=self._bo_config,
        )
        bo_result = bo_optimizer.optimize()
        bo_trials = bo_result.trials

        # Result aggregation: find best across all trials
        combined = llm_trials + bo_trials
        best_idx = max(range(len(combined)), key=lambda i: combined[i].objective)
        best_trial = combined[best_idx]

        return HybridResult(
            llm_trials=llm_trials,
            bo_trials=bo_trials,
            narrowed_space=narrowed_space,
            best_params=best_trial.params,
            best_objective=best_trial.objective,
            best_trial_id=best_idx,
            objective_name=self._objective.name,
            llm_config=self._llm_config,
            bo_config=self._bo_config,
        )

    def _narrow_search_space(self, trials: list[TrialResult]) -> SearchSpace:
        """Build a narrowed SearchSpace from the top Phase 1 trials.

        Selects the top ``ceil(len(trials) * top_k_ratio)`` trials by
        objective value. For linear-scale parameters, the margin is applied
        in the original scale. For log-scale parameters, the margin is
        applied in log space and the result is converted back with exp.

        If the computed ``new_low >= new_high`` (e.g., all top-k trials
        share the same value and margin_ratio=0), falls back to the
        original parameter bounds.

        Parameters
        ----------
        trials:
            All Phase 1 trials sorted or unsorted (will be sorted internally).

        Returns
        -------
        SearchSpace
            New SearchSpace whose bounds are a subset of the original bounds.
        """
        n_top = max(1, math.ceil(len(trials) * self._top_k_ratio))
        top_trials = sorted(trials, key=lambda t: t.objective, reverse=True)[:n_top]

        new_params: list[HyperParameter] = []
        for hp in self._search_space.parameters:
            values = [float(t.params[hp.name]) for t in top_trials]

            if hp.log_scale:
                log_low = math.log(hp.low)
                log_high = math.log(hp.high)
                log_range = log_high - log_low
                log_vals = [math.log(v) for v in values]
                new_log_low = min(log_vals) - self._margin_ratio * log_range
                new_log_high = max(log_vals) + self._margin_ratio * log_range
                new_low = math.exp(max(new_log_low, log_low))
                new_high = math.exp(min(new_log_high, log_high))
            else:
                linear_range = hp.high - hp.low
                new_low = max(hp.low, min(values) - self._margin_ratio * linear_range)
                new_high = min(hp.high, max(values) + self._margin_ratio * linear_range)

            # Fallback to original bounds if narrowing collapses
            if new_low >= new_high:
                new_low = hp.low
                new_high = hp.high

            new_params.append(
                HyperParameter(
                    name=hp.name,
                    param_type=hp.param_type,
                    low=new_low,
                    high=new_high,
                    log_scale=hp.log_scale,
                )
            )

        return SearchSpace(parameters=new_params)

    def _print_narrowing(
        self, original: SearchSpace, narrowed: SearchSpace
    ) -> None:
        """Print a summary of the search space narrowing to stdout.

        Parameters
        ----------
        original:
            Original search space before narrowing.
        narrowed:
            Narrowed search space after Phase 1.
        """
        print(f"  top_k_ratio={self._top_k_ratio}, margin_ratio={self._margin_ratio}")
        for orig_hp, narr_hp in zip(original.parameters, narrowed.parameters):
            print(
                f"  {orig_hp.name}: [{orig_hp.low:.4g}, {orig_hp.high:.4g}]"
                f" → [{narr_hp.low:.4g}, {narr_hp.high:.4g}]"
            )
