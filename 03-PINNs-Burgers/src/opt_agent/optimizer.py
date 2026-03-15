"""LLM-based hyperparameter optimizer (Facade)."""

from collections.abc import Callable
from typing import Optional

from opt_agent.chain import BaseChain, GeminiChain
from opt_agent.config import LLMConfig, LLMIterationMeta, LLMResult
from opt_tool.base import BaseOptimizer
from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


_INT_PARAMS = {"n_hidden_layers", "n_neurons", "epochs_adam"}

#: Callback called after each LLM-guided iteration.
#: Arguments: (meta, trial, all_trials_so_far)
IterationCallback = Callable[
    [LLMIterationMeta, TrialResult, list[TrialResult]], None
]


class LLMOptimizer(BaseOptimizer):
    """Optimize hyperparameters using an LLM (Gemini via LangChain).

    Provides the same interface as BayesianOptimizer for drop-in comparison.
    Three phases:
    - Phase 1: Sobol quasi-random initial exploration (via BaseOptimizer).
    - Phase 2: LLM-guided sequential search (_run_sequential_search).
    - Phase 3: Result aggregation (_build_result).

    The LLM chain can be injected for testing (Strategy pattern).
    When chain=None, GeminiChain is built automatically from environment
    variables GEMINI_API_KEY and GEMINI_MODEL_NAME.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective,
        config: LLMConfig = LLMConfig(),
        chain: BaseChain | None = None,
        on_iteration: Optional[IterationCallback] = None,
    ) -> None:
        """Initialize LLMOptimizer.

        Parameters
        ----------
        search_space:
            Search space defining hyperparameter bounds.
        objective:
            Objective function to maximize.
        config:
            LLM optimization configuration.
        chain:
            LLM chain implementation. If None, GeminiChain is built from
            GEMINI_API_KEY and GEMINI_MODEL_NAME environment variables.
        on_iteration:
            Optional callback invoked after each LLM-guided iteration.
            Called with (meta, trial, all_trials_so_far) immediately after
            the proposed point is evaluated.  Use this hook to write
            per-iteration reports or log streaming results.

        Raises
        ------
        ValueError
            If chain is None and GEMINI_API_KEY is not set.
        """
        super().__init__(search_space, objective, config)
        self._on_iteration = on_iteration
        self._metas: list[LLMIterationMeta] = []

        if chain is not None:
            self._chain: BaseChain = chain
        else:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. "
                    "Set it in .env or pass a chain argument."
                )
            model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")
            self._chain = GeminiChain(model_name=model_name, api_key=api_key)

    def _run_sequential_search(
        self, initial_trials: list[TrialResult]
    ) -> list[TrialResult]:
        """Phase 2: Iteratively ask the LLM for proposals and evaluate them.

        Out-of-bounds values are clamped to SearchSpace limits.
        Integer parameters (n_hidden_layers, n_neurons, epochs_adam) are
        converted with int().

        Parameters
        ----------
        initial_trials:
            Trial results from Phase 1.

        Returns
        -------
        list[TrialResult]
            All trials (initial + LLM-guided).
        """
        trials: list[TrialResult] = list(initial_trials)
        self._metas = []

        for i in range(self._config.n_iterations):
            proposal = self._chain.invoke(
                search_space=self._search_space,
                trials=trials,
                objective_name=self._objective.name,
                iteration_id=i,
            )

            # Clamp and round proposed parameters
            params = self._clamp_params(proposal.proposed_params)

            trial_id = len(trials)
            trial = self._objective(params=params, trial_id=trial_id, is_initial=False)
            trials.append(trial)
            label = f"[LLM    {i + 1:>{len(str(self._config.n_iterations))}}/{self._config.n_iterations}]"
            self._log_trial(trial, label)

            meta = LLMIterationMeta(
                iteration_id=i,
                analysis_report=proposal.analysis_report,
                proposed_params=params,
                reasoning=proposal.reasoning,
            )
            self._metas.append(meta)

            if self._on_iteration is not None:
                self._on_iteration(meta, trial, list(trials))

        return trials

    def _build_result(self, trials: list[TrialResult]) -> LLMResult:
        """Phase 3: Select the best trial and build LLMResult.

        Parameters
        ----------
        trials:
            All trials from Phase 1 and Phase 2.

        Returns
        -------
        LLMResult
            Final result with best params and all metadata.
        """
        best = max(trials, key=lambda t: t.objective)
        return LLMResult(
            trials=trials,
            best_params=best.params,
            best_objective=best.objective,
            best_trial_id=best.trial_id,
            objective_name=self._objective.name,
            llm_config=self._config,
            iteration_metas=self._metas,
        )

    def _clamp_params(
        self,
        proposed: dict[str, float | int],
    ) -> dict[str, float | int]:
        """Clamp proposed parameters to SearchSpace bounds and round integers.

        Parameters
        ----------
        proposed:
            Raw parameters proposed by the LLM (may be out of bounds).

        Returns
        -------
        dict[str, float | int]
            Clamped and type-corrected parameters.
        """
        clamped: dict[str, float | int] = {}
        for hp in self._search_space.parameters:
            val = float(proposed.get(hp.name, hp.low))
            val = max(float(hp.low), min(float(hp.high), val))
            if hp.param_type == "int" or hp.name in _INT_PARAMS:
                clamped[hp.name] = int(val)
            else:
                clamped[hp.name] = val
        return clamped
