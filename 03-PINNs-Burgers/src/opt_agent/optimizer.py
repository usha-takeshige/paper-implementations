"""LLM-based hyperparameter optimizer (Facade)."""

from bo.objective import ObjectiveFunction
from bo.result import TrialResult
from bo.space import SearchSpace
from opt_agent.chain import BaseChain, GeminiChain
from opt_agent.config import LLMConfig, LLMIterationMeta, LLMResult


_INT_PARAMS = {"n_hidden_layers", "n_neurons", "epochs_adam"}


class LLMOptimizer:
    """Optimize hyperparameters using an LLM (Gemini via LangChain).

    Provides the same interface as BayesianOptimizer for drop-in comparison.
    Three phases:
    - Phase 1: Sobol quasi-random initial exploration.
    - Phase 2: LLM-guided sequential search.
    - Phase 3: Result aggregation.

    The LLM chain can be injected for testing (Strategy pattern).
    When chain=None, GeminiChain is built automatically from environment
    variables GEMINI_API_KEY and GEMINI_MODEL_NAME.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        config: LLMConfig = LLMConfig(),
        chain: BaseChain | None = None,
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

        Raises
        ------
        ValueError
            If chain is None and GEMINI_API_KEY is not set.
        """
        self._search_space = search_space
        self._objective = objective
        self._config = config

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

    def optimize(self) -> LLMResult:
        """Run the full LLM-based optimization loop.

        Returns
        -------
        LLMResult
            Aggregated result including all trials and LLM metadata.
        """
        # Phase 1: Sobol initial exploration
        initial_trials = self._run_initial_exploration()

        # Phase 2: LLM-guided search
        all_trials, metas = self._run_llm_guided_search(initial_trials)

        # Phase 3: Aggregate results
        return self._aggregate_results(all_trials, metas)

    def _run_initial_exploration(self) -> list[TrialResult]:
        """Phase 1: Sample n_initial points with Sobol and evaluate each.

        Returns
        -------
        list[TrialResult]
            Trial results for the initial Sobol samples.
        """
        samples = self._search_space.sample_sobol(
            n=self._config.n_initial,
            seed=self._config.seed,
        )
        trials: list[TrialResult] = []
        for i in range(self._config.n_initial):
            params = self._search_space.from_tensor(samples[i])
            trial = self._objective(params=params, trial_id=i, is_initial=True)
            trials.append(trial)
            self._log_trial(trial, label="initial")
        return trials

    def _run_llm_guided_search(
        self,
        initial_trials: list[TrialResult],
    ) -> tuple[list[TrialResult], list[LLMIterationMeta]]:
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
        tuple[list[TrialResult], list[LLMIterationMeta]]
            All trials (initial + LLM-guided) and per-iteration LLM metadata.
        """
        trials: list[TrialResult] = list(initial_trials)
        metas: list[LLMIterationMeta] = []

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
            self._log_trial(trial, label=f"LLM [{i + 1}/{self._config.n_iterations}]")

            meta = LLMIterationMeta(
                iteration_id=i,
                analysis_report=proposal.analysis_report,
                proposed_params=params,
                reasoning=proposal.reasoning,
            )
            metas.append(meta)

        return trials, metas

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

    def _aggregate_results(
        self,
        trials: list[TrialResult],
        metas: list[LLMIterationMeta],
    ) -> LLMResult:
        """Phase 3: Select the best trial and build LLMResult.

        Parameters
        ----------
        trials:
            All trials from Phase 1 and Phase 2.
        metas:
            LLM metadata from each Phase 2 iteration.

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
            llm_config=self._config,
            objective_name=self._objective.name,
            iteration_metas=metas,
        )

    @staticmethod
    def _log_trial(trial: TrialResult, label: str) -> None:
        """Print a trial result summary to stdout."""
        print(
            f"  [{label}] trial={trial.trial_id:>3}  "
            f"obj={trial.objective:+.4e}  "
            f"rel_l2={trial.rel_l2_error:.4e}  "
            f"params={trial.params}"
        )
