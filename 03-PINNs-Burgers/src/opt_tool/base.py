"""Base classes for hyperparameter optimizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from opt_tool.objective import ObjectiveFunction
from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


@dataclass(frozen=True)
class BaseOptimizerConfig:
    """Common configuration for all optimizers.

    Holds shared hyperparameters controlling the optimization loop:
    number of initial Sobol samples, number of sequential iterations, and seed.
    Subclasses extend this with algorithm-specific fields.
    """

    n_initial: int = 5
    n_iterations: int = 20
    seed: int = 42


@dataclass(frozen=True)
class BaseOptimizationResult:
    """Common result fields shared by all optimizers.

    Stores the trials, best configuration, and objective function name that
    are common to every optimization result. Subclasses extend with
    algorithm-specific fields (e.g. bo_config or llm_config).
    """

    trials: list[TrialResult]
    best_params: dict[str, float | int]
    best_objective: float
    best_trial_id: int
    objective_name: str


class BaseOptimizer(ABC):
    """Abstract base class implementing the Template Method pattern for 3-phase optimization.

    Provides a common skeleton for hyperparameter optimization loops:
    - Phase 1: Sobol quasi-random initial exploration (shared implementation)
    - Phase 2: Algorithm-specific sequential search (abstract, implemented by subclasses)
    - Phase 3: Result aggregation (abstract, implemented by subclasses)

    Subclasses implement ``_run_sequential_search`` and ``_build_result``.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        config: BaseOptimizerConfig,
    ) -> None:
        """Initialize the optimizer.

        Parameters
        ----------
        search_space:
            Search space defining the hyperparameter domain.
        objective:
            Callable that evaluates a set of hyperparameters.
        config:
            Optimizer configuration (n_initial, n_iterations, seed, etc.).
        """
        self._search_space = search_space
        self._objective = objective
        self._config = config

    def optimize(
        self,
        warm_start_trials: list[TrialResult] | None = None,
    ) -> BaseOptimizationResult:
        """Run the full optimization loop and return the result.

        Executes the three-phase template:
        Phase 1 — Sobol initial sampling (shared implementation).
        Phase 2 — Sequential search (algorithm-specific, calls _run_sequential_search).
        Phase 3 — Result aggregation (algorithm-specific, calls _build_result).

        Parameters
        ----------
        warm_start_trials:
            Optional list of already-evaluated trials to reuse as initial data.
            Trials outside the current search space bounds are silently dropped.
            Remaining slots (up to n_initial) are filled with Sobol sampling.

        Returns
        -------
        BaseOptimizationResult
            Full optimization result (concrete subtype depends on subclass).
        """
        initial_trials = self._run_initial_exploration(warm_start_trials)
        all_trials = self._run_sequential_search(initial_trials)
        return self._build_result(all_trials)

    def _run_initial_exploration(
        self,
        warm_start_trials: list[TrialResult] | None = None,
    ) -> list[TrialResult]:
        """Phase 1: Reuse warm-start trials then fill remaining slots with Sobol.

        Warm-start trials that fall within the current search space bounds are
        adopted without re-evaluation. Any remaining slots up to ``n_initial``
        are filled by sampling from the Sobol sequence and evaluating the
        objective. If no warm-start trials are provided, the behaviour is
        identical to the original pure-Sobol initialisation.

        Parameters
        ----------
        warm_start_trials:
            Optional already-evaluated trials to reuse as initial data.

        Returns
        -------
        list[TrialResult]
            Trial results for the initial phase (warm-start + Sobol fill-up).
        """
        # Filter warm-start trials that lie within the current search space
        valid_warm: list[TrialResult] = []
        if warm_start_trials:
            valid_warm = [
                t for t in warm_start_trials
                if all(
                    hp.low <= t.params[hp.name] <= hp.high
                    for hp in self._search_space.parameters
                )
            ]

        # Determine how many new Sobol samples are needed to reach n_initial
        n_fill = max(0, self._config.n_initial - len(valid_warm))

        trials: list[TrialResult] = []

        # Adopt warm-start trials without re-evaluation
        for t in valid_warm:
            trials.append(t)
            self._log_trial(t, "[Warm    ]")

        # Sample and evaluate Sobol fill-up points
        if n_fill > 0:
            samples = self._search_space.sample_sobol(
                n=n_fill,
                seed=self._config.seed,
            )
            n_width = len(str(n_fill))
            for i in range(n_fill):
                params = self._search_space.from_tensor(samples[i])
                trial_id = len(valid_warm) + i
                trial = self._objective(params=params, trial_id=trial_id, is_initial=True)
                trials.append(trial)
                label = f"[Initial {i + 1:>{n_width}}/{n_fill}]"
                self._log_trial(trial, label)

        return trials

    @abstractmethod
    def _run_sequential_search(
        self, initial_trials: list[TrialResult]
    ) -> list[TrialResult]:
        """Phase 2: Run algorithm-specific sequential search.

        Parameters
        ----------
        initial_trials:
            Trial results from Phase 1 (Sobol initial exploration).

        Returns
        -------
        list[TrialResult]
            All trials including initial_trials and newly evaluated points.
        """
        ...

    @abstractmethod
    def _build_result(self, trials: list[TrialResult]) -> BaseOptimizationResult:
        """Phase 3: Aggregate all trials into the final result object.

        Parameters
        ----------
        trials:
            All trials from Phase 1 and Phase 2.

        Returns
        -------
        BaseOptimizationResult
            Final result (concrete subtype depends on subclass).
        """
        ...

    @staticmethod
    def _log_trial(trial: TrialResult, label: str) -> None:
        """Print a single trial result to stdout.

        Parameters
        ----------
        trial:
            The trial result to log.
        label:
            Short label displayed at the beginning of the line
            (e.g. "[Initial 1/5]" or "[BO     1/20]").
        """
        params_str = "  ".join(
            f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in trial.params.items()
        )
        print(
            f"{label}  {params_str}"
            f"  | L2={trial.rel_l2_error:.4e}"
            f"  | Time={trial.elapsed_time:.2f}s"
            f"  | Obj={trial.objective:.4e}"
        )
