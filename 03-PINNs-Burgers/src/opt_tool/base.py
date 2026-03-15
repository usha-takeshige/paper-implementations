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

    def optimize(self) -> BaseOptimizationResult:
        """Run the full optimization loop and return the result.

        Executes the three-phase template:
        Phase 1 — Sobol initial sampling (shared implementation).
        Phase 2 — Sequential search (algorithm-specific, calls _run_sequential_search).
        Phase 3 — Result aggregation (algorithm-specific, calls _build_result).

        Returns
        -------
        BaseOptimizationResult
            Full optimization result (concrete subtype depends on subclass).
        """
        initial_trials = self._run_initial_exploration()
        all_trials = self._run_sequential_search(initial_trials)
        return self._build_result(all_trials)

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
        n_width = len(str(self._config.n_initial))
        for i in range(self._config.n_initial):
            params = self._search_space.from_tensor(samples[i])
            trial = self._objective(params=params, trial_id=i, is_initial=True)
            trials.append(trial)
            label = f"[Initial {i + 1:>{n_width}}/{self._config.n_initial}]"
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
