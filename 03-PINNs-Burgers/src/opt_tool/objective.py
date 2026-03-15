"""Abstract base class for hyperparameter optimization objective functions."""

from abc import ABC, abstractmethod

from opt_tool.result import TrialResult


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions.

    Subclasses implement ``_compute_objective`` to define the scalar value
    derived from evaluation metrics after model training.
    Optimizers maximize the returned objective value.

    Common evaluation logic is delegated to subclasses via ``__call__``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the objective function."""
        ...

    @abstractmethod
    def __call__(
        self, params: dict[str, float | int], trial_id: int, is_initial: bool
    ) -> TrialResult:
        """Evaluate the objective at the given hyperparameters.

        Parameters
        ----------
        params:
            Hyperparameter values in the original scale.
        trial_id:
            Sequential trial identifier (0-based).
        is_initial:
            True if this trial comes from the Sobol initial sampling phase.

        Returns
        -------
        TrialResult
            Contains objective, rel_l2_error, elapsed_time, and metadata.
        """
        ...
