"""Objective functions for Bayesian optimization of PINN hyperparameters."""

import time
from abc import ABC, abstractmethod

import numpy as np
import torch

from PINNs_Burgers import (
    BoundaryData,
    BurgersPINNSolver,
    CollocationPoints,
    NetworkConfig,
    PDEConfig,
    TrainingConfig,
)

from bo.result import TrialResult


class ObjectiveFunction(ABC):
    """Abstract base class for PINN objective functions.

    Subclasses implement ``_compute_objective`` to define the scalar value
    derived from ``rel_l2_error`` and ``elapsed_time`` after PINN training.
    ``BayesianOptimizer`` maximizes the returned objective value.

    Common PINN evaluation logic (training, inference, error computation)
    is provided by ``__call__`` in this base class.
    """

    def __init__(
        self,
        pde_config: PDEConfig,
        boundary_data: BoundaryData,
        collocation: CollocationPoints,
        x_mesh: np.ndarray,
        t_mesh: np.ndarray,
        usol: np.ndarray,
        base_training_config: TrainingConfig,
    ) -> None:
        """Initialize with fixed problem data shared across all trials.

        Parameters
        ----------
        pde_config:
            Fixed PDE configuration (ν etc.).
        boundary_data:
            Initial and boundary condition data for PINN training.
        collocation:
            Collocation points for PDE residual computation.
        x_mesh:
            Spatial evaluation grid (meshgrid format), shape (Nx, Nt).
        t_mesh:
            Temporal evaluation grid (meshgrid format), shape (Nx, Nt).
        usol:
            Reference solution, shape (Nx, Nt). Used to compute rel_l2_error.
        base_training_config:
            Base training config providing fixed values for n_u, n_f,
            and epochs_lbfgs. n_hidden_layers, n_neurons, lr, epochs_adam
            are overridden by the hyperparameter dict passed to __call__.
        """
        self._pde_config = pde_config
        self._boundary_data = boundary_data
        self._collocation = collocation
        self._x_mesh = x_mesh
        self._t_mesh = t_mesh
        self._usol = usol
        self._base_training_config = base_training_config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the objective function."""
        ...

    @abstractmethod
    def _compute_objective(self, rel_l2_error: float, elapsed_time: float) -> float:
        """Compute the scalar objective from evaluation metrics.

        Parameters
        ----------
        rel_l2_error:
            Relative L2 error of the PINN prediction.
        elapsed_time:
            Training elapsed time in seconds.

        Returns
        -------
        float
            Scalar objective value. Higher is better (BayesianOptimizer maximizes).
        """
        ...

    def __call__(
        self, params: dict[str, float | int], trial_id: int, is_initial: bool
    ) -> TrialResult:
        """Train a PINN with the given hyperparameters and return a TrialResult.

        Processing flow:
        1. Build NetworkConfig and TrainingConfig from params.
        2. Run BurgersPINNSolver.solve_forward() with perf_counter timing.
        3. Compute u_pred on the evaluation grid with torch.no_grad().
        4. Compute rel_l2_error = ||u_pred - usol||_F / ||usol||_F.
        5. Compute objective via _compute_objective(rel_l2_error, elapsed_time).
        6. Return TrialResult.

        Parameters
        ----------
        params:
            Hyperparameter values in the original scale. Must contain keys:
            n_hidden_layers, n_neurons, lr, epochs_adam.
        trial_id:
            Sequential trial identifier (0-based).
        is_initial:
            True if this trial comes from the Sobol initial sampling phase.

        Returns
        -------
        TrialResult
            Contains objective, rel_l2_error, elapsed_time, and metadata.
        """
        network_config = NetworkConfig(
            n_hidden_layers=int(params["n_hidden_layers"]),
            n_neurons=int(params["n_neurons"]),
        )
        training_config = TrainingConfig(
            n_u=self._base_training_config.n_u,
            n_f=self._base_training_config.n_f,
            lr=float(params["lr"]),
            epochs_adam=int(params["epochs_adam"]),
            epochs_lbfgs=self._base_training_config.epochs_lbfgs,
        )

        solver = BurgersPINNSolver(self._pde_config, network_config, training_config)
        start = time.perf_counter()
        result = solver.solve_forward(self._boundary_data, self._collocation)
        elapsed_time = time.perf_counter() - start

        model = result.model
        device = next(model.parameters()).device
        x_flat = torch.tensor(
            self._x_mesh.flatten(), dtype=torch.float32, device=device
        ).unsqueeze(1)
        t_flat = torch.tensor(
            self._t_mesh.flatten(), dtype=torch.float32, device=device
        ).unsqueeze(1)
        with torch.no_grad():
            u_pred_flat = model(t_flat, x_flat).cpu().numpy()
        u_pred = u_pred_flat.reshape(self._x_mesh.shape)

        rel_l2_error = float(
            np.linalg.norm(u_pred - self._usol) / np.linalg.norm(self._usol)
        )
        objective = self._compute_objective(rel_l2_error, elapsed_time)

        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=objective,
            rel_l2_error=rel_l2_error,
            elapsed_time=elapsed_time,
            is_initial=is_initial,
        )


class AccuracyObjective(ObjectiveFunction):
    """Objective that maximizes accuracy only.

    objective = -rel_l2_error

    Higher objective means lower relative L2 error (better accuracy).
    Elapsed training time is not considered.
    """

    @property
    def name(self) -> str:
        """Return the objective name."""
        return "Accuracy  -rel_l2_error"

    def _compute_objective(self, rel_l2_error: float, _elapsed_time: float) -> float:
        """Return the negated relative L2 error."""
        return -rel_l2_error


class AccuracySpeedObjective(ObjectiveFunction):
    """Objective that maximizes both accuracy and training speed.

    objective = 1 / max(rel_l2_error × elapsed_time, 1e-10)

    Higher objective means more accurate and faster training.
    """

    @property
    def name(self) -> str:
        """Return the objective name."""
        return "Accuracy × Speed  1/(rel_l2_error × elapsed_time)"

    def _compute_objective(self, rel_l2_error: float, elapsed_time: float) -> float:
        """Return the inverse of the error-time product."""
        return 1.0 / max(rel_l2_error * elapsed_time, 1e-10)
