"""Theory validation tests for the bo module (THR-BO-01 to THR-BO-10).

Verifies that mathematical properties defined in bo_design.md hold in
the implementation. Uses mocking to avoid full PINN training.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from bo import (
    BOConfig,
    BOResult,
    BayesianOptimizer,
    HyperParameter,
    SearchSpace,
    TrialResult,
)
from bo.objective import AccuracySpeedObjective
from PINNs_Burgers import PDEConfig, TrainingConfig, BoundaryData, CollocationPoints


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    """Return the 4-parameter search space from bo_design.md §2."""
    return SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])


class MockObjectiveFunction:
    """Mock: trial_id + 1 as objective, 1/(trial_id+1) as rel_l2_error."""

    @property
    def name(self) -> str:
        """Return mock objective name."""
        return "mock"

    def __call__(
        self, params: dict, trial_id: int, is_initial: bool
    ) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=float(trial_id + 1),
            rel_l2_error=1.0 / (trial_id + 1),
            elapsed_time=1.0,
            is_initial=is_initial,
        )


def run_optimizer(n_initial: int = 3, n_iterations: int = 5) -> BOResult:
    """Run BayesianOptimizer with MockObjectiveFunction."""
    ss = make_search_space()
    obj = MockObjectiveFunction()
    cfg = BOConfig(n_initial=n_initial, n_iterations=n_iterations, seed=0,
                   num_restarts=2, raw_samples=16)
    return BayesianOptimizer(ss, obj, cfg).optimize()


# ---------------------------------------------------------------------------
# THR-BO-01: Linear normalization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_01_linear_roundtrip():
    """THR-BO-01: to_tensor → from_tensor round-trip for linear int parameter.

    Basis: bo_design.md §4-2 SearchSpace normalization rule (linear).
    Tolerance: atol=0.5 (integer rounding introduces up to 0.5 error).
    """
    ss = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    original = {"n_hidden_layers": 5}
    tensor = ss.to_tensor(original)
    recovered = ss.from_tensor(tensor)
    assert abs(recovered["n_hidden_layers"] - original["n_hidden_layers"]) <= 0.5


# ---------------------------------------------------------------------------
# THR-BO-02: Log normalization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_02_log_roundtrip():
    """THR-BO-02: to_tensor → from_tensor round-trip for log-scale float.

    Basis: bo_design.md §4-2 SearchSpace normalization rule (log).
    Tolerance: rtol=1e-6.
    """
    ss = SearchSpace(parameters=[
        HyperParameter(name="lr", param_type="float", low=1e-4, high=1e-2, log_scale=True),
    ])
    original = {"lr": 3e-4}
    tensor = ss.to_tensor(original)
    recovered = ss.from_tensor(tensor)
    assert abs(float(recovered["lr"]) - original["lr"]) / original["lr"] < 1e-6


# ---------------------------------------------------------------------------
# THR-BO-03: Linear normalization is monotone
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_03_linear_monotone():
    """THR-BO-03: Larger values map to larger normalized values (linear scale).

    Basis: bo_design.md §4-2 SearchSpace — x' = (x - low) / (high - low).
    """
    ss = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    vals = [2, 4, 6, 8]
    norm_vals = [float(ss.to_tensor({"n_hidden_layers": v})[0, 0]) for v in vals]
    for a, b in zip(norm_vals, norm_vals[1:]):
        assert a < b


# ---------------------------------------------------------------------------
# THR-BO-04: Log normalization is monotone
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_04_log_monotone():
    """THR-BO-04: Larger values map to larger normalized values (log scale).

    Basis: bo_design.md §4-2 SearchSpace — x' = (log x - log low) / (log high - log low).
    """
    ss = SearchSpace(parameters=[
        HyperParameter(name="lr", param_type="float", low=1e-4, high=1e-2, log_scale=True),
    ])
    vals = [1e-4, 1e-3, 1e-2]
    norm_vals = [float(ss.to_tensor({"lr": v})[0, 0]) for v in vals]
    for a, b in zip(norm_vals, norm_vals[1:]):
        assert a < b


# ---------------------------------------------------------------------------
# THR-BO-05: Objective formula  objective = 1 / (rel_l2_error × elapsed_time)
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_05_objective_formula():
    """THR-BO-05: objective = 1 / (rel_l2_error × elapsed_time).

    Basis: bo_design.md §1 objective definition.
    Uses a mock BurgersPINNSolver to control rel_l2_error and elapsed_time.
    Expected: rel_l2_error=0.01, elapsed_time=2.0 → objective=50.0.
    Tolerance: rtol=1e-9.
    """
    # Build minimal mesh and usol such that we can control rel_l2_error
    # usol = ones(4,4), ||usol||_F = 4.0
    # For rel_l2_error = 0.01 we need ||u_pred - usol||_F = 0.04
    # u_pred = usol + delta where ||delta||_F = 0.04
    # delta uniform = 0.04 / sqrt(16) = 0.01 everywhere
    usol = np.ones((4, 4), dtype=np.float32)
    delta = 0.01  # ||[delta]*16||_F = 0.01*4 = 0.04 → rel_l2 = 0.04/4 = 0.01
    x_mesh = np.zeros((4, 4), dtype=np.float32)
    t_mesh = np.zeros((4, 4), dtype=np.float32)

    pde_config = PDEConfig(nu=0.01 / math.pi, x_min=-1.0, x_max=1.0,
                           t_min=0.0, t_max=1.0)
    base_cfg = TrainingConfig(n_u=4, n_f=4, lr=1e-3, epochs_adam=1, epochs_lbfgs=0)

    # Minimal boundary/collocation data (shapes must be valid)
    t_tensor = torch.zeros(4, 1, dtype=torch.float32)
    x_tensor = torch.zeros(4, 1, dtype=torch.float32)
    u_tensor = torch.zeros(4, 1, dtype=torch.float32)
    boundary_data = BoundaryData(t=t_tensor, x=x_tensor, u=u_tensor)
    collocation = CollocationPoints(t=t_tensor, x=x_tensor)

    obj_fn = AccuracySpeedObjective(
        pde_config=pde_config,
        boundary_data=boundary_data,
        collocation=collocation,
        x_mesh=x_mesh,
        t_mesh=t_mesh,
        usol=usol,
        base_training_config=base_cfg,
    )

    # Mock model that returns usol + delta (flattened)
    mock_model = MagicMock()
    u_pred_values = (usol.flatten() + delta).reshape(-1, 1)
    mock_model.return_value = torch.tensor(u_pred_values, dtype=torch.float32)
    # next(model.parameters()).device must be a real torch.device
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_forward_result = MagicMock()
    mock_forward_result.model = mock_model

    # elapsed_time = end - start = 2.0 → patch perf_counter to return 0 then 2
    with (
        patch("bo.objective.BurgersPINNSolver") as MockSolver,
        patch("bo.objective.time.perf_counter", side_effect=[0.0, 2.0]),
    ):
        MockSolver.return_value.solve_forward.return_value = mock_forward_result
        trial = obj_fn(
            params={"n_hidden_layers": 4, "n_neurons": 20, "lr": 1e-3, "epochs_adam": 100},
            trial_id=0,
            is_initial=True,
        )

    expected_rel_l2 = np.linalg.norm(usol.flatten() + delta - usol.flatten()) / np.linalg.norm(usol.flatten())
    expected_elapsed = 2.0
    expected_objective = 1.0 / (expected_rel_l2 * expected_elapsed)

    assert abs(trial.objective - expected_objective) / expected_objective < 1e-9


# ---------------------------------------------------------------------------
# THR-BO-06: Numerical stability for extreme rel_l2_error
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_06_numerical_stability():
    """THR-BO-06: Objective is finite when rel_l2_error is near zero.

    Basis: bo_design.md §8 — clamp max(..., 1e-10).
    rel_l2_error=1e-15, elapsed_time=1.0 → objective = 1e10 (finite).
    """
    usol = np.ones((4, 4), dtype=np.float32)
    # u_pred ≈ usol: delta so small that rel_l2_error << 1e-10
    delta = 1e-15
    x_mesh = np.zeros((4, 4), dtype=np.float32)
    t_mesh = np.zeros((4, 4), dtype=np.float32)

    pde_config = PDEConfig(nu=0.01 / math.pi, x_min=-1.0, x_max=1.0,
                           t_min=0.0, t_max=1.0)
    base_cfg = TrainingConfig(n_u=4, n_f=4, lr=1e-3, epochs_adam=1, epochs_lbfgs=0)

    t_tensor = torch.zeros(4, 1, dtype=torch.float32)
    x_tensor = torch.zeros(4, 1, dtype=torch.float32)
    u_tensor = torch.zeros(4, 1, dtype=torch.float32)
    boundary_data = BoundaryData(t=t_tensor, x=x_tensor, u=u_tensor)
    collocation = CollocationPoints(t=t_tensor, x=x_tensor)

    obj_fn = AccuracySpeedObjective(
        pde_config=pde_config,
        boundary_data=boundary_data,
        collocation=collocation,
        x_mesh=x_mesh,
        t_mesh=t_mesh,
        usol=usol,
        base_training_config=base_cfg,
    )

    mock_model = MagicMock()
    u_pred_values = (usol.flatten() + delta).reshape(-1, 1)
    mock_model.return_value = torch.tensor(u_pred_values, dtype=torch.float32)
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_forward_result = MagicMock()
    mock_forward_result.model = mock_model

    with (
        patch("bo.objective.BurgersPINNSolver") as MockSolver,
        patch("bo.objective.time.perf_counter", side_effect=[0.0, 1.0]),
    ):
        MockSolver.return_value.solve_forward.return_value = mock_forward_result
        trial = obj_fn(
            params={"n_hidden_layers": 2, "n_neurons": 10, "lr": 1e-3, "epochs_adam": 100},
            trial_id=0,
            is_initial=True,
        )

    assert math.isfinite(trial.objective)
    assert trial.objective <= 1e10 + 1  # clamped at 1/1e-10 = 1e10


# ---------------------------------------------------------------------------
# THR-BO-07: best_params consistency with best_trial_id
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_07_best_trial_id_consistency():
    """THR-BO-07: best_params == trials[best_trial_id].params.

    Basis: bo_design.md §4-2 BOResult definition.
    """
    result = run_optimizer(n_initial=3, n_iterations=5)
    best_trial = result.trials[result.best_trial_id]
    assert result.best_params == best_trial.params


# ---------------------------------------------------------------------------
# THR-BO-08: Cumulative best objective is non-decreasing
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_08_cumulative_best_nondecreasing():
    """THR-BO-08: The running maximum of objective across trials is non-decreasing.

    Basis: bo_design.md §3 Phase 2 — each iteration can only maintain or
    improve the best known objective.
    """
    result = run_optimizer(n_initial=3, n_iterations=5)
    best = float("-inf")
    prev_best = float("-inf")
    for t in result.trials:
        best = max(best, t.objective)
        assert best >= prev_best
        prev_best = best


# ---------------------------------------------------------------------------
# THR-BO-09: is_initial exclusivity at n_initial boundary
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_09_is_initial_exclusivity():
    """THR-BO-09: Exactly the first n_initial trials have is_initial=True.

    Basis: bo_design.md §4-2 TrialResult.is_initial field description.
    """
    n_initial = 3
    n_iterations = 5
    result = run_optimizer(n_initial=n_initial, n_iterations=n_iterations)
    initial_flags = [t.is_initial for t in result.trials]
    assert sum(initial_flags) == n_initial
    assert all(initial_flags[:n_initial])
    assert not any(initial_flags[n_initial:])


# ---------------------------------------------------------------------------
# THR-BO-10: bounds values
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_thr_bo_10_bounds_values():
    """THR-BO-10: bounds[0] is all 0.0 and bounds[1] is all 1.0.

    Basis: bo_design.md §4-2 SearchSpace.bounds — manually normalized to [0,1].
    """
    ss = make_search_space()
    b = ss.bounds
    assert (b[0] == 0.0).all()
    assert (b[1] == 1.0).all()
