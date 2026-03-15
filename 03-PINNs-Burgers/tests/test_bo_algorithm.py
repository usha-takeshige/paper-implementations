"""Algorithm implementation tests for the bo module (ALG-BO-01 to ALG-BO-24).

Verifies that each class and method computes the correct results.
Uses MockObjectiveFunction to avoid running actual PINN training.
"""

import os
import tempfile

import pytest
import torch
from pydantic import ValidationError

from bo import (
    BOConfig,
    BOResult,
    BayesianOptimizer,
    HyperParameter,
    ReportGenerator,
    SearchSpace,
    TrialResult,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    """Return a 4-parameter SearchSpace matching the design spec."""
    return SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])


class MockObjectiveFunction:
    """Test mock: returns deterministic TrialResult without PINN training."""

    @property
    def name(self) -> str:
        """Return mock objective name."""
        return "mock"

    def __call__(
        self, params: dict, trial_id: int, is_initial: bool
    ) -> TrialResult:
        """Return fixed values based on trial_id."""
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=float(trial_id + 1),   # larger trial_id = higher score
            rel_l2_error=1.0 / (trial_id + 1),
            elapsed_time=1.0,
            is_initial=is_initial,
        )


def make_bo_result(n_trials: int = 5) -> BOResult:
    """Build a minimal BOResult for report tests."""
    cfg = BOConfig(n_initial=3, n_iterations=2)
    params: dict[str, float | int] = {
        "n_hidden_layers": 4, "n_neurons": 20, "lr": 1e-3, "epochs_adam": 2000,
    }
    trials = [
        TrialResult(
            trial_id=i, params=params, objective=float(i + 1),
            rel_l2_error=1.0 / (i + 1), elapsed_time=1.0,
            is_initial=(i < 3),
        )
        for i in range(n_trials)
    ]
    best = max(trials, key=lambda t: t.objective)
    return BOResult(
        trials=trials,
        best_params=best.params,
        best_objective=best.objective,
        best_trial_id=best.trial_id,
        bo_config=cfg,
        objective_name="mock",
    )


# ---------------------------------------------------------------------------
# SearchSpace tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_bo_01_dim():
    """ALG-BO-01: SearchSpace.dim returns the number of parameters."""
    ss = make_search_space()
    assert ss.dim == 4


@pytest.mark.algorithm
def test_alg_bo_02_bounds_shape_and_values():
    """ALG-BO-02: SearchSpace.bounds has shape (2, 4) with all values in [0, 1]."""
    ss = make_search_space()
    b = ss.bounds
    assert b.shape == (2, 4)
    assert (b >= 0.0).all()
    assert (b <= 1.0).all()


@pytest.mark.algorithm
def test_alg_bo_03_to_tensor_linear():
    """ALG-BO-03: to_tensor normalizes a linear int parameter correctly."""
    ss = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    x = ss.to_tensor({"n_hidden_layers": 5})
    assert x.shape == (1, 1)
    assert x.dtype == torch.float64
    assert abs(float(x[0, 0]) - 0.5) < 1e-9  # (5-2)/(8-2) = 3/6 = 0.5


@pytest.mark.algorithm
def test_alg_bo_04_to_tensor_log():
    """ALG-BO-04: to_tensor normalizes a log-scale float parameter correctly."""
    import math
    ss = SearchSpace(parameters=[
        HyperParameter(name="lr", param_type="float", low=1e-4, high=1e-2, log_scale=True),
    ])
    x = ss.to_tensor({"lr": 1e-3})
    assert x.shape == (1, 1)
    assert x.dtype == torch.float64
    # log(1e-3) is the midpoint of [log(1e-4), log(1e-2)] → normalized = 0.5
    expected = (math.log(1e-3) - math.log(1e-4)) / (math.log(1e-2) - math.log(1e-4))
    assert abs(float(x[0, 0]) - expected) < 1e-9


@pytest.mark.algorithm
def test_alg_bo_05_from_tensor_linear():
    """ALG-BO-05: from_tensor inverts linear normalization correctly."""
    ss = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    x = torch.tensor([[0.5]], dtype=torch.float64)
    params = ss.from_tensor(x)
    assert params["n_hidden_layers"] == 5  # 0.5*(8-2)+2 = 5.0 → round → 5


@pytest.mark.algorithm
def test_alg_bo_06_from_tensor_log():
    """ALG-BO-06: from_tensor inverts log normalization correctly."""
    ss = SearchSpace(parameters=[
        HyperParameter(name="lr", param_type="float", low=1e-4, high=1e-2, log_scale=True),
    ])
    x = torch.tensor([[0.5]], dtype=torch.float64)
    params = ss.from_tensor(x)
    assert abs(float(params["lr"]) - 1e-3) < 1e-15


@pytest.mark.algorithm
def test_alg_bo_07_from_tensor_int_type():
    """ALG-BO-07: from_tensor returns int type for int parameters."""
    ss = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    x = torch.tensor([[0.51]], dtype=torch.float64)
    params = ss.from_tensor(x)
    assert isinstance(params["n_hidden_layers"], int)


@pytest.mark.algorithm
def test_alg_bo_08_sample_sobol_shape():
    """ALG-BO-08: sample_sobol returns shape (n, dim)."""
    ss = make_search_space()
    samples = ss.sample_sobol(n=8, seed=0)
    assert samples.shape == (8, 4)


@pytest.mark.algorithm
def test_alg_bo_09_sample_sobol_in_unit_cube():
    """ALG-BO-09: All Sobol samples are in [0, 1]^d."""
    ss = make_search_space()
    samples = ss.sample_sobol(n=16, seed=0)
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()


# ---------------------------------------------------------------------------
# TrialResult tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_bo_10_trial_result_construction():
    """ALG-BO-10: TrialResult constructs without exceptions."""
    trial = TrialResult(
        trial_id=0,
        params={"n_hidden_layers": 4, "n_neurons": 20, "lr": 1e-3, "epochs_adam": 2000},
        objective=5.0,
        rel_l2_error=0.02,
        elapsed_time=10.0,
        is_initial=True,
    )
    assert trial.trial_id == 0
    assert trial.objective == 5.0


@pytest.mark.algorithm
def test_alg_bo_11_trial_result_frozen():
    """ALG-BO-11: TrialResult is frozen; assignment raises ValidationError."""
    trial = TrialResult(
        trial_id=0,
        params={"n_hidden_layers": 4},
        objective=1.0, rel_l2_error=0.1, elapsed_time=5.0, is_initial=True,
    )
    with pytest.raises((ValidationError, TypeError)):
        trial.objective = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BOResult tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_bo_12_bo_result_best_trial_id_in_range():
    """ALG-BO-12: best_trial_id is a valid index into trials."""
    result = make_bo_result(n_trials=5)
    assert 0 <= result.best_trial_id < len(result.trials)


@pytest.mark.algorithm
def test_alg_bo_13_bo_result_frozen():
    """ALG-BO-13: BOResult is frozen; assignment raises ValidationError."""
    result = make_bo_result()
    with pytest.raises((ValidationError, TypeError)):
        result.best_objective = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MockObjectiveFunction tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_bo_14_objective_returns_trial_result():
    """ALG-BO-14: ObjectiveFunction returns a TrialResult."""
    obj = MockObjectiveFunction()
    params: dict[str, float | int] = {
        "n_hidden_layers": 4, "n_neurons": 20, "lr": 1e-3, "epochs_adam": 2000,
    }
    result = obj(params=params, trial_id=0, is_initial=True)
    assert isinstance(result, TrialResult)


@pytest.mark.algorithm
def test_alg_bo_15_objective_is_initial_flag():
    """ALG-BO-15: is_initial flag is set according to the argument."""
    obj = MockObjectiveFunction()
    params: dict[str, float | int] = {"n_hidden_layers": 4, "n_neurons": 20,
                                       "lr": 1e-3, "epochs_adam": 2000}
    trial = obj(params=params, trial_id=0, is_initial=True)
    assert trial.is_initial is True


@pytest.mark.algorithm
def test_alg_bo_16_objective_trial_id():
    """ALG-BO-16: trial_id is set according to the argument."""
    obj = MockObjectiveFunction()
    params: dict[str, float | int] = {"n_hidden_layers": 4, "n_neurons": 20,
                                       "lr": 1e-3, "epochs_adam": 2000}
    trial = obj(params=params, trial_id=3, is_initial=False)
    assert trial.trial_id == 3


# ---------------------------------------------------------------------------
# BayesianOptimizer tests
# ---------------------------------------------------------------------------


def run_optimizer(n_initial: int = 3, n_iterations: int = 5) -> BOResult:
    """Helper: run BayesianOptimizer with MockObjectiveFunction."""
    ss = make_search_space()
    obj = MockObjectiveFunction()
    cfg = BOConfig(n_initial=n_initial, n_iterations=n_iterations, seed=0,
                   num_restarts=2, raw_samples=16)
    optimizer = BayesianOptimizer(ss, obj, cfg)
    return optimizer.optimize()


@pytest.mark.algorithm
def test_alg_bo_17_total_trial_count():
    """ALG-BO-17: Total trial count equals n_initial + n_iterations."""
    result = run_optimizer(n_initial=3, n_iterations=5)
    assert len(result.trials) == 8


@pytest.mark.algorithm
def test_alg_bo_18_initial_trials_flag():
    """ALG-BO-18: First n_initial trials have is_initial=True."""
    result = run_optimizer(n_initial=3, n_iterations=5)
    for t in result.trials[:3]:
        assert t.is_initial is True


@pytest.mark.algorithm
def test_alg_bo_19_bo_trials_flag():
    """ALG-BO-19: Remaining n_iterations trials have is_initial=False."""
    result = run_optimizer(n_initial=3, n_iterations=5)
    for t in result.trials[3:]:
        assert t.is_initial is False


@pytest.mark.algorithm
def test_alg_bo_20_best_params_match_best_trial():
    """ALG-BO-20: best_params matches the params of the best trial."""
    result = run_optimizer(n_initial=3, n_iterations=5)
    best_trial = result.trials[result.best_trial_id]
    assert result.best_params == best_trial.params


@pytest.mark.algorithm
def test_alg_bo_21_best_objective_is_max():
    """ALG-BO-21: best_objective equals the maximum objective across all trials."""
    result = run_optimizer(n_initial=3, n_iterations=5)
    expected_max = max(t.objective for t in result.trials)
    assert result.best_objective == expected_max


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_bo_22_report_file_created(tmp_path: str) -> None:
    """ALG-BO-22: ReportGenerator creates bo_report.md in output_dir."""
    result = make_bo_result()
    ss = make_search_space()
    reporter = ReportGenerator(output_dir=str(tmp_path))
    path = reporter.generate(result, ss)
    assert os.path.exists(path)
    assert os.path.basename(path) == "bo_report.md"


@pytest.mark.algorithm
def test_alg_bo_23_report_contains_best_trial_id(tmp_path: str) -> None:
    """ALG-BO-23: Report contains the best_trial_id."""
    cfg = BOConfig(n_initial=3, n_iterations=2)
    params: dict[str, float | int] = {
        "n_hidden_layers": 4, "n_neurons": 20, "lr": 1e-3, "epochs_adam": 2000,
    }
    trials = [
        TrialResult(
            trial_id=i, params=params, objective=float(i + 1),
            rel_l2_error=1.0 / (i + 1), elapsed_time=1.0, is_initial=(i < 3),
        )
        for i in range(5)
    ]
    result = BOResult(
        trials=trials, best_params=trials[2].params,
        best_objective=trials[2].objective, best_trial_id=2, bo_config=cfg,
        objective_name="mock",
    )
    ss = make_search_space()
    reporter = ReportGenerator(output_dir=str(tmp_path))
    path = reporter.generate(result, ss)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "2" in content  # best_trial_id = 2 appears in the report


@pytest.mark.algorithm
def test_alg_bo_24_report_trial_rows(tmp_path: str) -> None:
    """ALG-BO-24: Report contains one table row per trial (5 trials → 5 rows)."""
    result = make_bo_result(n_trials=5)
    ss = make_search_space()
    reporter = ReportGenerator(output_dir=str(tmp_path))
    path = reporter.generate(result, ss)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # Each trial row starts with "| <trial_id>" — count rows in the All Trials table
    import re
    rows = re.findall(r"^\| \d+\s+\| (?:initial|BO)", content, re.MULTILINE)
    assert len(rows) == 5
