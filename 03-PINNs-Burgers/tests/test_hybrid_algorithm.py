"""Algorithm implementation tests for the hybrid module (ALG-HYB-01 to ALG-HYB-13).

Verifies that HybridOptimizer and _narrow_search_space compute correct results.
Uses MockChain and MockObjectiveFunction to avoid Gemini API calls and PINN training.
"""

import dataclasses
import math

import pytest

from bo.result import BOConfig
from hybrid import HybridOptimizer, HybridResult
from opt_agent.chain import BaseChain
from opt_agent.config import LLMConfig
from opt_agent.proposal import LLMProposal
from opt_tool.result import TrialResult
from opt_tool.space import HyperParameter, SearchSpace


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    """Return the 4-parameter SearchSpace for PINN hyperparameter optimization."""
    return SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])


class MockObjectiveFunction:
    """Deterministic mock objective; returns TrialResult without PINN training."""

    @property
    def name(self) -> str:
        """Return mock objective name."""
        return "mock"

    def __call__(
        self,
        params: dict,
        trial_id: int,
        is_initial: bool,
    ) -> TrialResult:
        """Return a TrialResult with objective = trial_id + 1."""
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=float(trial_id + 1),
            rel_l2_error=1.0 / (trial_id + 1),
            elapsed_time=1.0,
            is_initial=is_initial,
        )


class MockChain(BaseChain):
    """Test mock chain that counts invocations and returns fixed params."""

    def __init__(self, fixed_params: dict[str, float | int]) -> None:
        """Initialize with fixed params and zero call count.

        Parameters
        ----------
        fixed_params:
            Parameters to return on every invoke call.
        """
        self.call_count = 0
        self._fixed_params = fixed_params

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Increment call counter and return fixed LLMProposal."""
        self.call_count += 1
        return LLMProposal(
            analysis_report="mock analysis",
            proposed_params=self._fixed_params,
            reasoning="mock reasoning",
        )


def make_fixed_params() -> dict[str, float | int]:
    """Return midpoint params within the standard search space bounds."""
    return {
        "n_hidden_layers": 5,
        "n_neurons": 50,
        "lr": 1e-3,
        "epochs_adam": 2_000,
    }


def make_optimizer(
    chain: BaseChain | None = None,
    n_llm_initial: int = 3,
    n_llm_iterations: int = 3,
    n_bo_initial: int = 3,
    n_bo_iterations: int = 5,
    top_k_ratio: float = 0.3,
    margin_ratio: float = 0.1,
) -> HybridOptimizer:
    """Build a HybridOptimizer with small configs for testing."""
    if chain is None:
        chain = MockChain(make_fixed_params())
    llm_config = LLMConfig(n_initial=n_llm_initial, n_iterations=10, seed=42)
    bo_config = BOConfig(
        n_initial=n_bo_initial, n_iterations=n_bo_iterations,
        acquisition="EI", seed=42, num_restarts=2, raw_samples=16,
    )
    return HybridOptimizer(
        search_space=make_search_space(),
        objective=MockObjectiveFunction(),
        llm_config=llm_config,
        bo_config=bo_config,
        chain=chain,
        n_llm_iterations=n_llm_iterations,
        top_k_ratio=top_k_ratio,
        margin_ratio=margin_ratio,
    )


# ---------------------------------------------------------------------------
# A-1: Algorithm implementation tests (ALG-HYB-01 to ALG-HYB-07)
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_hyb_01_llm_chain_called_n_iterations_times() -> None:
    """ALG-HYB-01: MockChain.invoke is called exactly n_llm_iterations times."""
    chain = MockChain(make_fixed_params())
    n_llm_iterations = 3
    make_optimizer(chain=chain, n_llm_iterations=n_llm_iterations).optimize()
    assert chain.call_count == n_llm_iterations


@pytest.mark.algorithm
def test_alg_hyb_02_llm_trials_count() -> None:
    """ALG-HYB-02: len(llm_trials) == llm_config.n_initial + n_llm_iterations."""
    n_initial = 3
    n_llm_iterations = 4
    result = make_optimizer(
        n_llm_initial=n_initial, n_llm_iterations=n_llm_iterations
    ).optimize()
    assert len(result.llm_trials) == n_initial + n_llm_iterations


@pytest.mark.algorithm
def test_alg_hyb_03_bo_trials_count() -> None:
    """ALG-HYB-03: len(bo_trials) == bo_config.n_initial + bo_config.n_iterations."""
    n_bo_initial = 3
    n_bo_iterations = 5
    result = make_optimizer(
        n_bo_initial=n_bo_initial, n_bo_iterations=n_bo_iterations
    ).optimize()
    assert len(result.bo_trials) == n_bo_initial + n_bo_iterations


@pytest.mark.algorithm
def test_alg_hyb_04_best_objective_is_max_across_all_trials() -> None:
    """ALG-HYB-04: best_objective equals the maximum across all Phase 1 + 2 trials."""
    result = make_optimizer().optimize()
    combined = result.llm_trials + result.bo_trials
    expected_max = max(t.objective for t in combined)
    assert result.best_objective == expected_max


@pytest.mark.algorithm
def test_alg_hyb_05_best_params_match_best_trial() -> None:
    """ALG-HYB-05: best_params matches the params of the trial at best_trial_id."""
    result = make_optimizer().optimize()
    combined = result.llm_trials + result.bo_trials
    assert result.best_params == combined[result.best_trial_id].params


@pytest.mark.algorithm
def test_alg_hyb_06_hybrid_result_is_frozen() -> None:
    """ALG-HYB-06: HybridResult is frozen; assignment raises FrozenInstanceError."""
    result = make_optimizer().optimize()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        result.best_objective = 0.0  # type: ignore[misc]


@pytest.mark.algorithm
def test_alg_hyb_07_hybrid_result_constructs_successfully() -> None:
    """ALG-HYB-07: HybridResult constructs without raising exceptions."""
    result = make_optimizer().optimize()
    assert isinstance(result, HybridResult)


# ---------------------------------------------------------------------------
# A-2: Pipeline guarantee tests (ALG-HYB-08 to ALG-HYB-13)
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_alg_hyb_08_narrowed_bounds_contained_in_original() -> None:
    """ALG-HYB-08: All narrowed parameter bounds are subsets of original bounds."""
    result = make_optimizer().optimize()
    original = make_search_space()
    for orig_hp, narr_hp in zip(original.parameters, result.narrowed_space.parameters):
        assert narr_hp.low >= orig_hp.low, (
            f"{orig_hp.name}: narrowed low {narr_hp.low} < original low {orig_hp.low}"
        )
        assert narr_hp.high <= orig_hp.high, (
            f"{orig_hp.name}: narrowed high {narr_hp.high} > original high {orig_hp.high}"
        )


@pytest.mark.algorithm
def test_alg_hyb_09_bo_trials_within_narrowed_space() -> None:
    """ALG-HYB-09: All Phase 2 trial params are within the narrowed_space bounds.

    For integer parameters, the effective bounds allow rounding to the nearest
    integer: values up to round(hp.high) and down to round(hp.low) are valid.
    """
    result = make_optimizer().optimize()
    for trial in result.bo_trials:
        for hp in result.narrowed_space.parameters:
            val = float(trial.params[hp.name])
            if hp.param_type == "int":
                # Integer params are decoded by rounding, so the effective
                # bounds are round(hp.low) to round(hp.high).
                effective_low = float(round(hp.low))
                effective_high = float(round(hp.high))
            else:
                effective_low = hp.low
                effective_high = hp.high
            assert val >= effective_low - 1e-9, (
                f"{hp.name}: value {val} < effective low {effective_low}"
            )
            assert val <= effective_high + 1e-9, (
                f"{hp.name}: value {val} > effective high {effective_high}"
            )


@pytest.mark.algorithm
def test_alg_hyb_10_linear_scale_narrowing_math() -> None:
    """ALG-HYB-10: Linear scale narrowing applies margin correctly (atol=1e-9)."""
    # Single linear parameter space
    space = SearchSpace(parameters=[
        HyperParameter(name="epochs_adam", param_type="int", low=500, high=5_000),
    ])
    hp = space.parameters[0]
    val = 2_000.0
    margin_ratio = 0.1
    linear_range = hp.high - hp.low

    # Build a single trial with known value
    trial = TrialResult(
        trial_id=0, params={"epochs_adam": int(val)},
        objective=1.0, rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
    )

    class SingleParamObjective:
        @property
        def name(self) -> str:
            return "mock"

        def __call__(self, params, trial_id, is_initial):
            return TrialResult(
                trial_id=trial_id, params=params,
                objective=float(trial_id + 1),
                rel_l2_error=0.1, elapsed_time=1.0, is_initial=is_initial,
            )

    optimizer = HybridOptimizer(
        search_space=space,
        objective=SingleParamObjective(),  # type: ignore[arg-type]
        llm_config=LLMConfig(n_initial=1, n_iterations=10, seed=0),
        bo_config=BOConfig(n_initial=1, n_iterations=1, seed=0, num_restarts=2, raw_samples=8),
        chain=MockChain({"epochs_adam": int(val)}),
        n_llm_iterations=1,
        top_k_ratio=1.0,  # use all trials
        margin_ratio=margin_ratio,
    )
    # Direct test of _narrow_search_space with a known trial
    narrowed = optimizer._narrow_search_space([trial])
    narr_hp = narrowed.parameters[0]

    expected_low = max(hp.low, val - margin_ratio * linear_range)
    expected_high = min(hp.high, val + margin_ratio * linear_range)

    assert abs(narr_hp.low - expected_low) < 1e-9
    assert abs(narr_hp.high - expected_high) < 1e-9


@pytest.mark.algorithm
def test_alg_hyb_11_log_scale_narrowing_math() -> None:
    """ALG-HYB-11: Log scale narrowing applies margin in log space (rtol=1e-9)."""
    space = SearchSpace(parameters=[
        HyperParameter(name="lr", param_type="float", low=1e-4, high=1e-2, log_scale=True),
    ])
    hp = space.parameters[0]
    val = 1e-3
    margin_ratio = 0.1
    log_range = math.log(hp.high) - math.log(hp.low)

    trial = TrialResult(
        trial_id=0, params={"lr": val},
        objective=1.0, rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
    )

    class SingleParamObjective:
        @property
        def name(self) -> str:
            return "mock"

        def __call__(self, params, trial_id, is_initial):
            return TrialResult(
                trial_id=trial_id, params=params,
                objective=float(trial_id + 1),
                rel_l2_error=0.1, elapsed_time=1.0, is_initial=is_initial,
            )

    optimizer = HybridOptimizer(
        search_space=space,
        objective=SingleParamObjective(),  # type: ignore[arg-type]
        llm_config=LLMConfig(n_initial=1, n_iterations=10, seed=0),
        bo_config=BOConfig(n_initial=1, n_iterations=1, seed=0, num_restarts=2, raw_samples=8),
        chain=MockChain({"lr": val}),
        n_llm_iterations=1,
        top_k_ratio=1.0,
        margin_ratio=margin_ratio,
    )
    narrowed = optimizer._narrow_search_space([trial])
    narr_hp = narrowed.parameters[0]

    new_log_low = math.log(val) - margin_ratio * log_range
    new_log_high = math.log(val) + margin_ratio * log_range
    expected_low = math.exp(max(new_log_low, math.log(hp.low)))
    expected_high = math.exp(min(new_log_high, math.log(hp.high)))

    assert abs(narr_hp.low - expected_low) / expected_low < 1e-9
    assert abs(narr_hp.high - expected_high) / expected_high < 1e-9


@pytest.mark.algorithm
def test_alg_hyb_12_fallback_to_original_bounds_when_collapsed() -> None:
    """ALG-HYB-12: Falls back to original bounds when new_low >= new_high."""
    space = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    hp = space.parameters[0]

    # All trials at the same value with margin=0 → would collapse
    # Use margin_ratio=0 and top_k_ratio=1.0 to force collapse
    val = 5
    trials = [
        TrialResult(
            trial_id=i, params={"n_hidden_layers": val},
            objective=float(i + 1), rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
        )
        for i in range(3)
    ]

    class SingleParamObjective:
        @property
        def name(self) -> str:
            return "mock"

        def __call__(self, params, trial_id, is_initial):
            return TrialResult(
                trial_id=trial_id, params=params,
                objective=float(trial_id + 1),
                rel_l2_error=0.1, elapsed_time=1.0, is_initial=is_initial,
            )

    optimizer = HybridOptimizer(
        search_space=space,
        objective=SingleParamObjective(),  # type: ignore[arg-type]
        llm_config=LLMConfig(n_initial=1, n_iterations=10, seed=0),
        bo_config=BOConfig(n_initial=1, n_iterations=1, seed=0, num_restarts=2, raw_samples=8),
        chain=MockChain({"n_hidden_layers": val}),
        n_llm_iterations=1,
        top_k_ratio=1.0,
        margin_ratio=0.0,  # no margin → new_low == new_high == val → fallback
    )
    narrowed = optimizer._narrow_search_space(trials)
    narr_hp = narrowed.parameters[0]

    assert narr_hp.low == hp.low
    assert narr_hp.high == hp.high


@pytest.mark.algorithm
def test_alg_hyb_13_top_k_ratio_selects_correct_trials() -> None:
    """ALG-HYB-13: Only top-k trials are used for narrowing (verified via param range)."""
    # 3 trials with objective 1, 2, 3 and distinct param values
    # With top_k_ratio=0.4 → top 2 trials (objective=2,3)
    space = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int", low=2, high=8),
    ])
    hp = space.parameters[0]

    trials = [
        TrialResult(
            trial_id=0, params={"n_hidden_layers": 2},
            objective=1.0, rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
        ),
        TrialResult(
            trial_id=1, params={"n_hidden_layers": 5},
            objective=2.0, rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
        ),
        TrialResult(
            trial_id=2, params={"n_hidden_layers": 7},
            objective=3.0, rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
        ),
    ]

    class SingleParamObjective:
        @property
        def name(self) -> str:
            return "mock"

        def __call__(self, params, trial_id, is_initial):
            return TrialResult(
                trial_id=trial_id, params=params,
                objective=float(trial_id + 1),
                rel_l2_error=0.1, elapsed_time=1.0, is_initial=is_initial,
            )

    optimizer = HybridOptimizer(
        search_space=space,
        objective=SingleParamObjective(),  # type: ignore[arg-type]
        llm_config=LLMConfig(n_initial=1, n_iterations=10, seed=0),
        bo_config=BOConfig(n_initial=1, n_iterations=1, seed=0, num_restarts=2, raw_samples=8),
        chain=MockChain({"n_hidden_layers": 5}),
        n_llm_iterations=1,
        top_k_ratio=0.4,   # ceil(3 * 0.4) = 2 → top 2 trials (objective=2,3)
        margin_ratio=0.0,  # no margin to clearly identify which trials were used
    )
    narrowed = optimizer._narrow_search_space(trials)
    narr_hp = narrowed.parameters[0]

    # Top 2 trials have values 5 and 7 (objective=2 and 3)
    # With margin_ratio=0: new_low=max(hp.low, 5)=5, new_high=min(hp.high, 7)=7
    assert narr_hp.low == 5.0
    assert narr_hp.high == 7.0
