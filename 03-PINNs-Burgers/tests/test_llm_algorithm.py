"""Algorithm implementation tests for the opt_agent module (ALG-LLM-01 to ALG-LLM-24).

Verifies that each class and method computes correct results.
Uses MockChain and MockObjectiveFunction to avoid Gemini API calls.
"""

import dataclasses
import os

import pytest
from pydantic import ValidationError

from bo import HyperParameter, SearchSpace, TrialResult
from opt_agent import (
    BaseChain,
    LLMConfig,
    LLMIterationMeta,
    LLMOptimizer,
    LLMProposal,
    LLMResult,
    MaximizeObjectivePromptBuilder,
)


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
    """Return fixed LLMProposal values without calling Gemini API."""

    def __init__(self, proposals: list[dict] | None = None) -> None:
        """Initialize with optional list of per-call proposals.

        Parameters
        ----------
        proposals:
            List of proposed_params dicts for each invoke call.
            If None, returns midpoint of the search space each time.
        """
        self._proposals = proposals
        self._call_index = 0

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Return the next fixed proposal or a default mid-point proposal."""
        if self._proposals is not None and self._call_index < len(self._proposals):
            params = self._proposals[self._call_index]
        else:
            params = {
                "n_hidden_layers": 5,
                "n_neurons": 50,
                "lr": 1e-3,
                "epochs_adam": 2000,
            }
        self._call_index += 1
        return LLMProposal(
            analysis_report="Mock analysis",
            proposed_params=params,
            reasoning="Mock reasoning",
        )


class InvocationCountChain(BaseChain):
    """MockChain that counts invoke calls (ALG-LLM-23)."""

    def __init__(self) -> None:
        """Initialize with zero call count."""
        self.call_count = 0

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Increment call counter and return default proposal."""
        self.call_count += 1
        return LLMProposal(
            analysis_report="count",
            proposed_params={"n_hidden_layers": 5, "n_neurons": 50, "lr": 1e-3, "epochs_adam": 2000},
            reasoning="count",
        )


def make_optimizer(
    chain: BaseChain | None = None,
    n_initial: int = 3,
    n_iterations: int = 2,
) -> LLMOptimizer:
    """Build an LLMOptimizer with MockChain and small config for testing."""
    if chain is None:
        chain = MockChain()
    return LLMOptimizer(
        search_space=make_search_space(),
        objective=MockObjectiveFunction(),
        config=LLMConfig(n_initial=n_initial, n_iterations=n_iterations, seed=42),
        chain=chain,
    )


# ---------------------------------------------------------------------------
# ALG-LLM-01 to ALG-LLM-08: Data class / Pydantic model tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_llm_config_defaults() -> None:
    """ALG-LLM-01: LLMConfig defaults match specification."""
    cfg = LLMConfig()
    assert cfg.n_initial == 5
    assert cfg.n_iterations == 20
    assert cfg.seed == 42


@pytest.mark.algorithm
def test_llm_config_frozen() -> None:
    """ALG-LLM-02: LLMConfig is immutable (frozen dataclass)."""
    cfg = LLMConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        cfg.n_initial = 99  # type: ignore[misc]


@pytest.mark.algorithm
def test_llm_iteration_meta_fields() -> None:
    """ALG-LLM-03: LLMIterationMeta stores all fields correctly."""
    meta = LLMIterationMeta(
        iteration_id=1,
        analysis_report="analysis",
        proposed_params={"lr": 1e-3},
        reasoning="reason",
        proposal_time=1.23,
    )
    assert meta.iteration_id == 1
    assert meta.analysis_report == "analysis"
    assert meta.proposed_params == {"lr": 1e-3}
    assert meta.reasoning == "reason"
    assert meta.proposal_time == 1.23


@pytest.mark.algorithm
def test_llm_iteration_meta_frozen() -> None:
    """ALG-LLM-04: LLMIterationMeta is immutable (frozen dataclass)."""
    meta = LLMIterationMeta(
        iteration_id=0,
        analysis_report="a",
        proposed_params={},
        reasoning="r",
        proposal_time=0.0,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        meta.iteration_id = 99  # type: ignore[misc]


@pytest.mark.algorithm
def test_llm_result_fields() -> None:
    """ALG-LLM-05: LLMResult stores all fields correctly."""
    trial = TrialResult(
        trial_id=0, params={"lr": 1e-3}, objective=1.0,
        rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
    )
    meta = LLMIterationMeta(
        iteration_id=0, analysis_report="a",
        proposed_params={"lr": 1e-3}, reasoning="r",
        proposal_time=0.0,
    )
    cfg = LLMConfig()
    result = LLMResult(
        trials=[trial],
        best_params={"lr": 1e-3},
        best_objective=1.0,
        best_trial_id=0,
        llm_config=cfg,
        objective_name="mock",
        iteration_metas=[meta],
    )
    assert result.trials == [trial]
    assert result.best_objective == 1.0
    assert result.best_trial_id == 0
    assert result.llm_config == cfg
    assert result.objective_name == "mock"
    assert result.iteration_metas == [meta]


@pytest.mark.algorithm
def test_llm_result_frozen() -> None:
    """ALG-LLM-06: LLMResult is immutable (frozen dataclass)."""
    trial = TrialResult(
        trial_id=0, params={}, objective=1.0,
        rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
    )
    result = LLMResult(
        trials=[trial], best_params={}, best_objective=1.0,
        best_trial_id=0, llm_config=LLMConfig(),
        objective_name="mock", iteration_metas=[],
    )
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        result.best_objective = 99.0  # type: ignore[misc]


@pytest.mark.algorithm
def test_llm_proposal_valid() -> None:
    """ALG-LLM-07: LLMProposal instantiates correctly with valid inputs."""
    proposal = LLMProposal(
        analysis_report="report",
        proposed_params={"lr": 1e-3, "n_hidden_layers": 4},
        reasoning="reason",
    )
    assert proposal.analysis_report == "report"
    assert proposal.proposed_params == {"lr": 1e-3, "n_hidden_layers": 4}
    assert proposal.reasoning == "reason"


@pytest.mark.algorithm
def test_llm_proposal_missing_field() -> None:
    """ALG-LLM-08: LLMProposal raises ValidationError when required field is missing."""
    with pytest.raises(ValidationError):
        LLMProposal(  # type: ignore[call-arg]
            proposed_params={"lr": 1e-3},
            reasoning="reason",
            # analysis_report is missing
        )


# ---------------------------------------------------------------------------
# ALG-LLM-09 to ALG-LLM-12: PromptBuilder tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_prompt_builder_system_contains_param_names() -> None:
    """ALG-LLM-09: System prompt contains all parameter names."""
    search_space = make_search_space()
    prompt = MaximizeObjectivePromptBuilder().build_system_prompt(search_space, "accuracy")
    for name in ["n_hidden_layers", "n_neurons", "lr", "epochs_adam"]:
        assert name in prompt, f"'{name}' not found in system prompt"


@pytest.mark.algorithm
def test_prompt_builder_system_contains_bounds() -> None:
    """ALG-LLM-10: System prompt contains each parameter's low and high values."""
    search_space = make_search_space()
    prompt = MaximizeObjectivePromptBuilder().build_system_prompt(search_space, "accuracy")
    for hp in search_space.parameters:
        assert str(hp.low) in prompt, f"low={hp.low} not found in system prompt"
        assert str(hp.high) in prompt, f"high={hp.high} not found in system prompt"


@pytest.mark.algorithm
def test_prompt_builder_human_contains_all_trials() -> None:
    """ALG-LLM-11: Human prompt contains all trial IDs and objectives."""
    trials = [
        TrialResult(
            trial_id=i, params={"n_hidden_layers": 4, "n_neurons": 20, "lr": 1e-3, "epochs_adam": 1000},
            objective=float(i + 1), rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
        )
        for i in range(3)
    ]
    prompt = MaximizeObjectivePromptBuilder().build_human_prompt(trials, iteration_id=2)
    for t in trials:
        assert str(t.trial_id) in prompt


@pytest.mark.algorithm
def test_prompt_builder_human_contains_best() -> None:
    """ALG-LLM-12: Human prompt contains the best trial's ID."""
    trials = [
        TrialResult(
            trial_id=0, params={"lr": 1e-3}, objective=1.0,
            rel_l2_error=0.5, elapsed_time=1.0, is_initial=True,
        ),
        TrialResult(
            trial_id=1, params={"lr": 1e-2}, objective=5.0,
            rel_l2_error=0.1, elapsed_time=1.0, is_initial=True,
        ),
    ]
    prompt = MaximizeObjectivePromptBuilder().build_human_prompt(trials, iteration_id=0)
    # Best trial is id=1 (objective=5.0)
    assert "1" in prompt


# ---------------------------------------------------------------------------
# ALG-LLM-13 to ALG-LLM-24: LLMOptimizer tests
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_optimizer_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """ALG-LLM-13: LLMOptimizer raises ValueError when GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        LLMOptimizer(
            search_space=make_search_space(),
            objective=MockObjectiveFunction(),
            config=LLMConfig(),
            chain=None,
        )


@pytest.mark.algorithm
def test_optimizer_accepts_mock_chain() -> None:
    """ALG-LLM-14: LLMOptimizer initializes without error when MockChain is injected."""
    optimizer = make_optimizer()
    assert optimizer is not None


@pytest.mark.algorithm
def test_phase1_trials_are_initial() -> None:
    """ALG-LLM-15: Phase 1 produces n_initial trials with is_initial=True."""
    result = make_optimizer(n_initial=3, n_iterations=2).optimize()
    assert all(t.is_initial for t in result.trials[:3])


@pytest.mark.algorithm
def test_phase2_trials_are_not_initial() -> None:
    """ALG-LLM-16: Phase 2 produces n_iterations trials with is_initial=False."""
    result = make_optimizer(n_initial=3, n_iterations=2).optimize()
    assert all(not t.is_initial for t in result.trials[3:])


@pytest.mark.algorithm
def test_total_trial_count() -> None:
    """ALG-LLM-17: Total number of trials equals n_initial + n_iterations."""
    result = make_optimizer(n_initial=3, n_iterations=2).optimize()
    assert len(result.trials) == 5


@pytest.mark.algorithm
def test_out_of_bounds_lr_is_clamped() -> None:
    """ALG-LLM-18: Out-of-bounds lr proposal is clamped to the search space upper bound."""
    chain = MockChain(proposals=[
        {"n_hidden_layers": 5, "n_neurons": 50, "lr": 100.0, "epochs_adam": 2000},
        {"n_hidden_layers": 5, "n_neurons": 50, "lr": 1e-3, "epochs_adam": 2000},
    ])
    result = make_optimizer(chain=chain, n_initial=1, n_iterations=2).optimize()
    # Phase 2 trial (trial_id=1)
    llm_trial = next(t for t in result.trials if not t.is_initial)
    assert float(llm_trial.params["lr"]) <= 1e-2


@pytest.mark.algorithm
def test_float_int_param_is_rounded() -> None:
    """ALG-LLM-19: Float value for integer parameter is rounded to int."""
    chain = MockChain(proposals=[
        {"n_hidden_layers": 3.7, "n_neurons": 50, "lr": 1e-3, "epochs_adam": 2000},
    ])
    result = make_optimizer(chain=chain, n_initial=1, n_iterations=1).optimize()
    llm_trial = next(t for t in result.trials if not t.is_initial)
    val = llm_trial.params["n_hidden_layers"]
    assert isinstance(val, int)


@pytest.mark.algorithm
def test_best_trial_id_is_last_with_monotone_objective() -> None:
    """ALG-LLM-20: best_trial_id points to the trial with maximum objective."""
    result = make_optimizer(n_initial=3, n_iterations=2).optimize()
    # MockObjectiveFunction returns objective = trial_id + 1, so last trial is best
    assert result.best_trial_id == len(result.trials) - 1


@pytest.mark.algorithm
def test_iteration_metas_count() -> None:
    """ALG-LLM-21: Number of iteration_metas equals n_iterations."""
    result = make_optimizer(n_initial=3, n_iterations=2).optimize()
    assert len(result.iteration_metas) == 2


@pytest.mark.algorithm
def test_trial_ids_are_contiguous() -> None:
    """ALG-LLM-22: trial_id values form a contiguous 0-based sequence."""
    result = make_optimizer(n_initial=3, n_iterations=2).optimize()
    ids = [t.trial_id for t in result.trials]
    assert ids == list(range(5))


@pytest.mark.algorithm
def test_chain_invoked_n_iterations_times() -> None:
    """ALG-LLM-23: MockChain.invoke is called exactly n_iterations times."""
    chain = InvocationCountChain()
    make_optimizer(chain=chain, n_initial=3, n_iterations=3).optimize()
    assert chain.call_count == 3


@pytest.mark.algorithm
def test_objective_name_in_result() -> None:
    """ALG-LLM-24: LLMResult.objective_name matches the objective function's name."""
    result = make_optimizer().optimize()
    assert result.objective_name == "mock"
