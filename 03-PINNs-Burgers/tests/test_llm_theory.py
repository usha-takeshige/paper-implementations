"""Theoretical property tests for the opt_agent module (THR-LLM-01 to THR-LLM-06).

Validates design-guaranteed properties of the LLM-based optimizer.
All tests use MockChain to avoid Gemini API calls.
"""

import pytest

from bo import HyperParameter, SearchSpace, TrialResult
from opt_agent import (
    BaseChain,
    LLMConfig,
    LLMOptimizer,
    LLMProposal,
    MaximizeObjectivePromptBuilder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    """Return the 4-parameter SearchSpace for PINN optimization."""
    return SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])


class MockObjectiveFunction:
    """Deterministic mock returning TrialResult without PINN training."""

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
    """Return fixed mid-point proposals without calling Gemini API."""

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Return a default mid-point proposal."""
        return LLMProposal(
            analysis_report="Mock analysis",
            proposed_params={
                "n_hidden_layers": 5,
                "n_neurons": 50,
                "lr": 1e-3,
                "epochs_adam": 2000,
            },
            reasoning="Mock reasoning",
        )


def make_optimizer(seed: int = 42) -> LLMOptimizer:
    """Build a small LLMOptimizer for theory tests."""
    return LLMOptimizer(
        search_space=make_search_space(),
        objective=MockObjectiveFunction(),
        config=LLMConfig(n_initial=3, n_iterations=2, seed=seed),
        chain=MockChain(),
    )


# ---------------------------------------------------------------------------
# THR-LLM-01: Sobol reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_sobol_phase_reproducibility() -> None:
    """THR-LLM-01: Phase 1 params are identical across runs with the same seed.

    Basis: LLMConfig.seed controls Sobol sampling (llm_opt_design.md §LLMConfig).
    """
    result1 = make_optimizer(seed=42).optimize()
    result2 = make_optimizer(seed=42).optimize()
    initial1 = [t.params for t in result1.trials if t.is_initial]
    initial2 = [t.params for t in result2.trials if t.is_initial]
    assert initial1 == initial2


# ---------------------------------------------------------------------------
# THR-LLM-02: best_objective is global maximum
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_best_objective_is_global_max() -> None:
    """THR-LLM-02: LLMResult.best_objective equals the max across all trials.

    Basis: llm_opt_design.md §LLMResult — best_objective is the maximum objective.
    """
    result = make_optimizer().optimize()
    expected_max = max(t.objective for t in result.trials)
    assert result.best_objective == expected_max


# ---------------------------------------------------------------------------
# THR-LLM-03: best_params consistency with best_trial_id
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_best_params_matches_best_trial() -> None:
    """THR-LLM-03: best_params equals the params of the trial at best_trial_id.

    Basis: llm_opt_design.md §LLMResult — best_params derived from best_trial_id.
    """
    result = make_optimizer().optimize()
    best_trial = next(t for t in result.trials if t.trial_id == result.best_trial_id)
    assert result.best_params == best_trial.params


# ---------------------------------------------------------------------------
# THR-LLM-04: All Phase 2 params within search space bounds
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_phase2_params_within_bounds() -> None:
    """THR-LLM-04: All LLM-proposed params are clamped within search space bounds.

    Basis: llm_opt_design.md §7 — out-of-bounds values are clamped.
    """
    search_space = make_search_space()
    result = LLMOptimizer(
        search_space=search_space,
        objective=MockObjectiveFunction(),
        config=LLMConfig(n_initial=2, n_iterations=3, seed=42),
        chain=MockChain(),
    ).optimize()
    llm_trials = [t for t in result.trials if not t.is_initial]
    for trial in llm_trials:
        for hp in search_space.parameters:
            val = float(trial.params[hp.name])
            assert val >= hp.low, f"{hp.name}={val} below low={hp.low}"
            assert val <= hp.high, f"{hp.name}={val} above high={hp.high}"


# ---------------------------------------------------------------------------
# THR-LLM-05: System prompt completeness
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_system_prompt_contains_all_param_info() -> None:
    """THR-LLM-05: System prompt contains all parameter names, types, and bounds.

    Basis: llm_opt_design.md §PromptBuilder — system prompt includes full space definition.
    """
    search_space = make_search_space()
    prompt = MaximizeObjectivePromptBuilder().build_system_prompt(search_space, "accuracy")
    for hp in search_space.parameters:
        assert hp.name in prompt
        assert hp.param_type in prompt
        assert str(hp.low) in prompt
        assert str(hp.high) in prompt


# ---------------------------------------------------------------------------
# THR-LLM-06: iteration_meta.proposed_params matches evaluated trial params
# ---------------------------------------------------------------------------


@pytest.mark.theory
def test_iteration_meta_params_match_trial_params() -> None:
    """THR-LLM-06: LLMIterationMeta.proposed_params matches the evaluated trial's params.

    Basis: llm_opt_design.md §7 — proposed_params is clamped/rounded before evaluation.
    After clamping and rounding, meta.proposed_params must equal trial.params.
    Tolerance: atol=1e-9 (per test design §5).
    """
    result = make_optimizer().optimize()
    n_initial = 3
    for meta in result.iteration_metas:
        trial_idx = n_initial + meta.iteration_id
        trial = result.trials[trial_idx]
        for key, val in meta.proposed_params.items():
            assert abs(float(val) - float(trial.params[key])) < 1e-9, (
                f"Mismatch for '{key}': meta={val}, trial={trial.params[key]}"
            )
