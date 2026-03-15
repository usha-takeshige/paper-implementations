"""Behavioral checks for the opt_agent module (BHV-LLM-01 to BHV-LLM-03).

Visual and qualitative checks using MockChain and a synthetic objective.
Run with:
    uv run python tests/check_llm_behavior.py

Output is saved to tests/llm_behavior_output/.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bo import HyperParameter, SearchSpace, TrialResult
from opt_agent import BaseChain, LLMConfig, LLMOptimizer, LLMProposal


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "llm_behavior_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


class SyntheticObjective:
    """Synthetic objective that scores by distance to a known optimum."""

    OPTIMUM = {"n_hidden_layers": 6, "n_neurons": 80, "lr": 5e-3, "epochs_adam": 4000}

    @property
    def name(self) -> str:
        """Return objective name."""
        return "synthetic"

    def __call__(self, params: dict, trial_id: int, is_initial: bool) -> TrialResult:
        """Compute a distance-based objective."""
        score = (
            -abs(params.get("n_hidden_layers", 5) - 6) / 6.0
            - abs(params.get("n_neurons", 50) - 80) / 90.0
            - abs(float(params.get("lr", 1e-3)) - 5e-3) / 1e-2
            - abs(params.get("epochs_adam", 2000) - 4000) / 4500.0
        )
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=float(score),
            rel_l2_error=max(0.0, -score),
            elapsed_time=0.01,
            is_initial=is_initial,
        )


class GuidedMockChain(BaseChain):
    """MockChain that nudges proposals toward the synthetic optimum."""

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """Return proposals gradually approaching the optimum."""
        factor = min(1.0, (iteration_id + 1) / 10.0)
        n_hidden = int(2 + factor * 4)
        n_neurons = int(10 + factor * 70)
        lr = 1e-4 * (10 ** (factor * 2))
        epochs = int(500 + factor * 3500)
        return LLMProposal(
            analysis_report=(
                f"Iteration {iteration_id}: exploring toward promising region. "
                f"Best objective so far: {max(t.objective for t in trials):.4e}"
            ),
            proposed_params={
                "n_hidden_layers": n_hidden,
                "n_neurons": n_neurons,
                "lr": lr,
                "epochs_adam": epochs,
            },
            reasoning=(
                f"Increasing network size and learning rate toward estimated optimum "
                f"(factor={factor:.2f})."
            ),
        )


# ---------------------------------------------------------------------------
# BHV-LLM-01: Convergence curve
# ---------------------------------------------------------------------------


def check_convergence(result: object) -> None:
    """BHV-LLM-01: Plot cumulative best objective to verify monotone non-decrease."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))

    best_so_far: list[float] = []
    current_best = float("-inf")
    for t in result.trials:  # type: ignore[attr-defined]
        current_best = max(current_best, t.objective)
        best_so_far.append(current_best)

    trial_ids = list(range(len(result.trials)))  # type: ignore[attr-defined]
    n_initial = result.llm_config.n_initial  # type: ignore[attr-defined]

    ax.axvline(x=n_initial - 0.5, color="gray", linestyle="--", alpha=0.6,
               label="Sobol / LLM boundary")
    sns.lineplot(x=trial_ids, y=best_so_far, ax=ax, marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    ax.set_title(
        "BHV-LLM-01: LLM Optimization Convergence Curve\n"
        "CHECK: Curve should be monotonically non-decreasing. "
        "Phase 2 should show improvement over Phase 1."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bhv_llm_01_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [BHV-LLM-01] Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-LLM-02: Parameter distribution scatter
# ---------------------------------------------------------------------------


def check_param_distribution(result: object, search_space: SearchSpace) -> None:
    """BHV-LLM-02: Scatter plot showing Sobol vs LLM proposals in parameter space."""
    sns.set_theme(style="whitegrid")
    param_names = [hp.name for hp in search_space.parameters]

    initial_rows = []
    llm_rows = []
    for t in result.trials:  # type: ignore[attr-defined]
        x = search_space.to_tensor(t.params).squeeze(0).tolist()
        if t.is_initial:
            initial_rows.append(x)
        else:
            llm_rows.append(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [(0, 1), (2, 3)]  # (n_hidden_layers vs n_neurons), (lr vs epochs_adam)
    for ax, (i, j) in zip(axes, pairs):
        if initial_rows:
            arr = np.array(initial_rows)
            ax.scatter(arr[:, i], arr[:, j], label="Sobol initial", alpha=0.8, marker="o", s=70)
        if llm_rows:
            arr = np.array(llm_rows)
            ax.scatter(arr[:, i], arr[:, j], label="LLM proposal", alpha=0.8, marker="^", s=70)
        ax.set_xlabel(f"{param_names[i]} (normalized)")
        ax.set_ylabel(f"{param_names[j]} (normalized)")
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "BHV-LLM-02: Parameter Distribution — Sobol vs LLM Proposals\n"
        "CHECK: All points in [0,1]^2. LLM proposals should differ from Sobol pattern."
    )
    path = os.path.join(OUTPUT_DIR, "bhv_llm_02_param_distribution.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [BHV-LLM-02] Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-LLM-03: LLM analysis reports
# ---------------------------------------------------------------------------


def check_llm_reports(result: object) -> None:
    """BHV-LLM-03: Print LLM analysis reports and reasoning for each iteration."""
    print("\n" + "=" * 70)
    print("BHV-LLM-03: LLM Analysis Reports")
    print("CHECK: Each iteration should have a non-empty, contextual report.")
    print("=" * 70)
    for meta in result.iteration_metas:  # type: ignore[attr-defined]
        print(f"\n--- Iteration {meta.iteration_id} ---")
        print(f"Analysis:  {meta.analysis_report[:200]}")
        print(f"Reasoning: {meta.reasoning[:200]}")
        assert len(meta.analysis_report) > 0, "analysis_report is empty"
        assert len(meta.reasoning) > 0, "reasoning is empty"
    print("\n[BHV-LLM-03] All iteration reports are non-empty.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all behavioral checks with GuidedMockChain."""
    print("=== LLM Optimizer Behavioral Checks ===\n")
    search_space = make_search_space()
    optimizer = LLMOptimizer(
        search_space=search_space,
        objective=SyntheticObjective(),
        config=LLMConfig(n_initial=5, n_iterations=10, seed=42),
        chain=GuidedMockChain(),
    )
    result = optimizer.optimize()

    print(f"\nBest trial: #{result.best_trial_id}")
    print(f"  Best objective: {result.best_objective:.4e}")
    print(f"  Best params:    {result.best_params}")

    print("\nGenerating behavioral check plots ...")
    check_convergence(result)
    check_param_distribution(result, search_space)
    check_llm_reports(result)

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
