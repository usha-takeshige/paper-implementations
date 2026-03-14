"""Behavior tests for the bo module (BHV-BO-01 to BHV-BO-04).

Visual checks using a synthetic objective function — no PINN training required.
Outputs graphs that a human should inspect for correctness.

Confirmation items:
  BHV-BO-01: BO convergence curve (best objective vs trial)
  BHV-BO-02: Initial sample vs BO proposal distribution (scatter)
  BHV-BO-03: GP surrogate model quality (1D cross-section)
  BHV-BO-04: Sobol initial sampling coverage (histograms)

Usage:
    cd 03-PINNs-Burgers
    uv run python tests/check_bo_behavior.py

Output: tests/behavior_output/
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from bo import (
    BOConfig,
    BayesianOptimizer,
    BOResult,
    HyperParameter,
    SearchSpace,
    TrialResult,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "behavior_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic objective function (no PINN training)
# ---------------------------------------------------------------------------


def synthetic_objective(params: dict) -> tuple[float, float]:
    """Synthetic objective: n=50, lr=1e-3 gives minimum error.

    Returns (rel_l2_error, elapsed_time).
    """
    n = float(params["n_neurons"])
    lr = float(params["lr"])
    rel_l2_error = 0.1 * (1 + abs(n - 50) / 50 + abs(math.log10(lr) + 3))
    elapsed_time = 0.01 * n
    return rel_l2_error, elapsed_time


class SyntheticObjectiveFunction:
    """Wraps synthetic_objective as a TrialResult-returning callable."""

    def __call__(
        self, params: dict, trial_id: int, is_initial: bool
    ) -> TrialResult:
        """Return TrialResult from synthetic function."""
        rel_l2_error, elapsed_time = synthetic_objective(params)
        objective = 1.0 / max(rel_l2_error * elapsed_time, 1e-10)
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=objective,
            rel_l2_error=rel_l2_error,
            elapsed_time=elapsed_time,
            is_initial=is_initial,
        )


# ---------------------------------------------------------------------------
# Run BO with synthetic objective
# ---------------------------------------------------------------------------


def run_bo() -> tuple[BOResult, SearchSpace]:
    """Run BayesianOptimizer with the synthetic objective function."""
    search_space = SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])
    bo_config = BOConfig(
        n_initial=8, n_iterations=15, acquisition="EI", seed=42,
        num_restarts=5, raw_samples=128,
    )
    obj = SyntheticObjectiveFunction()
    optimizer = BayesianOptimizer(search_space, obj, bo_config)
    return optimizer.optimize(), search_space


# ---------------------------------------------------------------------------
# BHV-BO-01: Convergence curve
# ---------------------------------------------------------------------------


def check_bhv_bo_01_convergence(result: BOResult) -> None:
    """BHV-BO-01: BO convergence — best objective should improve over trials.

    Pass criteria: The curve should generally trend upward (right) and plateau.
    """
    sns.set_theme(style="whitegrid")
    best_so_far: list[float] = []
    current_best = float("-inf")
    for t in result.trials:
        current_best = max(current_best, t.objective)
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(9, 4))
    n_initial = result.bo_config.n_initial
    ax.axvline(x=n_initial - 0.5, color="gray", linestyle="--", alpha=0.7,
               label="Sobol / BO boundary")
    sns.lineplot(x=list(range(len(best_so_far))), y=best_so_far,
                 ax=ax, linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Objective So Far")
    ax.set_title(
        "BHV-BO-01: BO Convergence Curve (Synthetic Objective)\n"
        "CHECK: Curve should trend upward and plateau — indicates BO converges."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bo_convergence_behavior.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-BO-02: Scatter of initial vs BO proposals
# ---------------------------------------------------------------------------


def check_bhv_bo_02_scatter(result: BOResult) -> None:
    """BHV-BO-02: BO proposals should concentrate in higher-objective regions.

    Pass criteria: BO proposal points (triangle) should cluster near higher
    objective values compared to random Sobol initial samples (circle).
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))

    initial = [(t.trial_id, t.objective) for t in result.trials if t.is_initial]
    bo_pts  = [(t.trial_id, t.objective) for t in result.trials if not t.is_initial]

    if initial:
        ids, objs = zip(*initial)
        ax.scatter(ids, objs, label="Sobol initial", marker="o", alpha=0.8, s=70)
    if bo_pts:
        ids, objs = zip(*bo_pts)
        ax.scatter(ids, objs, label="BO proposal", marker="^", alpha=0.8, s=70)

    ax.axhline(y=result.best_objective, color="red", linestyle="--", alpha=0.6,
               label=f"Best={result.best_objective:.3e}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective  1/(error × time)")
    ax.set_title(
        "BHV-BO-02: Objective per Trial — Sobol vs BO Proposals\n"
        "CHECK: BO proposals (▲) should lean toward higher objective values."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bo_objective_scatter_behavior.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-BO-03: GP surrogate quality (1D cross-section)
# ---------------------------------------------------------------------------


def check_bhv_bo_03_gp_surrogate(result: BOResult, search_space: SearchSpace) -> None:
    """BHV-BO-03: GP surrogate should pass through observed points.

    Fits a 1D GP on n_neurons vs objective (marginalizing other dims to median).
    Pass criteria: GP mean should pass through all observation points;
    uncertainty should be large far from observations.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    sns.set_theme(style="whitegrid")

    # Extract n_neurons normalized values and objectives
    n_neurons_idx = [hp.name for hp in search_space.parameters].index("n_neurons")
    train_X = torch.tensor([
        [float(search_space.to_tensor(t.params)[0, n_neurons_idx])]
        for t in result.trials
    ], dtype=torch.float64)
    train_Y = torch.tensor([[t.objective] for t in result.trials], dtype=torch.float64)

    # Fit 1D GP
    gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Predict over [0, 1]
    test_x = torch.linspace(0, 1, 100, dtype=torch.float64).unsqueeze(1)
    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(test_x)
        mean = posterior.mean.squeeze().numpy()
        std = posterior.variance.clamp(min=0).sqrt().squeeze().numpy()

    # Map normalized n_neurons back to real scale for the x-axis
    # Use the HP definition directly to avoid passing a 1-dim tensor to from_tensor
    hp = search_space.parameters[n_neurons_idx]
    n_neurons_vals = [
        int(round(float(x.item()) * (hp.high - hp.low) + hp.low))
        for x in test_x.squeeze()
    ]
    obs_n_neurons = [int(t.params["n_neurons"]) for t in result.trials]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(n_neurons_vals, mean - 2 * std, mean + 2 * std,
                    alpha=0.2, label="GP ±2σ")
    ax.plot(n_neurons_vals, mean, linewidth=1.5, label="GP mean")
    ax.scatter(obs_n_neurons, [t.objective for t in result.trials],
               color="red", zorder=5, label="Observations", s=50)
    ax.set_xlabel("n_neurons")
    ax.set_ylabel("Objective")
    ax.set_title(
        "BHV-BO-03: GP Surrogate Quality (1D cross-section: n_neurons)\n"
        "CHECK: GP mean (blue) should pass near observations (red dots);\n"
        "uncertainty (shaded) should be larger away from observations."
    )
    ax.legend()
    path = os.path.join(OUTPUT_DIR, "bo_gp_surrogate_behavior.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# BHV-BO-04: Sobol sampling coverage
# ---------------------------------------------------------------------------


def check_bhv_bo_04_sobol_coverage(result: BOResult, search_space: SearchSpace) -> None:
    """BHV-BO-04: Initial Sobol samples should cover the search space uniformly.

    Pass criteria: Histograms for each parameter should be roughly flat
    (uniform distribution) for the Sobol initial samples.
    """
    sns.set_theme(style="whitegrid")
    param_names = [hp.name for hp in search_space.parameters]

    initial_trials = [t for t in result.trials if t.is_initial]
    if not initial_trials:
        print("  No initial trials found; skipping BHV-BO-04.")
        return

    # Normalized values for each parameter
    norm_values: dict[str, list[float]] = {name: [] for name in param_names}
    for t in initial_trials:
        x = search_space.to_tensor(t.params).squeeze(0)
        for i, name in enumerate(param_names):
            norm_values[name].append(float(x[i]))

    fig, axes = plt.subplots(1, len(param_names), figsize=(14, 3))
    for ax, name in zip(axes, param_names):
        sns.histplot(norm_values[name], bins=max(3, len(initial_trials) // 2),
                     ax=ax, stat="density")
        ax.set_xlabel(f"{name}\n(normalized [0,1])")
        ax.set_ylabel("Density")
        ax.set_title(name)

    fig.suptitle(
        "BHV-BO-04: Sobol Initial Sample Coverage (normalized search space)\n"
        "CHECK: Each histogram should be roughly uniform — no strong clustering."
    )
    path = os.path.join(OUTPUT_DIR, "bo_sobol_coverage_behavior.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all behavior checks for the bo module."""
    print("=== BO Behavior Checks (Synthetic Objective) ===\n")
    print("Running Bayesian optimization with synthetic objective ...")
    result, search_space = run_bo()
    print(f"  Total trials: {len(result.trials)}")
    print(f"  Best objective: {result.best_objective:.4e}")
    print(f"  Best trial_id:  {result.best_trial_id}\n")

    print("Generating behavior check plots ...")
    check_bhv_bo_01_convergence(result)
    check_bhv_bo_02_scatter(result)
    check_bhv_bo_03_gp_surrogate(result, search_space)
    check_bhv_bo_04_sobol_coverage(result, search_space)

    print(f"\nDone. Plots saved to {OUTPUT_DIR}/")
    print("\nManual inspection required:")
    print("  BHV-BO-01: Convergence curve should trend upward / plateau.")
    print("  BHV-BO-02: BO proposals should concentrate near higher objectives.")
    print("  BHV-BO-03: GP mean should pass near observations; std large far from data.")
    print("  BHV-BO-04: Sobol histograms should be roughly uniform.")


if __name__ == "__main__":
    main()
