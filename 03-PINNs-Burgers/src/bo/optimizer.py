"""Bayesian optimizer using BoTorch SingleTaskGP."""

import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo.objective import ObjectiveFunction
from bo.result import BOConfig, BOResult, TrialResult
from bo.space import SearchSpace


class BayesianOptimizer:
    """Bayesian optimizer using BoTorch's SingleTaskGP and optimize_acqf.

    Implements a sequential Bayesian optimization loop:
    Phase 1 — Sobol initial sampling (n_initial evaluations).
    Phase 2 — GP-guided search: fit SingleTaskGP, optimize acquisition
              function, evaluate next candidate, repeat for n_iterations.
    Phase 3 — Aggregate all trials into BOResult.

    BoTorch components used:
    - SingleTaskGP with Standardize outcome transform (Matern 5/2 kernel)
    - fit_gpytorch_mll for GP hyperparameter fitting
    - LogExpectedImprovement or UpperConfidenceBound acquisition function
    - optimize_acqf with gradient-based multi-start optimization
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        config: BOConfig,
    ) -> None:
        """Initialize BayesianOptimizer.

        Parameters
        ----------
        search_space:
            Search space defining the hyperparameter domain.
        objective:
            Callable that evaluates a set of hyperparameters.
        config:
            BO configuration (n_initial, n_iterations, acquisition, etc.).
        """
        self._search_space = search_space
        self._objective = objective
        self._config = config

    def _log_trial(self, trial: TrialResult, label: str) -> None:
        """Print a single trial result to stdout.

        Parameters
        ----------
        trial:
            The trial result to log.
        label:
            Short label displayed at the beginning of the line (e.g. "[Initial 1/5]").
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

    def optimize(self) -> BOResult:
        """Run the full Bayesian optimization loop and return BOResult.

        Phase 1 — Initial Sobol sampling (config.n_initial points):
          1. Draw n_initial quasi-random points with SearchSpace.sample_sobol.
          2. Convert each to a parameter dict with SearchSpace.from_tensor.
          3. Evaluate ObjectiveFunction and record TrialResult.
          4. Build initial train_X (n_initial, dim) and train_Y (n_initial, 1).

        Phase 2 — GP-based sequential search (config.n_iterations rounds):
          5. Fit SingleTaskGP(train_X, train_Y, Standardize(m=1)).
          6. Fit GP kernel parameters with fit_gpytorch_mll.
          7. Build LogExpectedImprovement or UpperConfidenceBound.
          8. Optimize acquisition function with optimize_acqf to get x_next.
          9. Evaluate ObjectiveFunction at x_next and record TrialResult.
          10. Append x_next to train_X and objective to train_Y.

        Phase 3 — Collect results:
          11. Select the trial with maximum objective.
          12. Return BOResult.

        Returns
        -------
        BOResult
            Full optimization result with all trials and best configuration.
        """
        trials: list[TrialResult] = []

        # ------------------------------------------------------------------ #
        # Phase 1: Initial Sobol sampling
        # ------------------------------------------------------------------ #
        sobol_points = self._search_space.sample_sobol(
            n=self._config.n_initial, seed=self._config.seed
        )  # (n_initial, dim)

        n_init_width = len(str(self._config.n_initial))
        for i in range(self._config.n_initial):
            x_i = sobol_points[i]  # (dim,)
            params = self._search_space.from_tensor(x_i)
            trial = self._objective(params=params, trial_id=i, is_initial=True)
            trials.append(trial)
            label = f"[Initial {i + 1:>{n_init_width}}/{self._config.n_initial}]"
            self._log_trial(trial, label)

        train_X = sobol_points.clone().to(torch.float64)  # (n_initial, dim)
        train_Y = torch.tensor(
            [[t.objective] for t in trials], dtype=torch.float64
        )  # (n_initial, 1)

        # ------------------------------------------------------------------ #
        # Phase 2: GP-based sequential search
        # ------------------------------------------------------------------ #
        n_iter_width = len(str(self._config.n_iterations))
        for iteration in range(self._config.n_iterations):
            trial_id = self._config.n_initial + iteration

            # Fit Gaussian process surrogate
            gp = SingleTaskGP(
                train_X, train_Y, outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Build acquisition function
            if self._config.acquisition == "EI":
                acq_func = LogExpectedImprovement(
                    model=gp, best_f=train_Y.max()
                )
            else:
                acq_func = UpperConfidenceBound(
                    model=gp, beta=self._config.ucb_beta
                )

            # Optimize acquisition to get next candidate (q=1: one point at a time)
            x_next, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self._search_space.bounds,
                q=1,
                num_restarts=self._config.num_restarts,
                raw_samples=self._config.raw_samples,
            )  # (1, dim)

            params = self._search_space.from_tensor(x_next)
            trial = self._objective(params=params, trial_id=trial_id, is_initial=False)
            trials.append(trial)
            label = f"[BO     {iteration + 1:>{n_iter_width}}/{self._config.n_iterations}]"
            self._log_trial(trial, label)

            train_X = torch.cat([train_X, x_next.to(torch.float64)], dim=0)
            train_Y = torch.cat(
                [train_Y, torch.tensor([[trial.objective]], dtype=torch.float64)],
                dim=0,
            )

        # ------------------------------------------------------------------ #
        # Phase 3: Aggregate results
        # ------------------------------------------------------------------ #
        best_trial = max(trials, key=lambda t: t.objective)
        return BOResult(
            trials=trials,
            best_params=best_trial.params,
            best_objective=best_trial.objective,
            best_trial_id=best_trial.trial_id,
            bo_config=self._config,
        )
