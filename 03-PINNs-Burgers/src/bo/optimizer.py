"""Bayesian optimizer using BoTorch SingleTaskGP."""

import time

import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo.result import BOConfig, BOResult
from opt_tool.base import BaseOptimizer
from opt_tool.result import TrialResult
from opt_tool.space import SearchSpace


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimizer using BoTorch's SingleTaskGP and optimize_acqf.

    Implements a sequential Bayesian optimization loop:
    Phase 1 — Sobol initial sampling (n_initial evaluations) via BaseOptimizer.
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
        objective,
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
        super().__init__(search_space, objective, config)

    def _run_sequential_search(
        self, initial_trials: list[TrialResult]
    ) -> list[TrialResult]:
        """Phase 2: GP-guided sequential search.

        Rebuilds train_X from the same Sobol seed used in Phase 1 (deterministic).
        Fits a SingleTaskGP at each iteration, optimizes the acquisition function,
        evaluates the next candidate, and appends to trials.

        Parameters
        ----------
        initial_trials:
            Trial results from Phase 1 (Sobol initial exploration).

        Returns
        -------
        list[TrialResult]
            All trials including initial_trials and newly evaluated BO points.
        """
        trials: list[TrialResult] = list(initial_trials)

        # Reconstruct train_X using the same Sobol seed; draw_sobol_samples is deterministic
        train_X = self._search_space.sample_sobol(
            n=self._config.n_initial, seed=self._config.seed
        ).to(torch.float64)  # (n_initial, dim)
        train_Y = torch.tensor(
            [[t.objective] for t in initial_trials], dtype=torch.float64
        )  # (n_initial, 1)

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
            t0 = time.perf_counter()
            x_next, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self._search_space.bounds,
                q=1,
                num_restarts=self._config.num_restarts,
                raw_samples=self._config.raw_samples,
            )  # (1, dim)
            proposal_time = time.perf_counter() - t0

            params = self._search_space.from_tensor(x_next)
            trial = self._objective(params=params, trial_id=trial_id, is_initial=False)
            trial = trial.model_copy(update={"proposal_time": proposal_time})
            trials.append(trial)
            label = f"[BO     {iteration + 1:>{n_iter_width}}/{self._config.n_iterations}]"
            self._log_trial(trial, label)

            train_X = torch.cat([train_X, x_next.to(torch.float64)], dim=0)
            train_Y = torch.cat(
                [train_Y, torch.tensor([[trial.objective]], dtype=torch.float64)],
                dim=0,
            )

        return trials

    def _build_result(self, trials: list[TrialResult]) -> BOResult:
        """Phase 3: Select the best trial and build BOResult.

        Parameters
        ----------
        trials:
            All trials from Phase 1 and Phase 2.

        Returns
        -------
        BOResult
            Full optimization result with all trials and best configuration.
        """
        best_trial = max(trials, key=lambda t: t.objective)
        return BOResult(
            trials=trials,
            best_params=best_trial.params,
            best_objective=best_trial.objective,
            best_trial_id=best_trial.trial_id,
            objective_name=self._objective.name,
            bo_config=self._config,
        )
