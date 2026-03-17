"""Visualization utilities for hyperparameter optimization results.

Generic functions that work with any single-phase optimizer result
(Bayesian Optimization, LLM-based, etc.) by accepting ``list[TrialResult]``
and metadata directly rather than optimizer-specific result types.

For PINN-specific heatmaps, import from ``opt_viz.pinn_heatmap``.
For Hybrid optimizer plots, import from ``hybrid``.
"""

from opt_viz.plots import plot_convergence, plot_objective_scatter, plot_parallel_coords

__all__ = [
    "plot_convergence",
    "plot_objective_scatter",
    "plot_parallel_coords",
]
