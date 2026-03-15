"""Search space definition for hyperparameter optimization."""

import math
from typing import Literal

import torch
from botorch.utils.sampling import draw_sobol_samples
from pydantic import BaseModel, ConfigDict, Field


class HyperParameter(BaseModel):
    """Definition of a single hyperparameter.

    Stores the name, type, bounds, and scale of one hyperparameter to be
    optimized. Used by SearchSpace to define the full optimization domain.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="パラメータ名（NetworkConfig / TrainingConfig のフィールド名）"
    )
    param_type: Literal["int", "float"] = Field(description="パラメータの型")
    low: float = Field(description="探索範囲の下限（実際のパラメータスケール）")
    high: float = Field(description="探索範囲の上限（実際のパラメータスケール）")
    log_scale: bool = Field(
        default=False, description="True の場合、対数スケールで探索する"
    )


class SearchSpace(BaseModel):
    """Search space for hyperparameter optimization.

    Manages the set of hyperparameters to tune and handles normalized
    conversion to/from the [0, 1]^d unit hypercube that BoTorch expects.
    Linear parameters use (x - low) / (high - low) normalization;
    log-scale parameters use log-domain linear normalization.
    """

    model_config = ConfigDict(frozen=True)

    parameters: list[HyperParameter] = Field(
        description="チューニング対象のハイパーパラメータ一覧"
    )

    @property
    def dim(self) -> int:
        """Return the dimensionality of the search space."""
        return len(self.parameters)

    @property
    def bounds(self) -> torch.Tensor:
        """Return normalized bounds tensor of shape (2, dim) with all values in [0, 1].

        The lower bound row is all zeros and the upper bound row is all ones,
        since inputs are manually normalized to [0, 1]^d.
        """
        lower = torch.zeros(self.dim, dtype=torch.float64)
        upper = torch.ones(self.dim, dtype=torch.float64)
        return torch.stack([lower, upper], dim=0)

    def sample_sobol(self, n: int, seed: int) -> torch.Tensor:
        """Sample n points using a Sobol sequence for initial exploration.

        Uses BoTorch's draw_sobol_samples for reproducible quasi-random
        sampling that covers the search space more uniformly than random.

        Parameters
        ----------
        n:
            Number of points to sample.
        seed:
            Random seed for Sobol sequence reproducibility.

        Returns
        -------
        torch.Tensor
            Shape (n, dim), dtype float64, values in [0, 1]^d.
        """
        bounds = self.bounds
        # draw_sobol_samples returns shape (n, 1, dim) with q=1
        samples = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed)
        return samples.squeeze(1).to(torch.float64)

    def to_tensor(self, params: dict[str, float | int]) -> torch.Tensor:
        """Convert a parameter dict to a normalized BoTorch input tensor.

        Applies linear normalization x' = (x - low) / (high - low) for
        linear-scale parameters, and log normalization
        x' = (log x - log low) / (log high - log low) for log-scale.

        Parameters
        ----------
        params:
            Parameter values in the original (un-normalized) scale.

        Returns
        -------
        torch.Tensor
            Shape (1, dim), dtype float64, values in [0, 1].
        """
        x = torch.zeros(self.dim, dtype=torch.float64)
        for i, hp in enumerate(self.parameters):
            val = float(params[hp.name])
            if hp.log_scale:
                x[i] = (math.log(val) - math.log(hp.low)) / (
                    math.log(hp.high) - math.log(hp.low)
                )
            else:
                x[i] = (val - hp.low) / (hp.high - hp.low)
        return x.unsqueeze(0)  # shape (1, dim)

    def from_tensor(self, x: torch.Tensor) -> dict[str, float | int]:
        """Convert a normalized tensor back to a parameter dict.

        Inverts to_tensor: applies linear or log de-normalization.
        Integer parameters are rounded to the nearest integer.

        Parameters
        ----------
        x:
            Normalized tensor of shape (dim,) or (1, dim).

        Returns
        -------
        dict[str, float | int]
            Parameter values in the original scale.
        """
        if x.dim() == 2:
            x = x.squeeze(0)
        result: dict[str, float | int] = {}
        for i, hp in enumerate(self.parameters):
            val_norm = float(x[i].clamp(0.0, 1.0))
            if hp.log_scale:
                val = math.exp(
                    val_norm * (math.log(hp.high) - math.log(hp.low)) + math.log(hp.low)
                )
            else:
                val = val_norm * (hp.high - hp.low) + hp.low
            if hp.param_type == "int":
                result[hp.name] = int(round(val))
            else:
                result[hp.name] = float(val)
        return result
