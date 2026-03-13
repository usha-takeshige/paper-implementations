"""Data classes for boundary conditions and collocation points."""

import torch
from pydantic import BaseModel, ConfigDict, Field


class BoundaryData(BaseModel):
    """初期・境界条件データ点 {(t_u^i, x_u^i, u^i)} を保持する。

    式(8) に対応。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: torch.Tensor = Field(description="時間座標，形状 (N_u, 1)")
    x: torch.Tensor = Field(description="空間座標，形状 (N_u, 1)")
    u: torch.Tensor = Field(description="観測値 u(t, x)，形状 (N_u, 1)")


class CollocationPoints(BaseModel):
    """コロケーション点 {(t_f^j, x_f^j)} を保持する。

    式(9) に対応。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: torch.Tensor = Field(description="時間座標，形状 (N_f, 1)")
    x: torch.Tensor = Field(description="空間座標，形状 (N_f, 1)")
