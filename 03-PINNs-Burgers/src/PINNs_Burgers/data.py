"""Data classes for boundary conditions and collocation points."""

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundaryData(BaseModel):
    """初期・境界条件データ点 {(t_u^i, x_u^i, u^i)} を保持する。

    式(8) に対応。各テンソルは形状 (N_u, 1) であり，N_u は全フィールドで一致する必要がある。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: torch.Tensor = Field(description="時間座標，形状 (N_u, 1)")
    x: torch.Tensor = Field(description="空間座標，形状 (N_u, 1)")
    u: torch.Tensor = Field(description="観測値 u(t, x)，形状 (N_u, 1)")

    @model_validator(mode="after")
    def validate_shapes(self) -> "BoundaryData":
        """全テンソルが形状 (N_u, 1) であり，N_u が一致することを検証する。"""
        for name, tensor in [("t", self.t), ("x", self.x), ("u", self.u)]:
            if tensor.ndim != 2 or tensor.shape[1] != 1:
                raise ValueError(
                    f"BoundaryData.{name} must have shape (N, 1), got {tuple(tensor.shape)}"
                )
        n = self.t.shape[0]
        if self.x.shape[0] != n or self.u.shape[0] != n:
            raise ValueError(
                "BoundaryData.t, x, u must have the same number of rows, "
                f"got t={self.t.shape[0]}, x={self.x.shape[0]}, u={self.u.shape[0]}"
            )
        return self


class CollocationPoints(BaseModel):
    """コロケーション点 {(t_f^j, x_f^j)} を保持する。

    式(9) に対応。t と x は同一形状 (N_f, 1) である必要がある。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: torch.Tensor = Field(description="時間座標，形状 (N_f, 1)")
    x: torch.Tensor = Field(description="空間座標，形状 (N_f, 1)")

    @model_validator(mode="after")
    def validate_shapes(self) -> "CollocationPoints":
        """t と x が同一形状 (N_f, 1) であることを検証する。"""
        for name, tensor in [("t", self.t), ("x", self.x)]:
            if tensor.ndim != 2 or tensor.shape[1] != 1:
                raise ValueError(
                    f"CollocationPoints.{name} must have shape (N, 1), got {tuple(tensor.shape)}"
                )
        if self.t.shape[0] != self.x.shape[0]:
            raise ValueError(
                "CollocationPoints.t and x must have the same number of rows, "
                f"got t={self.t.shape[0]}, x={self.x.shape[0]}"
            )
        return self
