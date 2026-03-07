from __future__ import annotations
import torch
from torch import Tensor
from rbf_gen.kernels import Kernel


class RBFBasis:
    def __init__(self, centers: Tensor, kernel: Kernel):
        self.centers = centers
        self.kernel = kernel

    @classmethod
    def from_uniform(cls, K: int, bounds: Tensor, kernel: Kernel) -> RBFBasis:
        lb, ub = bounds[0], bounds[1]
        centers = torch.rand(K, lb.shape[0]) * (ub - lb) + lb
        return cls(centers, kernel)

    @classmethod
    def from_quasi_random(cls, K: int, bounds: Tensor, kernel: Kernel) -> RBFBasis:
        lb, ub = bounds[0], bounds[1]
        d = lb.shape[0]
        sobol = torch.quasirandom.SobolEngine(dimension=d)
        centers = sobol.draw(K).float() * (ub - lb) + lb
        return cls(centers, kernel)

    def compute_matrix(self, X: Tensor) -> Tensor:
        # X: (N, d) -> Phi: (N, K)
        diff = X.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, d)
        r = torch.norm(diff, dim=-1)  # (N, K)
        return self.kernel(r)

    def compute_vector(self, x: Tensor) -> Tensor:
        # x: (d,) -> phi: (K,)
        return self.compute_matrix(x.unsqueeze(0)).squeeze(0)
