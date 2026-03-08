from abc import ABC, abstractmethod
import torch
from torch import Tensor


class Kernel(ABC):
    @abstractmethod
    def __call__(self, r: Tensor) -> Tensor:
        pass


class GaussianKernel(Kernel):
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def __call__(self, r: Tensor) -> Tensor:
        return torch.exp(-(self.epsilon ** 2) * r ** 2)


class ThinPlateSplineKernel(Kernel):
    def __call__(self, r: Tensor) -> Tensor:
        # phi(r) = r^2 * log(r), limit phi(0) = 0
        safe_r = torch.where(r > 0, r, torch.ones_like(r))
        result = safe_r ** 2 * torch.log(safe_r)
        return torch.where(r > 0, result, torch.zeros_like(r))
