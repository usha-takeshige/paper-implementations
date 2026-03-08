from dataclasses import dataclass

from botorch.models import SingleTaskGP
from torch import Tensor


@dataclass
class GPConfig:
    kernel: str = "matern52"
    noise_var: float | None = None


class GPModelManager:
    def __init__(self, config: GPConfig) -> None:
        raise NotImplementedError

    def build(self, train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
        raise NotImplementedError

    def update(self, model: SingleTaskGP, new_X: Tensor, new_Y: Tensor) -> SingleTaskGP:
        raise NotImplementedError

    def get_length_scales(self, model: SingleTaskGP) -> Tensor:
        raise NotImplementedError

    def predict(self, model: SingleTaskGP, X: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
