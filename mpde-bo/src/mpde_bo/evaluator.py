import torch
from collections.abc import Callable
from torch import Tensor

from mpde_bo.optimizer import BOResult


class N90Evaluator:
    def __init__(
        self,
        budget: int = 200,
        n_functions: int = 100,
        threshold_ratio: float = 0.9,
        n_important: int = 2,
        n_unimportant: int = 8,
        grid_size: int = 100,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    def evaluate(
        self,
        algorithm: Callable[..., BOResult],
    ) -> float:
        raise NotImplementedError
