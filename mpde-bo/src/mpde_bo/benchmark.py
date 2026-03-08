import torch
from torch import Tensor


class BenchmarkFunction:
    def __init__(
        self,
        n_important: int,
        n_unimportant: int,
        grid_size: int = 100,
        a: tuple[float, ...] | None = None,
        sigma: tuple[float, ...] | None = None,
        a_s: float = 0.1,
        sigma_s: float = 25.0,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def n_important(self) -> int:
        raise NotImplementedError

    @property
    def n_unimportant(self) -> int:
        raise NotImplementedError

    @property
    def grid_size(self) -> int:
        raise NotImplementedError

    @property
    def optimal_value(self) -> float:
        raise NotImplementedError

    @property
    def peak_centers(self) -> list[Tensor]:
        raise NotImplementedError

    @property
    def peak_widths(self) -> list[float]:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> float:
        raise NotImplementedError
