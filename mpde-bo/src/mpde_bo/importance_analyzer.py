from botorch.models import SingleTaskGP
from torch import Tensor


class ImportanceAnalyzer:
    def compute_ice(
        self,
        model: SingleTaskGP,
        train_X: Tensor,
        param_indices: list[int],
        x_S_grid: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def compute_mpde(
        self,
        model: SingleTaskGP,
        train_X: Tensor,
        param_indices: list[int],
        x_S_grid: Tensor | None = None,
    ) -> float:
        raise NotImplementedError

    def compute_apde(
        self,
        model: SingleTaskGP,
        train_X: Tensor,
        param_indices: list[int],
        x_S_grid: Tensor | None = None,
    ) -> float:
        raise NotImplementedError
