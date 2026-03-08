from dataclasses import dataclass, field

from botorch.models import SingleTaskGP
from torch import Tensor

VALID_ACQUISITIONS = {"EI", "UCB", "PI"}


@dataclass
class AcquisitionConfig:
    type: str = "EI"
    ucb_beta: float = 2.0
    num_restarts: int = 10
    raw_samples: int = 512

    def __post_init__(self) -> None:
        if self.type not in VALID_ACQUISITIONS:
            raise ValueError(
                f"Unknown acquisition type '{self.type}'. "
                f"Choose from {VALID_ACQUISITIONS}."
            )


class AcquisitionOptimizer:
    def __init__(self, config: AcquisitionConfig, bounds: Tensor) -> None:
        raise NotImplementedError

    def maximize(
        self,
        model: SingleTaskGP,
        train_Y: Tensor,
        fixed_features: dict[int, float],
    ) -> Tensor:
        raise NotImplementedError
