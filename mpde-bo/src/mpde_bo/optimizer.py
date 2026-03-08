from collections.abc import Callable
from dataclasses import dataclass

from botorch.models import SingleTaskGP
from torch import Tensor

from mpde_bo.acquisition_optimizer import AcquisitionOptimizer
from mpde_bo.gp_model_manager import GPModelManager
from mpde_bo.parameter_classifier import ParameterClassification, ParameterClassifier


@dataclass
class BOResult:
    best_x: Tensor
    best_y: float
    train_X: Tensor
    train_Y: Tensor


class MPDEBOOptimizer:
    def __init__(
        self,
        gp_manager: GPModelManager,
        classifier: ParameterClassifier,
        acq_optimizer: AcquisitionOptimizer,
        bounds: Tensor,
    ) -> None:
        raise NotImplementedError

    def optimize(
        self,
        f: Callable[[Tensor], float],
        T: int,
        train_X: Tensor,
        train_Y: Tensor,
        callback: Callable[[int, SingleTaskGP, ParameterClassification], None] | None = None,
    ) -> BOResult:
        raise NotImplementedError
