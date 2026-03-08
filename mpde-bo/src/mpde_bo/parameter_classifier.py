from dataclasses import dataclass

from botorch.models import SingleTaskGP
from torch import Tensor

from mpde_bo.importance_analyzer import ImportanceAnalyzer


@dataclass
class ClassificationConfig:
    eps_l: float
    eps_e: float


@dataclass
class ParameterClassification:
    important: list[int]
    unimportant: list[int]


class ParameterClassifier:
    def __init__(self, analyzer: ImportanceAnalyzer, config: ClassificationConfig) -> None:
        raise NotImplementedError

    def classify(self, model: SingleTaskGP, train_X: Tensor) -> ParameterClassification:
        raise NotImplementedError
