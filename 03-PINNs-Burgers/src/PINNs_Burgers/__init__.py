"""PINNs Burgers equation solver package."""

from .api import BurgersPINNSolver
from .config import NetworkConfig, PDEConfig, TrainingConfig
from .data import BoundaryData, CollocationPoints
from .results import ForwardResult, InverseResult

__all__ = [
    "BurgersPINNSolver",
    "PDEConfig",
    "NetworkConfig",
    "TrainingConfig",
    "BoundaryData",
    "CollocationPoints",
    "ForwardResult",
    "InverseResult",
]
