"""共有フィクスチャ"""

import pytest
import torch
from torch import Tensor


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator().manual_seed(42)


@pytest.fixture
def simple_train_data() -> tuple[Tensor, Tensor]:
    """
    2次元入力、二次関数目的値のデータ (n=10)
    f(x) = -(x[0]-0.5)^2 - (x[1]-0.5)^2
    """
    gen = torch.Generator().manual_seed(0)
    train_X = torch.rand(10, 2, dtype=torch.double, generator=gen)
    train_Y = -(train_X[:, 0:1] - 0.5) ** 2 - (train_X[:, 1:2] - 0.5) ** 2
    return train_X, train_Y


@pytest.fixture
def important_unimportant_data() -> tuple[Tensor, Tensor]:
    """
    N=4 次元, n=50 観測
    次元 0,1 のみ目的値に影響 (sin による急峻な変動)、次元 2,3 は無関係

    GPModelManager / ImportanceAnalyzer / ParameterClassifier の
    Property テスト・Integration テストで共用する。
    """
    gen = torch.Generator().manual_seed(1)
    N, n = 4, 50
    train_X = torch.rand(n, N, dtype=torch.double, generator=gen)
    train_Y = (
        torch.sin(5.0 * train_X[:, 0]) + torch.sin(5.0 * train_X[:, 1])
    ).unsqueeze(-1)
    return train_X, train_Y


@pytest.fixture
def fitted_gp_model(simple_train_data: tuple[Tensor, Tensor]):
    """simple_train_data で学習済みの SingleTaskGP"""
    from mpde_bo.gp_model_manager import GPConfig, GPModelManager

    train_X, train_Y = simple_train_data
    manager = GPModelManager(GPConfig(kernel="matern52"))
    return manager.build(train_X, train_Y)


@pytest.fixture
def gp_manager_matern():
    from mpde_bo.gp_model_manager import GPConfig, GPModelManager

    return GPModelManager(GPConfig(kernel="matern52"))


@pytest.fixture
def classification_config():
    from mpde_bo.parameter_classifier import ClassificationConfig

    return ClassificationConfig(eps_l=1.0, eps_e=0.1)


@pytest.fixture
def bounds_2d() -> Tensor:
    """shape (2, 2): 各次元 [0, 1]"""
    return torch.stack([torch.zeros(2), torch.ones(2)])


@pytest.fixture
def bounds_4d() -> Tensor:
    """shape (2, 4): 各次元 [0, 1]"""
    return torch.stack([torch.zeros(4), torch.ones(4)])
