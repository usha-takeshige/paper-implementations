"""MPDEBOOptimizer のテスト (test_cases.md §5, テスト 10-1〜10-7)"""

import pytest
import torch
from torch import Tensor

from mpde_bo.acquisition_optimizer import AcquisitionConfig, AcquisitionOptimizer
from mpde_bo.gp_model_manager import GPConfig, GPModelManager
from mpde_bo.importance_analyzer import ImportanceAnalyzer
from mpde_bo.optimizer import BOResult, MPDEBOOptimizer
from mpde_bo.parameter_classifier import ClassificationConfig, ParameterClassifier


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def bounds_2d() -> Tensor:
    return torch.stack([torch.zeros(2), torch.ones(2)])


@pytest.fixture
def optimizer_2d(bounds_2d) -> MPDEBOOptimizer:
    """2 次元探索空間の MPDEBOOptimizer (テスト高速化設定)"""
    gp_manager = GPModelManager(GPConfig(kernel="matern52"))
    analyzer = ImportanceAnalyzer()
    classifier = ParameterClassifier(
        analyzer=analyzer,
        config=ClassificationConfig(eps_l=1.0, eps_e=0.01),
    )
    acq_optimizer = AcquisitionOptimizer(
        config=AcquisitionConfig(type="EI", num_restarts=2, raw_samples=16),
        bounds=bounds_2d,
    )
    return MPDEBOOptimizer(
        gp_manager=gp_manager,
        classifier=classifier,
        acq_optimizer=acq_optimizer,
        bounds=bounds_2d,
    )


@pytest.fixture
def init_data_2d(simple_train_data) -> tuple[Tensor, Tensor]:
    return simple_train_data  # shape (10, 2), (10, 1)


def quadratic(x: Tensor) -> float:
    """f(x) = -(x[0]-0.5)^2 - (x[1]-0.5)^2, 最大値 0.0 at (0.5, 0.5)"""
    return (-(x[0] - 0.5) ** 2 - (x[1] - 0.5) ** 2).item()


# ── optimize ──────────────────────────────────────────────────────────────────

class TestOptimize:
    def test_returns_bo_result(self, optimizer_2d, init_data_2d):
        """10-1: 返り値が BOResult インスタンスである"""
        X, Y = init_data_2d
        result = optimizer_2d.optimize(quadratic, T=2, train_X=X, train_Y=Y)
        assert isinstance(result, BOResult)

    def test_total_observation_count(self, optimizer_2d, init_data_2d):
        """10-2: 評価回数が n0 + T になる"""
        X, Y = init_data_2d
        n0 = X.shape[0]
        T = 3
        result = optimizer_2d.optimize(quadratic, T=T, train_X=X, train_Y=Y)
        assert result.train_X.shape[0] == n0 + T

    def test_train_y_shape(self, optimizer_2d, init_data_2d):
        """10-3: train_Y の shape が (n0+T, 1) である"""
        X, Y = init_data_2d
        n0 = X.shape[0]
        T = 3
        result = optimizer_2d.optimize(quadratic, T=T, train_X=X, train_Y=Y)
        assert result.train_Y.shape == (n0 + T, 1)

    def test_best_x_within_bounds(self, optimizer_2d, init_data_2d, bounds_2d):
        """10-4 [Property]: best_x が探索境界内にある"""
        X, Y = init_data_2d
        result = optimizer_2d.optimize(quadratic, T=2, train_X=X, train_Y=Y)
        assert (result.best_x >= bounds_2d[0]).all()
        assert (result.best_x <= bounds_2d[1]).all()

    def test_best_y_equals_train_y_max(self, optimizer_2d, init_data_2d):
        """10-5: best_y が train_Y 全体の最大値と一致する"""
        X, Y = init_data_2d
        result = optimizer_2d.optimize(quadratic, T=2, train_X=X, train_Y=Y)
        assert result.best_y == pytest.approx(result.train_Y.max().item(), abs=1e-8)

    def test_callback_called_t_times(self, optimizer_2d, init_data_2d):
        """10-6: callback が T 回呼ばれる"""
        X, Y = init_data_2d
        call_count = {"n": 0}

        def callback(t, model, classification):
            call_count["n"] += 1

        T = 4
        optimizer_2d.optimize(quadratic, T=T, train_X=X, train_Y=Y, callback=callback)
        assert call_count["n"] == T

    @pytest.mark.integration
    def test_improvement_over_initial(self, optimizer_2d, init_data_2d):
        """10-7 [Integration]: 単峰性関数で初期値より改善される"""
        X, Y = init_data_2d
        initial_best = Y.max().item()
        result = optimizer_2d.optimize(quadratic, T=15, train_X=X, train_Y=Y)
        assert result.best_y > initial_best
