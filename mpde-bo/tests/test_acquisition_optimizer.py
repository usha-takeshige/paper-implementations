"""AcquisitionOptimizer のテスト (test_cases.md §4, テスト 9-1〜9-7)"""

import pytest
import torch
from torch import Tensor

from mpde_bo.acquisition_optimizer import AcquisitionConfig, AcquisitionOptimizer
from mpde_bo.gp_model_manager import GPConfig, GPModelManager


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def manager() -> GPModelManager:
    return GPModelManager(GPConfig(kernel="matern52"))


@pytest.fixture
def model_and_train_y(manager, simple_train_data):
    """2 次元の学習済みモデルと train_Y"""
    X, Y = simple_train_data
    model = manager.build(X, Y)
    return model, Y


@pytest.fixture
def bounds_2d() -> Tensor:
    return torch.stack([torch.zeros(2), torch.ones(2)])


def make_optimizer(acquisition: str = "EI", **kwargs) -> AcquisitionOptimizer:
    config = AcquisitionConfig(
        type=acquisition,
        num_restarts=2,   # テスト高速化のため最小値
        raw_samples=16,
        **kwargs,
    )
    return AcquisitionOptimizer(config=config, bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double))


# ── maximize ──────────────────────────────────────────────────────────────────

class TestMaximize:
    def test_output_shape(self, model_and_train_y, bounds_2d):
        """9-1: 候補点の shape が (N,) = (2,) である"""
        model, train_Y = model_and_train_y
        optimizer = make_optimizer("EI")
        candidate = optimizer.maximize(model, train_Y, fixed_features={})
        assert candidate.shape == (2,)

    def test_candidate_within_bounds(self, model_and_train_y, bounds_2d):
        """9-2 [Property]: 候補点が探索境界 [0, 1]^2 の内側にある"""
        model, train_Y = model_and_train_y
        optimizer = make_optimizer("EI")
        candidate = optimizer.maximize(model, train_Y, fixed_features={})
        assert (candidate >= 0.0).all()
        assert (candidate <= 1.0).all()

    def test_fixed_features_are_respected(self, model_and_train_y):
        """9-3: fixed_features で指定した次元の値が候補点に保持される"""
        model, train_Y = model_and_train_y
        fixed = {1: 0.5}
        optimizer = make_optimizer("EI")
        candidate = optimizer.maximize(model, train_Y, fixed_features=fixed)
        assert candidate[1].item() == pytest.approx(0.5, abs=1e-6)

    def test_ei_runs(self, model_and_train_y):
        """9-4: EI 獲得関数で正常に候補点が返る"""
        model, train_Y = model_and_train_y
        optimizer = make_optimizer("EI")
        candidate = optimizer.maximize(model, train_Y, fixed_features={})
        assert candidate is not None

    def test_ucb_runs(self, model_and_train_y):
        """9-5: UCB 獲得関数で正常に候補点が返る"""
        model, train_Y = model_and_train_y
        optimizer = make_optimizer("UCB", ucb_beta=2.0)
        candidate = optimizer.maximize(model, train_Y, fixed_features={})
        assert candidate is not None

    def test_pi_runs(self, model_and_train_y):
        """9-6: PI 獲得関数で正常に候補点が返る"""
        model, train_Y = model_and_train_y
        optimizer = make_optimizer("PI")
        candidate = optimizer.maximize(model, train_Y, fixed_features={})
        assert candidate is not None

    def test_invalid_acquisition_raises(self):
        """9-7: 不正な獲得関数名で ValueError が送出される"""
        with pytest.raises(ValueError):
            AcquisitionConfig(type="INVALID")
