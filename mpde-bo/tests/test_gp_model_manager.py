"""GPModelManager のテスト (test_cases.md §1, テスト 1-1〜4-3)"""

import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RBFKernel
from torch import Tensor

from mpde_bo.gp_model_manager import GPConfig, GPModelManager


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def manager_matern() -> GPModelManager:
    return GPModelManager(GPConfig(kernel="matern52"))


@pytest.fixture
def manager_rbf() -> GPModelManager:
    return GPModelManager(GPConfig(kernel="rbf"))


@pytest.fixture
def train_data_3d() -> tuple[Tensor, Tensor]:
    """3 次元入力、n=5 の最小フィクスチャ"""
    gen = torch.Generator().manual_seed(10)
    X = torch.rand(5, 3, dtype=torch.double, generator=gen)
    Y = X.sum(dim=-1, keepdim=True)
    return X, Y


# ── build ──────────────────────────────────────────────────────────────────────

class TestBuild:
    def test_returns_singletaskgp(self, manager_matern, train_data_3d):
        """1-1: SingleTaskGP インスタンスが返る"""
        X, Y = train_data_3d
        model = manager_matern.build(X, Y)
        assert isinstance(model, SingleTaskGP)

    def test_model_is_in_eval_mode(self, manager_matern, train_data_3d):
        """1-2: 返り値のモデルが eval モードである"""
        X, Y = train_data_3d
        model = manager_matern.build(X, Y)
        assert not model.training

    def test_matern52_kernel(self, manager_matern, train_data_3d):
        """1-3: matern52 指定時に MaternKernel(nu=2.5) が使われる"""
        X, Y = train_data_3d
        model = manager_matern.build(X, Y)
        base_kernel = model.covar_module.base_kernel
        assert isinstance(base_kernel, MaternKernel)
        assert base_kernel.nu == pytest.approx(2.5)

    def test_rbf_kernel(self, manager_rbf, train_data_3d):
        """1-4: rbf 指定時に RBFKernel が使われる"""
        X, Y = train_data_3d
        model = manager_rbf.build(X, Y)
        base_kernel = model.covar_module.base_kernel
        assert isinstance(base_kernel, RBFKernel)

    def test_invalid_kernel_raises(self, train_data_3d):
        """1-5: 不正なカーネル名で ValueError が送出される"""
        X, Y = train_data_3d
        manager = GPModelManager(GPConfig(kernel="unknown"))
        with pytest.raises(ValueError):
            manager.build(X, Y)

    def test_ard_lengthscale_shape(self, manager_matern, train_data_3d):
        """1-6: ARD 長さスケールの shape が (1, N) である"""
        X, Y = train_data_3d
        N = X.shape[-1]
        model = manager_matern.build(X, Y)
        ls = model.covar_module.base_kernel.lengthscale
        assert ls.shape == (1, N)


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_observation_count_increases(
        self, manager_matern, simple_train_data, train_data_3d
    ):
        """2-1: 更新後のモデルの train_X 行数が元 + 追加分になる"""
        # simple_train_data は conftest 経由で N=2 なので train_data_3d を使う
        gen = torch.Generator().manual_seed(20)
        X = torch.rand(8, 3, dtype=torch.double, generator=gen)
        Y = X.sum(dim=-1, keepdim=True)
        X_new = torch.rand(2, 3, dtype=torch.double, generator=gen)
        Y_new = X_new.sum(dim=-1, keepdim=True)

        model = manager_matern.build(X, Y)
        updated = manager_matern.update(model, X_new, Y_new)
        assert updated.train_inputs[0].shape[0] == 10

    def test_returns_new_instance(self, manager_matern, train_data_3d):
        """2-2: update は元のモデルとは別のインスタンスを返す"""
        X, Y = train_data_3d
        gen = torch.Generator().manual_seed(21)
        X_new = torch.rand(1, 3, dtype=torch.double, generator=gen)
        Y_new = X_new.sum(dim=-1, keepdim=True)

        model = manager_matern.build(X, Y)
        updated = manager_matern.update(model, X_new, Y_new)
        assert updated is not model

    def test_updated_model_is_eval(self, manager_matern, train_data_3d):
        """2-3: 更新後のモデルが eval モードである"""
        X, Y = train_data_3d
        gen = torch.Generator().manual_seed(22)
        X_new = torch.rand(1, 3, dtype=torch.double, generator=gen)
        Y_new = X_new.sum(dim=-1, keepdim=True)

        model = manager_matern.build(X, Y)
        updated = manager_matern.update(model, X_new, Y_new)
        assert not updated.training


# ── get_length_scales ──────────────────────────────────────────────────────────

class TestGetLengthScales:
    def test_shape(self, manager_matern, train_data_3d):
        """3-1: shape が (N,) である"""
        X, Y = train_data_3d
        N = X.shape[-1]
        model = manager_matern.build(X, Y)
        ls = manager_matern.get_length_scales(model)
        assert ls.shape == (N,)

    def test_all_positive(self, manager_matern, train_data_3d):
        """3-2 [Property]: 全長さスケールが正の値である"""
        X, Y = train_data_3d
        model = manager_matern.build(X, Y)
        ls = manager_matern.get_length_scales(model)
        assert (ls > 0).all()

    def test_irrelevant_dims_have_larger_lengthscale(
        self, manager_matern, important_unimportant_data
    ):
        """3-3 [Property]: 目的値に無関係な次元は長さスケールが大きい
        (次元 0,1 が重要, 次元 2,3 が非重要)
        """
        X, Y = important_unimportant_data
        model = manager_matern.build(X, Y)
        ls = manager_matern.get_length_scales(model)
        # 非重要次元 (2, 3) の長さスケール平均 > 重要次元 (0, 1) の長さスケール平均
        important_mean = ls[[0, 1]].mean()
        unimportant_mean = ls[[2, 3]].mean()
        assert unimportant_mean > important_mean


# ── predict ────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_output_shapes(self, manager_matern, train_data_3d):
        """4-1: 事後平均・分散の shape が (m,) である"""
        X, Y = train_data_3d
        model = manager_matern.build(X, Y)
        gen = torch.Generator().manual_seed(30)
        X_test = torch.rand(10, 3, dtype=torch.double, generator=gen)
        mu, sigma2 = manager_matern.predict(model, X_test)
        assert mu.shape == (10,)
        assert sigma2.shape == (10,)

    def test_variance_nonnegative(self, manager_matern, train_data_3d):
        """4-2 [Property]: 事後分散が非負である"""
        X, Y = train_data_3d
        model = manager_matern.build(X, Y)
        gen = torch.Generator().manual_seed(31)
        X_test = torch.rand(20, 3, dtype=torch.double, generator=gen)
        _, sigma2 = manager_matern.predict(model, X_test)
        assert (sigma2 >= 0).all()

    def test_variance_small_at_training_points(self, manager_matern, train_data_3d):
        """4-3 [Property]: 訓練点での不確実性が非訓練点より小さい"""
        X_train, Y_train = train_data_3d
        model = manager_matern.build(X_train, Y_train)

        gen = torch.Generator().manual_seed(32)
        X_far = torch.rand(20, 3, dtype=torch.double, generator=gen) + 10.0  # 訓練範囲外

        _, sigma2_train = manager_matern.predict(model, X_train)
        _, sigma2_far = manager_matern.predict(model, X_far)
        assert sigma2_train.mean() < sigma2_far.mean()
