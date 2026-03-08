"""ImportanceAnalyzer のテスト (test_cases.md §2, テスト 5-1〜7-3)"""

import pytest
import torch
from torch import Tensor

from mpde_bo.gp_model_manager import GPConfig, GPModelManager
from mpde_bo.importance_analyzer import ImportanceAnalyzer


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer() -> ImportanceAnalyzer:
    return ImportanceAnalyzer()


@pytest.fixture
def manager() -> GPModelManager:
    return GPModelManager(GPConfig(kernel="matern52"))


@pytest.fixture
def model_and_data_important(
    manager, important_unimportant_data
):
    """次元 0,1 が重要な学習済みモデルとデータ"""
    X, Y = important_unimportant_data
    model = manager.build(X, Y)
    return model, X


@pytest.fixture
def model_and_data_simple(manager, simple_train_data):
    """simple_train_data の学習済みモデルとデータ"""
    X, Y = simple_train_data
    model = manager.build(X, Y)
    return model, X


@pytest.fixture
def x_s_grid_1d() -> Tensor:
    """1 次元グリッド, g=20 点"""
    return torch.linspace(0.0, 1.0, 20, dtype=torch.double).unsqueeze(-1)


# ── compute_ice ────────────────────────────────────────────────────────────────

class TestComputeICE:
    def test_output_shape(
        self, analyzer, model_and_data_simple, x_s_grid_1d
    ):
        """5-1: shape が (n, g) = (10, 20) である"""
        model, X = model_and_data_simple
        ice = analyzer.compute_ice(model, X, param_indices=[0], x_S_grid=x_s_grid_1d)
        assert ice.shape == (X.shape[0], x_s_grid_1d.shape[0])

    def test_irrelevant_dim_produces_flat_ice(
        self, analyzer, model_and_data_important, x_s_grid_1d
    ):
        """5-2 [Property]: 無関係な次元 (2) は ICE の行が均一に近い"""
        model, X = model_and_data_important
        ice = analyzer.compute_ice(model, X, param_indices=[2], x_S_grid=x_s_grid_1d)
        # 各行の max - min を計算し、平均が小さいことを確認
        row_range = (ice.max(dim=1).values - ice.min(dim=1).values).mean().item()
        assert row_range < 0.1

    def test_important_dim_produces_varying_ice(
        self, analyzer, model_and_data_important, x_s_grid_1d
    ):
        """5-3 [Property]: 重要な次元 (0) は少なくとも 1 行が変動する"""
        model, X = model_and_data_important
        ice = analyzer.compute_ice(model, X, param_indices=[0], x_S_grid=x_s_grid_1d)
        row_ranges = ice.max(dim=1).values - ice.min(dim=1).values
        assert row_ranges.max().item() > 0.01

    def test_auto_grid(self, analyzer, model_and_data_simple):
        """5-4: x_S_grid=None でも実行できる"""
        model, X = model_and_data_simple
        ice = analyzer.compute_ice(model, X, param_indices=[0], x_S_grid=None)
        assert ice.ndim == 2
        assert ice.shape[0] == X.shape[0]


# ── compute_mpde ───────────────────────────────────────────────────────────────

class TestComputeMPDE:
    def test_irrelevant_dim_mpde_near_zero(
        self, analyzer, model_and_data_important
    ):
        """6-1 [Property]: 無関係な次元 (2) の MPDE が 0 に近い"""
        model, X = model_and_data_important
        mpde = analyzer.compute_mpde(model, X, param_indices=[2])
        assert mpde == pytest.approx(0.0, abs=0.1)

    def test_important_dim_mpde_positive(
        self, analyzer, model_and_data_important
    ):
        """6-2 [Property]: 重要な次元 (0) の MPDE が正の値である"""
        model, X = model_and_data_important
        mpde = analyzer.compute_mpde(model, X, param_indices=[0])
        assert mpde > 0.01

    def test_returns_float(self, analyzer, model_and_data_simple):
        """6-3: スカラー (float) が返る"""
        model, X = model_and_data_simple
        mpde = analyzer.compute_mpde(model, X, param_indices=[0])
        assert isinstance(mpde, float)

    def test_mpde_geq_apde(self, analyzer, model_and_data_important):
        """6-4 [Property]: MPDE >= APDE (method.md §5 の主張)"""
        model, X = model_and_data_important
        for dim in [0, 1]:
            mpde = analyzer.compute_mpde(model, X, param_indices=[dim])
            apde = analyzer.compute_apde(model, X, param_indices=[dim])
            assert mpde >= apde - 1e-8  # 数値誤差の許容


# ── compute_apde ───────────────────────────────────────────────────────────────

class TestComputeAPDE:
    def test_returns_float(self, analyzer, model_and_data_simple):
        """7-1: スカラー (float) が返る"""
        model, X = model_and_data_simple
        apde = analyzer.compute_apde(model, X, param_indices=[0])
        assert isinstance(apde, float)

    def test_irrelevant_dim_apde_near_zero(
        self, analyzer, model_and_data_important
    ):
        """7-2 [Property]: 無関係な次元 (2) の APDE が 0 に近い"""
        model, X = model_and_data_important
        apde = analyzer.compute_apde(model, X, param_indices=[2])
        assert apde == pytest.approx(0.0, abs=0.1)

    def test_nonnegative(self, analyzer, model_and_data_simple):
        """7-3 [Property]: APDE が非負である"""
        model, X = model_and_data_simple
        for dim in range(X.shape[-1]):
            apde = analyzer.compute_apde(model, X, param_indices=[dim])
            assert apde >= 0.0
