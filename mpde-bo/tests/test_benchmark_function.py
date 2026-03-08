"""BenchmarkFunction のテスト (test_cases.md §6, テスト 11-1〜13-3)"""

import math

import pytest
import torch
from torch import Tensor

from mpde_bo.benchmark import BenchmarkFunction


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def func_default(generator) -> BenchmarkFunction:
    """論文デフォルトパラメータ: d=2 重要, s=8 非重要, M=100"""
    return BenchmarkFunction(
        n_important=2,
        n_unimportant=8,
        grid_size=100,
        generator=generator,
    )


@pytest.fixture
def func_small(generator) -> BenchmarkFunction:
    """小規模テスト用: d=1 重要, s=1 非重要, M=20"""
    return BenchmarkFunction(
        n_important=1,
        n_unimportant=1,
        grid_size=20,
        generator=generator,
    )


# ── __call__ ──────────────────────────────────────────────────────────────────

class TestCall:
    def test_returns_float(self, func_default, generator):
        """11-1: スカラー (float) が返る"""
        N = 2 + 8  # n_important + n_unimportant
        x = torch.rand(N, dtype=torch.double, generator=generator)
        result = func_default(x)
        assert isinstance(result, float)

    def test_batch_input_raises(self, func_default, generator):
        """11-2: バッチ入力 (2D Tensor) で ValueError が送出される"""
        N = 2 + 8
        x_batch = torch.rand(3, N, dtype=torch.double, generator=generator)
        with pytest.raises(ValueError):
            func_default(x_batch)

    def test_finite_value_within_domain(self, func_default):
        """11-3 [Property]: 定義域内 [0, M]^N の点で有限値が返る"""
        N = 2 + 8
        M = 100
        gen = torch.Generator().manual_seed(99)
        for _ in range(10):
            x = torch.rand(N, dtype=torch.double, generator=gen) * M
            result = func_default(x)
            assert math.isfinite(result)


# ── optimal_value ──────────────────────────────────────────────────────────────

class TestOptimalValue:
    def test_nonnegative(self, func_default):
        """12-1 [Property]: optimal_value が非負である"""
        assert func_default.optimal_value >= 0.0

    @pytest.mark.integration
    def test_matches_grid_search(self, func_small):
        """12-2 [Integration]: グリッド全探索の最大値と optimal_value が近似一致する
        (d=1, M=20 の小規模関数で確認)
        """
        M = func_small.grid_size
        N = func_small.n_important + func_small.n_unimportant
        # 粗いグリッドで最大値を探索
        grid = torch.linspace(0.0, float(M), steps=50, dtype=torch.double)
        coords = torch.meshgrid(*[grid] * N, indexing="ij")
        points = torch.stack([c.flatten() for c in coords], dim=-1)
        grid_max = max(func_small(p) for p in points)
        assert func_small.optimal_value == pytest.approx(grid_max, rel=0.05)


# ── コンストラクタ (ピーク分離制約) ────────────────────────────────────────────

class TestConstructor:
    def test_peak_separation_constraint(self):
        """13-1 [Property]: 全ピーク対が分離制約 ‖μ_i - μ_j‖ > max{2σ_i, 2σ_j} を満たす"""
        gen = torch.Generator().manual_seed(5)
        func = BenchmarkFunction(n_important=2, n_unimportant=0, grid_size=100, generator=gen)
        # 内部のピーク位置・幅を公開プロパティ経由で取得
        mus = func.peak_centers    # list of Tensor, len = d+1
        sigmas = func.peak_widths  # list of float, len = d+1

        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                dist = torch.norm(mus[i] - mus[j]).item()
                min_sep = max(2 * sigmas[i], 2 * sigmas[j])
                assert dist > min_sep, (
                    f"Peaks {i} and {j} violate separation: dist={dist:.3f}, min_sep={min_sep:.3f}"
                )

    def test_reproducibility(self):
        """13-2: 同一 generator シードで同一の optimal_value が得られる"""
        gen_a = torch.Generator().manual_seed(77)
        gen_b = torch.Generator().manual_seed(77)
        func_a = BenchmarkFunction(n_important=2, n_unimportant=3, grid_size=50, generator=gen_a)
        func_b = BenchmarkFunction(n_important=2, n_unimportant=3, grid_size=50, generator=gen_b)
        assert func_a.optimal_value == pytest.approx(func_b.optimal_value, abs=1e-8)

    def test_default_params_instantiate(self):
        """13-3: デフォルトパラメータでエラーなくインスタンス生成できる"""
        func = BenchmarkFunction(n_important=2, n_unimportant=8)
        assert func is not None
