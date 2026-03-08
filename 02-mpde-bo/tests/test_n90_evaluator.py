"""N90Evaluator のテスト (test_cases.md §7, テスト 14-1〜14-4)"""

import pytest
import torch
from torch import Tensor

from mpde_bo.benchmark import BenchmarkFunction
from mpde_bo.evaluator import N90Evaluator
from mpde_bo.optimizer import BOResult


# ── ヘルパー: テスト用アルゴリズム ─────────────────────────────────────────────

def random_algorithm(
    f: BenchmarkFunction,
    T: int,
    train_X: Tensor,
    train_Y: Tensor,
) -> BOResult:
    """ランダムサンプリングのみを行う比較用アルゴリズム"""
    N = train_X.shape[-1]
    M = float(f.grid_size)
    gen = torch.Generator().manual_seed(0)

    all_X = train_X.clone()
    all_Y = train_Y.clone()

    for _ in range(T):
        x_new = torch.rand(N, dtype=torch.double, generator=gen) * M
        y_new = torch.tensor([[f(x_new)]], dtype=torch.double)
        all_X = torch.cat([all_X, x_new.unsqueeze(0)], dim=0)
        all_Y = torch.cat([all_Y, y_new], dim=0)

    best_idx = all_Y.argmax().item()
    return BOResult(
        best_x=all_X[best_idx],
        best_y=all_Y.max().item(),
        train_X=all_X,
        train_Y=all_Y,
    )


def oracle_algorithm(
    f: BenchmarkFunction,
    T: int,
    train_X: Tensor,
    train_Y: Tensor,
) -> BOResult:
    """最初の 1 回で最適点を評価するオラクルアルゴリズム"""
    N = train_X.shape[-1]
    M = float(f.grid_size)

    # 粗いグリッドで近似最適点を探す (テスト用なので精度は問わない)
    gen = torch.Generator().manual_seed(42)
    candidates = torch.rand(200, N, dtype=torch.double, generator=gen) * M
    best_x = max(candidates, key=lambda x: f(x))
    best_y_val = f(best_x)

    all_X = torch.cat([train_X, best_x.unsqueeze(0)], dim=0)
    all_Y = torch.cat([
        train_Y,
        torch.tensor([[best_y_val]], dtype=torch.double),
    ], dim=0)

    return BOResult(
        best_x=best_x,
        best_y=all_Y.max().item(),
        train_X=all_X,
        train_Y=all_Y,
    )


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_evaluator() -> N90Evaluator:
    """テスト高速化のため n_functions=5, budget=10 の小規模 N90Evaluator"""
    return N90Evaluator(
        budget=10,
        n_functions=5,
        threshold_ratio=0.9,
        n_important=1,
        n_unimportant=1,
        grid_size=20,
        generator=torch.Generator().manual_seed(0),
    )


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_positive_finite_value(self, small_evaluator):
        """14-1 [Property]: N90 が 0 より大きく budget 以下の有限値である"""
        n90 = small_evaluator.evaluate(random_algorithm)
        assert 0 < n90 <= small_evaluator.budget

    @pytest.mark.integration
    def test_oracle_has_small_n90(self, small_evaluator):
        """14-2 [Integration]: オラクルアルゴリズムの N90 が初期データ数以下である"""
        n0 = 3  # small_evaluator 内の初期観測数
        n90 = small_evaluator.evaluate(oracle_algorithm)
        assert n90 <= n0 + 1  # 1 回目の追加評価で達成

    @pytest.mark.integration
    def test_random_has_larger_n90_than_oracle(self, small_evaluator):
        """14-3 [Integration]: ランダムアルゴリズムの N90 > オラクルの N90"""
        n90_random = small_evaluator.evaluate(random_algorithm)
        n90_oracle = small_evaluator.evaluate(oracle_algorithm)
        assert n90_random >= n90_oracle

    def test_reproducibility(self):
        """14-4: 同一シードで同一の N90 スコアが返る"""
        ev_a = N90Evaluator(
            budget=8,
            n_functions=3,
            n_important=1,
            n_unimportant=1,
            grid_size=20,
            generator=torch.Generator().manual_seed(11),
        )
        ev_b = N90Evaluator(
            budget=8,
            n_functions=3,
            n_important=1,
            n_unimportant=1,
            grid_size=20,
            generator=torch.Generator().manual_seed(11),
        )
        assert ev_a.evaluate(random_algorithm) == pytest.approx(
            ev_b.evaluate(random_algorithm), abs=1e-8
        )
