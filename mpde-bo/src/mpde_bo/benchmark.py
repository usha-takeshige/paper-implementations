"""論文のベンチマーク目的関数を提供するモジュール (method.md §7)。"""

import torch
from torch import Tensor

# method.md §7 末尾のデフォルトパラメータ
_DEFAULT_A = (0.3, 1.2, 0.6, 0.7, 0.7)
_DEFAULT_SIGMA = (20.0, 5.0, 5.0, 6.0, 6.0)
_MAX_PLACEMENT_TRIALS = 10_000


def _gaussian(x: Tensor, mu: Tensor, sigma: float) -> Tensor:
    """多次元等方ガウス関数の値を返す。exp(-||x-μ||^2 / (2σ^2))"""
    return torch.exp(-((x - mu).pow(2).sum()) / (2.0 * sigma ** 2))


class BenchmarkFunction:
    """method.md §7 に記述された多峰性モデル目的関数。

    以下の分解で構成される。

    .. code-block:: text

        f(x) = f_d(x_d) + f_s(x_s)

        f_d(x_d) = a_0 * N(x_d; μ_0, σ_0^2 I)
                 + Σ_{i=1}^{d} a_i * N(x_d; μ_i, σ_i^2 I)

        f_s(x_s) = a_s * N(x_s; μ_s, σ_s^2 I),  μ_s = M/2

    ここで N(x; μ, σ^2 I) = exp(-||x-μ||^2 / (2σ^2)) はガウスカーネル関数。

    ピーク位置 μ_i は分離制約 ``‖μ_i - μ_j‖ > max{2σ_i, 2σ_j}`` を
    満たすようにランダムに配置される。

    Args:
        n_important: 重要パラメータ数 d。
        n_unimportant: 非重要パラメータ数 s。
        grid_size: 探索空間の幅 M。各次元の範囲は ``[0, M]``。
        a: 重要次元のピーク高さ ``(a_0, ..., a_d)``。
           None のとき論文デフォルト値を使用。
        sigma: 重要次元のピーク幅 ``(σ_0, ..., σ_d)``。
               None のとき論文デフォルト値を使用。
        a_s: 非重要次元のピーク高さ。
        sigma_s: 非重要次元のピーク幅。
        generator: 再現性のための乱数生成器。
    """

    def __init__(
        self,
        n_important: int,
        n_unimportant: int,
        grid_size: int = 100,
        a: tuple[float, ...] | None = None,
        sigma: tuple[float, ...] | None = None,
        a_s: float = 0.1,
        sigma_s: float = 25.0,
        generator: torch.Generator | None = None,
    ) -> None:
        self._n_important = n_important
        self._n_unimportant = n_unimportant
        self._grid_size = grid_size
        self._a_s = a_s
        self._sigma_s = sigma_s

        n_peaks = n_important + 1  # ベースピーク(a_0) + 各重要次元のピーク
        a_vals = list((_DEFAULT_A[:n_peaks] if a is None else a))
        s_vals = list((_DEFAULT_SIGMA[:n_peaks] if sigma is None else sigma))

        self._a: list[float] = a_vals
        self._sigma_vals: list[float] = s_vals

        # ピーク位置をランダム配置（分離制約あり）
        gen = generator
        M = float(grid_size)
        self._peak_centers: list[Tensor] = self._place_peaks(
            n_peaks, n_important, M, s_vals, gen
        )
        # 非重要次元のピーク位置（中央固定）
        self._mu_s = torch.full((n_unimportant,), M / 2.0, dtype=torch.double)

    # ── public properties ──────────────────────────────────────────────────

    @property
    def n_important(self) -> int:
        """重要パラメータ数。"""
        return self._n_important

    @property
    def n_unimportant(self) -> int:
        """非重要パラメータ数。"""
        return self._n_unimportant

    @property
    def grid_size(self) -> int:
        """探索空間の幅 M。"""
        return self._grid_size

    @property
    def optimal_value(self) -> float:
        """f の近似最大値。各ピーク中心での f 値の最大として計算する。"""
        best = -float("inf")
        for mu in self._peak_centers:
            x = torch.cat([mu, self._mu_s])
            val = self(x)
            if val > best:
                best = val
        return best

    @property
    def peak_centers(self) -> list[Tensor]:
        """重要次元のピーク位置リスト (長さ n_important+1)。"""
        return self._peak_centers

    @property
    def peak_widths(self) -> list[float]:
        """重要次元のピーク幅リスト (長さ n_important+1)。"""
        return self._sigma_vals

    # ── callable ───────────────────────────────────────────────────────────

    def __call__(self, x: Tensor) -> float:
        """目的関数を評価する。x の shape は ``(N,)`` でなければならない。"""
        if x.ndim != 1:
            raise ValueError(
                f"x must be 1-D, got shape {tuple(x.shape)}. "
                "For batch evaluation, call in a loop."
            )
        x_d = x[: self._n_important]
        x_s = x[self._n_important :]
        return (self._f_d(x_d) + self._f_s(x_s)).item()

    # ── private helpers ────────────────────────────────────────────────────

    def _f_d(self, x_d: Tensor) -> Tensor:
        """重要パラメータ部分の関数値。"""
        val = torch.zeros(1, dtype=torch.double)
        for a_i, mu_i, sigma_i in zip(self._a, self._peak_centers, self._sigma_vals):
            val = val + a_i * _gaussian(x_d, mu_i, sigma_i)
        return val

    def _f_s(self, x_s: Tensor) -> Tensor:
        """非重要パラメータ部分の関数値。"""
        return self._a_s * _gaussian(x_s, self._mu_s, self._sigma_s)

    @staticmethod
    def _place_peaks(
        n_peaks: int,
        n_dims: int,
        M: float,
        sigmas: list[float],
        gen: torch.Generator | None,
    ) -> list[Tensor]:
        """分離制約を満たすようにピーク位置をランダム配置する。"""
        placed: list[Tensor] = []
        for i in range(n_peaks):
            for _ in range(_MAX_PLACEMENT_TRIALS):
                mu = torch.rand(n_dims, dtype=torch.double, generator=gen) * M
                ok = all(
                    torch.norm(mu - other).item() > max(2 * sigmas[i], 2 * sigmas[j])
                    for j, other in enumerate(placed)
                )
                if ok:
                    placed.append(mu)
                    break
            else:
                # 制約を緩めて中央付近に配置（フォールバック）
                placed.append(torch.rand(n_dims, dtype=torch.double, generator=gen) * M)
        return placed
