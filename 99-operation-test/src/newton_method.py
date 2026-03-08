"""ニュートン法による数値微分モジュール。

離散データの数値微分（NumericalDifferentiator）と、
ニュートン法による方程式の求根（NewtonSolver）を提供する。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class NumericalDifferentiator:
    """有限差分法による数値微分クラス。

    離散データ点 (x, y) に対して前進差分・後退差分・中心差分のいずれかで
    数値微分を計算する。中心差分は O(h^2) の精度を持ち最も高精度。

    Parameters
    ----------
    method : str
        差分法の種類。"forward", "backward", "central" のいずれか。

    Raises
    ------
    ValueError
        method が上記3種以外の場合。
    """

    _VALID_METHODS = {"forward", "backward", "central"}

    def __init__(self, method: str = "central") -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method は 'forward', 'backward', 'central' のいずれかを指定してください。"
                f" 受け取った値: {method!r}"
            )
        self.method = method

    def differentiate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """数値微分を計算する。

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            独立変数の配列。
        y : np.ndarray, shape (n,)
            従属変数の配列。

        Returns
        -------
        dy_dx : np.ndarray, shape (n,)
            各点における数値微分値。

        Raises
        ------
        ValueError
            x と y の長さが異なる場合。
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError(
                f"x と y の配列長が一致しません: x.shape={x.shape}, y.shape={y.shape}"
            )

        n = len(x)
        dy_dx = np.empty(n)

        if self.method == "forward":
            dy_dx[:-1] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            dy_dx[-1] = dy_dx[-2]  # 端点は一つ前の値で補完
        elif self.method == "backward":
            dy_dx[1:] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            dy_dx[0] = dy_dx[1]  # 端点は一つ後の値で補完
        else:  # central
            dy_dx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
            dy_dx[0] = (y[1] - y[0]) / (x[1] - x[0])        # 先端は前進差分
            dy_dx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])   # 末端は後退差分

        return dy_dx


class NewtonSolver:
    """ニュートン法による方程式の求根クラス。

    f(x) = 0 の根を反復的に求める。解析的微分が与えられない場合は
    中心差分による数値微分を使用する。

    Parameters
    ----------
    tol : float
        収束判定の許容誤差 ε。|Δx| < tol で収束とみなす。
    max_iter : int
        最大反復回数。
    h : float
        数値微分に使用するステップ幅。
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 100,
        h: float = 1e-5,
    ) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.h = h

    def solve(
        self,
        f: Callable[[float], float],
        x0: float,
        f_prime: Callable[[float], float] | None = None,
    ) -> float:
        """ニュートン法で f(x) = 0 の根を求める。

        Parameters
        ----------
        f : Callable
            根を求める関数。
        x0 : float
            初期値。
        f_prime : Callable or None
            解析的微分。None の場合は中心差分による数値微分を使用。

        Returns
        -------
        x : float
            収束した根の近似値。

        Raises
        ------
        ZeroDivisionError
            微分値がゼロになった場合。
        RuntimeError
            最大反復回数内に収束しなかった場合。
        """
        x = float(x0)

        for _ in range(self.max_iter):
            f_val = f(x)

            if f_prime is not None:
                f_grad = f_prime(x)
            else:
                f_grad = (f(x + self.h) - f(x - self.h)) / (2 * self.h)

            if f_grad == 0.0:
                raise ZeroDivisionError(
                    f"微分値がゼロになりました (x={x})。多重根の可能性があります。"
                )

            delta = f_val / f_grad
            x -= delta

            if abs(delta) < self.tol:
                return x

        raise RuntimeError(
            f"最大反復回数 {self.max_iter} 回以内に収束しませんでした。"
            f" 最終値: x={x}, f(x)={f(x)}"
        )
