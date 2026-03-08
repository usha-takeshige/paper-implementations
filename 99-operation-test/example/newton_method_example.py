"""ニュートン法の使用例。

NewtonSolver と NumericalDifferentiator の基本的な使い方を示す。

実行方法::

    uv run python -m example.newton_method_example
"""

import math

import numpy as np

from src.newton_method import NumericalDifferentiator, NewtonSolver


# ── 例1: 解析的微分ありで √2 を求める ─────────────────────────────────────
# f(x) = x^2 - 2 = 0  →  x = √2
solver = NewtonSolver(tol=1e-10)

root = solver.solve(
    f=lambda x: x**2 - 2,
    x0=1.5,
    f_prime=lambda x: 2 * x,
)
print(f"例1 解析的微分あり: √2 ≈ {root:.10f}  (math.sqrt(2) = {math.sqrt(2):.10f})")


# ── 例2: 数値微分のみで √2 を求める ──────────────────────────────────────
root_num = solver.solve(
    f=lambda x: x**2 - 2,
    x0=1.5,
)
print(f"例2 数値微分のみ:   √2 ≈ {root_num:.10f}")


# ── 例3: 三角関数の根 (cos(x) = x) ───────────────────────────────────────
# g(x) = cos(x) - x = 0  →  不動点 x ≈ 0.739085
root_cos = solver.solve(
    f=lambda x: math.cos(x) - x,
    x0=1.0,
    f_prime=lambda x: -math.sin(x) - 1,
)
print(f"例3 cos(x) = x の根: x ≈ {root_cos:.10f}")


# ── 例4: 数値微分（NumericalDifferentiator）で sin(x) を微分 ──────────────
x = np.linspace(0, 2 * math.pi, 9)
y = np.sin(x)

for method in ("forward", "backward", "central"):
    diff = NumericalDifferentiator(method=method)
    dy_dx = diff.differentiate(x, y)
    cos_x = np.cos(x)
    mae = np.mean(np.abs(dy_dx - cos_x))
    print(f"例4 {method:8s} 差分: MAE vs cos(x) = {mae:.6f}")
