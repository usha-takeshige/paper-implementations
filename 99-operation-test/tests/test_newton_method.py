import numpy as np
import pytest

from newton_method import NumericalDifferentiator, NewtonSolver


# ---------------------------------------------------------------------------
# TC-ND: NumericalDifferentiator
# ---------------------------------------------------------------------------


def test_central_diff_sin():
    """TC-ND-01: 中心差分による sin(x) の微分が cos(x) に近いことを確認する。"""
    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x)
    diff = NumericalDifferentiator(method="central")
    dy_dx = diff.differentiate(x, y)
    np.testing.assert_allclose(dy_dx[1:-1], np.cos(x)[1:-1], atol=1e-4)


def test_forward_diff_sin():
    """TC-ND-02: 前進差分による sin(x) の微分が cos(x) に近いことを確認する。"""
    x = np.linspace(0, 2 * np.pi, 10000)
    y = np.sin(x)
    diff = NumericalDifferentiator(method="forward")
    dy_dx = diff.differentiate(x, y)
    np.testing.assert_allclose(dy_dx[:-1], np.cos(x)[:-1], atol=1e-3)


def test_backward_diff_sin():
    """TC-ND-03: 後退差分による sin(x) の微分が cos(x) に近いことを確認する。"""
    x = np.linspace(0, 2 * np.pi, 10000)
    y = np.sin(x)
    diff = NumericalDifferentiator(method="backward")
    dy_dx = diff.differentiate(x, y)
    np.testing.assert_allclose(dy_dx[1:], np.cos(x)[1:], atol=1e-3)


def test_linear_function():
    """TC-ND-04: 線形関数 f(x) = 3x + 2 の微分が全点で 3.0 になることを確認する。"""
    x = np.linspace(-5, 5, 100)
    y = 3 * x + 2
    diff = NumericalDifferentiator(method="central")
    dy_dx = diff.differentiate(x, y)
    np.testing.assert_allclose(dy_dx, 3.0, atol=1e-10)


def test_central_diff_more_accurate():
    """TC-ND-05: 中心差分の誤差が前進・後退差分より小さいことを確認する。"""
    x = np.linspace(0.1, np.pi - 0.1, 50)
    y = np.sin(x)
    dy_true = np.cos(x)

    central = NumericalDifferentiator(method="central")
    forward = NumericalDifferentiator(method="forward")

    err_central = np.max(np.abs(central.differentiate(x, y) - dy_true))
    err_forward = np.max(np.abs(forward.differentiate(x, y)[:-1] - dy_true[:-1]))

    assert err_central < err_forward


def test_invalid_method():
    """TC-ND-06: 不正な method 指定で ValueError が送出されることを確認する。"""
    with pytest.raises(ValueError):
        NumericalDifferentiator(method="invalid")


def test_mismatched_input_lengths():
    """TC-ND-07: x と y の配列長が異なる場合に ValueError が送出されることを確認する。"""
    diff = NumericalDifferentiator(method="central")
    with pytest.raises(ValueError):
        diff.differentiate(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# TC-NS: NewtonSolver
# ---------------------------------------------------------------------------


def test_sqrt2_with_analytical_derivative():
    """TC-NS-01: 解析的微分を与えたとき √2 を高精度で求めることを確認する。"""
    solver = NewtonSolver()
    root = solver.solve(
        f=lambda x: x**2 - 2,
        x0=1.0,
        f_prime=lambda x: 2 * x,
    )
    np.testing.assert_allclose(root, np.sqrt(2), rtol=1e-8)


def test_sqrt2_with_numerical_derivative():
    """TC-NS-02: 数値微分のみで √2 が求まることを確認する。"""
    solver = NewtonSolver()
    root = solver.solve(f=lambda x: x**2 - 2, x0=1.0)
    np.testing.assert_allclose(root, np.sqrt(2), rtol=1e-6)


def test_cubic_equation():
    """TC-NS-03: 3次方程式 x^3 - x - 2 = 0 の根を求めることを確認する。"""
    solver = NewtonSolver()
    root = solver.solve(f=lambda x: x**3 - x - 2, x0=1.5)
    np.testing.assert_allclose(root**3 - root - 2, 0.0, atol=1e-8)


def test_convergence_from_negative_x0():
    """TC-NS-04: 負の初期値 x0=-1.0 から x^2 - 4 = 0 の根 -2 に収束することを確認する。"""
    solver = NewtonSolver()
    root = solver.solve(f=lambda x: x**2 - 4, x0=-1.0)
    np.testing.assert_allclose(root, -2.0, atol=1e-8)


def test_no_convergence_raises_runtime_error():
    """TC-NS-05: 最大反復回数内に収束しない場合に RuntimeError が送出されることを確認する。"""
    solver = NewtonSolver(max_iter=5)
    with pytest.raises(RuntimeError):
        solver.solve(f=lambda x: x**3 - 2 * x + 2, x0=0.0)


def test_zero_derivative_raises():
    """TC-NS-06: 微分値がゼロの場合に ZeroDivisionError が送出されることを確認する。"""
    solver = NewtonSolver()
    with pytest.raises(ZeroDivisionError):
        solver.solve(
            f=lambda x: x**2,
            x0=0.0,
            f_prime=lambda x: 2 * x,
        )


def test_tolerance_control():
    """TC-NS-07: tol を小さくするほど根の精度が高くなることを確認する。"""
    f = lambda x: x**2 - 2
    x0 = 1.0
    solver_loose = NewtonSolver(tol=1e-3)
    solver_tight = NewtonSolver(tol=1e-10)

    err_loose = abs(solver_loose.solve(f, x0) - np.sqrt(2))
    err_tight = abs(solver_tight.solve(f, x0) - np.sqrt(2))

    assert err_tight < err_loose
