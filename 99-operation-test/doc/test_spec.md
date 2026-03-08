# テスト仕様書：ニュートン法による数値微分

## 概要

`NumericalDifferentiator` および `NewtonSolver` クラスに対するテスト仕様。
アルゴリズム設計書（`newton_method.md`）に記載された仕様との整合性を担保する。

---

## テスト対象モジュール

| クラス | テスト対象メソッド |
|---|---|
| `NumericalDifferentiator` | `__init__`, `differentiate` |
| `NewtonSolver` | `__init__`, `solve` |

---

## TC-ND: NumericalDifferentiator のテスト

### TC-ND-01: 中心差分による sin(x) の微分

- **テストID**: `test_central_diff_sin`
- **対象**: `NumericalDifferentiator(method="central").differentiate(x, y)`
- **目的**: 中心差分が最も精度の高い $O(h^2)$ 近似を実現することを確認する
- **入力**:
  - `x = np.linspace(0, 2π, 1000)`
  - `y = np.sin(x)`
- **期待される出力**: `dy_dx ≈ cos(x)`（端点を除く内点）
- **検証方法**: `np.testing.assert_allclose(dy_dx[1:-1], np.cos(x)[1:-1], atol=1e-4)`

---

### TC-ND-02: 前進差分による sin(x) の微分

- **テストID**: `test_forward_diff_sin`
- **対象**: `NumericalDifferentiator(method="forward").differentiate(x, y)`
- **目的**: 前進差分が $O(h)$ 精度で微分を返すことを確認する
- **入力**:
  - `x = np.linspace(0, 2π, 10000)`（$h \approx 6.3 \times 10^{-4}$ で `atol=1e-3` を満たす）
  - `y = np.sin(x)`
- **期待される出力**: 端点（最後の要素）を除いた内点で `dy_dx ≈ cos(x)`
- **検証方法**: `np.testing.assert_allclose(dy_dx[:-1], np.cos(x)[:-1], atol=1e-3)`
- **備考**: 前進差分の誤差は $O(h)$ のため、1000点では誤差 $\approx 3 \times 10^{-3}$ となり許容誤差を超える。10000点が必要。

---

### TC-ND-03: 後退差分による sin(x) の微分

- **テストID**: `test_backward_diff_sin`
- **対象**: `NumericalDifferentiator(method="backward").differentiate(x, y)`
- **目的**: 後退差分が $O(h)$ 精度で微分を返すことを確認する
- **入力**:
  - `x = np.linspace(0, 2π, 10000)`（$h \approx 6.3 \times 10^{-4}$ で `atol=1e-3` を満たす）
  - `y = np.sin(x)`
- **期待される出力**: 端点（最初の要素）を除いた内点で `dy_dx ≈ cos(x)`
- **検証方法**: `np.testing.assert_allclose(dy_dx[1:], np.cos(x)[1:], atol=1e-3)`

---

### TC-ND-04: 線形関数の微分（定数勾配）

- **テストID**: `test_linear_function`
- **対象**: `NumericalDifferentiator(method="central").differentiate(x, y)`
- **目的**: $f(x) = 3x + 2$ の微分が全点で正確に 3.0 になることを確認する
- **入力**:
  - `x = np.linspace(-5, 5, 100)`
  - `y = 3 * x + 2`
- **期待される出力**: `dy_dx ≈ 3.0`（全点）
- **検証方法**: `np.testing.assert_allclose(dy_dx, 3.0, atol=1e-10)`

---

### TC-ND-05: 中心差分が前進・後退差分より精度が高いことの確認

- **テストID**: `test_central_diff_more_accurate`
- **目的**: 同じデータに対して中心差分の最大誤差が前進・後退差分より小さいことを確認する
- **入力**:
  - `x = np.linspace(0.1, π - 0.1, 50)`
  - `y = np.sin(x)`、真値 `dy_true = np.cos(x)`
- **検証方法**:
  ```python
  err_central = np.max(np.abs(central.differentiate(x, y) - dy_true))
  err_forward = np.max(np.abs(forward.differentiate(x, y)[:-1] - dy_true[:-1]))
  assert err_central < err_forward
  ```

---

### TC-ND-06: 不正な method 指定での例外

- **テストID**: `test_invalid_method`
- **対象**: `NumericalDifferentiator(method="invalid")`
- **期待される挙動**: `ValueError` が送出される
- **検証方法**: `pytest.raises(ValueError)`

---

### TC-ND-07: 入力配列の長さ不一致での例外

- **テストID**: `test_mismatched_input_lengths`
- **対象**: `differentiate(x, y)` で `len(x) != len(y)` の場合
- **入力**: `x = np.array([0, 1, 2])`, `y = np.array([0, 1])`
- **期待される挙動**: `ValueError` が送出される
- **検証方法**: `pytest.raises(ValueError)`

---

## TC-NS: NewtonSolver のテスト

### TC-NS-01: x^2 - 2 = 0 の根（解析的微分あり）

- **テストID**: `test_sqrt2_with_analytical_derivative`
- **対象**: `NewtonSolver().solve(f, x0, f_prime)`
- **目的**: 解析的微分を与えたとき、$\sqrt{2}$ を高精度で求めることを確認する
- **入力**:
  - `f = lambda x: x**2 - 2`
  - `f_prime = lambda x: 2 * x`
  - `x0 = 1.0`
- **期待される出力**: `root ≈ √2 = 1.41421356...`
- **検証方法**: `np.testing.assert_allclose(root, np.sqrt(2), rtol=1e-8)`

---

### TC-NS-02: x^2 - 2 = 0 の根（数値微分）

- **テストID**: `test_sqrt2_with_numerical_derivative`
- **対象**: `NewtonSolver().solve(f, x0)` （`f_prime=None`）
- **目的**: 数値微分のみで $\sqrt{2}$ が求まることを確認する
- **入力**:
  - `f = lambda x: x**2 - 2`
  - `x0 = 1.0`
- **期待される出力**: `root ≈ √2`
- **検証方法**: `np.testing.assert_allclose(root, np.sqrt(2), rtol=1e-6)`

---

### TC-NS-03: x^3 - x - 2 = 0 の根

- **テストID**: `test_cubic_equation`
- **対象**: `NewtonSolver().solve(f, x0)`
- **目的**: 3次方程式の根（$x = \phi \approx 1.5214$）を求めることを確認する
- **入力**:
  - `f = lambda x: x**3 - x - 2`
  - `x0 = 1.5`
- **期待される出力**: `root ≈ 1.5213797...`
- **検証方法**: `np.testing.assert_allclose(f(root), 0.0, atol=1e-8)`

---

### TC-NS-04: 負の初期値から正の根への収束

- **テストID**: `test_convergence_from_negative_x0`
- **対象**: `NewtonSolver().solve(f, x0=-1.0)`
- **目的**: 初期値が根と反対側にあっても収束することを確認する
- **入力**:
  - `f = lambda x: x**2 - 4`（根は $x = \pm 2$）
  - `x0 = -1.0`
- **期待される出力**: `root ≈ -2.0`
- **検証方法**: `np.testing.assert_allclose(root, -2.0, atol=1e-8)`

---

### TC-NS-05: 収束しない場合に RuntimeError が送出される

- **テストID**: `test_no_convergence_raises_runtime_error`
- **対象**: `NewtonSolver(max_iter=5).solve(f, x0)`
- **目的**: 最大反復回数内に収束しなかったとき `RuntimeError` が送出されることを確認する
- **入力**:
  - `f = lambda x: x**3 - 2 * x + 2`（根への収束が困難な関数）
  - `x0 = 0.0`、`max_iter = 5`
- **期待される挙動**: `RuntimeError` が送出される
- **検証方法**: `pytest.raises(RuntimeError)`

---

### TC-NS-06: 微分値がゼロの場合に ZeroDivisionError が送出される

- **テストID**: `test_zero_derivative_raises`
- **対象**: `NewtonSolver().solve(f, x0)`
- **目的**: $f'(x) = 0$ となる点で `ZeroDivisionError` が送出されることを確認する
- **入力**:
  - `f = lambda x: x**2`（$x = 0$ が多重根、$f'(0) = 0$）
  - `f_prime = lambda x: 2 * x`
  - `x0 = 0.0`
- **期待される挙動**: `ZeroDivisionError` が送出される
- **検証方法**: `pytest.raises(ZeroDivisionError)`

---

### TC-NS-07: tol パラメータによる精度制御

- **テストID**: `test_tolerance_control`
- **目的**: `tol` を変えたとき、より厳しい許容誤差ほど根の精度が高くなることを確認する
- **入力**:
  - `f = lambda x: x**2 - 2`、`x0 = 1.0`
  - `solver_loose = NewtonSolver(tol=1e-3)`
  - `solver_tight = NewtonSolver(tol=1e-10)`
- **検証方法**:
  ```python
  err_loose = abs(solver_loose.solve(f, x0) - np.sqrt(2))
  err_tight = abs(solver_tight.solve(f, x0) - np.sqrt(2))
  assert err_tight < err_loose
  ```

---

## テストマトリクス

| テストID | 対象クラス | 正常系/異常系 | 確認内容 |
|---|---|---|---|
| TC-ND-01 | `NumericalDifferentiator` | 正常系 | 中心差分の精度 |
| TC-ND-02 | `NumericalDifferentiator` | 正常系 | 前進差分の精度 |
| TC-ND-03 | `NumericalDifferentiator` | 正常系 | 後退差分の精度 |
| TC-ND-04 | `NumericalDifferentiator` | 正常系 | 線形関数の微分 |
| TC-ND-05 | `NumericalDifferentiator` | 正常系 | 差分法間の精度比較 |
| TC-ND-06 | `NumericalDifferentiator` | 異常系 | 不正な method 指定 |
| TC-ND-07 | `NumericalDifferentiator` | 異常系 | 配列長の不一致 |
| TC-NS-01 | `NewtonSolver` | 正常系 | 解析的微分による収束 |
| TC-NS-02 | `NewtonSolver` | 正常系 | 数値微分による収束 |
| TC-NS-03 | `NewtonSolver` | 正常系 | 3次方程式の求根 |
| TC-NS-04 | `NewtonSolver` | 正常系 | 負の初期値からの収束 |
| TC-NS-05 | `NewtonSolver` | 異常系 | 非収束時の `RuntimeError` |
| TC-NS-06 | `NewtonSolver` | 異常系 | 微分値ゼロの `ZeroDivisionError` |
| TC-NS-07 | `NewtonSolver` | 正常系 | `tol` による精度制御 |
