# ニュートン法による数値微分

## 概要

ニュートン法（Newton's Method）は、方程式 $f(x) = 0$ の根を反復的に求めるアルゴリズムであり、
数値微分の文脈では関数の勾配（微分値）を利用して最適解へ収束させる手法として広く用いられる。

本ドキュメントでは、ニュートン法の数学的基礎と、離散データに対する数値微分への応用を解説する。

---

## 1. ニュートン法の数学的基礎

### 1.1 テイラー展開による導出

関数 $f(x)$ を点 $x_n$ 周りで1次のテイラー展開すると：

$$
f(x) \approx f(x_n) + f'(x_n)(x - x_n)
$$

$f(x) = 0$ とおいて $x$ について解くと、更新式が得られる：

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

これがニュートン法の基本反復式である。

### 1.2 幾何学的解釈

各ステップで、現在の点 $(x_n, f(x_n))$ における接線を引き、
その接線が $x$ 軸と交わる点を次の近似解 $x_{n+1}$ とする。

```
f(x)
 |        /
 |       /
 |    * /   ← f(x_n) での接線
 |     /
 |    /  *  ← f(x_{n+1})
 |   /      *
---+--+--+--+------ x
       x_{n+1} x_n
```

### 1.3 収束性

- **収束次数**: 2次収束（近傍での誤差が2乗のオーダーで減少）
- **収束条件**: $f'(x_n) \neq 0$ かつ初期値が根の近傍にあること
- **収束判定**: $|x_{n+1} - x_n| < \varepsilon$ または $|f(x_n)| < \varepsilon$

---

## 2. 数値微分

### 2.1 有限差分法

離散データ点 $(x_i, y_i)$ に対して数値微分を行う手法。

#### 前進差分（Forward Difference）

$$
f'(x_i) \approx \frac{f(x_{i+1}) - f(x_i)}{h}
$$

誤差オーダー: $O(h)$

#### 後退差分（Backward Difference）

$$
f'(x_i) \approx \frac{f(x_i) - f(x_{i-1})}{h}
$$

誤差オーダー: $O(h)$

#### 中心差分（Central Difference）

$$
f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}
$$

誤差オーダー: $O(h^2)$（最も精度が高い）

ここで $h = x_{i+1} - x_i$ はステップ幅。

### 2.2 数値微分とニュートン法の組み合わせ

$f'(x)$ の解析的な式が不明な場合、数値微分を用いてニュートン法を適用できる（**数値ニュートン法**または**秘密ニュートン法**とも呼ぶ）：

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'_{\text{num}}(x_n)}
$$

ただし：

$$
f'_{\text{num}}(x_n) \approx \frac{f(x_n + h) - f(x_n - h)}{2h}
$$

---

## 3. アルゴリズム

### 3.1 ニュートン法（解析的微分あり）

```
入力: f, f', x0, ε, max_iter
出力: 根の近似値 x*

1. x ← x0
2. for i = 1, 2, ..., max_iter:
   a. Δx ← f(x) / f'(x)
   b. x ← x - Δx
   c. if |Δx| < ε: break
3. return x
```

### 3.2 ニュートン法（数値微分を使用）

```
入力: f, x0, h, ε, max_iter
出力: 根の近似値 x*

1. x ← x0
2. for i = 1, 2, ..., max_iter:
   a. f_val ← f(x)
   b. f_grad ← (f(x + h) - f(x - h)) / (2h)   ← 中心差分
   c. Δx ← f_val / f_grad
   d. x ← x - Δx
   e. if |Δx| < ε: break
3. return x
```

---

## 4. クラス・関数の定義

### 4.1 `NumericalDifferentiator`

離散データに対して数値微分を行うクラス。

```python
class NumericalDifferentiator:
    def __init__(self, method: str = "central"):
        """
        Parameters
        ----------
        method : str
            差分法の種類。"forward", "backward", "central" のいずれか。
        """

    def differentiate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray, shape (n,)
            独立変数の配列（等間隔でなくても可）
        y : np.ndarray, shape (n,)
            従属変数の配列

        Returns
        -------
        dy_dx : np.ndarray, shape (n,)
            各点における数値微分値
        """
```

### 4.2 `NewtonSolver`

ニュートン法で方程式の根を求めるクラス。

```python
class NewtonSolver:
    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 100,
        h: float = 1e-5,
    ):
        """
        Parameters
        ----------
        tol : float
            収束判定の許容誤差。
        max_iter : int
            最大反復回数。
        h : float
            数値微分のステップ幅。
        """

    def solve(
        self,
        f: Callable[[float], float],
        x0: float,
        f_prime: Callable[[float], float] | None = None,
    ) -> float:
        """
        Parameters
        ----------
        f : Callable
            根を求める関数 f(x) = 0
        x0 : float
            初期値
        f_prime : Callable or None
            解析的微分。None の場合は数値微分を使用。

        Returns
        -------
        x : float
            収束した根の近似値

        Raises
        ------
        RuntimeError
            最大反復回数内に収束しなかった場合。
        ZeroDivisionError
            微分値がゼロになった場合。
        """
```

---

## 5. パラメータ一覧

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `tol` | `1e-8` | 収束許容誤差 $\varepsilon$ |
| `max_iter` | `100` | 最大反復回数 |
| `h` | `1e-5` | 数値微分のステップ幅 |
| `method` | `"central"` | 差分法の種類（`forward` / `backward` / `central`） |

---

## 6. 使用例

```python
import numpy as np
from newton_method import NewtonSolver, NumericalDifferentiator

# 例1: x^2 - 2 = 0 の根（√2 の計算）
solver = NewtonSolver(tol=1e-10)
root = solver.solve(f=lambda x: x**2 - 2, x0=1.0)
print(f"√2 ≈ {root:.10f}")  # 1.4142135624

# 例2: 離散データの数値微分
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

diff = NumericalDifferentiator(method="central")
dy_dx = diff.differentiate(x, y)
# dy_dx ≈ cos(x)
```

---

## 7. 注意事項

- **ステップ幅 $h$ の選択**: 小さすぎると丸め誤差が増大し、大きすぎると打ち切り誤差が増大する。
  一般に $h \approx \sqrt{\varepsilon_{\text{machine}}} \approx 10^{-8}$ 程度が適切（64bit浮動小数点の場合）。
- **初期値の重要性**: ニュートン法は初期値が根から遠い場合に発散することがある。
- **多重根**: $f'(x^*) = 0$ となる多重根では収束が遅くなる（1次収束になる）。
