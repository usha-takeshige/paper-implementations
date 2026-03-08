# MPDE-BO ユーザー向け API 設計

本ドキュメントでは、[method.md](method.md) に記述されたアルゴリズムの各ステップに対応する
ユーザー向け関数のインターフェースを定義する。

BoTorch を基盤 GP ライブラリとして使用し、MPDE-BO 固有の計算（MPDE, ICE, パラメータ分類）を
その上に組み合わせる構成とする。

---

## 依存ライブラリと役割分担

| 役割                      | ライブラリ                    |
| ------------------------- | ----------------------------- |
| GP モデル・フィッティング | `botorch` / `gpytorch`        |
| カーネル定義              | `gpytorch.kernels`            |
| 獲得関数（EI, UCB, PI）   | `botorch.acquisition`         |
| 獲得関数の最大化          | `botorch.optim.optimize_acqf` |
| ICE / MPDE 計算           | 本実装（論文固有）            |
| パラメータ分類            | 本実装（論文固有）            |

---

## 対応関係の概要

| アルゴリズムステップ (method.md §6)      | 関数                           |
| ---------------------------------------- | ------------------------------ |
| ステップ 1: GP モデル構築                | `build_gp_model`               |
| ステップ 3: ARD 長さスケール取得         | `get_length_scales`            |
| ステップ 4: MPDE 計算                    | `compute_ice` / `compute_mpde` |
| ステップ 5: 重要パラメータ分類           | `classify_parameters`          |
| ステップ 6: 獲得関数最大化               | `maximize_acquisition`         |
| ステップ 7: 非重要パラメータサンプリング | `sample_unimportant`           |
| ステップ 10: GP 更新                     | `update_gp_model`              |
| 全体ループ (ステップ 1–12)               | `run_mpde_bo`                  |

---

## 1. テンソル規約

BoTorch の規約に従い、全関数で `torch.Tensor` / `torch.double` を使用する。

```python
import torch
from torch import Tensor

# 入力データ
train_X: Tensor  # shape (n, N)  — n: 観測数, N: 総パラメータ数, dtype=torch.double
train_Y: Tensor  # shape (n, 1)  — 単一出力, dtype=torch.double

# 探索空間の境界
bounds: Tensor   # shape (2, N) — bounds[0]: 下限, bounds[1]: 上限
                 # 例: torch.stack([torch.zeros(N), torch.full((N,), M)])
```

---

## 2. データ型

```python
from dataclasses import dataclass
from botorch.models import SingleTaskGP

@dataclass
class ParameterClassification:
    """classify_parameters の返り値 (method.md §6 ステップ 5)"""
    important: list[int]    # 重要パラメータのインデックス (X^⊤)
    unimportant: list[int]  # 非重要パラメータのインデックス (X^⊥)

@dataclass
class BOResult:
    """run_mpde_bo の返り値"""
    best_x: Tensor   # 近似最大点 x*、shape (N,)
    best_y: float    # f(x*) の観測値
    train_X: Tensor  # 全評価点の履歴、shape (n_total, N)
    train_Y: Tensor  # 全観測値の履歴、shape (n_total, 1)
```

---

## 3. GP モデル (method.md §2, §3)

### `build_gp_model`

```python
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

def build_gp_model(
    train_X: Tensor,          # shape (n, N)
    train_Y: Tensor,          # shape (n, 1)
    kernel: str = "matern52", # "matern52" | "rbf"
) -> SingleTaskGP:
    """
    データから ARD カーネルベースの GP モデルを構築・フィッティングする。
    アルゴリズム ステップ 1 に対応。

    内部実装:
        SingleTaskGP(train_X, train_Y, covar_module=<kernel>)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    カーネル (gpytorch.kernels):
        "matern52" (デフォルト):
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=N))
            → Matérn 5/2 + ARD (method.md §3.3)
        "rbf":
            ScaleKernel(RBFKernel(ard_num_dims=N))
            → RBF + ARD (method.md §3.2)

    ハイパーパラメータ (σ, ℓ_{1:N}, ノイズ分散) は
    ExactMarginalLogLikelihood の最大化で決定される。

    Returns:
        eval モードの SingleTaskGP インスタンス
    """
```

### `update_gp_model`

```python
def update_gp_model(
    model: SingleTaskGP,
    new_X: Tensor,   # 新規観測点、shape (q, N)
    new_Y: Tensor,   # 新規観測値、shape (q, 1)
) -> SingleTaskGP:
    """
    新しい観測点を含むデータで GP を再構築し、
    ハイパーパラメータを更新して返す。
    アルゴリズム ステップ 10 に対応。

    既存の train_X / train_Y に new_X / new_Y を結合し、
    build_gp_model を再実行する。
    """
```

### `gp_predict`

```python
def gp_predict(
    model: SingleTaskGP,
    X: Tensor,   # shape (m, N)
) -> tuple[Tensor, Tensor]:
    """
    任意の点 X における GP 事後分布を返す。
    method.md §2 の事後平均・分散の式に対応。

    内部実装:
        posterior = model.posterior(X)
        mean      = posterior.mean.squeeze(-1)      # (m,)
        variance  = posterior.variance.squeeze(-1)  # (m,)

    Returns:
        mu    : shape (m,)  — 事後平均 μ(x | D_t)
        sigma2: shape (m,)  — 事後分散 σ²(x | D_t)
    """
```

---

## 4. ARD 長さスケール取得 (method.md §3, アルゴリズム ステップ 3)

### `get_length_scales`

```python
def get_length_scales(
    model: SingleTaskGP,
) -> Tensor:
    """
    学習済みモデルから各次元の ARD 長さスケール ℓ_{1:N} を取得する。
    アルゴリズム ステップ 3 に対応。

    内部実装:
        model.covar_module.base_kernel.lengthscale  # shape (1, N)
        → squeeze して (N,) を返す

    Returns:
        shape (N,)  — 各パラメータの長さスケール ℓ_i
        値が大きい → そのパラメータの目的関数への影響が小さい (method.md §3.2)
    """
```

---

## 5. パラメータ重要度 (method.md §5)

ICE・MPDE は BoTorch モデルの事後予測を利用した論文固有の計算。

### `compute_ice`

```python
def compute_ice(
    model: SingleTaskGP,
    train_X: Tensor,              # 観測データ X_C の候補、shape (n, N)
    param_indices: list[int],     # 注目する S の次元インデックス
    x_S_grid: Tensor,             # x_S のグリッド点、shape (g, |S|)
) -> Tensor:
    """
    個別条件付き期待値 (ICE) を計算する。
    method.md §5.2 の ICE^i(x_S) に対応。

    各観測点 x_C^i を固定し、x_S を x_S_grid 上で変化させたときの
    GP 事後平均 μ(x_S, x_C^i | D_t) を返す。

    内部実装:
        n 個の観測点 × g グリッド点の組み合わせを構築し
        gp_predict(model, X_combined) で一括推論する。

    Returns:
        shape (n, g)  — ICE^i(x_S_grid[j])
    """
```

### `compute_mpde`

```python
def compute_mpde(
    model: SingleTaskGP,
    train_X: Tensor,              # shape (n, N)
    param_indices: list[int],     # 注目する S の次元インデックス
    x_S_grid: Tensor | None = None,  # None → 各次元を等間隔 50 点で自動生成
) -> float:
    """
    最大部分従属効果 (MPDE) を計算する。
    method.md §5.2 の e_S* に対応。アルゴリズム ステップ 4 に対応。

        e_S* = max_i [ max_{x_S} μ(x_S, x_C^i | D_t)
                     - min_{x_S} μ(x_S, x_C^i | D_t) ]

    compute_ice の結果の各行 (インスタンス i) で
    max - min を取り、さらに全インスタンスの最大値を返す。

    Returns:
        e_S*  — スカラー (Python float)
    """
```

### `compute_apde`  *(比較用: method.md §5.1)*

```python
def compute_apde(
    model: SingleTaskGP,
    train_X: Tensor,
    param_indices: list[int],
    x_S_grid: Tensor | None = None,
) -> float:
    """
    平均部分従属効果 (APDE) を計算する。
    method.md §5.1 の ê_S に対応。

        ê_S = max_{x_S} (1/n Σ_i μ(x_S, x_C^i | D_t))
            - min_{x_S} (1/n Σ_i μ(x_S, x_C^i | D_t))

    注意: 交互作用がある場合に重要パラメータを過小評価する可能性がある
    (method.md §5.1 ⚠️)。MPDE-BO 本体では使用せず、比較目的で提供する。
    """
```

---

## 6. 重要パラメータ分類 (method.md §6 ステップ 5)

### `classify_parameters`

```python
def classify_parameters(
    model: SingleTaskGP,
    train_X: Tensor,
    eps_l: float,   # ε_ℓ: ARD 長さスケール閾値
    eps_e: float,   # ε_e: MPDE 閾値
) -> ParameterClassification:
    """
    各パラメータを重要 (X^⊤) / 非重要 (X^⊥) に分類する。
    アルゴリズム ステップ 5 に対応。

    分類条件 (method.md §6 表):
        ℓ_i < ε_ℓ  かつ  ê_i* > ε_e  → 重要 (important)
        それ以外                       → 非重要 (unimportant)

    内部実装:
        1. get_length_scales(model) で ℓ_{1:N} を取得
        2. 各 i について compute_mpde(model, train_X, [i]) で ê_i* を計算
        3. 条件で分類して ParameterClassification を返す
    """
```

---

## 7. 獲得関数の最大化 (method.md §4, アルゴリズム ステップ 6)

### `maximize_acquisition`

```python
from botorch.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
)
from botorch.optim import optimize_acqf

AcquisitionType = Literal["EI", "UCB", "PI"]

def maximize_acquisition(
    model: SingleTaskGP,
    train_Y: Tensor,                  # shape (n, 1)  — best_f の計算に使用
    bounds: Tensor,                   # shape (2, N)  — 全次元の探索境界
    fixed_features: dict[int, float], # 非重要次元の固定値 {dim_idx: value}
    acquisition: AcquisitionType = "EI",
    ucb_beta: float = 2.0,            # UCB 専用パラメータ β
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> Tensor:
    """
    重要パラメータ空間 X^⊤ 上で獲得関数を最大化し、次の試行点を返す。
    アルゴリズム ステップ 6 に対応。

    非重要パラメータは fixed_features で固定した上で optimize_acqf を呼ぶ。
    これにより重要次元のみを最適化できる。

    獲得関数 (botorch.acquisition):
        "EI" (デフォルト):
            LogExpectedImprovement(model, best_f=train_Y.max())
            → method.md §4 の q(x|D_t) に対応（数値安定版）
        "UCB":
            UpperConfidenceBound(model, beta=ucb_beta)
        "PI":
            ProbabilityOfImprovement(model, best_f=train_Y.max())

    内部実装:
        optimize_acqf(
            acq_function=<acqf>,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
        )

    Returns:
        x_t^⊤ — shape (N,)、非重要次元は fixed_features の値で埋められる
    """
```

---

## 8. 非重要パラメータのサンプリング (method.md §6 ステップ 7)

### `sample_unimportant`

```python
def sample_unimportant(
    unimportant_dims: list[int],
    bounds: Tensor,                    # shape (2, N)
    generator: torch.Generator | None = None,
) -> dict[int, float]:
    """
    非重要パラメータ x^⊥ を一様分布からランダムサンプリングする。
    アルゴリズム ステップ 7 に対応。

        x_t^⊥ ~ Uniform(X^⊥)

    Returns:
        {dim_idx: value}  — maximize_acquisition の fixed_features に直接渡せる形式
    """
```

---

## 9. メインループ (method.md §6 アルゴリズム全体)

### `run_mpde_bo`

```python
from collections.abc import Callable

def run_mpde_bo(
    f: Callable[[Tensor], float],
    T: int,                          # 総評価予算
    train_X: Tensor,                 # 初期観測点 D_0 の X、shape (n0, N)
    train_Y: Tensor,                 # 初期観測点 D_0 の Y、shape (n0, 1)
    eps_l: float,                    # ε_ℓ: ARD 長さスケール閾値
    eps_e: float,                    # ε_e: MPDE 閾値
    bounds: Tensor,                  # shape (2, N) — 探索空間の境界
    kernel: str = "matern52",        # build_gp_model に渡すカーネル種別
    acquisition: AcquisitionType = "EI",
    ucb_beta: float = 2.0,
    num_restarts: int = 10,
    raw_samples: int = 512,
    generator: torch.Generator | None = None,
    callback: Callable[
        [int, SingleTaskGP, ParameterClassification], None
    ] | None = None,
) -> BOResult:
    """
    MPDE-BO アルゴリズムを実行する (method.md §6 全ステップ)。

    アルゴリズム:
        1. train_X, train_Y で build_gp_model → model_0
        2. t = 1, ..., T の各ループで:
            3. get_length_scales(model) で ℓ_{1:N} を取得
            4. classify_parameters が内部で compute_mpde を呼ぶ
            5. classify_parameters(model, train_X, eps_l, eps_e) で分類
            6. sample_unimportant → fixed_features (非重要次元の値)
            7. maximize_acquisition → x_t (重要次元を最適化、非重要次元は固定)
            8. x_t = maximize_acquisition の返り値 (全次元が揃っている)
            9. y_t = f(x_t); train_X, train_Y に追加
           10. update_gp_model で GP を再構築
        12. train_Y.argmax() に対応する train_X を x* として返す

    Args:
        f         : shape (N,) の Tensor を受け取り float を返す目的関数
        T         : 評価回数の上限（初期データ n0 回を除く追加評価数）
        train_X   : 初期観測点（ランダムサンプリング等で事前に取得）
        train_Y   : 初期観測値
        eps_l     : 長さスケール閾値（小さいほど感度が高い）
        eps_e     : MPDE 閾値（大きいほど感度が高い）
        bounds    : torch.stack([lower, upper]) — shape (2, N)
        callback  : 各イテレーション後に呼ばれるフック（進捗観察用）

    Returns:
        BOResult — 近似最大点 x*、観測値 f(x*)、全観測履歴
    """
```

---

## 10. ベンチマーク関数 (method.md §7)

### `make_objective`

```python
def make_objective(
    n_important: int,                       # 重要パラメータ数 d
    n_unimportant: int,                     # 非重要パラメータ数 s
    grid_size: int = 100,                   # M: グリッド幅
    a: tuple[float, ...] | None = None,     # (a_0, ..., a_d)、None → 論文デフォルト値
    sigma: tuple[float, ...] | None = None, # (σ_0, ..., σ_d)、None → 論文デフォルト値
    a_s: float = 0.1,
    sigma_s: float = 25.0,
    generator: torch.Generator | None = None,
) -> Callable[[Tensor], float]:
    """
    method.md §7 に記述されたモデル関数を生成する。

        f(x) = f_d(x_d) + f_s(x_s)

        f_d(x_d): 多峰性混合ガウス関数 (method.md §7 上段)
            a_0·N(μ_0, σ_0²·I) + Σ_{i=1}^{d} a_i·N(μ_i, σ_i²·I)
            ピーク分離制約: ‖μ_i - μ_j‖ > max{2σ_i, 2σ_j}

        f_s(x_s): 緩やかな変動（非重要パラメータ）
            a_s·N(μ_s, σ_s²·I),  μ_s = M/2

    デフォルト値 (method.md §7 末尾):
        (a_0,...,a_4) = (0.3, 1.2, 0.6, 0.7, 0.7)
        (σ_0,...,σ_4) = (20, 5, 5, 6, 6)
        a_s=0.1, σ_s=25, μ_s=M/2

    μ_i はピーク分離制約を満たしながらランダム配置される。

    Returns:
        f: Tensor (N,) → float
    """
```

---

## 11. 評価指標 (method.md §8)

### `compute_n90`

```python
def compute_n90(
    algorithm: Callable[[Callable, int, Tensor, Tensor], BOResult],
    budget: int = 200,             # 最大評価回数 T
    n_functions: int = 100,        # 目的関数のサンプル数
    threshold_ratio: float = 0.9,  # 達成目標の割合（デフォルト 90%）
    n_important: int = 2,
    n_unimportant: int = 8,
    grid_size: int = 100,
    generator: torch.Generator | None = None,
) -> float:
    """
    N90 評価指標を計算する (method.md §8)。

        N90 = Percentile_90 [ min{t : f(x_t) ≥ 0.9·f(x*)} ]_{100関数}

    異なる μ_i を持つ 100 個の目的関数に対して algorithm を実行し、
    最適値の 90% 以上を達成するのに必要な評価回数の 90 パーセンタイルを返す。

    Args:
        algorithm: (f, T, train_X_init, train_Y_init) → BOResult の
                   シグネチャを持つ BO アルゴリズム関数。
                   MPDE-BO のみでなく、比較手法も渡せるよう汎用化している。

    Returns:
        N90 スコア（小さいほど効率が良い）
    """
```

---

## 設計方針

### BoTorch との統合方針

| BoTorch が担う部分                     | 本実装が担う部分     |
| -------------------------------------- | -------------------- |
| GP フィッティング (`fit_gpytorch_mll`) | ICE / MPDE 計算      |
| カーネル定義 (`gpytorch.kernels`)      | パラメータ重要度分類 |
| 獲得関数 (`botorch.acquisition`)       | 探索空間の分割・結合 |
| 獲得関数最大化 (`optimize_acqf`)       | ベンチマーク関数生成 |
| 事後分布取得 (`model.posterior`)       | N90 評価             |

### `fixed_features` による次元固定

ステップ 6–7 の重要/非重要パラメータの分離は、
`optimize_acqf(..., fixed_features={dim: val})` で実現する。
これにより BoTorch の最適化ルーティンをそのまま活用しつつ、
MPDE-BO の探索空間分割を自然に表現できる。

### 拡張ポイント

- **カーネル**: `kernel="matern52"` / `"rbf"` で切り替え。`covar_module` を直接渡すことも想定（上級ユーザー向け）
- **獲得関数**: `acquisition="EI"` / `"UCB"` / `"PI"` で切り替え (method.md §4 注記)
- **再現性**: 乱数に関わる全関数が `torch.Generator` を受け付け、シードを固定した再現実験が可能
- **比較実験**: `compute_n90` は任意のアルゴリズムを受け付けるため、標準 BO との比較が容易
