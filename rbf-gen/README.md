# RBF-Gen

**"Knowledge-guided generative surrogate modeling for high-dimensional design optimization under scarce data"**（Bingran Wang et al., UC San Diego / Samsung Electronics）の実装。

## 概要

RBF-Gen は、データが希少な状況でドメイン知識を活用するサロゲートモデリングフレームワーク。

核となるアイデアは**過完備RBF系の零空間の活用**。RBF中心数 K がデータ点数 N より多いため、補間条件を満たす重みベクトルは無限に存在する。小さなジェネレーターネットワーク（MLP）を学習させて、補間条件を保ちながらドメイン知識（単調性・正値性・Lipschitz 上限など）を満足する関数を生成する。

**主な特長：**
- ジェネレーターの出力によらず、補間条件は構造上常に満足される
- ドメイン知識はペナルティ項・KL 発散項として微分可能な形で組み込まれる
- ジェネレーターがアンサンブルを生成するため、不確実性推定が可能

## 数学的背景

N 個の学習データ {(x_i, y_i)}、K 個の RBF 中心 {c_j}（K > N）、補間行列 Phi（N×K）に対して：

```
任意の補間解: w = w0 + Null @ alpha

  w0        : 最小ノルム特解         (K,)
  Null      : Phi の零空間基底       (K, K-N)
  alpha     : 自由係数ベクトル       (K-N,)
```

ジェネレーター G は潜在変数 z ~ N(0, I) を alpha にマッピングする：

```
f_z(x) = Phi(x)^T @ (w0 + Null @ G(z))
```

`Phi @ Null = 0` が成り立つため、補間条件 `f_z(x_i) = y_i` は z の値に関係なく常に満たされる。

ジェネレーターは次の損失を最小化するよう学習される：

```
L = sum_i lambda_i * pen_i(f_z) + sum_j gamma_j * KL(p_gen(s_j) || p_target(s_j))
```

数学的な詳細は [doc/method.md](doc/method.md) を参照。

## ディレクトリ構成

```
rbf-gen/
  src/rbf_gen/
    kernels.py      # RBF カーネル関数（ガウス、薄板スプライン）
    rbf.py          # RBFBasis: 中心配置・補間行列の計算
    null_space.py   # NullSpaceDecomposition: SVD による零空間計算
    generator.py    # Generator: 潜在変数 z -> alpha の MLP
    model.py        # RBFGenModel: forward, predict_mean, predict_std の統合
    losses.py       # ペナルティ項・KL 発散項・RBFGenLoss
    trainer.py      # RBFGenTrainer: Adam オプティマイザによる学習ループ
  tests/
    test_kernels.py
    test_rbf.py
    test_null_space.py
    test_generator.py
    test_model.py
    test_losses.py
    test_trainer.py
  doc/
    method.md         # 数学的導出
    class_diagram.md  # Mermaid クラス図
    test_spec.md      # テスト仕様
```

## セットアップ

```bash
uv sync
```

## 使い方

```python
import torch
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition
from rbf_gen.generator import Generator
from rbf_gen.model import RBFGenModel
from rbf_gen.losses import MonotonicityPenalty, PositivityPenalty, RBFGenLoss
from rbf_gen.trainer import RBFGenTrainer

# --- 学習データ ---
X = torch.tensor([[0.1], [0.4], [0.7], [0.9], [1.0]])  # (N, d)
y = torch.tensor([0.2, 0.5, 0.6, 0.8, 1.0])            # (N,)
N, d = X.shape

# --- Step 1: 過完備 RBF 基底の構築 (K > N) ---
K = 20
bounds = torch.tensor([[0.0], [1.0]])
kernel = GaussianKernel(epsilon=1.0)
rbf_basis = RBFBasis.from_quasi_random(K, bounds, kernel)

# --- Step 2: 零空間の計算 ---
Phi = rbf_basis.compute_matrix(X)
null_decomp = NullSpaceDecomposition()
null_decomp.fit(Phi, y)

# --- Step 3: ジェネレーターの構築 ---
latent_dim = K - N
generator = Generator(latent_dim=latent_dim, null_dim=latent_dim, hidden_dims=[64, 64])
model = RBFGenModel(rbf_basis=rbf_basis, null_decomp=null_decomp, generator=generator)

# --- Step 4: ドメイン知識の定義 ---
eval_grid = torch.linspace(0, 1, 50).unsqueeze(1)
penalty_terms = [
    MonotonicityPenalty(increasing=True, weight=1.0),
    PositivityPenalty(lower_bound=0.0, weight=1.0),
]
loss_fn = RBFGenLoss(penalty_terms=penalty_terms, kl_terms=[])

# --- Step 5: ジェネレーターの学習 ---
trainer = RBFGenTrainer(model=model, loss_fn=loss_fn, n_epochs=500, batch_size=32, eval_grid=eval_grid)
trainer.train()

# --- 推論 ---
x_test = torch.linspace(0, 1, 100).unsqueeze(1)
mean = model.predict_mean(x_test, n_samples=200)   # (100,)  アンサンブル平均
std  = model.predict_std(x_test, n_samples=200)    # (100,)  不確実性
```

## モジュール

### カーネル関数 (`kernels.py`)

| クラス | 式 |
|---|---|
| `GaussianKernel(epsilon)` | `phi(r) = exp(-epsilon^2 * r^2)` |
| `ThinPlateSplineKernel` | `phi(r) = r^2 * log(r)`（`phi(0) = 0` の処理済み） |

### RBFBasis (`rbf.py`)

| メソッド | 説明 |
|---|---|
| `from_uniform(K, bounds, kernel)` | K 個の中心を一様ランダムに配置 |
| `from_quasi_random(K, bounds, kernel)` | Sobol 列で K 個の中心を配置 |
| `compute_matrix(X)` | 補間行列 Phi (N, K) を返す |
| `compute_vector(x)` | 1 点の RBF ベクトル phi (K,) を返す |

### NullSpaceDecomposition (`null_space.py`)

Phi の完全 SVD を用いて以下を計算：
- `w0`：最小ノルム特解
- `null_basis`：零空間基底行列 (K, K-N)

### Generator (`generator.py`)

`z -> alpha` のための小さな MLP。隠れ層の構成は `hidden_dims` で指定可能。デフォルト活性化関数は Tanh。

### RBFGenModel (`model.py`)

| メソッド | 説明 |
|---|---|
| `forward(x, z)` | 点 x での f_z を評価。(N_eval,) または (N_eval, B) を返す |
| `predict_mean(x, n_samples)` | ランダム z のアンサンブル平均 |
| `predict_std(x, n_samples)` | アンサンブル標準偏差（不確実性） |
| `sample_z(batch_size)` | z ~ N(0, I) のサンプリング |

### 損失関数 (`losses.py`)

**ペナルティ項**（`PenaltyTerm` のサブクラス）：

| クラス | 強制する制約 |
|---|---|
| `MonotonicityPenalty(increasing, dim)` | 単調増加 or 単調減少 |
| `PositivityPenalty(lower_bound)` | f(x) >= lower_bound |
| `LipschitzPenalty(L)` | `|f(x) - f(y)| / |x - y| <= L` |
| `SmoothnessPenalty` | 二階差分の小ささ（滑らかさ） |
| `ConvexityPenalty(convex)` | 凸性 or 凹性 |
| `BoundaryPenalty(points, values)` | 境界条件 f(x_b) = v_b |

**KL 発散項**（`KLDivergenceTerm` のサブクラス）：

| クラス | 対象の統計量 |
|---|---|
| `PointValueKL(x0, mean, std)` | z に対する f_z(x0) の分布 |
| `RegionalAverageKL(region, mean, std)` | 領域平均の分布 |
| `ExtremalValueKL(region, mean, std, use_max)` | 最大値 or 最小値の分布 |
| `GradientMagnitudeKL(x0, mean, std)` | 勾配の大きさ `||grad f_z(x0)||` の分布 |
| `CurvatureKL(x0, dim, mean, std)` | 曲率 `d^2 f_z / dx_dim^2` の分布 |

### RBFGenTrainer (`trainer.py`)

Adam オプティマイザによる学習ループ。各ステップで z のバッチをサンプリングし、ペナルティ項と KL 項の和を最小化する。

## テスト

```bash
uv run pytest tests/
uv run pytest tests/ -v   # 詳細出力
```

## 論文

> Bingran Wang, Mark Sperry, Qi Zhou, John T. Hwang.
> *Knowledge-guided generative surrogate modeling for high-dimensional design optimization under scarce data.*
