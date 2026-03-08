# MPDE-BO

**MPDE-BO** (Maximum Partial Dependence Effect Bayesian Optimization) は、
高次元かつ評価コストの大きいブラックボックス関数の最大化を効率化するベイズ最適化アルゴリズムの実装です。

ARD (Automatic Relevance Determination) カーネルを持つガウス過程と、
論文固有の重要度指標 **MPDE** (Maximum Partial Dependence Effect) を組み合わせることで、
パラメータを「重要」と「非重要」に動的に分類しながら探索を進めます。

---

## 背景・アルゴリズム概要

### 問題設定

探索空間 $\mathcal{X} \in [0, M]^N$（$M$: グリッド数, $N$: 総パラメータ数）において、
ブラックボックス関数 $f$ の最大値 $x^* = \arg\max_{x \in \mathcal{X}} f(x)$ を最小試行回数で求めます。

探索空間は重要・非重要パラメータの直和に分解されることを前提とします：

$$\mathcal{X} = \mathcal{X}^\top \oplus \mathcal{X}^\bot, \quad f(x) = f_d(x^\top) + \epsilon f_s(x^\bot) \quad (\epsilon \ll 1)$$

### MPDE による重要度定量化

従来の APDE（平均部分従属効果）はパラメータ間の交互作用がある場合に重要パラメータを過小評価します。
MPDE はこれを改善し、個別条件付き期待値 (ICE) の最大を取ることで交互作用がある場合も正確に重要度を捉えます：

$$e_S^* = \max_{i \in [1,n]} \Bigl[\max_{x_S} \hat{f}(x_S, x_C^i) - \min_{x_S} \hat{f}(x_S, x_C^i)\Bigr]$$

### アルゴリズムの流れ

```
1.  D_0 を用いて ARD カーネルベースの GP モデルを構築
2.  for t = 1, 2, ..., T do
3.      各パラメータ i の ARD 長さスケール ℓ_i を取得
4.      f̂_{t-1} と D_{t-1} から各パラメータの MPDE ê_i* を計算
5.      閾値処理によって探索空間を分割:
            (ℓ_i < ε_ℓ) かつ (ê_i* > ε_e)  → x⊤（重要パラメータ）
            それ以外                          → x⊥（非重要パラメータ）
6.      EI を最大化して重要パラメータの次の試行点を決定
7.      非重要パラメータはランダムサンプリングで選択
8.      x_t = x_t⊤ + x_t⊥,  y_t = f(x_t)
9.      D_t を更新し GP を再構築
11. end for
12. return D_T 内の最大データ点 x*
```

---

## 実装の特徴

| 項目             | 内容                                                               |
| ---------------- | ------------------------------------------------------------------ |
| **GPライブラリ** | [BoTorch](https://botorch.org/) / [GPyTorch](https://gpytorch.ai/) |
| **カーネル**     | Matérn 5/2 + ARD（デフォルト）、RBF + ARD も選択可能               |
| **獲得関数**     | EI (`LogExpectedImprovement`、デフォルト)、UCB、PI                 |
| **探索空間分割** | `optimize_acqf(..., fixed_features=...)` で非重要次元を固定        |
| **再現性**       | 全乱数処理が `torch.Generator` に対応                              |

---

## ディレクトリ構成

```
mpde-bo/
├── src/
│   └── mpde_bo/
│       ├── __init__.py
│       ├── gp_model_manager.py      # GP の構築・更新・予測
│       ├── importance_analyzer.py   # ICE / MPDE / APDE の計算
│       ├── parameter_classifier.py  # パラメータ重要度分類
│       ├── acquisition_optimizer.py # 獲得関数の最大化
│       ├── optimizer.py             # メインループ (MPDEBOOptimizer)
│       ├── benchmark.py             # ベンチマーク目的関数
│       └── evaluator.py             # N90 評価指標
├── tests/                           # pytest テストスイート
├── examples/
│   └── run_mpde_bo.py               # 実行例
├── doc/
│   ├── method.md                    # アルゴリズムの数学的記述
│   ├── api_design.md                # 関数レベル API 設計
│   ├── class_design.md              # クラス設計・責務分担
│   └── test_cases.md                # テストケース設計
├── pyproject.toml
└── README.md
```

---

## クラス設計

アルゴリズムの各責務を独立したクラスに分割し、SOLID 原則に基づいて設計されています。

| クラス                 | 責務                                        |
| ---------------------- | ------------------------------------------- |
| `MPDEBOOptimizer`      | アルゴリズム全体のループ制御                |
| `GPModelManager`       | GP モデルの構築・更新・ARD 長さスケール取得 |
| `ImportanceAnalyzer`   | ICE / MPDE / APDE の計算（論文固有）        |
| `ParameterClassifier`  | ARD + MPDE の閾値処理によるパラメータ分類   |
| `AcquisitionOptimizer` | 獲得関数の生成と最大化                      |
| `BenchmarkFunction`    | 論文のモデル目的関数の生成と評価            |
| `N90Evaluator`         | N90 評価指標の計算                          |

---

## インストール

Python 3.13 以上が必要です。

```bash
# 通常インストール
pip install -e .

# 開発用 (pytest を含む)
pip install -e ".[dev]"
```

[uv](https://docs.astral.sh/uv/) を使う場合：

```bash
uv sync
uv sync --group dev   # テスト込み
```

---

## クイックスタート

```python
import torch
from mpde_bo.acquisition_optimizer import AcquisitionConfig, AcquisitionOptimizer
from mpde_bo.benchmark import BenchmarkFunction
from mpde_bo.gp_model_manager import GPConfig, GPModelManager
from mpde_bo.importance_analyzer import ImportanceAnalyzer
from mpde_bo.optimizer import MPDEBOOptimizer
from mpde_bo.parameter_classifier import ClassificationConfig, ParameterClassifier

torch.manual_seed(42)

# 問題設定: 重要 2 次元 + 非重要 3 次元, グリッド幅 100
N_IMPORTANT, N_UNIMPORTANT, M = 2, 3, 100
N = N_IMPORTANT + N_UNIMPORTANT

objective = BenchmarkFunction(
    n_important=N_IMPORTANT,
    n_unimportant=N_UNIMPORTANT,
    grid_size=M,
)

bounds = torch.stack([torch.zeros(N, dtype=torch.double),
                      torch.full((N,), M, dtype=torch.double)])

# 初期観測点 (10 点をランダムサンプリング)
train_X = (bounds[1] - bounds[0]) * torch.rand(10, N, dtype=torch.double) + bounds[0]
train_Y = torch.tensor([[objective(x)] for x in train_X], dtype=torch.double)

# コンポーネントを組み立て
optimizer = MPDEBOOptimizer(
    gp_manager=GPModelManager(config=GPConfig(kernel="matern52")),
    classifier=ParameterClassifier(
        analyzer=ImportanceAnalyzer(),
        config=ClassificationConfig(eps_l=50.0, eps_e=0.05),
    ),
    acq_optimizer=AcquisitionOptimizer(
        config=AcquisitionConfig(type="EI", num_restarts=5, raw_samples=256),
        bounds=bounds,
    ),
    bounds=bounds,
)

# 最適化実行 (100 ステップ)
result = optimizer.optimize(f=objective, T=100, train_X=train_X, train_Y=train_Y)

print(f"最適点:    {result.best_x.tolist()}")
print(f"最大観測値: {result.best_y:.4f}")
print(f"真の最適値: {objective.optimal_value:.4f}")
print(f"達成率:    {result.best_y / objective.optimal_value * 100:.1f}%")
```

完全な実行例は [`examples/run_mpde_bo.py`](examples/run_mpde_bo.py) を参照してください。

---

## 主要な API

### `MPDEBOOptimizer.optimize`

```python
result: BOResult = optimizer.optimize(
    f,           # Callable[[Tensor], float]  目的関数
    T,           # int  追加評価回数
    train_X,     # Tensor (n0, N)  初期観測点
    train_Y,     # Tensor (n0, 1)  初期観測値
    callback,    # Callable[[int, SingleTaskGP, ParameterClassification], None] | None
)
```

### `BOResult`

| 属性      | 型                    | 内容                |
| --------- | --------------------- | ------------------- |
| `best_x`  | `Tensor (N,)`         | 近似最大点 $x^*$    |
| `best_y`  | `float`               | 最大観測値 $f(x^*)$ |
| `train_X` | `Tensor (n_total, N)` | 全評価点の履歴      |
| `train_Y` | `Tensor (n_total, 1)` | 全観測値の履歴      |

### 獲得関数の切り替え

`AcquisitionConfig(type=...)` で `"EI"` (デフォルト)、`"UCB"`、`"PI"` を選択できます。

```python
AcquisitionConfig(type="UCB", ucb_beta=2.0)
```

---

## ベンチマーク評価指標 N90

$N_{90}$ は、異なるピーク位置を持つ 100 個の目的関数に対して
最適値の 90% 以上を達成するまでに必要な評価回数の **90 パーセンタイル値** です。
値が小さいほどアルゴリズムの効率が高いことを示します。

$$N_{90} = \text{Percentile}_{90}\!\left[\min\!\left\{t : f(x_t) \geq 0.9 \cdot f(x^*)\right\}\right]_{100\text{関数}}$$

```python
from mpde_bo.evaluator import N90Evaluator

evaluator = N90Evaluator(budget=200, n_functions=100, n_important=2, n_unimportant=8)
score = evaluator.evaluate(algorithm)  # 小さいほど良い
```

---

## テスト

```bash
pytest
pytest --cov=mpde_bo   # カバレッジレポート付き
```

テストは `tests/` 以下に各クラスごとに分かれています（Unit / Integration / Property の 3 種別）。
詳細は [`doc/test_cases.md`](doc/test_cases.md) を参照してください。

---

## ドキュメント

| ファイル                                     | 内容                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| [`doc/method.md`](doc/method.md)             | アルゴリズムの数学的記述（GP、MPDE、全ステップの擬似コード） |
| [`doc/api_design.md`](doc/api_design.md)     | 関数レベルの API 仕様（引数・返り値・内部実装の詳細）        |
| [`doc/class_design.md`](doc/class_design.md) | クラス設計・責務分担・クラス図                               |
| [`doc/test_cases.md`](doc/test_cases.md)     | テストケース設計（フィクスチャ・テストファイル構成）         |

---

## 依存ライブラリ

| ライブラリ              | 役割                       |
| ----------------------- | -------------------------- |
| `torch`                 | テンソル演算・自動微分     |
| `botorch`               | 獲得関数・最適化ルーティン |
| `gpytorch`              | GP モデル・カーネル定義    |
| `pytest` / `pytest-cov` | テスト（開発用）           |
