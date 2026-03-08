# テストケース設計

本ドキュメントでは、[class_design.md](class_design.md) で定義した各クラスに対する
テストケースを記述する。TDD で開発を進めるため、実装前に期待する振る舞いを明確にする。

---

## テストの分類

| 種別 | 説明 |
|---|---|
| **Unit** | 1 クラスを単体でテスト。外部依存はモックまたは最小フィクスチャ |
| **Integration** | 複数クラスを組み合わせた動作を確認 |
| **Property** | 数学的性質（不変条件）を検証 |

---

## 1. `GPModelManager`

### `build`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 1-1 | 正常系: モデルが返る | `train_X` shape `(5, 3)`, `train_Y` shape `(5, 1)` | `SingleTaskGP` インスタンスが返る | Unit |
| 1-2 | モデルが eval モード | 同上 | `model.training == False` | Unit |
| 1-3 | matern52 カーネル | `kernel="matern52"` | `model.covar_module.base_kernel` が `MaternKernel(nu=2.5)` | Unit |
| 1-4 | rbf カーネル | `kernel="rbf"` | `model.covar_module.base_kernel` が `RBFKernel` | Unit |
| 1-5 | 不正カーネル名 | `kernel="unknown"` | `ValueError` が送出される | Unit |
| 1-6 | ARD の次元数 | N=4 次元の `train_X` | `lengthscale.shape == (1, 4)` | Unit |

### `update`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 2-1 | 観測数が増える | 既存 model, `new_X` shape `(2, N)` | 返り値モデルの `train_X.shape[0]` が元 + 2 | Unit |
| 2-2 | 返り値が新しいインスタンス | 同上 | `updated_model is not model` | Unit |
| 2-3 | eval モード | 同上 | `updated_model.training == False` | Unit |

### `get_length_scales`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 3-1 | shape が正しい | N=4 次元モデル | 返り値の shape が `(4,)` | Unit |
| 3-2 | 値が正 | 任意の学習済みモデル | 全要素 > 0 | Property |
| 3-3 | 影響ゼロ次元は長さスケールが大きい | 目的値に無関係な次元を含むデータ | その次元の長さスケールが他次元より大きい | Property |

### `predict`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 4-1 | shape が正しい | `X` shape `(10, N)` | `mu.shape == (10,)`, `sigma2.shape == (10,)` | Unit |
| 4-2 | 分散が非負 | 任意の `X` | `sigma2` の全要素 ≥ 0 | Property |
| 4-3 | 訓練点での不確実性が小さい | `X = train_X` (ノイズなしモデル) | `sigma2` の全要素 ≈ 0 | Property |

---

## 2. `ImportanceAnalyzer`

### `compute_ice`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 5-1 | shape が正しい | n=5 観測, g=20 グリッド | 返り値の shape が `(5, 20)` | Unit |
| 5-2 | 無関係次元では行が均一 | 目的値に無関係な次元 `param_indices=[k]` | 各行の `max - min` ≈ 0 | Property |
| 5-3 | 重要次元では行が変動する | 目的値に影響する次元 `param_indices=[k]` | 少なくとも 1 行の `max - min` > 0 | Property |
| 5-4 | グリッド自動生成 | `x_S_grid=None` | エラーなく実行できる | Unit |

### `compute_mpde`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 6-1 | 無関係次元は MPDE ≈ 0 | 目的値に無関係な次元 | `mpde ≈ 0`（許容誤差 1e-3） | Property |
| 6-2 | 重要次元は MPDE > 0 | 目的値に影響する次元 | `mpde > 0` | Property |
| 6-3 | スカラーが返る | 任意の入力 | 返り値が `float` 型 | Unit |
| 6-4 | MPDE ≥ APDE | 同一モデル・データ | `compute_mpde(...) >= compute_apde(...)` | Property |

**補足 (6-4 の根拠)**: MPDE は個別 ICE の最大値を取るのに対し、APDE は平均 PDP を使うため、
交互作用がある場合に APDE が MPDE を下回ることが method.md §5 で示されている。

### `compute_apde`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 7-1 | スカラーが返る | 任意の入力 | 返り値が `float` 型 | Unit |
| 7-2 | 無関係次元は APDE ≈ 0 | 目的値に無関係な次元 | `apde ≈ 0` | Property |
| 7-3 | 非負 | 任意の入力 | `apde >= 0` | Property |

---

## 3. `ParameterClassifier`

### `classify`

セットアップ: 次元 0,1 が重要・次元 2,3 が非重要なデータを用意し、
GP モデルを学習させてから分類する。

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 8-1 | 返り値の型 | 任意のモデル | `ParameterClassification` インスタンス | Unit |
| 8-2 | インデックスに重複がない | 任意のモデル | `important ∩ unimportant == []` | Property |
| 8-3 | 全次元が分類される | N=4 次元 | `len(important) + len(unimportant) == 4` | Property |
| 8-4 | 重要次元が検出される | 次元 0,1 のみ目的値に影響するデータ | `0 in important and 1 in important` | Integration |
| 8-5 | 非重要次元が検出される | 次元 2,3 は目的値に無関係 | `2 in unimportant and 3 in unimportant` | Integration |
| 8-6 | eps_l を非常に大きくすると全て非重要 | `eps_l=1e9, eps_e=0` | `len(unimportant) == N` | Unit |
| 8-7 | eps_e を非常に大きくすると全て非重要 | `eps_l=0, eps_e=1e9` | `len(unimportant) == N` | Unit |

---

## 4. `AcquisitionOptimizer`

### `maximize`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 9-1 | shape が正しい | N=4 次元 | 返り値の shape が `(4,)` | Unit |
| 9-2 | 候補点が探索境界内 | `bounds = [[0]*N, [1]*N]` | `all(bounds[0] <= x <= bounds[1])` | Property |
| 9-3 | fixed_features が固定される | `fixed_features={2: 0.5, 3: 0.3}` | `candidate[2] ≈ 0.5`, `candidate[3] ≈ 0.3` | Unit |
| 9-4 | EI で動作する | `acquisition="EI"` | エラーなく候補点が返る | Unit |
| 9-5 | UCB で動作する | `acquisition="UCB"` | エラーなく候補点が返る | Unit |
| 9-6 | PI で動作する | `acquisition="PI"` | エラーなく候補点が返る | Unit |
| 9-7 | 不正獲得関数名 | `acquisition="INVALID"` | `ValueError` が送出される | Unit |

---

## 5. `MPDEBOOptimizer`

### `optimize`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 10-1 | 返り値の型 | 任意の設定 | `BOResult` インスタンス | Unit |
| 10-2 | 評価回数が正確 | `T=5`, 初期データ n0=3 | `result.train_X.shape[0] == 8` | Unit |
| 10-3 | 観測値が記録される | 任意の設定 | `result.train_Y.shape == (n0+T, 1)` | Unit |
| 10-4 | best_x が境界内 | `bounds=[[0]*N, [1]*N]` | `all(0 <= result.best_x <= 1)` | Property |
| 10-5 | best_y が train_Y の最大値 | 任意の設定 | `result.best_y == result.train_Y.max().item()` | Unit |
| 10-6 | callback が T 回呼ばれる | `T=5`, callback を渡す | callback の呼び出し回数が 5 | Unit |
| 10-7 | 単峰性関数で最適解に近づく | `f(x) = -(x-0.5)^2`, `T=30` | `result.best_y > initial_best_y` | Integration |

---

## 6. `BenchmarkFunction`

### `__call__`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 11-1 | スカラーが返る | shape `(N,)` の Tensor | 返り値が `float` 型 | Unit |
| 11-2 | バッチ入力はエラー | shape `(m, N)` の Tensor | `ValueError` が送出される | Unit |
| 11-3 | 定義域内で有限値 | `x ∈ [0, M]^N` | `isfinite(f(x)) == True` | Property |

### `optimal_value`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 12-1 | 非負 | 任意の設定 | `f.optimal_value >= 0` | Property |
| 12-2 | グリッド全探索での最大値と一致 | `n_important=1, grid_size=100` | `optimal_value ≈ max(f(x) for x in grid)` | Integration |

### コンストラクタ (ピーク分離制約)

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 13-1 | ピーク分離制約が満たされる | `n_important=2`, 任意シード | 全ペア `‖μ_i - μ_j‖ > max{2σ_i, 2σ_j}` | Property |
| 13-2 | 再現性 | 同一 `generator` | 同一の `optimal_value` | Unit |
| 13-3 | デフォルトパラメータで動作 | 引数なし（デフォルト値） | エラーなくインスタンス生成 | Unit |

---

## 7. `N90Evaluator`

### `evaluate`

| # | テストケース | 入力 | 期待する出力 / 状態 | 種別 |
|---|---|---|---|---|
| 14-1 | 正の有限値が返る | 任意の設定 | `0 < N90 < budget` | Property |
| 14-2 | 完璧なアルゴリズム → N90 が小さい | 常に最適点を返す oracle | `N90 <= n_initial` | Integration |
| 14-3 | ランダムアルゴリズム → N90 が大きい | ランダムサンプリング | `N90 > N90_of_mpde_bo`（期待値として） | Integration |
| 14-4 | 再現性 | 同一 `generator` | 同一の `N90` スコア | Unit |

---

## フィクスチャ設計

テスト間で共有するフィクスチャを以下に定義する。

```
# conftest.py で定義するフィクスチャ

@pytest.fixture
def simple_train_data():
    """2次元入力、単純な二次関数目的値のデータ (n=10)"""
    # train_X: shape (10, 2), train_Y: shape (10, 1)
    # f(x) = -(x[0]-0.5)^2 - (x[1]-0.5)^2

@pytest.fixture
def important_unimportant_data():
    """
    次元 0,1 のみ目的値に影響し、次元 2,3 は無関係なデータ (N=4, n=30)
    ParameterClassifier / ImportanceAnalyzer の Property テストに使用
    """

@pytest.fixture
def fitted_gp_model(simple_train_data):
    """simple_train_data で学習済みの SingleTaskGP"""

@pytest.fixture
def gp_config_matern():
    GPConfig(kernel="matern52")

@pytest.fixture
def classification_config():
    ClassificationConfig(eps_l=1.0, eps_e=0.1)
```

---

## テストファイル構成

```
tests/
├── conftest.py                    # 共有フィクスチャ
├── test_gp_model_manager.py       # GPModelManager のテスト (§1)
├── test_importance_analyzer.py    # ImportanceAnalyzer のテスト (§2)
├── test_parameter_classifier.py   # ParameterClassifier のテスト (§3)
├── test_acquisition_optimizer.py  # AcquisitionOptimizer のテスト (§4)
├── test_mpde_bo_optimizer.py      # MPDEBOOptimizer のテスト (§5)
├── test_benchmark_function.py     # BenchmarkFunction のテスト (§6)
└── test_n90_evaluator.py          # N90Evaluator のテスト (§7)
```
