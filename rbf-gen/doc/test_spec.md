# RBF-Gen テスト仕様

テストファイルとモジュールの対応、および各テストの意図を記述する。

---

## `tests/test_kernels.py` ← `kernels.py`

### GaussianKernel

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_gaussian_output_shape` | 入力テンソルと同形状を返す | `φ(r).shape == r.shape` |
| `test_gaussian_at_zero` | r=0 のとき φ(0) = 1 | `φ(0) == 1.0` |
| `test_gaussian_decreasing` | r が増加するにつれて単調減少 | `φ(r1) > φ(r2)` for `r1 < r2` |
| `test_gaussian_epsilon_effect` | ε が大きいほど急峻に減少する | `φ_large_eps(r) < φ_small_eps(r)` for `r > 0` |
| `test_gaussian_positive` | 任意の r > 0 に対して正値 | `φ(r) > 0` |

### ThinPlateSplineKernel

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_tps_output_shape` | 入力テンソルと同形状を返す | `φ(r).shape == r.shape` |
| `test_tps_at_zero` | r=0 のとき φ(0) = 0（極限値） | `φ(0) == 0.0` |
| `test_tps_positive_for_nonzero` | r > 1 のとき正値 | `φ(r) > 0` for `r > 1` |
| `test_tps_numerical_stability` | r が非常に小さい値でも NaN/Inf にならない | `isfinite(φ(1e-10))` |

---

## `tests/test_rbf.py` ← `rbf.py`

### RBFBasis の中心配置

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_from_uniform_center_count` | 指定した K 個の中心が生成される | `centers.shape == (K, d)` |
| `test_from_uniform_in_bounds` | すべての中心が指定領域内に収まる | `lb <= centers <= ub` |
| `test_from_quasi_random_center_count` | 指定した K 個の中心が生成される | `centers.shape == (K, d)` |

### RBFBasis の行列計算

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_compute_matrix_shape` | 補間行列の形状が (N, K) | `Φ.shape == (N, K)` |
| `test_compute_matrix_values` | 各要素が `φ(‖x_i - c_j‖)` と一致 | 手動計算値との一致 |
| `test_compute_vector_shape` | 評価ベクトルの形状が (K,) | `φ(x).shape == (K,)` |
| `test_compute_vector_values` | 各要素が `φ(‖x - c_j‖)` と一致 | 手動計算値との一致 |

---

## `tests/test_null_space.py` ← `null_space.py`

### 核となる数学的保証

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_particular_solution_satisfies_interpolation` | w0 が補間条件を満たす | `‖Φ @ w0 - y‖ < tol` |
| `test_null_basis_in_kernel` | 零空間基底が Φ の核に属する | `‖Φ @ null_basis‖ < tol` |
| `test_null_basis_orthonormal` | 零空間基底が正規直交 | `null_basis.T @ null_basis ≈ I` |
| `test_general_solution_satisfies_interpolation` | 任意の α で補間条件が成立する | `‖Φ @ (w0 + N @ α) - y‖ < tol` for random α |
| `test_shapes` | w0 は (K,)、null_basis は (K, K-N) | shape アサーション |
| `test_underdetermined_requirement` | K <= N のとき例外を発生させる | `raises(ValueError)` |

---

## `tests/test_generator.py` ← `generator.py`

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_output_shape` | 出力の形状が (B, null_dim) | `G(z).shape == (B, K-N)` |
| `test_different_z_different_output` | 異なる z が異なる α を生成する | `G(z1) != G(z2)` |
| `test_gradient_flows` | 損失から z を通じて勾配が流れる | `z.grad` が None でない |
| `test_batch_consistency` | バッチ処理が個別処理と一致する | ループ処理との数値一致 |

---

## `tests/test_model.py` ← `model.py`

### 補間条件の保証（RBF-Gen の根幹）

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_interpolation_condition_random_z` | ランダムな z でも訓練点での値が y_i に一致する | `|f_z(x_i) - y_i| < tol` for any z |
| `test_interpolation_condition_after_training` | 学習後も補間条件が崩れていない | `|f_z(x_i) - y_i| < tol` |

### 推論

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_forward_output_shape` | 単一 z での出力形状 | `f_z(x).shape == (N_eval,)` |
| `test_predict_mean_shape` | アンサンブル平均の形状 | shape アサーション |
| `test_predict_std_nonnegative` | 不確実性推定が非負 | `std >= 0` |
| `test_sample_z_distribution` | sample_z が正規分布に従う | Kolmogorov-Smirnov 検定 |

---

## `tests/test_losses.py` ← `losses.py`

### ペナルティ項

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_monotonicity_zero_for_increasing` | 単調増加関数でペナルティ=0 | `pen ≈ 0` |
| `test_monotonicity_positive_for_decreasing` | 非単調な関数でペナルティ>0 | `pen > 0` |
| `test_monotonicity_direction` | increasing=False で非増加を強制 | 減少方向でペナルティ=0 |
| `test_positivity_zero_above_bound` | f >= m のとき penalty=0 | `pen ≈ 0` |
| `test_positivity_positive_below_bound` | f < m の点があるとき penalty>0 | `pen > 0` |
| `test_lipschitz_zero_within_bound` | 傾きが L 以下のとき penalty=0 | `pen ≈ 0` |
| `test_lipschitz_positive_exceeds_bound` | 傾きが L を超えるとき penalty>0 | `pen > 0` |
| `test_smoothness_zero_for_linear` | 線形関数で曲率ペナルティ=0 | `pen ≈ 0` |
| `test_smoothness_positive_for_bumpy` | 急峻な関数で曲率ペナルティ>0 | `pen > 0` |
| `test_convexity_zero_for_convex` | 凸関数でペナルティ=0 | `pen ≈ 0` |
| `test_convexity_positive_for_concave` | 凹関数でペナルティ>0 | `pen > 0` |
| `test_boundary_zero_exact_match` | 境界値と完全一致でペナルティ=0 | `pen ≈ 0` |
| `test_boundary_positive_mismatch` | 境界値と不一致でペナルティ>0 | `pen > 0` |

### KL発散項

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_kl_returns_scalar` | 出力がスカラー | `kl.shape == ()` |
| `test_kl_nonnegative` | KL発散は常に非負 | `kl >= 0` |
| `test_kl_zero_for_matching_distribution` | 生成分布がターゲットと一致するとき小さい | `kl < threshold` |
| `test_point_value_kl_shape` | PointValueKL の出力形状 | スカラー |
| `test_regional_average_kl_shape` | RegionalAverageKL の出力形状 | スカラー |
| `test_extremal_value_kl_max_vs_min` | use_max=True/False で符号が変わる | 両者の区別 |
| `test_gradient_kl_requires_grad` | GradientMagnitudeKL が autograd を使う | 計算グラフが存在する |

### RBFGenLoss（統合）

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_loss_is_scalar` | 合計損失がスカラー | `loss.shape == ()` |
| `test_loss_weighted_sum` | 各項の重みが正しく反映される | 重みを 0 にするとその項が消える |
| `test_loss_nonnegative` | 合計損失が非負 | `loss >= 0` |
| `test_loss_gradient_flows_to_generator` | 損失から Generator のパラメータへ勾配が流れる | `param.grad is not None` |

---

## `tests/test_trainer.py` ← `trainer.py`

| テスト名 | 検証内容 | 期待値 |
|---|---|---|
| `test_loss_decreases` | 学習によって損失が減少する | `loss_final < loss_initial` |
| `test_interpolation_preserved_after_training` | 学習後も補間条件が保たれる | `|f_z(x_i) - y_i| < tol` |
| `test_trainer_runs_without_error` | smoke test：例外なく完走する | 例外なし |

---

## テストの共通事項

- テストデータは `N=5, K=15, d=2` など小さい値で構成する
- 数値許容誤差は原則 `atol=1e-5` を使用する
- 乱数シードは各テストの先頭で `torch.manual_seed(42)` で固定する
- pytest の `fixture` でモデル・データを共有し、重複を排除する
