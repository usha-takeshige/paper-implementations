# テスト設計書：BoTorch ベイズ最適化モジュール（`src/bo/`）

## 1. テスト設計の方針

**参照ファイル**：`/doc/bo_design.md`

`SearchSpace` の正規化・逆正規化の数学的正確性と、目的関数の定義式（`objective = 1 / (rel_l2_error × T)`）の整合性を重点的に検証する。`BayesianOptimizer` 本体は BoTorch に委譲する部分が多いため、「BO ループが正しい回数・順序で試行を生成するか」「最良結果の集計が正しいか」を確認する。`ObjectiveFunction` は `BurgersPINNSolver` を内部で呼び出すため、テスト用には軽量なモックオブジェクトで代替する。

---

## 2. 単体テスト：A-1 アルゴリズム実装テスト（`tests/test_bo_algorithm.py`）

> 各クラス・メソッドが計算として正しく動作するかを検証する。
> `@pytest.mark.algorithm` マーカーを付与する。

### テストケース一覧

| テストID | 対象クラス.メソッド | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|-------------------|---------|---------|-------------------|
| ALG-BO-01 | `SearchSpace.dim` | パラメータ数を正しく返す | 4 パラメータの `SearchSpace` | `dim == 4` |
| ALG-BO-02 | `SearchSpace.bounds` | 形状と値域が仕様通り | 4 パラメータの `SearchSpace` | shape `(2, 4)`、全要素が `[0, 1]`（下限 0、上限 1） |
| ALG-BO-03 | `SearchSpace.to_tensor` | linear パラメータを `[0, 1]` に正規化する | `low=2, high=8` で値 `5`（linear int） | `tensor([[0.5]], dtype=float64)` |
| ALG-BO-04 | `SearchSpace.to_tensor` | log パラメータを `[0, 1]` に正規化する | `low=1e-4, high=1e-2` で値 `1e-3`（log float） | `tensor([[0.5]], dtype=float64)`（対数中点） |
| ALG-BO-05 | `SearchSpace.from_tensor` | 正規化済みテンソルを実スケールに逆変換する（linear） | 正規化値 `0.5`、`low=2, high=8`（linear int） | `{"n_hidden_layers": 5}` |
| ALG-BO-06 | `SearchSpace.from_tensor` | 正規化済みテンソルを実スケールに逆変換する（log） | 正規化値 `0.5`、`low=1e-4, high=1e-2`（log float） | `{"lr": 1e-3}`（±1e-15 許容） |
| ALG-BO-07 | `SearchSpace.from_tensor` | int 型は四捨五入して返す | 正規化値 `0.51`、`low=2, high=8`（linear int） | 結果が `int` 型である |
| ALG-BO-08 | `SearchSpace.sample_sobol` | 出力形状が正しい | `n=8, seed=0`、4 パラメータ | shape `(8, 4)` |
| ALG-BO-09 | `SearchSpace.sample_sobol` | 全点が `[0, 1]^d` に収まる | `n=16, seed=0`、4 パラメータ | 全要素が `[0, 1]` |
| ALG-BO-10 | `TrialResult` | 全フィールドが正常に構築できる | 有効な値を持つ kwargs | 例外が発生しない |
| ALG-BO-11 | `TrialResult` | frozen であり値の書き換えができない | `TrialResult` インスタンス | `trial.objective = 1.0` が `ValidationError` を送出する |
| ALG-BO-12 | `BOResult` | `best_trial_id` のインデックスが `trials` の範囲内 | 有効な `BOResult` | `0 <= best_trial_id < len(trials)` |
| ALG-BO-13 | `BOResult` | frozen であり値の書き換えができない | `BOResult` インスタンス | `result.best_objective = 0.0` が `ValidationError` を送出する |
| ALG-BO-14 | `ObjectiveFunction.__call__` | 戻り値の型が `TrialResult` | モック ObjectiveFunction（後述） | `isinstance(result, TrialResult)` |
| ALG-BO-15 | `ObjectiveFunction.__call__` | `is_initial` フラグが引数通りに設定される | `is_initial=True` で呼び出し | `trial.is_initial == True` |
| ALG-BO-16 | `ObjectiveFunction.__call__` | `trial_id` が引数通りに設定される | `trial_id=3` で呼び出し | `trial.trial_id == 3` |
| ALG-BO-17 | `BayesianOptimizer.optimize` | 全試行数が `n_initial + n_iterations` | `n_initial=3, n_iterations=5`、モック目的関数 | `len(result.trials) == 8` |
| ALG-BO-18 | `BayesianOptimizer.optimize` | 最初の `n_initial` 試行が `is_initial=True` | `n_initial=3`、モック目的関数 | `result.trials[:3]` の全要素で `is_initial == True` |
| ALG-BO-19 | `BayesianOptimizer.optimize` | 残りの試行が `is_initial=False` | `n_iterations=5`、モック目的関数 | `result.trials[3:]` の全要素で `is_initial == False` |
| ALG-BO-20 | `BayesianOptimizer.optimize` | `best_params` が最大 objective を持つ試行と一致する | モック目的関数 | `result.best_params == result.trials[result.best_trial_id].params` |
| ALG-BO-21 | `BayesianOptimizer.optimize` | `best_objective` が全試行の最大値と一致する | モック目的関数 | `result.best_objective == max(t.objective for t in result.trials)` |
| ALG-BO-22 | `ReportGenerator.generate` | ファイルが output_dir に作成される | `BOResult`（モック）、`output_dir` | `bo_report.md` が存在する |
| ALG-BO-23 | `ReportGenerator.generate` | レポートに best_trial_id が含まれる | `BOResult`（`best_trial_id=2`） | ファイル内に `"2"` が含まれる |
| ALG-BO-24 | `ReportGenerator.generate` | レポートに全試行数分の行が含まれる | 5 試行の `BOResult` | テーブル行が 5 行ある（ヘッダーを除く） |

### モックオブジェクトの定義

`ObjectiveFunction` を直接呼び出すテストでは、`BurgersPINNSolver` の実行を避けるために以下のモッククラスを使用する。

```python
class MockObjectiveFunction:
    """テスト用モック。呼び出し番号に応じた固定値を返す。"""

    def __call__(
        self, params: dict, trial_id: int, is_initial: bool
    ) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=float(trial_id + 1),   # trial_id が大きいほど高いスコア
            rel_l2_error=1.0 / (trial_id + 1),
            elapsed_time=1.0,
            is_initial=is_initial,
        )
```

---

## 3. 単体テスト：A-2 理論的性質テスト（`tests/test_bo_theory.py`）

> `bo_design.md` が定義する数学的性質が実装で成立するかを検証する。
> `@pytest.mark.theory` マーカーを付与する。

### テストケース一覧

| テストID | 設計書の性質（根拠） | 対象クラス.メソッド | 検証内容 | 許容誤差 |
|---------|------------------|-------------------|---------|---------|
| THR-BO-01 | 正規化のラウンドトリップ（`bo_design.md` §4-2 SearchSpace） | `SearchSpace.to_tensor` → `from_tensor` | linear パラメータでラウンドトリップすると元の値に戻る | `atol=0.5`（int 四捨五入の誤差以内） |
| THR-BO-02 | 正規化のラウンドトリップ（`bo_design.md` §4-2 SearchSpace） | `SearchSpace.to_tensor` → `from_tensor` | log パラメータでラウンドトリップすると元の値に戻る | `rtol=1e-6` |
| THR-BO-03 | linear 正規化の単調性（`bo_design.md` §4-2 SearchSpace 正規化規則） | `SearchSpace.to_tensor` | パラメータ値が大きいほど正規化値も大きい（linear スケール） | 厳密な大小比較 |
| THR-BO-04 | log 正規化の単調性（`bo_design.md` §4-2 SearchSpace 正規化規則） | `SearchSpace.to_tensor` | パラメータ値が大きいほど正規化値も大きい（log スケール） | 厳密な大小比較 |
| THR-BO-05 | 目的関数の定義式（`bo_design.md` §1） | `ObjectiveFunction.__call__` | `objective == 1 / (rel_l2_error × elapsed_time)` が成立する | `rtol=1e-9` |
| THR-BO-06 | 数値安定性クランプ（`bo_design.md` §8） | `ObjectiveFunction.__call__` | `rel_l2_error` に極小値（1e-15）を渡しても `objective` が有限値になる | `objective < inf` |
| THR-BO-07 | `best_trial_id` の整合性（`bo_design.md` §4-2 BOResult） | `BayesianOptimizer.optimize` | `best_params` が `trials[best_trial_id].params` と一致する | 完全一致 |
| THR-BO-08 | 最良値の単調非減少性（`bo_design.md` §3 Phase 2） | `BayesianOptimizer.optimize` | 各試行時点の累積最良 objective が非減少系列である | 厳密な `<=` 比較 |
| THR-BO-09 | is_initial の排他性（`bo_design.md` §4-2 TrialResult） | `BayesianOptimizer.optimize` | `is_initial=True` の試行と `False` の試行が `n_initial` で分かれている | 完全一致 |
| THR-BO-10 | `bounds` の値域（`bo_design.md` §4-2 SearchSpace） | `SearchSpace.bounds` | 下限行 `bounds[0]` が全次元で 0.0、上限行 `bounds[1]` が全次元で 1.0 | 厳密な `==` 比較 |

### THR-BO-05 の補足

`ObjectiveFunction` の内部で `BurgersPINNSolver` を呼ぶ部分をモックに差し替え、`rel_l2_error=0.01`, `elapsed_time=2.0` の固定値を返すスタブを使用する。期待値：`objective = 1 / (0.01 × 2.0) = 50.0`。

### THR-BO-06 の補足

`rel_l2_error=1e-15`, `elapsed_time=1.0` のスタブを使用。期待値：`objective == 1 / max(1e-15, 1e-10) = 1e10`（クランプにより有限値）。

---

## 4. 振る舞いテスト：グラフ出力による確認（`tests/check_bo_behavior.py`）

> 自動判定が困難な視覚的・定性的な性質を、グラフを出力して人間が確認する。
> 実行方法：`uv run python tests/check_bo_behavior.py`
> 出力先：`tests/behavior_output/`

### 確認項目一覧

| チェックID | 確認すべき観点 | 設計書の根拠 | グラフの種類 | 合格の目安 |
|----------|-------------|-----------|------------|----------|
| BHV-BO-01 | BO の収束性：試行が進むにつれて最良 objective が向上するか | `bo_design.md` §3 Phase 2（逐次探索） | 折れ線グラフ（試行番号 vs 累積最良 objective） | グラフが全体的に右上がり、またはプラトーに収束する |
| BHV-BO-02 | 初期サンプルと BO 提案点の分布の違い | `bo_design.md` §4-2 TrialResult（is_initial） | 散布図（試行番号 vs objective、`is_initial` で色分け） | BO 提案点が初期サンプルより高い objective に集中する傾向が見える |
| BHV-BO-03 | GP サロゲートモデルの予測精度 | `bo_design.md` §4-2 BayesianOptimizer（BoTorch SingleTaskGP） | 1D 断面図（1 パラメータを変化させ、GP 予測平均 ± std を可視化） | GP 予測が観測点を通過し、観測点から遠い領域で標準偏差が大きい |
| BHV-BO-04 | 探索空間のカバレッジ | `bo_design.md` §4-2 SearchSpace（sample_sobol） | ヒストグラム（各パラメータの試行値の分布） | 初期 Sobol サンプルが探索空間を偏りなくカバーしている |

### 実行に使う合成目的関数

`BurgersPINNSolver` の代わりに以下の合成関数を使用する（短時間で動作確認できるようにするため）。

```python
def synthetic_objective(params: dict) -> tuple[float, float]:
    """
    n_neurons が大きく lr が適切な範囲にあるほど高スコアを返す合成関数。
    BO が有意義な探索を行っているかを視覚的に確認するためのスタンドイン。

    Returns: (rel_l2_error, elapsed_time)
    """
    n = params["n_neurons"]
    lr = params["lr"]
    # 擬似誤差：n=50, lr=1e-3 付近で最小化
    rel_l2_error = 0.1 * (1 + abs(n - 50) / 50 + abs(math.log10(lr) + 3))
    elapsed_time = 0.01 * n  # ニューロン数に比例する擬似実行時間
    return rel_l2_error, elapsed_time
```

---

## 5. 許容誤差の設定方針

| テストID | 許容誤差 | 設定根拠 |
|---------|---------|---------|
| THR-BO-01 | `atol=0.5` | int 型の四捨五入によって最大 0.5 の誤差が生じる |
| THR-BO-02 | `rtol=1e-6` | float64 の丸め誤差は最大 ~1e-15 程度であり、log スケール変換後も 1e-6 以内 |
| THR-BO-05 | `rtol=1e-9` | float64 の除算における丸め誤差は ~1e-15 であり、十分な余裕を持った許容誤差 |

---

## 6. pytest 設定（`pyproject.toml` への追記）

```toml
[tool.pytest.ini_options]
markers = [
    "algorithm: アルゴリズム実装の正確性テスト（計算の正しさ）",
    "theory: 論文の理論的性質が成立するかのテスト（数学的保証）",
]
```

※ `pyproject.toml` の既存 markers に追記する形で対応する。

---

## 7. テストファイル構成

```
tests/
├── test_algorithm.py          # 既存（PINNs アルゴリズムテスト）
├── test_theory.py             # 既存（PINNs 理論テスト）
├── check_behavior.py          # 既存（PINNs 振る舞いテスト）
├── test_bo_algorithm.py       # 新規（BO アルゴリズムテスト：ALG-BO-*）
├── test_bo_theory.py          # 新規（BO 理論テスト：THR-BO-*）
└── check_bo_behavior.py       # 新規（BO 振る舞いテスト：BHV-BO-*）
```
