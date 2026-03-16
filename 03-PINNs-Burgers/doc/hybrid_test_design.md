# テスト設計書：Hybrid LLM + BO Hyperparameter Optimizer

## 1. テスト設計の方針

**参照ファイル**：`/doc/hybrid_design.md`

ハイブリッド最適化モジュールは論文アルゴリズムではなく工学的なパイプラインであるため、テストの焦点は「設計書に記述されたパイプライン動作の保証」に置く。
A-1 テストでは `HybridOptimizer.optimize()` の各ステップが正しく実行されることを検証する（LLM 呼び出し回数・試行数・best_objective の正確性・HybridResult の不変性）。
A-2 テストでは探索空間の絞り込みが設計書通りの性質（包含関係・log スケール処理・マージン適用）を満たすことを検証する。
すべてのテストで `MockChain` と `MockObjectiveFunction` を使用し、実際の PINN 学習・Gemini API 呼び出しを行わない。

---

## 2. 単体テスト：A-1 アルゴリズム実装テスト（`test_hybrid_algorithm.py`）

> 各アルゴリズムステップが正しく実装されているかを検証する。
> `@pytest.mark.algorithm` マーカーを付与する。

### テストケース一覧

| テストID | 対象メソッド / クラス | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|--------------------|---------|---------|--------------------|
| ALG-HYB-01 | `HybridOptimizer.optimize` | Phase 1 で LLM が `n_llm_iterations` 回呼ばれること | `n_llm_iterations=3`、`MockChain`（呼び出しカウンタ付き） | `MockChain.invoke` が 3 回呼ばれること |
| ALG-HYB-02 | `HybridOptimizer.optimize` | `llm_trials` の件数が `llm_config.n_initial + n_llm_iterations` に等しいこと | `n_initial=3`、`n_llm_iterations=4` | `len(result.llm_trials) == 7` |
| ALG-HYB-03 | `HybridOptimizer.optimize` | `bo_trials` の件数が `bo_config.n_initial + bo_config.n_iterations` に等しいこと | `n_initial=3`、`n_iterations=5` | `len(result.bo_trials) == 8` |
| ALG-HYB-04 | `HybridOptimizer.optimize` | `best_objective` が全試行（Phase 1 + 2）の最大目的関数値と一致すること | 任意の MockObjectiveFunction | `result.best_objective == max(t.objective for t in result.llm_trials + result.bo_trials)` |
| ALG-HYB-05 | `HybridOptimizer.optimize` | `best_params` が `best_objective` を持つ試行のパラメータと一致すること | 任意の MockObjectiveFunction | `result.best_params == combined[result.best_trial_id].params` |
| ALG-HYB-06 | `HybridResult` | `HybridResult` が frozen であること（代入で例外が送出されること） | 正常な `HybridResult` インスタンス | `best_objective = 0.0` で `FrozenInstanceError` が送出されること |
| ALG-HYB-07 | `HybridResult` | `HybridResult` が正常に構築できること | 最小限のフィールドを持つ引数 | 例外が送出されないこと |

---

## 3. 単体テスト：A-2 パイプライン保証テスト（`test_hybrid_algorithm.py`）

> `_narrow_search_space` が設計書通りの性質を満たすことと、パイプライン全体の空間整合性を検証する。
> A-1 と同一ファイル（`test_hybrid_algorithm.py`）に記載し、`@pytest.mark.algorithm` マーカーを付与する。

### テストケース一覧

| テストID | 検証する性質（根拠） | 対象メソッド | 検証内容 | 許容誤差 |
|---------|----------------|------------|---------|---------|
| ALG-HYB-08 | 絞り込み後の各パラメータ境界が元の SearchSpace の境界に包含されること（`hybrid_design.md` §3-2 `_narrow_search_space`） | `HybridOptimizer._narrow_search_space` | `narrowed_space.parameters[i].low >= original_space.parameters[i].low` かつ `narrowed_space.parameters[i].high <= original_space.parameters[i].high` がすべての i で成立すること | — |
| ALG-HYB-09 | Phase 2 の全試行パラメータが `narrowed_space` の境界内に収まること（`hybrid_design.md` §3-2 `optimize()`） | `HybridOptimizer.optimize` | `result.bo_trials` の全 `TrialResult.params` について、各パラメータが `narrowed_space` の境界を超えないこと | — |
| ALG-HYB-10 | linear スケールパラメータの絞り込みがマージン付きで正しく計算されること（`hybrid_design.md` §3-2 `_narrow_search_space`） | `HybridOptimizer._narrow_search_space` | 既知の値を持つ 1 件の試行（上位 k=1）を入力したとき、`new_low = max(hp.low, val - margin*range)` が成立すること | `atol=1e-9` |
| ALG-HYB-11 | log スケールパラメータの絞り込みが log 空間でマージン計算されること（`hybrid_design.md` §3-2 `_narrow_search_space`） | `HybridOptimizer._narrow_search_space` | 既知の値を持つ 1 件の試行（上位 k=1）を入力したとき、log 空間でのマージン付き境界が正しく計算されること | `rtol=1e-9` |
| ALG-HYB-12 | `new_low >= new_high` となる場合に元の境界にフォールバックすること（`hybrid_design.md` §9 注意点） | `HybridOptimizer._narrow_search_space` | 上位試行がすべて同一値を持ち、`top_k_ratio` と `margin_ratio` が 0 の場合 | 絞り込み後の境界が元の `[hp.low, hp.high]` と一致すること |
| ALG-HYB-13 | `top_k_ratio` に従って上位試行のみが絞り込みに使用されること（`hybrid_design.md` §3-2 `_narrow_search_space`） | `HybridOptimizer._narrow_search_space` | 3 件の試行（objective: 1, 2, 3）、`top_k_ratio=0.4`（上位 2 件） | 絞り込みに objective=2, 3 の試行のみが使用されること（パラメータ範囲で確認） |

---

## 4. 振る舞いテスト：グラフ出力による確認（`check_hybrid_behavior.py`）

> 自動判定が困難な視覚的・定性的な性質を，グラフを出力して人間が確認する。
> 実際の PINN 学習と LLM 呼び出しを必要とするため、`uv run python tests/check_hybrid_behavior.py` で単独実行する。
> GEMINI_API_KEY が必要（`.env` ファイルに設定）。

### 確認項目一覧

| チェックID | 確認すべき観点 | 根拠（`hybrid_design.md`） | グラフの種類 | 合格の目安 |
|----------|-------------|--------------------------|------------|----------|
| BHV-HYB-01 | 全試行（Phase 1 + Phase 2）の累積最良目的関数値が単調非減少であること | `optimize()` の Phase 1 → Phase 2 の流れ | 折れ線グラフ（x 軸: 試行インデックス、y 軸: 累積最良値、Phase 境界を縦線で表示） | Phase 2 開始後に Phase 1 の最良値を更新または維持していること |
| BHV-HYB-02 | 絞り込まれた探索空間が元の探索空間より狭いこと（各パラメータの幅で確認） | `_narrow_search_space()` のマージン付き絞り込み | バーチャート（各パラメータの元幅 vs 絞り込み後の幅、log スケールパラメータは log 空間で比較） | 全パラメータで絞り込み後の幅が元の幅以下であること |
| BHV-HYB-03 | Phase 2（BO）の試行が Phase 1（LLM）の試行より高い目的関数値に収束する傾向があること | BO による局所収束の設計意図 | 散布図（x 軸: 試行インデックス、y 軸: 目的関数値、Phase 1/2 を色分け） | Phase 2 の後半試行の目的関数値が Phase 1 の平均を上回る傾向があること |

---

## 5. テスト実装のための MockChain 設計

```python
class MockChain(BaseChain):
    """テスト用の MockChain。呼び出し回数をカウントし、固定値を返す。"""

    def __init__(self, fixed_params: dict[str, float | int]) -> None:
        self.call_count = 0
        self._fixed_params = fixed_params

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        self.call_count += 1
        return LLMProposal(
            analysis_report="mock analysis",
            proposed_params=self._fixed_params,
            reasoning="mock reasoning",
        )
```

`MockChain.call_count` を ALG-HYB-01 の検証に使用する。`fixed_params` は `SearchSpace` の境界内の値を設定する。

---

## 6. 許容誤差の設定方針

| テストID | 許容誤差 | 設定根拠 |
|---------|---------|---------|
| ALG-HYB-10 | `atol=1e-9` | `_narrow_search_space` の計算は Python の `float` 演算のみで構成されるため、浮動小数点誤差（`≈1e-15`）に対して十分余裕を持たせた |
| ALG-HYB-11 | `rtol=1e-9` | `math.log` / `math.exp` の精度（倍精度 float）に対して余裕を持たせた |

---

## 7. pytest 設定（`pyproject.toml`）

既存の設定を継承する（新規マーカーの追加は不要）。

```toml
[tool.pytest.ini_options]
markers = [
    "algorithm: アルゴリズム実装の正確性テスト（計算の正しさ）",
    "theory: 論文の理論的性質が成立するかのテスト（数学的保証）",
]
```

---

## 8. テストファイル構成

| ファイル | 内容 |
|--------|------|
| `tests/test_hybrid_algorithm.py` | A-1（ALG-HYB-01〜07）および A-2（ALG-HYB-08〜13）の pytest テスト |
| `tests/check_hybrid_behavior.py` | B テスト（BHV-HYB-01〜03）のグラフ出力スクリプト（単独実行） |
