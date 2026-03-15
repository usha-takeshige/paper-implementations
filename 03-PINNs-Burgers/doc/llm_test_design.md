# テスト設計書：LLM-based Hyperparameter Optimizer

## 1. テスト設計の方針

**参照ファイル**：`/doc/llm_opt_design.md`

設計の中心的な検証観点は 3 点ある。第一に、`LLMOptimizer` が `MockChain` 経由で Gemini API を呼ばずに動作し、ループの各フェーズ（Sobol 初期探索 → LLM 主導探索 → 結果集約）が正しく連携するか。第二に、`best_trial_id` の整合性など結果集約の正確性が成立するか。第三に、LLM が探索範囲外の値を提案した場合のクランプ・丸め処理が確実に機能するかを重点的に検証する。

---

## 2. 単体テスト：A-1 アルゴリズム実装テスト（`test_llm_algorithm.py`）

> 各クラス・メソッドが計算として正しく動作するかを検証する。
> `@pytest.mark.algorithm` マーカーを付与する。

### テストケース一覧

| テストID | 対象クラス / メソッド | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|---------------------|---------|---------|-------------------|
| ALG-LLM-01 | `LLMConfig` | デフォルト値が仕様通りに設定される | 引数なしでインスタンス化 | `n_initial=5`, `n_iterations=20`, `seed=42` |
| ALG-LLM-02 | `LLMConfig` | frozen により変更不可 | `config.n_initial = 99` を代入 | `FrozenInstanceError` / `dataclasses.FrozenInstanceError` を送出 |
| ALG-LLM-03 | `LLMIterationMeta` | 全フィールドが正しく格納される | 任意の値でインスタンス化 | 各フィールドが入力値と一致 |
| ALG-LLM-04 | `LLMIterationMeta` | frozen により変更不可 | `meta.iteration_id = 99` を代入 | `FrozenInstanceError` を送出 |
| ALG-LLM-05 | `LLMResult` | 全フィールドが正しく格納される | 任意の値でインスタンス化 | 各フィールドが入力値と一致 |
| ALG-LLM-06 | `LLMResult` | frozen により変更不可 | `result.best_objective = 99.0` を代入 | `FrozenInstanceError` を送出 |
| ALG-LLM-07 | `LLMProposal` | 正しい型の値でインスタンス化できる | 有効な `analysis_report`, `proposed_params`, `reasoning` を渡す | インスタンス化成功、各フィールドが入力値と一致 |
| ALG-LLM-08 | `LLMProposal` | 必須フィールド欠落で ValidationError を送出する | `analysis_report` を省略 | `pydantic.ValidationError` を送出 |
| ALG-LLM-09 | `PromptBuilder.build_system_prompt` | 出力にすべてのパラメータ名が含まれる | 4 次元 `SearchSpace` と `objective_name` を渡す | `"n_hidden_layers"`, `"n_neurons"`, `"lr"`, `"epochs_adam"` がすべて含まれる |
| ALG-LLM-10 | `PromptBuilder.build_system_prompt` | 出力に各パラメータの探索範囲が含まれる | 4 次元 `SearchSpace` を渡す | 各パラメータの `low`, `high` 値が文字列として含まれる |
| ALG-LLM-11 | `PromptBuilder.build_human_prompt` | 出力に全トライアルの結果が含まれる | 3 件の `TrialResult` リストと `iteration_id=2` を渡す | `trial_id` 0, 1, 2 の objective 値がすべて含まれる |
| ALG-LLM-12 | `PromptBuilder.build_human_prompt` | 出力に現在の最良点情報が含まれる | 複数トライアルを渡す | 最大 objective を持つトライアルの `trial_id` が含まれる |
| ALG-LLM-13 | `LLMOptimizer.__init__` | `GEMINI_API_KEY` 未設定時に `ValueError` を送出 | `chain=None` かつ環境変数未設定 | `ValueError` を送出 |
| ALG-LLM-14 | `LLMOptimizer.__init__` | `MockChain` を `chain` 引数で注入できる | `chain=MockChain()` を渡す | 例外なく初期化される |
| ALG-LLM-15 | `LLMOptimizer.optimize` | Phase 1 で `n_initial` 件のトライアルが `is_initial=True` で生成される | `MockChain`, `config.n_initial=3` | `result.trials[:3]` の `is_initial` がすべて `True` |
| ALG-LLM-16 | `LLMOptimizer.optimize` | Phase 2 で `n_iterations` 件のトライアルが `is_initial=False` で生成される | `MockChain`, `config.n_iterations=2` | `result.trials[3:]` の `is_initial` がすべて `False` |
| ALG-LLM-17 | `LLMOptimizer.optimize` | 全トライアル数が `n_initial + n_iterations` 件になる | `config.n_initial=3`, `config.n_iterations=2` | `len(result.trials) == 5` |
| ALG-LLM-18 | `LLMOptimizer.optimize` | 探索範囲外の提案がクランプされる | `MockChain` が `lr=100.0`（上限 `1e-2` 超）を提案 | 評価時の `params["lr"] <= 1e-2` が保証される |
| ALG-LLM-19 | `LLMOptimizer.optimize` | 整数パラメータが丸められる | `MockChain` が `n_hidden_layers=3.7` を提案 | 評価時の `params["n_hidden_layers"] == 4`（または `3`） かつ `isinstance(..., int)` |
| ALG-LLM-20 | `LLMOptimizer.optimize` | `LLMResult.best_trial_id` が最大 objective のトライアル ID を指す | `MockChain` + 単調増加する objective を返す `MockObjective` | `result.best_trial_id == len(result.trials) - 1` |
| ALG-LLM-21 | `LLMOptimizer.optimize` | `iteration_metas` の件数が `n_iterations` と一致する | `config.n_iterations=2` | `len(result.iteration_metas) == 2` |
| ALG-LLM-22 | `LLMOptimizer.optimize` | `trial_id` が 0 始まりで連続する | `config.n_initial=3`, `config.n_iterations=2` | `[t.trial_id for t in result.trials] == [0, 1, 2, 3, 4]` |
| ALG-LLM-23 | `LLMOptimizer.optimize` | `MockChain.invoke` がちょうど `n_iterations` 回呼ばれる | `InvocationCountChain` を注入, `n_iterations=3` | `chain.call_count == 3` |
| ALG-LLM-24 | `LLMOptimizer.optimize` | `LLMResult.objective_name` が目的関数名と一致する | `objective.name = "mock"` | `result.objective_name == "mock"` |

---

## 3. 単体テスト：A-2 理論的性質テスト（`test_llm_theory.py`）

> 設計上の保証として述べられている性質が実装において成立するかを検証する。
> `@pytest.mark.theory` マーカーを付与する。
> 各テストには根拠となる設計上の保証を明記する。

### テストケース一覧

| テストID | 設計上の性質（根拠） | 対象クラス / メソッド | 検証内容 | 許容誤差 |
|---------|------------------|--------------------|---------| --------|
| THR-LLM-01 | Phase 1 は Sobol 準乱数で `seed` 固定のため再現性がある（`llm_opt_design.md` § LLMConfig） | `LLMOptimizer.optimize` | 同一 `seed` で 2 回実行した Phase 1 の `params` が完全一致する | 完全一致 |
| THR-LLM-02 | `LLMResult.best_objective` は全トライアルの最大値（`llm_opt_design.md` § LLMResult） | `LLMOptimizer.optimize` | `result.best_objective == max(t.objective for t in result.trials)` が成立する | 完全一致 |
| THR-LLM-03 | `LLMResult.best_params` は `best_trial_id` のトライアルのパラメータ（`llm_opt_design.md` § LLMResult） | `LLMOptimizer.optimize` | `result.best_params == result.trials[result.best_trial_id].params` が成立する | 完全一致 |
| THR-LLM-04 | クランプ後のパラメータは探索空間の境界内に収まる（`llm_opt_design.md` § 7） | `LLMOptimizer.optimize` | Phase 2 の全 `TrialResult.params` について `low <= v <= high` が各次元で成立する | 完全一致 |
| THR-LLM-05 | `PromptBuilder` のプロンプトは `SearchSpace` のすべてのパラメータ情報を含む（`llm_opt_design.md` § PromptBuilder） | `PromptBuilder.build_system_prompt` | 探索空間の全パラメータ名、型、上下限が出力文字列に含まれる | 文字列包含チェック |
| THR-LLM-06 | `LLMIterationMeta.proposed_params` は対応する `TrialResult` の評価点と一致する（クランプ・丸め後）（`llm_opt_design.md` § 7） | `LLMOptimizer.optimize` | `meta.proposed_params` をクランプ・丸めした結果が `trials[n_initial + meta.iteration_id].params` と一致する | `atol=1e-9` |

---

## 4. 振る舞いテスト：グラフ出力による確認（`check_llm_behavior.py`）

> 自動判定が困難な視覚的・定性的な性質を、グラフを出力して人間が確認する。
> `assert` を使わず、グラフタイトルに確認観点を表示する。
> 実行方法：`uv run python tests/check_llm_behavior.py`

### 確認項目一覧

| チェックID | 確認すべき観点 | 設計上の根拠 | グラフの種類 | 合格の目安 |
|----------|-------------|-----------|------------|----------|
| BHV-LLM-01 | LLM 最適化の収束曲線が単調非減少である | `llm_opt_design.md` § LLMResult | 折れ線グラフ（x: iteration, y: cumulative best objective） | Phase 2 に入ってから目的関数値が改善または横ばいの傾向がある |
| BHV-LLM-02 | LLM の提案点がパラメータ空間内に分布している | `llm_opt_design.md` § 7「パラメータのクランプ」 | 散布図（Phase 1: Sobol 点, Phase 2: LLM 提案点を異なるマーカーで表示） | Phase 2 の点が探索空間内に収まっており、Sobol 点と異なる分布を示す |
| BHV-LLM-03 | LLM の分析レポートと提案理由が各イテレーションで生成されている | `llm_opt_design.md` § LLMIterationMeta | テキスト出力（各イテレーションの `analysis_report` と `reasoning` を番号付きで表示） | 各イテレーションで空でない文字列が出力されており、内容が文脈に沿っている |

---

## 5. 許容誤差の設定方針

| テストID | 許容誤差 | 設定根拠 |
|---------|---------|---------|
| THR-LLM-06 | `atol=1e-9` | クランプ・丸め処理は整数演算 + float64 範囲内の演算のため、実質的に誤差なし。ただし float 比較のため微小な許容値を設ける |
| その他 | 完全一致 | frozen dataclass のフィールド比較および文字列包含チェックは数値誤差を含まない |

---

## 6. テスト用フィクスチャ設計

### `MockChain`

```python
class MockChain(BaseChain):
    """固定値の LLMProposal を返す MockChain。Gemini API を呼ばない。"""

    def __init__(self, proposals: list[dict] | None = None) -> None:
        """
        proposals: 各 invoke 呼び出しで返すパラメータのリスト。
        None の場合はすべて探索空間の中点を返す。
        """
        ...

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """インデックスに対応する固定 proposal を返す。"""
        ...
```

### `InvocationCountChain`

```python
class InvocationCountChain(BaseChain):
    """invoke の呼び出し回数を記録する MockChain。ALG-LLM-23 向け。"""

    call_count: int = 0

    def invoke(self, ...) -> LLMProposal:
        self.call_count += 1
        ...
```

### `make_search_space`

```python
def make_search_space() -> SearchSpace:
    """PINNs ハイパーパラメータ最適化の 4 次元 SearchSpace を返す。"""
    return SearchSpace(parameters=[
        HyperParameter(name="n_hidden_layers", param_type="int",   low=2,    high=8),
        HyperParameter(name="n_neurons",       param_type="int",   low=10,   high=100),
        HyperParameter(name="lr",              param_type="float", low=1e-4, high=1e-2, log_scale=True),
        HyperParameter(name="epochs_adam",     param_type="int",   low=500,  high=5_000),
    ])
```

### `MockObjectiveFunction`

```python
class MockObjectiveFunction:
    """決定論的な Mock 目的関数。PINN 学習を実行せずに TrialResult を返す。"""

    @property
    def name(self) -> str:
        return "mock"

    def __call__(self, params: dict, trial_id: int, is_initial: bool) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            params=params,
            objective=float(trial_id + 1),
            rel_l2_error=1.0 / (trial_id + 1),
            elapsed_time=1.0,
            is_initial=is_initial,
        )
```

---

## 7. pytest 設定（`pyproject.toml`）

```toml
[tool.pytest.ini_options]
markers = [
    "algorithm: アルゴリズム実装の正確性テスト（計算の正しさ）",
    "theory: 設計上保証された性質が成立するかのテスト",
]
```

---

## 8. テストファイル構成

```
tests/
├── test_llm_algorithm.py   # A-1: ALG-LLM-01 〜 ALG-LLM-24（@pytest.mark.algorithm）
├── test_llm_theory.py      # A-2: THR-LLM-01 〜 THR-LLM-06（@pytest.mark.theory）
└── check_llm_behavior.py   # B: BHV-LLM-01 〜 BHV-LLM-03（手動確認・グラフ出力）
```
