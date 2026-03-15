# 実装設計書：LLM-based Hyperparameter Optimizer

## 1. 設計の概要

**手法名**：LLM-based Hyperparameter Optimization（LangChain + Gemini）
**参照ファイル**：本ドキュメントが設計の起点（BO モジュールと比較実験するための新規モジュール）

**設計の方針**
LLM による最適化ループを BOモジュール（`src/bo/`）と同一のインターフェースで提供し、比較実験コードの変更を最小にする。
LangChain の呼び出しを `BaseChain` 抽象クラスで隠蔽し、テスト時は `MockChain` に差し替えられる構造とする（依存性逆転の原則）。
設定値・推論メタ情報・最適化結果はすべてイミュータブルなデータクラスで管理し、ループ内の副作用を防ぐ。

---

## 2. パブリック API（アルゴリズムステップに基づく）

```python
class LLMOptimizer:
    """
    LLM（Gemini）を用いてハイパーパラメータを最適化するオプティマイザ。
    BayesianOptimizer と同一のインターフェースを提供し、比較実験を容易にする。
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        config: LLMConfig = LLMConfig(),
    ) -> None:
        """
        オプティマイザを初期化する。
        .env から GEMINI_API_KEY / GEMINI_MODEL_NAME を読み込み、
        GeminiChain を構築する。
        """
        ...

    def optimize(self) -> LLMResult:
        """
        最適化ループ全体を実行する。
        Phase 1（Sobol 初期探索）→ Phase 2（LLM 主導探索）→ Phase 3（結果集約）の順に実行する。
        """
        ...
```

---

## 3. クラス設計

### 3-1. クラス一覧

| クラス名 | 種別 | 責務（単一責任の原則） | 対応するアルゴリズムの概念 |
|---------|------|---------------------|--------------------------|
| `LLMConfig` | frozen dataclass | LLM 最適化の設定値保持 | Phase 1/2 の制御パラメータ |
| `LLMIterationMeta` | frozen dataclass | 1 イテレーション分の LLM 推論メタ情報保持 | Phase 2 の LLM 出力 |
| `LLMResult` | frozen dataclass | 最適化全体の結果集約 | Phase 3 の出力 |
| `LLMProposal` | Pydantic BaseModel | LLM 構造化出力スキーマの定義と検証 | Phase 2 の LLM 応答パース |
| `BaseChain` | 抽象クラス | LLM 呼び出しの抽象インターフェース定義 | Phase 2 の LLM 推論 |
| `GeminiChain` | 具象クラス | LangChain を使った Gemini API 呼び出し実装 | Phase 2 の LLM 推論（本番） |
| `PromptBuilder` | 具象クラス（静的メソッド） | 探索履歴をプロンプト文字列に変換 | Phase 2 のプロンプト構築 |
| `LLMOptimizer` | 具象クラス | 最適化ループ全体のオーケストレーション | Phase 1/2/3 の実行制御 |

---

### 3-2. 各クラスの定義

#### `LLMConfig`

**種別**：frozen dataclass
**責務**：LLM 最適化ループの制御パラメータを保持する
**対応するアルゴリズムの概念**：Phase 1/2 の実行回数・乱数シード制御

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMConfig:
    n_initial: int = 5         # Sobol 初期探索点数（BO の BOConfig.n_initial と揃える）
    n_iterations: int = 20     # LLM 主導イテレーション回数（BO の BOConfig.n_iterations と揃える）
    seed: int = 42             # Sobol サンプリングの乱数シード
```

**SOLIDチェック**
- S: 設定値の保持のみを責務とする
- O: フィールド追加で拡張可能（既存コードの変更不要）
- D: 具体値を保持するため抽象への依存は不要

---

#### `LLMIterationMeta`

**種別**：frozen dataclass
**責務**：LLM 主導フェーズの 1 イテレーション分の推論メタ情報を保持する
**対応するアルゴリズムの概念**：Phase 2 の LLM 出力（分析レポート・提案・理由）

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMIterationMeta:
    iteration_id: int                          # イテレーション番号（0 始まり）
    analysis_report: str                       # LLM が生成した現状分析（自然言語）
    proposed_params: dict[str, float | int]    # LLM が提案したパラメータ（実スケール）
    reasoning: str                             # 提案理由（自然言語）
```

**SOLIDチェック**
- S: 推論メタ情報の保持のみを責務とする
- O: フィールド追加で拡張可能

---

#### `LLMResult`

**種別**：frozen dataclass
**責務**：最適化全体の結果を集約する
**対応するアルゴリズムの概念**：Phase 3 の出力（BO の `BOResult` に対応）

```python
from dataclasses import dataclass
from src.bo.result import TrialResult  # 既存データクラスを流用

@dataclass(frozen=True)
class LLMResult:
    trials: list[TrialResult]                   # 全トライアル（初期 + LLM 主導）
    best_params: dict[str, float | int]         # 最良ハイパーパラメータ
    best_objective: float                        # 最良目的関数値
    best_trial_id: int                           # 最良トライアル ID
    llm_config: LLMConfig                        # 使用した設定
    objective_name: str                          # 目的関数名
    iteration_metas: list[LLMIterationMeta]      # LLM 主導フェーズの推論メタ情報
```

**SOLIDチェック**
- S: 結果の集約・保持のみを責務とする
- O: フィールド追加で拡張可能（レポート生成は別クラスが担う）

---

#### `LLMProposal`

**種別**：Pydantic BaseModel
**責務**：LLM の構造化出力スキーマを定義し、レスポンスのバリデーションを行う
**対応するアルゴリズムの概念**：Phase 2 の LLM 応答パース

```python
from pydantic import BaseModel, Field

class LLMProposal(BaseModel):
    analysis_report: str = Field(
        description="現在の探索状況の分析（自然言語）"
    )
    proposed_params: dict[str, float | int] = Field(
        description="次に探索するパラメータ値（実スケール）"
    )
    reasoning: str = Field(
        description="提案パラメータの選択理由（自然言語）"
    )
```

**SOLIDチェック**
- S: LLM 出力スキーマの定義とバリデーションのみを責務とする
- O: フィールド追加で LLM に要求する情報を拡張可能

---

#### `BaseChain`

**種別**：抽象クラス（ABC）
**責務**：LLM 呼び出しのインターフェースを定義する
**対応するアルゴリズムの概念**：Phase 2 の LLM 推論呼び出し

```python
from abc import ABC, abstractmethod
from src.bo.space import SearchSpace
from src.bo.result import TrialResult

class BaseChain(ABC):
    @abstractmethod
    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """
        現在の探索履歴を受け取り、次の探索点と分析レポートを返す。
        """
        ...
```

**SOLIDチェック**
- S: LLM 呼び出しのインターフェース定義のみを責務とする
- O: 新しい LLM（Claude など）は新しい具象クラスを追加するだけで対応可能
- I: 単一メソッド `invoke` のみを要求（最小インターフェース）
- D: `LLMOptimizer` はこの抽象に依存し、具体実装（`GeminiChain`）に依存しない

---

#### `GeminiChain`

**種別**：具象クラス（`BaseChain` のサブクラス）
**責務**：LangChain を使って Gemini API を呼び出し、`LLMProposal` を返す
**対応するアルゴリズムの概念**：Phase 2 の LLM 推論（本番実装）

```python
class GeminiChain(BaseChain):
    def __init__(self, model_name: str, api_key: str) -> None:
        """
        ChatGoogleGenerativeAI と with_structured_output を使って
        LangChain LCEL チェーンを構築する。
        """
        ...

    def invoke(
        self,
        search_space: SearchSpace,
        trials: list[TrialResult],
        objective_name: str,
        iteration_id: int,
    ) -> LLMProposal:
        """
        システムプロンプト（問題設定）+ ユーザーメッセージ（探索履歴）を構築し
        Gemini API に送信する。レスポンスを LLMProposal として返す。
        """
        ...
```

**LangChain 構成**:
- モデル: `ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)`
- 出力: `llm.with_structured_output(LLMProposal)` で JSON 構造化出力を取得
- プロンプト: `ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])`
- チェーン: `prompt | llm.with_structured_output(LLMProposal)`（LCEL）

**SOLIDチェック**
- S: Gemini API 呼び出しのみを責務とする
- L: `BaseChain` の契約（`invoke` のシグネチャ）を守る
- D: `PromptBuilder`（具象）に依存するが、テスト時は `MockChain` に差し替え可能

---

#### `PromptBuilder`

**種別**：具象クラス（静的メソッドのみ）
**責務**：探索空間・履歴・目的関数の情報をプロンプト文字列に変換する
**対応するアルゴリズムの概念**：Phase 2 のプロンプト構築

```python
class PromptBuilder:
    @staticmethod
    def build_system_prompt(
        search_space: SearchSpace,
        objective_name: str,
    ) -> str:
        """
        問題設定を記述したシステムプロンプトを構築する。
        含む情報: タスク説明、探索空間の各パラメータ（名前・型・範囲・スケール）、
        目的関数の説明、出力フォーマットの指定。
        """
        ...

    @staticmethod
    def build_human_prompt(
        trials: list[TrialResult],
        iteration_id: int,
    ) -> str:
        """
        現在の探索履歴を記述したユーザーメッセージを構築する。
        含む情報: 現在のイテレーション番号、全トライアル結果テーブル、現在の最良点。
        """
        ...
```

**SOLIDチェック**
- S: プロンプト文字列の構築のみを責務とする（API 呼び出しは `GeminiChain` が担う）
- O: プロンプト内容の変更はこのクラスの変更だけで済む

---

#### `LLMOptimizer`

**種別**：具象クラス
**責務**：最適化ループ全体のオーケストレーションと依存クラスの組み立て（Facade）
**対応するアルゴリズムの概念**：Phase 1/2/3 の実行制御

```python
class LLMOptimizer:
    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        config: LLMConfig = LLMConfig(),
        chain: BaseChain | None = None,  # None の場合 .env から GeminiChain を自動構築
    ) -> None: ...

    def optimize(self) -> LLMResult:
        """
        Phase 1: run_initial_exploration  — Sobol 準乱数で n_initial 点を評価
        Phase 2: run_llm_guided_search    — LLM に n_iterations 回提案させて評価
        Phase 3: aggregate_results        — 最良点を選択して LLMResult を生成
        """
        ...
```

**SOLIDチェック**
- S: 最適化ループの制御と依存組み立てのみを責務とする
- O: 新しいフェーズは新メソッドの追加で対応
- D: `BaseChain`（抽象）に依存（`GeminiChain` の具体実装には依存しない）

---

## 4. デザインパターン

| パターン名 | 適用箇所（クラス名） | 採用理由 |
|-----------|-------------------|---------|
| Strategy | `BaseChain`, `GeminiChain` | LLM の種類をテスト時に差し替えられるようにするため |
| Facade | `LLMOptimizer` | 複数の内部クラスを単一の `optimize()` API に集約し、利用側の複雑性を隠蔽するため |
| Value Object（データクラス） | `LLMConfig`, `LLMIterationMeta`, `LLMResult` | 設定・推論結果は不変であるべきで、データの保持が責務であることを明示するため |

### Strategy パターンの詳細

**適用箇所**：`BaseChain` → `GeminiChain`
**採用理由**：LLM の種類（Gemini / Claude / Mock）を `LLMOptimizer` から切り離すことで、テスト時に API を呼び出さない `MockChain` に差し替えられる。Gemini から別モデルへの変更も `GeminiChain` の置き換えのみで対応できる。
**代替案と却下理由**：`LLMOptimizer` 内に直接 Gemini 呼び出しを埋め込む案は、単体テストで Gemini API への実呼び出しが発生しテストが不安定になるため却下した。

### Facade パターンの詳細

**適用箇所**：`LLMOptimizer`
**採用理由**：`GeminiChain`・`PromptBuilder`・`SearchSpace`・`ObjectiveFunction` を利用者が直接組み立てる必要がなく、`LLMOptimizer(search_space, objective).optimize()` の 1 呼び出しで完結させる。BO モジュールの `BayesianOptimizer` と同一の使い方ができるため比較実験コードが簡潔になる。
**代替案と却下理由**：依存性注入のみで Facade を設けない案は、利用者が内部クラスの組み立て順序を知る必要が生じるため却下した。

---

## 5. クラス図（Mermaid）

```mermaid
classDiagram
    class LLMOptimizer {
        +optimize() LLMResult
        -_run_initial_exploration() list~TrialResult~
        -_run_llm_guided_search(trials) list~TrialResult~
        -_aggregate_results(trials, metas) LLMResult
    }

    class BaseChain {
        <<abstract>>
        +invoke(search_space, trials, objective_name, iteration_id)* LLMProposal
    }

    class GeminiChain {
        +invoke(search_space, trials, objective_name, iteration_id) LLMProposal
    }

    class PromptBuilder {
        +build_system_prompt(search_space, objective_name)$ str
        +build_human_prompt(trials, iteration_id)$ str
    }

    class LLMConfig {
        +n_initial int
        +n_iterations int
        +seed int
    }

    class LLMIterationMeta {
        +iteration_id int
        +analysis_report str
        +proposed_params dict
        +reasoning str
    }

    class LLMResult {
        +trials list~TrialResult~
        +best_params dict
        +best_objective float
        +best_trial_id int
        +llm_config LLMConfig
        +objective_name str
        +iteration_metas list~LLMIterationMeta~
    }

    class LLMProposal {
        +analysis_report str
        +proposed_params dict
        +reasoning str
    }

    class SearchSpace {
        +sample_sobol(n, seed) Tensor
        +from_tensor(x) dict
    }

    class ObjectiveFunction {
        <<abstract>>
        +__call__(params, trial_id, is_initial) TrialResult
    }

    class TrialResult {
        +trial_id int
        +params dict
        +objective float
        +rel_l2_error float
        +elapsed_time float
        +is_initial bool
    }

    LLMOptimizer --> BaseChain : depends on
    LLMOptimizer --> SearchSpace : uses
    LLMOptimizer --> ObjectiveFunction : uses
    LLMOptimizer --> LLMConfig : uses
    LLMOptimizer ..> LLMResult : produces
    LLMOptimizer ..> LLMIterationMeta : produces

    BaseChain <|-- GeminiChain : implements
    GeminiChain --> PromptBuilder : uses
    GeminiChain ..> LLMProposal : produces

    LLMResult --> LLMConfig : contains
    LLMResult --> TrialResult : contains
    LLMResult --> LLMIterationMeta : contains
    LLMIterationMeta --> LLMProposal : derived from
```

---

## 6. 依存ライブラリ

| ライブラリ | 用途 | 対応する処理 |
|-----------|------|------------|
| `langchain-google-genai` | `ChatGoogleGenerativeAI` によるGemini との通信 | Phase 2 LLM 呼び出し |
| `langchain-core` | `ChatPromptTemplate`, LCEL（`|` 演算子）, `with_structured_output` | Phase 2 チェーン構築・プロンプト構築 |
| `python-dotenv` | `.env` ファイルからの `GEMINI_API_KEY` / `GEMINI_MODEL_NAME` 読み込み | `LLMOptimizer.__init__` での環境変数取得 |
| `pydantic` | `LLMProposal` の構造化出力スキーマ定義とバリデーション | Phase 2 LLM 応答パース |

---

## 7. 実装上の注意点

| 項目 | 内容 | 対応するクラス / メソッド |
|------|------|----------------------|
| パラメータのクランプ | LLM が探索範囲外の値を提案した場合、`SearchSpace` の境界値にクランプしてから評価する | `LLMOptimizer._run_llm_guided_search` |
| 整数パラメータの丸め | `n_hidden_layers`, `n_neurons`, `epochs_adam` は `int()` で丸める | `LLMOptimizer._run_llm_guided_search` |
| API エラーハンドリング | Gemini API 呼び出し失敗時は最大 3 回リトライし、失敗した場合は `RuntimeError` を送出する | `GeminiChain.invoke` |
| テスト時の差し替え | `LLMOptimizer(chain=MockChain())` のように `chain` 引数で差し替える | `LLMOptimizer.__init__` |
| 環境変数の未設定 | `GEMINI_API_KEY` が未設定の場合は `ValueError` を送出する | `LLMOptimizer.__init__` |

---

## 8. `.env` 仕様

```dotenv
GEMINI_API_KEY=<your_api_key>
GEMINI_MODEL_NAME=gemini-2.0-flash
```

`python-dotenv` の `load_dotenv()` で読み込み、`GeminiChain.__init__` に渡す。

---

## 9. BO との比較設計

LLM 最適化は以下の条件を BO と完全に揃えて公平な比較を行う。

| 条件 | 値 |
|------|----|
| 初期探索点 | Sobol（`n_initial=5`, `seed=42`） |
| イテレーション数 | `n_iterations=20` |
| 目的関数 | `AccuracyObjective` または `AccuracySpeedObjective` |
| 探索空間 | 4次元（`n_hidden_layers`, `n_neurons`, `lr`, `epochs_adam`） |
| 結果データ構造 | `TrialResult`（BO と共通） |
