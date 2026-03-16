# PINNs — Burgers Equation

Physics-Informed Neural Networks (PINNs) による 1 次元粘性 Burgers 方程式の順問題・逆問題への応用実装。

**論文**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

---

## 概要

### 解く問題

**Burgers 方程式**（Eq. 1）

$$u_t + u \cdot u_x - \nu \cdot u_{xx} = 0, \quad x \in [-1, 1],\ t \in [0, 1]$$

$$u(0, x) = -\sin(\pi x), \quad u(t, \pm 1) = 0$$

| 問題 | 目的 |
|------|------|
| **順問題 (Forward)** | 既知の粘性係数 $\nu$ のもとで速度場 $u(t, x)$ を学習 |
| **逆問題 (Inverse)** | 観測データから $\nu$ を未知パラメータとして同定 |

### アルゴリズムの概要

1. **Algorithm 1**: 自動微分で PDE 残差 $f = u_t + u \cdot u_x - \nu \cdot u_{xx}$ を計算
2. **Algorithm 2** (順問題): 損失 $L = L_{\text{data}} + L_{\text{phys}}$ を Adam + L-BFGS で最小化
3. **Algorithm 3** (逆問題): $\theta$ と $\nu$ を同時に Adam で最適化

---

## プロジェクト構成

```
03-PINNs-Burgers/
├── src/PINNs_Burgers/
│   ├── api.py          # BurgersPINNSolver (Facade)
│   ├── config.py       # PDEConfig, NetworkConfig, TrainingConfig
│   ├── data.py         # BoundaryData, CollocationPoints
│   ├── network.py      # PINN (tanh 活性化, Xavier 初期化)
│   ├── residual.py     # PDEResidualComputer (Algorithm 1)
│   ├── loss.py         # LossFunction (L_data + L_phys)
│   ├── solver.py       # ForwardSolver, InverseSolver
│   └── results.py      # ForwardResult, InverseResult
├── src/opt_tool/                  # bo / opt_agent 共通基盤
│   ├── base.py         # BaseOptimizerConfig, BaseOptimizationResult, BaseOptimizer (Template Method)
│   ├── result.py       # TrialResult (Pydantic)
│   ├── space.py        # SearchSpace, HyperParameter
│   ├── objective.py    # ObjectiveFunction ABC
│   └── report_utils.py # 共通 Markdown ユーティリティ
├── src/bo/
│   ├── objective.py    # PINNObjectiveFunction, AccuracyObjective, AccuracySpeedObjective
│   ├── optimizer.py    # BayesianOptimizer (BoTorch GP + acquisition)
│   ├── result.py       # BOConfig, BOResult
│   └── report.py       # ReportGenerator (Markdown レポート)
├── src/opt_agent/
│   ├── config.py       # LLMConfig, LLMResult, LLMIterationMeta
│   ├── optimizer.py    # LLMOptimizer (Facade, BaseOptimizer 継承)
│   ├── chain.py        # BaseChain ABC, GeminiChain (LangChain + Gemini)
│   ├── prompt.py       # PromptBuilder ABC, MaximizeObjectivePromptBuilder, NarrowSearchSpacePromptBuilder
│   ├── proposal.py     # LLMProposal (Pydantic 構造化出力スキーマ)
│   └── report.py       # IterationReportWriter (逐次 Markdown レポート)
├── src/hybrid/
│   ├── optimizer.py    # HybridOptimizer (LLM 探索 + BO 収束の 2 フェーズ最適化)
│   └── result.py       # HybridResult
├── example/
│   ├── forward_problem.py      # 順問題の使用例
│   ├── inverse_problem.py      # 逆問題の使用例
│   ├── bo_forward.py           # ベイズ最適化によるハイパーパラメータ探索
│   ├── opt_agent_forward.py    # LLM エージェントによるハイパーパラメータ探索
│   └── hybrid_forward.py       # Hybrid（LLM + BO）によるハイパーパラメータ探索
├── tests/
│   ├── test_bo_algorithm.py       # BO アルゴリズム実装テスト (ALG-BO-*)
│   ├── test_bo_theory.py          # BO 理論的性質テスト (THR-BO-*)
│   ├── test_llm_algorithm.py      # LLM アルゴリズム実装テスト (ALG-LLM-*)
│   ├── test_llm_theory.py         # LLM 理論的性質テスト (THR-LLM-*)
│   ├── test_hybrid_algorithm.py   # Hybrid アルゴリズム実装テスト (ALG-HYB-*)
│   ├── check_bo_behavior.py       # BO 振る舞い確認スクリプト（グラフ出力）
│   ├── check_llm_behavior.py      # LLM 振る舞い確認スクリプト（グラフ出力）
│   └── check_hybrid_behavior.py   # Hybrid 振る舞い確認スクリプト（グラフ出力）
├── doc/
│   ├── paper_method.md      # 論文アルゴリズム抽出
│   ├── imp_design.md        # PINNs 実装設計書
│   ├── test_design.md       # PINNs テスト設計書
│   ├── bo_design.md         # BO モジュール設計書
│   └── llm_opt_design.md    # LLM 最適化モジュール設計書
└── data/
    └── burgers_shock.mat    # 参照解データ
```

---

## セットアップ

```bash
cd 03-PINNs-Burgers
uv sync
```

**依存ライブラリ**: `torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pydantic`, `botorch`, `langchain-google-genai`

LLM エージェント最適化を使用する場合は `.env` ファイルに以下を設定する。

```bash
GEMINI_API_KEY=<your-api-key>
GEMINI_MODEL_NAME=gemini-2.0-flash
```

---

## 使い方

### 順問題

既知の $\nu = 0.01/\pi$ で $u(t, x)$ を学習する。

```bash
uv run python example/forward_problem.py
```

出力先: `example/output/`

| ファイル | 内容 |
|---------|------|
| `forward_loss_curve.png` | 学習損失の推移（対数スケール） |
| `forward_solution_heatmap.png` | 予測解 vs 参照解のヒートマップ |
| `forward_error_heatmap.png` | 点ごとの絶対誤差 |

### 逆問題

観測データから $\nu$ を同定する（初期値 $\nu_0 = 0.005$、真値 $\nu^* = 0.01/\pi$）。

```bash
uv run python example/inverse_problem.py
```

出力先: `example/output/`

| ファイル | 内容 |
|---------|------|
| `inverse_loss_curve.png` | 学習損失の推移 |
| `inverse_nu_trajectory.png` | $\nu$ の推定軌跡と真値との比較 |

### ベイズ最適化によるハイパーパラメータ探索

BoTorch ベースのガウス過程 BO で、順問題に最適なネットワーク構造・学習率を自動探索する。

```bash
uv run python example/bo_forward.py
```

出力先: `example/bo_output/`

| ファイル | 内容 |
|---------|------|
| `bo_convergence.png` | 累積ベスト目的関数値の推移 |
| `bo_objective_scatter.png` | 各試行の目的関数値（Sobol 初期 vs BO 提案） |
| `bo_parallel_coords.png` | ハイパーパラメータの並行座標プロット |
| `bo_best_solution_heatmap.png` | 最良パラメータによる予測解 vs 参照解 |
| `bo_report.md` | Markdown サマリーレポート |

**探索空間**

| パラメータ | 範囲 | スケール |
|-----------|------|---------|
| `n_hidden_layers` | 2 〜 8 | linear |
| `n_neurons` | 10 〜 100 | linear |
| `lr` | 1e-4 〜 1e-2 | log |
| `epochs_adam` | 500 〜 5000 | linear |

**目的関数の切り替え**（Strategy パターン）

| クラス | 目的関数 | 用途 |
|--------|---------|------|
| `AccuracyObjective` | $-\text{rel\_l2\_error}$ | 精度のみ最大化 |
| `AccuracySpeedObjective` | $1 / (\text{rel\_l2\_error} \times t)$ | 精度と学習速度を同時に最大化 |

`bo_forward.py` の `objective = AccuracyObjective(...)` を `AccuracySpeedObjective(...)` に変更するだけで切り替えられる。

### LLM エージェントによるハイパーパラメータ探索

Google Gemini（LangChain 経由）を用いた LLM エージェントが、試行履歴を分析して次の探索点を自然言語で推論しながらハイパーパラメータを最適化する。

```bash
uv run python example/opt_agent_forward.py
```

出力先: `example/opt_agent_output/`

| ファイル | 内容 |
|---------|------|
| `opt_agent_convergence.png` | 累積ベスト目的関数値の推移 |
| `opt_agent_objective_scatter.png` | 各試行の目的関数値（Sobol 初期 vs LLM 提案） |
| `opt_agent_parallel_coords.png` | ハイパーパラメータの並行座標プロット |
| `opt_agent_best_solution_heatmap.png` | 最良パラメータによる予測解 vs 参照解 |
| `opt_agent_report.md` | LLM の分析・推論を含む逐次 Markdown レポート |

**最適化フロー**

1. **Phase 1（Sobol 初期探索）**: 準乱数列で `n_initial` 点をサンプリングして評価
2. **Phase 2（LLM 誘導探索）**: 試行履歴全体を LLM に提示し、分析・推論に基づく次点を提案させて評価を繰り返す

**LLMConfig**

| パラメータ | デフォルト | 意味 |
|-----------|-----------|------|
| `n_initial` | 5 | Phase 1 の初期サンプル数 |
| `n_iterations` | 20 | Phase 2 の LLM 提案回数 |
| `seed` | 42 | Sobol 乱数シード（再現性） |

**プロンプトの切り替え**（Strategy パターン）

`GeminiChain` は `PromptBuilder` を DI で受け取り、LLM への指示内容を差し替えられる。

| クラス | 指示内容 | 用途 |
|--------|---------|------|
| `MaximizeObjectivePromptBuilder` | 目的関数の最大値を目指す | `LLMOptimizer` のスタンドアロン利用 |
| `NarrowSearchSpacePromptBuilder` | 探索空間を広く探索し有望領域を特定する | `HybridOptimizer` の Phase 1 |

`GeminiChain` は `prompt_builder` 省略時に `MaximizeObjectivePromptBuilder` を使用する。

---

### Hybrid（LLM + BO）によるハイパーパラメータ探索

LLM エージェントによる広域探索（Phase 1）と、上位試行から絞り込んだ探索空間での BO（Phase 2）を組み合わせた 2 フェーズ最適化。

```bash
uv run python example/hybrid_forward.py
```

出力先: `example/hybrid_output/`

| ファイル | 内容 |
|---------|------|
| `hybrid_convergence.png` | 累積ベスト目的関数値の推移（両フェーズ） |
| `hybrid_objective_scatter.png` | 全試行の目的関数値（フェーズ別に色分け） |
| `hybrid_space_comparison.png` | 元の探索空間 vs 絞り込み後の探索空間の比較 |
| `hybrid_best_solution_heatmap.png` | 最良パラメータによる予測解 vs 参照解 |

**最適化フロー**

1. **Phase 1（LLM 広域探索）**: `NarrowSearchSpacePromptBuilder` を使った LLM エージェントが元の探索空間を広く探索し有望領域を特定する
2. **探索空間の絞り込み**: Phase 1 の上位 `top_k_ratio` 割の試行をもとに各パラメータの新しい上下限を算出する（`margin_ratio` で余裕を付加）
3. **Phase 2（BO 精密探索）**: 絞り込まれた探索空間で BoTorch ベースの GP-BO を実行し、最適点へ収束させる

**HybridOptimizer のパラメータ**

| パラメータ | デフォルト | 意味 |
|-----------|-----------|------|
| `n_llm_iterations` | 10 | Phase 1 の LLM 提案回数 |
| `top_k_ratio` | 0.3 | 探索空間絞り込みに使う上位試行の割合 |
| `margin_ratio` | 0.1 | 上下限に加える余裕（元範囲に対する比率） |

---

### Python API

```python
from PINNs_Burgers import (
    BurgersPINNSolver,
    PDEConfig, NetworkConfig, TrainingConfig,
    BoundaryData, CollocationPoints,
)

solver = BurgersPINNSolver(
    pde_config=PDEConfig(nu=0.01 / 3.14159, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0),
    network_config=NetworkConfig(n_hidden_layers=5, n_neurons=20),
    training_config=TrainingConfig(n_u=100, n_f=10_000, lr=1e-3, epochs_adam=2000, epochs_lbfgs=50),
)

# 順問題
result = solver.solve_forward(boundary_data, collocation)  # -> ForwardResult

# 逆問題
result = solver.solve_inverse(observations, collocation, nu_init=0.005)  # -> InverseResult
print(result.nu)  # 推定された粘性係数
```

---

## テスト

```bash
# 単体テスト（全件）
uv run pytest tests/

# アルゴリズム実装テストのみ
uv run pytest tests/ -m algorithm

# 理論的性質テストのみ
uv run pytest tests/ -m theory

# 振る舞い確認（グラフ出力・目視確認用）
uv run python tests/check_bo_behavior.py
uv run python tests/check_llm_behavior.py
uv run python tests/check_hybrid_behavior.py
```

---

## 実装上のポイント

| 項目 | 内容 |
|------|------|
| ネットワーク構造 | 5 隠れ層 × 20 ニューロン、活性化関数 tanh（論文 Section 6.1） |
| 重み初期化 | Xavier 一様初期化（論文 Section 4） |
| 順問題の最適化 | Adam (2000 epoch) → L-BFGS (50 epoch)（Algorithm 2） |
| 逆問題の最適化 | $\theta$ と $\nu$ を Adam で同時最適化（Algorithm 3） |
| 微分計算 | PyTorch 自動微分（`create_graph=True` による高階微分） |
| 訓練データ | $N_u = 100$（境界・初期条件）、$N_f = 10{,}000$（コロケーション点） |
