# PINNs Burgers equation 
Physics Informed Neural Networks（PINNs）を用いて、Burgers Equationのサロゲートモデルを作るプロジェクトです。

# ディレクトリ構成
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
├── src/bo/
│   ├── objective.py    # ObjectiveFunction ABC, AccuracyObjective, AccuracySpeedObjective
│   ├── optimizer.py    # BayesianOptimizer (BoTorch GP + acquisition)
│   ├── result.py       # BOResult, TrialResult, BOConfig
│   ├── report.py       # ReportGenerator (Markdown レポート)
│   └── search_space.py # SearchSpace, HyperParameter
├── src/opt_agent/
│   ├── config.py       # LLMConfig, LLMResult, LLMIterationMeta
│   ├── optimizer.py    # LLMOptimizer (Facade)
│   ├── chain.py        # BaseChain ABC, GeminiChain (LangChain + Gemini)
│   ├── prompt.py       # PromptBuilder (システム・ヒュープロンプト構築)
│   ├── proposal.py     # LLMProposal (Pydantic 構造化出力スキーマ)
│   └── report.py       # IterationReportWriter (逐次 Markdown レポート)
├── example/
│   ├── forward_problem.py      # 順問題の使用例
│   ├── inverse_problem.py      # 逆問題の使用例
│   ├── bo_forward.py           # ベイズ最適化によるハイパーパラメータ探索
│   └── opt_agent_forward.py    # LLM エージェントによるハイパーパラメータ探索
├── tests/
│   ├── test_algorithm.py    # アルゴリズム実装テスト
│   ├── test_theory.py       # 理論的性質テスト
│   └── check_behavior.py    # 振る舞い確認スクリプト（グラフ出力）
├── doc/
│   ├── paper_method.md      # 論文アルゴリズム抽出
│   ├── imp_design.md        # 実装設計書
│   └── test_design.md       # テスト設計書
└── data/
    └── burgers_shock.mat    # 参照解データ
```