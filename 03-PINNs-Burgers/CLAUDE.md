# PINNs Burgers equation 
Physics Informed Neural Networks（PINNs）を用いて、Burgers Equationのサロゲートモデルを作るプロジェクトです。
ベイズ最適化とLLMを利用したハイパーパラメータ最適化が行えます。

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
│   ├── prompt.py       # PromptBuilder (システム・ヒューマンプロンプト構築)
│   ├── proposal.py     # LLMProposal (Pydantic 構造化出力スキーマ)
│   └── report.py       # IterationReportWriter (逐次 Markdown レポート)
├── example/
│   ├── forward_problem.py      # 順問題の使用例
│   ├── inverse_problem.py      # 逆問題の使用例
│   ├── bo_forward.py           # ベイズ最適化によるハイパーパラメータ探索
│   └── opt_agent_forward.py    # LLM エージェントによるハイパーパラメータ探索
├── tests/
│   ├── test_bo_algorithm.py    # BO アルゴリズム実装テスト (ALG-BO-*)
│   ├── test_bo_theory.py       # BO 理論的性質テスト (THR-BO-*)
│   ├── test_llm_algorithm.py   # LLM アルゴリズム実装テスト (ALG-LLM-*)
│   ├── test_llm_theory.py      # LLM 理論的性質テスト (THR-LLM-*)
│   ├── check_bo_behavior.py    # BO 振る舞い確認スクリプト（グラフ出力）
│   └── check_llm_behavior.py   # LLM 振る舞い確認スクリプト（グラフ出力）
├── doc/
│   ├── paper_method.md      # 論文アルゴリズム抽出
│   ├── imp_design.md        # PINNs 実装設計書
│   ├── test_design.md       # PINNs テスト設計書
│   ├── bo_design.md         # BO モジュール設計書
│   └── llm_opt_design.md    # LLM 最適化モジュール設計書
└── data/
    └── burgers_shock.mat    # 参照解データ
```