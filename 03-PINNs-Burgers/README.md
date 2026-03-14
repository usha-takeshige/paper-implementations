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
├── example/
│   ├── forward_problem.py   # 順問題の使用例
│   └── inverse_problem.py   # 逆問題の使用例
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

---

## セットアップ

```bash
cd 03-PINNs-Burgers
uv sync
```

**依存ライブラリ**: `torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pydantic`

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
uv run python tests/check_behavior.py
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
