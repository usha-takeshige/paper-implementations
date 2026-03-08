"""MPDE-BO アルゴリズムの実行例 (method.md §6 のフローに対応)。

このスクリプトは method.md §6 に記載されたアルゴリズムの各ステップを
そのまま追えるように、各コンポーネントを組み立てて最適化を実行します。

探索設定:
    - 総パラメータ数 N = 5 (重要 2 次元 + 非重要 3 次元)
    - グリッド数 M = 100  →  探索空間 [0, 100]^5
    - 初期観測数 n0 = 10、最大評価数 T = 30
"""

import torch

from mpde_bo.acquisition_optimizer import AcquisitionConfig, AcquisitionOptimizer
from mpde_bo.benchmark import BenchmarkFunction
from mpde_bo.gp_model_manager import GPConfig, GPModelManager
from mpde_bo.importance_analyzer import ImportanceAnalyzer
from mpde_bo.optimizer import MPDEBOOptimizer
from mpde_bo.parameter_classifier import ClassificationConfig, ParameterClassifier

# ── 再現性のためのシード固定 ─────────────────────────────────────────────────
torch.manual_seed(42)

# =============================================================================
# 0. 問題設定 (method.md §1, §7)
# =============================================================================
N_IMPORTANT = 2    # 重要パラメータ数 d
N_UNIMPORTANT = 3  # 非重要パラメータ数 s
N = N_IMPORTANT + N_UNIMPORTANT  # 総パラメータ数
M = 100            # グリッド数（探索空間 [0, M]^N）
N0 = 10            # 初期観測数
T = 100             # 最大評価ステップ数

# 目的関数  f(x) = f_d(x^⊤) + ε f_s(x^⊥)
objective = BenchmarkFunction(
    n_important=N_IMPORTANT,
    n_unimportant=N_UNIMPORTANT,
    grid_size=M,
)
print("=" * 60)
print(f"探索空間: [0, {M}]^{N}  (重要 {N_IMPORTANT} 次元 + 非重要 {N_UNIMPORTANT} 次元)")
print(f"真の最適値 f(x*) ≈ {objective.optimal_value:.4f}")
print("=" * 60)

# =============================================================================
# ステップ 1: D_0 を用いて ARD カーネルベースの GP モデルを構築 (method.md §6, l.1)
# =============================================================================
# 初期観測点を探索空間内の一様ランダムサンプリングで生成
bounds = torch.stack([
    torch.zeros(N, dtype=torch.double),
    torch.full((N,), M, dtype=torch.double),
])
train_X = (bounds[1] - bounds[0]) * torch.rand(N0, N, dtype=torch.double) + bounds[0]
train_Y = torch.tensor(
    [[objective(x)] for x in train_X], dtype=torch.double
)
print(f"\n初期観測 D_0: {N0} 点をランダムサンプリング")
print(f"初期最大観測値: {train_Y.max().item():.4f}")

# GP モデルマネージャ: Matérn 5/2 + ARD カーネル (method.md §3)
gp_manager = GPModelManager(config=GPConfig(kernel="matern52"))

# =============================================================================
# ステップ 2~10 を担うコンポーネントを組み立て
# =============================================================================

# ステップ 3~4: ARD 長さスケールの取得・MPDE 計算 (method.md §5, §6 l.3-4)
analyzer = ImportanceAnalyzer()

# ステップ 5: 閾値によるパラメータ分類 (method.md §6 l.5)
#   ℓ_i < ε_ℓ  AND  ê_i* > ε_e  →  重要 (X^⊤)
#   それ以外                      →  非重要 (X^⊥)
classifier = ParameterClassifier(
    analyzer=analyzer,
    config=ClassificationConfig(eps_l=50.0, eps_e=0.05),
)

# ステップ 6: EI を最大化して重要パラメータの次の試行点を決定 (method.md §4, §6 l.6)
acq_optimizer = AcquisitionOptimizer(
    config=AcquisitionConfig(type="EI", num_restarts=5, raw_samples=256),
    bounds=bounds,
)

# オーケストレータ: ステップ 2~11 のループ全体を管理
optimizer = MPDEBOOptimizer(
    gp_manager=gp_manager,
    classifier=classifier,
    acq_optimizer=acq_optimizer,
    bounds=bounds,
)

# =============================================================================
# ステップ 2~11: メインループ (method.md §6 l.2-11)
# コールバックで各ステップの状態を表示
# =============================================================================
print("\n--- 最適化ループ開始 ---")
print(f"{'ステップ':>6} | {'観測値':>10} | {'累積最大':>10} | 重要次元")

# objective をラッパーで包んで各評価値をキャプチャする。
# model.train_targets は GP 内部の正規化済み値なので使用しない。
_observed_ys: list[float] = []

def _tracked_objective(x):
    y = objective(x)
    _observed_ys.append(y)
    return y

def callback(t, _model, cls):
    """各ステップの情報を表示するコールバック。"""
    latest_y = _observed_ys[-1] if _observed_ys else float("nan")
    current_best = max(_observed_ys) if _observed_ys else float("nan")
    print(f"  t={t:>3}  | {latest_y:>10.4f} | {current_best:>10.4f} | {cls.important}")

result = optimizer.optimize(
    f=_tracked_objective,
    T=T,
    train_X=train_X,
    train_Y=train_Y,
    callback=callback,
)

# =============================================================================
# ステップ 12: D_T 内の最大データ点 x* を返す (method.md §6 l.12)
# =============================================================================
print("\n--- 最適化完了 ---")
print(f"最終最大観測値:     {result.best_y:.4f}")
print(f"真の最適値 f(x*): {objective.optimal_value:.4f}")
print(f"達成率:             {result.best_y / objective.optimal_value * 100:.1f}%")
print(f"\n最適点 x*:\n  {result.best_x.tolist()}")
print(f"  (重要次元 x^⊤: {result.best_x[:N_IMPORTANT].tolist()})")
print(f"  (非重要次元 x^⊥: {result.best_x[N_IMPORTANT:].tolist()})")
print(f"\n総評価回数: {len(result.train_Y)} 回 (初期 {N0} + ループ {T})")
