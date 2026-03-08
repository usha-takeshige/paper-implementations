# RBF-Gen 実装ガイド

## 論文情報

**タイトル:** Knowledge-guided generative surrogate modeling for high-dimensional design optimization under scarce data
**著者:** Bingran Wang et al. (UC San Diego, Samsung Electronics)
**概要:** 過完備RBF表現の零空間をジェネレーターネットワークで探索することで、データが希少な状況でもドメイン知識を活用した高精度サロゲートモデルを構築するフレームワーク。

数学的な詳細は [doc/method.md](doc/method.md) を参照。

---

## アーキテクチャ概要

### 3ステップの実装構造

```
Step 1: RBF基底の構築
  - K個の中心 {c_j} を設計領域に配置（K > N）
  - 補間行列 Φ ∈ R^{N×K} を構築

Step 2: 零空間の計算
  - SVD/擬似逆行列で最小ノルム特解 w_0 を計算
  - Φ の零空間基底 Null ∈ R^{K×(K-N)} を計算

Step 3: ジェネレーターの学習
  - G: z → α （小さなMLP、z ~ N(0,I)）
  - f_z(x) = Φ(x)^T (w_0 + Null * G(z))
  - ドメイン知識をペナルティ・KL項として損失に組み込む
```

### モジュール構成（予定）

```
src/
  rbf_gen/
    rbf.py          # RBF基底・補間行列の計算
    null_space.py   # SVDによる零空間計算
    generator.py    # ジェネレーターMLPの定義
    losses.py       # ペナルティ項・KL発散項
    model.py        # RBFGenモデル（統合クラス）
    kernels.py      # RBFカーネル関数（ガウス、薄板スプライン等）
tests/
  test_rbf.py
  test_null_space.py
  test_generator.py
  test_losses.py
```

---

## 主要な数学的コンポーネント

### カーネル関数
- ガウスカーネル: `φ(r) = exp(-ε² r²)`
- 薄板スプライン: `φ(r) = r² log(r)`

### 損失関数の構成
```python
L = λ_mono * pen_mono + λ_pos * pen_pos + λ_KL * KL(p_gen || p_target)
```

ペナルティ項の種類: 単調性、正値性、Lipschitz上限、滑らかさ、凸性/凹性、境界条件

KL発散項: 特定点での値分布、領域平均、極値、勾配大きさ、曲率

### ハイパーパラメータの目安
| パラメータ | 推奨値 | 説明 |
|---|---|---|
| K（RBF中心数） | 2D〜5D | D=設計変数の次元数 |
| 潜在変数 z の次元 | K-N 程度 | 零空間の次元に合わせる |
| ネットワーク構造 | 小さなMLP（2〜3層） | 零空間探索用 |

---

## 実装の注意点

- **補間条件の保証:** `Null * G(z)` は必ず `Φ Null = 0` を満たす → どんな z でも補間条件が自動的に成立
- **KL近似:** バッチサンプリングで経験分布を構築し、ガウス分布ターゲットとのKLを計算
- **零空間計算:** `torch.linalg.svd` または `scipy.linalg.null_space` を使用
- **数値安定性:** 薄板スプラインは `r=0` で `r² log(r) → 0` の処理が必要

---

## 開発方針：テスト駆動開発（TDD）

**実装はすべてテストファーストで進める。**

1. 実装前にテストを書き、RED（失敗）を確認する
2. テストが通る最小限の実装を書き、GREEN にする
3. リファクタリングして REFACTOR する

各モジュールのテスト仕様は [doc/test_spec.md](doc/test_spec.md) に定義する。
実装に着手する前に必ず対応するテストを作成すること。

---

## Python環境

```bash
# 環境構築
uv sync

# テスト実行
uv run pytest tests/

# 詳細出力
uv run pytest tests/ -v

# 新パッケージ追加
uv add <package>
```

主要依存パッケージ（予定）: `torch`, `numpy`, `scipy`

---

## ケーススタディ（論文の実験）

1. **1D カンチレバービーム最適化** - シンプルなベンチマーク、D=2〜80
2. **2D シェル厚み最適化** - ライスナー・マインドリンシェルモデル使用
3. **半導体エッチングプロセス** - 実データ34点、17入力変数、5出力QoI
