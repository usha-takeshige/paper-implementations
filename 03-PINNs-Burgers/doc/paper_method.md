# 論文アルゴリズム抽出：Physics-Informed Neural Networks による Burgers 方程式の順問題・逆問題への応用

## 1. 論文の概要

**一言でいうと**
ニューラルネットワークの損失関数に支配方程式の残差を組み込む Physics-Informed Neural Networks（PINNs）を用いて，1次元粘性 Burgers 方程式の順問題（速度場の予測）と逆問題（動粘性係数の同定）を解く．

**解決する課題**
- 従来の数値解法（有限差分法・有限要素法）はメッシュ生成を必要とし，複雑な形状や高次元問題での計算コストが高い
- 少量の境界・初期条件データのみから物理的に整合した解を得ることが困難

**提案手法の名称**
Physics-Informed Neural Networks（PINNs）

---

## 2. アルゴリズム

### Algorithm 1: PDE 残差の計算

**処理内容**
コロケーション点集合 $\{(t_f^j, x_f^j)\}_{j=1}^{N_f}$ と学習済みパラメータ $\theta$，動粘性係数 $\nu$ を入力として，Burgers 方程式の PDE 残差ベクトル $f \in \mathbb{R}^{N_f}$ を計算する．各偏微分は自動微分（automatic differentiation）により解析的に求められる．

**ステップ**

| ステップ | 処理内容 |
|---------|---------|
| 1 | $t, x \leftarrow \{(t_f^j, x_f^j)\}$ を計算グラフに登録（自動微分の対象として宣言） |
| 2 | $u \leftarrow u_\theta(t, x)$（ネットワークの順伝播） |
| 3 | $u_t \leftarrow \partial u / \partial t$（$u$ を $t$ に関して自動微分，計算グラフを保持） |
| 4 | $u_x \leftarrow \partial u / \partial x$（$u$ を $x$ に関して自動微分，計算グラフを保持） |
| 5 | $u_{xx} \leftarrow \partial u_x / \partial x$（$u_x$ を $x$ に関して自動微分，2階微分に相当） |
| 6 | $f \leftarrow u_t + u \cdot u_x - \nu \cdot u_{xx}$（Burgers 方程式の残差を構成） |
| 7 | return $f$ |

**関連数式**

| 式番号 | 数式 | 変数の説明 |
|--------|------|------------|
| (6) | $f(t, x; \theta) := \partial u_\theta/\partial t + u_\theta \cdot \partial u_\theta/\partial x - \nu \cdot \partial^2 u_\theta/\partial x^2$ | $f$：PDE 残差，$u_\theta$：ネットワーク出力，$\nu$：動粘性係数 |

**備考**
Step 3–5 における「計算グラフの保持」とは，$u_t$ や $u_x$ 自体をさらに微分できるよう，それらを導出した計算経路を破棄しないことを意味する（`create_graph=True` に相当）．

---

### Algorithm 2: PINN 順問題ソルバー

**処理内容**
境界・初期条件データ $D_u$ とコロケーション点 $D_f$ を入力として，損失関数 $L(\theta) = L_{\text{data}}(\theta) + L_{\text{phys}}(\theta)$ を Adam（Phase 1）と L-BFGS（Phase 2）の二段階最適化で最小化し，最適パラメータ $\theta^*$ を求める．

**ステップ**

| ステップ | フェーズ | 処理内容 |
|---------|---------|---------|
| 1 | 初期化 | $\theta \leftarrow$ Xavier 一様初期化によりネットワーク重みを初期化 |
| 2 | 初期化 | optimizer_1 $\leftarrow$ Adam($\theta$, lr=$\eta$)，optimizer_2 $\leftarrow$ L-BFGS($\theta$) |
| 3–10 | Phase 1（Adam） | $e = 1, \ldots, E_{\text{Adam}}$ について：$u_{\text{pred}}$ を順伝播，$L_{\text{data}}$ 計算，Algorithm 1 で $f$ を計算，$L_{\text{phys}}$ 計算，$L = L_{\text{data}} + L_{\text{phys}}$ 計算，$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L$ で更新 |
| 11–14 | Phase 2（L-BFGS） | $e = 1, \ldots, E_{\text{LBFGS}}$ について：$L$ を再計算（手順 4–8 と同様），$\theta \leftarrow \theta - H^{-1} \nabla_\theta L$ で更新（$H$：近似ヘッセ行列） |
| 15 | 出力 | return $\theta^*$ |

**関連数式**

| 式番号 | 数式 | 変数の説明 |
|--------|------|------------|
| (5) | $u_\theta(t, x) = (W_L \circ \sigma \circ W_{L-1} \circ \sigma \circ \cdots \circ \sigma \circ W_1)([t, x]^T)$ | $W_l$：第 $l$ 層の重み行列・バイアス，$\sigma$：活性化関数（tanh） |
| (7) | $L(\theta) = L_{\text{data}}(\theta) + L_{\text{phys}}(\theta)$ | $L_{\text{data}}$：データ損失，$L_{\text{phys}}$：物理損失 |
| (8) | $L_{\text{data}} = \frac{1}{N_u} \sum_{i=1}^{N_u} \|u_\theta(t_u^i, x_u^i) - u^i\|^2$ | $N_u$：データ点数，$(t_u^i, x_u^i, u^i)$：初期・境界条件データ点 |
| (9) | $L_{\text{phys}} = \frac{1}{N_f} \sum_{j=1}^{N_f} \|f(t_f^j, x_f^j; \theta)\|^2$ | $N_f$：コロケーション点数，$(t_f^j, x_f^j)$：コロケーション点 |
| (10) | $\theta^* = \arg\min_\theta L(\theta)$ | $\theta^*$：最適パラメータ |

**備考**
Phase 1 の Adam は広い損失景観を効率よく探索し，Phase 2 の L-BFGS は局所的な曲率情報を活用して高精度な極小点に収束する（Raissi et al., 2019 で採用された標準的な手順）．

---

### Algorithm 3: PINN 逆問題ソルバー（動粘性係数の同定）

**処理内容**
観測データ $D_{\text{obs}}$ とコロケーション点 $D_f$ を入力として，ネットワークパラメータ $\theta$ と動粘性係数 $\nu$ を同時に学習可能変数として宣言し，単一の最適化ループ内で $(\theta^*, \nu^*)$ を同定する．Algorithm 2 との差異は Step 1（$\nu$ を学習変数として宣言）と Step 9（$\theta$ と $\nu$ を同時更新）のみ．

**ステップ**

| ステップ | 処理内容 |
|---------|---------|
| 1 | $\theta \leftarrow$ Xavier 一様初期化，$\nu \leftarrow \nu_0$（$\nu$ を $\theta$ と並ぶ学習可能変数として宣言） |
| 2 | optimizer $\leftarrow$ Adam($[\theta, \nu]$, lr=$\eta$)（$\theta$ と $\nu$ を同一オプティマイザで更新） |
| 3–10 | $e = 1, \ldots, E$ について：$u_{\text{pred}}$ 計算，$L_{\text{data}}$ 計算，$f = u_t + u \cdot u_x - \nu \cdot u_{xx}$ を計算（$\nu$ が学習変数のため $\partial f/\partial \nu = -u_{xx}$ が計算グラフに含まれる），$L_{\text{phys}}$ 計算，$L$ 計算，$(\theta, \nu) \leftarrow (\theta, \nu) - \eta \cdot \nabla_{(\theta,\nu)} L$ で同時更新 |
| 11 | return $\theta^*, \nu^*$ |

**関連数式**

| 式番号 | 数式 | 変数の説明 |
|--------|------|------------|
| (6) | $f(t, x; \theta) := \partial u_\theta/\partial t + u_\theta \cdot \partial u_\theta/\partial x - \nu \cdot \partial^2 u_\theta/\partial x^2$ | $\nu$ が学習変数の場合，$\partial f/\partial \nu = -\partial^2 u_\theta/\partial x^2$ が自動微分で得られる |
| (11) | $(\theta^*, \nu^*) = \arg\min_{\theta, \nu} L(\theta, \nu)$ | $\nu^*$：推定された動粘性係数 |

---

## 3. 記号・変数一覧

| 記号 | 意味 | 初出箇所 |
|------|------|----------|
| $u(t, x)$ | 流体速度場（スカラー） | 式 (1) |
| $\nu$ | 動粘性係数（$> 0$） | 式 (1) |
| $u_\theta(t, x)$ | ネットワークによる $u(t, x)$ の近似 | Section 3.1 |
| $\theta$ | ネットワークのパラメータ全体 | Section 3.1 |
| $L$ | 隠れ層の数 | 式 (5) |
| $N$ | 各隠れ層の幅（ニューロン数） | Section 3.1 |
| $W_l$ | 第 $l$ 層の重み行列およびバイアスベクトル | 式 (5) |
| $\sigma$ | 活性化関数（tanh） | 式 (5) |
| $f(t, x; \theta)$ | Burgers 方程式の PDE 残差 | 式 (6) |
| $L_{\text{data}}(\theta)$ | データ損失（初期・境界条件の MSE） | 式 (8) |
| $L_{\text{phys}}(\theta)$ | 物理損失（PDE 残差の MSE） | 式 (9) |
| $L(\theta)$ | 総損失 | 式 (7) |
| $N_u$ | 初期・境界条件データ点数 | 式 (8) |
| $N_f$ | コロケーション点数 | 式 (9) |
| $\{(t_u^i, x_u^i, u^i)\}$ | 初期・境界条件データ集合 | 式 (8) |
| $\{(t_f^j, x_f^j)\}$ | コロケーション点集合（ラベルなし） | 式 (9) |
| $\nu_0$ | 逆問題における $\nu$ の初期推定値 | Algorithm 3 Step 1 |
| $e_{\text{rel}}$ | 相対 L2 誤差 | 式 (12) |

---

## 4. 手法のポイント（論文記載のもの）

- PDE の残差を損失関数として組み込むことで，支配方程式を「教師信号」として利用し，物理法則を正則化項として学習に組み込む
- 偏微分 $\partial u_\theta/\partial t$，$\partial u_\theta/\partial x$，$\partial^2 u_\theta/\partial x^2$ はすべて自動微分（automatic differentiation）により解析的に計算され，数値微分と異なり離散化誤差を含まない
- 活性化関数として $\tanh$ を採用する理由は，$\tanh$ が無限回連続微分可能（$C^\infty$ 級）であり，PDE に含まれる高階偏微分の計算において数値的に安定した結果を与えるため
- 逆問題においては，PDE 残差 (6) が $\nu$ について線形であるため，勾配 $\partial L/\partial \nu$ が自動微分によって自然に計算される
- ネットワーク重みの更新と物理パラメータの同定が単一の最適化ループ内で同時に実行される（Algorithm 3）
- コロケーション点は観測ラベルを必要とせず，時空間ドメイン $[-1,1] \times [0,1]$ 上の点座標のみで構成される

---

## 5. 論文が明示する課題・制限

- 複数の損失項の重み付けの難しさ：$L_{\text{data}}$ と $L_{\text{phys}}$ のスケールが異なる場合，一方の損失が支配的になり学習が不安定になることがある
- 衝撃波など急峻な解への対応：$\nu$ が非常に小さい場合，コロケーション点の均一サンプリングでは残差が過小評価される（適応的コロケーション点再サンプリングが有効な対策として知られる）
- 順問題における計算コスト：従来の有限差分法と比較して，PINNs は順問題では必ずしも計算効率に優れるわけではなく，逆問題やパラメータ同定での強みが際立つ
