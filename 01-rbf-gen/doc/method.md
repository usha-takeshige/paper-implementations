# RBF-Gen: 数学的ロジック整理

## 1. 問題設定

**入力：** $N$ 個のデータ点 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$
- $\mathbf{x}_i \in \mathbb{R}^d$：$d$ 次元の設計変数
- $y_i \in \mathbb{R}$：関心量（QoI）

**目標：** $\hat{f} : \mathbb{R}^d \to \mathbb{R}$ を構築する（$N \ll d$ の状況で）

---

## 2. Step 1：緩和されたRBF基底の構築

### 古典的RBFとの違い

| | 古典的RBF | RBF-Gen |
|---|---|---|
| 中心数 | $N$（データ点と一致） | $K > N$（独立に配置） |
| 線形系 | 正方系（唯一解） | 過小決定系（無限解） |

### 過完備RBF表現

$K$ 個の中心 $\{c_j\}_{j=1}^K$ を設計領域に均一/準ランダムに配置し、

$$\hat{f}(\mathbf{x}) = \sum_{j=1}^K w_j \phi(\|\mathbf{x} - c_j\|)$$

補間条件 $\hat{f}(\mathbf{x}_i) = y_i$ を課すと、

$$\Phi w = y, \quad \Phi_{ij} = \phi(\|\mathbf{x}_i - c_j\|) \in \mathbb{R}^{N \times K}$$

$K > N$ なので、これは **過小決定系（underdetermined system）** → 無限に多くの解が存在する

### カーネル関数の選択肢

- ガウスカーネル：$\phi(r) = \exp(-\epsilon^2 r^2)$
- 薄板スプライン：$\phi(r) = r^2 \log r$

---

## 3. Step 2：零空間（Null Space）の計算

任意の解は以下で表現できる：

$$w = w_0 + N\alpha$$

| 記号 | 意味 | 次元 |
|---|---|---|
| $w_0$ | $\Phi w = y$ の最小ノルム特解 | $\mathbb{R}^K$ |
| $N$ | $\Phi$ の零空間の基底（$\Phi N = 0$） | $\mathbb{R}^{K \times (K-N)}$ |
| $\alpha$ | 自由係数ベクトル | $\mathbb{R}^{K-N}$ |

### 重要な性質

$\alpha$ の値に関わらず、補間条件 $\hat{f}(\mathbf{x}_i) = y_i$ は常に満たされる：

$$\Phi(w_0 + N\alpha) = \Phi w_0 + \underbrace{\Phi N}_{=0} \alpha = y \quad \checkmark$$

---

## 4. Step 3：ジェネレーターネットワーク

### 定式化

潜在変数 $z \sim \mathcal{N}(0, I)$ を零空間係数 $\alpha$ にマッピングするニューラルネットワーク $G(z; \theta)$：

$$\hat{f}_z(\mathbf{x}) = \Phi(\mathbf{x})^\top (w_0 + N \cdot G(z))$$

ここで $\Phi(\mathbf{x}) \in \mathbb{R}^K$ は評価点 $\mathbf{x}$ における各RBF中心への距離を格納したベクトル。

### 意味
- $z$ をサンプリングするたびに、**補間条件を満たしながら**、異なる形状の関数が得られる
- ジェネレーターは「補間条件を守りつつ、どんな形状を作るか」を学習する

---

## 5. Step 4：損失関数

### 全体構造

$$\mathcal{L}_\text{gen} = \sum_i \lambda_i \text{pen}_i(\hat{f}_z) + \sum_j \gamma_j \text{KL}(p_\text{gen}(s_j) \| p_\text{target}(s_j))$$

| 種類 | 役割 | 使用場面 |
|---|---|---|
| ペナルティ項 $\text{pen}_i$ | 点ごとの構造的制約 | 単調性・正値性・凸性など |
| KL発散項 | 生成関数の統計量の分布を制御 | 値域・曲率・勾配の分布など |

---

### ペナルティ項（Penalty Terms）

**単調性（増加 or 減少）**

$$\text{pen}_\text{mono} = \frac{1}{|\mathcal{G}|} \sum_{x_k \in \mathcal{G}} \text{ReLU}\left(\pm(\hat{f}_z(x_{k+1}) - \hat{f}_z(x_k))\right)$$

- $+$：非減少を強制
- $-$：非増加を強制

**正値性 / 下限制約**

$$\text{pen}_\text{pos} = \text{ReLU}\left(m - \min_{x \in \mathcal{G}} \hat{f}_z(x)\right)$$

**Lipschitz / 勾配の上限**

$$\text{pen}_\text{Lip} = \sum_{(x,y) \in \mathcal{P}} \text{ReLU}\left(\frac{|\hat{f}_z(x) - \hat{f}_z(y)|}{\|x - y\|} - L\right)$$

**滑らかさ / 曲率**

$$\text{pen}_\text{curv} = \sum_{x_k \in \mathcal{G}} \left(\hat{f}_z(x_{k+1}) - 2\hat{f}_z(x_k) + \hat{f}_z(x_{k-1})\right)^2$$

**凸性 / 凹性**

$$\text{pen}_\text{conv} = \sum_{x_k \in \mathcal{G}} \text{ReLU}(-\Delta^2 \hat{f}_z(x_k))$$

- 凸性：$\Delta^2 f \geq 0$ を強制
- 凹性：符号を反転

**境界条件**

$$\text{pen}_\text{bnd} = \sum_{x_b \in \mathcal{B}} (\hat{f}_z(x_b) - v_b)^2$$

---

### KL発散項（Distributional Priors）

**特定点での値の分布**

$$\text{KL}(p_\text{gen}(\hat{f}_z(x_0)) \| \mathcal{N}(\mu, \sigma^2))$$

**領域平均の分布**

$$\text{KL}\left(p_\text{gen}\left(\frac{1}{|R|}\sum_{x \in R} \hat{f}_z(x)\right) \| p_\text{target}\right)$$

**極値の分布**

$$\text{KL}\left(p_\text{gen}\left(\max_{x \in R} \hat{f}_z(x)\right) \| p_\text{target}\right)$$

**勾配の大きさの分布**

$$\text{KL}\left(p_\text{gen}(\|\nabla \hat{f}_z(x)\|) \| p_\text{target}\right)$$

**曲率の分布**

$$\text{KL}\left(p_\text{gen}\left(\frac{\partial^2 \hat{f}_z}{\partial x_i^2}\right) \| p_\text{target}\right)$$

---

### KL発散の計算手順（バッチ近似）

1. $z^{(1)}, \ldots, z^{(B)}$ を $\mathcal{N}(0, I)$ からサンプリング
2. 各 $z^{(b)}$ について統計量 $s^{(b)} = s(\hat{f}_{z^{(b)}})$ を計算
3. 経験分布 $p_\text{gen}$ を $\{s^{(b)}\}$ から構築
4. ターゲット分布 $p_\text{target}$（通常 $\mathcal{N}(\mu_t, \sigma_t^2)$）とのKLを計算

---

## 6. カンチレバービーム問題での具体的な損失関数

$$\mathcal{L} = \lambda_\text{mono} \underbrace{\sum_{j=1}^N \sum_{k=1}^{G-1} \max(0, \Delta f_j^{(k)})}_{\mathcal{L}_\text{mono}} + \lambda_\text{pos} \underbrace{\sum_{x \in \mathcal{G}} \max(0, -f(x))}_{\mathcal{L}_\text{pos}} + \lambda_\text{KL} \underbrace{\text{KL}(p_\text{gen}(f(x)) \| \mathcal{N}(\mu_t, \sigma_t^2))}_{\mathcal{L}_\text{dist}}$$

**スライスベースの事前情報の構築：**
- 各設計次元について1Dスライスを切り取る（他の変数は公称値に固定）
- そのスライス上の真の値 $\pm 30\%$ からガウス分布 $\mathcal{N}(\mu_t, \sigma_t^2)$ を構築
- 生成関数の分布がこのターゲットに近づくようにKL項で誘導

---

## 7. ハイパーパラメータの選択指針

| パラメータ | 推奨設定 | 役割 |
|---|---|---|
| $K$（RBF中心数） | $K \in [2D, 5D]$ | 零空間の次元 $= K - N$ を決める |
| $\alpha$ のスケール | 滑らかさと多様性のバランスで調整 | 零空間への投影の振れ幅 |
| $\lambda_i, \gamma_j$ | 問題依存（手動チューニング） | 各損失項の重み |

---

## 8. 推論（サロゲートの使い方）

**アンサンブル平均**

$$\hat{f}(\mathbf{x}) = \mathbb{E}_z[\hat{f}_z(\mathbf{x})] \approx \frac{1}{B} \sum_{b=1}^B \hat{f}_{z^{(b)}}(\mathbf{x})$$

**不確実性推定**

$$\text{Var}[\hat{f}(\mathbf{x})] \approx \frac{1}{B} \sum_{b=1}^B (\hat{f}_{z^{(b)}}(\mathbf{x}) - \bar{f}(\mathbf{x}))^2$$

---

## 9. 全体フロー
```
データ {(x_i, y_i)} + ドメイン知識
         ↓
[1] K個のRBF中心を設計領域に配置
         ↓
[2] 補間行列 Φ (N×K) を構築
         ↓
[3] 最小ノルム特解 w_0 と零空間基底 N を計算
    （SVD または疑似逆行列を使用）
         ↓
[4] ジェネレーター G: z → α を定義（小さなMLP）
         ↓
[5] ドメイン知識をペナルティ項・KL発散項として定式化
         ↓
[6] L_gen を最小化するよう G のパラメータ θ を学習
         ↓
[7] 学習済みGを使いアンサンブル平均でサロゲート構築
         ↓
[8] サロゲートで最適化問題を解く
```

---

## 10. ベイズ的解釈

$$\underbrace{p(f \mid \mathcal{D}, \text{knowledge})}_{\text{求めたいもの}} \propto \underbrace{p(\mathcal{D} \mid f)}_{\text{データへの適合}} \cdot \underbrace{p(f \mid \text{knowledge})}_{\text{ドメイン知識の事前分布}}$$

| ベイズの要素 | RBF-Gen での実装 |
|---|---|
| 尤度項 | 補間条件（ハードな等式制約） |
| 事前分布 | ペナルティ項・KL発散項（ソフトな制約） |
| 事後サンプル | ジェネレーターによる関数サンプル |