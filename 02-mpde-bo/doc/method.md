# MPDE-BO アルゴリズムの数学的記述

## 1. 問題設定

- 探索空間: $\mathcal{X} \in [0, M]^N$（$M$: グリッド数, $N$: 総パラメータ数）
- 目的: $x^* = \arg\max_{x \in \mathcal{X}} f(x)$ を最小試行回数で求める
- $f$ はブラックボックス関数（評価コスト大、勾配計算不可、非凸）

探索空間は重要・非重要パラメータの直和に分解される：
$$\mathcal{X} = \mathcal{X}^\top \oplus \mathcal{X}^\bot$$
$$f(x) = f_d(x^\top) + \epsilon f_s(x^\bot), \quad \epsilon \ll 1$$

---

## 2. ガウス過程（GP）回帰

$n$ 個のデータ点 $x_{1:n}$ に対してGPは以下を満たす：
$$f(x_{1:n}) \sim \mathcal{N}(m(x_{1:n}),\ K(x_{1:n}, x_{1:n}))$$

- $K_{i,j} = k(x_i, x_j)$（カーネル関数による分散共分散行列）
- 簡単のため $m(\cdot) = 0$ と仮定

データセット $D_t = \{(x_i, y_i)\}$ が与えられたとき、任意の点 $x'$ における事後分布：
$$\mu(x' \mid D_t) = k(x', x_{1:n})\, K(x_{1:n}, x_{1:n})^{-1}\, y_{1:n}$$
$$\sigma^2(x' \mid D_t) = k(x', x') - k(x', x_{1:n})\, K(x_{1:n}, x_{1:n})^{-1}\, k(x_{1:n}, x')$$

---

## 3. カーネル関数

### RBFカーネル
$$k_\text{RBF}(x, x' \mid \sigma_\text{RBF}, \ell)
= \sigma_\text{RBF}^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)$$

- ハイパーパラメータ: 分散 $\sigma_\text{RBF}^2$、長さスケール $\ell$（全次元共通）
- 各成分の重要度を個別に評価できない

### ARDカーネル（RBFの一般化）
$$k_\text{ARD}(x, x' \mid \sigma_\text{ARD}, \ell_{1:N})
= \sigma_\text{ARD}^2 \exp\!\left(-\frac{1}{2}\sum_{i=1}^{N}\frac{(x_i - x_i')^2}{\ell_i^2}\right)$$

- ハイパーパラメータ: 分散 $\sigma_\text{ARD}^2$、各次元の長さスケール $\ell_i$
- $\ell_i$ が大きい → $i$ 番目のパラメータの目的関数への影響が小さい
- データから $\ell_i$ を推定することで各パラメータの重要度を定量化できる

### Matérn 5/2 カーネル（本実装で使用）

$$k_{\text{Matérn}5/2}(x, x' \mid \sigma, \ell_{1:N})
= \sigma^2\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right)$$

ここで $r = \sqrt{\sum_{i=1}^{N}(x_i - x_i')^2 / \ell_i^2}$（ARD距離）。

- RBFより滑らかさが弱く、非滑らかな目的関数に対してより頑健
- ARD長さスケール $\ell_i$ により各次元の重要度を定量化できる（ARDカーネルと同等の解釈性）
- 論文では RBF/ARD カーネルを前提としているが、**本実装では Matérn 5/2 + ARD を採用する**

---

## 4. 獲得関数（期待改善量 EI）

$$q(x \mid D_t) = \mathbb{E}\!\left(\max\{0,\ \hat{f}_t(x) - f(x^+)\} \mid D_t\right)$$

- $f(x^+) = \max_{x \in x_{1:n}} f(x)$（過去の最大観測値）
- 次の試行点: $x_{t+1} = \arg\max_{x \in \mathcal{X}} q(x \mid D_t)$

> **実装上の注記（論文からのカスタマイズ）**
>
> 論文では獲得関数として EI を用いるが、**本実装では獲得関数をユーザーが選択できるよう拡張する**。
> EI はデフォルト値として保持しつつ、UCB（Upper Confidence Bound）や PI（Probability of Improvement）など
> 他の獲得関数も利用可能にする予定。

---

## 5. パラメータ重要度の定量化

### 5.1 部分従属プロット（PDP）と平均部分従属効果（APDE）

集合 $S \subset \{1,\ldots,p\}$、補集合 $C = \bar{S}$ に対して：

**部分従属関数（推定）**：
$$\hat{f}_S(x_S) = \frac{1}{n}\sum_{i=1}^{n} \hat{f}(x_S, x_C^i)$$

**APDE（平均部分従属効果）**：
$$\hat{e}_S = \max_{x_S} \hat{f}_S(x_S) - \min_{x_S} \hat{f}_S(x_S)$$

⚠️ **制限**: 目的関数にパラメータ間の交互作用がある場合、
平均化の効果で重要なパラメータを過小評価する恐れがある

### 5.2 個別条件付き期待値（ICE）と最大部分従属効果（MPDE）

$i$ 番目のインスタンス $x_C^i$ を固定したときのICE：
$$\text{ICE}^i(x_S) = \hat{f}(x_S, x_C^i)$$

各インスタンスにおける成分 $S$ の効果：
$$e_S^i = \max_{x_S} \hat{f}(x_S, x_C^i) - \min_{x_S} \hat{f}(x_S, x_C^i)$$

**MPDE（最大部分従属効果）**：
$$e_S^* = \max_{i \in [1,n]} e_S^i$$

これは以下の近似に対応：
$$e_S^* \approx \max_{x_C}\!\left\{\max_{x_S} f(x_S, x_C) - \min_{x_S} f(x_S, x_C)\right\}$$

✅ **特長**: 交互作用がある場合も、成分 $S$ のみを変化させたときの
目的関数値の**最大変化量**を正確に捉えられる

---

## 6. MPDE-BO アルゴリズム

**入力**: 目的関数 $f$、総評価予算 $T$、初期データセット $D_0$、
閾値 $\epsilon_\ell$（ARD長さスケール）、$\epsilon_e$（MPDE）

**出力**: 近似最大値 $x^* = \arg\max_{x \in \mathcal{X}} f(x)$
```
1:  D_0 を用いて ARD カーネルベースの GP モデル f̂_0 を構築
2:  for t = 1, 2, ..., T do
3:      各パラメータ i の ARD 長さスケール ℓ_i を取得
4:      f̂_{t-1} と D_{t-1} から各パラメータ i の MPDE ê_i* を計算
5:      閾値処理によって探索空間を分割：
            (ℓ_i < ε_ℓ) かつ (ê_i* > ε_e)  → x⊤（重要パラメータ空間）
            それ以外                          → x⊥（非重要パラメータ空間）
6:      EI を最大化して重要パラメータの次の試行点を決定：
            x_t⊤ = argmax_{x ∈ X} q(x | f̂_{t-1})
7:      非重要パラメータはランダムサンプリングで選択：
            x_t⊥ ~ Uniform(x⊥)
8:      x_t = x_t⊤ + x_t⊥
9:      D_t = D_{t-1} ∪ {(x_t, f(x_t))}
10:     D_t を用いてカーネルハイパーパラメータを更新し GP を再構築
11: end for
12: return D_t 内の最大データ点 x*
```

**スパース分類の条件（ステップ5）**:

| 条件 | 判定 | 処理 |
|------|------|------|
| $\ell_i < \epsilon_\ell$ かつ $\hat{e}_i^* > \epsilon_e$ | 重要 ($\mathcal{X}^\top$) | EIで最適化 |
| それ以外 | 非重要 ($\mathcal{X}^\bot$) | ランダムサンプリング |

---

## 7. モデル関数

重要パラメータ $x_d$ に対する関数（多峰性を持つ混合ガウス関数）：
$$f_d(x_d) = a_0\,\mathcal{N}(\mu_0, \sigma_0^2 I_d)
+ \sum_{i=1}^{d} a_i\,\mathcal{N}(\mu_i, \sigma_i^2 I_d)$$

ピーク分離の制約：
$$\|\mu_i - \mu_j\|_2 > \max\{2\sigma_i, 2\sigma_j\}, \quad i,j \geq 1$$

非重要パラメータ $x_s$ に対する関数（緩やかな変動）：
$$f_s(x_s) = a_s\,\mathcal{N}(\mu_s, \sigma_s^2 I_s)$$

目的関数全体：
$$f(x) = f_d(x_d) + f_s(x_s)$$

実験設定での値：
- $(a_0, a_1, a_2, a_3, a_4) = (0.3, 1.2, 0.6, 0.7, 0.7)$
- $(\sigma_0, \sigma_1, \sigma_2, \sigma_3, \sigma_4) = (20, 5, 5, 6, 6)$
- $a_s = 0.1,\ \sigma_s = 25,\ \mu_s = M/2$（各成分）
- 非重要パラメータの特性値変化: 探索空間全体で **10%未満**

---

## 8. 評価指標

$N_{90}$: 異なる $\mu_i$ を持つ100個の目的関数に対して、
最適値の90%以上を達成するために必要な実験回数の**90パーセンタイル値**

$$N_{90} = \text{Percentile}_{90}\!\left[\min\!\left\{t : f(x_t) \geq 0.9 \cdot f(x^*)\right\}\right]_{100\text{関数}}$$