# テスト設計書：Physics-Informed Neural Networks（PINNs）— Burgers 方程式

## 1. テスト設計の方針

**参照ファイル**：`/doc/paper_method.md`、`/doc/imp_design.md`

論文の核心は「PDE 残差を損失関数に組み込む」という自動微分の利用にあるため，`PDEResidualComputer` が式(6) を正しく計算できることを最優先に検証する．加えて，式(7) の `L = L_data + L_phys` の加法的分解と，逆問題において `ν` が計算グラフに組み込まれ勾配が正しく流れることを理論的性質テストで保証する．振る舞いテストでは，論文の Figure 1・4 に対応する解場の可視化と損失収束曲線によって定性的な再現性を確認する．

---

## 2. 単体テスト：A-1 アルゴリズム実装テスト（`test_algorithm.py`）

> 各アルゴリズムステップが計算として正しく動作するかを検証する。
> `@pytest.mark.algorithm` マーカーを付与する。

### テストケース一覧

| テストID | 対象メソッド / クラス | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|--------------------|---------|---------|--------------------|
| ALG-01 | `PINN.forward` | 出力形状が `(N, 1)` であること | `t, x: (N, 1)` テンソル | shape `(N, 1)`, requires_grad なし |
| ALG-02 | `PINN.__init__` | Xavier 一様初期化が各 Linear 層に適用されること | `NetworkConfig(n_hidden_layers=2, n_neurons=4)` | 全 weight テンソルが非ゼロで分散が層サイズに依存する |
| ALG-03 | `PDEResidualComputer.compute` | 出力形状が `(N_f, 1)` であること | `N_f=10` のコロケーション点 | shape `(N_f, 1)` |
| ALG-04 | `PDEResidualComputer.compute` | `create_graph=True` により `f` が微分可能であること（$u_{xx}$ の計算が可能） | 任意のコロケーション点 | `f.grad_fn` が `None` でないこと |
| ALG-05 | `LossFunction.compute` | 戻り値が `(L_total, L_data, L_phys)` の 3 要素タプルであること | 任意の `BoundaryData`, `CollocationPoints` | 3 要素のタプル，各要素がスカラー Tensor |
| ALG-06 | `LossFunction.compute` | `L_total = L_data + L_phys` が成立すること | 任意の入力 | `L_total` と `L_data + L_phys` の差が `1e-6` 未満 |
| ALG-07 | `LossFunction.compute` | `L_data`, `L_phys` が非負であること（MSE のため） | 任意の入力 | `L_data >= 0`, `L_phys >= 0` |
| ALG-08 | `ForwardSolver._get_trainable_params` | `ν` が学習パラメータに含まれないこと | `PINN` インスタンス | 返却リストに `ν` テンソルが含まれないこと |
| ALG-09 | `InverseSolver._get_trainable_params` | `ν` が学習パラメータに含まれること | `PINN` インスタンス, `nu_init=0.05` | 返却リストに `requires_grad=True` の `ν` テンソルが含まれること |
| ALG-10 | `BoundaryData` | 不正な形状のテンソルでバリデーションエラーが発生すること | `t: (N, 2)` の 2 次元目が 1 以外 | `ValidationError` が送出されること |
| ALG-11 | `CollocationPoints` | `t`, `x` の形状が一致しない場合にバリデーションエラーが発生すること | `t: (10, 1)`, `x: (5, 1)` | `ValidationError` が送出されること |


---

## 3. 単体テスト：A-2 理論的性質テスト（`test_theory.py`）

> 論文が保証する数学的・統計的性質が実装において成立するかを検証する。
> `@pytest.mark.theory` マーカーを付与する。
> 各テストケースには，対応する論文の命題・性質を明記する。

### テストケース一覧

| テストID | 論文の性質（根拠） | 対象クラス / メソッド | 検証内容 | 許容誤差 |
|---------|-----------------|--------------------|---------| --------|
| THR-01 | 式(6): PDE 残差の定義 $f = u_t + u \cdot u_x - \nu \cdot u_{xx}$ | `PDEResidualComputer.compute` | $u(t,x) = \text{const}$ を入力したとき $f = 0$ になること | `atol=1e-5` |
| THR-02 | 式(8): $L_{\text{data}} = 0 \Leftrightarrow u_\theta = u_{\text{true}}$ | `LossFunction.compute` | 完全一致データを与えたとき `L_data = 0` | `atol=1e-6` |
| THR-03 | 式(9): $L_{\text{phys}} = 0 \Leftrightarrow f = 0$ | `LossFunction.compute` | PDE 残差が 0 のとき `L_phys = 0` | `atol=1e-6` |
| THR-04 | 式(6): $\partial f/\partial \nu = -u_{xx}$（逆問題の勾配） | `PDEResidualComputer.compute` | $\nu$ に対する `f` の勾配が $-u_{xx}$ に等しいこと | `rtol=1e-4` |
| THR-05 | Algorithm 3 Step 1: $\nu$ を学習可能変数として宣言 | `InverseSolver` | $L$ を $\nu$ に関して `backward()` したとき `ν.grad` が `None` でないこと | — |
| THR-06 | 式(5): tanh 活性化関数 | `PINN.forward` | 隠れ層の出力値が $(-1, 1)$ に収まること（$\tanh$ の値域） | `atol=0` |


---

## 4. 振る舞いテスト：グラフ出力による確認（`check_behavior.py`）

> 自動判定が困難な視覚的・定性的な性質を，グラフを出力して人間が確認する。
> 各グラフには「確認すべき観点」をタイトルまたは注釈として表示する。

### 確認項目一覧

| チェックID | 確認すべき観点 | 論文の根拠 | グラフの種類 | 合格の目安 |
|----------|-------------|---------|------------|----------|
| BHV-01 | 損失曲線（L_total, L_data, L_phys）が単調減少傾向にあること | Algorithm 2 Step 3–14，式(7) | 折れ線グラフ（y軸対数スケール） | 学習終了時の L_total が初期値の 1/10 以下に収束していること |
| BHV-02 | 推定した $u_\theta(t,x)$ の等高線図が参照解 `usol` と視覚的に一致すること | 論文 Figure 1（順問題の解場），式(12) | ヒートマップ（2 枚: 推定解・参照解）| 衝撃波の位置・形状が概ね一致すること |
| BHV-03 | 推定解と参照解の点別絶対誤差 $|u_\theta - u_{\text{ref}}|$ が小さいこと | 式(12) $e_{\text{rel}}$ | ヒートマップ（誤差場） | 誤差が全体的に 0 に近い薄い色であること（衝撃波近傍を除く） |
| BHV-04 | 逆問題における $\nu$ の推定値がエポックごとに真値 $\nu^* = 0.01/\pi \approx 0.00318$ に近づくこと | Algorithm 3，式(11) | 折れ線グラフ（ν の推移 + 真値の水平線） | 学習終了時の $\nu^*$ が真値の $\pm 5\%$ 以内に収束していること |
| BHV-05 | 初期条件 $u(0, x) = -\sin(\pi x)$ がネットワーク出力で再現されること | 論文 Section 5.1（式(2)-(4)），Fig.1 の $t=0$ 断面 | 折れ線グラフ（推定 vs 参照，$t=0$） | 2 曲線がほぼ重なること |


---

## 5. 許容誤差の設定方針

| テストID | 許容誤差 | 設定根拠 |
|---------|---------|---------|
| THR-01 | `atol=1e-5` | ネットワーク重みをゼロ化した状態での float32 演算誤差（$\approx 10^{-7}$）に，自動微分チェーンの積み上げ誤差を考慮して余裕を持たせた |
| THR-02 | `atol=1e-6` | MSE の定義から完全一致時は厳密に 0 になるが，float32 の演算精度（$\approx 10^{-7}$）を考慮 |
| THR-03 | `atol=1e-6` | THR-02 と同様の根拠 |
| THR-04 | `rtol=1e-4` | 自動微分の数値的精度は理論値と一致するが，float32 の丸め誤差が連鎖するため |
| ALG-06 | `atol=1e-6` | $L = L_{\text{data}} + L_{\text{phys}}$ の加法的定義から，float32 演算の丸め誤差のみを許容 |

---

## 6. pytest 設定（`pyproject.toml`）

```toml
[tool.pytest.ini_options]
markers = [
    "algorithm: アルゴリズム実装の正確性テスト（計算の正しさ）",
    "theory: 論文の理論的性質が成立するかのテスト（数学的保証）",
]
```
