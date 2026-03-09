# 気になった論文を気ままに実装するリポジトリ

## 実装一覧

### 01: rbf-gen
**"Knowledge-guided generative surrogate modeling for high-dimensional design optimization under scarce data"**
（Bingran Wang et al., UC San Diego / Samsung Electronics）

[査読付きの雑誌](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/doi/10.1115/1.4070934/1230105/Knowledge-Guided-Generative-Surrogate-Modeling-for?redirectedFrom=fulltext)
[arXiv](https://arxiv.org/abs/2603.00052)

データが極端に少ない設計最適化問題において、専門家のドメイン知識をサロゲートモデルに体系的に組み込むフレームワーク「RBF-Gen」を提案した論文。放射基底関数（radial basis function: RBF）の零空間を生成ネットワークで探索することで、「データを必ず補間しつつ、物理的に妥当な関数を生成する」という両立を実現している。

### 02: mpde-bo
**"Sparse modeling based Bayesian optimization for experimental design"**
(Ryuji Masui, Unseo Lee, Ryo Nakayama and Taro Hitosugi)

[査読付きの雑誌](https://pubs.rsc.org/en/content/articlelanding/2025/ma/d5ma00346f)

各パラメータの各データに対するICE（Individual Conditional Expectation）幅の最大値を、目的関数の単位で設定した閾値と比較することで、研究者が直感的に設定できる形で探索軸を削れるようにしようというアイディア

## リポジトリ構成

各論文の実装はそれぞれ独立したディレクトリで管理するモノレポ構成です。

```
paper-implementations/
├── <id>-<paper-name>/
     ├── doc/          # ドキュメント
     ├── example/      # 使用例
     ├── src/          # 実装コード
     ├── tests/        # テストコード
     ├── pyproject.toml
     └── README.md

```

## 運用ルール

### 基本方針
- モノレポで管理する
- 共通のルールに関してはルートディレクトリで管理し、個別のルールに関してはプロジェクトごとのディレクトリで管理を行う

### 新しい論文を追加するとき
1. スクリプトを実行する: `bash scripts/new-project.sh <id>-<paper-name>`
   - 標準ディレクトリ構造（`src/`, `tests/`, `doc/`, `example/`）と `pyproject.toml` を自動生成
   - `.github/workflows/test.yml` の `filters` に該当ディレクトリを自動追加
```bash
scripts/new-project.sh <id>-<paper-name>
```
2. `cd <id>-<paper-name> && uv sync` で Python 環境を構築する
```bash
cd <id>-<paper-name> && uv sync
```
3. AIエージェントを用いた実装を行うときには、該当の論文のディレクトリに入った状態で作業を行う

※ 運用のテストを行うときには、`99-operation-test`ディレクトリを使用する。
