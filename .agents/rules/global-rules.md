---
trigger: always_on
---

# Research Monorepo - Agent Rules

## 重要：スコープ制限
- 作業は必ず指定された論文ディレクトリ (`papers/<paper-name>/`) 内のみで行う
- 他の論文ディレクトリは参照・編集しないこと
- ルートレベルの設定ファイルは読み取り可能だが、変更は慎重に

## ディレクトリ構造
- `papers/<paper-name>/src/` : 実装コード
- `papers/<paper-name>/tests/` : テストコード
- `papers/<paper-name>/pyproject.toml` : パッケージ設定

## Python環境
- パッケージ管理は `uv` を使用
- 作業ディレクトリで `uv sync` して環境を構築
- 新パッケージ追加: `uv add <package>`
- テスト実行: `uv run pytest tests/`

## ブランチ・PRルール
- ブランチ名: `paper/<paper-name>/<feature>`
- コミットは論文ディレクトリ内の変更のみを含める

## コミットメッセージルール

コミットメッセージは以下のプレフィックスを使用する（Conventional Commits準拠）。

### プレフィックス一覧

| プレフィックス | 用途                                     | 例                                         |
| -------------- | ---------------------------------------- | ------------------------------------------ |
| `feat`         | 新機能・新しいアルゴリズムの実装         | `feat: add multi-head attention`           |
| `fix`          | バグ修正                                 | `fix: correct softmax numerical stability` |
| `test`         | テストの追加・修正                       | `test: add unit test for encoder layer`    |
| `refactor`     | 動作を変えないコード整理                 | `refactor: extract scaled dot product`     |
| `docs`         | READMEや数式コメントの追加・修正         | `docs: add paper reference to README`      |
| `chore`        | 依存関係・設定ファイルの変更             | `chore: add torch to pyproject.toml`       |
| `perf`         | パフォーマンス改善                       | `perf: vectorize attention computation`    |
| `wip`          | 作業途中の一時コミット（PRには含めない） | `wip: drafting decoder block`              |

### フォーマット
```
<prefix>(<scope>): <変更内容の要約（英語・命令形・50文字以内）> #<Issue番号>

# 必要に応じて本文を追加（72文字で折り返し）
# 論文の数式やアルゴリズム番号を参照する場合はここに記載
# 例: Implements Eq. (1) from Section 3.2
```

### ルール
- サマリーは英語・命令形（"Add"、"Fix"、"Update"など）で書く
- Issue番号は必ずサマリー末尾に `#<番号>` の形式で付与する
- Issueが存在しない場合は作業前にIssueを作成してから着手する
- 1コミット1目的とし、複数の変更を混在させない
- `wip` コミットはPRを出す前に `git rebase -i` で整理する
- スコープを付ける場合は `feat(encoder): ...` の形式を使う