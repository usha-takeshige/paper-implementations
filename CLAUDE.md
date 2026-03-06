# Research Monorepo - Agent Rules

## 重要：スコープ制限
- 作業は必ず指定された論文ディレクトリ (`<paper-name>/`) 内のみで行う
- 他の論文ディレクトリは参照・編集しないこと
- ルートレベルの設定ファイルは読み取り可能だが、変更は慎重に

## ディレクトリ構造
- `<paper-name>/src/` : 実装コード
- `<paper-name>/tests/` : テストコード
- `<paper-name>/pyproject.toml` : パッケージ設定

## Python環境
- パッケージ管理は `uv` を使用
- 作業ディレクトリで `uv sync` して環境を構築
- 新パッケージ追加: `uv add <package>`
- テスト実行: `uv run pytest tests/`

## ブランチ・PRルール
- ブランチ名: `<issue-number>-<paper-name>/<feature>`
- コミットは論文ディレクトリ内の変更のみを含める

## ワークフロー
詳細は @.claude/rules/git-workflow.md を参照
