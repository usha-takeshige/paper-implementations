# Git Commit Rules

## コミットメッセージルール

コミットメッセージは以下のプレフィックスを使用する（Conventional Commits準拠）。

### プレフィックス一覧

| プレフィックス | 用途                               | 例                                      |
|------------|------------------------------------|-----------------------------------------|
| `feat`     | 新機能・新しいアルゴリズムの実装       | `feat: add multi-head attention #3`        |
| `fix`      | バグ修正                            | `fix: correct softmax numerical stability #1` |
| `test`     | テストの追加・修正                   | `test: add unit test for encoder layer #12` |
| `refactor` | 動作を変えないコード整理              | `refactor: extract scaled dot product #1`  |
| `docs`     | READMEや数式コメントの追加・修正      | `docs: add paper reference to README #1`   |
| `chore`    | 依存関係・設定ファイルの変更          | `chore: add torch to pyproject.toml #1`    |
| `perf`     | パフォーマンス改善                   | `perf: vectorize attention computation #1` |

### フォーマット
```
<prefix>: <変更内容の要約（英語・命令形・50文字以内）> <issue番号>

# 必要に応じて本文を追加（72文字で折り返し）
# 論文の数式やアルゴリズム番号を参照する場合はここに記載
# 例: Implements Eq. (1) from Section 3.2
```

### ルール
- サマリーは英語・命令形（"Add"、"Fix"、"Update"など）で書く
- 1コミット1目的とし、複数の変更を混在させない
- スコープを付ける場合は `feat(encoder): ...` の形式を使う
