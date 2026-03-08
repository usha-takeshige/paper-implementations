#!/usr/bin/env bash
# new-project.sh - Create a new project directory with standard structure.
#
# Usage: bash scripts/new-project.sh <project-name>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Argument validation ---
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <project-name>" >&2
  exit 1
fi

PROJECT_NAME="$1"
PROJECT_DIR="$REPO_ROOT/$PROJECT_NAME"

# --- Check for existing directory ---
if [[ -e "$PROJECT_DIR" ]]; then
  echo "Error: '$PROJECT_DIR' already exists. Aborting to avoid overwrite." >&2
  exit 1
fi

echo "Creating project: $PROJECT_NAME"

# --- Create standard subdirectories ---
for dir in src tests doc example; do
  mkdir -p "$PROJECT_DIR/$dir"
  echo "  created $PROJECT_NAME/$dir/"
done

# --- Copy .claude/settings.json ---
SETTINGS_SRC="$REPO_ROOT/.claude/settings.json"
SETTINGS_DST="$PROJECT_DIR/.claude/settings.json"

if [[ -f "$SETTINGS_SRC" ]]; then
  mkdir -p "$PROJECT_DIR/.claude"
  cp "$SETTINGS_SRC" "$SETTINGS_DST"
  echo "  copied  .claude/settings.json"
else
  echo "  Warning: '$SETTINGS_SRC' not found. Skipping."
fi

# --- Generate pyproject.toml ---
# Convert hyphens to underscores for the package name used in wheel config.
PACKAGE_NAME="${PROJECT_NAME//-/_}"

cat > "$PROJECT_DIR/pyproject.toml" << EOF
[project]
name = "$PROJECT_NAME"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/$PACKAGE_NAME"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
]
EOF
echo "  created $PROJECT_NAME/pyproject.toml"

echo ""
echo "Done. Next steps:"
echo "  cd $PROJECT_NAME && uv sync"
