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

# Extract paper name: strip leading "<id>-" prefix if present (e.g. "27-transformer" -> "transformer")
if [[ "$PROJECT_NAME" =~ ^[0-9]+-(.+)$ ]]; then
  PAPER_NAME="${BASH_REMATCH[1]}"
else
  PAPER_NAME="$PROJECT_NAME"
fi

# --- Check for existing directory ---
if [[ -e "$PROJECT_DIR" ]]; then
  echo "Error: '$PROJECT_DIR' already exists. Aborting to avoid overwrite." >&2
  exit 1
fi

echo "Creating project: $PROJECT_NAME"

# --- Create standard subdirectories ---
for dir in src "src/$PAPER_NAME" tests doc example; do
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
PACKAGE_NAME="${PAPER_NAME//-/_}"

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

# --- Update .github/workflows/test.yml filters ---
WORKFLOW_FILE="$REPO_ROOT/.github/workflows/test.yml"
if [[ -f "$WORKFLOW_FILE" ]]; then
  python3 << PYEOF
import re

filepath = "$WORKFLOW_FILE"
project_name = "$PROJECT_NAME"

with open(filepath) as f:
    content = f.read()

entry = "            {}:\n              - '{}/**'\n".format(project_name, project_name)

matches = list(re.finditer(r"              - '[^']+/\*\*'\n", content))
if matches:
    insert_pos = matches[-1].end()
    content = content[:insert_pos] + entry + content[insert_pos:]
    with open(filepath, "w") as f:
        f.write(content)
    print("  updated .github/workflows/test.yml")
else:
    print("  Warning: Could not find filter insertion point in test.yml")
PYEOF
else
  echo "  Warning: '$WORKFLOW_FILE' not found. Skipping."
fi

echo ""
echo "Done. Next steps:"
echo "  cd $PROJECT_NAME && uv sync"
