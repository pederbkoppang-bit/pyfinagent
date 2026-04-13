#!/usr/bin/env bash
# Post-commit hook: auto-update CHANGELOG.md with latest commit info
# Triggered by Claude Code PostToolUse hook on "git commit"
# Does NOT create a new commit (avoids infinite loop)

set -euo pipefail

CHANGELOG="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}/CHANGELOG.md"
MAX_ROWS=20

# Get latest commit info
HASH=$(git log -1 --format="%h" 2>/dev/null || echo "unknown")
MSG=$(git log -1 --format="%s" 2>/dev/null | head -c 100 || echo "unknown")
DATE=$(date +%Y-%m-%d)

# Skip if CHANGELOG doesn't exist or no "Recent Activity" section
if [ ! -f "$CHANGELOG" ]; then
    exit 0
fi

if ! grep -q "### Recent Activity" "$CHANGELOG"; then
    exit 0
fi

NEW_ROW="| ${DATE} | \`${HASH}\` | ${MSG} |"

# Insert new row after the table header (line after |------|--------|--------|)
# Then trim to MAX_ROWS entries
python3 - "$CHANGELOG" "$NEW_ROW" "$MAX_ROWS" << 'PYEOF'
import sys

changelog_path = sys.argv[1]
new_row = sys.argv[2]
max_rows = int(sys.argv[3])

with open(changelog_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the table header separator line (|------|--------|--------|)
insert_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("|------") and "Recent Activity" in "".join(lines[max(0,i-3):i]):
        insert_idx = i + 1
        break

if insert_idx is None:
    sys.exit(0)

# Don't add duplicate (same hash already in table)
hash_marker = new_row.split("`")[1] if "`" in new_row else ""
for line in lines[insert_idx:insert_idx+max_rows]:
    if hash_marker and hash_marker in line:
        sys.exit(0)

# Insert new row
lines.insert(insert_idx, new_row + "\n")

# Count table rows and trim excess
table_rows = []
for i in range(insert_idx, len(lines)):
    if lines[i].strip().startswith("|") and not lines[i].strip().startswith("|--"):
        table_rows.append(i)
    elif not lines[i].strip().startswith("|"):
        break

if len(table_rows) > max_rows:
    for idx in sorted(table_rows[max_rows:], reverse=True):
        del lines[idx]

with open(changelog_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
PYEOF
