#!/usr/bin/env bash
# Post-commit hook: auto-update CHANGELOG.md with latest commit info
# Triggered by Claude Code PostToolUse hook on "git commit"
# Does NOT create a new commit (avoids infinite loop)
# Auto-bumps patch version daily (v6.3.0 -> v6.4.0 on first commit of new day)

set -euo pipefail

CHANGELOG="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}/CHANGELOG.md"
MAX_ROWS=20

# Get latest commit info
HASH=$(git log -1 --format="%h" 2>/dev/null || echo "unknown")
MSG=$(git log -1 --format="%s" 2>/dev/null | head -c 100 || echo "unknown")
DATE=$(date +%Y-%m-%d)

# Skip changelog/drift commits to prevent infinite loop
if echo "$MSG" | grep -qiE "^chore: (auto-changelog|changelog drift)"; then
    exit 0
fi

# Skip if CHANGELOG doesn't exist or no "Recent Activity" section
if [ ! -f "$CHANGELOG" ]; then
    exit 0
fi

if ! grep -q "### Recent Activity" "$CHANGELOG"; then
    exit 0
fi

NEW_ROW="| ${DATE} | \`${HASH}\` | ${MSG} |"

# Insert new row + auto-bump version
python3 - "$CHANGELOG" "$NEW_ROW" "$MAX_ROWS" "$DATE" "$MSG" << 'PYEOF'
import sys
import re

changelog_path = sys.argv[1]
new_row = sys.argv[2]
max_rows = int(sys.argv[3])
today = sys.argv[4]
commit_subject = sys.argv[5] if len(sys.argv) > 5 else "Continued Development"

# Condense the subject for a version-header title: strip a leading
# "prefix: " scope marker and cap length so it fits the What's New card.
def _header_title(subject: str) -> str:
    s = subject.strip()
    # Drop "chore: ", "fix: ", "Phase 4.4.4.2: ", etc.
    s = re.sub(r"^[A-Za-z0-9.]+:\s*", "", s)
    if len(s) > 72:
        s = s[:69].rstrip() + "..."
    return s or "Continued Development"

header_title = _header_title(commit_subject)

with open(changelog_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# --- Auto-bump version ---
# Find the first ### vX.Y.Z header
version_idx = None
current_version = None
for i, line in enumerate(lines):
    m = re.match(r"^### v(\d+)\.(\d+)\.(\d+)\b", line)
    if m:
        version_idx = i
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        current_version = (major, minor, patch)
        break

# Check if the current version header already has today's date
# If not, bump patch version and add new header
if version_idx is not None and current_version is not None:
    version_line = lines[version_idx]
    if today not in version_line:
        # Bump patch version (6.4.0 -> 6.4.1)
        new_major, new_minor, new_patch = current_version[0], current_version[1], current_version[2] + 1
        new_version_header = f"### v{new_major}.{new_minor}.{new_patch} \u2014 {header_title} ({today})\n"
        # Insert new version header before the old one, with a separator
        lines.insert(version_idx, "\n")
        lines.insert(version_idx, new_version_header)

# --- Insert bullet point under today's version header ---
# Find the current version header (may have just been inserted above).
# Insert a "- **subject**" bullet right after the header line so the
# What's New card in the frontend always shows meaningful content.
# Skip commits prefixed with "chore:" since those are auto-generated
# (changelog updates, drift fixes) and clutter the summary.
if not commit_subject.lower().startswith("chore:"):
    # Re-find the version header (may have shifted due to bump insert)
    for i, line in enumerate(lines):
        m = re.match(r"^### v\d+\.\d+\.\d+\b", line)
        if m and today in line:
            # Find the insertion point: right after the header, before the
            # next header or the "### Recent Activity" section. Skip any
            # existing blank line immediately after the header.
            bullet_idx = i + 1
            while bullet_idx < len(lines) and lines[bullet_idx].strip() == "":
                bullet_idx += 1

            # Build bullet: "- **Subject** (hash)"
            bullet_text = re.sub(r"^[A-Za-z0-9.]+:\s*", "", commit_subject.strip())
            if len(bullet_text) > 100:
                bullet_text = bullet_text[:97].rstrip() + "..."
            bullet_line = f"- **{bullet_text}**\n"

            # Don't duplicate (check if the same bullet text is already there)
            already = False
            for j in range(bullet_idx, min(bullet_idx + 20, len(lines))):
                if lines[j].strip().startswith("### "):
                    break
                if bullet_text in lines[j]:
                    already = True
                    break
            if not already:
                lines.insert(bullet_idx, bullet_line)
            break

# --- Insert changelog row ---
# Re-find the table header separator (may have shifted due to version insert)
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
