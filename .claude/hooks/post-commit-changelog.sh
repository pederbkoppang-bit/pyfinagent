#!/usr/bin/env bash
# Post-commit hook: auto-update CHANGELOG.md with latest commit info
# Triggered by Claude Code PostToolUse hook on "git commit"
# Does NOT create a new commit (avoids infinite loop)
# Auto-bumps patch version daily (v6.3.0 -> v6.4.0 on first commit of new day)

set -euo pipefail

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
CHANGELOG="$PROJECT_ROOT/CHANGELOG.md"
MAX_ROWS=20

# Get latest commit info
HASH=$(git log -1 --format="%h" 2>/dev/null || echo "unknown")
MSG=$(git log -1 --format="%s" 2>/dev/null | head -c 100 || echo "unknown")
# Capture the full body too so the Python classifier can detect BREAKING CHANGE markers.
BODY=$(git log -1 --format="%B" 2>/dev/null || echo "")
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

# Insert new row + auto-bump version (semver-aware)
python3 - "$CHANGELOG" "$NEW_ROW" "$MAX_ROWS" "$DATE" "$MSG" "$BODY" << 'PYEOF'
import sys
import re

changelog_path = sys.argv[1]
new_row = sys.argv[2]
max_rows = int(sys.argv[3])
today = sys.argv[4]
commit_subject = sys.argv[5] if len(sys.argv) > 5 else "Continued Development"
commit_body = sys.argv[6] if len(sys.argv) > 6 else ""


def classify_commit(subject: str, body: str) -> str:
    """Return 'major' / 'minor' / 'patch' / 'none' per Conventional Commits +
    phase-X.Y conventions documented in CLAUDE.md.

    - BREAKING CHANGE in body OR feat!: / fix!: prefix -> major
    - feat: / feat(scope): -> minor
    - fix: / bug: / perf: -> patch
    - phase-X.0: -> minor (new top-level phase kickoff)
    - phase-X.Y: -> patch (sub-step)
    - chore: / docs: / refactor: / test: / style: / ci: / build: -> none
    - anything else -> patch (default safety)
    """
    s = subject.strip()
    b = body or ""
    # Major: explicit ! suffix on type prefix, or BREAKING CHANGE: line in body
    if re.match(r"^[a-z]+(?:\([^)]*\))?!:", s):
        return "major"
    if re.search(r"^BREAKING CHANGE:", b, re.MULTILINE):
        return "major"
    # phase-X.0: -> minor; phase-X.Y: (Y>=1) -> patch
    m_phase = re.match(r"^phase-(\d+)\.(\d+)(?:\.\d+)?:", s)
    if m_phase:
        return "minor" if m_phase.group(2) == "0" else "patch"
    # Conventional Commits type prefixes
    m_type = re.match(r"^([a-z]+)(?:\([^)]*\))?:", s)
    if m_type:
        t = m_type.group(1)
        if t == "feat":
            return "minor"
        if t in ("fix", "bug", "perf"):
            return "patch"
        if t in ("chore", "docs", "refactor", "test", "style", "ci", "build"):
            return "none"
    return "patch"


bump_type = classify_commit(commit_subject, commit_body)
# Back-compat alias used by the bullet-injection block below: "none" means
# no version row AND no bullet (same semantics as the old is_chore flag).
is_chore = bump_type == "none"

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

# Bump version per Conventional-Commits classifier (see classify_commit
# above). "none" -> no version header (just Recent Activity row).
# "patch" -> Z+1, "minor" -> Y+1/Z=0, "major" -> X+1/Y=0/Z=0.
if bump_type != "none" and version_idx is not None and current_version is not None:
    major, minor, patch = current_version
    if bump_type == "major":
        new_major, new_minor, new_patch = major + 1, 0, 0
    elif bump_type == "minor":
        new_major, new_minor, new_patch = major, minor + 1, 0
    else:  # patch
        new_major, new_minor, new_patch = major, minor, patch + 1
    new_version_header = f"### v{new_major}.{new_minor}.{new_patch} \u2014 {header_title} ({today})\n"
    # Insert new version header before the old one, with a separator.
    lines.insert(version_idx, "\n")
    lines.insert(version_idx, new_version_header)

# --- Insert bullet point under today's newest version header ---
# Every meaningful commit gets a header (just inserted above), so the
# bullet appears under its own header. Chore commits are skipped.
if not is_chore:
    for i, line in enumerate(lines):
        m = re.match(r"^### v\d+\.\d+\.\d+\b", line)
        if m and today in line:
            bullet_idx = i + 1
            while bullet_idx < len(lines) and lines[bullet_idx].strip() == "":
                bullet_idx += 1

            bullet_text = re.sub(r"^[A-Za-z0-9.]+:\s*", "", commit_subject.strip())
            if len(bullet_text) > 100:
                bullet_text = bullet_text[:97].rstrip() + "..."
            bullet_line = f"- **{bullet_text}**\n"

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

# Auto-commit the changelog update so it's included in the next push.
# The "chore: auto-changelog" prefix is in the skip-list above (line 18)
# so this commit will NOT re-trigger the hook (no infinite loop).
if git diff --quiet "$CHANGELOG" 2>/dev/null; then
    # No changes (duplicate detected or nothing to write)
    exit 0
fi
git add "$CHANGELOG" 2>/dev/null || exit 0
git commit -m "chore: auto-changelog hook entry for ${HASH}" 2>/dev/null || exit 0
