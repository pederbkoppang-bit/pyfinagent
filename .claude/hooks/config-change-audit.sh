#!/usr/bin/env bash
# phase-4.14.27: ConfigChange audit log.
#
# Fires whenever .claude/settings.json or similar config is written.
# Appends a line to handoff/config_change_audit.jsonl with timestamp,
# changed path, and the git blob sha before+after (when available).

set -euo pipefail

PATH_CHANGED="${CLAUDE_CONFIG_PATH:-unknown}"
# phase-4.16.2: audit JSONL in handoff/audit/ (layout convention).
AUDIT="${CLAUDE_PROJECT_DIR:-$(pwd)}/handoff/audit/config_change_audit.jsonl"
mkdir -p "$(dirname "$AUDIT")"

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
sha=$(git -C "${CLAUDE_PROJECT_DIR:-$(pwd)}" hash-object "$PATH_CHANGED" 2>/dev/null || echo "-")
printf '{"ts":"%s","path":"%s","sha":"%s","event":"config-change"}\n' "$ts" "$PATH_CHANGED" "$sha" >> "$AUDIT"

exit 0
