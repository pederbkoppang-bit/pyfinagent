#!/usr/bin/env bash
# phase-4.14.27: InstructionsLoaded research-gate reminder.
#
# Fires once per session when CLAUDE.md / project instructions are
# loaded. Prints a short reminder to stderr so the operator sees the
# mandatory MAS harness research-gate rule on every session start.
# Does NOT block -- purely informational.

set -euo pipefail

cat >&2 <<'BANNER'
[phase-4.14.27] research-gate rule loaded:
  - EVERY masterplan step MUST spawn `researcher` BEFORE writing contract.md.
  - EVERY GENERATE MUST be followed by a fresh `qa` spawn -- never self-evaluate.
  - 5 handoff artifacts (contract, experiment_results, evaluator_critique, harness_log append, masterplan status flip) are non-skippable.
  - See `docs/runbooks/per-step-protocol.md` for the full cycle.
BANNER

# phase-4.16.2: audit JSONL in handoff/audit/ (layout convention).
AUDIT="${CLAUDE_PROJECT_DIR:-$(pwd)}/handoff/audit/instructions_loaded_audit.jsonl"
mkdir -p "$(dirname "$AUDIT")"
ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
printf '{"ts":"%s","event":"instructions-loaded","research_gate_reminded":true}\n' "$ts" >> "$AUDIT"

exit 0
