#!/usr/bin/env bash
# phase-67.5: SessionStart tripwire for the masterplan 67.4 Fable revert.
#
# Purpose: make the Sunday 2026-07-12 revert SELF-ENFORCING. Any session
# starting on/after that date with `model: fable` still pinned in either
# Layer-3 agent file gets a loud injected warning naming 67.4 as top P0.
#
# Fail-open by design: every error path exits 0 silently (SessionStart
# cannot block startup; a broken tripwire must never break a session).
# Dry-run knob: TRIPWIRE_FAKE_TODAY=YYYY-MM-DD overrides the date.
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
TODAY="${TRIPWIRE_FAKE_TODAY:-}"
if [ -z "$TODAY" ]; then
  TODAY="$(date +%F 2>/dev/null)" || exit 0
fi
[ -n "$TODAY" ] || exit 0

# ISO dates compare lexicographically; warn only on/after the window end.
if [[ "$TODAY" < "2026-07-12" ]]; then
  exit 0
fi

PINNED=""
for f in "$ROOT/.claude/agents/researcher.md" "$ROOT/.claude/agents/qa.md"; do
  if grep -q "^model: fable" "$f" 2>/dev/null; then
    PINNED="$PINNED ${f##*/}"
  fi
done
[ -n "$PINNED" ] || exit 0

WARN="TRIPWIRE (phase-67.5): today is $TODAY -- the free Fable 5 window ended 2026-07-12 and 'model: fable' is STILL pinned in:$PINNED. Masterplan step 67.4 (revert to model: opus, KEEP every artifact improvement) is the TOP P0 for this session, unless the operator has recorded 'FABLE PERMANENT: AUTHORIZE' in handoff/harness_log.md. Every Fable turn may now draw Max usage credits."

# Single-line JSON additionalContext (v2.1.163+ shape). If a future schema
# drifts, plain stdout also becomes session context per the hooks docs, so
# the JSON string itself still surfaces the warning text.
ESCAPED="$(printf '%s' "$WARN" | sed 's/\\/\\\\/g; s/"/\\"/g')"
printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$ESCAPED"
exit 0
