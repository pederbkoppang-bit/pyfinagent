#!/usr/bin/env bash
# phase-67.5, REPURPOSED 2026-07-31: SessionStart advisory for a Fable pin.
#
# HISTORY: this hook was originally the self-enforcing tripwire for the
# masterplan 67.4 revert -- it fired on/after 2026-07-12 (the end of the
# temporary free-Fable window) and named 67.4 as the session's top P0.
# That premise is DEAD. Fable 5 is now a standing part of the Max plan
# (verified 2026-07-31 against support.claude.com article 15424964), so
# there is no window to expire and no revert to enforce. Left unchanged,
# this hook would have injected a FALSE P0 ordering a revert of a pin that
# is now legitimate.
#
# CURRENT PURPOSE: a Fable pin is still worth surfacing, for a BUDGET
# reason rather than a calendar one -- Fable draws the same weekly Max
# budget as every other model and burns it faster, and past 50% of the
# weekly limit it transitions to metered usage credits, which violates the
# standing away-ops $0-metered constraint. So: advisory only, no P0, no
# revert instruction, and NO date gate.
#
# Fail-open by design: every error path exits 0 silently (SessionStart
# cannot block startup; a broken advisory must never break a session).
set -u

ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"

PINNED=""
for f in "$ROOT/.claude/agents/researcher.md" "$ROOT/.claude/agents/qa.md"; do
  if grep -q "^model: fable" "$f" 2>/dev/null; then
    PINNED="$PINNED ${f##*/}"
  fi
done
[ -n "$PINNED" ] || exit 0

WARN="FABLE BUDGET ADVISORY: 'model: fable' is pinned in:$PINNED. This is a legitimate pin -- Fable 5 is a standing part of the Max plan, and the old free-window/scheduled-revert doctrine is retired (CLAUDE.md 'Fable 5 policy'). It is flagged only because Fable shares ONE weekly Max budget with the main session, the harness subagents, and every Workflow fan-out, and burns it faster than Opus; past 50% of the weekly limit it transitions to METERED usage credits, which breaks the standing \$0-metered away-ops constraint. Prefer Fable selectively on hard gate evaluations and keep bulk fan-out on Opus. No action required."

# Single-line JSON additionalContext (v2.1.163+ shape). If a future schema
# drifts, plain stdout also becomes session context per the hooks docs, so
# the JSON string itself still surfaces the warning text.
ESCAPED="$(printf '%s' "$WARN" | sed 's/\\/\\\\/g; s/"/\\"/g')"
printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$ESCAPED"
exit 0
