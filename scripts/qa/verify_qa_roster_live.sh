#!/usr/bin/env bash
# phase-23.3.0: post-restart verification that the qa.md "1b. Frontend
# lint + typecheck" section (added in phase-23.2.24) is live in the
# Q/A subagent's snapshot.
#
# CONTEXT (per Anthropic Claude Code docs https://code.claude.com/docs/en/sub-agents):
#   "Subagents are loaded at session start. If you add or edit a
#    subagent file directly on disk, restart your session to load it."
#
# Therefore: verifying on-disk state is necessary but NOT sufficient.
# The behavioral verification (Q/A actually running ESLint as part of
# its rubric) requires a NEW Claude Code session.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QA_MD="$REPO_ROOT/.claude/agents/qa.md"
SECTION_HEADER="### 1b. Frontend lint + typecheck"

echo "================================================================"
echo " QA roster live-state verification (phase-23.3.0)"
echo "================================================================"

# 1. On-disk verification
echo
echo "[1/3] On-disk state of $QA_MD:"
if [ ! -f "$QA_MD" ]; then
    echo "  FAIL: qa.md missing"
    exit 1
fi
if grep -qF "$SECTION_HEADER" "$QA_MD"; then
    echo "  OK: '$SECTION_HEADER' found in qa.md"
    grep -A 2 -F "$SECTION_HEADER" "$QA_MD" | sed 's/^/    /'
else
    echo "  FAIL: '$SECTION_HEADER' NOT in qa.md"
    exit 1
fi

# 2. Git status verification
echo
echo "[2/3] Git status of phase-23.2.24 commit:"
COMMIT=$(cd "$REPO_ROOT" && git log --grep "phase-23.2.24:" --format=%H -n 1)
if [ -z "$COMMIT" ]; then
    echo "  FAIL: phase-23.2.24 commit not found in local history"
    exit 1
fi
echo "  Local commit: $COMMIT"
ORIGIN_HAS=$(cd "$REPO_ROOT" && git branch -r --contains "$COMMIT" 2>/dev/null | grep -c "origin/main" || true)
if [ "$ORIGIN_HAS" -ge 1 ]; then
    echo "  OK: commit is on origin/main (next session pulling main has the new rubric)"
else
    echo "  WARN: commit not yet visible on origin/main; run 'git push origin main' first"
fi

# 3. Operator self-disclosure prompt (manual step)
echo
echo "[3/3] Behavioral verification (manual, requires NEW Claude Code session):"
cat <<'PROMPT'

  After running `/clear` (or restarting the Claude Code app), spawn a
  fresh Q/A subagent and paste this prompt verbatim:

  -------- BEGIN OPERATOR PROMPT --------
  Self-disclosure: does your agent definition's deterministic-checks
  section include a subsection titled "1b. Frontend lint + typecheck"?
  Reply YES or NO. If YES, quote the first 3 lines of that subsection
  verbatim from your system prompt (NOT from a Read of qa.md from disk).
  Do not perform any other work; this is a roster verification only.
  --------- END OPERATOR PROMPT ---------

  Expected: YES + the 3 lines starting with "phase-23.2.24: a runtime
  React Rules-of-Hooks violation shipped in phase-23.2.23..."

  If the response is NO or the quoted lines do not match, the snapshot
  did not pick up the new rubric. Recovery: full Claude Code app
  restart (not just /clear), then retry.

PROMPT

echo "================================================================"
echo " On-disk + git checks PASSED. Behavioral check is operator-driven."
echo "================================================================"
exit 0
