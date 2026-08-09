#!/usr/bin/env bash
# SessionStart: keep Playwright able to authenticate, every session.
#
# WHY. UI verification is BINDING (CLAUDE.md Critical Rules; qa.md 1c), but the
# Q/A subagent carries only browser_navigate/snapshot/screenshot/console -- it
# has NO Write tool, so it cannot mint its own session cookie. Before
# 2026-08-09 the documented path was a SECOND dev server on :3100 with
# LIGHTHOUSE_SKIP_AUTH; that was abandoned because a second `next dev` breaks
# the operator's :3000 (auto-memory feedback_second_next_dev_breaks_operator_3000),
# and NOTHING replaced it -- which is why UI verification silently stopped
# working and every UI claim quietly degraded to an API cross-check.
#
# This hook removes the failure mode by minting the storage state at session
# start, so a Q/A spawned at any point in the session finds a valid cookie
# already on disk. The cookie's TTL is 1h; a long session can outlive it, which
# is why the mint script is idempotent and safe to re-run by hand.
#
# FAIL-OPEN, ALWAYS. This is a convenience for verification tooling, never a
# gate on the session. Any failure prints a warning and exits 0 -- consistent
# with the hook discipline of never breaking the session for a non-safety
# concern. A stale/absent cookie surfaces later as a /login redirect, which
# qa.md requires be treated as NO EVIDENCE rather than a passing capture.
set -uo pipefail

REPO="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$REPO" 2>/dev/null || exit 0

MINTER="scripts/qa/mint_playwright_storage_state.py"
[ -f "$MINTER" ] || { echo "playwright-auth: $MINTER absent -- skipping"; exit 0; }

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 2>/dev/null)"
[ -n "$PY" ] || { echo "playwright-auth: no python -- skipping"; exit 0; }

if out=$("$PY" "$MINTER" 2>&1); then
  # Report shape only -- never the token.
  echo "playwright-auth: storage state refreshed ($(echo "$out" | grep -c . ) lines); UI verification is armed."
else
  echo "playwright-auth: WARNING -- could not mint a Playwright session cookie."
  echo "$out" | head -3
  echo "playwright-auth: UI verification will hit /login. Per qa.md that is NO EVIDENCE,"
  echo "playwright-auth: not a passing capture. Re-run: python $MINTER"
fi
exit 0
