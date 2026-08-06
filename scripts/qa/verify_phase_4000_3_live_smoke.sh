#!/usr/bin/env bash
# verify_phase_4000_3_live_smoke.sh -- step 4000.3 gate.
# Bounded to this step's own artifacts (masterplan 4000.3 criterion 1):
# live_check_4000.3.md in handoff/current/ OR handoff/archive/phase-4000.3/,
# plus the three harness_log precondition greps (criterion 2). Content markers
# are asserted IN-SECTION-adjacent form (the live_check is one document; the
# markers chosen are unique to their sections). $1 overrides the artifact path
# (mutation demo).
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CUR="$REPO/handoff/current/live_check_4000.3.md"
ARC="$REPO/handoff/archive/phase-4000.3/live_check_4000.3.md"
LOG="$REPO/handoff/harness_log.md"

if [ "${1:-}" != "" ]; then ARTIFACT="$1"
elif [ -f "$CUR" ]; then ARTIFACT="$CUR"
elif [ -f "$ARC" ]; then ARTIFACT="$ARC"
else echo "FAIL: live_check_4000.3.md not found"; exit 1; fi
[ -f "$ARTIFACT" ] || { echo "FAIL: not a file: $ARTIFACT"; exit 1; }
echo "artifact: $ARTIFACT"
fails=0
need() {  # need <label> <file> <marker>...
  local label="$1" file="$2"; shift 2
  local m
  for m in "$@"; do
    if ! grep -qF -- "$m" "$file"; then
      echo "FAIL [$label]: missing marker: $m"; fails=$((fails+1)); return
    fi
  done
  echo "ok   [$label]"
}

# criterion 2: the three preconditions, dated before the window, ALL grepped
# from harness_log.md (Q/A 4000.3 cycle-2: the single-writer leg was wrongly
# grepped from the live_check, leaving the log's own record unguarded).
need "log: RAIL TIERS"        "$LOG" "RAIL TIERS: AS CONFIGURED"
need "log: window auth"       "$LOG" "4000.3 LIVE-WINDOW AUTHORIZATION" "END-STATE DECISION RULE"
need "log: single-writer"     "$LOG" "single-writer window"
need "lc: single-writer"      "$ARTIFACT" "single-writer"
# criterion 3: E1-E6 machine verdicts + envelope + rules + explicit E6 figure
need "lc: E-verdicts"         "$ARTIFACT" '"check": "E1"' '"check": "E2"' '"check": "E3"' '"check": "E4"' '"check": "E5"' '"check": "E6"'
need "lc: envelope"           "$ARTIFACT" '"modelUsage"' '"total_cost_usd"'
need "lc: rail rule inline"   "$ARTIFACT" "provider = 'claude-code' OR agent = 'cc_rail' OR agent LIKE 'cc_rail:%'"
need "lc: E6 explicit figure" "$ARTIFACT" '"pct_of_100usd_monthly_credit"'
# criterion 4: cap statement from the smoke's own counter
need "lc: cap"                "$ARTIFACT" "window_rail_calls = 0 <= 30" "tickers = 1 <= 2"
# criterion 5: post-window end-state capture + rule application. The VALUE is
# asserted on the line(s) immediately following the POST-WINDOW header, so a
# capture flipped to false cannot pass on the strength of the pre-window
# events earlier in the document (Q/A 4000.3 cycle-1 finding 1: the file-wide
# marker was vacuous -- it also matched the pre_flag/flag_put lines).
need "lc: end state header"   "$ARTIFACT" "POST-WINDOW GET /api/settings" "STAYS"
# Uniqueness first: a decoy earlier header could otherwise satisfy the
# after-header value check (Q/A cycle-2 NOTE, closed here).
HDRS="$(grep -cF 'POST-WINDOW GET /api/settings' "$ARTIFACT")"
if [ "$HDRS" != "1" ]; then
  echo "FAIL [lc: end state]: expected exactly 1 POST-WINDOW header, found $HDRS"
  fails=$((fails+1))
fi
POSTVAL="$(awk '/POST-WINDOW GET \/api\/settings/{n=NR} n && NR>n && NR<=n+2 && /paper_use_claude_code_route/{print; exit}' "$ARTIFACT")"
if printf '%s' "$POSTVAL" | grep -qF '"paper_use_claude_code_route": true'; then
  echo "ok   [lc: end-state value inside post-window block]"
else
  echo "FAIL [lc: end state]: the post-window capture block does not carry the true value ($POSTVAL)"
  fails=$((fails+1))
fi
# criterion 6: the as-run bracket captures (filtered form, disclosed), the
# fresh full capture, the reconciliation, and the cycle-3 flip protocol gated
# on the FULL foreign set.
need "lc: an brackets"        "$ARTIFACT" "As-run PRE capture, verbatim (14:23:11Z" "As-run POST capture, verbatim (14:26:21Z" "Fresh FULL capture, verbatim" "Reconciliation, every path accounted" "FLIP PROTOCOL (cycle-3"
# criterion 7: restart statement re-derived (cycle-2 rewrite)
need "lc: restart"            "$ARTIFACT" "Therefore NO restart" "parent 89530 + child 89533"
# criterion 8: FAILed checks -> queued step id
need "lc: queued step"        "$ARTIFACT" "masterplan step 4000.10 (added in the same edit"

if [ "$fails" -gt 0 ]; then echo "RESULT: FAIL ($fails)"; exit 1; fi
echo "RESULT: PASS"
exit 0
