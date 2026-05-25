#!/usr/bin/env bash
# phase-38.4 (OPEN-13) -- bash smoke test for harness_log_gate.py.
# Exercises the 3 decision paths: proceed (gate disabled), passed
# (token present), skip (token missing + gate enabled).
#
# Usage: bash .claude/hooks/lib/harness_log_gate_test.sh
# Exit code: 0 on PASS, 1 on FAIL.
#
# Used by masterplan 38.4.verification: this script is the first half
# of the verification command; pytest test_phase_38_4_hook_gate.py is
# the second.

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HELPER="$REPO/.claude/hooks/lib/harness_log_gate.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ ! -f "$HELPER" ]; then
    echo "FAIL: helper missing at $HELPER" >&2
    exit 1
fi

# Case 1: gate DISABLED (env var unset) -> proceed regardless of log contents.
unset HARNESS_LOG_GATE_ENABLED || true
LOG="$TMP/empty.log"
: > "$LOG"
result=$(python3 "$HELPER" "$LOG" "12.3")
if [ "$result" != "proceed" ]; then
    echo "FAIL: case 1 expected 'proceed' (gate disabled), got '$result'" >&2
    exit 1
fi
echo "PASS: case 1 -- gate disabled returns proceed"

# Case 2: gate ENABLED + token present -> passed.
export HARNESS_LOG_GATE_ENABLED=true
LOG="$TMP/with_token.log"
cat > "$LOG" << 'EOF'
## Cycle 99 -- 2026-05-25 -- phase=12.3 result=PASS
- Step: dummy
EOF
result=$(python3 "$HELPER" "$LOG" "12.3")
if [ "$result" != "passed" ]; then
    echo "FAIL: case 2 expected 'passed', got '$result'" >&2
    exit 1
fi
echo "PASS: case 2 -- gate enabled + token present returns passed"

# Case 3: gate ENABLED + token MISSING -> skip.
LOG="$TMP/no_token.log"
cat > "$LOG" << 'EOF'
## Cycle 99 -- 2026-05-25 -- phase=99.9 result=PASS
- Different step
EOF
result=$(python3 "$HELPER" "$LOG" "12.3")
if [ "$result" != "skip" ]; then
    echo "FAIL: case 3 expected 'skip', got '$result'" >&2
    exit 1
fi
echo "PASS: case 3 -- gate enabled + token missing returns skip"

# Case 4: gate ENABLED + log file MISSING -> proceed (fail-open).
result=$(python3 "$HELPER" "$TMP/nonexistent.log" "12.3")
if [ "$result" != "proceed" ]; then
    echo "FAIL: case 4 expected 'proceed' (missing log file), got '$result'" >&2
    exit 1
fi
echo "PASS: case 4 -- missing log file returns proceed (fail-open)"

# Case 5: prefix-match guard -- step 38.6 must NOT match phase=38.6.1.
LOG="$TMP/prefix.log"
cat > "$LOG" << 'EOF'
## Cycle 99 -- 2026-05-25 -- phase=38.6.1 result=PASS
EOF
result=$(python3 "$HELPER" "$LOG" "38.6")
if [ "$result" != "skip" ]; then
    echo "FAIL: case 5 (prefix-match guard) expected 'skip', got '$result'" >&2
    exit 1
fi
echo "PASS: case 5 -- prefix-match guard (38.6 does not match phase=38.6.1)"

echo "ALL PASS"
exit 0
