#!/usr/bin/env bash
# aggregate.sh -- Pre-go-live aggregate smoketest (masterplan step 4.9).
#
# Runs every other phase's verification command and reports a single pass/fail
# for phase-4 Production Readiness. Fails fast on the first red signal so the
# operator sees the exact blocker instead of a wall of green.
#
# Success criteria (see masterplan step 4.9):
#   - every_other_phase_status_is_done
#   - each_done_phase_verification_command_reruns_green
#   - pytest_backend_tests_passes_with_zero_failures
#   - frontend_tsc_noemit_exits_zero
#   - frontend_next_build_exits_zero
#   - phase_4_6_smoketest_passes_all_10_steps
#   - no_open_critical_incidents_in_handoff_harness_log
#   - evaluator_critique_pass

set -u  # do not use -e; we want to report every failing check, not bail.

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_ROOT" || { echo "cannot cd to $PROJECT_ROOT"; exit 2; }

FAIL=0
# phase-53.5: SMOKE_PORTABLE=1 runs the credential-free PORTABLE subset -- it
# SKIPs the live-env-coupled legs (check #2 done-phase reruns that curl
# localhost / read live BQ; check #3's live/state/always-fail test files) so the
# smoke is green on a fresh checkout / CI runner with no backend, no .venv-live
# state, no BQ creds. Default (unset) is byte-identical to before (full audit).
PORTABLE="${SMOKE_PORTABLE:-0}"
export SMOKE_PORTABLE="$PORTABLE"
log() { printf '[aggregate] %s\n' "$*"; }
pass() { log "PASS: $1"; }
fail() { log "FAIL: $1"; FAIL=1; }
skip() { log "SKIP: $1"; }

# --- 1. Masterplan state ------------------------------------------------------
log "checking masterplan phase statuses ..."
python3 - <<'PYEOF' || fail "masterplan has non-done blockers"
import json, sys
mp = json.load(open(".claude/masterplan.json"))
blockers = next(p for p in mp["phases"] if p["id"] == "phase-4").get("depends_on", [])
issues = []
for p in mp["phases"]:
    # phase-53.5: "deferred" is an INTENTIONAL non-blocker (e.g. phase-5 crypto
    # removal) -- it must not fail the gate. Only a genuinely-incomplete blocker
    # (pending/in-progress/blocked) counts.
    if p["id"] in blockers and p.get("status") not in ("done", "deferred"):
        issues.append(f"{p['id']} status={p.get('status')}")
if issues:
    print("non-done blockers:", issues)
    sys.exit(1)
print("all blockers done (deferred counts as an intentional non-blocker)")
PYEOF
[ $? -eq 0 ] && pass "every_other_phase_status_is_done"

# --- 2. Re-run every done phase's verification command ----------------------
# phase-53.5: PORTABLE mode SKIPS this leg. Re-running all ~488 done-phase
# verification commands is a FULL pre-go-live / historical-drift audit, NOT a
# portable smoke -- empirically (53.5), ~30 of them fail on a clean rerun because
# they invoke live MCP servers / a booted backend, import since-moved or -removed
# modules (paper_metrics_v2, reconciliation), or read transient handoff artifacts
# that drift out of existence. Run with SMOKE_PORTABLE unset for the full audit.
# (The CI e2e-smoke.yml lane does not invoke aggregate.sh for the same reason.)
if [ "$PORTABLE" = "1" ]; then
  skip "each_done_phase_verification_command_reruns_green (PORTABLE: full 488-command done-phase rerun is a live/historical-drift audit, not a portable smoke; run with SMOKE_PORTABLE unset)"
else
  log "re-running verification commands for every done phase ..."
  python3 - <<'PYEOF'
import json, subprocess, sys
mp = json.load(open(".claude/masterplan.json"))
cmds = []
for p in mp["phases"]:
    for s in (p.get("steps") or []):
        if not isinstance(s, dict):
            continue  # guard malformed/legacy non-dict step entries
        v = s.get("verification")
        if not isinstance(v, dict):
            continue
        cmd = v.get("command")
        if s.get("status") == "done" and cmd:
            cmds.append((p["id"], s["id"], cmd))
fails = []
for phase_id, step_id, cmd in cmds:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        fails.append(f"{phase_id}/{step_id}")
        sys.stderr.write(f"  FAIL {phase_id}/{step_id}: {cmd[:80]}...\n")
        sys.stderr.write((r.stderr or r.stdout)[:500])
if fails:
    print("verification reruns failed:", fails); sys.exit(1)
print(f"verified {len(cmds)} done-phase verification commands OK")
PYEOF
  [ $? -eq 0 ] && pass "each_done_phase_verification_command_reruns_green" \
               || fail "each_done_phase_verification_command_reruns_green"
fi

# --- 3. Backend pytest -------------------------------------------------------
log "running backend pytest ..."
# phase-53.5: in PORTABLE mode, ignore the 6 live/state/always-fail test files
# (need live BQ / running services / time-sensitive state) so the credential-free
# subset is deterministic. Durable fix = a `requires_live` pytest marker (follow-up).
PYTEST_IGNORE=""
if [ "$PORTABLE" = "1" ]; then
  for f in test_phase_23_2_10_watchdog_no_fire_7d test_phase_23_2_12_layer1_pipeline_active \
           test_phase_23_2_14_no_reentrant_locks test_phase_23_2_16_shortlist_doc_presence \
           test_agent_map_live_model test_rainbow_canary; do
    PYTEST_IGNORE="$PYTEST_IGNORE --ignore=backend/tests/${f}.py"
  done
fi
if [ -d backend/tests ] && [ -f .venv/bin/activate ]; then
  ( source .venv/bin/activate && python -m pytest backend/tests/ -q $PYTEST_IGNORE ) \
    && pass "pytest_backend_tests_passes_with_zero_failures" \
    || fail "pytest_backend_tests_passes_with_zero_failures"
else
  fail "pytest: backend/tests or .venv missing"
fi

# --- 4. Frontend tsc + build -------------------------------------------------
log "running frontend tsc --noEmit ..."
( cd frontend && npx --no-install tsc --noEmit ) \
  && pass "frontend_tsc_noemit_exits_zero" \
  || fail "frontend_tsc_noemit_exits_zero"

log "running frontend next build ..."
( cd frontend && npm run build --silent ) \
  && pass "frontend_next_build_exits_zero" \
  || fail "frontend_next_build_exits_zero"

# --- 5. Phase 4.6 smoketest ---------------------------------------------------
SMOKETEST_SCRIPT="scripts/smoketest/phase-4.6.sh"
if [ -x "$SMOKETEST_SCRIPT" ]; then
  log "running phase-4.6 smoketest ..."
  bash "$SMOKETEST_SCRIPT" && pass "phase_4_6_smoketest_passes_all_10_steps" \
                            || fail "phase_4_6_smoketest_passes_all_10_steps"
else
  log "SKIP: $SMOKETEST_SCRIPT not yet implemented (phase 4.6 pending)"
fi

# --- 6. Harness log -- no OPEN critical incident in the recent tail ----------
# phase-53.5: match ACTUAL incident markers, not the bare word "critical" (which
# appears in benign prose like "0 critical", "2 critical findings", node_modules
# "1 critical" vuln counts). An open incident is a literal HARNESS HALT or a
# "CRITICAL INCIDENT" marker in the recent log tail.
if [ -f handoff/harness_log.md ]; then
  if tail -80 handoff/harness_log.md | grep -qiE 'HARNESS HALT|CRITICAL INCIDENT|HARNESS HALTED'; then
    fail "no_open_critical_incidents_in_handoff_harness_log"
  else
    pass "no_open_critical_incidents_in_handoff_harness_log"
  fi
else
  fail "handoff/harness_log.md missing"
fi

# --- 7. Evaluator critique ---------------------------------------------------
if [ -f handoff/current/evaluator_critique.md ]; then
  if grep -iE '^## Verdict.*(FAIL|BLOCK)' handoff/current/evaluator_critique.md >/dev/null; then
    fail "evaluator_critique_pass"
  else
    pass "evaluator_critique_pass"
  fi
else
  fail "handoff/current/evaluator_critique.md missing"
fi

# --- Summary -----------------------------------------------------------------
if [ $FAIL -eq 0 ]; then
  log "=== AGGREGATE SMOKETEST PASS ==="
  exit 0
else
  log "=== AGGREGATE SMOKETEST FAIL ==="
  exit 1
fi
