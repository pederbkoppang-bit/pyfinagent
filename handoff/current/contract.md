# Contract -- BLOCKER-3: Exercise HITL C/C gate end-to-end (task #42)

## Research gate

- Researcher spawn: 2026-04-24. Brief at `handoff/current/blocker-3-research-brief.md`.
- JSON envelope: tier=moderate, external_sources_read_in_full=7 (floor 5), urls_collected=17, recency_scan_performed=true, internal_files_inspected=9, gate_passed=true.
- Key external finding: industry consensus (Netflix, Snowflake, DataRobot, H2O, SageMaker MLOps) is that a promotion drill must exercise the full write-path including the audit-log row; shadow-mode + HITL-button paths both need real telemetry proving they fired.
- Key internal findings: (a) `backend/autoresearch/monthly_champion_challenger.py::record_approval` writes ONLY to the JSON state file -- NO BQ write at all, meaning criterion "BQ log row" cannot be satisfied without a code change; (b) the code path uses dependency-injection for side effects (`slack_fn` is an injectable Callable); (c) `strategy_deployments_log` table schema is: strategy_id, status, sharpe, dsr, pbo, max_dd, deployed_at, allocation_pct, notes -- currently 0 rows; (d) stale state short-circuits re-runs -- drill must use a dedicated state_path to stay out of real state.

## Hypothesis

The C/C gate pipeline is assembled but has never been exercised end-to-end:
1. Gate fires (Sortino delta / PBO / DD ratio thresholds) -- tested in unit tests.
2. Pending state persists to JSON -- tested in unit tests.
3. Slack ping sent -- slack_fn injection exists but no production caller passes a real client.
4. Approval handler transitions JSON -- tested via direct function call.
5. **BQ audit row written -- NOT IMPLEMENTED TODAY.**
6. **End-to-end drill exercising 1-5 in one shot -- does not exist today.**

Goal of this cycle: close the gap by adding the BQ write (step 5) behind the same dependency-injection pattern as slack_fn, then building a hermetic drill that proves steps 1-5 in one run without touching real production state or requiring Peder to click a button.

## Planned change (minimum scope, dependency-injection-first)

1. **`backend/autoresearch/monthly_champion_challenger.py::record_approval`** --
   add optional `bq_fn: Callable[[dict], None] | None = None` parameter. On a
   successful terminal transition (status in {"approved", "rejected", "expired"}),
   if `bq_fn` is provided, call `bq_fn(log_row)` where `log_row` is a dict with
   keys matching the `strategy_deployments_log` schema (strategy_id = the
   challenger_id, status, sharpe=None, dsr=None, pbo, max_dd=None, deployed_at,
   allocation_pct=0.0, notes). Same try/except fail-open pattern as slack_fn.
   Zero behavior change when `bq_fn` is `None`.

2. **`backend/api/monthly_approval_api.py::post_monthly_approval`** -- wire a
   real `bq_fn` that inserts the row into `pyfinagent_pms.strategy_deployments_log`
   via a small helper function `_default_bq_logger`. Fail-open: if BQ unavailable,
   log warning and proceed. Passed as `bq_fn=_default_bq_logger` to `record_approval`.

3. **`scripts/go_live_drills/hitl_gate_drill.py`** (new) -- hermetic end-to-end
   drill:
   - Uses `state_path=Path(tempfile.mkdtemp())/"drill_state.json"` to stay out of
     `handoff/logs/monthly_approval_state.json`.
   - Picks the last Friday of a PAST month (e.g., 2026-03-27) so `is_last_trading_friday`
     returns True deterministically (don't depend on today's date).
   - Synthesizes champion/challenger return series that pass the three gates
     (positive Sortino delta, low PBO, DD ratio well under 1.2).
   - Captures all slack_fn + bq_fn calls in lists (hermetic; does NOT post to
     real Slack or BQ).
   - Invokes `run_monthly_sortino_gate(...)`; asserts `fired=True, gate_pass=True,
     approval_pending=True`; asserts slack_fn captured 1 call; asserts JSON state
     has `status=pending` for the month key.
   - Invokes `record_approval(month_key, status="approved", state_path=...,
     bq_fn=capture_list.append)`; asserts JSON state has `status=approved` and a
     `resolved_at_iso`; asserts bq_fn captured 1 call with `status="approved"`
     and `strategy_id="DRILL-YYYY-MM"`.
   - Cleans up the temp state path on exit.
   - Exits 0 prints `PASS` on green; exits 1 prints FAIL lines otherwise.

4. **Do NOT** in this cycle:
   - Post to real Slack (the drill uses a capture list; real-Slack e2e is a
     Peder-interactive follow-up).
   - Write to real BQ (the drill uses a capture list; the API endpoint now
     does the real write in production, but the drill runs hermetically).
   - Lift the `actual_replacement=False` invariant (that's a separate SR 11-7
     compliance gate, not in scope).
   - Change the gate thresholds or the Sortino/PBO/DD formulas.

## Immutable success criteria

1. `backend/autoresearch/monthly_champion_challenger.py::record_approval`
   function signature contains the keyword argument `bq_fn`.
2. Same file: on terminal transitions (approved/rejected/expired), `record_approval`
   calls `bq_fn(...)` when provided, inside a `try/except` fail-open wrapper.
3. `backend/api/monthly_approval_api.py` contains a function named
   `_default_bq_logger` and passes `bq_fn=_default_bq_logger` into `record_approval`.
4. `scripts/go_live_drills/hitl_gate_drill.py` exists and `ast.parse` succeeds.
5. `python scripts/go_live_drills/hitl_gate_drill.py` exits 0 and prints `PASS`.
6. Drill output contains `step1_gate_fired`, `step2_pending`, `step3_approved`,
   `step4_bq_row_written` evidence lines (one per state transition).
7. `python -c "import ast; ast.parse(open('backend/autoresearch/monthly_champion_challenger.py').read())"` exits 0.
8. `python -c "import ast; ast.parse(open('backend/api/monthly_approval_api.py').read())"` exits 0.
9. NO regression: `python -c "from backend.autoresearch import monthly_champion_challenger"` imports cleanly.
10. NO regression: `python -c "from backend.api import monthly_approval_api"` imports cleanly.

## Verification command (Q/A reproduces this)

```bash
source .venv/bin/activate
grep -c "bq_fn" backend/autoresearch/monthly_champion_challenger.py       # >= 2 (signature + call site)
grep -c "_default_bq_logger" backend/api/monthly_approval_api.py          # >= 2 (def + call)
python -c "import ast; ast.parse(open('backend/autoresearch/monthly_champion_challenger.py').read())" && echo SYNTAX_MCC_OK
python -c "import ast; ast.parse(open('backend/api/monthly_approval_api.py').read())" && echo SYNTAX_API_OK
python -c "import ast; ast.parse(open('scripts/go_live_drills/hitl_gate_drill.py').read())" && echo SYNTAX_DRILL_OK
python -c "from backend.autoresearch import monthly_champion_challenger; print('IMPORT_MCC_OK')"
python -c "from backend.api import monthly_approval_api; print('IMPORT_API_OK')"
python scripts/go_live_drills/hitl_gate_drill.py
```

All must succeed (counts >= 2, imports clean, drill prints PASS) for PASS.

## Explicit non-goals and deferred work

- **Real-Slack end-to-end** (human click path): researcher recommended Option B
  with a TEST-labelled Slack ping. Deferred to a Peder-interactive follow-up
  cycle; NOT a pre-production blocker because the Slack handler is already
  unit-tested and the drill exercises the full internal pipeline it feeds.
- **Real BQ write in drill**: drill captures `bq_fn` calls for hermeticity. The
  production API endpoint writes to real BQ. Testing the real BQ write against
  `strategy_deployments_log` once it's wired is a post-drill sanity check
  Peder can do via `curl POST /api/harness/monthly-approval/2026-XX` plus a BQ
  SELECT.
- **Gate threshold tuning** (Sortino delta / PBO / DD ratio): these are
  already unit-tested with the existing values; tuning is a research task not
  a pre-prod blocker.
- **SR 11-7 compliance pass** to lift `actual_replacement=False`: out of scope
  for this phase; separate compliance/legal gate.

## References

- `handoff/current/blocker-3-research-brief.md` (research deliverable)
- `backend/autoresearch/monthly_champion_challenger.py` (target file)
- `backend/api/monthly_approval_api.py` (target file)
- `scripts/go_live_drills/hitl_gate_drill.py` (new drill)
- BQ: `pyfinagent_pms.strategy_deployments_log` (9-column schema verified,
  0 rows currently)
- `handoff/logs/monthly_approval_state.json` (real state path -- drill stays
  OUT of this path)
