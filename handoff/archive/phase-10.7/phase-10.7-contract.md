# Sprint Contract — phase-10.7 (Rollback kill-switch wiring)

**Step id:** 10.7 **Date:** 2026-04-20 **Tier:** moderate **Harness-required:** true

## Why

Phase-10.6 gated **promotion** behind HITL approval. Phase-10.7 is the safety counterpart: when a challenger's DD breaches `DD_TRIGGER = 0.10`, auto-demote with NO human approval (bright line). Decision logged as `auto_demoted` to JSONL + ledger + state file.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.7-research-brief.md` — 7 sources in full, 17 URLs, three-variant, recency, gate_passed=true.

Critical findings:
- **`DD_TRIGGER = 0.10` already at `promoter.py:15`** — import, don't re-derive
- **`kill_switch.py` is system-wide pause** — do NOT call `KillSwitchState.pause()` for per-challenger demotion
- **`compliance_logger.write_rationale()` rejects automated events** (requires `approver_id`) — use separate JSONL audit
- **Canonical audit pattern** at `kill_switch.py:77-87` (JSONL append) — mirror exactly
- **Ledger sink:** `notes` column free-text, no schema change
- **State augmentation:** add `"auto_demoted"` as terminal status alongside existing `approved/rejected/expired`

## Immutable success criteria (masterplan-verbatim)

Test command: `python scripts/harness/phase10_rollback_test.py`

1. `challenger_dd_breach_auto_demotes` — `auto_demote_on_dd_breach(dd=-0.11)` returns `demoted=True, decision="auto_demoted"`; sub-threshold dd=-0.05 returns `demoted=False, decision="no_breach"`
2. `demotion_logged_with_auto_demoted_decision` — JSONL audit file contains a record with `"decision": "auto_demoted"` and `"challenger_id"` matching the input
3. `no_human_approval_required_for_demotion` — function signature has no `slack_fn` / `approver_id` kwargs; call completes without any Slack or HITL-state interaction

## Plan

1. Create `backend/autoresearch/rollback.py`:
   - Public `auto_demote_on_dd_breach(*, challenger_id, challenger_current_dd, dd_threshold=DD_TRIGGER, state_path=None, audit_path=None, ledger_path=None, week_iso=None, now=None) -> dict`
   - Returns `{demoted, decision, challenger_id, dd, threshold, ts}`
   - `dd_threshold` default imports from `backend.autoresearch.promoter.DD_TRIGGER` (single source of truth)
   - Idempotent: if state file already has `status="auto_demoted"` for this challenger, return `{demoted: True, decision: "already_demoted"}` without re-writing sinks
   - Writes three sinks on breach:
     - `handoff/demotion_audit.jsonl` — append JSONL record `{"ts": iso, "event": "auto_demoted", "challenger_id": ..., "dd": ..., "threshold": ..., "decision": "auto_demoted"}`
     - `monthly_approval_state.json` — upsert `{status: "auto_demoted", demoted_at_iso: ..., dd: ...}` keyed by challenger_id under a `demotions` sub-dict (to not collide with monthly month keys)
     - `weekly_ledger.append_row(..., notes="auto_demoted:challenger_id:dd=X.XX")` when `week_iso` is provided; preserves any prior notes via concatenation
   - Fail-open on individual sink failure (log warning, continue; function is best-effort for non-critical sinks, but returns truthful `demoted` flag)
   - ASCII-only logger messages; UTF-8 file writes
2. Create `scripts/harness/phase10_rollback_test.py` — 3 cases matching masterplan success_criteria verbatim, each in `tempfile.TemporaryDirectory` with injectable `now`
3. Create `tests/autoresearch/test_rollback.py` — ≥6 pytest cases incl. edge:
   - 3 CLI cases
   - `test_sub_threshold_no_demote`
   - `test_idempotent_second_call_no_op`
   - `test_jsonl_appends_not_overwrites`
   - `test_ledger_notes_preserved`
4. Verify: ast + immutable CLI + pytest new file + neighbor (autoresearch + slack_bot + metrics)
5. Spawn fresh Q/A. Cycle-2 flow if CONDITIONAL/FAIL. **After Q/A's mutation testing, Main checks `.pyc` cleanliness before re-running tests (lesson from 10.6 cycle-2).**
6. Log, flip, close task

## References

- `handoff/current/phase-10.7-research-brief.md`
- `backend/autoresearch/promoter.py:15,40-49` (DD_TRIGGER + on_dd_breach)
- `backend/services/kill_switch.py:77-87` (JSONL audit pattern to mirror)
- `backend/autoresearch/monthly_champion_challenger.py` (state file sibling)
- `backend/autoresearch/weekly_ledger.py:21-30` (notes column sink)
- phase-10.6 cycle-2 lesson: stale `.pyc` caused false test-failure during Q/A mutations; remember to clear `__pycache__` between mutations

## Carry-forwards (out of scope)

- Wire the `on_dd_breach` callback path from live paper-trader into `auto_demote_on_dd_breach` — separate integration step (phase-10.7.1?)
- Resume / un-demote logic (HITL-gated reversal) — deferred
- Metrics dashboard surface — phase-10.9 harness tab
