# Sprint Contract -- phase-25.R -- Strategy auto-switching policy (closes red-line goal-c)

**Cycle:** phase-25 cycle 17 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.R
**Priority:** P1
**Depends on:** 25.C3 (done)
**Audit basis:** bucket 24.13 F-3 -- "the goal-c strategy-switching mechanism does not exist"; closes red-line goal-c

## Research-gate

Researcher spawned this cycle (agent a850a1d4697c5ae72). Brief at
`handoff/current/research_brief.md`. Gate envelope: 6 sources read in full,
16 URLs, recency scan performed, gate_passed=true.

Key research conclusions:
- **Two-path policy:** Path A (Promoter ops-authorized, no HITL) -- when `Promoter.promote()` returns `{promoted: True}` for a trial that cleared the shadow+DSR gate, write to registry with `status="active"` AND supersede the prior active row AND fire P0 Slack. Path B (Monthly HITL) -- already implemented in 25.C3 via `record_approval` -- untouched.
- **No `autonomous_loop.py` change needed.** Criterion 2 is satisfied by 25.B3's `load_promoted_params(bq)` wiring at line 132.
- **Reuse `save_promoted_strategy` for the registry write** (status arg flowed in via the row dict). Add `update_promoted_strategy_status` call on the prior active row (if any) to flip it to `superseded` -- this delivers the atomic switch semantics SHARP arxiv 2605.06822 validates (highest rval selected atomically; conservative gate already applied at the promoter level).
- **Slack formatter pattern:** mirror `format_escalation_alert` at `formatters.py:679-741` -- Block Kit header + sections + context footer. P0 visual (`:rotating_light:`) + `level: error` semantic.
- **`Promoter` is `@dataclass(frozen=True)`.** Adding `write_to_registry` as an instance method is valid (frozen only prevents attribute mutation). bq_client passed per-call.

## Hypothesis

Adding (a) `Promoter.write_to_registry(bq_client, trial, *, week_iso, slack_fn=None)`
that writes the new active row + supersedes the prior active + fires a P0
Slack alert when `promoter.promote(trial)["promoted"]==True`, and (b) a new
`format_strategy_switch(event)` Block Kit formatter -- closes the goal-c
gap WITHOUT touching the existing monthly HITL path (25.C3) or the daily
cycle reader (25.B3 already wired).

## Success criteria (verbatim from masterplan)

1. `promoter_writes_registry_with_status_active_on_gate_clear`
2. `autonomous_loop_uses_registry_as_primary_strategy_source`
3. `format_strategy_switch_slack_notification_implemented`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_R.py`

Live check (per masterplan):
`Live: a strategy switch event posts P0 Slack alert and is reflected in next-cycle decisions`

## Plan

1. **Promoter registry write** -- `backend/autoresearch/promoter.py`:
   - Add instance method `write_to_registry(self, bq_client, trial: dict, *, week_iso: str, slack_fn: Callable | None = None) -> dict`.
   - Run `self.promote(trial)` first; if `promoted=False`, return that verdict unchanged (no registry write, no Slack).
   - If `promoted=True`:
     - Try to fetch the prior active row via `bq_client.get_latest_promoted_strategy(status_filter=["active"])`. If found AND its `strategy_id` differs from the new trial's `trial_id`, call `bq_client.update_promoted_strategy_status(prior_id, "superseded", week_iso=prior_week)` -- per-row try/except (fail-open).
     - Build the new row dict with `status="active"` (NOT `pending`) + `allocation_pct = self.position_size(trial, capital=1.0)` (fractional; callers can scale). Other fields: `strategy_id=trial['trial_id']`, `week_iso`, `params` (JSON-string), `dsr`, `pbo`, `promoted_at=now`, `sortino_monthly` (from trial if present, else 0.0).
     - Call `bq_client.save_promoted_strategy(row)`. Per-call try/except (fail-open).
     - If a Slack callback was provided, build a payload via `format_strategy_switch({...})` and call `slack_fn(blocks, channel?)` -- per-call try/except.
   - Return a dict `{promoted: True, prior_strategy_id, new_strategy_id, alert_sent: bool}`.
2. **Slack formatter** -- `backend/slack_bot/formatters.py`:
   - Add `format_strategy_switch(event: dict) -> list[dict]` returning Block Kit JSON.
   - Required event keys: `new_strategy_id`, `prior_strategy_id` (may be None), `dsr`, `pbo`, `allocation_pct`, `switched_at` (ISO timestamp), `week_iso`.
   - Header block: `":rotating_light: Strategy Auto-Switch (P0)"`.
   - Section: bold "New active strategy" + fields {strategy_id, week, DSR, PBO, allocation %}.
   - Section: "Superseded" with prior_strategy_id (or "(none -- first promotion)" when null).
   - Context footer: "phase-25.R auto-switching policy * Closes red-line goal-c".
   - Mirror `format_escalation_alert`'s shape (rose-toned content via emoji + structured fields).
3. **Verifier** -- `tests/verify_phase_25_R.py` -- 10+ claims:
   - Claim 1: `Promoter.write_to_registry` instance method exists with the signature above.
   - Claim 2: `format_strategy_switch` exists in `formatters.py` and returns a list of dicts.
   - Claim 3: **Behavioral happy path** -- mock bq_client + slack_fn. Trial passes the gate. Assert: `save_promoted_strategy` called once with `row["status"] == "active"`; `update_promoted_strategy_status` called once on prior active id with new_status="superseded"; slack_fn called once with a list of blocks; return dict has `promoted=True`, `prior_strategy_id=<prior id>`, `new_strategy_id=<new>`.
   - Claim 4: **Behavioral gate-fail** -- trial fails the gate (e.g., shadow_days too low). Assert: `save_promoted_strategy` NOT called; `slack_fn` NOT called; return dict has `promoted=False` + reason.
   - Claim 5: **Behavioral first-promotion** -- prior active row does NOT exist (bq returns None). Assert: `save_promoted_strategy` called once with `status="active"`; `update_promoted_strategy_status` NOT called; slack_fn called.
   - Claim 6: **Behavioral fail-open BQ** -- `save_promoted_strategy` raises. Assert: function still returns a dict (no crash); slack_fn may or may not be called (document expectation: slack only after successful write to avoid lying about state).
   - Claim 7: **format_strategy_switch shape** -- output is a list with at least 3 dicts (header + section + context). Header contains `Strategy Auto-Switch`. Section contains `new_strategy_id`. Context contains `phase-25.R` or `goal-c`.
   - Claim 8: **format_strategy_switch with None prior** -- formatter handles `prior_strategy_id=None` gracefully (renders "first promotion" or similar; no Python None leaking to user).
   - Claim 9: `autonomous_loop.py:132` (or thereabouts) contains `load_promoted_params(bq)` call (criterion 2 -- registry-as-primary, satisfied by 25.B3).
   - Claim 10: `Promoter.write_to_registry` does NOT mutate any frozen-dataclass attribute (`@dataclass(frozen=True)` invariant preserved). Verifier inspects function source.
   - Claim 11: prior-active supersession passes `new_status="superseded"` (literal value).

## Non-goals

- No HITL approval path change (25.C3 owns that path).
- No `autonomous_loop.py` edit (25.B3 already wired the registry-as-primary).
- No allocation ramp-up logic (instant flip, per SHARP atomic-selection finding).
- No new BQ schema; reuses 25.A3 `promoted_strategies` table.
- No real-time signal-based regime-detection trigger (the gate is the trigger).

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/autoresearch/promoter.py:19-52` -- existing Promoter dataclass
- `backend/db/bigquery_client.py:save_promoted_strategy` (25.A3), `get_latest_promoted_strategy` (25.B3), `update_promoted_strategy_status` (25.C3)
- `backend/slack_bot/formatters.py:679-741` (`format_escalation_alert`) -- canonical Block Kit shape
- `backend/services/autonomous_loop.py:132` -- registry reader (criterion 2 wire, already done)
- `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md` -- red-line goal-c definition
- CLAUDE.md `Critical Rules` -- 30s BQ timeout (covered by 25.A3/C3 helpers)
