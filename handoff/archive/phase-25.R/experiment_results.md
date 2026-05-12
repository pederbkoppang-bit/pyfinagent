---
step: phase-25.R
cycle: 73
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_R.py'
title: Strategy auto-switching policy -- closes red-line goal-c (P1)
audit_basis: phase-24.13 F-3 (strategy-switching mechanism does not exist)
depends_on: 25.C3 (done, commit b51bb893)
---

# Experiment Results -- phase-25.R

## Code changes

### `backend/autoresearch/promoter.py`
- New imports: `json`, `logging`, `datetime`, `timezone`.
- New module-level `logger`.
- New instance method `Promoter.write_to_registry(bq_client, trial, *, week_iso, slack_fn=None) -> dict`:
  - Runs `self.promote(trial)` first; on gate-fail returns `{promoted: False, reason, ...}` and skips all side effects.
  - On gate-pass:
    - Looks up prior active row via `bq_client.get_latest_promoted_strategy(status_filter=["active"])`.
    - If a different prior strategy exists, flips it to `superseded` via `bq_client.update_promoted_strategy_status(prior_id, "superseded", week_iso=prior_week)`. Per-call try/except (fail-open).
    - Builds the new row dict (`status="active"`) and writes via `bq_client.save_promoted_strategy(row)`. Per-call try/except (fail-open).
    - If `slack_fn` is provided AND the BQ write succeeded, builds Block Kit payload via `format_strategy_switch(...)` and invokes `slack_fn(blocks)`. Per-call try/except. **Slack is NEVER fired after a failed write** -- the alert never lies about state.
  - Returns `{promoted, reason?, prior_strategy_id, new_strategy_id, alert_sent}`.
- `@dataclass(frozen=True)` invariant preserved -- the method does NOT mutate `self.*`.

### `backend/slack_bot/formatters.py`
- New `format_strategy_switch(event: dict) -> list[dict]` returns a 6-block Block Kit payload:
  - Header: `":rotating_light: Strategy Auto-Switch (P0)"`.
  - Section: bold "New active strategy" with id, week, switched_at.
  - Section with 6 mrkdwn fields: Strategy ID, Week, DSR, PBO, Allocation %, Switched at.
  - Section: "Superseded: \`<id>\`" or "Superseded: \`(none -- first promotion)\`".
  - Divider + context footer: ":robot_face: phase-25.R auto-switching policy * Closes red-line goal-c".
- Numeric fields use `_fmt_num` helper to handle None / non-numeric gracefully.

### `tests/verify_phase_25_R.py` (new file)
- 11 immutable claims with 6 behavioral round-trips:
  - Claims 1-2: structural (Promoter signature, formatter signature).
  - Claim 3: **Behavioral happy path** -- gate passes, prior active exists. Asserts `save_promoted_strategy` called once with `status="active"`, `update_promoted_strategy_status` called with `"superseded"`, `slack_fn` called once, return dict has expected ids.
  - Claim 4: **Behavioral gate-fail** -- shadow_days too low. Asserts NO writes + NO slack.
  - Claim 5: **Behavioral first-promotion** -- prior active=None. Asserts NO supersession call, save+slack still fire.
  - Claim 6: **Behavioral fail-open BQ** -- save raises. Asserts NO crash AND `slack_fn` NOT called (don't lie about state).
  - Claim 7: format_strategy_switch shape -- >=3 blocks, header present, new_id present, phase-25.R attribution.
  - Claim 8: None prior renders gracefully (no literal "None" leaks; uses "(none -- first promotion)" sentinel).
  - Claim 9: `autonomous_loop.py:132` uses `load_promoted_params(bq)` (criterion 2, satisfied by 25.B3).
  - Claim 10: Promoter remains `@dataclass(frozen=True)` and `write_to_registry` does NOT mutate `self.*`.
  - Claim 11: supersession call uses literal `"superseded"` string.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_R.py
write_to_registry: registry write fail-open for trial_boom: RuntimeError('BQ blew up')
PASS: promoter_write_to_registry_signature
PASS: format_strategy_switch_slack_notification_implemented
PASS: promoter_writes_registry_with_status_active_on_gate_clear
PASS: gate_fail_skips_registry_and_slack
PASS: first_promotion_skips_supersession_and_still_fires_slack
PASS: bq_failure_does_not_crash_and_does_not_lie_via_slack
PASS: format_strategy_switch_block_kit_shape
PASS: format_strategy_switch_handles_none_prior
PASS: autonomous_loop_uses_registry_as_primary_strategy_source
PASS: promoter_remains_frozen_dataclass
PASS: supersession_uses_superseded_literal

11/11 claims PASS, 0 FAIL
```

(The "write_to_registry: registry write fail-open for trial_boom" line is the
expected `logger.warning` from claim 6's fail-open behavioral test -- it
proves the fail-open path actually runs and logs as designed.)

## Backend gates

- `python -c "import ast; ast.parse(open('backend/autoresearch/promoter.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/slack_bot/formatters.py').read())"` -- OK
- 6 behavioral round-trips exercise actual code (happy / gate-fail / first-promotion / fail-open / formatter-shape / None-prior).

## Hypothesis verdict

CONFIRMED. All three immutable success criteria mapped:
- Criterion 1 (`promoter_writes_registry_with_status_active_on_gate_clear`) -- claim 3 behavioral round-trip asserts `save_promoted_strategy` called with `row["status"] == "active"` only on gate pass.
- Criterion 2 (`autonomous_loop_uses_registry_as_primary_strategy_source`) -- claim 9 grep on `load_promoted_params(bq)` wire (criterion satisfied by 25.B3; preserved by 25.R).
- Criterion 3 (`format_strategy_switch_slack_notification_implemented`) -- claims 2 + 7 + 8 cover signature, shape, and graceful None-prior handling.

Red-line goal-c ("dynamically shift strategy to whichever is making the most money") is now wired end-to-end: friday_promotion writes `pending` (25.A3) -> daily loop reads via load_promoted_params (25.B3) -> monthly HITL flips to `active` via record_approval (25.C3) -> OR promoter auto-flips to `active` + supersedes prior + fires P0 Slack (25.R).

## Live-check

Per masterplan: "Live: a strategy switch event posts P0 Slack alert and is reflected in next-cycle decisions".

Live evidence pending capture in `handoff/current/live_check_25.R.md`. Required: an operator-triggered or scheduler-fired `Promoter.write_to_registry(bq_client, trial, week_iso, slack_fn=slack_post_fn)` call with a real trial that clears the shadow+DSR gate; verify (a) Slack channel received the P0 block, (b) BQ `promoted_strategies` shows new row `status='active'` AND prior row `status='superseded'`, (c) next daily cycle's `load_promoted_params` log line shows the new params merged.

## Non-regressions

- `Promoter.promote` / `position_size` / `on_dd_breach` unchanged.
- `format_strategy_switch` is purely additive in formatters.py; no existing formatter touched.
- `autonomous_loop.py` unchanged (criterion 2 preserved by 25.B3).
- 25.C3 HITL path unaffected (uses `update_promoted_strategy_status` directly, not `write_to_registry`).
- Fail-open semantics consistent with 25.A3 / 25.C3 / friday_promotion: a BQ or Slack failure logs a warning but never crashes the caller.

## Downstream

Red-line goal-c is now closed end-to-end. Remaining red-line gap is **goal-d** (real-time `profit_per_llm_dollar` metric) which is covered by step **25.Q**.

## Next phase

Q/A pending.
