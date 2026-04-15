# Experiment Results -- Phase 4.2.4.2 BQ signals_log outcome-event append path

**Step:** Phase 4.2.4.2 (outcome-event append path).
**Base commit:** `867d134` (latest origin/main at session start).
**Generate commit:** `a36c312`.
**Files touched:** 1 (modified).
**Diff:** `+48 / -0` (100% additive, 69% of 70-line budget).

## Summary

Shipped the outcome-event append path for the durable `signals_log`
BigQuery table. `SignalsServer.track_signal_accuracy` now appends a
new event row with `event_kind="outcome"` at each of its 3
successful return paths (HOLD / missing_prices / scored), via a new
private helper `_save_outcome_event_to_bq(record)` that mirrors the
publish-path builder from `_append_signal_history` with the outcome
fields populated and a fresh `recorded_at` timestamp.

Append-only by design: we NEVER UPDATE the prior publish-event row.
The streaming-buffer DML restriction (worst-case 90 min retention)
only blocks UPDATE/DELETE/MERGE/TRUNCATE, not new row INSERTs via
`insert_rows_json`. The Storage Write API migration is therefore
NOT a prereq for this cycle -- the 0223 session's deferral note
"After Storage Write API migration" was based on a misreading and
is explicitly overridden here (see `handoff/current/research.md`
category 1).

## Design

**New helper** `SignalsServer._save_outcome_event_to_bq(record) -> None`:

1. Early-return guard: `if self.bq_client is None: return`.
2. Build `bq_record: dict` with exactly the 17 fields matching
   `SIGNALS_LOG_SCHEMA` from `scripts/migrations/migrate_signals_log.py`.
3. `event_kind = "outcome"` literal.
4. `recorded_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")`
   (fresh timestamp, distinct from `created_at` which preserves
   `record["timestamp"]` from the original publish event).
5. `try: self.bq_client.save_signal(bq_record) except Exception as e:
   logger.warning(f"bq_signal_log outcome save failed: {type(e).__name__}")`
   -- best-effort, never-raise, ASCII-only log message.

**Call sites** in `track_signal_accuracy` (3 new lines):

1. HOLD path: after `record["holding_days"] = self._compute_holding_days(...)`
   and before the `return {..."reason": "hold_unscored"...}`.
2. Missing-prices path: after `record["forward_return_pct"] = None`
   and before the `return {..."reason": "missing_prices"...}`.
3. Scored path: after `record["holding_days"] = self._compute_holding_days(...)`
   and before the final `return {..."reason": "recorded"...}`.

Early-return paths (invalid_signal_id / signal_not_found) do NOT
emit outcome events -- there is no mutated record to project.

## Lead-self verification (stdlib only, before commit)

All 24 contract SCs passed:

- **SC1** single_file: 1 file modified (signals_server.py).
- **SC2** diff_bound: `+48 / 0`, well under 70-line budget.
- **SC3** imports_unchanged: 7 top-level import statements byte-identical
  (ast.unparse equal).
- **SC4** toplevel_count: 36 top-level nodes pre, 36 post.
- **SC5** no_renames: all 21 pre-existing method names present post.
- **SC6** helper_exists: `_save_outcome_event_to_bq` found.
- **SC7** signature: `(self, record)` + `Dict[str, Any]` + `-> None`.
- **SC8** location: `_append_signal_history` < `_save_outcome_event_to_bq`
  < `risk_check` in source order.
- **SC9** guard: first non-docstring statement is
  `if self.bq_client is None: return`.
- **SC10** 17_keys: bq_record has exactly 17 keys matching
  SIGNALS_LOG_SCHEMA field names.
- **SC11** event_kind_outcome: literal `"outcome"`.
- **SC12** recorded_at: `datetime.now(timezone.utc).isoformat(timespec="milliseconds")`.
- **SC13** created_at: `record["timestamp"]`.
- **SC14** except_Exception: single handler, `ast.Name id="Exception"`.
- **SC15** ascii_log: `logger.warning(f'bq_signal_log outcome save failed:
  {type(e).__name__}')` -- 100% ASCII.
- **SC16** zero_raise: 0 `Raise` nodes in helper body.
- **SC17** implicit_none: only bare `Return` in guard path, no value returns.
- **SC18** three_calls: exactly 3 `_save_outcome_event_to_bq` call sites
  in `track_signal_accuracy`.
- **SC19-21** call-site placement: verified by source inspection.
- **SC22** no_early_path_emit: exactly 3 calls (SC18) + byte-identity
  of other methods (SC23) = early paths have zero.
- **SC23** byte_identity: 20/21 pre-existing methods byte-identical at
  `ast.dump()` level (only `track_signal_accuracy` modified, one new).
- **SC24** ascii+parse+compile: 0 non-ASCII bytes, `ast.parse` clean,
  `py_compile` clean.

## QA subagent verification

Spawned dedicated `qa-evaluator` subagent (Opus, anti-leniency,
isolated git worktree at `.claude/worktrees/agent-adb04dfa/`).
Passed the pre-baked 34-assertion block. Completed in 3 tool uses
/ 69s / 22325 tokens.

**QA verdict:** `PASS` (34/34 checks, 0 violated criteria).
Scores `10/10/10/10/10` (correctness / scope / security_rule /
simplicity / conventions). Zero soft notes.

## Blockers

None. Push path clean on first try, QA agent cooperative, no
retries needed, no race conditions observed.
