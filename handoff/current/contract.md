# Contract -- Phase 4.2.4.2 BQ `signals_log` outcome-event append path

**Depends on:** Phase 4.2.4 publish-event write path (commit `e8d3bb3`).
**Research gate:** `handoff/current/research.md` (this cycle, 6 queries +
0223 archive inheritance).

## Goal

Wire `SignalsServer.track_signal_accuracy` into the durable
`signals_log` BigQuery table by appending a new event row with
`event_kind="outcome"` at each of the 3 successful return paths
after the in-memory record has been mutated to reflect the outcome.
No DML UPDATE. No schema change. No read-path change.

## Files touched

1. `backend/agents/mcp_servers/signals_server.py` (+N, tight budget)
   - NEW method `_save_outcome_event_to_bq(record: Dict[str, Any]) -> None`
     inserted after `_append_signal_history` and before `risk_check`.
   - 3 new call-site lines in `track_signal_accuracy` at the 3
     successful return paths (HOLD / missing_prices / scored).

No other files modified. No new imports. `json`, `datetime`,
`timezone`, `logger`, `Dict`, `Any`, `Optional` are already imported
at top of file (lines 16-29).

## Diff budget

- `<= 70` lines added to `signals_server.py`
- `<= 0` lines deleted from `signals_server.py`
- Net budget: `+70 / 0`, 100% additive.

Rationale: the publish-event wire-up in 0223 used ~30 lines inline;
this cycle wraps the builder in a method (~40 lines) and adds 3
call-site lines (3 lines). 40 + 3 = 43 + docstring + formatting
headroom = ~70 line budget.

## Success criteria (SC1-24)

### A. Scope discipline (SC1-5)

- **SC1.** Exactly one file modified: `backend/agents/mcp_servers/signals_server.py`.
- **SC2.** Net diff bound: `+N added / 0 deleted`, `N <= 70`.
- **SC3.** Zero new imports in modified file (top-of-file `import`
  and `from ... import` statements byte-identical pre/post).
- **SC4.** Zero new top-level statements. All new code is inside
  the `SignalsServer` class.
- **SC5.** No rename of any existing method, attribute, parameter,
  or local variable.

### B. New helper method `_save_outcome_event_to_bq` (SC6-13)

- **SC6.** Method exists as `SignalsServer._save_outcome_event_to_bq`
  (exactly this name, leading underscore).
- **SC7.** Method signature is `(self, record: Dict[str, Any]) -> None`.
  Return annotation is `None`; parameter annotation is `Dict[str, Any]`.
- **SC8.** Method is defined AFTER `_append_signal_history` and BEFORE
  `risk_check` in source order (stable location for BQ-adjacent helpers).
- **SC9.** Method body's first non-docstring statement is an early-return
  guard: `if self.bq_client is None: return`.
- **SC10.** Method builds a `bq_record: dict` with exactly 17 keys
  matching the `SIGNALS_LOG_SCHEMA` field names:
  `signal_id`, `ticker`, `signal_type`, `confidence`, `signal_date`,
  `entry_price`, `factors_json`, `created_at`, `outcome`, `scored`,
  `hit`, `exit_price`, `exit_date`, `forward_return_pct`,
  `holding_days`, `recorded_at`, `event_kind`.
- **SC11.** `bq_record["event_kind"]` literal string is `"outcome"`.
- **SC12.** `bq_record["recorded_at"]` is computed via
  `datetime.now(timezone.utc).isoformat(timespec="milliseconds")`
  (a fresh timestamp, NOT `record["timestamp"]`). This is the
  event-time of the outcome projection, distinct from the
  publish-time `created_at`.
- **SC13.** `bq_record["created_at"]` is sourced from
  `record["timestamp"]` (the original publish timestamp, preserved).

### C. Defensive boundary semantics (SC14-17)

- **SC14.** Method wraps `self.bq_client.save_signal(bq_record)` in
  `try: ... except Exception as e: logger.warning(...)`. The
  `except` clause catches `Exception` (not a narrower class).
- **SC15.** `logger.warning` call message is ASCII-only (per
  `.claude/rules/security.md`). No Unicode arrows, no em-dashes.
  Message pattern matches the publish-path `bq_signal_log` log
  prefix for grep-ability, e.g. `f"bq_signal_log outcome save
  failed: {type(e).__name__}"`.
- **SC16.** Method has zero `Raise` AST nodes in its body. Never
  raises under any branch.
- **SC17.** Method returns `None` in all paths (implicit, from
  the absence of any `Return` node with a value).

### D. `track_signal_accuracy` call-site additions (SC18-22)

- **SC18.** Exactly 3 new lines `self._save_outcome_event_to_bq(record)`
  added inside `track_signal_accuracy`. Each line appears in the
  source between the end of the record mutations and the start of
  the return dict construction for its respective path.
- **SC19.** HOLD path call-site: the new line appears between
  `record["holding_days"] = self._compute_holding_days(...)` and
  the `return {` that follows it (HOLD branch).
- **SC20.** Missing-prices path call-site: the new line appears
  between `record["forward_return_pct"] = None` (end of the
  missing_prices record mutations block) and the `return {` that
  follows it.
- **SC21.** Scored path call-site: the new line appears between
  `record["holding_days"] = self._compute_holding_days(...)` (end
  of the scored path mutations block) and the final `return {`
  of the method.
- **SC22.** No new lines added in the two early-return paths
  (invalid_signal_id / signal_not_found). Early-return paths do
  NOT emit outcome events (no mutated record to project).

### E. Byte-identity + global invariants (SC23-24)

- **SC23.** Every `SignalsServer` method EXCEPT `track_signal_accuracy`
  is `ast.dump()`-byte-identical between base commit `867d134`
  and HEAD of this cycle. Specifically including the publish-path
  scaffold: `publish_signal`, `_append_signal_history`,
  `_parse_iso_date`, `get_signal_history`, `validate_signal`,
  `generate_signal`, `_compute_holding_days`, etc.
- **SC24.** `ast.parse(open(file).read())` succeeds. `py_compile`
  succeeds. Zero non-ASCII bytes in the modified file.

## Adversarial probes (for QA evaluator)

- **A1.** `_save_outcome_event_to_bq({...full record...})` with
  `self.bq_client=None` -> returns None, no BQ call, no log.
- **A2.** Same with `bq_client` mocked, `save_signal` raising
  `RuntimeError` -> method catches, logs warning once, returns
  None, does NOT reraise.
- **A3.** Same with `save_signal` raising `ConnectionError` ->
  method catches via `Exception`, logs warning, returns None.
- **A4.** `_save_outcome_event_to_bq` built record has exactly
  17 keys; no extra, no missing.
- **A5.** `bq_record["event_kind"] == "outcome"`.
- **A6.** `bq_record["recorded_at"] != bq_record["created_at"]`
  (the two timestamps are distinct; recorded_at is fresh now(),
  created_at is from record["timestamp"]).
- **A7.** `track_signal_accuracy("", 100.0)` with empty signal_id
  -> invalid_signal_id early return, ZERO calls to
  `_save_outcome_event_to_bq`.
- **A8.** `track_signal_accuracy("unknown", 100.0)` with missing
  signal_id -> signal_not_found early return, ZERO calls to
  `_save_outcome_event_to_bq`.
- **A9.** A normal scored BUY -> ONE call to
  `_save_outcome_event_to_bq` with `record["outcome"]` populated
  as `"hit"` / `"miss"` / `"neutral"`.
- **A10.** A HOLD path -> ONE call to `_save_outcome_event_to_bq`
  with `record["outcome"] == "unscored"`.
- **A11.** A missing_prices path (entry=0) -> ONE call to
  `_save_outcome_event_to_bq` with `record["outcome"] == "error"`.
- **A12.** `logger.warning` message in `_save_outcome_event_to_bq`
  is ASCII-only (0 bytes > 127) and does not contain Unicode
  arrows (U+2192) or em-dashes (U+2014).

## Research-phase overrides (explicit)

The 0223 session's deferral note "After Storage Write API migration"
is overridden by this cycle's research gate (see
`handoff/current/research.md` category 1). The Storage Write API is
a prereq for DML UPDATE, not for append-only INSERT. We are
append-only by design, so no Storage Write API migration is required
for this cycle.

## Out of scope (explicitly deferred)

- DML UPDATE on signals_log rows (deferred indefinitely;
  event-sourced design means we never UPDATE).
- Storage Write API migration for `save_report`, `save_outcome`,
  `save_signal`, paper-trading inserts (deferred to a separate cycle).
- Read-path projection "latest outcome per signal_id" via window
  function / QUALIFY pattern (deferred to Phase 4.2.4.3).
- Schema changes to `signals_log` (17 fields unchanged).
- Migration script changes (byte-identical).
- `BigQueryClient.save_signal` changes (byte-identical).
- In-memory state changes in `track_signal_accuracy` (untouched
  except for the 3 new call-site lines).

## Verification recipe (for QA)

```python
import ast
p = 'backend/agents/mcp_servers/signals_server.py'
tree = ast.parse(open(p).read())
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == 'SignalsServer')
methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}

# SC6: helper exists
assert '_save_outcome_event_to_bq' in methods

# SC7: signature
m = methods['_save_outcome_event_to_bq']
assert [a.arg for a in m.args.args] == ['self', 'record']
assert isinstance(m.returns, ast.Constant) and m.returns.value is None

# SC16: zero Raise
assert not any(isinstance(n, ast.Raise) for n in ast.walk(m))

# SC18: 3 new call-site lines in track_signal_accuracy
ts = methods['track_signal_accuracy']
calls = [n for n in ast.walk(ts)
         if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Attribute)
         and n.func.attr == '_save_outcome_event_to_bq']
assert len(calls) == 3

# SC24: ASCII
assert sum(1 for b in open(p,'rb').read() if b > 127) == 0
```

## Anti-leniency

QA must independently run assertions in an isolated worktree (post-
commit snapshot), not trust my self-verification. QA must spawn
with `subagent_type="qa-evaluator"` (dedicated type, anti-leniency,
Opus). QA must fetch origin/main first and check out the GENERATE
commit's file before running the assertion block (per 2026-04-14-2300
finding: qa-evaluator subagent runs in isolated worktree at
`.claude/worktrees/agent-<hash>/`, sees only committed state).
