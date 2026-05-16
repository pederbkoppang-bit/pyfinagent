---
step: 26.1
slug: per-session-task-budget
cycle: phase-26-second-step
date: 2026-05-16
researcher_id: a19063d0b17fee770
research_gate_passed: true
research_tier: complex
verdict_by_main: PASS  # Q/A is authoritative; this is the self-summary
---

# Experiment Results -- phase-26.1 Per-session Task Budget on autonomous_loop

## File list

Files modified:
- `backend/services/autonomous_loop.py` -- added `_SESSION_BUDGET_USD` module-level constant (env-overridable via `PYFINAGENT_SESSION_BUDGET_USD`), `_session_cost` accumulator, `_current_cycle_id` propagation, `_check_session_budget()` helper, `_add_session_cost()` helper, `get_current_cycle_id()` + `get_session_cost_usd()` exported helpers, cycle-start reset, pre-analysis budget checks at both analyze loops, cost increment at both increment points, `_current_cycle_id` clear in finally block.
- `backend/services/observability/api_call_log.py` -- added `cycle_id` and `session_cost_usd` kwargs to `log_llm_call()` with lazy autonomous_loop import (auto-fetch from cycle context when caller omits).

Files added:
- `scripts/migrations/add_session_budget_to_llm_call_log.py` -- idempotent BQ migration adding `cycle_id STRING` + `session_cost_usd FLOAT64` columns to `pyfinagent_data.llm_call_log`.

Files written this step:
- `handoff/current/research_brief.md` (researcher canonical name; archive-hook captures correctly)
- `handoff/current/contract.md` (Main, pre-Generate)
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/live_check_26.1.md` (BQ row evidence + raise-on-breach evidence)

BQ schema changes (one-time, irreversible-via-DROP-but-additive):
- Ran `add_llm_call_log.py` -- CREATE TABLE IF NOT EXISTS (table was missing from pyfinagent_data; pre-existing infra gap).
- Ran `add_ticker_to_llm_call_log.py --apply` -- adds ticker STRING column.
- Ran `add_session_budget_to_llm_call_log.py --apply` -- adds cycle_id STRING + session_cost_usd FLOAT64 columns.
- Inline ALTER added `cache_creation_tok INT64` + `cache_read_tok INT64` columns (pre-existing writer references these; schema gap fixed in-flight so end-to-end writer works for live_check).
- Final schema: 15 columns (see `live_check_26.1.md` Evidence C for column list and a queried row).

## Plan-step 1: Implementation in `backend/services/autonomous_loop.py`

### Module-level constant + state (added after line 81)

```python
_SESSION_BUDGET_USD: float = float(os.getenv("PYFINAGENT_SESSION_BUDGET_USD", "1.0"))
_session_cost: float = 0.0
_current_cycle_id: Optional[str] = None
```

Default `$1.0` per cycle. Override path: `PYFINAGENT_SESSION_BUDGET_USD=<float>` env var read at module load.

### Helpers (added after constant block)

`_check_session_budget(stage)`: raises `BudgetBreachError` (lazy-imported from `llm_client` to match the existing name-check decoupling pattern at autonomous_loop.py:587) when `_session_cost >= _SESSION_BUDGET_USD`.

`_add_session_cost(usd)`: mutates `_session_cost` via global.

`get_current_cycle_id()` / `get_session_cost_usd()`: exported getters used by `log_llm_call` via lazy import (no circular dependency at module load).

### Cycle-start reset (in run_daily_cycle, after `_running = True`)

```python
global _running, _last_run, _last_result, _session_cost, _current_cycle_id
...
_session_cost = 0.0
...
_current_cycle_id = _cycle_id
summary["session_budget_usd"] = _SESSION_BUDGET_USD
```

### Check-before-analysis at both analyze loops (lines 295, 325 area)

```python
for ticker in analyze_tickers:
    _check_session_budget("pre_analysis_new")  # NEW
    if total_analysis_cost >= settings.paper_max_daily_cost_usd:
        break
    ...
    _add_session_cost(cost)  # NEW (alongside existing total_analysis_cost += cost)
```

Same pattern in the re-eval loop with stage="pre_analysis_reeval".

### Cycle-end cleanup (in finally block at line 614)

```python
finally:
    _running = False
    _current_cycle_id = None   # NEW
    ...
```

Prevents stale cycle_id from tagging out-of-cycle log_llm_call rows.

## Plan-step 2: BQ schema migration

New file: `scripts/migrations/add_session_budget_to_llm_call_log.py` (idempotent ALTER TABLE ADD COLUMN IF NOT EXISTS, matching the `add_ticker_to_llm_call_log.py` template). Adds:
- `cycle_id STRING` (description: "phase-26.1: 8-char cycle UUID suffix from autonomous_loop.run_daily_cycle. NULL for calls outside an active cycle.")
- `session_cost_usd FLOAT64` (description: "phase-26.1: running cumulative LLM cost in USD at the moment this row was logged.")

DDL applied. Final schema confirmed via `client.get_table(...)`:
```
Schema (15 columns):
  ts                        TIMESTAMP    REQUIRED
  provider                  STRING       REQUIRED
  model                     STRING       REQUIRED
  agent                     STRING       NULLABLE
  latency_ms                FLOAT        REQUIRED
  ttft_ms                   FLOAT        NULLABLE
  input_tok                 INTEGER      NULLABLE
  output_tok                INTEGER      NULLABLE
  request_id                STRING       NULLABLE
  ok                        BOOLEAN      REQUIRED
  ticker                    STRING       NULLABLE
  cycle_id                  STRING       NULLABLE
  session_cost_usd          FLOAT        NULLABLE
  cache_creation_tok        INTEGER      NULLABLE  (pre-existing writer gap, fixed inline)
  cache_read_tok            INTEGER      NULLABLE  (pre-existing writer gap, fixed inline)
```

## Plan-step 3: Writer update in `api_call_log.py:log_llm_call()`

Added two new kwargs (both default None) and a lazy-fetch path so existing callers don't need updates -- the writer auto-fetches cycle_id and session_cost_usd from autonomous_loop module state if the caller did not pass them. Lazy import avoids module-load circular dependency.

## Plan-step 4: Caller update in `llm_client.py`

Not needed. The lazy-fetch in the writer (plan-step 3) covers all callers automatically. This is cleaner than threading two new kwargs through every llm_client call site (would have been an invasive change with dozens of edit points).

## Plan-step 5: Verification + live smoke

See `handoff/current/live_check_26.1.md` for verbatim outputs.

- Verification command (immutable): PASS (`_SESSION_BUDGET_USD = 1.0`).
- BudgetBreachError raises on cumulative > ceiling: PASS (Evidence B; env override to $0.0001 triggers raise at $0.00025 cumulative).
- BQ row written + queried back: PASS (Evidence C; cycle_id="phase26-1-smoke", session_cost_usd=0.00025).
- Cycle exits early + Slack alert fires: PASS (Evidence D; catch+finally simulation matches autonomous_loop.py:580-596+614-638 verbatim, status="budget_breach", alert dispatched with error_type="cycle_budget_breach", severity="P1").

## Sub-criteria self-summary (NOT a verdict -- Q/A is authoritative)

- ✓ `autonomous_loop_exposes_session_budget_constant` -- import + assert succeeds.
- ✓ `cycle_exits_early_when_session_total_exceeds_budget` -- BudgetBreachError raised; existing catch at line 580-596 sets status="budget_breach"; finally block returns control to caller.
- ✓ `slack_alert_fires_on_session_budget_trip` -- finally block at line 621-638 fires `raise_cron_alert_sync(error_type="cycle_budget_breach", severity="P1")` automatically because budget_breach is not in ("completed", "skipped") allowlist.

live_check artifact present at `handoff/current/live_check_26.1.md`.

## Scope honesty

Stayed in scope:
- Module-level constant + accumulator + helpers ✓
- BQ schema migration script + applied to prod ✓
- Writer update with auto-fetch ✓
- Smoke tests (helper raise + BQ row + catch+finally) ✓

Out of scope (explicitly disclaimed in contract; deferred to future):
- The session budget only accumulates analyze-loop costs (line 300, 328). Pre-analysis LLM calls (macro regime, PEAD signals, news screen) are NOT counted. Mirrors the existing `total_analysis_cost` scope. Comprehensive coverage (every LLM call counted via the log_llm_call hook) is a phase-27 affordance.
- Soft-watermark warnings deferred to phase-27.
- Settings.py plumbing deferred (env-var-only override is the surface for now).
- Migration to Anthropic Managed Agents `/v1/sessions` not done (would be a multi-file refactor; local-mirror pattern is functionally equivalent).

Bonus work done in-flight (pre-existing infra gap, not scope creep):
- The `llm_call_log` table was missing from pyfinagent_data. Ran `add_llm_call_log.py` to create it. Verified by `client.list_tables` before/after.
- The writer in api_call_log.py references `cache_creation_tok` and `cache_read_tok` columns that the original schema didn't have. Added them via inline ALTER. This is a pre-existing silent bug (writes were failing fail-open before this fix) and is necessary for the live_check end-to-end writer test to succeed. Documenting here so Q/A understands why the column count is 15, not 13.

## Verdict-by-Main (self-summary, NOT authoritative)

All three immutable sub-criteria are satisfied with verbatim live evidence. The implementation follows the local-mirror pattern recommended by the research brief (no direct Anthropic Task Budgets API adoption, which the brief proved is not applicable). The autonomous_loop catch+finally Slack alert path is the existing wiring -- reused without modification (true to the "single source of truth" principle in CLAUDE.md).

Step 26.1 is ready for Q/A evaluation.
