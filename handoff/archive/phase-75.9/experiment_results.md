# Experiment results -- Step 75.9 (BQ fail-closed dedup, parameterization, timeout sweep, cost guard)

Date: 2026-07-23. **Execution model (operator directive + step executor tag):
GENERATE was delegated to a Sonnet-4.6 executor agent; Main (this session)
wrote the contract, reviewed the diff, independently re-measured every
headline figure below, and authored this artifact. The executor's own
run-then-write draft is preserved verbatim at
`handoff/current/experiment_results_75.9_draft.md` (its section 2 lists 8
explicit deviations, all reviewed and endorsed by Main -- none silently
absorbed).**

## What was built (per contract; details + per-site tables in the draft)

- **(a) data-bq-01**: `_get_existing_price_dates` + `_get_existing_fundamentals`
  now log + re-raise on query exception (fail-closed); a SUCCESSFUL empty
  result still returns `set()` (cold-start insert-all unchanged, proven by a
  dedicated test). `_get_existing_macro` untouched (frozen; queued 75.9.1).
- **(b) data-bq-02**: `get_agent_memories` -> ScalarQueryParameter (agent_type
  + limit); both data_ingestion ticker queries -> ArrayQueryParameter +
  `IN UNNEST(@tickers)`. Table-name f-strings retained (identifiers are not
  parameterizable -- official BQ doc).
- **(c) data-bq-03/gap3-08/gap6-09**: timeout added to every untimed BQ
  `.result()`: 18 sites in `bigquery_client.py` (DML sites at 60 -- all 5,
  a disclosed widening of the contract's two examples under its own stated
  rule), the corrected external sites (paper_trader, cycle_health,
  metrics/sortino, api/paper_trading, api/performance_api,
  services/pead_signal, services/sector_calendars, skill_optimizer x2,
  slot_accounting, api/harness_autoresearch, monthly_approval_api), and 13
  migration files / 20 sites at 60. `cost_budget_watcher.py` confirmed
  phantom (zero BQ calls) -- untouched. ThreadPool `future.result()` sites
  excluded by design.
- **(d) data-bq-06**: `MAX_BYTES_BILLED_DEFAULT = 5 * 1024 ** 3` (= 5368709120,
  documented) + one shared `_job_config()` factory; all of
  bigquery_client's own query paths adopt it (24 pre-existing QueryJobConfig
  sites + 4 previously-bare paths -- a disclosed, additive widening of
  criterion 4's "at least one").
- **(e) py-core-03**: skill_optimizer's bare `except: pass` -> warning +
  degraded return mirroring its sibling; slot_accounting reuses one
  module-level client, timeout=30.
- **(f) perf-11**: zero-arg `@lru_cache get_bq_client()`; adopted at ALL 20
  inline construction sites in api/paper_trading.py (contract said "8" --
  measured 20; migrating all stays inside the named file),
  api/performance_api.py, api/reports.py.

## Change surface (measured)

`git diff --stat HEAD`: **37 files changed, 935 insertions(+), 351 deletions(-)**
(30 .py files + masterplan 75.9.1 insert + handoff artifacts). New:
`backend/tests/test_phase_75_bq_discipline.py` (45 tests),
`backend/governance`-adjacent files: none. Boundary held: no schema/table
changes, no historical_macro-path behavior change, no .env edits.

Out-of-worklist changes, all disclosed + reviewed:
1. `backend/tests/test_phase_slack_digest_71.py` -- test double's `result()`
   signature accepts `**kwargs` (consumer-contract ripple of the timeout
   sweep; production untouched).
2. `backend/services/observability/api_call_log.py` +
   `backend/tests/test_phase_66_3_cost_truth.py` -- a REAL pre-existing
   test-isolation bug surfaced by the regression run: the LLM buffer never
   got the phase-56.2 `reset_*_for_test` fix its sibling buffer has, so any
   full-suite run whose wall-clock crosses the 60s flush window before that
   file drains the row a test just injected. Executor isolated it by
   prefix-bisection WITH ZERO 75.9 SOURCE EDITS PRESENT (not a 75.9
   regression), then added the mirroring additive helper
   `reset_llm_buffer_for_test()`.
3. Six pre-existing F401 dead imports removed in touched files (75.5
   precedent): four found by the executor (reports.py traceback,
   bigquery_client.py local timezone re-import, metrics/sortino.py Any,
   create_strategy_deployments_view.py Tuple) and TWO found by Main's
   independent git-derived-scope lint that the executor's hand-derived list
   missed (`api_call_log.py` dataclasses.field, `test_phase_slack_digest_71.py`
   pytest) -- both proven pre-existing at HEAD via `git show HEAD:` lint.
   The executor's "All checks passed!" claim was therefore measured over an
   incomplete scope; Main's re-lint over the full 30-file git-derived scope
   is the figure of record below.

## Verification (ALL figures independently re-measured by Main, not
transcribed from the executor)

- Immutable command: `.venv/bin/python -m pytest backend/tests/test_phase_75_bq_discipline.py -q`
  -> **45 passed, exit 0** (Main re-run).
- Ruff `--select F821,F401,F811` over the git-derived 30-file scope
  (non-empty guard: scope_files=30) + the new test file -> **"All checks
  passed!", exit 0** (Main re-run, after the 2 extra F401 removals).
- Full-suite regression (Main re-run): **10 failed / 1370 passed / 12
  skipped / 5 xfailed / 1 xpassed** -- the fail set is BYTE-IDENTICAL to the
  pre-75.9 baseline 10 (symmetric difference EMPTY), and 1370 = 1325
  baseline + exactly the 45 new tests. Zero regressions attributable to the
  step. (Executor's first run had exposed 5 extra failures; its two
  disclosed fixes -- #7/#8 above -- returned the set to baseline.)
- `ast.parse`: clean on the touched files (executor draft section 6; Main
  spot-relied on ruff+pytest which parse everything in scope).

## Mutation matrix

- Executor matrix (scripted, exactly-once + byte-restore asserted;
  scratchpad `mutation_matrix_75_9.py`): **10 applied / 10 KILLED / 0
  survivors** -- dedup revert, STUB blank-fixture (crit-1 fixture can
  represent the failure), parameterization strip, timeout drop in EACH
  scanned group (client/external/migration), scan-at-missing-path (errors,
  not skips-green), cost-cap None, lru removal, bare-pass restore.
- Main independent spot-checks: **M1 KILLED** (1 failed), **M6 KILLED**
  (3 failed) -- with one honest correction: Main's FIRST M6 attempt mutated
  the literal `5368709120` which exists only in a COMMENT (code uses
  `5 * 1024 ** 3`), and correctly SURVIVED -- an invalid mutant carrying
  zero information, disclosed rather than dropped (cycle-131 N3 precedent).
  The re-run against the real constant killed 3 tests.
- Per the cycle-131 rule this licenses only the named kills, not a global
  no-vacuous-guards claim.

## Not verified live

- No live BQ query executed (all offline mocks); the timeout/cost-cap
  behavior on real jobs lands on the next natural query cycle. No backend
  restart performed. No UI surface. Migration scripts edited but NOT re-run
  (idempotent DDL; client-side kwarg only).
