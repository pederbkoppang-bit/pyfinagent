# BLOCKER-3 HITL C/C gate e2e -- Evaluator Critique

**Cycle:** BLOCKER-3 (task #42) -- 2026-04-24
**Verdict:** PASS (single cycle, no respawn)
**Q/A agent:** qa (merged deterministic + LLM-judgment)

## Harness-compliance audit (5-item, all PASS)

1. Researcher before contract -- PASS. `blocker-3-research-brief.md` predates `contract.md` (17:58). JSON envelope: tier=moderate, external_sources_read_in_full=7 (>=5), urls_collected=17, recency_scan=true, internal_files_inspected=9, gate_passed=true.
2. Contract before code -- PASS. contract.md 17:58 predates monthly_champion_challenger.py 17:58, monthly_approval_api.py 18:00, hitl_gate_drill.py 18:00.
3. experiment_results.md with verbatim output -- PASS.
4. Log-last -- PASS (harness_log.md ends with blocker-2 entry).
5. First-cycle Q/A, no verdict-shopping -- PASS.

## Deterministic checks (all PASS)

| # | Check | Result |
|---|---|---|
| 1 | `grep -c "bq_fn" monthly_champion_challenger.py` | 9 (>=2) |
| 2 | `grep -c "_default_bq_logger" monthly_approval_api.py` | 3 (>=2) |
| 3 | syntax: `ast.parse(monthly_champion_challenger.py)` | OK |
| 4 | syntax: `ast.parse(monthly_approval_api.py)` | OK |
| 5 | syntax: `ast.parse(hitl_gate_drill.py)` | OK |
| 6 | `from backend.autoresearch import monthly_champion_challenger` | OK |
| 7 | `from backend.api import monthly_approval_api` | OK |
| 8 | Drill run: 4 steps (gate_fired, pending, approved, bq_row) | PASS |

## Mutation-resistance (both tests fired correctly)

A) **Drill tampered** -- change `len(bq_calls) != 1` to `len(bq_calls) != 99`.
   Drill FAILs step4 as expected. Restored (manual text restore since
   drill is untracked) -> PASS again.

B) **Production path tampered** -- remove the `_emit_deployment_log_row(...)`
   call from `record_approval` approved-branch in
   `monthly_champion_challenger.py`. Drill FAILs step4 with
   "expected 1 bq_fn call, got 0". Restored -> PASS again.

Both confirm the drill is not rigged AND the fix is actually wired.

## LLM judgment

- **Task #42 satisfied**: state visible in `monthly_approval_state.json`
  (step2+step3 disk reads) AND BQ log row emitted (step4).
- **Production path is real**: Q/A read `_default_bq_logger` and
  confirmed it constructs a real `bigquery.Client` and INSERTs a
  parameterized query into `pyfinagent_pms.strategy_deployments_log`.
  Drill uses a capture `bq_fn` only for hermeticity; the production
  API endpoint wires the real logger.
- **Scope honesty verified**: deferring the real-Slack-click path to a
  Peder-interactive follow-up is defensible because (a) `slack_fn` is
  dependency-injected and captured in the drill, (b) an E2E Slack
  click requires human interaction by design and isn't something a
  hermetic drill can simulate.

## Minor observations (not blockers)

- The drill file is currently untracked (new file). Q/A used
  manual text-restore for the mutation test instead of
  `git checkout`. File is still on disk, tests pass.
- `_default_bq_logger` is fail-open: a BQ write failure in production
  won't block the approval. This is the right tradeoff (UI state is
  the source of truth for the approval itself; BQ is audit).

## Violated criteria

None.

## Verdict

PASS. Main may now append the cycle entry to `handoff/harness_log.md`
and flip task #42 to completed.
