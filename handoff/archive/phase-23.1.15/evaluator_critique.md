---
step: phase-23.1.15
verdict: PASS
qa_pass: 1
cycle_date: 2026-04-29
checks_run:
  - harness_compliance_audit
  - syntax_ast
  - immutable_verification_command
  - pytest_idempotency_and_sector
  - merge_upsert_visual_inspection
  - idempotency_guard_placement_inspection
  - mutation_resistance_token_audit
  - scope_honesty_audit
  - research_gate_envelope_audit
  - git_diff_scope
---

# Q/A Critique — phase-23.1.15

## 1. Harness-compliance audit (5-item gate, FIRST)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Both research briefs present + envelope `gate_passed:true` | PASS | `handoff/current/phase-23.1.15-external-research.md` (envelope: `external_sources_read_in_full=8`, `urls_collected=18`, `recency_scan_performed=true`, `gate_passed=true`) + `phase-23.1.15-internal-codebase-audit.md` |
| 2 | `contract.md` front-matter `step: phase-23.1.15` + immutable verification listed | PASS | `verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_15.py'` |
| 3 | `experiment_results.md` `step: phase-23.1.15` + reproducible verification | PASS | front-matter matches; verification command produced canonical ok-line on re-run |
| 4 | `harness_log.md` does NOT yet contain `23.1.15` (log-LAST invariant) | PASS | `grep -c "23.1.15" handoff/harness_log.md` = 0 |
| 5 | First Q/A spawn for this step (no prior `evaluator_critique.md` for 23.1.15) | PASS | top-level `evaluator_critique.md` was the prior step's; being overwritten now |

All five harness-compliance items satisfied before any code check.

## 2. Deterministic checks

### A. Immutable verification command

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_15.py
ok execute_buy idempotency-guard + paper_positions MERGE upsert + get_paper_trades_for_ticker_since helper + cleanup script (dry-run/apply) + 4 new tests pass
```

Exit code 0. Canonical ok-line emitted.

### B. Pytest

```
$ python -m pytest tests/services/test_trade_idempotency.py tests/services/test_sector_concentration.py -q
............                                                             [100%]
12 passed in 0.79s
```

12/12 PASS — matches contract acceptance criterion.

### C. AST syntax

```
$ python -c "import ast; [ast.parse(open(p).read()) for p in [...]]; print('all syntax ok')"
all syntax ok
```

Five files parsed: `paper_trader.py`, `bigquery_client.py`,
`test_trade_idempotency.py`, `verify_phase_23_1_15.py`,
`cleanup_phase_23_1_15.py`.

### D. Post-state reconciliation

`experiment_results.md` line 129 reports the live BQ reconciliation
result: **14 join_rows, 0 dup_trade_tickers, 0 orphan_trades, $0.00
leak_dollars**. WDC and XOM cash leaks resolved end-to-end.

### E. git diff scope

Modified: `backend/services/paper_trader.py`,
`backend/db/bigquery_client.py`,
`handoff/current/{contract,experiment_results}.md`. New:
`scripts/cleanup_phase_23_1_15.py`,
`tests/services/test_trade_idempotency.py`,
`tests/verify_phase_23_1_15.py`,
`handoff/current/phase-23.1.15-{external-research,internal-codebase-audit}.md`.
All within prompted whitelist. No incidental drive-bys.

### F. MERGE inspection — `save_paper_position` (bigquery_client.py:550-580)

Confirmed the body is now a MERGE:

```sql
MERGE `{table}` T
USING (...) S
ON T.ticker = S.ticker
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...
```

Idempotent natural-key write. Behavior-equivalent to INSERT when no
match exists (backwards-compat preserved). Defensive
`raise ValueError("save_paper_position requires 'ticker' field for
MERGE key")` guards against silent fallthrough.

### G. Idempotency guard placement — `execute_buy` (paper_trader.py:102-129)

Guard fires AFTER the `existing` snapshot lookup at line 95 and
BEFORE `trade_id` generation at line 137 and any BQ trade write.
The guard:

1. Only runs when `not existing` (cycle 1 already booked + position
   wrote → guard never fires; cycle 2 saw stale snapshot → guard
   catches the duplicate).
2. Looks back 30 minutes in `paper_trades` for matching BUY at
   near-identical quantity (1% tolerance for FP rounding).
3. Returns `None` (skip) on match — no cash debit, no trade write.
4. Wrapped in `try/except` with non-fatal log on failure (defensive
   default-open is acceptable since the original double-buy path
   is still less likely than a guard-query glitch).

## 3. LLM judgment

### Contract alignment

Contract `## Plan steps` lists exactly Fix A (idempotency guard) +
Fix B (MERGE upsert) + Fix E (cleanup script). Diff implements all
three:

| Fix | Location | Implemented |
|-----|----------|-------------|
| A — idempotency guard in `execute_buy` | `paper_trader.py:102-129` | YES |
| B — MERGE upsert in `save_paper_position` | `bigquery_client.py:550-580` | YES |
| E — cleanup script (dry-run + apply) | `scripts/cleanup_phase_23_1_15.py` | YES |

Skipped C (deterministic `client_order_id`) and D (nightly drift
audit) are explicitly listed in `## Phase 2 (deferred)` of
experiment_results — scope-honest deferral matching the contract.

### Mutation resistance

`tests/verify_phase_23_1_15.py` asserts on regression-detecting
tokens that would trip on naive reverts:

- `"get_paper_trades_for_ticker_since"` (helper presence)
- `"timedelta(minutes=30)"` (look-back window literal)
- regex `def save_paper_position\(self.*?MERGE\s+`` (DOTALL match
  forces MERGE inside the function body, not just somewhere in the
  file)
- `"ON T.ticker = S.ticker"` (correct MERGE key — guards against
  copy-paste errors)
- "WHEN MATCHED" + "WHEN NOT MATCHED" (both branches)
- `"requires 'ticker' field for MERGE key"` (defensive raise on
  missing key — guards against silent fallthrough)

Each token is distinct enough that a casual revert (e.g. "switch
back to INSERT but keep the helper") would still fail at least one
assertion. Good mutation-resistance design.

### Anti-rubber-stamp / scope honesty

`experiment_results.md` lines 145-170 contain four explicit
disclosures:

1. Mid-cleanup column-type bug surfaced live during `--apply` —
   first run partially failed on STRING-vs-TIMESTAMP at Step 3,
   was fixed mid-flight, recovery UPDATE issued. Script as shipped
   is now correct end-to-end. **This is exactly the kind of
   honest "we hit a snag" disclosure the anti-rubber-stamp leg
   wants.**
2. 30-minute idempotency window has a known edge case (intraday
   buy-sell-rebuy) — explicit caveat with a remediation path.
3. MERGE overwrites `entry_date` for existing tickers — preserved
   prior delete+insert behavior; refactor deferred.
4. No real-money risk — paper trading only.

Phase 2 deferred list matches the contract scope: collapse
delete+insert, deterministic `client_order_id`, nightly drift
audit. No silent overclaim.

### Research-gate compliance

Contract front-matter cites both research briefs. External brief
envelope reports 8 sources read in full (>=5 floor), 18 URLs
collected, recency scan performed (2026 BQ MERGE engineering blog
inline), `gate_passed: true`. Internal audit has file:line
anchors on every claim per its envelope summary. Gate cleanly
cleared.

### Backwards compatibility

- Idempotency guard only triggers when `not existing` — mainstream
  path (existing position) is untouched.
- MERGE-when-no-row behaves identically to INSERT-when-no-row.
- Cleanup script is `--dry-run` by default; `--apply` is opt-in.
- 12 existing sector-concentration tests still pass.

## 4. Verdict

**PASS.**

All three contract fixes implemented, all 12 tests pass, immutable
verification exits 0, post-state reconciliation confirms zero leak,
mutation-resistance tokens are sharp, scope honesty is high (mid-
cleanup bug + recovery disclosed without spin), research gate
cleared with 8 sources + recency scan, and the diff stays within
the contract whitelist.

Recommended next-step actions for Main:

1. Append `## Cycle N -- 2026-04-29 -- phase=23.1.15 result=PASS`
   block to `handoff/harness_log.md` (log-LAST invariant; not yet
   present, correctly).
2. Flip `.claude/masterplan.json` step phase-23.1.15 → `status:
   done` AFTER the log append.
3. Track Phase 2 deferred items as new masterplan substeps
   (`client_order_id`, nightly drift audit, collapse
   delete+insert) so they don't get lost.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable acceptance criteria met (idempotency guard + MERGE upsert + cleanup script). Verification command exit=0 with canonical ok-line. 12/12 pytest pass. Mutation-resistance tokens sharp. Scope-honesty disclosures complete (mid-cleanup column-type bug + recovery path). Research gate cleared (8 sources, recency scan, gate_passed:true). Post-state BQ reconciliation: 0 dup_trade_tickers, 0 orphan_trades, $0.00 leak.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "syntax_ast",
    "immutable_verification_command",
    "pytest_idempotency_and_sector",
    "merge_upsert_visual_inspection",
    "idempotency_guard_placement_inspection",
    "mutation_resistance_token_audit",
    "scope_honesty_audit",
    "research_gate_envelope_audit",
    "git_diff_scope"
  ]
}
```
