# Contract -- phase-82.54

**Step:** 82.54 (P1) -- a second live phantom-column defect the schema sweep
cannot see.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.54.md`,
`gate_passed: true`, **audit_class** with `dry: true` after 8 rounds / 2 dry,
8 sources read in full, 24 URLs, 13 internal files.

---

## 1. TWO PREMISES I WROTE INTO THIS STEP ARE REFUTED

I queued 82.54 yesterday-in-session with claims I had not derived. The gate
refuted two of them, and I verified both refutations myself.

1. **"the tile has been permanently NULL... reads to an operator as no LLM
   spend"** -- FALSE. `llm_tokens_today` has **ZERO consumers**: no reference
   anywhere in `frontend/src`, no `getCostBudgetToday` in `api.ts`, no
   `CostBudgetToday` in `types.ts`. The phase-15.1 tile was removed, and
   `docs/architecture/api-route-audit-2026-04-26.md` already says "Zero callers
   anywhere". **Nothing coerces `None` to `0`, so no operator was ever shown a
   false $0.** The defect is real; the consequence I asserted is not.
2. **"this may be the phase-75.5.1 $25/day metric"** -- FALSE. That metric is
   `spend.py::fetch_llm_spend`, a different function that dry-runs CLEAN and
   uses the correct columns. It reads dark because
   `settings.cost_budget_use_llm_spend_enabled` defaults OFF, not because of
   this bug.

This is the same failure mode as the rest of the week, one layer earlier: **I
put an underived consequence into a queued step, where a future executor would
have inherited it as established.**

## 2. What IS true, measured by me

```
token columns: ['input_tok', 'output_tok', 'cache_creation_tok', 'cache_read_tok']
ts type: TIMESTAMP
2026-08-05 -> {'calls': 154, 'naive': 353896, 'with_cache': 9159745}
today      -> {'calls': 0, 'naive': None, 'with_cache': None}
```

- The defect is confirmed at `$0` by dry run: *"Unrecognized name: input_tokens;
  Did you mean input_tok?"*
- **The blindness is proven first-hand:** `extract_sql_literals` returns **0**
  literals for this file. The `JoinedStr` branch keeps only `Constant` parts, so
  `FROM \`{project}.pyfinagent_data.llm_call_log\`` reassembles as
  `FROM \`.pyfinagent_data.llm_call_log\``, and `_FQ_TABLE_RE` requires a
  non-empty project group -- so `tables_in_sql` returns `[]` and `derive_scope`'s
  loop never runs.
- **`ts` is TIMESTAMP, day-partitioned, and the predicate prunes to 0 bytes.**
  Unlike 82.39's `created_at` and 82.21's `report_date`, there is no date trap
  here. **Do NOT rewrite the WHERE clause.**

## 3. THE FIX IS NOT A RENAME

Cache tokens are ~82.2M of the table's ~90.2M. Measured on 2026-08-05, the
naive `input_tok + output_tok` sum is **353,896** while including cache columns
gives **9,159,745** -- a **25.9x** difference. Renaming the two columns would
ship a number that under-reports by an order of magnitude and looks plausible.

**DECISION: sum all four token columns, and expose the breakdown.**
Rationale: the field is named `llm_tokens_today` -- it is a TOKEN count, not a
billed-cost figure, and cache-read/creation tokens are real tokens the provider
counted. Cost *weighting* (where cache reads are ~10x cheaper) already lives in
`spend.py::fetch_llm_spend`, which is correct and untouched by this step.
Returning one conflated number with no breakdown is what let a 26x error hide,
so the components are exposed alongside the total.

## 4. Immutable success criteria (verbatim)

1. "the repaired query is validated by a BigQuery dry run and reported valid,
   asserted by a test that FAILS against the current input_tokens/output_tokens
   projection"
2. "a fixture proves the repaired query returns a non-null token total for a
   period where llm_call_log demonstrably has rows, so the fix is not merely
   syntactically valid"
3. "every column identifier this file selects is derived from the source and
   checked against the live schema, the derived set asserted non-empty, and any
   further mismatch fixed or queued"
4. "a run in which the BQ call fails emits an operator-visible signal rather
   than only a warning log and a null tile, asserted by a test capturing the
   emitted signal"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_54_cost_budget_columns.py -q`

### Criterion 2 is VACUOUS AS WRITTEN, and the guard must exceed it

The query is an aggregate with **no GROUP BY**, so it always returns exactly one
row, and `COALESCE(..., 0)` makes that row always non-NULL. Proven: **today has
ZERO rows and the repaired query still returns `tokens=0, calls=0` -- non-NULL.**
So "returns a non-null token total" cannot fail.

The guard therefore asserts a **POSITIVE** total over a **FIXED** window with a
`calls > 0` precondition, so it cannot pass on an empty day. Not pinned to
"yesterday" -- the gate measured that 2026-07-26 has 1 call and 0 tokens.

*(The same aggregate shape also makes `if not rows: return None, None` dead code.)*

## 5. Plan

- **D1** -- extract the SQL to a **plain string literal** constant so the sweep
  can resolve the table (the 82.39 lesson), with the correct four-column
  projection and the breakdown exposed.
- **D2** -- alert on failure via `raise_cron_alert_sync` at **P1**, imported
  function-locally, reusing the 82.39/82.48 seam.
- **D3** -- criterion 3's derivation: enumerate every column identifier the file
  selects, structurally, assert the set non-empty, and recall-test it against a
  known alias (`tokens`, `calls`) since the gate measured that aliases are the
  classic false positive -- its own regex sweep produced **9 false positives out
  of 10** hits.
- **D4** -- queue the third live defect (§6).

## 6. A THIRD live defect, found by the gate and queued not fixed

`backend/services/observability/spend.py:115` calls
`raise_cron_alert_sync(..., detail=...)` but the signature is **`details`**.
Verified by reading both. That raises `TypeError`, which is swallowed by the
surrounding `except -> logger.debug`, so **that alert has never fired**.
Compounded: it is `severity="P2"`, and only `_CRITICAL_SEVERITIES` reach the
bot-token fallback while `slack_webhook_url` is empty -- so even with the kwarg
fixed it would not deliver. It is the only malformed site of 15 audited
repo-wide. **This guards the cost-budget hard-block being fail-open**, which is
why it is P1-class and gets its own step rather than a fix smuggled in here.

## 7. Non-scope

No change to the WHERE clause (`ts` is TIMESTAMP and already prunes to 0 bytes).
No change to `spend.py` (§6). No change to
`cost_budget_use_llm_spend_enabled`. The audit found 25 f-string-invisible
sites; the dry run proves exactly **one** is a real defect -- the rest are
recorded in the brief, and widening the sweep is 82.55's job. No live positions.

## 8. References

- `handoff/current/research_brief_82.54.md` (audit-class, dry after 2 rounds)
- Gould, Su, Devanbu, ICSE 2004 -- dynamically-generated queries fail only at runtime
- Google Cloud: dry-run validation, cost best practices
- Internal: `backend/api/cost_budget_api.py`,
  `backend/db/schema_oracle.py:63,199-208,477,550-566`,
  `backend/services/observability/spend.py:108-127`,
  `backend/services/observability/alerting.py:54,253-259`,
  `backend/tests/test_phase_82_39_outcome_rebuild_query.py`
