# Experiment Results -- phase-82.54 (cycles 1-3)

**Step:** 82.54 (P1). **Date:** 2026-08-06.
**Contract:** `handoff/current/contract_82.54.md`.
**Research brief:** `handoff/current/research_brief_82.54.md`
(`gate_passed: true`, **audit_class**, `dry: true` after 8 rounds / 2 dry).

---

## 1. TWO PREMISES I WROTE INTO THIS STEP ARE REFUTED

I queued 82.54 myself, earlier today, with consequences I had not derived. The
gate refuted two; I verified both refutations.

1. **"the tile reads to an operator as no LLM spend"** -- FALSE.
   `llm_tokens_today` has **ZERO consumers**: nothing in `frontend/src`, no
   `getCostBudgetToday` in `api.ts`, no `CostBudgetToday` in `types.ts`. The
   phase-15.1 tile was removed, and an existing route audit already says "Zero
   callers anywhere". Nothing coerces `None` to `0`, so **no operator was ever
   shown a false $0.**
2. **"this may be the phase-75.5.1 $25/day metric"** -- FALSE. That is
   `spend.py::fetch_llm_spend`, a different function that dry-runs clean. It
   reads dark because its settings flag defaults OFF.

The defect is real. The consequences I asserted were not. **This is the week's
failure mode one layer earlier: I put an underived claim into a QUEUED STEP,
where a future executor would have inherited it as established fact.**

## 2. Measured, by me

```
token columns: ['input_tok', 'output_tok', 'cache_creation_tok', 'cache_read_tok']
ts type: TIMESTAMP
2026-08-05 -> {'calls': 154, 'naive': 353896, 'with_cache': 9159745}
today      -> {'calls': 0, 'naive': None, 'with_cache': None}

NEW dry run VALID, bytes= 0
OLD dry run 400 -> Unrecognized name: input_tokens; Did you mean input_tok?
sweep literals in this file (was 0): 1
```

## 3. THE FIX IS NOT A RENAME

Cache tokens dominate: **353,896 vs 9,159,745 -- 25.9x**. Renaming the two
columns would have shipped a number under-reporting by an order of magnitude
and looking entirely plausible. **DECISION: sum all four token columns and
expose the breakdown**, because the field is a TOKEN count, not a billed-cost
figure -- cost weighting already lives correctly in `spend.py`. Returning one
conflated number with no components is what let a 26x error hide.

`ts` is TIMESTAMP and day-partitioned, so the predicate prunes to **0 bytes**.
Unlike 82.39's `created_at` and 82.21's `report_date` there is no STRING-date
trap; the WHERE clause is deliberately untouched.

## 4. CRITERION 2 IS VACUOUS AS WRITTEN, and the guard exceeds it

No GROUP BY + `COALESCE` means the query always returns exactly one non-NULL
row. **Measured: today has ZERO calls and still returns `tokens=0, calls=0`.**
So "returns a non-null total" cannot fail. The guard asserts a **POSITIVE**
total over a **FIXED** day with a `calls > 0` precondition, plus a companion
test pinning that a zero-traffic day is still non-null -- which is why non-null
proves nothing. *(The same shape also makes `if not rows: return None, None`
dead code.)*

## 5. Verbatim verification output (REGENERATED at cycle 3)

The cycle-2 Q/A found this block stale -- it still carried cycle-1's `12 passed`
while the file had 13 tests, contradicting section 7 of this same artifact. A
"verbatim" capture must be regenerated, never carried forward. It is now
regenerated as the last action of every cycle, with the ruff DERIVATION shown
rather than abbreviated (cycle 2 applied that remedy additively in section 10
and never to this block, which is the defect repeating one level up).

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_54_cost_budget_columns.py -q
..............                                                           [100%]
14 passed in 14.83s

$ python -m pytest backend/tests/ -q -k "cost_budget or 82_54 or 82_12 or spend or budget or 82_39"
97 passed, 2694 deselected, 1 xfailed, 1 warning in 30.64s

$ FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files -o --exclude-standard -- '*.py'; } | sort -u )
$ test -n "$FILES" || exit 1
$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811
All checks passed!
exit=0
```

Derived sizes, regenerated last:

```
$ git diff --numstat -- backend/api/cost_budget_api.py
107	16	backend/api/cost_budget_api.py

$ wc -l backend/tests/test_phase_82_54_cost_budget_columns.py
     368 backend/tests/test_phase_82_54_cost_budget_columns.py
$ python3 -c "ast walk for test_ functions"
14
```

## 6. Mutation matrix

| # | Mutant | Result |
|---|---|---|
| M1 | revert to the phantom columns | KILLED |
| M2 | drop the cache pair from the total (26x undercount) | KILLED |
| M3 | make the SQL an f-string again (blinds the sweep) | KILLED |
| M4 | delete the alert call | KILLED |
| M5 | downgrade the alert to P2 | KILLED |

**5 of 5 killed.** Licenses "these 5 died", not "no survivor exists".

### A defect my own recall test caught

`test_the_derivation_excludes_aliases_a_RECALL_TEST` FAILED on first run. My
identifier derivation filtered out any name appearing in the alias set -- but in
the pre-fix query `COALESCE(SUM(input_tokens), 0) AS input_tokens` makes
`input_tokens` **both a read and an alias**, so the filter erased the genuine
phantom read. The derivation was blind to exactly the defect it existed to find.
Fixed by relying on the `AS <alias>` substitution alone, which is what
`schema_oracle` does. **Precision bought with recall is not precision**, and the
recall test is the only reason I know.

## 7. Files changed

| File | Change |
|---|---|
| `backend/api/cost_budget_api.py` | `+107 / -16` -- plain-literal SQL, four-column projection with breakdown, P1 alert |
| `backend/tests/test_phase_82_54_cost_budget_columns.py` | NEW, 368 lines, 14 tests |
| `.claude/masterplan.json` | queues 82.58 (§8) |

## 8. A THIRD live defect, queued as 82.58 (P1)

`spend.py` calls `raise_cron_alert_sync(..., detail=...)` against a `details=`
signature -- verified by reading both. That raises `TypeError` into its own
swallowing `except -> logger.debug`, so **the alert has never fired**, and its
failure was never even visible at INFO. It is the only malformed site of 15
audited repo-wide.

It matters more than a typo: the message it fails to send says the cost-budget
spend fetch is degraded and *"the daily/monthly budget hard-block cannot trip
while this persists"*. Compounded -- it is `severity="P2"`, and only
`_CRITICAL_SEVERITIES` reach the bot-token fallback while `slack_webhook_url` is
empty, so a kwarg-only fix yields an alert that is built correctly and then
dropped. Both must change, which is why it is its own step.

## 9. Non-scope

The audit found **25** f-string-invisible SQL sites; the dry run proves exactly
**one** is a real defect -- this one. The gate's own regex flagged 10 and **9
were false positives** (string literals, CTE names, INTERVAL units), which is
itself the evidence that widening the sweep needs care. That widening is 82.55.
No WHERE-clause change. No `spend.py` change. No live positions.

---

## 10. Cycle-2 corrections (Q/A CONDITIONAL -> fixed)

Three findings. Two were right and one I refuted by measurement.

**F1 -- the live endpoint was still serving the PRE-FIX module, and I had not
checked.** `uvicorn` pid 60478 started 2026-08-05 17:38:35, before my edit, with
no `--reload`, so `curl /api/cost-budget/today` returned
`"llm_tokens_today": null` -- the fail-open path -- while the repaired query
returns 0. The Q/A even ruled out a cache artifact by waiting out the 60s TTL.
I had run 12 green tests and never touched the running system. CLAUDE.md's rule
is explicit: a claim about the running system needs a restart plus a re-curl.

Fixed: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`, health-poll,
re-curl. **And the restart immediately exposed a 500 my green tests had not
caught** -- see below.

**F2 -- "expose the breakdown" was FALSE at the API surface**, and the claim was
load-bearing. The function returned only `(tokens, calls)`, discarding the four
components, and `CostBudgetToday` had no breakdown fields -- while the contract,
this artifact, AND the production docstring all said the components were
exposed. My entire stated rationale for summing four columns is that *a single
conflated number is what let a 26x undercount hide*. Shipping the conflation
while claiming the fix would have reproduced the defect and its excuse together.
Now genuinely exposed, pinned by
`test_the_breakdown_reaches_the_RESPONSE_MODEL_not_just_the_query`.

**F3 -- REFUTED by measurement.** The Q/A judged my ruff scope structurally
excluded the untracked new test file. It did not: my command unions
`git ls-files --others --exclude-standard`, and re-running it prints both files.
The artifact rendered only `echo "$FILES" | xargs ...` without the derivation
line, which is what made it look narrower. The fix is to SHOW the derivation,
not to change the gate:

```
$ FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files -o --exclude-standard -- '*.py'; } | sort -u )
$ test -n "$FILES" || exit 1
$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811
All checks passed!   exit=0
```

**A NOTE the Q/A found and I fixed:** `_selected_identifiers` used
`[a-z_][a-z0-9_]{2,}`, a 3-character floor blind to `ok` and `ts` -- **both real
columns of this very table**. No such name is in the current SELECT list, so the
criterion held, but the docstring's universal claim over-reached by one
character class. Widened to `{1,}` with the keyword stoplist extended.

### The 500 the restart exposed, and what it says about my method

After exposing the breakdown, the live endpoint returned **HTTP 500** while all
13 tests stayed green. Cause: a SECOND unpack site,
`tokens, calls = await asyncio.to_thread(...)`, that my `str.replace(..., 1)`
never touched -- the single-occurrence assumption, which is a named recurring
failure in this project.

Two things caught it and neither was a test: the **live curl**, and an assertion
I had written to fail if the replace matched nothing (it fired on an indentation
mismatch rather than silently no-op'ing). A subsequent regex pass then produced
a duplicate-kwarg `SyntaxError`, which a count-based assertion caught before it
could ship. I stopped regex-editing at that point and fixed the region by
reading it.

**LIVE PROOF, after restart:**

```
$ launchctl list | grep com.pyfinagent.backend$
79058	-15	com.pyfinagent.backend

$ curl -s http://127.0.0.1:8000/api/cost-budget/today
{
    "llm_tokens_today": 0,
    "llm_input_tokens_today": 0,
    "llm_output_tokens_today": 0,
    "llm_cache_creation_tokens_today": 0,
    "llm_cache_read_tokens_today": 0,
    ...
}
```

`0` rather than `null` is correct: today has zero calls, and the pre-fix module
returned `null` from the fail-open path.

---

## 11. Cycle-3 corrections (Q/A CONDITIONAL #2 -> fixed)

Four findings, all correct.

**F1 -- THE GUARD I ADDED TO CLOSE F2 WAS ITSELF ILLUSORY.** It asserted the
four names exist in `CostBudgetToday.model_fields` -- a CLASS-LEVEL SCHEMA FACT,
true of every possible response. The Q/A's M_C (keep the field, stop populating
it) SURVIVED all 13 tests. **A guard written to fix a stop-one-seam-short
finding stopped one seam short.**

**F2 -- the HTTP 500 class had ZERO coverage.** The Q/A measured it: no test
anywhere referenced `get_cost_budget_today`, so unpack arity, kwarg wiring and
breakdown population were ALL live-check-only. Its M_B (revert to the 2-target
unpack -- byte-for-byte the regression that shipped a 500) survived every test.

Both closed by ONE test that drives the real endpoint with the fetch patched and
asserts response VALUES. Re-measured, all three mutants now die:

```
control: GREEN
[KILLED] M_B revert to the 2-target unpack (the shipped 500)
[KILLED] M_C keep the field but stop populating it
[KILLED] M_D drop a component from the response model
restored: GREEN
```

Writing it also exposed that I had guessed the cache seam twice
(`api.cache` does not exist; `delete()` does not exist -- it is `invalidate()`).
Both failed loudly rather than silently passing.

**F3 -- section 5 did not reproduce** (`12 passed` vs 13; `95` vs 96). Stale
cycle-1 transcription, arithmetically explained by +1 here and +13 from a
foreign session's untracked test file. Regenerated above, WITH the ruff
derivation this time -- cycle 2 had applied that remedy additively in section 10
and never to the offending block.

**F4 -- the skip trapdoor, undisclosed.** 5 of the tests are conditional on
`PYFIN_SKIP_LIVE_BQ`, and they include BOTH criterion-1 dry runs and BOTH
criterion-2 fixtures -- so with the opt-out set the verification command exits 0
with those criteria never executed. Not set in this environment (0 skips), so
the criteria are genuinely demonstrated today. Now disclosed here AND enforced
by a NON-skippable guard that fails if the opt-out is active, so the gate cannot
silently empty.
