# Experiment Results -- phase-82.39 (cycles 1-3)

**Step:** 82.39 (P1). **Date:** 2026-08-06.
**Contract:** `handoff/current/contract_82.39.md`.
**Research brief:** `handoff/current/research_brief_82.39.md`
(`gate_passed: true`, **audit_class**, `dry: true` after 10 rounds / 2 dry).

---

## 1. What was built

### D1 -- a real seam, then the repair

`build_ledger_fetch_sql()` extracted in
`backend/slack_bot/jobs/_production_fns.py`; `_fetch` calls it. The SQL was
inline before, so the only way a test could reach it was to copy it -- and a
copied string proves nothing about what production issues.

Repaired: `created_at` + `realized_pnl_pct AS pnl`, predicate
`SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)`,
`realized_pnl_pct IS NOT NULL` retained. `SAFE.TIMESTAMP` because `created_at`
is a STRING column; the idiom is per-column and NOT portable (on a native
TIMESTAMP column it 400s).

### D2 -- fail-open, no longer fail-silent

`_alert_fetch_failure()` dispatches `raise_cron_alert_sync` at **P1** from the
`except` branch, reusing the seam phase-82.11 established. The broad `except`
stays -- a nightly job must not crash the scheduler. P1 and never P2: with
`slack_webhook_url` empty a P2 is logged and dropped, which is the same
invisibility this step removes.

### D3 -- the criterion-4 sweep, with its recall stated

`derive_scope` re-run; `scope` non-empty, `unknown_columns` empty, and the
instrument's measured recall limit recorded in a guard so a clean sweep can
never be mistaken for a clean repo.

### D4 -- two phase-82.12 guards rewritten, preserving intent

See §5.

---

## 2. Verbatim verification command output

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_39_outcome_rebuild_query.py -q
................                                                         [100%]
16 passed in 6.23s
```

Regression + lint (scope DERIVED from git, asserted non-empty, piped through
`xargs` so word-splitting cannot silently lint zero files):

```
$ python -m pytest backend/tests/ -q -k "production_fns or slack or job or schema or 82_12 or 82_39 or outcome or nightly"
217 passed, 1 skipped, 2504 deselected, 1 warning in 18.35s

$ FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files -o --exclude-standard -- '*.py'; } | sort -u )
$ test -n "$FILES" || exit 1
$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811
All checks passed!
ruff exit=0
```

Derived sizes. **These are regenerated from live command output as the LAST
action before this artifact is frozen** -- the cycle-2 Q/A caught the numstat
block stale after a later edit, so they are no longer typed or copied forward.
(Corrected post-PASS, cycle-3 Q/A NOTE 1: this read "caught them stale",
plural. The Q/A raised exactly ONE such WARN -- the numstat. The plural
over-credited the evaluator for a second figure I found myself.)

```
$ git diff --numstat -- backend/slack_bot/jobs/_production_fns.py backend/tests/test_phase_82_12_string_column_guards.py
129	9	backend/slack_bot/jobs/_production_fns.py
81	11	backend/tests/test_phase_82_12_string_column_guards.py

$ wc -l backend/tests/test_phase_82_39_outcome_rebuild_query.py
     423 backend/tests/test_phase_82_39_outcome_rebuild_query.py
$ python3 -c "...ast walk for test_ functions..."
16
```

---

## 3. LIVE evidence (dry runs are free; Google: "you are not charged")

```
OLD (pre-fix):      DRY RUN 400 -> Unrecognized name: timestamp at [5:27]
NEW (rolling):      DRY RUN VALID, bytes=6737
NEW (fixed window): DRY RUN VALID, bytes=6737

live row counts:
  rolling 30d:              3 rows
  2026-06-01..2026-07-01:  20 rows
   sample: {'trade_id': 'c21925a7-...', 'ticker': '000660.KS', 'action': 'SELL',
            'created_at': '2026-06-05T19:02:20.831295+00:00', 'pnl': -9.9217}
```

The rolling window returning **3** is why criterion 2's fixture is pinned to a
FIXED window: the research gate measured that a rolling fixture returns 0 after
2026-08-26, so it would pass today and rot silently.

Schemas re-measured live:

| Table | Rows | Cols | Finding |
|---|---|---|---|
| `paper_trades` | 65 | 18 | `timestamp` absent, `realized_pnl` absent; real: `created_at` (STRING), `realized_pnl_pct` (FLOAT) |
| `outcome_tracking` | **0** | 9 | the WRITE half is still broken -- 82.48 |

Sweep, before and after:

```
before: tables_resolved=1  sql_literals=13  scope=0  unknown_columns=2
after:  tables_resolved=1  sql_literals=13  scope=1  unknown_columns=0
        SCOPE: created_at date backend/slack_bot/jobs/_production_fns.py
```

---

## 4. Mutation matrix

Script: `scratchpad/mutate_82_39.py`. Target asserted present before replace,
hash checked after write, always restored, restored tree re-verified.

| # | Mutant (production site) | Result |
|---|---|---|
| M1 | revert the SELECT to the phantom columns | KILLED |
| M2 | revert the predicate to `TIMESTAMP_TRUNC` on a phantom column | KILLED |
| M3 | drop the `IS NOT NULL` predicate | KILLED |
| M4 | delete the alert call from the failure branch | KILLED |
| M5 | downgrade the alert to P2 (logged and dropped) | KILLED |
| M6 | make the alert fire unconditionally, on success too | KILLED |
| M7 | make the SQL an f-string with an interpolated table | KILLED |
| M8 | let the windowed `replace` be a silent no-op | KILLED |
| M9 | hoist the emitter import to module scope | KILLED |
| M10 | close the `cost_budget_api` step (82.54) without fixing it | KILLED |
| M11 | close the sweep-recall step (82.55) without fixing it | KILLED |

**11 of 11 killed.** That licenses "these 11 died", not "no survivor exists".

### Two survivors I had to chase, and what each exposed

**M10 survived the first corrected run** because my guard matched any open step
whose *name* mentioned `cost_budget_api` -- and 82.55 mentions it as motivating
evidence. So closing the step that actually repairs the defect left the guard
green. I was matching on **mention** rather than **ownership**: another claim
about a set whose membership rule I had not written down. Fixed by
discriminating on the step's own criteria (only the owning step commits to
repairing the `input_tokens/output_tokens` projection); verified that 82.54's
criteria contain that string and 82.55's do not.

**M9 and the first M10/M11 were SKIP-BROKEN or wrong as constructed** -- M9's
target string did not exist (I guessed the import line), and M10/M11 mutated the
step `id` while the guard matches on `name` + `status`. A mutant that cannot be
applied, or that changes something the guard never reads, proves nothing; both
were corrected and re-run rather than counted.

### A near-miss I caused and then caught

**Extracting the seam almost blinded the sweep this step depends on.** My first
`build_ledger_fetch_sql` was an f-string interpolating the table name and the
WHERE predicate. `schema_oracle.extract_sql_literals` reassembles an f-string
from its CONSTANT parts only -- so interpolating the table dropped
`tables_resolved` from 1 to **0**, and interpolating the predicate erased the
`SAFE.TIMESTAMP(created_at)` date semantics, leaving `scope` empty. The fix for
the defect would have made the file invisible to the instrument that found it,
while every other guard stayed green.

The production query is therefore a fully plain literal, and
`test_the_production_sql_stays_visible_to_the_sweep` pins it structurally (the
AST node must be `ast.Constant`, not `ast.JoinedStr`). M7 confirms it dies.

---

## 4b. Cycle-2 corrections (Q/A CONDITIONAL -> fixed)

Cycle-1 verdict: CONDITIONAL. All four criteria MET and independently
reproduced by the Q/A; three WARN-level findings, all mine. Verbatim at
`handoff/current/evaluator_critique_82.39.md`.

**B1 -- I wrote the `A or B` escape hatch, in the guard that names the defect.**
`test_the_phantom_columns_are_gone_from_the_production_sql` contained:

```python
assert not re.search(r"\btimestamp\b", sql, re.I) or "SAFE.TIMESTAMP" in sql
```

The right disjunct is unconditionally True while the repaired query uses
`SAFE.TIMESTAMP`, so the line could never fail. The Q/A executed the
counterfactual: restoring the phantom `timestamp` column to the SELECT list left
it **passing**. Split into two independent assertions. Re-measured: that test now
appears in the kill list for the restore-`timestamp` mutant, which it did not
before.

**B2 -- my own docstrings defeated my own guard, and my claim about it was
measurably false.** Section 5 item 2 asserted "deleting the query cannot satisfy
it". The 82.12 fixed-branch scanned the RAW FILE TEXT for `created_at` /
`realized_pnl_pct` -- and this step's explanatory docstrings contain both.
Reproduced exactly: deleting the entire query body leaves `created_at` on **7**
lines and `realized_pnl_pct` on **1**, all prose, and the guard passed on a
source with no query at all. Now scans the **parsed SQL literals** via
`extract_sql_literals`. Re-measured under the same deletion:

```
prose survives the deletion: created_at x7, realized_pnl_pct x1
82.12 fixed-branch guard under the deletion: FAILED (good)
```

The sentence in section 5 is corrected rather than left standing.

**B3 -- a test wrote a probe into a shipped package.** The recall probe was
written to `backend/db/_recall_probe_82_12.py` and unlinked in `finally`.
Cleanup works, but a hard kill bypassing `finally` would leave a file selecting
phantom columns inside `backend/db/`, where the sweep would flag it forever as a
self-inflicted defect and an `add -A` hook would commit it. It cannot go in
`tmp_path` -- `derive_scope` does `path.relative_to(_REPO_ROOT)` and raises
outside the repo -- so it now goes to `handoff/logs/`, which is gitignored
(`.gitignore:76`) and is not importable code. Verified: no residue in either
location after a full run, `git status backend/db/` clean.

None of the three touched a criterion or production behaviour. Post-fix:
`46 passed`, ruff exit 0 over a git-derived non-empty scope.

## 4c. Cycle-3 correction (Q/A CONDITIONAL #2 -> fixed)

**One WARN, and it is the same class as B2 one level up.** The cycle-2 Q/A
re-ran the command this artifact *labels as its own source*:

```
$ git diff --numstat -- backend/tests/test_phase_82_12_string_column_guards.py
```

The artifact carried `60  11`; the command returned `81  11`. My cycle-2 B2/B3
edits added ~21 lines to that file and I did not re-derive the block. So a
command-labelled capture in the artifact that gets ARCHIVED as the durable
record understated the change by 26%.

It was true when taken and the cycle-2 changes are disclosed qualitatively in
§4b, so it is staleness rather than fabrication -- but "true when I wrote it" is
exactly the excuse this harness exists to refuse, and it is my third
underived-figure finding today.

**The fix is procedural, not a one-off edit.** Every number in §2 and §7 is now
regenerated by running the commands as the LAST action before the artifact is
frozen, so a later edit cannot leave them behind. Retitled to "cycles 1-3",
since the doc was still headed "(cycle 1)" while carrying two later sections.

**Also recorded, from the cycle-2 notes, because it is a real residual and not a
finding I get to drop:** the rewritten 82.12 fixed-branch still has a narrow
hatch -- a SQL literal whose only occurrences of `created_at` /
`realized_pnl_pct` sit inside SQL *comments* would satisfy it. Far weaker than
the docstring-prose hatch it replaced, and criteria 1-2 (a live dry run of the
production builder string, plus a live fixed-window count of exactly 20 rows)
are the primary behavioural coverage. Stated rather than quietly inherited.

**And a provenance note worth carrying forward:** the Q/A observed that
`_production_fns.py`'s mtime is later than the cycle-1 verdict, so *mtime alone*
would not prove "no production change in cycle 2". The claim holds on stronger
evidence (numstat identical to the cycle-1 record, all four criteria
re-reproducing) -- but a future cycle wanting a hard guarantee should record a
content hash, not a timestamp.

## 5. Two phase-82.12 guards I deliberately rewrote

Closing this step turns two currently-green tests RED. Measured, not predicted.
Both were rewritten preserving intent; neither was deleted.

1. `test_query_selecting_nonexistent_columns_is_detected` asserted the live
   defect **is** flagged. Repairing the query removes the flag, so the guard
   would have punished the repair it existed to demand. Its intent -- *"the
   checker can SEE this defect class"* -- is now driven on a **synthetic**
   instance (a temporary probe file), which is strictly stronger: it survives
   the live defect being fixed and still fails if the checker stops detecting.
2. `test_the_nonexistent_column_defect_is_queued_as_its_own_step` required an
   **OPEN** step naming all four signature tokens. Measured over all 1115 steps,
   82.39 was the only match (82.48 carries 2 of 4), so flipping 82.39 to done
   breaks it. Rewritten to the disjunction 82.39's own criterion 4 uses:
   **fixed OR queued**. Those are disjoint states -- one a property of the
   source, one of the masterplan -- so this is not the `A or B` escape hatch
   where A is a subset of B. The fixed branch additionally asserts the real
   columns are present in the **parsed SQL literal** (via
   `extract_sql_literals`), not in the raw file text -- see the retraction in
   section 4b, where a raw-text version of this same sentence was measured
   FALSE.

---

## 6. Corrections to the step's own text

**The step's consequence claim is FALSE and is corrected rather than repeated.**
It says outcome tracking "feeds agent memories (BM25) and the learning loop, so
a long-dead rebuild may mean the reflection corpus has been frozen". Measured:
`outcome_tracker.py` has **0 references** to `outcome_tracking`; the real writer
is `autonomous_loop.py`, gated by `paper_learn_loop_enabled = False`. The defect
is real -- the job produces nothing -- but the blast radius is smaller than
stated.

**Duration is a LOWER BOUND, not a count.** Dead since the file's first commit
(`2301b977`, 2026-05-11) = 87 days. But `IdempotencyStore` is an in-memory
`set()` and `heartbeat` sinks to `logger.info`, so there is **no durable
receipt**. "87 nightly runs failed" is NOT claimable and is not claimed.

**A pre-existing lint finding cleared, disclosed:** `from pathlib import Path`
was unused in `_production_fns.py`. Verified pre-existing by running ruff
against `git show HEAD:` -- I did not introduce it. Removed because the required
lint gate covers every file this step touches, and a red gate would block the
close for a reason unrelated to this work.

---

## 7. Files changed

| File | Change |
|---|---|
| `backend/slack_bot/jobs/_production_fns.py` | `+129 / -9` -- seam, repaired SQL, P1 alert on the swallow |
| `backend/tests/test_phase_82_39_outcome_rebuild_query.py` | NEW, 423 lines, 16 tests |
| `backend/tests/test_phase_82_12_string_column_guards.py` | `+81 / -11` -- two guards rewritten (see §5) |
| `.claude/masterplan.json` | queues 82.54 + 82.55; 82.39's own entry untouched |
| `handoff/current/*_82.39.md` | contract, brief, results |

## 8. Queued out of scope

- **82.54 (P1)** -- a SECOND live phantom-column defect of the identical class:
  `cost_budget_api.py` selects `input_tokens` / `output_tokens` from
  `llm_call_log`, whose real columns are `input_tok` / `output_tok` (measured
  live: 5519 rows, 15 columns, `input_tokens` present=False). Same fail-open
  swallow, so the token tile has been permanently null. **`derive_scope` cannot
  see it** -- which is why criterion 4's clean report is not a clean repo.
- **82.55 (P2)** -- the sweep's recall: `tables_resolved=1` of 33 oracle tables,
  because ~20 backend files build table refs with f-strings and `scripts/` is
  never scanned. 82.54 is the proof this matters.

## 9. What this step does NOT do -- read before closing

**The WRITE half is still broken. The job will STILL write 0 rows.**
`make_outcome_write_fn` emits 5 keys of which only `ticker` exists on
`outcome_tracking`, and both REQUIRED columns are unsupplied; per Google's
streaming documentation a schema mismatch means **none** of the rows insert, and
`insert_rows_json` **returns** errors rather than raising. `_compute_outcomes`
also crashes on a NULL `pnl`. Both are **82.48**. This step keeps the
`realized_pnl_pct IS NOT NULL` predicate, which happens to keep NULLs out of
`_compute_outcomes`, but that is a side effect and not a fix.

No live positions touched; paper trading untouched.
