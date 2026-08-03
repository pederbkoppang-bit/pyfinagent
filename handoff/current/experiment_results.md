# experiment_results -- step 82.0

**GENERATE phase.** Contract: `handoff/current/contract.md`.
Research: `handoff/current/research_brief_82.0.md` (gate_passed=true, 8 sources
read in full, 22 URLs, 19 internal files).

NOTE: this rolling file previously held phase-80.2 content -- it was never
archived when that step closed. Flagging rather than silently overwriting;
this is the same rolling-file/archive drift class as phase-81.

## What was built

The freeze was NOT a broken job. Root cause (research-verified, then
independently re-verified by Main): `ingest_macro` had **no scheduled caller at
any point in the repo's history**, and `settings.backtest_end_date`
("2025-12-31") was threaded into the FRED `observation_end` parameter, so the
handful of manual runs asked for data ending on the backtest cap and inserted
zero rows while reporting success.

| # | Change | File |
|---|---|---|
| 1 | `macro_ingest_end_date` setting; macro end date severed from `backtest_end_date` | `backend/config/settings.py:245-251` |
| 2 | `_resolve_macro_end_date()`; `ingest_macro(end_date=None)` defaults to today; caller stops forwarding the backtest cap | `backend/backtest/data_ingestion.py` |
| 3 | `_get_existing_macro` now fails **CLOSED** (was bare `except -> set()`) | `backend/backtest/data_ingestion.py` |
| 4 | Per-series freshness SLA replaces the global-max gate, with date coercion so it is not vacuous on the STRING column, failing closed on unparseable dates | `backend/backtest/cache.py:40-51` + `preload_macro` |
| 5 | Run-receipt JSONL on every attempt, incl. pre-ingest failures | `backend/backtest/data_ingestion.py`, `backend/backtest/macro_cron.py` |
| 6 | `realtime_start` vintage column + versioned idempotent migration | `scripts/migrations/add_macro_realtime_start.py` |
| 7 | The scheduled caller that never existed, wired into app startup | `backend/backtest/macro_cron.py` (new), `backend/main.py` |

`self.settings` was also retained on `DataIngestionService.__init__` -- the
service previously unpacked only project/dataset, so no setting could be read
at call time.

## Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_0_macro_ingestion.py -q
................                                                         [100%]
16 passed in 0.82s
```

## CYCLE 2 -- Q/A returned FAIL on cycle 1; what changed

The cycle-1 Q/A verdict was **FAIL** on criterion 4, and it was correct. The
blocking finding, independently re-verified by Main before acting:

`historical_macro.date` is a **STRING** column
(`('date','STRING','REQUIRED')`, declared at
`scripts/migrations/migrate_backtest_data.py:68`; live rows return
`type(date) = str`, e.g. `'2023-07-03'`). Both the pre-existing phase-25.D7
gate and the cycle-1 per-series rewrite tested `isinstance(rd, datetime.date)`,
which is FALSE for every production row. So `per_series_max` was empty on
every real query, `stale` could never be non-empty, and **preload_macro never
refused anything**.

Three consequences, all corrected here:

1. **The delivered guard was vacuous.** Fixed: dates are now coerced
   (`_coerce_date` handles str / date / datetime), and an unparseable date
   column now fails CLOSED instead of silently disabling the gate.
2. **The cycle-1 fixture could not represent the production failure.** It
   passed `datetime.date`, a type the query never returns, so the test was
   green for every possible production state including a fully dead table --
   and it was the SOLE coverage of criterion 4. Fixed: `_macro_row` now emits
   the production STRING shape, plus two new regression pins
   (`test_gate_is_not_vacuous_on_the_production_date_type`,
   `test_unparseable_dates_fail_closed`).
3. **The mutation matrix was mis-scoped.** Mutating the code only proved
   discrimination inside the fixture's own type space. A mutation test
   inherits its fixture's blind spots; mutating the code does not test the
   fixture.

**FALSE CLAIM WITHDRAWN.** Cycle 1 asserted preload_macro "returned 0 before
this step". It did not -- it returned **4412**, because the pre-fix gate was
vacuous for the same isinstance reason. That number was inferred from reading
the threshold logic and reported as measured. The honest delta is
**4412 -> 4729 (+317 rows)**. The corresponding causal story was also removed
from two production comments (`backend/main.py`, `backend/backtest/macro_cron.py`)
where it had been shipped as fact.

**The real defect is worse than the one originally described.** Nothing was
hanging and nothing was being refused: backtests were being silently trained
on 212-day-old macro features, and had been for as long as the guard existed.

Lint gate (also red in cycle 1, both findings introduced by this diff) is now
clean over the git-derived 7-file scope:

```
$ FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u )
$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811
All checks passed!
```

`test_receipt_written_on_zero_row_run` was additionally rewritten to assert on
the real receipts file rather than a monkeypatched stub (Main's own flagged
concern; the Q/A had accepted the weaker version).

## Live evidence

Full artifact: `handoff/current/live_check_82.0.md`. Headline: the table
advanced from 4412 rows / `MAX(date)=2025-12-31` to **4729** rows with every
series inside its per-series SLA; the live FRED request carried
`observation_end=2026-08-03` and inserted 317 rows. Gate non-vacuity is now
demonstrated in BOTH directions against the production STRING type: current
data caches 4729 across 7 series; a stale-GDP fixture in production shape
returns 0 with `GDP(newest=2018-05-17 age=3000d limit=225d)`.


## CYCLE 3 -- disposition of the cycle-2 CONDITIONAL

Cycle-2 verdict: **CONDITIONAL**. All 6 immutable criteria assessed MET with
15/15 injected mutants killed; six findings capped it below PASS. Verbatim
verdicts for both cycles are preserved in
`handoff/current/evaluator_critique_82.0.md` (+ the raw returns in
`evaluator_critique_82.0_cycle1.json` / `_cycle2.json`).

**F1 -- retracted claim surviving in a forward-looking artifact [was BLOCK-for-PASS]. FIXED.**
Pending step 82.3's description still instructed a future executor
"preload_macro() returns 0 today ... Do NOT attempt this step while
preload_macro() returns 0" -- the exact claim cycle 2 withdrew. Corrected: the
precondition is now stated as DATA VALIDITY (non-zero row count AND every
series inside its SLA), the old text is quoted and explicitly withdrawn, and
the newly-reachable refusal path is called out. 

**CYCLE 4 CORRECTION -- this disposition was itself wrong, twice.** The
cycle-3 Q/A caught both errors and they are the SAME failure class as the
cycle-1 "returned 0" slip: a past-tense claim written without running the
check that would prove it.

1. **"The research brief ... is annotated" was FALSE when written.** It was
   not annotated -- `research_brief_82.0.md:281` still carried the withdrawn
   claim as **CONFIRMED**, with an mtime unchanged since the PLAN phase. That
   brief is NOT inert: pending steps **82.8, 82.9 and 82.10** are each named
   "DEFECT from the 82.0 research brief" (MEASURED against masterplan.json:
   3 of 3 -- 82.12 is a DIFFERENT lineage, named "DEFECT CLASS sweep, surfaced
   by the 82.0 cycle-1 Q/A FAIL", and an earlier draft of this sentence wrongly
   included it) and point their executors at the very table containing the
   false row. NOW ACTUALLY DONE: the row is struck through
   and marked WITHDRAWN, with an annotation block stating the measured truth
   and confirming every other row in that table still stands.
2. **"Two occurrences remain by design" did not reproduce.** A repo-wide
   census returns far more than 2. Worse, the count is SELF-MUTATING -- writing
   the census changes it. See the stable-invariant framing below.

### Census of the retracted claim

**A raw total here is a self-mutating claim** -- writing the census adds
occurrences of the phrase, so any number is stale the moment it is committed.
I measured 11, then 12 one command later, purely because this section exists.
That is the third instance in this step of the same failure class, so the
census is stated as a STABLE INVARIANT instead of a count:

> **Live carriers asserting the claim AS FACT: ZERO.**

Every remaining occurrence falls into one of four categories, none of which
asserts it:

| Category | Where | Why it is not a live carrier |
|---|---|---|
| IMMUTABLE criteria text | `masterplan.json` (82.0 `live_check`), `contract.md` (its verbatim copy) | Cannot be edited by protocol; overturned in `live_check_82.0.md` and annotated in `contract.md` |
| The retraction itself | `masterplan.json` (82.3 description), `experiment_results.md` | Quotes the claim in order to withdraw it |
| Verbatim evaluator record | `evaluator_critique_82.0.md`, `evaluator_critique_82.0_cycle2.json` | Must stay byte-exact -- it is the Q/A's own words |
| Annotated dated record | `research_brief_82.0.md:281` | Struck through, marked WITHDRAWN, annotation block states the measured truth |

Reproduce with:

```
grep -rnE "returns 0 today|preload_macro\` returns 0|preload_macro\(\) returns 0|returns 0 at .cache\.py" \
  .claude handoff/current backend scripts docs CLAUDE.md | grep -v .venv
```

and check each hit against the table above. The invariant to verify is that no
hit asserts the claim as a present fact -- not that the total equals any
particular number.

**F2 -- test suite forging records into the operational ledger [WARN]. FIXED.**
The receipts directory is now an injectable override
(`DataIngestionService._receipts_dir_override`), an autouse fixture redirects
every service built in this module to `tmp_path`, and a new regression pin
(`test_tests_do_not_write_to_the_operational_receipts_ledger`) fails if a test
ever writes to `handoff/logs/` again. MEASURED: two consecutive full-suite runs
now move the real ledger by **0 lines** (was 13 -> 37 during one evaluation).
`test_receipt_is_valid_jsonl`'s unused `tmp_path` and no-op `monkeypatch` --
both correctly identified as vestigial -- are gone.
DISCLOSED, NOT REWRITTEN: the ledger holds 37 records, of which exactly ONE
(`rows_inserted=317`) is a genuine ingest and **36 are test residue** written
before this fix. MEASURED partition (an earlier version of this sentence
mis-partitioned its own set -- it said "28 ok + 8 skipped", which both counted
the genuine ingest as residue and omitted the partial-failure record):

| bucket | n |
|---|---|
| genuine ingest (`rows_inserted=317`, outcome `ok`) | 1 |
| residue, outcome `ok` | 27 |
| residue, outcome `skipped_no_api_key` | 8 |
| residue, outcome `partial_failed=FEDFUNDS,CPIAUCSL` | 1 |
| **total** | **37** | Rewriting an append-only
audit log to make it look clean is a worse habit than disclosing it, so they
stay. The file is gitignored and rolls forward.

**F3 -- newly-reachable per-cutoff BQ fallback [WARN]. FIXED + DISCLOSED.**
This is a real defect introduced by this step, not a documentation gap. Arming
a previously-vacuous gate makes `preload_macro` able to return 0 for the first
time ever; `backtest_engine.py:308` DISCARDS that return value, so a refusal
surfaces only as a WARNING and `cached_macro` falls through to the per-cutoff
BQ query -- CLAUDE.md's ~40-minute path. Measured headroom at the 5-day daily
SLA was ONE day (DGS10 newest 2026-07-30 = 4d), against a cron that has never
been observed firing. FRED daily series skip weekends and US market holidays,
so a long weekend plus one missed run would have tripped the gate on a
perfectly healthy feed. Daily bounds widened 5 -> 12 days: still catches a
genuinely dead feed by orders of magnitude (the observed failure was 212 days)
while tolerating holidays and several missed runs. The residual risk --
`backtest_engine.py:308` ignoring the return value -- is NOT fixed here (it is
outside this step's surface) and is queued as **82.13**.

**F4 -- mis-attributed kill mechanism [WARN]. FIXED.**
`test_gate_is_not_vacuous_on_the_production_date_type` asserted only
`preload_macro() == 0`, which the fail-closed branch added in the same diff
also satisfies -- so a re-introduced isinstance bug would have been "killed"
by the wrong mechanism and passed for the wrong reason. It now pins the
DISCRIMINATING behaviour: the refusal must come from the per-series SLA
evaluation and name the stale series (`past their per-series SLA`,
`GDP(newest=`), which a vacuous gate cannot produce.

**F5 -- cycle-1 FAIL never persisted [WARN]. FIXED.**
A genuine protocol breach by Main: `evaluator_critique.md` still held
phase-80.2 content and no `evaluator_critique_82.0.*` existed, so the Q/A had
to reconstruct the prior verdict from the author's own summary of it -- which
defeats the point of an independent record. Both verdicts are now transcribed
verbatim to `evaluator_critique_82.0.md` with the raw structured returns
alongside.

**F6 -- commit scope not disclosed [WARN]. FIXED (disclosure below).**

## Commit scope -- what `git add -A` would actually ship

**Stated as a classification, not a count.** The total moves every time this
step writes anything -- including the status flip itself -- so a frozen number
here is stale on commit (same self-mutating-claim trap as the phrase census
above). Reproduce with `git add -An`:

| class | belongs to this step? | note |
|---|---|---|
| `backend/{config/settings,main,backtest/{cache,data_ingestion,macro_cron}}.py`, `backend/tests/test_phase_82_0_*`, `scripts/migrations/add_macro_realtime_start.py`, `scripts/harness/build_evaluator_critique.py` | YES | the change surface |
| `handoff/current/*`, `handoff/harness_log.md` | YES | this step's artifacts |
| `.claude/masterplan.json` | PARTLY | phase-82 is 14 steps, but 8 of them are defects DISCOVERED during this work, not part of 82.0's own change |
| `.claude/agent-memory/{qa,researcher}/*` | NO | written by the subagents during their own runs |
| `handoff/archive/phase-81.2/*` | NO | the PRIOR step's snapshots, untracked at its close |
| `handoff/autoresearch/*-ERROR-*.md` | NO | the failing nightly job -- the subject of queued step 82.11 |
| `handoff/{audit,away_ops,*.jsonl}`, `.claude/.archive-baseline.json` | NO | runtime state mutated by hooks on every tool call |

Everything in the NO rows rides along under this step's commit subject. That is
the standing behaviour of `git add -A` in the auto-commit hook, not something
this step introduced -- but it is disclosed rather than left implicit.

## Defects discovered during GENERATE (each needs its own step)

1. **FRED API key logged in plaintext (SECURITY).** `httpx` logs the full
   request URL at INFO and `data_ingestion.py:313` puts `api_key=` in the query
   string. `LOG_LEVEL` defaults to INFO, so the key has been written to backend
   logs on every ingest, and it appeared in this session's console output.
   **The key should be rotated.** Fix: POST body, or suppress the `httpx`
   logger around this call. NOT fixed here -- needs its own research gate.
2. **`run_macro_ingest` first-run bug, self-inflicted and caught live.**
   `BigQueryClient()` was called without its required `settings` arg; the
   top-level fail-open swallowed it and returned a clean `0`. That is precisely
   the invisible-failure mode this step exists to kill, so the fail-open path
   now writes a failure receipt. Fixed within this step.
3. From the research brief, out of scope: `sortino.py:108` queries
   `pyfinagent_data.historical_macro` (wrong dataset -- the table is in
   `financial_reports`) for `DGS3MO`/`DTB3` (not in `FRED_SERIES`), so its
   tier-1 lookup has always been dead; `data_server.py:185` serves stale rows
   stamped `as_of: today`; `compute_freshness` is reachable only from HTTP
   handlers the frontend calls, so nothing pages when a feed dies.

## Scope honesty

- **No live trading behaviour changed.** `macro_regime.py:23` reads FRED
  directly via `backend/tools/fred_data.py`, not this table, so the live buy
  funnel is untouched. This step alters backtest inputs and adds a scheduler
  registration.
- `backtest_end_date` itself was NOT modified -- backtests still read it. Only
  the macro feed was severed from it.
- The cron registration is verified against a stub scheduler AND by a source
  scan of `main.py`. The job has NOT been observed firing on its trigger; that
  needs a backend restart plus waiting for 08:10 ET.
- The conservative vintage backfill (`DATE(ingested_at)`) is not the true
  publication vintage for pre-existing rows. It cannot be earlier than the
  truth, so it cannot manufacture look-ahead, but it makes point-in-time
  backtests over the historical span slightly pessimistic.


## CYCLE 5 -- disposition of the cycle-4 FAIL

Cycle-4 verdict: **FAIL**, correctly applying the 3rd-consecutive-CONDITIONAL
rule. Cycle 4 stated explicitly that **all 6 immutable criteria are MET**, each
backed by a guard it proved can fail (C1->3 failed, C2->1, C3->2, C4->2 and ->2
under two independent mutations, C5->2, C6 source-level removal of the
`"realtime_start": vintage,` line -> the vintage test fails), and that the macro
repair **must not be reverted**. `retry_count` is now 2/3 (two FAIL verdicts:
cycles 1 and 4); `certified_fallback` remains false.

The four close items it named, each verified in the same turn it was claimed:

**(A) Cycle-3 verdict never transcribed -- a RECURRENCE of cycle-2 F5, which I
had declared FIXED. FIXED, structurally.** Cycles 3 and 4 are now persisted as
`_cycle3.json` / `_cycle4.json`, and `evaluator_critique_82.0.md` is no longer
hand-maintained -- it is **GENERATED from the persisted returns**, so it cannot
silently lag them again. Proof run: every persisted return has a matching
section carrying the same verdict. This was the worst finding of the four,
because declaring F5 fixed while leaving cycle 3 untranscribed forced the next
evaluator to take a prior verdict from my own summary of it -- destroying the
independence the transcription rule exists to protect.

**(B) My cycle-4 annotation split the table it was repairing. FIXED.** The
blockquote sat between table rows, so the final row -- `LIVE analysis pipeline
| REFUTED -- not degraded`, the row that establishes this is NOT a live-money
defect -- was swallowed as literal pipe-text. Annotation moved below the table
and the orphaning blank line removed. Measured: 6 data rows now render inside
the table, including the LIVE row. A correction that damages the artifact it
corrects is worse than the error.

**(C) "including the two" enumerated THREE. FIXED** -> now names the three
explicitly with their step ids (82.8 sortino, 82.9 data_server, 82.10
cycle_health), and additionally points at the LIVE row so an executor sees the
urgency bound.

**(D) "82.8/82.9/82.10/82.12 each named 'DEFECT from the 82.0 research brief'"
-- MEASURED FALSE. FIXED.** Verified against masterplan.json before rewriting:
3 of 4. 82.12 is a different lineage ("DEFECT CLASS sweep, surfaced by the 82.0
cycle-1 Q/A FAIL"). Claim now says 82.8/82.9/82.10 and states the measurement.

### The pattern, stated plainly

This step has produced **five** false or unreproduced claims across four
evaluation cycles -- "returned 0" (c1), "is annotated" + "two occurrences"
(c3), "including the two" + "each named" (c4) -- and C and D were written
*inside the section created to end that class*. The product code has been
correct and converging since cycle 2; the claims about it took four more
passes. Every instance was prose, none was code, and every one was caught by
an independent evaluator rather than by me.

The durable fixes are structural rather than resolutions to be more careful:
the critique file is now generated rather than remembered, and counts over
artifacts under edit are stated as reproducible invariants rather than numbers.
Recorded in auto-memory `feedback_verify_own_completed_action_claims` (with the
new self-mutating-count sub-lesson) and
`reference_vacuous_type_guards_on_bq_string_columns`.
