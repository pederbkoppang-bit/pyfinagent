# experiment_results -- step 86.69 (GENERATE, 2026-08-17)

Contract: `contract_86.69.md`. Research gate: PASSED (`wf_8515024d-2fe`; 9
sources in full, 31 URLs; brief 40,219 chars). Verbatim command evidence:
`live_check_86.69.md`.

## The shape of this GENERATE, stated up front

The product fix was ALREADY WRITTEN (phase-61.2, tested, 33/33 green) and
DARK. This step's GENERATE is therefore: (a) the cause and the fabrication
sites verified from source at HEAD; (b) the numbered ASK put to the
attending operator; (c) the operator's approval EXECUTED as an
operator-token action (never step authorship -- criterion 7 intact); (d)
the in-force chain proven; (e) the measurement queries staged against the
frozen baselines, accruing from tonight's cycle.

## Criterion 1 -- the CAUSE, demonstrated

Hypothesis (i) CONFIRMED; (ii)/(iii) REFUTED (brief A1-A5, re-verified):
phase-60.1 (`fa62b5fe`, 2026-06-11) restored the FULL pipeline after
gemini-2.0-flash's 2026-06-01 server-side retirement had silently pushed
every analysis onto the healthy lite path. The `_path` provenance field is
the deploy marker: last unstamped row 2026-06-10 18:38Z, first stamped
2026-06-11 10:17Z, zero-scores start that same day -- the break date is
2026-06-11, one to four days earlier than the audit-basis window, REPORTED
as a correction not adopted silently. All 211 empties are `_path=full`;
lite produced 0/19. The full path rides the CC rail (`$.rail=claude_code`:
257 rows, 84.4% zero-score) whose `--json-schema` is post-hoc validated,
not constrained -- the Gemini-shaped `_SYNTHESIS_STRUCTURED_CONFIG` is
honoured by nothing, and `final_synthesis.error='Failed to parse final
report.'` appears on 211/211 empties vs 0/38 healthy.

## Criterion 2 -- the fabrication sites, READ and quoted

`backend/services/autonomous_loop.py`:
- `:2172` `rec = synthesis.get("recommendation", {})` -- on the error dict
  this is `{}`;
- `:2179` `"recommendation": rec.get("action", "HOLD") ...` -- `{}` yields
  the fabricated uppercase `HOLD`;
- `:2191-2192` `"final_score": synthesis.get("final_weighted_score",
  synthesis.get("final_score", 0))` -- the error dict yields `0`.
The log-co-occurrence inference in the q1 diagnosis is REPLACED by this
source evidence.

> **CORRECTION (2026-08-17, measured).** This paragraph originally ended "The
> lite writer is exonerated (title-case `Hold`)", using letter-case as the
> discriminator between writers. **That does not hold.** All six post-arm rows
> are `_path=full` and carry title-case `Hold`, because the MODEL returned
> `"Hold"` as its action string (`$.final_synthesis.recommendation.action` =
> `Hold`, `error` NULL). The full path produces title-case too. The reliable
> discriminators are the stamped `_path` field (phase-60.1) and
> `final_score = 0.0` co-occurring with a non-null `final_synthesis.error` --
> not the case of the recommendation. The `:2179` fabrication site still yields
> UPPERCASE `HOLD` as described; what fails is the converse inference.

## Criterion 3 -- absence as absence, and the consumer set

The phase-61.2 machinery (verified from source + its 33 tests):
- Flag ON, parse failure: `:2163-2171` raises `SynthesisDegradedError`
  BEFORE assembly -- the fabricated row is never built; the except routes
  to the LITE fallback, which produces a REAL scored row (measured 36.8%
  BUY conversion -- this is also the recovery lever for criterion 5).
- Flag ON, both paths fail: `:2252-2267` returns the honest degraded dict
  -- `recommendation=None, final_score=None, $._degraded + reason` --
  persisted as a NULL/NULL row then converted to None so it NEVER enters
  `decide_trades` (consumer proof in the 61.2 tests; NULL is already
  plumbed to the frontend; BQ `UPPER(recommendation)` on NULL yields NULL,
  never 'HOLD'; `_DOWNGRADE_RECS` hazard is covered by the documented
  unsafe-combination guard on the sibling flag).
- Flag OFF = legacy fabrication, deliberately pinned by
  `test_flag_off_legacy_fabrication_unchanged` -- the dark-flag doctrine,
  which is why this step COULD NOT simply edit the default path (the
  green suite encodes that policy; changing it belongs to the flag, not
  to a silent default flip).

## Criterion 7 -- the ASK, and how the approval was executed

ASK-1 (arm the flag) + ASK-3 (restart now vs session end) were put to the
ATTENDING operator via AskUserQuestion. Operator: **"Yes -- arm it"** and
**"Now"**. Execution followed the away-ops rail-1 token protocol (the
danger-hook blocked the write until the token was applied -- the gate
worked): token `ARM-SYNTHESIS-INTEGRITY-86.69` recorded in
`handoff/away_ops/pending_tokens.json` (same in-session channel as the
2026-07-07 precedent recorded in that file), `tokens_cursor` touched,
THEN `PAPER_SYNTHESIS_INTEGRITY_ENABLED=true` appended to `backend/.env`.
The STEP promoted nothing; the operator did, through the machine gate
built for exactly this. ASK-2 (the 2,859-failure CC-rail structured-output
surface) is queued as its own masterplan step.

## The in-force chain (committed-is-not-in-force, discharged)

`backend/.env` mtime 15:06:04 -> `launchctl kickstart -k` -> new pid 14280
with lstart 15:06:17 (old pid 47562 gone; PID CHANGED asserted) -> health
endpoint OK -> the same pydantic loader on the same file yields `True`.

> **TWO CORRECTIONS to the paragraph above, measured 2026-08-17T18:2xZ.
> These REPLACE the corresponding readings; they do not sit beside them.
> Full evidence in `live_check_86.69.md` §5.**
>
> 1. **`15:06:04` and `15:06:17` are LOCAL (CEST=UTC+2), not UTC** --
>    `stat -f %Sm` and `ps -o lstart` both print local time. The true
>    instants are **`13:06:04Z`** (env write) and **`13:06:17Z`** (pid
>    14280 start): `date -u -r $(stat -f %m backend/.env)` ->
>    `2026-08-17T13:06:04Z`. Any post-arm query keyed on "15:06Z" would
>    silently drop a two-hour window; today that window is provably empty
>    (zero `analysis_results` rows exist for 2026-08-15/16/17), so no
>    evidence was lost, but the cutoff used below is `13:06:17Z`.
> 2. **The flag is no longer held by pid 14280.** A later restart replaced
>    it with **pid 41635, started `13:57:16Z`** -- still AFTER the
>    `13:06:04Z` env write, so the in-force chain survives, re-verified
>    against the RUNNING process (`Settings().paper_synthesis_integrity_enabled`
>    -> `True`; `backend/.env:88` carries the line). A verifier checking
>    pid 14280 will find it gone; that is a restart, not a regression.
Direct memory read is impossible (`GET /api/settings/` does not expose the
flag -- the SAME blind spot the q1 doc measured for the diversity flags;
queued with ASK-2's step). The chain write-BEFORE-boot + loader-reads-file
is the in-force evidence, stated as a chain rather than claimed as a read.

## Criteria 4+5 -- MEASURED, and the measurement does NOT discriminate

The first post-arm cycle ran `cycle-1786989600`, 2026-08-17T18:00:00Z, held by
pid 41635 (the process proven to load the flag). It produced **6 analyses**.
Everything below is from `scratchpad/measure_86_69.py`, which prints every
query beside its result.

### C4 -- zero-score share

```sql
SELECT CASE WHEN analysis_date >= TIMESTAMP('2026-08-17 13:06:17 UTC') THEN 'POST_ARM'
            WHEN DATE(analysis_date) <= '2026-06-10' THEN 'PRE' ELSE 'POST' END AS regime,
       COUNT(*) n, COUNTIF(final_score = 0.0) zero_score,
       ROUND(100*COUNTIF(final_score = 0.0)/COUNT(*),1) zero_pct
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE DATE(analysis_date) BETWEEN '2026-05-01' AND CURRENT_DATE() GROUP BY regime
```

| regime | n | zero_score | zero_pct |
|---|---|---|---|
| PRE | 238 | 87 | **36.6%** |
| POST | 281 | 219 | **77.9%** |
| **POST_ARM** | **6** | **0** | **0.0%** |

### C5 -- BUY conversion among rows that produced a real score

| regime | scored rows | buys | buy_pct |
|---|---|---|---|
| PRE | 151 | 87 | **57.6%** |
| POST | 62 | 11 | **17.7%** |
| **POST_ARM** | **6** | **0** | **0.0%** |

### THE FROZEN BASELINES DO NOT REPRODUCE -- reported, not reconciled

`live_check_86.69.md` §4 froze `PRE 95/251 = 37.8%; POST 211/260 = 81.2%`.
Re-running that same query today gives `PRE 87/238 = 36.6%; POST 219/281 =
77.9%`. POST growing is expected. **PRE SHRINKING (251 -> 238) is not**, because
2026-05-01..2026-06-10 is a closed historical window. Table reconciles
(54 + 238 + 286 = 578). **No cause is asserted.** The direction is unaffected,
but a baseline that cannot be regenerated is not a baseline, so both readings
are given with their queries.

### *** THE HONEST CONCLUSION: THIS CYCLE PROVES NOTHING ABOUT THE FIX ***

Two independent reasons, both measured:

**1. The guard never ran.** Zero `Failed to parse final report` and zero
`falling back to lite` this cycle; `final_synthesis.error` is NULL on all six
rows and every row is `_path=full`. `paper_synthesis_integrity_enabled` guards
the *parse-failure* branch, so with no parse failures **the armed flag was
never entered**. (There WERE 6 `returned invalid JSON` events this cycle -- but
those are AGENT-level, the 86.108 class, which degrade debate inputs; the final
synthesis parsed 6/6.)

**2. The pre-arm days were already clean, so 0/6 is not a signal.** The
zero-score rate per write-day is extremely volatile, and it had already
recovered before the flag was armed:

| day | n | zero | % | buys |
|---|---|---|---|---|
| 2026-08-08 | 6 | 6 | 100 | 0 |
| 2026-08-09 | 12 | 7 | 58 | 0 |
| **2026-08-10** | 6 | **0** | **0** | 2 |
| 2026-08-11 | 6 | 1 | 17 | 1 |
| 2026-08-12 | 6 | 6 | 100 | 0 |
| 2026-08-13 | 6 | 1 | 17 | 1 |
| **2026-08-14** | 6 | **0** | **0** | 0 |
| **2026-08-17 (POST-ARM)** | 6 | **0** | **0** | **0** |

**2026-08-10 and 2026-08-14 were both 0/6 with the flag OFF.** Tonight's 0/6
sits inside the pre-arm distribution, so it is not distinguishable from it.
And on the C5 half tonight is *worse* than two pre-arm days: 0 buys against 2
on 08-10 and 1 each on 08-11 and 08-13.

**Consequence for the criteria, stated plainly rather than argued around:**
criterion 4's number is delivered (0.0% against the two baselines, with the
queries), but the comparison it is set against -- a POST *average* spanning
2026-06-11..2026-08-17 -- averages across a regime that had already partly
recovered. Criterion 5's decomposition likewise cannot be evidenced from n=6
where the pre-arm comparator already contains 0/6 days. **One cycle is not a
measurement of this fix.** The step needs either several more post-arm cycles
or a cycle in which the full path actually fails; the evaluator should judge
criteria 4 and 5 on that basis, and Main is not claiming them as met.

## Criteria 4+5 -- original staging note (superseded by the section above)

Baselines frozen with their queries (live_check §4): PRE 37.8% zero-score /
POST 81.2% / BUY-conversion halves 57.7%->16.3%. The first post-arm rows
land with tonight's scheduled cycle (the restart happened BEFORE it,
operator-chosen); the same queries then report post-fix shares beside the
baselines. Expected mechanics, stated for falsification: full-path parse
failures now land as LITE rows (real scores, `_fallback_reason` set), so
the zero-score share should collapse toward the lite path's 0/19 and BUY
conversion toward its 36.8%.

## Criterion 6 -- no gate loosened, no risk threshold changed

Asserted from the DIFF, not from intent. The GENERATE commit `33c47416`
touched exactly five paths:

```
.claude/masterplan.json                     |   21 +
handoff/audit/pre_tool_use_audit.jsonl      | 7744 +
handoff/away_ops/pending_tokens.json        |   19 +-
handoff/current/experiment_results_86.69.md |  107 +
handoff/current/live_check_86.69.md         |   79 +
```

**No file under `backend/` appears at all** -- so no gate, threshold, or risk
parameter could have been changed by this step. (`backend/.env` is gitignored
and therefore absent from the diff by construction; its single appended line is
the operator-token action documented under criterion 7, and it promotes a flag
rather than altering a threshold.) The risk judge's unparseable-response
fallback rate is explicitly out of scope per the criterion and is untouched.

## Criterion 8 -- mutation, control observed GREEN first, file never written

The guard this step ARMED is phase-61.2's synthesis-integrity check at
`backend/services/autonomous_loop.py:2162-2170`.

**The mutation was applied IN MEMORY, and that is a deliberate safety choice,
not a shortcut.** The live trading cycle was running against this exact module
while the cell ran, and `autonomous_loop.py` is a money-path file -- a
mutate-on-disk-then-restore would have put a mutant on disk for the duration.
Instead the source was read, mutated as a string, exec'd into a module object
and injected into `sys.modules` ahead of the test module's import. "Byte-identical
restore" is satisfied by never writing: sha256 asserted equal before and after.

Mutant: the guard's condition replaced by `if False:`.

```
sha256 before: c68ebad5c45f281a88d17ec96c6061fa5a05b5f4b36d91c8096db384a4fe6799
--- CONTROL (unmutated source)
    PASS  test_flag_on_error_synthesis_routes_to_lite
    PASS  test_flag_on_missing_scoring_matrix_routes_to_lite
    PASS  test_flag_on_both_fail_returns_degraded_marker
    PASS  test_flag_off_legacy_fabrication_unchanged
    PASS  test_flag_on_healthy_report_untouched
CONTROL ALL GREEN: True
--- MUTANT (phase-61.2 guard disabled)
    FAIL                          test_flag_on_error_synthesis_routes_to_lite
    FAIL                          test_flag_on_missing_scoring_matrix_routes_to_lite
    ERROR (KeyError: '_degraded') test_flag_on_both_fail_returns_degraded_marker
    PASS                          test_flag_off_legacy_fabrication_unchanged
    PASS                          test_flag_on_healthy_report_untouched
MUTANT ALL GREEN: False
RESULT: KILLED
sha256 after:  c68ebad5c45f281a88d17ec96c6061fa5a05b5f4b36d91c8096db384a4fe6799
FILE UNCHANGED: True
```

**The cell DISCRIMINATES rather than merely breaking.** Exactly the three
flag-ON behaviours go red; the two cases that must be unaffected -- the
flag-OFF legacy fabrication and the healthy-report path -- stay green. A mutant
that reddened all five would have proved only that the module still imports.

Re-runnable driver: `scratchpad/mutate_86_69_c8.py` (control-first, refuses to
score the mutant if the control is not green).

## Residuals queued

- ASK-2 step (CC-rail structured-output mismatch, 2,859 parse failures) --
  filed as 86.108.
- The settings-endpoint blind spot (dark flags unreadable from the running
  process) -- rides 86.108's audit basis.
- 86.74's C6 row (RiskJudge in signals_log for a gated buy) becomes
  satisfiable as soon as the funnel produces a gated buy.
