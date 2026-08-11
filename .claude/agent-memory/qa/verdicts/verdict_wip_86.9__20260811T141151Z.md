STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.9
WRITTEN: 2026-08-11T14:11:51Z

# Q/A write-first record -- step 86.9, CYCLE 2

Cycle-1 CONDITIONAL (wf_28cf4dbb-9aa). Remediation ca78e00a + 61d16d25.
Main self-reported a 5th defect and asked it be graded hardest.

## A. HARNESS COMPLIANCE

- mtimes: research_brief 15:45:17 < contract 15:48:21 < experiment_results/live_check
  16:09:54 (local). ORDER OK.
- Gate: wf_6f5558d5-56b, 8 sources / 21 URLs / recency present. DISCLOSED: WebSearch
  budget 200/200 exhausted -> three-variant discipline did NOT run (research-gate.md
  calls a single-variant search a protocol breach; disclosed, not hidden, and the
  load-bearing findings are internal measurements I re-derived myself).
- harness_log: exactly ONE `## Cycle` row for phase=86.9 (line 34081, Cycle 1221,
  result=CONDITIONAL). So this is the 2nd, NOT the 3rd -> auto-FAIL does NOT bind.
  (`grep -c "phase=86\.9 "` = 2 but hit #2 is prose about the grep, not a row.)
- masterplan: 86.9 status=pending, retry_count=0/max_retries=3. Not flipped. OK.
- No verdict-shopping: evidence CHANGED across 2 commits. OK.

### FINDING A1 [WARN, HARNESS-COMPLIANCE BREACH -- cycle-1 missed it]
contract_86.9.md §4 is headed "Immutable success criteria -- VERBATIM". Re-derived
from .claude/masterplan.json: **5 of 6 differ.**

| # | masterplan (authority) | contract §4 as shipped |
|---|---|---|
| 1 | "...(or an endpoint it serves) **and record the pid and its start time, since the setting is read at cycle start**" | "...(or an endpoint it serves), **not from .env or a new import**" |
| 2 | "...the raise was **INSUFFICIENT rather than closing on the config change alone**" | "...the raise **did NOT fix it**" |
| 3 | "...cycles run AFTER **the rail was repaired 2026-08-09 -- the 2310-2320s figure predates that fix**" | "...cycles run AFTER **the raise**" |
| 4 | "...whether a longer outer budget **increases the window in which a hang goes unnoticed**" | "...whether a longer budget **merely delays the same failure**" |
| 5 | "...recommended or withdrawn **-- a budget raise that leaves 26% of rail time being discarded is treating the symptom**" | clause dropped |
| 6 | identical | identical |

Not cosmetic: c1 DROPPED the very clause whose measurement produced Main's own 5th
defect; c3 substitutes a DIFFERENT qualifying event; c4 substitutes a DIFFERENT
QUESTION (failure latency vs non-DETECTION window). CLAUDE.md requires the criteria be
"copied verbatim from .claude/masterplan.json". Cycle-1's check
`masterplan_diff_criteria_unamended` proved the SOURCE was unamended -- a different
thing from checking the COPY.

## B. DETERMINISTIC

- Immutable command -> stdout `10800.0`, **EXIT=0**. REPRODUCED.
- 1a lint: `git diff --name-only HEAD -- '*.py'` and `git ls-files --others -- '*.py'`
  both EMPTY -> N/A BY DERIVATION (step changed no code). 1b/1c N/A.
- `git status --short`: only audit JSONL + health.jsonl + researcher memory. ZERO
  production files changed. "NOTHING WAS CHANGED" confirmed.

### Criterion 1 -- MET, independently reproduced
`curl http://127.0.0.1:8000/api/settings/` -> http=200 size=1356,
paper_cycle_max_seconds=10800.0, paper_analyze_top_n=5, paper_screen_top_n=10.
`lsof -nP -iTCP:8000 -sTCP:LISTEN` -> Python **66306**.
`ps -o pid=,lstart=,etime= -p 66306` -> `66306 man. 10 aug. 21.33.01 2026 18:40:31`.
pid AND start time both recorded in live_check. MET.

### Criterion 3 -- MET, reproduced EXACTLY
My own `python scripts/diagnostics/measure_analysis_phase.py`:
wall=4532.113s, per-ticker mean **1315.2s**, median 1296.6, serial 7891.1,
parallelism **1.85**, PROJECTED cycle **4492s**, cc_rail started=152 timed_out=1
rate=0.0066, `agent latency : None`. Only lines_parsed moved (74398 vs 73231) --
live log appending. Every figure in the artifact reproduces.

### Criterion 4 -- MET, independently reproduced
`grep -nE "asyncio\.(timeout|wait_for)" backend/services/autonomous_loop.py` -> :426
(comment), :509 (comment), :514 `async with asyncio.timeout(_cycle_timeout)`. NO
asyncio.wait_for anywhere. `_run_single_analysis` def :2088, called :1229 inside
`run_daily_cycle` (def :349) with NO enclosing timeout. No per-ticker cap. TRUE.

### Criterion 6 -- MET (partially permission-blocked)
Key NAMES symmetric diff of backend/.env vs backend/.env.bak.20260809T155016 -> EMPTY
("KEY SETS IDENTICAL"), 51 vs 51 keys. `PAPER_CYCLE_MAX_SECONDS` 7200.0 -> 10800.0.
`PAPER_MAX_PER_SECTOR` unchanged at 5. Backup retained + referenced.
BLOCKED: a full value-by-value `diff` of the two .env files was DENIED by the
permission system (twice). So "exactly ONE changed value" is corroborated by key-set
identity + the single key I could read, NOT exhaustively re-derived. Disclosed, not
worked around.

## C. THE 5th DEFECT, GRADED HARDEST -- Main's conclusion is WRONG, in its own
## disfavour. The predecessor IS identifiable and criterion 2 is STRONGER than claimed.

Main: "**Can I recover the budget that predecessor held? No.**" / "the value in force
is unrecoverable post-hoc" / criterion 2 "satisfied but weakly".

I recovered it in two greps.

1. `grep "Application startup complete" backend.log` -> **exactly 1 hit**:
   2026-08-10 21:33:04 (pid 66306) -- AFTER the cycle ended 21:15:34.
2. Archive `handoff/logs/backend.log.20260810T064130Z.gz`, last startups:
   `Started server process [84494]` 2026-08-09 17:08:08;
   `Started server process [6644]`  2026-08-09 18:56:03;
   **`Started server process [43839]` 2026-08-09 22:11:55** <- LAST before the cycle.
=> **The 2026-08-10 20:00:02 cycle ran under pid 43839, started 2026-08-09 22:11:55
CEST -- 6h21m AFTER the .env write (2026-08-09T13:50Z = 15:50 CEST, corroborated by
the backup filename .env.bak.20260809T155016).** A fresh process builds Settings from
backend/.env on its first get_settings(); `_scheduled_run()` (paper_trading.py:1485-87)
calls `get_settings()` at fire time and passes it to `run_daily_cycle`, whose
`:406 settings = settings or get_settings()` uses that object, and `:507` reads
paper_cycle_max_seconds from it. So the cycle ran under 10800.0 -- MEASURED.

Independent corroboration that the lru_cache is cleared post-write even without a
restart: `AnalysisOrchestrator construction` lines at 2026-08-09 16:07:06 / 16:12:28 /
16:40:51 are emitted 1-4 lines AFTER `_get_settings_fresh.cache_clear()` +
`_get_settings_fresh()` (autonomous_loop.py:2137-2138).
TZ cross-check: log "2026-08-10 21:33:04" == `ps lstart` "21.33.01" for pid 66306, so
log stamps are LOCAL CEST.

ANSWER TO THE QUESTION ASKED: criterion 2 is **MET**, and "satisfied but weakly" is
honest in DIRECTION (it under-claims, never over-claims) but **FALSE in its central
premise**. The remaining genuine weakness is a different one, which Main states
correctly: 4,532s completes under 7200 too, so the cycle does not show the raise was
NECESSARY. Criterion 2 does not ask that.

### FINDING A2 [WARN] -- §8 census is labelled grep output and is not
Artifact: "Below is the output of `grep -rn "paper_cycle_max_seconds|_CYCLE_BUDGET_FALLBACK_SEC" backend/ scripts/`."
- LITERALLY as written (BRE, no -E): **0 hits** -- `|` is a literal char.
- With `-E`: 18 rows. Symmetric difference vs the 10-row table:
  - in grep, ABSENT from table: test_phase_85_4_cycle_loudness.py:244,
    test_phase_85_5_cycle_lock_split_brain.py:356 & :363,
    test_phase_85_6_anchor_deadlock.py:374, test_phase_38_6_restart_survivable.py:161,
    cycle_lock.py:28, :57, :83.
  - in table, UNPRODUCIBLE by that grep: `scripts/diagnostics/measure_analysis_phase.py:263`
    (that file contains the token **0** times; :263 is `--budget-sec default=7200.0`)
    and `backend/.env:70`.
The remediation replaced a typed COUNT with a curated table wearing a derivation's
label. Same defect class, one layer up.

### FINDING A3 [WARN] -- "_cycle_timeout is never logged" is FALSE
`gzcat archive | grep 7200` -> **5 hits**, three of them
`"Paper trading cycle TIMED OUT after 7200s"` (08-04 22:00:01, 08-06 22:00:01,
08-07 22:00:01) from autonomous_loop.py:1896. So the budget IS logged -- on the
TIMEOUT path only. The parenthetical ("no cycle-START budget record") is accurate;
the bolded assertion is not. Bonus: those 3 records independently corroborate that
both pre-raise overrun cycles ran under 7200s, evidence §7 did not use. 86.54 remains
worth filing (a failure-only record is not observability).

### FINDING A4 [NOTE] -- "the archive holding that cycle"
live_check: grep ran over "backend.log and the archive holding that cycle". The
archive rotated 2026-08-10 06:41:30Z (08:41 local); the live log runs 08-10 08:41:30
-> 08-11 16:16 and contains the 20:00 cycle (31 hits for "2026-08-10 20:00:0"). The
archive does NOT hold that cycle -- §3 of experiment_results says so itself. Same
archive-misdescription class as cycle-1 defect #4, surviving in the artifact that was
not rewritten.

### FINDING A5 [NOTE] -- "sole inner cap" uncorrected and mis-prescribed
§4: "The sole inner cap is a per-call 150s at claude_code_client.py:593."
:593 `def __init__(self, model_name: str, timeout_s: int = 150)` -- accurate.
But `timeout_s=120` at services/autonomous_loop.py:2960 and :3044, BOTH inside
`_run_claude_analysis` (def :2829), which IS the analysis-path handler (routed at
:2470/:2573/:2582). Cycle-1 raised this and prescribed "qualify to the analysis
phase"; the fix was not applied, and the prescription was itself wrong -- those sites
are IN the analysis phase. Does not change criterion 4: these are per-CALL caps, not
per-TICKER, and a ticker makes many calls.

### Criterion 4's second half -- partial gap [NOTE]
Masterplan asks "whether a longer outer budget increases the window in which a hang
goes UNNOTICED". Answered in substance ("delays a hung ticker's failure by 3,600s;
it does not remove it") but the detection dimension the audit_basis names -- the
phase-85.4 completed-age alarm, "loud within 96h either way" -- is never engaged.
The contract's mis-transcribed c4 asked the latency question instead, so the gap and
FINDING A1 are the same root cause.

### Criterion 5 -- MET in form
#24 RECOMMENDED with provenance dated to research_brief_85.4.md:321 as PRE-fix and
reconciled against the 0.66% post-fix rate; #25 "NOT recommended now, NOT withdrawn".
Both dispositions unambiguous.

## D. WHAT MAIN GOT RIGHT (belongs in the record)
- All 4 cycle-1 remediations landed and I verified each: 8554/8529 now at
  experiment_results:177/:189 (brief :397 confirms the figures); the FOUR->derived
  census; the #24 pre-fix dating; "6 archives" -> ONE archive / 6 cycles.
- The self-incriminating retraction in §1 (trailing-slash probe) is true and useful.
- §9 "What is NOT claimed" and the MEASURED/INFERRED table are real scope honesty --
  they are why every finding here is WARN/NOTE and none is a criterion miss.
- Zero production files changed, exactly as the artifact says.

## E. DISPOSITION
All 6 masterplan criteria MET (1/3/4/6 independently reproduced by me; 2 strengthened
beyond the artifact's own claim; 5 met in form). No production change. No money-path
risk. FAIL is not warranted. PASS is blocked by A1 (harness-compliance: the contract's
"VERBATIM" criteria are not verbatim, 5 of 6, two materially) plus A3/A6 false claims
in bold in two artifacts. -> **CONDITIONAL**, 2nd for this step-id, so the
3rd-CONDITIONAL auto-FAIL does not bind (one `## Cycle` row exists).

COMPLETED: 2026-08-11T14:20:33Z
