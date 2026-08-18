STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.47
WRITTEN: 2026-08-18T01:32:02Z
COMPLETED: 2026-08-18T01:44:21Z

# Q/A write-first record -- step 86.47 (drought census / BUY funnel)

Spawn: Workflow rail, agentType qa. Read qa.md in full at 01:32Z.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command, git status scope, ruff, runtime smoke,
   re-run the step's own rerunnable checks (--verify, bare, --sql)
C. Claim audit: re-derive every number in experiment_results/live_check from BigQuery itself
D. Guard-vacuity: mutate the 13 _check() invariants in drought_census_86_47.py
E. Criteria 1-6 MET/NOT MET with cited evidence

## Findings log (appended as established)

### Prior-attempt / prior-verdict evidence (gathered, NOT applied as a trigger)
- `qa_wip.py 86.47 --spawned-at 2026-08-18T01:32:02Z`: source_present=True,
  attempt_number=2, attempt_number_status=ok, attempt_number_is_lower_bound=False,
  prior_attempts=1, records_retained=2 (gauge, not a counter), identity_checked=True.
- `verdict_history_86_21.py --step 86.47 --evidence-only`: status=ok,
  detail="1 verdict(s) from the ledger", verdicts=FAIL.
- CROSS-CHECK: prior_attempts (1) == ledger verdict count (1) -> ledger is CURRENT,
  not stale. sequence: [FAIL]. This is the cycle-2 fresh-respawn.
- Verdict-shopping test: evidence CHANGED (census gained 14 _check assertions, the
  refusal funnel is new, criterion-1 normalisation corrected, path-coverage corrected).
  Not a verdict-shop.

### B. DETERMINISTIC (all run in .venv, 2026-08-18)
- IMMUTABLE COMMAND: `ast.parse(backend/services/autonomous_loop.py)` -> `parsed`, EXIT=0.
- `drought_census_86_47.py --verify` -> "OK: all 13 invariants hold", EXIT=0.
- `drought_census_86_47.py` (bare) EXIT=0; `--sql` EXIT=0.
- ruff F821,F401,F811 over DERIVED scope
  (`git diff --name-only HEAD -- '*.py'` UNION `git ls-files --others --exclude-standard -- '*.py'`
  = backend/api/sovereign_api.py, backend/services/autonomous_loop.py,
  scripts/qa/drought_census_86_47.py) -> "All checks passed!", EXIT=0. Non-empty set asserted.
- The step's deliverable `scripts/qa/drought_census_86_47.py` is UNTRACKED (new file);
  `git ls-files --error-unmatch` errors, so the derived scope had to union `--others`.

### C. CLAIM AUDIT -- every BigQuery figure INDEPENDENTLY RE-DERIVED (google-cloud-bigquery, ADC)
ALL REPRODUCE EXACTLY. Nothing failed to reproduce.
- paper_trades: 26 distinct trade days, max=2026-08-13, latest row = 2026-08-13 DELL BUY.
  (The step text's "last trade 2026-07-31 (NTAP)" is indeed stale; 2026-07-31 NTAP BUY is
  the second-latest.)
- risk_judge_decision populated: paper_trades BUY 19/34, SELL 0/32; analysis_results 18/580.
- JSON_QUERY judge coverage: 382/526 since 2026-05-01; 256/275 post-break; 13/13 window.
  JSON_VALUE control on the same predicate = 0/526 -> the stated mechanism for cycle 1's
  false negative is REAL and reproducible.
- Silence window: 13 rows, all path=full, all rec=HOLD, 8 REJECT @0%,
  5 approvals @2/2/5/3/2%. Exact match to WINDOW[] including tickers and pcts.
- paper_positions qty>0: DELL Technology 4.806437, NTAP Technology 5.346643. Exact.
- _path coverage: 288/580 all-time; 288/288 from 2026-06-11; MIN(date with _path)=2026-06-11.
- FUNNEL: all 8 cells exact (17/0, 8/0, 221/100, 2/0, 3/3, 211/0, 45/1, 19/7).
- SYNTH_ERROR: 'Failed to parse final report.', path=full, n=219, 2026-06-11..2026-08-13. Exact.
- DAILY tail: all 11 rows exact, 2026-08-11 present.
- pyfinagent_data.risk_intervention_log COUNT(*) = 0. Confirmed.
- pyfinagent_data.analysis_results -> 404 Table not found. The dataset-relocation claim holds.
- STATED REASON verified in the data: all 8 REJECT reasonings mention sector AND
  concentration AND "60" AND Technology. The two quoted sentences are each verbatim in
  exactly 1 of 13 rows (009150.KS and HPE on 2026-08-17) -- quoted honestly as "verbatim
  from the judge's reasoning", and the GROUND generalises across all 8.
- The 60% figure is corroborated in source: `backend/agents/skills/risk_judge.md:30`
  documents portfolio_sector_exposure.threshold_pct (60.0).
- Arithmetic re-derived: 23 weekday / 79 weekday-days = 0.2911; weekend trade-days are
  exactly 2026-04-26 Sun, 2026-05-16 Sat, 2026-05-17 Sun; cycle-1's 26/79 = 0.3291.
  2026-08-15/16 are Sat/Sun, so "the days between are a weekend" holds.
  Sensitivity table exact: 0.0291/0.6813, 0.1250/0.1762, 0.0222/0.7467, 0.4558/0.0004.
  need_healthy=5, need_post=102, total BUYs 111, 211/275 = 76.7%.

### D. MUTATION MATRIX (14 cells + control + null; scratchpad copy, tree untouched)
CONTROL: scratchpad copy output byte-identical to in-tree run (`diff` clean), --verify exit 0.
NULL MUTANT (whitespace): survives, exit 0 -> relocation is inert, matrix measures the subject.
KILLED (6): M1 FAILED-cell BUY 0->99 (the exact cycle-1 killer -- now dies, exit 1);
  M3 a window row becomes BUY; M4 a REJECT becomes APPROVE; M5 path coverage claimed 100%;
  M6 judge coverage in window 13->0; (M1 also kills on bare run).
SURVIVED (8) -- all at exit 0 with "OK: all 13 invariants hold":
  M2  neuter one invariant (`True or ...` on window_has_no_BUY_recommendation)
  M7  daily-tail CONTENT falsified (2026-08-11 -> 6/0/44)
  M8  SYNTH_ERROR fully falsified -> PHASE-1 headline prints
      "PHASE 1 2020-01-01..2020-01-02 -- 'Everything was fine.' 3 rows, path=lite"
  M9  RJD paper_trades BUY 19/34 -> 34/34
  M10 SECTOR_CAP_PCT 60.0 -> 5.0
  M11 POSITIONS tickers DELL/NTAP -> AAPL/MSFT
  M12 HEALTHY-NULL cell gutted (A_pre ok unmarked BUYs 100 -> 1)
  M13 JUDGE_COVERAGE post-break 256/275 -> 3/275

M12 IS THE MATERIAL ONE -- behavioural differential captured verbatim:
      Under the HEALTHY null (p=0.0177)
      the silence IS surprising -- P(0 in 13) = 0.7928, and only 168 analyses
      are needed to reach that bar, so 13 is already MORE than enough:
      BUY supply has NOT returned to the pre-break rate.
  P=0.79 is the opposite of surprising and 13 < 168, yet the conclusion sentence is
  printed UNCONDITIONALLY. The criterion-6 CONCLUSION is hardcoded prose, not derived
  from the computed statistic, and NO invariant guards the healthy-null cell.
  This is the same defect class the artifact claims to have fixed ("a prose number that
  contradicts the computation it summarises").

GUARD-COUNT DEFECT: `grep -c '_check('` = 14 calls, but N_INVARIANTS = 13 (hardcoded).
  So --verify prints "OK: all 13 invariants hold" while running 14, and M2 proves a
  DELETED guard is invisible to the count. "13 assertions"/"13 invariants" is restated in
  experiment_results (x2) and live_check §9. Unmeasured count in production source.

M9 differential: prints "paper_trades BUY 34/34 = 100.0%" and then, unconditionally,
  "=> too sparse to key a funnel on" -- criterion 3's own conclusion contradicting its
  own printed evidence, exit 0.
M14 (added cell): DAILY_TAIL 2026-08-17 analyses 7 -> 70 SURVIVES. `n_an` -- the
  denominator of EVERY criterion-6 probability -- is derived from DAILY_TAIL and is
  cross-checked against nothing. Output: header "P(0 in 76)", all four probabilities
  corrupted (0.6813->0.1061, 0.1762->0.0000, 0.7467->0.1812, 0.0004->0.0000), the row
  label still reading "matches all 13 window rows", and §4 still printing
  "REACHED THE GATE 13/13". Two representations of the same 13 disagree on one page,
  exit 0, "OK: all 13 invariants hold". `len(WINDOW)==13` and `n_an==13` are never tied.
MATRIX TOTAL: 15 cells + control + null -> 6 KILLED, 9 SURVIVED.
  The census docstring (lines 18-23) claims "Every recorded figure is now guarded."
  KNOWN-MEMBER RECALL TEST over its own recorded constants REFUTES that: DAILY_TAIL
  content, SYNTH_ERROR, RJD paper_trades cells, SECTOR_CAP_PCT, POSITIONS tickers,
  JUDGE_COVERAGE non-window rows, the healthy-null FUNNEL cell and n_an are all unguarded.

### THE HARD CLAIM DEFECT (reproduced, not inferred)
`experiment_results_86.47.md:131` -- "Also fixed: ... and the contract's \"~97\" against
the census's 102." The spawn context states it more strongly: "The contract no longer
restates the number at all."
MEASURED: `grep -n "97" handoff/current/contract_86.47.md` -> line 119 still reads
  "artifact states the power requirement (~97 analyses, ~16 days) rather than"
while `drought_census_86_47.py` computes need_post = 102. The contract WAS edited in
cycle 2 (mtime 03:30:59, AFTER the census at 03:29:29 -- the P5 paragraph at line 113
gained the "~62 orders of magnitude / 10^-61.7 / p=0.452" text), so the file was open and
the P6 sentence was left behind. A false past-tense remediation claim, and it violates the
very principle the same sentence states ("a figure restated in prose is a figure that can
go stale").

Related NOTE: `experiment_results:71-73` says the hardcoded "~48 orders of magnitude" is
"Now derived from the computed value." The shipped census contains NEITHER figure -- it was
DELETED -- and ~62 now lives as PROSE in contract:113, using p=0.452 (100/221) where the
census/live_check use 0.4558 (103/226) for the same "pre-break rate" label.

### A. HARNESS COMPLIANCE -- all 5 CLEAN
1. research-gate-before-contract: research_brief_86.47.md 03:04:55 < contract; envelope
   brief_status=COMPLETE, gate_passed=true, external_sources_read_in_full=14 (floor 5),
   urls_collected=48 (floor 10), recency_scan_performed=true, coverage.audit_class=true,
   rounds=14, dry_rounds=2, K_required=2, dry=true. "Recency scan (2024-2026) -- PERFORMED"
   at brief:113.
2. contract-before-generate: research < contract holds. Contract mtime is 90s AFTER the
   census only because of the cycle-2 re-edit; the cycle-1 critique records the original
   ordering (contract 03:08:53 < census 03:09:03). Not a violation.
   All SIX immutable criteria verified VERBATIM present in the contract by exact string
   match against .claude/masterplan.json.
3. experiment_results present (7,872 bytes).
4. log-last: `grep -F "phase=86.47" handoff/harness_log.md` -> no rows; masterplan 86.47
   status=pending, retry_count=0, max_retries=3. Clean.
5. no-verdict-shopping: evidence CHANGED materially between spawns. Not a shop.

### CRITERION 5 SCOPE AUDIT (independently verified)
Peer-session files: sovereign_api.py mtime 2026-08-17 15:54:50, autonomous_loop.py
2026-08-17 21:42:56 -- both inside the disclosed "15:54-22:19" range. Diffs are a
`1y` red-line window and a /reports final_summary fallback; `git diff | grep -c 86.47` = 0.
Authored file scan: no secrets, no subprocess/eval/exec/shell, no requests, no file writes,
no broad except, no execute_buy/execute_sell/kill_switch/stop_loss/paper_max references.
Runtime smoke: `import backend.services.autonomous_loop` OK, `import backend.api.sovereign_api`
OK, `curl :8000/api/health` -> 200. No UI claims in this step, so gate 1c does not apply and
I took no browser capture.

### E. CRITERIA
C1 MET   -- base rate re-derived from BQ (I reproduced: 26 trade days, last 2026-08-13 DELL
            BUY); normalisation rule stated with both endpoints; 3 weekend trade-days named
            and excluded; 23/79 = 0.2911 exact; the step text's premise refuted, not inherited.
            NOTE: the anomaly question is answered in ANALYSIS units, never restated in the
            trade units criterion 1 names. Permissive clause, so not a miss.
C2 MET   -- window stated (2026-08-14, 2026-08-17); 13 analyses, 0 BUY-class, 13/13 reached
            the gate, 8 REFUSED at 0%, 5 approved at 2-5%; queries printed by --sql. I
            reproduced all 13 rows exactly and confirmed ALL 8 REJECT reasonings carry
            sector + concentration + "60" + Technology. The 60% is corroborated in source at
            backend/agents/skills/risk_judge.md:30 (threshold_pct 60.0).
            NOTE: the reason-producing query ($.reasoning) is NOT among the 9 --sql prints.
C3 MET   -- population PROVED before use: 18/580 = 3.1% (analysis_results), 19/34 BUY and
            0/32 SELL (paper_trades), risk_intervention_log 0 rows -- all reproduced. Funnel
            then derived another way (JSON_QUERY), exactly as the criterion instructs. I
            independently confirmed the mechanism: JSON_VALUE 0/526 vs JSON_QUERY 382/526.
C4 MET   -- every count split by $._path with its epoch; 288/580 = 49.7% all-time, 100% only
            from 2026-06-11 (I confirmed MIN(date with _path) = 2026-06-11); the consequence
            stated (pre-break baseline cell is path-UNKNOWN).
            NOTE: the research brief still asserts "_path ... 100% of rows" at lines 42, 362,
            516, unannotated; 362's scoped form ("every day since 2026-06-01") is measurably
            false -- rows 06-01..06-10 carry no _path.
C5 MET   -- nothing loosened, nothing promoted, no .env, only file authored is the untracked
            read-only census. No miscalibration claim; outcome evidence correctly demanded.
C6 MET (WARN) -- explicit base-rate check IS stated, with a four-null sensitivity table
            INCLUDING the healthy-funnel null (0.4558 -> 0.0004) and the power bounds (5 vs
            102); every cell reproduced exactly. PHASE-1's causal claim carries its own check
            (10^-61.7 over 236 at p=0.452) in the contract. WARN: the conclusion SENTENCE is
            hardcoded prose, not derived; M12 and M14 both invert it while exiting 0.

### VERDICT REASONING
No criterion is missed and every figure reproduces, so FAIL is wrong. Two WARN-level
findings -- a falsified past-tense remediation claim, and a completeness claim about the
guard set that a known-member recall test refutes -- cap this at CONDITIONAL.

