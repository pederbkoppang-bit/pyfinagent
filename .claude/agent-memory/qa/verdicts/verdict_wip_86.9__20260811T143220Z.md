STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.9
WRITTEN: 2026-08-11T14:32:20Z

# Q/A cycle 3 for step 86.9 (paper_cycle_max_seconds raise)

Spawn context: cycle 3, attempt 3 of 5 under the 86.32 budget. Prior verdicts C, C.
Main claims 5 cycle-2 findings fixed via commit 473f9b43. ZERO production files
changed by this step (config/.env only).

## Plan
- A. harness-compliance audit (5 items)
- B. deterministic: immutable command exit code; git status/diff scope; re-derive
  the §8 pinned grep; re-derive the contract §4 generation claim
- C. per-criterion MET/NOT MET, with the graded-hardest focus on §7 balance,
  criterion 4 both halves, and 86.54 vacuity check

## Findings log (appended as established)

### B. Deterministic (established 14:35Z)
- IMMUTABLE CMD: `bash -c 'source .venv/bin/activate && python -c "...paper_cycle_max_seconds"'`
  -> stdout `10800.0`, **exit=0**.
- git status: only audit jsonl + researcher memory + my own WIP are dirty. NO production
  file modified by this step. `git diff --stat 473f9b43 HEAD` = masterplan.json(3) +
  CHANGELOG(12) + harness_log(52) only.
- 473f9b43 touched: contract_86.9, experiment_results_86.9, live_check_86.9,
  operator_asks doc, masterplan.json(86.54 audit_basis only). NO production code.
- CRITERIA IMMUTABILITY: md5 of 86.9 success_criteria = 92da59702eb0 across the last
  25 commits touching masterplan.json; len=6; status=pending throughout. UNAMENDED.
- 3rd-CONDITIONAL COUNT: header-anchored
  `grep -cE '^## Cycle [0-9]+ -- .* -- phase=86\.9 result='` = **2** (Cycle 1221, 1222,
  both CONDITIONAL). Naive `grep -c 'phase=86.9 '` = 3, inflated by a prose quote at
  line 34087. Main's claim REPRODUCES. => a CONDITIONAL here would be the 3rd =>
  MUST be FAIL per qa.md. Verdict space is {PASS, FAIL}.

### FIX-1 (contract §4 byte-for-byte) -- VERIFIED FIXED
Programmatic compare of the 6 blockquoted lines vs masterplan success_criteria:
ALL SIX BYTE-EQUAL (incl. c1's "record the pid and its start time" clause and c4's
"goes unnoticed" detection wording). Blocker resolved.

### FIX-3 (§8 regenerated census) -- VERIFIED FIXED
`/usr/bin/grep -rnIE "paper_cycle_max_seconds|_CYCLE_BUDGET_FALLBACK_SEC" backend/ scripts/`
-> 18 rows. Symmetric difference vs the pasted fenced block = **0 in both directions**,
and ORDER-EQUAL too. The grep-binary claim also reproduces: shell `grep` here yields 18
as well once `-I` is present (Main's 18-vs-26 was the pre-`-I` form).

### FIX-4 (inner caps) -- source-verified
`asyncio.timeout` appears 3x in backend/services/autonomous_loop.py but only :514 is a
call (:426, :509 are comments). No `asyncio.wait_for`. cycle_health.py:61 = 93_600.0
(26h), :80 = 345_600.0 (96h) -- Main's table MATCHES source.

### 86.54 audit_basis (asked to check for vacuity)
The corrected basis still quotes `grep -c 'cycle_timeout|effective cycle budget'
backend.log` WITHOUT -E, which is vacuous-by-construction (basic grep treats | as a
literal). MEASURED both forms against backend.log: literal form -> 0 (exit 1); correct
`-E` form -> **also 0** (exit 1); positive control 'Application startup complete' -> 1.
So the FILED DEFECT SURVIVES a correct command -- not vacuous. Residual: the quoted
command is still the vacuous shape Main just fixed one artifact over. NOTE-level.

### A. Harness compliance -- CLEAN
- research gate: envelope brief_status COMPLETE, gate_passed true, 8 sources (>=5),
  21 URLs (>=10), recency_scan_performed true, 14 internal files. DISCLOSED GAP:
  three-variant search discipline did NOT run (WebSearch 200/200 exhausted) -- stated
  prominently in contract §1. NOTE, not a blocker; the load-bearing findings are
  internal measurements that I re-derived myself.
- contract-before-generate: first git appearance research_brief+contract = 26037c1e
  2026-08-11 15:48:57; experiment_results+live_check = 38ae0f9c 15:51:27. CORRECT ORDER.
- experiment_results present (19,302 b). live_check present (5,499 b).
- log-last: masterplan 86.9 status=pending (NOT flipped). harness_log carries the two
  in-progress CONDITIONAL rows only. OK.
- no-verdict-shopping: evidence CHANGED (473f9b43: contract +41, results +220,
  live_check +45). Legitimate cycle-3 respawn.

### Criterion-by-criterion (all independently re-derived by me)
- c1 MET. curl /api/settings/ -> 10800.0; analyze_top_n=5; screen_top_n=10. Listener
  pid 66306 (lsof). `ps -o pid=,lstart= -p 66306` -> man. 10 aug. 21.33.01 2026.
- c2 MET. measure_analysis_phase.py reproduces CYCLE started=2026-08-10 20:00:02.593000
  terminal=completed wall=4532.113s. The pid-43839 chain VERIFIED BY ME: live backend.log
  spans 2026-08-10 08:41:30 -> now and contains exactly ONE startup (66306 @21:33);
  archive backend.log.20260810T064130Z.gz spans 08-04 20:27:15 -> 08-10 08:41:28
  (contiguous) and its LAST startup is `Started server process [43839]` immediately
  above a 2026-08-09 22:11:55,250 record. backend/.env mtime 9 aug 15:50, 6129 b vs
  backup 6128 b (+1 byte = 7200.0->10800.0), unmodified since. So 43839 constructed
  Settings with 10800.0. MEASURED, as claimed.
- c3 MET. My own run of the named script is numerically IDENTICAL (mean 1315.2s,
  median 1296.6s, serial 7891.1s, parallelism 1.85, projected 4492s, cc_rail
  152/1/0.0066). lines_parsed differs (75385 vs 73231) because the log grew -- expected.
- c4 MET, BOTH halves. Latency: `_run_single_analysis` = backend/services/autonomous_loop.py
  lines 2088-2305; occurrences of timeout / wait_for / asyncio.timeout / TimeoutError
  inside it = 0 / 0 / 0 / 0. Only :514 is a real asyncio.timeout call (:426, :509 are
  comments). Detection: cycle_health.py:61 = 93_600.0, :80 = 345_600.0; 3600/93600 =
  3.85% ("3.8%"), 3600/345600 = 1.04% ("1.0%"). Both check out.
- c5 MET IN FORM (#24 RECOMMENDED, #25 explicitly "NOT recommended now, NOT withdrawn"),
  but see F1/F2/F3 -- its supporting evidence misstates its own source.
- c6 MET. analyze_top_n=5 live; .env.bak.20260809T155016 retained and referenced.
  NOTE: direct read of backend/.env was DENIED by the permission system; corroborated
  via the endpoint + the +1-byte delta vs backup instead.

## FINDINGS (all reproducible; ground truth = research_brief_86.9.md:380-386 table)
Table: c1 08-05 0.2339 (no overrun, 5670s proj) | c2 08-06 0.1486 TIMEOUT |
c3 08-07 0.1808 TIMEOUT | c4 0.0 | c5 0.0 | c6 0.0988 | c7 08-10 0.0066.

F1 [Contradiction, MATERIAL] experiment_results §7 line 238: "overrun cycles ran a
9.9%-23.4% rail-timeout rate". The overrun cycles are #2 = 14.9% and #3 = 18.1%. The
range endpoints belong to #6 (9.88%) and #1 (23.39%), NEITHER of which overran. #1
carries the HIGHEST rate in the set and COMPLETED (5,670s projected) -- a direct
counterexample to §7(b)'s "The overruns were produced by rail timeouts", which the
widened range conceals. Main's OWN contract §3:65 states the correct pair
"(18.1% / 14.9% vs a 0.66% baseline)", so the artifact contradicts its own contract.

F2 [Contradiction] §5 line 196: "five other measured cycles ran 9.9%-23.4%" -- FOUR
(#1,#2,#3,#6). Of the six non-#7 cycles, TWO ran 0.0%. Overstates prevalence 4/6 -> 5/6
in the direction that supports the recommendation. Figure adopted verbatim from the
cycle-2 critique (evaluator_critique_86.9.md:44) without re-derivation.

F3 [Unjustified_Inference] §7 lines 239-240: "32 x 150s = 4,800s ... the waste is 3.6x
the problem it caused" divides SUBPROCESS-seconds by WALL-seconds. The brief
(:413-417) does the parallelism conversion (~1.85 -> ~2,600s wall) and states the
estimate caveat AGAINST INTEREST; corrected ratio ~1.95x. §7 dropped both, inflating
the multiple ~1.85x.

F4 [NOTE] 86.54 audit_basis still quotes a non-`-E` grep. Measured: correct `-E` form
returns the SAME 0 (positive control = 1), so the filed defect is REAL, not vacuous.
Hazard is forward-looking: an executor copying that command to verify the FIX would get
a false zero.

## VERDICT REASONING
All six criteria MET in substance; harness compliance clean; zero production files
changed; the five cycle-2 findings are genuinely fixed and I verified each checkable
one. F1/F2/F3 are fixable-claim defects => my judgment is CONDITIONAL. Header-anchored
prior count = 2 consecutive CONDITIONALs (Cycle 1221, 1222), no intervening PASS/FAIL
=> qa.md 3rd-CONDITIONAL auto-FAIL converts this to FAIL.

COMPLETED: 2026-08-11T14:47:05Z
