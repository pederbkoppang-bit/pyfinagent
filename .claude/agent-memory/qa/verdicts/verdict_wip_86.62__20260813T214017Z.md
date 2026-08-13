STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.62
WRITTEN: 2026-08-13T21:40:17Z

# Q/A write-first record -- step 86.62 (ATTEMPT 1)

Role: Layer-3 Q/A evaluator. Crash-survival record, NOT a verdict.

## Attempt counter
`python scripts/qa/qa_wip.py 86.62` -> records_retained: 1, prior_records: [].
The one retained record is MY OWN (this file). => **ATTEMPT 1**, prior-verdict
sequence: (none). Secondary cross-check `grep -c "phase=86\.62" handoff/harness_log.md`
-> **0**. Ledger and log AGREE. 3rd-CONDITIONAL rule not triggered.

## Harness-compliance audit (5 items) -- CLEAN
1. research-gate-before-contract: brief exists; envelope `brief_status: COMPLETE`,
   `gate_passed: true`, `external_sources_read_in_full: 6` (floor 5),
   `urls_collected: 19` (floor 10), `recency_scan_performed: true`. PASS
2. contract-before-generate (mtimes, UTC):
   research_brief 21:34:22Z < contract 21:36:42Z < experiment_results 21:38:58Z
   < live_check 21:39:28Z. Correct order. PASS
3. experiment_results present. PASS
4. log-last: harness_log rows for 86.62 = 0; masterplan status = `pending`. PASS
5. no-verdict-shopping: attempt 1, no prior verdict. N/A. PASS

## Deterministic
- Immutable cmd -> stdout `4`, **EXIT=0**. Matches live_check's "4, exit 0".
- Lint gate 1a: N/A -- diff touches **0** `*.py` (commit c6519b43 = 3 handoff .md;
  `git diff --stat HEAD -- backend/` EMPTY). Per the empty-set guard I do NOT report
  a lint pass. 1b (frontend) N/A. 1c (UI) N/A -- no UI claim. 1d N/A -- no backend change.

## Population re-derivation (cat backend.log + gunzip -c all 6 rotated .gz, one pass)
| quantity | CLAIMED | RE-DERIVED | |
|---|---|---|---|
| ^{"timestamp" lines | 912,459 | **912,572** | +113, live log grew; NOTE not finding |
| days / range | 21, 07-24..08-13 | **21, 2026-07-24..2026-08-13** | EXACT |
| cycles ("Paper trading: Step 1") | 19 | **19 on 17 days** | EXACT |
| 404 promoted_strategies | 19 / 17d | **19 / 17d** | EXACT |
| MetaCoordinator decision | 14 | **14** (10 perf_opt + 4 skill_opt) | EXACT |
| p95 breaches | 10 of 14 (71.4%), 9d | **10 of 14, 9d** | EXACT |
| p95 min/max | 2,750 / 13,341 | **2750 / 13341** | EXACT (2750,3420,3973,6267,6500,6602,6701,6812,7776,13341; only threshold 500) |
| rate limit broad | 68 / 19d | **68 / 19d** | EXACT |
| in social_sentiment | 27 / 14d | **27 / 14d** | EXACT |
| quant_opt | "0 times in 21 days" | **17 occurrences** | **F1** |

Every headline number reproduced exactly. That is genuinely strong.

## F1 (BLOCKING, criterion 1) -- p95 remedial-action claim is WRONG AT SOURCE
contract:37-38 "`quant_opt` -- the action the p95 breach is supposed to trigger -- fired
0 times in 21 days." live_check:53-54 "`quant_opt` appears 0 times in 21 days -- the
remedial action the breach is meant to trigger has never fired."
`backend/agents/meta_coordinator.py::decide()` is an EARLY-RETURN ladder:
  :157 if health.p95_latency_ms > self.latency_threshold_ms:
  :159     return CoordinatorDecision(action="perf_opt", ...)     <- Priority 1
  :165 if health.sharpe_ratio < self.sharpe_target and days_since_last_quant_opt >= ...:
  :170     return CoordinatorDecision(action="quant_opt", ...)    <- Priority 2, LOW SHARPE
(a) wrong action: p95 -> **perf_opt**, not quant_opt;
(b) inverted fact: perf_opt fired **10 of 10** -- the artifact's own verbatim quote one
    line above ("MetaCoordinator decision: perf_opt (reason=p95 latency ...)") says so;
(c) literally false count: `quant_opt` occurs 17x (substring of module `quant_optimizer`).
    Under the charitable reading (`MetaCoordinator decision: quant_opt`) the count IS 0,
    but the gloss is still wrong.
MISSED CORRECT FINDING: the early return means a chronic p95 breach **STARVES**
quant_opt and skill_opt -- on 10 of 14 decisions Priority 2/3 were never evaluated.

## F2 (WARN, criterion 2) -- the labelled INFERENCE is REFUTED by a 15-second check
experiment_results:85-90 (labelled "INFERENCE ... NOT measured", "I did not verify which
endpoints populate those entries"), nominated at :173-174 as the argument for a future
threshold change. Two measurements, both contradict it:
(a) LIVE `curl :8000/api/observability/latency?window=300`, backend idle, no cycle:
    p50 5.2 / **p95 2680.2** / p99 4594.1, 37 reqs;
    /api/paper-trading/portfolio **p95 4724.7ms**, /status 2232.1, /api/health 5.4.
    Threshold breached 5.4x with ZERO batch traffic present.
(b) HISTORICAL: endpoint mix in the 300s before each of the 10 breaches (147,416 uvicorn
    access lines). All ten windows dominated by FRONTEND DASHBOARD POLLING; no
    analysis/agent endpoint in any top-6. The 6267ms window (2026-08-11 21:21:28, the
    cycle this step is named for) = 111 reqs: live-prices 17, portfolio 16, snapshots 16,
    kill-switch 16, freshness 15, gate 15.
Correct reading: the interactive endpoints ARE the slow ones (portfolio p95 4.7s). The
500ms threshold measures what it claims; the breach is a TRUE POSITIVE about user-visible
latency, not batch contamination. Criterion 2's LITERAL requirements remain MET.

## F3 (BLOCKING, criterion 4) -- the consumer was never read, and the answer is half-wrong
Criterion 4 prescribes a METHOD: "determined by **reading the consumer**".
Citation census across the three deliverables (avg_sentiment / analysis.py / NO_DATA):
  contract 0/0/0 · experiment_results 0/0/0 · live_check 0/0/0
POSITIVE CONTROL: `avg_sentiment` DOES appear in research_brief_86.62.md, so the probe
is not returning a false zero -- the researcher surfaced it and GENERATE dropped it.
Actual consumer `backend/tasks/analysis.py:251`:
  social_sentiment_score=social_data_dict.get("avg_sentiment") if isinstance(...) else None
Producer `backend/tools/social_sentiment.py:73-81` has TWO rate-limit branches:
  if not feed:
      if fallback_articles: return _score_fallback_articles(...)   # -> avg_sentiment 0.0  ZEROES
      return {"ticker","signal":"NO_DATA","summary"}               # NO avg_sentiment key -> .get()=None  OMITS
The codebase does **BOTH**. The artifact asserts flatly "the production path ZEROES"
(contract:42, experiment_results:138) and never reports the omitting branch -- on the
exact dichotomy ("zeroes ... VERSUS omitting") the criterion exists to resolve.

## Criterion roll-up
- C1 **NOT MET** -- recurrence measurement is exemplary and "not transient" is properly
  earned for all three, but degradation 2's causal account carries F1 in BOTH the
  contract and the operator-facing live_check.
- C2 **MET** -- re-derivation exact; population correctly stated and verified at source
  (main.py:617 global middleware records EVERY request; perf_tracker.py:59 window 300,
  :63 window filter, :76 latencies; meta_coordinator.py:157/:266-267). WARN = F2.
- C3 **MET** -- object `sunny-might-477607-p8:pyfinagent_data.promoted_strategies`;
  notFound-vs-403 discrimination sound; explicit YES; consequence NIL verified: 
  decide_trades signature at portfolio_manager.py:164-172 has no best_params;
  autonomous_loop.py:500-505 sets exactly TWO summary fields (best_params_sharpe,
  strategy_params); corroborated independently by strategy_registry.py:38 and
  strategy_candidate_producer.py:35 comments. Well evidenced.
- C4 **NOT MET** -- F3.
- C5 **MET** -- I TESTED the author's reading and SIDE WITH IT. Under the strict reading
  the trailing clause "speculation in either direction is recorded as untested" could
  never fire (if demonstration/refutation were mandatory there would be no speculation
  left to record) -- a clause that cannot fire is vacuous, so the permissive reading is
  the one that gives every clause work. 86.60 demonstrated with mechanism; 86.47 refused
  and recorded UNTESTED with the measurement named (scorer replay, abstain vs zero).
  CAVEAT: the 86.60 mechanism rides on the same "0.0 in the neutral band" story F3 shows
  is only the fallback-articles branch -- the link is real for that branch only.
- C6 **MET** -- `git diff --stat HEAD -- backend/` EMPTY; commit c6519b43 touches 0
  backend files; DEFAULT_LATENCY_THRESHOLD_MS = 500.0 intact at meta_coordinator.py:120;
  last commit touching that file is phase-23.8.3 (unrelated). No threshold moved.

## Attack #1 (68/19 vs 27/14) -- author UPHELD, not a finding
Both reproduce EXACTLY. All 68 broad "rate limit" lines are Alpha Vantage; 27 carry
"in social_sentiment". Criterion 4 names "the social-sentiment rate limit", so 27/14 IS
the criterion's population. Pinning the disagreement instead of silently reconciling is
correct practice. (Residual 41 non-social AV limits = a different, larger degradation,
out of scope here.)

## ADJACENT finding -- properly quarantined, NOT smuggled in
contract:105-106, experiment_results:178-190, live_check:113-124 each state explicitly
that the cycle-self-reports-CLEAN finding is owned by NO criterion and is not claimed as
a deliverable. Confirmed. Good practice.

## Minor NOTEs (no verdict effect)
- contract header says "Date: 2026-08-14 (~01:40 CEST)"; mtime is 2026-08-13 23:36 CEST.
- I did not independently query BQ to confirm promoted_strategies is absent; I relied on
  the 19 reproduced verbatim 404s. Bound on MY verification, not a finding.

## Separation of duties
Main edited `.claude/agents/qa.md` today (phase-86.75), incl. the attempt counter I ran.
INERT for this evaluation: (a) attempt-counter source change is moot -- qa_wip AND
harness_log both say attempt 1; (b) the removed "prior verdict is ground truth" clause is
moot -- no prior verdict exists for 86.62; (c) every finding above rests on re-derived
measurement and source reads, not on qa.md's discretionary text. Operator review still
appropriate as a standing structural matter.

## Verdict issued: FAIL (2 criteria missed: C1, C4)

COMPLETED: 2026-08-13T21:52:10Z
