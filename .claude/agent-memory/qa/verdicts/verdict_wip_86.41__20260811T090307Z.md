STATUS: INCOMPLETE -- not a verdict
STEP: 86.41
WRITTEN: 2026-08-11T09:03:07Z

# Q/A write-first record -- step 86.41 (EVALUATE)

Launch: Workflow structured-output rail. Read `.claude/agents/qa.md` in full at 09:03Z.

## A. Harness compliance (5 items) -- CLEAN so far
- research_brief_86.41.md exists, mtime 10:41:48 CEST; contract 10:45:45; code 10:58:32 -> ORDER OK
- commits: contract b8d2ea96 10:46:04 < guard 73dcf2c8 10:52:32 < tests 678d979f 10:58:54 < artifacts c9314945 11:02:38
- IMMUTABLE CRITERIA UNAMENDED: md5 of verification block for 86.41 at b8d2ea96 == at HEAD
  (07e298ad8c052e7235ae295a69768642). Commit 48741e44 edited the step NAME only (premise refutation).
- harness_log.md: grep -F "86.41" => 0 hits (log-last respected); masterplan status=pending, retry_count=0
- No prior evaluator_critique for 86.41 -> cycle 1, no verdict-shopping possible

## B. Deterministic
- IMMUTABLE COMMAND reproduced: `parsed`, exit=0. Scope limit acknowledged by Main (parses
  autonomous_loop.py, untouched by this step) -- I agree, reaches criterion 1 only.
- `git status --short`: no uncommitted production/test changes. Only agent-memory + audit jsonl churn.
- Step scope (git diff --name-only b8d2ea96..HEAD -- '*.py'):
  backend/agents/orchestrator.py, backend/tests/test_phase_86_41_quant_isolation.py,
  scripts/qa/mutation_matrix_86_41.py, scripts/qa/derive_lite_fallback_census_86_38.py
  (the last one changed by 8e95fb88 at 10:47:36, labelled phase-86.38 -- the peer/filer's
  classifier correction, disclosed in the artifact.)
- CRITERION 5 pre-check: `git diff b8d2ea96..HEAD -- backend/services/autonomous_loop.py` = 0 bytes.
  Last commit touching that file: 7a7184d2 2026-08-11 09:58:08 (phase-86.38), BEFORE this step's contract.

## C. Census re-derivation (criterion 2)
- I re-ran `python scripts/qa/derive_lite_fallback_census_86_38.py` MYSELF: exit=0.
  Coverage block reproduces byte-for-byte with the artifact's quote (416/416 ... total accounted 442),
  TOTAL 67 full / 9 lite / 11.8% over 10 days. REPRODUCES.
- Independent grep-equivalent recall test (zgrep 'falling back to lite Claude analyzer'):
  416,12,5,1,3,2 + backend.log 3 = 442. MATCHES the instrument exactly.

### FINDING C1 (measured, WARN) -- the coverage assertion is VACUOUS against the
### exact defect it names, and the artifact OVERCLAIMS what it protects
experiment_results_86.41.md:73 states: "The 416-event assertion protects against a parser that
drops rows." MEASURED FALSE. In the current instrument `raw += 1` and `per_file_parsed += 1`
happen in the SAME unconditional branch, so `parsed == raw` is structurally guaranteed unless a
future edit inserts a skip BETWEEN them. I reproduced the historical defect shape (read-level
JSON-only filter, which is what actually dropped the 416) by monkeypatching `read_lines` in an
in-memory import (no file written):
    MUTANT read_lines=JSON-only  rc = 0
      backend.log.20260612T104931Z.gz  raw=0  parsed=0  ok
      ... total accounted: 9   (vs 442 unmutated)
    TOTAL 67 9 11.8%   <-- unchanged, census still printed, exit 0
The assertion stayed GREEN while 433 of 442 events vanished. So it protects against a drop that
occurs AFTER counting, not against the read-level filter that caused the 416 loss.
Materiality: the literal criterion-2 requirements ("re-derived with the instrument", "coverage
assertion shown passing") are still MET, and the population is independently corroborated by my
own grep, so the NUMBERS stand. The defect is in the CLAIM about the guard.

## D. Claim re-derivation (the 34 / 17 / 18 family)
- "42 retained logs" REPRODUCES: 35 *.log + 6 *.gz in handoff/logs + ./backend.log = 42.
- "34 raw events" REPRODUCES as a LINE count of
  /QuantAgent failed for (\S+): .*NoneType' object has no attribute 'get'/ over those 42 files.

### FINDING D1 (measured, WARN) -- "17 have no 429 in window" DOES NOT REPRODUCE;
### it is a line-vs-event category error, and there is no unattributed class
Each real occurrence emits TWO matching lines exactly 17 lines apart (verified in
backend.log.20260810T064130Z.gz: DDOG 36367 orchestrator INFO + 36384 autonomous_loop WARNING;
CRWD 38288/38305; AAPL 71775/71792). Per-file line counts are all even (12,6,6,4,4,2 = 34) => 17
distinct events.
My independent 25-line-window measurement over the same 42 files:
    total NoneType-get LINES: 34
      with SEC-429 cue in window: 34    without: 0
      cue kinds: failed_to_fetch_cik_429=20, sec429_retrying=14
      distinct tickers: 13 (AAPL COHR CRWD DDOG DELL DVA INTC MU NTAP PANW SNDK STX WDC)
The artifact (experiment_results_86.41.md:109-111) states "of 34 raw events, 0 are Vertex; 10
carry Failed to fetch CIK map: 429; 7 carry Quant: SEC 429 rate-limit; 17 have no 429 in window."
10 and 7 are EVENT counts (mine: 20 and 14 LINES = 10 and 7 events); the residual "17 have no
429" is 34(lines) - 17(events) and describes NOTHING. There is no measured population of 17
un-attributed events: every distinct event carries the upstream cue.
Direction: this makes criterion 3's conclusion STRONGER, not weaker. But it is an unreproduced
number carried into a GENERATE artifact from a peer session ("verified independently by the
filing session") in a step whose own title says RE-DERIVE EVERY NUMBER, and whose contract
promised to re-derive. It is also the same normalisation class the artifact itself flags.
- "13 distinct tickers" REPRODUCES exactly.
- Ticker-count sanity: my 13 includes AAPL, whose 2026-08-06 event ended in
  "Analysis failed for AAPL: [RuntimeError] Step 'quant'" -- i.e. NOT a lite fallback. So the
  34/17 population is not co-extensive with the 9 lite fallbacks; the artifact does not claim it is.

(continued below)
