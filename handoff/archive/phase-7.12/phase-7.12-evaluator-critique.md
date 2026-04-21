# Q/A Critique — phase-7 / 7.12 (Feature integration & IC evaluation)

**Q/A id:** qa_712_v1
**Date:** 2026-04-20
**Cycle:** 1 (first Q/A on 7.12; no prior critique)
**Verdict:** **PASS**

---

## 1. Protocol audit (5 items)

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Research gate passed, >=5 sources in full | PASS | `phase-7.12-research-brief.md` JSON envelope: `gate_passed: true`, `external_sources_read_in_full: 7`, `recency_scan_performed: true`. Three-variant queries documented. |
| 2 | Contract mtime < experiment_results mtime | PASS | contract=`Apr 20 00:47:32`, experiment_results=`Apr 20 00:49:46`. 2m14s delta; contract was frozen before generate. |
| 3 | Experiment-results verbatim + IC-sanity + advisory-handling | PASS | Contains verbatim `test -f` / `ls` output, IC math block (+1 / -1 / tie rank / summary), and an explicit "Advisory handling" section naming `adv_71`, `adv_72`, and the adv_73 non-implementation. |
| 4 | Log-last: harness_log last block is 7.11, NOT 7.12 yet | PASS | Last block in `handoff/harness_log.md` is `phase=7.11 result=PASS` at 00:45 UTC. 7.12 append will happen AFTER this Q/A PASS and BEFORE masterplan status flip. |
| 5 | First Q/A on 7.12 (not verdict-shopping) | PASS | No prior `phase-7.12-evaluator-critique*.md` exists. This is the sole Q/A on this step. |

**Protocol audit: 5/5 PASS.**

---

## 2. Deterministic checks (A-H)

| # | Check | Command | Exit | Result |
|---|---|---|---|---|
| A | features.py exists | `test -f backend/alt_data/features.py` | 0 | PASS |
| B | IC TSV present | `ls backend/backtest/experiments/results/alt_data_ic_*.tsv \| head -n 1` | 0 | PASS — `alt_data_ic_20260419T224855.tsv` |
| C | Regression | `pytest backend/tests/ --ignore=backend/tests/test_paper_trading_v2.py -q` | 0 | PASS — **152 passed, 1 skipped, 1 warning in 11.99s** (unchanged baseline) |
| D | ASCII decode | `file backend/alt_data/features.py` | 0 | PASS — "Python script text executable, ASCII text" |
| E | Scope | `git status --short` | n/a | PASS — only `backend/alt_data/` (new dir) + new TSV + handoff trio; all other M/D entries pre-existed this cycle |
| F | IC math | `python -c "from backend.alt_data.features import compute_ic, _spearman_rank; ..."` | 0 | PASS — `compute_ic([1,2,3,4,5],[5,4,3,2,1]) -> {'ic': -0.9999999999999998, 'n': 5}` (float-equivalent -1.0, acceptable); `_spearman_rank([1,2,2,3]) -> [1.0, 2.5, 2.5, 4.0]` (exact tie-aware) |
| G | TSV header | `head -1 alt_data_ic_*.tsv` | 0 | PASS — `feature_name\tticker\tstart\tend\twindow_days\tic\tic_std\tic_ir\tn\tnotes` (10 cols, exact) |
| H | `ast.parse` | `python -c "import ast; ast.parse(...)"` | 0 | PASS — sanity check only (not an immutable criterion) |

**Deterministic: 8/8 PASS.**

**Immutable criteria (from contract + masterplan):**
- `test -f backend/alt_data/features.py` -> PASS (A)
- `ls backend/backtest/experiments/results/alt_data_ic_*.tsv | head -n 1` -> PASS (B)

Both immutable criteria satisfied.

Minor note on F: `-0.9999999999999998` instead of an exact `-1.0` is a floating-point normalization artifact from the Pearson-on-ranks path. Numerically equivalent and agrees with the experiment-results summary within float tolerance. Not a contract breach (the IC value is not an immutable criterion; the file-existence checks are).

---

## 3. LLM judgment

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Advisory-aware notes | PASS | Source line 404: `note_parts = ["Senate only adv_71"]` (congress branch). Line 430: `notes += "; adv_72 cusip unresolved"` (13F unresolved-CUSIP branch). Lines 19-20 of docstring also reference these exactly. |
| 2 | FINRA absent (adv_73 gate) | PASS | `grep alt_finra_short_volume features.py` -> no matches. `run_ic_evaluation` has no branch for FINRA. Docstring (line 21) calls out `adv_73` explicitly as not-implemented. |
| 3 | Dry-run writes only the header | PASS | `wc -l alt_data_ic_20260419T224855.tsv` -> 1 line. Exactly the 10-column header, zero data rows. |
| 4 | No scipy dependency | PASS | `grep "^import scipy\|^from scipy" features.py` -> no matches. `_spearman_rank` + `_pearson` are pure-Python implementations. |
| 5 | No top-level network call | PASS | `resolve_cusip_to_ticker` is a `def` at line 178 (not invoked at import). `_fetch_forward_returns` is a `def` at line 286 (not invoked at import). `import requests` is nested INSIDE `resolve_cusip_to_ticker` (line 188) and wrapped in try/except (fail-open). No module-top network I/O. |
| 6 | Phase-7 closes 13/13 | PASS | 12 of 13 phase-7 substeps were `status=done` before this cycle; flipping 7.12 -> done closes phase-7. |
| 7 | Contract alignment | PASS | Module shipped matches the contract's hypothesis verbatim: aggregate_congress + aggregate_13f + resolve_cusip + compute_ic + summarize_ic + _fetch_forward_returns + run_ic_evaluation + _cli. Dry-run mode writes header-only TSV as promised. |
| 8 | Scope honesty | PASS | experiment_results.md §"Known caveats" discloses 6 real scope bounds (0-row dry-run, OpenFIGI unauth tier, pure-Python Spearman perf, yfinance dependency, no significance test, live IC expectations). No overclaim. |

**LLM judgment: 8/8 PASS.**

---

## 4. Violations

**None.** All immutable criteria met. No contract drift. No verdict-shopping. No rubber-stamp concerns (the regression run is reproducible, math is exact on known test cases, advisories line-read confirmed in source, dry-run output empirically matches the header-only claim).

`violated_criteria: []`
`violation_details: []`
`certified_fallback: false`

---

## 5. Decision

**PASS (qa_712_v1).**

Main may proceed to:
1. Append the phase-7.12 block to `handoff/harness_log.md` (log-last rule).
2. Flip `.claude/masterplan.json` phase-7/7.12 `pending -> done`.
3. Phase-7 closes (13/13 substeps done).

`checks_run: [protocol_audit, research_gate, contract_mtime, results_verbatim, advisory_coverage, deterministic_A_through_H, regression_152_1, scope_diff, llm_judgment_8_of_8]`
