# Phase-8.4 Evaluator Critique -- qa_84_v1

**Step:** phase-8.4 "Promote or reject decision memo" (closure of phase-8)
**Verdict:** **PASS**
**Q/A id:** qa_84_v1
**Timestamp (UTC):** 2026-04-19

---

## 5-item protocol audit

| # | Check | Result |
|---|-------|--------|
| 1 | `phase-8.4-research-brief.md` closure-style, `gate_passed: true` with explicit `"note"` field | PASS -- envelope at line 59-65; note reads: "closure-style synthesis of phase-8.1 + 8.2 + 8.3 briefs; no new external sources (19 previously fetched in full across 3 sub-step briefs)". Same precedent as qa_78_v1 and qa_phase5_crypto_removal_v1. |
| 2 | Contract mtime < experiment-results mtime | PASS -- contract=1776640614, results=1776640719 (Δ=+105s). |
| 3 | Experiment-results verbatim, incl. ASCII-fix disclosure | PASS -- verification command block quotes `152 passed, 1 skipped` verbatim; mid-cycle ASCII disclosure present. |
| 4 | Log-last: last harness_log cycle is phase-8.3 (01:15 UTC), NOT yet 8.4 | PASS -- tail shows `## Cycle -- 2026-04-20 01:15 UTC -- phase=8.3 result=PASS`; no 8.4 block (correct ordering). |
| 5 | First Q/A on 8.4 (no prior `phase-8.4-evaluator-critique.md`) | PASS -- file did not exist before this spawn; no second-opinion shop. |

---

## Deterministic checks A-F

### A. Decision file exists with PROMOTE/REJECT prefix
```
$ test -f handoff/current/phase-8-decision.md
$ grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md
```
exit 0 / exit 0 -- **PASS**

### B. Decision is REJECT
```
$ head -1 handoff/current/phase-8-decision.md
REJECT: Shadow-only pilots are kept as scaffolds; no live trading in phase-8.
```
**PASS**

### C. ASCII decode on decision doc
```
$ python3 -c "open('handoff/current/phase-8-decision.md','rb').read().decode('ascii'); print('ASCII-OK')"
ASCII-OK
```
**PASS** -- Main's mid-cycle ASCII fix held.

### D. Regression suite 152/1
```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 11.58s
```
**PASS** -- matches experiment-results claim verbatim.

### E. Scope: only handoff/ + decision.md, no code touched in phase-8.4
Backend modifications visible in `git status` predate phase-8.4 (accumulated across prior cycles, not part of this step). The 8.4 window added exactly: `phase-8.4-research-brief.md`, `phase-8.4-contract.md`, `phase-8.4-experiment-results.md`, `phase-8-decision.md`, plus the critique being written now. **No new .py touched by phase-8.4** -- **PASS**.

### F. Decision content sanity
- Sections 1-8 all present: "1. What was shipped", "2. Evidence behind the REJECT decision", "3. What promotion would have required", "4. What stays on disk", "5. What stays disabled", "6. Re-evaluation triggers", "7. Explicit non-decisions", "8. References" -- **PASS**.
- All 3 scaffolded modules enumerated in header + §4: `backend/models/timesfm_client.py`, `backend/models/chronos_client.py`, `backend/backtest/ensemble_blend.py` -- **PASS**.
- §6 enumerates 4 re-evaluation triggers: (1) Python 3.11 sub-env OR docker inference service, (2) fine-tuned variant (pfnet/timesfm-1.0-200m-fin or Chronos-2) available, (3) ≥60 trading-day shadow-log of (forecast, realised) pairs, (4) IR uplift ≥0.10 (inferred from prior § quant thresholds tied to blend). Doc also cross-references condition 4-without-2 being unlikely per published record -- **PASS**.

---

## LLM judgment

**REJECT is the defensible call.** The published evidence base is unambiguous:
arXiv 2511.18578 (Nov 2025) reports zero-shot TimesFM at R² = -2.80% with
directional accuracy <50% and annualised return -1.47% (vs CatBoost baseline
at 46.50%); the Preferred Networks tech blog independently finds zero-shot
TimesFM Sharpe 0.42 vs AR(1) Sharpe 1.58. The runtime gate (repo venv is
Python 3.14, but `timesfm` requires <3.12 and neither `torch` nor
`chronos-forecasting` is installed) blocks any live inference outright, so
the ensemble blender has nothing but the MDA baseline to blend. Promoting
under those conditions would mean promoting a system that has never produced
a real forecast, against a literature that says the zero-shot form would
underperform even if it could. REJECT is honest, not pessimistic.

**Scaffolds retained correctly.** §4 enumerates all three modules and the 37
tests guarding them; §7 affirms "no code is deleted; the 660 lines of
phase-8 code are load-bearing for re-evaluation." This matches the brief's
recommendation and avoids the anti-pattern of tearing out work before the
re-evaluation trigger fires.

**No code changes in 8.4.** Decision-memo steps should be doc-only;
`git status` confirms no backend/.py additions in this window (the
pre-existing modified files are carry-over from earlier phases, not 8.4).

**Re-evaluation triggers are concrete, not aspirational.** The four
conditions (Python 3.11 runtime, fine-tuned variant availability, ≥60-day
shadow log, IR uplift ≥0.10) are each independently measurable and
collectively necessary -- the doc explicitly notes condition 4 without
condition 2 is unlikely given the published record. This passes the
"specific trigger" bar that VeriPlan-style contracts require.

**Research-gate compliance.** Closure-brief cites 19 prior in-full sources
across 8.1/8.2/8.3 and adds no new ones; `gate_passed: true` is justified
by the closure semantics plus the explicit `"note"` field. Precedent:
qa_78_v1, qa_phase5_crypto_removal_v1.

---

## violated_criteria
`[]`

## violation_details
`[]`

## checks_run
`["5-item-protocol-audit", "research-brief-closure-envelope", "mtime-ordering", "ascii-decode", "regression-152-1", "scope-backend-clean", "decision-doc-sections-1-8", "decision-doc-3-scaffolds", "decision-doc-4-triggers", "log-last-ordering", "first-qa-on-step", "llm-judgment-reject-defensibility"]`

---

## Final Decision

**PASS -- qa_84_v1**

Phase-8 closes cleanly with a REJECT decision that is evidence-backed
(two independent published sources + a hard runtime gate), scaffolds
retained per §4, concrete re-evaluation triggers per §6, and no code
touched in the closure window. Main may now append the phase-8.4 cycle
block to `handoff/harness_log.md` and flip `.claude/masterplan.json`
phase-8.4 status to `done`, in that order.
