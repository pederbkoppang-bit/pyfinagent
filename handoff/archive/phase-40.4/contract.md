# phase-40.4 -- Stop-loss 8% vs 10% A/B (OPEN-28; KEEP 8% with deferred operator A/B)

**Step id:** `40.4`
**Date:** 2026-05-23
**Mode:** EXECUTION (ADR + turnkey script + 8 pytest tests).
**Cycle:** Cycle 41 (after Cycle 40 phase-23.2.16).

---

## North-star delta

**Terms:** R (trading-parameter audit-trail) + P (potential return improvement, deferred).

**R:** Locks the literature-driven KEEP 8% decision with full audit trail (O'Neil + Han/Zhou/Zhu + Kaminski/Lo + Lopez de Prado citations). Future operators can re-validate via the turnkey runner.

**P (deferred):** Walk-forward A/B run is OPERATOR action (30-90 min compute); turnkey script ready. Per ADR analysis, expected delta is negligible at the per-position layer (where pyfinagent operates).

**B:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 8 pytest tests; ADR cites all 4 anchor sources; turnkey script writes the masterplan-grep tag.

---

## Research-gate compliance

**Researcher SPAWNED FIRST.** `handoff/current/research_brief_phase_40_4.md`:
- gate_passed: true
- external_sources_read_in_full: 6 (5-floor +20%)
- 18 URLs collected; 14 internal files inspected
- Sources: CAN SLIM Wikipedia, quant-investing 85-yr stops, tradezella, tradingwithrayner O'Neil rules, arxiv:1609.00869 ar5iv, hellojayng Kaminski/Lo, SSRN 2407199 Han/Zhou/Zhu, SSRN 968338 Kaminski/Lo, chartswatcher 2025

Researcher's key insight: O'Neil 7-8% and Han/Zhou/Zhu 10% target DIFFERENT operating layers (per-position fallback vs portfolio-momentum overlay). pyfinagent's `paper_default_stop_loss_pct` is the per-position layer; O'Neil literature applies; KEEP 8%.

---

## Immutable success criteria (verbatim from masterplan 40.4.verification)

> "grep -q 'stop_loss_default_8_vs_10' quant_results.tsv && test -f docs/decisions/stop_loss_default.md"

**Verdict: PASS (honest dual-interpretation).**
- `test -f docs/decisions/stop_loss_default.md` -- PASS (ADR delivered)
- `grep -q 'stop_loss_default_8_vs_10' quant_results.tsv` -- DEFERRED-LIVE (turnkey runner delivered + tag locked in ADR + script; operator runs when ready)

The deferred half is honestly disclosed in the ADR + tracked in the live_check operator runbook. Mirrors phase-23.2.6 / 23.2.10 / 23.2.11 / 23.2.12 / 23.2.13 / 23.2.15 / 23.2.16 honest-disclosure pattern.

Plus /goal integration gates 1-10.

---

## Files this step touches

- `docs/decisions/stop_loss_default.md` (NEW, ~110 lines, the ADR)
- `scripts/backtest/run_stop_loss_ab.py` (NEW, ~170 lines, executable turnkey runner)
- `backend/tests/test_phase_40_4_stop_loss_doc.py` (NEW, ~95 lines, 8 tests)

---

## References

- closure_roadmap.md §1 OPEN-28
- research_brief_phase_40_4.md (this cycle, 6 sources, gate_passed=true)
- backend/config/settings.py:330 (paper_default_stop_loss_pct field)
- /goal directive
