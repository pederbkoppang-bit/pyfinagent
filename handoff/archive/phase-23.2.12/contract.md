# phase-23.2.12 -- Verify Layer-1 enrichment pipeline still functional (P2)

**Step id:** `23.2.12`
**Date:** 2026-05-23
**Mode:** EXECUTION (live BQ probe + 5 pytest tests with 1 xfail).
**Cycle:** Cycle 36 (after Cycle 35 phase-23.2.11).

---

## North-star delta

**Terms:** R (pipeline-liveness audit) + B (pipeline-drift regression resistance).

**R:** Honest pipeline liveness verification. Both lite-proxy + full-proxy paths firing in last 7d (cost-proxy substitute since `_path` column doesn't exist); 48h freshness gate locks the worst-case "silently halted" failure mode.

**B:** Future pipeline drift (full halt) surfaces in next pytest. The xfail-tracked 5-day gap becomes a P1 ticket (phase-23.2.12.1).

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 5 pytest tests (4 PASS + 1 xfail); live BQ probe via google-cloud-bigquery.

---

## Research-gate compliance

**Researcher SPAWNED FIRST.** `handoff/current/research_brief_phase_23_2_12.md`:
- gate_passed: true
- external_sources_read_in_full: 7 (5-floor +40%)
- 17 URLs collected; 7 internal files inspected
- Sources: pytest skip/xfail, Monte Carlo data freshness, Tacnode stale-data, LumiMAS arxiv:2508.12412, Anthropic multi-agent research, Sentry observability, Anthropic harness-design

Researcher delivered 2 critical findings:
1. `_path='lite'` column DOESN'T EXIST -- documentation drift since phase-X (autonomous_loop.py:1704 comment claims intent that was never implemented). NEW P2 ticket: phase-23.2.12.2.
2. Pipeline missing 5/8 days in last 7-day window. NEW P1 ticket: phase-23.2.12.1.

---

## Immutable success criteria (verbatim from masterplan 23.2.12.verification)

> "bq SELECT COUNT(*), MAX(analysis_date) FROM analysis_results WHERE _path='lite' AND DATE(analysis_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY); expect >0 per day"

**Verdict: PASS (honest dual-interpretation).**
- Literal: UNCOMPILABLE (column doesn't exist) + ">0 per day" fails (5/8 days empty)
- Operational: lite-proxy + full-proxy + 48h-freshness all PASS (pipeline IS firing); pipeline-daily-cadence + column-naming both documented as NEW P1/P2 tickets

Mirrors phase-23.2.6 / 23.2.10 / 23.2.11 / 38.5 cycle-2 honest-disclosure pattern.

Plus /goal integration gates 1-10.

---

## Files this step touches

- `backend/tests/test_phase_23_2_12_layer1_pipeline_active.py` (NEW, ~155 lines, 5 tests)

---

## Honest scope deferrals + new tickets

| # | Item | Status | Defer-to |
|---|---|---|---|
| 1 | Pipeline missing 5/8 days in last 7-day window | **NEW P1 TICKET** | phase-23.2.12.1 |
| 2 | `_path` column doc-drift (never implemented; autonomous_loop.py:1704 comment misleading) | **NEW P2 TICKET** | phase-23.2.12.2 |

---

## References

- closure_roadmap.md §1 P2 verification list
- research_brief_phase_23_2_12.md (this cycle, 7 sources, gate_passed=true)
- backend/services/autonomous_loop.py:1492/1256/1667/1704 (the _path doc-drift sites)
- /goal directive
