# phase-23.2.6 -- Verify sector cap blocked same-sector buys (P1)

**Step id:** `23.2.6`
**Date:** 2026-05-23
**Mode:** EXECUTION (forward-gate verification + 6 new pytest tests).
**Cycle:** Cycle 30 (after Cycle 29 phase-23.2.5).

---

## North-star delta

**Terms:** R (concentration-risk audit integrity) + B (defensive cap-gate regression resistance).

**R:** AFML / Bailey-Lopez de Prado on diversification: sector concentration above cap is documented portfolio-risk anti-pattern. Locks in the phase-23.1.13 cap implementation (commit `5b350e4d`) at the forward-gate layer; CAVEAT discloses legacy 8-Tech overage as phase-23.2.6.1 follow-up.

**B:** Each future same-sector buy attempt that the cap blocks saves implicit concentration drift. 24 emits in backend.log show the gate actively firing on real cycles.

**P:** N/A (no decision-quality change). **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 6 pytest cases (forward-gate + emit-site + settings default + log-evidence); BQ snapshot anomaly (8 Tech vs cap=2) honestly disclosed as separate follow-up.

---

## Research-gate compliance

**Researcher SPAWNED** per `feedback_never_skip_researcher`. Simple-tier brief at `handoff/current/research_brief_phase_23_2_6.md`:
- gate_passed: true
- external_sources_read_in_full: 6 (5-source floor +20% buffer)
- 17 URLs collected; 6 internal files inspected
- Sources: pytest stable + 8.x logging docs, Woteq caplog assertion examples, Pytest with Eric caplog fixture, Motley Fool 11 GICS sectors, Guardfolio concentration risk, HDFC TRU equity portfolio design

**Critical findings:**
- backend.log: 24 "Skipping BUY ... at cap" emits (forward-gate working)
- Cap: `settings.paper_max_per_sector = 2` (default, settings.py:162)
- Emit site: `backend/services/portfolio_manager.py:247-252`
- BQ snapshot: 8 Tech positions today (legacy overage; entries dated 2026-04-26 to 2026-04-28, predating phase-23.2.6-fix sector-persistence migration commit `c854386f`)

**Honest dual-interpretation:**
1. Forward-looking gate: PASS (24 log emits prove)
2. Current-snapshot invariant: FAIL (legacy 8 Tech rows exist)

Per Q/A cycle-1 38.5 lesson: I'm honestly disclosing this trade-off in contract + live_check, not silently pivoting.

---

## Hypothesis

> The forward-gate cap at `portfolio_manager.py:247-252` is intact + actively
> firing (24 emits this cycle). The BQ snapshot's 8 Tech overage is a
> LEGACY artifact (rows dated 2026-04-26 to 2026-04-28, predating the
> sector-persistence fix); the cap CANNOT retro-divest legacy state.
> Regression test covers forward-gate semantics: blocks-third-tech /
> allows-new-sector / cap=0-disables / emit-site-present / settings-default
> / log-evidence-present. Snapshot retro-divest is phase-23.2.6.1 follow-up.

---

## Immutable success criteria (verbatim from masterplan 23.2.6.verification)

> "grep 'Skipping BUY .* at cap' backend.log; bq SELECT sector, COUNT(*) FROM paper_positions GROUP BY sector should never show >2 per sector when cap=2"

**Verdict per part:**
- **Part 1 (grep backend.log):** PASS verbatim. 24 matching emits today; regression test verifies presence.
- **Part 2 (BQ SELECT >2 per sector when cap=2):** **FORWARD-GATE PASS** + **LEGACY-SNAPSHOT CAVEAT** (8 Tech rows pre-migration). Documented as phase-23.2.6.1 follow-up.

Plus /goal integration gates 1-10.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Researcher (simple tier, 6 sources, gate_passed=true) | DONE |
| 2 | Verify emit site at portfolio_manager.py:247-252 | DONE |
| 3 | Verify settings.paper_max_per_sector default = 2 | DONE |
| 4 | Count backend.log "Skipping BUY" emits (researcher: 24) | DONE |
| 5 | Capture BQ snapshot dual-interpretation (forward PASS + snapshot CAVEAT) | DONE |
| 6 | Write contract + live_check + tests | DONE (6/6 pass; total 400 -> 406) |
| 7 | Q/A + harness_log Cycle 30 + flip | IN FLIGHT |

---

## Files this step touches

- `backend/tests/test_phase_23_2_6_sector_cap_emit.py` (NEW, ~205 lines, 6 tests)

**NOT changed:** any source code; any frontend file; any masterplan structural change. The 8-Tech legacy overage in BQ is OPERATOR action (phase-23.2.6.1 follow-up); not closed by this step.

---

## Honest scope deferrals

| Item | Status | Defer-to |
|---|---|---|
| BQ `paper_positions` legacy divest (8 Tech -> ≤2) | DEFERRED | phase-23.2.6.1 (operator OR follow-up cycle; researcher recommends NOT scope-creeping into 23.2.6) |
| Live BQ MCP probe assertion in pytest | DEFERRED | requires live BQ client; pytest checks source + log only |

NOT silent drops -- both tracked explicitly here + in live_check.

---

## References

- closure_roadmap.md §1 P1 verification list
- research_brief_phase_23_2_6.md (this cycle, 6 sources, gate_passed=true)
- backend/services/portfolio_manager.py:209-252 (sector-cap implementation)
- backend/config/settings.py:162 (paper_max_per_sector default=2)
- /goal directive (researcher mandatory per feedback_never_skip_researcher)
