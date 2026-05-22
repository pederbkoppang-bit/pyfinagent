# phase-38.3 -- Startup banner logs deep_think_model

**Step id:** `phase-38.3`
**Date:** 2026-05-22
**Mode:** EXECUTION (small observability addition; ~25 LOC).
**Cycle:** Cycle 20 (after Cycle 19 phase-37.4).

---

## North-star delta

**Terms:** B (defensive Burn-protection) + R (audit-trail).

**B:** Catches future model-default regressions at boot time. Without this banner, the phase-34.1e claude-opus-4-7 default silently regressed cycles for days. With the banner, a fresh checkout / restart with a wrong env override produces a startup WARNING that an operator can grep / alert on within seconds.

**R:** SR-11-7 model-routing observability + OWASP LLM v2 audit-trail compliance. Routing-affecting config now logged at startup (per 12-Factor §XI Logs + Portkey 2026 LLM-observability guide).

**P:** N/A (no trading-logic change).

**Caltech arxiv:2502.15800 discount:** N/A (no decision-quality change).

**How measured:** `grep -c "phase-38.3 model routing" backend.log` after next restart = 1 per startup. WARNING fires when deep_think_model is non-Gemini.

---

## Research-gate compliance

**Researcher SPAWNED** per `feedback_never_skip_researcher`. Simple-tier brief at `handoff/current/research_brief_phase_38_3.md`:
- gate_passed: true
- external_sources_read_in_full: 7 (5-source floor +40% buffer)
- 5 internal files inspected
- 3-variant queries + recency scan performed
- 6 of the 7 sources are 2026-current (12-Factor canonical, OWASP LLM 2026, SR-11-7 ModelOp, Portkey 2026, Honeycomb modern-OTel, Medium audit-trail)

Researcher delivered the exact log-line shape (provider-detect classifier + INFO + WARNING) -- applied verbatim with one comment refinement citing closure_roadmap §3 OPEN-12.

---

## Hypothesis

> If we insert a parallel deep-think tier banner block in `backend/main.py`
> immediately after the existing standard-tier banner (line 152), using
> the SAME provider-detect + INFO + WARNING pattern, THEN every backend
> restart will emit two greppable lines (`phase-31.1 model routing` +
> `phase-38.3 model routing`), and any future model-default regression
> will surface as a startup WARNING the operator can spot in
> backend.log immediately.

---

## Immutable success criteria (verbatim from masterplan 38.3.verification)

1. `backend_main_py_emits_both_standard_and_deep_think_banners` -- PASS via test 1 + test 4 (greppable both prefixes).
2. `fresh_restart_shows_both_lines` -- PASS (code-path) + DEFERRED-LIVE (next backend restart; operator runbook in live_check).

Plus /goal integration gates 1-10.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Researcher (simple tier, 7 sources, gate_passed=true) | DONE |
| 2 | Write contract | IN FLIGHT |
| 3 | Edit backend/main.py:152 -- insert parallel deep-think banner | DONE |
| 4 | Write `backend/tests/test_phase_38_3_deep_think_banner.py` (5 tests) | DONE |
| 5 | pytest verify (count >= 331; achieved 336) | DONE |
| 6 | live_check + Q/A + harness_log Cycle 20 + flip | IN FLIGHT |

---

## Files this step touches

- `backend/main.py` +25 lines (after line 152; new deep-think provider-detect block)
- `backend/tests/test_phase_38_3_deep_think_banner.py` (NEW, ~70 lines, 5 tests)

**NOT changed:** any frontend file; any config / agents / services file; the existing standard-tier banner at line 127-152.

---

## References

- closure_roadmap.md §3 OPEN-12 (the observability gap diagnosis)
- research_brief_phase_38_3.md (this cycle's researcher output, 7 sources)
- backend/main.py:127-152 (the reference pattern this step mirrors)
- backend/config/settings.py:30 (the Field this step's banner reads from; phase-37.2 = gemini-2.5-pro)
- handoff/archive/phase-34.1/ (original observability-gap mention)
- /goal directive (10 integration gates; researcher mandatory per feedback_never_skip_researcher)
