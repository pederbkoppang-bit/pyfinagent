# Contract — phase-43.0 (Production-Ready DoD audit)

**Date:** 2026-06-01. **Tier:** moderate. **Step:** phase-43.0 (P1).

## N* delta (N* = Profit − Risk − Burn)

**Risk↓** (governance/honesty): a refreshed, evidence-backed go/no-go audit prevents a
premature PRODUCTION_READY declaration (a watermelon). No P/B delta. $0 (read-only checks).

## Research-gate summary

`researcher` ran FIRST (gate **PASSED**: 5 external sources read in full, 14 URLs,
recency scan, 12 internal files). Brief: `handoff/current/research_brief.md`. The DoD gate
is **26 criteria** (14 backend = `master_roadmap_to_production.md §6`; 12 UX =
`frontend_ux_master_design.md §6`). External grounding: production-readiness is a
scored/tri-state gate where "done" = executed-and-evidenced (Google ML Test Score); open
criteria are legitimate with an agreed remediation plan (Google SRE PRR); honest red beats
padded green (watermelon literature).

**Current tally (deterministic checks ran 2026-06-01):** Backend **8/14 PASS**
(DoD-3,4,8,10,11,12,13,14 — DoD-14 newly closes, OWASP 10/10). 5 LIVE-BLOCKED
(DoD-2,5,6,7,9 — need live cycles = operator LLM spend). 1 OPERATOR-GATED (DoD-1 —
autoresearch/ablation cron exit=1, owner-gated phase-39.1). UX **0/12** (need phase-44.x
build + Playwright/Lighthouse behind the NextAuth wall = operator-gated). Watermelon
finding: full backend run = 16 failed / 711 passed, all ENVIRONMENT-COUPLED (live-BQ
freshness probes, a moved fixture-doc ×7, canary/wiring) — NOT logic regressions; must be
surfaced honestly.

## Honest framing (this step CANNOT fully close autonomously — by design)

Immutable criterion #1 (`all_14_DoD_criteria_PASS`) is NOT met (8/14) and #4
(`operator_approval_recorded`) needs the REMOTE operator to type "PRODUCTION_READY:
APPROVED". Per Message-B step 2 ("run the DoD audit; close what's autonomously closable;
honestly mark the live-blocked criteria"), the deliverable is the honest refreshed audit
`production_ready_audit_2026-06-01.md`. The step stays **pending** (verdict =
NOT_PRODUCTION_READY); the operator asks go to `cycle_block_summary.md`. Then the run
continues to the autonomously-closable phase-53.x.

## Immutable success criteria — VERBATIM from masterplan phase-43.0 (do NOT edit)

1. all_14_DoD_criteria_PASS
2. audit_file_carries_verbatim_evidence_per_criterion
3. qa_confirms_no_silent_drops
4. operator_approval_recorded_for_PRODUCTION_READY_declaration

(`verification.live_check`: production_ready_audit_<date>.md is the deliverable; live
verification = read it.)

**Achievable this cycle:** #2 (verbatim per-criterion evidence) + #3 (Q/A confirms no
silent drops). **NOT achievable autonomously:** #1 (8/14, 5 live-blocked + 1 operator-gated
+ UX 0/12) + #4 (operator away). The audit honestly records all of this.

## Plan steps

1. Write `handoff/current/production_ready_audit_2026-06-01.md`: per-criterion (all 26)
   verbatim definition + current state (PASS / LIVE-BLOCKED / OPERATOR-GATED) + the exact
   evidence/command. Surface the 16 env-coupled test failures honestly (quarantine note,
   not a claimed-green suite). Headline verdict: NOT_PRODUCTION_READY (8/14 backend,
   0/12 UX). Re-probe DoD-5 freshness on a stable backend (the prior read was SIGTERM-tainted).
2. Append the operator ask to `cycle_block_summary.md` (approve after live cycles; the 5
   live-blocked + UX-DoD need operator LLM spend + visual confirmation behind NextAuth).
3. Fresh qa: confirm criteria 2+3 (verbatim evidence + no silent drops) + that the
   NOT_PRODUCTION_READY verdict is honest (not a watermelon and not a false-fail).
4. Append `harness_log.md` (result=CONDITIONAL — audit complete + honest; step gated on
   operator approval + live cycles). Do NOT flip 43.0 to done.

## Guardrails

- $0 / read-only (no live cycles, no LLM spend, no BQ writes). Honest classification — no
  watermelon (do not claim all-green from the collect-only count; document the 16 env-coupled
  failures). Do NOT seek/forge the operator approval. No emoji; ASCII.

## References

`handoff/current/research_brief.md`; `master_roadmap_to_production.md §6` (backend DoD);
`frontend_ux_master_design.md §6` (UX DoD); `production_ready_audit_2026-05-28.md` (prior);
Google ML Test Score / SRE PRR / watermelon-reporting / Bailey-LdP DSR.
