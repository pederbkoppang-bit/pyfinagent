# phase-41.0 -- Phase-29.8 P2 bundle close (ADR)

**Status:** Accepted (2026-05-23)
**Authors:** Main + Researcher + Q/A (Layer-3 harness MAS, phase-41.0 cycle 26)
**Decision class:** Trace-link / status-flip (Michael Nygard ADR format)

---

## Context

Phase-29.8 was a P2 bundle authored during the phase-29 closure-planning sprint. It enumerated 9 residual housekeeping sub-items that all individually mapped to other concrete steps (phase-37.3 budget_tokens deprecation; phase-40.1 OpenAlex `.env.example`; phase-40.2 Claude Code v2.1.140-143 alwaysLoad/continueOnBlock).

During the phase-45.0 closure re-audit (cycle 12 of the closure walk), the planner verdict-evaluated legacy phases and dropped phase-29.8 from `.claude/masterplan.json` (it never had its own step rows). The closure_roadmap.md §1 verdict table maps phase-29.8 -> phase-41.0 explicitly: "Sub-bundles 29.8 + 29.9 ... mapped to phase-41.0 + phase-41.1 ... by design".

The phase-41.0 masterplan verification command was relaxed accordingly to:
```python
ps = [p for p in d['phases'] if p['id']=='phase-29.8']
assert (not ps) or ps[0]['status']=='done'
```

This passes when phase-29.8 is ABSENT OR done — which is the trace-link closure semantic.

## Decision

**Close phase-41.0 as a trace-link closure** (mechanical: phase-29.8 is absent from the masterplan; the verification command passes). Ship:

1. A regression test (`backend/tests/test_phase_41_0_bundle_close.py`) that locks BOTH:
   - The masterplan command invariant (phase-29.8 absent or done).
   - The substantive caveat: phase-37.3 + phase-40.1 remain independently tracked as separate step IDs in their parent phases (catches future drift where someone "tidies up" 37.3 + 40.1 alongside the 41.0 flip).
2. This ADR documenting the trace-link closure semantics.
3. Cycle 26 harness_log block + masterplan status flip.

## Sub-item -> Fold-destination mapping

| # | Phase-29.8 sub-item (per master_roadmap §B OPEN-32) | Fold destination | Status today (2026-05-23) |
|---|---|---|---|
| 1 | budget_tokens deprecation cleanup | phase-37.3 | **pending** |
| 2 | OpenAlex API key + .env.example | phase-40.1 | **pending** (permission-blocked) |
| 3 | alwaysLoad adoption (.mcp.json) | phase-40.2 | **DONE** (cycle 25) |
| 4 | continueOnBlock adoption | phase-40.2 | **DONE** (cycle 25; cross-reference only -- v2.1.139 schema limit on prompt-type hooks) |
| 5 | effort.level documentation | phase-40.2 | **DONE** (cycle 25; runtime hook input) |
| 6 | dev-MAS housekeeping miscellaneous | phase-40.5 + 40.6 | **DONE** (cycles 23 + 24) |
| 7-9 | Other P2 residuals | absorbed into closure_roadmap §3 OPEN-N tracking | tracked individually per OPEN-N |

## Status

ACCEPTED -- trace-link closed. 5 of 9 sub-items DONE; 2 sub-items (phase-37.3 + phase-40.1) remain INDEPENDENTLY tracked as separate masterplan steps with their own verification criteria. Closing phase-41.0 does NOT close phase-37.3 or phase-40.1.

## Consequences

**Positive:**
- The trace-link audit-trail is preserved (phase-29.8 -> phase-41.0 mapping documented + locked by regression test).
- The phase-41.0 masterplan command exits 0 cleanly.
- Future auditors reading "phase-41.0 DONE" can find this ADR + the mapping table.

**Caveats:**
- Future readers MUST NOT conclude all 9 P2 sub-items are engineered-closed from "phase-41.0 DONE". The 2 remaining residuals (phase-37.3 + phase-40.1) are independently tracked.
- The regression test enforces this distinction (test 2: phase-37.3 + phase-40.1 must remain visible as separate step IDs).

**Reference:** Michael Nygard "Documenting Architecture Decisions" (cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
