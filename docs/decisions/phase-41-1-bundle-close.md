# phase-41.1 -- Phase-29.9 P3 bundle close (ADR)

**Status:** Accepted (2026-05-23)
**Authors:** Main + Researcher + Q/A (Layer-3 harness MAS, phase-41.1 cycle 27)
**Decision class:** Trace-link / status-flip (Nygard ADR format -- mirror of phase-41.0)

---

## Context

Phase-29.9 was a P3 bundle authored during the phase-29 closure-planning sprint. It enumerated 10 residual housekeeping sub-items spanning vendor-released models (Gemini 3.1 / GPT-5.5), stress-test doctrine, agent-prompt refinements, sandbox-blocked tooling, and future tracking. During the phase-45.0 closure re-audit (cycle 12), the planner verdict-evaluated legacy phases and dropped phase-29.9 from `.claude/masterplan.json` phases array. The closure_roadmap.md §1 verdict table maps phase-29.9 -> phase-41.1 explicitly.

The phase-41.1 masterplan verification command:
```python
ps = [p for p in d['phases'] if p['id']=='phase-29.9']
assert (not ps) or ps[0]['status']=='done'
```
passes today (phase-29.9 absent). This is the trace-link closure semantic.

## Decision

**Close phase-41.1 as a trace-link closure** (mirror of phase-41.0 cycle 26). Ship:

1. Regression test (`backend/tests/test_phase_41_1_bundle_close.py`) locking both the masterplan invariant + the substantive caveat (phase-40.3 stress-test doctrine remains independently pending).
2. This ADR documenting the trace-link closure semantics + sub-item -> fold-destination mapping.
3. Cycle 27 harness_log block + masterplan status flip.

## Sub-item -> Fold-destination mapping (10 P3 items)

| # | Phase-29.9 P3 sub-item | Fold destination | Status today (2026-05-23) |
|---|---|---|---|
| 1 | Researcher multi-subagent fork doc | `.claude/agents/researcher.md:193` | **DONE** (engineered into agent prompt) |
| 2 | Q/A cycle-2-flow surfacing | `.claude/agents/qa.md:198` | **DONE** (engineered into agent prompt) |
| 3 | Gemini 3.1 model docs (released 2026-02-19) | adoption is owner-only | **VENDOR-RELEASED, ADOPTION DEFERRED** |
| 4 | GPT-5.5 model docs (released 2026-04-23) | adoption is owner-only | **VENDOR-RELEASED, ADOPTION DEFERRED** |
| 5 | Stress-test doctrine harness-free Opus 4.7 cycle | phase-40.3 | **pending** (independently tracked) |
| 6 | Scaffolding-pruning audit | phase-40.3 (folded) | **pending** (folded into 40.3) |
| 7 | Mythos Preview eval | sandbox-blocked / future | **TRACKED INDEPENDENTLY** |
| 8 | anthropic-docs MCP eval | sandbox-blocked / future | **TRACKED INDEPENDENTLY** |
| 9 | Browserbase eval | sandbox-blocked / future | **TRACKED INDEPENDENTLY** |
| 10 | futurelab eval | sandbox-blocked / future | **TRACKED INDEPENDENTLY** |

**Bucket roll-up:** 2/10 engineered-done; 2/10 vendor-released (adoption owner-only); 1/10 absorbed (sub-item 6 folded into 40.3); 1/10 independently pending (phase-40.3); 4/10 sandbox-blocked / future tracking.

## Status

ACCEPTED -- trace-link closed. The 10 P3 sub-items are accounted for across 4 buckets per the table above. Closing phase-41.1 does NOT close phase-40.3 or change vendor-adoption decisions.

## Consequences

**Positive:**
- Trace-link audit trail preserved (phase-29.9 -> phase-41.1 mapping documented + locked by regression test).
- phase-41.1 masterplan command exits 0 cleanly.
- Future auditors find this ADR + table.

**Caveats:**
- Future readers MUST NOT conclude "phase-41.1 DONE" means all 10 P3 sub-items are engineered-closed. phase-40.3 remains independently pending; Gemini 3.1 / GPT-5.5 adoption is owner-only; 4 sandbox-blocked items continue to be tracked.
- The regression test (specifically `test_phase_41_1_residual_40_3_remains_visible_separately`) enforces this.

**References:**
- Nygard 2011 (cognitect.com/blog/2011/11/15/documenting-architecture-decisions) -- ADR original spec.
- Anthropic harness-design blog 2026-03 -- stress-test doctrine that motivates the deferral.
- Mirror of `docs/decisions/phase-41-0-bundle-close.md` (cycle 26).
