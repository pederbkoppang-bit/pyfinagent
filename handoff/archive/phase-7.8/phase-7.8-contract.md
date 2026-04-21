# Sprint Contract — phase-7 / 7.8 (Satellite/geospatial proxies — deferred)

**Step id:** 7.8 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** simple (closure)

## Hypothesis

Criterion `grep -q 'Phase 8' docs/compliance/alt-data.md` is already satisfied by phase-7.0's compliance doc, which carries a dedicated Section 8 "Open Items / Deferred" explicitly calling out "Satellite / geospatial proxies -- DEFERRED to Phase 8" with budget rationale (Planet Labs, Maxar, Spire enterprise licenses). Closure cycle verifies + flips status.

## Immutable criterion (verbatim)

- `grep -q 'Phase 8' docs/compliance/alt-data.md`

## Plan

1. Re-run the grep to confirm current state.
2. Write experiment results.
3. Q/A.
4. Log-last + flip.

## Out of scope

- No new doc sections (the Section-8 deferral from 7.0 is already sufficient).
- No satellite vendor integration (deferred to Phase 8 by construction).

## References

- `docs/compliance/alt-data.md` Section 8
- `handoff/archive/phase-7.0-*` (Q/A qa_70_v1 explicitly validated this token via the bonus check)
- `handoff/current/phase-7.8-research-brief.md`
