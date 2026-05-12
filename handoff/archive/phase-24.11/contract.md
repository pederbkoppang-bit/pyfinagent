# Sprint Contract — phase-24.11 — Frontend↔Backend Wiring

**Cycle:** phase-24 cycle 10
**Date:** 2026-05-12
**Step ID:** 24.11
**Priority:** P2

## Research-gate
`gate_passed: true` (tier=moderate). 5 sources: Next.js data fetching, React useEffect guidance, TanStack Query, FastAPI client generation, pydantic-to-typescript.

```json
{"tier":"moderate","external_sources_read_in_full":5,"snippet_only_sources":7,"urls_collected":12,"recency_scan_performed":true,"internal_files_inspected":7,"gate_passed":true}
```

## Hypothesis
Wiring mostly clean. Orphan: /paper-trading/learnings has UI but no backend. Type drift minimal but non-zero.

**Researcher verdict: CONFIRMED.**
- Orphan: `frontend/src/app/paper-trading/learnings/page.tsx:6-9` says "Live data hookup lands in a follow-up backend step". `VirtualFundLearnings` component renders only empty states.
- 119 backend routes total in main.py:379-413; api.ts exposes 83 functions
- Type drift at 2 points: `ReportSummary.analysis_date` (Pydantic datetime / TS string — benign), `SynthesisReport` enrichment fields (Pydantic Optional[dict] / TS fully-typed sub-interfaces — TS more precise; codegen-from-Pydantic would regress)
- 7 `unknown`/`Record<string, unknown>` return types in api.ts (`getReport`, `getSignal`, `getMacroIndicators`, etc.) — escape hatches defeating type safety
- `GoLiveGate` exported from component, not types.ts; sovereign interfaces inline in api.ts:568-709

## Success criteria (verbatim)
1. findings_md_exists
2-10. common pack
11. findings_audits_type_drift_between_pydantic_and_typescript
12. findings_audits_every_page_to_endpoint_mapping
13. findings_audits_learnings_page_backend_hookup_gap
14. findings_audits_auth_header_propagation

**Verifier:** `python3 tests/verify_phase_24_11.py`

## Plan
1. Findings
2. Results
3. Q/A
4. Cycle 51 log
5. live_check_24.11.md
6. Flip
