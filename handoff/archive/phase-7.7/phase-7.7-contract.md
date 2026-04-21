# Sprint Contract — phase-7 / 7.7 (Revelio license doc)

**Step id:** 7.7 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Research-gate summary

6 sources in full (Revelio data page, WRDS vendor page, Revelio research tier, U of T library notice, Hyperstart DPA guide, Coresignal industry comparison), 16 URLs, three-variant queries, recency scan 2024–2026. `gate_passed: true`. Brief at `handoff/current/phase-7.7-research-brief.md`.

## Hypothesis

Doc-only step. Write `docs/compliance/revelio-license.md` following the 8-section template established by `reddit-license.md`. Revelio-specific content: MSA-based contract (not OAuth), Sentiment + Workforce Dynamics datasets, batch delivery (S3/Snowflake), GDPR Art. 28 DPA pending.

## Immutable criterion

- `test -f docs/compliance/revelio-license.md`

## Plan

1. Write `docs/compliance/revelio-license.md` (8 sections, ASCII-only).
2. Verify + regression.
3. Q/A. Log. Flip.

## Out of scope

- No code. No MSA negotiation. No live Revelio API call.
- No data ingestion module (that's a future phase-7.12 or beyond step if/when Revelio is signed).

## References

- `handoff/current/phase-7.7-research-brief.md`
- `docs/compliance/reddit-license.md` (template)
- `docs/compliance/alt-data.md` row 7.7 + Sec. 8 open items
- `.claude/masterplan.json` → phase-7 / 7.7
