# Contract — Cycle 4.15.6 — Batches + Files + Citations + Search results compliance

Step: phase-4.15.6 Batches API + Files API + Citations + Search results
Ran at: 2026-04-18 (UTC)

## Research gate (3-agent MAS)

Spawning `researcher` (merged) to cover external docs + internal code.

## Hypothesis

Claim: pyfinagent uses ZERO of the four doc features (Batches,
Files API, Citations, search_result content blocks). Phase-4.14
MF-31, MF-33, MF-34, and v2 cluster A2 capture the gaps. This
cycle verifies the zero-usage claim with live checks and documents
every documented-but-unused pattern so go-live review can't miss
them.

## Success criteria (immutable, from masterplan)

1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page

## Scope

Write `docs/audits/compliance-batches-files-citations.md` with a
pattern-per-row table covering every pattern in:
- Batches API (creation, limits, retrieval, results format, 50%
  discount, rate limits, ZDR ineligibility)
- Files API (beta header, size limits, MIME types, file_id reuse,
  download restrictions, ZDR ineligibility)
- Citations (document block wrapping, enabled flag, char_location
  vs page_location vs content_block_location, cited_text free,
  compatibility with caching, incompatibility with structured
  outputs)
- Search results content block ({type, title, url/source, content})

Each row: Status / Evidence / Deviation / Risk / Recommended fix /
MF-# mapping.

## Anti-patterns guarded

- Do NOT claim "already covered in phase-4.10-4.13" — that was
  docs-read-only. This cycle needs live code-check evidence.
- Do NOT overlook ZDR gating — adopting any of the 4 features
  makes the org no longer ZDR-eligible.

## Out of scope

- Implementation (audit-only)
- Managed Agents Files (covered in phase-4.15.14)

## Risk

- PDF + citations + search_result together could materially change
  the SEC filing ingestion architecture. Flag as pre-requisites
  before MF-31 / MF-33 / MF-34 land.

## References

- Phase-4.11 platform_overview.md (batches + files + citations
  mention)
- Phase-4.12 prompting_skills_mcp_resources.md (release-notes
  deltas)
- Phase-4.13 messages_sidebar_sweep.md (MF-31, MF-32, MF-33, MF-34)
