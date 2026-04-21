# Sprint Contract — phase-7 / step 7.0 (Compliance & legal foundation)

**Step id:** 7.0 — first step of phase-7 Alt-Data & Scraping Expansion.
**Cycle:** 1. **Date:** 2026-04-19. **Tier:** moderate.

Parallel-safe: phase-scoped handoff files.

## Research-gate summary

8 sources fetched in full (Nixon Peabody Van Buren analysis, ZwillGen hiQ wrap-up, FKKS X Corp ruling analysis, TLDRFiling EDGAR rate-limit best-practices, FINRA short-sale-volume catalog + daily files, Tendem.ai 2026 web-scraping legal overview, PainOnSocial Reddit-scraping guide), three-variant queries, 2024–2026 recency scan including Reddit v. Perplexity DMCA §1201 2025 angle. Internal audit confirms `backend/alt_data/` is greenfield and `.claude/rules/security.md` already sets the EDGAR User-Agent format. Brief at `handoff/current/phase-7.0-research-brief.md`. `gate_passed: true`.

## Hypothesis

A single doc `docs/compliance/alt-data.md` captures the legal framework for every phase-7 ingestion step (7.1–7.12) in one place. Sections cover: CFAA post-Van Buren narrowing, ToS/breach-of-contract post-hiQ settlement, copyright-preemption under X Corp v. Bright Data, DMCA §1201 anti-circumvention threat from Reddit v. Perplexity, GDPR/CCPA minimum-viable. Per-source policy table keys each of 12 pending steps to a rule-set. A deferred "Phase 8" section future-proofs step 7.8 (its immutable verification is `grep -q 'Phase 8' docs/compliance/alt-data.md`).

## Immutable success criteria (from .claude/masterplan.json)

List-form assertion set:
- `test -f docs/compliance/alt-data.md`
- `grep -q 'Van Buren' docs/compliance/alt-data.md`
- `grep -q 'hiQ' docs/compliance/alt-data.md`
- `grep -q 'X Corp' docs/compliance/alt-data.md`

All 4 must pass. Not edited. The Van Buren / hiQ / X Corp citations must be substantive (each has a subsection with the case caption + holding + relevance-to-pyfinagent note), not just tokens.

**Non-blocking extra:** include a "Phase 8" deferral section so the future step 7.8 verification becomes trivial. This is explicitly **not** part of 7.0's immutable criteria — noted here so 7.8's closure later is frictionless.

## Plan steps

1. Create `docs/compliance/` directory if absent.
2. Write `docs/compliance/alt-data.md` with the 9-section TOC from the research brief:
   §1 Purpose & Scope · §2 Legal Framework · §3 Landmark Cases · §4 Per-Source Policy Table · §5 Scraping Disciplines · §6 Audit Trail Requirements · §7 Risk Register · §8 Open Items / Deferred (Phase 8 note) · §9 Review Cadence · References.
3. Each landmark-case subsection cites the case caption (ALL three required tokens literally on their own line), the holding, and what it means for pyfinagent.
4. Per-source policy table: one row per phase-7 step (7.1 Congress, 7.2 13F, 7.3 FINRA, 7.4 ETF flows, 7.5 Reddit WSB, 7.6 Twitter/X, 7.7 licensed employee sentiment, 7.8 deferred satellite, 7.9 Google Trends, 7.10 licensed hiring, 7.11 shared infra, 7.12 feature integration).
5. Run the 4 immutable assertions; capture verbatim output.
6. Write `phase-7.0-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No code. Doc-only step.
- No test files (the 4 verification assertions are shell greps, not pytest).
- No automation of future compliance checks (that would be a separate step).
- ASCII-only per `.claude/rules/security.md` (applies to docs too, defense-in-depth).

## References

- `handoff/current/phase-7.0-research-brief.md`
- `.claude/rules/security.md` (EDGAR User-Agent + ASCII-only)
- `.claude/masterplan.json` → phase-7 (all 13 step verifications, so policy table rows key to actual downstream expectations)
- External legal sources (see research brief URLs)
