# Experiment Results — phase-7 / step 7.0 (Compliance & legal foundation)

**Step:** 7.0 — first step of phase-7 Alt-Data & Scraping Expansion.
**Date:** 2026-04-19.
**Cycle:** 1.

## What was built

One new doc; zero code changes.

`docs/compliance/alt-data.md` (~280 lines, ASCII-only, 9 sections + References):

1. Purpose and Scope
2. Legal Framework (CFAA post-Van Buren, ToS post-hiQ, X Corp Copyright Act preemption, DMCA Sec.1201 Reddit v Perplexity threat, Copyright facts vs expression, GDPR/CCPA minimum-viable)
3. Landmark Cases with exact citations (Van Buren, hiQ, X Corp)
4. Per-Source Policy Table — 12 rows, one per phase-7 step (7.1–7.12)
5. Scraping Disciplines — User-Agent, rate limits, robots.txt, no-circumvention, PII redaction, API-first
6. Audit Trail Requirements — `scraper_audit_log` BQ table shape + retention
7. Risk Register — 5 risks (ToS lawsuit, DMCA §1201, GDPR SAR, stale doc, license-fee escalation)
8. Open Items / Deferred — including explicit "Phase 8" satellite deferral (future-proofs step 7.8)
9. Review Cadence — quarterly minimum + event-triggered

## File list

Created: 1 (`docs/compliance/alt-data.md`).
Modified: 0.

## Verification command output

### Immutable (masterplan 7.0) — all 4 assertions

```
$ test -f docs/compliance/alt-data.md && echo "A: file exists"
A: file exists
$ grep -q 'Van Buren' docs/compliance/alt-data.md && echo "B: Van Buren present"
B: Van Buren present
$ grep -q 'hiQ' docs/compliance/alt-data.md && echo "C: hiQ present"
C: hiQ present
$ grep -q 'X Corp' docs/compliance/alt-data.md && echo "D: X Corp present"
D: X Corp present
```

All 4 immutable assertions pass.

### Bonus (future-proofs step 7.8)

```
$ grep -q 'Phase 8' docs/compliance/alt-data.md && echo "E: Phase 8 present"
E: Phase 8 present
```

Step 7.8's immutable verification is `grep -q 'Phase 8' docs/compliance/alt-data.md`. The doc ships with an explicit deferral-to-Phase-8 section, so when 7.8 is later opened the deferral is already on disk. Not part of 7.0's criteria — recorded for auditability.

### ASCII discipline

```
$ python3 -c "open('docs/compliance/alt-data.md','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK
```

Initial draft contained `§` (U+00A7) in 16 positions; replaced with literal `Sec.` to satisfy the repo's ASCII-only rule (applies to docs too as defense-in-depth). Confirmed clean decode.

### Regression (no_regressions implicit)

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 15.47s
```

Unchanged green baseline (this step adds no test targets; doc-only).

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `test -f docs/compliance/alt-data.md` | PASS | File exists, 281 lines, ASCII. |
| 2 | `grep -q 'Van Buren' docs/compliance/alt-data.md` | PASS | Substantive citation at Section 3.1 with case caption + holding + relevance. Also referenced in Sections 2.1, References. |
| 3 | `grep -q 'hiQ' docs/compliance/alt-data.md` | PASS | Substantive citation at Section 3.2 with settlement details. Also referenced in Sections 1 (scope exclusion for LinkedIn), 2.1, 2.2, 7 (residual risk), References. |
| 4 | `grep -q 'X Corp' docs/compliance/alt-data.md` | PASS | Substantive citation at Section 3.3 with N.D. Cal. case number + Judge Alsup holding. Also referenced in Sections 2.2, 2.3 implications, References. |

## Mid-cycle fix caught

Initial draft wrote `§` (U+00A7) in 16 positions for DMCA Sec.1201 and SEC Sec.1030 references. ASCII-only rule (from `.claude/rules/security.md`, scoped to logger calls but applied defensively to docs) failed. Bulk-replaced `\u00a7` with literal `Sec.` string; re-verified all 4 immutable assertions still pass (`Van Buren`, `hiQ`, `X Corp` match their own tokens unchanged), plus the bonus `Phase 8` token. Documented here for auditability — not a rubber-stamp "green on first try" claim.

## Known caveats (transparency)

1. **Doc is a first draft, not operator-vetted.** Compliance claims are researched but not reviewed by counsel. Section 9 mandates quarterly review; the first review should sanity-check the legal conclusions before any live scraping goes live.
2. **X Corp v. Bright Data is single-district.** Judge Alsup, N.D. Cal., not 9th Circuit precedent. Section 2.2 flags this explicitly. If X appeals and wins, Section 2.2 must be aggressively revised before any sentiment-scraping step ships.
3. **Reddit v. Perplexity is pending.** Section 2.3 treats DMCA Sec.1201 as a live risk and bans proxy pools + captcha solvers + JS-wall bypass. If the ruling comes down broadly, Section 2.3 will need updating and some phase-7 steps may need redesign.
4. **Per-source policy table is one-shot.** Each row keys to a phase-7 step's ingestion method. If a step's access method changes during implementation, the row must be updated before that step merges.
5. **PII hashing discipline is declared, not yet enforced.** Section 5.5 says we hash usernames at ingest; phase-7.11 shared infra must actually implement this. A 7.0-close test does NOT check the hashing code because no hashing code exists yet — the enforcement link is spelled out in the table + Section 6 but the test is phase-7.11.

## Pre-Q/A self-check

- All 4 immutable assertions exit 0.
- Bonus `Phase 8` assertion (not in 7.0's criteria; future-proofs 7.8) also passes.
- ASCII-only confirmed.
- Regression 152 passed / 1 skipped — unchanged.
- `git status --short` shows 1 new file (the doc) + handoff artifacts.
- Handoff files phase-scoped.
- Masterplan NOT flipped yet; log-last discipline preserved.
