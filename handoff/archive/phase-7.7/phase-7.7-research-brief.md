# Research Brief: phase-7.7 — Revelio Labs Employee Sentiment License Doc

**Tier:** simple
**Date:** 2026-04-19
**Researcher:** researcher agent

---

## Queries run (three-variant discipline)

1. **Current-year frontier:** "Revelio Labs employee data licensing API product offering 2026"
2. **Last-2-year window:** "Revelio Labs WRDS workforce data research academic license 2024 2025"
3. **Year-less canonical:** "Revelio Labs workforce data MSA pricing tiers enterprise license"
4. Supplemental: "Revelio Labs API delivery format REST Snowflake data feed integration"
5. Supplemental: "alternative data vendor employee sentiment LinkUp Thinknum Coresignal comparison workforce intelligence 2025"
6. Supplemental: "licensed alternative data compliance data processing agreement sub-processor retention vendor 2025"

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.reveliolabs.com/data/ | 2026-04-19 | Official product page | WebFetch | Six products: Workforce Dynamics, Employee Transitions, Job Postings (COSMOS), Sentiment Analysis, Layoff Notices, Individual Work History; all delivered via API, Data Feed, and Dashboard |
| https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/revelio-labs/ | 2026-04-19 | Official vendor doc (WRDS) | WebFetch | Six WRDS datasets; 4.5M+ companies, 1.1B+ profiles, 3.4K+ skills; monthly updates from 2007; access requires WRDS subscription; contact research@reveliolabs.com |
| https://www.reveliolabs.com/products/research/ | 2026-04-19 | Official product page | WebFetch | Research/academic tier: project-specific one-time delivery or WRDS institutional license; all datasets include CUSIP/GVKEY/ISIN/ticker mapping; delivery as flat files (parquet/CSV) via S3, Snowflake, GCS, or zipped link |
| https://mdl.library.utoronto.ca/mdl-blog/new-data-revelio-labs-workforce-data | 2026-04-19 | Library/institutional doc | WebFetch | U of T licensed 6 datasets 2007-2024; size 1-4TB per dataset; delivery into SciNet supercomputing environment; access gated by application process; confirms Sentiment dataset contains positive/negative text mapped to employee characteristics |
| https://www.hyperstart.com/blog/dpa-agreement/ | 2026-04-19 | Authoritative legal blog | WebFetch | Eight mandatory DPA elements under GDPR Art.28(3): processing instructions, confidentiality, security, subprocessor management, data subject rights, breach notification, retention/deletion, audit rights |
| https://coresignal.com/alternative-data-providers/ | 2026-04-19 | Industry comparison | WebFetch | Coresignal (859M profiles, API/Snowflake), Bright Data (668M profiles), Thinknum (investor-focused, employee sentiment); Revelio not directly compared but positioned as research-grade; as of Mar 2026 |

---

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://aws.amazon.com/marketplace/pp/prodview-m7h5in35nozha | AWS listing | Auth wall; snippet confirms $85k/year for Human Capital Dynamics dataset via ADX |
| https://aws.amazon.com/marketplace/pp/prodview-r33ewahy5tv62 | AWS listing | Auth wall; snippet confirms trial tier available |
| https://www.reveliolabs.com/ | Homepage | Snippet sufficient; substantive content in product page |
| https://sourceforge.net/software/product/Revelio-Labs/ | Review site | Low authority; no additional content needed |
| https://wrds-www.wharton.upenn.edu/documents/1904/Revelio_Labs_data_on_WRDS.pdf | PDF | Not fetched; WRDS page covers same information |
| https://www.reveliolabs.com/public-labor-statistics/ | Product page | Out-of-scope (free public tier, not licensed commercial feed) |
| https://coresignal.com/jobs-data-providers/ | Comparison | Snippet sufficient for competitor context |
| https://brightdata.com/blog/web-data/best-alternative-data-providers | Blog | Snippet sufficient; no Revelio-specific detail |
| https://www.data-dictionary.reveliolabs.com/faq.html | FAQ | Snippet confirmed delivery format; full fetch blocked |
| https://www.eur.nl/en/library/news/new-wrds-data-available-revelio-labs | Library notice | Snippet sufficient (confirms Jun 2025 Erasmus license) |

---

## Recency scan (2024-2026)

Searched with year-locked queries "2026" and "2024 2025". Findings:

- **Jun 2025:** Erasmus University Rotterdam licensed Revelio on WRDS — confirms continued commercial availability and institutional licensing model (no product discontinuation).
- **Mar 2026:** Coresignal comparison page (most recent authoritative comparison) positions Revelio in the research-grade tier against Coresignal/Bright Data for investor use cases; no new competitor has displaced Revelio for sentiment + transitions use cases.
- **AWS Marketplace (2026 active):** Both Revelio listings are live as of search date; trial tier added since original listing.
- No new case law or regulatory action specific to Revelio Labs data licensing found in the 2024-2026 window.
- alt-data.md open item ("clarify Revelio DPA PII posture") remains unresolved in any public source — must be addressed in license doc and direct vendor outreach.

---

## Key findings

1. **Six distinct datasets**: Workforce Dynamics (headcounts/inflows/outflows, weekly), Employee Transitions (talent movement, monthly), Job Postings/COSMOS (4.1B postings, weekly), Sentiment Analysis (reviews with text + ratings, monthly), Layoff Notices (WARN Act, monthly), Individual Work History (career progression, daily). (Source: reveliolabs.com/data/, 2026-04-19)

2. **Three delivery channels**: REST API (real-time), Data Feed (parquet/CSV via S3, Snowflake, GCS, or zip), and Dashboard UI. Snowflake Marketplace listing confirmed live. (Source: reveliolabs.com/data/ + data-dictionary FAQ snippet, 2026-04-19)

3. **Pricing**: AWS Marketplace lists Human Capital Dynamics at $85,000/year for a 1-year subscription; enterprise pricing is custom. Academic/research tier is available at a discount for project-specific one-time delivery. (Source: AWS snippet + reveliolabs.com/products/research/, 2026-04-19)

4. **Competitor landscape**: LinkUp (job posting data, MSA-based), Thinknum (investor-facing, employee sentiment + job postings), Coresignal (859M profiles, API-first). Revelio is differentiated by WARN layoff data, WRDS academic distribution, and built-in CUSIP/GVKEY/ISIN ticker mapping. (Source: coresignal.com comparison, 2026-04-19)

5. **DPA requirements (GDPR Art.28(3))**: Eight mandatory sections — processing instructions/scope, confidentiality obligations, security measures, subprocessor management, data subject rights assistance, breach notification, retention/deletion, audit rights. Alt-data.md open item on Revelio PII posture requires these sections be explicit in the license doc. (Source: hyperstart.com/blog/dpa-agreement/, 2026-04-19)

6. **No PII in core workforce aggregates**: Revelio's Workforce Dynamics and Job Postings are company-level aggregates with no individual identifiers. Individual Work History and Sentiment datasets contain profile-level data from public sources; Revelio standardizes these and strips direct identifiers, but some quasi-identifiers (seniority, geo, skills) may constitute personal data under GDPR. (Source: U of T library doc + WRDS page, 2026-04-19)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `docs/compliance/alt-data.md` | 287 | Alt-data compliance framework; Section 4 row 7.7 commits to Revelio MSA; open item on Revelio/LinkUp DPA PII posture | Active; row 7.7 points to `docs/compliance/revelio-license.md` (not yet created) |
| `docs/compliance/reddit-license.md` | 129 | 8-section template for per-vendor license docs | Active; canonical template for 7.7 doc |
| `docs/compliance/2026-regulatory-memo.md` | (not read; out of scope) | Regulatory memo | Exists; not relevant to license format |

---

## Consensus vs debate

**Consensus**: Revelio is the leading academic/research-grade workforce dataset vendor; WRDS distribution is widely adopted. Delivery via parquet/CSV flat files is standard for institutional clients; API access is available but less common for bulk historical pulls.

**Debate/uncertainty**: Exact commercial MSA terms are not public. The $85k/year AWS price is a list price for one specific dataset; enterprise bundles covering all six datasets are negotiated. Sub-processor list and DPA specifics must be requested directly from Revelio (research@reveliolabs.com).

---

## Pitfalls (from literature + internal)

- Individual Work History and Sentiment datasets contain quasi-identifying data; license doc must confirm Revelio's DPA covers the GDPR Art.28 obligations noted above.
- alt-data.md Section 8 open item ("clarify per Revelio DPA whether they cover our use — signal generation only, no re-distribution") must be resolved before the first live call.
- WARN Act layoff data is US-only; if non-US expansion is later planned, this dataset has a geographic gap.
- Revelio normalizes profiles from public sources; the "no re-distribution" constraint in their standard terms means pyfinagent may not share raw Revelio data with third parties even if the signal is useful.

---

## Application to pyfinagent

- `docs/compliance/alt-data.md` line 160: row 7.7 already commits to `docs/compliance/revelio-license.md` — the doc must be created this step.
- `docs/compliance/alt-data.md` line 258: open item on Revelio DPA PII posture feeds directly into Section 5 (Retention & PII) and Section 8 (References) of the new doc.
- reddit-license.md (lines 1-129) is the canonical 8-section template; revelio-license.md must match its structure with Revelio-specific content substituted.
- Recommended pyfinagent datasets for phase-7.7: **Sentiment Analysis** (monthly, employee reviews) as the primary signal; **Workforce Dynamics** (weekly, headcounts/outflows) as the secondary confirming signal for layoff/hiring trend detection.

---

## Design proposal: `docs/compliance/revelio-license.md` (8-section structure)

Matching reddit-license.md exactly:

**Section 1 — Scope**: Ingestion of Revelio Labs Sentiment Analysis dataset (employee reviews with ratings and text) and Workforce Dynamics dataset (headcounts, inflows, outflows). Delivery method: parquet flat files via S3 data feed or Snowflake share. Use: internal signal generation and backtesting only.

**Section 2 — Access method**: Revelio Labs MSA / data feed agreement. Delivery: S3 bucket or Snowflake (preferred for BigQuery integration). Credentials: `REVELIO_API_KEY` env var (or Snowflake connector credentials). No OAuth click-through — MSA is the controlling contract.

**Section 3 — Terms-of-Service / MSA record**: Advisory `adv_70_oauth_tos` applies. Tracking table: MSA execution (pending), DPA signed (pending), first data delivery (not started), first live signal run (phase-7.12). Mirrors reddit-license.md table pattern.

**Section 4 — Rate limits**: Per MSA tier. API: contact Revelio for QPM cap. Data feed: batch delivery (weekly/monthly depending on dataset); no per-request rate limit applies. Client-side: no live API calls in phase-7.7 scope; scaffold only.

**Section 5 — Retention & PII**: Workforce Dynamics is company-level aggregate — no individual PII. Sentiment dataset contains quasi-identifying fields (seniority, geo, role). Raw review text: 90-day retention then redact, matching alt-data.md Sec.6.2. No direct identifiers expected from Revelio (they strip names before delivery); confirm in DPA. GDPR/CCPA basis: aggregated sentiment index; no SAR obligation on aggregates.

**Section 6 — Permitted use**: Internal research, backtesting, and signal generation. No redistribution of raw Revelio data. No training proprietary LLMs on Revelio content without explicit addendum. Revenue-threshold trigger: same as reddit-license.md Sec.6 — re-assess if live trading revenue > $10k/year with material Revelio contribution.

**Section 7 — Review cadence**: Quarterly, next 2026-07-20. Event-triggered: any new Revelio dataset added, any MSA renewal, any Revelio pricing-tier change.

**Section 8 — References**: reveliolabs.com/data/, WRDS vendor page, DPA guide, alt-data.md Sec.4 row 7.7, alt-data.md Sec.8 open items, advisory adv_70_oauth_tos, this research brief.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total including snippet-only (16 total)
- [x] Recency scan (last 2 years) performed and reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (alt-data.md:160, alt-data.md:258, reddit-license.md:1-129)

Soft checks:
- [x] Internal exploration covered relevant compliance modules (alt-data.md, reddit-license.md)
- [x] Competitor landscape noted (LinkUp, Thinknum, Coresignal)
- [x] All claims cited per-claim
- [ ] Revelio DPA/MSA exact terms: not public; must be requested from vendor (gap documented, not gate-blocking for a doc-only step)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "gate_passed": true
}
```
