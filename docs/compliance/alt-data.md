# Alt-Data & Scraping -- Compliance Framework

**Version:** 1.0 -- 2026-04-19
**Owner:** Peder (GitHub: pederbkoppang-bit)
**Step:** phase-7.0 masterplan record
**Scope:** All alt-data ingestion steps under phase-7 (7.1 through 7.12) and any
future source added behind the same infrastructure.

---

## 1. Purpose and Scope

This document is the legal-compliance gate for every external-data ingestion
that pyfinagent ships as part of phase-7 Alt-Data & Scraping Expansion. No
scraper, API client, or licensed-feed consumer may be deployed to production
unless it satisfies the per-source policy (Section 4) and the scraping
discipline (Section 5) below.

Scope of covered data streams:

- Public government data (SEC EDGAR 13F, US Senate/House financial disclosures,
  FINRA short-sale volume).
- Public web data behind no login (ETF issuer pages, Google Trends, retail
  news + sentiment indexes).
- Public social platforms with explicit API terms (Reddit, X/Twitter).
- Licensed third-party vendors (Revelio Labs, LinkUp, SafeGraph, etc.) where
  the license agreement is the controlling legal authority.

Out of scope: satellite / geospatial imagery (see Section 8 -- deferred to
**Phase 8**), private messaging platforms, LinkedIn profile data (post-hiQ
settlement risk too high), paywalled institutional research.

---

## 2. Legal Framework

### 2.1 Computer Fraud and Abuse Act (CFAA)

Post-Van Buren and post-hiQ, the CFAA is substantially narrowed for our use
case:

- **Van Buren v. United States** narrowed "exceeds authorized access" to
  apply only when the accessor crosses a gate (e.g. a folder they're not
  permitted to open). Misusing data you are authorized to read is NOT a
  CFAA violation.
- **hiQ Labs v. LinkedIn** extended the reasoning to public-web scraping: if
  no authorization is required (no login, no paywall), then there is no
  authorization that can be exceeded.

Practical rule: **scraping of publicly accessible URLs is not a federal
criminal offense** under CFAA. This does NOT exempt us from ToS-breach civil
liability or DMCA Sec.1201 anti-circumvention (see 2.2 and 2.3).

### 2.2 Terms of Service / Breach of Contract (post-X Corp v. Bright Data)

- **X Corp v. Bright Data** held that when the scraped data is non-copyrighted
  factual user-generated content, Copyright Act preemption defeats ToS-based
  trespass/tortious-interference claims. The ruling is a single N.D. Cal.
  district decision (Judge Alsup, 2024), not 9th-Circuit precedent.
- **Residual risk (hiQ-style):** a platform may still sue for breach of a
  click-through ToS where a user "clicked I agree" to access the site. Our
  scrapers must not require authentication; no clicking of an "I agree"
  button happens in the pipeline -> no contract formation -> X Corp reasoning
  defeats this vector as well.
- **Conservative rule:** we still honor `robots.txt`; rate-limit as if the
  site is paying for each request; do not bypass captcha or JS-challenge
  pages (see 2.3).

### 2.3 DMCA Section 1201 (anti-circumvention)

Reddit v. Perplexity (filed 2025, pending 2026) is the first major case
testing DMCA Sec.1201 against rate-limit / API-access-control bypassing. We
must assume Sec.1201 liability is a live risk for any alt-data source that:

- Requires an API key we then pool or multiplex.
- Uses a distinct residential-proxy rotation to bypass per-IP rate limits.
- Solves a CAPTCHA or bypasses a JS wall.

Rule: **we do none of these**. If a source's free tier is insufficient, we
upgrade to a paid tier or drop the source. No residential-proxy pools, no
captcha solvers, no JS-challenge bypass.

### 2.4 Copyright (facts vs creative expression)

US copyright does not protect facts. Stock prices, trade dates, filing
timestamps, short-volume counts, 13F holdings -- all uncopyrightable facts.

But the *presentation* of those facts can be copyrighted (e.g. a
Seeking Alpha analyst's commentary, a hedge fund's PDF layout). Rule: we
ingest facts and factual summaries, not verbatim creative expression. If we
need to store raw text for dedup/embedding, we store it under fair-use
research-purpose rationale AND redact any attribution/author text from
public-facing outputs.

### 2.5 Privacy -- GDPR and CCPA

- **GDPR:** Reddit and Twitter/X are pseudonymous; usernames are personal data
  under EU law if they can be linked to a real person. Mitigation: we hash
  usernames at ingest (sha256) and do not store IP addresses, email, real
  names, or geolocation. Our retention for raw text is <= 90 days.
- **CCPA:** California residents may request deletion of personal info we
  hold. Because we hash usernames at ingest and discard raw text after 90
  days, the operational burden is minimal. A signed deletion-request
  endpoint is tracked in phase-7.11 infra work.

---

## 3. Landmark Cases (exact citations)

### Van Buren v. United States, 593 U.S. ___, 141 S.Ct. 1648 (2021)

Holding: "exceeds authorized access" under 18 U.S.C. Sec.1030(e)(6) applies only
to access of files, folders, or databases that are off-limits to the
accessor. Using authorized access for an improper purpose is not a CFAA
violation. Opinion delivered by Justice Barrett; 6-3 majority.

Relevance to pyfinagent: scraping public URLs (no credential required) does
not fall under Sec.1030 at all, because there is no "authorization" that could
be exceeded.

### hiQ Labs, Inc. v. LinkedIn Corp., 31 F.4th 1180 (9th Cir. 2022)

Holding (on remand from SCOTUS): scraping publicly accessible URLs is not
access "without authorization" under the CFAA. Settlement: December 2022,
hiQ paid $500,000 and accepted a permanent injunction against further
LinkedIn scraping plus data deletion -- NOT a CFAA liability, but a
breach-of-contract outcome on the click-through ToS LinkedIn required.

Relevance to pyfinagent: we do not log into any source, so there is no
ToS contract formation. We still treat the hiQ settlement as a standing
reminder that sources can sue under contract theory even where CFAA fails.
Never scrape LinkedIn.

### X Corp v. Bright Data Ltd., No. 3:23-cv-03698-WHA (N.D. Cal. May 9, 2024)

Holding: The Copyright Act preempts state-law claims (trespass to chattels,
tortious interference with contract, unjust enrichment) against scraping of
publicly posted user-generated content, because X is only a non-exclusive
licensee of the content under its ToS. Judge William Alsup.

Relevance to pyfinagent: the strongest 2024 authority protecting public
retail-sentiment and social-signal scraping from state-law tort claims.
Single-district ruling; we treat it as guidance, not binding precedent. We
still rate-limit and honor robots.txt.

---

## 4. Per-Source Policy Table

Rows correspond 1:1 to phase-7 masterplan steps.

| Step | Source | Access method | Legal basis | Rate limit | Compliance owner |
|------|--------|---------------|-------------|------------|------------------|
| 7.1 | US Senate/House financial disclosures | HTTP scraper + Congress.gov API | Government-public (no CFAA issue; STOCK Act mandates publication) | 8 req/s | phase-7.1 step |
| 7.2 | SEC 13F quarterly holdings | EDGAR API with `User-Agent: pyfinagent/1.0 peder.bkoppang@hotmail.no` | Government-public; SEC's own terms | 8 req/s (EDGAR ceiling is 10) | phase-7.2 step |
| 7.3 | FINRA daily short-sale volume | FINRA Equity API (developer.finra.org) -- NOT the daily TXT download (non-commercial label) | Licensed developer key for commercial use | per-key quota | phase-7.3 step |
| 7.4 | ETF flows (issuer pages + ETF.com / ETFGI aggregation) | HTTP scraper on issuer pages; licensed feed for aggregators if budget permits | Public issuer pages = X Corp reasoning; aggregator feeds require license | 1 req/s per domain | phase-7.4 step |
| 7.5 | Reddit WSB sentiment | Reddit Data API v1 with OAuth app key | Reddit ToS 2024 permits non-commercial + small-volume research; commercial use requires paid enterprise tier | API default | phase-7.5 step |
| 7.6 | Twitter/X sentiment | X API v2 with OAuth app key (paid tier for volume) | X Corp v. Bright Data reasoning + paid-tier terms | API default | phase-7.6 step |
| 7.7 | Employee sentiment | Revelio Labs (licensed feed, not scrape) | Revelio MSA | per-license | phase-7.7 step; license doc at `docs/compliance/revelio-license.md` |
| 7.8 | Satellite / geospatial | **DEFERRED to Phase 8** (see Section 8) | -- | -- | deferred |
| 7.9 | Google Trends extended | pytrends + Google's undocumented endpoints -- treat as fragile | Google ToS permits personal research; commercial use grey-area. Minimum viable: weekly pulls, no keyword-injection | <= 5 req/min | phase-7.9 step |
| 7.10 | Hiring signals | LinkUp (licensed feed, not LinkedIn scrape) | LinkUp MSA | per-license | phase-7.10 step |
| 7.11 | Shared scraper infrastructure | HTTP client library + audit-log BQ table | n/a (infra) | configurable per source | phase-7.11 step |
| 7.12 | Feature integration + IC eval | internal-only | n/a (offline analytics) | n/a | phase-7.12 step |

---

## 5. Scraping Disciplines (mandatory for every step)

### 5.1 User-Agent
Every outbound request must set:

```
User-Agent: pyfinagent/1.0 peder.bkoppang@hotmail.no
```

This matches the SEC EDGAR-required format per `.claude/rules/security.md`
and gives sources an accountable contact.

### 5.2 Rate limits
Each source has a documented per-second cap in Section 4. Infra enforces it
at the HTTP-client layer (phase-7.11); no step may exceed its row cap.

### 5.3 robots.txt
Honor `robots.txt` on every domain. Do not scrape any path disallowed to
`User-Agent: *` or to `User-Agent: pyfinagent/1.0`.

### 5.4 No circumvention
Do not use residential proxy pools, captcha solvers, JS-wall bypass, or
anti-fingerprint browsers. If a source's free tier is insufficient, upgrade
the license or drop the source.

### 5.5 PII redaction
Hash usernames (sha256) before persistence. Never store IP, email, real name,
or geolocation. Truncate raw text after 90 days (retention policy in
Sec.6.2).

### 5.6 API-first
When a source offers a documented API, prefer it over HTML scraping. API
access carries an explicit ToS we can comply with; HTML scraping invites
fragility and legal ambiguity.

---

## 6. Audit Trail Requirements

### 6.1 scraper_audit_log BQ table
Phase-7.11 creates `pyfinagent_data.scraper_audit_log` with at minimum:

```
request_id STRING NOT NULL
source STRING NOT NULL                -- matches step id (e.g. "7.5.reddit")
url STRING NOT NULL
method STRING                          -- GET / POST
status_code INT64
latency_ms FLOAT64
user_agent STRING
ip_hash STRING                         -- sha256 of our egress IP, opt
ts TIMESTAMP NOT NULL
bytes_returned INT64
error STRING                           -- if non-200
```

Every live request writes one row. Partition: `DATE(ts)`.

### 6.2 Retention
- `scraper_audit_log`: keep 2 years for legal auditability.
- Raw-text columns on intel tables: 90 days, then text-redact (keep embeddings + hashes).
- Username columns: always hashed; raw usernames are never written.

### 6.3 Compliance evidence
Monthly the operator runs a query that counts rate-limit violations, HTTP
4xx/5xx rates per source, and anomalous User-Agent drift. Output tracked in
`docs/compliance/alt-data-audit-YYYY-MM.md` (created by phase-7.11 or later).

---

## 7. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------:|-------:|-----------|
| R1 | Platform lawsuit on ToS grounds despite X Corp reasoning | low | high (injunction + discovery cost) | No login; no click-through; fallback to paid API if source fights |
| R2 | DMCA Sec.1201 claim (Reddit v Perplexity style) for rate-limit bypass | low | high | No proxy pools, no captcha solvers, no JS bypass (Section 5.4) |
| R3 | GDPR subject-access request on retained raw text | medium | medium | 90-day raw-text retention; hashed usernames; written SAR runbook (phase-7.11) |
| R4 | Stale compliance doc after new case law | high | medium | Quarterly review cadence (Section 9); each review appends to References |
| R5 | License-fee escalation from Revelio / LinkUp / pytrends | medium | medium | Cost-budget guard in phase-8.5 / phase-10.7 authorizes spend only within the 15-slot daily Claude-routine budget |

---

## 8. Open Items / Deferred (including Phase 8)

- **7.8 Satellite/geospatial proxies** -- DEFERRED to **Phase 8**. Scope:
  parking-lot-count imagery, oil-storage-tank levels, cargo-port counts.
  Deferred because compliant sources (Planet Labs, Maxar, Spire) all require
  enterprise licenses that exceed the current LLM-API budget. Revisit when
  a budget unlock is approved.
- Licensed-feed PII posture -- clarify per Revelio and LinkUp data-processing
  agreements whether they cover our use (signal generation only, no
  re-distribution).
- The X Corp ruling is single-district. Track for a 9th-Circuit appeal.
- Reddit v Perplexity is pending; if the court rules broadly on DMCA Sec.1201,
  revise Section 2.3 aggressively.

---

## 9. Review Cadence

- Minimum: quarterly. Operator opens this doc, checks Sections 2, 3, 7, 8 for
  new case law or license changes, appends a "Reviewed YYYY-MM-DD" line.
- Event-triggered: any new scraping step added to phase-7 or later.
- Any vendor license renewal triggers a same-day re-read of Section 4 rows
  covered by that license.

## References

- Van Buren v. United States, 593 U.S. ___, 141 S.Ct. 1648 (2021)
- hiQ Labs, Inc. v. LinkedIn Corp., 31 F.4th 1180 (9th Cir. 2022)
- X Corp v. Bright Data Ltd., No. 3:23-cv-03698-WHA (N.D. Cal. May 9, 2024)
- Reddit, Inc. v. Perplexity AI, Inc., filed N.D. Cal. 2025 (pending)
- SEC EDGAR access policy: https://www.sec.gov/about/data-publication/sec-edgar-data-publication-best-practices
- FINRA daily short-sale volume data: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data
- Reddit Data API Terms: https://redditinc.com/policies/data-api-terms
- X (Twitter) Developer Agreement: https://developer.twitter.com/en/developer-terms/agreement-and-policy
- `.claude/rules/security.md` (internal): User-Agent + ASCII-only logger rules
- `handoff/current/phase-7.0-research-brief.md` (internal): primary sources + snippet-only URLs
