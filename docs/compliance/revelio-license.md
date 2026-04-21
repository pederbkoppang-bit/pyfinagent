# Revelio Labs -- Per-Vendor License Record

**Parent:** `docs/compliance/alt-data.md` Sec. 4 row 7.7
**Step:** phase-7.7 masterplan record
**Version:** 1.0 -- 2026-04-20
**Owner:** Peder (GitHub: pederbkoppang-bit)

---

## 1. Scope

Ingestion of Revelio Labs employee-sentiment and workforce-dynamics data for
internal signal generation and backtesting. In-scope datasets (of Revelio's 6):

- **Sentiment Analysis** -- employee review sentiment, monthly from 2009,
  mapped to seniority/geo/role.
- **Workforce Dynamics** -- headcount + outflow deltas, weekly from 2007,
  company-level aggregates.

Out of scope at the time of this doc: Transitions, Job Postings (COSMOS -- see
LinkUp phase-7.10), Layoff Notices (WARN), Individual Work History. Any
addition requires MSA amendment and a re-review of this doc.

## 2. Access method

- **Protocol:** Licensed commercial data feed under a Revelio MSA (Master
  Service Agreement). No OAuth flow; MSA is the controlling contract.
- **Delivery:** parquet/CSV via S3, Snowflake share, or Google Cloud Storage.
  Preferred for pyfinagent: GCS bucket + BQ external table or direct BQ load.
- **Credentials:** `REVELIO_API_KEY` env var (for REST tier) + cloud-bucket
  access keys (for batch delivery).
- **Not self-service.** First delivery requires a signed MSA between pyfinagent
  and Revelio Labs.
- **Vendor contact:** `research@reveliolabs.com` (research tier) or sales for
  enterprise.

## 3. Terms-of-Service / MSA record

Advisory `adv_70_oauth_tos` applies by analogy: any MSA execution IS contract
formation. Tracking table:

| Gate | Status | Date |
|------|--------|------|
| Initial vendor outreach | pending | |
| MSA executed | pending | (before first delivery) |
| DPA executed (GDPR Art. 28) | pending | (before first delivery) |
| First data delivery | pending | |
| First live signal run | not started | phase-7.12 or later |

Before any live data delivery, the above gates are completed and this table is
updated. MSA + DPA both reviewed by counsel (outside scope of pyfinagent).

## 4. Rate limits

- **Batch delivery:** weekly (Workforce Dynamics) or monthly (Sentiment);
  no per-request rate limit.
- **REST tier** (if adopted): QPM cap per MSA tier (negotiate; 100 QPM typical
  for research-tier endpoints).
- **File-size ceiling:** 1-4 TB per historical backfill per dataset (U of T
  library citation); ingest incrementally into BQ.
- **Retry:** delivered files are point-in-time; any missed delivery is
  requested from Revelio directly, not retried programmatically.

## 5. Retention & PII

- **Workforce Dynamics:** company-level aggregates; no PII.
- **Sentiment Analysis:** review-level with quasi-identifiers (seniority
  bucket, geography, role family). Individual employees are NOT identifiable
  in the delivered data, but aggregated quasi-identifiers may constitute
  personal data under GDPR when combined.
- **Raw review text retention:** 90 days, then redact per
  `docs/compliance/alt-data.md` Sec. 6.2.
- **Company identifiers** (`CUSIP`, `GVKEY`, `ISIN`, `ticker`) are not
  personal data; retained indefinitely.
- **GDPR Art. 28 DPA:** Revelio as processor must sign a DPA with pyfinagent
  as controller. Eight mandatory Art. 28(3) clauses: processing instructions,
  confidentiality, security, subprocessors, data-subject rights, breach
  notification, retention/deletion, audit rights. Pending execution.
- **No email, no phone, no IP** is delivered by Revelio or stored.

## 6. Permitted use

**Permitted under Revelio MSA (expected terms, pending execution):**

- Internal research and backtesting of trading signals.
- Publication of aggregated results without raw Revelio data.
- Development of sentiment/headcount features for the quant pipeline.

**Prohibited (per typical Revelio MSA terms):**

- Redistribution of raw Revelio data to third parties.
- Training proprietary LLMs on Revelio data without a written addendum.
- Sharing Revelio data with other pyfinagent installations (strict
  per-installation licensing).

**Revenue-threshold trigger:** if pyfinagent moves to real-capital trading
AND Revelio-derived signals contribute materially AND revenue exceeds
$10k/year, re-assess MSA tier with vendor. Tracked as phase-7.12 open item.

## 7. Review cadence

- **Quarterly review:** next 2026-07-20; operator opens this doc and checks
  sections 2, 3, 5, 6 against current MSA + Revelio product updates.
- **Event-triggered reviews:**
  - MSA renewal or amendment -- same-day re-read.
  - New Revelio dataset added to scope (e.g. WARN or Transitions) -- new
    addendum + re-approved DPA before ingestion.
  - Any GDPR/CCPA enforcement action against Revelio -- immediate re-read
    of Sec. 5 and Sec. 6.
  - Revenue-threshold trigger hit (see Sec. 6).

## 8. References

- Revelio Labs product catalog: https://www.reveliolabs.com/data/
- Revelio Research tier: https://www.reveliolabs.com/products/research/
- WRDS vendor page: https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/revelio-labs/
- DPA guide (Art. 28 sections): https://www.hyperstart.com/blog/dpa-agreement/
- Industry comparison (Revelio vs Coresignal vs Thinknum): https://coresignal.com/alternative-data-providers/
- Internal: `docs/compliance/alt-data.md` Sec. 2.2 (ToS framework), Sec. 2.5
  (GDPR/CCPA), Sec. 4 row 7.7, Sec. 5 (scraping disciplines -- mostly N/A
  since this is batch delivery), Sec. 6.2 (retention), Sec. 8 (open items).
- Internal: `handoff/current/phase-7.7-research-brief.md` (6 sources in full).
- Internal: `docs/compliance/reddit-license.md` (per-vendor template).

---

**Reviewed:** 2026-04-20 (initial authoring)
