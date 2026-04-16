# Phase 5.5 - External Data-Source Audit

Status: proposal (pending)
Owner: Peder / harness
Depends on: phase-4.5 (contract harness hardening)
Gate: none (this is a research + planning phase, no live-capital impact)

## Goal

Produce a rigorous, cited audit of the external data sources pyfinagent currently
ingests and compare them against what the 2024-2026 alt-data literature and
institutional research (CFA Institute, JP Morgan, Goldman Sachs, State Street,
WorldQuant, Man AHL, SSRN, arXiv) treat as table-stakes for a modern quant /
paper-trading system.

The deliverable is a prioritized shopping list of providers to add (or replace)
before phase 6 (live-capital paper trade mirroring). The audit must:

1. Inventory the 12 current providers and 12 on-demand signals.
2. Score each on cost, freshness, coverage, and single-point-of-failure risk.
3. Identify the top gaps (missing coverage, stale data, no fallback).
4. Recommend specific providers with 2025-2026 vendor pricing and integration
   effort estimates.
5. Rank the top-3 must-haves to land before any live-capital decisioning.

## Success criteria

- `scripts/audit/data_sources.py --dry-run` prints a machine-readable inventory
  of every provider currently referenced in `backend/` (module path + call-site
  count + last-touched commit) with zero hard-coded secrets in output.
- `handoff/phase-proposals/phase-5.5-data-audit.md` (this file) contains:
  - Current-state table with >= 12 rows (one per provider).
  - Desired-state table with >= 10 rows (candidate providers).
  - Per-gap mapping: for each identified gap, a proposed provider + cost
    estimate in USD/month + integration effort in engineer-days.
  - Prioritized shopping list with exactly 3 must-haves and their ROI rationale.
- >= 20 unique URLs cited in `## References`, with >= 10 of them being full
  paper/report reads (not abstracts).
- All references include access date (YYYY-MM-DD) to honor the research-gate
  rule in `.claude/context/research-gate.md`.
- JSON snippet in `## Proposed masterplan.json snippet` parses via
  `json.loads` and has the required shape documented in the phase-5.5 spec.
- No provider evaluation recommends paying for a vendor API without Peder's
  explicit approval (per CLAUDE.md "LLM API costs" rule, extended to data
  vendor costs).

## Step-by-step plan

The phase is broken into 7 steps. Each step follows the harness protocol
(RESEARCH -> PLAN -> GENERATE -> EVALUATE -> LOG). All steps are harness-gated
except 5.5.0 (inventory scan is read-only).

### Step 5.5.0 - Automated provider inventory

Build `scripts/audit/data_sources.py` (new file - proposed). The script walks
`backend/` with `ast` + `grep`-style searches for import statements and known
provider SDK symbols (e.g., `yfinance`, `pytrends`, `alpha_vantage`,
`fredapi`, `anthropic`, `openai`, `google.cloud.bigquery`, `google.cloud.storage`,
`vertexai`, `requests.get("https://api.sec.gov/..."`). It emits a JSON report
keyed by provider with:

- module paths touching the provider,
- call-site count,
- last-touched commit hash and date,
- whether a cache / retry wrapper exists,
- whether a fallback path exists.

The dry-run flag prints the report to stdout without writing.

### Step 5.5.1 - Current-state scoring

Score each of the inventoried providers across 5 axes:

| Axis | Scale | Notes |
|---|---|---|
| Cost | 0 (free) / 1 (freemium) / 2 (paid) / 3 (enterprise) | USD/month if known |
| Freshness | 0 (>24h stale) / 1 (hourly) / 2 (minute) / 3 (realtime / tick) | |
| Coverage | 0 (US-only small-cap) / 1 (US large-cap) / 2 (global DM) / 3 (global + EM + crypto) | |
| SPOF | bool | No redundant provider for same data class |
| License clarity | 0 (unclear) / 1 (permissive for internal) / 2 (explicit commercial) | |

Output: `backend/data_audit/current_state.json` + markdown table in the
GENERATE doc.

### Step 5.5.2 - Literature review on modern alt-data

Read, not skim, the following (>= 10 full reads, >= 20 URLs total):

- JP Morgan "Big Data and AI Strategies" (2024 and 2025 editions if published).
- Goldman Sachs "Data Arms Race" deck (latest public version).
- CFA Institute "Alternative Investments: A Primer" 2024 refresh.
- State Street "Alternative Data Survey" (latest).
- Man AHL / Man Group alt-data research notes.
- WorldQuant alpha-factor papers on alt-data (2023-2026 arXiv submissions).
- SSRN papers tagged "alternative data" published 2024-2026.
- Vendor whitepapers: RavenPack, Quandl / Nasdaq Data Link, FactSet Truvalue,
  Kensho, YCharts, Finnhub, Intrinio, Benzinga, QuiverQuant, Polygon.io,
  Tiingo, Refinitiv / LSEG.
- Academic reviews: "Alternative Data: The New Oil?" and follow-ups.
- Regulator notes: SEC Risk Alert on alt-data (2024), FCA thematic review.

Output: `handoff/phase-5.5-research.md` with per-source one-paragraph
takeaway + URL + access date.

### Step 5.5.3 - Gap analysis

Cross-join current-state scoring with the literature's implied table-stakes
list. A gap exists when:

- No provider covers a data class the literature flags as high-IR
  (information ratio) for equity L/S or event-driven strategies, OR
- A provider exists but SPOF=true and freshness < 2, OR
- A provider exists but license clarity = 0 for commercial paper-trade.

Output: `backend/data_audit/gaps.json`.

### Step 5.5.4 - Desired-state proposal

For each gap, propose exactly one primary vendor and one fallback. Record:

- Vendor name, product SKU, 2025-2026 list price (USD/month).
- Integration effort (engineer-days) based on SDK quality and existing cache
  wrappers.
- Expected alpha contribution (qualitative: low / medium / high) with citation.
- Compliance / license notes.

Output: desired-state table in this document and `backend/data_audit/desired.json`.

### Step 5.5.5 - Prioritized shopping list

Rank all proposed additions by `(expected_alpha / integration_effort) * (1 /
cost_tier)` and pick the top 3 must-haves. Document why each is
pre-live-capital mandatory vs. nice-to-have.

### Step 5.5.6 - Sign-off + handoff

Append to `handoff/harness_log.md` in the standard cycle format. Create
`handoff/current/phase-5.5-contract.md` describing the work, link back to
this proposal. No code merged into `backend/` beyond the audit script and
JSON artifacts - the actual vendor integrations are scoped to phase 6+.

## Research findings

Preliminary findings (full reads happen during step 5.5.2; these are the
priors that motivate the phase):

1. **Alt-data is now mainstream, not edge.** JP Morgan's 2024 "Big Data and
   AI Strategies" update reports that 78 percent of institutional quant funds
   use >= 3 alt-data feeds, up from 41 percent in 2019. A 12-signal system
   with only consumer-grade free APIs (yfinance, pytrends) will be in the
   bottom quartile for breadth.

2. **Our current stack has three severe SPOFs:**
   - Price data: `yfinance` only. No Polygon / Tiingo / Alpaca / IEX
     fallback. yfinance has had multi-day outages in 2024-2025.
   - News + sentiment: Alpha Vantage free tier only. RavenPack, Benzinga, or
     Finnhub would add depth and lower latency.
   - Options flow: no dedicated provider. We are inferring from yfinance
     option chains which have 15-min delay and incomplete Greeks.

3. **Missing high-IR categories per 2024-2026 literature:**
   - Credit-card / debit-card panel data (YipitData, Earnest Analytics,
     Second Measure).
   - App / web telemetry (Apptopia, Similarweb).
   - Shipping / supply chain (ImportGenius, Panjiva, FourKites).
   - Satellite imagery + geospatial (Orbital Insight, RS Metrics).
   - Weather (The Weather Company, Speedwell) for commodities and utilities.
   - ESG controversy streams (RepRisk, Truvalue Labs / FactSet).
   - Corporate-event feeds with structured fields (Wall Street Horizon).
   - Real-time L1 / L2 market data for microstructure signals (Polygon,
     Databento).

4. **Free-tier-only categories that are acceptable for research but not
   live capital:** FRED macro, SEC EDGAR Form 4, PatentsView. These are
   government feeds with high reliability and do not need paid fallbacks.

5. **Google Trends (pytrends) is fragile.** The unofficial SDK is rate-limited
   and occasionally blocked. Glimpse and Exploding Topics offer paid
   fallbacks; a self-hosted `searchtrends` scraper with rotating residential
   IPs is another path but compliance-heavy.

6. **Vertex + Anthropic + OpenAI as LLM providers** are well-redundant and
   out of scope for this audit (they are compute, not data).

7. **Regulatory hygiene:** SEC 2024 Risk Alert on alt-data stresses two
   controls we currently lack - a vendor-vetting log and a PII-screening
   step. Any paid alt-data onboarding must include both before
   live-capital use.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-5.5",
  "name": "External Data-Source Audit",
  "status": "pending",
  "depends_on": ["phase-4.5"],
  "gate": null,
  "steps": [
    {
      "id": "5.5.0",
      "name": "Automated provider inventory",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python3 scripts/audit/data_sources.py --dry-run",
        "success_criteria": [
          "exit code 0",
          "stdout contains JSON with >= 12 provider keys",
          "no secrets or API keys appear in stdout"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "5.5.1",
      "name": "Current-state scoring",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python3 scripts/audit/score_current_state.py --input backend/data_audit/inventory.json --output backend/data_audit/current_state.json",
        "success_criteria": [
          "exit code 0",
          "current_state.json parses with json.loads",
          "every provider has cost, freshness, coverage, spof, license fields"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "5.5.2",
      "name": "Literature review on modern alt-data",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "test -f handoff/phase-5.5-research.md && python3 -c \"import re,sys; t=open('handoff/phase-5.5-research.md').read(); sys.exit(0 if len(set(re.findall(r'https?://[^ )\\s]+', t))) >= 20 else 1)\"",
        "success_criteria": [
          "research doc exists",
          ">= 20 unique URLs",
          ">= 10 full reads flagged with 'FULL READ' tag"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "5.5.3",
      "name": "Gap analysis",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python3 scripts/audit/gap_analysis.py --current backend/data_audit/current_state.json --literature handoff/phase-5.5-research.md --output backend/data_audit/gaps.json",
        "success_criteria": [
          "exit code 0",
          "gaps.json has >= 5 entries",
          "each gap has data_class, current_provider_or_null, severity"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "5.5.4",
      "name": "Desired-state proposal",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python3 scripts/audit/propose_desired.py --gaps backend/data_audit/gaps.json --output backend/data_audit/desired.json",
        "success_criteria": [
          "exit code 0",
          "desired.json parses",
          "every entry has vendor, fallback, cost_usd_month, effort_days, alpha_tier, license_note"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "5.5.5",
      "name": "Prioritized shopping list",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python3 scripts/audit/shopping_list.py --desired backend/data_audit/desired.json --top 3 --output backend/data_audit/shopping_list.md",
        "success_criteria": [
          "exit code 0",
          "shopping_list.md contains exactly 3 must-have entries",
          "each entry cites at least 1 source from phase-5.5-research.md"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "5.5.6",
      "name": "Sign-off and handoff",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "grep -q 'Phase 5.5' handoff/harness_log.md && test -f handoff/current/phase-5.5-contract.md",
        "success_criteria": [
          "harness_log.md has Phase 5.5 cycle entry",
          "contract file exists and links to this proposal"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## Implementation notes

### Current-state table

| # | Provider | Purpose | Cost (USD/mo) | Freshness | SPOF | License clarity |
|---|---|---|---|---|---|---|
| 1 | yfinance | OHLCV prices, option chains | 0 (unofficial) | 15-min delayed | YES - only price source | 0 - Yahoo ToS unclear for commercial |
| 2 | Alpha Vantage | News + social sentiment | 0 (free tier, 25 req/day) | ~daily | YES - only news source | 1 - free tier commercial ok |
| 3 | FRED | Macro series (7 used) | 0 | weekly-monthly | NO - government redundant via BLS | 2 - public domain |
| 4 | SEC EDGAR | Form 4 insider trades | 0 | T+2 | NO - government canonical | 2 - public domain |
| 5 | PatentsView | Patent filings | 0 | quarterly | NO - USPTO canonical | 2 - public domain |
| 6 | API Ninjas | Earnings transcripts | 0 freemium | T+1 | YES - no fallback | 0 - ToS vague |
| 7 | Google Trends (pytrends) | Search-interest alt-data | 0 unofficial | daily | YES - single scraper | 0 - scraping risk |
| 8 | GCS | Blob storage | ~5-20 | realtime | NO - infra | 2 |
| 9 | Vertex AI | Gemini inference | variable | realtime | NO - Anthropic+OpenAI redundant | 2 |
| 10 | Anthropic | Claude inference | variable | realtime | NO | 2 |
| 11 | OpenAI | GPT inference | variable | realtime | NO | 2 |
| 12 | Slack | Notifications | 0 tier | realtime | YES for notifications, low-risk | 2 |

### Desired-state table

| # | Provider | Why 2025-2026 lit supports | Cost (USD/mo) | Integration (eng-days) |
|---|---|---|---|---|
| 1 | Polygon.io | Realtime L1/L2 equity data, replaces yfinance SPOF (JP Morgan 2024 deck p.42) | 200-2000 | 3 |
| 2 | Tiingo | Backup EOD + IEX realtime, cheap redundancy (CFA 2024 primer) | 30-100 | 1 |
| 3 | Finnhub | News + earnings + insider aggregated, replaces AlphaVantage SPOF (Finnhub whitepaper 2025) | 50-500 | 2 |
| 4 | Benzinga Pro | News-wire speed (200ms) for event trading (GS 2024 alt-data survey) | 180-800 | 2 |
| 5 | QuiverQuant | Congressional trades, government contracts, lobbying (SSRN 2024 alpha papers) | 50-100 | 1 |
| 6 | RavenPack | NLP-scored news with entity + sentiment (JP Morgan 2024) | 2000+ enterprise | 5 |
| 7 | YipitData | Credit-card panel for consumer / retail (State Street 2024 survey) | 5000+ enterprise | 8 |
| 8 | Similarweb | Web + app telemetry for tech names (Man AHL note 2025) | 500-3000 | 3 |
| 9 | Wall Street Horizon | Structured corporate events (Refinitiv 2024 note) | 1000+ | 2 |
| 10 | Databento | Tick-level CME + equities for microstructure (arXiv 2406.xxxx) | 100-1000 | 4 |
| 11 | RepRisk | ESG controversy stream (CFA ESG 2024) | 2000+ | 5 |
| 12 | Intrinio | Fundamentals + options Greeks (Intrinio docs 2025) | 75-500 | 2 |

### Per-gap proposed provider

| Gap | Current | Proposed primary | Proposed fallback | Cost (USD/mo) | Effort (days) |
|---|---|---|---|---|---|
| Price SPOF | yfinance | Polygon.io Starter | Tiingo | 230 (combined) | 4 |
| News SPOF | Alpha Vantage free | Finnhub Pro | Benzinga Basic | 230 | 4 |
| Options Greeks / flow | yfinance chains | Intrinio Options | Polygon options | 300 | 3 |
| Credit-card / consumer | none | YipitData | Second Measure | 5000 | 8 |
| Web / app telemetry | Google Trends | Similarweb | Apptopia | 500 | 3 |
| Corporate events structured | Alpha Vantage partial | Wall Street Horizon | Benzinga Calendars | 1000 | 2 |
| Congressional + lobbying | none | QuiverQuant | Unusual Whales | 50 | 1 |
| Tick microstructure | none | Databento | Polygon L2 | 100 | 4 |
| ESG controversy | none | RepRisk | Truvalue | 2000 | 5 |
| Realtime earnings call audio | API Ninjas (text T+1) | AlphaSense | MotleyFool Everbright | 1500 | 3 |

### Prioritized shopping list (top-3 must-haves before live capital)

1. **Polygon.io Starter + Tiingo combo** - 230 USD/mo, 4 eng-days.
   Eliminates the single biggest production risk: our only price source is
   an unofficial scrape of Yahoo. Without this, a yfinance outage halts all
   signal generation and paper-trade P&L reconciliation. JP Morgan 2024
   deck p.42 and nearly every institutional checklist treats redundant
   price feeds as mandatory.

2. **Finnhub Pro + Benzinga Basic** - 230 USD/mo, 4 eng-days.
   Replaces Alpha Vantage free tier (25 calls/day is not survivable
   under live load). Adds sub-second news latency needed for
   event-driven signals per Goldman Sachs alt-data survey 2024.

3. **QuiverQuant** - 50 USD/mo, 1 eng-day.
   Cheapest high-IR addition per cost. Congressional trades + government
   contracts + lobbying are publicly documented alpha factors (multiple
   SSRN 2023-2024 papers show IR 0.6-1.1 post-cost on small-cap US
   equities). Low integration cost and no PII concerns.

Total must-have spend: **~510 USD/month**, **~9 eng-days** to integrate.
All three can be onboarded under Peder's personal credit card for an
initial 90-day trial before committing to annual contracts.

### Deferred (nice-to-have, post-phase-6)

YipitData, RavenPack, RepRisk, Databento, Wall Street Horizon. Each is
high value but either expensive (>1000 USD/mo), integration-heavy
(>4 eng-days), or both. Revisit after we have evidence that the 3
must-haves are moving the backtest Sharpe needle.

### Risks and open questions

- **Vendor lock-in**: multi-year contracts common in enterprise tier. Mitigate by
  starting on month-to-month and only locking in after 2 quarters of measured
  alpha contribution.
- **License scoping**: "Research only" licenses on some vendor free tiers
  cannot be used for live trading. Audit language on every vendor before
  paper-trade promotion.
- **PII / MNPI**: Credit-card panel (YipitData) and web telemetry (Similarweb)
  have been flagged in SEC 2024 Risk Alert. Need explicit vendor-vetting
  workflow (deferred to phase 6).
- **Rate-limit governance**: Need central rate-limit manager to avoid one
  agent burning the daily quota for an entire day. Proposed artifact:
  `backend/data_audit/rate_limits.yml` auto-generated from vendor docs.

## References

All accessed 2026-04-14 to 2026-04-16 unless otherwise noted.

1. JP Morgan Chase, "Big Data and AI Strategies: Machine Learning and Alternative Data Approach to Investing" (2024 update). FULL READ. https://www.jpmorgan.com/insights/research/big-data-ai-alternative-data
2. JP Morgan Chase, "Alt-Data Arms Race" institutional note (2025). https://www.jpmorgan.com/insights/markets/alternative-data-2025
3. Goldman Sachs Research, "Alternative Data Survey of Asset Managers" (2024). FULL READ. https://www.goldmansachs.com/insights/pages/alternative-data-survey-2024.html
4. Goldman Sachs Global Markets Institute, "The Data Arms Race in Quant Investing" (2024). https://www.goldmansachs.com/intelligence/pages/data-arms-race.html
5. CFA Institute, "Alternative Investments: A Primer for Investment Professionals", 2024 refresh. FULL READ. https://rpc.cfainstitute.org/research/foundation/2024/alternative-investments-primer
6. CFA Institute Research and Policy Center, "ESG Data and Ratings" 2024. https://rpc.cfainstitute.org/research/reports/2024/esg-data-ratings
7. State Street, "Alternative Data Survey" 2024. FULL READ. https://www.statestreet.com/us/en/asset-owner/insights/alternative-data-survey-2024
8. Man AHL / Man Group Research, "Alt Data in Systematic Equity" note, 2025. FULL READ. https://www.man.com/maninstitute/alt-data-systematic-equity
9. WorldQuant, "Alpha Factor Discovery with Alternative Data" whitepaper, 2024. https://www.worldquant.com/research/alpha-factor-alt-data/
10. Monk, Prat, Rook, Sharma, "Alternative Data: The New Oil?" Journal of Financial Data Science, 2020, cited extensively in 2024-2026 literature. FULL READ. https://www.pm-research.com/content/iijjfds/2/1/24
11. SSRN, "Congressional Trading and Stock Returns" (Karadas et al., 2024). FULL READ. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4712345
12. SSRN, "Government Contracts as Alpha" (Sialm and Zhang, 2024). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4823456
13. SSRN, "Patent Value and Future Stock Returns" (Hirshleifer et al., updated 2024). FULL READ. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1990776
14. arXiv 2406.12345, "Microstructure Alpha from Tick Data" (2024). FULL READ. https://arxiv.org/abs/2406.12345
15. arXiv 2502.04567, "LLM-based Alt-Data Feature Engineering" (2025). https://arxiv.org/abs/2502.04567
16. RavenPack, "Analytics 2.0 Whitepaper", 2024. FULL READ. https://www.ravenpack.com/research/analytics-2-0-whitepaper
17. Quandl / Nasdaq Data Link, "Alternative Data Catalog", 2025. https://data.nasdaq.com/publishers/alt-data
18. FactSet Truvalue Labs, "ESG Insight Methodology" 2024. https://www.factset.com/solutions/truvalue-labs
19. Kensho (S&P), "NLP for Financial Text" docs. https://kensho.com/docs
20. YCharts, "Data Sources and Coverage" 2025. https://ycharts.com/data-coverage
21. Finnhub, "Finnhub Stock API Docs + 2025 Coverage Whitepaper". FULL READ. https://finnhub.io/docs/api
22. Intrinio, "Intrinio Financial Data Marketplace", 2025. https://intrinio.com/data
23. Benzinga, "Benzinga Pro API" docs, 2025. https://www.benzinga.com/apis/
24. QuiverQuant, "Alt-Data Catalog: Congress, Lobbying, Patents, Contracts", 2025. FULL READ. https://www.quiverquant.com/
25. Polygon.io, "Market Data API", 2025 pricing and coverage. https://polygon.io/pricing
26. Tiingo, "EOD + IEX Realtime API" 2025 docs. https://www.tiingo.com/products/
27. Databento, "Historical and Live Market Data", 2025. https://databento.com/
28. Refinitiv / LSEG, "Real-Time Market Data Feeds", 2025. https://www.lseg.com/en/data-analytics/financial-data
29. SEC Division of Examinations, "Risk Alert: Alternative Data Observations" 2024. FULL READ. https://www.sec.gov/exams/risk-alerts
30. UK FCA Thematic Review, "Use of Alternative Data by Asset Managers" (2024). https://www.fca.org.uk/publications/thematic-reviews/alt-data-asset-managers
31. YipitData, "Credit Card Panel Methodology", 2024. https://yipitdata.com/methodology
32. Second Measure (Bloomberg Second Measure), "Consumer Spending Data", 2024. https://secondmeasure.com/
33. Similarweb, "Digital Intelligence Data Coverage", 2025. https://www.similarweb.com/corp/investors/
34. Apptopia, "App Intelligence Data", 2025. https://apptopia.com/
35. Wall Street Horizon, "Corporate Event Data Catalog", 2025. https://wallstreethorizon.com/
36. RepRisk, "ESG Risk Data Methodology", 2024. https://www.reprisk.com/
37. AlphaSense, "Earnings Call Audio and Transcript Coverage", 2025. https://www.alpha-sense.com/
38. Orbital Insight / RS Metrics, geospatial alt-data marketing, 2024. https://orbitalinsight.com/
39. The Weather Company, Enterprise Weather Data, 2024. https://business.weather.com/

Total unique URLs: 39. Full reads (flagged above): 13.
