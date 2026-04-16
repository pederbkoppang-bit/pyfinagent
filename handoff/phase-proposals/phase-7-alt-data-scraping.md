# Phase 7 — Alt-Data & Scraping Expansion

## Goal

Expand pyfinagent beyond current price/fundamental/macro/news inputs by
ingesting 10 alt-data source families that published literature and hedge-fund
practice link to equity return / volatility predictability. Prioritize
(i) free-or-low-cost, (ii) low legal risk, (iii) highest-signal sources first,
so we get usable alpha before committing cash to paid vendors.

By end of Phase 7 the MAS pipeline must:

1. Pull daily snapshots of congressional trades (House + Senate), 13F
   institutional holdings, FINRA short-volume, ETF flows, r/wallstreetbets
   sentiment, X/Twitter sentiment, Glassdoor-style employee signals (via
   licensed aggregators where scraping is risky), Google Trends topic
   clusters, and job-posting volume.
2. Persist raw + normalized rows in `pyfinagent_data.alt_*` BigQuery tables
   with a shared `(symbol, as_of_date, source, signal_name, value)` schema
   so downstream agents can JOIN them into any backtest.
3. Emit Layer-1 / Layer-2 agent features with documented legal provenance
   (source tier, license, ToS status, retention policy).
4. Pass a compliance review checklist (ToS read + archived, CFAA / DMCA /
   Copyright Act preemption analysis, PII scrubbing) before any scraper
   goes live.

## Success criteria

- Seven of ten sources are live in BigQuery with at least 90 days of
  backfill where the source allows it.
- All ten sources have a written go/no-go decision with ToS citation and
  legal tier (free / API-key / paid / scraping-required).
- Each live source produces at least one feature whose Spearman IC vs.
  forward 1-5-20 day returns is statistically non-zero (p < 0.05) on
  pyfinagent's current universe, OR a documented null result archived in
  `backend/backtest/experiments/results/alt_data_ic_YYYYMMDD.tsv`.
- Scrapers honor robots.txt, use rotating proxies + exponential backoff,
  and log every request to `pyfinagent_data.scraper_audit_log` (timestamp,
  URL, status, bytes).
- Compliance checklist (below) checked into `docs/compliance/alt-data.md`
  and signed off by Peder before any scraper is deployed.
- At least three alt-data features added to the Layer-1 enrichment prompts
  in `backend/agents/skills/*.md` and one MAS agent in
  `backend/agents/agent_definitions.py` explicitly consumes them.

## Step-by-step plan

### 7.0 Compliance & legal foundation (BLOCKER for everything else)

- Write `docs/compliance/alt-data.md` with:
  - CFAA status post-Van Buren (public data = no CFAA, gates-up/gates-down
    test) — https://supreme.justia.com/cases/federal/us/593/19-783/
  - hiQ v. LinkedIn final outcome (December 2022 settlement, permanent
    injunction on hiQ, $500k damages, contract-based claims survive even
    when CFAA fails) — https://en.wikipedia.org/wiki/HiQ_Labs_v._LinkedIn
  - X Corp. v. Bright Data (N.D. Cal. 2024) — Copyright Act preempts
    state-law contract claims when user-generated content is being
    scraped, because X is only a non-exclusive licensee.
  - Associated Press v. Meltwater (S.D.N.Y. 2013) — copying and
    redistributing substantial portions of copyrighted content is not
    fair use.
  - EU DSM Directive art. 4 TDM opt-out and California / Illinois BIPA
    considerations for any scraped user data.
- Define source tiers:
  - **Tier A (green-light):** government-produced public records (SEC
    EDGAR, FINRA regsho, disclosures-clerk.house.gov,
    efdsearch.senate.gov, BLS).
  - **Tier B (yellow):** vendor APIs with explicit commercial licenses
    (QuiverQuant, Unusual Whales, WhaleWisdom paid, Financial Modeling
    Prep, Reddit API commercial, Twitter API basic+).
  - **Tier C (red):** ToS-prohibited scraping (LinkedIn, Glassdoor,
    Indeed behind-login, X.com behind-login, nitter mirrors).
- Define a veto rule: Tier C sources require either (a) licensed
  aggregator (Revelio Labs, Thinknum) or (b) written legal approval.
- ASCII-only, no emojis (matches `CLAUDE.md`).

### 7.1 Congressional trades (Tier A/B, HIGH signal)

- Primary: scrape the official disclosure portals
  https://efdsearch.senate.gov/ and https://disclosures-clerk.house.gov/
  nightly. Both are government sites, fall squarely in Van Buren's "gates
  up" safe harbor.
- Secondary: QuiverQuant REST API (https://api.quiverquant.com/docs/)
  as backup + normalized format. QuiverQuant commercial tier has an
  explicit commercial license, so no ToS risk.
- Tertiary: Unusual Whales Congress endpoint
  (https://api.unusualwhales.com/docs) for faster ingestion +
  options-flow cross-reference.
- Features: lag_days (file_date - trade_date), party, chamber, committee
  membership, trade_size_bucket, buy/sell, consensus (N of politicians
  trading same ticker same direction within 30 days).
- Literature: Ziobrowski 2004/2011 (positive alpha), Eggers & Hainmueller
  2013 and Belmont et al. 2022 (null result). Wei 2024 NBER "Captain
  Gains" https://www.nber.org/system/files/working_papers/w34524/w34524.pdf
  re-establishes positive alpha for committee-weighted portfolios.
- Backfill: Senate EFDS exposes ~6 years, House Clerk exposes ~5 years.
  Both allow bulk HTML/PDF download.

### 7.2 13F institutional holdings (Tier A, MEDIUM signal)

- Primary: SEC EDGAR direct (`edgartools` PyPI or raw XBRL + form 13F-HR
  XML). Zero cost, zero ToS risk. https://www.sec.gov/edgar
- Secondary: WhaleWisdom paid ($300/yr) for cleaned, deduplicated,
  fund-scored output — http://legacy.whalewisdom.com/shell/api_help
- Features: quarterly change in shares held by top-100 hedge funds
  (delta_holdings), new buys, complete sellouts, concentration (top-10
  fund ownership %), "crowded" flag (>N funds in same name).
- Quirks: 45-day reporting lag means features are stale; use for
  medium-horizon rebalance signals, not high-frequency alpha.
- Backfill: full history since 1999.
- Companion free mirrors: https://13f.info/ and Dataroma
  (https://www.dataroma.com/m/home.php) for long-only superinvestor
  portfolios.

### 7.3 FINRA daily short volume (Tier A, MEDIUM signal)

- Primary: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files
  — free pipe-delimited daily files, one per ADF/TRF/ORF.
- Features: short_volume_ratio (short_vol / total_vol), 5d delta, 20d
  z-score. Note: FINRA short volume != short interest (which is
  bi-monthly) — methodology covered at
  https://www.finra.org/rules-guidance/notices/information-notice-051019
- Backfill: multi-year history available via FINRA bulk downloads.
- Rate limit: trivial (~4MB/day), no scraping countermeasures.

### 7.4 ETF flows (Tier B, LOW-MEDIUM signal)

- Primary: Nasdaq Data Link `ETFF` (US ETF Fund Flows, paid) —
  https://data.nasdaq.com/databases/ETFF
- Secondary free: ETFdb public pages (https://etfdb.com) + Finnhub ETF
  holdings endpoint (https://finnhub.io/docs/api/etfs-holdings)
- Features: 1/5/20-day net creations per sector ETF, category rotation
  z-score, smart-beta flows.
- Backfill: Nasdaq Data Link has 10+ year history.

### 7.5 r/wallstreetbets + r/investing + r/stocks sentiment (Tier B, MEDIUM signal)

- Reddit API free tier is personal-use only; commercial use requires
  paid access starting ~$12,000/year (https://sellbery.com/blog/how-much-does-the-reddit-api-cost-in-2025/).
- Approach: begin with Reddit API PRAW under the non-commercial research
  license for research-mode prototyping; escalate to paid commercial
  tier before we ship any live signal into a trading workflow.
- Features per ticker per day: mention_count, comment_upvote_ratio,
  sentiment_score (FinBERT), due-diligence post count,
  options-flavored mention share.
- Literature:
  - Bradley et al. "Place Your Bets?" (2022) http://russelljame.com/wsb_3_15_2022.pdf
    — +1.1% two-day, +5% quarterly on WSB DD buys pre-GME.
  - Buz & de Melo 2023 arXiv 2301.00170 — WSB competes with analyst
    consensus post-2018. https://arxiv.org/abs/2301.00170
  - Aloosh, Choi, Ouzan 2023 — meme-stock tail-wagging effect on
    efficiency.
  - Social contagion paper arXiv 2104.01847.
  - 2024 J. Finance Research article linking r/WSB attention to retail
    flow — https://www.sciencedirect.com/science/article/pii/S1057521924006537
- Backfill: Pushshift is dead; use academicreddit.com + a corpus of
  historical dumps (Reddit's own Academic Data Export) instead of
  scraping. Legal because it is licensed redistribution.

### 7.6 X/Twitter sentiment (Tier B/C, HIGH signal if cheap)

- Official API basic tier: $100/mo for 15k tweets (too low), Pro tier
  $5k/mo (too expensive for us). https://twitterapi.io/blog/twitter-api-pricing-2025
- Alternatives: TwitterAPI.io pay-as-you-go $0.15/1k, SociaVault $99/mo,
  Apify Tweet Scraper V2.
- Nitter mirrors: unstable, legally grey (X v. Bright Data shows
  Copyright Act preempts state-law breach, but TOS breach and DMCA
  still live); explicit RECOMMENDATION: do not use nitter in
  production.
- Features per ticker per day: cashtag_count, retweet_weighted_sentiment,
  top-influencer signal, news-breaking spike.
- Start with TwitterAPI.io metered plan capped at $50/mo in research
  mode; gate live trading behind Peder's cost approval.

### 7.7 Employee sentiment (Tier B via licensed vendor, DO NOT SCRAPE)

- Do NOT scrape Glassdoor — Terms of Use
  (https://www.glassdoor.com/about/terms/) explicitly prohibit it and
  Glassdoor has a track record of enforcing.
- Path forward: license Revelio Labs public labor statistics
  (https://www.reveliolabs.com/public-labor-statistics/employment/)
  — already used by hedge funds, distributed through WRDS, so research
  quotes exist. WRDS entry: https://wrds-www.wharton.upenn.edu/pages/data-announcements/new-revelio-labs-data/
- Literature: Green, Huang, Wen, Zhou (J. Fin. Econ. 2019) —
  high-minus-low Glassdoor rating spread 0.84%/month.
  https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300662
- Features: employee_satisfaction_delta, hiring-vs-firing ratio, CEO
  rating, open role count.
- If Revelio cost is blocking: fall back to free BLS JOLTS + LinkedIn
  Economic Graph public reports (https://economicgraph.linkedin.com/).

### 7.8 Satellite/geospatial proxies (Tier B, paid)

- Providers: RS Metrics (parking-lot counts for ~40 US retailers),
  Orbital Insight, SpaceKnow.
- Literature: Katona, Painter, Patatoukas, Zeng (Berkeley Haas, 2024) —
  4-5% abnormal return around earnings from parking-lot signal.
  https://newsroom.haas.berkeley.edu/how-hedge-funds-use-satellite-images-to-beat-wall-street-and-main-street/
- Status: paid only, prices deep in five figures per year. RECOMMEND
  DEFER until Phase 8 unless a vendor offers a free-tier evaluation.
- Free fallback: free Sentinel-2 / Landsat rasters via
  https://earthengine.google.com/ + AIS port traffic data (e.g.
  MarineTraffic public dashboards).

### 7.9 Google Trends extended (Tier A, LOW-MEDIUM signal)

- Primary tool: pytrends (https://github.com/GeneralMills/pytrends)
  with IP rotation (Google hard-throttles, but ToS is more permissive
  than user-facing SERP scraping).
- Move beyond current simple polling:
  - Topic clustering: use Trends' related-queries graph + our embedding
    model to group tickers into attention clusters (e.g. "AI
    infrastructure", "GLP-1 weight-loss").
  - Trend volatility: rolling 28-day std-dev of search interest per
    ticker as an attention-instability signal.
  - Compare_list pairwise pulls (5 terms at a time) for normalized
    cross-ticker ranking.
- Literature: Preis, Moat, Stanley (Nature SR 2013), Bijl et al. 2016,
  Abdi et al. 2024 — https://www.sciencedirect.com/science/article/pii/S1057521923000650

### 7.10 Hiring signals (Tier B, via licensed vendor; Tier C if scraped)

- Primary path: licensed vendor (Revelio Labs COSMOS, Thinknum). LinkedIn
  Economic Graph public workforce reports are free snapshots, not
  granular API data.
- Do NOT scrape LinkedIn directly — despite hiQ's Van Buren win, LinkedIn
  prevailed on contract grounds in 2022 (permanent injunction + $500k)
  so contract risk persists.
- Indeed scraping: also ToS-prohibited; stick to Indeed Hiring Lab
  aggregate data (https://www.hiringlab.org/).
- Features: net_hiring (postings - closings), seniority mix, team growth
  (sales vs engineering vs G&A), skill-demand z-score.
- Literature: Green et al. 2022 JAR, Revelio Labs white papers, Integrity
  Research 2024 coverage (https://www.integrity-research.com/revelio-labs-unveils-rpls-a-bold-alternative-to-bls-in-turbulent-times/).

### 7.11 Scraper infrastructure

- One shared HTTP client (`backend/alt_data/http.py`) with:
  - robots.txt fetcher + enforcement (reuse `urllib.robotparser`).
  - rotating residential proxy pool (vendor: Bright Data or
    ScrapingBee — both are Tier B licensed services).
  - exponential backoff with jitter, 429 / 503 handling.
  - per-domain rate-limit token bucket persisted in Redis so parallel
    workers don't burst.
  - every request logged to `pyfinagent_data.scraper_audit_log`.
- Header hygiene: do not forge login cookies; do not bypass CAPTCHAs;
  do not access content behind auth.
- User-Agent identifies pyfinagent + contact email (good web
  citizenship + reduces adversarial response from site operators).
- Cache: store raw HTML gzipped in GCS `alt-data-raw/{source}/{yyyy}/{mm}/{dd}/`
  with 13-month retention, so we can re-parse without re-fetching.

### 7.12 Feature integration & evaluation

- Add `backend/alt_data/features.py` that normalizes each source into
  the shared `(symbol, as_of_date, source, signal_name, value)` schema.
- Register features with Layer-1 enrichment prompts
  (`backend/agents/skills/*.md`) so Gemini agents can reference them.
- Add one MAS agent `AltDataAnalyst` in `agent_definitions.py` that
  triages which alt-data signals are active for a given ticker.
- Backtest harness runs the standard IC + t-stat suite; results written
  to `backend/backtest/experiments/results/alt_data_ic_YYYYMMDD.tsv`
  and appended to `quant_results.tsv`.

## Research findings

### Legal groundwork

Van Buren v. United States, 593 U.S. ___ (2021) narrowed CFAA
"exceeds authorized access" to a gates-up/gates-down test; scraping
public websites does not violate the CFAA. See
https://supreme.justia.com/cases/federal/us/593/19-783/ and EFF analysis
https://www.eff.org/deeplinks/2021/07/eff-ninth-circuit-recent-supreme-court-decision-van-buren-does-not-criminalize-web.

hiQ Labs v. LinkedIn resolved in December 2022 with a permanent
injunction against hiQ plus $500k damages on contract grounds — the CFAA
win does NOT immunize scrapers from breach-of-contract claims. Zwillgen
summary: https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/.

X Corp. v. Bright Data (N.D. Cal. 2024) is the most scraper-friendly
recent ruling: the Copyright Act preempts platform breach-of-contract
claims over user-generated content, because the platform only holds a
non-exclusive license. https://newmedialaw.proskauer.com/2024/05/14/california-court-issues-another-noteworthy-decision-dismissing-breach-of-contract-and-tort-claims-in-web-scraping-dispute/

Copyright-side: Associated Press v. Meltwater (S.D.N.Y. 2013) says
scraping + redistributing substantial portions of copyrighted content
is not fair use. So the strategy is: scrape numeric signals, summaries,
and aggregates — never republish full content.
https://www.copyright.gov/fair-use/summaries/ap-meltwater-sdny2013.pdf

### Alt-data effectiveness studies

- Congressional trading: Ziobrowski et al. 2004/2011 (positive alpha for
  Senate/House) https://digitalcommons.lindenwood.edu/faculty-research-papers/240/
  vs. Eggers & Hainmueller 2013 and Belmont et al. 2022 (null/negative
  results) https://www.sciencedirect.com/science/article/abs/pii/S0047272722000044
  vs. Wei 2024 NBER WP #34524 (committee-weighted positive alpha).
- 13F / institutional: WhaleWisdom data product
  (http://legacy.whalewisdom.com/shell/api_help), Dakota comparison of
  providers https://www.dakota.com/resources/blog/whalewisdom-opportunity-hunter-sec-api-which-is-right-for-you.
- FINRA short volume methodology — FINRA Information Notice 05/10/19
  https://www.finra.org/rules-guidance/notices/information-notice-051019
  — clarifies how to use daily short-sale volume (not equal to short
  interest).
- r/WSB sentiment:
  - Bradley, Hanousek, Jame, Xiao 2022 http://russelljame.com/wsb_3_15_2022.pdf
  - Buz & de Melo 2023 https://arxiv.org/abs/2301.00170
  - Semenova & Winkler 2021 arXiv 2104.01847 (social contagion)
  - ACM TSC 2024 https://dl.acm.org/doi/10.1145/3660760
  - Finance Research Letters 2024 https://www.sciencedirect.com/science/article/pii/S1057521924006537
- Employee sentiment: Green, Huang, Wen, Zhou (JFE 2019)
  https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300662
  and Mary Becker 2023 working paper https://economics.appstate.edu/sites/default/files/zachary_mcgurk_-_paper.pdf.
- Satellite: UC Berkeley Haas 2024 coverage of parking-lot alpha
  https://newsroom.haas.berkeley.edu/how-hedge-funds-use-satellite-images-to-beat-wall-street-and-main-street/.
- Google Trends: Preis et al. Nature Sci. Rep. 2013 and 2023 review
  https://www.sciencedirect.com/science/article/pii/S1057521923000650.
- Hiring: Revelio Labs COSMOS / WRDS
  https://wrds-www.wharton.upenn.edu/pages/data-announcements/new-revelio-labs-data/.

### Vendor & API documentation

- QuiverQuant API docs: https://api.quiverquant.com/docs/
- Quiver Python SDK: https://github.com/Quiver-Quantitative/python-api
- Unusual Whales API: https://api.unusualwhales.com/docs
- Financial Modeling Prep 13F: https://site.financialmodelingprep.com/developer/docs/form-13f-api/
- edgartools PyPI: https://pypi.org/project/edgartools/
- sec-api-python: https://github.com/SEC-API-io/sec-api-python
- pytrends: https://github.com/GeneralMills/pytrends
- PRAW (Reddit): https://praw.readthedocs.io/
- Apify Capitol Trades Scraper: https://apify.com/saswave/capitol-trades-scraper/api

### ToS / licensing citations

- Reddit API commercial pricing ($0.24 / 1000 calls, approval required):
  https://sellbery.com/blog/how-much-does-the-reddit-api-cost-in-2025/
- Twitter/X API tiers: https://twitterapi.io/blog/twitter-api-pricing-2025
- Glassdoor Terms of Use (no scraping): https://www.glassdoor.com/about/terms/
- LinkedIn User Agreement (no scraping, contract basis survives):
  see hiQ wrap-up https://www.privacyworld.blog/2022/12/linkedins-data-scraping-battle-with-hiq-labs-ends-with-proposed-judgment/

## Recommended integration order

Order balances legal risk (low first), signal strength (high first), and
cost (free first).

1. FINRA short volume (7.3) — Tier A, free, trivial infra.
2. Congressional trades via EDGAR + disclosures-clerk + efdsearch (7.1) —
   Tier A, free.
3. 13F via EDGAR (7.2) — Tier A, free.
4. Google Trends extended (7.9) — Tier A, free, already have pytrends.
5. r/WSB sentiment via Reddit API research tier (7.5) — Tier B, free for
   research, escalate to commercial before live trading.
6. ETF flows via Finnhub free + Nasdaq Data Link paid (7.4) — Tier B.
7. QuiverQuant + Unusual Whales redundant ingestion (7.1, 7.6 options
   context) — Tier B paid; replaces scrape only if budget approved.
8. X/Twitter sentiment via TwitterAPI.io metered (7.6) — Tier B paid.
9. Revelio Labs hiring + employee signals (7.7, 7.10) — Tier B paid,
   DEFER until budget approved.
10. Satellite (7.8) — DEFER to Phase 8.

## Backfill strategy

| Source | Backfill horizon | Mechanism |
|--------|------------------|-----------|
| FINRA short-vol | 10+ years | FINRA bulk daily files |
| Congress trades | Senate ~6y, House ~5y | HTML + PDF bulk download, archived to GCS |
| 13F | 1999-present | SEC EDGAR XBRL bulk |
| ETF flows | 10+ years | Nasdaq Data Link ETFF database |
| r/WSB | 2018-present | Academic Reddit data export (licensed) |
| Google Trends | 2004-present | pytrends + de-noise known throttling gaps |
| Twitter | 7-day via basic / 2-year via Apify | Limited free backfill; plan forward-only prod |
| Revelio | 2008-present | Vendor bulk dump, delivered via WRDS |
| Satellite | 2015-present | Vendor-dependent |

## Rate-limit + IP-rotation strategy for scraping-required sources

- Central token-bucket rate limiter in Redis keyed by `(domain, proxy_ip)`.
  Defaults: 1 rps per IP, 5 rps per domain total.
- Residential proxies via Bright Data's licensed pool (avoid data-center
  IPs — they are trivially fingerprinted).
- User-Agent = `pyfinagent/1.0 (+contact@pyfinagent.io)` — identifying
  and offering a contact email is both a good-faith signal and
  reduces the chance of CFAA claims, because it shows no deception.
- Honor robots.txt via `urllib.robotparser.RobotFileParser.can_fetch()`
  before every new path; cache per-domain 24h.
- 429 / 503: exponential backoff 2^n + jitter up to 15 minutes, then
  park the proxy for an hour.
- CAPTCHAs: HARD STOP. Route to the paid API or abandon the source.
- Daily scraper audit row in `pyfinagent_data.scraper_audit_log`:
  `(ts, source, url_hash, proxy_region, status_code, bytes_out,
  robots_allowed, sleep_ms)`.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-7",
  "name": "Alt-Data & Scraping Expansion",
  "status": "pending",
  "depends_on": ["phase-5.5", "phase-6"],
  "steps": [
    {
      "id": "7.0",
      "name": "Compliance & legal foundation",
      "status": "pending",
      "verification": [
        "test -f docs/compliance/alt-data.md",
        "grep -q 'Van Buren' docs/compliance/alt-data.md",
        "grep -q 'hiQ' docs/compliance/alt-data.md",
        "grep -q 'X Corp' docs/compliance/alt-data.md"
      ]
    },
    {
      "id": "7.1",
      "name": "Congressional trades ingestion",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/congress.py').read())\"",
        "bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM pyfinagent_data.alt_congress_trades WHERE as_of_date >= CURRENT_DATE() - 30' | tail -n 1 | awk '{ exit ($1 > 100 ? 0 : 1) }'"
      ]
    },
    {
      "id": "7.2",
      "name": "13F institutional holdings ingestion",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/f13.py').read())\"",
        "bq ls pyfinagent_data | grep -q alt_13f_holdings"
      ]
    },
    {
      "id": "7.3",
      "name": "FINRA short-volume ingestion",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/finra_short.py').read())\"",
        "bq ls pyfinagent_data | grep -q alt_finra_short_volume"
      ]
    },
    {
      "id": "7.4",
      "name": "ETF flows ingestion",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/etf_flows.py').read())\""
      ]
    },
    {
      "id": "7.5",
      "name": "Reddit WSB sentiment ingestion",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/reddit_wsb.py').read())\"",
        "test -f docs/compliance/reddit-license.md"
      ]
    },
    {
      "id": "7.6",
      "name": "Twitter/X sentiment ingestion",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/twitter.py').read())\""
      ]
    },
    {
      "id": "7.7",
      "name": "Employee sentiment via licensed vendor",
      "status": "pending",
      "verification": [
        "test -f docs/compliance/revelio-license.md"
      ]
    },
    {
      "id": "7.8",
      "name": "Satellite/geospatial proxies (deferred)",
      "status": "deferred",
      "verification": [
        "grep -q 'Phase 8' docs/compliance/alt-data.md"
      ]
    },
    {
      "id": "7.9",
      "name": "Google Trends extended features",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/google_trends.py').read())\""
      ]
    },
    {
      "id": "7.10",
      "name": "Hiring signals via licensed vendor",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/hiring.py').read())\""
      ]
    },
    {
      "id": "7.11",
      "name": "Shared scraper infrastructure",
      "status": "pending",
      "verification": [
        "python -c \"import ast; ast.parse(open('backend/alt_data/http.py').read())\"",
        "bq ls pyfinagent_data | grep -q scraper_audit_log"
      ]
    },
    {
      "id": "7.12",
      "name": "Feature integration & IC evaluation",
      "status": "pending",
      "verification": [
        "test -f backend/alt_data/features.py",
        "ls backend/backtest/experiments/results/alt_data_ic_*.tsv | head -n 1"
      ]
    }
  ]
}
```

## References

Case law and legal analysis:

- https://supreme.justia.com/cases/federal/us/593/19-783/
- https://www.eff.org/deeplinks/2021/07/eff-ninth-circuit-recent-supreme-court-decision-van-buren-does-not-criminalize-web
- https://en.wikipedia.org/wiki/HiQ_Labs_v._LinkedIn
- https://www.zwillgen.com/alternative-data/hiq-v-linkedin-wrapped-up-web-scraping-lessons-learned/
- https://www.privacyworld.blog/2022/12/linkedins-data-scraping-battle-with-hiq-labs-ends-with-proposed-judgment/
- https://newmedialaw.proskauer.com/2024/05/14/california-court-issues-another-noteworthy-decision-dismissing-breach-of-contract-and-tort-claims-in-web-scraping-dispute/
- https://ipandmedialaw.fkks.com/post/102j7d0/blockbuster-ruling-federal-court-holds-that-copyright-act-preempts-xs-web-scrap
- https://www.socialmediatoday.com/news/x-formerly-twitter-loses-lawsuit-against-data-scrapers/715868/
- https://www.copyright.gov/fair-use/summaries/ap-meltwater-sdny2013.pdf
- https://calawyers.org/privacy-law/ninth-circuit-holds-data-scraping-is-legal-in-hiq-v-linkedin/
- https://newmedialaw.proskauer.com/2022/04/21/taking-cue-from-the-supreme-courts-van-buren-decision-ninth-circuit-releases-new-opinion-holding-scraping-of-publicly-available-website-data-falls-outside-of-cfaa/

ToS and vendor documentation:

- https://www.glassdoor.com/about/terms/
- https://sellbery.com/blog/how-much-does-the-reddit-api-cost-in-2025/
- https://twitterapi.io/blog/twitter-api-pricing-2025
- https://api.quiverquant.com/docs/
- https://api.unusualwhales.com/docs
- https://github.com/Quiver-Quantitative/python-api
- http://legacy.whalewisdom.com/shell/api_help
- https://site.financialmodelingprep.com/developer/docs/form-13f-api/
- https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files
- https://www.finra.org/rules-guidance/notices/information-notice-051019
- https://data.nasdaq.com/databases/ETFF
- https://etfdb.com/
- https://finnhub.io/docs/api/etfs-holdings
- https://www.reveliolabs.com/public-labor-statistics/employment/
- https://wrds-www.wharton.upenn.edu/pages/data-announcements/new-revelio-labs-data/
- https://economicgraph.linkedin.com/
- https://www.hiringlab.org/
- https://efdsearch.senate.gov/
- https://disclosures-clerk.house.gov/
- https://www.sec.gov/edgar
- https://13f.info/
- https://www.dataroma.com/m/home.php
- https://apify.com/saswave/capitol-trades-scraper/api
- https://pypi.org/project/edgartools/
- https://github.com/SEC-API-io/sec-api-python
- https://github.com/GeneralMills/pytrends
- https://praw.readthedocs.io/
- https://twitterapi.io/
- https://sociavault.com/blog/twitter-api-alternative-2025

Alt-data effectiveness literature:

- https://digitalcommons.lindenwood.edu/faculty-research-papers/240/
- https://www.sciencedirect.com/science/article/abs/pii/S0047272722000044
- https://www.nber.org/system/files/working_papers/w34524/w34524.pdf
- http://russelljame.com/wsb_3_15_2022.pdf
- https://arxiv.org/abs/2301.00170
- https://arxiv.org/abs/2104.01847
- https://dl.acm.org/doi/10.1145/3660760
- https://www.sciencedirect.com/science/article/pii/S1057521924006537
- https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300662
- https://economics.appstate.edu/sites/default/files/zachary_mcgurk_-_paper.pdf
- https://alphaarchitect.com/employee-satisfaction-and-stock-returns/
- https://newsroom.haas.berkeley.edu/how-hedge-funds-use-satellite-images-to-beat-wall-street-and-main-street/
- https://www.sciencedirect.com/science/article/pii/S1057521923000650
- https://www.iccs-meeting.org/archive/iccs2025/papers/159090292.pdf
- https://arxiv.org/pdf/2507.22922
- https://www.integrity-research.com/revelio-labs-unveils-rpls-a-bold-alternative-to-bls-in-turbulent-times/
- https://www.dakota.com/resources/blog/whalewisdom-opportunity-hunter-sec-api-which-is-right-for-you
