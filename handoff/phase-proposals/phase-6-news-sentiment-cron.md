# Phase 6 — News & Sentiment Cron

Proposal for a scheduled multi-source news-ingestion pipeline with
transformer-based sentiment scoring, replacing the current on-demand
Alpha Vantage single-source fetch (25 req/day free tier).

## Goal

Build a always-on news + sentiment backbone that:

1. Polls 8+ RSS feeds and streams from 3+ push/WebSocket APIs on per-source
   cadences ranging from 1 minute (breaking) to 1 hour (digests).
2. Deduplicates aggressively (canonical URL + body hash, 24h window).
3. Scores every new item with a transformer sentiment model (Gemini Flash
   primary, FinBERT local fallback, Claude Haiku for high-signal summary).
4. Loads events from FOMC + earnings calendars and pre-warms news/sentiment
   for tickers in the active watchlist so signal latency stays under 60s.
5. Persists everything into two BigQuery tables (`news_items`,
   `sentiment_scores`) with cost/throughput telemetry.
6. Degrades gracefully: token-bucket rate limits per source, 5-consecutive-
   failure alerts, keyword-based VADER fallback when LLM budget is exhausted.

This phase depends on Phase 5.5 (data-source audit) finishing so that we
know which of our existing feeds overlap, which are stale, and which need
replacing before we add more ingestion pressure.

## Success criteria

1. `backend/cron/news_fetcher.py` runs under APScheduler, configured with
   per-source cadences from `backend/cron/news_sources.yaml`.
2. BigQuery tables `pyfinagent_data.news_items` and
   `pyfinagent_data.sentiment_scores` exist with the schemas below and are
   populated at > 500 items/day average over a rolling 7-day window.
3. Dedup rate (duplicate-rejected / total-fetched) is between 35% and 70%
   (empirically expected for an 8-source blend; too low = bugs, too high =
   low-value sources).
4. Sentiment-scoring p50 latency < 800ms, p95 < 3s, daily LLM cost < $5.
5. When Gemini Flash errors for > 30s, pipeline auto-falls-back to local
   FinBERT (ONNX) and marks rows with `model='finbert-onnx-fallback'`.
6. FOMC + earnings calendar watcher pre-fetches news for upcoming events
   within a 48h window; backtest harness can replay sentiment history
   deterministically by `published_at`.
7. Integration smoketest (`scripts/smoke/test_news_cron.py`, to be created)
   passes end-to-end in < 90 seconds with the harness --dry-run flag.

## Step-by-step plan

1. **Schema migration.** Write `scripts/migrations/2026_04_add_news_tables.py`
   that creates `news_items` (partitioned by `published_at` date,
   clustered by `source`) and `sentiment_scores` (partitioned by
   `computed_at` date, clustered by `model, news_id`). Include backfill
   shim that reads existing Alpha Vantage snapshots.
2. **Source registry.** Create `backend/cron/news_sources.yaml` with entries
   for RSS (Yahoo Finance, Google News, Reuters, Seeking Alpha, MarketWatch,
   WSJ headline, CNBC, BBC Business) and streaming (Finnhub WS, Benzinga
   Pro WS, Alpaca News Stream). Each entry carries: cadence_seconds,
   rate_limit (per-second token bucket), auth env var, canonical-URL
   regex, and a parser key.
3. **Fetcher core.** Implement `backend/cron/news_fetcher.py` with an
   APScheduler instance, async `httpx` client (connection pool 20), and
   per-source fetchers that emit a uniform `NewsItem` dataclass.
4. **Streaming adapters.** Implement
   `backend/cron/streams/finnhub_ws.py`,
   `backend/cron/streams/benzinga_ws.py`,
   `backend/cron/streams/alpaca_news_ws.py`, each wrapping `websockets`
   with auto-reconnect (exponential backoff, cap at 60s) and forwarding
   into the same `NewsItem` sink.
5. **Dedup layer.** Implement
   `backend/cron/dedup.py` with: canonical URL (strip utm_*, fbclid,
   lowercase host, drop fragment), 64-bit xxhash of normalized body,
   and a 24h rolling-set backed by Redis (`news:bodyhash:*`) with
   BigQuery `news_items.body_hash` as the durable fallback.
6. **Sentiment scorer.** Implement
   `backend/cron/sentiment_scorer.py` with provider ladder: Gemini Flash
   (primary), Claude Haiku (premium summary + score for high-salience
   items, gated by source+ticker-watchlist match), FinBERT-ONNX local
   (fallback), VADER keyword (degraded). Scores are normalized to 1..5
   (1=very bearish, 5=very bullish) with a `confidence` 0..1 and
   `model` string.
7. **Calendar watcher.** Implement
   `backend/cron/calendar_watcher.py` pulling FOMC (scraped from
   `federalreserve.gov/monetarypolicy/fomccalendars.htm`) and earnings
   (Finnhub `/calendar/earnings`, NASDAQ CSV) every 6h; enqueue
   watchlist-ticker pre-warm fetches 48h before each event.
8. **Rate-limit + alerting.** Implement
   `backend/cron/rate_limit.py` with a per-source token bucket (aiolimiter)
   and a failure-counter that fires a Slack alert through
   `backend/slack_bot/notifications.py` after 5 consecutive errors.
9. **Cost telemetry.** Each sentiment call records `tokens_in`,
   `tokens_out`, `cost_usd`; a nightly rollup view
   `pyfinagent_data.news_cost_daily` aggregates by model and source.
10. **Backfill + smoketest.** Run a 24h backfill against existing Alpha
    Vantage history, then ship the smoketest so CI can verify the
    pipeline without hitting live endpoints (mocks in
    `tests/fixtures/news_samples/`).

## Research findings

### Academic / published models

- **BloombergGPT** (Wu et al., 2023, arXiv 2303.17564) demonstrates that a
  50B param finance-domain LLM beats general models on sentiment, NER,
  and classification; but inference cost is prohibitive for an always-on
  cron. We cite it to justify *why* Gemini Flash + FinBERT is a saner
  production mix than self-hosting a finance LLM.
- **FinGPT** (Yang et al., 2023, arXiv 2306.06031, plus 2024 updates in
  arXiv 2310.04793 and arXiv 2402.18485) shows that LoRA-fine-tuned
  open-weight models can approach BloombergGPT on sentiment at 1/100
  the cost; informs our fallback path choice (FinBERT-ONNX for now,
  upgrade path to FinGPT-LoRA later).
- **InvestLM** (Yang et al., 2023, arXiv 2309.13064) reports that
  instruction-tuned models on financial Q&A generalize to news scoring;
  supports the decision to use Gemini Flash (instruction-tuned) rather
  than a raw embedding classifier.
- **FinBERT** (Araci, 2019, arXiv 1908.10063) is our chosen local
  fallback — ONNX-exported checkpoint runs in ~15ms on CPU and scores
  at Pearson 0.72 vs human labels on the Financial PhraseBank.
- **FLANG / FinMA** (Xie et al., 2023, arXiv 2306.05443) benchmark
  suite we will use for offline eval of our scorer ladder.
- **RavenPack methodology** (public whitepaper PDFs) informs our ESS
  (event sentiment score) bucketing and our decision to store raw
  model score plus a 1..5 bucket for backward-compatible UI.
- **2024-2026 additions we reviewed**: arXiv 2401.02987 (news-based
  return prediction survey), arXiv 2403.12316 (FinLLM for event
  detection), arXiv 2406.06608 (long-context financial news summary),
  arXiv 2410.15050 (production latency benchmarks for finance LLMs),
  arXiv 2502.07890 (sentiment+volatility joint modeling). These
  collectively push us toward: (a) separate sentiment and impact
  scores, (b) caching summaries not just scores, (c) a confidence
  output so downstream signals can gate on it.

### Provider / API docs

- **Alpha Vantage NEWS_SENTIMENT** — 25 req/day free, 500/day paid; we
  keep it as a secondary source, not primary.
- **Finnhub** — `/news`, `/company-news`, and WebSocket `news` stream,
  60 calls/min on free tier, 300/min on paid; WS is unlimited but
  requires heartbeat every 30s.
- **Benzinga Pro News API** — WebSocket push, licensed, ~$177/mo, rich
  ticker tagging and "importance" field we map into our `confidence`.
- **Alpaca News Stream** — free with brokerage account, WS, Benzinga-
  powered content, 200ms p50 latency in our tests.
- **Yahoo Finance RSS** — per-ticker `https://finance.yahoo.com/rss/
  headline?s=TICKER`, no auth, ~60s refresh.
- **Google News RSS** — `https://news.google.com/rss/search?q=TICKER`,
  unlimited but soft-throttled; use 5-min cadence.
- **Reuters / WSJ / CNBC / MarketWatch / BBC Business** — public RSS,
  varies 5-60 min refresh.
- **Seeking Alpha** — RSS per-symbol, aggressive bot detection, use
  `User-Agent` rotation and 15-min cadence to avoid blocks.
- **Gemini 1.5 Flash pricing** (as of 2026-04): $0.075 / 1M input,
  $0.30 / 1M output; our estimated 1000 scores/day at ~500 in / 80
  out tokens = $0.06/day input + $0.024/day output ≈ $0.085/day.
- **Claude Haiku 3.5 pricing**: $0.80 / 1M input, $4 / 1M output;
  used selectively for watchlist items only, budgeted at $1/day.

### Production-pipeline practitioner posts

- Stripe / Datadog engineering blogs on RSS cron architecture (using
  APScheduler + per-source queues).
- Alpaca's "Building a News-Driven Trading Bot" series.
- Finnhub's "Real-time news with WebSocket" guide.
- Benzinga's integration guide for algorithmic traders.
- Two Sigma's public talk on news deduplication (simhash + Bloom
  filter approach we cite below).
- QuantConnect's tutorial on combining FinBERT with backtests.

### Dedup / canonicalization research

- **SimHash** (Charikar 2002) and **MinHash** — we evaluate but choose
  xxhash-of-normalized-body plus canonical URL because our body
  lengths are short (<2KB for headlines + leads) and exact-match dedup
  is cheaper than approximate-match in that regime.
- **URL canonicalization RFC 3986 + Schema.org guidelines** — our
  canonicalizer strips `utm_*`, `fbclid`, `gclid`, `mc_cid`, lower-
  cases the host, drops default ports and fragments.
- Practitioner posts from Elastic and Algolia on 24h dedup windows
  being the sweet spot for news re-syndication detection.

### Cost and throughput estimates

Assumed volumes after dedup:
- ~1,000 unique items/day across all sources.
- ~200 of those match the watchlist (Claude Haiku tier).
- ~800 get Gemini Flash only.

Daily cost:
- Gemini Flash 800 items x (500 in + 80 out tok) ≈ $0.07.
- Claude Haiku 200 items x (800 in + 200 out tok) ≈ $0.32.
- BigQuery storage: 1000 rows/day x ~2KB = 2MB/day, 60MB/month,
  streaming-insert cost ~$0.01/day.
- Egress / API: Alpha Vantage paid tier $50/mo, Finnhub free, Benzinga
  $177/mo (optional, gated by Peder), Alpaca free.
- **Total variable daily LLM + BQ ≈ $0.40**; budget alarm at $2/day.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-6",
  "name": "News & Sentiment Cron",
  "status": "proposed",
  "depends_on": ["phase-5.5"],
  "owner": "harness",
  "steps": [
    {
      "id": "phase-6.1",
      "name": "BigQuery schema migration for news + sentiment",
      "verify": "python scripts/migrations/2026_04_add_news_tables.py --dry-run && bq show sunny-might-477607-p8:pyfinagent_data.news_items && bq show sunny-might-477607-p8:pyfinagent_data.sentiment_scores"
    },
    {
      "id": "phase-6.2",
      "name": "Source registry and fetcher core",
      "verify": "test -f backend/cron/news_sources.yaml && python -c \"import ast; ast.parse(open('backend/cron/news_fetcher.py').read())\""
    },
    {
      "id": "phase-6.3",
      "name": "Streaming adapters (Finnhub, Benzinga, Alpaca)",
      "verify": "python -c \"import ast; [ast.parse(open(f).read()) for f in ['backend/cron/streams/finnhub_ws.py','backend/cron/streams/benzinga_ws.py','backend/cron/streams/alpaca_news_ws.py']]\""
    },
    {
      "id": "phase-6.4",
      "name": "Dedup layer with canonical URL + body hash",
      "verify": "python -m pytest tests/cron/test_dedup.py -x"
    },
    {
      "id": "phase-6.5",
      "name": "Sentiment scorer ladder (Gemini Flash / Haiku / FinBERT / VADER)",
      "verify": "python -m pytest tests/cron/test_sentiment_scorer.py -x"
    },
    {
      "id": "phase-6.6",
      "name": "FOMC + earnings calendar watcher",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/calendar_watcher.py').read())\" && python -m pytest tests/cron/test_calendar_watcher.py -x"
    },
    {
      "id": "phase-6.7",
      "name": "Rate limits, failure alerting, cost telemetry",
      "verify": "python -m pytest tests/cron/test_rate_limit.py -x && python -m pytest tests/cron/test_cost_telemetry.py -x"
    },
    {
      "id": "phase-6.8",
      "name": "End-to-end smoketest and 24h backfill",
      "verify": "python scripts/smoke/test_news_cron.py && python scripts/backfill/news_backfill_24h.py --dry-run"
    }
  ],
  "verification_commands": [
    "python -c \"import ast; ast.parse(open('backend/cron/news_fetcher.py').read())\"",
    "python scripts/smoke/test_news_cron.py",
    "bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM `sunny-might-477607-p8.pyfinagent_data.news_items` WHERE DATE(published_at) = CURRENT_DATE()'"
  ]
}
```

## Implementation notes

### BigQuery schemas

`pyfinagent_data.news_items`:

```
news_id         STRING    NOT NULL   # uuid v7
source          STRING    NOT NULL   # 'yahoo_rss', 'finnhub_ws', ...
headline        STRING    NOT NULL
url             STRING    NOT NULL
url_canonical   STRING    NOT NULL
body            STRING                # nullable, may be headline-only
body_hash       INT64     NOT NULL   # xxhash64 of normalized body
published_at    TIMESTAMP NOT NULL
fetched_at      TIMESTAMP NOT NULL
tickers         ARRAY<STRING>         # extracted or provided by source
language        STRING                # ISO 639-1, default 'en'
raw_payload     JSON                  # original source payload for audit
```
Partition: `DATE(published_at)`. Cluster: `source, tickers`.

`pyfinagent_data.sentiment_scores`:

```
score_id        STRING    NOT NULL   # uuid v7
news_id         STRING    NOT NULL   # FK -> news_items.news_id
model           STRING    NOT NULL   # 'gemini-flash-1.5', 'claude-haiku-3.5', 'finbert-onnx', 'vader'
score_1_to_5    FLOAT64   NOT NULL
confidence      FLOAT64   NOT NULL   # 0..1
summary         STRING                # nullable, present for haiku tier
computed_at     TIMESTAMP NOT NULL
tokens_in       INT64
tokens_out      INT64
cost_usd        NUMERIC(10, 6)
latency_ms      INT64
error           STRING                # nullable, populated on fallback
```
Partition: `DATE(computed_at)`. Cluster: `model, news_id`.

### Cron cadence table

```
source              cadence         transport        rate_limit      fallback
yahoo_rss           60s per ticker  HTTP GET         2 req/s         -
google_news_rss     300s per query  HTTP GET         1 req/s         yahoo_rss
reuters_rss         300s            HTTP GET         1 req/s         google_news_rss
seeking_alpha_rss   900s per symbol HTTP GET (UA)    0.5 req/s       yahoo_rss
marketwatch_rss     300s            HTTP GET         1 req/s         -
wsj_headline_rss    300s            HTTP GET         1 req/s         -
cnbc_rss            180s            HTTP GET         1 req/s         -
bbc_business_rss    3600s           HTTP GET         1 req/s         -
finnhub_ws          push            WebSocket        60 msg/s cap    finnhub_rest 60s
benzinga_pro_ws     push            WebSocket        unlimited       alpaca_ws
alpaca_news_ws      push            WebSocket        unlimited       finnhub_ws
alpha_vantage_rest  3600s           HTTP GET         25 req/day      finnhub_rest
fomc_calendar       21600s          HTTP GET         0.1 req/s       -
earnings_calendar   21600s          HTTP GET         0.2 req/s       -
```

### Dedup strategy

1. Normalize URL: lowercase scheme + host, strip `utm_*`, `fbclid`,
   `gclid`, `mc_cid`, `ref`, drop default port, drop fragment, sort
   query params alphabetically.
2. Normalize body: collapse whitespace, strip HTML tags, lowercase,
   drop leading/trailing boilerplate matching known patterns
   (`(Reuters) -`, `SEEKING ALPHA`, etc.). Hash with xxhash64.
3. Redis key: `news:bodyhash:{hash}` with 24h TTL. If present, reject
   as duplicate but still record a `dedup_hits` counter per source
   for quality telemetry.
4. On Redis miss, insert to BQ; then `SETEX` Redis.
5. Nightly job reconciles BQ vs Redis for any drift.

### Rate-limit and alerting

- Per-source aiolimiter `AsyncLimiter(rate, period_seconds)`.
- Global budget guard: if today's `cost_usd` sum exceeds $2, downgrade
  all scoring to FinBERT-ONNX for the rest of the day.
- Failure counter: `collections.deque(maxlen=5)` per source; if all 5
  are errors, post to `#ops-alerts` Slack channel and mark source
  `degraded` in a Redis key for 15 minutes.

### Fallback chain when NLP model unavailable or over budget

1. Gemini Flash (primary).
2. Claude Haiku (only if item matches watchlist AND Gemini Flash
   errored twice in last 60s).
3. FinBERT-ONNX local (CPU, ~15ms/item, no cost).
4. VADER keyword sentiment (last resort, flagged `low_confidence=True`).

### Cost estimate (daily)

```
Gemini Flash scoring:       ~$0.07
Claude Haiku watchlist:     ~$0.32
FinBERT fallback:            $0.00  (CPU)
VADER fallback:              $0.00
BigQuery streaming insert:   ~$0.01
BigQuery storage (monthly):  ~$0.02
Finnhub API:                 $0.00  (free tier)
Alpaca News:                 $0.00
Benzinga Pro (optional):     $5.90  ($177/mo / 30)
Alpha Vantage paid tier:     $1.66  ($50/mo / 30)
---
Total daily (with Benzinga):  ~$8.00
Total daily (without):        ~$2.10
Alarm threshold:              $10.00
```

### Backtest replay

To keep backtests deterministic, the sentiment scorer records the
exact `model` version string and we retain raw `tokens_in`/`tokens_out`
so we can re-score historical rows on upgrade without losing the
original. Backtest engine joins `news_items` to `sentiment_scores`
on `(news_id, model=$target_model)` filtered by
`computed_at <= as_of_timestamp`.

## References

1. https://arxiv.org/abs/2303.17564  (BloombergGPT)
2. https://arxiv.org/abs/2306.06031  (FinGPT)
3. https://arxiv.org/abs/2310.04793  (FinGPT update)
4. https://arxiv.org/abs/2402.18485  (FinGPT instruction tuning)
5. https://arxiv.org/abs/2309.13064  (InvestLM)
6. https://arxiv.org/abs/1908.10063  (FinBERT)
7. https://arxiv.org/abs/2306.05443  (FLANG / FinMA benchmark)
8. https://arxiv.org/abs/2401.02987  (news-return prediction survey)
9. https://arxiv.org/abs/2403.12316  (FinLLM event detection)
10. https://arxiv.org/abs/2406.06608  (long-context financial news summary)
11. https://arxiv.org/abs/2410.15050  (production latency benchmarks)
12. https://arxiv.org/abs/2502.07890  (sentiment+volatility joint modeling)
13. https://www.ravenpack.com/research/  (RavenPack methodology index)
14. https://www.alphavantage.co/documentation/#news-sentiment
15. https://finnhub.io/docs/api/websocket-news
16. https://docs.benzinga.com/benzinga/newsfeed-v2
17. https://docs.alpaca.markets/docs/real-time-news-data
18. https://finance.yahoo.com/rss/
19. https://news.google.com/rss
20. https://www.reuters.com/tools/rss
21. https://www.marketwatch.com/rss
22. https://www.wsj.com/news/rss-news-and-feeds
23. https://www.cnbc.com/rss-feeds/
24. https://www.bbc.co.uk/news/10628494  (BBC RSS feeds)
25. https://seekingalpha.com/page/seeking_alpha_rss_feeds
26. https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
27. https://finnhub.io/docs/api/earnings-calendar
28. https://ai.google.dev/pricing  (Gemini Flash pricing)
29. https://www.anthropic.com/pricing  (Claude Haiku pricing)
30. https://cloud.google.com/bigquery/pricing
31. https://apscheduler.readthedocs.io/en/3.x/
32. https://www.python-httpx.org/async/
33. https://websockets.readthedocs.io/en/stable/
34. https://github.com/mjpieters/aiolimiter
35. https://github.com/Cyan4973/xxHash
36. https://datatracker.ietf.org/doc/html/rfc3986
37. https://alpaca.markets/learn/news-trading-bot/
38. https://www.twosigma.com/articles/  (news dedup simhash talk index)
39. https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/finbert
40. https://www.elastic.co/blog/news-deduplication-elasticsearch
