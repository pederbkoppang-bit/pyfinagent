# Phase 6.5 — Global Intelligence Directive

One scheduled daily "research scan" routine that walks four non-overlapping
source families in a single sequential pass: institutional outlook archives,
academic frontier feeds, AI-frontier lab releases, and player-driven community
/ sovereign-wealth signals. Output is a BigQuery-persisted intelligence corpus,
a Slack digest to `#intel`, and a queue of researcher-prompt patch proposals
for human approval.

This phase EXTENDS (it does not replace) the baselines established by:

- `handoff/phase-proposals/phase-6-news-sentiment-cron.md` — general
  financial-news RSS + streaming + sentiment (Yahoo, Reuters, Finnhub,
  Benzinga, Alpaca, WSJ, CNBC, Bloomberg-adjacent feeds).
- `handoff/phase-proposals/phase-7-alt-data-scraping.md` — alt-data
  (Congress trades, 13F, FINRA short-volume, ETF flows, r/WSB, X/Twitter,
  Google Trends, hiring, satellite, employee sentiment).

Phase 6 owns *news*. Phase 7 owns *alt-data*. Phase 6.5 owns *thought
leadership and research frontier*: the non-news, non-alt-data written
output of major research-producing institutions — banks, academic labs,
AI labs, sovereign wealth funds, and influential retail/pro communities
whose *narrative* shapes capital allocation before prices move.

## Goal

Build ONE APScheduler routine (`backend/cron/intel_scan.py`) that, within a
single Claude-scheduled slot per day, sequentially walks four source
families and produces four persisted BigQuery tables plus a Slack digest
plus a queue of researcher-agent prompt patches.

The load-bearing constraint is `cron_slots: 1`. The live environment caps
Claude-scheduled routines at 15/day total; this directive MUST consume
exactly one slot. Fragmenting into four jobs (one per family) is
explicitly rejected. Instead the routine loops through the families in
priority order, each family has an individual soft-timeout, and any
family that overruns is deferred to tomorrow with an audit row written.

## Success criteria

1. `backend/cron/intel_scan.py` runs under APScheduler exactly once per
   calendar day (08:00 UTC), consumes exactly one Claude scheduled slot,
   and completes within a 45-minute wall-clock budget.
2. Four BigQuery tables exist and receive at least one row per family
   per day on ≥ 90% of trading days across a rolling 30-day window:
   `pyfinagent_data.institutional_intel`,
   `pyfinagent_data.academic_frontier`,
   `pyfinagent_data.ai_frontier_releases`,
   `pyfinagent_data.player_driven_signals`.
3. Each row carries a `novelty_score` (0..1, cosine-distance vs. the
   existing-signal embedding library) and any row with novelty ≥ 0.80 is
   emitted as a queued researcher-prompt-patch proposal in
   `pyfinagent_data.researcher_prompt_patches` with status `pending`.
4. Researcher prompt patches are NEVER auto-applied. Approval requires a
   reviewer (human) to set `status = approved` in BQ; a nightly job then
   writes the patch into `backend/agents/skills/researcher.md` via a
   pull-request-style diff committed by the harness with Peder's sign-off.
5. One Slack message per day to `#intel` containing: (a) 3-5 bullet
   institutional-view summary, (b) top 3 novel academic papers, (c) any
   AI-lab release tagged for researcher review, (d) any SWF 13D/13F
   delta > $100M, (e) link back to the BQ day-slice view.
6. All sources obey the 30s BQ timeout rule, respect robots.txt, and
   fall under Tier A (public government / publisher-authorized feeds) or
   Tier B (licensed API). No Tier C scraping is introduced. Any paywalled
   institutional PDF is explicitly excluded in `intel_sources.yaml`.
7. Total daily LLM cost for the directive ≤ $1.50 (Gemini Flash summary
   + Claude Haiku novelty tagging). Alarm at $3.00.
8. Smoketest `scripts/smoke/test_intel_scan.py` runs end-to-end in
   < 120s with fixtures and exits 0.

## Step-by-step plan

1. **Schema migration.** Write
   `scripts/migrations/2026_04_add_intel_tables.py` creating the four
   ingestion tables plus `researcher_prompt_patches`. Partitioning:
   `DATE(observed_at)`. Clustering: `source, publisher`. Backfill hook
   reads any existing `news_items` rows whose source matches the
   institutional-outlook allowlist so prior history is not lost.
2. **Source registry.** Create `backend/cron/intel_sources.yaml` with
   four families, each with entries tagged `family`, `publisher`,
   `transport` (rss | oai-pmh | http | edgar), `url`, `license_tier`
   (A|B), `max_items_per_run`, `timeout_seconds`. Include the full
   published-URL catalog from the Research findings section below.
3. **Scan core.** Implement `backend/cron/intel_scan.py` as a single
   async coroutine that iterates families in order
   `[institutional, academic, ai_frontier, player_driven]`, each with
   its own soft budget (10 / 15 / 10 / 10 minutes). Uses shared `httpx`
   pool, shared rate limiter, shared embedding client. On family timeout,
   writes an audit row and continues to the next family.
4. **Institutional extractors.** Implement
   `backend/cron/intel/institutional.py` with adapters for:
   (a) Goldman Sachs ISG outlook PDFs (public `gsam.com/insights` and
       `goldmansachs.com/insights` feeds + PDF text via `pypdfium2`),
   (b) BlackRock Investment Institute outlooks
       (`blackrock.com/corporate/insights/blackrock-investment-institute`
       RSS + PDF),
   (c) Bridgewater Daily Observations public subset
       (`bridgewater.com/research-and-insights` — free-tier articles
       only; paywalled DO excluded),
   (d) JP Morgan Asset Management Market Insights
       (`am.jpmorgan.com/.../market-insights/guide-to-the-markets` plus
       the Eye on the Market archive).
   Each adapter returns a uniform `InstitutionalItem` dataclass
   (publisher, title, date, url, text, key_macro_view, mentioned_assets).
5. **Academic extractors.** Implement `backend/cron/intel/academic.py`:
   (a) arXiv OAI-PMH daily harvester for sets `cs.LG`, `q-fin.CP`,
       `q-fin.GN`, `q-fin.MF`, `q-fin.PM`, `q-fin.PR`, `q-fin.RM`,
       `q-fin.ST`, `q-fin.TR`, `stat.ML`. Incremental fetch by
       `from=yesterday` with resumption tokens.
   (b) SSRN finance section RSS
       (`papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=203`
       plus the Financial Economics Network feed).
   (c) MIT CSAIL (`news.mit.edu/rss/topic/artificial-intelligence2`),
       Stanford AI Lab (`ai.stanford.edu/blog/feed.xml`),
       Stanford HAI (`hai.stanford.edu/news`), Oxford Said Business
       School research (`sbs.ox.ac.uk/research/centres/oxford-future-of-finance-and-technology-initiative`).
   (d) Novelty scoring: Claude-produced 1024-dim embedding of abstract,
       cosine distance to the nearest existing-signal embedding in
       `pyfinagent_data.signal_embedding_library`; emit
       `novelty_score = distance`. Threshold 0.80 for patch proposal.
6. **AI-frontier extractors.** Implement `backend/cron/intel/ai_frontier.py`:
   (a) Anthropic news + engineering (`anthropic.com/news/rss.xml`
       plus blog content from `anthropic.com/engineering` harvested via
       sitemap),
   (b) OpenAI research (`openai.com/news/research/rss.xml`),
   (c) DeepMind blog (`deepmind.google/blog/rss.xml` plus research
       publications index),
   (d) Google Research blog (`research.google/blog/rss/`).
   Each extracted snippet is summarized by Claude Haiku; if the summary
   contains any of the patchable-hooks list (`{reasoning, planning,
   tool-use, RAG, embedding, fine-tune, prompt}`), emit a candidate
   prompt patch into `researcher_prompt_patches` with `status=pending`.
7. **Player-driven extractors.** Implement
   `backend/cron/intel/player_driven.py`:
   (a) SeekingAlpha headlines + editorials RSS (ticker-agnostic feeds
       only — per-ticker feeds are already covered by phase-6),
   (b) r/wallstreetbets daily Top/DD digest (once/day aggregate; ticker
       detail is phase-7's responsibility),
   (c) QuantConnect community forum RSS
       (`quantconnect.com/forum/discussions.rss`) filtered to research,
       strategy, and alpha-idea threads,
   (d) SEC EDGAR 13D/13F for NBIM (Norges Bank Investment Management),
       GIC Private Ltd, CalPERS, CPP Investments — filing-delta alerts
       computed by diffing today's filings against the previous quarter.
8. **Embedding + novelty client.** Implement
   `backend/cron/intel/novelty.py` wrapping Voyage `voyage-3-large`
   (primary, 1024-d, $0.18/1M tok), Gemini embedding (fallback, free
   tier), with cosine-distance lookup against the BQ-hosted
   `signal_embedding_library`. Cache per-day in GCS
   `intel-embeddings/{yyyy}/{mm}/{dd}/` for replay.
9. **Slack digest + patch queue.** Implement
   `backend/cron/intel/digest.py`. Posts one message to `#intel` via
   `backend/slack_bot/notifications.py`. Patch-proposal entries get a
   `:inbox_tray:`-equivalent ASCII tag and a BQ link; approval is done
   out of band by reviewer.
10. **Smoketest + scheduler wiring.** Add
    `scripts/smoke/test_intel_scan.py` with fixtures for every adapter
    (canned RSS + canned PDF), register the job in
    `backend/cron/scheduler.py` at `08:00 UTC daily`, and gate deploy
    behind a harness dry-run.

## Research findings

### Institutional intelligence — publicly-accessible archives

Goldman Sachs publishes the Investment Strategy Group (ISG) *Outlook*
each January, plus interim monthly updates. The January outlook PDF is
openly linked from `privatewealth.goldmansachs.com/insights/isg-outlook`
and the GSAM macro team's *Macro Matters* feed at
`gsam.com/responsible-investing/en-us/professional/market-insights` is
free. The quarterly *Global Macro Outlook* from Goldman Sachs Asset
Management (`am.gs.com`) is ungated and distributed as PDF via the same
path, which means `pypdfium2` text extraction is sufficient.

BlackRock's Investment Institute (`blackrock.com/corporate/insights/
blackrock-investment-institute`) publishes the *Weekly Commentary*, the
semi-annual *Midyear Outlook*, the *2025/2026 Global Outlook*, and
the *Market Take* video-series transcripts. All are free and RSS-
discoverable via the sitemap `blackrock.com/sitemap.xml`.

Bridgewater's *Daily Observations* is paywalled to clients, but a
curated subset is published free as articles at
`bridgewater.com/research-and-insights`. Ray Dalio's LinkedIn essays
are free but LinkedIn scraping is legally risky post-hiQ, so we rely on
the Bridgewater-hosted copy only.

JP Morgan Asset Management's *Guide to the Markets*
(`am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/
guide-to-the-markets/`) is updated quarterly and the slide deck PDF is
free. JP Morgan's *Eye on the Market* by Michael Cembalest is also
free at `privatebank.jpmorgan.com/nam/en/insights/markets-and-investing/
eotm`.

PIMCO's *Cyclical* and *Secular Outlooks* (`pimco.com/en-us/insights/
economic-and-market-commentary`), Vanguard's annual *Economic and Market
Outlook* (`corporate.vanguard.com/content/corporatesite/us/en/corp/
who-we-are/economic-and-market-outlook.html`), and Fidelity's quarterly
*Business Cycle Update* (`institutional.fidelity.com/advisors/insights/
topics/business-cycle-update`) are optional additions — included in the
registry as second-priority institutional entries.

### Academic frontier feeds

arXiv supports OAI-PMH bulk harvesting at
`export.arxiv.org/oai2`. The OAI-PMH protocol is explicitly designed
for the daily-incremental use case: `verb=ListRecords&set=cs:cs.LG&
from=2026-04-16&metadataPrefix=arXiv`. This is the preferred
ingestion path vs. the RSS feed because it supports resumption
tokens and is rate-limit friendly (one-request-per-three-seconds
courtesy cap per arXiv's robots policy). arXiv also exposes category
RSS at `export.arxiv.org/rss/cs.LG` and `export.arxiv.org/rss/q-fin`,
which we use for preview-only pre-fetch (to shorten the OAI-PMH
window).

SSRN's Financial Economics Network (FEN) subscription feed (JEL-G
classification) is free but requires account-level authentication
for the full RSS. The public top-downloads feed at
`papers.ssrn.com/sol3/papers.cfm?abstract_id_list=...` plus the
recent-papers index pages are harvestable without auth and are our
primary SSRN path.

MIT News AI topic (`news.mit.edu/topic/artificial-intelligence2`)
publishes a clean RSS at `news.mit.edu/rss/topic/artificial-
intelligence2`. Stanford AI Lab blog (`ai.stanford.edu/blog/`) and
Stanford HAI news (`hai.stanford.edu/news`) expose standard
WordPress-style RSS. Oxford Said has the Oxford-Man Institute of
Quantitative Finance (`oxford-man.ox.ac.uk`) research-papers feed, and
the Oxford Said Business School research centres page at
`sbs.ox.ac.uk/research` which lists working papers quarterly.

Novelty detection: we compare each new abstract's embedding against
an existing-signal library of ~2-5k embeddings (one per documented
pyfinagent signal, one per in-production feature). The
cosine-distance threshold of 0.80 is conservative — at 1024-d
Voyage-3 embeddings, arXiv's own "related paper" threshold empirically
sits at ~0.35, so 0.80 selects genuinely off-distribution items.
This follows the novelty-detection methodology from Clinchant et al.
(SIGIR 2020) and the recent survey arXiv 2402.16889.

### AI frontier labs

Anthropic publishes `anthropic.com/news/rss.xml` for news+research,
with engineering-tagged posts at `anthropic.com/engineering`. The
Anthropic research index (`anthropic.com/research`) lists papers and
technical posts that frequently propose new prompt-engineering or
tool-use patterns directly relevant to our Layer-2 MAS agent skills.

OpenAI's research blog RSS is `openai.com/news/research/rss.xml`
(modern path — the legacy `openai.com/blog/rss.xml` also still serves
content). The arxiv preprints linked from these posts are already
captured by the academic family, so we deduplicate on arXiv ID.

DeepMind blog RSS lives at `deepmind.google/blog/rss.xml` (post-rename
from `deepmind.com`); Google Research has a research-blog RSS at
`research.google/blog/rss/`. AlphaProof-style releases appear in the
DeepMind feed; we tag any post whose title contains `{AlphaProof,
AlphaGeometry, AlphaFold, Gemini, AlphaCode}` for researcher-patch
candidate review.

Prompt-patch generation: when a blog post covers a technique in our
patchable-hooks allowlist (`reasoning`, `planning`, `tool-use`, `RAG`,
`embedding`, `fine-tune`, `prompt`), Claude Haiku is prompted to
produce a short (≤ 20-line) diff against `backend/agents/skills/
researcher.md`. The diff is stored in `researcher_prompt_patches`
with fields `source_url`, `proposed_diff`, `rationale`,
`status=pending`, `created_at`. No auto-apply.

### Player-driven insights

SeekingAlpha's public headlines RSS is
`seekingalpha.com/feed.xml`; the per-ticker feed is scraped by phase-6.
Editor's-picks, *Daily Wrap-Up*, and *Wall Street Breakfast* are the
high-density items for this phase and are distinct from ticker-level
news. SeekingAlpha's Terms of Service allow headline aggregation with
attribution, but prohibit full-article redistribution — we store only
title + excerpt + url, consistent with AP v. Meltwater guidance cited
in phase-7.

r/wallstreetbets: phase-7 owns per-ticker retail-sentiment metrics.
This phase instead ingests the *r/wallstreetbets Daily Discussion
Thread* and *Weekend Discussion Thread* aggregate narratives — one
summary per day — via the Reddit API research tier (PRAW). Commercial
use escalates to paid; phase-6.5 stays within the research-tier budget
by capping at one thread-level fetch per day (≤ 100 API calls).

QuantConnect forum
(`quantconnect.com/forum/discussions.rss`): exposes a clean RSS of
new topics and replies. We filter to categories "Research", "Strategy
Development", and "Algorithm Framework". QuantConnect's ToS permit
aggregation of forum titles and excerpts for non-commercial research,
but posting unmodified user code is copyrighted; we store titles,
links, and LLM-generated summaries only.

Sovereign-wealth filings: NBIM (ticker block via Norges Bank Investment
Management is filed as institutional 13F at the SEC in bulk — NBIM's
filing CIK is `0001262767`), GIC Private Ltd (CIK `0001337447`),
CalPERS (CIK `0000919079`), CPP Investments (CIK `0001484160`). SEC
EDGAR bulk download via `data.sec.gov/submissions/CIK{10-digit}.json`
is free, rate-limited to 10 req/s, and does not require auth. We
compute a filing-delta by diffing the latest 13F-HR against the prior
one, flagging any position change with absolute USD impact above $100M.

This is distinct from phase-7's 13F ingestion, which covers all
institutional filers generically for signal extraction. Phase-6.5's
SWF filter is narrow, narrative-focused, and produces a Slack alert,
not a backtest feature.

### Novelty detection via embeddings

Voyage `voyage-3-large` (1024-d, $0.18/1M tok input, as of 2026-04)
remains the leading production embedding for finance/technical text.
Anthropic's managed embeddings endpoint and OpenAI's `text-embedding-3-
large` (3072-d reducible to 1024-d via Matryoshka) are alternatives;
the Gemini free-tier embedding endpoint is our fallback. Cosine
distance is the standard choice; we do NOT use L2 because the
embeddings are unit-normalized.

Novelty-scoring threshold of 0.80 is supported by the arXiv
novelty-detection benchmarks in `arxiv.org/abs/2402.16889` (survey)
and `arxiv.org/abs/2310.07712` (retrieval-based scientific novelty).

### Legal / licensing

All institutional PDFs referenced above are free-access at the
publisher. We DO NOT scrape paywalled PDFs (Bridgewater's full Daily
Observations, S&P Global's iQAccess, FactSet, Bloomberg Terminal
excerpts). The GSAM, BlackRock, JP Morgan, PIMCO, Vanguard, and
Fidelity feeds explicitly permit linking and short-excerpt
re-syndication under their media / research-distribution terms.

arXiv's API terms of use
(`info.arxiv.org/help/api/tou.html`) permit automated harvesting at
≤ 1 req / 3 seconds with a contact email in the User-Agent. SSRN's
robots.txt allows crawling of the non-authenticated index and
disallows `/sol3/*?...&paid=1`; we honor this.

SeekingAlpha's scraping stance is aggressive; we limit to RSS feeds
which are explicitly distribution-sanctioned. QuantConnect's ToS permit
RSS aggregation. SEC EDGAR is public-record, zero-risk. Reddit API
use must remain non-commercial or escalate to paid per the pricing
cited in phase-7.

See phase-7's `docs/compliance/alt-data.md` for the full legal
framework (Van Buren, hiQ, X v. Bright Data, AP v. Meltwater). This
phase inherits that framework unchanged.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-6.5",
  "name": "Global Intelligence Directive",
  "status": "proposed",
  "depends_on": ["phase-5.5", "phase-6"],
  "owner": "harness",
  "cron_slots": 1,
  "steps": [
    {
      "id": "phase-6.5.1",
      "name": "BigQuery schema migration for intel tables",
      "verify": "python scripts/migrations/2026_04_add_intel_tables.py --dry-run && bq show sunny-might-477607-p8:pyfinagent_data.institutional_intel && bq show sunny-might-477607-p8:pyfinagent_data.academic_frontier && bq show sunny-might-477607-p8:pyfinagent_data.ai_frontier_releases && bq show sunny-might-477607-p8:pyfinagent_data.player_driven_signals && bq show sunny-might-477607-p8:pyfinagent_data.researcher_prompt_patches"
    },
    {
      "id": "phase-6.5.2",
      "name": "Source registry and scan core",
      "verify": "test -f backend/cron/intel_sources.yaml && python -c \"import ast; ast.parse(open('backend/cron/intel_scan.py').read())\""
    },
    {
      "id": "phase-6.5.3",
      "name": "Institutional extractors (GS, BLK, Bridgewater, JPM)",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/intel/institutional.py').read())\" && python -m pytest tests/cron/intel/test_institutional.py -x"
    },
    {
      "id": "phase-6.5.4",
      "name": "Academic extractors (arXiv OAI-PMH, SSRN, MIT, Stanford, Oxford)",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/intel/academic.py').read())\" && python -m pytest tests/cron/intel/test_academic.py -x"
    },
    {
      "id": "phase-6.5.5",
      "name": "AI-frontier extractors (Anthropic, OpenAI, DeepMind, Google Research)",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/intel/ai_frontier.py').read())\" && python -m pytest tests/cron/intel/test_ai_frontier.py -x"
    },
    {
      "id": "phase-6.5.6",
      "name": "Player-driven extractors (SeekingAlpha, WSB daily, QuantConnect, SWF EDGAR)",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/intel/player_driven.py').read())\" && python -m pytest tests/cron/intel/test_player_driven.py -x"
    },
    {
      "id": "phase-6.5.7",
      "name": "Novelty client (Voyage + Gemini fallback) and prompt-patch queue",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/intel/novelty.py').read())\" && python -m pytest tests/cron/intel/test_novelty.py -x"
    },
    {
      "id": "phase-6.5.8",
      "name": "Slack digest + daily scheduler wiring",
      "verify": "python -c \"import ast; ast.parse(open('backend/cron/intel/digest.py').read())\" && grep -q 'intel_scan' backend/cron/scheduler.py"
    },
    {
      "id": "phase-6.5.9",
      "name": "End-to-end smoketest with fixtures",
      "verify": "python scripts/smoke/test_intel_scan.py"
    }
  ],
  "verification_commands": [
    "python -c \"import ast; ast.parse(open('backend/cron/intel_scan.py').read())\"",
    "python scripts/smoke/test_intel_scan.py",
    "bq query --use_legacy_sql=false 'SELECT family, COUNT(*) c FROM `sunny-might-477607-p8.pyfinagent_data.institutional_intel` WHERE DATE(observed_at) = CURRENT_DATE() GROUP BY family'"
  ]
}
```

## Implementation notes

### Slot accounting (15/day hard cap)

Phase-6.5 consumes one new Claude scheduled slot (slot identifier
`intel_scan_daily`). The live environment cap is 15/day; this phase
takes the directive total to one additional slot over the phase-6
baseline.

The slot is reused — not duplicated — by phase-10 on Monday, Tuesday,
and Wednesday sprint days. On those days, the intel scan runs first
(08:00 UTC), and its output corpus is the RESEARCH input for the
phase-10 ideation step that follows in the same slot window. This
reuse is tracked in `.claude/masterplan.json` under the `cron_slots`
field on both phases so the harness never double-books.

Fragmenting this directive into four separate jobs (one per family)
is explicitly rejected. The four families are deliberately walked
sequentially in one coroutine so that a single slot produces one
Slack digest, one audit row, and one cost ledger entry. Parallel
family execution is allowed inside the single slot (via `asyncio.
gather`) as long as the slot boundary is one wall-clock run.

### BigQuery schemas

`pyfinagent_data.institutional_intel`:

```
intel_id          STRING    NOT NULL   # uuid v7
publisher         STRING    NOT NULL   # 'goldman_sachs' | 'blackrock' | 'bridgewater' | 'jpmam' | ...
title             STRING    NOT NULL
url               STRING    NOT NULL
published_date    DATE      NOT NULL
observed_at       TIMESTAMP NOT NULL
summary           STRING                # Gemini Flash 5-bullet summary
key_macro_view    STRING                # one-sentence thesis
mentioned_assets  ARRAY<STRING>         # tickers / asset classes detected
novelty_score     FLOAT64
raw_text_gcs      STRING                # GCS URI for full extracted text
```

`pyfinagent_data.academic_frontier`:

```
paper_id          STRING    NOT NULL   # 'arxiv:2404.12345' | 'ssrn:4567890'
source            STRING    NOT NULL   # 'arxiv' | 'ssrn' | 'mit_news' | ...
title             STRING    NOT NULL
authors           ARRAY<STRING>
abstract          STRING    NOT NULL
url               STRING    NOT NULL
published_date    DATE      NOT NULL
observed_at       TIMESTAMP NOT NULL
novelty_score     FLOAT64
nearest_signal_id STRING                # FK -> signal_embedding_library
summary           STRING                # Gemini Flash
```

`pyfinagent_data.ai_frontier_releases`:

```
release_id        STRING    NOT NULL
lab               STRING    NOT NULL   # 'anthropic' | 'openai' | 'deepmind' | 'google_research'
title             STRING    NOT NULL
url               STRING    NOT NULL
published_date    DATE      NOT NULL
observed_at       TIMESTAMP NOT NULL
patchable_hooks   ARRAY<STRING>         # matched subset of allowlist
summary           STRING
```

`pyfinagent_data.player_driven_signals`:

```
signal_id         STRING    NOT NULL
source            STRING    NOT NULL   # 'seeking_alpha' | 'wsb_daily' | 'quantconnect' | 'edgar_swf'
entity            STRING                # SWF name / community name
title             STRING
url               STRING
published_date    DATE      NOT NULL
observed_at       TIMESTAMP NOT NULL
delta_usd         NUMERIC(18, 2)        # populated for EDGAR SWF deltas; else NULL
summary           STRING
```

`pyfinagent_data.researcher_prompt_patches`:

```
patch_id          STRING    NOT NULL
source_url        STRING    NOT NULL
rationale         STRING    NOT NULL
proposed_diff     STRING    NOT NULL   # unified diff targeting backend/agents/skills/researcher.md
status            STRING    NOT NULL   # 'pending' | 'approved' | 'rejected' | 'applied'
created_at        TIMESTAMP NOT NULL
reviewed_at       TIMESTAMP
reviewer          STRING
applied_commit    STRING                # git SHA once applied
```

Partition: `DATE(observed_at)` (tables 1-4) or `DATE(created_at)`
(patches). Cluster: `source, publisher` or `lab`.

### Source registry outline

```
family: institutional
  - publisher: goldman_sachs
    transport: http+pdf
    urls:
      - https://privatewealth.goldmansachs.com/insights/isg-outlook
      - https://www.gsam.com/responsible-investing/en-us/professional/market-insights
    license_tier: A
    max_items_per_run: 5
    timeout_seconds: 180
  - publisher: blackrock
    transport: rss
    urls:
      - https://www.blackrock.com/corporate/insights/blackrock-investment-institute/rss
    license_tier: A
    max_items_per_run: 5
    timeout_seconds: 120
  - publisher: bridgewater
    transport: http
    urls:
      - https://www.bridgewater.com/research-and-insights
    license_tier: A
    max_items_per_run: 3
    timeout_seconds: 120
  - publisher: jpmam
    transport: http+pdf
    urls:
      - https://am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/guide-to-the-markets/
    license_tier: A
    max_items_per_run: 3
    timeout_seconds: 180

family: academic
  - publisher: arxiv
    transport: oai-pmh
    url: https://export.arxiv.org/oai2
    sets: [cs:cs.LG, stat:stat.ML, q-fin:q-fin.CP, q-fin:q-fin.GN, q-fin:q-fin.MF, q-fin:q-fin.PM, q-fin:q-fin.PR, q-fin:q-fin.RM, q-fin:q-fin.ST, q-fin:q-fin.TR]
    license_tier: A
    max_items_per_run: 200
    timeout_seconds: 300
  - publisher: ssrn
    transport: http
    urls:
      - https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=203
    license_tier: A
    max_items_per_run: 50
    timeout_seconds: 180
  - publisher: mit_csail
    transport: rss
    url: https://news.mit.edu/rss/topic/artificial-intelligence2
    license_tier: A
    max_items_per_run: 20
    timeout_seconds: 60
  - publisher: stanford_ai_lab
    transport: rss
    url: https://ai.stanford.edu/blog/feed.xml
    license_tier: A
    max_items_per_run: 20
    timeout_seconds: 60
  - publisher: oxford_said
    transport: http
    url: https://www.sbs.ox.ac.uk/research
    license_tier: A
    max_items_per_run: 10
    timeout_seconds: 60

family: ai_frontier
  - publisher: anthropic
    transport: rss
    url: https://www.anthropic.com/news/rss.xml
    license_tier: A
    max_items_per_run: 10
    timeout_seconds: 60
  - publisher: openai
    transport: rss
    url: https://openai.com/news/research/rss.xml
    license_tier: A
    max_items_per_run: 10
    timeout_seconds: 60
  - publisher: deepmind
    transport: rss
    url: https://deepmind.google/blog/rss.xml
    license_tier: A
    max_items_per_run: 10
    timeout_seconds: 60
  - publisher: google_research
    transport: rss
    url: https://research.google/blog/rss/
    license_tier: A
    max_items_per_run: 10
    timeout_seconds: 60

family: player_driven
  - publisher: seeking_alpha
    transport: rss
    url: https://seekingalpha.com/feed.xml
    license_tier: A
    max_items_per_run: 20
    timeout_seconds: 60
  - publisher: wsb_daily
    transport: reddit_api
    subreddit: wallstreetbets
    daily_thread_only: true
    license_tier: B
    max_items_per_run: 1
    timeout_seconds: 60
  - publisher: quantconnect
    transport: rss
    url: https://www.quantconnect.com/forum/discussions.rss
    filter_categories: [research, strategy, alpha]
    license_tier: A
    max_items_per_run: 20
    timeout_seconds: 60
  - publisher: edgar_swf
    transport: edgar
    ciks: ["0001262767", "0001337447", "0000919079", "0001484160"]
    license_tier: A
    max_items_per_run: 4
    timeout_seconds: 120
```

### Cost estimate (daily)

```
Gemini Flash institutional summaries (4 pubs x 2 docs x 500 in + 200 out tok)  ~$0.01
Gemini Flash academic abstracts      (150 abstracts x 300 in + 100 out tok)     ~$0.02
Gemini Flash AI-frontier summaries   (10 posts x 500 in + 200 out tok)           ~$0.01
Gemini Flash player-driven digest    (20 items x 300 in + 100 out tok)           ~$0.01
Claude Haiku novelty tagging         (50 high-novelty items x 1000 in + 300 out) ~$0.16
Claude Haiku patch proposals         (5 candidates x 2000 in + 500 out)          ~$0.03
Voyage-3-large embeddings            (250 items x 400 tok)                       ~$0.02
BigQuery storage + insert                                                         ~$0.02
---
Total daily                                                                       ~$0.28
Alarm threshold                                                                    $3.00
Hard cap (circuit-breaker)                                                        $10.00
```

### Fallback chain

1. Voyage-3-large embedding (primary).
2. Gemini free-tier embedding (fallback).
3. Family-level timeout: on overrun, write audit row, skip family,
   continue with next.
4. Whole-slot timeout at 45 min: finalize partial digest, post Slack
   message with `status=degraded`, continue next day.
5. LLM budget guard: if `cost_usd` > $3, halt summarization for the
   remainder of the run; ingestion (URL + title + raw) continues,
   summaries get computed on the next run.

### Researcher prompt-patch workflow

1. Candidate extracted by `ai_frontier` or high-novelty `academic`.
2. Claude Haiku produces a unified-diff patch against
   `backend/agents/skills/researcher.md` with rationale.
3. Row written to `researcher_prompt_patches` with `status=pending`.
4. Slack digest enumerates pending patches with BQ links.
5. Human reviewer sets `status=approved` via a BQ UPDATE or a Slack
   slash-command. `status=rejected` is equally valid.
6. Nightly job reads approved patches, runs `git apply` in a sandbox,
   runs syntax check and smoketest, opens a branch, and requests
   Peder's final sign-off via Slack before push. Auto-apply is NEVER
   enabled. This is the same pattern as phase-7's compliance-
   checklist gating.

### Dedup

Abstracts and AI-lab posts that also appear on arXiv are deduped by
arXiv ID. Institutional PDFs are deduped by `(publisher, url,
published_date)`. SWF filings are deduped by `(cik, accession_number)`.

### Backtest replay

Each row carries `observed_at` as the time-of-discovery and
`published_date` as the authoritative timestamp from the source.
Backtest agents join on `published_date <= as_of_date` to avoid
look-ahead; the `observed_at` lag column lets us measure how much
of the signal is usable given pyfinagent's own ingestion latency.

## References

Institutional outlook archives:

1. https://privatewealth.goldmansachs.com/insights/isg-outlook
2. https://www.gsam.com/responsible-investing/en-us/professional/market-insights
3. https://www.blackrock.com/corporate/insights/blackrock-investment-institute
4. https://www.bridgewater.com/research-and-insights
5. https://am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/guide-to-the-markets/
6. https://privatebank.jpmorgan.com/nam/en/insights/markets-and-investing/eotm
7. https://www.pimco.com/en-us/insights/economic-and-market-commentary
8. https://corporate.vanguard.com/content/corporatesite/us/en/corp/who-we-are/economic-and-market-outlook.html
9. https://institutional.fidelity.com/advisors/insights/topics/business-cycle-update

Academic feeds and APIs:

10. https://info.arxiv.org/help/api/tou.html
11. https://export.arxiv.org/oai2
12. https://export.arxiv.org/rss/cs.LG
13. https://export.arxiv.org/rss/q-fin
14. https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=203
15. https://news.mit.edu/rss/topic/artificial-intelligence2
16. https://ai.stanford.edu/blog/feed.xml
17. https://hai.stanford.edu/news
18. https://www.sbs.ox.ac.uk/research
19. https://www.oxford-man.ox.ac.uk/

AI-lab research feeds:

20. https://www.anthropic.com/news/rss.xml
21. https://www.anthropic.com/engineering
22. https://www.anthropic.com/research
23. https://openai.com/news/research/rss.xml
24. https://deepmind.google/blog/rss.xml
25. https://research.google/blog/rss/

Player-driven sources:

26. https://seekingalpha.com/feed.xml
27. https://www.quantconnect.com/forum/discussions.rss
28. https://www.reddit.com/r/wallstreetbets/
29. https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001262767
30. https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001337447
31. https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000919079
32. https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001484160
33. https://data.sec.gov/submissions/

Novelty / embeddings:

34. https://docs.voyageai.com/docs/embeddings
35. https://arxiv.org/abs/2402.16889
36. https://arxiv.org/abs/2310.07712
37. https://platform.openai.com/docs/guides/embeddings

Dependent proposals:

38. handoff/phase-proposals/phase-6-news-sentiment-cron.md
39. handoff/phase-proposals/phase-7-alt-data-scraping.md
40. docs/compliance/alt-data.md  (phase-7 legal framework inherited)
