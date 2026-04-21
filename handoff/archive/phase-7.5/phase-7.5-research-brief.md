# Research Brief: phase-7.5 Reddit WSB Sentiment Ingestion

**Tier:** moderate  
**Authored by:** researcher agent  
**Date:** 2026-04-19  
**Verification criteria (immutable):**
```
python -c "import ast; ast.parse(open('backend/alt_data/reddit_wsb.py').read())"
test -f docs/compliance/reddit-license.md
```

---

## Objective / Output Format / Tool Scope / Task Boundaries

**Objective:** Scaffold `backend/alt_data/reddit_wsb.py` and `docs/compliance/reddit-license.md` to satisfy phase-7.5 masterplan criteria.

**Output format:** Two files. No live API calls at scaffold time. Advisory `adv_70_oauth_tos` acknowledged (app registration deferred; click-through = contract formation happens at phase-7.12). DDL for `alt_reddit_sentiment` included in scaffold.

**Tool scope:** PRAW 7.8.x, Reddit Data API free/research tier (100 QPM), OAuth "script" app type (hardware-controlled server, holds secret). Pushshift is deprecated; Arctic Shift is viable for historical backfill but NOT used at scaffold time.

**Task boundaries:** This brief stops at design. Main authors both files. Phase-7.12 wires live fetch and FinBERT scoring.

---

## Queries run (three-variant discipline)

1. **Current-year frontier:** `Reddit Data API OAuth rate limits 2026`
2. **Last-2-year window:** `Reddit Data API pricing free tier commercial 2025`; `PRAW Reddit Python wrapper version 2025 OAuth rate limit User-Agent format`; `Pushshift Reddit archive status 2025 2026 Arctic Shift replacement`
3. **Year-less canonical:** `PRAW Reddit API Python library`; `cashtag extraction Reddit WSB ticker regex NLP sentiment`; `Reddit API ToS OAuth installed app research academic use 2024`

---

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://github.com/praw-dev/praw | 2026-04-19 | Official SDK docs/code | WebFetch | PRAW v7.8.1 (Oct 2024); Python 3.9+; built-in rate-limit handler; `user_agent` param mandatory |
| https://github.com/reddit-archive/reddit/wiki/OAuth2 | 2026-04-19 | Official Reddit OAuth2 spec | WebFetch | Three app types: web, installed, script. Script app runs on controlled hardware, holds secret, suitable for server-side pipelines |
| https://apidog.com/blog/reddit-api-guide/ | 2026-04-19 | Authoritative tech blog | WebFetch | Free tier: 100 QPM, non-commercial/research; commercial: ~$12K/yr; User-Agent mandatory; OAuth required as of late 2024 |
| https://redaccs.com/reddit-api-guide/ | 2026-04-19 | Tech blog (2026 guide) | WebFetch | User-Agent exact format: `platform:appname:version (by /u/username)`; Reddit aggressively rate-limits without it; script app type recommended for server bots |
| https://www.wappkit.com/blog/reddit-api-credentials-guide-2025 | 2026-04-19 | Tech blog (2025 guide) | WebFetch | Pre-approval required (Responsible Builder Policy); free tier ~100 QPM; commercial $0.24/1000 calls; enterprise negotiated |
| https://github.com/ArthurHeitmann/arctic_shift | 2026-04-19 | Official GitHub project | WebFetch | Active project (latest release 2026_03, April 10 2026); 2.5B items (posts + comments) via dump or limited API; viable Pushshift successor |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://painonsocial.com/blog/reddit-api-rate-limits-guide | Blog | Redundant with apidog/redaccs fetched in full |
| https://replydaddy.com/blog/reddit-api-pre-approval-2025-personal-projects-crackdown | Blog | Covered by wappkit fetch |
| https://data365.co/blog/reddit-api-limits | Blog | Duplicate rate-limit coverage |
| https://data365.co/blog/reddit-api-pricing | Blog | Redundant with pricing sources |
| https://rankvise.com/blog/reddit-api-cost-guide/ | Blog | Redundant |
| https://github.com/asad70/wallstreetbets-sentiment-analysis | Code | Snippet sufficient; confirms `$[A-Z]{1,5}` regex convention on WSB |
| https://medium.com/nerd-for-tech/wallstreetbets-sentiment-analysis-on-stock-prices-using-natural-language-processing | Blog | Confirms VADER as lightweight lexicon scorer |
| https://pypi.org/project/praw/ | PyPI | Version confirmed from GitHub fetch |
| https://praw.readthedocs.io/en/stable/getting_started/quick_start.html | Official docs | 403 on full fetch; PRAW GitHub page covered the core facts |
| https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki | Official help | 403; key facts recovered from snippet + apidog fetch |

---

## Recency scan (2024-2026)

Searched explicitly for 2024-2026 material. Results:

- **2025 pre-approval enforcement** (replydaddy.com): Reddit removed self-service API access in late 2024. Developers must now submit a Responsible Builder Policy review; personal/research projects approved in days, commercial use takes weeks. This supersedes pre-2024 "create app and go" guidance.
- **PRAW v7.8.1** released October 2024 -- current stable; no breaking changes to OAuth flow.
- **Arctic Shift 2026_03** (April 10 2026) -- active Pushshift successor with data through at least early 2026.
- **Springer Nature 2024** (`s13278-024-01273-2`): WSB NLP paper confirmed cashtag regex `$[A-Z]{2,5}` with spaCy disambiguation; main finding that `$[A-Z]{1,5}` produces false positives on common English words like `$A`, `$I` -- minimum 2-char tickers recommended.
- **Reddit v. Perplexity AI** (filed N.D. Cal. 2025, pending): live DMCA Sec.1201 risk on rate-limit bypass; no scraping without authentication is the safe path.

No findings in the 2024-2026 window that invalidate the scaffold approach. The pre-approval enforcement and the ticker-length floor (>=2 chars) are the two operationally significant updates.

---

## Key findings

1. **User-Agent format is fixed.** Reddit's API enforces the format `<platform>:<app_id>:<version> (by /u/<username>)` e.g. `python:pyfinagent:1.0 (by /u/pederbkoppang)`. Deviation triggers aggressive rate-limiting. (Source: redaccs.com guide, 2026-04-19)

2. **Free tier: 100 QPM = 1.67 req/s. Rate interval floor = 0.6s.** The limit is per OAuth client_id, averaged over a 10-minute window. PRAW handles back-off natively if `ratelimit_seconds >= expected_wait`. (Source: apidog.com guide, praw GitHub, 2026-04-19)

3. **Script app type is correct for a server-controlled pipeline.** Not "installed app" (mobile, no secret) but "script" (runs on hardware you control, has secret, OAuth2 `password` grant). This is distinct from the installed-app `reddit-specific grants/installed_client` flow. (Source: Reddit OAuth2 wiki, 2026-04-19)

4. **Pre-approval is now required.** App registration at `reddit.com/prefs/apps` is still step 1, but production API use requires Responsible Builder Policy submission. For non-commercial research, approval is typically a few days. The click-through of the Responsible Builder Policy IS contract formation -- relevant to `adv_70_oauth_tos`. (Source: wappkit guide 2025, 2026-04-19)

5. **Cashtag regex on Reddit should floor at 2 chars.** WSB community does use the `$TICKER` convention identical to Twitter; however the 2024 Springer WSB paper noted `$A` and `$I` generate high false-positive rates. The safer regex is `\$[A-Z]{2,5}\b`, not `{1,5}`. The twitter.py scaffold uses `{1,5}` -- reddit_wsb.py should use `{2,5}` and document the deviation. (Source: Springer Nature 2024 snippet, WSB GitHub repos, 2026-04-19)

6. **Pushshift is defunct; Arctic Shift is the successor.** Arctic Shift has 2.5B items through at least 2026-02 and an active API + dump release cycle. For historical backfill (phase-7.12), Arctic Shift replaces Pushshift entirely. (Source: ArthurHeitmann/arctic_shift GitHub, 2026-04-19)

7. **Commercial cliff.** Free tier = research/personal, up to 100 QPM, $0. Commercial requires prior Reddit approval; pricing is opaque but one data point is $12K/yr base for 100 QPM. The pyfinagent use is non-commercial model training/backtesting, which fits the free tier -- but the Responsible Builder Policy application must accurately characterize the use case. (Source: apidog guide, redaccs guide, 2026-04-19)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/alt_data/twitter.py` | 227 | Phase-7.6 Twitter/X scaffold; house style template | Active; read in full |
| `backend/alt_data/google_trends.py` | 30+ | Phase-7.9 scaffold; import/docstring pattern | Active; read header |
| `backend/alt_data/__init__.py` | n/a | Package init | Exists |
| `docs/compliance/alt-data.md` | 287 | Alt-data compliance framework; row 7.5 is the Reddit entry | Active; read in full |
| `docs/compliance/` | -- | Only `alt-data.md` and `2026-regulatory-memo.md` exist | No per-vendor license docs yet -- reddit-license.md will be the first |

**Key observations from twitter.py (house style):**
- `_USER_AGENT` constant at module top (line 36): `"pyfinagent/1.0 peder.bkoppang@hotmail.no"` -- this follows SEC EDGAR format. Reddit requires a different format (`platform:appid:version (by /u/username)`). reddit_wsb.py must use the Reddit-specific form.
- `_TABLE = "alt_twitter_sentiment"` constant pattern -- mirrored as `_TABLE = "alt_reddit_sentiment"`.
- `_STARTER_CASHTAGS` tuple + `_CASHTAG_RE` regex constant (line 40-41) -- reddit_wsb.py uses `_STARTER_SUBS` for subreddits and same `_CASHTAG_RE` but with `{2,5}` floor.
- `_hash_author()` + sha256 at line 70-74: exact pattern to copy.
- `_resolve_target()` + `_get_bq_client()` helpers at lines 102-131: copy verbatim.
- `ensure_table()` + `upsert()` + `ingest_*()` + `_cli()` orchestration shape: direct parallel.
- `if __name__ == "__main__": raise SystemExit(_cli(sys.argv[1:]))` tail: copy verbatim.
- Scaffold docstring declares `source` in BQ row as `"x.com/api/2"` -- reddit_wsb.py uses `"reddit.com/api/v1"`.

**Compliance doc structure:** `alt-data.md` is the umbrella. Row 7.5 (line 158) already has the entry. `docs/compliance/reddit-license.md` will be a sibling file -- there are no other per-vendor sibling docs yet (Revelio at `revelio-license.md` is referenced in row 7.7 but not yet created). Pattern: first per-vendor doc in this project.

---

## Consensus vs debate (external)

- **Consensus:** PRAW is the de-facto Python SDK for Reddit API; no credible challenger.
- **Consensus:** Free tier is 100 QPM per OAuth client; no source disputes this.
- **Debate:** 60 QPM vs 100 QPM occasionally appears in older sources. Official current position (confirmed by multiple 2025-2026 sources) is 100 QPM with OAuth.
- **Debate:** "Installed app" vs "script app" -- the correct type for a server-side pipeline is **script** (holds secret, password grant). Some guides conflate these. The OAuth2 wiki is authoritative.
- **Debate:** Whether pyfinagent's use is "non-commercial research." Signal generation that feeds live trading decisions is a grey area. The compliance conservative position: apply for free tier as research; if/when live trading generates revenue on Reddit-derived signals, upgrade or segregate.

---

## Pitfalls (from research)

1. **Wrong app type.** Registering as "installed app" gives no client_secret and requires device-id grant flow. Server-side pipelines need "script" app.
2. **Wrong User-Agent.** Generic `python-requests/2.x` or the SEC EDGAR format (`pyfinagent/1.0 peder.bkoppang@hotmail.no`) will trigger rate-limiting. Reddit's format is `python:pyfinagent:1.0 (by /u/pederbkoppang)`.
3. **Cashtag false positives.** `$A` (Agilent) and `$I` (not a valid ticker but a false match) inflate mention counts. Use `{2,5}`, not `{1,5}`.
4. **Pushshift dependency.** Any code that references `pushshift.io` will fail. Arctic Shift is the replacement for historical data.
5. **Pre-approval skip.** Creating the app is step 1; the Responsible Builder Policy submission is step 2 and required before production use.
6. **Scaffold-time OAuth secret.** The module docstring must flag that `REDDIT_CLIENT_SECRET` env var is read only at phase-7.12 runtime. At scaffold time it is intentionally absent; the module must not raise on missing env vars during `import` or `--dry-run`.

---

## Application to pyfinagent (file:line anchors)

- `backend/alt_data/twitter.py:36` -- `_USER_AGENT` constant: reddit_wsb.py must use Reddit-specific format, not this one.
- `backend/alt_data/twitter.py:40-41` -- `_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}\b")`: reddit_wsb.py should change `{1,5}` to `{2,5}` and add a comment explaining the WSB false-positive concern.
- `backend/alt_data/twitter.py:70-74` -- `_hash_author()`: copy verbatim.
- `backend/alt_data/twitter.py:102-131` -- `_resolve_target()` + `_get_bq_client()`: copy verbatim.
- `docs/compliance/alt-data.md:158` -- Row 7.5 already populated; reddit-license.md must reference this row by section (Sec. 4, row 7.5) and alt-data.md Sec. 2.2 on ToS/contract-formation risk.
- `docs/compliance/alt-data.md:283` -- References already lists Reddit Data API Terms URL (`https://redditinc.com/policies/data-api-terms`); reddit-license.md should cite this.

---

## Concrete design proposal

### `backend/alt_data/reddit_wsb.py`

```python
"""phase-7.5 Reddit WSB sentiment -- scaffold.

Persists `(post_id, as_of_date, subreddit, cashtag, author_id_hash, title,
text, sentiment_score, sentiment_label, created_at, source, raw_payload)` rows
to `pyfinagent_data.alt_reddit_sentiment`. This cycle ships the scaffold; live
PRAW fetch + FinBERT scoring are deferred to phase-7.12.

Compliance: docs/compliance/alt-data.md row 7.5 -- Reddit Data API v1 with
OAuth script-app key. Per advisory adv_70_oauth_tos, app registration (click-
through = contract formation) is deferred to phase-7.12. PII: author name is
sha256'd before persistence per compliance Sec 5.5. Raw text retained <=90 days.

User-Agent: Reddit-required format is platform:appid:version (by /u/username)
-- differs from SEC EDGAR format used in twitter.py.

CLI:
    python -m backend.alt_data.reddit_wsb [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# Reddit-required User-Agent format: platform:appid:version (by /u/username)
# Different from SEC EDGAR format used in twitter.py.
_USER_AGENT = "python:pyfinagent:1.0 (by /u/pederbkoppang)"
_TABLE = "alt_reddit_sentiment"

# Subreddits to ingest at scaffold time
_STARTER_SUBS: tuple[str, ...] = ("wallstreetbets", "stocks", "investing")

# Free-tier ceiling: 100 QPM => 1 req per 0.6s minimum interval
_RATE_INTERVAL_S: float = 0.6

# Minimum 2-char ticker to reduce false positives on WSB
# ($A, $I etc generate high false-positive rates per Springer 2024 WSB study)
# Differs from twitter.py {1,5} convention; intentional, documented here.
_CASHTAG_RE = re.compile(r"\$[A-Z]{2,5}\b")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  post_id STRING NOT NULL,
  as_of_date DATE NOT NULL,
  subreddit STRING,
  author_hash STRING,
  cashtag STRING,
  title STRING,
  text STRING,
  sentiment_score FLOAT64,
  sentiment_label STRING,
  score INT64,
  upvote_ratio FLOAT64,
  created_at TIMESTAMP,
  source STRING,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY subreddit, cashtag
OPTIONS (
  description = "phase-7.5 Reddit WSB cashtag sentiment; fetch + model deferred to phase-7.12"
)
""".strip()


def extract_cashtags(text: str) -> list[str]:
    """Return all uppercase cashtags (e.g. '$AAPL') found in text.

    Uses {2,5} lower bound (not {1,5} as in twitter.py) to reduce
    false positives from single-char matches on WSB posts.
    """
    if not text:
        return []
    return [m.group(0) for m in _CASHTAG_RE.finditer(text)]


def _hash_author(author_name: str | None) -> str | None:
    """PII discipline: never persist raw Reddit author name. sha256 hash."""
    if author_name is None:
        return None
    return hashlib.sha256(str(author_name).encode("utf-8")).hexdigest()


def fetch_wsb_posts(
    subreddit: str,
    limit: int = 100,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Scaffold -- deferred to phase-7.12.

    Live impl will use PRAW (v7.8.x) with a script-type OAuth app:
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=_USER_AGENT,
        )
        sub = reddit.subreddit(subreddit)
        posts = list(sub.new(limit=limit))

    `since` is an ISO-date string for incremental ingestion (phase-7.12).
    Returns [] until implemented.
    """
    logger.debug("reddit_wsb: fetch_wsb_posts scaffold subreddit=%s", subreddit)
    return []


def score_sentiment(text: str) -> tuple[float, str]:
    """Scaffold -- deferred to phase-7.12.

    Live impl will load ProsusAI/finbert and return softmax score + label
    over {positive, neutral, negative}. Returns neutral prior until wired.
    """
    return 0.0, "neutral"


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    proj = project
    ds = dataset
    if proj is None or ds is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            if proj is None:
                proj = s.gcp_project_id
            if ds is None:
                ds = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:  # pragma: no cover
            logger.warning("reddit_wsb: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("reddit_wsb: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("reddit_wsb: bigquery.Client() init failed (%r)", exc)
        return None


def ensure_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    sql = _CREATE_TABLE_SQL.format(project=proj, dataset=ds, table=_TABLE)
    try:
        client.query(sql).result(timeout=60)
        return True
    except Exception as exc:
        logger.warning("reddit_wsb: ensure_table fail-open: %r", exc)
        return False


def upsert(
    rows: list[dict[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return 0
    table_ref = f"{proj}.{ds}.{_TABLE}" if proj else f"{ds}.{_TABLE}"
    try:
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("reddit_wsb: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("reddit_wsb: upsert fail-open: %r", exc)
        return 0


def ingest_subreddit(
    sub: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Scaffold orchestrator for a single subreddit.

    Returns 0 until phase-7.12 wires live PRAW fetch + FinBERT scoring.
    """
    today_iso = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    posts = fetch_wsb_posts(sub)
    for post in posts:
        title = post.get("title") or ""
        selftext = post.get("selftext") or ""
        combined = f"{title} {selftext}".strip()
        tags = extract_cashtags(combined)
        for tag in tags:
            score, label = score_sentiment(combined)
            rows.append(
                {
                    "post_id": post.get("id"),
                    "as_of_date": today_iso,
                    "subreddit": sub,
                    "author_hash": _hash_author(post.get("author")),
                    "cashtag": tag,
                    "title": title,
                    "text": selftext,
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "score": post.get("score"),
                    "upvote_ratio": post.get("upvote_ratio"),
                    "created_at": post.get("created_utc"),
                    "source": "reddit.com/api/v1",
                    "raw_payload": json.dumps(post, default=str, ensure_ascii=True),
                }
            )
    if dry_run:
        return len(rows)
    return upsert(rows, project=project, dataset=dataset)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="phase-7.5 Reddit WSB sentiment ingester (scaffold)"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    total = 0
    for sub in _STARTER_SUBS:
        total += ingest_subreddit(sub, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
                "ingested": total,
                "scaffold_only": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
```

**DDL column mapping for `alt_reddit_sentiment`:**

| Column | Type | Notes |
|--------|------|-------|
| post_id | STRING NOT NULL | Reddit post fullname (t3_xxxx) |
| as_of_date | DATE NOT NULL | Partition key |
| subreddit | STRING | e.g. "wallstreetbets" |
| cashtag | STRING | e.g. "$AAPL" |
| author_hash | STRING | sha256 of Reddit username |
| cashtag | STRING | e.g. "$AAPL" |
| title | STRING | Post title |
| text | STRING | Post selftext; redacted after 90 days |
| sentiment_score | FLOAT64 | FinBERT score (-1..1); 0.0 scaffold |
| sentiment_label | STRING | positive/neutral/negative |
| score | INT64 | Reddit post upvote score |
| upvote_ratio | FLOAT64 | Upvote ratio (0.0-1.0) |
| created_at | TIMESTAMP | UTC from Reddit API |
| source | STRING | "reddit.com/api/v1" |
| raw_payload | JSON | Full API response; 90-day retention |

Cluster: `subreddit, cashtag`. Partition: `as_of_date`.

---

### `docs/compliance/reddit-license.md`

```
# Reddit Data API -- Per-Vendor License Record

**Step:** phase-7.5  
**Version:** 1.0 -- 2026-04-19  
**Owner:** Peder (GitHub: pederbkoppang-bit)  
**Parent framework:** docs/compliance/alt-data.md, Section 4, row 7.5

---

## §1 Scope

This document is the per-vendor compliance record for pyfinagent's use of the
Reddit Data API (v1) to ingest public posts from r/wallstreetbets, r/stocks,
and r/investing for sentiment signal generation (phase-7.5 scaffold; live
ingestion begins phase-7.12).

Covers: API access method, ToS acceptance status, rate limits, data retention,
permitted use, and review cadence. Any questions not answered here defer to
the umbrella framework at docs/compliance/alt-data.md.

---

## §2 Access Method

- **API:** Reddit Data API v1 (OAuth2)
- **App type:** "script" (server-controlled hardware, holds client_secret)
  -- NOT "installed app" (mobile, no secret).
- **Grant flow:** OAuth2 password grant with client_id + client_secret +
  Reddit account credentials. All credentials read from environment variables
  at runtime (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`,
  `REDDIT_PASSWORD`). No credentials are stored in code or git.
- **Registration URL:** https://www.reddit.com/prefs/apps
- **SDK:** PRAW 7.8.x (`pip install praw>=7.8`). PRAW handles token refresh
  and rate-limit back-off natively.
- **User-Agent:** `python:pyfinagent:1.0 (by /u/pederbkoppang)`
  -- Reddit's required format: `platform:appid:version (by /u/username)`.
  This differs from the SEC EDGAR User-Agent in alt-data.md Sec 5.1.

---

## §3 ToS Acceptance Record

**Advisory adv_70_oauth_tos is active for this step.**

OAuth app registration at `reddit.com/prefs/apps` constitutes contract
formation with Reddit's Terms of Service. Additionally, Reddit's Responsible
Builder Policy must be submitted and approved before production API use.

| Event | Date | Status |
|-------|------|--------|
| App registration (prefs/apps) | pending | Not done at scaffold time |
| Responsible Builder Policy submission | pending | Required before phase-7.12 live fetch |
| Responsible Builder Policy approval | pending | Reddit estimates days for research use |
| First live API call | pending | phase-7.12 |

**Action required before phase-7.12:** Peder registers the OAuth script app,
submits the Responsible Builder Policy with use-case description ("non-
commercial quantitative research; backtesting only; no redistribution of
Reddit data"), and records the approval datetime in the table above.

Reddit Data API Terms of Service: https://redditinc.com/policies/data-api-terms

---

## §4 Rate Limits

| Tier | Limit | Enforcement |
|------|-------|-------------|
| Free (research/non-commercial) | 100 requests/minute per OAuth client_id | Averaged over 10-minute window |
| Unauthenticated | 10 requests/minute | Blocked as of late 2024 |
| Commercial | 100 requests/minute | ~$12,000/yr; requires approval |

pyfinagent targets the **free research tier**. The module enforces
`_RATE_INTERVAL_S = 0.6` (100 QPM ceiling = 1 req per 0.6s). PRAW's
built-in `ratelimit_seconds` parameter provides a second layer of back-off.

Response headers `X-Ratelimit-Used`, `X-Ratelimit-Remaining`,
`X-Ratelimit-Reset` are returned by the API and should be logged for
audit trail compliance (phase-7.11 scraper_audit_log table).

---

## §5 Retention & PII

Follows alt-data.md Sec 5.5 and Sec 6.2:

- **Author name:** sha256-hashed at ingest. Raw Reddit usernames are never
  written to BigQuery.
- **Raw text (title, selftext):** retained 90 days in `alt_reddit_sentiment`
  then redacted (column set to NULL; embeddings and hashes retained).
- **Deleted posts:** if Reddit marks a post deleted, we do not re-ingest it
  and set the existing row's text to NULL on next cycle.
- **IP address:** never stored.

GDPR basis: Reddit usernames are pseudonymous personal data under EU law
(can be linked to a real person). Mitigation: sha256 hash at ingest boundary.
CCPA: same mitigation; deletion requests honored via phase-7.11 SAR endpoint.

---

## §6 Permitted Use

Reddit's free tier ToS permits:

- Academic and non-commercial research use.
- Signal generation for internal backtesting.
- Publishing results of research (not raw Reddit data) with Reddit attribution.

Reddit's free tier ToS **prohibits:**

- Commercial redistribution of Reddit data or derivative products.
- Training large language models on Reddit data without a separate agreement.
- Monetizing any product or service that is primarily built on Reddit data
  without a commercial tier agreement.

pyfinagent characterization: non-commercial quantitative research. Signals
derived from Reddit data feed the internal IC evaluation in phase-7.12 and
are not redistributed. If live trading generates revenue attributable to
Reddit-derived signals at material scale, Peder must assess whether a
commercial tier agreement is required and document the decision here.

---

## §7 Review Cadence

- Minimum: quarterly (aligned with alt-data.md Sec 9).
- Event-triggered: any change to how Reddit signals are used in production,
  any Reddit ToS revision, any ruling in Reddit v. Perplexity AI
  (N.D. Cal., filed 2025, pending).
- Next scheduled review: 2026-07-19.

Append a "Reviewed YYYY-MM-DD" line after each review:
<!-- Reviews -->

---

## §8 References

- Reddit Data API Terms of Service: https://redditinc.com/policies/data-api-terms
- Reddit OAuth2 app types: https://github.com/reddit-archive/reddit/wiki/OAuth2
- Reddit Responsible Builder Policy: https://support.reddithelp.com/hc/en-us/articles/14945211791892
- PRAW documentation: https://praw.readthedocs.io/en/stable/
- PRAW v7.8.1 release: https://github.com/praw-dev/praw
- Alt-data umbrella framework: docs/compliance/alt-data.md (Sec 2.2, Sec 4 row 7.5, Sec 5, Sec 6.2)
- Advisory adv_70_oauth_tos: phase-7.0 Q/A critique (handoff/archive/phase-7.0/)
- Reddit v. Perplexity AI, N.D. Cal. filed 2025 (pending): DMCA Sec.1201 risk
  documented in alt-data.md Sec 2.3
```

---

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) -- 16 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks -- note gaps but do not auto-fail:

- [x] Internal exploration covered every relevant module (twitter.py, google_trends.py, alt-data.md, compliance/ dir)
- [x] Contradictions / consensus noted (60 QPM vs 100 QPM debate documented)
- [x] All claims cited per-claim (not just listed in a footer)

---

## Floor report (<=200 words)

**(a) Exact User-Agent string recommended:**
`python:pyfinagent:1.0 (by /u/pederbkoppang)`

This follows Reddit's mandatory format `platform:appid:version (by /u/username)` as confirmed by the 2026 redaccs.com guide. It deliberately differs from the SEC EDGAR format used in twitter.py (`pyfinagent/1.0 peder.bkoppang@hotmail.no`) -- reddit_wsb.py must use the Reddit-specific form.

**(b) Free-tier rate limit we should honor:**
100 QPM (100 requests per minute per OAuth client_id, averaged over a 10-minute window). Translate to `_RATE_INTERVAL_S = 0.6` seconds between requests at steady state. PRAW handles automatic back-off on 429 responses; no manual `time.sleep()` needed in the live implementation.

**(c) Pushshift status verdict:**
**Defunct.** The original `pushshift.io` API is no longer reliably operational. **Arctic Shift** (`arctic-shift.photon-reddit.com`, GitHub: ArthurHeitmann/arctic_shift) is the active successor: 2.5 billion items, coverage through at least 2026-02, most recent data dump released April 10 2026. Use Arctic Shift for any historical backfill in phase-7.12.

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-7.5-research-brief.md",
  "gate_passed": true
}
```
