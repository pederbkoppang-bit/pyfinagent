# Reddit Data API -- Per-Vendor License Record

**Parent:** `docs/compliance/alt-data.md` Sec. 4 row 7.5
**Step:** phase-7.5 masterplan record
**Version:** 1.0 -- 2026-04-20
**Owner:** Peder (GitHub: pederbkoppang-bit)

---

## 1. Scope

Ingestion of public Reddit posts from the following subreddits for internal
signal generation and backtesting:

- r/wallstreetbets
- r/stocks
- r/investing

Starter set codified in `backend/alt_data/reddit_wsb.py::_STARTER_SUBS`. No
other subreddits are in scope under this license record. Any addition requires
updating this doc + the scaffold plus re-running the Reddit Responsible Builder
Policy review.

## 2. Access method

- **Protocol:** OAuth 2.0, **script app** (not installed app).
- **SDK:** PRAW >= 7.8.1 (phase-7.12 runtime).
- **User-Agent:** `python:pyfinagent:1.0 (by /u/pederbkoppang)`
  - Reddit mandates this exact format; spoofing or omission triggers
    throttling and eventual bans.
  - Differs from the SEC EDGAR format used elsewhere in `docs/compliance/
    alt-data.md` Sec. 5.1.
- **Credentials:** `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` env vars.
  - Registered at `https://www.reddit.com/prefs/apps` (app type = script).
  - Env vars are read ONLY inside `fetch_wsb_posts`; never at import time.
- **App registration + RBP submission** gated to phase-7.12. Scaffold does
  not perform either action.

## 3. Terms-of-Service record

Advisory `adv_70_oauth_tos` (Q/A qa_70_v1, 2026-04-19) is active: the Reddit
developer-app click-through IS contract formation. Tracking table:

| Gate | Status | Date |
|------|--------|------|
| Reddit app registration | pending | (before phase-7.12 first live call) |
| Responsible Builder Policy submission | pending | (before phase-7.12 first live call) |
| RBP approval | pending | (expected days for non-commercial) |
| First live call | not started | phase-7.12 |

Before any live call, the above gates are completed and this table is updated.

## 4. Rate limits

- **Free tier:** 100 QPM per OAuth client_id, averaged over a 10-minute window.
- **Unauthenticated:** 10 QPM (not used; OAuth required for every call).
- **Per-listing cap:** 1,000 posts per subreddit via pagination.
- **Client-side cap:** `_RATE_INTERVAL_S = 0.6` (= 100 QPM steady-state).
- **Response headers:** `X-Ratelimit-Used`, `X-Ratelimit-Remaining`,
  `X-Ratelimit-Reset` must be logged per request in phase-7.12 live impl.
- **PRAW back-off:** native 429 back-off is enabled by default; we do not
  disable it.

## 5. Retention & PII

- **Author identifier:** sha256 hash of the Reddit username. Raw username is
  never persisted. Implementation: `_hash_author` in the scaffold.
- **Raw text retention:** 90 days, then text redacted to the null string; the
  `author_hash`, `cashtag`, `sentiment_score`, `score`, and `upvote_ratio`
  columns are kept beyond 90 days for longitudinal analytics.
- **Deleted-author handling:** `author == "[deleted]"` maps to `author_hash
  = None`; we never generate a sha256 of the literal "[deleted]".
- **No IP, no geolocation, no email** is collected from Reddit posts or
  persisted anywhere.
- **GDPR / CCPA basis:** aggregated sentiment index; per-user SAR is
  serviceable because author_hash is reproducible from raw username without
  re-fetching.

## 6. Permitted use

**Permitted:**

- Internal research and backtesting of trading signals.
- Publication of aggregated results (e.g. "cashtag X appeared Y times per
  day") without raw Reddit text.
- Development of FinBERT-scored sentiment features for the quant pipeline.

**Prohibited (per Reddit Developer Agreement 2024):**

- Redistribution of raw Reddit content to third parties.
- Training proprietary LLMs on Reddit content.
- Building a commercial product that charges end-users a fee for Reddit-
  derived features without separately negotiating commercial access.

**Revenue-threshold trigger:** if live paper trading is promoted to
real-capital trading AND Reddit-derived signals contribute materially to
strategy P&L AND revenue exceeds $10k/year, re-assess the tier and contact
Reddit for a commercial agreement. Tracked as an open item on phase-7.12.

## 7. Review cadence

- **Quarterly review:** next review 2026-07-20; operator opens this doc and
  checks sections 2, 3, 4, 6 against current Reddit policy.
- **Event-triggered reviews:**
  - Any new subreddit added to `_STARTER_SUBS` -- same-day re-read + re-
    submission to RBP if scope shifts.
  - Reddit v. Perplexity AI (N.D. Cal., filed 2025, pending as of 2026-04-20)
    -- any ruling triggers a re-read of Sec. 2.3 of alt-data.md and Sec. 4 of
    this doc.
  - Any rate-limit or ToS change announced by Reddit (watch the Reddit Help
    Center developer category).

## 8. References

- Reddit Data API wiki (canonical): https://github.com/reddit-archive/reddit/wiki/API
- Reddit Data API Terms: https://redditinc.com/policies/data-api-terms
- Reddit Developer Agreement 2024 (Responsible Builder Policy)
- PRAW documentation: https://praw.readthedocs.io/
- Arctic Shift (Pushshift successor, for historical backfill): https://arctic-shift.photon-reddit.com/
- Internal: `docs/compliance/alt-data.md` Sec. 2.2 (ToS / X Corp framework),
  Sec. 2.3 (DMCA Sec.1201 / Reddit v Perplexity), Sec. 4 row 7.5, Sec. 5.5
  (PII redaction), Sec. 6.2 (retention).
- Internal: `handoff/current/phase-7.5-research-brief.md` (8 sources in full).
- Internal: advisory `adv_70_oauth_tos` (Q/A qa_70_v1, 2026-04-19).

---

**Reviewed:** 2026-04-20 (initial authoring)
