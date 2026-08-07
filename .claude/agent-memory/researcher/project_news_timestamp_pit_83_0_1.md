---
name: news-timestamp-pit-83-0-1
description: Step 83.0.1 news PIT integrity -- BQ ALLOWS a NULLABLE partition column; the step premise was INCOMPLETE (3 more wall-clock fabrications upstream in the adapters make fetcher.py's fallback dead code); RuleA next-session derivation
metadata:
  type: project
---

Step 83.0.1 (news corpus point-in-time integrity), researched 2026-08-07.

**BigQuery permits a NULLABLE time-unit PARTITIONING column.** Verbatim, from
`cloud.google.com/bigquery/docs/partitioned-tables` Limitations: *"The
partitioning column must be either a scalar DATE, TIMESTAMP, or DATETIME
column. While the mode of the column can be REQUIRED or NULLABLE, it cannot be
REPEATED."* `ALTER COLUMN ... DROP NOT NULL` lists exactly three restrictions
(must have the constraint / not external tables / top-level only) and **no
partition-column exclusion**. NULL-keyed rows land in the `__NULL__` partition.
The only forbidden in-place partitioning change is unpartitioned -> partitioned.

**Why:** the step assumed this might be impossible and might force a quarantine
table or an owner-approved DROP+recreate. It does not. Verdict (A).

**How to apply:** whenever a REQUIRED partition column blocks a NULL-bearing
design, relax it in place -- but check `streaming_buffer` first (an active
buffer from `insert_rows_json` is the usual blocker for metadata mutation).

---

**THE PREMISE WAS INCOMPLETE -- the class matters more than the instance.**
The step named ONE fabrication site (`fetcher.py`'s `or _now_iso()`). There are
FOUR. Three sit upstream in the production adapters:
`sources/finnhub.py:143-145` (`else: published_at = datetime.now(...)`),
`sources/benzinga.py:146-149` and `sources/alpaca.py:145-148` (both
`str(a or b or "") or datetime.now(...)`).

**Consequence:** because each adapter already substitutes a non-empty ISO
string, `raw.get("published_at") or _now_iso()` in the fetcher **never fires
for any real source** -- the named defect is DEAD CODE on the live path, and
fixing only it fixes nothing. Same class as
[[feedback_guards_stop_one_seam_short]].

A 4th failure mode none of them handle: benzinga/alpaca pass a *malformed but
non-empty* vendor string straight through. The quarantine predicate must be
**parse-based**, not presence-based.

**Trap:** 83.0's `test_c6_finnhub_benzinga_byte_unchanged` does
`git diff HEAD --name-only` on those adapters -- editing them turns that test
RED in the working tree and green again only after commit.

---

**`effective_trade_date` derivation.** `exchange_calendars==4.13.2` is
installed + pinned (`backend/requirements.txt:23`); `pandas_market_calendars`
and `holidays` are ABSENT. Reuse `backend/backtest/markets.py:168`
`get_trading_calendar()`.

RuleA = *first session strictly after the publication UTC date*:
`cal.date_to_session(pub_date + 1day, direction="next")`. Measured over
2022-01-01..2026-06-30: **0 violations, exactly 1 session of separation.**
The tempting alternative (`next_session(date_to_session(d, "next"))`)
**over-embargoes weekend/holiday news by one extra session** (Saturday news
gets Tuesday instead of Monday) -- it discards a genuinely tradable open.

Note `markets.py:204` `is_trading_day` fails **OPEN** (returns True when xcals
is missing). That is right for "don't block a trade" and **wrong for an
embargo** -- the derivation must fail CLOSED into quarantine.

---

**External anchor worth reusing:** EarningsInOne, arXiv:2606.29734 (2026-06).
*"NextOpen: 09:30am open on the following day, standard entry anchor in both
communities"*; *"quantitative surprise ... is largely eliminated by next market
open; qualitative ECT sentiment peaks on the next trading day, real and
tradeable"*. So a one-session embargo is FREE for a slow thematic channel and
FATAL for a fast numeric one -- record the latter as a priced concession.
Its `/html/` URL 404s; fetch via `curl` + **pdfplumber** (installed, 0.11.9).
Contrast FNSPID (arXiv:2402.06698): the most-cited modern news corpus documents
NO timestamp provenance, NO alignment rule and reports ZERO drop counts.
