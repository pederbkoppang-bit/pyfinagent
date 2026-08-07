# Research Brief -- step 83.0.1 (tier: moderate)

**Topic:** Point-in-time integrity for the news corpus: kill the fabricated
`published_at` wall-clock fallback, stop fabricating `ingested_at` on backfill,
and derive/persist `effective_trade_date` = next trading session after the
publication-day session (entry at session OPEN).

**Gate:** PASSED. 5 external sources read in full, 23 URLs, recency scan done,
17 internal files/regions inspected. `coverage.audit_class=false`.

---

## 1. VERDICT ON THE CRITICAL DESIGN QUESTION: **(A)** -- no redesign needed

**BigQuery permits a NULLABLE time-unit partitioning column, and permits
relaxing REQUIRED -> NULLABLE in place on it.** No quarantine table, no DDL
recreate, no owner-approved DROP, no data movement.

> "The partitioning column must be either a scalar `DATE`, `TIMESTAMP`, or
> `DATETIME` column. **While the mode of the column can be `REQUIRED` or
> `NULLABLE`**, it cannot be `REPEATED` (array-based)."
> -- *Introduction to partitioned tables* > Limitations > Time-unit
> column-partitioned tables, https://cloud.google.com/bigquery/docs/partitioned-tables (2026-08-07)

> "`__NULL__`: Contains rows with NULL values in the partitioning column."
> -- same page, "Time-unit column partitioning". (2026-08-07)

> "The only supported modification you can make to a column's mode is changing
> it from `REQUIRED` to `NULLABLE`. Changing a column's mode from `REQUIRED` to
> `NULLABLE` is also called **column relaxation**."
> -- *Modifying table schemas*, https://cloud.google.com/bigquery/docs/managing-table-schemas (2026-08-07)

> "**ALTER COLUMN DROP NOT NULL statement** -- Removes a NOT NULL constraint
> from a column in a table in BigQuery. ... **Details:** If a column does not
> have a NOT NULL constraint the query returns an error. This statement is not
> supported for **external tables**."
> -- *DDL statements in GoogleSQL*, https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language (2026-08-07)

**Restriction audit (load-bearing).** The DDL page lists exactly three
constraints on `DROP NOT NULL`: (1) the column must currently carry the
constraint -- `published_at` is `REQUIRED`, satisfied; (2) not supported on
**external tables** -- `news_articles` is native, N/A; (3) "the name of the
**top level** column you're altering. Modifying subfields is not supported" --
`published_at` is top-level. **No partition-column exclusion exists on either
page, and the partitioned-tables Limitations section affirmatively blesses
`NULLABLE`.** The only in-place partitioning change BigQuery forbids is
converting an *unpartitioned* table to a partitioned one -- a different
operation.

**Live preconditions verified read-only (`client.get_table`, 2026-08-07):**

```
sunny-might-477607-p8:pyfinagent_data.news_articles
num_rows: 9   created 2026-08-07 07:48:46Z   modified 09:10:31Z
time_partitioning: TimePartitioning(field='published_at', type_='DAY')
clustering: ['source','ticker']   require_partition_filter: None
streaming_buffer: None            <- no active buffer; ALTER is unobstructed TODAY
published_at TIMESTAMP REQUIRED   ingested_at TIMESTAMP REQUIRED
```

`streaming_buffer: None` matters -- `insert_rows_json` parks rows in a streaming
buffer and an active buffer is the usual blocker for metadata mutation. Run the
ALTER **before** the next ingest.

### `ingested_at` (scope b): leave it REQUIRED

Criterion 3 asserts only the decidable property "the ingest timestamp is never a
value **earlier than** the publication timestamp". The **backfill-run-timestamp
reading satisfies it with no schema change**: a backfill run stamping the real
wall-clock moment of the run is truthful (that ingest genuinely happened),
trivially `>= published_at` for a historical article, and is exactly what the
criterion's negative case ("fails if `ingested_at` is written as the article's
own era") targets. `bq_writer.py:229-231` already documents this reasoning for
the sentiment table. **Recommend: no ALTER on `ingested_at`.**

---

## 2. THE STEP PREMISE IS INCOMPLETE -- three more fabrication sites

The step names one site (`fetcher.py`, the `or _now_iso()` fallback). **There
are four.** The other three sit UPSTREAM in the real production adapters:

| File:line | Code | Effect |
|---|---|---|
| `backend/news/fetcher.py:102` | `published_at=str(raw.get("published_at") or _now_iso())` | the named defect |
| `backend/news/sources/finnhub.py:143-145` | `if isinstance(ts,(int,float)) and ts>0: ... else: published_at = datetime.now(timezone.utc).isoformat()` | fabricates when `row["datetime"]` missing/<=0 |
| `backend/news/sources/benzinga.py:146-149` | `str(row.get("created") or row.get("updated") or "") or datetime.now(timezone.utc).isoformat()` | fabricates when both absent |
| `backend/news/sources/alpaca.py:145-148` | `str(row.get("created_at") or row.get("updated_at") or "") or datetime.now(timezone.utc).isoformat()` | fabricates when both absent |

**Consequence: fixing only `fetcher.py:102` fixes nothing on the real sources.**
By the time an adapter row reaches `_normalize`, `published_at` is already a
fabricated non-empty ISO string, so `raw.get("published_at") or _now_iso()`
never fires. For finnhub/benzinga/alpaca the fetcher fallback is **dead code**;
it only fires for adapters that omit the key entirely (the stub). This is the
`feedback_guards_stop_one_seam_short` class exactly.

A **fourth** failure mode exists that none of the four sites handle: benzinga
and alpaca pass the vendor string through unvalidated, so a *malformed but
non-empty* date (e.g. `"n/a"`) is neither caught nor fabricated -- it becomes a
garbage `published_at`. The quarantine predicate must be **parse-based**, not
merely presence-based: `missing OR unparseable -> NULL + quarantine`.

**Cycle trap:** 83.0's `test_c6_finnhub_benzinga_byte_unchanged`
(`test_phase_83_0_news_corpus_persistence.py:345-358`) runs
`git diff HEAD --name-only -- finnhub.py benzinga.py` and asserts empty. Editing
those files turns that test **RED in the working tree** and green again only
after commit. Expect it; do not "fix" it by reverting the source-level repair.

---

## 3. `effective_trade_date`: use "first session STRICTLY AFTER the publication
UTC date" (RuleA)

`exchange_calendars==4.13.2` is installed and pinned
(`backend/requirements.txt:23`); `pandas_market_calendars` and `holidays` are
**absent**. The repo already uses xcals at `backend/backtest/markets.py:168-213`
(`get_trading_calendar` -> `xcals.get_calendar(MARKET_CONFIG[m]["exchange"])`,
`is_trading_day` via `cal.is_session`). **Reuse that seam -- do not add a dep.**

Two candidate rules were tested live over every calendar day 2022-01-01..2026-06-30:

```python
# RuleA (RECOMMENDED)
cal.date_to_session(pd.Timestamp(pub_utc_date) + pd.Timedelta(days=1), direction="next")
# RuleB (over-embargoes)
cal.next_session(cal.date_to_session(pub_date, direction="next"))
```

| pub date | note | RuleA | RuleB |
|---|---|---|---|
| 2026-08-05 | Wed intraday | 2026-08-06 | 2026-08-06 |
| 2026-08-07 | Fri | 2026-08-10 | 2026-08-10 |
| 2026-08-08 | **Sat** | 2026-08-10 | 2026-08-11 |
| 2022-01-17 | **MLK holiday** | 2022-01-18 | 2022-01-19 |
| 2022-12-25 | **Christmas Sun** | 2022-12-27 | 2022-12-28 |

RuleB throws away a genuinely tradable session for weekend/holiday news (news
published Saturday IS knowable before Monday's open). **RuleA measured: 0
violations (`eff <= pub` never occurred) and `min sessions in (pub_date, eff] ==
1` exactly** -- i.e. RuleA is provably a >=1-session embargo and never more.

**Honesty proof for RuleA (UTC).** Entry is the session OPEN on a date `D' >
D`(=UTC publication date). US session opens are 13:30/14:30 UTC, so entry
`>= D+1 @ 13:30Z` which strictly exceeds any instant on date `D` (max
`D @ 23:59:59Z`). Entry therefore ALWAYS follows publication. Extract the date
in **UTC** -- it is both the safer choice and identical to `DATE(published_at)`,
the partition expression.

---

## 4. Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://cloud.google.com/bigquery/docs/partitioned-tables | 2026-08-07 | official doc | curl + tag-strip (JS-rendered; `feedback_gcloud_docs_fetch`) | "the mode of the column can be REQUIRED or NULLABLE"; "`__NULL__`: Contains rows with NULL values in the partitioning column" |
| 2 | https://cloud.google.com/bigquery/docs/managing-table-schemas | 2026-08-07 | official doc | curl + tag-strip | "The only supported modification ... REQUIRED to NULLABLE ... column relaxation"; also "You cannot add a REQUIRED column to an existing table schema" |
| 3 | https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language | 2026-08-07 | official doc | curl + tag-strip | `ALTER COLUMN [IF EXISTS] column DROP NOT NULL`; restrictions = has-constraint / not-external / top-level only |
| 4 | https://arxiv.org/html/2402.06698v1 (FNSPID, 15.7M articles) | 2026-08-07 | peer-reviewed preprint | WebFetch (native arXiv HTML) | **Negative finding:** no timestamp/UTC handling documented, "no explicit alignment rule is stated", **zero counts of dropped records**, no look-ahead discussion |
| 5 | https://arxiv.org/pdf/2606.29734 (EarningsInOne, Yu/Liu/Zhang/He, 2026-06-29) | 2026-08-07 | peer-reviewed preprint | curl + **pdfplumber 0.11.9** (arXiv `/html/` 404s for this ID; step-3 of the fetch chain) | "**NextOpen**: 09:30am open on the following day, **standard entry anchor in both communities**"; "AMC trades at next session open"; "quantitative surprise ... **is largely eliminated by next market open**; qualitative ECT sentiment **peaks on the next trading day, real and tradeable**" |

## 5. Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.sciencedirect.com/science/article/abs/pii/S0922142524000124 | paper (overnight earnings / preopening price discovery) | paywalled abstract |
| https://www.sciencedirect.com/science/article/abs/pii/S0278425422001181 | paper (timing of information arrival, overnight returns) | paywalled |
| https://arxiv.org/pdf/2511.15123 | preprint (causal inference in event studies) | superseded by #5 for entry timing |
| https://arxiv.org/pdf/2605.31201 | preprint (Point-in-Time Financial RAG) | PIT-adjacent, not timestamp-integrity |
| https://arxiv.org/pdf/2406.15667 | preprint (high-frequency event studies) | intraday scope, explicitly out of step scope |
| https://arxiv.org/pdf/2512.23847 | preprint (Detecting Lookahead Bias in LLM Forecasts) | LLM-memorisation bias, different mechanism |
| https://arxiv.org/pdf/2309.17322 | preprint (look-ahead bias, GPT sentiment) | same |
| https://arxiv.org/pdf/2605.24564 | preprint (Summoning the Oracle to Slay It) | same |
| https://arxiv.org/pdf/2510.10526 | preprint (LLM+RL sentiment trading) | no timestamp discipline |
| https://bookdown.org/mike/data_analysis/sec-event-studies.html | textbook chapter | community tier |
| https://www.eventstudytools.com/other-event-study-types | industry | community tier |
| https://paperswithbacktest.com/course/look-ahead-bias-llm-trading | blog | community tier; source of the 4pm-cutoff convention note |
| https://huggingface.co/datasets/rangeva/financial-news-dataset | dataset card | schema reference only |
| https://github.com/felixdrinkall/financial-news-dataset | dataset repo | schema reference only |
| https://cloud.google.com/bigquery/docs/samples/bigquery-relax-column | doc sample | code sample of #2 |
| https://cloud.google.com/bigquery/docs/samples/bigquery-relax-column-load-append | doc sample | load-job relaxation path, not needed |
| https://cloud.google.com/bigquery/docs/creating-partitioned-tables | official doc | creation, not alteration |
| https://cloud.google.com/bigquery/docs/managing-tables | official doc | general table ops |

**Query-variant discipline (3 per topic):** current-year/last-2-year
(`"...2025 2026 look-ahead bias"`), year-less canonical (`"event study news
announcement next open entry timing overnight embargo point-in-time backtest"`,
`"news dataset missing publication timestamp quarantine data quality financial
NLP"`), and vendor-scoped (`"BigQuery ALTER COLUMN DROP NOT NULL ... partitioning
column"`, domain-filtered to cloud.google.com).

## 6. Recency scan (2024-2026)

Performed. **Found 4 findings that supersede or sharpen the canonical sources.**

1. **EarningsInOne (2026-06)** supersedes the classic PEAD entry convention: it
   measures BOTH anchors on the same events and finds the fast numeric channel
   "largely eliminated by next market open" while the **slow language channel
   peaks on the next trading day**. This is the single most decision-relevant
   new result for this step (see §7.1).
2. **Martineau (2022), cited in #5:** "multi-day PEAD has been non-existent for
   non-microcap stocks since 2006 ... near-instantaneous price jumps replace
   gradual drift" (Christensen et al. 2025). This *qualifies* the step's
   premise -- see the adversarial note in §7.
3. **FNSPID (2024-02)** is the canonical modern news corpus and documents
   **none** of timestamp provenance, alignment rule, or drop counts -- evidence
   that this step's discipline is above field norm, not below it.
4. **BigQuery docs (fetched 2026-08-07)** are current; no deprecation of
   `ALTER COLUMN DROP NOT NULL` and no new partition-column restriction.

## 7. Key findings

1. **Next-session-open is the field-standard anchor, not conservatism.**
   "**NextOpen**: 09:30am open on the following day, **standard entry anchor in
   both communities**" (EarningsInOne 2026, arXiv:2606.29734, accessed
   2026-08-07). The step's rule is the mainstream convention.
2. **The one-session embargo costs pyfinagent's channel ~nothing.** "quantitative
   surprise is fast: IC peaks at announcement and decays to near zero by the
   next market open; **qualitative sentiment is slow: predictive content peaks
   on the next trading day**" (ibid.). pyfinagent's thematic/weeks-to-months
   channel is the *slow* one; it peaks exactly where the embargo puts entry.
3. **Fabricated timestamps destroy the ability to tell the two channels apart.**
   "without precise event-level timestamps, numeric signal exhausted within
   minutes is **indistinguishable from** language signal maturing overnight"
   (ibid.). This is the sharpest available statement of *why* 83.0.1 is P0.
4. **The after-close mapping rule is explicit:** "AMC trades at **next session
   open** unless entered via extended hours, BMO at the regular open" (ibid.).
   pyfinagent has no extended-hours path, so next-session-open is the only
   available anchor for both AMC and intraday news.
5. **[ADVERSARIAL] A one-session embargo does kill the fast channel entirely.**
   Martineau (2022) / Christensen et al. (2025), via #5: prices "fully reflect
   earnings surprises on the announcement date itself". So this step forecloses
   any future numeric-surprise strategy on this corpus. That is the correct
   trade for a daily-cycle $0 system (the step says so), but the contract should
   record it as a **deliberate, priced concession**, not an invisible one.
6. **Field norm is worse, not better.** FNSPID (15.7M records, the most-cited
   modern corpus) reports "zero counts of dropped records" and no alignment
   rule. Criterion 6's "RECORD the quarantine count rather than assert a
   threshold" is the right call and is above field practice.
7. **A 4pm-cutoff intraday convention exists** (headline before 16:00 -> day t;
   after -> t+1) but requires timestamp precision the step explicitly rules out
   for backfilled vendor data. RuleA is strictly safer in the same direction.

## 8. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/news/fetcher.py` | 102 | **named defect**: `published_at=str(raw.get("published_at") or _now_iso())` | line RE-DERIVED post-83.0 (step text said 99) |
| `backend/news/fetcher.py` | 105 | `ingested_at=_now_iso()` unconditional | scope (b); truthful for live, needs backfill-run value |
| `backend/news/fetcher.py` | 91-92 | `_now_iso()` helper | the mutation target |
| `backend/news/fetcher.py` | 95-118 | `_normalize(raw, source_name, provenance)` | the single insertion point for NULL + quarantine + `effective_trade_date` |
| `backend/news/fetcher.py` | 138-187 | `run_once(source_names, dry_run, dedup, provenance)` | already threads `provenance`; add counter reset/report here |
| `backend/news/fetcher.py` | 58-74 | `NormalizedArticle` TypedDict | must gain `effective_trade_date`; `published_at` type -> `str | None` |
| `backend/news/fetcher.py` | 197-234 | `StubSource` (3 articles, all with `published_at`) | fixture base; needs a no-timestamp sibling |
| `backend/news/sources/finnhub.py` | 143-145 | **2nd fabrication** `else: published_at = datetime.now(...)` | NOT in step scope as written |
| `backend/news/sources/benzinga.py` | 146-149 | **3rd fabrication** `... or datetime.now(...)` | NOT in step scope as written |
| `backend/news/sources/alpaca.py` | 145-148 | **4th fabrication** `... or datetime.now(...)` | NOT in step scope as written |
| `backend/news/bq_writer.py` | 50-52, 72-80 | `_WRITE_FAILURES` + `_write_lock` + `_record_failure` | **the counter idiom to REUSE** |
| `backend/news/bq_writer.py` | 58-69 | `write_failure_count()` / `reset_write_failures_for_test()` | accessor pattern to mirror |
| `backend/news/bq_writer.py` | 148-187 | `_serialize_article` | must pass `published_at=None` through unmodified + add `effective_trade_date` |
| `backend/news/bq_writer.py` | 169 | `"article_id": article.get("article_id") or ""` | **trap**: the `or ""` idiom must NOT be copied to `published_at` |
| `backend/news/bq_writer.py` | 229-231 | comment already distinguishing truthful-wall-clock from fabrication | cite it in the contract |
| `backend/backtest/markets.py` | 168-190 | `get_trading_calendar(market)` -> xcals | **reuse for the calendar** |
| `backend/backtest/markets.py` | 192-213 | `is_trading_day` via `cal.is_session`, fail-OPEN | note: fail-open is wrong for an embargo -- see §9 |
| `scripts/migrations/add_news_sentiment_schema.py` | 69-92 | `DDL_NEWS_ARTICLES` (`published_at TIMESTAMP NOT NULL`, `PARTITION BY DATE(published_at)`) | must drop `NOT NULL` for fresh creates |
| `scripts/migrations/add_news_sentiment_schema.py` | 122-133 | `REQUIRED_MODES` map | flip `news_articles.published_at` -> `"NULLABLE"` |
| `scripts/migrations/add_news_sentiment_schema.py` | 136-156 | `verify_post_condition` | already the right read-back gate; reuse verbatim |
| `scripts/smoketest/phase6_e2e.py` | 85-117, 262-300 | `--backfill` flag -> `provenance="backfill"` | **the backfill entry point already exists** |
| `backend/tests/test_phase_83_0_news_corpus_persistence.py` | 136-152 | `_FakeClient` + `_ROW` | **the seam + fixture convention to copy** |
| `backend/tests/test_phase_83_0_news_corpus_persistence.py` | 173-189 | negative controls (success / empty input) | the negative-control convention to copy |
| `backend/tests/test_phase_83_0_news_corpus_persistence.py` | 345-358 | `test_c6_finnhub_benzinga_byte_unchanged` | **goes RED while 83.0.1 edits those files** |
| `backend/tests/test_phase_83_0_news_corpus_persistence.py` | 70-82 | `_resolve_schema` live-or-snapshot oracle + non-empty assert | reuse; **update the snapshot** to NULLABLE |

## 9. Pitfalls

1. **Fail-open calendar is wrong here.** `markets.py:204` returns `True` when
   xcals is missing -- correct for "don't block a trade", **dangerous for an
   embargo** (a missing calendar would silently produce a same-day
   `effective_trade_date`). The derivation must **fail CLOSED**: no calendar ->
   no `effective_trade_date` -> quarantine. xcals is pinned, so this is a guard,
   not a likely path.
2. **`or ""` / `or 0` coercion**, the `feedback_fabricated_safe_80_36` class.
   `_serialize_article` uses `x or ""` on six columns. Applying that idiom to
   `published_at` would convert NULL back into a value. Discriminate on
   **presence**, not truthiness.
3. **`insert_rows_json` is STRICT** (`bq_writer.py:25-31`): adding
   `effective_trade_date` to the row dict **before** the column exists in BQ
   fails the WHOLE batch. Migration must land before the writer change.
4. **A zero-row counter is not evidence.** If the fixture corpus has no
   missing-timestamp article, the quarantine counter reads 0 and every guard
   passes vacuously (`feedback_a_green_suite_can_be_blind`). Assert the counter
   **strictly increases** from a known-missing fixture, per 83.0's C4 idiom.
5. **`DATE(published_at)` on a NULL is NULL**, and NULL rows land in `__NULL__`.
   Any downstream query with a partition-date predicate will silently exclude
   quarantined rows -- correct here (they must never reach a backtest), but it
   must be *stated*, not accidental.
6. **The 9 existing rows** were written under the old fabricating path. They are
   suspect provenance; the live_check should show them alongside new rows so the
   operator can see the change of behaviour.

## 10. Application to pyfinagent (external findings -> file:line)

- Finding 1/4 (NextOpen is standard; AMC->next session open) -> implement RuleA
  in `backend/news/fetcher.py:95-118` using `backend/backtest/markets.py:168`
  `get_trading_calendar("US")`, fail-closed.
- Finding 2 (slow channel peaks next trading day) -> the contract's
  justification section; the embargo is free for this channel.
- Finding 3 (fabricated timestamps collapse fast/slow) -> the P0 rationale for
  killing all **four** fabrication sites (§2), not just `fetcher.py:102`.
- Finding 5 [ADVERSARIAL] -> record the foreclosed fast-channel strategy as a
  priced concession in `contract.md`.
- Finding 6 (FNSPID reports no drop counts) -> criterion 6's RECORD-don't-
  threshold design; mirror `bq_writer.py:58-69` accessors.
- Verdict (A) -> `scripts/migrations/add_news_sentiment_schema.py:71,122-133`.

---

## 11. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (3 official vendor docs
      via curl+tag-strip per `feedback_gcloud_docs_fetch`; 1 arXiv native HTML;
      1 arXiv PDF via pdfplumber per research-gate.md step 3)
- [x] 10+ unique URLs total -- **23**
- [x] Recency scan (2024-2026) performed + reported (§6, 4 findings)
- [x] Full pages read, not abstracts
- [x] file:line anchors for every internal claim (§8)

Soft checks:
- [x] Internal exploration covered fetcher, all 3 adapters, bq_writer, the
      migration, the 83.0 test file, the calendar seam, the backfill entry point
- [x] Contradictions noted (§7.5 adversarial; RuleA vs RuleB; 4pm-cutoff variant)
- [x] Claims cited per-claim
- Gap: `handoff/archive/phase-83.0/contract.md` contains **82.54's** contract
  (archive rotation lag), so 83.0's own contract text was unavailable; the 83.0
  decisions were recovered from the test file + migration + `bq_writer` comments
  instead, which are authoritative.

## 12. Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 18,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 17,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Verdict (A): BigQuery explicitly permits a NULLABLE time-unit partitioning column ('the mode of the column can be REQUIRED or NULLABLE') and ALTER COLUMN DROP NOT NULL has no partition-column restriction, so published_at can be relaxed in place; NULL rows land in __NULL__. Table is 9 rows with an empty streaming buffer, so the ALTER is unobstructed today. ingested_at should stay REQUIRED -- a real backfill-run timestamp satisfies criterion 3. Biggest finding: the step premise is incomplete -- three MORE wall-clock fabrications exist upstream in finnhub.py:145, benzinga.py:146-149 and alpaca.py:145-148, which make fetcher.py:102 dead code for all real sources. effective_trade_date should use 'first session strictly after the publication UTC date' via the installed exchange_calendars 4.13.2 (0 violations, exactly 1 session of separation, measured over 2022-2026). EarningsInOne (arXiv:2606.29734) confirms NextOpen is the standard anchor and that the slow language channel peaks on the next trading day, so the embargo is free for pyfinagent's thematic channel.",
  "brief_path": "handoff/current/research_brief_83.0.1.md",
  "gate_passed": true
}
```
