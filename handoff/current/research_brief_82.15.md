# Research Brief — step 82.15: wire the READ side of `realtime_start` (point-in-time macro)

Tier: **moderate**. Audit-class: **no**. Date: 2026-08-03. Status: COMPLETE.

## Question

82.0 added + populated `financial_reports.historical_macro.realtime_start` but
never wired any reader (`grep -n realtime_start backend/backtest/*.py
backend/services/*.py` → nothing). Every macro-conditioned backtest still reads
values that did not exist at the cutoff. Deliverables: exact change at BOTH read
paths; MEASURED answer to the NULL-vintage/coverage question; a defensible
treatment for pre-migration rows with the cost stated plainly.

---

## 1. MEASURED: the live table (queried 2026-08-03, `us-central1`)

`sunny-might-477607-p8.financial_reports.historical_macro` — 4729 rows,
7 series. Schema:

| column | type |
|---|---|
| series_id | STRING |
| date | **STRING** (not DATE — see §4 trap) |
| value | FLOAT |
| ingested_at | TIMESTAMP |
| market | STRING |
| base_currency | STRING |
| `realtime_start` | **DATE** |

**`realtime_start` IS NULL: 0 rows / 4729.** The premise in the spawn prompt
("pre-migration rows were backfilled") is right, but there are no NULLs left to
special-case — the migration `add_macro_realtime_start.py` UPDATE completed. The
problem is not NULLs, it is that **every historical row carries a 2026 vintage**.

### Vintage cohorts

| `realtime_start` | rows | series | observation-date span |
|---|---|---|---|
| 2026-03-22 | 1652 | 7 | 2023-01-01 .. 2025-12-31 |
| 2026-03-25 | 2760 | 7 | 2018-01-01 .. 2022-12-30 |
| 2026-08-03 | 317 | 7 | 2026-01-01 .. 2026-07-31 |

Per-series `realtime_start` MIN is **2026-03-22 for all 7 series** (CPIAUCSL,
DGS10, FEDFUNDS, GDP, T10Y2Y, UMCSENT, UNRATE).

### THE CRUX — strict-filter survival, measured

Query: `COUNTIF(date <= cutoff AND realtime_start <= DATE(cutoff))`.

| cutoff | obs-only rows (series) | **strict PIT rows (series)** |
|---|---|---|
| 2018-06-01 | 236 (7) | **0 (0)** |
| 2020-01-01 | 1107 (7) | **0 (0)** |
| 2023-01-01 | 2765 (7) | **0 (0)** |
| 2025-06-01 | 4096 (7) | **0 (0)** |
| 2026-03-23 | 4535 (7) | 1652 (7) |
| 2026-03-26 | 4541 (7) | 4412 (7) |
| 2026-04-01 | 4554 (7) | 4412 (7) |
| 2026-08-01 | 4729 (7) | 4412 (7) |

**A naive strict filter returns ZERO macro rows for every cutoff before
2026-03-22 — i.e. it blanks 100% of the macro features across the entire
2018–2025 backtest window.** The team lead's suspicion is confirmed with
numbers: shipping the obvious one-line fix destroys the backtest. Only the
~4.5-month tail from 2026-03-22 onward has usable strict vintages, and inside
that tail the vintages are *wrong in the pessimistic direction* (a 2018 CPI
print is marked as first-known 2026-03-25).

### What "blanked" actually looks like downstream

`backend/backtest/historical_data.py:268-275`:

```python
        # ── Macro ────────────────────────────────────────────────
        if macro:
            features["fed_funds_rate"]     = macro.get("FEDFUNDS", {}).get("value")
            features["cpi_yoy"]            = macro.get("CPIAUCSL", {}).get("value")
            features["unemployment_rate"]  = macro.get("UNRATE", {}).get("value")
            features["yield_curve_spread"] = macro.get("T10Y2Y", {}).get("value")
            features["consumer_sentiment"] = macro.get("UMCSENT", {}).get("value")
            features["treasury_10y"]       = macro.get("DGS10", {}).get("value")
```

An empty `macro` dict makes the `if macro:` guard **false**, so the six keys are
**not set at all** — they are absent from the feature dict, not `None`. Any
consumer doing `features["cpi_yoy"]` raises `KeyError`; any consumer doing
`features.get("cpi_yoy")` silently gets `None`. Either way the failure is
**silent-by-omission across the whole sample**, which is the worst shape: a
backtest would run to completion and report a Sharpe with the macro block
quietly amputated. This is exactly the `absence-becomes-affirmative` class the
project has been bitten by before.

---

## 2. The exact change — BOTH read paths in `backend/backtest/cache.py`

### (a) `preload_macro()` — `cache.py:240`, SELECT at `:254-258`

```python
    query = f"""
        SELECT series_id, value, date
        FROM `{_table("historical_macro")}`
        ORDER BY series_id, date DESC
    """
```

**`realtime_start` is NOT in the projection.** The per-series list built at
`:348-357` stores only `{"value": row["value"], "date": row["date"]}`, so
`_macro_full` physically cannot support a vintage filter. **The fast path is
un-fixable without changing this SELECT and this dict.** Required:

- add `realtime_start` to the SELECT list at `:255`;
- carry it into the cached entry at `:351` →
  `{"value": ..., "date": ..., "realtime_start": row.get("realtime_start")}`.

Note `_macro_full` is documented at `:68` as `series_id -> [{value, date}, ...]`
— update that comment or the next reader inherits the same blind spot.

### (b) fast path inside `cached_macro()` — `cache.py:471`, loop at `:478-486`

```python
    if _macro_full:
        _cache_stats["hits"] += 1
        result = {}
        for series_id, entries in _macro_full.items():
            # entries sorted by date DESC — find first entry <= cutoff_date
            for entry in entries:
                if str(entry["date"]) <= cutoff_date:
                    result[series_id] = entry
                    break
```

The `break` on the first `date <= cutoff` is correct for a DESC-sorted list
*only* while the predicate is monotone in `date`. Adding a vintage predicate
makes it **non-monotone** — a row can pass on `date` and fail on
`realtime_start` — so the condition must become an `and` and the loop must
CONTINUE scanning rather than break on a date-only match:

```python
                if str(entry["date"]) <= cutoff_date and _vintage_ok(entry, cutoff_date):
                    result[series_id] = entry
                    break
```

(Keep the `break` — it now fires on the first entry passing BOTH predicates,
which is still the max-date visible row, because the list is date-DESC. What
must NOT happen is `break`-ing on a date-only match and discarding the series.)

### (c) BQ fallback inside `cached_macro()` — `cache.py:494-515`

```python
    query = f"""
        SELECT series_id, value, date
        FROM (
            SELECT series_id, value, date,
                   ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY date DESC) as rn
            FROM `{_table("historical_macro")}`
            WHERE date <= @cutoff
        )
        WHERE rn = 1
    """
```

Add the vintage predicate to the inner `WHERE`. **Type trap:** `date` is STRING
and the bound param is `ScalarQueryParameter("cutoff", "STRING", ...)` at
`:509`, but `realtime_start` is **DATE**. `realtime_start <= @cutoff` with a
STRING param will fail with a type error. Write
`realtime_start <= DATE(@cutoff)` (or add a second DATE-typed param). Also
`ROW_NUMBER() ... ORDER BY date DESC` ties are unbroken today; if a vintage
tiebreak is ever wanted it goes here (`ORDER BY date DESC, realtime_start DESC`).

`preload_macro` is called from `backtest_engine.py:317` and
`scripts/diag_label_pin.py:27`; `cached_macro` from
`historical_data.py:48` (→ the 6 features) and
`backend/agents/mcp_servers/data_server.py:185` (`get_macro`, cutoff =
`date.today()` — unaffected by any vintage rule since today ≥ every vintage).

`backend/metrics/sortino.py:108` queries
`{project}.pyfinagent_data.historical_macro` — **that table does not exist**
(verified: `404 NotFound` on `pyfinagent_data.historical_macro`; the real table
is in `financial_reports`). That call is dead and fails open to the DTB3 CSV /
0.045 fallback. Out of scope for 82.15, but do not "fix" the vintage there
believing it runs.

**No existing project convention for PIT joins on macro.** The only prior art:
`backend/tools/screener.py:42` and `backend/backtest/candidate_selector.py:121`
both *raise* `NotImplementedError` when `as_of` is passed (PIT universe
membership deliberately unimplemented — fail-loud, not fail-silent). That is a
useful precedent for the treatment recommended in §5.
`backend/econ_calendar/sources/fred_releases.py:55-56` already sends
`realtime_start`/`realtime_end` to the FRED releases endpoint, so the ALFRED
vocabulary is already in the codebase.

---

## 3. MEASURED: revisions and observed lags

**Duplicate `(series_id, date)` pairs: 0 / 4729.** By construction —
`data_ingestion.py:295-312` (`_get_existing_macro`) skips any `(series_id,
date)` already present, so **this table can never record a revision.** It holds
whichever print we happened to ingest first, and for the 2018–2025 span that is
the *fully revised* value as of March 2026. Consequence for the fix: wiring
`realtime_start` removes the **publication-lag** look-ahead only. The
**revision** look-ahead is structurally out of reach without an ALFRED backfill.
Say this out loud in the contract; do not let 82.15 be reported as "look-ahead
fixed".

Observed lag (`realtime_start - date`) at the data frontier — the newest
observation per series, which is the tightest lag this table can evidence:

| series | newest obs | vintage | lag |
|---|---|---|---|
| DGS10 | 2026-07-30 | 2026-08-03 | 4d |
| T10Y2Y | 2026-07-31 | 2026-08-03 | 3d |
| CPIAUCSL | 2026-06-01 | 2026-08-03 | 63d |
| FEDFUNDS | 2026-06-01 | 2026-08-03 | 63d |
| UMCSENT | 2026-06-01 | 2026-08-03 | 63d |
| UNRATE | 2026-06-01 | 2026-08-03 | 63d |
| **GDP** | **2026-04-01** | **2026-08-03** | **124d** |

The GDP 124d confirms the ~120d figure in the 82.0 migration docstring. **But
these are upper bounds, not true FRED release lags** — every 2026 row shares one
vintage (2026-08-03) because the whole cohort was ingested in a single batch
today, so the lag is inflated by "when our ingest happened to run", not by when
FRED published. The true lags are shorter (H.15 daily ≈ 1 business day; monthly
series dated at month START and released mid-following-month ≈ 45d; GDP advance
≈ 30d after quarter END = ~120d from quarter-start dating). Erring long is the
safe direction.

---

## 4. External research

### Read in full (7; ≥5 required — counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://raw.githubusercontent.com/mortada/fredapi/master/README.md | 2026-08-03 | official-adjacent lib docs | curl (raw MD) | Canonical ALFRED semantics: "every *observation* can have three dates: *date*, *realtime_start* and *realtime_end*. date: the date the value is for; realtime_start: **the first date the value is valid**; realtime_end: the last date the value is valid." Worked GDP-2014Q1 example: three prints (17149.6 → 17101.3 → 17016.0) with `realtime_start` 2014-04-30 / 05-29 / 06-25. |
| https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists | 2026-08-03 | official (Fed) | curl + tag-strip (WebFetch 403) | "The real-time data set consists of vintages, or snapshots, of time series of major macroeconomic variables… to verify empirical results, to analyze policy, or to forecast." Canonical cite = Croushore & Stark, *J. Econometrics* 105 (2001) 111-30. Note the RTDSM stores **complete vintage history** AND a separate "first-, second-, third-release values" product — i.e. the gold standard is a per-vintage row, not a single value + a date. |
| https://arxiv.org/html/2607.04958v1 | 2026-08-03 | peer-review-track preprint (2026) | WebFetch | Formalizes look-ahead-freedom as **temporal non-interference**: "for every decision emitted at epoch t, perturbing any base datum whose availability exceeds t leaves that decision unchanged." Verification via effects that are "a **conservative upper bound on the availability** of every base datum whose value can flow into the term's result", enforced at a `T-Decide` rule. Checker is **fail-closed** — "conservatively rejects" what it cannot prove. **[ADVERSARIAL to our approach]** it assumes "availability is exogenous, fixed by the data source… ingested immediately and unconditionally" and *does not address* unknown/missing availability — the framework "treats availability as given and complete—not inferred, estimated, or partially known." |
| https://v1-docs.xtdb.com/concepts/bitemporality/ | 2026-08-03 | official docs | WebFetch | Bitemporal split: **transaction time** = "the point at which data arrives into the database… an audit trail"; **valid time** = "what users will typically use for query purposes". Valid time "supports retroactive and proactive operations — data can be written to the past or future — while transaction time always moves forward." Our `ingested_at` is transaction time; `date` is valid time; **`realtime_start` is a third thing — availability/knowledge time — and backfilling it from transaction time conflates the two.** |
| https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/ | 2026-08-03 | curriculum (CFA L2) | WebFetch | "Look-ahead bias emanates from the use of unavailable information by an investor during the historical periods over which a backtest is conducted." Names data revisions explicitly: "macroeconomic data can be revised severally. The revised data often replaces the old data, and an analyst will, as such, use information that was unavailable to them." Remedy: "To avert its occurrence, we use point-in-time data." |
| https://www.pfolio.io/academy/look-ahead-bias | 2026-08-03 | practitioner academy | WebFetch | Magnitude: Bailey & López de Prado — "look-ahead bias can inflate annualised returns by 100–500 basis points"; Asness/Frazzini/Pedersen — restated-data quality strategies beat as-reported by "approximately 100 basis points per year". Standard mitigation = "apply conservative reporting lags to all non-price data" (60–90d quarterly, 90–180d annual). Warns the fix is unglamorous: "The corrected return is always lower than the contaminated return." |
| https://starqube.com/point-in-time-data/ | 2026-08-03 | industry vendor | WebFetch | "Strategies relying on GDP data… show **15–25% higher Sharpe ratios** in backtests using final revised figures compared to initial releases." Notably **silent** on proxy construction when release timestamps are unavailable — asked directly, the page offers no proxy methodology. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://alfred.stlouisfed.org/help/downloaddata | official | HTTP 403 on WebFetch AND on curl-with-UA (bot wall) |
| https://fred.stlouisfed.org/docs/api/fred/series_observations.html | official | HTTP 403 on both paths (same wall) |
| https://macrosynergy.com/academy/notebooks/point-in-time-economics/ | industry | JS "Checking your browser" interstitial; snippet is load-bearing (see below) |
| https://macrosynergy.com/academy/what-are-macro-quantamental-indicators/ | industry | same JS wall |
| https://www.sciencedirect.com/science/article/abs/pii/S0304407601000720 | peer-reviewed | paywall (Croushore & Stark 2001) |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=244554 | preprint | abstract only |
| https://www.philadelphiafed.org/the-economy/macroeconomics/a-real-time-data-set-for-macroeconomists-does-the-data-vintage-matter | official | duplicate of RTDSM landing page content |
| https://arxiv.org/abs/2605.24564 | preprint 2026 | recency scan; LLM-specific look-ahead, off our axis |
| https://arxiv.org/pdf/2605.23959 | preprint 2026 | recency scan; decision-time leakage benchmark |
| https://arxiv.org/html/2607.11889 | preprint 2026 | recency scan; PiT language models |
| https://www.bloomberg.com/company/press/bloomberg-introduces-point-in-time-economic-data-to-power-quantitative-research-and-strategy-development/ | vendor press | paywall/press release |
| https://info.ceicdata.com/ceic-launches-point-in-time-data | vendor press | marketing |
| https://github.com/mortada/fredapi | repo | superseded by the raw README read in full |
| https://v1-docs.xtdb.com (TSQL2 chapter, JUXT, Medium bitemporal x2, softwarepatternslexicon) | mixed | redundant with the XTDB primary |
| https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879 | blog | Medium wall |
| https://www.researchgate.net/publication/399953316_Look-Ahead-Bench... | preprint index | RG wall |

**URLs collected: 24+ unique.** Query variants run: (a) current-year —
"point-in-time macro data publication lag proxy backfill vintage unknown **2026**
backtest"; "arXiv **2025 2026** look-ahead bias point-in-time…"; (b) year-less
canonical — "bitemporal modeling point-in-time database design time series valid
time transaction time"; "real-time data set for macroeconomists Croushore Stark
vintage publication lag"; "ALFRED realtime_start realtime_end vintage
point-in-time FRED data backtest"; (c) topic — "look-ahead bias macroeconomic
data revisions real-time data backtest strategy performance".

### Recency scan (2024–2026)

**Performed. Result: 4 new findings, one of which changes the recommendation.**
(1) `arXiv:2607.04958v1` (2026) formalizes look-ahead-freedom as temporal
non-interference with a **fail-closed** checker and an explicitly *conservative
upper bound* on availability — this validates the conservative-proxy direction
but also states the framework assumes availability is **known and complete**,
which ours is not; that is the honest limit of our fix and I record it as the
adversarial finding. (2) A 2025–26 cluster (`2605.23959`, `2605.24564`,
`2607.11889`, Look-Ahead-Bench) has moved the field toward *benchmarking*
leakage rather than merely warning about it — relevant later if 82.15 wants a
regression test rather than a code review. (3) Vendors (Bloomberg PiT Economic
Releases, CEIC PiT, both 2026) now sell exactly the vintage warehouse we are
approximating — confirming that a hand-rolled proxy is the normal position for a
non-institutional shop, not a shortcut. (4) Macrosynergy (snippet, JS-walled)
states the practitioner method verbatim: *"a standard publication lag for each
country is estimated by comparing the last release date with the last
observation date in the revised series, and this same lag is then applied to all
past observations"* — and warns of exactly our risk: *"Hindsight errors occur
when macro-quantamental factors are adopted on the basis of significant
empirical relationships found in revised proxy data, even though no such
significance exists for true point-in-time information."* Older canonical work
(Croushore & Stark 2001) is not superseded; it remains the framing citation.

### Consensus vs debate

**Consensus:** point-in-time data is the only real remedy (CFA curriculum, pfolio,
StarQube, Philly Fed); a *conservative publication lag applied uniformly to past
observations* is the accepted approximation when true vintages are unavailable
(pfolio's "conservative reporting lags", Macrosynergy's "standard publication
lag… applied to all past observations"); conservatism must run in the direction
that withholds information (arXiv:2607.04958's "conservative upper bound",
fail-closed).

**Debate / gap:** nobody in the read-in-full set endorses using the *ingest write
date* as the vintage for backdated rows. The bitemporal literature is explicit
that transaction time and knowledge time are different axes (XTDB), and
arXiv:2607.04958 declines to model unknown availability at all. So the 82.0
backfill is defensible as an *upper bound* but is not a usable PIT key — which
is precisely what the measurement in §1 shows empirically.

---

## 5. RECOMMENDED TREATMENT (and what it costs)

### Rejected: the naive strict filter

`WHERE date <= cutoff AND realtime_start <= cutoff` — **measured cost: 0 macro
rows at every cutoff before 2026-03-22**, i.e. all six macro features silently
vanish from 2018–2025 (and vanish *by omission*, not as `None`, per
`historical_data.py:269`). This is not a conservative fix; it is a silent
amputation of the feature block across the whole usable sample. Do not ship it.

### Rejected: restrict the backtest window to ≥ 2026-03-22

Honest and fail-loud (precedent: `screener.py:42` /
`candidate_selector.py:121` raise `NotImplementedError` rather than fake PIT),
but leaves ~4.5 months of sample — statistically worthless for 82.3's
incumbent-vs-candidate comparison. Keep it only as the documented fallback if
the operator rejects proxies on principle.

### RECOMMENDED: effective-vintage = `MIN(realtime_start, date + publication_lag[series])`

One uniform rule, no special-casing of the two backfill dates, no NULL branch
(there are no NULLs). Rationale:

- For the 2018–2025 backfilled cohort, `realtime_start` (2026-03-22/25) is far
  later than `date + lag`, so the proxy wins → the sample survives.
- For live-ingested rows, `realtime_start` is a genuine first-observation stamp
  and is normally earlier than a deliberately-long `date + lag`, so truth wins.
- `MIN` means the estimate can only *admit* data the upper-bound would have
  withheld — the error direction is stated, bounded by the lag table, and
  auditable; it is not hidden.

Lag table: add `MACRO_PUBLICATION_LAG_DAYS: dict[str, int]` beside the existing
`MACRO_SERIES_MAX_AGE_DAYS` in `cache.py:23-52` (same idiom, same file, one
cited comment per series). Suggested conservative values, each traceable to
FRED's dating convention and cross-checked against the §3 frontier measurements:
`DGS10`/`T10Y2Y` = 2; `CPIAUCSL`/`UNRATE`/`FEDFUNDS`/`UMCSENT` = 50 (month-START
dating + mid-following-month release ≈ 45d, rounded long); `GDP` = 125
(quarter-START dating + advance estimate ≈ 120d; our own frontier row measures
124d). A series absent from the table must fall back to a **long** default (say
95d) — never 0 — so an added series fails toward pessimism.

Three code sites, all in `backend/backtest/cache.py`:

1. `:255` — add `realtime_start` to the `preload_macro` SELECT.
2. `:351` — carry it into the cached entry dict (and fix the `:68` comment).
3. `:481-485` — `and` the vintage predicate into the fast-path loop; keep the
   `break`, but only after BOTH predicates pass (the predicate is no longer
   monotone in `date`).
4. `:494-515` — add `AND realtime_start <= DATE(@cutoff)`-equivalent to the
   fallback's inner `WHERE`; **use `DATE(@cutoff)`**, since `@cutoff` is bound
   STRING at `:509` while `realtime_start` is DATE.

Both paths must apply the *same* effective-vintage expression, or the fast path
and the fallback disagree — a divergence nobody would notice because the
fallback only fires when the preload is skipped.

Plus one guard: make an empty macro result **loud**. `preload_macro` already
fails closed at `:317-325`; mirror it. Today, zero macro rows at a cutoff is
indistinguishable from "macro block ran fine" downstream of
`historical_data.py:269`.

### What this costs — stated plainly

1. **It fixes publication lag, not revisions.** With 0 duplicate
   `(series_id, date)` pairs and a dedupe that guarantees there never will be
   any, we serve a *revised* value stamped with an *estimated first-release*
   date. Croushore & Stark and the CFA curriculum both treat revision bias as
   real and separate. 82.15 must not be reported as "look-ahead eliminated".
2. **The lag table is an estimate.** Per-series error of days (daily series) to
   weeks (monthly/quarterly). Erring long costs realism (a strategy is denied
   data it genuinely had); erring short reinstates look-ahead. Choose long.
3. **Results will get worse, and that is the point.** Expect macro-conditioned
   Sharpe to fall — the read-in-full sources bracket the effect at 100–500bp of
   annualised return (Bailey & López de Prado via pfolio) and 15–25% of Sharpe
   for GDP-dependent strategies (StarQube). "The corrected return is always
   lower than the contaminated return."
4. **Comparability break with 82.3.** Step 82.3 (pending) backtests three
   candidates against the `triple_barrier` incumbent. Flipping macro semantics
   mid-comparison makes those runs non-comparable. Recommend shipping behind a
   default-OFF settings flag (project idiom; e.g. `macro_point_in_time`), running
   ON/OFF once to quantify the delta, then flipping ON before 82.3 — or running
   all of 82.3 in one flag state and recording which.
5. **The real fix is a one-off ALFRED backfill**, and it is reachable:
   `backend/econ_calendar/sources/fred_releases.py:55-56` already sends
   `realtime_start`/`realtime_end` to FRED. Pulling true vintages (and true
   revision rows) for 7 series × 2018–2026 would replace both the proxy and the
   revision gap. That is its own masterplan step, not 82.15 — queue it.

---

## 6. Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL (7)
- [x] 10+ unique URLs total (24+)
- [x] Recency scan (2024–2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft:
- [x] Internal exploration covered every macro read path + all consumers
- [x] Contradictions noted (arXiv:2607.04958 declines to model unknown
      availability; no source endorses ingest-date-as-vintage)
- [x] Claims cited per-claim
- [!] Two official FRED/ALFRED doc pages are bot-walled (403 on WebFetch AND
      curl) — ALFRED semantics were obtained from the `fredapi` README, which
      quotes the field definitions and a worked GDP example verbatim. Flagging
      rather than hiding.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 17,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "summary": "The naive strict PIT filter is measured-fatal: realtime_start MIN is 2026-03-22 for all 7 series, so `realtime_start <= cutoff` returns 0 rows at every cutoff before then, blanking all six macro features across 2018-2025 -- and blanking them by omission (historical_data.py:269 `if macro:`), not as None. Recommend effective_vintage = MIN(realtime_start, date + MACRO_PUBLICATION_LAG_DAYS[series]) applied identically at all four cache.py sites (:255 SELECT missing realtime_start, :351 cached dict, :481-485 fast-path loop where the DESC `break` becomes non-monotone, :494-515 fallback WHERE needing DATE(@cutoff) because date is STRING and realtime_start is DATE). Costs, stated: fixes publication lag only -- 0 duplicate (series,date) pairs and a dedupe that guarantees none, so revisions are structurally uncapturable; lag table is an estimate; Sharpe will fall (100-500bp / 15-25% per sources); breaks comparability with pending step 82.3 unless flag-gated. Real fix is a one-off ALFRED backfill via the realtime_start/end params already used at fred_releases.py:55-56 -- queue as its own step.",
  "brief_path": "handoff/current/research_brief_82.15.md",
  "gate_passed": true
}
```
