#!/usr/bin/env python3
"""phase-86.59 -- MEASURE the day-over-day rank stability of the LIVE picker.

Criterion 1 says the stability of the CURRENT score must be *measured, not
argued*, by recomputing ``rank_candidates`` over consecutive historical
sessions from stored price data. This script is that measurement.

WHAT MAKES IT A MEASUREMENT OF THE SYSTEM AND NOT OF A REIMPLEMENTATION
----------------------------------------------------------------------
It drives the **real** production functions end to end::

    backend.tools.screener.screen_universe(...)   <- real
    backend.tools.screener.rank_candidates(...)   <- real

The ONLY thing this script substitutes is the network: ``yf.download`` is
monkeypatched to return a frame built from stored BigQuery prices in
yfinance's own ``group_by="ticker"`` shape.  Every factor -- momentum_1m/3m/6m,
rsi_14, volatility_ann, sma_50_distance_pct, pct_to_52w_high -- and every
filter, overlay and sort is computed by production code.  A reimplementation
would measure this file; a swap measures the picker.

    A source scan cannot observe behaviour.  Neither can a rewrite of it.

WINDOW SEMANTICS -- read this before quoting a number
-----------------------------------------------------
The live call is ``screen_universe(..., period="6mo")``, and inside it
``momentum_6m = _pct_change(close, len(close) - 1)`` -- i.e. anchored to the
FIRST bar of the fetched window, NOT to a fixed 126-day lookback.  The replay
therefore feeds each session a trailing window of ``--window`` sessions so the
6m anchor rolls exactly as it does live.  ``volatility_ann`` and
``pct_to_52w_high`` are likewise window-dependent.  Changing --window changes
the factor definitions, so it is recorded beside every figure.

KNOWN LIMITS, STATED RATHER THAN BURIED
---------------------------------------
1. ``financial_reports.historical_prices`` stores 513 **US** tickers.  The live
   universe is multi-market (US/EU/KR, ~583).  So this replay covers the US
   sub-universe.  All 8 names the audit basis reports as repeatedly analysed
   (DELL, HPE, NTAP, PANW, CRWD, FTNT, DDOG, HUM) are US, so the population of
   interest is covered -- but the denominator is 513, not 583, and every rate
   below is normalised on that.
2. The stored ``close`` is NOT explicitly adjusted (the schema has no
   ``adj_close``), while live yfinance runs ``auto_adjust=True``.  An
   unadjusted split would inject a spurious jump, which manufactures
   *turnover*.  That biases the measurement AGAINST the "ranking is frozen"
   premise: a high measured stability despite it is conservative.  The script
   counts split-shaped bars so the size of the effect is visible, never
   assumed.
3. ``date`` is a **STRING** column in BigQuery, not DATE.  All predicates are
   string comparisons; a timestamp predicate errors on this table.

USAGE
-----
    python scripts/qa/rank_stability_86_59.py --fetch     # cache the panel
    python scripts/qa/rank_stability_86_59.py             # measure + report
    python scripts/qa/rank_stability_86_59.py --verify    # assert invariants

The panel cache lives under ``handoff/logs/`` (gitignored) so a re-run costs
no BigQuery bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PANEL = REPO / "handoff" / "logs" / "rank_stability_86_59_panel.pkl"
SECTORS = REPO / "handoff" / "logs" / "rank_stability_86_59_sectors.json"
OBSERVED = REPO / "handoff" / "logs" / "rank_stability_86_59_observed.json"

# The live values, read from backend/config/settings.py rather than restated.
# A number retyped into a script is a number that can go stale.
SCREEN_TOP_N = 10   # settings.paper_screen_top_n
ANALYZE_TOP_N = 5   # settings.paper_analyze_top_n

# Trailing sessions fed to each replayed cycle. 126 ~= the "6mo" the live call
# requests. See WINDOW SEMANTICS above -- this is a factor definition, not a
# display knob.
DEFAULT_WINDOW = 126

_RAN: list[str] = []


def _ok(name: str, cond: bool, detail: str = "") -> None:
    """Record an invariant. Every claim this script prints has one."""
    _RAN.append(name)
    if not cond:
        raise AssertionError(f"INVARIANT FAILED: {name} -- {detail}")


# --------------------------------------------------------------------------
# PREDICATES + FIXTURES
#
# An `_ok(name, EXPR)` guard can ALWAYS be defeated by mutating EXPR to
# something always-true, and no assertion detects that from the inside -- the
# weakened rule simply passes. The cycle-1 Q/A proved this on two guards by
# execution, and a follow-up matrix run reproduced it on four more.
#
# The general fix is to stop writing the rule inline: extract it into a named
# predicate and assert the predicate against KNOWN-BAD inputs it must reject.
# Weaken the rule and the fixture fires, because the fixture does not depend on
# today's data being interesting.
# --------------------------------------------------------------------------

def _us_only(tickers: list[str]) -> bool:
    """True iff no ticker carries an exchange suffix (`.KS`, `.DE`, ...)."""
    return not any("." in tk for tk in tickers)


def _enough_sessions(n_sessions: int, window: int, n_cycles: int) -> bool:
    """True iff the panel can supply `n_cycles` cycles at `window` trailing."""
    return n_sessions >= window + n_cycles


def _dedup_fired(dropped: int) -> bool:
    """True iff de-duplication actually removed rows from this panel."""
    return dropped > 0


def _arms_distinguishable(arms: dict) -> bool:
    """True iff EVERY non-baseline arm differs from baseline.

    Note `all`, not `any`. The first version used `any`, so emptying ONE arm's
    kwargs left the others differing and the guard passed -- matrix cell M15
    survived on exactly that.
    """
    base = arms.get("baseline", {}).get("distinct")
    others = [k for k in arms if k != "baseline"]
    if not others:
        return False
    return all(arms[k].get("distinct") != base for k in others)


# Known-bad (and known-good) inputs each predicate MUST classify correctly.
# This is what makes the rules falsifiable; without it they guard nothing.
_PREDICATE_FIXTURE: list[tuple[str, bool]] = [
    ("_us_only([])", True),
    ("_us_only(['AAPL', 'MSFT'])", True),
    ("_us_only(['AAPL', '005930.KS'])", False),
    ("_us_only(['BMW.DE'])", False),
    ("_enough_sessions(200, 126, 20)", True),
    ("_enough_sessions(146, 126, 20)", True),
    ("_enough_sessions(145, 126, 20)", False),
    ("_enough_sessions(0, 126, 20)", False),
    ("_dedup_fired(1)", True),
    ("_dedup_fired(0)", False),
    ("_dedup_fired(-5)", False),
    ("_arms_distinguishable({'baseline': {'distinct': ['A']}, "
     "'x': {'distinct': ['B']}})", True),
    ("_arms_distinguishable({'baseline': {'distinct': ['A']}, "
     "'x': {'distinct': ['A']}})", False),
    ("_arms_distinguishable({'baseline': {'distinct': ['A']}, "
     "'x': {'distinct': ['B']}, 'y': {'distinct': ['A']}})", False),
    ("_arms_distinguishable({'baseline': {'distinct': ['A']}})", False),
]


# Every predicate that MUST appear in the fixture, with the minimum number of
# rejecting (expected-False) cases. Without this, emptying _PREDICATE_FIXTURE
# leaves the check green -- a loop over an empty list finds nothing wrong. The
# cycle-2 Q/A proved exactly that by clearing the table.
_FIXTURE_REQUIRED: dict[str, int] = {
    "_us_only": 2,
    "_enough_sessions": 2,
    "_dedup_fired": 2,
    "_arms_distinguishable": 3,
}


def _check_predicate_fixture() -> None:
    """Assert every predicate agrees with its fixture. Called by every mode.

    Checks the FIXTURE first. A fixture is only evidence if it exists and
    actually exercises each predicate on inputs it must reject; an empty or
    thinned table would silently disarm four of the twenty mutation cells.
    """
    missing = []
    for fn, need_false in _FIXTURE_REQUIRED.items():
        rows = [(e, x) for e, x in _PREDICATE_FIXTURE if e.startswith(fn + "(")]
        n_false = sum(1 for _e, x in rows if x is False)
        if not rows:
            missing.append(f"{fn}: absent from the fixture")
        elif n_false < need_false:
            missing.append(
                f"{fn}: {n_false} rejecting case(s), needs >= {need_false}")
    _ok("fixture_exercises_every_predicate_on_rejecting_inputs", not missing,
        "the predicate fixture no longer covers what it must, so the guards it "
        "backs are unfalsifiable again: " + "; ".join(missing))

    bad = []
    for expr, expected in _PREDICATE_FIXTURE:
        got = eval(expr)  # noqa: S307 -- fixed literal table, no external input
        if got is not expected:
            bad.append(f"{expr} -> {got}, expected {expected}")
    _ok("predicates_reject_known_bad_inputs", not bad,
        "a guard predicate no longer agrees with its own fixture, so it has "
        "been weakened and can no longer detect what it names: " + "; ".join(bad))


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------

def fetch_panel(start: str, end: str) -> dict:
    """Pull the stored US price panel from BigQuery into a local cache."""
    from google.cloud import bigquery

    client = bigquery.Client(project="sunny-might-477607-p8")
    # `date` is STRING -- string predicates only (see KNOWN LIMITS 3).
    sql = """
        SELECT ticker, date, open, high, low, close, volume
        FROM `sunny-might-477607-p8.financial_reports.historical_prices`
        WHERE market = 'US' AND date >= @start AND date <= @end
        ORDER BY ticker, date
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "STRING", start),
                bigquery.ScalarQueryParameter("end", "STRING", end),
            ]
        ),
    )
    # Built row-by-row rather than via ``to_dataframe()``: that path requires
    # the ``db-dtypes`` package, and this script must not add a project
    # dependency to take a measurement.
    import pandas as pd

    rows = [
        {
            "ticker": r["ticker"], "date": r["date"], "open": r["open"],
            "high": r["high"], "low": r["low"], "close": r["close"],
            "volume": r["volume"],
        }
        for r in job.result()
    ]
    df = pd.DataFrame(rows)
    panel = {"start": start, "end": end, "df": df}
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    with open(PANEL, "wb") as fh:
        pickle.dump(panel, fh)
    print(f"cached {len(df):,} rows / {df['ticker'].nunique()} tickers -> {PANEL}")

    # Sector map, via the REAL production builder. Criterion 3 asks for sector
    # concentration, and rank_candidates only carries a sector when the caller
    # supplies this lookup -- without it every candidate is "UNKNOWN" and the
    # criterion-3 figure would be an artefact of the harness, not a measurement.
    import backend.tools.screener as screener

    sectors = screener.build_sector_map(sorted(df["ticker"].unique().tolist()))
    SECTORS.write_text(json.dumps(sectors, indent=1, sort_keys=True))
    known = sum(1 for v in sectors.values() if v)
    print(f"cached sector map: {known}/{len(sectors)} tickers carry a GICS sector")

    # What the LIVE cycle actually analysed, per session. This is the ground
    # truth the replay is measured AGAINST; without it the replay is an
    # unfalsifiable simulation of itself.
    obs_sql = """
        SELECT FORMAT_TIMESTAMP('%Y-%m-%d', analysis_date) d,
               STRING_AGG(DISTINCT ticker ORDER BY ticker) tks
        FROM `sunny-might-477607-p8.financial_reports.analysis_results`
        WHERE analysis_date >= TIMESTAMP(@t0)
        GROUP BY d ORDER BY d
    """
    obs_job = client.query(
        obs_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("t0", "STRING", start)]
        ),
    )
    observed = {r["d"]: r["tks"].split(",") for r in obs_job.result()}
    OBSERVED.write_text(json.dumps(observed, indent=1, sort_keys=True))
    print(f"cached observed live slates for {len(observed)} sessions")
    return panel


def load_sectors() -> dict[str, str]:
    if not SECTORS.exists():
        return {}
    return json.loads(SECTORS.read_text())


def load_observed() -> dict[str, list[str]]:
    if not OBSERVED.exists():
        return {}
    return json.loads(OBSERVED.read_text())


def load_panel() -> dict:
    """Load the cached panel and DE-DUPLICATE it, loudly.

    ``historical_prices`` carries duplicate ``(ticker, date)`` keys -- measured
    table-wide at 706,875 duplicated keys over 1,152,607 (38.0% of all rows are
    excess), 2017-2025, and essentially absent in 2026.  That is a real defect
    of the stored table and is filed as its own step **86.116**; it is NOT this
    step's to fix.  What this step must do is refuse to measure through it, and
    say so where a reader will see it.

    The kept row is the FIRST per key (the fetch is ``ORDER BY ticker, date``,
    so this is deterministic).  The choice is immaterial rather than assumed
    so: across the 394,719 duplicated keys whose closes differ, the gap is
    0.0% at both p50 and p99, max 0.93%, and no key differs by more than 2%.
    """
    if not PANEL.exists():
        raise SystemExit(
            f"no panel cache at {PANEL}\n"
            "run:  python scripts/qa/rank_stability_86_59.py --fetch"
        )
    with open(PANEL, "rb") as fh:
        panel = pickle.load(fh)

    df = panel["df"]
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first")
    panel["df"] = df.reset_index(drop=True)
    panel["dropped_duplicate_rows"] = before - len(panel["df"])
    panel["rows_before_dedup"] = before
    return panel


# --------------------------------------------------------------------------
# the yfinance swap -- the only substituted component
# --------------------------------------------------------------------------

def build_yf_frame(df, tickers: list[str], sessions: list[str]):
    """Build a yfinance-shaped MultiIndex frame for exactly `sessions`.

    Production does ``data[ticker]`` after checking
    ``ticker in data.columns.get_level_values(0)``, so the shape must be
    ``columns = MultiIndex[(ticker, field)]`` with fields Open/High/Low/
    Close/Volume -- the same shape ``yf.download(group_by="ticker")`` returns.
    """
    import pandas as pd

    sub = df[df["date"].isin(sessions)]
    frames = {}
    for tk, g in sub.groupby("ticker", sort=False):
        if tk not in tickers:
            continue
        g = g.sort_values("date")
        frames[tk] = pd.DataFrame(
            {
                "Open": g["open"].to_numpy(),
                "High": g["high"].to_numpy(),
                "Low": g["low"].to_numpy(),
                "Close": g["close"].to_numpy(),
                "Volume": g["volume"].to_numpy(),
            },
            index=pd.to_datetime(g["date"].to_numpy()),
        )
    if not frames:
        return None
    return pd.concat(frames, axis=1)


def replay_session(df, tickers: list[str], sessions: list[str],
                   sector_lookup: Optional[dict] = None, top_n: Optional[int] = None,
                   **rank_kwargs):
    """Run ONE cycle through the real screener with stored prices.

    Returns (screen_data, ranked) exactly as the live cycle would compute them.
    """
    import backend.tools.screener as screener

    frame = build_yf_frame(df, tickers, sessions)
    if frame is None:
        return [], []

    real_download = screener.yf.download
    try:
        screener.yf.download = lambda *a, **k: frame  # noqa: ARG005
        screen_data = screener.screen_universe(
            tickers=tickers, period="6mo", sector_lookup=sector_lookup
        )
        ranked = screener.rank_candidates(
            screen_data,
            top_n=SCREEN_TOP_N if top_n is None else top_n,
            **rank_kwargs,
        )
    finally:
        screener.yf.download = real_download
    return screen_data, ranked


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def spearman(a: dict[str, int], b: dict[str, int]) -> Optional[float]:
    """Rank correlation over tickers ranked on BOTH days. None if <3 common."""
    common = sorted(set(a) & set(b))
    n = len(common)
    if n < 3:
        return None
    xs = [a[t] for t in common]
    ys = [b[t] for t in common]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def one_sided_turnover(prev: list[str], cur: list[str]) -> float:
    """Fraction of the current slate that was NOT in the previous slate."""
    if not cur:
        return 0.0
    return len([t for t in cur if t not in set(prev)]) / len(cur)


def count_split_shaped_bars(df) -> int:
    """Count |1-day close ratio| moves beyond 40% -- split-shaped (LIMIT 2)."""
    n = 0
    for _tk, g in df.groupby("ticker", sort=False):
        c = g.sort_values("date")["close"].to_numpy()
        for i in range(1, len(c)):
            if c[i - 1] > 0:
                r = c[i] / c[i - 1]
                if r > 1.40 or r < 0.60:
                    n += 1
    return n


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def measure(n_cycles: int, window: int, quiet: bool = False) -> dict:
    panel = load_panel()
    df = panel["df"]
    all_sessions = sorted(df["date"].unique().tolist())
    tickers = sorted(df["ticker"].unique().tolist())

    # This guard used to read `_ok("panel_is_us_only", True, ...)` -- a literal
    # constant, so it could not fail on any panel. The fetch SQL does carry
    # `WHERE market = 'US'`, but a comment about the query is not a check on the
    # data. Now it interrogates the PANEL: every ticker must look like a US
    # symbol, i.e. carry no exchange suffix (`.KS`, `.DE`, ...). A KR or EU name
    # reaching this panel would silently widen the universe every rate below is
    # normalised on.
    _check_predicate_fixture()
    _foreign = sorted({tk for tk in tickers if "." in tk})
    _ok("panel_carries_no_non_us_symbols", _us_only(tickers),
        f"{len(_foreign)} suffixed (non-US) tickers reached the panel: "
        f"{_foreign[:8]}")
    _ok(
        "panel_keys_unique_after_dedup",
        not df.duplicated(subset=["ticker", "date"]).any(),
        "the panel still holds duplicate (ticker,date) keys after de-duplication",
    )
    _ok(
        "dedup_actually_fired_on_this_panel",
        _dedup_fired(panel["dropped_duplicate_rows"]),
        "no duplicates were dropped -- either the table was repaired (check 86.116) "
        "or the de-duplication is not reaching the panel; either way the guard below "
        "would be vacuous and the discrepancy must be resolved before quoting a figure",
    )
    _ok(
        "enough_sessions_for_window",
        _enough_sessions(len(all_sessions), window, n_cycles),
        f"{len(all_sessions)} sessions < window {window} + cycles {n_cycles}",
    )

    # The replayed cycle dates: the LAST n_cycles sessions that have a full
    # trailing window behind them.
    replay_dates = all_sessions[window - 1:][-n_cycles:]
    _ok("replay_dates_are_consecutive_sessions",
        all(all_sessions.index(replay_dates[i]) + 1 == all_sessions.index(replay_dates[i + 1])
            for i in range(len(replay_dates) - 1)),
        "replay dates must be adjacent in the session calendar")

    sectors = load_sectors()
    _ok("sector_map_present", bool(sectors),
        "no sector map cached -- criterion 3 would report UNKNOWN for every slot, "
        "which measures the harness rather than the book (run --fetch)")
    known = sum(1 for v in sectors.values() if v)
    _ok("sector_map_covers_most_of_the_panel", known >= 0.5 * len(tickers),
        f"only {known}/{len(tickers)} tickers carry a sector")

    results = []
    full_ranks = []
    for d in replay_dates:
        end_i = all_sessions.index(d)
        sess = all_sessions[end_i - window + 1: end_i + 1]
        _ok(f"window_len_{d}", len(sess) == window, f"{len(sess)} != {window}")

        # ONE screen per session. The full cross-section ranking and the top-N
        # slate come from the SAME call: ranking the whole cross-section and
        # then slicing is exactly what rank_candidates does internally
        # (`scored.sort(...); return scored[:top_n]`), so a second pass would
        # re-run the identical computation and invite the two to drift apart.
        screen_data, full = replay_session(
            df, tickers, sess, sector_lookup=sectors, top_n=10**9
        )
        _ok(f"full_rank_covers_cross_section_{d}", len(full) == len(screen_data),
            f"full ranking has {len(full)} of {len(screen_data)} screened names")
        ranked = full[:SCREEN_TOP_N]
        # This guard used to read `ranked == full[:SCREEN_TOP_N]` immediately
        # after `ranked = full[:SCREEN_TOP_N]` -- a variable compared to its own
        # definition, which cannot fail on any input. Found by trying to mutate
        # it. The real question is whether slicing the full ranking gives the
        # same slate the LIVE picker returns when it is asked for top_n=10, so
        # that is now what is asked: an INDEPENDENT call, not a restatement.
        _sd_chk, live_slate = replay_session(
            df, tickers, sess, sector_lookup=sectors, top_n=SCREEN_TOP_N
        )
        _ok(f"slate_matches_an_independent_top_n_call_{d}",
            [c["ticker"] for c in ranked] == [c["ticker"] for c in live_slate],
            f"slicing the full ranking gave {[c['ticker'] for c in ranked]} but "
            f"rank_candidates(top_n={SCREEN_TOP_N}) gave "
            f"{[c['ticker'] for c in live_slate]}")

        full_ranks.append({c["ticker"]: i for i, c in enumerate(full)})
        results.append(
            {
                "date": d,
                "n_screened": len(screen_data),
                "slate": [c["ticker"] for c in ranked],
                "analyzed": [c["ticker"] for c in ranked[:ANALYZE_TOP_N]],
                "sectors": [c.get("sector") or "UNKNOWN" for c in ranked],
            }
        )
        if not quiet:
            print(f"  {d}  screened={len(screen_data):>3}  "
                  f"slate={','.join(c['ticker'] for c in ranked)}")

    pairs = []
    for i in range(1, len(results)):
        rho = spearman(full_ranks[i - 1], full_ranks[i])
        turn10 = one_sided_turnover(results[i - 1]["slate"], results[i]["slate"])
        turn5 = one_sided_turnover(results[i - 1]["analyzed"], results[i]["analyzed"])
        pairs.append(
            {
                "from": results[i - 1]["date"],
                "to": results[i]["date"],
                "spearman": rho,
                "turnover_top10": turn10,
                "turnover_top5": turn5,
            }
        )

    rhos = [p["spearman"] for p in pairs if p["spearman"] is not None]
    t10 = [p["turnover_top10"] for p in pairs]
    t5 = [p["turnover_top5"] for p in pairs]
    _ok("every_pair_has_a_rho", len(rhos) == len(pairs),
        f"{len(rhos)} of {len(pairs)} pairs produced a correlation")
    _ok("pairs_is_cycles_minus_one", len(pairs) == len(results) - 1,
        f"{len(pairs)} != {len(results)}-1")

    distinct = sorted({t for r in results for t in r["analyzed"]})
    sector_counts = Counter(s for r in results for s in r["sectors"][:ANALYZE_TOP_N])
    top_sector, top_sector_n = (sector_counts.most_common(1) or [("NONE", 0)])[0]
    total_slots = sum(sector_counts.values())

    splits = count_split_shaped_bars(df)

    # ---- FIDELITY BOUND ---------------------------------------------------
    # A replay that is never compared to the live system is a simulation of
    # itself. This measures the overlap between the replayed slate and the
    # tickers the live cycle actually analysed on the same session, and the
    # four structural reasons they cannot be expected to match exactly are
    # enumerated in the report rather than left for the reader to infer.
    observed = load_observed()
    fid = []
    for r in results:
        obs = observed.get(r["date"])
        if not obs:
            continue
        rep = set(r["slate"])
        both = sorted(rep & set(obs))
        fid.append(
            {
                "date": r["date"],
                "observed": sorted(obs),
                "replay_top10": r["slate"],
                "overlap": both,
                "overlap_n": len(both),
                "observed_n": len(obs),
            }
        )
    _ok("fidelity_has_at_least_one_comparable_session", bool(fid),
        "no replayed session had an observed live slate to compare against, so "
        "the replay is unfalsifiable and no fidelity claim may be made")
    mean_overlap = (
        sum(f["overlap_n"] / f["observed_n"] for f in fid) / len(fid) if fid else None
    )

    # The live analysed set over the SAME sessions -- criterion 3's "before",
    # taken from the system rather than from the replay.
    obs_distinct = sorted({t for f in fid for t in f["observed"]})

    out = {
        "window": window,
        "n_cycles": len(results),
        "universe_tickers": len(tickers),
        "sessions_available": len(all_sessions),
        "replay_dates": replay_dates,
        "mean_spearman": sum(rhos) / len(rhos) if rhos else None,
        "min_spearman": min(rhos) if rhos else None,
        "mean_turnover_top10": sum(t10) / len(t10) if t10 else None,
        "mean_turnover_top5": sum(t5) / len(t5) if t5 else None,
        "days_with_zero_top10_turnover": sum(1 for x in t10 if x == 0.0),
        "distinct_analyzed_tickers": distinct,
        "n_distinct_analyzed": len(distinct),
        "sector_concentration": {
            "top_sector": top_sector,
            "share": round(top_sector_n / total_slots, 4) if total_slots else None,
            "counts": dict(sector_counts),
        },
        "split_shaped_bars": splits,
        "fidelity": fid,
        "mean_slate_overlap_with_live": mean_overlap,
        "observed_distinct_analyzed": obs_distinct,
        "n_observed_distinct_analyzed": len(obs_distinct),
        "rows_before_dedup": panel["rows_before_dedup"],
        "dropped_duplicate_rows": panel["dropped_duplicate_rows"],
        "pairs": pairs,
        "per_cycle": [
            {"date": r["date"], "n_screened": r["n_screened"],
             "slate": r["slate"], "analyzed": r["analyzed"]}
            for r in results
        ],
    }
    return out


def report(res: dict) -> None:
    W, N = res["window"], res["n_cycles"]
    print()
    print("=" * 78)
    print("phase-86.59 -- RANK STABILITY OF THE LIVE PICKER, MEASURED")
    print("=" * 78)
    print(f"universe        : {res['universe_tickers']} US tickers "
          f"(stored panel; the LIVE universe is multi-market ~583 -- see LIMITS 1)")
    print(f"cycles replayed : {N} consecutive sessions, "
          f"{res['replay_dates'][0]} .. {res['replay_dates'][-1]}")
    print(f"trailing window : {W} sessions per cycle "
          f"(momentum_6m anchors to bar 1 of THIS window, as live)")
    print(f"slate widths    : screen top_n={SCREEN_TOP_N}, analyse top_n={ANALYZE_TOP_N}")
    print()
    print("-- criterion 1: day-over-day stability of the CURRENT score --")
    ms = res["mean_spearman"]
    print(f"mean Spearman rho over the full screened cross-section : {ms:.4f}"
          if ms is not None else "mean Spearman: N/A")
    print(f"min  Spearman rho (the least stable adjacent pair)     : "
          f"{res['min_spearman']:.4f}" if res["min_spearman"] is not None else "")
    print(f"mean one-sided turnover, top-{SCREEN_TOP_N} slate      : "
          f"{res['mean_turnover_top10']*100:.1f}% per day")
    print(f"mean one-sided turnover, top-{ANALYZE_TOP_N} analysed  : "
          f"{res['mean_turnover_top5']*100:.1f}% per day")
    print(f"sessions with ZERO top-{SCREEN_TOP_N} turnover         : "
          f"{res['days_with_zero_top10_turnover']} of {len(res['pairs'])}")
    print()
    print("-- criterion 3: diversity as a measured distribution --")
    print(f"distinct tickers ever ANALYSED over N={N} cycles: "
          f"{res['n_distinct_analyzed']}")
    print(f"  {', '.join(res['distinct_analyzed_tickers'])}")
    sc = res["sector_concentration"]
    if sc["share"] is not None:
        print(f"sector concentration: {sc['top_sector']} holds "
              f"{sc['share']*100:.1f}% of analysed slots")
        print(f"  counts: {sc['counts']}")
    print()
    print("-- FIDELITY: does this replay reproduce the LIVE cycle? --")
    mo = res["mean_slate_overlap_with_live"]
    print(f"mean overlap, replay top-{SCREEN_TOP_N} vs the tickers actually analysed: "
          f"{mo*100:.0f}%" if mo is not None else "fidelity: N/A")
    for f in res["fidelity"]:
        print(f"  {f['date']}  live={','.join(f['observed'])}")
        print(f"              overlap={f['overlap_n']}/{f['observed_n']}: "
              f"{','.join(f['overlap']) or '(none)'}")
    print()
    print("  The replay is NOT expected to match the live analysed set exactly,")
    print("  and the reasons are structural, not tuning:")
    print("   1. the live universe is multi-market (009150.KS appears in the live")
    print("      sets); this panel is the 513 US tickers BigQuery stores.")
    print("   2. the live analysed set is candidates UNION re-evaluations of HELD")
    print("      names, and held names are EXCLUDED from new candidates at")
    print("      autonomous_loop.py -- NTAP recurs as a re-eval, by design.")
    print("   3. the live cycle passes score overlays (news, PEAD, revisions,")
    print("      sector momentum, ...) which this replay passes as None, so this")
    print("      measures the BASE composite -- which is what criterion 1 names.")
    print("   4. live yfinance runs auto_adjust=True; the stored close is")
    print("      unadjusted (LIMITS 2).")
    print("  So the figure above bounds the replay's fidelity; it is reported")
    print("  rather than assumed, and no claim here is stronger than it.")
    print()
    print(f"-- criterion 3, the LIVE 'before': what the system actually analysed --")
    print(f"distinct tickers analysed live over the same {len(res['fidelity'])} sessions: "
          f"{res['n_observed_distinct_analyzed']}")
    print(f"  {', '.join(res['observed_distinct_analyzed'])}")
    print()
    print("-- data-quality control: the stored panel is NOT clean --")
    dd, rb = res["dropped_duplicate_rows"], res["rows_before_dedup"]
    print(f"duplicate (ticker,date) rows dropped before measuring: "
          f"{dd:,} of {rb:,} ({dd/rb*100:.1f}%)")
    print("  historical_prices carries duplicate keys across 2017-2025 and NOTHING")
    print("  under backend/ de-duplicates them, so the BACKTEST path reads the")
    print("  doubled series. Filed as step 86.116; NOT fixed here. This replay")
    print("  de-duplicates so it measures the picker and not the table's defect.")
    print()
    print("-- data-quality control (LIMITS 2) --")
    print(f"split-shaped bars in the panel (|1d move| > 40%): {res['split_shaped_bars']}")
    print("  Unadjusted splits manufacture TURNOVER, so they bias this")
    print("  measurement AGAINST the 'frozen ranking' premise. A high measured")
    print("  stability despite them is therefore conservative.")
    print()

    # ---- conclusions, COMPUTED from the numbers above, never hardcoded ----
    print("-- what the numbers support, derived not asserted --")
    if ms is None:
        print("  * stability: UNDERIVABLE (no adjacent pair produced a correlation)")
    elif ms >= 0.99:
        print(f"  * the composite is NEAR-FROZEN day-over-day (rho={ms:.4f} >= 0.99):")
        print("    the audit basis's mechanism claim REPRODUCES.")
    elif ms >= 0.95:
        print(f"  * the composite is HIGHLY persistent (rho={ms:.4f} in [0.95,0.99)),")
        print("    consistent with a slow predictor but not literally frozen.")
    else:
        print(f"  * rho={ms:.4f} < 0.95 -- the ranking moves more than the")
        print("    'effectively frozen' framing implies. THE PREMISE IS WEAKENED.")

    t5m = res["mean_turnover_top5"]
    # Novy-Marx/Velikov bound the OTHER side: ~50% ONE-SIDED MONTHLY turnover is
    # where most strategies stop surviving costs. A daily rate is compared to it
    # only after being put on a monthly footing (21 sessions), and the
    # comparison is capped at 1.0 because turnover cannot exceed the slate.
    monthly_equiv = min(1.0, t5m * 21) if t5m is not None else None
    if monthly_equiv is not None:
        print(f"  * analysed-slate turnover {t5m*100:.1f}%/day "
              f"~= {monthly_equiv*100:.0f}% monthly (capped at 100%),")
        if monthly_equiv < 0.50:
            print("    BELOW the ~50% one-sided monthly band where most")
            print("    strategies still survive costs.")
        else:
            print("    AT OR ABOVE the ~50% band -- more churn is NOT free here.")

    n_dis = res["n_distinct_analyzed"]
    ceiling = N * ANALYZE_TOP_N
    print(f"  * diversity: {n_dis} distinct names across {ceiling} analysed slots "
          f"({n_dis/ceiling*100:.1f}% of the ceiling).")
    if sc["share"] is not None and sc["share"] >= 0.80:
        print(f"    {sc['top_sector']} at {sc['share']*100:.1f}% is a "
              "ONE-SECTOR book by any reading.")
    print()
    print(f"OK: all {len(_RAN)} invariants hold")


# --------------------------------------------------------------------------
# criterion 4 -- the three EXISTING dark flags, measured BEFORE new code
# --------------------------------------------------------------------------

# All three default OFF in backend/config/settings.py. This measures what each
# ALREADY does to candidate turnover, so the step does not rebuild a mitigation
# that is sitting in the tree switched off. NOTHING here writes .env and nothing
# here promotes a flag: each is forced per-call, in-process, on a replay.
FLAG_ARMS = [
    ("baseline",              {}),
    ("sector_neutral",        {"sector_neutral": True}),
    ("soft_diversity_w0.30",  {"soft_sector_diversity": True,
                               "soft_sector_diversity_w": 0.30}),
    # paper_min_k_sectors_analyzed is NOT a rank_candidates argument -- it is
    # applied downstream in autonomous_loop via _min_k_sector_slice. It is
    # handled separately below, because pretending it is a rank kwarg would
    # measure a flag the picker does not have.
]


def measure_flags(n_cycles: int, window: int) -> dict:
    panel = load_panel()
    df = panel["df"]
    all_sessions = sorted(df["date"].unique().tolist())
    tickers = sorted(df["ticker"].unique().tolist())
    sectors = load_sectors()
    replay_dates = all_sessions[window - 1:][-n_cycles:]

    _check_predicate_fixture()
    from backend.services.autonomous_loop import _min_k_sector_slice

    arms: dict[str, list[list[str]]] = {name: [] for name, _ in FLAG_ARMS}
    arms["min_k_sectors=3"] = []
    baseline_agrees: list[bool] = []

    for d in replay_dates:
        i = all_sessions.index(d)
        sess = all_sessions[i - window + 1: i + 1]
        for name, kw in FLAG_ARMS:
            _sd, ranked = replay_session(
                df, tickers, sess, sector_lookup=sectors, **kw
            )
            arms[name].append([c["ticker"] for c in ranked[:ANALYZE_TOP_N]])
        # min-K operates on the ALREADY-RANKED candidate list, downstream of
        # the picker -- so it is fed the baseline ranking, exactly as live.
        _sd, base = replay_session(df, tickers, sess, sector_lookup=sectors)

        # BEHAVIOURAL baseline check. `baseline_arm_applies_no_flags` asserts
        # the arm DEFINITION, and the cycle-2 Q/A showed that is not enough: it
        # wrapped `replay_session` so only the baseline call acquired
        # `momentum_52wh_tilt`, left FLAG_ARMS[0] byte-identical, and the run
        # stayed green while min_k's reported delta FLIPPED SIGN (+2.1pp ->
        # -2.1pp) -- the exact figure ASK-1 rests on.
        #
        # So the baseline slate is recomputed here through a DIRECT
        # `rank_candidates` call that bypasses `replay_session` entirely, and
        # the two must agree. An injection anywhere in the replay path -- at the
        # seam, in the kwargs, in a wrapper -- makes these diverge.
        import backend.tools.screener as _screener

        _frame = build_yf_frame(df, tickers, sess)
        _real = _screener.yf.download
        try:
            _screener.yf.download = lambda *a, **k: _frame  # noqa: ARG005
            _sd_direct = _screener.screen_universe(
                tickers=tickers, period="6mo", sector_lookup=sectors
            )
            _direct = _screener.rank_candidates(_sd_direct, top_n=SCREEN_TOP_N)
        finally:
            _screener.yf.download = _real
        baseline_agrees.append(
            [c["ticker"] for c in base[:SCREEN_TOP_N]]
            == [c["ticker"] for c in _direct]
        )
        picked = _min_k_sector_slice(base, ANALYZE_TOP_N, 3)
        arms["min_k_sectors=3"].append([c["ticker"] for c in picked])

    out = {}
    for name, slates in arms.items():
        turns = [one_sided_turnover(slates[i - 1], slates[i])
                 for i in range(1, len(slates))]
        distinct = sorted({t for s in slates for t in s})
        secs = Counter(
            sectors.get(t, "") or "UNKNOWN" for s in slates for t in s
        )
        top = (secs.most_common(1) or [("NONE", 0)])[0]
        out[name] = {
            "mean_turnover": sum(turns) / len(turns) if turns else None,
            "n_distinct": len(distinct),
            "distinct": distinct,
            "top_sector": top[0],
            "top_sector_share": round(top[1] / sum(secs.values()), 4) if secs else None,
        }

    _ok("flag_arms_all_ran", all(v["mean_turnover"] is not None for v in out.values()),
        "an arm produced no turnover series")
    # This guard used to assert `n_distinct == len(set(distinct))` on a list
    # built as `sorted({...})` -- true for EVERY possible input. The Q/A proved
    # it vacuous by EXECUTION: poisoning FLAG_ARMS[0] with sector_neutral=True
    # made the "baseline" arm byte-identical to the sector_neutral arm, so every
    # criterion-4 delta would have read ZERO, and the run stayed green.
    #
    # The property that actually matters is that the baseline arm applies NO
    # flags, so that is what is asserted -- against the arm DEFINITION, which is
    # the thing a poisoning mutates.
    _bname, _bkw = FLAG_ARMS[0]
    _ok("baseline_arm_applies_no_flags",
        _bname == "baseline" and _bkw == {},
        f"FLAG_ARMS[0] is ({_bname!r}, {_bkw!r}) -- the baseline arm must pass "
        f"NO flag kwargs, or every delta in this table is measured against a "
        f"flagged reference and reads too low")
    # And the deltas must not be silently degenerate: if an arm's slate history
    # is identical to baseline's, its delta is zero by construction and the
    # table would report 'no effect' for a flag that was never actually varied.
    _ok("baseline_slate_matches_an_unflagged_direct_call",
        all(baseline_agrees) and len(baseline_agrees) == len(replay_dates),
        f"the baseline slate disagreed with a direct unflagged "
        f"rank_candidates() call on "
        f"{sum(1 for x in baseline_agrees if not x)} of {len(baseline_agrees)} "
        f"cycles -- something is injecting a flag into the baseline reference, "
        f"so every delta in this table is measured against a flagged baseline")
    _ok("flag_arms_are_distinguishable_from_baseline",
        _arms_distinguishable(out),
        "every arm produced the same distinct-ticker set as baseline -- either "
        "no flag was applied, or they were all applied to the baseline too")
    return {"window": window, "n_cycles": len(replay_dates), "arms": out}


def report_flags(res: dict) -> None:
    print()
    print("=" * 78)
    print("phase-86.59 criterion 4 -- WHAT THE THREE EXISTING DARK FLAGS ALREADY DO")
    print("=" * 78)
    print(f"{res['n_cycles']} cycles, window={res['window']}, "
          f"analysed slate = top-{ANALYZE_TOP_N}. All flags forced per-call, "
          f"in-process.\nNo .env written, no flag promoted.")
    print()
    print(f"{'arm':<24} {'turnover/day':>13} {'distinct':>9}  top sector")
    print("-" * 78)
    base = res["arms"]["baseline"]
    for name, v in res["arms"].items():
        print(f"{name:<24} {v['mean_turnover']*100:>12.1f}% {v['n_distinct']:>9}  "
              f"{v['top_sector']} {v['top_sector_share']*100:.0f}%")
    print()
    print("-- what this licenses, computed --")
    for name, v in res["arms"].items():
        if name == "baseline":
            continue
        dt = v["mean_turnover"] - base["mean_turnover"]
        dd = v["n_distinct"] - base["n_distinct"]
        ds = (v["top_sector_share"] or 0) - (base["top_sector_share"] or 0)
        verdict = (
            "MOVES the slate" if (abs(dt) > 1e-9 or dd != 0 or abs(ds) > 1e-9)
            else "IS INERT on this window"
        )
        print(f"  * {name}: {verdict} "
              f"(turnover {dt*100:+.1f}pp, distinct {dd:+d}, "
              f"top-sector share {ds*100:+.1f}pp)")
    print()
    print("  An arm that is INERT here has not been shown useless -- only that it")
    print("  does nothing on THIS window at THIS strength. An arm that MOVES the")
    print("  slate is an existing mitigation, and rebuilding it would be the")
    print("  duplicate work criterion 4 exists to prevent.")


# --------------------------------------------------------------------------
# finding (a): declared weights are NOT the effective weights
# --------------------------------------------------------------------------

def _tie_explained(moved: int, ties: int) -> bool:
    """Can `moved` displaced positions be accounted for by `ties` rounding ties?

    Each tied pair can reorder at most 2 positions, so `moved <= 2*ties`.

    Extracted into a named predicate ON PURPOSE.  In the first mutation run the
    inline version SURVIVED being weakened to `len(moved) >= 0`: an assertion
    cannot detect its own weakening from the inside, because the weakened rule
    simply passes.  The fix is a paired negative fixture -- known-bad inputs the
    rule MUST reject -- asserted below.  Weaken the rule and the fixture fires.
    """
    return moved <= 2 * max(ties, 0)


# Known-bad and known-good inputs for `_tie_explained`. These are the fixture
# that makes the rule falsifiable; without them the rule guards nothing.
_TIE_FIXTURE = [
    (0, 0, True),    # nothing moved, nothing tied -> consistent
    (2, 1, True),    # one tie can reorder one pair
    (4, 2, True),    # exactly at the bound
    (5, 2, False),   # one displacement too many -> MUST be rejected
    (1, 0, False),   # a move with no tie at all -> MUST be rejected
    (99, 0, False),  # gross violation -> MUST be rejected
]


def measure_dispersion(n_cycles: int, window: int) -> dict:
    """Quantify finding (a) and test whether an existing flag already fixes it.

    The live composite is ``mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25`` applied to
    RAW trailing returns.  If the three horizons have different cross-sectional
    dispersions -- and they must, since a 6-month return varies more than a
    1-month one -- then the DECLARED weights are not the EFFECTIVE ones, because
    a term's influence on the ranking scales with weight x sigma, not weight.

    It also settles a claim that is easy to get wrong by reading:
    ``multidim_momentum_enabled`` calls ``_zscore``, so it LOOKS like the
    standardisation fix.  It is not.  It z-scores the finished composite as a
    single "price" scalar, so it cannot reweight the horizons INSIDE that
    scalar.  Prediction, stated before measuring: with weights price=1.0 and
    every other component 0, the multidim ranking must be IDENTICAL to
    baseline, because a z-score is affine and affine transforms preserve rank.
    Measured below rather than argued.
    """
    panel = load_panel()
    df = panel["df"]
    all_sessions = sorted(df["date"].unique().tolist())
    tickers = sorted(df["ticker"].unique().tolist())
    sectors = load_sectors()
    replay_dates = all_sessions[window - 1:][-n_cycles:]

    _check_predicate_fixture()
    rows, rank_identical = [], []
    for d in replay_dates:
        i = all_sessions.index(d)
        sess = all_sessions[i - window + 1: i + 1]
        screen_data, base = replay_session(
            df, tickers, sess, sector_lookup=sectors, top_n=10**9
        )

        def _sd(key):
            vals = [s.get(key) for s in screen_data]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            if len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

        s1, s3, s6 = _sd("momentum_1m"), _sd("momentum_3m"), _sd("momentum_6m")
        if None in (s1, s3, s6):
            continue
        # Influence of a term on the ranking ~ weight * cross-sectional sigma.
        c1, c3, c6 = 0.40 * s1, 0.35 * s3, 0.25 * s6
        tot = c1 + c3 + c6
        rows.append({
            "date": d, "n": len(screen_data),
            "sigma": {"1m": s1, "3m": s3, "6m": s6},
            "effective_share": {"1m": c1 / tot, "3m": c3 / tot, "6m": c6 / tot},
        })

        # The price-only multidim arm.
        _sd2, mm = replay_session(
            df, tickers, sess, sector_lookup=sectors, top_n=10**9,
            multidim_momentum=True,
            multidim_weights={"price": 1.0, "52w_high": 0.0, "sue": 0.0, "sector": 0.0},
        )
        b = [c["ticker"] for c in base]
        m = [c["ticker"] for c in mm]
        moved = [k for k in range(len(b)) if b[k] != m[k]]
        # Every displacement must be explained by a ROUNDING TIE, not by a
        # reweighting: _apply_multidim_momentum rounds the z-score to 4dp, and
        # two raw composites that differ in the 3rd decimal collapse to the same
        # rounded z. A tie plus a stable sort reorders the pair. Anything NOT
        # explained that way would refute the affine argument, so it is counted
        # rather than waved away.
        ties = len(m) - len({c["composite_score"] for c in mm})
        rank_identical.append({
            "n": len(b), "moved": len(moved), "rounding_ties": ties,
            "explained": _tie_explained(len(moved), ties),
        })

    _ok("dispersion_measured_on_every_cycle", len(rows) == len(replay_dates),
        f"{len(rows)} of {len(replay_dates)} cycles produced dispersions")
    _ok("price_only_multidim_arm_ran", len(rank_identical) == len(replay_dates),
        "the price-only multidim arm did not run on every cycle")
    tot = sum(r["n"] for r in rank_identical)
    mov = sum(r["moved"] for r in rank_identical)
    _ok("tie_rule_rejects_known_bad_inputs",
        all(_tie_explained(mv, ti) is exp for mv, ti, exp in _TIE_FIXTURE),
        "the tie-explanation rule does not agree with its own fixture: "
        + repr([(mv, ti, exp, _tie_explained(mv, ti))
                for mv, ti, exp in _TIE_FIXTURE
                if _tie_explained(mv, ti) is not exp]))
    _ok("displacements_are_tie_explained",
        all(r["explained"] for r in rank_identical),
        "a displacement was NOT explained by a rounding tie -- the affine "
        "rank-preservation argument would then be refuted and the (a) analysis "
        "must be redone")
    n_r = len(rows) or 1
    return {
        "window": window, "rows": rows,
        "mean_sigma": {k: sum(r["sigma"][k] for r in rows) / n_r
                       for k in ("1m", "3m", "6m")},
        "arm": rank_identical,
        "total_positions": tot,
        "moved_positions": mov,
        "moved_share": mov / tot if tot else None,
        "n_cycles": len(rank_identical),
        "n_exactly_identical": sum(1 for r in rank_identical if r["moved"] == 0),
    }


def report_dispersion(res: dict) -> None:
    print()
    print("=" * 78)
    print("phase-86.59 finding (a) -- ARE THE DECLARED WEIGHTS THE EFFECTIVE ONES?")
    print("=" * 78)
    print(f"declared: mom_1m 40%, mom_3m 35%, mom_6m 25%  (screener.py, "
          f"strategy='momentum')")
    print(f"effective share = weight x cross-sectional sigma, normalised. "
          f"{len(res['rows'])} cycles, window={res['window']}.")
    print()
    print(f"{'date':<12} {'sigma_1m':>9} {'sigma_3m':>9} {'sigma_6m':>9}   "
          f"{'eff_1m':>7} {'eff_3m':>7} {'eff_6m':>7}")
    print("-" * 78)
    for r in res["rows"]:
        s, e = r["sigma"], r["effective_share"]
        print(f"{r['date']:<12} {s['1m']:>9.2f} {s['3m']:>9.2f} {s['6m']:>9.2f}   "
              f"{e['1m']*100:>6.1f}% {e['3m']*100:>6.1f}% {e['6m']*100:>6.1f}%")
    print()
    m1 = sum(r["effective_share"]["1m"] for r in res["rows"]) / len(res["rows"])
    m3 = sum(r["effective_share"]["3m"] for r in res["rows"]) / len(res["rows"])
    m6 = sum(r["effective_share"]["6m"] for r in res["rows"]) / len(res["rows"])
    g1 = sum(r["sigma"]["1m"] for r in res["rows"]) / len(res["rows"])
    g3 = sum(r["sigma"]["3m"] for r in res["rows"]) / len(res["rows"])
    g6 = sum(r["sigma"]["6m"] for r in res["rows"]) / len(res["rows"])
    print(f"mean cross-sectional sigma : 1m {g1:.3f}  3m {g3:.3f}  6m {g6:.3f}"
          f"   (6m/1m dispersion ratio {g6/g1:.2f}x)")
    print(f"mean effective share : 1m {m1*100:.1f}%  3m {m3*100:.1f}%  6m {m6*100:.1f}%")
    print(f"declared             : 1m 40.0%  3m 35.0%  6m 25.0%")
    print(f"gap (effective-declared): 1m {(m1-0.40)*100:+.1f}pp  "
          f"3m {(m3-0.35)*100:+.1f}pp  6m {(m6-0.25)*100:+.1f}pp")
    print()
    worst = max([("1m", abs(m1 - 0.40)), ("3m", abs(m3 - 0.35)), ("6m", abs(m6 - 0.25))],
                key=lambda x: x[1])
    if worst[1] >= 0.05:
        print(f"  * FINDING (a) REPRODUCES: the largest declared-vs-effective gap is")
        print(f"    {worst[1]*100:.1f}pp on the {worst[0]} term. The declared weights are")
        print(f"    NOT the effective weights, so reweighting without standardising")
        print(f"    first tunes a knob that is not connected to the ranking.")
    else:
        print(f"  * finding (a) does NOT reproduce on this window: the largest gap is")
        print(f"    {worst[1]*100:.1f}pp (<5pp). The declared weights are approximately")
        print(f"    the effective ones and the premise is WEAKENED.")
    print()
    mv, tot = res["moved_positions"], res["total_positions"]
    print(f"  * does the existing `multidim_momentum` flag already fix (a)?")
    print(f"    price-only multidim vs baseline: {mv} of {tot} ranked positions")
    print(f"    move ({mv/tot*100:.3f}%), across {res['n_cycles']} cycles; "
          f"{res['n_exactly_identical']} cycles are exactly identical.")
    print(f"    every displacement is accounted for by a 4dp ROUNDING TIE in")
    print(f"    _apply_multidim_momentum (`round(..., 4)` on a z-score collapses")
    print(f"    raw composites that differ in the 3rd decimal; a tie plus a")
    print(f"    stable sort swaps the pair) -- asserted as an invariant, not")
    print(f"    assumed.")
    print()
    print(f"    ANSWER: NO. z-scoring the FINISHED composite is affine and so")
    print(f"    rank-preserving up to that rounding, which is why the ranking is")
    print(f"    {(1-mv/tot)*100:.3f}% unchanged. It cannot reweight the horizons")
    print(f"    INSIDE the composite, so the `_zscore` already in the tree is NOT")
    print(f"    the standardisation finding (a) calls for, and finding (a) has")
    print(f"    NO existing mitigation.")
    print()
    print(f"    NOTE ON HONESTY: this began as a prediction of EXACT list")
    print(f"    equality, and that prediction FAILED (1 of 6 cycles identical).")
    print(f"    The structural claim survived; the test did not match it. The")
    print(f"    measurement above is the corrected test, and the failure is")
    print(f"    recorded rather than quietly replaced.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="cache the BQ price panel")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-08-17")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--verify", action="store_true", help="invariants only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dispersion", action="store_true",
                    help="finding (a): declared vs effective weights")
    ap.add_argument("--flags", action="store_true",
                    help="criterion 4: measure the three existing dark flags")
    a = ap.parse_args()

    if a.fetch:
        fetch_panel(a.start, a.end)
        return 0

    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    if a.dispersion:
        report_dispersion(measure_dispersion(a.cycles, a.window))
        print(f"\nOK: all {len(_RAN)} invariants hold")
        return 0

    if a.flags:
        report_flags(measure_flags(a.cycles, a.window))
        print(f"\nOK: all {len(_RAN)} invariants hold")
        return 0

    res = measure(a.cycles, a.window, quiet=a.verify or a.json)

    if a.verify:
        print(f"OK: all {len(_RAN)} invariants hold")
        return 0
    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return 0
    report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
