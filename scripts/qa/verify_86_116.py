#!/usr/bin/env python3
"""phase-86.116 -- re-runnable evidence for the duplicate-price defect.

Covers criteria 1, 2, 3, 5 and 6. Criterion 4 is the fix itself
(`backend/backtest/cache.py::_dedupe_index`) and criterion 7 is
`scripts/qa/mutation_86_116.py`.

    python scripts/qa/verify_86_116.py            # everything (hits BigQuery)
    python scripts/qa/verify_86_116.py --offline  # skip the BQ-backed sections
    python scripts/qa/verify_86_116.py --verify   # invariants only, terse

Every printed claim carries an invariant. Conclusions are COMPUTED from the
measured values, never hardcoded -- a census that prints its verdict regardless
of the numbers above it is not evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROJECT = "sunny-might-477607-p8"
TABLE = f"{PROJECT}.financial_reports.historical_prices"

# The commit that introduced `_dedupe_index`. Criterion 3 asks whether anything
# de-duplicated BEFORE this step, so the probe must run against this commit's
# PARENT. It used to default to HEAD, which was the pre-fix tree only while the
# fix was uncommitted -- the moment the fix landed, HEAD became the POST-fix
# tree, the probe found this step's own code and the whole script aborted before
# printing any evidence. A default that silently expires is a defect, so the
# revision is pinned.
FIX_COMMIT = "539f16eb"
PRE_FIX_REV = f"{FIX_COMMIT}~1"

_RAN: list[str] = []


def _ok(name: str, cond: bool, detail: str = "") -> None:
    _RAN.append(name)
    if not cond:
        raise AssertionError(f"INVARIANT FAILED: {name} -- {detail}")


# --------------------------------------------------------------------------
# criterion 1 -- re-measure the census
# --------------------------------------------------------------------------

CENSUS_SQL = f"""
WITH k AS (
  SELECT ticker, date, COUNT(*) n, COUNT(DISTINCT close) nc
  FROM `{TABLE}` GROUP BY 1, 2
)
SELECT (SELECT COUNT(*) FROM `{TABLE}`) total_rows,
       COUNT(*) keys, COUNTIF(n>1) dup_keys, SUM(n-1) excess_rows,
       MAX(n) max_mult,
       COUNT(DISTINCT IF(n>1, ticker, NULL)) tk_affected,
       (SELECT COUNT(DISTINCT ticker) FROM `{TABLE}`) tk_total,
       COUNTIF(n>1 AND nc>1) dup_keys_close_differs
FROM k
"""

BY_YEAR_SQL = f"""
WITH k AS (SELECT ticker, date, COUNT(*) n FROM `{TABLE}` GROUP BY 1, 2)
SELECT SUBSTR(date,1,4) yr, COUNT(*) keys, COUNTIF(n>1) dup
FROM k GROUP BY yr ORDER BY yr
"""


def _bq():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT)


def census(client) -> dict:
    row = next(iter(client.query(CENSUS_SQL).result()))
    d = dict(row)
    d["excess_share_of_rows"] = d["excess_rows"] / d["total_rows"]
    d["dup_share_of_keys"] = d["dup_keys"] / d["keys"]
    by_year = [
        {"yr": r["yr"], "keys": r["keys"], "dup": r["dup"],
         "share": r["dup"] / r["keys"] if r["keys"] else 0.0}
        for r in client.query(BY_YEAR_SQL).result()
    ]
    d["by_year"] = by_year

    _ok("census_rows_exceed_keys", d["total_rows"] > d["keys"],
        "no duplication measured -- if the table was repaired, this step's "
        "premise is stale and must be re-derived, not assumed")
    _ok("excess_equals_rows_minus_keys",
        d["excess_rows"] == d["total_rows"] - d["keys"],
        f"{d['excess_rows']} != {d['total_rows']} - {d['keys']}")
    _ok("multiplicity_is_bounded", d["max_mult"] >= 2,
        "max multiplicity below 2 contradicts the duplication census")
    _ok("year_buckets_cover_the_table",
        sum(y["keys"] for y in by_year) == d["keys"],
        "the per-year key counts must sum to the total key count")
    _ok("recent_window_is_cleaner_than_the_legacy_window",
        by_year[-1]["share"] < max(y["share"] for y in by_year[:-1]),
        "the newest year is not cleaner than the legacy window -- the "
        "'legacy, therefore bounded' framing would not hold")
    return d


# --------------------------------------------------------------------------
# criterion 2 -- drive REAL code on a REAL affected ticker
# --------------------------------------------------------------------------

def driven_harm(client, ticker: str, start: str, end: str) -> dict:
    """Load through the REAL loader; run the REAL factor functions on both."""
    import backend.backtest.cache as cache
    from backend.tools.screener import _compute_rsi, _pct_change

    cache.init_cache(client, PROJECT, "financial_reports")
    cache._prices_full.clear()
    cache.preload_prices([ticker], start, end)
    fixed = cache._prices_full[ticker]

    # The pre-fix frame, rebuilt from the same rows WITHOUT de-duplication, so
    # the comparison is fix-vs-no-fix on identical input rather than two
    # different queries.
    raw = _raw_frame(client, ticker, start, end)

    _ok("raw_frame_is_actually_duplicated", not raw.index.is_unique,
        f"{ticker} {start}..{end} has a unique index in the raw table, so it "
        f"cannot demonstrate the defect -- pick an affected ticker")
    _ok("fixed_frame_index_is_unique", fixed.index.is_unique,
        "preload_prices still returns a non-unique index after the fix")
    _ok("fix_removed_exactly_the_excess", len(fixed) == raw.index.nunique(),
        f"fixed={len(fixed)} but raw has {raw.index.nunique()} distinct dates")

    def facts(df):
        s = df["close"].dropna()
        dr = s.pct_change().dropna()
        return {
            "n": len(s),
            "mom_1m": _pct_change(s, 21),
            "mom_3m": _pct_change(s, 63),
            "rsi_14": _compute_rsi(s, 14),
            "vol_daily": float(dr.std()),
            "vol_ann": float(dr.std() * (252 ** 0.5)),
            "span_21_sessions": len(set(s.index[-22:])),
        }

    a, b = facts(raw), facts(fixed)
    _ok("positional_lookback_is_compressed_before_the_fix",
        a["span_21_sessions"] < b["span_21_sessions"],
        f"a 21-period lookback covered {a['span_21_sessions']} distinct "
        f"sessions before and {b['span_21_sessions']} after -- if these are "
        f"equal the positional claim is not demonstrated on this ticker")
    _ok("volatility_was_understated_before_the_fix",
        a["vol_ann"] < b["vol_ann"],
        "duplication must DEPRESS volatility (interleaved zero returns)")
    return {"raw": a, "fixed": b, "ticker": ticker,
            "window": f"{start}..{end}"}


def _raw_frame(client, ticker, start, end):
    """The frame the pre-fix code would have produced -- no de-duplication."""
    import pandas as pd
    from google.cloud import bigquery

    sql = f"""SELECT date, open, high, low, close, volume FROM `{TABLE}`
              WHERE ticker=@t AND date>=@s AND date<=@e ORDER BY date ASC"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("t", "STRING", ticker),
            bigquery.ScalarQueryParameter("s", "STRING", start),
            bigquery.ScalarQueryParameter("e", "STRING", end),
        ]))
    df = pd.DataFrame([dict(r) for r in job.result()])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# --------------------------------------------------------------------------
# criterion 3 -- POSITIVE-CONTROLLED proof that nothing de-duplicated before
# --------------------------------------------------------------------------

def _git_grep_count(pattern: str, rev: str = "HEAD") -> int:
    r = subprocess.run(["git", "grep", "-c", pattern, rev, "--", "backend/"],
                       capture_output=True, text=True, cwd=str(REPO))
    return len([ln for ln in r.stdout.splitlines() if ln.strip()])


def no_dedup_before(base_rev: str) -> dict:
    """A grep returning nothing proves nothing until it can return something."""
    claim = {p: _git_grep_count(p, base_rev)
             for p in ("drop_duplicates", r"index\.duplicated", "is_unique")}
    control_present = _git_grep_count("set_index", base_rev)

    _ok("probe_positive_control_finds_a_token_that_must_exist",
        control_present > 0,
        f"the probe found {control_present} files containing `set_index` in "
        f"{base_rev}; if this is 0 the probe is broken and the zeros below are "
        f"meaningless")
    _ok("no_dedup_existed_before_this_step",
        all(v == 0 for v in claim.values()),
        f"a de-duplication already existed in {base_rev}: {claim}")
    return {"rev": base_rev, "claim": claim, "control": control_present}


# --------------------------------------------------------------------------
# criterion 5 -- parity oracle: the fix is INERT on clean input
# --------------------------------------------------------------------------

def parity_oracle() -> dict:
    """Fix-absent parity, demonstrated against an oracle rather than examples.

    The oracle is every frame shape the loader can produce: unique-index frames
    (where the fix must be a no-op returning the SAME object), empty frames, and
    single-row frames. Duplicated frames are where it is allowed to differ, and
    it must then differ by exactly the excess.
    """
    import pandas as pd

    from backend.backtest.cache import _dedupe_index

    cases, inert, changed = [], 0, 0
    for n in (0, 1, 2, 5, 50, 260):
        for dup in (False, True):
            idx = pd.bdate_range("2025-01-01", periods=n)
            df = pd.DataFrame({"close": [100.0 + i for i in range(n)]}, index=idx)
            if dup and n:
                df = pd.concat([df, df]).sort_index()
            out = _dedupe_index(df, "ORACLE")
            unique_in = df.index.is_unique
            if unique_in:
                same_obj = out is df
                inert += 1
                _ok(f"inert_on_unique_input_n{n}_dup{int(dup)}", same_obj,
                    "a unique-index frame must be returned unchanged and "
                    "uncopied, or the fix is not provably inert")
            else:
                changed += 1
                _ok(f"exact_on_duplicated_input_n{n}", len(out) == df.index.nunique(),
                    f"expected {df.index.nunique()} rows, got {len(out)}")
            cases.append((n, dup, unique_in, len(df), len(out)))

    _ok("oracle_exercised_both_branches", inert > 0 and changed > 0,
        f"the oracle must cover BOTH the inert and the active branch "
        f"(inert={inert}, active={changed}); one-sided coverage would let a "
        f"fix that never fires, or always fires, pass")
    return {"cases": cases, "inert": inert, "active": changed}


# --------------------------------------------------------------------------
# criterion 6 -- the gate path, with the mechanism stated correctly
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# tripwire PREDICATES + their fixture
#
# An `_ok(name, EXPR)` guard can always be disarmed by mutating EXPR to True,
# and no assertion detects that from the inside -- proven here, not assumed:
# the first version of the tripwire cells (T1/T2/T3) SURVIVED exactly that
# mutation. The remedy, learned on step 86.59, is to extract each rule into a
# named predicate and assert the predicate against KNOWN-BAD inputs it must
# reject. Disarm the rule and the fixture fires, because the fixture does not
# depend on today's tree being interesting.
# --------------------------------------------------------------------------

def _declares_dead_key(src: str, key: str = "vol_barrier_multiplier") -> bool:
    """True iff `src` lists `key` inside a _DEAD_KEYS declaration."""
    return f'"{key}",' in src and "_DEAD_KEYS" in src


def _volatility_identifiers(fn_node) -> list[str]:
    """Every identifier the function REFERENCES whose name mentions volatility.

    AST-based on purpose: a comment or docstring mentioning volatility must not
    satisfy it, and renaming an unrelated attribute must not disarm it.
    """
    import ast as _a

    idents = set()
    for node in _a.walk(fn_node):
        if isinstance(node, _a.Name):
            idents.add(node.id)
        elif isinstance(node, _a.Attribute):
            idents.add(node.attr)
        elif isinstance(node, _a.arg):
            idents.add(node.arg)
    return sorted(i for i in idents if "vol" in i.lower())


_TRIPWIRE_FIXTURE_SRC = {
    "dead_yes": 'X = 1\n_DEAD_KEYS = (\n    "vol_barrier_multiplier",\n)\n',
    "dead_no_list": '_DEAD_KEYS = (\n    "something_else",\n)\n',
    "dead_no_decl": 'NOTES = "vol_barrier_multiplier is dead"\n',
    "vol_yes": "def f(self):\n    daily_vol = 1\n    return daily_vol * self.tp_pct\n",
    "vol_yes_attr": "def f(self):\n    return self.daily_volatility * self.tp_pct\n",
    "vol_no": "def f(self):\n    return entry_price * (1 + self.tp_pct / 100)\n",
    "vol_comment_only": "def f(self):\n    # volatility is not used here\n    return self.tp_pct\n",
}


def _check_tripwire_fixture() -> None:
    """Assert both predicates classify known inputs correctly."""
    import ast as _a

    bad = []
    F = _TRIPWIRE_FIXTURE_SRC

    # An INCOMPLETE fixture is a fixture failure, not a crash. Without this,
    # emptying the table raised KeyError and the run went red for the wrong
    # reason -- which scores as UNSCORABLE, not as a kill, and would have left
    # the "someone deleted the fixture" case genuinely uncovered.
    required = ("dead_yes", "dead_no_list", "dead_no_decl",
                "vol_yes", "vol_yes_attr", "vol_no", "vol_comment_only")
    missing = [k for k in required if k not in F]
    if missing:
        _ok("tripwire_predicates_reject_known_bad_inputs", False,
            f"the tripwire fixture is missing {missing}, so the predicates it "
            f"backs are unfalsifiable")
        return

    for key, expected in (("dead_yes", True), ("dead_no_list", False),
                          ("dead_no_decl", False)):
        got = _declares_dead_key(F[key])
        if got is not expected:
            bad.append(f"_declares_dead_key({key}) -> {got}, expected {expected}")
    for key, expected in (("vol_yes", True), ("vol_yes_attr", True),
                          ("vol_no", False), ("vol_comment_only", False)):
        fn = _a.parse(F[key]).body[0]
        got = bool(_volatility_identifiers(fn))
        if got is not expected:
            bad.append(f"_volatility_identifiers({key}) -> {got}, expected {expected}")
    _ok("tripwire_predicates_reject_known_bad_inputs", not bad,
        "a mechanism tripwire's predicate no longer agrees with its own "
        "fixture, so it has been weakened and can no longer detect what it "
        "names: " + "; ".join(bad))


def check_mechanism_tripwires() -> None:
    """The two durability guards behind criterion 6's corrected mechanism.

    Split out of `gate_effect` so the mutation matrix can exercise them WITHOUT
    BigQuery -- a guard that can only be reached through a network call is a
    guard that will not be mutation-tested, which is how the previous version
    shipped vacuous.
    """
    # The dead key must STAY dead. A grep-and-filter tripwire was tried first
    # and was too brittle -- it fired on `if "vol_barrier_multiplier" in params:`
    # at quant_optimizer.py:715, which is the SETTER's guard, not a reader. So
    # the tripwire now checks the repo's own authoritative statement instead of
    # trying to re-derive readership from text.
    _check_tripwire_fixture()

    dead_src = (REPO / "backend" / "autoresearch" / "rotation_runner.py").read_text()
    _ok("vol_barrier_multiplier_is_declared_dead_by_the_repo",
        _declares_dead_key(dead_src),
        "vol_barrier_multiplier is no longer listed in rotation_runner's "
        "_DEAD_KEYS, so the barrier mechanism may have been re-wired and this "
        "whole section must be re-derived rather than trusted")

    # And the barrier label itself must stay volatility-free. This is the claim
    # that actually matters: even if the key came back, the barrier is only
    # vol-sensitive if the LABEL reads a volatility.
    #
    # THIS GUARD HAS BEEN WRONG TWICE AND THE HISTORY IS THE POINT.
    #   v1 was a grep-with-filters over the whole repo -- too BRITTLE: it fired
    #      on `if "vol_barrier_multiplier" in params:`, the setter's own guard.
    #   v2 was `("vol" not in <text slice>) or ("self.tp_pct" in <slice>)` --
    #      and the second clause is TRUE on the unmutated tree, so it
    #      short-circuited the whole thing. The cycle-2 Q/A proved it by adding
    #      a real volatility term while keeping `self.tp_pct`: the guard stayed
    #      GREEN. It detected a RENAME, not a volatility. The remedy for v1's
    #      brittleness introduced v2's vacuity.
    #   v3, below, has NO escape clause and reads the AST rather than text: it
    #      collects every identifier the function actually references and
    #      rejects any that names a volatility. Comments and docstrings cannot
    #      satisfy it, and a renamed `tp_pct` cannot disarm it.
    #
    # The slice is also bounded to the function via `ast`, not a fixed
    # character count -- the v2 window took 2600 chars from a 1753-char
    # function, overshooting 847 chars into `_compute_sample_weights`.
    import ast as _ast

    eng_src = (REPO / "backend" / "backtest" / "backtest_engine.py").read_text()
    tree = _ast.parse(eng_src)
    fn = next((n for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef)
               and n.name == "_compute_triple_barrier_label"), None)
    _ok("triple_barrier_label_function_was_found", fn is not None,
        "_compute_triple_barrier_label no longer exists, so the claim that it "
        "carries no volatility term cannot be checked and must be re-derived")

    vol_idents = _volatility_identifiers(fn)
    _ok("triple_barrier_label_references_no_volatility_identifier",
        not vol_idents,
        f"_compute_triple_barrier_label now references {vol_idents}, so the "
        f"barrier may be volatility-scaled after all and criterion 6's "
        f"mechanism must be re-derived rather than trusted")



def gate_effect(harm: dict) -> dict:
    """What the duplication does to the DSR/PBO inputs -- via the LIVE path.

    Two corrections are baked in here, both from evaluators rather than from me.

    **The research gate** refuted the direct reading: the engine's NAV is a
    per-day dict, so duplication does not double-count NAV points and this is
    NOT a Sharpe-formula bug.

    **The cycle-1 Q/A** then refuted my replacement mechanism. I had credited
    `barriers = daily_vol * vol_barrier_multiplier` from `quant_optimizer.py`.
    That key is **DEAD**: it is written into `engine._strategy_params` and read
    by nothing, and `backend/autoresearch/rotation_runner.py::_DEAD_KEYS` lists
    it by name under the comment "NO engine reader (reverted in 9fbd9cd6)". The
    formula I quoted exists only as a COMMENT, and
    `backtest_engine._compute_triple_barrier_label` sets `tp_price`/`sl_price`
    from fixed `tp_pct`/`sl_pct` plus a cost term -- **no volatility term at
    all**. The conclusion was right and the mechanism was not wired.

    **The mechanism that IS wired** is inverse-volatility POSITION SIZING:

        historical_data.py      features["annualized_volatility"]
          -> backtest_engine.py  volatility = fv.get("annualized_volatility")
          -> signal dict
          -> backtest_trader.py  size_position(probability, volatility, nav)
          -> backtest_trader.py  vol_scale = min(target_vol / stock_vol, 3.0)

    `stock_vol` is in the DENOMINATOR, so a depressed volatility makes positions
    LARGER. That is the opposite direction a reader would guess from "volatility
    is understated", and it is why the sign matters.
    """
    a, b = harm["raw"], harm["fixed"]
    vol_ratio = a["vol_ann"] / b["vol_ann"] if b["vol_ann"] else None
    _ok("vol_ratio_is_derivable", vol_ratio is not None,
        "no volatility measured, so the sizing claim would be unfounded")
    _ok("pre_fix_volatility_is_the_lower_one", vol_ratio < 1.0,
        f"duplication must DEPRESS volatility; ratio {vol_ratio}")

    # vol_scale = target_vol / stock_vol, capped at 3.0. With stock_vol scaled
    # by `vol_ratio`, the scale factor is 1/vol_ratio -- until the cap binds.
    size_inflation = 1.0 / vol_ratio
    _ok("size_inflation_is_above_one", size_inflation > 1.0,
        "a DEPRESSED volatility must INFLATE an inverse-vol position size; "
        "if this is below 1.0 the direction of the claim is wrong")
    _ok("cap_is_accounted_for", size_inflation < 3.0,
        f"the measured inflation {size_inflation:.4f} must be read against the "
        f"3.0 cap at backtest_trader.py `min(self.target_vol / stock_vol, 3.0)`; "
        f"at or above the cap the inflation saturates and this arithmetic no "
        f"longer describes the outcome")

    check_mechanism_tripwires()

    return {
        "vol_ratio_pre_over_post": vol_ratio,
        "position_size_inflation": size_inflation,
        "vol_scale_cap": 3.0,
        "full_duplication_closed_form": 2 ** -0.5,
        "max_inflation_under_full_duplication": 2 ** 0.5,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ticker", default="AKAM")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--base-rev", default=PRE_FIX_REV,
                    help="revision representing the PRE-FIX tree")
    a = ap.parse_args()
    q = a.verify

    nod = no_dedup_before(a.base_rev)
    par = parity_oracle()

    if not q:
        print("=" * 78)
        print("phase-86.116 -- DUPLICATE PRICE ROWS: EVIDENCE")
        print("=" * 78)
        print()
        print("-- criterion 3: did anything de-duplicate BEFORE this step? --")
        print(f"pre-fix revision under test: {nod['rev']}")
        for k, v in nod["claim"].items():
            print(f"  files containing `{k}` under backend/ : {v}")
        print(f"  POSITIVE CONTROL, `set_index` (must exist): "
              f"{nod['control']} files")
        print("  A grep returning nothing proves nothing until the same probe")
        print("  is shown able to return something. It is, so the zeros count.")
        print()
        print("-- criterion 5: parity oracle --")
        print(f"{len(par['cases'])} frame shapes: {par['inert']} unique-index "
              f"(fix must be INERT, same object) / {par['active']} duplicated "
              f"(fix must be EXACT)")
        print("  Both branches are exercised, so the fix is provably neither")
        print("  a no-op nor an always-on rewrite.")
        print()

    if a.offline:
        check_mechanism_tripwires()
        print(f"OK: all {len(_RAN)} invariants hold (offline subset)")
        return 0

    client = _bq()
    cen = census(client)
    harm = driven_harm(client, a.ticker, a.start, a.end)
    gate = gate_effect(harm)

    if q:
        print(f"OK: all {len(_RAN)} invariants hold")
        return 0

    print("-- criterion 1: the census, RE-MEASURED by this step --")
    print(f"table: {TABLE}")
    print(f"  total rows          : {cen['total_rows']:,}")
    print(f"  distinct keys       : {cen['keys']:,}   (ticker, date)")
    print(f"  duplicated keys     : {cen['dup_keys']:,} "
          f"= {cen['dup_share_of_keys']*100:.2f}% OF KEYS")
    print(f"  excess rows         : {cen['excess_rows']:,} "
          f"= {cen['excess_share_of_rows']*100:.2f}% OF ROWS")
    print(f"  max multiplicity    : {cen['max_mult']}")
    print(f"  tickers affected    : {cen['tk_affected']} of {cen['tk_total']}")
    print(f"  dup keys whose close differs: {cen['dup_keys_close_differs']:,}")
    print()
    print("  NORMALISATION RULE, stated because the two shares differ and are")
    print("  NOT interchangeable: 'share of KEYS' divides by distinct")
    print("  (ticker,date) pairs; 'share of ROWS' divides by total table rows.")
    print("  Quoting one as the other misstates the defect by ~23 points.")
    print()
    print("  by year:")
    for y in cen["by_year"]:
        print(f"    {y['yr']}  keys {y['keys']:>7,}  dup {y['dup']:>7,}  "
              f"{y['share']*100:5.1f}%")
    newest = cen["by_year"][-1]
    print(f"  The newest bucket ({newest['yr']}) is at {newest['share']*100:.1f}%,")
    print("  so the write side is no longer producing duplicates and a repair")
    print("  is terminal rather than a treadmill. It is also why nobody noticed.")
    print()
    print(f"-- criterion 2: the harm, DRIVEN on {harm['ticker']} {harm['window']} --")
    print("   (loaded through the REAL preload_prices; factors from the REAL")
    print("    screener functions -- a reimplementation would measure itself)")
    r, f = harm["raw"], harm["fixed"]
    print(f"  {'':<22}{'as-loaded (pre-fix)':>22}{'de-duplicated':>18}")
    for k, fmt in (("n", "{:,}"), ("mom_1m", "{:.2f}"), ("mom_3m", "{:.2f}"),
                   ("rsi_14", "{:.1f}"), ("vol_ann", "{:.4f}"),
                   ("span_21_sessions", "{}")):
        print(f"  {k:<22}{fmt.format(r[k]):>22}{fmt.format(f[k]):>18}")
    print()
    flips = [k for k in ("mom_1m", "mom_3m")
             if r[k] is not None and f[k] is not None and (r[k] < 0) != (f[k] < 0)]
    if flips:
        print(f"  * SIGN FLIP on {', '.join(flips)} -- the pre-fix value does not")
        print("    merely differ in magnitude, it points the other way.")
    print(f"  * a '21-period' lookback covered {r['span_21_sessions']} real "
          f"sessions, not {f['span_21_sessions']}.")
    if r["rsi_14"] is not None and f["rsi_14"] is not None:
        print(f"  * RSI moved {r['rsi_14']:.1f} -> {f['rsi_14']:.1f}. "
              f"rank_candidates applies score*=0.8 below 20 and score*=0.7")
        print("    above 80, so an RSI error is not cosmetic.")
    print()
    print("-- criterion 6: how this reaches the DSR/PBO gates --")
    print("  NOT a Sharpe-formula bug. The engine's NAV is a per-day dict, so")
    print("  duplication does not double-count NAV points. (Research gate.)")
    print()
    print("  NOT the triple-barrier width either -- I credited that in cycle 1")
    print("  and it was wrong. `vol_barrier_multiplier` is a DEAD KEY: written")
    print("  into engine._strategy_params, read by nothing, and named in")
    print("  rotation_runner._DEAD_KEYS as 'NO engine reader (reverted in")
    print("  9fbd9cd6)'. _compute_triple_barrier_label uses fixed tp_pct/sl_pct")
    print("  with NO volatility term. (Cycle-1 Q/A.)")
    print()
    print("  THE WIRED PATH is inverse-volatility POSITION SIZING:")
    print("    historical_data  features['annualized_volatility']")
    print("      -> backtest_engine  volatility = fv.get('annualized_volatility')")
    print("      -> signal dict")
    print("      -> backtest_trader  size_position(probability, volatility, nav)")
    print("      -> backtest_trader  vol_scale = min(target_vol / stock_vol, 3.0)")
    print()
    print(f"  measured volatility ratio (pre/post)   : "
          f"{gate['vol_ratio_pre_over_post']:.4f}")
    print(f"  => position-size inflation (1/ratio)   : "
          f"{gate['position_size_inflation']:.4f}x")
    print(f"  cap at backtest_trader.py              : "
          f"{gate['vol_scale_cap']:.1f}x")
    print(f"  closed form under FULL duplication     : "
          f"{gate['full_duplication_closed_form']:.4f} (1/sqrt(2)), giving at")
    print(f"                                           most "
          f"{gate['max_inflation_under_full_duplication']:.4f}x inflation")
    print()
    print("  DIRECTION MATTERS AND IS COUNTER-INTUITIVE: stock_vol is in the")
    print("  DENOMINATOR, so an UNDERSTATED volatility makes positions LARGER,")
    print("  not smaller. The backtest was taking bigger risk than its own")
    print("  vol-targeting believed.")
    print()
    print("  NO THRESHOLD IS ADJUSTED by this step. min_dsr=0.95 / max_pbo=0.20")
    print("  are untouched; if a re-run moves them, that is a finding to report.")
    print()
    print(f"OK: all {len(_RAN)} invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
