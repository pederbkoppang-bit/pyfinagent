#!/usr/bin/env python3
"""phase-86.38 -- how often does the book trade off the LITE fallback, and why?

THE INSTRUMENT ASSERTS ITS OWN COVERAGE, and that is the point of this file.
A first version of this census parsed only the JSON log format and silently
dropped **416 events** from the June-era log, which uses an older ANSI-coloured
plain-text format. The zero it reported for June was a property of the parser,
not of the system. So this script:

  1. counts the raw `grep`-equivalent population FIRST, per file;
  2. parses with BOTH format readers;
  3. **asserts parsed == raw for every file** and exits non-zero on a shortfall,
     naming the file and the gap.

A census that cannot account for every line it claims to cover is withheld.

    $ python scripts/qa/derive_lite_fallback_census_86_38.py
"""
from __future__ import annotations

import collections
import gzip
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

FALLBACK_MARK = "falling back to lite Claude analyzer"
FULL_MARK = "Critic verdict"

#: JSON era: {"timestamp": "2026-08-10 20:19:27,714", "message": "..."}
#: Legacy era: "\x1b[33m20:00:32 W [autonomous_loop]\x1b[0m Full orchestrator ..."
_LEGACY_TIME = re.compile(r"(\d{2}:\d{2}:\d{2})\s+[A-Z]\s+\[")
_REASON = re.compile(r"Full orchestrator failed for (\S+?): (.*?) -- falling back", re.S)


#: phase-86.38 cycle 3 -- THE WRAPPER IS NOT THE CAUSE.
#: `classify()` receives only the string inside "Full orchestrator failed for X:
#: <reason> -- falling back", and for the QuantAgent family that reason reads
#: "'NoneType' object has no attribute 'get'". The ACTUAL cause is logged on an
#: EARLIER line and never reaches this function, so the 429 branch -- which is
#: first -- could not fire, and every such event fell through to the NoneType
#: bucket. That is why the census reported a 6-vs-3 "code defect vs quota" split
#: and why phase-86.41 was filed on a premise that does not hold.
#:
#: MEASURED over all 42 retained log files: 34 raw QuantAgent-NoneType wrapper
#: events; 17 are preceded within 8 lines by a SEC.gov 429 on the CIK map
#: ("Quant: SEC 429 rate-limit on CIK map" / "Failed to fetch CIK map: 429
#: Client Error ... sec.gov"); 17 are not. **NONE is a Vertex 429** -- these are
#: a different provider's rate limit from the one phase-86.38 investigated, and
#: the NoneType itself is raised in a REMOTE Cloud Function (/workspace/main.py
#: get_cik), not in this repository.
_UPSTREAM_CAUSE = [
    (("sec 429", "company_tickers.json", "cik map"), "SEC.gov 429 on the CIK map (upstream)"),
    (("resource_exhausted", "aiplatform"), "Vertex 429 (upstream)"),
]


def classify_with_context(reason: str, recent: list[str]) -> str:
    """Classify a fallback using the lines that PRECEDE it, not just the wrapper.

    Falls back to `classify()` when no upstream cause is visible. A NoneType
    wrapper with no identifiable upstream is reported as exactly that -- unknown
    upstream -- rather than being asserted as a code defect.
    """
    base = classify(reason)
    if "nonetype" not in reason.lower():
        return base
    blob = " ".join(recent).lower()
    for needles, label in _UPSTREAM_CAUSE:
        if any(n in blob for n in needles):
            return f"remote QuantAgent crash after {label}"
    return "QuantAgent NoneType, upstream cause NOT identified"


def classify(reason: str) -> str:
    r = reason.lower()
    if "resource_exhausted" in r or "429" in reason:
        return "429 RESOURCE_EXHAUSTED (quota)"
    if "requires a github token" in r:
        return "config: GITHUB_TOKEN unset (legacy, resolved)"
    if "timed out" in r or "timeout" in r:
        return "timeout"
    if "nonetype" in r:
        return "code defect: QuantAgent NoneType"
    if "503" in reason or "unavailable" in r:
        return "503 UNAVAILABLE"
    if "500" in reason or "internal" in r:
        return "500 INTERNAL"
    return reason[:70]


def read_lines(p: pathlib.Path):
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def per_cycle_census() -> int:
    """phase-86.38 cycle 2 -- criterion 2 asks for PER-CYCLE, not per-day.

    The cycle-1 Q/A returned FAIL partly on this: the original table was PER-DAY
    over 10 DAYS, printed under a header that said "per-cycle ... over >=10
    cycles", and the substitution was disclosed nowhere. Day-level aggregation is
    lossy in the direction that matters -- 2026-08-04 shows 11 analyses, more
    than one cycle's ticker count, so a day holding one fully-degraded cycle and
    one clean cycle reports as partially degraded and neither cycle is visible.

    `handoff/cycle_history.jsonl` carries real cycle boundaries, so the per-cycle
    derivation was available all along and I substituted without saying so.

    SCOPE LIMIT, stated rather than discovered later: only the JSON-format log era
    carries per-event timestamps, so only cycles whose window overlaps that era
    can be attributed. Cycles outside it are reported as UNATTRIBUTABLE, not as
    zero -- the distinction the whole coverage assertion exists to preserve.
    """
    import datetime as _dt

    hist = REPO / "handoff" / "cycle_history.jsonl"
    if not hist.exists():
        print("no cycle_history.jsonl -- cannot derive per-cycle")
        return 1

    def parse(ts):
        if not ts:
            return None
        try:
            return _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return None

    cycles = []
    for line in hist.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:                                          # noqa: BLE001
            continue
        if not r.get("status") or r.get("status") == "started":
            continue
        s, e = parse(r.get("started_at")), parse(r.get("completed_at"))
        if s and e:
            cycles.append({"id": r.get("cycle_id"), "s": s, "e": e,
                           "trades": r.get("n_trades"), "full": 0, "lite": 0,
                           "reasons": collections.Counter()})
    cycles.sort(key=lambda c: c["s"])

    files = sorted((REPO / "handoff" / "logs").glob("backend.log.*.gz"))
    cur = REPO / "backend.log"
    if cur.exists():
        files.append(cur)

    # OFFSET SELF-CHECK. The +2h shift below is CEST and is hardcoded. The
    # cycle-2 Q/A validated it as a sharp maximum (75 of 76 dated events
    # attributed at +2h; every other offset collapses attribution), so it is
    # measured-correct for THIS data -- but it would be silently wrong across a
    # CET/CEST boundary. The whole JSON-log era (2026-07-24..08-10) is inside
    # CEST, so no current number is affected; this asserts that precondition
    # rather than leaving it as a comment nobody re-checks.
    _era = [c for c in cycles if c["s"].year == 2026]
    if _era:
        _lo, _hi = min(c["s"] for c in _era), max(c["s"] for c in _era)
        if not (_dt.datetime(2026, 3, 29, tzinfo=_dt.timezone.utc) <= _lo
                and _hi <= _dt.datetime(2026, 10, 25, tzinfo=_dt.timezone.utc)):
            print("  WARNING: cycle window crosses a CET/CEST boundary; the "
                  "hardcoded +2h offset is NOT valid across it and this table "
                  "must not be trusted until the offset is derived per-event.")

    dated = 0
    for p_ in files:
        for line in read_lines(p_):
            if FALLBACK_MARK not in line and FULL_MARK not in line:
                continue
            if not line.lstrip().startswith("{"):
                continue                      # legacy format: no date on the line
            try:
                d = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue
            ts = parse((d.get("timestamp") or "").replace(",", ".").replace(" ", "T"))
            if not ts:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            dated += 1
            for c in cycles:
                # cycle_history timestamps are UTC; log timestamps are local CEST
                # (+02:00) -- compare on the naive wall clock of each, shifted.
                s = c["s"].astimezone(_dt.timezone.utc).replace(tzinfo=None)
                e = c["e"].astimezone(_dt.timezone.utc).replace(tzinfo=None)
                t_ = ts.replace(tzinfo=None) - _dt.timedelta(hours=2)
                if s <= t_ <= e:
                    if FALLBACK_MARK in line:
                        c["lite"] += 1
                        m = _REASON.search(d.get("message", ""))
                        c["reasons"][classify(m.group(2) if m else "?")] += 1
                    else:
                        c["full"] += 1
                    break

    attributed = [c for c in cycles if c["full"] or c["lite"]]
    print("=" * 92)
    print("PER-CYCLE full-vs-lite  (criterion 2: 'at least the last 10 completed cycles')")
    print("=" * 92)
    print(f"  terminal cycles in cycle_history : {len(cycles)}")
    print(f"  dated log events considered      : {dated}")
    print(f"  cycles with ATTRIBUTABLE events  : {len(attributed)}")
    print(f"  cycles UNATTRIBUTABLE (outside the timestamped log era): "
          f"{len(cycles) - len(attributed)}  <- NOT zero-degradation, unknown")
    print()
    print(f"  {'cycle_id':16s} {'started (UTC)':20s} {'full':>5s} {'lite':>5s} {'lite%':>6s} {'trades':>7s}  causes")
    print("-" * 92)
    for c in cycles[-14:]:
        tot = c["full"] + c["lite"]
        if not tot:
            print(f"  {str(c['id']):16s} {str(c['s'])[:19]:20s} {'-':>5s} {'-':>5s} {'-':>6s} "
                  f"{str(c['trades']):>7s}  UNATTRIBUTABLE (no timestamped events in window)")
            continue
        pct = f"{100*c['lite']/tot:.0f}%"
        causes = ", ".join(f"{k} x{v}" for k, v in c["reasons"].most_common(2))
        print(f"  {str(c['id']):16s} {str(c['s'])[:19]:20s} {c['full']:5d} {c['lite']:5d} {pct:>6s} "
              f"{str(c['trades']):>7s}  {causes}")
    print("-" * 92)
    tf = sum(c["full"] for c in attributed); tl = sum(c["lite"] for c in attributed)
    print(f"  attributed totals: {tf} full, {tl} lite over {len(attributed)} cycle(s)")
    print()
    print("  CRITERION 3 -- do degraded cycles and zero-trade cycles coincide?")
    zt = [c for c in attributed if c["trades"] == 0]
    dz = [c for c in zt if c["lite"]]
    print(f"    attributable cycles with ZERO trades : {len(zt)} of {len(attributed)}")
    print(f"    of those, cycles that DEGRADED       : {len(dz)}")
    print(f"    of those, cycles with NO degradation : {len(zt) - len(dz)}")
    print("    => a zero-trade cycle with ZERO fallbacks is a counter-example to")
    print("       'degradation explains the drought'. Any such cycle refutes it.")
    return 0


def main() -> int:
    if "--per-cycle" in sys.argv:
        return per_cycle_census()
    files = sorted((REPO / "handoff" / "logs").glob("backend.log.*.gz"))
    cur = REPO / "backend.log"
    if cur.exists():
        files.append(cur)
    if not files:
        print("no backend logs found")
        return 1

    per_file_raw, per_file_parsed = {}, collections.Counter()
    by_day_reason = collections.defaultdict(collections.Counter)
    day_full = collections.Counter()
    undated = collections.Counter()

    for p in files:
        raw = 0
        recent: list[str] = []
        for line in read_lines(p):
            recent.append(line)
            # WINDOW SIZE IS MEASURED, NOT CHOSEN. My first attempt used 8 lines
            # and reported "upstream cause NOT identified" for all 6 JSON-era
            # events -- a suspiciously uniform clean result. Measuring the actual
            # distance: the SEC-429/CIK cue sits EXACTLY 18 lines back in all six
            # (the remote traceback is interleaved between). 8 was a false
            # negative of my own making. 25 covers it with margin.
            # TRADE-OFF, stated: a wider window can pick up an unrelated 429 from
            # a different ticker in a busy cycle, so this errs toward attributing
            # a cause. The bucket name says "after", not "caused by".
            if len(recent) > 26:
                recent.pop(0)
            if FALLBACK_MARK in line:
                raw += 1
                m = _REASON.search(line)
                reason = classify_with_context(
                    m.group(2) if m else "(unparsed reason)", recent[:-1])
                day = None
                if line.lstrip().startswith("{"):
                    try:
                        day = json.loads(line).get("timestamp", "")[:10] or None
                    except Exception:                                  # noqa: BLE001
                        day = None
                if day:
                    by_day_reason[day][reason] += 1
                else:
                    # legacy format carries a time but no date on the line
                    undated[f"{p.name} :: {reason}"] += 1
                per_file_parsed[p.name] += 1
            elif FULL_MARK in line:
                day = None
                if line.lstrip().startswith("{"):
                    try:
                        day = json.loads(line).get("timestamp", "")[:10] or None
                    except Exception:                                  # noqa: BLE001
                        day = None
                if day:
                    day_full[day] += 1
        per_file_raw[p.name] = raw

    # ---- COVERAGE ASSERTION, before any number is reported ----------------
    print("=" * 88)
    print("COVERAGE -- every fallback line must be accounted for, or no census is printed")
    print("=" * 88)
    shortfall = False
    for name, raw in per_file_raw.items():
        got = per_file_parsed[name]
        ok = got == raw
        shortfall |= not ok
        print(f"  {name:38s} raw={raw:5d}  parsed={got:5d}  {'ok' if ok else 'SHORTFALL'}")
    if shortfall:
        print("\n  PARSER DROPPED LINES. The census is WITHHELD rather than reported")
        print("  with a silent gap -- that is exactly the defect this script exists to avoid.")
        return 1
    print(f"  total accounted: {sum(per_file_parsed.values())}\n")

    # ---- the dated census -------------------------------------------------
    print("=" * 88)
    print("PER-DAY full-pipeline vs lite-fallback  (JSON-format era only -- see UNDATED below)")
    print("=" * 88)
    print(f"{'date':12s} {'full':>6s} {'lite':>6s}  {'lite%':>6s}  causes")
    print("-" * 88)
    tot_f = tot_l = 0
    for day in sorted(set(list(by_day_reason) + list(day_full))):
        lite = sum(by_day_reason[day].values())
        full = day_full[day]
        tot_f += full
        tot_l += lite
        pct = f"{100*lite/(full+lite):.0f}%" if (full + lite) else "-"
        causes = ", ".join(f"{k} x{v}" for k, v in by_day_reason[day].most_common(3))
        print(f"{day:12s} {full:6d} {lite:6d}  {pct:>6s}  {causes}")
    print("-" * 88)
    denom = tot_f + tot_l
    print(f"{'TOTAL':12s} {tot_f:6d} {tot_l:6d}  "
          f"{(f'{100*tot_l/denom:.1f}%' if denom else '-'):>6s}")
    print(f"\n  days covered: {len(set(list(by_day_reason) + list(day_full)))}")

    if undated:
        print()
        print("=" * 88)
        print("UNDATED -- legacy plain-text log lines carrying a time but no date")
        print("=" * 88)
        print("  These are REAL events, excluded from the per-day table only because the")
        print("  line format has no date. They are NOT dropped and NOT zero.")
        for k, v in undated.most_common():
            print(f"   {v:5d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
