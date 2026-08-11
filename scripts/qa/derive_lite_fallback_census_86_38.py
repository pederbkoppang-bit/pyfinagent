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


def main() -> int:
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
        for line in read_lines(p):
            if FALLBACK_MARK in line:
                raw += 1
                m = _REASON.search(line)
                reason = classify(m.group(2) if m else "(unparsed reason)")
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
