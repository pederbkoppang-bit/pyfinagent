#!/usr/bin/env python3
"""phase-86.108 criterion 1 (second half): the rail split, delivered as an ERA
BUCKET and labelled as one.

## Read this before quoting anything below as a rail attribution

**A per-event `claude_code`-vs-`gemini` split of the invalid-JSON corpus is not
derivable, and this script does not produce one.** The reasons are measured,
not assumed:

  - a JSON marker record's entire field set is `timestamp/level/module/message`;
    `grep -c '"model"'` over the corpus returns **0**. No log line carries a
    rail, a provider or a model.
  - `pyfinagent_data.llm_call_log` DOES carry `provider`, but the warning lines
    carry no `request_id`, so any join is time-proximity, not identity.
  - that table's `ok` column is "true on 2xx", and an invalid-JSON body **is**
    a 2xx. So `llm_call_log` can tell you the rail mix of a WINDOW; it can
    never identify which call produced a parse failure.

What this script therefore reports is a **time co-occurrence**: for each
rotation window, how many invalid-JSON lines that window's log contains, beside
what fraction of that window's logged LLM calls went out on each provider. Two
series over the same clock. Reading a causal split out of it would be exactly
the fabrication the contract said to refuse.

The **fix for the underlying gap ships in this same step**: every new failure
is recorded by `backend/agents/parse_failure_ledger.py` with the rail stamped
at record time, so the attribution that cannot be recovered retrospectively is
available on every event from here on.

## The eras

Each rotated `backend.log.<UTC>.gz` covers the window ending at the timestamp
in its own name and beginning at the previous file's. The live `backend.log`
covers everything after the last rotation and is therefore **still growing** --
its count is printed with the file's size and mtime so the number is
attributable to a moment rather than presented as final.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
MARKER = "returned invalid JSON"
_STAMP = re.compile(r"backend\.log\.(\d{8}T\d{6}Z)\.gz$")

# Measured 2026-08-17 against pyfinagent_data.llm_call_log (7,248 rows,
# partitioned on ts). `--sql` prints this verbatim and it is EXECUTABLE as
# printed -- paste it into the BigQuery MCP or `bq query --use_legacy_sql=false`.
#
# An earlier revision of this constant carried a prose placeholder where the
# era rows belong and pointed at a `--refresh-help` flag that was never
# implemented, so the advertised re-derivation path did not exist. The numbers
# were true and independently confirmed; the path to re-deriving them was not.
RAIL_QUERY = """
WITH eras AS (
  SELECT 'e1_upto_20260612T104931Z' AS era, TIMESTAMP('2000-01-01') AS lo, TIMESTAMP('2026-06-12 10:49:31') AS hi UNION ALL
  SELECT 'e2_upto_20260706T225648Z', TIMESTAMP('2026-06-12 10:49:31'), TIMESTAMP('2026-07-06 22:56:48') UNION ALL
  SELECT 'e3_upto_20260724T064045Z', TIMESTAMP('2026-07-06 22:56:48'), TIMESTAMP('2026-07-24 06:40:45') UNION ALL
  SELECT 'e4_upto_20260729T171222Z', TIMESTAMP('2026-07-24 06:40:45'), TIMESTAMP('2026-07-29 17:12:22') UNION ALL
  SELECT 'e5_upto_20260804T182713Z', TIMESTAMP('2026-07-29 17:12:22'), TIMESTAMP('2026-08-04 18:27:13') UNION ALL
  SELECT 'e6_upto_20260810T064130Z', TIMESTAMP('2026-08-04 18:27:13'), TIMESTAMP('2026-08-10 06:41:30') UNION ALL
  SELECT 'e7_upto_20260814T155315Z', TIMESTAMP('2026-08-10 06:41:30'), TIMESTAMP('2026-08-14 15:53:15') UNION ALL
  SELECT 'e8_live_after_20260814T155315Z', TIMESTAMP('2026-08-14 15:53:15'), TIMESTAMP('2030-01-01')
)
SELECT e.era,
       COUNTIF(l.provider='claude-code') AS claude_code,
       COUNTIF(l.provider='anthropic')   AS anthropic,
       COUNTIF(l.provider='gemini')      AS gemini,
       COUNT(l.ts) AS total
FROM eras e
LEFT JOIN `sunny-might-477607-p8.pyfinagent_data.llm_call_log` l
  ON l.ts > e.lo AND l.ts <= e.hi
GROUP BY e.era ORDER BY e.era
"""

# era-end stamp -> measured provider mix for that window
RAIL_MIX: dict[str, dict[str, int]] = {
    "20260612T104931Z": {"claude_code": 0,  "anthropic": 211,  "gemini": 210, "total": 421},
    "20260706T225648Z": {"claude_code": 11, "anthropic": 2587, "gemini": 57,  "total": 2655},
    "20260724T064045Z": {"claude_code": 10, "anthropic": 1266, "gemini": 187, "total": 1463},
    "20260729T171222Z": {"claude_code": 2,  "anthropic": 169,  "gemini": 23,  "total": 194},
    "20260804T182713Z": {"claude_code": 6,  "anthropic": 402,  "gemini": 82,  "total": 490},
    "20260810T064130Z": {"claude_code": 4,  "anthropic": 813,  "gemini": 144, "total": 961},
    "20260814T155315Z": {"claude_code": 7,  "anthropic": 562,  "gemini": 69,  "total": 638},
    "LIVE":             {"claude_code": 0,  "anthropic": 375,  "gemini": 51,  "total": 426},
}


def count_marker(p: pathlib.Path) -> int:
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if MARKER in line)


def main(as_json: bool) -> int:
    rotated = sorted((REPO / "handoff" / "logs").glob("backend.log.*.gz"))
    live = REPO / "backend.log"
    rows = []
    prev = "(beginning of corpus)"
    for p in rotated:
        m = _STAMP.search(p.name)
        stamp = m.group(1) if m else p.name
        rows.append({
            "era_end": stamp, "era_start": prev, "file": p.name,
            "invalid_json_lines": count_marker(p), "live": False,
        })
        prev = stamp
    if live.exists():
        st = live.stat()
        rows.append({
            "era_end": "LIVE", "era_start": prev, "file": live.name,
            "invalid_json_lines": count_marker(live), "live": True,
            "bytes_at_read": st.st_size,
            "mtime_at_read": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
            "read_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        })

    for r in rows:
        mix = RAIL_MIX.get(r["era_end"])
        if mix and mix["total"]:
            r["rail_mix"] = mix
            r["cc_pct_of_logged_calls"] = round(100 * mix["claude_code"] / mix["total"], 2)
            r["gemini_pct_of_logged_calls"] = round(100 * mix["gemini"] / mix["total"], 2)
        else:
            r["rail_mix"] = None

    if as_json:
        print(json.dumps({"rows": rows, "query": RAIL_QUERY.strip()}, indent=2))
        return 0

    print("ERA BUCKET -- invalid-JSON lines beside that window's LLM-call rail mix")
    print("THIS IS A TIME CO-OCCURRENCE, NOT A PER-EVENT ATTRIBUTION. See the")
    print("module docstring: no log line carries a rail, and llm_call_log's `ok`")
    print("is 2xx-shaped, so it is blind to an invalid body.\n")
    hdr = f"{'era ends':18s} {'lines':>6s} {'calls':>6s} {'cc':>5s} {'anthr':>6s} {'gem':>5s} {'cc%':>6s}"
    print(hdr)
    print("-" * len(hdr))
    total = 0
    for r in rows:
        total += r["invalid_json_lines"]
        mix = r["rail_mix"]
        if mix:
            print(f"{r['era_end']:18s} {r['invalid_json_lines']:6d} {mix['total']:6d} "
                  f"{mix['claude_code']:5d} {mix['anthropic']:6d} {mix['gemini']:5d} "
                  f"{r['cc_pct_of_logged_calls']:5.1f}%")
        else:
            print(f"{r['era_end']:18s} {r['invalid_json_lines']:6d} {'-':>6s} "
                  f"{'-':>5s} {'-':>6s} {'-':>5s} {'-':>6s}")
    rot = sum(r["invalid_json_lines"] for r in rows if not r["live"])
    print("-" * len(hdr))
    print(f"{'ROTATED ONLY':18s} {rot:6d}   <- reproduces the filed 2,859")
    print(f"{'INCL. LIVE':18s} {total:6d}")
    for r in rows:
        if r["live"]:
            print(f"\nLIVE LOG IS STILL GROWING: {r['bytes_at_read']:,} bytes, "
                  f"mtime {r['mtime_at_read']}, read at {r['read_at']}.")
            print("Its count is a reading, not a final figure -- re-running this")
            print("script later will legitimately return a larger number.")

    # The one claim this table DOES support is computed here rather than
    # written as prose. An earlier draft of this section asserted that the two
    # largest eras were "the eras with the fewest claude-code-tagged calls";
    # that was false (era 20260729 has 2, fewer than era 20260706's 11) and it
    # was false because it was typed instead of derived.
    zero_cc = [r for r in rows if (r["rail_mix"] or {}).get("claude_code") == 0]
    zero_cc_lines = sum(r["invalid_json_lines"] for r in zero_cc)
    rot_zero_cc = sum(r["invalid_json_lines"] for r in zero_cc if not r["live"])
    cc_total = sum((r["rail_mix"] or {}).get("claude_code", 0) for r in rows)
    calls_total = sum((r["rail_mix"] or {}).get("total", 0) for r in rows)

    print("\nWHAT THE TABLE DOES AND DOES NOT SUPPORT (computed, not asserted)")
    print(f"  Eras with ZERO claude-code-tagged calls: "
          f"{', '.join(r['era_end'] for r in zero_cc) or 'none'}")
    print(f"  Invalid-JSON lines in those eras: {zero_cc_lines} "
          f"({100*rot_zero_cc/rot:.1f}% of the rotated 2,859)")
    print("  SUPPORTED: that share of the corpus falls in windows where the")
    print("             call log recorded no claude-code-tagged traffic at all,")
    print("             so it cannot be attributed to the CC rail under this")
    print("             tagging -- an UPPER bound on CC attribution, nothing more.")
    print(f"  Across every era, claude-code-tagged calls are {cc_total} of "
          f"{calls_total} ({100*cc_total/calls_total:.2f}%).")
    print("  NOT SUPPORTED: any statement of the form 'N of the failures came")
    print("             from the CC rail'. No such number exists in this data.")
    print("  CAVEAT: the CC rail is known to UNDER-tag -- its calls have been")
    print("             logged as provider='anthropic' with the wrong model")
    print("             before (phase-78). So 'cc%' is a FLOOR on CC-rail share")
    print("             and the bound above is correspondingly loose.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--sql", action="store_true",
                    help="print the executable BigQuery SQL that produced RAIL_MIX, "
                         "then exit -- run it to re-derive the rail columns")
    args = ap.parse_args()
    if args.sql:
        print(RAIL_QUERY.strip())
        raise SystemExit(0)
    raise SystemExit(main(as_json=args.json))
