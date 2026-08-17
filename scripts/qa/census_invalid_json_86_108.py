"""phase-86.108 criterion 1 -- the invalid-JSON census, with its rule STATED.

WHY THIS EXISTS AS A COMMITTED SCRIPT. Criterion 1 asks for the per-agent
rates "re-derived with the population rule and command stated". A number in a
handoff document is not re-runnable; this is. Every figure it prints carries
the glob and the match rule that produced it, and it refuses to print a rate
whose denominator does not exist in the corpus.

FOUR CORRECTIONS THIS SCRIPT ENCODES, each measured rather than assumed:

  C-a  THE CORPUS IS MIXED-FORMAT. `CompactFormatter` lines carry ANSI colour
       and a `HH:MM:SS L [module]` prefix; `JsonFormatter` lines are objects
       with timestamp/level/module/message. A `"module":`-keyed parse sees only
       the JSON share and LOOKS complete. Measured 2026-08-17: 2371 compact vs
       501 json, i.e. such a parser reports 17.4% of the truth.

  C-b  THE AGENT LABELS IN THE ORIGINAL FILING ARE A MATCH-RULE ARTIFACT.
       Taking the last token before the marker collapses distinct agents:
       "Analyst 926" is not one agent, it is Conservative (309) + Neutral (310)
       + Aggressive (307). Likewise "Advocate" is Devil's Advocate and "Judge"
       is Risk Judge. **There is no agent called "Analyst".** This script
       therefore extracts the phrase between the formatter prefix and the
       marker, and strips the prefix rather than guessing a word count -- an
       earlier draft took "the last three capitalised words" and produced
       buckets like `W orchestrator Critic`, which is the same class of error
       one level down.

  C-c  THE TOTAL COUNTS LINES, NOT EVENTS. The Critic path double-logs:
       `Critic returned invalid JSON` and `...treating as PASS with draft.`
       appear in equal numbers -- one failure, two lines. The second wording
       was REMOVED by phase-75, so the corpus spans multiple code generations
       and any rate over the whole of it mixes builds.

  C-d  NO RATE IS DERIVABLE. There is no synthesis-attempt denominator in this
       corpus (`Synthesis complete` / `Running Synthesis` / `Analysis complete`
       all return 0). The filed "9.2%" is 264/2859 -- a composition SHARE of
       invalid-JSON LINES. This script prints shares labelled as shares and
       refuses to print a failure rate.

RAIL ATTRIBUTION IS NOT DERIVABLE FROM THIS CORPUS AND THE SCRIPT SAYS SO.
A JSON marker record's entire field set is timestamp/level/module/message;
`grep -c '"model"'` returns 0. No line carries a rail, provider or model, so a
per-event claude_code-vs-gemini split would be fabricated. Criterion 1's split
must be delivered era-bucketed on `paper_use_claude_code_route`, labelled as
such. See `contract_86.108.md` P2.

    $ python scripts/qa/census_invalid_json_86_108.py
    $ python scripts/qa/census_invalid_json_86_108.py --rotated-only
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
MARKER = "returned invalid JSON"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# CompactFormatter prefix: `19:11:09 W [debate] `. Anchored, so a message that
# merely contains a bracket cannot be mistaken for a prefix.
_COMPACT_PREFIX = re.compile(r"^\d{2}:\d{2}:\d{2}\s+\w+\s+\[[^\]]+\]\s*")


def corpus(rotated_only: bool) -> list[pathlib.Path]:
    """The glob, returned so the caller can print it beside the counts."""
    files = sorted((REPO / "handoff" / "logs").glob("backend.log.*.gz"))
    if not rotated_only:
        live = REPO / "backend.log"
        if live.exists():
            files.append(live)
    return files


def _lines(p: pathlib.Path):
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def message_of(raw: str) -> tuple[str, str]:
    """Return (message_text, formatter) for one log line.

    The formatter is reported because C-a is only visible if the two shapes are
    counted separately.
    """
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            return str(json.loads(stripped).get("message", "")), "json"
        except ValueError:
            return _ANSI.sub("", stripped), "json-unparseable"
    clean = _ANSI.sub("", stripped)
    return _COMPACT_PREFIX.sub("", clean), "compact"


def agent_of(message: str) -> str:
    """The agent phrase, i.e. everything before the marker.

    Deliberately NOT a fixed word count -- that is what collapsed three
    analysts into one label in the original filing, and what produced
    prefix-polluted buckets in this script's own first draft.
    """
    i = message.find(MARKER)
    return message[:i].strip() if i > 0 else "(no agent phrase)"


def main(rotated_only: bool) -> int:
    files = corpus(rotated_only)
    print("CORPUS (the glob, stated):")
    for p in files:
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size:,} bytes)")
    print(f"MATCH RULE: fixed substring {MARKER!r}; ANSI stripped and the "
          "CompactFormatter prefix removed before the agent phrase is taken.")
    print()

    total = 0
    by_agent: collections.Counter = collections.Counter()
    by_format: collections.Counter = collections.Counter()
    by_wording: collections.Counter = collections.Counter()
    for p in files:
        for raw in _lines(p):
            if MARKER not in raw:
                continue
            total += 1
            msg, fmt = message_of(raw)
            by_format[fmt] += 1
            by_agent[agent_of(msg)] += 1
            j = msg.find(MARKER)
            by_wording[msg[j:][:60]] += 1

    print(f"TOTAL matching LINES = {total}")
    print("  LINES, not events -- see the double-log table below.")
    print()
    print("BY FORMATTER (C-a):")
    for k, v in by_format.most_common():
        print(f"  {k:18s} {v:6d}   {100*v/total:5.1f}% of lines")
    print()
    print("BY AGENT (C-b -- full phrase, not a word count):")
    for k, v in by_agent.most_common(15):
        print(f"  {v:6d}  {k}")
    print()
    print("BY WORDING (C-c -- equal counts mean one event emitting two lines):")
    for k, v in by_wording.most_common(6):
        print(f"  {v:6d}  {k!r}")
    print()
    print("NO FAILURE RATE IS PRINTED (C-d): this corpus contains no")
    print("synthesis-attempt denominator, so any percentage here is a")
    print("composition share of LINES. Quoting one as a rate is unreproducible.")
    print()
    print("RAIL: not derivable from these logs -- no line carries a rail,")
    print("provider or model. Criterion 1's split must be era-bucketed on")
    print("paper_use_claude_code_route and labelled as such.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rotated-only", action="store_true",
                    help="exclude the live backend.log (reproduces the "
                         "research gate's corpus, which ended 2026-08-14)")
    args = ap.parse_args()
    raise SystemExit(main(rotated_only=args.rotated_only))
