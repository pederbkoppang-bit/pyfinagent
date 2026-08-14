#!/usr/bin/env python3
"""Is the Layer-3 rail's StructuredOutput drop a TURN-BUDGET EXHAUSTION?

WHY THIS EXISTS
---------------
`scripts/qa/rail_drop_rate.py` measures the RATE of

    agent({schema}): subagent completed without calling StructuredOutput
    (after in-conversation nudge)

and says, in its own header, that the mechanism is UNPROVEN -- "size,
wall-clock, effort and the documented preamble-suppression trigger were each
tested and refuted", with the rate splitting by MODEL.

This script tests a hypothesis those probes did not: the subagent runs out of
TURNS. Every custom agent in `.claude/agents/*.md` carries a `maxTurns:`
frontmatter key. The built-in agent types (`general-purpose`, `Explore`) and the
default workflow subagent carry none.

THE MEASUREMENT (2026-08-14, 572 run records / 1325 agent spawns on disk):

    agentType         frontmatter   dropped   turns on a dropped spawn
    qa                maxTurns 30    39/302   30, 30, 30 ... (all 39)
    researcher        maxTurns 40     9/93    40, 40, 40 ... (all 9)
    general-purpose   (none)          0/252   -- reaches 63 turns, never drops
    Explore           (none)          0/263   -- reaches 56 turns, never drops

Not "near the cap". AT the cap, every time, with no exceptions in either role.
The agent spends its last turn on ordinary work, the runtime has no turn left in
which the schema call could be emitted, and the run dies with tokens spent and
nothing returned.

THE MODEL SPLIT IS CONFOUNDED, AND THIS SCRIPT SHOWS THAT TOO. Holding the model
constant at claude-opus-5[1m]: qa 39/290, researcher 8/89, general-purpose 0/19,
Explore 0/109, default workflow subagent 0/289. The clean 0.0% previously
attributed to claude-opus-4-8[1m] is an artefact of WHAT THAT MODEL RAN --
223 of its 258 spawns were uncapped `general-purpose`. Model and agentType are
near-collinear in this corpus, so a model-keyed reading of the same data reports
a model effect. Keyed on agentType the separation is total.

THIS IS A RECURRENCE, NOT A NEW DEFECT. `.claude/agents/qa.md` records
"maxTurns 30 (phase-59.1): the old 12 cap caused mid-evaluation stalls (20-26
tool-uses per evaluation); 30 gives headroom", and `researcher.md` records
"maxTurns 40 (phase-59.1): complex briefs hit the old 30 cap mid-write". The
same failure was diagnosed once and answered by raising the cap to a number that
the workload has since outgrown. So a fix that only raises the number again is
on a clock -- see WHAT THIS SCRIPT DOES NOT ESTABLISH.

    python3 scripts/qa/rail_turn_cap.py            # report
    python3 scripts/qa/rail_turn_cap.py --json
    python3 scripts/qa/rail_turn_cap.py --verify   # exit 1 if the claim breaks

POPULATION RULE -- stated here so no ratio in the output floats free.
  Universe   : every `*/workflows/wf_*.json` run record under this project's
               ~/.claude/projects/ tree, and for each, every entry in
               `workflowProgress` with `type == "workflow_agent"`.
  Turn count : distinct `requestId` over `type == "assistant"` lines of that
               agent's `subagents/workflows/<runId>/agent-<agentId>.jsonl`.
               A requestId is one API round-trip, which is what a turn is.
               Agents whose transcript is missing are EXCLUDED and counted
               separately -- they are never silently treated as zero.
  Dropped    : the RUN's `status == "failed"`. Read from the named `status`
               field, never by scanning the record: a run record embeds the
               dispatched workflow SOURCE, and both workflow files quote the
               drop string in comments, so any blob scan matches itself. That
               trap already produced 38 phantom drops out of 81 once
               (commit f88f8190) and this script does not reopen it.
  Cap        : `maxTurns:` parsed from the YAML frontmatter of
               `.claude/agents/<agentType>.md`. Absent file or absent key =>
               no cap, reported as such and never defaulted to a number.

TWO CONTROLS, BECAUSE A CLEAN SEPARATION IS EXACTLY WHAT A BROKEN PROBE ALSO
PRODUCES.
  C1 (turn counter is not vacuous): the counter must return a POSITIVE turn
     count for a majority of transcripts and must never return 0 for a
     transcript that contains an assistant line. A counter that returned 0
     everywhere would make "every drop is at the cap" false, not vacuously
     true -- but it would make the SUCCESS side look artificially far from the
     cap, so it is checked in the direction that can actually mislead.
  C2 (the cap is a real ceiling, not a coincidence of the counter): NO spawn of
     a capped agentType -- dropped or completed -- may exceed its cap. If the
     counter were inflating turns, successful spawns would breach the cap too.
     This control is derived from the completed population, which the
     hypothesis says nothing about, so it is not built from the pattern it
     tests.

WHAT THIS SCRIPT DOES NOT ESTABLISH, AND THE HONEST LIMITS
  - It does NOT prove the cap is the whole cause. It proves exhaustion is
    NECESSARY on every observed drop (48/48) and that no uncapped spawn has
    ever dropped in 515 tries. It cannot rule out a second mechanism that only
    fires at the cap.
  - The turn distribution of capped roles is RIGHT-CENSORED at the cap. The
    observed p50 (qa 18) is a censored median, so "what cap would be enough" is
    NOT answerable from this data alone -- the tail beyond 30 was never
    observed, only truncated. Any new number chosen from these percentiles
    inherits that censoring, which is how phase-59.1's 30 was chosen and why it
    recurred. Sizing needs an uncensored sample (raise the cap, re-measure) or
    a mechanism that does not depend on guessing the tail.
  - `logs` is empty on dropped runs (phase-86.81, finding I-2), so this script
    cannot see how many retry attempts a lost run consumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_SLUG = "-Users-ford--openclaw-workspace-pyfinagent"
RECORDS_GLOB = "*/workflows/wf_*.json"
REPO = Path(__file__).resolve().parents[2]

# A run whose status is one of these is a lost run for our purposes.
FAILED_STATUS = "failed"

# Cardinality floors. These have NO opt-out flag: a run of this script over an
# empty or truncated corpus must FAIL, not report a serene zero. A guard that
# can pass over nothing is not a guard (see the project's
# feedback_zero_assertion_guard_passes_vacuously).
MIN_AGENTS = 200
MIN_DROPS = 5
MIN_CAPPED_TYPES = 1


def projects_root() -> Path:
    return Path(os.path.expanduser("~/.claude/projects")) / PROJECT_SLUG


def parse_cap(agent_type: str | None) -> int | None:
    """maxTurns from .claude/agents/<type>.md frontmatter, or None if uncapped.

    Only the frontmatter block (between the first two `---` lines) is scanned,
    so a `maxTurns` mentioned in the body prose cannot be mistaken for the pin.
    qa.md talks about maxTurns in its "Verification budget" bullet; that line
    must not be read as a setting.
    """
    if not agent_type:
        return None
    path = REPO / ".claude" / "agents" / f"{agent_type}.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*maxTurns\s*:\s*(\d+)\s*$", line)
        if mm:
            return int(mm.group(1))
    return None


def count_turns(transcript: Path) -> tuple[int, int]:
    """(distinct requestIds, assistant lines seen) for one agent transcript."""
    seen: set[str] = set()
    assistant_lines = 0
    with transcript.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "assistant":
                continue
            assistant_lines += 1
            rid = obj.get("requestId")
            if rid:
                seen.add(rid)
    return len(seen), assistant_lines


def called_structured_output(transcript: Path) -> bool:
    """True iff a tool_use block named StructuredOutput was ever emitted.

    Reads the tool_use NAME, not the raw text: the string appears in prompts and
    in grep output, so a substring scan over the file would over-count.
    """
    with transcript.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            content = obj.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "StructuredOutput"
                ):
                    return True
    return False


def collect() -> dict:
    root = projects_root()
    spawns: list[dict] = []
    missing_transcripts = 0
    records = 0

    for record_path in sorted(root.glob(RECORDS_GLOB)):
        try:
            rec = json.loads(record_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        records += 1
        run_id = rec.get("runId")
        status = rec.get("status")
        session_dir = record_path.parent.parent
        tdir = session_dir / "subagents" / "workflows" / str(run_id)

        for entry in rec.get("workflowProgress") or []:
            if entry.get("type") != "workflow_agent":
                continue
            agent_id = entry.get("agentId")
            transcript = tdir / f"agent-{agent_id}.jsonl"
            if not transcript.is_file():
                missing_transcripts += 1
                continue
            turns, assistant_lines = count_turns(transcript)
            spawns.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "dropped": status == FAILED_STATUS,
                    "agent_type": entry.get("agentType"),
                    "cap": parse_cap(entry.get("agentType")),
                    "model": entry.get("model"),
                    "turns": turns,
                    "assistant_lines": assistant_lines,
                    "tool_calls": entry.get("toolCalls"),
                    "tokens": entry.get("tokens"),
                    "structured_output": called_structured_output(transcript),
                    "timestamp": rec.get("timestamp"),
                    "workflow": rec.get("workflowName"),
                }
            )

    return {
        "records": records,
        "spawns": spawns,
        "missing_transcripts": missing_transcripts,
    }


def analyse(data: dict) -> dict:
    spawns = data["spawns"]
    by_type: dict[str | None, list[dict]] = defaultdict(list)
    for s in spawns:
        by_type[s["agent_type"]].append(s)

    rows = []
    for atype, group in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        cap = group[0]["cap"]
        dropped = [s for s in group if s["dropped"]]
        completed = [s for s in group if not s["dropped"]]
        at_cap_dropped = [s for s in dropped if cap is not None and s["turns"] == cap]
        over_cap = [s for s in group if cap is not None and s["turns"] > cap]
        rows.append(
            {
                "agent_type": atype,
                "cap": cap,
                "n": len(group),
                "dropped": len(dropped),
                "dropped_exactly_at_cap": len(at_cap_dropped),
                "dropped_turn_values": sorted({s["turns"] for s in dropped}),
                "any_spawn_over_cap": len(over_cap),
                "completed_max_turns": max((s["turns"] for s in completed), default=0),
                "completed_median_turns": (
                    sorted(s["turns"] for s in completed)[len(completed) // 2]
                    if completed
                    else 0
                ),
                "completed_at_cap": sum(
                    1 for s in completed if cap is not None and s["turns"] == cap
                ),
            }
        )

    # Model x agentType, to show the confound rather than assert it.
    cross: dict[str, dict[str, str]] = defaultdict(dict)
    for s in spawns:
        key = str(s["model"])
        cell = cross[key].setdefault(str(s["agent_type"]), [0, 0])
        cell[0] += 1
        if s["dropped"]:
            cell[1] += 1

    dropped_all = [s for s in spawns if s["dropped"]]
    capped = [s for s in spawns if s["cap"] is not None]
    uncapped = [s for s in spawns if s["cap"] is None]

    controls = {
        # C1: the turn counter is not returning zeros.
        "c1_zero_turns_with_assistant_lines": sum(
            1 for s in spawns if s["turns"] == 0 and s["assistant_lines"] > 0
        ),
        "c1_positive_turn_counts": sum(1 for s in spawns if s["turns"] > 0),
        # C2: no spawn of a capped type ever exceeds its cap.
        "c2_capped_spawns_over_cap": sum(1 for s in capped if s["turns"] > s["cap"]),
        # Positive control on the drop detector, from the completed population.
        "structured_output_called_completed": sum(
            1 for s in spawns if not s["dropped"] and s["structured_output"]
        ),
        "completed_total": sum(1 for s in spawns if not s["dropped"]),
        "structured_output_called_dropped": sum(
            1 for s in dropped_all if s["structured_output"]
        ),
        "dropped_total": len(dropped_all),
    }

    claim = {
        "every_drop_is_at_its_cap": all(
            s["cap"] is not None and s["turns"] == s["cap"] for s in dropped_all
        ),
        "drops_on_uncapped_types": sum(1 for s in uncapped if s["dropped"]),
        "uncapped_spawns": len(uncapped),
        "uncapped_max_turns": max((s["turns"] for s in uncapped), default=0),
        "capped_spawns": len(capped),
    }

    return {
        "records": data["records"],
        "spawns": len(spawns),
        "missing_transcripts": data["missing_transcripts"],
        "by_agent_type": rows,
        "model_x_agent_type": {k: dict(v) for k, v in cross.items()},
        "controls": controls,
        "claim": claim,
    }


def render(a: dict) -> str:
    out: list[str] = []
    w = out.append
    w("Layer-3 rail: is the StructuredOutput drop a TURN-BUDGET EXHAUSTION?")
    w("=" * 78)
    w(f"run records read      : {a['records']}")
    w(f"agent spawns analysed : {a['spawns']}")
    w(f"transcripts missing   : {a['missing_transcripts']}  (excluded, not zeroed)")
    w("")
    w("POPULATION RULE: one row per workflowProgress entry of type")
    w("workflow_agent; turns = distinct requestId over assistant lines of that")
    w("agent's transcript; dropped = the RUN's named `status` field == failed;")
    w("cap = maxTurns in .claude/agents/<type>.md frontmatter only.")
    w("")
    w(
        f"  {'agentType':<18}{'cap':>5}{'n':>6}{'drop':>6}"
        f"{'@cap':>6}{'>cap':>6}{'ok p50':>8}{'ok max':>8}{'ok@cap':>8}"
    )
    for r in a["by_agent_type"]:
        cap = "-" if r["cap"] is None else str(r["cap"])
        w(
            f"  {str(r['agent_type']):<18}{cap:>5}{r['n']:>6}{r['dropped']:>6}"
            f"{r['dropped_exactly_at_cap']:>6}{r['any_spawn_over_cap']:>6}"
            f"{r['completed_median_turns']:>8}{r['completed_max_turns']:>8}"
            f"{r['completed_at_cap']:>8}"
        )
    w("")
    w("  @cap  = dropped spawns whose turn count EQUALS the cap exactly")
    w("  >cap  = spawns of any outcome exceeding the cap (must be 0: control C2)")
    w("  ok*   = the completed population, which the hypothesis does not predict")
    w("")
    w("Turn counts observed on dropped spawns, per role:")
    for r in a["by_agent_type"]:
        if r["dropped"]:
            w(
                f"  {str(r['agent_type']):<18} cap={r['cap']}  "
                f"observed={r['dropped_turn_values']}"
            )
    w("")
    w("MODEL x agentType -- dropped/n, showing the confound explicitly:")
    for model, cells in sorted(a["model_x_agent_type"].items()):
        parts = ", ".join(
            f"{t}={c[1]}/{c[0]}" for t, c in sorted(cells.items()) if c[0]
        )
        w(f"  {model:<26} {parts}")
    w("")
    c = a["controls"]
    w("CONTROLS")
    w(
        f"  C1 turn counter alive     : {c['c1_positive_turn_counts']} spawns with "
        f"turns>0; {c['c1_zero_turns_with_assistant_lines']} zero-with-assistant-lines "
        f"(must be 0)"
    )
    w(
        f"  C2 cap is a real ceiling  : {c['c2_capped_spawns_over_cap']} capped spawns "
        f"exceed their cap (must be 0)"
    )
    w(
        f"  detector positive control : StructuredOutput emitted by "
        f"{c['structured_output_called_completed']}/{c['completed_total']} completed "
        f"spawns vs {c['structured_output_called_dropped']}/{c['dropped_total']} dropped"
    )
    w("")
    cl = a["claim"]
    w("CLAIM")
    w(f"  every dropped spawn sits EXACTLY at its cap : {cl['every_drop_is_at_its_cap']}")
    w(
        f"  drops among UNCAPPED agent types           : "
        f"{cl['drops_on_uncapped_types']}/{cl['uncapped_spawns']} "
        f"(uncapped spawns reach {cl['uncapped_max_turns']} turns)"
    )
    w("")
    w("NOT ESTABLISHED: that the cap is the whole cause; and the capped roles'")
    w("turn distribution is RIGHT-CENSORED at the cap, so no percentile in this")
    w("output can size a replacement cap. See the module docstring.")
    return "\n".join(out)


def verify(a: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    c, cl = a["controls"], a["claim"]

    if a["spawns"] < MIN_AGENTS:
        problems.append(
            f"cardinality floor: {a['spawns']} spawns < {MIN_AGENTS} required "
            "-- refusing to report on a truncated corpus"
        )
    if cl["capped_spawns"] < MIN_AGENTS // 4:
        problems.append(
            f"cardinality floor: only {cl['capped_spawns']} capped spawns"
        )
    if c["dropped_total"] < MIN_DROPS:
        problems.append(
            f"cardinality floor: {c['dropped_total']} drops < {MIN_DROPS} "
            "-- with no drops in the corpus this check would pass vacuously"
        )
    capped_types = sum(1 for r in a["by_agent_type"] if r["cap"] is not None)
    if capped_types < MIN_CAPPED_TYPES:
        problems.append("no agent type carries a maxTurns cap; nothing to test")

    if c["c1_zero_turns_with_assistant_lines"]:
        problems.append(
            f"C1 FAILED: {c['c1_zero_turns_with_assistant_lines']} transcripts have "
            "assistant lines but zero counted turns -- the turn counter is broken"
        )
    if c["c1_positive_turn_counts"] < a["spawns"] // 2:
        problems.append("C1 FAILED: fewer than half of spawns have a positive turn count")
    if c["c2_capped_spawns_over_cap"]:
        problems.append(
            f"C2 FAILED: {c['c2_capped_spawns_over_cap']} capped spawns exceed their "
            "cap -- the cap is not the ceiling this analysis assumes"
        )
    if c["structured_output_called_completed"] <= c["structured_output_called_dropped"]:
        problems.append(
            "detector control FAILED: completed spawns do not emit StructuredOutput "
            "more often than dropped ones -- the detector does not discriminate"
        )

    if not cl["every_drop_is_at_its_cap"]:
        problems.append(
            "CLAIM BROKEN: a dropped spawn is not at its cap -- turn exhaustion is no "
            "longer necessary on every drop, and the diagnosis must be revisited"
        )
    if cl["drops_on_uncapped_types"]:
        problems.append(
            f"CLAIM BROKEN: {cl['drops_on_uncapped_types']} uncapped spawns dropped -- "
            "the cap cannot be the only mechanism"
        )
    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="exit 1 if a control or the turn-exhaustion claim breaks",
    )
    args = ap.parse_args()

    analysis = analyse(collect())

    if args.json:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print(render(analysis))

    if args.verify:
        ok, problems = verify(analysis)
        print()
        if ok:
            print("VERIFY: PASS -- controls green, turn-exhaustion claim holds.")
            return 0
        print("VERIFY: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
