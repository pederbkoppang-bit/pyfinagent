#!/usr/bin/env python3
"""Is the Layer-3 rail's StructuredOutput drop a TURN-BUDGET EXHAUSTION?

WHY THIS EXISTS
---------------
`scripts/qa/rail_drop_rate.py` measures the RATE of

    agent({schema}): subagent completed without calling StructuredOutput
    (after in-conversation nudge)

and used to say, in its own header, that the mechanism was UNPROVEN -- "size,
wall-clock, effort and the documented preamble-suppression trigger were each
tested and refuted", with the rate splitting by MODEL. (That header, and the
twin comment blocks in both workflow files, were superseded by phase-86.84 on
the strength of what this script measures. All four of those refutations
STAND; the cause is a fifth hypothesis none of them tested.)

This script tests that fifth hypothesis: the subagent runs out of TURNS. Every custom agent in `.claude/agents/*.md` carries a `maxTurns:`
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
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_SLUG = "-Users-ford--openclaw-workspace-pyfinagent"
RECORDS_GLOB = "*/workflows/wf_*.json"
REPO = Path(__file__).resolve().parents[2]

# The three run statuses actually present in the corpus. They are DISJOINT and
# are read from the record's named `status` field, never inferred from each
# other -- see the F4 note in collect().
FAILED_STATUS = "failed"        # the lost runs: the StructuredOutput drop
COMPLETED_STATUS = "completed"
KILLED_STATUS = "killed"        # "Workflow aborted" -- operator/runtime stop

# Cardinality floors. These have NO opt-out flag: a run of this script over an
# empty or truncated corpus must FAIL, not report a serene zero. A guard that
# can pass over nothing is not a guard (see the project's
# feedback_zero_assertion_guard_passes_vacuously).
MIN_AGENTS = 200
MIN_DROPS = 5
MIN_CAPPED_TYPES = 1
# phase-86.84 cycle-5 (Q/A finding QM2/QM7/QM8): once the boundary has been
# crossed, the re-measurement must rest on a non-trivial sample -- and a
# post-removal NON-EMITTER is the one signal verify() previously asserted
# nothing about, which is exactly the axis the 86.81 retry can mask.
MIN_POST_REMOVAL_SPAWNS = 10


def projects_root() -> Path:
    return Path(os.path.expanduser("~/.claude/projects")) / PROJECT_SLUG


# ── THE CAP IN FORCE IS A FUNCTION OF TIME, NOT OF TODAY'S FILE ─────────────
# Discovered the moment phase-86.84 removed the caps: this script read the LIVE
# frontmatter to explain HISTORICAL runs, so the remediation made its own
# verifier go red -- "no agent type carries a cap; nothing to test", plus 48
# drops reclassified as uncapped. The measurement was scoring yesterday's runs
# against today's file.
#
# Each run must be scored against the cap that was actually in force WHEN IT
# RAN. The corpus spans exactly one cap change, so the timeline is two entries
# per role. Extend it -- do not edit the historical numbers -- if a cap is ever
# reintroduced.
# V-7 (cycle-2 Q/A): the boundary is NOT the file-edit instant. The Agent-tool
# roster snapshots at session start, so a cap removed at 17:35Z is still IN FORCE
# for every spawn of the session that was already running. Using the edit instant
# would score those spawns as uncapped and, if one of them exhausted, the
# verifier would go red with "CLAIM BROKEN: a dropped spawn is not at its cap" --
# indicting the diagnosis when the real fault was the boundary. Fails loud, but
# misdiagnoses. So the boundary is the START OF THE SESSION THAT LAUNCHED THE RUN.
#
# F-E (cycle-3): the first attempt at that was a HARDCODED CALENDAR CONSTANT --
# `CAP_REMOVED_AT = "2026-08-15T00:00:00Z"`, standing in for "the first session
# after the edit". That is a PREDICTION about a future event, and it is wrong in
# BOTH directions:
#   * a spawn of the PRE-removal session running past midnight is still capped by
#     its roster snapshot but would score cap=None -- and a drop there reddens the
#     verifier against the DIAGNOSIS when the real fault is the boundary;
#   * a spawn of the POST-removal session before midnight is genuinely uncapped
#     but would score against the phase-59.1 pins -- so the uncensored sample the
#     removal exists to produce would be read back as censored evidence.
# The cap a spawn ran under is a property of ITS SESSION, not of the wall clock,
# and sessions overlap -- so no single instant can separate them. The boundary is
# therefore DERIVED FROM DISK, per run: a run is post-removal iff the SESSION
# DIRECTORY owning its run record was born after the edit. The edit instant is a
# commit that has already happened -- a historical fact, not a forecast -- which
# is the difference that matters.
CAP_EDIT_AT = "2026-08-14T17:37:50Z"  # commit 85127353; `git log -1 --format=%cI`
HISTORICAL_CAPS = {"qa": 30, "researcher": 40}  # set by phase-59.1

# phase-86.84 cycle-10: the ONE constant both the erased-attempt role
# classifier (collect) and its coupling pin (verify) read. A drift here moves
# both together, so the pin goes red against the workflow files; a drift in
# the workflow files reddens the pin directly. Single-sourced precisely so the
# cycle-9 silent-zero (classifier drifted, pin none the wiser) cannot recur.
ROLE_MARKERS = {
    "qa": ("IMMUTABLE SUCCESS CRITERIA", ".claude/workflows/qa-verdict.js"),
    "researcher": ("OBJECTIVE:", ".claude/workflows/research-gate.js"),
}

# A frontmatter cap that parses to something non-integral is neither "absent"
# nor a usable number. Treat it as a pin so the guard reddens rather than
# silently reporting the role uncapped.
SENTINEL_UNPARSEABLE_CAP = -1


def session_started_at(session_dir: Path) -> str | None:
    """When the session owning a run record started -- the directory's birth time.

    The Agent-tool roster, and therefore the maxTurns pin, is snapshotted at
    SESSION START. So this -- not the run's own timestamp -- is what decides which
    cap was in force for its spawns.

    Returns None when no birth time is available (a filesystem without
    st_birthtime, or an unreadable directory). Callers treat None as HISTORICAL,
    the conservative direction: it keeps the run inside the claim being tested
    rather than quietly excusing it from it.
    """
    try:
        birth = getattr(session_dir.stat(), "st_birthtime", None)
    except OSError:
        return None
    if birth is None:
        return None
    return datetime.datetime.fromtimestamp(
        birth, datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_is_post_removal(session_dir: Path) -> bool:
    """True iff this session's roster snapshot was taken after the caps were removed."""
    started = session_started_at(session_dir)
    return started is not None and started > CAP_EDIT_AT


def effective_cap(agent_type: str | None, post_removal: bool) -> int | None:
    """The maxTurns actually in force for a run, by the SESSION that launched it.

    For a pre-removal session the phase-59.1 pins applied. For a post-removal
    session the live frontmatter governs (and should be uncapped).
    """
    if not post_removal:
        return HISTORICAL_CAPS.get(agent_type or "")
    return parse_cap(agent_type)


def parse_cap(agent_type: str | None) -> int | None:
    """maxTurns from .claude/agents/<type>.md frontmatter as it stands NOW.

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

    # PARSE THE YAML; DO NOT PATTERN-MATCH THE LINE. The first version of this
    # used `^\s*maxTurns\s*:\s*(\d+)\s*$` and the phase-86.84 cycle-2 Q/A killed
    # it: `maxTurns: 30  # restored` is a LIVE integer pin that the regex misses,
    # so the remediation guard reported "all pins removed: True" over a restored
    # cap. That shape is not exotic -- every other line of these frontmatter
    # blocks is a `#` comment, so "restore the pin with a note" is the most
    # likely way it would come back. A quoted `"30"` slipped through too.
    # The loader reads YAML, so the guard must read YAML.
    try:
        import yaml  # noqa: PLC0415 -- optional dep, fall back below

        block = yaml.safe_load(m.group(1))
    except ImportError:
        block = None
    except yaml.YAMLError:
        # A frontmatter block that does not parse is a LOUD problem, not a
        # licence to report "uncapped". Fall through to the scan below.
        block = None

    if isinstance(block, dict):
        if "maxTurns" not in block:
            return None
        raw = block["maxTurns"]
        if raw is None:
            return None
        try:
            # Coerce quoted scalars: `maxTurns: "30"` is still a pin as far as
            # this guard is concerned. Over-detecting a pin is the safe
            # direction -- it can only make the check redder, never greener.
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return SENTINEL_UNPARSEABLE_CAP

    # FALLBACK PATH -- and it must not be quietly weaker than the YAML path.
    # phase-86.84 cycle-3 finding F-C: bare `python3` on this machine is
    # /usr/bin/python3, which has NO PyYAML, so the SHIPPED verification command
    # takes this branch. The first version matched a digit-shaped value, so
    # `!!int 30`, `&anchor 30`, `*alias` and `0x1e` all read as UNCAPPED -- live
    # pins the guard would have missed on the only path that actually runs. A
    # fix that does not execute under its own verification command is not a fix.
    #
    # So this branch does NOT try to interpret the value. ANY top-level
    # `maxTurns` key with a non-empty, non-null value is treated as A PIN. Over-
    # detection is the safe direction: it can only make the remediation check
    # redder, never greener, and the only cost of a false positive is a loud
    # failure a human then reads.
    for line in m.group(1).splitlines():
        head = line.split("#", 1)[0].rstrip()
        if not re.match(r"[ \t]*maxTurns[ \t]*:", head):
            continue
        if re.match(r"[ \t]+maxTurns", head):  # indented => nested, not a pin
            continue
        value = head.split(":", 1)[1].strip().strip("'\"")
        if value in ("", "null", "~", "None"):
            return None
        mm = re.search(r"(\d+)", value)
        return int(mm.group(1)) if mm else SENTINEL_UNPARSEABLE_CAP
    return None


def cap_parser_path() -> str:
    """Which branch parse_cap() will actually take, so it is never silent."""
    try:
        import yaml  # noqa: F401, PLC0415

        return "yaml"
    except ImportError:
        return "fallback"


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
        # F-E: scored by the SESSION's roster snapshot, not by the wall clock.
        post_removal = session_is_post_removal(session_dir)

        for entry in rec.get("workflowProgress") or []:
            if entry.get("type") != "workflow_agent":
                continue
            agent_id = entry.get("agentId")
            transcript = tdir / f"agent-{agent_id}.jsonl"
            if not transcript.is_file():
                missing_transcripts += 1
                continue
            turns, assistant_lines = count_turns(transcript)
            # phase-86.84 cycle-9 REPLACEMENT of a cycle-8 comment whose
            # premise measured FALSE. The old text said the 529 entries made
            # the command "permanently red (the corpus is append-only)". The
            # cycle-8 Q/A proved otherwise, and Main re-derived it: a
            # same-runId RE-DISPATCH REWRITES the run record (birth != mtime,
            # entries replaced), unlike the 86.81 in-script retry which
            # APPENDS (birth == mtime, both entries visible). So error-bearing
            # entries can VANISH from workflowProgress while their transcripts
            # remain on disk as orphans -- see the erased-transcript
            # accounting in collect()'s tail. The `errored` field still
            # matters for entries that DO carry an error (7 historical ones
            # exist), and the never-had-a-chance-to-emit exclusion below is
            # correct for them; it simply was not what cleared cycle 7's red.
            spawns.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "errored": bool(entry.get("error")),
                    "dropped": status == FAILED_STATUS,
                    # phase-86.84 cycle-1 Q/A, finding F4: there is a THIRD run
                    # status (`killed`, "Workflow aborted"). `not dropped` is
                    # not the same thing as `completed`, and bucketing killed
                    # runs into the ok* columns contaminated them. Carry the
                    # three states explicitly and never infer one from another.
                    "completed": status == COMPLETED_STATUS,
                    "killed": status == KILLED_STATUS,
                    "agent_type": entry.get("agentType"),
                    "cap": effective_cap(entry.get("agentType"), post_removal),
                    "post_removal": post_removal,
                    "session_started_at": session_started_at(session_dir),
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

    # phase-86.84 cycle-9: ERASED-ENTRY transcripts. A same-runId
    # re-dispatch REPLACES workflowProgress, so a prior attempt's transcript
    # can sit on disk with NO entry pointing at it -- invisible to every
    # entry-keyed population above (measured: the two 529-killed evaluator
    # attempts of this step, 38 and 10 turns, neither emitting
    # StructuredOutput). They are collected SEPARATELY: their role is
    # classified from the transcript's own first user message, their turns
    # counted the same way, and they are NEVER merged into the realised turn
    # distribution -- an attempt truncated by a server error or superseded by
    # a re-dispatch is right-censored by its death, and folding it in would
    # bias the uncensored sample the re-measurement exists to publish.
    erased: list[dict] = []
    for record_path in sorted(root.glob(RECORDS_GLOB)):
        try:
            rec = json.loads(record_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        run_id = rec.get("runId")
        session_dir = record_path.parent.parent
        tdir = session_dir / "subagents" / "workflows" / str(run_id)
        if not tdir.is_dir():
            continue
        consumed = {f"agent-{e.get('agentId')}.jsonl"
                    for e in (rec.get("workflowProgress") or [])
                    if e.get("type") == "workflow_agent"}
        for tr in tdir.glob("agent-*.jsonl"):
            if tr.name in consumed:
                continue
            turns, _lines = count_turns(tr)
            role = None
            try:
                with tr.open(encoding="utf-8", errors="replace") as fh:
                    for raw in fh:
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("type") != "user":
                            continue
                        c = (obj.get("message") or {}).get("content")
                        text = c if isinstance(c, str) else " ".join(
                            b.get("text", "") for b in (c or [])
                            if isinstance(b, dict))
                        for r_name, (marker, _src) in ROLE_MARKERS.items():
                            if marker in text:
                                role = r_name
                                break
                        break
            except OSError:
                pass
            erased.append({
                "run_id": run_id, "transcript": tr.name, "role": role,
                "turns": turns,
                "structured_output": called_structured_output(tr),
                "post_removal": session_is_post_removal(session_dir),
            })

    return {
        "records": records,
        "spawns": spawns,
        "missing_transcripts": missing_transcripts,
        "erased_transcripts": erased,
    }


def analyse(data: dict) -> dict:
    spawns = data["spawns"]
    by_type: dict[str | None, list[dict]] = defaultdict(list)
    for s in spawns:
        by_type[s["agent_type"]].append(s)

    rows = []
    for atype, group in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        # phase-86.84 cycle-4 (external audit 2026-08-17, finding D1): this row
        # previously took its cap from group[0] -- the alphabetically-first run
        # record on disk. The moment any POST-removal session sorts first, the
        # row reads uncapped while 300+ historical spawns in the same group ran
        # capped, and verify()'s capped_types floor then counts ZERO capped
        # roles -> exit 1 "no agent type carries a cap; nothing to test" over a
        # corpus holding 395 capped spawns. A sampled representative is not a
        # population property: score each spawn against ITS OWN era-correct cap
        # and report the group's caps as the SET actually observed.
        caps_present = sorted({s["cap"] for s in group if s["cap"] is not None})
        dropped = [s for s in group if s["dropped"]]
        # F4: `completed` is its own status, NOT "everything that did not drop".
        completed = [s for s in group if s["completed"]]
        at_cap_dropped = [
            s for s in dropped if s["cap"] is not None and s["turns"] == s["cap"]
        ]
        over_cap = [
            s for s in group if s["cap"] is not None and s["turns"] > s["cap"]
        ]
        rows.append(
            {
                "agent_type": atype,
                # Single historical cap -> that number (the common case, per
                # HISTORICAL_CAPS). No capped spawn ever -> None. Multiple
                # distinct caps -> the list, so a mixed history is visible
                # rather than averaged away.
                "cap": (
                    caps_present[0]
                    if len(caps_present) == 1
                    else (caps_present or None)
                ),
                "caps_present": caps_present,
                "capped_n": sum(1 for s in group if s["cap"] is not None),
                "uncapped_n": sum(1 for s in group if s["cap"] is None),
                # phase-86.84 cycle-6: an INDEPENDENT second derivation of the
                # uncapped-past-historical-cap count, computed here from the
                # by-type grouping so verify() can cross-check the
                # post_removal_turns report against a different code path. A
                # single-site inversion of either computation now produces a
                # visible disagreement instead of a silently wrong published
                # number (the cycle-5 Q/A proved past_old_cap was
                # reported-but-unguarded).
                "uncapped_past_hist_cap": sum(
                    1 for s in group
                    if s["cap"] is None
                    and s["turns"] > HISTORICAL_CAPS.get(str(atype), 10 ** 9)
                ),
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
                    1
                    for s in completed
                    if s["cap"] is not None and s["turns"] == s["cap"]
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

    killed_all = [s for s in spawns if s["killed"]]
    completed_all = [s for s in spawns if s["completed"]]

    controls = {
        # C1: the turn counter is not returning zeros.
        "c1_zero_turns_with_assistant_lines": sum(
            1 for s in spawns if s["turns"] == 0 and s["assistant_lines"] > 0
        ),
        "c1_positive_turn_counts": sum(1 for s in spawns if s["turns"] > 0),
        # C2: no spawn of a capped type ever exceeds its cap.
        "c2_capped_spawns_over_cap": sum(1 for s in capped if s["turns"] > s["cap"]),
        # Positive control on the drop detector. Denominator is the COMPLETED
        # population only (F4) -- killed runs are neither successes nor drops.
        "structured_output_called_completed": sum(
            1 for s in completed_all if s["structured_output"]
        ),
        "completed_total": len(completed_all),
        "structured_output_called_dropped": sum(
            1 for s in dropped_all if s["structured_output"]
        ),
        "dropped_total": len(dropped_all),
        # C3 NEGATIVE CONTROL (phase-86.84 cycle-1 Q/A, contributed by the
        # evaluator and adopted here). A `killed` run is a termination that is
        # NOT turn exhaustion, so its spawns should sit nowhere near a cap. If
        # killed spawns started landing at the cap, the "at cap" signal would be
        # measuring something generic about long runs rather than exhaustion.
        "c3_killed_spawns": len(killed_all),
        "c3_killed_turn_values": sorted(s["turns"] for s in killed_all),
        "c3_killed_at_cap": sum(
            1 for s in killed_all if s["cap"] is not None and s["turns"] == s["cap"]
        ),
    }

    # NOTE-A (cycle-1 Q/A): "0 drops in N uncapped" is inflated if most of that N
    # was never at risk. An uncapped spawn that finished in 9 turns could not
    # have exhausted a 30-turn cap, so it is not evidence about the cap. The
    # AT-RISK subset is the uncapped spawns that ran past the smallest live cap.
    live_caps = sorted({s["cap"] for s in capped if s["cap"] is not None})
    smallest_cap = live_caps[0] if live_caps else None
    uncapped_at_risk = (
        [s for s in uncapped if smallest_cap is not None and s["turns"] > smallest_cap]
    )

    # F5 (cycle-1 Q/A): run status is a PROXY. The mechanism is "sat at the cap
    # and never emitted the schema call", and that can happen inside a run that
    # ultimately COMPLETED, because the phase-86.81 retry can absorb it. Measure
    # the mechanism directly rather than through the run's outcome.
    at_cap = [s for s in capped if s["turns"] == s["cap"]]
    at_cap_non_emitters = [s for s in at_cap if not s["structured_output"]]

    claim = {
        "every_drop_is_at_its_cap": all(
            s["cap"] is not None and s["turns"] == s["cap"] for s in dropped_all
        ),
        "drops_on_uncapped_types": sum(1 for s in uncapped if s["dropped"]),
        "uncapped_spawns": len(uncapped),
        "uncapped_max_turns": max((s["turns"] for s in uncapped), default=0),
        "capped_spawns": len(capped),
        "smallest_live_cap": smallest_cap,
        "uncapped_at_risk": len(uncapped_at_risk),
        "uncapped_at_risk_drops": sum(1 for s in uncapped_at_risk if s["dropped"]),
        "capped_drop_rate_pct": (
            round(100.0 * len(dropped_all) / len(capped), 1) if capped else 0.0
        ),
        "at_cap_spawns": len(at_cap),
        "at_cap_non_emitters": len(at_cap_non_emitters),
        "at_cap_non_emitters_in_completed_runs": sum(
            1 for s in at_cap_non_emitters if s["completed"]
        ),
        "at_cap_non_emitter_runs": sorted(
            {s["run_id"] for s in at_cap_non_emitters if s["completed"]}
        ),
    }

    # THE REMEDIATION'S OWN CHECK. The diagnosis and the fix are verified by the
    # same command deliberately: a green run must mean both "the mechanism is
    # still what we said" and "the caps are still gone". If someone restores a
    # pin, this goes red without anyone having to remember why.
    # F-E: the boundary is OBSERVED, not declared. `cap_edit_at` is the commit
    # that removed the pins; `first_post_removal_run` is the earliest run record
    # on disk owned by a session that started after it -- i.e. the first spawn
    # that actually ran uncapped. Until one exists this is None, and None is the
    # honest reading: "no run on disk is past the boundary yet". A number here
    # would be a forecast dressed as a measurement.
    post_removal_spawns = [s for s in spawns if s.get("post_removal")]
    post_removal_ts = [s["timestamp"] for s in post_removal_spawns if s.get("timestamp")]

    # phase-86.84 cycle-4: THE COMMITTED RE-MEASUREMENT. The realised turn
    # distribution of post-removal (uncapped) spawns is the uncensored sample
    # the removal existed to produce -- report it per formerly-capped role so
    # the right-censoring argument is checked against lived data on every run,
    # not against a promise. Percentile = value at floor(frac*(n-1)) of the
    # sorted list; stated so the number cannot float free of its rule.
    def _q(sorted_vals: list[int], frac: float) -> int:
        if not sorted_vals:
            return 0
        return sorted_vals[int(frac * (len(sorted_vals) - 1))]

    post_removal_turns = []
    for role, old_cap in sorted(HISTORICAL_CAPS.items()):
        g = [s for s in post_removal_spawns if s["agent_type"] == role]
        turns = sorted(s["turns"] for s in g)
        erased_for_role = [e for e in data.get("erased_transcripts", [])
                           if e.get("role") == role and e.get("post_removal")]
        post_removal_turns.append(
            {
                "agent_type": role,
                # cycle-9: attempts whose entries a re-dispatch ERASED. On
                # disk, real, and invisible to every entry-keyed count --
                # REPORTED here so the trigger cannot lie by omission, and
                # deliberately NOT folded into the turn distribution (each is
                # right-censored by the death that orphaned it). No hard
                # floor: a re-dispatch is the caller's documented recovery
                # act and its loss is already recorded in the verdict ledger
                # as NO_VERDICT; a red here would fire on every legitimate
                # recovery. Advisory visibility, never silence.
                "erased_n": len(erased_for_role),
                "erased_non_emitters": sum(
                    1 for e in erased_for_role if not e["structured_output"]),
                "historical_cap": old_cap,
                "n": len(g),
                "dropped": sum(1 for s in g if s["dropped"]),
                # phase-86.84 cycle-6 (cycle-5 Q/A, Invalid_Precondition): a
                # spawn in a KILLED run -- an operator abort -- never had the
                # chance to emit StructuredOutput, and counting it here
                # re-committed the exact killed-vs-completed conflation this
                # file fixed as F4. The three run statuses stay explicit:
                # killed is NAMED, dropped is named, and non_emitters counts
                # only spawns that ran to completion WITHOUT emitting -- the
                # one shape that genuinely signals a new loss mechanism.
                "killed_n": sum(1 for s in g if s["killed"]),
                "errored_n": sum(1 for s in g if s.get("errored")),
                # The exclusion list now enumerates the WHOLE
                # never-had-a-chance-to-emit family -- dropped (run failed),
                # killed (operator abort), errored (server-side API error on
                # the agent entry itself) -- so a counted non-emitter means
                # exactly one thing: the agent ran to a natural completion and
                # still never emitted StructuredOutput. Closed by
                # construction, not by member (cycle-7 Q/A: the killed fix was
                # instance-not-class).
                "non_emitters": sum(
                    1 for s in g
                    if not s["structured_output"]
                    and not s["killed"] and not s["dropped"]
                    and not s.get("errored")
                ),
                "median_turns": _q(turns, 0.5),
                "p90_turns": _q(turns, 0.9),
                "max_turns": turns[-1] if turns else 0,
                "past_old_cap": sum(1 for t in turns if t > old_cap),
            }
        )

    remediation = {
        "erased_transcripts": data.get("erased_transcripts", []),
        # cycle-10 (cycle-9 Q/A): orphans whose role could not be classified
        # were silently dropped from every per-role row. They are aggregated
        # here so the channel is never invisible -- 41 of 44 current orphans
        # are pre-removal role=None bulk from two old runs, and that number is
        # a fact about the corpus, not about the formerly-capped roles.
        "erased_unclassified": sum(
            1 for e in data.get("erased_transcripts", [])
            if e.get("role") is None),
        "erased_unclassified_post_removal": sum(
            1 for e in data.get("erased_transcripts", [])
            if e.get("role") is None and e.get("post_removal")),
        "cap_edit_at": CAP_EDIT_AT,
        "first_post_removal_run": min(post_removal_ts) if post_removal_ts else None,
        "post_removal_spawns": len(post_removal_spawns),
        "post_removal_turns": post_removal_turns,
        "live_caps": {r: parse_cap(r) for r in sorted(HISTORICAL_CAPS)},
        "historical_caps": dict(HISTORICAL_CAPS),
        "all_pins_removed": all(parse_cap(r) is None for r in HISTORICAL_CAPS),
        # F-C: which branch parse_cap actually took. Never leave this implicit --
        # the shipped command resolves `python3` to an interpreter without
        # PyYAML, and a guard whose strength depends on an undisclosed
        # interpreter is a guard nobody can audit.
        "cap_parser": cap_parser_path(),
    }

    return {
        "records": data["records"],
        "spawns": len(spawns),
        "missing_transcripts": data["missing_transcripts"],
        "remediation": remediation,
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
    w(
        f"  C3 negative control       : {c['c3_killed_spawns']} spawns in `killed` runs "
        f"sit at turns {c['c3_killed_turn_values']}; {c['c3_killed_at_cap']} at a cap "
        f"(must be 0 -- a non-exhaustion stop should land nowhere near one)"
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
    w(
        f"  ... of which AT RISK (>{cl['smallest_live_cap']} turns)        : "
        f"{cl['uncapped_at_risk_drops']}/{cl['uncapped_at_risk']}  "
        f"vs a {cl['capped_drop_rate_pct']}% drop rate on capped spawns"
    )
    w("     ^ QUOTE THIS RATIO, NOT THE RAW UNCAPPED TOTAL. A spawn that")
    w("       finished well under the smallest cap could never have exhausted")
    w("       it, so it is not evidence about the cap.")
    w("")
    w(
        f"  spawns sitting AT a cap                    : {cl['at_cap_spawns']}, of which "
        f"{cl['at_cap_non_emitters']} never emitted StructuredOutput"
    )
    w(
        f"  ... inside runs that COMPLETED anyway      : "
        f"{cl['at_cap_non_emitters_in_completed_runs']} "
        f"{cl['at_cap_non_emitter_runs']}"
    )
    w("     ^ run status is a PROXY. These are exhaustions the 86.81 retry")
    w("       absorbed, so the true at-cap non-emitter population is larger")
    w("       than the failed-run count. They strengthen the mechanism.")
    w("")
    r = a["remediation"]
    w("REMEDIATION (phase-86.84) -- checked by the same command as the diagnosis")
    w(f"  caps removed at : {r['cap_edit_at']}  (commit 85127353)")
    if r["first_post_removal_run"] is None:
        w("  first uncapped  : NONE ON DISK YET -- no session started after the")
        w("                    edit has produced a run record, so every run below")
        w("                    is scored against the phase-59.1 pins. The realised")
        w("                    uncapped turn distribution is NOT YET MEASURABLE.")
    else:
        w(f"  first uncapped  : {r['first_post_removal_run']}  "
          f"({r['post_removal_spawns']} spawn(s) past the boundary)")
        w("  REALISED UNCAPPED TURN DISTRIBUTION (the committed re-measurement --")
        w("  the uncensored sample; percentile rule: sorted[int(frac*(n-1))]):")
        for pr in r.get("post_removal_turns", []):
            w(
                f"    {pr['agent_type']:<12} n={pr['n']:>3}  dropped={pr['dropped']}  "
                f"non-emitters={pr['non_emitters']}  killed={pr.get('killed_n', 0)}  "
                f"errored={pr.get('errored_n', 0)}  "
                f"erased={pr.get('erased_n', 0)}"
                f"(non-emit {pr.get('erased_non_emitters', 0)})  "
                f"p50={pr['median_turns']}  p90={pr['p90_turns']}  "
                f"max={pr['max_turns']}  "
                f">old-cap({pr['historical_cap']})={pr['past_old_cap']}"
            )
        w(f"    unclassified orphans: {r.get('erased_unclassified', 0)} "
          f"(post-removal {r.get('erased_unclassified_post_removal', 0)}) -- "
          "role=None erased attempts, visible in the DEFAULT report, not only --json")
    w("     ^ DERIVED FROM DISK, per run, from the birth time of the session")
    w("       directory owning the run record -- never a calendar constant. The")
    w("       roster snapshots at SESSION START, and sessions overlap, so no")
    w("       single instant separates capped from uncapped runs.")
    w(f"  in force before : {r['historical_caps']}  (phase-59.1 pins)")
    w(f"  live now        : {r['live_caps']}")
    w(f"  all pins removed: {r['all_pins_removed']}  (must be True)")
    w(f"  cap parser used : {r['cap_parser']}  "
      f"({'PyYAML' if r['cap_parser'] == 'yaml' else 'no PyYAML on this interpreter'})")
    w("    Both paths detect every pin shape probed (bare, trailing comment,")
    w("    quoted, !!int, anchor, hex, float, tab, zero). The fallback does not")
    w("    interpret the value: any top-level maxTurns key with a non-null value")
    w("    is a pin. Over-detection is the safe direction.")
    w("  Runs are scored against the cap in force WHEN THEY RAN, not against")
    w("  today's file -- otherwise removing the caps would erase the evidence")
    w("  for removing them.")
    w("")
    w("NOT ESTABLISHED: that the cap is the whole cause; and the capped roles'")
    w("turn distribution is RIGHT-CENSORED at the cap, so no percentile in this")
    w("output can size a replacement cap. See the module docstring.")
    return "\n".join(out)


def verify(a: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    c, cl = a["controls"], a["claim"]

    r = a["remediation"]
    if not r["all_pins_removed"]:
        still = {k: v for k, v in r["live_caps"].items() if v is not None}
        problems.append(
            f"REMEDIATION REVERTED: a maxTurns pin is live again on {still}. "
            "phase-86.84 removed these deliberately -- see the rationale block in "
            ".claude/agents/qa.md. Raising a cap is not a fix; the distribution is "
            "right-censored at whatever the cap is."
        )

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
    if c["c3_killed_at_cap"]:
        problems.append(
            f"C3 FAILED: {c['c3_killed_at_cap']} spawns in `killed` runs sit exactly at "
            "a cap -- 'at cap' may be measuring something generic about long runs "
            "rather than turn exhaustion"
        )
    if cl["uncapped_at_risk"] < 10:
        problems.append(
            f"cardinality floor: only {cl['uncapped_at_risk']} uncapped spawns ran past "
            f"the smallest cap ({cl['smallest_live_cap']}) -- the uncapped comparison "
            "has too few AT-RISK cases to carry weight, whatever the raw total says"
        )
    if cl["uncapped_at_risk_drops"]:
        problems.append(
            f"CLAIM BROKEN: {cl['uncapped_at_risk_drops']} at-risk uncapped spawns "
            "dropped"
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

    # ── The orphan classifier's coupling is PINNED (phase-86.84 cycle-10) ──
    # The erased-attempt role classifier matches two prompt literals emitted
    # by the workflow launch scripts. Nothing pinned that coupling, so a
    # renamed prompt header would silently zero the erased counter (cycle-9
    # Q/A, executed: marker drifted one word -> erased qa=(0,0), still green).
    # verify() now asserts the literals still appear in the files that emit
    # them; a red here means the classifier must be retargeted, not that the
    # rail regressed.
    # cycle-11 (cycle-10 Q/A, F1): "in file bytes" was satisfiable by a //
    # comment retaining the retired literal while the emitted header drifted
    # (executed MUT-A survivor) -- and these files ALREADY quote retired
    # literals in comments because criterion 5 mandates quote-then-replace
    # (qa-verdict.js:106/:609/:628, research-gate.js:880). The pin now
    # requires the literal on a NON-COMMENT line (first non-whitespace not
    # //, * or /*). Residual, stated rather than hidden: a literal in a
    # TRAILING same-line comment after code still passes -- narrower than
    # the bytes check, not airtight against deliberate evasion; matrix cell
    # S16 drives the exact MUT-A evasion and must stay red.
    for _role, (marker, wf_file) in ROLE_MARKERS.items():
        wf_path = REPO / wf_file
        try:
            wf_src = wf_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            wf_src = ""
        non_comment = [
            ln for ln in wf_src.splitlines()
            if not ln.lstrip().startswith(("//", "*", "/*"))
        ]
        if not any(marker in ln for ln in non_comment):
            problems.append(
                f"ORPHAN-CLASSIFIER COUPLING BROKEN: {wf_file} no longer "
                f"contains the prompt literal {marker!r} on any NON-COMMENT "
                "line (a comment quoting the retired text does not emit it) "
                "-- retarget the classifier in collect() or the erased "
                "counter will silently read zero"
            )

    # ── The unclassified-orphan aggregates are CROSS-CHECKED (cycle-11) ────
    # cycle-10 Q/A, F2: erased_unclassified / _post_removal were computed
    # into the remediation dict and referenced NOWHERE else -- not rendered,
    # not verified, no cell; hardcoding either to 0 was invisible to every
    # mode (executed MUT-C survivor). verify() now recomputes both from the
    # erased_transcripts actually collected, and render() prints them in the
    # default report.
    _rem = a["remediation"]
    _et = _rem.get("erased_transcripts", [])
    _want_total = sum(1 for e in _et if e.get("role") is None)
    _want_post = sum(1 for e in _et
                     if e.get("role") is None and e.get("post_removal"))
    if _rem.get("erased_unclassified") != _want_total:
        problems.append(
            f"UNCLASSIFIED-ORPHAN AGGREGATE INCONSISTENT: stored "
            f"{_rem.get('erased_unclassified')!r} != recount {_want_total} "
            "from erased_transcripts -- the role=None channel is misreported"
        )
    if _rem.get("erased_unclassified_post_removal") != _want_post:
        problems.append(
            f"UNCLASSIFIED-ORPHAN POST-REMOVAL AGGREGATE INCONSISTENT: stored "
            f"{_rem.get('erased_unclassified_post_removal')!r} != recount "
            f"{_want_post} from erased_transcripts"
        )

    # ── The re-measurement itself is guarded (phase-86.84 cycle-5) ──────────
    # The cycle-4 Q/A proved by mutation that verify() asserted NOTHING over
    # remediation.post_removal_turns: zeroed percentiles, a vanished role block
    # and a non-emitter flood all left this function green (QM1/QM2/QM7/QM8).
    # The non-emitter axis is exactly the one the 86.81 retry can mask, so a
    # re-measurement without a floor is an illusory guard.
    if r["first_post_removal_run"] is not None:
        prt = r.get("post_removal_turns") or []
        if not prt:
            problems.append(
                "RE-MEASUREMENT MISSING: the boundary has been crossed but "
                "post_removal_turns is empty -- the committed uncensored sample "
                "is not being produced"
            )
        total_post = sum(row["n"] for row in prt)
        if prt and total_post < MIN_POST_REMOVAL_SPAWNS:
            problems.append(
                f"cardinality floor: only {total_post} post-removal qa/researcher "
                f"spawns < {MIN_POST_REMOVAL_SPAWNS} -- refusing to certify the "
                "re-measurement on a truncated sample (or a broken role filter)"
            )
        for row in prt:
            if row["n"] <= 0:
                continue
            if row["non_emitters"]:
                problems.append(
                    f"POST-REMOVAL NON-EMITTER: {row['non_emitters']} uncapped "
                    f"{row['agent_type']} spawn(s) never emitted StructuredOutput. "
                    "The cap was the proven mechanism and it is gone, so this is a "
                    "NEW loss mechanism (or the 86.81 retry absorbing losses "
                    "again). Revisit the diagnosis; do NOT re-pin a cap."
                )
            if row["dropped"]:
                problems.append(
                    f"POST-REMOVAL DROP: {row['dropped']} uncapped "
                    f"{row['agent_type']} run(s) failed -- same revisit rule"
                )
            if row["median_turns"] < 1:
                problems.append(
                    f"re-measurement broken for {row['agent_type']}: p50 "
                    f"{row['median_turns']} with n={row['n']} -- a real spawn "
                    "takes at least one turn, so the percentile computation is "
                    "not reading real data"
                )
            if not (row["median_turns"] <= row["p90_turns"] <= row["max_turns"]):
                problems.append(
                    f"re-measurement broken for {row['agent_type']}: p50 "
                    f"{row['median_turns']} / p90 {row['p90_turns']} / max "
                    f"{row['max_turns']} are not monotone"
                )
            # phase-86.84 cycle-6: the report must AGREE with the independent
            # by-type derivation. For qa/researcher, "uncapped" and
            # "post-removal" are the same population (a pre-removal spawn of a
            # formerly-capped role always carries its historical cap), so a
            # role-filter break, a truncated sample, or an emptied set on
            # either side produces a visible disagreement here.
            brow = next((r2 for r2 in a["by_agent_type"]
                         if r2["agent_type"] == row["agent_type"]), None)
            if brow is not None:
                if row["n"] != brow["uncapped_n"]:
                    problems.append(
                        f"re-measurement disagrees with the by-type derivation "
                        f"for {row['agent_type']}: post_removal n={row['n']} vs "
                        f"uncapped_n={brow['uncapped_n']} -- one of the two "
                        "computations is broken (or a cap was re-pinned)"
                    )
                if row["past_old_cap"] != brow["uncapped_past_hist_cap"]:
                    problems.append(
                        f"past_old_cap disagrees with its independent "
                        f"derivation for {row['agent_type']}: "
                        f"{row['past_old_cap']} vs "
                        f"{brow['uncapped_past_hist_cap']} -- the published "
                        "number no longer reflects the computation"
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
