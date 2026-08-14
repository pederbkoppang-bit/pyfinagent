#!/usr/bin/env python3
"""phase-86.31 -- the Q/A write-first WIP artifact: path contract + the CALLER
that checks the completion marker.

WHY THIS EXISTS
---------------
The two Layer-3 rails were asymmetric, and it was an accident of a hook rather
than a design decision. The RESEARCHER writes its brief incrementally, so a
dropped run still leaves a complete artifact on disk (measured 2026-08-10: peer
run `wf_e03b2ad2-e7e` dropped after 224,317 tokens and left a correct 1,187-line
brief). The Q/A rail is the opposite: `qa-write-guard.sh` denies Write/Edit for
`agent_type == "qa"` outside `.claude/agent-memory/qa/`, so a drop destroyed the
ENTIRE evaluation -- including, on `wf_e03ec2d0-c07`, a genuine surviving mutant
that had to be recovered by hand-parsing the transcript jsonl.

THE SINK IS INSIDE THE Q/A's OWN MEMORY DIR, AND THAT IS THE WHOLE DESIGN.
No allowlist is added and no deny is removed: the guard ALREADY permits this
path, so the blast radius of phase-86.31 is a prompt/doc change plus tests.
The obvious sink -- `handoff/current/evaluator_critique_<sid>.json` -- was
REJECTED because it is the FIRST candidate of
`.claude/hooks/lib/verdict_gate.py::resolve_verdict_source`, which
`auto-commit-and-push.sh` consults to decide whether to HOLD the push. Letting
the evaluator write there would let it authorise its own push by writing PASS:
a textbook confused deputy. Measured 2026-08-10: nothing under `.claude/hooks/`
reads `.claude/agent-memory/qa/` except the guard's own definition of the
constant, so this sink carries no authority.

THE `verdicts/` SUBDIRECTORY IS LOAD-BEARING, NOT COSMETIC.
`scripts/housekeeping/audit_memory.py` globs the memory corpus NON-recursively
(`root.glob("*.md")`) and fails the run on any top-level file MEMORY.md does not
link. Measured 2026-08-10 against the live corpus: a top-level
`verdict_wip_PROBE.md` added TWO problems (`NO POINTER` + `MALFORMED
FRONTMATTER`); the same file under `verdicts/` produced output BYTE-IDENTICAL to
baseline. So the WIP files live one level down, where the memory auditor cannot
see them and they cannot rot its signal.

BORN INERT
----------
The artifact carries `STATUS: INCOMPLETE` from its FIRST byte and the final act
of a successful run rewrites it to `STATUS: COMPLETE`. This is SQLite's
atomic-commit shape: the journal's page-count starts at zero so a torn record is
*inert*, not *ambiguous*. Atomic rename fixes torn VISIBILITY but not semantic
INCOMPLETENESS -- a fully-flushed half-analysis is still a half-analysis, so the
marker has to be explicit rather than implied by the file existing.

A RECOVERED WIP IS EVIDENCE, NEVER A VERDICT -- INCLUDING A `COMPLETE` ONE.
Crash-only doctrine: a crashed process's partial output is INFORMATION, never
its RESULT. The marker distinguishes "the analysis reached its end" from "the
analysis was truncated"; it does NOT promote either into a verdict. That rule is
what keeps a post-drop respawn from quietly becoming verdict-shopping, and it is
why `report()` below has NO `verdict` key to scrape and always carries
`is_verdict: false`.

    $ python scripts/qa/qa_wip.py 86.31 --spawned-at 2026-08-10T12:00:00Z
    $ python scripts/qa/qa_wip.py 86.31            # report (JSON, identity UNCHECKED)
    $ python scripts/qa/qa_wip.py 86.31 --body     # + the recovered prose

Exit codes: 0 COMPLETE, 3 INCOMPLETE/UNMARKED, 4 ABSENT,
5 STALE or IDENTITY_UNKNOWN (a PRIOR cycle's artifact), 2 bad step id.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

#: The guard's allowed prefix. Duplicated from qa-write-guard.sh DELIBERATELY:
#: this module must not import the hook, and the checker asserts the two agree.
MEMORY_DIR = ".claude/agent-memory/qa/"
#: One level below the memory dir, out of audit_memory.py's non-recursive glob.
WIP_SUBDIR = "verdicts"

INCOMPLETE_MARKER = "STATUS: INCOMPLETE -- not a verdict"
COMPLETE_MARKER = "STATUS: COMPLETE -- write-first record, still NOT a verdict"

#: phase-86.31 cycle 2, from the cycle-1 Q/A's third finding. The artifact is
#: at a FIXED path per step, so a cycle-2 spawn that drops BEFORE its first
#: write leaves cycle-1's file sitting there -- and a `COMPLETE` marker on
#: pre-fix evidence reads as current. The header therefore carries WHEN it was
#: written, and `report(..., spawned_at=...)` reports STALE when the artifact
#: predates the spawn being recovered from. A fixed-path artifact with no
#: identity is a stale-read waiting to happen; the marker alone cannot fix it.
WRITTEN_KEY = "WRITTEN"
COMPLETED_KEY = "COMPLETED"
STEP_KEY = "STEP"

_HEADER_RE = re.compile(r"(?m)^\s*(WRITTEN|COMPLETED|STEP):\s*(\S.*?)\s*$")

#: A step id is a dotted numeric path. Anything else is refused rather than
#: sanitised: a resolver that silently repairs a hostile id is how traversal
#: gets in (CWE-22), and no legitimate caller has one.
_STEP_ID_RE = re.compile(r"\A[0-9]+(?:\.[0-9]+)*\Z")

_STATUS_LINE_RE = re.compile(r"\A\s*STATUS:\s*(INCOMPLETE|COMPLETE)\b")


class BadStepId(ValueError):
    """Raised for a step id that is not a dotted numeric path."""


#: phase-86.36. A run-unique component in the NAME. The fixed per-step path made
#: the retry's first write destroy the prior attempt -- and because 86.31
#: requires that write to happen before any analysis, the safety property and
#: the hazard were the same write. MEASURED pre-fix: 4,386 -> 124 bytes, and in
#: production verdict_wip_86.34.md 4,921 -> 796 between two tool calls of a
#: single observer.
#:
#: A COUNTER WAS REJECTED. Deriving `c<N>` requires read-then-write on the
#: directory; two concurrent Claude sessions (routine here) race and silently
#: land on the same N. The stamp needs no coordination and no caller change:
#: the writer already knows its own spawn time because it writes WRITTEN:.
#: Precedent: Airflow `attempt={try_number}.log`, journald `.journal~`, k8s
#: retaining exactly one prior -- a path component plus BOUNDED retention, never
#: locking.
_RUN_STAMP_RE = re.compile(r"\A[0-9]{8}T[0-9]{6}Z\Z")
#: Current record + this many prior attempts. Bounded on the k8s/journald
#: precedent: nobody retains every attempt, and this directory is read by the
#: memory tooling.
DEFAULT_KEEP = 3


def _sid_or_raise(step_id: str) -> str:
    sid = str(step_id).strip()
    if not _STEP_ID_RE.match(sid):
        raise BadStepId(
            f"refusing step id {sid!r}: expected a dotted numeric path such as "
            f"'86.31' (a resolver that repairs a hostile id is how traversal gets in)"
        )
    return sid


def resolve_wip_path(step_id: str, repo: pathlib.Path | None = None,
                     run_stamp: str | None = None) -> pathlib.Path:
    """Return the WIP path for `step_id`. Raises BadStepId on anything else.

    `run_stamp` (phase-86.36, compact UTC `YYYYMMDDTHHMMSSZ`) makes the name
    unique per run. Omitted, this returns the LEGACY fixed path unchanged --
    deliberately, so pre-86.36 artifacts written by a live peer session still
    resolve and every phase-86.31 assertion keeps its exact subject.
    """
    sid = _sid_or_raise(step_id)
    root = pathlib.Path(repo) if repo is not None else REPO
    sink = root / MEMORY_DIR / WIP_SUBDIR
    if run_stamp is None:
        return sink / f"verdict_wip_{sid}.md"
    stamp = str(run_stamp).strip()
    if not _RUN_STAMP_RE.match(stamp):
        raise BadStepId(
            f"refusing run stamp {stamp!r}: expected compact UTC "
            f"'YYYYMMDDTHHMMSSZ'. Refused rather than sanitised, for the same "
            f"reason as the step id."
        )
    return sink / f"verdict_wip_{sid}__{stamp}.md"


def source_present(repo: pathlib.Path | None = None) -> bool:
    """Does the counting source itself EXIST?

    phase-86.21 criterion 6. `list_wip_records` returns `[]` for a missing sink,
    which is indistinguishable from a genuine first attempt -- and the attempt
    counter that drives the 3rd-attempt auto-FAIL is fed by exactly that number.
    So a deleted or unmounted sink SILENTLY disables the escalation rule, which
    is the precise failure mode 86.21 was filed to remove from the harness_log
    grep. Replacing the grep with a ledger did not fix it; the ledger inherited
    it. MEASURED by scripts/qa/mutate_counter_source_86_21.py: with the sink
    deleted, report() was BYTE-IDENTICAL to a genuine first attempt.

    A count of zero is only a fact about ATTEMPTS when this returns True.
    """
    root = pathlib.Path(repo) if repo is not None else REPO
    return (root / MEMORY_DIR / WIP_SUBDIR).is_dir()


def list_wip_records(step_id: str, repo: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every retained record for `step_id`, NEWEST FIRST.

    Includes the legacy un-stamped path when present. Ordering is by the
    WRITTEN stamp inside each file, falling back to mtime, so a record is
    ordered by when the RUN happened rather than by when the bytes last moved.
    """
    sid = _sid_or_raise(step_id)
    root = pathlib.Path(repo) if repo is not None else REPO
    sink = root / MEMORY_DIR / WIP_SUBDIR
    if not sink.is_dir():
        return []
    found = [p for p in sink.glob(f"verdict_wip_{sid}.md") if p.is_file()]
    found += [p for p in sink.glob(f"verdict_wip_{sid}__*.md") if p.is_file()]

    def _key(p: pathlib.Path):
        try:
            dt = _parse_ts(headers(p.read_text(encoding="utf-8", errors="replace"))
                           .get(WRITTEN_KEY, ""))
        except Exception:
            dt = None
        return dt or datetime.datetime.fromtimestamp(p.stat().st_mtime,
                                                     datetime.timezone.utc)

    return sorted(set(found), key=_key, reverse=True)


def prune_wip_records(step_id: str, repo: pathlib.Path | None = None,
                      keep: int = DEFAULT_KEEP) -> list[pathlib.Path]:
    """Delete all but the `keep` newest records. Returns what was removed.

    Retention is BOUNDED on purpose: an unbounded sink is a slow leak in a
    directory the memory tooling reads. `keep < 1` is refused -- keeping zero
    records would reintroduce exactly the data loss this step removes.
    """
    if keep < 1:
        raise ValueError(
            f"refusing keep={keep}: retaining zero records is the defect "
            f"phase-86.36 exists to remove, not a configuration of it"
        )
    records = list_wip_records(step_id, repo)
    removed = []
    for p in records[keep:]:
        try:
            p.unlink()
            removed.append(p)
        except OSError:
            pass
    return removed


def classify(text: str) -> str:
    """COMPLETE / INCOMPLETE / UNMARKED, decided by the FIRST non-blank line.

    UNMARKED is deliberately NOT folded into INCOMPLETE: a file with no marker
    at all was not written by the write-first path, and saying so is more useful
    to a reader than silently degrading it to "partial".
    """
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _STATUS_LINE_RE.match(line)
        return m.group(1) if m else "UNMARKED"
    return "UNMARKED"


def _parse_ts(raw: str):
    """Lenient ISO-8601 parse. Returns None rather than raising -- an
    unparseable stamp must degrade to 'identity unknown', never to a crash in
    a recovery path that only runs when something has already gone wrong."""
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(s)
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)


def headers(text: str) -> dict:
    """The `KEY: value` identity stamps in the artifact header."""
    return {m.group(1): m.group(2) for m in _HEADER_RE.finditer(text)}


def report(step_id: str, repo: pathlib.Path | None = None,
           spawned_at: str | None = None) -> dict:
    """Machine-readable recovery report.

    Carries NO `verdict` key by construction -- there is nothing here for a
    caller to scrape as one -- and states `is_verdict: false` explicitly.

    `spawned_at` (ISO-8601) is the time of the spawn you are recovering FROM.
    Pass it: without it this function cannot tell a current artifact from a
    previous cycle's, and the fixed per-step path guarantees they collide.
    """
    # phase-86.36: several records may now coexist. Pick the one belonging to
    # THIS spawn -- the newest whose WRITTEN is not before it -- and report the
    # rest as priors instead of destroying them. With a single legacy file this
    # selects that file, so every phase-86.31 assertion keeps its exact subject.
    records = list_wip_records(step_id, repo)
    spawn_dt_sel = _parse_ts(spawned_at) if spawned_at else None
    path = resolve_wip_path(step_id, repo)
    if records:
        path = records[0]
        if spawn_dt_sel is not None:
            # OLDEST-first. The record belonging to a spawn is the EARLIEST one
            # written at or after it -- walking newest-first instead picks the
            # most recent record for every spawn, so recovering an older cycle
            # silently hands back a newer cycle's file. That was a real bug in
            # the first version of this selection, caught by the criterion-3
            # assertion "spawn 1 resolves to cycle 1's record".
            for cand in reversed(records):
                w = _parse_ts(headers(cand.read_text(encoding="utf-8",
                                                     errors="replace"))
                              .get(WRITTEN_KEY, ""))
                if w is not None and w >= spawn_dt_sel:
                    path = cand
                    break

    out = {
        "step_id": str(step_id).strip(),
        "path": str(path),
        "exists": path.exists(),
        "status": "ABSENT",
        "is_verdict": False,
        "recoverable": False,
        "bytes": 0,
        "written_at": "",
        "completed_at": "",
        "identity_checked": bool(spawned_at),
        "guidance": "",
        # Priors are LISTED, never merged: each is its own attempt and reading
        # two half-analyses as one is how a recovered partial turns into a
        # confident wrong answer.
        "prior_records": [str(p) for p in records if p != path],
        "records_retained": len(records),
        # phase-86.21 criterion 6. Without this, records_retained==0 conflates
        # "no prior attempts" with "the counting source is gone", and any caller
        # deriving an attempt number from it fails OPEN and silently.
        "source_present": source_present(repo),
    }
    if not path.exists():
        out["guidance"] = (
            "ABSENT: nothing was recovered. Re-run the Q/A. This is NO VERDICT, never PASS."
        ) if out["source_present"] else (
            "SOURCE MISSING: the WIP sink " + MEMORY_DIR + WIP_SUBDIR + "/ DOES NOT EXIST, "
            "so records_retained=0 is NOT a statement about prior attempts -- it is a "
            "statement that the counter has no input. Do NOT derive an attempt number from "
            "it: treat the count as UNKNOWN and say so in the verdict notes. This is NO "
            "VERDICT, never PASS."
        )
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    out["bytes"] = len(text.encode("utf-8"))
    out["status"] = classify(text)
    hdr = headers(text)
    out["written_at"] = hdr.get(WRITTEN_KEY, "")
    out["completed_at"] = hdr.get(COMPLETED_KEY, "")

    # Identity gate, BEFORE the guidance table: a stale artifact is not
    # evidence for this cycle no matter what its marker says.
    if spawned_at:
        spawn_dt = _parse_ts(spawned_at)
        written_dt = _parse_ts(out["written_at"])
        if spawn_dt is None:
            out["status"] = "IDENTITY_UNKNOWN"
        elif written_dt is None:
            out["status"] = "IDENTITY_UNKNOWN"
        elif written_dt < spawn_dt:
            out["status"] = "STALE"
    if out["status"] in ("STALE", "IDENTITY_UNKNOWN"):
        out["recoverable"] = False
        out["guidance"] = {
            "STALE": (
                f"STALE: this artifact was written at {out['written_at']}, BEFORE the spawn "
                f"you are recovering from ({spawned_at}). It belongs to a PREVIOUS cycle and "
                "is NOT evidence for this one -- a spawn that dropped before its first write "
                "leaves the prior cycle's file in place at the same path. Do not read it as "
                "current. An errored/empty rail return is NO VERDICT, NEVER PASS."
            ),
            "IDENTITY_UNKNOWN": (
                "IDENTITY_UNKNOWN: the artifact carries no parseable WRITTEN stamp, or the "
                "spawned_at you passed is unparseable, so it CANNOT be shown to belong to "
                "this cycle. Treat it as untrusted. An errored/empty rail return is NO "
                "VERDICT, NEVER PASS."
            ),
        }[out["status"]]
        return out

    out["recoverable"] = out["status"] in ("COMPLETE", "INCOMPLETE")
    out["guidance"] = {
        "COMPLETE": (
            "COMPLETE: the analysis reached its end. It is EVIDENCE FOR THE NEXT SPAWN "
            "ONLY -- never transcribe it into evaluator_critique.md, never feed it to "
            "the verdict gate, never count it as a verdict. An errored/empty rail return "
            "is NO VERDICT, NEVER PASS."
        ),
        "INCOMPLETE": (
            "INCOMPLETE: the run was truncated, so the analysis is partial and may "
            "contradict itself. Treat as EVIDENCE FOR THE NEXT SPAWN ONLY. An "
            "errored/empty rail return is NO VERDICT, NEVER PASS."
        ),
        "UNMARKED": (
            "UNMARKED: no STATUS marker on the first non-blank line, so this file was "
            "not produced by the write-first path. Treat it as untrusted notes. An "
            "errored/empty rail return is NO VERDICT, NEVER PASS."
        ),
    }[out["status"]]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step_id")
    ap.add_argument("--body", action="store_true",
                    help="also dump the recovered prose under an explicit banner")
    ap.add_argument("--spawned-at", default=None, metavar="ISO8601",
                    help="ISO-8601 time of the spawn you are recovering FROM. PASS IT: "
                         "without it a previous cycle's artifact at the same path cannot "
                         "be distinguished from this one's.")
    ns = ap.parse_args(argv)

    try:
        rep = report(ns.step_id, spawned_at=ns.spawned_at)
    except BadStepId as exc:
        print(f"BAD STEP ID: {exc}", file=sys.stderr)
        return 2
    if not ns.spawned_at:
        print("WARNING: no --spawned-at given, so this report CANNOT tell a current "
              "artifact from a previous cycle's. See qa_wip.report(spawned_at=...).",
              file=sys.stderr)

    print(json.dumps(rep, indent=2, sort_keys=True))
    if ns.body and rep["exists"]:
        print("\n===== RECOVERED WIP BODY -- EVIDENCE ONLY, NOT A VERDICT =====")
        print(pathlib.Path(rep["path"]).read_text(encoding="utf-8", errors="replace"))
        print("===== END RECOVERED WIP BODY -- STILL NOT A VERDICT =====")
    return {"COMPLETE": 0, "INCOMPLETE": 3, "UNMARKED": 3, "ABSENT": 4,
            "STALE": 5, "IDENTITY_UNKNOWN": 5}[rep["status"]]


if __name__ == "__main__":
    sys.exit(main())
