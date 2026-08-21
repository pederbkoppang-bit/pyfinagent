#!/usr/bin/env python3
"""phase-90.1 -- resolve what an attempt PRODUCED and what it COST.

WHY THIS EXISTS
---------------
`handoff/audit/attempt_budget_audit.jsonl` records that a launch happened and
nothing else. Measured 2026-08-20 over the live ledger, the row key set is
exactly {ts, type, step_id, workflow, tool_use_id, session_id,
attempt_number_inclusive, note} -- no outcome field and no token field. Two
consequences follow mechanically:

- the gate spends budget identically on a graded attempt and on a rail drop,
  and drops are NOT cheap: over 617 Workflow run records the 46 StructuredOutput
  drops cost a mean of 191,796 tokens against a completed run's 242,997 (8.8M
  tokens in total);
- `attempt_budget.DEFAULT_MAX_TOKENS` cannot bind, because `attempt_gate` calls
  `state.record(outcome)` with no `tokens=` and `Attempt.tokens` defaults to 0,
  so `tokens_used` is a constant 0.

Both are ACCOUNTING defects. This module adds no policy: it reads the Workflow
run record that already exists and says what the attempt produced and cost.

THE JOIN KEY IS `startTime`, NEVER `timestamp`
----------------------------------------------
A run record carries two clocks. `timestamp` is written at COMPLETION and a run
can last 15+ minutes (observed durationMs 1,002,277), so it is useless as a
launch key. `startTime` (epoch ms) is the launch moment and matches the
PreToolUse row's `ts` almost exactly.

EVERY FIGURE BELOW IS CORPUS-DEPENDENT AND DRIFTS. State the corpus with the
number or do not state the number. Re-derive with
`python3 scripts/harness/attempt_outcomes.py --backfill --dry-run`; the sweep
that produces the ambiguity threshold is step 90.10's to make re-runnable.

Measured 2026-08-21 over 106 attempt rows and 635 run records: joining on
`(args.step_id, |startTime - row.ts| <= tol)` resolves 101 of 106 UNIQUELY with
ZERO ambiguous matches at every tolerance from 30s through 385s; the 5 that miss
are the synthetic `999.2` pipetest rows, which have no run record at all. The
same join on `timestamp` resolves 2 of 106 -- the field is the bug, not the key.
**Ambiguity first appears at 386s** (386s -> 1 ambiguous, 387s -> 2, 899s -> 11),
which is why the default tolerance is 30s: ~13x headroom over the observed worst
case and still an order of magnitude short of ambiguity.

CORRECTED phase-90.1 cycle 5. This paragraph previously read "resolves 83 of 89
... resolves 9 of 89 ... Ambiguity first appears at 900s". The 900s figure was
re-measured as 386s during cycle 4 and the cycle-4 record CLAIMED this docstring
had been corrected; `git show --stat a252b025 -- scripts/harness/attempt_outcomes.py`
is EMPTY -- the commit never touched this file, and the stale 900s then propagated
into masterplan step 90.10's audit_basis and into the mutation-matrix comment.
The `timestamp` figure did not reproduce either (9 filed, 2 measured today, 1
measured by the cycle-1 Q/A). Both numbers are REPLACED here, not annotated.

An ambiguous or absent match resolves to UNKNOWN. It is never guessed.

THE VOCABULARY IS TWO FIELDS, NOT ONE
--------------------------------------
`outcome` is the closed five the step's criterion 1 fixes:
PASS | CONDITIONAL | FAIL | NO_VERDICT | UNKNOWN.

But NO_VERDICT alone conflates four measurably different things -- 46
StructuredOutput drops (expensive), 2 args-unparseable failures (ZERO tokens; a
caller bug, not a rail drop), 5 `killed` aborts, and "no run record found",
which is a fifth thing again. So the finer class rides in a SEPARATE
`outcome_reason` field rather than being crushed into the five. The precedent is
Kubernetes, which carries a machine reason (BackoffLimitExceeded /
DeadlineExceeded / PodFailurePolicy) beside a closed terminal condition rather
than inventing a condition per cause.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
It does not reconcile the run record against `handoff/verdict_ledger.jsonl`.
Of 120 run_ids shared between them, 7 disagree: 2 ledger rows say NO_VERDICT
where the rail returned a real verdict, and 5 say FAIL where the rail returned
CONDITIONAL (the documented 3rd-CONDITIONAL conversion). `outcome` here is the
RAIL'S RAW RETURN. The ledger is Main's transcription and is a different
population; fixing it is step 90.5, not this one.

It also decides nothing about whether a drop SHOULD count against a budget --
the published literature openly disagrees (DeepSWE excludes infra-terminated
rollouts; arXiv:2607.12227 scores them r=0). Recording the outcome keeps both
computable. Deciding it inside the counter does not.

    python3 scripts/harness/attempt_outcomes.py --backfill --dry-run
    python3 scripts/harness/attempt_outcomes.py --backfill

Testing overrides: ATTEMPT_GATE_RUN_RECORDS=<dir root>,
ATTEMPT_GATE_MASTERPLAN=<path>, ATTEMPT_GATE_LEDGER=<path>.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Where Claude Code writes Workflow run records. Overridable for tests ONLY.
DEFAULT_RUN_RECORD_ROOT = (
    Path.home() / ".claude" / "projects"
    / "-Users-ford--openclaw-workspace-pyfinagent"
)

#: The closed five. criterion 1 fixes this vocabulary; do not extend it here --
#: extend `outcome_reason` instead.
OUTCOMES = ("PASS", "CONDITIONAL", "FAIL", "NO_VERDICT", "UNKNOWN")

#: Default join tolerance, seconds. See the module docstring for why 30.
DEFAULT_TOLERANCE_S = 30

#: The four keys a launch row carries as null placeholders and `--backfill`
#: fills in. They are the ONLY keys the backfill may write onto an existing row,
#: and only while that row is unsettled (outcome null or UNKNOWN).
RESOLUTION_KEYS = frozenset({"outcome", "outcome_reason", "total_tokens", "run_id"})


def run_record_root() -> Path:
    return Path(os.environ.get("ATTEMPT_GATE_RUN_RECORDS")
                or DEFAULT_RUN_RECORD_ROOT)


def masterplan_path() -> Path:
    return Path(os.environ.get("ATTEMPT_GATE_MASTERPLAN")
                or REPO / ".claude" / "masterplan.json")


def ledger_path() -> Path:
    return Path(os.environ.get("ATTEMPT_GATE_LEDGER")
                or REPO / "handoff" / "audit" / "attempt_budget_audit.jsonl")


def parse_ts(raw: str) -> datetime.datetime | None:
    """The ledger's `ts` shape, then ISO as a fallback. Never raises."""
    if not raw:
        return None
    try:
        return datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _args_of(record: dict) -> dict:
    """Workflow args, whether stored as an object or a JSON string."""
    a = record.get("args")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except (json.JSONDecodeError, ValueError):
            return {}
    return a if isinstance(a, dict) else {}


def load_run_records(root: Path | None = None) -> list[dict]:
    """Every Workflow run record, normalised to the fields this module uses.

    A record that will not parse is SKIPPED rather than raising: an unreadable
    record can only make an attempt resolve to UNKNOWN, which is the direction
    that under-counts tokens and therefore allows more, never less.
    """
    root = run_record_root() if root is None else root
    out = []
    for f in glob.glob(str(root / "*" / "workflows" / "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- see docstring
            continue
        if not isinstance(d, dict):
            continue
        args = _args_of(d)
        start = d.get("startTime")
        out.append({
            "run_id": d.get("runId"),
            "step_id": str(args.get("step_id") or "").strip(),
            "start_ms": start if isinstance(start, (int, float)) else None,
            "total_tokens": int(d.get("totalTokens") or 0),
            "status": d.get("status"),
            "error": str(d.get("error") or ""),
            "workflow": d.get("workflowName"),
            "result": d.get("result"),
        })
    return out


#: The workflow whose return IS a verdict. Every other rail spends the same
#: ledger allowance (finding I-7) but can never produce one, so it must not be
#: filed under the same reason as a Q/A that tried and dropped.
EVALUATION_WORKFLOW = "qa-verdict"


def classify(record: dict) -> tuple[str, str]:
    """(outcome, outcome_reason) for ONE matched run record.

    A returned verdict wins. Otherwise the record's own workflow, `status` and
    `error` say which KIND of no-verdict this was -- the distinction the single
    NO_VERDICT value cannot carry, and the reason this function returns a PAIR.

    The reasons are not cosmetic; each was measured over the 617 live run
    records and they behave differently:

    - `graded`                  -- the rail returned a verdict.
    - `not_an_evaluation`       -- a NON-Q/A rail (research-gate and friends).
      It COMPLETED successfully; it simply is not an evaluation and never had a
      verdict to give. 16 of the live ledger's 18 no-verdict attempt rows are
      this, and calling them "drops" would badly overstate the drop rate.
      This is also the persistent discriminator step 90.6 needs to separate the
      two budgets now sharing one counter.
    - `structured_output_drop`  -- the documented rail drop. EXPENSIVE: mean
      191,796 tokens against a completed run's 242,997.
    - `args_unparseable`        -- costs ZERO tokens. A caller bug, not a rail
      drop; merging the two is exactly what a binary vocabulary would do.
    - `killed`                  -- aborted.
    - `completed_without_result` -- the workflow ended `completed` and returned
      nothing at all. Rare (2 live rows) and worth its own name precisely
      because it looks like success everywhere else.
    """
    res = record.get("result")
    if isinstance(res, dict):
        v = res.get("verdict")
        if isinstance(v, str) and v.upper() in OUTCOMES and v.upper() != "UNKNOWN":
            return v.upper(), "graded"
    err = record.get("error") or ""
    if record.get("status") == "killed":
        return "NO_VERDICT", "killed"
    if "not parseable as JSON" in err:
        return "NO_VERDICT", "args_unparseable"
    if "without calling StructuredOutput" in err:
        return "NO_VERDICT", "structured_output_drop"
    wf = str(record.get("workflow") or "")
    if wf and wf != EVALUATION_WORKFLOW:
        return "NO_VERDICT", "not_an_evaluation"
    if record.get("status") == "completed" and res is None:
        return "NO_VERDICT", "completed_without_result"
    return "NO_VERDICT", "no_verdict_other"


def resolve_row(row: dict, records: list[dict],
                tolerance_s: int = DEFAULT_TOLERANCE_S) -> dict:
    """Resolve ONE attempt row against the run records.

    Returns the fields to merge onto the row. UNKNOWN is used only when no
    record matched, or when more than one did -- an ambiguous match is never
    broken by picking the nearest, because a wrong outcome is worse than an
    admitted unknown.
    """
    unknown = {"outcome": "UNKNOWN", "outcome_reason": "no_run_record",
               "total_tokens": 0, "run_id": None}
    t = parse_ts(str(row.get("ts") or ""))
    sid = str(row.get("step_id") or "")
    if t is None or not sid:
        return dict(unknown, outcome_reason="unresolvable_row")
    hits = []
    for r in records:
        if r["step_id"] != sid or r["start_ms"] is None:
            continue
        rt = datetime.datetime.fromtimestamp(r["start_ms"] / 1000,
                                             datetime.timezone.utc)
        if abs((rt - t).total_seconds()) <= tolerance_s:
            hits.append(r)
    if not hits:
        return unknown
    if len(hits) > 1:
        return dict(unknown, outcome_reason="ambiguous_match")
    rec = hits[0]
    outcome, reason = classify(rec)
    return {"outcome": outcome, "outcome_reason": reason,
            "total_tokens": int(rec["total_tokens"]), "run_id": rec["run_id"]}


def read_rows(path: Path) -> list[dict]:
    """Ledger rows, preserving order. A corrupt line is kept as a marker.

    Mirrors `attempt_gate.read_ledger`'s rule: a corrupt row must not silently
    shrink the count. Here it must additionally survive a rewrite untouched,
    so the raw text is carried.
    """
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append({"__parsed__": json.loads(line), "__raw__": line})
        except json.JSONDecodeError:
            rows.append({"__parsed__": None, "__raw__": line})
    return rows


def backfill(path: Path | None = None, records: list[dict] | None = None,
             dry_run: bool = False,
             tolerance_s: int = DEFAULT_TOLERANCE_S) -> dict:
    """Resolve every attempt row; enrich IN PLACE, additive-only.

    ADDITIVE-ONLY IS PROVEN, NOT ASSERTED. Before writing, every enriched row is
    projected back onto its ORIGINAL key set and compared to the original row;
    any difference aborts the whole write. Row count and order are checked the
    same way, and corrupt lines are passed through verbatim. So this can add
    fields and can never rewrite history -- which is what lets an append-only
    stream be enriched at all.

    phase-90.1 cycle-2, criterion 1 BLOCK -- WHY `RESOLUTION_KEYS` EXISTS.
    The first version made the two halves of its own commit incompatible. The
    gate writes a launch row with the four resolution keys PRESENT and null
    ("completed later by --backfill", which is correct: the outcome genuinely is
    unknown at PreToolUse). But the projection above then saw `outcome: None`
    becoming `outcome: "FAIL"` as an existing field CHANGING, and aborted the
    entire write. The very first launch after that commit -- the cycle-1 Q/A's
    own row -- made `--backfill` exit 1, so the "re-runnable backfill" criterion
       1 asks for was falsified within minutes of shipping. Every self-test and
    matrix fixture seeded the OLD row shape, so the drives stayed green
    regardless: a fixture that predates the writer cannot detect the writer.

    The rule now distinguishes SETTLED from UNSETTLED rather than PRESENT from
    ABSENT. A row is unsettled while its `outcome` is null or UNKNOWN, and only
    then may the four resolution keys be written. Once a row carries a real
    outcome it is FROZEN and nothing may touch it -- which is the immutability
    that actually matters, and it is checked separately below. UNKNOWN stays
    writable on purpose: a launch still in flight has no run record yet, and
    freezing that answer would make a transient absence permanent.
    """
    path = ledger_path() if path is None else path
    records = load_run_records() if records is None else records
    rows = read_rows(path)
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    frozen = 0
    enriched: list[str] = []
    resolved = 0
    for r in rows:
        parsed = r["__parsed__"]
        if parsed is None:
            enriched.append(r["__raw__"])          # corrupt line: verbatim
            continue
        if parsed.get("type", "attempt") != "attempt":
            enriched.append(r["__raw__"])          # extension rows: verbatim
            continue
        settled = str(parsed.get("outcome") or "") not in ("", "UNKNOWN")
        if settled:
            # FROZEN. A row that already carries a real outcome is history and
            # is passed through byte-for-byte; it is not even re-resolved.
            enriched.append(r["__raw__"])
            counts[parsed["outcome"]] = counts.get(parsed["outcome"], 0) + 1
            reasons[str(parsed.get("outcome_reason") or "already_settled")] = \
                reasons.get(str(parsed.get("outcome_reason") or "already_settled"), 0) + 1
            resolved += 1
            frozen += 1
            continue
        res = resolve_row(parsed, records, tolerance_s)
        merged = dict(parsed)
        merged.update(res)
        # --- the additive-only proof, per row -----------------------------
        # Everything EXCEPT the four resolution keys must be byte-identical.
        # Those four are the launch placeholders this pass exists to fill, and
        # they are only writable while the row is unsettled (checked above).
        projection = {k: merged[k] for k in parsed
                      if k in merged and k not in RESOLUTION_KEYS}
        original = {k: v for k, v in parsed.items() if k not in RESOLUTION_KEYS}
        if projection != original:
            raise AssertionError(
                f"backfill would MUTATE an existing field on row ts={parsed.get('ts')!r} "
                f"step_id={parsed.get('step_id')!r}: {projection!r} != {original!r}. "
                "Refusing to write; the ledger is append-only and enrichment "
                "may only ADD keys or FILL the unsettled resolution keys "
                f"{sorted(RESOLUTION_KEYS)}.")
        counts[res["outcome"]] = counts.get(res["outcome"], 0) + 1
        reasons[res["outcome_reason"]] = reasons.get(res["outcome_reason"], 0) + 1
        resolved += 1
        enriched.append(json.dumps(merged, ensure_ascii=False))
    if len(enriched) != len(rows):
        raise AssertionError("backfill changed the row COUNT -- refusing to write")
    if not dry_run and rows:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text("\n".join(enriched) + "\n", encoding="utf-8")
    return {"rows_total": len(rows), "attempt_rows": resolved,
            "already_settled_passed_through": frozen,
            "outcome_counts": counts, "reason_counts": reasons,
            "dry_run": dry_run, "ledger": str(path),
            "tolerance_s": tolerance_s}


def resolved_rows(step_id: str, rows: list[dict],
                  records: list[dict] | None = None) -> list[dict]:
    """The step's attempt rows with outcome/total_tokens present.

    A row that already carries a resolved `outcome` is trusted as-is (that is
    the point of persisting it). Only unresolved rows cost a run-record scan,
    and the scan is lazy: with nothing to resolve, no records are loaded.
    """
    mine = [r for r in rows
            if str(r.get("step_id")) == step_id
            and r.get("type", "attempt") == "attempt"]
    need = [r for r in mine if not r.get("outcome")]
    if need:
        records = load_run_records() if records is None else records
        for r in need:
            r.update(resolve_row(r, records))
    return mine


def masterplan_step_ids(path: Path | None = None) -> set[str]:
    """Every id ANYWHERE in the plan of record, plus its `phase-`-stripped form.

    phase-90.1 cycle-2, criterion 4 BLOCK. The first version walked exactly
    `phases[].steps[]` and nothing else. The plan of record is NOT that shape
    everywhere: real steps also live under `phases[].subphases[]`, and the
    shallow walk missed 123 dotted ids -- among them **10 that are `pending`
    AND `harness_required`**: 38.13 and 46.0 through 46.8. Because a missing id
    DENIES, that shipped a gate which blocked ten real pending steps while its
    own denial text asserted they were "not a step in .claude/masterplan.json".
    Fail-CLOSED, and the exact opposite of the fail-open discipline the module
    documents.

    The reason it was not caught is worth keeping: the blast-radius measurement
    that cleared this change used THE SAME shallow walk, so the control shared
    the defect it was meant to detect and could only ever agree. A recall check
    over the plan's own members -- now `assert_membership_recall()` below, and
    driven by the gate's self-test -- is the check that cannot share it, because
    it derives its expectation from the file rather than from this function.

    So: recurse over the whole document and collect every `id`. Over-inclusion
    is the SAFE direction here (it can only ADMIT), which is why this walks
    everything rather than trying to enumerate the container keys that happen
    to exist today. Duplicate-id resolution across the `phase-` prefix remains
    step 90.7's.
    """
    path = masterplan_path() if path is None else path
    try:
        mp = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    ids: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            raw = node.get("id")
            if isinstance(raw, str) and raw.strip():
                sid = raw.strip()
                ids.add(sid)
                if sid.startswith("phase-"):
                    ids.add(sid[len("phase-"):])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(mp)
    return ids


def assert_membership_recall(path: Path | None = None) -> dict:
    """Every id the plan contains must be ADMITTED. Derived from the file.

    This is the check the cycle-1 blast-radius measurement could not be: it
    builds its expectation by re-reading the plan with an INDEPENDENT walk
    (collect every dotted id, however nested) instead of reusing
    `masterplan_step_ids`, so the two cannot share a traversal bug. If a real
    member is not admitted, that is a recall failure and it is loud.
    """
    path = masterplan_path() if path is None else path
    try:
        mp = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "members": 0, "missing": []}
    dotted = re.compile(r"\A[0-9]+(?:\.[0-9]+)*\Z")
    members: set[str] = set()

    def collect(node) -> None:
        if isinstance(node, dict):
            raw = node.get("id")
            if isinstance(raw, str):
                sid = raw.strip()
                if sid.startswith("phase-"):
                    sid = sid[len("phase-"):]
                if dotted.match(sid):
                    members.add(sid)
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(mp)
    known = masterplan_step_ids(path)
    missing = sorted(members - known,
                     key=lambda s: [int(p) for p in s.split(".")])
    return {"ok": not missing, "members": len(members),
            "missing": missing[:20], "missing_total": len(missing)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--backfill" in argv:
        summary = backfill(dry_run="--dry-run" in argv)
        print(json.dumps(summary, indent=2, sort_keys=True))
        unknown = summary["outcome_counts"].get("UNKNOWN", 0)
        print(f"\nUNKNOWN = {unknown} (used ONLY where no run record matched; "
              "an ambiguous match also resolves UNKNOWN and is never guessed)")
        return 0
    print(__doc__.split("\n\n", 1)[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
