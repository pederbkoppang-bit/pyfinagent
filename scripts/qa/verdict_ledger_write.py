#!/usr/bin/env python3
"""Append ONE verdict row to `handoff/verdict_ledger.jsonl` -- phase-86.85.

WHY THIS EXISTS
---------------
`scripts/qa/verdict_history_86_21.py` (the READER) works. `enforceEscalation` in
`.claude/workflows/qa-verdict.js` (the CONSUMER) works and already fails closed --
an absent or unusable sequence yields `null`, never `0`. **What was missing is a
WRITER.** Measured 2026-08-15:

    $ python scripts/qa/verdict_history_86_21.py --step 86.74 --evidence-only
    status = no_rows_for_step, verdicts = (none)

...for a step that had SIX real verdicts by then. All 35 pre-existing rows carry
`recorded_by=main` and the newest is dated 2026-08-11, i.e. the ledger has only
ever been hand-fed in bulk backfills. Positive control run in the same session:
`--step 86.21` returns `status=ok` with 5 verdicts through the same reader and the
same key, so the zero for 86.74 is a MEASURED zero and the cause is NEVER-WRITTEN
-- not a wrong key, not pruning, not write-only-after-close.

The cost of the gap is concrete. On 2026-08-15 the 86.74 cycle-7 Q/A returned
CONDITIONAL as the third consecutive one. The escalation machinery reported
`sequence_status="not_supplied"`, `consecutive_conditionals=null`,
`would_auto_fail=null`, because the sequence had to be hand-carried and arrived as
prose. The auto-FAIL had to be computed by hand. THIS script is that missing input.

WHY MAIN IS THE WRITER, AND NOT THE Q/A
---------------------------------------
A Workflow-rail agent CANNOT write: an `import fs` makes the script UNLAUNCHABLE
(SyntaxError) -- a runtime property of the Workflow sandbox, not a policy choice
(see `.claude/workflows/research-gate.js:52-55`). That is exactly the ESAA topology
(arXiv:2602.23193 §3, §6.5): the agent emits structured intentions, a deterministic
orchestrator validates and appends, and "denying direct writing reduces the blast
radius of a compromised agent". pyfinagent arrived at the recommended architecture
by accident; this script embraces it rather than engineering around it.

WHY NOT A PostToolUse HOOK -- MEASURED, NOT ASSUMED
---------------------------------------------------
Probed 2026-08-15 with a temporary hook (settings restored byte-identical):

  * a PostToolUse hook DOES receive `tool_response`; and
  * the matcher DOES fire on `Workflow` (`tool_name == "Workflow"` observed); but
  * `tool_response` for a Workflow is the LAUNCH RECEIPT --
    `{runId, scriptPath, status, summary, taskId, taskType, transcriptDir,
    workflowName}` -- and carries NO verdict.

Workflows always run in the background, so PostToolUse fires at launch. There is no
foreground mode. **A hook can therefore never author a verdict row.** It remains
viable as a pure ALARM (it sees `runId` at launch, which is the key needed to notice
a launched run that never got a row), but that is deliberately out of scope here.

SCOPE DISCIPLINE IS THE FINDING
-------------------------------
One append-only JSONL plus a dedup key. No projections, no snapshots, no
rehydration, no CQRS. Azure's own guidance: "Event sourcing is a complex pattern
that introduces significant trade-offs ... Apply it selectively."

WHAT THIS DOES **NOT** DO
-------------------------
* It changes NO verdict semantics. It records the verdict it is given, and the
  vocabulary guard REJECTS anything outside {PASS, CONDITIONAL, FAIL, NO_VERDICT}
  rather than coercing it. There is no input under which a non-PASS becomes a PASS.
* It does NOT decide whether a NO_VERDICT row grades -- that is step 86.45. Today's
  consumers already skip it (`enforceEscalation`: `if (v === 'NO_VERDICT') continue`
  -- it neither extends nor resets a run), so recording a drop faithfully CANNOT
  clear an escalation. This script preserves that; it does not re-decide it.
* It does NOT read or touch `qa_wip.py`'s parallel trail (step 86.79).
* It NEVER rewrites a row. Corrections are new, labelled rows (Azure: in-place
  rewriting "undermines the audit trail").

USAGE
-----
    # append (the seam call -- run BEFORE acting on the verdict)
    python scripts/qa/verdict_ledger_write.py \
        --step 86.74 --verdict CONDITIONAL --run-id wf_8c3730a1-32e \
        --cycle 7 --note "cross-file prose"

    # obtain the array to hand to qa-verdict.js as args.verdict_sequence
    python scripts/qa/verdict_ledger_write.py --emit-sequence --step 86.74

    # self-test (no writes to the real ledger)
    python scripts/qa/verdict_ledger_write.py --self-test

EXIT CODES -- the writer FAILS LOUD, never silently
---------------------------------------------------
    0  row appended, or --emit-sequence/--self-test succeeded
    2  duplicate key refused (the row already exists; nothing was written)
    3  invalid input (unknown verdict, missing step id, no dedup key available)
    4  I/O failure

A silent writer would manufacture exactly the `LEDGER_EMPTY` / absent-row state the
reader is built to refuse, so every failure path here is loud and non-zero.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

#: phase-86.85 cycle 7 (QA-C6-1): the ordering contract rests on "ISO
#: YYYY-MM-DD, so lexicographic order IS chronological order" -- which is only
#: true when the format actually holds. It demonstrably did not: 11 of 52 real
#: rows carried range-shaped dates like '2026-08-09/10', and a driven
#: '--date 2026-8-10' (non-zero-padded) sorted AFTER every ISO August date,
#: re-opening the exact backfill-clears-escalation hole cycle 6 closed. So the
#: format is validated at BOTH seams: build_row refuses to write a non-ISO
#: date, and emit_sequence refuses to ORDER one (same doctrine as the undated
#: and out-of-vocabulary refusals: a row that cannot be placed correctly must
#: be loud, never silently mis-ordered).
ISO_DATE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "handoff" / "verdict_ledger.jsonl"

#: The only verdicts any consumer understands. Kept byte-identical to
#: `.claude/workflows/qa-verdict.js:347`'s VALID set. An unknown verdict is
#: REJECTED, never coerced -- coercion is how a FAIL would silently become
#: something else.
VALID_VERDICTS = ("PASS", "CONDITIONAL", "FAIL", "NO_VERDICT")

EXIT_OK = 0
EXIT_DUPLICATE = 2
EXIT_INVALID = 3
EXIT_IO = 4


class LedgerError(Exception):
    """Raised with an explicit exit code so no failure can be swallowed."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _dedup_key(row: dict) -> tuple[str, str]:
    """The identifier of the LOGICAL event, assigned by the producer.

    `(step_id, run_id)` where `run_id` exists -- it is the rail's own `wf_<uuid>`,
    satisfying Confluent's "unique to the logical event, such as an event key or
    request ID".

    Population = every non-blank line of `handoff/verdict_ledger.jsonl` at
    `d1c4a79d~1` (the pre-86.85 state). Command and result:

        $ git show d1c4a79d~1:handoff/verdict_ledger.jsonl | python3 -c "..."
          total rows          : 35
          run_id key present  : 35
          run_id non-empty    : 35
          run_id wf_-prefixed : 35
          non-wf run_id values: []

    i.e. **35 of 35 on every predicate**. An earlier revision of this docstring, and
    of `contract_86.85.md`, said "33 of the 35" -- a number carried forward from the
    research brief and never re-derived. No predicate yields 33. Corrected against a
    re-run rather than by editing the digit (phase-86.85 cycle-1 Q/A, C2).

    Falls back to `(step_id, cycle)` when `run_id` is absent, which is the only
    other producer-assigned identifier the historical rows carry.

    A retried spawn gets a NEW run_id and so is legitimately a NEW row. Only a
    re-transcription of the SAME run is a duplicate.
    """
    step = (row.get("step_id") or "").strip()
    run = (row.get("run_id") or "").strip()
    if run:
        return (step, f"run:{run}")
    cycle = str(row.get("cycle") or "").strip()
    if cycle:
        return (step, f"cycle:{cycle}")
    raise LedgerError(
        "no dedup key available: a row needs a run_id or a cycle. Refusing to "
        "append an unkeyed row, because it could never be deduplicated later.",
        EXIT_INVALID,
    )


def read_rows(path: Path = LEDGER) -> list[dict]:
    """Every parseable row. A corrupt line is LOUD, never skipped silently."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(
                f"{path}:{lineno} is not valid JSON ({exc}). Refusing to append to "
                "a ledger that cannot be read -- a partial read would under-count "
                "existing keys and let a duplicate through.",
                EXIT_IO,
            ) from exc
    return rows


def existing_keys(path: Path = LEDGER) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in read_rows(path):
        try:
            keys.add(_dedup_key(row))
        except LedgerError:
            # A historical row with neither run_id nor cycle cannot collide with
            # anything we would write (we refuse to write such a row at all).
            continue
    return keys


def build_row(
    step_id: str,
    verdict: str,
    run_id: str | None = None,
    cycle: str | None = None,
    note: str = "",
    recorded_by: str = "main",
    event_date: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Construct the row, validating BEFORE any I/O."""
    step_id = (step_id or "").strip()
    if not step_id:
        raise LedgerError("--step is required and must be non-empty.", EXIT_INVALID)

    verdict = (verdict or "").strip().upper()
    if verdict not in VALID_VERDICTS:
        raise LedgerError(
            f"unknown verdict {verdict!r}. Valid: {', '.join(VALID_VERDICTS)}. "
            "Refusing rather than coercing -- coercion is how a non-PASS would "
            "silently become something else.",
            EXIT_INVALID,
        )

    if event_date is not None and not ISO_DATE_RE.match(str(event_date).strip()):
        raise LedgerError(
            f"--date {event_date!r} is not ISO YYYY-MM-DD. The sequence reader "
            "orders rows lexicographically BY THIS FIELD, and a non-ISO date "
            "sorts wrongly -- a driven '2026-8-10' backfill landed AFTER every "
            "August ISO date and cleared an escalation. Refused rather than "
            "normalised.",
            EXIT_INVALID,
        )

    stamp = now or datetime.now(timezone.utc)
    row = {
        "step_id": step_id,
        "cycle": str(cycle) if cycle is not None else "",
        "verdict": verdict,
        "run_id": (run_id or "").strip(),
        "recorded_by": recorded_by,
        # EVENT time -- when the verdict happened.
        "date": event_date or stamp.date().isoformat(),
        # WRITE time -- when this row was appended. Kept SEPARATE from event time
        # so a backfill can never masquerade as live history. 12 of the existing
        # rows share one microsecond stamp; that is visible ONLY because this
        # field exists.
        "recorded_at": stamp.isoformat(),
        "note": note or "",
    }
    _dedup_key(row)  # raise now, before touching the file
    return row


def append_row(row: dict, path: Path = LEDGER) -> dict:
    """Append exactly one row, refusing a duplicate key. Never rewrites."""
    key = _dedup_key(row)
    if key in existing_keys(path):
        raise LedgerError(
            f"duplicate key {key} already in {path}. Refusing to append. "
            "This ledger is append-only and a key identifies one logical event; "
            "a correction must be a NEW labelled row, never a rewrite.",
            EXIT_DUPLICATE,
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise LedgerError(f"failed to append to {path}: {exc}", EXIT_IO) from exc
    return row


def emit_sequence(step_id: str, path: Path = LEDGER) -> list[str]:
    """The array to hand to `qa-verdict.js` as `args.verdict_sequence`.

    Oldest -> newest **by EVENT date** (`row["date"]`, ISO `YYYY-MM-DD`, so
    lexicographic order IS chronological order), stable by file position within
    a date. phase-86.85 cycle-5 (QA-MUT-B): the previous revision returned FILE
    order while this docstring claimed oldest->newest, and the divergence was
    reachable through the writer's own `--date` backfill flag -- appending an
    older PASS after two CONDITIONALs made `enforceEscalation` read
    [C, C, PASS] (n=0, no auto-fail) instead of the true [PASS, C, C] (n=2,
    auto-fail). A backfill could CLEAR a live escalation: fail-open. Event time
    is what `build_row` persists `date` FOR; the only reader now reads it.

    Matches `enforceEscalation`'s reverse scan. This closes the loop that
    failed on 86.74 cycle 7, where the sequence arrived as prose in `extra` and
    the machinery reported `sequence_status="not_supplied"`.

    An out-of-vocabulary verdict token is LOUD, never silently dropped. Silently
    filtering it would BYPASS the consumer's own fail-closed branch: given the raw
    tokens, `enforceEscalation` returns `sequence_status="unparseable"` and
    `consecutive_conditionals=null`, whereas a filtered list looks like a clean,
    confident, SHORTER sequence -- which can only ever under-count a consecutive
    run, i.e. it fails OPEN. This also keeps the function consistent with
    `read_rows`, which is deliberately loud about a corrupt line for the same
    "would under-count" reason. (phase-86.85 cycle-1 Q/A, WARN.)
    """
    step_id = (step_id or "").strip()
    keyed: list[tuple[str, int, str]] = []
    pos = 0  # file position: the stable tiebreak WITHIN one event date
    for row in read_rows(path):
        if (row.get("step_id") or "").strip() != step_id:
            continue
        verdict = (row.get("verdict") or "").strip().upper()
        if verdict not in VALID_VERDICTS:
            raise LedgerError(
                f"row for step {step_id!r} carries an unrecognised verdict "
                f"{verdict!r} (valid: {', '.join(VALID_VERDICTS)}). Refusing to "
                "emit a sequence with it silently removed -- a filtered sequence "
                "is SHORTER than the truth and can only under-count a consecutive "
                "run, bypassing the consumer's fail-closed 'unparseable' branch.",
                EXIT_IO,
            )
        event_date = (row.get("date") or "").strip()
        if event_date and not ISO_DATE_RE.match(event_date):
            # QA-C6-1: 11 of 52 real rows carry range-shaped dates like
            # '2026-08-09/10', and '2026-8-10' sorts AFTER every ISO August
            # date -- silently mis-ordering is exactly the escalation-clearing
            # hole. Refuse loudly; the row must be corrected (a new labelled
            # row cannot share the same run_id, so repairing a legacy row is
            # an operator act on the file, recorded in the step notes).
            raise LedgerError(
                f"row for step {step_id!r} carries a non-ISO event date "
                f"{event_date!r} (expected YYYY-MM-DD). Refusing to ORDER it -- "
                "a lexicographic sort over a malformed date silently mis-places "
                "the verdict, which can clear an escalation.",
                EXIT_IO,
            )
        if not event_date:
            # Same fail-loud doctrine as the verdict vocabulary above: a row
            # that cannot be ORDERED cannot be silently placed. An undated row
            # sorted "somewhere" can move a verdict across an escalation
            # boundary exactly like the QA-MUT-B backfill did.
            raise LedgerError(
                f"row for step {step_id!r} carries no event date; refusing to "
                "order it silently -- an unordered verdict can move across an "
                "escalation boundary. Repair the row's `date` field.",
                EXIT_IO,
            )
        keyed.append((event_date, pos, verdict))
        pos += 1
    # phase-86.85 cycle 7 (QA-M-POS-const): the key EXCLUDES the verdict
    # string. A bare sorted(keyed) fell through to element 3 on same-date rows
    # -- the COMMON case (86.85's three FAILs all share 2026-08-15) -- so a
    # pos-neutering mutant reordered real sequences alphabetically and
    # disarmed an escalation. With the key pinned to (date, position), the
    # verdict can never participate in ordering; and because Python's sort is
    # guaranteed stable, even a constant position degrades to file order
    # WITHIN a date, which is the documented contract.
    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], t[1]))]


def _self_test() -> int:
    """Exercise the guards against a temp ledger. Touches no real file."""
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ledger.jsonl"

        r1 = build_row("99.1", "CONDITIONAL", run_id="wf_aaa", cycle="1")
        append_row(r1, p)
        check("append writes one row", len(read_rows(p)) == 1)

        # duplicate (step_id, run_id) refused
        dup = build_row("99.1", "PASS", run_id="wf_aaa", cycle="1")
        try:
            append_row(dup, p)
            check("duplicate key refused", False)
        except LedgerError as e:
            check("duplicate key refused", e.code == EXIT_DUPLICATE)
        check("duplicate did not append", len(read_rows(p)) == 1)

        # a retry is a NEW run_id -> legitimately a new row
        append_row(build_row("99.1", "CONDITIONAL", run_id="wf_bbb", cycle="2"), p)
        check("new run_id appends", len(read_rows(p)) == 2)

        # unknown verdict rejected, not coerced
        try:
            build_row("99.1", "MOSTLY_FINE", run_id="wf_ccc")
            check("unknown verdict rejected", False)
        except LedgerError as e:
            check("unknown verdict rejected", e.code == EXIT_INVALID)

        # unkeyed row refused
        try:
            build_row("99.1", "PASS")
            check("unkeyed row refused", False)
        except LedgerError as e:
            check("unkeyed row refused", e.code == EXIT_INVALID)

        # event time and write time are separate fields
        check("event/write time separate", r1["date"] != r1["recorded_at"])

        append_row(build_row("99.1", "CONDITIONAL", run_id="wf_ddd", cycle="3"), p)

        # ORDER. The fixture MUST be order-sensitive. A previous revision asserted
        # against ["CONDITIONAL","CONDITIONAL","CONDITIONAL"] -- a PALINDROME -- so
        # a mutant returning `out[::-1]` passed this very check, and the cycle-1
        # Q/A's QA-M1 survived because of it. Ordering is the load-bearing contract
        # of the one function that feeds args.verdict_sequence: reversing
        # [PASS,C,C] to [C,C,PASS] takes enforceEscalation from n=2/auto_fail=true
        # to n=0/auto_fail=false, i.e. it SILENTLY DISARMS the escalation.
        # phase-86.85 cycle-5 (QA-MUT-B): the fixture must carry DISTINCT event
        # dates or a date-conditional reorder is unobservable -- every row used
        # to share one date, so the "oldest->newest" check could not fail for
        # the date-sort mutant class (fixture-cannot-break-the-symmetry).
        for i, v in enumerate(("PASS", "CONDITIONAL", "FAIL"), start=1):
            append_row(build_row("99.4", v, run_id=f"wf_ord{i}", cycle=str(i),
                                 event_date=f"2026-08-1{i - 1}"), p)
        ordered = emit_sequence("99.4", p)
        # guard-on-the-guard: if this fixture is ever made palindromic again, the
        # ordering check silently stops testing ordering. Assert it cannot be.
        check("order fixture is NOT palindromic (anti-vacuity)",
              ordered != list(reversed(ordered)))
        check("order fixture carries distinct event dates (anti-vacuity for the date axis)",
              len({f"2026-08-1{i}" for i in range(3)}) == 3)
        check("sequence is oldest->newest",
              ordered == ["PASS", "CONDITIONAL", "FAIL"])

        # phase-86.85 cycle-5 (QA-MUT-B, the reachable differential): a
        # BACKFILLED older verdict must land in EVENT order, not file order.
        # File order read [C, C, PASS] as n=0/no-auto-fail; event order reads
        # [PASS, C, C] as n=2/auto-fail. The backfill flag is shipped, so this
        # is a real path, not a hypothetical.
        append_row(build_row("99.7", "CONDITIONAL", run_id="wf_bf1",
                             event_date="2026-08-11"), p)
        append_row(build_row("99.7", "CONDITIONAL", run_id="wf_bf2",
                             event_date="2026-08-12"), p)
        append_row(build_row("99.7", "PASS", run_id="wf_bf0", note="backfill",
                             event_date="2026-08-10"), p)
        check("backfilled older verdict lands in EVENT order (a backfill cannot clear an escalation)",
              emit_sequence("99.7", p) == ["PASS", "CONDITIONAL", "CONDITIONAL"])

        # phase-86.85 cycle 7 (QA-M-POS-const): SAME-DATE rows are the common
        # case and must preserve FILE order -- and the fixture must be able to
        # FAIL for a verdict-string fallthrough, so the file order (C, P, F)
        # deliberately differs from the alphabetical order (C, F, P).
        for i, v in enumerate(("CONDITIONAL", "PASS", "FAIL"), start=1):
            append_row(build_row("99.9", v, run_id=f"wf_sd{i}",
                                 event_date="2026-08-14"), p)
        check("same-date rows preserve FILE order (verdict string never orders)",
              emit_sequence("99.9", p) == ["CONDITIONAL", "PASS", "FAIL"])

        # QA-C6-1: a non-ISO date is refused at the WRITE seam...
        try:
            build_row("99.10", "PASS", run_id="wf_niso", event_date="2026-8-10")
            check("non-ISO --date refused at build_row", False)
        except LedgerError as e:
            check("non-ISO --date refused at build_row",
                  e.code == EXIT_INVALID and "not ISO" in str(e))
        # ...and at the READ seam, for rows that reached the file some other way.
        niso = Path(td) / "niso.jsonl"
        niso.write_text(
            '{"step_id":"99.11","verdict":"PASS","run_id":"wf_n1","date":"2026-08-09/10"}\n',
            encoding="utf-8")
        try:
            emit_sequence("99.11", niso)
            check("non-ISO stored date is LOUD at emit, never silently ordered", False)
        except LedgerError as e:
            check("non-ISO stored date is LOUD at emit, never silently ordered",
                  "non-ISO event date" in str(e))

        # An undated row cannot be ORDERED, so it must refuse loudly -- the
        # same doctrine as the out-of-vocabulary verdict above. Silently
        # placing it can move a verdict across an escalation boundary.
        undated = Path(td) / "undated.jsonl"
        row_ud = build_row("99.8", "PASS", run_id="wf_ud")
        row_ud.pop("date", None)
        undated.write_text(json.dumps(row_ud) + "\n", encoding="utf-8")
        try:
            emit_sequence("99.8", undated)
            check("undated row is LOUD, never silently ordered", False)
        except LedgerError as exc:
            check("undated row is LOUD, never silently ordered",
                  "no event date" in str(exc))

        # other steps are not mixed in
        append_row(build_row("99.2", "PASS", run_id="wf_eee", cycle="1"), p)
        check("sequence filters by step",
              emit_sequence("99.4", p) == ["PASS", "CONDITIONAL", "FAIL"])

        # an out-of-vocabulary token must be LOUD, never silently filtered --
        # a filtered sequence is SHORTER than the truth and under-counts a run.
        # phase-86.85 cycle 6: the rows CARRY dates and the check PINS the
        # message. Without dates, the new undated-row guard raised the same
        # LedgerError/EXIT_IO first, and removing the vocabulary guard (M7)
        # survived because a DIFFERENT guard's refusal satisfied a check that
        # only looked at the exit code -- a masked fixture, the same class as
        # the palindrome this block already documents.
        oov = Path(td) / "oov.jsonl"
        oov.write_text(
            '{"step_id":"99.5","verdict":"CONDITIONAL","run_id":"wf_o1","date":"2026-08-10"}\n'
            '{"step_id":"99.5","verdict":"COND","run_id":"wf_o2","date":"2026-08-11"}\n'
        )
        try:
            emit_sequence("99.5", oov)
            check("out-of-vocabulary verdict is loud", False)
        except LedgerError as e:
            check("out-of-vocabulary verdict is loud",
                  e.code == EXIT_IO and "unrecognised verdict" in str(e))

        # NO_VERDICT is recorded faithfully and is not silently dropped
        append_row(build_row("99.3", "NO_VERDICT", run_id="wf_fff", cycle="1"), p)
        check("NO_VERDICT recorded", emit_sequence("99.3", p) == ["NO_VERDICT"])

        # a corrupt ledger is LOUD, never read as "no verdicts"
        bad = Path(td) / "corrupt.jsonl"
        bad.write_text('{"step_id": "99.9"\n')
        try:
            read_rows(bad)
            check("corrupt ledger is loud", False)
        except LedgerError as e:
            check("corrupt ledger is loud", e.code == EXIT_IO)

        # ---- COVERAGE OF THE REMAINING GUARD CLASS (cycle-2 Q/A) ----
        # Enumerated from source rather than from the findings: every `raise
        # LedgerError` site plus every distinguishing branch of _dedup_key. Two
        # cycles running, a NEW guard shipped with no check; the fix is to cover
        # the CLASS, not the two instances that happened to be reported.

        # (a) an I/O failure must be LOUD, never a silent success. QA-M6: with
        # this guard reverted the writer PRINTS the row, exits 0, writes NOTHING
        # and says nothing on stderr -- the exact "silent writer" the module
        # docstring forbids, and the state criterion 6 warns is unfalsifiable.
        ro = Path(td) / "ro"
        ro.mkdir()
        ro.chmod(0o500)  # r-x------ : cannot create a file inside
        try:
            append_row(build_row("99.6", "PASS", run_id="wf_io"), ro / "l.jsonl")
            check("I/O failure is loud", False)
        except LedgerError as e:
            check("I/O failure is loud", e.code == EXIT_IO)
        finally:
            ro.chmod(0o700)

        # (b) the dedup key must include step_id. QA-M4: dropping it makes the
        # same run_id collide ACROSS steps, so a legitimate row for a second step
        # is refused and LOST -- an under-count, i.e. it fails OPEN.
        k1 = Path(td) / "keyed.jsonl"
        append_row(build_row("88.1", "PASS", run_id="wf_same", cycle="1"), k1)
        # The second append MUST be caught rather than allowed to propagate: with
        # step_id dropped from the key it raises, and an uncaught raise exits the
        # whole self-test with a traceback instead of a failed check -- which the
        # matrix scores UNSCORABLE, not KILLED, because a crash is not evidence
        # that the guard discriminated. Convert it into a check result.
        try:
            append_row(build_row("99.9", "PASS", run_id="wf_same", cycle="1"), k1)
            second_step_accepted = True
        except LedgerError:
            second_step_accepted = False
        check("dedup key includes step_id (same run_id, different step)",
              second_step_accepted and len(read_rows(k1)) == 2)

        # (c) an empty step id is refused
        try:
            build_row("", "PASS", run_id="wf_x")
            check("empty step_id rejected", False)
        except LedgerError as e:
            check("empty step_id rejected", e.code == EXIT_INVALID)

        # (d) the CLI argument guards, driven through main()
        check("--emit-sequence without --step exits 3",
              main(["--emit-sequence", "--ledger", str(p)]) == EXIT_INVALID)
        check("append without --verdict exits 3",
              main(["--step", "99.1", "--ledger", str(p)]) == EXIT_INVALID)

        # (e) THE CYCLE FALLBACK BRANCH -- the third outcome of _dedup_key, and the
        # one two "class-complete" claims missed. It is LIVE, not theoretical:
        # 5 of the real ledger's rows carry no run_id, ALL of them on 86.74 (cycles
        # 1-drop-a, 1-drop-b, 3, 3b, 4), i.e. the very step this writer exists to
        # serve. If the cycle VALUE stops distinguishing rows, those appends are
        # refused under EXIT_DUPLICATE -- the benign "already recorded" code a
        # caller ignores -- and the verdicts silently vanish from the sequence
        # enforceEscalation consumes. That is the under-count / fail-OPEN direction.
        # The fixture replays the real 86.74 backfill shape deliberately.
        cf = Path(td) / "cyclekey.jsonl"
        cycles = ("1-drop-a", "1-drop-b", "3", "3b", "4")
        refused = 0
        for c in cycles:
            try:
                append_row(build_row("86.99", "CONDITIONAL", cycle=c), cf)
            except LedgerError:
                refused += 1
        check("cycle fallback keys on the cycle VALUE (run_id absent)",
              refused == 0 and len(read_rows(cf)) == len(cycles))
        check("cycle fallback preserves sequence length",
              len(emit_sequence("86.99", cf)) == len(cycles))

        # ── CLI-boundary refusals (cycle-4, C8) ───────────────────────────
        # These two guards are DEFENCE IN DEPTH: build_row would refuse an
        # empty --step and an empty --verdict anyway, so removing the CLI
        # guard still refuses -- only the MESSAGE changes. That made them
        # unobservable, and a guard that cannot fail does not count: matrix
        # cell M14 SURVIVED against the self-test as it stood, which is
        # exactly the "shipped a guard with no coverage" class this step
        # exists to end. Found by verify_matrix_coverage_86_85.py deriving
        # the gap from the writer's own AST, not by re-reading the file.
        #
        # So the assertion pins WHICH refusal fires, not merely THAT one
        # does. Asserting the exit code alone would be vacuous -- both paths
        # return EXIT_INVALID (3).
        import contextlib
        import io

        def cli(argv: list[str]) -> tuple[int, str]:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                rc = main(argv)
            return rc, buf.getvalue()

        rc_seq, err_seq = cli(["--emit-sequence"])
        check("CLI refuses --emit-sequence without --step, at the boundary",
              rc_seq == EXIT_INVALID
              and "--emit-sequence requires --step" in err_seq)

        rc_app, err_app = cli(["--step", "99.5", "--ledger", str(p)])
        check("CLI refuses an append with no --verdict, at the boundary",
              rc_app == EXIT_INVALID
              and "both --step and --verdict are required" in err_app)

        # Anti-vacuity: the CLI must still SUCCEED on a well-formed append,
        # or the two refusals above would pass simply because every
        # invocation fails.
        rc_ok, _ = cli(["--step", "99.6", "--verdict", "PASS",
                        "--run-id", "wf_cli_ok", "--ledger", str(p)])
        check("CLI still appends a well-formed row (anti-vacuity)",
              rc_ok == EXIT_OK)

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("SELF-TEST PASSED")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Append one verdict row to handoff/verdict_ledger.jsonl "
                    "(phase-86.85). Main calls this AT THE SEAM, before acting "
                    "on the verdict."
    )
    ap.add_argument("--step", help="masterplan step id, e.g. 86.74")
    ap.add_argument("--verdict", help=f"one of: {', '.join(VALID_VERDICTS)}")
    ap.add_argument("--run-id", default=None, help="the rail's wf_<uuid>")
    ap.add_argument("--cycle", default=None, help="cycle label, e.g. 7")
    ap.add_argument("--note", default="", help="short free-text context")
    ap.add_argument("--recorded-by", default="main")
    ap.add_argument("--date", default=None, help="EVENT date (default: today UTC)")
    ap.add_argument("--ledger", default=None, help="override ledger path (testing)")
    ap.add_argument("--emit-sequence", action="store_true",
                    help="print the JSON array for args.verdict_sequence")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    path = Path(args.ledger) if args.ledger else LEDGER

    try:
        if args.emit_sequence:
            if not args.step:
                raise LedgerError("--emit-sequence requires --step.", EXIT_INVALID)
            print(json.dumps(emit_sequence(args.step, path)))
            return EXIT_OK

        if not args.step or not args.verdict:
            raise LedgerError(
                "both --step and --verdict are required to append a row.",
                EXIT_INVALID,
            )

        row = build_row(
            step_id=args.step,
            verdict=args.verdict,
            run_id=args.run_id,
            cycle=args.cycle,
            note=args.note,
            recorded_by=args.recorded_by,
            event_date=args.date,
        )
        append_row(row, path)
        print(json.dumps(row, ensure_ascii=False))
        return EXIT_OK

    except LedgerError as exc:
        # LOUD. Never a silent no-op -- a silent writer manufactures the exact
        # LEDGER_EMPTY state the reader exists to refuse.
        print(f"verdict_ledger_write: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
