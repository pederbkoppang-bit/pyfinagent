#!/usr/bin/env python3
"""phase-86.71 -- the PreToolUse gate that finally WIRES attempt_budget.py.

WHERE THIS SITS
---------------
Registered in `.claude/settings.json` as a PreToolUse hook on the `Workflow`
tool -- the seam where every Layer-3 run actually originates (measured: 655
PreToolUse rows for Workflow in `handoff/audit/pre_tool_use_audit.jsonl`,
2026-05-28..2026-08-17). It reads the hook's stdin JSON, attributes the launch
to a masterplan step via `tool_input.args.step_id` (object OR JSON-string args
-- 80.6% of production launches are strings), and:

- BELOW the ceiling: appends ONE attempt row to the append-only ledger
  `handoff/audit/attempt_budget_audit.jsonl` and allows (exit 0). The row is
  written on ATTEMPT, not outcome -- a dropped spawn costs full tokens and
  returns nothing, which is exactly why the old verdict-keyed counters failed
  (attempt_budget.py's docstring carries the measurements).
- AT the ceiling: DENIES the launch (exit 2, the only blocking exit code per
  the official hooks doc), writes
  `handoff/current/escalation_attempt_budget_<sid>.md` with the module's own
  escalation summary, and spends ZERO tokens. **A denial is not a verdict** --
  it can never pass or fail a step; it only stops the loop and hands the
  decision to the operator.
- A launch with NO recoverable step_id is not a step attempt: allowed, not
  recorded, and said out loud on stderr. (Self-audit and ad-hoc workflows land
  here by design.)

FAIL-OPEN, LOUDLY -- AND THE BOUND STATED
-----------------------------------------
Per the official hook docs, timeouts, schema errors and missing scripts all
fail OPEN; only exit 2 blocks. So this gate cannot be a hard guarantee against
a crashed hook, and it does not pretend to be: any internal error exits 0 with
a `[attempt-gate] INTERNAL ERROR -- failing OPEN` line on stderr, because
breaking every Workflow launch on a gate bug would be worse than one uncounted
attempt. The audit stream makes a silent gap visible after the fact.

THE PASS EXCEPTION (deliberate, narrow)
---------------------------------------
If the verdict ledger already carries a PASS for the step, the gate ALLOWS
regardless of count: `BudgetState.disposition()` checks PASS before exhaustion
so a re-grade of a passed step (the 86.93 workflow) is never budget-blocked.
This can only ever ADMIT MORE evaluation of an already-passed step; it can
never admit work a Q/A refused.

OPERATOR EXTENSION (audited, not unforgeable)
---------------------------------------------
`--operator-extend <sid> --by N --reason "..."` appends an extension row that
raises the allowance for ONE step. It is an AUDITED act (the row records who,
when, why, in a git-tracked stream), not a cryptographic one -- a determined
agent could run it, exactly as it could edit any file. The protection here is
that the act cannot be silent, which is the same protection every hook in this
repo relies on. The deny message quotes the exact command so the operator
never has to hand-edit an append-only audit file.

    echo '<hook stdin json>' | python3 scripts/harness/attempt_gate.py
    python3 scripts/harness/attempt_gate.py --status 86.84
    python3 scripts/harness/attempt_gate.py --operator-extend 86.84 --by 2 --reason "..."
    python3 scripts/harness/attempt_gate.py --self-test

Testing override: ATTEMPT_GATE_LEDGER=<path> (testing only; the same shape as
verdict_ledger_write's --ledger). ATTEMPT_GATE_VERDICT_LEDGER overrides the
verdict-ledger source for the PASS exception, same caveat.
"""
from __future__ import annotations

import json
import os
import re
import sys
import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "harness"))
sys.path.insert(0, str(REPO / "scripts" / "qa"))

from attempt_budget import (  # noqa: E402
    BudgetState, Outcome, Disposition, DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_TOKENS,
)

LEDGER = Path(os.environ.get("ATTEMPT_GATE_LEDGER",
                             REPO / "handoff" / "audit" / "attempt_budget_audit.jsonl"))
VERDICT_LEDGER = Path(os.environ.get("ATTEMPT_GATE_VERDICT_LEDGER",
                                     REPO / "handoff" / "verdict_ledger.jsonl"))
ESCALATION_DIR = Path(os.environ.get("ATTEMPT_GATE_ESCALATION_DIR",
                                     REPO / "handoff" / "current"))

#: A dotted numeric step id, same refusal rule as qa_wip.py: anything else is
#: not attributed rather than sanitised.
_STEP_ID_RE = re.compile(r"\A[0-9]+(?:\.[0-9]+)*\Z")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_step_id(tool_input: dict) -> str | None:
    """step_id from Workflow args: dict, JSON string, or regex salvage."""
    args = tool_input.get("args")
    sid = None
    if isinstance(args, dict):
        sid = args.get("step_id") or args.get("stepId")
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                sid = parsed.get("step_id") or parsed.get("stepId")
        except (json.JSONDecodeError, ValueError):
            m = re.search(r'"step_id"\s*:\s*"([^"]+)"', args[:600])
            sid = m.group(1) if m else None
    if sid is None:
        return None
    sid = str(sid).strip()
    return sid if _STEP_ID_RE.match(sid) else None


def workflow_label(tool_input: dict) -> str:
    sp = str(tool_input.get("scriptPath") or "")
    if sp:
        return Path(sp).name
    return str(tool_input.get("name") or "unnamed")


def read_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A corrupt attempt row must not silently shrink the count -- treat
            # it as an attempt of unknown shape (over-counting escalates EARLY,
            # the safe direction, mirroring qa_wip's prune-before-unlink rule).
            rows.append({"step_id": "__corrupt__", "type": "attempt"})
    return rows


def append_row(row: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def verdict_outcomes(step_id: str) -> list[Outcome]:
    """Outcomes already recorded for the step, for the PASS exception only."""
    try:
        from verdict_ledger_write import emit_sequence  # noqa: PLC0415
        seq = emit_sequence(step_id, VERDICT_LEDGER)
        return [Outcome(v) for v in seq if v in Outcome.__members__]
    except Exception:
        return []  # no ledger, unreadable, or malformed -> no PASS exception


def build_state(step_id: str, rows: list[dict]) -> BudgetState:
    mine = [r for r in rows if r.get("step_id") == step_id]
    attempts = [r for r in mine if r.get("type", "attempt") == "attempt"]
    extensions = sum(int(r.get("extra_attempts") or 0)
                     for r in mine if r.get("type") == "operator_extension")
    state = BudgetState(step_id=step_id,
                        max_attempts=DEFAULT_MAX_ATTEMPTS + extensions,
                        max_tokens=DEFAULT_MAX_TOKENS)
    verdicts = verdict_outcomes(step_id)
    for i, _r in enumerate(attempts):
        outcome = verdicts[i] if i < len(verdicts) else Outcome.NO_VERDICT
        state.record(outcome)
    # A PASS recorded in the verdict ledger counts even when the attempts
    # ledger is younger than the step (this gate starts counting from its
    # wiring date; history is not backfillable -- the danger-hook discarded
    # tool_input for all 185,020 historical rows).
    if Outcome.PASS in verdicts and not any(
            a.outcome is Outcome.PASS for a in state.attempts):
        state.record(Outcome.PASS, note="verdict-ledger PASS predating this gate's rows")
    return state


def decide(step_id: str, rows: list[dict]) -> tuple[str, BudgetState]:
    state = build_state(step_id, rows)
    d = state.disposition()
    if d is Disposition.ESCALATE:
        return "deny", state
    return "allow", state


def write_escalation(state: BudgetState) -> Path:
    p = ESCALATION_DIR / f"escalation_attempt_budget_{state.step_id}.md"
    body = state.escalation_summary() or (
        f"# BUDGET EXHAUSTED -- step {state.step_id}\n")
    body += (
        "\n## How to proceed (operator)\n\n"
        "A further attempt requires an AUDITED extension row:\n\n"
        f"    python3 scripts/harness/attempt_gate.py --operator-extend {state.step_id} "
        "--by 1 --reason \"<why another attempt is warranted>\"\n\n"
        "The denial itself is NOT a verdict: the step remains exactly as the\n"
        "last Q/A left it.\n"
        f"\n*(written {_now()} by attempt_gate.py at the deny)*\n"
    )
    p.write_text(body, encoding="utf-8")
    return p


def handle_hook() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print("[attempt-gate] INTERNAL ERROR -- unreadable stdin, failing OPEN",
              file=sys.stderr)
        return 0
    try:
        if payload.get("tool_name") != "Workflow":
            return 0
        tool_input = payload.get("tool_input") or {}
        sid = extract_step_id(tool_input)
        if sid is None:
            print("[attempt-gate] Workflow launch carries no step_id -- not a "
                  "step attempt; allowed and not counted.", file=sys.stderr)
            return 0
        rows = read_ledger()
        decision, state = decide(sid, rows)
        if decision == "deny":
            esc = write_escalation(state)
            # The first version called esc.relative_to(REPO) unconditionally;
            # with the escalation dir overridden outside the repo (the test
            # environment) that RAISED, fell into the fail-open handler, and
            # turned a deny into an allow -- found by mutation_matrix_86_71's
            # own control run. The message path must never be able to defeat
            # the decision it reports.
            esc_shown = (esc.relative_to(REPO) if esc.is_relative_to(REPO)
                         else esc)
            print(
                f"[attempt-gate] DENIED: step {sid} has used "
                f"{state.attempts_used}/{state.max_attempts} attempts "
                f"(cumulative, cross-session; increments on ATTEMPT, not outcome). "
                f"This launch was stopped BEFORE any tokens were spent. "
                f"A denial is not a verdict. Operator escalation written to "
                f"{esc_shown}; to authorize another attempt run: "
                f"python3 scripts/harness/attempt_gate.py --operator-extend {sid} "
                f"--by 1 --reason \"...\"",
                file=sys.stderr)
            return 2
        append_row({
            "ts": _now(), "type": "attempt", "step_id": sid,
            "workflow": workflow_label(tool_input),
            "tool_use_id": payload.get("tool_use_id") or "",
            "session_id": payload.get("session_id") or "",
            "attempt_number_inclusive": state.attempts_used + 1,
            "note": "recorded at launch (PreToolUse); outcome unknown at this seam",
        })
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[attempt-gate] INTERNAL ERROR -- {type(exc).__name__}: {exc} "
              "-- failing OPEN (the launch proceeds UNCOUNTED; see the hooks "
              "doc: only exit 2 blocks, so a broken gate must not break the "
              "harness)", file=sys.stderr)
        return 0


def cmd_status(sid: str) -> int:
    rows = read_ledger()
    decision, state = decide(sid, rows)
    print(json.dumps({
        "step_id": sid,
        "attempts_used": state.attempts_used,
        "max_attempts": state.max_attempts,
        "tokens_used": state.tokens_used,
        "disposition": state.disposition().value,
        "next_launch": decision,
        "ledger": str(LEDGER),
    }, indent=2))
    return 0


def cmd_extend(sid: str, by: int, reason: str) -> int:
    if not reason.strip():
        print("--reason is required: an unexplained extension is exactly the "
              "silent act this gate exists to prevent.", file=sys.stderr)
        return 2
    if by < 1:
        print("--by must be >= 1", file=sys.stderr)
        return 2
    append_row({"ts": _now(), "type": "operator_extension", "step_id": sid,
                "extra_attempts": int(by), "reason": reason.strip(),
                "recorded_by": "operator"})
    print(f"extension recorded: step {sid} allowance +{by}")
    return 0


def _self_test() -> int:
    import tempfile
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "attempts.jsonl"
        global LEDGER, VERDICT_LEDGER
        old_l, old_v = LEDGER, VERDICT_LEDGER
        LEDGER = led
        VERDICT_LEDGER = Path(td) / "verdicts.jsonl"  # absent -> no PASS exception
        try:
            # below ceiling: allow
            d, s = decide("9.1", read_ledger(led))
            check("fresh step -> allow", d == "allow" and s.attempts_used == 0)
            for i in range(DEFAULT_MAX_ATTEMPTS):
                append_row({"ts": _now(), "type": "attempt", "step_id": "9.1"}, led)
            d, s = decide("9.1", read_ledger(led))
            check(f"at ceiling ({DEFAULT_MAX_ATTEMPTS}) -> deny", d == "deny")
            # cross-process shape: rows written above are re-READ from disk here
            check("count survives re-read from disk",
                  build_state("9.1", read_ledger(led)).attempts_used
                  == DEFAULT_MAX_ATTEMPTS)
            # extension raises allowance
            append_row({"ts": _now(), "type": "operator_extension",
                        "step_id": "9.1", "extra_attempts": 1,
                        "reason": "self-test"}, led)
            d, _ = decide("9.1", read_ledger(led))
            check("operator extension re-opens exactly one attempt", d == "allow")
            append_row({"ts": _now(), "type": "attempt", "step_id": "9.1"}, led)
            d, _ = decide("9.1", read_ledger(led))
            check("extension consumed -> deny again", d == "deny")
            # PASS exception
            VERDICT_LEDGER.write_text(json.dumps(
                {"step_id": "9.1", "verdict": "PASS", "run_id": "wf_x",
                 "date": "2026-08-17"}) + "\n", encoding="utf-8")
            d, s = decide("9.1", read_ledger(led))
            check("verdict-ledger PASS -> allow (re-grades never budget-blocked)",
                  d == "allow" and s.disposition() is Disposition.CLOSED_PASS)
            # a corrupt row over-counts, never under-counts
            led.write_text(led.read_text() + "not json\n", encoding="utf-8")
            check("corrupt row counts as an attempt (over-count is the safe direction)",
                  any(r["step_id"] == "__corrupt__" for r in read_ledger(led)))
            # denial writes no verdict anywhere
            check("deny path emits no verdict artifact (no such key exists)",
                  "verdict" not in json.dumps({"decision": "deny"}))
            # unattributable launches are not counted
            check("no step_id -> not attributed",
                  extract_step_id({"args": {"topic": "x"}}) is None)
            check("string args attribute correctly",
                  extract_step_id({"args": json.dumps({"step_id": "9.2"})}) == "9.2")
            check("malformed string args salvage the step id",
                  extract_step_id({"args": '{"step_id":"9.3","broken":['}) == "9.3")
            check("hostile step id refused",
                  extract_step_id({"args": {"step_id": "../evil"}}) is None)
        finally:
            LEDGER, VERDICT_LEDGER = old_l, old_v
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return handle_hook()
    if argv[0] == "--self-test":
        return _self_test()
    if argv[0] == "--status" and len(argv) == 2:
        return cmd_status(argv[1])
    if argv[0] == "--operator-extend":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--operator-extend", dest="sid")
        ap.add_argument("--by", type=int, default=1)
        ap.add_argument("--reason", default="")
        ns = ap.parse_args(argv)
        return cmd_extend(ns.sid, ns.by, ns.reason)
    print(__doc__.split("\n\n", 1)[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
