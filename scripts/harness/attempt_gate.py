#!/usr/bin/env python3
"""phase-86.71 -- the PreToolUse gate that finally WIRES attempt_budget.py.

WHERE THIS SITS
---------------
Registered in `.claude/settings.json` as a PreToolUse hook on the `Workflow`
tool -- the FIRST-CLASS seam where Layer-3 runs originate (655+ PreToolUse
Workflow rows in `handoff/audit/pre_tool_use_audit.jsonl`,
2026-05-28..2026-08-17). SCOPE BOUND, stated rather than implied (phase-86.71
cycle-1 Q/A): the Agent-tool fallback path -- `Agent(subagent_type:'qa')` /
`'researcher'`, CLAUDE.md's documented next move after a rail drop, 42+44
historical spawns -- is NOT gated by this hook. A budget that matters most
right after drops has a hole exactly there; gating the Agent tool needs
step-id attribution from free-text prompts (no structured args field) and is
deliberately left as its own decision rather than bolted on loosely here.
It reads the hook's stdin JSON, attributes the launch
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
from attempt_outcomes import (  # noqa: E402  -- phase-90.1
    masterplan_step_ids, resolved_rows,
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


def extract_step_id_claim(tool_input: dict) -> str | None:
    """The step id a launch CLAIMS, before any validation. None if none claimed.

    phase-90.1 splits this from `extract_step_id` because "no step id at all"
    and "a step id that resolves to nothing" are different events with different
    correct answers. The first is the documented escape hatch for self-audit and
    ad-hoc workflows (81 of 617 historical launches) and stays allowed-and-
    uncounted. The second is a claim the plan of record cannot corroborate, and
    it used to mint its own private 5-attempt allowance -- `999.2`, absent from
    every masterplan step, already holds 5 attempt rows and a written escalation
    file. Appending `.1` to a real id was enough to do it, through the ordinary
    args field, with no file edits.
    """
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
    return sid or None


def extract_step_id(tool_input: dict) -> str | None:
    """The claimed step id, but ONLY if it is real.

    Two gates, and they are different questions:
      1. SHAPE  -- `_STEP_ID_RE`, which refuses `../evil` and other hostile input.
      2. MEMBERSHIP -- the id must exist in `.claude/masterplan.json`.

    Shape alone is what let `86.118.1`, `86.1180` and `999.99` through: each is
    a perfectly well-formed dotted-numeric string that names no step. Returning
    None here makes the hook DENY loudly instead of silently granting a fresh
    allowance.

    Membership degrades OPEN and says so: if the plan of record cannot be read
    at all, every id passes membership rather than every launch being denied by
    an unreadable file. That direction can only allow more, never less, and it
    is consistent with this file's fail-open discipline.
    """
    sid = extract_step_id_claim(tool_input)
    if sid is None or not _STEP_ID_RE.match(sid):
        return None
    known = masterplan_step_ids()
    if not known:
        print("[attempt-gate] masterplan unreadable or empty -- skipping the "
              "step-id membership check for this launch (fail-open: this can "
              "only allow more, never less)", file=sys.stderr)
        return sid
    return sid if sid in known else None


def workflow_label(tool_input: dict) -> str:
    sp = str(tool_input.get("scriptPath") or "")
    if sp:
        return Path(sp).name
    return str(tool_input.get("name") or "unnamed")


def read_ledger(path: Path | None = None) -> list[dict]:
    # Resolved at CALL time: a def-time default binds the import-time LEDGER,
    # and the self-test's global rebinding then silently reads/writes the
    # PRODUCTION ledger -- measured: the first run of the cycle-3 self-test
    # appended its synthetic 9.4 extension row to the real audit stream
    # through exactly this. (That row remains, disclosed, append-only.)
    path = LEDGER if path is None else path
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


def append_row(row: dict, path: Path | None = None) -> None:
    path = LEDGER if path is None else path  # call-time resolution, see read_ledger
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def verdict_outcomes(step_id: str) -> list[Outcome]:
    """Outcomes already recorded for the step, for the PASS exception only."""
    try:
        from verdict_ledger_write import emit_sequence  # noqa: PLC0415
        seq = emit_sequence(step_id, VERDICT_LEDGER)
        return [Outcome(v) for v in seq if v in Outcome.__members__]
    except Exception as exc:  # noqa: BLE001
        # Fail-CLOSED by direction (an empty list can only REMOVE the PASS
        # allowance) -- but never silently: the writer's refusals are LOUD by
        # design, and swallowing them here un-louds them (cycle-10 Q/A of
        # 86.85 flagged exactly this). Say what was swallowed and why the
        # direction is safe.
        print(f"[attempt-gate] verdict-ledger read failed for step {step_id}: "
              f"{type(exc).__name__}: {exc} -- proceeding WITHOUT the PASS "
              "exception (fail-closed: this can only deny more, never allow "
              "more)", file=sys.stderr)
        return []


def build_state(step_id: str, rows: list[dict]) -> BudgetState:
    mine = [r for r in rows if r.get("step_id") == step_id]
    attempts = [r for r in mine if r.get("type", "attempt") == "attempt"]
    extensions = sum(int(r.get("extra_attempts") or 0)
                     for r in mine if r.get("type") == "operator_extension")
    state = BudgetState(step_id=step_id,
                        max_attempts=DEFAULT_MAX_ATTEMPTS + extensions,
                        max_tokens=DEFAULT_MAX_TOKENS)
    verdicts = verdict_outcomes(step_id)
    # phase-90.1: resolve this step's OWN rows so the attempt carries what it
    # produced and what it cost. Before this, `record()` was called with no
    # `tokens=` at all, so `tokens_used` was a constant 0 and the token half of
    # `exhausted` could never bind -- every escalation file on disk prints
    # "tokens used : 0 / 1,200,000" verbatim while real steps ran past 2.5M.
    #
    # Fail-open and loud: a resolver failure leaves tokens at 0, which is the
    # direction that ALLOWS more. Denying launches because a run record could
    # not be read would be the harness breaking itself over accounting.
    try:
        attempts = resolved_rows(step_id, attempts)
    except Exception as exc:  # noqa: BLE001
        print(f"[attempt-gate] outcome resolution failed for step {step_id}: "
              f"{type(exc).__name__}: {exc} -- proceeding with UNRESOLVED rows "
              "(fail-open: tokens read as 0, which can only allow more)",
              file=sys.stderr)
    for i, r in enumerate(attempts):
        # The row's OWN resolved outcome wins when present. The positional
        # fallback pairs the i-th attempt with the i-th verdict-ledger row
        # across two populations with different start dates -- the same
        # positional-parse defect attempt_budget.py's docstring records as the
        # reason its first 86.28 fixture was wrong. Keep it only for rows that
        # predate resolution.
        recorded = str(r.get("outcome") or "")
        if recorded in Outcome.__members__:
            outcome = Outcome(recorded)
        elif recorded == "UNKNOWN":
            outcome = Outcome.NO_VERDICT
        else:
            outcome = verdicts[i] if i < len(verdicts) else Outcome.NO_VERDICT
        tokens = r.get("total_tokens")
        state.record(outcome, tokens=int(tokens) if isinstance(tokens, int) else 0,
                     run_id=str(r.get("run_id") or ""))
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


def _slug(claim: str) -> str:
    """A filesystem-safe stand-in for an id we are refusing to trust.

    The denial path writes a file NAMED after the rejected claim, and the claim
    is attacker-controlled text straight out of `args`. `_STEP_ID_RE` is not a
    guard here: this path exists precisely for ids that FAILED it, so `../evil`
    reaches this function by design. Everything outside [A-Za-z0-9._-] is
    replaced and the result is capped, so a rejected id can never escape
    ESCALATION_DIR or name an arbitrary path.
    """
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in claim)
    safe = safe.strip("._-") or "unnamed"
    return safe[:64]


def _unknown_step_id_body(claim: str) -> str:
    """The escalation body for a claim the plan of record does not contain."""
    return (
        f"# UNRECOGNISED STEP ID -- {claim!r} -- OPERATOR DECISION REQUIRED\n"
        "\n"
        "A Workflow launch claimed this step id. It is not a step in\n"
        "`.claude/masterplan.json`.\n"
        "\n"
        "## THIS IS NOT A PASS, NOT A FAIL, AND NOT AN EXHAUSTION\n"
        "\n"
        "No budget was consumed and no verdict is implied. The launch was\n"
        "stopped before any tokens were spent.\n"
        "\n"
        "## Why an unrecognised id is refused rather than counted\n"
        "\n"
        "Every distinct id gets its own attempt allowance. An id that names no\n"
        "step therefore mints a FRESH allowance on demand -- appending `.1` to a\n"
        "real step id was enough to do it, through the ordinary `args` field,\n"
        "with no file edits. The live ledger already carries `999.2`, which is\n"
        "absent from every masterplan step and holds 5 attempt rows.\n"
        "\n"
        "## How to proceed (operator)\n"
        "\n"
        "- If this IS a step attempt: correct the id to a real masterplan step.\n"
        "- If it is NOT a step attempt (self-audit, ad-hoc workflow): omit\n"
        "  `step_id` from `args` entirely. That path is still allowed and\n"
        "  uncounted, by design.\n"
        "- If the step is real but not yet filed: file it in the masterplan\n"
        "  first. The plan of record is the allowance list.\n"
    )


#: Machine reasons a launch can be denied. Kubernetes carries a reason beside a
#: closed terminal condition (BackoffLimitExceeded / DeadlineExceeded /
#: PodFailurePolicy) rather than one generic failure; same idea here.
REASON_ATTEMPT_BUDGET = "attempt_budget"      # the pre-90.1 path -- name kept so
                                              # the four files already on disk are
                                              # not orphaned by this change
REASON_UNKNOWN_STEP_ID = "unknown_step_id"


def write_escalation(state: BudgetState,
                     reason: str = REASON_ATTEMPT_BUDGET,
                     body: str | None = None) -> Path:
    """Write the operator escalation for ONE denial, named by its reason.

    phase-90.1 fixes two defects in one place:

    1. The path was FIXED at `escalation_attempt_budget_<sid>.md` for every
       denial, so any future non-exhaustion denial would overwrite a real
       exhaustion record. Files at that exact path exist on disk today, and one
       of them (86.85) is hand-authored by the operator.
    2. The body fell back to a literal `# BUDGET EXHAUSTED -- step <sid>`
       whenever `disposition() != ESCALATE` -- i.e. precisely when the step was
       NOT exhausted. A denial for any other reason therefore forged an
       exhaustion record. That fallback is REMOVED: a caller that has no
       exhaustion summary must supply its own body, and a body is never
       invented on its behalf.
    """
    p = ESCALATION_DIR / f"escalation_{reason}_{state.step_id}.md"
    if body is None:
        body = state.escalation_summary()
        if not body:
            raise ValueError(
                f"write_escalation({reason!r}) has no exhaustion summary and no "
                "explicit body -- refusing to forge a '# BUDGET EXHAUSTED' "
                "record for a step that is not exhausted")
    if reason == REASON_ATTEMPT_BUDGET:
        body += (
            "\n## How to proceed (operator)\n\n"
            "A further attempt requires an AUDITED extension row:\n\n"
            f"    python3 scripts/harness/attempt_gate.py --operator-extend {state.step_id} "
            "--by 1 --reason \"<why another attempt is warranted>\"\n\n"
        )
    body += (
        "The denial itself is NOT a verdict: the step remains exactly as the\n"
        "last Q/A left it.\n"
        f"\n*(written {_now()} by attempt_gate.py at the deny, reason={reason})*\n"
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
        claim = extract_step_id_claim(tool_input)
        if claim is None:
            print("[attempt-gate] Workflow launch carries no step_id -- not a "
                  "step attempt; allowed and not counted.", file=sys.stderr)
            return 0
        sid = extract_step_id(tool_input)
        if sid is None:
            # phase-90.1: a launch that CLAIMS a step must name a real one.
            # Before this, an unresolvable id was "allowed and not counted",
            # which handed every unrecognised key its own private 5-attempt
            # allowance -- reachable by appending ".1" to a real id, through
            # the ordinary args field, with no file edits.
            esc = write_escalation(
                BudgetState(step_id=_slug(claim)),
                reason=REASON_UNKNOWN_STEP_ID,
                body=_unknown_step_id_body(claim))
            esc_shown = (esc.relative_to(REPO) if esc.is_relative_to(REPO)
                         else esc)
            print(
                f"[attempt-gate] DENIED: this launch claims step_id {claim!r}, "
                "which is not a step in .claude/masterplan.json. A claimed step "
                "must be a real one -- an unrecognised id would otherwise get a "
                "fresh, private attempt allowance. Fix the id, or omit step_id "
                "entirely if this launch is not a step attempt (that path is "
                "still allowed and uncounted). This launch was stopped BEFORE "
                f"any tokens were spent, and a denial is NOT a verdict. Written "
                f"to {esc_shown}",
                file=sys.stderr)
            return 2
        rows = read_ledger()
        decision, state = decide(sid, rows)
        if decision == "deny":
            esc = write_escalation(state, reason=REASON_ATTEMPT_BUDGET)
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
            # phase-90.1: the fields are PRESENT and explicitly null rather than
            # absent. The outcome genuinely is unknown at PreToolUse -- the run
            # has not happened -- so the record is completed later by
            # `attempt_outcomes.py --backfill`, which is the same shape
            # Kubernetes uses when it stamps a terminal condition onto a record
            # after the fact. An absent key and an unresolved one look identical
            # to a reader; a null one does not.
            "outcome": None,
            "outcome_reason": "unresolved_at_launch",
            "total_tokens": None,
            "run_id": None,
            "note": "recorded at launch (PreToolUse); outcome unknown at this seam",
        })
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[attempt-gate] INTERNAL ERROR -- {type(exc).__name__}: {exc} "
              "-- failing OPEN (the launch proceeds UNCOUNTED; see the hooks "
              "doc: only exit 2 blocks, so a broken gate must not break the "
              "harness)", file=sys.stderr)
        return 0


def _outcome_mix(state: BudgetState) -> dict:
    mix: dict[str, int] = {}
    for a in state.attempts:
        mix[a.outcome.value] = mix.get(a.outcome.value, 0) + 1
    return mix


def cmd_status(sid: str) -> int:
    rows = read_ledger()
    decision, state = decide(sid, rows)
    print(json.dumps({
        "step_id": sid,
        "attempts_used": state.attempts_used,
        "max_attempts": state.max_attempts,
        "tokens_used": state.tokens_used,
        "max_tokens": state.max_tokens,
        # phase-90.1: the gap between these two is the thing the old counter
        # could not see -- an attempt that cost full tokens and returned no
        # verdict. Printing them is what makes it auditable from the CLI
        # instead of only from inside the process.
        "verdicts_seen": state.verdicts_seen,
        "dropped": state.dropped,
        "outcome_mix": _outcome_mix(state),
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
        global LEDGER, VERDICT_LEDGER, ESCALATION_DIR
        old_l, old_v, old_e = LEDGER, VERDICT_LEDGER, ESCALATION_DIR
        LEDGER = led
        VERDICT_LEDGER = Path(td) / "verdicts.jsonl"  # absent -> no PASS exception
        # phase-90.1: the escalation dir must be redirected too. The first
        # revision of these checks called write_escalation with only LEDGER
        # rebound, and it wrote a real `escalation_unknown_step_id_9.9.md` into
        # the production `handoff/current/` -- the SAME leak class as the 9.4
        # extension row in read_ledger's docstring, in a second channel. A
        # self-test must contain EVERY output channel it touches, not the one
        # that happened to be noticed first.
        ESCALATION_DIR = Path(td) / "escalations"
        ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
        # phase-90.1, criterion 4's last clause. This self-test's ids -- 9.1
        # through 9.5 -- are NOT synthetic: all five are REAL masterplan steps.
        # So a membership check would let them pass SILENTLY, which is exactly
        # what the criterion forbids, and a leaked self-test row has already
        # raised a real step's allowance once (see read_ledger's docstring on
        # the 9.4 incident). Pointing the check at a SYNTHETIC plan of record
        # makes the test ids exempt BY CONSTRUCTION: they are real here and
        # nowhere else, and this test can never again touch a real allowance.
        synthetic_plan = Path(td) / "masterplan.json"
        synthetic_plan.write_text(json.dumps({"phases": [{"id": "phase-9", "steps": [
            {"id": f"9.{n}"} for n in range(1, 6)]}]}), encoding="utf-8")
        old_mp = os.environ.get("ATTEMPT_GATE_MASTERPLAN")
        os.environ["ATTEMPT_GATE_MASTERPLAN"] = str(synthetic_plan)
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
            # cycle-3 (86.71 cycle-2 Q/A, H4): the --reason requirement on the
            # ONLY ceiling-raising path had zero coverage anywhere.
            # cycle-4 fix of a TAUTOLOGY the cycle-3 Q/A mutation-proved:
            # before_rows was captured AFTER the cmd_extend call (which ran
            # inside the preceding check's argument), so the no-row check
            # compared the ledger length to itself. Capture BEFORE the action.
            before_rows = len(read_ledger(led))
            check("operator extension WITHOUT --reason is refused",
                  cmd_extend("9.4", 1, "   ") == 2)
            check("refused extension appends NO row",
                  len(read_ledger(led)) == before_rows)
            check("operator extension WITH a reason appends its labelled row",
                  cmd_extend("9.4", 1, "self-test reason") == 0
                  and any(r.get("type") == "operator_extension"
                          and r.get("step_id") == "9.4"
                          for r in read_ledger(led)))
            # cycle-5 (cycle-4 Q/A, V1/V2): the loud fail-closed swallow in
            # verdict_outcomes was observable ONLY in a hand-run demo -- every
            # automated drive pointed the verdict ledger at an ABSENT path,
            # where emit_sequence returns [] WITHOUT raising, so the except
            # branch was unreachable and both a silent revert (V1) and a
            # fail-OPEN `return [Outcome.PASS]` (V2) survived every check.
            # Drive the branch: a DIRECTORY raises IsADirectoryError (the
            # section-10 demo fixture, now automated). Assert BOTH properties:
            # the failure is loud, and it grants no PASS exception.
            import contextlib  # noqa: PLC0415
            import io  # noqa: PLC0415
            vdir = Path(td) / "verdict_isadir"
            vdir.mkdir()
            led2 = Path(td) / "attempts2.jsonl"
            LEDGER, VERDICT_LEDGER = led2, vdir
            for _ in range(DEFAULT_MAX_ATTEMPTS):
                append_row({"ts": _now(), "type": "attempt",
                            "step_id": "9.5"}, led2)
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                d, _ = decide("9.5", read_ledger(led2))
            check("verdict-ledger read error is LOUD on stderr (V1)",
                  "verdict-ledger read failed" in buf.getvalue())
            check("read error grants NO PASS exception -- at ceiling stays deny (V2)",
                  d == "deny")

            # ---- phase-90.1 ------------------------------------------------
            # criterion 4: a claimed id must be a REAL step. These ids are all
            # well-formed dotted-numerics, so the shape regex admits every one;
            # only membership separates them.
            check("a claimed step id ABSENT from the plan of record is refused "
                  "(90.1 c4: '9.9' is well-formed but names no step)",
                  extract_step_id({"args": {"step_id": "9.9"}}) is None)
            check("appending '.1' to a real id no longer mints an allowance "
                  "(90.1 c4: '9.1.1' refused)",
                  extract_step_id({"args": {"step_id": "9.1.1"}}) is None)
            check("a digit-appended near-miss is refused (90.1 c4: '9.10')",
                  extract_step_id({"args": {"step_id": "9.10"}}) is None)
            check("a REAL step id is still admitted (90.1 c4: the check denies "
                  "only what the plan does not contain)",
                  extract_step_id({"args": {"step_id": "9.1"}}) == "9.1")
            check("the CLAIM survives validation so the denial can name it",
                  extract_step_id_claim({"args": {"step_id": "9.9"}}) == "9.9"
                  and extract_step_id_claim({"args": {"topic": "x"}}) is None)
            check("a hostile claim cannot escape the escalation dir (90.1: the "
                  "denial path is REACHED by ids the shape regex refused)",
                  "/" not in _slug("../evil") and ".." not in _slug("../../x"))

            # criterion 2: no forged exhaustion record.
            fresh = BudgetState(step_id="9.1")
            forged = False
            try:
                write_escalation(fresh, reason="some_other_reason")
            except ValueError:
                forged = True
            check("write_escalation REFUSES to forge '# BUDGET EXHAUSTED' for a "
                  "step that is not exhausted (90.1 c2)", forged)
            esc_dir_before = {p.name for p in ESCALATION_DIR.iterdir()}
            check("the refusal wrote NO file at all",
                  not any(n.startswith("escalation_some_other_reason")
                          for n in esc_dir_before))
            unk = write_escalation(BudgetState(step_id="9.9"),
                                   reason=REASON_UNKNOWN_STEP_ID,
                                   body=_unknown_step_id_body("9.9"))
            check("a non-exhaustion denial writes its OWN reason-named path "
                  "(90.1 c2)", unk.name == "escalation_unknown_step_id_9.9.md")
            check("and its body says what actually happened, not 'BUDGET "
                  "EXHAUSTED'",
                  "UNRECOGNISED STEP ID" in unk.read_text(encoding="utf-8")
                  and "BUDGET EXHAUSTED" not in unk.read_text(encoding="utf-8"))

            # criterion 3: the token ceiling FIRES. Driven by EXECUTION, not by
            # reading the constant -- one row over the line, one attempt used.
            led3 = Path(td) / "attempts3.jsonl"
            LEDGER = led3
            append_row({"ts": _now(), "type": "attempt", "step_id": "9.1",
                        "outcome": "CONDITIONAL",
                        "total_tokens": DEFAULT_MAX_TOKENS + 1}, led3)
            d3, s3 = decide("9.1", read_ledger(led3))
            check(f"ONE attempt costing {DEFAULT_MAX_TOKENS + 1:,} tokens is "
                  "DENIED on the TOKEN ceiling with 4 of 5 attempts still "
                  "unused (90.1 c3)",
                  d3 == "deny" and s3.attempts_used == 1
                  and s3.tokens_used == DEFAULT_MAX_TOKENS + 1)
            led4 = Path(td) / "attempts4.jsonl"
            LEDGER = led4
            append_row({"ts": _now(), "type": "attempt", "step_id": "9.1",
                        "outcome": "CONDITIONAL",
                        "total_tokens": DEFAULT_MAX_TOKENS - 1}, led4)
            d4, s4 = decide("9.1", read_ledger(led4))
            check("and one token UNDER the ceiling is still allowed -- so the "
                  "check discriminates rather than always denying",
                  d4 == "allow" and s4.tokens_used == DEFAULT_MAX_TOKENS - 1)

            # criterion 1/5: the row's own outcome is what gets recorded, so a
            # NO_VERDICT attempt cannot be laundered into a graded one.
            led5 = Path(td) / "attempts5.jsonl"
            LEDGER = led5
            append_row({"ts": _now(), "type": "attempt", "step_id": "9.1",
                        "outcome": "NO_VERDICT", "total_tokens": 10}, led5)
            _, s5 = decide("9.1", read_ledger(led5))
            check("a NO_VERDICT row is recorded as a DROP, not as a verdict "
                  "(90.1 c1/c5: dropped=1, verdicts_seen=0)",
                  s5.dropped == 1 and s5.verdicts_seen == 0)
            led6 = Path(td) / "attempts6.jsonl"
            LEDGER = led6
            append_row({"ts": _now(), "type": "attempt", "step_id": "9.1",
                        "outcome": "PASS", "total_tokens": 10}, led6)
            _, s6 = decide("9.1", read_ledger(led6))
            check("and a graded row IS counted as a verdict -- the probe "
                  "discriminates (dropped=0, verdicts_seen=1)",
                  s6.dropped == 0 and s6.verdicts_seen == 1)
            LEDGER = led
            check("the self-test wrote every escalation into its OWN temp dir "
                  "-- nothing leaked into handoff/current/ (the 9.4 lesson, "
                  "applied to the second channel)",
                  ESCALATION_DIR != old_e
                  and all(p.parent == ESCALATION_DIR
                          for p in ESCALATION_DIR.iterdir()))
        finally:
            LEDGER, VERDICT_LEDGER, ESCALATION_DIR = old_l, old_v, old_e
            if old_mp is None:
                os.environ.pop("ATTEMPT_GATE_MASTERPLAN", None)
            else:
                os.environ["ATTEMPT_GATE_MASTERPLAN"] = old_mp
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
