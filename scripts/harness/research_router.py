#!/usr/bin/env python3
"""phase-86.72 cycle 2 -- the CONSUMER of `research_routing`.

The cycle-1 Q/A FAILED the step for, among other things, this exact absence:
qa-verdict.js surfaced `research_routing` beside the verdict and *nothing
anywhere consumed it* -- no spawn was caused, no re-research rounds were
counted, and qa.md told the judge "the caller enforces at most 2 re-research
rounds per step" while no code enforced anything. This module is that caller
machinery, named so the claim is true.

Contract
--------
    python3 scripts/harness/research_router.py --verdict-json <path|-> \
        [--step-id <sid>] [--audit-ledger <path>] [--print-launch]

Reads a qa-verdict.js RETURN OBJECT (the captured Workflow result, as JSON),
inspects `research_routing`, and decides:

    DISPATCH  -> research_needed is true AND the per-step re-research budget
                 (TMAX_ROUNDS=2) has headroom. Emits the exact Workflow launch
                 instruction (scriptPath research-gate.js + args built from
                 research_brief_spec) on stdout as JSON.
    REFUSE    -> the signal is present but the budget is exhausted (round
                 count >= TMAX_ROUNDS) or the spec is missing/malformed.
                 Exit 3, reason on stderr -- LOUD, never silent.
    NO_SIGNAL -> research_needed absent/false/null. Exit 0, decision printed.
                 A verdict without the signal causes NO spawn -- the other
                 arm of criterion 3.

Round counting is REAL, not asserted: prior re-research rounds for the step
are counted from the attempt-gate's append-only audit stream
(handoff/audit/attempt_budget_audit.jsonl), whose rows carry the launched
workflow's name -- every research-gate.js launch for the step since the gate
was wired (2026-08-17) is one round. Pre-gate history is invisible to this
counter and that bound is stated in the output (`rounds_floor: true`).

The router NEVER launches anything itself (it has no Workflow tool); it
emits the instruction for the caller (Main, run_harness.py, or a human) to
execute. It also never touches the verdict: routing is downstream of
grading, exactly as enforceResearchRouting is upstream of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT_LEDGER = REPO / "handoff" / "audit" / "attempt_budget_audit.jsonl"

#: At most this many re-research rounds per step (the Tmax the qa.md text and
#: the qa-verdict.js guidance both state; run_harness.py's F2 leg uses the
#: same bound). Stagnation (a round adding no new read-in-full sources) is
#: judged by the CALLER from consecutive briefs -- a router cannot read
#: brief deltas; it enforces the hard count.
TMAX_ROUNDS = 2

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_REFUSED = 3


def count_research_rounds(step_id: str, ledger: Path) -> tuple[int, bool]:
    """(rounds, is_floor). Counts research-gate.js launches for the step in
    the attempt-gate audit stream. A corrupt row counts as a round for its
    step only when its step_id is readable; the stream starts 2026-08-17 so
    the count is a FLOOR over all time (stated, never hidden)."""
    rounds = 0
    if not ledger.is_file():
        return 0, True
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # unattributable -- cannot belong to this step legibly
        if (row.get("step_id") == step_id
                and "research-gate" in str(row.get("workflow", ""))):
            rounds += 1
    return rounds, True


def route(verdict: dict, step_id: str | None, ledger: Path) -> tuple[int, dict]:
    routing = verdict.get("research_routing")
    sid = step_id or verdict.get("step_id")
    out: dict = {
        "decision": None,
        "step_id": sid,
        "tmax_rounds": TMAX_ROUNDS,
        "rounds_used": None,
        "rounds_floor": True,
        "launch": None,
        "reason": None,
    }
    if not isinstance(routing, dict) or routing.get("research_needed") is not True:
        out["decision"] = "NO_SIGNAL"
        out["reason"] = ("research_routing absent or research_needed is not true "
                         "-- no spawn is caused (the no-signal arm)")
        return EXIT_OK, out
    if not sid:
        out["decision"] = "REFUSE"
        out["reason"] = "signal present but no step_id -- refusing to dispatch blind"
        return EXIT_REFUSED, out
    spec = routing.get("research_brief_spec")
    required = ("objective", "output_format", "tool_scope", "task_boundaries")
    if not isinstance(spec, dict) or not all(
            isinstance(spec.get(k), str) and spec.get(k) for k in required):
        out["decision"] = "REFUSE"
        out["reason"] = ("research_needed=true but research_brief_spec is missing "
                         f"or lacks the 4 required keys {required} -- the judge "
                         "must supply the F2 shape; refusing a blind research spawn")
        return EXIT_REFUSED, out
    rounds, floor = count_research_rounds(sid, ledger)
    out["rounds_used"], out["rounds_floor"] = rounds, floor
    if rounds >= TMAX_ROUNDS:
        out["decision"] = "REFUSE"
        out["reason"] = (f"re-research budget exhausted: {rounds} >= "
                         f"TMAX_ROUNDS={TMAX_ROUNDS} counted rounds for {sid} "
                         "-- escalate to the operator instead of looping")
        return EXIT_REFUSED, out
    out["decision"] = "DISPATCH"
    out["launch"] = {
        "tool": "Workflow",
        "scriptPath": ".claude/workflows/research-gate.js",
        "args": {
            "step_id": sid,
            "topic": spec["objective"],
            "tier": "moderate",
            "internal_scope": (spec["tool_scope"] + " -- BOUNDARIES: "
                              + spec["task_boundaries"]),
            "brief_path": f"handoff/current/research_brief_{sid}.md",
            "extra": ("Requested by a Q/A verdict via research_routing "
                      "(phase-86.72). Output format wanted: "
                      + spec["output_format"]),
        },
    }
    out["reason"] = (f"round {rounds + 1} of {TMAX_ROUNDS} for {sid} -- "
                     "execute the launch object with the Workflow tool")
    return EXIT_OK, out


def _self_test() -> int:
    import tempfile
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    spec = {"objective": "o", "output_format": "f",
            "tool_scope": "t", "task_boundaries": "b"}
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "audit.jsonl"
        # no-signal arm: absent, false, and null all cause NO spawn
        for v in ({}, {"research_routing": None},
                  {"research_routing": {"research_needed": False}},
                  {"research_routing": {"research_needed": None}}):
            rc, out = route(v, "9.1", led)
            check(f"no-signal arm causes NO spawn ({v!r:.40})",
                  rc == EXIT_OK and out["decision"] == "NO_SIGNAL"
                  and out["launch"] is None)
        # signal arm: dispatches with the researcher target, never the judge
        rc, out = route({"research_routing": {"research_needed": True,
                                              "research_brief_spec": spec}},
                        "9.1", led)
        check("signal arm DISPATCHES", rc == EXIT_OK
              and out["decision"] == "DISPATCH")
        check("the routed target is research-gate.js, never qa-verdict.js",
              out["launch"]["scriptPath"].endswith("research-gate.js")
              and "qa-verdict" not in json.dumps(out["launch"]))
        check("the spec is ECHOED into the launch, not fabricated",
              out["launch"]["args"]["topic"] == "o"
              and "b" in out["launch"]["args"]["internal_scope"])
        # budget: 2 prior rounds -> REFUSE, loud
        led.write_text("".join(json.dumps({"step_id": "9.1",
                                           "workflow": "research-gate.js"}) + "\n"
                               for _ in range(2)), encoding="utf-8")
        rc, out = route({"research_routing": {"research_needed": True,
                                              "research_brief_spec": spec}},
                        "9.1", led)
        check("TMAX_ROUNDS=2 exhausted -> REFUSE at exit 3",
              rc == EXIT_REFUSED and out["decision"] == "REFUSE"
              and "exhausted" in out["reason"])
        # qa-verdict launches for the step do NOT count as research rounds
        led.write_text(json.dumps({"step_id": "9.2",
                                   "workflow": "qa-verdict.js"}) + "\n",
                       encoding="utf-8")
        rc, out = route({"research_routing": {"research_needed": True,
                                              "research_brief_spec": spec}},
                        "9.2", led)
        check("qa launches do not consume the research budget",
              out["rounds_used"] == 0 and out["decision"] == "DISPATCH")
        # malformed spec refuses rather than dispatching blind
        rc, out = route({"research_routing": {"research_needed": True,
                                              "research_brief_spec": {"objective": "o"}}},
                        "9.3", led)
        check("missing spec keys -> REFUSE, never a blind spawn",
              rc == EXIT_REFUSED and "4 required keys" in out["reason"])
        # missing step id refuses
        rc, out = route({"research_routing": {"research_needed": True,
                                              "research_brief_spec": spec}},
                        None, led)
        check("no step_id -> REFUSE", rc == EXIT_REFUSED)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdict-json", help="path to the captured qa-verdict return (or - for stdin)")
    ap.add_argument("--step-id", default=None)
    ap.add_argument("--audit-ledger", default=None, help="override (testing)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.verdict_json:
        print("--verdict-json is required (or --self-test)", file=sys.stderr)
        return EXIT_INVALID
    raw = (sys.stdin.read() if args.verdict_json == "-"
           else Path(args.verdict_json).read_text(encoding="utf-8"))
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"verdict JSON unparseable: {exc} -- refusing to route", file=sys.stderr)
        return EXIT_INVALID
    ledger = Path(args.audit_ledger) if args.audit_ledger else AUDIT_LEDGER
    rc, out = route(verdict, args.step_id, ledger)
    print(json.dumps(out, indent=2))
    if rc == EXIT_REFUSED:
        print(f"REFUSED: {out['reason']}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
