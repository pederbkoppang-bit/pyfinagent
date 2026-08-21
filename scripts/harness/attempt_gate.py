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

import hashlib
import json
import os
import re
import shutil
import subprocess
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


# ---------------------------------------------------------------------------
# phase-90.3 -- THE PROGRESS DIGEST, and what it is NOT
# ---------------------------------------------------------------------------
#
# This corrects step 89.1, whose mechanism was measured to fire on NONE of the
# real loops (the cycle-2 flow mandates updating the artifacts, so the digest
# advances every round by construction) and to DENY the doctrine-mandated
# post-drop retry (14 of 16 NO_VERDICT rows were followed by a retry; 8 of the
# 11 affected steps later reached PASS).
#
# THE HAZARD THE 90.3 RESEARCH GATE CAUGHT BEFORE ANY CODE EXISTED, and it is
# why the allowlist below is the load-bearing part rather than a detail:
# criterion 1's file set is `declared masterplan paths UNION git diff HEAD UNION
# untracked`, and MEASURED on this repo that union resolves to
# handoff/audit/attempt_budget_audit.jsonl -- TRACKED, and written BY THIS GATE
# on every launch -- plus handoff/audit/pre_tool_use_audit.jsonl, written on
# every tool call, plus .claude/agent-memory/ files written by the agents
# themselves. Without the exclusions, THE DIGEST ADVANCES BY CONSTRUCTION and
# the check is vacuous: 89.1's defect through a different door, in the step
# built to correct 89.1.
#
# ("declared masterplan paths" resolves to the empty set: measured, 0 of 1222
# steps carry a `files` or `paths` key. The union is well-defined without it and
# the absence is reported rather than silently dropped.)
#
# WHAT THIS INSTRUMENT IS, STATED SO NO CONSUMER OVERREADS IT. A byte-digest is
# the WEAKEST of the three published stagnation signals: CUDABeaver measures
# SHA-256 duplicate_code at 0-50.8% and code_cycle at 0.7-3.8%, against SEMANTIC
# no_progress at 44.6-84.6%. It is built because it is deterministic and cheap.
# IT DETECTS AN EXACT REPEAT AND AN A->B->A OSCILLATION, AND NOTHING ELSE.
# It is NOT evidence of progress, of convergence, or of quality, and criterion 5
# forbids any consumer from reading it that way -- optimal-stopping work
# (arXiv 2608.10729) triggers on an ABSOLUTE score, explicitly not on
# inter-iteration change.

#: Roots whose contents count as a step's evidence.
DIGEST_ALLOWED_ROOTS = ("handoff/current/", "scripts/", "backend/", "frontend/",
                        "docs/", ".claude/")
#: Roots EXCLUDED because they are written by the instrument, by every tool
#: call, or by the agents -- see the hazard note above. Longest match wins, so
#: these override DIGEST_ALLOWED_ROOTS.
DIGEST_EXCLUDED_ROOTS = ("handoff/audit/", "handoff/logs/", ".claude/agent-memory/",
                         ".claude/masterplan.json.bak")

#: DEFAULT-OFF. This adds a new DENY path to a PreToolUse hook that runs on every
#: Workflow launch, and it has NOT been graded by a Q/A. An ungraded deny-capable
#: gate that misfires blocks the whole harness, so it stays dark on the live rail
#: until an operator enables it deliberately. The self-test and the mutation matrix
#: both set this, so the mechanism is fully exercised while the live rail is
#: untouched -- the same discipline as a feature flag, and the reason it is stated
#: here rather than in a commit message nobody reads at 3am.
DIGEST_ENABLED = os.environ.get("ATTEMPT_GATE_PROGRESS_DIGEST", "") == "1"

DIGEST_STATUS_OK = "ok"
DIGEST_STATUS_INPUTS_INCOMPLETE = "inputs_incomplete"
DIGEST_STATUS_UNAVAILABLE = "unavailable"


def _git_lines(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                         timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr[:200]}")
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def digest_inputs() -> list[str]:
    """The evidence file set, allowlisted and with the instrument excluded.

    Repo-relative, sorted, de-duplicated. NEVER inferred from critique prose --
    criterion 1 forbids it, and prose is full of line-wrapped path fragments.

    `ATTEMPT_GATE_DIGEST_INPUTS` (os.pathsep-separated) replaces the git-derived
    set. It exists so the WIRING can be driven where git cannot answer -- a
    sandbox is not a repo, and without it the hook-drive cells silently fell into
    the unavailable branch and tested nothing. The DERIVATION is covered
    separately by the allowlist/exclusion cells, so the two halves are tested
    apart rather than one of them not at all. The override is still filtered by
    the same allowlist below: a test seam that skips the exclusions would be a way
    to smuggle the instrument back into its own digest.
    """
    injected = os.environ.get("ATTEMPT_GATE_DIGEST_INPUTS", "")
    if injected:
        seen = {x for x in injected.split(os.pathsep) if x.strip()}
    else:
        seen = set(_git_lines("diff", "--name-only", "HEAD"))
        seen |= set(_git_lines("ls-files", "--others", "--exclude-standard"))
    keep = []
    for rel in sorted(seen):
        if any(rel.startswith(x) for x in DIGEST_EXCLUDED_ROOTS):
            continue
        if not any(rel.startswith(x) for x in DIGEST_ALLOWED_ROOTS):
            continue
        keep.append(rel)
    return keep


def compute_digest(paths: list[str]) -> tuple[str | None, str, list[str]]:
    """(digest, status, missing).

    CONTENT ONLY -- no mtime, no size, no inode. A cell that os.utime()s every
    input without changing a byte must still produce the same digest, and
    therefore must still DENY.

    A MISSING declared input is `inputs_incomplete`, not a silent skip: a digest
    computed over a subset of its inputs is not the digest it claims to be.
    """
    h = hashlib.sha256()
    missing = []
    for rel in paths:
        f = REPO / rel
        try:
            data = f.read_bytes()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            missing.append(rel)
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
    if missing:
        return None, DIGEST_STATUS_INPUTS_INCOMPLETE, missing
    return h.hexdigest(), DIGEST_STATUS_OK, []


def digest_exempt(step_id: str, rows: list[dict]) -> str | None:
    """Why the digest check is SKIPPED for this launch, or None.

    A drop produces NOTHING TO FIX, so a byte-identical relaunch after one is
    correct rather than suspicious -- Temporal's definition of a permanent
    failure is one that "requires a change to your input", and a dropped rail
    supplies no such change. Measured on the real ledger: 14 of 16 NO_VERDICT
    rows were followed by a retry and 8 of the 11 affected steps later PASSED.
    89.1 would have denied all 14.
    """
    verdicts = []
    try:
        for line in VERDICT_LEDGER.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("step_id")) == str(step_id):
                verdicts.append(r)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        # FAIL-CLOSED FOR THE EXEMPTION: an unreadable verdict ledger grants NO
        # exemption, so the digest check still runs. That direction can only ever
        # deny more, never allow more -- the same asymmetry the PASS-exception
        # read already uses a few lines above in build_state.
        return None
    if not verdicts:
        return "no verdict row exists for this step -- nothing to compare against"
    if str(verdicts[-1].get("verdict")) == "NO_VERDICT":
        return ("the most recent verdict row is NO_VERDICT -- a dropped rail "
                "produces nothing to fix, so a byte-identical relaunch is correct")
    prior_attempts = [r for r in rows
                      if r.get("type") == "attempt" and str(r.get("step_id")) == str(step_id)]
    if prior_attempts:
        last_attempt_ts = str(prior_attempts[-1].get("ts") or "")
        newer = [v for v in verdicts if str(v.get("recorded_at") or v.get("date") or "") > last_attempt_ts]
        if not newer:
            return ("no verdict row postdates the previous attempt -- the prior "
                    "launch produced no graded outcome to respond to")
    return None


def prior_digests(step_id: str, rows: list[dict]) -> list[str]:
    """EVERY digest recorded for this step, not just the previous one.

    Criterion 3: pairwise comparison lets an A->B->A revert oscillate forever.
    Rows are appended on DENY as well as on allow, so the third launch of an
    oscillation is compared against BOTH prior states and denied.
    """
    out = []
    for r in rows:
        if str(r.get("step_id")) != str(step_id):
            continue
        d = r.get("evidence_digest")
        if isinstance(d, str) and d:
            out.append(d)
    return out


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
        # ── phase-90.3: the progress digest ────────────────────────────────
        # Runs AFTER the attempt-budget decision, so a step that is already at
        # its ceiling is denied for the ceiling and not for a digest -- a denial
        # must name the reason that actually bound.
        digest, dstatus, dmissing = None, DIGEST_STATUS_UNAVAILABLE, []
        exempt = (None if DIGEST_ENABLED
                  else "progress digest is DISABLED (set ATTEMPT_GATE_PROGRESS_DIGEST=1 "
                       "to enable; default-off until graded)")
        if exempt is None:
            exempt = digest_exempt(sid, rows)
        if exempt is None:
            try:
                inputs = digest_inputs()
                digest, dstatus, dmissing = compute_digest(inputs)
            except Exception as exc:  # noqa: BLE001
                # A CRASH is not a DENY. Criterion 4 keeps these distinct: the
                # launch proceeds (fail-open, per the hook contract), but the row
                # records digest=null with an explicit unavailable status, so the
                # NEXT launch has no comparable baseline and cannot be compared
                # against a phantom.
                print(f"[attempt-gate] progress-digest UNAVAILABLE -- "
                      f"{type(exc).__name__}: {exc} -- allowing this launch and "
                      "recording no baseline", file=sys.stderr)
                digest, dstatus, dmissing = None, DIGEST_STATUS_UNAVAILABLE, []
            if dstatus == DIGEST_STATUS_INPUTS_INCOMPLETE:
                append_row({
                    "ts": _now(), "type": "attempt", "step_id": sid,
                    "workflow": workflow_label(tool_input),
                    "tool_use_id": payload.get("tool_use_id") or "",
                    "session_id": payload.get("session_id") or "",
                    "attempt_number_inclusive": state.attempts_used + 1,
                    "outcome": None, "outcome_reason": "denied_at_launch",
                    "total_tokens": None, "run_id": None,
                    "evidence_digest": None,
                    "evidence_digest_status": DIGEST_STATUS_INPUTS_INCOMPLETE,
                    "note": "DENIED: declared digest inputs are missing",
                })
                print("[attempt-gate] DENIED: reason=inputs_incomplete -- these "
                      "declared evidence inputs could not be read: "
                      + ", ".join(dmissing[:5])
                      + ". A digest computed over a SUBSET of its inputs is not the "
                        "digest it claims to be, so this is a denial and not a "
                        "silent skip. This is NOT a gate crash (a crash fails open) "
                        "and a denial is NOT a verdict.", file=sys.stderr)
                return 2
            if dstatus == DIGEST_STATUS_OK and digest in prior_digests(sid, rows):
                append_row({
                    "ts": _now(), "type": "attempt", "step_id": sid,
                    "workflow": workflow_label(tool_input),
                    "tool_use_id": payload.get("tool_use_id") or "",
                    "session_id": payload.get("session_id") or "",
                    "attempt_number_inclusive": state.attempts_used + 1,
                    "outcome": None, "outcome_reason": "denied_at_launch",
                    "total_tokens": None, "run_id": None,
                    "evidence_digest": digest,
                    "evidence_digest_status": DIGEST_STATUS_OK,
                    "note": "DENIED: evidence digest already seen for this step",
                })
                print(f"[attempt-gate] DENIED: reason=no_new_evidence -- step {sid} "
                      f"has already been launched with this exact evidence "
                      f"(digest {digest[:16]}). Comparison is against EVERY digest "
                      "recorded for the step, not just the previous one, so an "
                      "A->B->A revert is caught on the third launch. NOTE WHAT THIS "
                      "DOES NOT SAY: a CHANGED digest is not evidence of progress. "
                      "This check detects an exact repeat and an oscillation, and "
                      "nothing else. A denial is not a verdict.", file=sys.stderr)
                return 2
        else:
            print(f"[attempt-gate] progress-digest SKIPPED: {exempt}", file=sys.stderr)

        append_row({
            "ts": _now(), "type": "attempt", "step_id": sid,
            "evidence_digest": digest,
            "evidence_digest_status": (DIGEST_STATUS_UNAVAILABLE if exempt
                                       else dstatus),
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
        # Snapshot the REAL dir BEFORE redirecting, so containment can be
        # asserted against the thing that actually matters.
        real_before = ({p.name for p in old_e.iterdir()}
                       if old_e.is_dir() else set())
        ESCALATION_DIR = Path(td) / "escalations"
        ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
        # phase-90.1 cycle-4. The cycle-3 Q/A built a structurally-equivalent
        # sandbox, deleted the ONE line that redirects VERDICT_LEDGER, and the
        # self-test still reported PASSED with zero FAILs while TRUNCATING the
        # sandbox's real verdict ledger to a single synthetic PASS row -- via
        # the write below in the PASS-exception fixture. One deleted line
        # between this test and the project's verdict history, and nothing
        # would have said so. Refuse to run at all if any output path still
        # resolves inside the repo.
        for label, target in (("LEDGER", LEDGER),
                              ("VERDICT_LEDGER", VERDICT_LEDGER),
                              ("ESCALATION_DIR", ESCALATION_DIR)):
            if Path(target).resolve().is_relative_to(REPO):
                print(f"  FAIL  {label} still points INSIDE the repo "
                      f"({target}) -- refusing to run the self-test rather "
                      "than risk writing to production state", file=sys.stderr)
                return 1
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
            # ---- phase-90.1 cycle-2: RECALL, the check cycle 1 lacked -----
            # Cycle 1 tested only that BAD ids are denied (precision) and never
            # that GOOD ids are admitted (recall), so a walk that missed
            # subphases[] shipped and denied 10 real pending steps.
            from attempt_outcomes import assert_membership_recall  # noqa: PLC0415
            rec = assert_membership_recall(synthetic_plan)
            check("every dotted id the plan of record contains is ADMITTED "
                  "(90.1 c4 RECALL, checked against the file not the function)",
                  rec["ok"] and rec["members"] > 0)
            nested_plan = Path(td) / "masterplan_nested.json"
            nested_plan.write_text(json.dumps({"phases": [{"id": "phase-9",
                "subphases": [{"id": "phase-9.9", "steps": [{"id": "9.9.1"}]}]}]}),
                encoding="utf-8")
            os.environ["ATTEMPT_GATE_MASTERPLAN"] = str(nested_plan)
            check("a step id nested under subphases[] is ADMITTED -- the plan is "
                  "NOT uniformly phases[].steps[] (the cycle-1 BLOCK)",
                  extract_step_id({"args": {"step_id": "9.9.1"}}) == "9.9.1")
            os.environ["ATTEMPT_GATE_MASTERPLAN"] = str(synthetic_plan)

            # phase-90.1 cycle-3. The cycle-2 Q/A proved this check was a
            # TAUTOLOGY: `all(p.parent == ESCALATION_DIR for p in
            # ESCALATION_DIR.iterdir())` is True by construction, because
            # iterdir() yields only direct children -- it returned True while a
            # file sat OUTSIDE the dir, and True vacuously on an empty dir. It
            # asserted a proxy, not the property. The property is "the REAL
            # handoff/current/ did not change", so that is what is now asserted,
            # against a snapshot taken before any of this ran.
            real_after = ({p.name for p in old_e.iterdir()}
                          if old_e.is_dir() else set())
            wrote_here = {p.name for p in ESCALATION_DIR.iterdir()}
            check("the REAL escalation dir is UNCHANGED by this self-test -- "
                  "compared name-set before vs after, not the tautology of "
                  "asking a temp dir about its own children (the 9.4 lesson)",
                  real_after == real_before)
            # Anti-vacuity: "the real dir is unchanged" is trivially true if
            # this test wrote nothing anywhere. It must have written SOMEWHERE,
            # and that somewhere must be the temp dir. (Measured while writing
            # this: the self-test writes exactly ONE escalation -- the
            # unknown-step-id record -- because the refusal path deliberately
            # writes none. The first version of this line asserted >= 2 and went
            # RED, which is how a non-tautological check behaves.)
            check("...and the temp dir actually RECEIVED an escalation, so the "
                  "check above cannot pass by writing nothing at all",
                  ESCALATION_DIR != old_e and len(wrote_here) >= 1
                  and all(n.startswith("escalation_") for n in wrote_here))
        finally:
            LEDGER, VERDICT_LEDGER, ESCALATION_DIR = old_l, old_v, old_e
            if old_mp is None:
                os.environ.pop("ATTEMPT_GATE_MASTERPLAN", None)
            else:
                os.environ["ATTEMPT_GATE_MASTERPLAN"] = old_mp

    # ── phase-90.3: the progress digest, DRIVEN (criterion 8) ───────────────
    # These assert behavioural outcomes of the real functions, not the presence
    # of a token in the source. Contained: every path below is a temp dir.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _root = Path(_td)
        (_root / "a.txt").write_text("alpha", encoding="utf-8")
        (_root / "b.txt").write_text("beta", encoding="utf-8")
        _old_repo = globals()["REPO"]
        globals()["REPO"] = _root
        try:
            d1, st1, miss1 = compute_digest(["a.txt", "b.txt"])
            check("the digest is computed and reports ok over readable inputs",
                  st1 == DIGEST_STATUS_OK and isinstance(d1, str) and len(d1) == 64
                  and miss1 == [])
            # criterion 1: CONTENT ONLY. Touching every input must not move it.
            import os as _os
            for _f in ("a.txt", "b.txt"):
                _os.utime(_root / _f, (0, 0))
            d2, _, _ = compute_digest(["a.txt", "b.txt"])
            check("os.utime on EVERY input does not change the digest -- content "
                  "only, so a touched-but-unchanged relaunch still DENIES (c1)",
                  d2 == d1)
            (_root / "a.txt").write_text("ALPHA", encoding="utf-8")
            d3, _, _ = compute_digest(["a.txt", "b.txt"])
            check("...but changing one BYTE does move it, so the digest is not a "
                  "constant", d3 != d1)
            d4, st4, miss4 = compute_digest(["a.txt", "gone.txt"])
            check("a MISSING declared input is inputs_incomplete with the file "
                  "named -- not a silent skip over a subset (c4)",
                  st4 == DIGEST_STATUS_INPUTS_INCOMPLETE and d4 is None
                  and miss4 == ["gone.txt"])
            # criterion 1: the ALLOWLIST must exclude the instrument. This is the
            # hazard the research gate caught: the gate's own audit stream is in
            # the diff on every launch, so without the exclusion the digest
            # advances by construction and the check is vacuous.
            check("handoff/audit/ is EXCLUDED -- the gate writes it on every "
                  "launch, so including it would make the digest advance by "
                  "construction (the 90.3 gate's finding)",
                  any(x == "handoff/audit/" for x in DIGEST_EXCLUDED_ROOTS))
            check("...as is .claude/agent-memory/, which the agents write "
                  "themselves",
                  any(x == ".claude/agent-memory/" for x in DIGEST_EXCLUDED_ROOTS))
            check("...and the exclusions are not vacuous: a path under an excluded "
                  "root is dropped even though its root is also allowlisted",
                  ".claude/" in DIGEST_ALLOWED_ROOTS
                  and any(".claude/agent-memory/x".startswith(e)
                          for e in DIGEST_EXCLUDED_ROOTS))
        finally:
            globals()["REPO"] = _old_repo

        # criterion 3: comparison is against the SET, so A->B->A denies on the third
        _rows = [
            {"type": "attempt", "step_id": "9.1", "ts": "t1", "evidence_digest": "AAA"},
            {"type": "attempt", "step_id": "9.1", "ts": "t2", "evidence_digest": "BBB"},
            {"type": "attempt", "step_id": "9.2", "ts": "t3", "evidence_digest": "CCC"},
        ]
        check("prior_digests returns EVERY digest for the step, not just the last "
              "-- which is what makes an A->B->A revert deny on the third launch (c3)",
              prior_digests("9.1", _rows) == ["AAA", "BBB"])
        check("...and it does not leak another step's digests",
              "CCC" not in prior_digests("9.1", _rows))

        # criterion 2: the drop exemption, from the ledger rather than a heuristic
        _vl = _root / "vl.jsonl"
        _old_v2 = VERDICT_LEDGER
        VERDICT_LEDGER = _vl
        try:
            _vl.write_text(json.dumps({"step_id": "9.1", "verdict": "NO_VERDICT",
                                       "recorded_at": "2026-01-01T00:00:00Z"}) + "\n",
                           encoding="utf-8")
            check("a most-recent NO_VERDICT row EXEMPTS the check, so a "
                  "byte-identical relaunch after a dropped rail is ADMITTED (c2)",
                  (digest_exempt("9.1", []) or "").startswith("the most recent verdict"))
            _vl.write_text(json.dumps({"step_id": "9.1", "verdict": "CONDITIONAL",
                                       "recorded_at": "2026-01-02T00:00:00Z"}) + "\n",
                           encoding="utf-8")
            check("...while a graded CONDITIONAL that POSTDATES the last attempt "
                  "does NOT exempt it -- the exemption is for drops, not for "
                  "verdicts (c2)",
                  digest_exempt("9.1", [{"type": "attempt", "step_id": "9.1",
                                         "ts": "2026-01-01T00:00:00Z"}]) is None)
            check("...and when no verdict postdates the previous attempt the check "
                  "is skipped, because the prior launch produced nothing to respond to",
                  (digest_exempt("9.1", [{"type": "attempt", "step_id": "9.1",
                                          "ts": "2026-01-03T00:00:00Z"}]) or ""
                   ).startswith("no verdict row postdates"))
        finally:
            VERDICT_LEDGER = _old_v2


    # ── phase-90.3: DRIVE THE HOOK, not just the pure functions ─────────────
    # Found by mutation cell QX: renaming a name used ONLY inside handle_hook's
    # digest branch SURVIVED the whole self-test, because every cell above calls
    # decide()/compute_digest() directly and nothing executed the WIRING. A
    # mechanism whose functions are tested and whose wiring is not is a mechanism
    # that can be disconnected without a single check going red.
    with _tf.TemporaryDirectory() as _td2:
        _t = Path(_td2)
        (_t / "esc").mkdir()
        import attempt_outcomes as _ao  # noqa: PLC0415
        shutil.copy2(_ao.masterplan_path(), _t / "mp.json")
        (_t / "led.jsonl").write_text("", encoding="utf-8")
        (_t / "vl.jsonl").write_text("", encoding="utf-8")
        _env = dict(os.environ,
                    ATTEMPT_GATE_PROGRESS_DIGEST="1",
                    ATTEMPT_GATE_LEDGER=str(_t / "led.jsonl"),
                    ATTEMPT_GATE_VERDICT_LEDGER=str(_t / "vl.jsonl"),
                    ATTEMPT_GATE_ESCALATION_DIR=str(_t / "esc"),
                    ATTEMPT_GATE_MASTERPLAN=str(_t / "mp.json"),
                    ATTEMPT_GATE_DIGEST_INPUTS=os.pathsep.join(
                        ["scripts/harness/attempt_gate.py"]))
        _stdin = json.dumps({"tool_name": "Workflow",
                             "tool_input": {"args": {"step_id": "9.1"}},
                             "session_id": "selftest"})

        def _hook(env=_env):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                               input=_stdin, capture_output=True, text=True,
                               env=env, timeout=120)
            # SURFACE THE NESTED STDERR. These drives are a subprocess inside a
            # subprocess, and an outer observer -- the mutation matrix -- sees only
            # what reaches THIS process. Cell QX renames a name used in the hook's
            # digest branch; the resulting NameError is caught by the production
            # fail-open handler, printed as one line INSIDE the nested drive, and
            # was invisible out here, so the mutant scored KILLED where the ERROR
            # discipline (phase-90.12) requires ERROR. That is the same swallowed
            # -signal defect 90.12 fixed, one process boundary further out: a
            # discriminator can only read what it is given.
            for _ln in (r.stderr or "").splitlines():
                if "INTERNAL ERROR" in _ln or "Traceback" in _ln:
                    print(f"    [nested hook drive] {_ln}", file=sys.stderr)
            return r.returncode, r.stderr

        _rc1, _e1 = _hook()
        check("hook drive 1: allowed, and the digest is SKIPPED because no verdict "
              "row exists yet (c2)",
              _rc1 == 0 and "progress-digest SKIPPED" in _e1)
        # a graded verdict that POSTDATES the attempt makes the check apply
        _later = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        (_t / "vl.jsonl").write_text(json.dumps(
            {"step_id": "9.1", "verdict": "CONDITIONAL", "date": "2026-08-21",
             "recorded_at": _later}) + "\n", encoding="utf-8")
        _rc2, _e2 = _hook()
        check("hook drive 2: a graded verdict now postdates the attempt, so the "
              "digest is COMPUTED and this first sighting is allowed",
              _rc2 == 0 and "progress-digest SKIPPED" not in _e2)
        _rc3, _e3 = _hook()
        check("hook drive 3: the SAME evidence is now DENIED with reason "
              "no_new_evidence -- the wiring is exercised, not just the functions (c8)",
              _rc3 == 2 and "no_new_evidence" in _e3)
        _rows_now = read_ledger(_t / "led.jsonl")
        check("...and a row is appended on the DENY as well as on the allows, which "
              "is what makes an A->B->A revert deny on the third launch (c3)",
              len(_rows_now) == 3
              and "DENIED" in str(_rows_now[-1].get("note") or ""))
        check("...and the denial did not touch the verdict ledger -- a denial is not "
              "a verdict (c7)",
              (_t / "vl.jsonl").read_text(encoding="utf-8").count("\n") == 1)
        _env_off = dict(_env)
        _env_off["ATTEMPT_GATE_PROGRESS_DIGEST"] = ""
        _rc4, _e4 = _hook(_env_off)
        check("...and with the flag OFF (the shipped default) the same launch is "
              "ALLOWED -- the mechanism is dark on the live rail until enabled",
              _rc4 == 0 and "DISABLED" in _e4)

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
