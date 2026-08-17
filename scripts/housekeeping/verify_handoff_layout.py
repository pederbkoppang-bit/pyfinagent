"""phase-4.16.2 handoff layout verifier.

Asserts:
1. `handoff/current/` contains only `_templates/*`, rolling files,
   and in-progress-step files (no done-step files).
2. `handoff/` root contains no `*_audit.json*` files (they belong in
   `handoff/audit/`).
3. `handoff/` root contains no `*.log` files (they belong in
   `handoff/logs/`).

Exit 0 when clean; 1 when any invariant is violated, with a
diff list printed to stdout so the caller sees exactly what to move.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# phase-75.11.4: the step-id recogniser is now SHARED with
# backfill_handoff_archive.py so the two readers cannot drift. Explicit
# sys.path insert because this file is also loaded by absolute path via
# importlib (backend/tests/test_phase_36_7_*.py::_load_script), where the
# containing directory is NOT on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_naming import is_archivable, resolve_step_id  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HANDOFF = REPO / "handoff"
CURRENT = HANDOFF / "current"
MASTERPLAN = REPO / ".claude" / "masterplan.json"

ROLLING_KEEP = {
    "contract.md",
    "experiment_results.md",
    "evaluator_critique.md",
    "research_brief.md",
    "research.md",
    "research_plan.md",
    "harness_log.md",
    # phase-81.0 -- MUST stay in sync with the identically-named set in
    # scripts/housekeeping/backfill_handoff_archive.py. The machine-readable
    # verdict is read by auto-commit-and-push.sh by literal path; omitting it
    # here is what let the 2026-07-24 sweep disarm the verdict gate.
    "evaluator_critique.json",
}

# phase-81.0: mirrors backfill_handoff_archive.py. Per-step verdict JSONs are
# the current convention and must not be reported as strays.
ROLLING_KEEP_PREFIXES = ("evaluator_critique_",)


def _is_rolling_keep(name: str) -> bool:
    """True if `name` legitimately lives in handoff/current/ (a hook reads it)."""
    if name in ROLLING_KEEP:
        return True
    return name.startswith(ROLLING_KEEP_PREFIXES) and name.endswith(".json")

# phase-75.11.4: RETIRED IN PLACE. This PREFIX-only pattern matched 0 of the
# 727 files in handoff/current/ on 2026-08-17, which made the done-step arm
# below UNREACHABLE -- the invariant this script exists to assert could not
# fire at all. Superseded by handoff_naming.resolve_step_id(), which accepts
# this legacy form AND the `<base>_<sid>.md` form every writer actually uses.
# Kept as a named constant so the drift is documented rather than deleted.
STEP_ID_RE = re.compile(r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$")

# phase-36.7: LIVE PRODUCTION STATE FILES that merely LOOK like audit output.
#
# This verifier used to print
#   "handoff/kill_switch_audit.jsonl is audit output; move to handoff/audit/"
# for the kill switch's ONLY persistence file. An operator/agent obeying that
# instruction (via backfill_handoff_archive.py) left the switch DISARMED on the
# next restart -- `_load_from_audit` replays that exact path to restore
# sod_nav/peak_nav, so after the move a 50% drawdown returned any_breached=False
# (measured 2026-07-26). The verifier DEMANDING the move is what made the defect
# self-perpetuating rather than a one-off, so the allowlist has to live here too.
#
# Keep byte-identical to `backfill_handoff_archive.py::HANDOFF_ROOT_KEEP`.
HANDOFF_ROOT_KEEP = {
    "kill_switch_audit.jsonl",
}

# phase-36.8: the audit ARCHIVES are safety-relevant and MUST NOT be pruned.
# Keep this set byte-identical between the two housekeeping scripts; the test
# backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py::
# test_phase_36_8_both_housekeeping_scripts_protect_the_audit_archives parses
# both by AST and fails if they drift.
#
# WHY NO CAP / NO OLDEST-FIRST PRUNING (criterion 3, decided on measurement):
# `kill_switch._load_from_audit` merges these files on every boot to restore the
# trailing high-water mark, and 100% of the live book's baselines come from them
# today. The TRUE peak (24666.57) lives in the OLDEST file, so an
# oldest-first cap would delete the row the kill switch depends on. Measured
# 2026-07-26: 897 rows across 5 files, boot cost 0.95 ms total (1.06 us/row) --
# so growth is accepted as a bounded cost rather than capped. All five files are
# git-tracked, which is the recoverability backstop.
AUDIT_KEEP_GLOBS = (
    "kill_switch_audit*.jsonl",
)


def _statuses() -> dict[str, str]:
    with MASTERPLAN.open() as f:
        mp = json.load(f)
    out: dict[str, str] = {}
    for p in mp.get("phases", []):
        for s in p.get("steps", []):
            sid = s.get("id")
            if sid:
                out[str(sid)] = str(s.get("status") or "pending")
    return out


def main() -> int:
    failures: list[str] = []
    # phase-75.11.4: files that carry NO step id are reported but do NOT fail.
    # See the `no_step_id` note below for why that is the documented invariant
    # rather than a loosened one.
    no_step_id: list[str] = []
    unknown_step: list[str] = []

    if CURRENT.exists():
        statuses = _statuses()
        for p in CURRENT.iterdir():
            if p.is_dir():
                continue
            if _is_rolling_keep(p.name) or p.name.startswith("."):
                continue
            ref = resolve_step_id(p.name)
            if ref is None:
                # phase-75.11.4: this used to be a FAILURE ("has no step-id
                # prefix; move to handoff/archive/misc/"). It was the single
                # largest false class -- 664 of 667 findings on 2026-08-17 --
                # and it is NOT the documented invariant. `.claude/rules/
                # research-gate.md` states the rule as "handoff/current/
                # contains NO files belonging to status=done steps"; a day
                # report or an incident note belongs to no step, so it
                # violates nothing. Reported for visibility, not counted.
                no_step_id.append(p.name)
                continue
            sid = ref.sid
            # Masterplan step ids are inconsistent across buckets: some are
            # bare `4.14.1`, others `phase-6.1`. Try both forms.
            status = statuses.get(sid) or statuses.get(f"phase-{sid}")
            if status is None:
                # An id no masterplan step claims (a PHASE id such as `78`, or
                # a step since removed). Not a violation, and specifically NOT
                # archivable -- the 2026-07-25 sweep moved exactly this class.
                unknown_step.append(f"{p.name} (sid={sid}, no such step)")
            elif is_archivable(status):
                failures.append(
                    f"current/{p.name} belongs to {status} step {sid}; "
                    f"move to handoff/archive/phase-{sid}/"
                )

    if HANDOFF.exists():
        for p in HANDOFF.iterdir():
            if p.is_dir():
                continue
            if p.name in HANDOFF_ROOT_KEEP:
                # phase-36.7: live state file, not layout dirt. Do not flag.
                continue
            if p.name.endswith(".log"):
                failures.append(
                    f"handoff/{p.name} is a log; move to handoff/logs/"
                )
            if p.name.endswith("_audit.json") or p.name.endswith("_audit.jsonl"):
                failures.append(
                    f"handoff/{p.name} is audit output; move to handoff/audit/"
                )

    # phase-75.11.4: print the non-failing classes FIRST so they are never
    # mistaken for a clean tree, and so a reader can see what the checker
    # deliberately does not police.
    if no_step_id:
        print(f"[info] {len(no_step_id)} file(s) in current/ carry no step id "
              "(rolling files, day reports, incident notes) -- not an invariant "
              "violation; left in place")
    if unknown_step:
        print(f"[warn] {len(unknown_step)} file(s) name a step the masterplan "
              "does not have -- NOT archivable, left in place:")
        for u in unknown_step:
            print(f"  - {u}")

    if failures:
        print(f"handoff layout FAIL -- {len(failures)} invariant violation(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("handoff layout OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
