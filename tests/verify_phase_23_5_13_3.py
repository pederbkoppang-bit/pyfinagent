"""phase-23.5.13.3 verifier — launchd-substep criterion amendment.

  1. The 5 launchd substeps (23.5.15..19) verification fields no longer
     contain `next_run is not None` (the unmeetable assertion).
  2. They DO contain the new in-set assertion against the bridge's
     documented status values.
  3. Phase-23.5.14's verification field is UNCHANGED (historical
     archive integrity; amendment is forward-only).
  4. The audit-trail JSONL at handoff/audit/criterion_amendments.jsonl
     contains a row with the required fields.

Exit 0 only when all 4 checks pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTERPLAN = REPO / ".claude" / "masterplan.json"
AUDIT = REPO / "handoff" / "audit" / "criterion_amendments.jsonl"

AMENDED_STEPS = ("23.5.15", "23.5.16", "23.5.17", "23.5.18", "23.5.19")
PRESERVED_STEP = "23.5.14"
EXPECTED_STATUS_SET_TOKENS = (
    '"running"',
    '"ok"',
    '"failed"',
    '"not_loaded"',
    '"unknown"',
)
REQUIRED_AUDIT_FIELDS = (
    "timestamp",
    "amendment_id",
    "amended_step_ids",
    "criterion_id",
    "prior_criterion_per_step",
    "new_criterion_template",
    "justification",
    "evidence_refs",
    "operator",
    "applies_forward_only",
    "retroactive_re_evaluation",
)


def _walk_steps(node, sid_to_field: dict[str, str]) -> None:
    if isinstance(node, dict):
        sid = node.get("id")
        if sid and "verification" in node:
            sid_to_field[sid] = node["verification"]
        for v in node.values():
            if isinstance(v, list):
                for it in v:
                    _walk_steps(it, sid_to_field)
            elif isinstance(v, dict):
                _walk_steps(v, sid_to_field)


def _load_steps() -> dict[str, str]:
    d = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    _walk_steps(d, out)
    return out


def check_amended_no_next_run(sid_to_field: dict[str, str]) -> tuple[bool, str]:
    failures = []
    for sid in AMENDED_STEPS:
        v = sid_to_field.get(sid, "")
        if "next_run" in v:
            failures.append(f"{sid} still references next_run")
    if failures:
        return False, "; ".join(failures)
    return True, f"5/5 amended substeps have no `next_run` reference"


def check_amended_has_status_set(sid_to_field: dict[str, str]) -> tuple[bool, str]:
    failures = []
    for sid in AMENDED_STEPS:
        v = sid_to_field.get(sid, "")
        missing = [t for t in EXPECTED_STATUS_SET_TOKENS if t not in v]
        if missing:
            failures.append(f"{sid} missing tokens {missing}")
    if failures:
        return False, "; ".join(failures)
    return True, f"5/5 amended substeps include the documented status-set check"


def check_phase_23_5_14_preserved(sid_to_field: dict[str, str]) -> tuple[bool, str]:
    v = sid_to_field.get(PRESERVED_STEP, "")
    if "next_run" not in v:
        return False, f"{PRESERVED_STEP} appears to have been retroactively amended (next_run missing)"
    return True, f"{PRESERVED_STEP} verification field still contains `next_run` (historical record preserved)"


def check_audit_trail() -> tuple[bool, str]:
    if not AUDIT.exists():
        return False, f"audit file missing: {AUDIT}"
    rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return False, "audit file is empty"
    target = next(
        (r for r in rows if r.get("amendment_id") == "phase-23.5.13.3-launchd-next_run"),
        None,
    )
    if target is None:
        return False, "amendment row 'phase-23.5.13.3-launchd-next_run' not found in audit trail"
    missing = [f for f in REQUIRED_AUDIT_FIELDS if f not in target]
    if missing:
        return False, f"audit row missing fields {missing}"
    if set(target["amended_step_ids"]) != set(AMENDED_STEPS):
        return False, f"audit row amended_step_ids mismatch: {target['amended_step_ids']}"
    if target.get("retroactive_re_evaluation") is not False:
        return False, f"audit row retroactive_re_evaluation must be False (got {target.get('retroactive_re_evaluation')!r})"
    return True, f"audit row present with all {len(REQUIRED_AUDIT_FIELDS)} required fields"


def main() -> int:
    sid_to_field = _load_steps()
    checks = [
        ("amended steps no longer assert next_run", lambda: check_amended_no_next_run(sid_to_field)),
        ("amended steps include status-set check",   lambda: check_amended_has_status_set(sid_to_field)),
        ("23.5.14 preserved (forward-only amendment)", lambda: check_phase_23_5_14_preserved(sid_to_field)),
        ("audit-trail row present + complete",        check_audit_trail),
    ]
    print("=== phase-23.5.13.3 verifier ===")
    failed = []
    for label, fn in checks:
        ok, info = fn()
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {label}: {info}")
        if not ok:
            failed.append(label)
    print()
    if failed:
        print(f"FAIL ({len(failed)}/{len(checks)}): {failed}")
        return 1
    print(f"PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
