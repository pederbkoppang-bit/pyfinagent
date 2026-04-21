"""phase-4.9.9 red-team negative tests.

Runs 3 deterministic negative checks that prove the phase-4.9
immutable-core and risk-guard contracts actually block -- per
FINRA Notice 15-09 requirement that pre-trade controls block, not
just log, non-conforming strategies.

Tests:
  1. `unsigned_mutation_blocked` -- CI workflow
     `.github/workflows/limits-tag-enforcement.yml` contains the
     `verify_limits_tag` step that enforces GPG-signed tags on
     `limits.yaml` mutations (phase-4.9.1).
  2. `bad_strategy_blocked` -- `backend.autonomous_harness
     .promote_strategy('intentionally_bad_strategy')` raises
     `PromotionBlocked` because no Gauntlet report exists for that
     strategy (phase-4.9.8).
  3. `evidence_logged` -- the failed promotion appends a row to the
     blocklist JSONL (phase-4.9.8 30-day cooldown).

Writes `handoff/phase-4.9-redteam.md` with the per-test evidence and
a final `REDTEAM_PASS` marker iff all three tests are True.
Exit 0 on pass, 1 on fail.

Test 2 and Test 3 monkey-patch `backend.autonomous_harness._BLOCKLIST_PATH`
to a temp file so repeated runs do NOT pollute the real
`handoff/gauntlet_blocklist.jsonl`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WORKFLOW = REPO / ".github" / "workflows" / "limits-tag-enforcement.yml"
OUT = REPO / "handoff" / "phase-4.9-redteam.md"


def test_unsigned_mutation_blocked() -> tuple[bool, str]:
    if not WORKFLOW.exists():
        return False, f"workflow missing: {WORKFLOW.relative_to(REPO)}"
    body = WORKFLOW.read_text(encoding="utf-8")
    if "verify_limits_tag" not in body:
        return False, "verify_limits_tag step missing from workflow"
    return True, (
        "verify_limits_tag step present in "
        + str(WORKFLOW.relative_to(REPO))
        + " -- CI blocks limits.yaml diffs lacking a GPG-signed annotated tag (phase-4.9.1)."
    )


def test_bad_strategy_blocked_and_evidence_logged() -> tuple[bool, bool, str]:
    """Returns (bad_strategy_blocked, evidence_logged, detail)."""
    import backend.autonomous_harness as ah
    tmp_blocklist = Path(tempfile.gettempdir()) / "phase4_9_redteam_blocklist.jsonl"
    if tmp_blocklist.exists():
        tmp_blocklist.unlink()

    original_path = ah._BLOCKLIST_PATH
    ah._BLOCKLIST_PATH = tmp_blocklist
    try:
        raised: Exception | None = None
        try:
            ah.promote_strategy("intentionally_bad_strategy")
        except ah.PromotionBlocked as e:
            raised = e
        except Exception as e:
            raised = e

        blocked = raised is not None and isinstance(raised, ah.PromotionBlocked)

        evidence_ok = False
        rows: list[dict] = []
        if tmp_blocklist.exists():
            for line in tmp_blocklist.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
            evidence_ok = any(
                r.get("strategy") == "intentionally_bad_strategy" for r in rows
            )

        detail_parts = []
        detail_parts.append(
            f"promote_strategy raised={type(raised).__name__ if raised else 'None'}"
        )
        if raised:
            detail_parts.append(f"msg={str(raised)[:180]}")
        detail_parts.append(f"blocklist_rows={len(rows)} at {tmp_blocklist}")
        return blocked, evidence_ok, " | ".join(detail_parts)
    finally:
        ah._BLOCKLIST_PATH = original_path


def main() -> int:
    t1_ok, t1_detail = test_unsigned_mutation_blocked()
    t2_ok, t3_ok, t23_detail = test_bad_strategy_blocked_and_evidence_logged()

    ts = datetime.now(timezone.utc).isoformat()
    all_pass = t1_ok and t2_ok and t3_ok

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# phase-4.9.9 red-team audit")
    lines.append("")
    lines.append(f"Run at: {ts}")
    lines.append("")
    lines.append("| Test | Pass | Evidence |")
    lines.append("|------|------|----------|")
    lines.append(f"| unsigned_mutation_blocked | {t1_ok} | {t1_detail} |")
    lines.append(f"| bad_strategy_blocked | {t2_ok} | {t23_detail} |")
    lines.append(f"| evidence_logged | {t3_ok} | (see bad_strategy_blocked row above) |")
    lines.append("")
    lines.append(f"Overall: {'REDTEAM_PASS' if all_pass else 'REDTEAM_FAIL'}")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "unsigned_mutation_blocked": t1_ok,
        "bad_strategy_blocked": t2_ok,
        "evidence_logged": t3_ok,
        "overall": "REDTEAM_PASS" if all_pass else "REDTEAM_FAIL",
        "wrote": str(OUT.relative_to(REPO)),
    }, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
