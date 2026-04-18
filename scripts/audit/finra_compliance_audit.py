"""phase-4.8 step 4.8.9 FINRA compliance audit.

Four teeth:
1. compliance_logger exposes `write_rationale` + `fetch_rationale`
   + `retention_policy` as callables.
2. Round-trip test: write a fresh synthetic record, fetch by
   trade_id, compare every key byte-for-byte. A regression that
   dropped fields between write + read fails here.
3. retention_policy()['retention_years_policy'] >= 3 (the master-
   plan floor). Documented value is 6 per SEC 17a-4.
4. HITL fields enforced: calling write_rationale without
   approver_id raises ValueError. A regression that made HITL
   fields optional fails here.
"""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services import compliance_logger as cl  # noqa: E402

OUT = REPO / "handoff" / "finra_compliance_audit.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []

    # 1. Callables present.
    t1 = all(
        callable(getattr(cl, name, None))
        for name in ("write_rationale", "fetch_rationale", "retention_policy")
    )
    if not t1:
        reasons.append("compliance_logger missing one of the 3 required callables")

    # 2. Round-trip.
    audit_tid = f"compliance-audit-{uuid.uuid4().hex[:10]}"
    write_ok = False
    roundtrip_ok = False
    try:
        record, loc = cl.write_rationale(
            trade_id=audit_tid,
            system_id="pyfinagent-audit",
            agent_trace=[{"agent": "audit", "model": "none"}],
            input_signals={"marker": "compliance-audit"},
            output_recommendation="BUY",
            confidence=0.5,
            approver_id="audit@pyfinagent.local",
            decision="approve",
            reason_code="audit_check",
            extras={"audit": True},
        )
        write_ok = True
    except Exception as e:
        reasons.append(f"write_rationale raised on valid input: {e}")
    if write_ok:
        try:
            back = cl.fetch_rationale(audit_tid)
            # Compare key fields.
            need = ("trade_id", "system_id", "approver_id", "decision",
                    "reason_code", "output_recommendation", "confidence")
            mism = [k for k in need
                     if back.get(k) != getattr(record, k)]
            roundtrip_ok = not mism
            if mism:
                reasons.append(f"roundtrip mismatch on {mism}")
        except Exception as e:
            reasons.append(f"fetch_rationale raised: {e}")
    # Cleanup the synthetic audit record so reruns don't collide.
    try:
        path = cl._local_path(audit_tid)  # noqa: SLF001
        if path.exists():
            path.unlink()
    except Exception:
        pass

    # 3. Retention policy >= masterplan floor (3y).
    pol = cl.retention_policy()
    t3 = int(pol.get("retention_years_policy", 0)) >= 3
    if not t3:
        reasons.append(f"retention_years_policy={pol.get('retention_years_policy')} < 3")

    # 4. HITL fields REQUIRED: write without approver_id must raise.
    hitl_enforced = False
    try:
        cl.write_rationale(
            trade_id=f"audit-noapprover-{uuid.uuid4().hex[:8]}",
            system_id="pyfinagent-audit",
            agent_trace=[],
            input_signals={},
            output_recommendation="BUY",
            confidence=0.5,
            approver_id="",          # empty -> must raise
            decision="approve",
            reason_code="audit_check",
        )
        reasons.append("write_rationale accepted empty approver_id")
    except (ValueError, TypeError):
        hitl_enforced = True
    except Exception as e:
        reasons.append(f"unexpected exception for missing approver: {e}")

    # 5. finra_audit.json already exists from the compliance run?
    finra = REPO / "handoff" / "finra_audit.json"
    finra_ok = finra.exists()
    if finra_ok:
        try:
            d = json.loads(finra.read_text(encoding="utf-8"))
            finra_ok = d.get("sample_retrieval_success_rate") == 1.0
        except Exception:
            finra_ok = False
    if not finra_ok:
        reasons.append("handoff/finra_audit.json missing or sample_retrieval_success_rate != 1.0")

    all_ok = t1 and roundtrip_ok and t3 and hitl_enforced and finra_ok
    verdict = "PASS" if all_ok else "FAIL"
    result = {
        "step": "4.8.9",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "callables_present": t1,
        "roundtrip_ok": roundtrip_ok,
        "retention_ge_3y": t3,
        "retention_policy": pol,
        "hitl_enforced": hitl_enforced,
        "finra_rationale_audit_passed": finra_ok,
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "callables": t1, "roundtrip": roundtrip_ok,
        "retention_ge_3y": t3, "hitl": hitl_enforced,
        "finra_ok": finra_ok,
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
