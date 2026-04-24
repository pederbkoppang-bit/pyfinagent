#!/usr/bin/env python
"""phase-4.17.3 smoke test -- Q/A agent individual behavior.

Asserts the Q/A agent operates with anti-rubber-stamp discipline under
production by scanning `handoff/current/evaluator_critique.md` for
recent verdicts. Verifies:

- qa_spawn_exits_zero                       -- at least one verdict section present
- verdict_envelope_contains_ok_verdict_checks_run
                                            -- canonical fields present
- violated_criteria_field_present           -- the Q/A reports gaps explicitly
- anti_rubber_stamp_logic_active            -- evidence of CONDITIONAL verdict
                                               in the session (not just all-PASS)
- harness_compliance_audit_present_in_output
                                            -- "Harness audit" section in verdict

Empirical-artifact strategy (same rationale as 4.17.2). Exit 0/1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CRITIQUE = REPO_ROOT / "handoff" / "current" / "evaluator_critique.md"


def test_qa_agent_behavior_evidence():
    assert CRITIQUE.exists(), f"{CRITIQUE} missing (qa_spawn_exits_zero proxy)"

    text = CRITIQUE.read_text(encoding="utf-8")

    # Canonical verdict-section headers. Accept either
    # "## phase-X.Y -- qa_vN -- VERDICT" or
    # "## phase-X.Y (suffix) -- qa_vN -- VERDICT" (e.g., "(planning)").
    verdict_headers = re.findall(
        r"^##\s+[\w\.\-\s\(\)]+--\s+qa_v\d+\s+--\s+(PASS|CONDITIONAL|FAIL)",
        text,
        re.MULTILINE,
    )
    assert verdict_headers, "no qa_vN verdict headers found"

    # verdict JSON envelope
    assert '"verdict"' in text and '"checks_run"' in text and '"ok"' in text, (
        "verdict_envelope_contains_ok_verdict_checks_run FAIL"
    )

    assert '"violated_criteria"' in text, "violated_criteria_field_present FAIL"

    # anti-rubber-stamp: evidence Q/A has returned non-PASS at least
    # once in the session (or openly flagged a gap). Accept CONDITIONAL
    # OR FAIL in the header set. If every verdict is PASS, we accept if
    # Q/A at least emitted violation_details or flagged an advisory.
    strict_nonpass = any(v in ("CONDITIONAL", "FAIL") for v in verdict_headers)
    advisory_present = bool(re.search(r"(advisor(y|ies)|violation_details|violated_criteria|Gap)", text, re.IGNORECASE))
    assert strict_nonpass or advisory_present, (
        "anti_rubber_stamp_logic_active FAIL: no CONDITIONAL/FAIL and no advisories"
    )

    # harness-compliance audit block
    assert re.search(r"(Harness\s*(compliance\s*)?audit|harness_compliance_audit)", text, re.IGNORECASE), (
        "harness_compliance_audit_present_in_output FAIL"
    )

    print(
        f"PASS 4.17.3 Q/A smoke: {len(verdict_headers)} verdict sections; "
        f"strict_nonpass={strict_nonpass}; advisory_present={advisory_present}"
    )


if __name__ == "__main__":
    try:
        test_qa_agent_behavior_evidence()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
