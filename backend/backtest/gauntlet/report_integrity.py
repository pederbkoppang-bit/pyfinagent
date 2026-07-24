"""phase-75.8.1: report-integrity predicates for gauntlet report.json consumers.

The SINGLE shared implementation of the fabricated-evidence rejections that
phase-75.8 introduced (stub fingerprint) and this step extends (dry_run-label
refusal) -- imported by BOTH consumers of `handoff/gauntlet/<strategy>/report.json`:

  - scripts/risk/promotion_gate.py   (the CLI gate; used the inline 75.8 block)
  - backend/autonomous_harness.py::promote_strategy  (trusted the report blindly)

Two consumers with byte-identical predicate logic is exactly the drift class the
SSOT rule exists for -- one copy hardens, the other stays exploitable. Research
basis (brief 75.8.1): fabricated intermediate artifacts are the most common
verifier exploit (SpecBench), and the guard must remain a DETERMINISTIC code
gate, not an LLM judgment (ImpossibleBench: monitors caught only 42-50%).

PURE LEAF by design: imports nothing beyond typing, so both the in-package
consumer and the sys.path-inserted scripts/risk consumer can use it with zero
import-cycle risk. Do NOT import consumers, services, or evaluator here.

Check ORDER is load-bearing: fingerprint FIRST, dry_run label SECOND.
test_phase_75_promotion_gate.py:245-257 feeds a report that is BOTH stub AND
dry_run:true and pins the 'stub fingerprint' reason string.
"""
from __future__ import annotations

from typing import Any


def is_dry_run_report(report: dict[str, Any]) -> bool:
    """True when the report labels itself dry-run. A dry-run report is
    synthetic evidence by construction (gauntlet.py:147 only ever writes
    dry_run:true today; live mode raises NotImplementedError) and must never
    authorise a real promotion."""
    return bool(report.get("dry_run"))


def has_stub_fingerprint(report: dict[str, Any]) -> bool:
    """True when every NON-SKIPPED regime has bt_drawdown exactly equal to
    drawdown -- the dry-run stub copies one into the other (gauntlet.py:97),
    and real live-vs-backtest drawdowns never match to the last bit on every
    regime. Skipped regimes are filtered like the evaluator does (they carry
    neither key, so None == None would false-positive); an empty non-skipped
    list is NOT fingerprinted (all([]) is True -- the vacuous case is guarded
    explicitly)."""
    non_skipped = [
        r for r in (report.get("per_regime", []) or [])
        if not r.get("skipped")
    ]
    return bool(non_skipped) and all(
        r.get("bt_drawdown") == r.get("drawdown") for r in non_skipped
    )


def check_report_integrity(report: dict[str, Any]) -> tuple[bool, str | None]:
    """Composite gate: (ok, reason). ok=False means the report is fabricated
    or synthetic evidence and MUST NOT drive a promotion.

    Fingerprint is checked FIRST (ordering pinned by the pre-existing
    promotion-gate test on a report that is both stub and dry_run:true); the
    fingerprint reason string is byte-identical to the 75.8 inline original.
    """
    if has_stub_fingerprint(report):
        return False, (
            "stub fingerprint: bt_drawdown == drawdown exactly for "
            "all non-skipped regimes -- report is dry-run/stub "
            "evidence, not a live gauntlet run"
        )
    if is_dry_run_report(report):
        return False, (
            "report labeled dry_run:true -- synthetic gauntlet evidence "
            "cannot authorise a real promotion"
        )
    return True, None


__all__ = ["check_report_integrity", "has_stub_fingerprint", "is_dry_run_report"]
