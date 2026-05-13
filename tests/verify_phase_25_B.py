"""phase-25.B verifier -- remove cosmetic aliasing patch (post-25.A cleanup).

Closes phase-24.4 F-2 (signal_attribution.py:131-154 is_lite_dup detection
became dead code after 25.A decoupled the RiskJudge with an independent
LLM call).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_B.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIG = REPO / "backend" / "services" / "signal_attribution.py"
DRAWER = REPO / "frontend" / "src" / "components" / "AgentRationaleDrawer.tsx"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not SIG.exists() or not DRAWER.exists():
        print(f"FAIL: required source file missing")
        return 1

    sig_src = SIG.read_text(encoding="utf-8")
    drawer_src = DRAWER.read_text(encoding="utf-8")

    # ---- Claim 1: is_lite_dup literal removed from signal_attribution.py.
    has_islitedup = "is_lite_dup" in sig_src
    results.append((
        "PASS" if not has_islitedup else "FAIL",
        "is_lite_dup_branch_removed_from_signal_attribution",
        "is_lite_dup variable / branch must not appear in signal_attribution.py",
    ))

    # ---- Claim 2: lite_path string literal removed from signal_attribution.py.
    has_litepath = '"lite_path"' in sig_src or "'lite_path'" in sig_src
    results.append((
        "PASS" if not has_litepath else "FAIL",
        "lite_path_field_removed_from_signal_attribution",
        "lite_path key must not appear in signal_attribution.py",
    ))

    # ---- Claim 3: lite_path field removed from Signal interface.
    has_field = "lite_path?:" in drawer_src
    results.append((
        "PASS" if not has_field else "FAIL",
        "lite_path_field_removed_from_signal_interface",
        "Signal interface in AgentRationaleDrawer.tsx must not declare lite_path?: boolean",
    ))

    # ---- Claim 4: lite-path badge removed.
    has_badge = "lite-path" in drawer_src
    results.append((
        "PASS" if not has_badge else "FAIL",
        "lite_path_amber_badge_removed_from_frontend",
        "the 'lite-path' badge string must not appear in AgentRationaleDrawer.tsx",
    ))

    # ---- Claim 5: conditional amber styling removed.
    has_amber_cond = "text-amber-200/80" in drawer_src
    results.append((
        "PASS" if not has_amber_cond else "FAIL",
        "conditional_amber_styling_removed",
        "the conditional text-amber-200/80 styling for lite-path must be removed",
    ))

    # ---- Claim 6: BEHAVIORAL no-regression -- extract_signals_from_analysis
    # over a post-25.A risk_assessment shape returns a RiskJudge entry with
    # the expected keys and no lite_path key.
    behavior_ok = False
    behavior_err = ""
    try:
        sys.path.insert(0, str(REPO))
        sys.modules.pop("backend.services.signal_attribution", None)
        from backend.services.signal_attribution import (  # type: ignore
            extract_signals_from_analysis,
        )

        analysis = {
            "ticker": "TEST",
            "recommendation": "BUY",
            "final_score": 7,
            "full_report": {
                "source": "claude-sonnet-4-6",
                "analysis": {
                    "action": "BUY",
                    "confidence": 75,
                    "score": 7,
                    "reason": "Strong momentum",
                },
            },
            "risk_assessment": {
                "decision": "APPROVE_REDUCED",
                "reasoning": "Volatility moderate; concentration acceptable; valuation rich -- size down.",
                "reason": "Volatility moderate; concentration acceptable; valuation rich -- size down.",
                "recommended_position_pct": 4.5,
                "risk_level": "MODERATE",
                "risk_limits": {"stop_loss_pct": 10.0, "max_drawdown_pct": 15.0},
            },
        }
        signals = extract_signals_from_analysis(analysis)
        risk_rows = [s for s in signals if s.get("agent") == "RiskJudge"]
        if not risk_rows:
            behavior_err = "no RiskJudge row in signal stack"
        elif "lite_path" in risk_rows[0]:
            behavior_err = "lite_path key still present on RiskJudge entry"
        elif risk_rows[0].get("weight") != 4.5:
            behavior_err = f"weight wrong: {risk_rows[0].get('weight')!r}"
        elif not risk_rows[0].get("rationale", "").startswith("Volatility moderate"):
            behavior_err = f"rationale wrong: {risk_rows[0].get('rationale')!r}"
        else:
            behavior_ok = True
    except Exception as e:
        behavior_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if behavior_ok else "FAIL",
        "behavioral_risk_judge_entry_clean_post_cleanup",
        f"RiskJudge entry must lack lite_path key and carry expected weight + rationale ({behavior_err})",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
