"""verify_phase_25_D -- Normalize per-agent contribution weights to 0-1 range.

Verifies:
  1. signal_attribution.py contains the /10.0 normalization on Trader, Quant,
     SignalStack sites + the _clamp01 helper.
  2. Drawer contains a TotalWeightSummary element + "Total contribution weight" label.
  3. The 22 existing signal_attribution unit tests pass (pytest invocation).
  4. Behavioral round-trip: ALL signals returned by extract_signals_from_analysis +
     extract_quant_signals have weight in [0.0, 1.0].
  5. Sample: final_score=7 yields Trader weight=0.7 (was 7.0).

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: backend normalization ──────────────────────────────────────
sa_src = (REPO / "backend/services/signal_attribution.py").read_text(encoding="utf-8")
has_clamp01 = re.search(r"def _clamp01\b", sa_src) is not None
# Trader: divide by 10 of score
trader_norm = re.search(r"trader_weight\s*=.*?/ 10\.0", sa_src) is not None
# Quant: divide by 10 of composite
quant_norm = re.search(r"quant_weight\s*=.*?float\(composite\) / 10\.0", sa_src) is not None
# SignalStack: divide by 10 of conviction
stack_norm = re.search(r"stack_weight\s*=.*?float\(conviction_score\) / 10\.0", sa_src) is not None

claim(
    "1. all_weights_normalized_to_0_to_1_in_signal_attribution",
    has_clamp01 and trader_norm and quant_norm and stack_norm,
    f"clamp01={has_clamp01} trader={trader_norm} quant={quant_norm} stack={stack_norm}",
)


# ── Claim 2: frontend drawer summary ────────────────────────────────────
drawer_src = (REPO / "frontend/src/components/AgentRationaleDrawer.tsx").read_text(encoding="utf-8")
has_component = "TotalWeightSummary" in drawer_src
has_label = "Total contribution weight" in drawer_src
has_wire = "<TotalWeightSummary signals={data.signals}" in drawer_src
claim(
    "2. total_contribution_weight_summary_displayed_at_top_of_drawer",
    has_component and has_label and has_wire,
    f"component={has_component} label={has_label} wire={has_wire}",
)


# ── Claim 3: pytest exit=0 on full signal_attribution suite ────────────
proc = subprocess.run(
    [
        str(REPO / ".venv/bin/python"),
        "-m",
        "pytest",
        "tests/services/test_signal_attribution.py",
        "-v",
        "--no-header",
        "--tb=short",
    ],
    cwd=str(REPO),
    env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    capture_output=True,
    text=True,
    timeout=60,
)
n_passed = proc.stdout.count(" PASSED")
exit_ok = proc.returncode == 0 and n_passed >= 22
claim(
    "3. pytest_signal_attribution_suite_all_pass",
    exit_ok,
    f"exit={proc.returncode} passed={n_passed} (expect 22+)",
)


# ── Claim 4: behavioral round-trip ─────────────────────────────────────
from backend.services.signal_attribution import (  # noqa: E402
    extract_signals_from_analysis,
    extract_quant_signals,
)

analysis = {
    "recommendation": "BUY",
    "final_score": 9,
    "trader_note": "Strong momentum",
    "risk_assessment": {"reasoning": "Position cap 2%", "recommended_position_pct": 0.02},
    "analyst_summary": "Synthesis text",
    "debate": {"bull_argument": "case A", "bear_argument": "case B", "bull_weight": 0.6},
}
candidate = {
    "ticker": "X",
    "composite_score": 12.5,  # intentionally > 10 to test clamp
    "conviction_score": 8.0,
    "regime_tag": "risk_on",
}

all_sigs = extract_signals_from_analysis(analysis) + extract_quant_signals(candidate)
weights = [(s.get("agent"), s.get("weight")) for s in all_sigs]
out_of_range = [(a, w) for (a, w) in weights if not (0.0 <= w <= 1.0)]
claim(
    "4. behavioral_all_weights_in_zero_one_range",
    not out_of_range,
    f"out_of_range={out_of_range} all_weights={weights}",
)


# ── Claim 5: Trader weight=0.9 for final_score=9 (clamp check) ─────────
trader = next((s for s in all_sigs if s["agent"] == "Trader"), None)
claim(
    "5. trader_weight_is_normalized_division_by_ten",
    trader is not None and abs(trader["weight"] - 0.9) < 1e-9,
    f"trader_weight={trader['weight'] if trader else 'None'} expected=0.9",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.D verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
