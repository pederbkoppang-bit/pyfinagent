"""verify_phase_25_C -- Surface Layer-1 28-skill outputs in drawer.

Verifies:
  1. `extract_layer1_signals(analysis, *, lite_mode=False) -> list[dict]` exists.
  2. Returns [] on a lite-shape analysis dict (no skill keys).
  3. Returns >=3 entries on a full-shape dict (insider + options + sector).
  4. Gate: explicit lite_mode=True returns [] even when skill keys are present.
  5. `_signal_to_weight` mapping correct (BUY=1.0, HOLD=0.5, N/A=0.0).
  6. Drawer renders `layer1_skills` via the Layer component.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: extract_layer1_signals exists with right signature ────────
sa_src = (REPO / "backend/services/signal_attribution.py").read_text(encoding="utf-8")
tree = ast.parse(sa_src)
fn_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "extract_layer1_signals":
        fn_node = node
        break
if fn_node:
    pos = [a.arg for a in fn_node.args.args]
    kw = [a.arg for a in fn_node.args.kwonlyargs]
    has_analysis = "analysis" in pos
    has_lite_mode = "lite_mode" in kw
else:
    pos = kw = []
    has_analysis = has_lite_mode = False

claim(
    "1. signal_attribution_extracts_layer1_skill_keys",
    bool(fn_node) and has_analysis and has_lite_mode,
    f"found={bool(fn_node)} pos={pos} kw={kw}",
)


# ── Setup: import functions ────────────────────────────────────────────
from backend.services.signal_attribution import (  # noqa: E402
    extract_layer1_signals,
    _signal_to_weight,
    group_signals_for_drawer,
)


# ── Claim 2: returns [] on lite-shape analysis ─────────────────────────
lite_analysis = {
    "recommendation": "BUY",
    "final_score": 7,
    "risk_assessment": {"reason": "ok"},
    "full_report": {"source": "claude-sonnet-4"},
}
out_lite = extract_layer1_signals(lite_analysis)
claim(
    "2. extract_layer1_signals_returns_empty_on_lite_shape",
    out_lite == [],
    f"got {out_lite}",
)


# ── Claim 3: returns 3+ entries on full-shape ──────────────────────────
full_analysis = {
    "recommendation": "BUY",
    "final_score": 7,
    "insider": {"signal": "BUY", "summary": "Strong insider buying"},
    "options": {"signal": "HOLD", "summary": "Mixed options flow"},
    "sector_analysis": {"signal": "BUY", "summary": "Sector rotation favorable"},
    "patent": {"signal": "N/A", "summary": ""},  # should be skipped
}
out_full = extract_layer1_signals(full_analysis)
agents = [s["agent"] for s in out_full]
claim(
    "3. drawer_renders_layer1_skills_sub_tree_when_full_pipeline_ran",
    len(out_full) >= 3 and "Insider" in agents and "Options" in agents and "Sector" in agents,
    f"count={len(out_full)} agents={agents}",
)


# ── Claim 4: gate -- explicit lite_mode=True short-circuits ────────────
out_gated = extract_layer1_signals(full_analysis, lite_mode=True)
claim(
    "4. gate_on_settings_lite_mode_false",
    out_gated == [],
    f"got {out_gated} (expected [])",
)


# ── Claim 5: signal->weight mapping ─────────────────────────────────────
correct_mapping = (
    _signal_to_weight("BUY") == 1.0
    and _signal_to_weight("SELL") == 1.0
    and _signal_to_weight("HOLD") == 0.5
    and _signal_to_weight("NEUTRAL") == 0.5
    and _signal_to_weight("N/A") == 0.0
    and _signal_to_weight("") == 0.0
    and _signal_to_weight("ERROR") == 0.0
)
claim(
    "5. signal_to_weight_mapping_correct",
    correct_mapping,
    f"BUY={_signal_to_weight('BUY')} HOLD={_signal_to_weight('HOLD')} NA={_signal_to_weight('N/A')}",
)


# ── Claim 6: drawer renders layer1_skills ──────────────────────────────
drawer_src = (REPO / "frontend/src/components/AgentRationaleDrawer.tsx").read_text(encoding="utf-8")
has_interface = "layer1_skills?: Signal[]" in drawer_src
has_render = bool(
    re.search(r'<Layer\s+title="Layer-1 Skills"\s+items={data\.tree\.layer1_skills', drawer_src)
)
claim(
    "6. drawer_renders_layer1_skills_layer",
    has_interface and has_render,
    f"interface={has_interface} render={has_render}",
)


# ── Bonus: group routing ────────────────────────────────────────────────
grouped = group_signals_for_drawer(out_full)
group_ok = "layer1_skills" in grouped and len(grouped["layer1_skills"]) == 3
claim(
    "7. group_signals_for_drawer_routes_layer1_skills_bucket",
    group_ok,
    f"layer1_skills bucket size={len(grouped.get('layer1_skills', []))}",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.C verification ===\n")
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
