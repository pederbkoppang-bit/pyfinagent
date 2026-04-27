"""phase-23.1.12 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. The hardcoded `settings.lite_mode = True` literal is no longer present in
   autonomous_loop.py (operator's choice is now respected).
2. _run_single_analysis branches on settings.lite_mode (the if-else exists).
3. _run_claude_analysis return dict has the "_path": "lite" marker.
4. OpsStatusBar.tsx aggregator collapses unknown -> amber.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    al = (repo / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    ops = (repo / "frontend/src/components/OpsStatusBar.tsx").read_text(encoding="utf-8")

    # 1. The hardcoded mutation is gone (allow the substring inside backtick-quoted
    # comments documenting the removal; reject only actual assignment statements).
    assignment_re = re.compile(r"^[^#`\n]*settings\.lite_mode\s*=\s*True", re.MULTILINE)
    matches = assignment_re.findall(al)
    assert not matches, \
        f"BUG: settings.lite_mode = True still active in autonomous_loop.py: {matches}"
    restore_re = re.compile(r"^[^#`\n]*settings\.lite_mode\s*=\s*original_lite", re.MULTILINE)
    assert not restore_re.findall(al), \
        "BUG: lite_mode restoration still present (means override still in place)"

    # 2. _run_single_analysis branches on lite_mode
    rsa_block_match = re.search(
        r"async def _run_single_analysis.*?(?=^async def )",
        al, re.DOTALL | re.MULTILINE,
    )
    assert rsa_block_match, "_run_single_analysis function not found"
    rsa = rsa_block_match.group(0)
    assert "if settings.lite_mode:" in rsa, \
        "_run_single_analysis must branch on settings.lite_mode"
    assert "AnalysisOrchestrator(settings)" in rsa, \
        "Full path must instantiate AnalysisOrchestrator with operator's settings (no Gemini-fallback override)"

    # 3. Lite path emits the _path marker
    assert '"_path": "lite"' in al, \
        "_run_claude_analysis must tag its output dict with _path=lite for the persist guard"

    # 4. OpsStatusBar collapses unknown -> amber
    assert 'b.band === "amber" || b.band === "unknown"' in ops, \
        "OpsStatusBar.tsx must collapse unknown -> amber in the worst-of-N aggregator"

    print("ok lite_mode override removed + branch path correct + _path marker + OpsStatusBar amber-on-unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
