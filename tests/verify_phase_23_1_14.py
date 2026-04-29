"""phase-23.1.14 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. autonomous_loop enriches legacy positions with sector via _fetch_ticker_meta
   before calling decide_trades.
2. paper-trading page lifts the tab gate on useLivePrices.
3. SummaryHero accepts liveNav + liveTotalPnlPct props.
4. paper-trading page computes liveNav + liveTotalPnlPct via useMemo.
5. New sector-concentration tests exist.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. autonomous_loop enrichment block
    al_src = (repo / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    assert "phase-23.1.14" in al_src, \
        "autonomous_loop.py missing phase-23.1.14 marker"
    assert "legacy_tickers" in al_src, \
        "autonomous_loop.py missing legacy_tickers list comprehension"
    assert re.search(r"asyncio\.to_thread\s*\(\s*_fetch_ticker_meta\s*,\s*legacy_tickers",
                     al_src), \
        "autonomous_loop.py missing asyncio.to_thread(_fetch_ticker_meta, legacy_tickers, ...)"
    assert "Enriched" in al_src and "legacy positions" in al_src, \
        "autonomous_loop.py missing 'Enriched ... legacy positions' info log"

    # 2 + 3 + 4. frontend page.tsx
    page_src = (repo / "frontend/src/app/paper-trading/page.tsx").read_text(encoding="utf-8")
    # gate lifted: useLivePrices no longer keyed on `tab === "positions"`
    assert 'tab === "positions" && positions.length > 0' not in page_src, \
        "page.tsx still gates useLivePrices on tab==='positions'"
    assert "positions.length > 0,\n  );" in page_src or \
           "positions.length > 0,\n    );" in page_src, \
        "page.tsx useLivePrices second arg should be `positions.length > 0`"
    # SummaryHero signature
    assert "liveNav: number | null" in page_src and "liveTotalPnlPct: number | null" in page_src, \
        "SummaryHero must accept liveNav + liveTotalPnlPct props"
    # liveNav memo body
    assert "const liveNav = useMemo" in page_src, \
        "page.tsx missing liveNav useMemo"
    assert "const liveTotalPnlPct = useMemo" in page_src, \
        "page.tsx missing liveTotalPnlPct useMemo"
    assert "starting_capital" in page_src, \
        "page.tsx liveTotalPnlPct must use starting_capital as the reference"

    # 5. tests file has both new tests
    tests_src = (repo / "tests/services/test_sector_concentration.py").read_text(encoding="utf-8")
    assert "test_legacy_position_with_enriched_sector_blocks_same_sector_buy" in tests_src, \
        "missing test_legacy_position_with_enriched_sector_blocks_same_sector_buy"
    assert "test_legacy_position_without_enrichment_falls_into_unknown" in tests_src, \
        "missing test_legacy_position_without_enrichment_falls_into_unknown"

    # 6. run pytest on the sector-concentration suite (8 tests after this phase)
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/services/test_sector_concentration.py", "-q", "--no-header"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, \
        f"pytest failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert " passed" in result.stdout, f"pytest output unexpected: {result.stdout}"

    print("ok autonomous_loop legacy-position sector enrichment + "
          "page.tsx live-derived NAV/Total-P&L scoreboards + "
          "useLivePrices gate lifted + 2 new sector-concentration tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
