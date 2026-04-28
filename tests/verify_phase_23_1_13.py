"""phase-23.1.13 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. paper_max_per_sector setting exists in Settings + FullSettings + SettingsUpdate
2. screen_universe accepts sector_lookup kwarg
3. decide_trades source contains the sector cap guard logic
4. autonomous_loop calls _fetch_ticker_meta to enrich candidates with sector
5. RiskMonitorCard source uses tickerMeta for the sector concentration check
6. Manage tab exposes paper_max_per_sector to the operator
7. /portfolio endpoint returns sector_breakdown
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. Settings field
    from backend.config.settings import Settings
    from backend.api.settings_api import FullSettings, SettingsUpdate
    s = Settings()
    assert hasattr(s, "paper_max_per_sector"), "Settings missing paper_max_per_sector"
    assert s.paper_max_per_sector == 2, f"default 2 expected, got {s.paper_max_per_sector}"
    assert "paper_max_per_sector" in FullSettings.model_fields, \
        "FullSettings missing paper_max_per_sector"
    assert "paper_max_per_sector" in SettingsUpdate.model_fields, \
        "SettingsUpdate missing paper_max_per_sector"

    # 2. screen_universe sector_lookup kwarg
    from backend.tools.screener import screen_universe
    sig = inspect.signature(screen_universe)
    assert "sector_lookup" in sig.parameters, \
        f"screen_universe missing sector_lookup kwarg; got {list(sig.parameters)}"

    # 3. decide_trades sector cap guard
    pm_src = (repo / "backend/services/portfolio_manager.py").read_text(encoding="utf-8")
    assert "max_per_sector" in pm_src, "portfolio_manager.py missing max_per_sector usage"
    assert "sector_counts" in pm_src, "portfolio_manager.py missing sector_counts dict"
    assert "at cap" in pm_src, "portfolio_manager.py missing 'at cap' log line"

    # 4. autonomous_loop ticker_meta enrichment
    al_src = (repo / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    assert "_fetch_ticker_meta" in al_src, \
        "autonomous_loop.py missing _fetch_ticker_meta call to enrich candidates"

    # 5. RiskMonitorCard sector concentration
    page_src = (repo / "frontend/src/app/paper-trading/page.tsx").read_text(encoding="utf-8")
    assert "sectorConcentrationHigh" in page_src, \
        "paper-trading/page.tsx missing sector concentration check"
    assert "tickerMeta" in page_src and "RiskMonitorCard" in page_src, \
        "RiskMonitorCard must use tickerMeta"

    # 6. Manage tab toggle
    assert 'field="paper_max_per_sector"' in page_src, \
        "Manage tab missing paper_max_per_sector PaperSettingNum"

    # 7. /portfolio sector_breakdown
    pt_src = (repo / "backend/api/paper_trading.py").read_text(encoding="utf-8")
    assert "sector_breakdown" in pt_src, \
        "paper_trading.py /portfolio endpoint missing sector_breakdown"

    print("ok paper_max_per_sector + screen_universe.sector_lookup + decide_trades cap + "
          "autonomous_loop ticker_meta enrichment + RiskMonitor sector check + "
          "Manage tab toggle + /portfolio sector_breakdown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
