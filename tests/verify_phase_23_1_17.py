"""phase-23.1.17 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. frontend/src/lib/useLiveNav.ts exists with the canonical signature.
2. frontend/src/app/page.tsx imports useLiveNav AND uses it.
3. frontend/src/app/paper-trading/page.tsx imports useLiveNav (refactored).
4. paper-trading page no longer has the inline `liveNav = useMemo` block.
5. scripts/repair_phase_23_1_17.py exists with mark_to_market call.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. Shared hook file
    hook_path = repo / "frontend/src/lib/useLiveNav.ts"
    assert hook_path.exists(), "frontend/src/lib/useLiveNav.ts missing"
    hook_src = hook_path.read_text(encoding="utf-8")
    assert "export function useLiveNav" in hook_src, "useLiveNav not exported"
    assert "liveNav: number | null" in hook_src and "liveTotalPnlPct: number | null" in hook_src, \
        "useLiveNav must return liveNav + liveTotalPnlPct"
    assert "phase-23.1.17" in hook_src, "useLiveNav.ts missing phase-23.1.17 marker"

    # 2. Home page wires it up
    home_src = (repo / "frontend/src/app/page.tsx").read_text(encoding="utf-8")
    assert 'import { useLiveNav } from "@/lib/useLiveNav"' in home_src, \
        "page.tsx must import useLiveNav"
    assert 'import { useLivePrices } from "@/lib/useLivePrices"' in home_src, \
        "page.tsx must import useLivePrices for the hook"
    assert "useLiveNav(ptStatus, positions, livePrices)" in home_src, \
        "page.tsx must call useLiveNav(ptStatus, positions, livePrices)"
    assert "navValue = liveNav ?? nav?.nav" in home_src, \
        "page.tsx NAV tile must prefer liveNav with BQ snapshot fallback"

    # 3 + 4. Paper-trading SHARED LAYOUT uses the hook.
    # phase-44.2 (cycle 63): /paper-trading route-split; the hook moved from
    # page.tsx into layout.tsx so all 6 sub-routes consume the same SSOT
    # NAV. page.tsx is now a redirect to /paper-trading/positions.
    pt_layout_path = repo / "frontend/src/app/paper-trading/layout.tsx"
    assert pt_layout_path.exists(), (
        "frontend/src/app/paper-trading/layout.tsx missing (phase-44.2)"
    )
    pt_src = pt_layout_path.read_text(encoding="utf-8")
    assert 'import { useLiveNav } from "@/lib/useLiveNav"' in pt_src, \
        "paper-trading/layout.tsx must import useLiveNav"
    assert "useLiveNav(status, positions, livePrices)" in pt_src, \
        "paper-trading layout must call useLiveNav(...)"
    # No inline duplication -- the explicit `const liveNav = useMemo` block must be absent.
    assert "const liveNav = useMemo" not in pt_src, \
        "paper-trading layout inline liveNav useMemo must be removed (use shared hook)"

    # 5. Repair script
    repair_path = repo / "scripts/repair_phase_23_1_17.py"
    assert repair_path.exists(), "scripts/repair_phase_23_1_17.py missing"
    repair_src = repair_path.read_text(encoding="utf-8")
    assert "trader.mark_to_market()" in repair_src, \
        "repair script must call trader.mark_to_market()"
    assert "save_daily_snapshot" in repair_src, \
        "repair script must call save_daily_snapshot to refresh redLine series"
    assert "--apply" in repair_src and "args.apply" in repair_src, \
        "repair script must require explicit --apply for mutation"

    print("ok useLiveNav shared hook + home page consumption + paper-trading refactor + "
          "repair script (mark_to_market + save_daily_snapshot)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
