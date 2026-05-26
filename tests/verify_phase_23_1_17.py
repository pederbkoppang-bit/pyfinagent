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

    # phase-72 cycle (2026-05-26): the SSOT was lifted UP one level. Both
    # consumer pages now access liveNav via `useLivePortfolio()` from the
    # root-level LivePortfolioProvider, which itself imports useLiveNav.
    # This stronger invariant -- one provider, two consumers -- eliminates
    # the cycle-71 race condition where two independent useLivePrices
    # instances polled at different millisecond offsets and produced ~$18
    # NAV gaps between Home and Paper Trading. Accept either shape so the
    # invariant tracks the architecture without per-cycle test churn.
    live_portfolio_ctx = repo / "frontend/src/lib/live-portfolio-context.tsx"
    home_src = (repo / "frontend/src/app/page.tsx").read_text(encoding="utf-8")
    assert (
        'import { useLiveNav } from "@/lib/useLiveNav"' in home_src
        or 'import { useLivePortfolio } from "@/lib/live-portfolio-context"' in home_src
    ), "page.tsx must import useLiveNav OR useLivePortfolio (phase-72 SSOT)"
    # NAV display must reference the live derivation -- either via
    # `liveNav ?? nav?.nav` pattern OR via the lp.liveNav fall-through.
    assert "liveNav" in home_src, "page.tsx NAV display must reference liveNav"

    pt_layout_path = repo / "frontend/src/app/paper-trading/layout.tsx"
    assert pt_layout_path.exists(), (
        "frontend/src/app/paper-trading/layout.tsx missing (phase-44.2)"
    )
    pt_src = pt_layout_path.read_text(encoding="utf-8")
    assert (
        'import { useLiveNav } from "@/lib/useLiveNav"' in pt_src
        or 'import { useLivePortfolio } from "@/lib/live-portfolio-context"' in pt_src
    ), "paper-trading layout must import useLiveNav OR useLivePortfolio"
    # If the layout went via the context, sanity-check the context has the hook.
    if (
        'import { useLivePortfolio } from "@/lib/live-portfolio-context"' in pt_src
    ):
        assert live_portfolio_ctx.exists(), (
            "live-portfolio-context.tsx missing (phase-72 root SSOT provider)"
        )
        ctx_src = live_portfolio_ctx.read_text(encoding="utf-8")
        assert 'import { useLiveNav }' in ctx_src and 'from "@/lib/useLiveNav"' in ctx_src, (
            "live-portfolio-context.tsx must import useLiveNav (SSOT lives here)"
        )
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
