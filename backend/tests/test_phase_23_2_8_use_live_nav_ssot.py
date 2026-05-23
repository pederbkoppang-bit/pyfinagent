"""phase-23.2.8 (P1) verification: home cockpit + paper-trading hero NAV in sync.

Per researcher (handoff/current/research_brief_phase_23_2_8.md, 6 sources):
  - `frontend/src/lib/useLiveNav.ts` exists + exports useLiveNav.
  - Home page (page.tsx:15) imports useLiveNav.
  - Paper-trading page (paper-trading/page.tsx:46) imports useLiveNav.
  - Both pages destructure { liveNav, liveTotalPnlPct }.
  - The NAV math (`cash + positionsValue`) appears ONLY in useLiveNav.ts
    (no re-inlining elsewhere).

This test enforces the SSOT invariant via source-grep so a future refactor
that re-inlines the NAV calc OR removes the hook usage gets caught at
lint-time. Per masterplan 23.2.8: "Manual: open both pages; NAV / Total P&L
should be byte-identical (post phase-23.1.17 useLiveNav SSOT)" -- the
manual check is operator-dependent; the source-grep is mutation-resistant.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "frontend" / "src" / "lib" / "useLiveNav.ts"
HOME_PAGE = REPO_ROOT / "frontend" / "src" / "app" / "page.tsx"
PT_PAGE = REPO_ROOT / "frontend" / "src" / "app" / "paper-trading" / "page.tsx"


def test_phase_23_2_8_use_live_nav_hook_exists_and_exports():
    """The SSOT hook file must exist + export useLiveNav."""
    assert HOOK.exists(), f"useLiveNav hook missing: {HOOK}"
    text = HOOK.read_text(encoding="utf-8")
    assert "export function useLiveNav" in text or "export const useLiveNav" in text, (
        "useLiveNav.ts must export useLiveNav (function or const)"
    )


def test_phase_23_2_8_home_page_imports_use_live_nav():
    """Home page must import useLiveNav from @/lib/useLiveNav."""
    assert HOME_PAGE.exists(), f"home page missing: {HOME_PAGE}"
    text = HOME_PAGE.read_text(encoding="utf-8")
    # Match the import line with whitespace tolerance
    pattern = r'import\s*\{[^}]*useLiveNav[^}]*\}\s*from\s*["\']@/lib/useLiveNav["\']'
    assert re.search(pattern, text), (
        f"home page must import useLiveNav from @/lib/useLiveNav; not found"
    )


def test_phase_23_2_8_paper_trading_page_imports_use_live_nav():
    """Paper-trading page must import useLiveNav from @/lib/useLiveNav."""
    assert PT_PAGE.exists(), f"paper-trading page missing: {PT_PAGE}"
    text = PT_PAGE.read_text(encoding="utf-8")
    pattern = r'import\s*\{[^}]*useLiveNav[^}]*\}\s*from\s*["\']@/lib/useLiveNav["\']'
    assert re.search(pattern, text), (
        f"paper-trading page must import useLiveNav from @/lib/useLiveNav; not found"
    )


def test_phase_23_2_8_both_pages_destructure_live_nav_and_pnl():
    """Both pages must destructure { liveNav, liveTotalPnlPct } from the hook.
    Catches a future refactor that imports the hook but doesn't use its
    return values (silent SSOT drift)."""
    home_text = HOME_PAGE.read_text(encoding="utf-8")
    pt_text = PT_PAGE.read_text(encoding="utf-8")
    # Either order of destructured fields acceptable
    pattern = r'\{\s*liveNav\s*,\s*liveTotalPnlPct\s*\}|\{\s*liveTotalPnlPct\s*,\s*liveNav\s*\}'
    assert re.search(pattern, home_text), (
        "home page must destructure { liveNav, liveTotalPnlPct } from useLiveNav"
    )
    assert re.search(pattern, pt_text), (
        "paper-trading page must destructure { liveNav, liveTotalPnlPct } from useLiveNav"
    )


def test_phase_23_2_8_nav_math_lives_only_in_hook():
    """Anti-drift: the NAV-computation pattern (`cash + positionsValue`) must
    appear ONLY in useLiveNav.ts. If a future commit re-inlines the math in
    a page, it silently breaks the SSOT invariant; this test catches it."""
    hook_text = HOOK.read_text(encoding="utf-8")
    # The NAV math must be present in the hook (sanity)
    nav_math_pattern = re.compile(r'cash\s*\+\s*positionsValue|positionsValue\s*\+\s*cash')
    assert nav_math_pattern.search(hook_text), (
        f"useLiveNav.ts must contain the NAV math 'cash + positionsValue' "
        f"(or its mirror); not found"
    )

    # And it must NOT appear in any frontend page/component
    frontend_src = REPO_ROOT / "frontend" / "src"
    leaks = []
    for path in frontend_src.rglob("*.tsx"):
        if path == HOOK or "useLiveNav" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if nav_math_pattern.search(text):
            # Permit comments mentioning the pattern; check for actual code
            # (we use a stricter test: not inside a // or /* ... */).
            for ln_idx, line in enumerate(text.splitlines(), 1):
                if nav_math_pattern.search(line):
                    stripped = line.strip()
                    if stripped.startswith("//") or stripped.startswith("*"):
                        continue
                    leaks.append(f"{path.relative_to(REPO_ROOT)}:{ln_idx}: {line.strip()[:80]}")
    assert not leaks, (
        "NAV math leak: 'cash + positionsValue' must live ONLY in useLiveNav.ts. "
        "Found in:\n" + "\n".join(leaks[:10])
    )


def test_phase_23_2_8_hook_return_shape_is_documented():
    """The hook must return `{ liveNav, liveTotalPnlPct }`. Verified by
    reading the return statement structure."""
    text = HOOK.read_text(encoding="utf-8")
    # Match the return shape (either `return {...}` or `return { liveNav, ... }`)
    assert "liveNav" in text, "useLiveNav.ts must reference liveNav"
    assert "liveTotalPnlPct" in text, "useLiveNav.ts must reference liveTotalPnlPct"
    # Look for a return statement that includes both
    return_pattern = re.compile(
        r'return\s*\{[^}]*liveNav[^}]*liveTotalPnlPct[^}]*\}|return\s*\{[^}]*liveTotalPnlPct[^}]*liveNav[^}]*\}',
        re.DOTALL,
    )
    assert return_pattern.search(text), (
        "useLiveNav.ts must `return { liveNav, liveTotalPnlPct }` (canonical SSOT shape)"
    )
