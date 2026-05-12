"""phase-25.B12 verifier — missing states + tab icons sweep.

Closes phase-24.12 audit F-2 (performance + sovereign degraded states)
and F-3 (paper-trading tab icons missing).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_B12.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PERF_PAGE = REPO / "frontend" / "src" / "app" / "performance" / "page.tsx"
SOV_PAGE = REPO / "frontend" / "src" / "app" / "sovereign" / "page.tsx"
PT_PAGE = REPO / "frontend" / "src" / "app" / "paper-trading" / "page.tsx"
ICONS = REPO / "frontend" / "src" / "lib" / "icons.ts"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PERF_PAGE, SOV_PAGE, PT_PAGE, ICONS):
        if not p.exists():
            print(f"FAIL: {p} not found")
            return 1

    perf_text = PERF_PAGE.read_text(encoding="utf-8")
    sov_text = SOV_PAGE.read_text(encoding="utf-8")
    pt_text = PT_PAGE.read_text(encoding="utf-8")
    icons_text = ICONS.read_text(encoding="utf-8")

    # Claim 1: performance/page uses PageSkeleton (no bare <p>Loading...</p>)
    perf_uses_skel = "<PageSkeleton />" in perf_text or "PageSkeleton" in perf_text
    perf_legacy = re.search(r'<p\s+className=["\']text-slate-400["\']>Loading', perf_text)
    results.append(("PASS" if perf_uses_skel and not perf_legacy else "FAIL",
                    "performance_page_uses_pageskeleton_not_bare_p_loading",
                    "performance/page.tsx must import PageSkeleton and replace bare <p>Loading...</p>"))

    # Claim 2: performance/page has rose-bordered error banner with Retry
    perf_err_banner = re.search(
        r'border-rose-500/30.*?bg-rose-950/30.*?Retry',
        perf_text,
        re.DOTALL,
    )
    results.append(("PASS" if perf_err_banner else "FAIL",
                    "performance_page_has_rose_error_banner_with_retry_button",
                    "performance/page.tsx error state must use rose-bordered banner + Retry button"))

    # Claim 3: sovereign/page declares redLineError state
    sov_err_state = re.search(r'redLineError', sov_text)
    results.append(("PASS" if sov_err_state else "FAIL",
                    "sovereign_page_declares_redline_error_state",
                    "sovereign/page.tsx must declare redLineError state for the RedLine fetch"))

    # Claim 4: sovereign/page error handler captures err message (not silent swallow)
    sov_err_set = re.search(r'setRedLineError\s*\(', sov_text)
    results.append(("PASS" if sov_err_set else "FAIL",
                    "sovereign_page_setredlineerror_called_in_catch",
                    "sovereign/page.tsx .catch must call setRedLineError"))

    # Claim 5: sovereign/page renders rose error banner conditionally
    sov_banner = re.search(
        r'\{redLineError\s*&&.*?bg-rose-950/30',
        sov_text,
        re.DOTALL,
    )
    results.append(("PASS" if sov_banner else "FAIL",
                    "sovereign_page_renders_rose_error_banner_when_redline_fails",
                    "sovereign/page.tsx must render rose-bordered error banner when redLineError set"))

    # Claim 6: paper-trading TABS has icon field for each entry
    tabs_with_icons = re.findall(
        r'\{\s*id\s*:\s*["\'][a-z-]+["\']\s*,\s*label\s*:\s*["\'][^"\']+["\']\s*,\s*icon\s*:\s*Tab\w+',
        pt_text,
    )
    results.append(("PASS" if len(tabs_with_icons) >= 6 else "FAIL",
                    "paper_trading_tabs_array_has_icon_field_per_entry",
                    f"paper-trading/page.tsx TABS array must have icon field on each entry; got {len(tabs_with_icons)}/6"))

    # Claim 7: icons.ts exports tab icon aliases via canonical barrel (NOT direct @phosphor-icons/react import in page)
    aliases_in_barrel = all(
        f"as {name}" in icons_text
        for name in ["TabPositions", "TabTrades", "TabNavChart", "TabRealityGap", "TabExitQuality", "TabManage"]
    )
    results.append(("PASS" if aliases_in_barrel else "FAIL",
                    "icons_ts_exports_all_tab_aliases_via_canonical_barrel",
                    "frontend/src/lib/icons.ts must export TabPositions/TabTrades/TabNavChart/TabRealityGap/TabExitQuality/TabManage"))

    # Claim 8: paper-trading page imports tab icons from @/lib/icons, NOT @phosphor-icons/react
    direct_import = re.search(
        r'import\s*\{[^}]*Tab\w+[^}]*\}\s*from\s*["\']@phosphor-icons/react["\']',
        pt_text,
    )
    barrel_import = re.search(
        r'import\s*\{[^}]*Tab\w+[^}]*\}\s*from\s*["\']@/lib/icons["\']',
        pt_text,
        re.DOTALL,
    )
    results.append(("PASS" if barrel_import and not direct_import else "FAIL",
                    "paper_trading_imports_tab_icons_via_canonical_barrel_no_phosphor_direct",
                    "paper-trading/page.tsx must import tab icons from @/lib/icons, NOT @phosphor-icons/react"))

    # Claim 9: phase-25.B12 attribution in all 3 page files + icons.ts
    attr_in_all = all(
        "phase-25.B12" in text
        for text in (perf_text, sov_text, pt_text, icons_text)
    )
    results.append(("PASS" if attr_in_all else "FAIL",
                    "phase_25_B12_attribution_in_all_four_files",
                    "phase-25.B12 attribution comment must appear in all 3 page files + icons.ts"))

    # --- Output ---
    print("=== phase-25.B12 (missing states + tab icons) verifier ===")
    fail = 0
    for flag, name, detail in results:
        prefix = "[PASS]" if flag == "PASS" else "[FAIL]"
        print(f"  {prefix} {name}")
        if flag == "FAIL" and detail:
            print(f"         -> {detail}")
            fail += 1
    total = len(results)
    passed = total - fail
    verdict = "PASS" if fail == 0 else "FAIL"
    print(f"{verdict} ({passed}/{total}) EXIT={0 if fail == 0 else 1}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
