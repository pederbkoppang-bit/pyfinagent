"""phase-4.6 step 4.6.6: Paper-trading 5 tabs render without error.

Usage:
    python scripts/smoketest/steps/frontend_tabs.py --base http://localhost:3000 \
        --tabs positions,trades,chart,reality-gap,exit-quality

Exits 0 on PASS, non-zero on FAIL. Emits JSON to stdout.

Design (evidence in handoff/current/contract.md research gate):
- Playwright (chromium-headless-shell) drives a real browser so we can
  (a) execute JS and render client-side tabs, (b) capture console
  errors (TypeError / ReferenceError), and (c) query the DOM for the
  rose-500 error banner.
- Backend calls are INTERCEPTED and mocked -- 4.6.6 is a frontend-
  rendering smoketest, not an integration test (backend integration is
  4.6.3 + 4.6.4). Mocking removes auth/CORS/BQ dependencies and makes
  the test hermetic and reproducible.
- Visits the SPA once, then clicks each of the 5 tab buttons and
  asserts (a) label visible, (b) no rose-500 error banner, (c) no
  console TypeError/ReferenceError.
"""
import argparse
import asyncio
import json
import re
import sys

TAB_LABEL = {
    "positions": "Positions",
    "trades": "Trades",
    "chart": "NAV Chart",
    "reality-gap": "Reality gap",
    "exit-quality": "Exit quality",
}
CONSOLE_BLACKLIST_RE = re.compile(r"\b(TypeError|ReferenceError)\b")
ROSE_BANNER_SEL = "div[class*='border-rose-500']"

# Hermetic mocks for backend endpoints touched by /paper-trading page.
MOCKS = {
    "/api/auth/session": {"user": {"email": "smoketest@local", "name": "Smoketest"}},
    "/api/paper-trading/status": {
        "status": "active",
        "portfolio": {"nav": 10000.0, "cash": 10000.0, "starting_capital": 10000.0,
                      "pnl_pct": 0.0, "benchmark_return_pct": 0.0,
                      "inception_date": "2026-01-01T00:00:00+00:00",
                      "updated_at": "2026-04-17T00:00:00+00:00"},
        "position_count": 0, "scheduler_active": True,
        "next_run": "2026-04-18T14:00:00+02:00",
        "loop": {"running": False, "last_run": "2026-04-17T00:00:00+00:00", "last_result": None},
        "last_run_ts": "2026-04-17T00:00:00+00:00",
    },
    "/api/paper-trading/portfolio": {
        "total_nav": 10000, "current_cash": 10000, "starting_capital": 10000,
        "total_pnl_pct": 0.0, "benchmark_return_pct": 0.0, "positions": [],
    },
    "/api/paper-trading/trades": {"trades": []},
    "/api/paper-trading/snapshots": {"snapshots": []},
    "/api/paper-trading/performance": {
        "sharpe_ratio": None, "max_drawdown_pct": 0.0,
        "total_return_pct": 0.0, "win_rate": None,
    },
    "/api/paper-trading/mfe-mae-scatter": {"trades": []},
    "/api/paper-trading/reality-gap": {"points": []},
    "/api/paper-trading/go-live-gate": {
        "gate": {"paper_14d": False, "sharpe_0_5": False, "sortino_0_7": False,
                 "max_dd_5pct": False, "risk_controls": False},
        "approved": False, "messages": [],
    },
    "/api/paper-trading/kill-switch": {"state": "active", "killed_at": None, "killed_by": None},
    "/api/paper-trading/cycle-health": {"scheduler_healthy": True, "data_fresh": True, "ok": True},
}


async def _handle_route(route):
    path = route.request.url.split("://", 1)[-1].split("/", 1)[-1]
    for prefix, body in MOCKS.items():
        # match both "api/..." (no leading /) and the full path
        if ("/" + path).endswith(prefix) or path.endswith(prefix.lstrip("/")):
            await route.fulfill(status=200, content_type="application/json",
                                body=json.dumps(body))
            return
    # Catch-all for any other /api/* (e.g. /api/paper-trading/kpi/*): generic healthy stub.
    if "/api/" in path:
        await route.fulfill(status=200, content_type="application/json",
                            body=json.dumps({"ok": True}))
        return
    await route.continue_()


async def _run(base: str, tabs: list[str], timeout_ms: int) -> dict:
    from playwright.async_api import async_playwright

    console_log: list[str] = []
    page_errors: list[str] = []

    result = {"step": "4.6.6", "base": base, "per_tab": [],
              "console_errors": [], "page_errors": [], "console_check": "browser"}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        page.on("console", lambda msg: console_log.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Intercept all network calls matching /api/*
        await page.route(re.compile(r".*/api/.*"), _handle_route)

        try:
            await page.goto(f"{base}/paper-trading", timeout=timeout_ms, wait_until="domcontentloaded")
        except Exception as e:
            result["verdict"] = "FAIL"
            result["reason"] = f"nav_error:{type(e).__name__}:{e}"
            await browser.close()
            return result

        # Wait for React to mount + initial data to settle.
        await page.wait_for_timeout(1500)

        for tab in tabs:
            label = TAB_LABEL.get(tab, "")
            entry = {"tab": tab, "label": label}
            try:
                btn = page.get_by_role("button", name=label, exact=True).first
                await btn.click(timeout=timeout_ms)
                await page.wait_for_timeout(300)
                visible = await btn.is_visible()
                banner_count = await page.locator(ROSE_BANNER_SEL).count()
                entry.update({
                    "http_status": 200,
                    "label_visible": visible,
                    "rose_error_banner": banner_count > 0,
                    "ok": visible and banner_count == 0,
                })
            except Exception as e:
                entry.update({"ok": False, "reason": f"{type(e).__name__}:{str(e)[:200]}"})
            result["per_tab"].append(entry)

        await browser.close()

    flat = "\n".join(console_log + page_errors)
    bad = CONSOLE_BLACKLIST_RE.findall(flat)
    result["console_errors"] = [c for c in console_log if CONSOLE_BLACKLIST_RE.search(c)]
    result["page_errors"] = [e for e in page_errors if CONSOLE_BLACKLIST_RE.search(e)]

    all_tabs_ok = all(t.get("ok") for t in result["per_tab"])
    no_bad = not bad
    result["verdict"] = "PASS" if (all_tabs_ok and no_bad) else "FAIL"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000")
    ap.add_argument("--tabs", default="positions,trades,chart,reality-gap,exit-quality")
    ap.add_argument("--timeout", type=int, default=20000)
    args = ap.parse_args()

    tabs = [t.strip() for t in args.tabs.split(",") if t.strip()]
    for t in tabs:
        if t not in TAB_LABEL:
            print(json.dumps({"step": "4.6.6", "verdict": "FAIL", "reason": f"unknown_tab:{t}"}))
            return 2

    result = asyncio.run(_run(args.base, tabs, args.timeout))
    print(json.dumps(result))
    if result["verdict"] == "PASS":
        print("FRONTEND_TABS_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
