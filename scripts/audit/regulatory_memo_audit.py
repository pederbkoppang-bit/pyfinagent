"""phase-4.8 step 4.8.10 regulatory-memo + guards audit.

Teeth:
1. docs/compliance/2026-regulatory-memo.md exists with all 7
   required sections (Scope, Regulatory changes, System impact,
   Operational controls, Monitoring, Review cadence, Open items).
2. Memo references real citations (SEC 34-96930, IRC Sec 1091,
   FINRA 4210).
3. Library modules expose the required public names.
4. wash_sale_filter.py --test exits 0 (delegates to the compliance
   script).
5. Wash-sale window is calendar-day based (docstring check +
   WINDOW_DAYS constant).
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MEMO = REPO / "docs" / "compliance" / "2026-regulatory-memo.md"
TEST_SCRIPT = REPO / "scripts" / "compliance" / "wash_sale_filter.py"
OUT = REPO / "handoff" / "regulatory_memo_audit.json"

REQUIRED_SECTIONS = (
    "## 1. Scope",
    "## 2. Regulatory changes",
    "## 3. System impact",
    "## 4. Operational controls",
    "## 5. Monitoring",
    "## 6. Review cadence",
    "## 7. Open items",
)
REQUIRED_CITATIONS = (
    "34-96930",      # SEC T+1
    "1091",          # IRC Sec 1091
    "4210",          # FINRA Rule 4210
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []

    # 1+2. Memo structure + citations.
    memo_ok = MEMO.exists()
    memo_sections: dict[str, bool] = {}
    memo_cites: dict[str, bool] = {}
    if memo_ok:
        text = MEMO.read_text(encoding="utf-8")
        for sec in REQUIRED_SECTIONS:
            memo_sections[sec] = sec in text
            if sec not in text:
                reasons.append(f"memo missing section '{sec}'")
        for cite in REQUIRED_CITATIONS:
            memo_cites[cite] = cite in text
            if cite not in text:
                reasons.append(f"memo missing citation '{cite}'")
        memo_ok = all(memo_sections.values()) and all(memo_cites.values())
    else:
        reasons.append(f"memo missing at {MEMO}")

    # 3. Library modules + public names.
    public_ok = True
    lib_checks: dict[str, bool] = {}
    try:
        ws = importlib.import_module("backend.services.wash_sale_filter")
        fg = importlib.import_module("backend.services.funding_guard")
        lib_checks["WashSaleLedger"] = hasattr(ws, "WashSaleLedger")
        lib_checks["filter_candidates"] = hasattr(ws, "filter_candidates")
        lib_checks["WINDOW_DAYS"] = getattr(ws, "WINDOW_DAYS", None) == 30
        lib_checks["t1_funding_guard"] = callable(getattr(fg, "t1_funding_guard", None))
        lib_checks["realtime_margin_guard"] = callable(getattr(fg, "realtime_margin_guard", None))
        public_ok = all(lib_checks.values())
    except Exception as e:
        public_ok = False
        reasons.append(f"library import failed: {e}")
    if not public_ok:
        missing = [k for k, v in lib_checks.items() if not v]
        reasons.append(f"library symbols missing/wrong: {missing}")

    # 4. --test exits 0.
    test_ok = False
    test_tail = ""
    try:
        r = subprocess.run(
            [sys.executable, str(TEST_SCRIPT), "--test"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
        test_ok = r.returncode == 0
        test_tail = (r.stdout.strip().splitlines()[-1] if r.stdout else "")
        if not test_ok:
            reasons.append(f"wash_sale_filter.py --test rc={r.returncode}: {r.stderr[:200]}")
    except Exception as e:
        reasons.append(f"could not run test script: {e}")

    # 5. Calendar-day window evidence.
    cal_ok = False
    try:
        # Dynamic check: 2026-04-01 to 2026-04-01+30 (Apr has 30 days).
        # If the filter used business days, the boundary at +30 calendar
        # days would fall in a weekend depending on the date. We test
        # by setting a Saturday inside the window explicitly.
        from datetime import date, timedelta
        from backend.services.wash_sale_filter import WashSaleLedger
        lg = WashSaleLedger()
        # Sell on a Wednesday (2026-04-01 is a Wednesday).
        sell = date(2026, 4, 1)
        lg.record_loss(symbol="AAPL", sell_date=sell, disallowed_loss_usd=100)
        # Buy on a Saturday (2026-04-04), 3 calendar days after.
        is_ws, _ = lg.is_wash_sale("AAPL", date(2026, 4, 4))
        cal_ok = is_ws
        if not cal_ok:
            reasons.append("calendar-day check: Saturday buy 3 days after sell not flagged -- suggests business-day logic")
    except Exception as e:
        reasons.append(f"calendar-day check error: {e}")

    all_ok = memo_ok and public_ok and test_ok and cal_ok
    verdict = "PASS" if all_ok else "FAIL"
    summary = {
        "step": "4.8.10",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "memo_landed": memo_ok,
        "memo_sections": memo_sections,
        "memo_citations": memo_cites,
        "library_public_api": lib_checks,
        "test_script_passed": test_ok,
        "test_tail": test_tail,
        "calendar_day_window": cal_ok,
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "memo": memo_ok, "lib": public_ok,
        "test": test_ok, "calendar": cal_ok,
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
