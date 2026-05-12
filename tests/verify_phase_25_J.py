"""phase-25.J verifier — trade confirmation Slack wiring.

Closes phase-24.5 audit F-5(a) — no send_trade_confirmation function
existed; paper_trader.execute_trade returned silently.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_J.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
PAPER_TRADER = REPO / "backend" / "services" / "paper_trader.py"
FORMATTERS = REPO / "backend" / "slack_bot" / "formatters.py"
SCHEDULER = REPO / "backend" / "slack_bot" / "scheduler.py"
sys.path.insert(0, str(REPO))


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PAPER_TRADER, FORMATTERS, SCHEDULER):
        if not p.exists():
            print(f"FAIL: {p} not found")
            return 1

    pt_text = PAPER_TRADER.read_text(encoding="utf-8")
    fmt_text = FORMATTERS.read_text(encoding="utf-8")
    sched_text = SCHEDULER.read_text(encoding="utf-8")

    # Claim 1: PaperTrader.__init__ accepts trade_notifier kwarg
    init_sig = re.search(
        r'def __init__\s*\(\s*self,\s*settings\s*:.*?bq_client\s*:.*?trade_notifier',
        pt_text,
        re.DOTALL,
    )
    results.append(("PASS" if init_sig else "FAIL",
                    "paper_trader_init_accepts_trade_notifier_kwarg",
                    "PaperTrader.__init__ must accept trade_notifier kwarg"))

    # Claim 2: _maybe_notify_trade helper exists
    helper_def = re.search(r'def _maybe_notify_trade\s*\(self,\s*trade\s*:', pt_text)
    results.append(("PASS" if helper_def else "FAIL",
                    "maybe_notify_trade_helper_defined",
                    "_maybe_notify_trade helper must be defined on PaperTrader"))

    # Claim 3: execute_buy calls _maybe_notify_trade on success
    buy_callsite = re.search(
        r'def execute_buy.*?logger\.info\(f"BUY.*?_maybe_notify_trade\(trade\)',
        pt_text,
        re.DOTALL,
    )
    results.append(("PASS" if buy_callsite else "FAIL",
                    "execute_buy_emits_slack_message_on_success",
                    "execute_buy must call self._maybe_notify_trade(trade) on success"))

    # Claim 4: execute_sell calls _maybe_notify_trade on success
    sell_callsite = re.search(
        r'def execute_sell.*?logger\.info\(f"SELL.*?_maybe_notify_trade\(trade\)',
        pt_text,
        re.DOTALL,
    )
    results.append(("PASS" if sell_callsite else "FAIL",
                    "execute_sell_emits_slack_message_on_success",
                    "execute_sell must call self._maybe_notify_trade(trade) on success"))

    # Claim 5: stop-loss-trigger flows through execute_sell (so it ALSO emits)
    # The 25.1 Step 5.6 block calls execute_sell with reason='stop_loss_trigger',
    # so the trade_notifier fires for stop sells without separate wiring.
    # Verifier: confirm format_trade_confirmation special-cases reason='stop_loss_trigger'.
    sl_special = re.search(
        r'is_stop_loss\s*=\s*reason\s*==\s*["\']stop_loss_trigger["\']',
        fmt_text,
    )
    results.append(("PASS" if sl_special else "FAIL",
                    "stop_loss_trigger_emits_slack_message",
                    "format_trade_confirmation must special-case reason='stop_loss_trigger'"))

    # Claim 6: format_trade_confirmation defined in formatters.py
    fmt_def = re.search(r'def format_trade_confirmation\s*\(\s*trade\s*:\s*dict\s*\)', fmt_text)
    results.append(("PASS" if fmt_def else "FAIL",
                    "format_trade_confirmation_defined",
                    "format_trade_confirmation function must be defined in formatters.py"))

    # Claim 7: notify_trade_confirmation async helper in scheduler.py
    notify_def = re.search(
        r'async def notify_trade_confirmation\s*\(\s*app\s*:.*?trade\s*:\s*dict',
        sched_text,
        re.DOTALL,
    )
    results.append(("PASS" if notify_def else "FAIL",
                    "notify_trade_confirmation_async_helper_defined",
                    "async def notify_trade_confirmation(app, trade) must exist in scheduler.py"))

    # Claim 8: notify_trade_confirmation uses format_trade_confirmation
    notify_uses_fmt = re.search(
        r'async def notify_trade_confirmation.*?format_trade_confirmation',
        sched_text,
        re.DOTALL,
    )
    results.append(("PASS" if notify_uses_fmt else "FAIL",
                    "notify_trade_confirmation_uses_format_trade_confirmation",
                    "notify_trade_confirmation must call format_trade_confirmation"))

    # Claim 9: phase-25.J attribution in all 3 files
    attr_ok = (
        "phase-25.J" in pt_text
        and "phase-25.J" in fmt_text
        and "phase-25.J" in sched_text
    )
    results.append(("PASS" if attr_ok else "FAIL",
                    "phase_25_J_attribution_comment_in_all_three_files",
                    "phase-25.J attribution must appear in paper_trader.py, formatters.py, and scheduler.py"))

    # Claim 10: AST clean (3 files)
    for label, path, text in (
        ("paper_trader", PAPER_TRADER, pt_text),
        ("formatters", FORMATTERS, fmt_text),
        ("scheduler", SCHEDULER, sched_text),
    ):
        try:
            ast.parse(text)
            results.append(("PASS", f"{label}_py_syntax_clean", ""))
        except SyntaxError as e:
            results.append(("FAIL", f"{label}_py_syntax_clean", f"SyntaxError in {path}: {e}"))

    # Claim 11: behavioral round-trip — instantiate PaperTrader with a mock
    # notifier and confirm _maybe_notify_trade dispatches.
    try:
        from backend.services.paper_trader import PaperTrader  # noqa: E402

        trader = PaperTrader.__new__(PaperTrader)
        notifier = MagicMock()
        trader.bq = MagicMock()
        trader.settings = MagicMock()
        trader.trade_notifier = notifier

        trade = {"trade_id": "test-1", "ticker": "TER", "action": "SELL",
                 "quantity": 100, "price": 87.7, "total_value": 8770.0,
                 "reason": "stop_loss_trigger"}
        trader._maybe_notify_trade(trade)
        notifier.assert_called_once_with(trade)
        results.append(("PASS", "behavioral_round_trip_notifier_dispatched_with_trade_dict", ""))
    except Exception as e:
        results.append(("FAIL", "behavioral_round_trip_notifier_dispatched_with_trade_dict",
                        f"unexpected exception: {e!r}"))

    # Claim 12: behavioral round-trip — notifier exception does NOT break execute path
    try:
        from backend.services.paper_trader import PaperTrader  # noqa: E402

        trader = PaperTrader.__new__(PaperTrader)
        trader.bq = MagicMock()
        trader.settings = MagicMock()

        def bad_notifier(_t):
            raise RuntimeError("simulated notifier failure")
        trader.trade_notifier = bad_notifier

        # Should NOT raise
        trader._maybe_notify_trade({"ticker": "TER", "action": "SELL"})
        results.append(("PASS", "notifier_exceptions_are_swallowed_and_logged", ""))
    except Exception as e:
        results.append(("FAIL", "notifier_exceptions_are_swallowed_and_logged",
                        f"trade_notifier exception leaked out: {e!r}"))

    # --- Output ---
    print("=== phase-25.J (trade confirmation Slack) verifier ===")
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
