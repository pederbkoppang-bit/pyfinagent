"""phase-25.2 verifier — backfill_missing_stops + same-cycle re-check.

Closes phase-24.1 audit F-5 (6 of 11 positions have stop_loss_price=None,
pre-date phase-23.1.8 fallback). Verifier checks code structure +
behavioral round-trip on the helper logic.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_2.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
PAPER_TRADER = REPO / "backend" / "services" / "paper_trader.py"
BACKFILL_SCRIPT = REPO / "scripts" / "maintenance" / "backfill_stops.py"

sys.path.insert(0, str(REPO))


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not PAPER_TRADER.exists():
        print(f"FAIL: {PAPER_TRADER} not found")
        return 1
    if not BACKFILL_SCRIPT.exists():
        print(f"FAIL: {BACKFILL_SCRIPT} not found")
        return 1

    pt_text = PAPER_TRADER.read_text(encoding="utf-8")
    script_text = BACKFILL_SCRIPT.read_text(encoding="utf-8")

    # Claim 1: backfill_missing_stops method defined on PaperTrader
    method_def = re.search(r'def backfill_missing_stops\s*\(', pt_text)
    results.append(("PASS" if method_def else "FAIL",
                    "backfill_missing_stops_method_defined",
                    "PaperTrader.backfill_missing_stops must be defined"))

    # Claim 2: backfill computes stop = entry * (1 - default_pct/100)
    formula_pattern = re.search(
        r'entry_price\s*\*\s*\(\s*1\.0\s*-\s*default_pct\s*/\s*100',
        pt_text,
    )
    results.append(("PASS" if formula_pattern else "FAIL",
                    "backfill_uses_paper_default_stop_loss_pct_against_avg_entry_price",
                    "Stop computation must be entry_price * (1.0 - default_pct / 100.0)"))

    # Claim 3: save_paper_position called from backfill (persists to BQ)
    save_in_backfill = re.search(
        r'def backfill_missing_stops.*?save_paper_position\s*\(',
        pt_text,
        re.DOTALL,
    )
    results.append(("PASS" if save_in_backfill else "FAIL",
                    "backfill_persists_via_save_paper_position",
                    "backfill_missing_stops must call self.bq.save_paper_position"))

    # Claim 4: returns dict with backfilled + skipped + count_backfilled + count_skipped
    return_fields_present = all(
        f'"{field}"' in pt_text or f"'{field}'" in pt_text
        for field in ["backfilled", "skipped", "count_backfilled", "count_skipped"]
    )
    results.append(("PASS" if return_fields_present else "FAIL",
                    "backfill_returns_dict_with_backfilled_skipped_counts",
                    "backfill_missing_stops must return dict with backfilled/skipped/count_backfilled/count_skipped"))

    # Claim 5: one-shot maintenance script exists at expected path
    has_script_entry = (
        "PaperTrader" in script_text
        and "backfill_missing_stops" in script_text
        and "argparse" in script_text
    )
    results.append(("PASS" if has_script_entry else "FAIL",
                    "backfill_stops_script_invokes_papertrader_method",
                    "scripts/maintenance/backfill_stops.py must invoke PaperTrader.backfill_missing_stops"))

    # Claim 6: script has --yes flag for non-interactive use
    has_yes_flag = '"--yes"' in script_text or "'--yes'" in script_text
    results.append(("PASS" if has_yes_flag else "FAIL",
                    "backfill_script_has_yes_flag_for_non_interactive_use",
                    "scripts/maintenance/backfill_stops.py must accept --yes for CI/non-interactive use"))

    # Claim 7: phase-25.2 attribution in paper_trader
    results.append(("PASS" if "phase-25.2" in pt_text else "FAIL",
                    "phase_25_2_attribution_in_paper_trader",
                    "Comment must reference phase-25.2 closure of phase-24.1 F-5"))

    # Claim 8: AST clean
    try:
        ast.parse(pt_text)
        results.append(("PASS", "paper_trader_py_syntax_clean", ""))
    except SyntaxError as e:
        results.append(("FAIL", "paper_trader_py_syntax_clean", f"SyntaxError: {e}"))

    try:
        ast.parse(script_text)
        results.append(("PASS", "backfill_stops_script_syntax_clean", ""))
    except SyntaxError as e:
        results.append(("FAIL", "backfill_stops_script_syntax_clean", f"SyntaxError: {e}"))

    # Claim 9: behavioral round-trip — mock trader.get_positions to return one
    # stop-less position, call backfill, confirm save_paper_position invoked
    # with stop_loss_price computed correctly.
    try:
        from backend.services.paper_trader import PaperTrader  # noqa: E402

        # Construct without DI: bypass __init__ via __new__
        trader = PaperTrader.__new__(PaperTrader)
        trader.bq = MagicMock()
        trader.settings = MagicMock()
        trader.settings.paper_default_stop_loss_pct = 8.0
        trader.get_positions = MagicMock(return_value=[
            {"ticker": "TER", "avg_entry_price": 100.0, "stop_loss_price": None, "current_price": 87.7},
            {"ticker": "FIX", "avg_entry_price": 50.0, "stop_loss_price": 46.0, "current_price": 51.0},
        ])

        result = trader.backfill_missing_stops()

        # Should backfill TER (no stop) and skip FIX (already has stop)
        assert result["count_backfilled"] == 1, f"expected 1 backfilled, got {result['count_backfilled']}"
        assert result["count_skipped"] == 1, f"expected 1 skipped, got {result['count_skipped']}"
        assert result["backfilled"][0]["ticker"] == "TER"
        # 100 * (1 - 8/100) = 92.0
        assert result["backfilled"][0]["stop_loss_price"] == 92.0, \
            f"expected stop 92.0, got {result['backfilled'][0]['stop_loss_price']}"
        # save_paper_position called once with TER
        trader.bq.save_paper_position.assert_called_once()
        results.append(("PASS", "behavioral_round_trip_backfills_stopless_skips_existing", ""))
    except Exception as e:
        results.append(("FAIL", "behavioral_round_trip_backfills_stopless_skips_existing",
                        f"unexpected exception: {e!r}"))

    # --- Output ---
    print("=== phase-25.2 (backfill missing stops) verifier ===")
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
