"""phase-25.2: one-shot script to backfill missing stop-loss prices.

Closes phase-24.1 audit finding F-5: 6 of 11 current positions
(ON, INTC, TER, DELL, GLW, CIEN) pre-date the phase-23.1.8 entry-path
fallback and have stop_loss_price=None.

Run (interactive confirm):
  source .venv/bin/activate
  python scripts/maintenance/backfill_stops.py

Run (non-interactive, e.g., in CI fixture):
  python scripts/maintenance/backfill_stops.py --yes

After backfill, the next autonomous cycle's Step 5.6 stop-loss
enforcement (phase-25.1) will sell positions already below their
newly-set stop (e.g., TER -12.30%).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill missing stop_loss_price on open positions.")
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--default-pct", type=float, default=None,
        help="Stop percentage below entry. Defaults to settings.paper_default_stop_loss_pct (8.0).",
    )
    args = parser.parse_args()

    from backend.services.paper_trader import PaperTrader

    trader = PaperTrader()

    positions = trader.get_positions()
    stop_less = [p for p in positions if not p.get("stop_loss_price")]
    print(f"Found {len(positions)} open positions; {len(stop_less)} have stop_loss_price=None.")

    if not stop_less:
        print("Nothing to backfill. Exiting.")
        return 0

    print()
    print("Positions to backfill:")
    for pos in stop_less:
        entry = float(pos.get("avg_entry_price") or 0.0)
        default_pct = args.default_pct or float(getattr(trader.settings, "paper_default_stop_loss_pct", 8.0))
        projected_stop = round(entry * (1.0 - default_pct / 100.0), 4) if entry > 0 else 0.0
        current = float(pos.get("current_price") or 0.0)
        will_trigger = "WILL TRIGGER STOP NEXT CYCLE" if (0 < projected_stop and 0 < current <= projected_stop) else ""
        print(f"  - {pos['ticker']}: entry=${entry:.4f}, projected_stop=${projected_stop:.4f}, current=${current:.4f} {will_trigger}")

    if not args.yes:
        print()
        try:
            answer = input("Proceed with backfill? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted by operator.")
            return 1

    print()
    print("Backfilling...")
    result = trader.backfill_missing_stops(default_pct=args.default_pct)

    print()
    print(f"Backfilled {result['count_backfilled']} positions:")
    for row in result["backfilled"]:
        print(f"  + {row['ticker']}: entry=${row['entry_price']:.4f} -> stop=${row['stop_loss_price']:.4f}")
    if result["skipped"]:
        print(f"Skipped {result['count_skipped']}: {', '.join(result['skipped'])}")

    print()
    print("phase-25.2 backfill complete. Next autonomous cycle's Step 5.6 will sell any position below its new stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
