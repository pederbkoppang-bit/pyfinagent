"""phase-5.6 Options snapshot ingestion CLI.

Module-as-script entrypoint:
    python -m backend.markets.options.options_ingestion --underlyings SPY QQQ --dry-run

Dry-run mode prints what WOULD be ingested per underlying and exits 0
without any BQ writes or alpaca-py calls. The non-dry-run path
(future cycle) would fetch the active option chain from Alpaca via
alpaca-py + write rows to `pyfinagent_hdw.options_snapshots`. Without
both creds and the BQ table created (separate user-action --apply via
`scripts/migrations/create_options_snapshots_table.py`), the script
fails-open and exits 0.

ASCII-only logging per `.claude/rules/security.md`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger("options_ingestion")

DEFAULT_UNDERLYINGS: tuple[str, ...] = ("SPY", "QQQ", "IWM")
TABLE_FQN = "pyfinagent_hdw.options_snapshots"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _has_alpaca_creds() -> bool:
    return bool(
        os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")
    )


def _dry_run_ingest(underlyings: list[str]) -> int:
    """Dry-run: log what would be ingested. No I/O. Returns 0."""
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        "DRY-RUN: would ingest options snapshots for %d underlyings", len(underlyings)
    )
    logger.info("DRY-RUN: target table = %s", TABLE_FQN)
    logger.info("DRY-RUN: snapshot ts = %s", ts)
    for u in underlyings:
        logger.info(
            "DRY-RUN: underlying=%s would fetch active option chain (~30-DTE focus)", u
        )
    logger.info("DRY-RUN: complete; no BQ writes performed")
    return 0


def _live_ingest(underlyings: list[str]) -> int:
    """Live ingest path. Fail-open when creds absent or alpaca-py options
    module missing or BQ table absent. Returns 0 on success-or-fail-open,
    non-zero only on truly unexpected exceptions."""
    if not _has_alpaca_creds():
        logger.warning(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing; fail-open exit 0"
        )
        return 0

    try:
        # Lazy import to avoid module-load overhead in dry-run.
        from alpaca.data.historical import OptionHistoricalDataClient  # type: ignore  # noqa: F401
    except Exception as exc:
        logger.warning(
            "alpaca-py options module unavailable (Level 3 SDK): %r; fail-open exit 0",
            exc,
        )
        return 0

    logger.warning(
        "live options ingest not yet wired -- requires Alpaca Options Level 3 keys "
        "+ pyfinagent_hdw.options_snapshots BQ table (run "
        "scripts/migrations/create_options_snapshots_table.py --apply first); "
        "exit 0"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="options_ingestion",
        description=(
            "Fetch active option chains for the requested underlyings and "
            "persist to pyfinagent_hdw.options_snapshots. --dry-run prints "
            "what would happen without any I/O."
        ),
    )
    parser.add_argument(
        "--underlyings",
        nargs="+",
        default=list(DEFAULT_UNDERLYINGS),
        help="Underlying tickers to fetch (default: SPY QQQ IWM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be ingested without making any BQ or network calls.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging.",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    underlyings = [u.strip().upper() for u in args.underlyings if u.strip()]
    if not underlyings:
        logger.error("no underlyings specified")
        return 2

    if args.dry_run:
        return _dry_run_ingest(underlyings)
    return _live_ingest(underlyings)


if __name__ == "__main__":
    sys.exit(main())
