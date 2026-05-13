"""verify_phase_25_D7 -- preload_macro() max-age guard.

Verifies:
  1. Module contains `MACRO_MAX_AGE_DAYS = 35` constant.
  2. preload_macro contains a comparison of max-date vs the threshold.
  3. Behavioral: stale rows (age > 35 days) -> WARNING log + return 0 + no cache.
  4. Behavioral: fresh rows (age < 35 days) -> populated cache + return > 0.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: MACRO_MAX_AGE_DAYS = 35 constant ──────────────────────────
src = (REPO / "backend/backtest/cache.py").read_text(encoding="utf-8")
has_constant = bool(re.search(r"MACRO_MAX_AGE_DAYS\s*=\s*35\b", src))
claim(
    "1. macro_max_age_days_constant_35",
    has_constant,
    "Found MACRO_MAX_AGE_DAYS=35" if has_constant else "Constant missing or wrong value",
)


# ── Claim 2: preload_macro contains the comparison ─────────────────────
has_compare = "age_days > MACRO_MAX_AGE_DAYS" in src
has_refuse = "refusing to cache" in src or "refuse to cache" in src
claim(
    "2. preload_macro_checks_max_age_days_35_before_caching",
    has_compare and has_refuse,
    f"compare={has_compare} refuse_msg={has_refuse}",
)


# ── Helpers for behavioral claims ──────────────────────────────────────
class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _mock_row(series: str, value: float, d: date):
    """Return an object that satisfies r.get / r["date"] / dict(r)."""
    return {"series_id": series, "value": value, "date": d}


# We need to mock `_bq_client.query(...).result(...)` to return rows
# and call preload_macro. The function imports `_bq_client` at module level,
# so we need to set it.

import backend.backtest.cache as cache_mod  # noqa: E402


def _setup_mock_bq(rows: list[dict]) -> MagicMock:
    """Install a mock _bq_client that returns the given rows."""
    mock_client = MagicMock()
    mock_query = MagicMock()
    mock_query.result = MagicMock(return_value=iter(rows))
    mock_client.query = MagicMock(return_value=mock_query)
    return mock_client


# ── Claim 3: stale data refuses cache ───────────────────────────────────
handler = _CapturingHandler()
handler.setLevel(logging.DEBUG)
cache_logger = logging.getLogger("backend.backtest.cache")
cache_logger.addHandler(handler)
cache_logger.setLevel(logging.DEBUG)

try:
    today = date.today()
    stale_rows = [
        _mock_row("CPIAUCSL", 295.5, today - timedelta(days=40)),
        _mock_row("UNRATE", 4.0, today - timedelta(days=45)),
    ]
    cache_mod._bq_client = _setup_mock_bq(stale_rows)
    cache_mod._project = "test-project"
    # Reset the cache to ensure preload runs fresh
    cache_mod._macro_full.clear()
    handler.records.clear()
    result = cache_mod.preload_macro()
    stale_records = [r for r in handler.records if r.levelno == logging.WARNING and "stale data" in r.getMessage()]
    rt3_ok = (
        result == 0
        and len(stale_records) >= 1
        and not cache_mod._macro_full  # nothing cached
    )
    rt3_detail = f"return={result} warning_records={len(stale_records)} cache_populated={bool(cache_mod._macro_full)}"
except Exception as e:
    rt3_ok = False
    rt3_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "3. warning_log_emitted_on_stale_data_refuse_to_preload",
    rt3_ok,
    rt3_detail,
)


# ── Claim 4: fresh data caches normally ────────────────────────────────
try:
    today = date.today()
    fresh_rows = [
        _mock_row("CPIAUCSL", 295.5, today - timedelta(days=5)),
        _mock_row("UNRATE", 4.0, today - timedelta(days=10)),
        _mock_row("UNRATE", 3.9, today - timedelta(days=40)),  # older row, ok
    ]
    cache_mod._bq_client = _setup_mock_bq(fresh_rows)
    cache_mod._macro_full.clear()
    handler.records.clear()
    result = cache_mod.preload_macro()
    rt4_ok = result == 3 and len(cache_mod._macro_full) == 2
    rt4_detail = f"return={result} series_count={len(cache_mod._macro_full)}"
except Exception as e:
    rt4_ok = False
    rt4_detail = f"Exception: {type(e).__name__}: {e}"
finally:
    cache_logger.removeHandler(handler)

claim(
    "4. fresh_data_caches_normally",
    rt4_ok,
    rt4_detail,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.D7 verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
