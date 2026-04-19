"""phase-3.3 Regime Detection.

VIX rolling-quantile classifier. Adopted over HMM for one concrete reason:
zero new runtime dependency. Research (2026-04-19 brief) shows VIX quantile
classification matches HMM accuracy on daily-bar US equities.

Classifier labels each trading day as `low_vol` / `medium_vol` / `high_vol`
based on that day's VIX close relative to a trailing 252-day rolling
quantile band (defaults: 33rd + 67th percentiles). Consecutive same-label
days are merged into one regime window -- this produces the
`[{name, start_date, end_date}, ...]` shape the downstream consumer
`backend/backtest/spot_checks.py::RegimeShiftTest` expects at line 181-194.

Fail-open: any yfinance / pandas exception -> static 2-regime pre/post-COVID
fallback (same shape `spot_checks.py:172-175` already produces when
`regime_detector=None`). Never raises out of `.detect()`.

Settings gate: `settings.regime_detection_enabled=False` by default; harness
wiring at `spot_checks_harness.py:80` instantiates this detector only when
the flag is True, preserving existing behavior unchanged.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


_FALLBACK_REGIMES = [
    {"name": "Pre-COVID", "start_date": "2018-01-01", "end_date": "2020-03-15"},
    {"name": "Post-COVID", "start_date": "2020-03-16", "end_date": "2025-12-31"},
]


@runtime_checkable
class RegimeDetector(Protocol):
    """Structural protocol consumed by `spot_checks.RegimeShiftTest`."""

    def detect(self) -> list[dict[str, Any]]:
        """Return regime windows as `[{name, start_date, end_date}, ...]`."""
        ...


class VIXRollingQuantileRegimeDetector:
    """Rolling-quantile VIX-based regime detector.

    Labels each day as `low_vol` / `medium_vol` / `high_vol` by comparing
    the day's VIX close to the trailing `window_days` rolling quantiles.
    Consecutive same-label days are merged into one regime window.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        *,
        low_q: float = 0.33,
        high_q: float = 0.67,
        window_days: int = 252,
        vix_symbol: str = "^VIX",
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.low_q = float(low_q)
        self.high_q = float(high_q)
        self.window_days = int(window_days)
        self.vix_symbol = vix_symbol

    def detect(self) -> list[dict[str, Any]]:
        try:
            closes = self._fetch_vix_closes()
            if closes is None or len(closes) < self.window_days:
                logger.warning(
                    "VIXRollingQuantileRegimeDetector: insufficient VIX data "
                    "(%s rows); using fallback",
                    0 if closes is None else len(closes),
                )
                return list(_FALLBACK_REGIMES)
            regimes = self._classify_and_merge(closes)
            if not regimes:
                logger.warning(
                    "VIXRollingQuantileRegimeDetector: classifier produced "
                    "no regimes; using fallback"
                )
                return list(_FALLBACK_REGIMES)
            return regimes
        except Exception as exc:
            logger.warning(
                "VIXRollingQuantileRegimeDetector fail-open: %r", exc
            )
            return list(_FALLBACK_REGIMES)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fetch_vix_closes(self):
        """Return a pandas Series of VIX closes indexed by date, or None."""
        try:
            import yfinance as yf  # local import: optional dep in tests
            import pandas as pd  # noqa: F401
        except Exception as exc:
            logger.debug("yfinance/pandas unavailable: %r", exc)
            return None
        try:
            ticker = yf.Ticker(self.vix_symbol)
            hist = ticker.history(start=self.start_date, end=self.end_date)
            if hist is None or hist.empty:
                return None
            closes = hist["Close"].dropna()
            return closes
        except Exception as exc:
            logger.debug("yfinance fetch failed: %r", exc)
            return None

    def classify_series(self, closes):
        """Classify a pd.Series of VIX closes day-by-day. Public for tests."""
        import pandas as pd  # noqa: F401

        rolling_low = closes.rolling(self.window_days, min_periods=1).quantile(self.low_q)
        rolling_high = closes.rolling(self.window_days, min_periods=1).quantile(self.high_q)

        labels = []
        for i, value in enumerate(closes):
            lo = rolling_low.iloc[i]
            hi = rolling_high.iloc[i]
            if value <= lo:
                labels.append("low_vol")
            elif value >= hi:
                labels.append("high_vol")
            else:
                labels.append("medium_vol")
        return labels

    def _classify_and_merge(self, closes) -> list[dict[str, Any]]:
        labels = self.classify_series(closes)
        return self._merge_runs(list(closes.index), labels)

    @staticmethod
    def _merge_runs(dates, labels) -> list[dict[str, Any]]:
        """Collapse consecutive same-label days into `[{name, start_date, end_date}]`."""
        if not dates or not labels or len(dates) != len(labels):
            return []
        out: list[dict[str, Any]] = []
        cur_label = labels[0]
        cur_start = _to_date_str(dates[0])
        cur_end = cur_start
        for i in range(1, len(labels)):
            if labels[i] == cur_label:
                cur_end = _to_date_str(dates[i])
            else:
                out.append(
                    {
                        "name": cur_label,
                        "start_date": cur_start,
                        "end_date": cur_end,
                    }
                )
                cur_label = labels[i]
                cur_start = _to_date_str(dates[i])
                cur_end = cur_start
        out.append(
            {"name": cur_label, "start_date": cur_start, "end_date": cur_end}
        )
        return out


def _to_date_str(d: Any) -> str:
    """Coerce pandas Timestamp / datetime / str to ISO YYYY-MM-DD."""
    try:
        if hasattr(d, "strftime"):
            return d.strftime("%Y-%m-%d")
        return str(d)[:10]
    except Exception:
        return str(d)[:10]


__all__ = [
    "RegimeDetector",
    "VIXRollingQuantileRegimeDetector",
]
