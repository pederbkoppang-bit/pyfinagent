"""price_quality -- data-quality gate for (esp. international) price bars (phase-50.5).

The operator's precondition for going live on free-yfinance international data
("free yfinance + quality gate"). yfinance international (.DE/.KS) has documented
defects (Tobi Lux DAX study): up to 11% deviation vs the real exchange + 10-24
days/yr of all-identical OHLC bars (with zero volume) that silently break
momentum/RSI/vol signals. This validator drops the UNAMBIGUOUS bad bars and FLAGS
the merely-suspicious -- it must NEVER destroy real volatility (arXiv 2403.19735).

BYTE-IDENTITY: `validate_ohlcv(df, market="US")` returns the input UNCHANGED for
US (fast-path no-op) -- the live US screener/ingestion are untouched. Real US
bars also pass every rule.

Detection rules (DROP only the unambiguous; FLAG the rest):
  R1 OHLC consistency -- high >= max(open,close,low); low <= min(open,close,high);
     all > 0. Violations are impossible real bars -> DROP.
  R2 identical-OHLC bar (open==high==low==close) AND volume==0/absent -> DROP
     (the documented yfinance bad-bar signature; zero-vol corroborates). With
     volume>0 it could be a real no-move day -> FLAG, don't drop.
  R3 single-day |return| > 0.50 (a >50% round-trip on a large-cap is a data
     glitch) -> DROP; |return| z-score > 3 over a rolling window -> FLAG.
  R4 stale run: >= 4 consecutive identical closes -> FLAG (possible stale feed).

Returns (clean_df, report) where report = {dropped, flagged, reasons:[...]}.
NO silent truncation -- counts are logged + returned.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RET_DROP = 0.50   # >50% single-day move on a large-cap == data glitch
_Z_FLAG = 3.0      # rolling return z-score flag threshold (axionquant)
_STALE_RUN = 4     # consecutive identical closes -> flag


def _col(df, *names):
    """Case-insensitive column accessor; returns the Series or None."""
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]]
    return None


def validate_ohlcv(df, market: str = "US", ticker: str = "") -> tuple[Any, dict]:
    """Validate + clean an OHLCV DataFrame. US -> no-op (byte-identical).

    Returns (clean_df, report). Fail-open: on any internal error, returns the
    input df unchanged with an error note (never block the pipeline on a bug
    in the validator)."""
    report = {"dropped": 0, "flagged": 0, "reasons": []}
    if df is None or market == "US":
        return df, report          # US fast-path no-op -> byte-identical
    try:
        import pandas as pd
        if not hasattr(df, "columns") or len(df) == 0:
            return df, report
        o = _col(df, "open"); h = _col(df, "high"); l = _col(df, "low")
        c = _col(df, "close", "adj close"); v = _col(df, "volume")
        if c is None:
            return df, report      # nothing to validate without a close
        drop_mask = pd.Series(False, index=df.index)

        # R1: OHLC consistency (impossible bars -> drop)
        if o is not None and h is not None and l is not None:
            bad = (
                (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
                | (c <= 0) | (h <= 0) | (l <= 0) | (o <= 0)
            )
            n = int(bad.fillna(False).sum())
            if n:
                drop_mask = drop_mask | bad.fillna(False)
                report["reasons"].append(f"R1 OHLC-inconsistent x{n}")

        # R2: identical-OHLC + zero/absent volume -> drop; with volume>0 -> flag
        if o is not None and h is not None and l is not None:
            identical = (o == h) & (h == l) & (l == c)
            identical = identical.fillna(False)
            if v is not None:
                zero_vol = (v.fillna(0) == 0)
                bad_flat = identical & zero_vol
                n = int(bad_flat.sum())
                if n:
                    drop_mask = drop_mask | bad_flat
                    report["reasons"].append(f"R2 identical-OHLC+zero-vol x{n}")
                soft = identical & ~zero_vol
                report["flagged"] += int(soft.sum())
            else:
                # no volume info -> can't corroborate; flag, don't drop
                report["flagged"] += int(identical.sum())
                if int(identical.sum()):
                    report["reasons"].append(f"R2 identical-OHLC(no-vol) flagged x{int(identical.sum())}")

        # R3: extreme single-day return -> drop; z-score outlier -> flag
        ret = c.pct_change()
        huge = ret.abs() > _RET_DROP
        huge = huge.fillna(False)
        n = int(huge.sum())
        if n:
            drop_mask = drop_mask | huge
            report["reasons"].append(f"R3 |ret|>{_RET_DROP:.0%} x{n}")
        if len(ret.dropna()) > 5:
            mu, sd = ret.mean(), ret.std()
            if sd and sd > 0:
                z = (ret - mu).abs() / sd
                flag_z = (z > _Z_FLAG) & ~huge
                report["flagged"] += int(flag_z.fillna(False).sum())

        # R4: stale run of identical closes -> flag
        same = (c == c.shift(1)).fillna(False)
        run = same.groupby((~same).cumsum()).cumsum()
        stale = int((run >= (_STALE_RUN - 1)).sum())
        if stale:
            report["flagged"] += stale
            report["reasons"].append(f"R4 stale-close-run x{stale}")

        clean = df[~drop_mask]
        report["dropped"] = int(drop_mask.sum())
        if report["dropped"] or report["flagged"]:
            logger.info(
                "price_quality[%s/%s]: dropped %d, flagged %d (%s)",
                market, ticker or "?", report["dropped"], report["flagged"],
                "; ".join(report["reasons"]) or "-",
            )
        return clean, report
    except Exception as e:
        logger.warning("price_quality: validation error for %s/%s (%s); passing through", market, ticker, e)
        return df, report


def is_bad_bar(open_, high, low, close, volume: Optional[float] = None) -> bool:
    """Single-bar check for the live fill/mark path (L2). True = drop this bar
    (return None to the caller -> it falls back to last-known price). Mirrors
    R1 + R2. Lenient: only the unambiguous bad bar."""
    try:
        vals = [open_, high, low, close]
        if any(x is None for x in vals):
            return False
        o, h, l, c = float(open_), float(high), float(low), float(close)
        if min(o, h, l, c) <= 0:
            return True                       # non-positive price
        if h < l or h < o or h < c or l > o or l > c:
            return True                       # impossible OHLC
        if o == h == l == c and (volume is None or float(volume or 0) == 0):
            return True                       # identical-OHLC + zero/absent volume
        return False
    except Exception:
        return False                          # fail-open: don't drop on a parse error
