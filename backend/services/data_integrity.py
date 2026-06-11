"""phase-60.3 (AW-9): deterministic decision-input integrity for non-USD markets.

The away week rendered KRW prices/market caps with '$' literals into the lite
LLM prompts ("Market Cap: $1630000.0B" -- $1.63 quadrillion), the lite Risk
Judge CORRECTLY flagged the corruption in prose ("physically impossible...
KRW/USD unit error", 066570.KS 2026-06-09) and nothing acted on it -- the BUY
executed and stopped out at -9.7%. Guard verdicts must be ENFORCED by code at
a chokepoint before execution (GuardAgent arXiv:2406.09187; deterministic
guardrails arXiv:2604.01483) -- prose-only flagging is the documented
anti-pattern.

Pure functions, no I/O beyond the cached FX lookup; callers fail open.

- `normalize_market_values` -- currency + USD conversion (REUSES
  fx_rates.get_fx_rate, 6h cache) + as-of staleness from regularMarketTime.
- `check_data_integrity` -- deterministic flags; `blocking=True` flags are
  enforced (candidate excluded pre-LLM) when paper_data_integrity_enabled.
- `render_market_lines` -- the prompt presentation: US tickers byte-identical
  to the historical f-string in BOTH flag states; non-US flag-ON renders
  USD-converted values + an as-of line.

Sanity bounds (researcher-cited, 2026-06-11): largest real market cap is
NVDA $4.854T -> $10T post-USD-normalization ceiling (2x headroom, still
catches the away week's $44.5T). P/E exactly 0 on a mega-cap is a
missing-data artifact, never a real value (tag, not block). Currency
mismatch: the yfinance suffix is ground truth for the listing market
(yfinance #2699); `info.currency` disagreeing with the suffix-implied
currency is a corruption tell.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.backtest.markets import market_for_symbol

logger = logging.getLogger(__name__)

# Post-USD-normalization plausibility ceiling (USD). NVDA, the largest real
# market cap, is $4.854T (2026-06-10); nothing real is within 2x of 10T.
MARKET_CAP_CEILING_USD = 10e12
# Large-cap floor for the P/E==0 missing-data tag (USD). P/E exactly 0 on
# an established large-cap (the 066570.KS row: $32B converted, pe=0.0) is a
# missing-data artifact; on a micro-cap it can be real (no earnings).
LARGE_CAP_FLOOR_USD = 10e9

_MARKET_CURRENCY = {
    "US": "USD", "KR": "KRW", "EU": "EUR", "NO": "NOK",
    "SE": "SEK", "DK": "DKK", "FI": "EUR", "IS": "ISK", "CA": "CAD",
}


def normalize_market_values(ticker: str, info: dict) -> dict:
    """Currency-normalize the yfinance info values used by the lite prompts.

    Never raises. `fx_available=False` (non-US, no rate) means the values
    CANNOT be unit-verified -- check_data_integrity turns that into a
    blocking flag so corrupted magnitudes never reach an LLM prompt.
    """
    market = market_for_symbol(ticker)
    expected_ccy = _MARKET_CURRENCY.get(market, "USD")
    info_ccy = str(info.get("currency") or "").upper() or None

    out = {
        "market": market,
        "is_us": market == "US",
        "currency": expected_ccy,
        "info_currency": info_ccy,
        "fx_rate": None,
        "fx_available": market == "US",
        "price_usd": None,
        "market_cap_usd": None,
        "as_of": None,
        "as_of_age_hours": None,
    }

    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    mcap = info.get("marketCap") or 0

    if market == "US":
        out["price_usd"] = float(price) if price else None
        out["market_cap_usd"] = float(mcap) if mcap else None
    else:
        try:
            from backend.services.fx_rates import get_fx_rate

            rate = get_fx_rate(expected_ccy, "USD")
        except Exception as exc:  # fail open -> unverifiable
            logger.warning("FX lookup failed for %s (%s): %s", ticker, expected_ccy, exc)
            rate = None
        if rate:
            out["fx_rate"] = float(rate)
            out["fx_available"] = True
            out["price_usd"] = float(price) * float(rate) if price else None
            out["market_cap_usd"] = float(mcap) * float(rate) if mcap else None

    # As-of staleness from the quote's own timestamp (NOT a hardcoded close
    # constant -- KRX adds extended sessions 2026-06-29).
    rmt = info.get("regularMarketTime")
    if rmt:
        try:
            ts = datetime.fromtimestamp(float(rmt), tz=timezone.utc)
            out["as_of"] = ts.isoformat()
            out["as_of_age_hours"] = round(
                (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0, 1
            )
        except (ValueError, TypeError, OSError):
            pass
    return out


def check_data_integrity(ticker: str, info: dict, normalized: dict) -> list[dict]:
    """Deterministic implausibility flags. blocking=True flags are enforced
    in code (pre-LLM exclusion) when paper_data_integrity_enabled."""
    flags: list[dict] = []
    mcap_raw = float(info.get("marketCap") or 0)
    pe = info.get("trailingPE")
    mcap_usd = normalized.get("market_cap_usd")

    # 1. Implausible market cap. For unit-verified values, check USD; for
    # unverifiable non-US values the RAW magnitude is what the away-week
    # prompts presented as dollars -- exactly the corruption class.
    effective_mcap = mcap_usd if mcap_usd is not None else mcap_raw
    if effective_mcap and effective_mcap > MARKET_CAP_CEILING_USD:
        flags.append({
            "flag": "implausible_market_cap",
            "blocking": True,
            "detail": f"market cap {effective_mcap:.3e} exceeds the {MARKET_CAP_CEILING_USD:.0e} USD ceiling "
                      f"(largest real cap ~4.9e12); KRW/USD unit corruption class (AW-9)",
        })

    # 2. Non-US with no FX rate: magnitudes cannot be unit-verified; the
    # away week showed what an unverified KRW number does inside a $-prompt.
    if not normalized.get("is_us") and not normalized.get("fx_available"):
        flags.append({
            "flag": "currency_unverified",
            "blocking": True,
            "detail": f"{normalized.get('currency')} values with no FX rate -- cannot present truthfully",
        })

    # 3. Currency mismatch: suffix-implied currency vs yfinance info.currency
    # (suffix is ground truth per yfinance #2699).
    info_ccy = normalized.get("info_currency")
    if info_ccy and info_ccy != normalized.get("currency"):
        flags.append({
            "flag": "currency_mismatch",
            "blocking": True,
            "detail": f"suffix market {normalized.get('market')} implies {normalized.get('currency')} "
                      f"but info.currency={info_ccy}",
        })

    # 4. P/E exactly 0 on a large-cap: missing-data artifact (the 066570.KS
    # row carried pe_ratio=0.0). Tag-only -- the value is treated as missing,
    # not as evidence of corruption.
    if pe is not None and float(pe) == 0.0 and (mcap_usd or 0) > LARGE_CAP_FLOOR_USD:
        flags.append({
            "flag": "missing_pe_large_cap",
            "blocking": False,
            "detail": "trailingPE exactly 0.0 on an established large-cap is a missing-data artifact, treated as missing",
        })
    return flags


def render_market_lines(
    ticker: str,
    current_price: float,
    market_cap: float,
    pe_ratio: float,
    normalized: dict,
    integrity_enabled: bool,
) -> str:
    """The Price/MarketCap/PE prompt line (+ as-of line for non-US).

    US tickers -- and EVERYTHING when the flag is OFF -- render the
    historical f-string byte-identically (do-no-harm).
    """
    if not integrity_enabled or normalized.get("is_us"):
        return f"Price: ${current_price:.2f} | Market Cap: ${market_cap/1e9:.1f}B | P/E: {pe_ratio:.1f}"

    ccy = normalized.get("currency", "?")
    if normalized.get("fx_available") and normalized.get("price_usd") is not None:
        rate = normalized.get("fx_rate")
        lines = (
            f"Price: ${normalized['price_usd']:.2f} (converted from {ccy} {current_price:,.0f} "
            f"@ {rate:.6f} {ccy}/USD) | "
            f"Market Cap: ${(normalized.get('market_cap_usd') or 0)/1e9:.1f}B | P/E: {pe_ratio:.1f}"
        )
    else:
        # Label-native fallback (a blocking currency_unverified flag will
        # normally exclude the candidate before any prompt is built; this
        # branch is defense-in-depth, never a $-labeled native magnitude).
        lines = (
            f"Price: {ccy} {current_price:,.0f} (NOT USD) | "
            f"Market Cap: {ccy} {market_cap/1e9:,.1f}B (NOT USD) | P/E: {pe_ratio:.1f}"
        )
    if normalized.get("as_of"):
        age = normalized.get("as_of_age_hours")
        age_txt = f"; quote is {age:.1f}h old" if age is not None else ""
        lines += f"\nData as-of: {normalized['as_of']} (exchange-local close{age_txt} -- NOT live)"
    return lines
