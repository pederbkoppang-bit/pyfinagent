"""phase-4.9 step 4.9.4 Gauntlet regime catalog.

Seven immutable historical black-swan windows every self-evolved
strategy must be backtested against before any promotion path.
Date ranges are sourced from primary authorities (NBER, SEC/CFTC,
SNB, Fed, BIS, Cboe) and are IMMUTABLE once shipped -- changing
them would invalidate historical gauntlet results. Corrections
require a new phase-4.9.x step with regression analysis.

Note on flash_crash_2010: it is a single-session intraday event.
Daily-bar backtests will show near-zero drawdown for this window
because open and close of 2010-05-06 were near-normal. The
`intraday_only=True` flag signals to the gauntlet runner (4.9.5)
that this regime must be skipped or specially handled unless
minute-bar data is available.

Sources (primary, cited per regime):
- gfc_2008:              NBER + Lehman bankruptcy records.
- flash_crash_2010:      SEC/CFTC Joint Final Report (2010-09-30).
- snb_chf_2015:          SNB Press Release 2015-01-15.
- covid_crash_2020:      S&P 500 peak/trough via St. Louis Fed.
- fed_hike_shock_2022:   Fed Implementation Note 2022-03-16.
- yen_carry_unwind_2024: BIS Bulletin 90.
- tariff_vol_2025:       Cboe Index Insights April 2025.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date


@dataclass(frozen=True)
class RegimeWindow:
    """One black-swan regime window for the gauntlet catalog.

    Frozen dataclass: attempting to set a field on an instance
    raises `FrozenInstanceError`. This is intentional -- the
    gauntlet catalog is part of the immutable-core contract
    (phase-4.9).

    Supports dict-style key access via `__contains__` /
    `__getitem__` / `keys()` so the masterplan verification
    command (`'start' in r`) works verbatim.
    """
    id: str
    name: str
    start: str
    end: str
    asset_classes: tuple[str, ...]
    region: str
    note: str
    primary_source_url: str
    intraday_only: bool = False

    def __contains__(self, key: str) -> bool:
        return key in {f.name for f in fields(self)}

    def __getitem__(self, key: str) -> object:
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    def start_date(self) -> date:
        return date.fromisoformat(self.start)

    def end_date(self) -> date:
        return date.fromisoformat(self.end)


REGIMES: tuple[RegimeWindow, ...] = (
    RegimeWindow(
        id="gfc_2008",
        name="2008 Global Financial Crisis",
        start="2008-09-15",
        end="2009-03-09",
        asset_classes=("equity", "rates", "credit", "FX"),
        region="Global",
        note=(
            "Lehman Brothers bankruptcy (2008-09-15) to S&P 500 "
            "closing low 676.53 on 2009-03-09 (-56.8% peak-to-trough "
            "from 2007-10-09). VIX intraday peak 89.53 on 2008-10-24; "
            "close high 80.86 on 2008-11-20. Key tickers: SPY, XLF, "
            "C, BAC, TLT, VIX."
        ),
        primary_source_url=(
            "https://www.nber.org/news/business-cycle-dating-"
            "committee-announcement-september-20-2010"
        ),
    ),
    RegimeWindow(
        id="flash_crash_2010",
        name="2010 Flash Crash",
        start="2010-05-06",
        end="2010-05-06",
        asset_classes=("equity",),
        region="US",
        note=(
            "Single-session intraday crash 14:32-15:07 EDT. DJIA "
            "fell ~1000 points (9%) then recovered by 15:07. VIX "
            "close 32.80 (+7.89 from 24.91). Stub quotes at $0.01 "
            "printed on ACN, PG. Key ticker: ES (E-mini S&P), SPY."
        ),
        primary_source_url=(
            "https://www.sec.gov/news/studies/2010/marketevents-"
            "report.pdf"
        ),
        intraday_only=True,
    ),
    RegimeWindow(
        id="snb_chf_2015",
        name="2015 SNB EUR/CHF Floor Removal",
        start="2015-01-15",
        end="2015-01-26",
        asset_classes=("FX", "equity"),
        region="Europe",
        note=(
            "SNB abandoned 1.20 EUR/CHF floor without warning at "
            "09:30 CET on 2015-01-15. EUR/CHF fell ~30% intraday "
            "(1.20 to 0.85 on some platforms). SMI -10% on day. "
            "FXCM, Alpari UK, Excel Markets suffered terminal or "
            "near-terminal losses. End is conservative 7-trading-day "
            "tail-risk window post-ECB QE 2015-01-22. Key pairs: "
            "EURCHF, USDCHF; index: SMI, EWL."
        ),
        primary_source_url=(
            "https://www.snb.ch/en/publications/communication/"
            "press-releases/2015/pre_20150115"
        ),
    ),
    RegimeWindow(
        id="covid_crash_2020",
        name="2020 COVID-19 Market Crash",
        start="2020-02-19",
        end="2020-03-23",
        asset_classes=("equity", "rates", "FX", "crypto", "oil"),
        region="Global",
        note=(
            "S&P 500 peak close 3386.15 on 2020-02-19 to trough "
            "close 2237.40 on 2020-03-23 (-33.9%). Fastest bear "
            "market in US history: peak to -20% in 16 calendar "
            "days. VIX all-time closing high 82.69 on 2020-03-16. "
            "Fed emergency cuts 2020-03-03 (-50bp) and 2020-03-15 "
            "(-100bp to 0%). Key tickers: SPY, QQQ, USO, VIX, BTC."
        ),
        primary_source_url=(
            "https://en.wikipedia.org/wiki/2020_stock_market_crash"
        ),
    ),
    RegimeWindow(
        id="fed_hike_shock_2022",
        name="2022 Fed Rate-Hike Shock",
        start="2022-01-03",
        end="2022-10-12",
        asset_classes=("equity", "rates", "crypto"),
        region="US",
        note=(
            "S&P 500 peak close 4796.56 on 2022-01-03 to bear-"
            "market trough close 3577.03 on 2022-10-12 (-25.4%). "
            "Fed liftoff 2022-03-16 (+25bp to 0.25-0.50%) then 10 "
            "more hikes. AGG -17% worst bond year since 1976; "
            "simultaneous equity+bond drawdown broke 60/40 "
            "correlation. USDJPY 115 -> 151. BTC -75% peak-to-"
            "trough. Key tickers: SPY, QQQ, AGG, TLT, BTC."
        ),
        primary_source_url=(
            "https://www.federalreserve.gov/newsevents/"
            "pressreleases/monetary20220316a1.htm"
        ),
    ),
    RegimeWindow(
        id="yen_carry_unwind_2024",
        name="2024 Yen Carry-Trade Unwind",
        start="2024-07-31",
        end="2024-08-09",
        asset_classes=("equity", "FX", "crypto"),
        region="Japan",
        note=(
            "BOJ surprise rate hike 2024-07-31 (0.10% -> 0.25%) + "
            "weak US jobs print 2024-08-02 forced carry unwind. "
            "Nikkei 225 -12.4% on 2024-08-05 (worst since Black "
            "Monday 1987). TOPIX -12%. VIX intraday above 65, "
            "close 38.57 (+15.2 from 23.39). USDJPY fell 158 -> "
            "141. JPMorgan estimated 65-75% of global carry "
            "positioning unwound by mid-August. BOJ capitulation "
            "signal 2024-08-07; S&P full recovery 2024-08-09. "
            "Key tickers: NKY, EWJ, USDJPY, VIX, BTC."
        ),
        primary_source_url="https://www.bis.org/publ/bisbull90.pdf",
    ),
    RegimeWindow(
        id="tariff_vol_2025",
        name="2025 Liberation Day Tariff Shock",
        start="2025-04-02",
        end="2025-04-09",
        asset_classes=("equity", "rates", "FX", "crypto"),
        region="Global",
        note=(
            "Trump reciprocal tariff announcement after close "
            "2025-04-02 (10-50%+ on most imports; 145% cumulative "
            "on China). Largest 2-day US loss in history: S&P -10% "
            "on 2025-04-03/04. VIX close 45.31 on 2025-04-04 "
            "(+23.8 from 21.5), intraday above 60 on 2025-04-07. "
            "S&P trough 2025-04-07. 90-day pause announced "
            "2025-04-09 -> S&P +9.52% (largest one-day gain since "
            "2020 recovery). Full pre-crash recovery 2025-05-13. "
            "Unusual: TLT sold off with stocks (dollar-confidence "
            "concerns). Key tickers: SPY, QQQ, FXI, DXY, TLT, BTC."
        ),
        primary_source_url=(
            "https://www.cboe.com/insights/posts/index-insights-april/"
        ),
    ),
)


__all__ = ["REGIMES", "RegimeWindow"]
