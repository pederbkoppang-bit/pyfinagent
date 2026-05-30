"""phase-50.3: curated international constituent lists (yfinance symbols).

STATIC + in-repo by design (research_brief 50.3): DAX-40 + KOSPI-200 are small
and rebalance ~2x/year, so a static list is more reliable than a runtime
Wikipedia/Yahoo scrape (which can silently collapse to [] on an HTML change).
Symbols are the EXACT yfinance form -- the suffix IS part of the ticker
(SAP.DE, 005930.KS), and `market` is derived from the suffix at use-time
(markets.market_for_symbol). KR numeric codes keep their leading zeros
(STRING -- never int()).

NOTE: AIR.PA (Airbus) is a DAX-40 member listed in PARIS, not Frankfurt -- it
is .PA, NOT .DE. This is exactly the trap the suffixed-symbol-as-ticker design
avoids (you cannot derive .DE from market='EU' alone).
"""
from __future__ import annotations

# DAX-40 (Deutscher Aktienindex) -- yfinance symbols. Mostly .DE (XETRA);
# AIR.PA is the documented Paris-listed exception.
DAX40: list[str] = [
    "ADS.DE", "AIR.PA", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE",
    "BNR.DE", "CBK.DE", "CON.DE", "1COV.DE", "DBK.DE", "DB1.DE", "DHL.DE",
    "DTE.DE", "DTG.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN3.DE",
    "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE",
    "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "SHL.DE",
    "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "ENR.DE",
]

# KOSPI large-cap seed (documented subset of KOSPI-200) -- .KS (KOSPI).
# 6-digit numeric codes; leading zeros are SIGNIFICANT (005930 = Samsung Elec).
KOSPI200: list[str] = [
    "005930.KS", "000660.KS", "207940.KS", "005380.KS", "005490.KS",
    "035420.KS", "051910.KS", "006400.KS", "035720.KS", "012330.KS",
    "000270.KS", "068270.KS", "105560.KS", "055550.KS", "028260.KS",
    "066570.KS", "003670.KS", "096770.KS", "017670.KS", "015760.KS",
    "034730.KS", "018260.KS", "032830.KS", "086790.KS", "009150.KS",
    "011200.KS", "010130.KS", "024110.KS", "316140.KS", "138040.KS",
    "030200.KS", "010950.KS", "009540.KS", "011170.KS", "036570.KS",
    "047050.KS", "021240.KS", "323410.KS", "302440.KS", "000810.KS",
]

# market code -> curated symbol list (US handled separately via the S&P scrape).
INTL_UNIVERSE: dict[str, list[str]] = {
    "EU": DAX40,
    "KR": KOSPI200,
}
