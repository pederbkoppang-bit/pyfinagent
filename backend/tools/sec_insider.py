"""
SEC Insider Trading tool — fetches Form 4 filings from SEC EDGAR.
Uses the official data.sec.gov/submissions API (CIK-based) and parses
Form 4 XML documents for actual transaction details (buy/sell/exercise).
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{filer_cik}/{accession}/{doc}"
SEC_HEADERS = {"User-Agent": "PyFinAgent/2.0 peder.bkoppang@hotmail.no"}

# In-memory CIK cache (ticker -> zero-padded CIK string)
_cik_cache: dict[str, str] = {}


async def _resolve_cik(client: httpx.AsyncClient, ticker: str) -> str | None:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK string."""
    upper = ticker.upper()
    if upper in _cik_cache:
        return _cik_cache[upper]

    resp = await client.get(SEC_TICKERS_URL)
    resp.raise_for_status()

    for entry in resp.json().values():
        t = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        _cik_cache[t] = cik

    return _cik_cache.get(upper)


def _parse_form4_xml(xml_text: str) -> list[dict]:
    """Parse a Form 4 XML and extract transaction details."""
    trades: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return trades

    # Reporting owner name
    owner_el = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
    owner_name = (owner_el.text or "").strip() if owner_el is not None else "Unknown"

    def _text(el) -> str:
        return (el.text or "").strip() if el is not None else ""

    # --- non-derivative transactions ---
    for txn in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(txn.find(".//transactionCoding/transactionCode"))
        acq_disp = _text(txn.find(".//transactionAmounts/transactionAcquiredDisposedCode/value"))
        date_str = _text(txn.find(".//transactionDate/value"))

        shares = 0
        try:
            shares = int(float(_text(txn.find(".//transactionAmounts/transactionShares/value"))))
        except (ValueError, TypeError):
            pass

        price = 0.0
        try:
            price = float(_text(txn.find(".//transactionAmounts/transactionPricePerShare/value")))
        except (ValueError, TypeError):
            pass

        # P = open-market purchase, S = sale, M = exercise, A = award, G = gift
        if code == "P":
            txn_type = "BUY"
        elif code == "S":
            txn_type = "SELL"
        elif acq_disp == "D":
            txn_type = "SELL"
        elif acq_disp == "A" and code not in ("G", "A"):
            txn_type = "BUY"
        else:
            txn_type = "OTHER"

        trades.append({
            "filer": owner_name, "date": date_str, "type": txn_type,
            "code": code, "shares": shares, "price": price,
        })

    # --- derivative transactions (options exercises, etc.) ---
    for txn in root.findall(".//derivativeTable/derivativeTransaction"):
        code = _text(txn.find(".//transactionCoding/transactionCode"))
        acq_disp = _text(txn.find(".//transactionAmounts/transactionAcquiredDisposedCode/value"))
        date_str = _text(txn.find(".//transactionDate/value"))

        shares = 0
        try:
            shares = int(float(_text(txn.find(".//transactionAmounts/transactionShares/value"))))
        except (ValueError, TypeError):
            pass

        if code in ("M", "P"):
            txn_type = "BUY"
        elif code == "S" or acq_disp == "D":
            txn_type = "SELL"
        else:
            txn_type = "OTHER"

        trades.append({
            "filer": owner_name, "date": date_str, "type": txn_type,
            "code": code, "shares": shares, "price": 0.0,
        })

    return trades


async def _fetch_form4(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    accession: str, doc: str,
) -> list[dict]:
    """Fetch and parse a single Form 4 XML document."""
    # Filer CIK is the first segment of the accession number (no leading zeros)
    filer_cik = accession.split("-")[0].lstrip("0") or "0"
    acc_no_dashes = accession.replace("-", "")
    # Strip XSLT prefix (e.g. "xslF345X05/") to get the raw XML filename
    raw_doc = doc.split("/")[-1] if "/" in doc else doc
    url = SEC_ARCHIVES_URL.format(filer_cik=filer_cik, accession=acc_no_dashes, doc=raw_doc)

    async with sem:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return _parse_form4_xml(resp.text)
        except Exception as exc:
            logger.debug("Failed to fetch Form 4 %s: %s", accession, exc)
            return []


async def get_insider_trades(ticker: str, months: int = 6) -> dict:
    """
    Fetch and analyse insider trades from SEC EDGAR Form 4 filings.
    Two-phase: list filings via submissions API, then fetch XMLs concurrently.
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=months * 30)).strftime("%Y-%m-%d")

    try:
        async with httpx.AsyncClient(timeout=30, headers=SEC_HEADERS) as client:
            cik = await _resolve_cik(client, ticker)
            if not cik:
                return {
                    "ticker": ticker, "trades": [], "signal": "UNKNOWN",
                    "summary": f"Could not resolve CIK for {ticker}.",
                }

            resp = await client.get(SEC_SUBMISSIONS_URL.format(cik=cik))
            resp.raise_for_status()
            data = resp.json()

        # Gather Form 4 filing metadata
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        form4_filings: list[tuple[str, str, str]] = []
        for i, form in enumerate(forms):
            if form not in ("4", "4/A"):
                continue
            filed = dates[i] if i < len(dates) else ""
            if filed < cutoff_date:
                continue
            acc = accessions[i] if i < len(accessions) else ""
            doc = docs[i] if i < len(docs) else ""
            if acc and doc:
                form4_filings.append((acc, doc, filed))
            if len(form4_filings) >= 20:
                break

        if not form4_filings:
            return {
                "ticker": ticker, "trades": [], "signal": "NEUTRAL",
                "summary": f"No Form 4 filings for {ticker} in {months} months.",
            }

        # Fetch Form 4 XMLs concurrently (max 5 at a time for SEC rate-limit)
        sem = asyncio.Semaphore(5)
        async with httpx.AsyncClient(timeout=30, headers=SEC_HEADERS) as client:
            tasks = [
                _fetch_form4(client, sem, acc, doc)
                for acc, doc, _ in form4_filings
            ]
            results = await asyncio.gather(*tasks)

        # Flatten and filter
        all_trades: list[dict] = []
        for txn_list in results:
            for t in txn_list:
                if t["date"] >= cutoff_date:
                    all_trades.append(t)

        if not all_trades:
            return {
                "ticker": ticker, "trades": [], "signal": "NEUTRAL",
                "summary": f"No insider transactions parsed for {ticker}.",
            }

        buys = [t for t in all_trades if t["type"] == "BUY"]
        sells = [t for t in all_trades if t["type"] == "SELL"]

        # Cluster buy detection: 3+ unique buyers within 30 days
        recent_cutoff = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_buyers = {t["filer"] for t in buys if t["date"] >= recent_cutoff}

        cluster_buy = len(recent_buyers) >= 3
        buy_sell_ratio = len(buys) / max(len(sells), 1)

        signal = "NEUTRAL"
        if cluster_buy:
            signal = "STRONG_BULLISH"
        elif buy_sell_ratio > 2:
            signal = "BULLISH"
        elif buy_sell_ratio < 0.3 and len(sells) > 3:
            signal = "BEARISH"

        # Format top 15 trades for output
        formatted = []
        for t in sorted(all_trades, key=lambda x: x["date"], reverse=True)[:15]:
            desc = f"{t['type']} {t.get('shares', 0):,} shares"
            if t.get("price"):
                desc += f" @ ${t['price']:.2f}"
            formatted.append({
                "filer": t["filer"], "date": t["date"], "type": t["type"],
                "form": "4", "shares": t.get("shares", 0),
                "price": t.get("price", 0.0), "description": desc,
            })

        return {
            "ticker": ticker,
            "total_filings": len(form4_filings),
            "total_transactions": len(all_trades),
            "buys": len(buys),
            "sells": len(sells),
            "buy_sell_ratio": round(buy_sell_ratio, 2),
            "cluster_buy": cluster_buy,
            "cluster_buyers": list(recent_buyers),
            "signal": signal,
            "trades": formatted,
            "summary": (
                f"{len(buys)} insider buys, {len(sells)} sells in {months}mo "
                f"({len(form4_filings)} Form 4 filings parsed). "
                f"Buy/Sell ratio: {buy_sell_ratio:.1f}. "
                f"{'CLUSTER BUY DETECTED (' + str(len(recent_buyers)) + ' execs in 30d)' if cluster_buy else 'No cluster buy.'} "
                f"Signal: {signal}."
            ),
        }

    except Exception as e:
        logger.error("Failed to fetch insider trades for %s: %s", ticker, e)
        return {"ticker": ticker, "trades": [], "summary": f"Error: {e}", "signal": "UNKNOWN"}
