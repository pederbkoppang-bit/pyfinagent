"""
SEC Insider Trading tool — fetches Form 4 filings from SEC EDGAR.
Uses the official data.sec.gov/submissions API (CIK-based) and parses
Form 4 XML documents for actual transaction details (buy/sell/exercise).
"""

import asyncio
import base64
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

    for attempt in range(3):
        resp = await client.get(SEC_TICKERS_URL)
        if resp.status_code == 429:
            wait = 2 ** attempt + 1
            logger.warning("SEC 429 rate-limit on CIK fetch, retrying in %ds...", wait)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        for entry in resp.json().values():
            t = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            _cik_cache[t] = cik
        return _cik_cache.get(upper)

    logger.error("SEC CIK fetch failed after 3 retries (429)")
    return None


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
    company_cik: str, accession: str, doc: str,
) -> list[dict]:
    """Fetch and parse a single Form 4 XML document."""
    # Use the company CIK (issuer), not the filer/agent CIK from the accession
    cik_stripped = company_cik.lstrip("0") or "0"
    acc_no_dashes = accession.replace("-", "")
    # Strip XSLT prefix (e.g. "xslF345X05/") to get the raw XML filename
    raw_doc = doc.split("/")[-1] if "/" in doc else doc
    url = SEC_ARCHIVES_URL.format(filer_cik=cik_stripped, accession=acc_no_dashes, doc=raw_doc)

    async with sem:
        try:
            for attempt in range(3):
                resp = await client.get(url)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt + 1)
                    continue
                resp.raise_for_status()
                return _parse_form4_xml(resp.text)
            logger.debug("Form 4 %s: 429 after retries", accession)
            return []
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

            for attempt in range(3):
                resp = await client.get(SEC_SUBMISSIONS_URL.format(cik=cik))
                if resp.status_code == 429:
                    logger.warning("SEC 429 on submissions API, retrying in %ds...", 2 ** attempt + 1)
                    await asyncio.sleep(2 ** attempt + 1)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            else:
                return {
                    "ticker": ticker, "trades": [], "signal": "UNKNOWN",
                    "summary": f"SEC rate limit (429) for {ticker} after retries.",
                }

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

        # Fetch Form 4 XMLs concurrently (max 3 at a time for SEC rate-limit)
        sem = asyncio.Semaphore(3)
        async with httpx.AsyncClient(timeout=30, headers=SEC_HEADERS) as client:
            tasks = [
                _fetch_form4(client, sem, cik, acc, doc)
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


# phase-4.14.16 (MF-34): Files API helper for SEC filings >32 MB.
#
# Form 4 XMLs are small (5-50 KB) and always fit inline, but the SEC
# insider tool is also used as the staging point for larger companion
# filings (10-K / 10-Q exhibits attached via the issuer's Form 4). When
# the caller collects one of those large filings, they can route it
# through `upload_large_filing_to_files_api(...)` to get a reusable
# `file_id` via the Anthropic beta Files API surface, cutting the
# input-token spend on follow-on calls (the file bytes are not
# re-sent; only the file_id is referenced).
#
# Beta header: `anthropic-beta: files-api-2025-04-14`. The Python SDK
# injects it automatically for `client.beta.files.upload(...)` calls.
# For `client.beta.messages.create(...)` calls that REFERENCE an
# uploaded file_id, callers must still pass `betas=["files-api-2025-04-14"]`
# explicitly.
#
# ZDR eligibility: Files API does NOT qualify for zero-data-retention
# today. This is documented in ARCHITECTURE.md; do not upload
# customer-PII-bearing filings via this path until ZDR status changes.

def upload_large_filing_to_files_api(
    client,
    filename: str,
    filing_bytes: bytes,
    mime_type: str = "application/pdf",
    size_threshold_bytes: int = 32_000_000,
) -> str:
    """Upload a large SEC filing and return its `file_id` for reuse.

    Guards on `size_threshold_bytes` (default 32 MB) so callers do
    not pay the Files-API round-trip for small filings that fit
    inline. Uses `client.beta.files.upload(...)`; the response
    attribute is `.id` (NOT `.file_id`) per Anthropic docs.
    """
    if len(filing_bytes) <= size_threshold_bytes:
        raise ValueError(
            "filing fits inline; keep payload as a document block instead "
            f"(size={len(filing_bytes)} <= threshold={size_threshold_bytes})"
        )
    uploaded = client.beta.files.upload(
        file=(filename, filing_bytes, mime_type),
    )
    file_id = uploaded.id
    return file_id


# phase-4.14.14 (MF-31): wrap insider-trades summary as Claude
# document block with citations enabled. Pure data helper -- no API
# call. Must not be combined with response_schema / output_config
# (guarded in ClaudeClient per phase-4.14.9).


def build_sec_document_block(ticker: str, result: dict) -> dict:
    """Return a Claude document block wrapping the insider-trades summary."""
    lines: list[str] = ["SEC insider trades for " + ticker + ":", ""]
    lines.append(result.get("summary", ""))
    trades = result.get("trades") or []
    if trades:
        lines.append("")
        lines.append("Recent transactions:")
        for t in trades[:25]:
            lines.append(
                "  - {d} {i} {s} {sh} @ {p} ({r})".format(
                    d=t.get("date", ""),
                    i=t.get("insider", ""),
                    s=t.get("side", ""),
                    sh=t.get("shares", ""),
                    p=t.get("price", ""),
                    r=t.get("role", ""),
                )
            )
    body = "\n".join(lines)
    return {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": body,
        },
        "title": "SEC insider trades -- " + ticker,
        "citations": {"enabled": True},
    }


# phase-4.14.17 (MF-34b): PDF-native document block for 10-K / 10-Q
# filings <= 32 MB. Preserves charts + tables (no text extraction).
# cache_control:ephemeral with ttl:"1h" per project convention.
def build_filing_pdf_block(ticker: str, pdf_bytes: bytes) -> dict:
    """Return a Claude PDF-native document block for a 10-K/10-Q filing."""
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode("ascii"),
        },
        "title": "SEC filing PDF -- " + ticker,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "citations": {"enabled": True},
    }

