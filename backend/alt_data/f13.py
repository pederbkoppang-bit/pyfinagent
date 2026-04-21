"""phase-7.2 13F institutional holdings ingestion.

Pulls the latest 13F-HR filing(s) for a given CIK via EDGAR, parses
`informationTable` XML, normalizes to a 19-column row shape, and persists
to `pyfinagent_data.alt_13f_holdings`.

Compliance (docs/compliance/alt-data.md row 7.2):
- User-Agent: `pyfinagent/1.0 peder.bkoppang@hotmail.no`
- Rate limit: 8 req/s (below EDGAR's 10 req/s ceiling).
- 60*2^attempt backoff on 403 (see research brief risk R1).
- No residential proxies, no CAPTCHA bypass.

Fail-open. ASCII-only logger messages.

CLI:
    python -m backend.alt_data.f13                    # default Berkshire
    python -m backend.alt_data.f13 --cik 0001067983 --last-n 1
    python -m backend.alt_data.f13 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
_TABLE = "alt_13f_holdings"
_RATE_INTERVAL_S = 1.0 / 8.0  # 8 req/s ceiling
_RAW_PAYLOAD_CAP_BYTES = 100_000
_EDGAR_NS = {
    "ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable",
}

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
_ARCHIVE_FILE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"

_DEFAULT_CIK = "0001067983"  # Berkshire Hathaway

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  holding_id STRING NOT NULL,
  as_of_date DATE NOT NULL,
  cik STRING NOT NULL,
  filer_name STRING,
  accession_number STRING NOT NULL,
  period_of_report DATE,
  filed_on DATE,
  ticker STRING,
  cusip STRING NOT NULL,
  nameOfIssuer STRING,
  titleOfClass STRING,
  value_usd_thousands INT64,
  sshPrnamt INT64,
  sshPrnamtType STRING,
  putCall STRING,
  investmentDiscretion STRING,
  votingAuthority_sole INT64,
  votingAuthority_shared INT64,
  votingAuthority_none INT64,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY cik, cusip
OPTIONS (
  description = "phase-7.2 13F-HR institutional holdings; value_usd_thousands is EDGAR <value> in thousands USD"
)
""".strip()


def _zero_pad_cik(cik: str | int) -> str:
    s = str(cik).strip()
    if not s.isdigit():
        # tolerate "CIK0001067983" prefix or stray chars
        s = re.sub(r"\D", "", s)
    return s.zfill(10)


def _rate_limit() -> None:
    time.sleep(_RATE_INTERVAL_S)


def _holding_id(accession_number: str, cusip: str, sshPrnamt: int | None) -> str:
    key = f"{accession_number}|{cusip}|{sshPrnamt if sshPrnamt is not None else ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _safe_payload(raw: Any) -> str:
    try:
        s = json.dumps(raw, default=str, ensure_ascii=True)
    except Exception:
        s = json.dumps({"_error": "serialize_failed"}, ensure_ascii=True)
    if len(s) > _RAW_PAYLOAD_CAP_BYTES:
        s = s[:_RAW_PAYLOAD_CAP_BYTES - 32] + '..."__truncated__"}'
    return s


def _normalize_date(val: Any) -> str | None:
    if not val or not isinstance(val, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(val[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _http_get(url: str, *, accept: str = "application/json", attempt: int = 0, max_attempts: int = 2) -> Any:
    """Generic GET with EDGAR-compliant User-Agent + 403 backoff. Returns bytes or None."""
    try:
        import requests  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("f13: requests missing: %r", exc)
        return None
    headers = {"User-Agent": _USER_AGENT, "Accept": accept}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except Exception as exc:
        logger.warning("f13: get fail-open url=%s err=%r", url, exc)
        return None
    if resp.status_code == 403 and attempt < max_attempts:
        time.sleep(60 * (2**attempt))
        return _http_get(url, accept=accept, attempt=attempt + 1, max_attempts=max_attempts)
    if 500 <= resp.status_code < 600 and attempt < max_attempts:
        time.sleep(5 * (2**attempt))
        return _http_get(url, accept=accept, attempt=attempt + 1, max_attempts=max_attempts)
    if resp.status_code != 200:
        logger.warning("f13: non-200 status=%s url=%s", resp.status_code, url)
        return None
    return resp.content


def fetch_13f_submissions(cik: str, *, last_n: int = 1) -> list[dict[str, Any]]:
    """Return the most recent last_n 13F-HR filings for a CIK (list of {accessionNumber, filingDate, reportDate, form})."""
    cik_padded = _zero_pad_cik(cik)
    url = _SUBMISSIONS_URL.format(cik=cik_padded)
    _rate_limit()
    payload = _http_get(url)
    if payload is None:
        return []
    try:
        data = json.loads(payload)
    except Exception as exc:
        logger.warning("f13: submissions parse fail-open: %r", exc)
        return []
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accession_numbers = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    out: list[dict[str, Any]] = []
    for i, f in enumerate(forms):
        if f != "13F-HR":
            continue
        out.append(
            {
                "accessionNumber": accession_numbers[i] if i < len(accession_numbers) else None,
                "filingDate": filing_dates[i] if i < len(filing_dates) else None,
                "reportDate": report_dates[i] if i < len(report_dates) else None,
                "form": f,
                "filer_name": data.get("name"),
            }
        )
        if len(out) >= last_n:
            break
    return out


def fetch_filing_index(cik: str, accession_number: str) -> dict[str, Any]:
    """Pull the index.json and identify the informationTable filename."""
    cik_int = str(int(_zero_pad_cik(cik)))
    accession_nodash = (accession_number or "").replace("-", "")
    url = _FILING_INDEX_URL.format(cik_int=cik_int, accession_nodash=accession_nodash)
    _rate_limit()
    payload = _http_get(url)
    if payload is None:
        return {}
    try:
        return json.loads(payload)
    except Exception as exc:
        logger.warning("f13: filing-index parse fail-open: %r", exc)
        return {}


def _find_information_table_filename(index_json: dict[str, Any]) -> str | None:
    """Scan index.directory.item list for type=="INFORMATION TABLE"."""
    items = (index_json.get("directory") or {}).get("item") or []
    for it in items:
        if not isinstance(it, dict):
            continue
        if (it.get("type") or "").upper() == "INFORMATION TABLE":
            return it.get("name")
    # Fallback heuristic: any xml file that is not primary_doc.xml
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or ""
        if name.endswith(".xml") and "primary_doc" not in name.lower():
            return name
    return None


def fetch_13f(cik: str, accession_number: str) -> bytes | None:
    """Discover + download the informationTable XML for a given filing."""
    idx = fetch_filing_index(cik, accession_number)
    if not idx:
        return None
    filename = _find_information_table_filename(idx)
    if not filename:
        logger.warning("f13: informationTable filename not found for %s/%s", cik, accession_number)
        return None
    cik_int = str(int(_zero_pad_cik(cik)))
    accession_nodash = (accession_number or "").replace("-", "")
    url = _ARCHIVE_FILE_URL.format(cik_int=cik_int, accession_nodash=accession_nodash, filename=filename)
    _rate_limit()
    return _http_get(url, accept="application/xml")


def _find_text(el: ET.Element, local_name: str) -> str | None:
    """Namespace-tolerant text extractor."""
    for ns_prefix in ("ns:",):
        found = el.find(f"{ns_prefix}{local_name}", _EDGAR_NS)
        if found is not None and found.text:
            return found.text.strip()
    # un-namespaced fallback
    found = el.find(local_name)
    if found is not None and found.text:
        return found.text.strip()
    return None


def _find_int(el: ET.Element, local_name: str) -> int | None:
    t = _find_text(el, local_name)
    if t is None:
        return None
    try:
        return int(t.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_information_table(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Parse informationTable XML to list of holding dicts (namespace-tolerant)."""
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("f13: xml parse fail-open: %r", exc)
        return []
    holdings: list[dict[str, Any]] = []
    # look for infoTable elements both with and without namespace
    it_elements = list(root.findall("ns:infoTable", _EDGAR_NS)) + list(root.findall("infoTable"))
    for it in it_elements:
        shrs = None
        shrs_type = None
        # shrsOrPrnAmt wrapper
        shrs_wrapper = it.find("ns:shrsOrPrnAmt", _EDGAR_NS)
        if shrs_wrapper is None:
            shrs_wrapper = it.find("shrsOrPrnAmt")
        if shrs_wrapper is not None:
            shrs = _find_int(shrs_wrapper, "sshPrnamt")
            shrs_type = _find_text(shrs_wrapper, "sshPrnamtType")
        va_wrapper = it.find("ns:votingAuthority", _EDGAR_NS)
        if va_wrapper is None:
            va_wrapper = it.find("votingAuthority")
        va_sole = va_shared = va_none = None
        if va_wrapper is not None:
            va_sole = _find_int(va_wrapper, "Sole")
            va_shared = _find_int(va_wrapper, "Shared")
            va_none = _find_int(va_wrapper, "None")
        holdings.append(
            {
                "nameOfIssuer": _find_text(it, "nameOfIssuer"),
                "titleOfClass": _find_text(it, "titleOfClass"),
                "cusip": _find_text(it, "cusip"),
                "value_usd_thousands": _find_int(it, "value"),
                "sshPrnamt": shrs,
                "sshPrnamtType": shrs_type,
                "putCall": _find_text(it, "putCall"),
                "investmentDiscretion": _find_text(it, "investmentDiscretion"),
                "votingAuthority_sole": va_sole,
                "votingAuthority_shared": va_shared,
                "votingAuthority_none": va_none,
            }
        )
    return holdings


def normalize(
    holdings: Iterable[dict[str, Any]],
    filer_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    today_iso = date.today().isoformat()
    acc = filer_meta.get("accessionNumber") or ""
    out: list[dict[str, Any]] = []
    for h in holdings or []:
        cusip = (h.get("cusip") or "").strip()
        if not cusip:
            continue
        row = {
            "as_of_date": today_iso,
            "cik": _zero_pad_cik(filer_meta.get("cik") or ""),
            "filer_name": filer_meta.get("filer_name"),
            "accession_number": acc,
            "period_of_report": _normalize_date(filer_meta.get("reportDate")),
            "filed_on": _normalize_date(filer_meta.get("filingDate")),
            "ticker": None,
            "cusip": cusip,
            "nameOfIssuer": h.get("nameOfIssuer"),
            "titleOfClass": h.get("titleOfClass"),
            "value_usd_thousands": h.get("value_usd_thousands"),
            "sshPrnamt": h.get("sshPrnamt"),
            "sshPrnamtType": h.get("sshPrnamtType"),
            "putCall": h.get("putCall"),
            "investmentDiscretion": h.get("investmentDiscretion"),
            "votingAuthority_sole": h.get("votingAuthority_sole"),
            "votingAuthority_shared": h.get("votingAuthority_shared"),
            "votingAuthority_none": h.get("votingAuthority_none"),
            "raw_payload": _safe_payload(h),
        }
        row["holding_id"] = _holding_id(acc, cusip, h.get("sshPrnamt"))
        out.append(row)
    return out


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    proj = project
    ds = dataset
    if proj is None or ds is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            if proj is None:
                proj = s.gcp_project_id
            if ds is None:
                ds = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:  # pragma: no cover
            logger.warning("f13: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("f13: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("f13: bigquery.Client() init failed (%r)", exc)
        return None


def ensure_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    """Idempotent CREATE TABLE IF NOT EXISTS. Returns True on success, False on fail-open."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    sql = _CREATE_TABLE_SQL.format(project=proj, dataset=ds, table=_TABLE)
    try:
        client.query(sql).result(timeout=60)
        return True
    except Exception as exc:
        logger.warning("f13: ensure_table fail-open: %r", exc)
        return False


def upsert_holdings(
    rows: list[dict[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return 0
    table_ref = f"{proj}.{ds}.{_TABLE}" if proj else f"{ds}.{_TABLE}"
    try:
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("f13: insert errors: %s", errors)
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("f13: upsert fail-open: %r", exc)
        return 0


def ingest_cik(
    cik: str,
    *,
    last_n: int = 1,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Full orchestrator for one CIK. Returns count upserted."""
    subs = fetch_13f_submissions(cik, last_n=last_n)
    if not subs:
        logger.info("f13: no 13F-HR filings for cik=%s", cik)
        return 0
    total = 0
    for sub in subs:
        acc = sub.get("accessionNumber") or ""
        xml = fetch_13f(cik, acc)
        holdings = parse_information_table(xml or b"")
        filer_meta = {
            "cik": cik,
            "accessionNumber": acc,
            "filingDate": sub.get("filingDate"),
            "reportDate": sub.get("reportDate"),
            "filer_name": sub.get("filer_name"),
        }
        rows = normalize(holdings, filer_meta)
        if not rows:
            continue
        if dry_run:
            total += len(rows)
            continue
        total += upsert_holdings(rows, project=project, dataset=dataset)
    return total


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.2 13F holdings ingester")
    ap.add_argument("--cik", default=_DEFAULT_CIK, help="CIK (10-digit zero-padded or numeric)")
    ap.add_argument("--last-n", type=int, default=1, help="number of recent 13F-HR filings to ingest")
    ap.add_argument("--dry-run", action="store_true", help="skip BQ writes (still creates the table)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Always ensure the table exists so `bq ls | grep -q alt_13f_holdings` passes.
    table_ready = ensure_table()
    logger.info("f13: ensure_table -> %s", table_ready)
    count = ingest_cik(args.cik, last_n=args.last_n, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "cik": args.cik,
                "last_n": args.last_n,
                "dry_run": args.dry_run,
                "table_ready": table_ready,
                "ingested": count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
