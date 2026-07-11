"""
Sector event calendars — FDA PDUFA dates + upcoming earnings.

Pure data-pull service ($0 LLM cost):
- FDA PDUFA scraped from rttnews.com (no API key, public HTML)
- Earnings from existing pyfinagent_data.calendar_events BQ table

Returns dict[ticker, SectorEvent]. Screener applies +20% boost for catalysts in
7 days, +10% for earnings in 1-3 days, FILTERS OUT tickers within ±1 day of a
binary FDA event (no new positions during the binary risk window).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime, timezone, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "sector_calendars"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_HOURS = 6

_RTTNEWS_URL = "https://www.rttnews.com/corpinfo/fdacalendar.aspx"
_USER_AGENT = "PyFinAgent/2.0 SectorCalendars (peder.bkoppang@hotmail.no)"
_TICKER_RE = re.compile(r"^[A-Z]{1,6}$")

# GICS sector inferred from event type for SectorEvent population
_EVENT_SECTOR = {
    "fda_pdufa": "Health Care",
    "earnings": "Multi",
}


class SectorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Equity ticker (uppercase). Sector ETF for macro events.")
    event_type: str = Field(description="fda_pdufa | earnings (extensible)")
    scheduled_date: date = Field(description="Event calendar date.")
    days_to_event: int = Field(description="Signed: positive=future, negative=past. Computed at fetch.")
    sector: str = Field(description="GICS sector name matching SECTOR_ETFS keys.")
    signal_direction: str = Field(
        description="positive_catalyst | binary_risk | macro_release | neutral",
    )
    drug_name: Optional[str] = Field(default=None)
    indication: Optional[str] = Field(default=None)
    source: str = Field(description="rttnews | bq_calendar")
    confidence: str = Field(default="confirmed", description="confirmed | estimated")
    metadata: dict = Field(default_factory=dict)


class _RTTNewsTableParser(HTMLParser):
    """Extract rows from the first <table> in the RTTNews FDA calendar page."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cells: list[str] = []
        self.current_cell: list[str] = []
        self.rows: list[list[str]] = []
        self._table_count = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_count += 1
            if self._table_count == 1:
                self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.cells = []
        elif self.in_row and tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self.in_table:
            self.in_table = False
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.cells:
                self.rows.append(self.cells)
        elif tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            text = " ".join("".join(self.current_cell).split())
            self.cells.append(text)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def _parse_rttnews_html(html_text: str, today: Optional[date] = None) -> list[SectorEvent]:
    """Parse the RTTNews FDA calendar HTML. Best-effort; returns empty on any failure."""
    if today is None:
        today = datetime.now(timezone.utc).date()
    parser = _RTTNewsTableParser()
    try:
        parser.feed(html_text)
    except Exception as e:
        logger.warning("RTTNews HTML parse failed: %s", e)
        return []

    events: list[SectorEvent] = []
    for row in parser.rows:
        # Heuristic columns: [date, company, ticker, drug, indication, ...] OR similar order.
        # Find the cell that looks like a ticker (1-6 uppercase letters).
        ticker = None
        for cell in row:
            cell_clean = cell.strip().upper()
            if _TICKER_RE.match(cell_clean):
                ticker = cell_clean
                break
        if not ticker:
            continue

        # Find a date cell (M/D/YYYY or M-D-YYYY)
        scheduled = None
        for cell in row:
            for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y"):
                try:
                    scheduled = datetime.strptime(cell.strip(), fmt).date()
                    break
                except ValueError:
                    continue
            if scheduled:
                break
        if not scheduled:
            continue

        days = (scheduled - today).days
        # Drug name + indication: take the longest text cells as best-effort
        text_cells = [c for c in row if c and not _TICKER_RE.match(c.strip().upper())]
        drug = text_cells[1] if len(text_cells) > 1 else None
        indication = text_cells[2] if len(text_cells) > 2 else None

        # Within ±1 day of an FDA event = binary risk; >1 day in future = positive catalyst
        if -1 <= days <= 1:
            direction = "binary_risk"
        elif 0 < days <= 30:
            direction = "positive_catalyst"
        else:
            direction = "neutral"

        events.append(SectorEvent(
            ticker=ticker,
            event_type="fda_pdufa",
            scheduled_date=scheduled,
            days_to_event=days,
            sector="Health Care",
            signal_direction=direction,
            drug_name=drug,
            indication=indication,
            source="rttnews",
            confidence="confirmed",
        ))
    return events


async def _fetch_fda_pdufa_events() -> list[SectorEvent]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html"}
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=20) as client:
            resp = await client.get(_RTTNEWS_URL)
        if resp.status_code != 200:
            logger.warning("RTTNews HTTP %s", resp.status_code)
            return []
    except Exception as e:
        logger.warning("RTTNews fetch failed: %s", e)
        return []
    return _parse_rttnews_html(resp.text)


def _fetch_earnings_events_sync(lookahead_days: int) -> list[SectorEvent]:
    """Sync because google.cloud.bigquery is sync. Wrapped via asyncio.to_thread."""
    today = datetime.now(timezone.utc).date()
    try:
        from backend.db.bigquery_client import BigQueryClient
        bq = BigQueryClient(get_settings())
        query = (
            "SELECT ticker, scheduled_at FROM `pyfinagent_data.calendar_events` "
            "WHERE event_type = 'earnings' "
            "AND scheduled_at >= CURRENT_TIMESTAMP() "
            f"AND scheduled_at <= TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {int(lookahead_days)} DAY) "
            "GROUP BY ticker, scheduled_at"
        )
        rows = list(bq.client.query(query).result())
    except Exception as e:
        logger.warning("Earnings calendar BQ query failed: %s", e)
        return []

    out: list[SectorEvent] = []
    for r in rows:
        t = (r.get("ticker") or "").upper().strip()
        if not _TICKER_RE.match(t):
            continue
        sched = r.get("scheduled_at")
        if hasattr(sched, "date"):
            sd = sched.date()
        elif isinstance(sched, str):
            try:
                sd = datetime.fromisoformat(sched.replace("Z", "+00:00")).date()
            except ValueError:
                continue
        else:
            continue
        days = (sd - today).days
        # Mild boost when earnings within 1-3 days; binary-risk on day-of (handled in apply)
        if 1 <= days <= 3:
            direction = "positive_catalyst"
        elif days == 0:
            direction = "binary_risk"
        else:
            direction = "neutral"
        out.append(SectorEvent(
            ticker=t,
            event_type="earnings",
            scheduled_date=sd,
            days_to_event=days,
            sector="Multi",
            signal_direction=direction,
            source="bq_calendar",
        ))
    return out


def _cache_path() -> Path:
    bucket = datetime.now(timezone.utc).strftime("%Y%m%d")
    return _CACHE_DIR / f"sector_calendars_{bucket}.json"


def _load_cache() -> Optional[dict[str, SectorEvent]]:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if age > timedelta(hours=_CACHE_TTL_HOURS):
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {t: SectorEvent.model_validate(v) for t, v in raw.items()}
    except Exception as e:
        logger.warning("Sector calendars cache unreadable: %s", e)
        return None


def _save_cache(events: dict[str, SectorEvent]) -> None:
    try:
        p = _cache_path()
        payload = {t: json.loads(e.model_dump_json()) for t, e in events.items()}
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("Sector calendars cache write failed: %s", e)


async def fetch_sector_events(use_cache: bool = True) -> dict[str, SectorEvent]:
    """Fetch FDA PDUFA + earnings events, return dict[ticker, SectorEvent]."""
    if use_cache:
        cached = _load_cache()
        if cached is not None:
            logger.info("Sector calendars cache hit: %d events", len(cached))
            return cached

    settings = get_settings()
    lookahead = int(getattr(settings, "sector_calendars_lookahead_days", 7))

    fda_task = _fetch_fda_pdufa_events()
    earnings_task = asyncio.to_thread(_fetch_earnings_events_sync, lookahead)
    fda_events, earnings_events = await asyncio.gather(fda_task, earnings_task, return_exceptions=False)

    out: dict[str, SectorEvent] = {}
    # FDA first (rarer + binary), earnings second (more frequent)
    for ev in fda_events + earnings_events:
        # First-seen wins; FDA binary_risk takes precedence over earnings catalyst on same ticker
        if ev.ticker in out and out[ev.ticker].event_type == "fda_pdufa":
            continue
        out[ev.ticker] = ev

    _save_cache(out)
    logger.info(
        "Sector calendars: fda=%d earnings=%d total=%d",
        len(fda_events), len(earnings_events), len(out),
    )
    return out


def apply_sector_events_to_score(
    base_score: float,
    ticker: str,
    sector: Optional[str],
    sector_events: Optional[dict[str, SectorEvent]],
) -> Optional[float]:
    """Apply sector event adjustment.

    Returns:
      - None if ticker has a binary-risk event within ±1 day (FILTER OUT)
      - score * 1.20 if positive catalyst within next 7 days (FDA)
      - score * 1.10 if earnings catalyst within 1-3 days
      - score unchanged otherwise
    """
    if not sector_events or not ticker or ticker not in sector_events:
        return base_score
    ev = sector_events[ticker]
    if ev.signal_direction == "binary_risk":
        return None
    if ev.signal_direction == "positive_catalyst":
        from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
        if ev.event_type == "fda_pdufa" and 0 < ev.days_to_event <= 7:
            return sign_safe_mult(base_score, 1.20)
        if ev.event_type == "earnings" and 1 <= ev.days_to_event <= 3:
            return sign_safe_mult(base_score, 1.10)
    return base_score
