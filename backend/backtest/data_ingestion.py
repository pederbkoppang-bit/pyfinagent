"""
Data ingestion service — downloads historical data from yfinance/FRED
and stores it permanently in BigQuery. Run once, replay forever.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
import httpx

from google.cloud import bigquery

from backend.config.settings import Settings

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = ["FEDFUNDS", "CPIAUCSL", "UNRATE", "GDP", "T10Y2Y", "UMCSENT", "DGS10"]

# Batch size for yfinance downloads and BQ streaming inserts
_YF_BATCH = 50
_BQ_BATCH = 500


class DataIngestionService:
    """Downloads historical data once and stores in BigQuery for backtest replay."""

    def __init__(self, bq_client, settings: Settings):
        self.client = bq_client
        self.project = settings.gcp_project_id
        self.dataset = settings.bq_dataset_reports

    def _table(self, name: str) -> str:
        return f"{self.project}.{self.dataset}.{name}"

    # ── Prices ───────────────────────────────────────────────────

    def _get_existing_price_dates(self, tickers: list[str]) -> set[tuple[str, str]]:
        """Return set of (ticker, date) already in BQ."""
        table = self._table("historical_prices")
        # Use parameterized IN clause
        ticker_list = ", ".join(f"'{t}'" for t in tickers[:100])
        query = f"""
            SELECT DISTINCT ticker, date
            FROM `{table}`
            WHERE ticker IN ({ticker_list})
        """
        try:
            rows = self.client.query(query).result()
            return {(r["ticker"], r["date"]) for r in rows}
        except Exception:
            return set()

    def ingest_prices(self, tickers: list[str], start_date: str, end_date: str) -> int:
        """Download OHLCV from yfinance and store in BQ. Returns row count inserted."""
        table = self._table("historical_prices")
        now = datetime.utcnow().isoformat()
        total_inserted = 0

        for i in range(0, len(tickers), _YF_BATCH):
            batch = tickers[i:i + _YF_BATCH]
            logger.info(f"Downloading prices batch {i // _YF_BATCH + 1} ({len(batch)} tickers)")

            try:
                data = yf.download(
                    batch, start=start_date, end=end_date,
                    group_by="ticker", auto_adjust=True,
                    threads=True, progress=False,
                )
            except Exception as e:
                logger.error(f"yfinance download failed for batch {i}: {e}")
                continue

            if data is None or data.empty:
                continue

            existing = self._get_existing_price_dates(batch)
            rows = []

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_df = data
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            continue
                        ticker_df = data[ticker]

                    ticker_df = ticker_df.dropna(subset=["Close"])  # type: ignore[arg-type]

                    for idx, row in ticker_df.iterrows():
                        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")  # type: ignore[arg-type]
                        if (ticker, date_str) in existing:
                            continue
                        rows.append({
                            "ticker": ticker,
                            "date": date_str,
                            "open": float(row.get("Open", 0)) if pd.notna(row.get("Open")) else None,
                            "high": float(row.get("High", 0)) if pd.notna(row.get("High")) else None,
                            "low": float(row.get("Low", 0)) if pd.notna(row.get("Low")) else None,
                            "close": float(row["Close"]),
                            "volume": int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else None,
                            "ingested_at": now,
                        })
                except Exception as e:
                    logger.warning(f"Failed to process {ticker}: {e}")

            # Stream insert in sub-batches
            for j in range(0, len(rows), _BQ_BATCH):
                sub = rows[j:j + _BQ_BATCH]
                errors = self.client.insert_rows_json(table, sub)
                if errors:
                    logger.error(f"BQ insert errors (prices): {errors[:3]}")
                else:
                    total_inserted += len(sub)

        logger.info(f"Ingested {total_inserted} price rows")
        return total_inserted

    # ── Fundamentals ─────────────────────────────────────────────

    def _get_existing_fundamentals(self, tickers: list[str]) -> set[tuple[str, str]]:
        table = self._table("historical_fundamentals")
        ticker_list = ", ".join(f"'{t}'" for t in tickers[:100])
        query = f"""
            SELECT DISTINCT ticker, report_date
            FROM `{table}`
            WHERE ticker IN ({ticker_list})
        """
        try:
            rows = self.client.query(query).result()
            return {(r["ticker"], r["report_date"]) for r in rows}
        except Exception:
            return set()

    def ingest_fundamentals(self, tickers: list[str]) -> int:
        """Download quarterly financials from yfinance and store in BQ."""
        table = self._table("historical_fundamentals")
        now = datetime.utcnow().isoformat()
        total_inserted = 0

        existing = set()
        for i in range(0, len(tickers), 100):
            existing |= self._get_existing_fundamentals(tickers[i:i + 100])

        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                sector = info.get("sector", "")
                industry = info.get("industry", "")

                # Quarterly financials (income statement)
                qf = t.quarterly_financials
                # Quarterly balance sheet
                qbs = t.quarterly_balance_sheet
                # Quarterly cash flow
                qcf = getattr(t, "quarterly_cashflow", None)

                if qf is None or qf.empty:
                    continue

                rows = []
                for col_date in qf.columns:
                    report_date = pd.Timestamp(col_date).strftime("%Y-%m-%d")
                    if (ticker, report_date) in existing:
                        continue

                    # Extract fields safely
                    def _get(df, field):
                        if df is not None and not df.empty and field in df.index:
                            val = df.loc[field, col_date] if col_date in df.columns else None
                            return float(val) if pd.notna(val) else None
                        return None

                    rows.append({
                        "ticker": ticker,
                        "report_date": report_date,
                        "filing_date": report_date,  # Approximation; true filing date not available from yfinance
                        "total_revenue": _get(qf, "Total Revenue"),
                        "net_income": _get(qf, "Net Income"),
                        "total_debt": _get(qbs, "Total Debt") or _get(qbs, "Long Term Debt"),
                        "total_equity": _get(qbs, "Total Equity Gross Minority Interest") or _get(qbs, "Stockholders Equity"),
                        "total_assets": _get(qbs, "Total Assets"),
                        "operating_cash_flow": _get(qcf, "Operating Cash Flow") if qcf is not None else None,
                        "shares_outstanding": _get(qbs, "Share Issued") or _get(qbs, "Ordinary Shares Number"),
                        "sector": sector,
                        "industry": industry,
                        "ingested_at": now,
                    })

                if rows:
                    errors = self.client.insert_rows_json(table, rows)
                    if errors:
                        logger.error(f"BQ insert errors (fundamentals {ticker}): {errors[:3]}")
                    else:
                        total_inserted += len(rows)

            except Exception as e:
                logger.warning(f"Failed fundamentals for {ticker}: {e}")

        logger.info(f"Ingested {total_inserted} fundamentals rows")
        return total_inserted

    # ── Macro ────────────────────────────────────────────────────

    def _get_existing_macro(self) -> set[tuple[str, str]]:
        table = self._table("historical_macro")
        query = f"SELECT DISTINCT series_id, date FROM `{table}`"
        try:
            rows = self.client.query(query).result()
            return {(r["series_id"], r["date"]) for r in rows}
        except Exception:
            return set()

    def ingest_macro(self, start_date: str, end_date: str, fred_api_key: str) -> int:
        """Download FRED macro series and store in BQ."""
        if not fred_api_key:
            logger.warning("FRED API key not configured, skipping macro ingestion")
            return 0

        table = self._table("historical_macro")
        now = datetime.utcnow().isoformat()
        existing = self._get_existing_macro()
        total_inserted = 0

        for series_id in FRED_SERIES:
            try:
                url = (
                    f"{FRED_BASE}?series_id={series_id}"
                    f"&api_key={fred_api_key}&file_type=json"
                    f"&observation_start={start_date}&observation_end={end_date}"
                    f"&sort_order=asc"
                )
                with httpx.Client(timeout=20) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                observations = data.get("observations", [])
                rows = []
                for obs in observations:
                    val = obs.get("value", ".")
                    if val == ".":
                        continue
                    date_str = obs["date"]
                    if (series_id, date_str) in existing:
                        continue
                    rows.append({
                        "series_id": series_id,
                        "date": date_str,
                        "value": float(val),
                        "ingested_at": now,
                    })

                if rows:
                    for j in range(0, len(rows), _BQ_BATCH):
                        sub = rows[j:j + _BQ_BATCH]
                        errors = self.client.insert_rows_json(table, sub)
                        if errors:
                            logger.error(f"BQ insert errors (macro {series_id}): {errors[:3]}")
                        else:
                            total_inserted += len(sub)

            except Exception as e:
                logger.warning(f"Failed FRED series {series_id}: {e}")

        logger.info(f"Ingested {total_inserted} macro rows")
        return total_inserted

    # ── Orchestrator ─────────────────────────────────────────────

    def run_full_ingestion(
        self,
        tickers: list[str],
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None,
        fred_api_key: str = "",
    ) -> dict:
        """Run full ingestion pipeline. Returns row counts per table."""
        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        logger.info(f"Starting full ingestion: {len(tickers)} tickers, {start_date} → {end_date}")

        prices = self.ingest_prices(tickers, start_date, end_date)
        fundamentals = self.ingest_fundamentals(tickers)
        macro = self.ingest_macro(start_date, end_date, fred_api_key)

        result = {
            "prices_inserted": prices,
            "fundamentals_inserted": fundamentals,
            "macro_inserted": macro,
            "tickers_count": len(tickers),
            "start_date": start_date,
            "end_date": end_date,
        }
        logger.info(f"Ingestion complete: {result}")
        return result

    def get_ingestion_status(self) -> dict:
        """Check current row counts in historical tables."""
        counts = {}
        for name in ["historical_prices", "historical_fundamentals", "historical_macro"]:
            try:
                table_ref = self.client.get_table(self._table(name))
                counts[name] = table_ref.num_rows
            except Exception:
                counts[name] = 0
        return counts
