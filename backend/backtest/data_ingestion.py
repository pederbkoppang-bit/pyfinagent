"""
Data ingestion service — downloads historical data from yfinance/FRED
and stores it permanently in BigQuery. Run once, replay forever.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import httpx

from google.cloud import bigquery

from backend.config.settings import Settings
from backend.backtest import markets  # phase-50.1: market -> ISO currency map

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
        # phase-82.0: retain the settings object. `_resolve_macro_end_date`
        # needs `macro_ingest_end_date`; previously only project/dataset were
        # unpacked here, so there was no way to read a setting at call time.
        self.settings = settings

    def _table(self, name: str) -> str:
        return f"{self.project}.{self.dataset}.{name}"

    def _ensure_tables_exist(self):
        """Create historical data tables if they don't exist (idempotent)."""
        from migrate_backtest_data import ALL_TABLES

        for name, ref, schema in ALL_TABLES:
            try:
                self.client.get_table(ref)
            except Exception:
                table = bigquery.Table(ref, schema=schema)
                self.client.create_table(table)
                logger.info(f"Auto-created BQ table: {name}")

    @staticmethod
    def _compute_dividends_per_share(qcf, qbs, col_date) -> float | None:
        """Compute per-share dividends from quarterly cash flow / balance sheet."""
        if qcf is None or qcf.empty:
            return None
        # yfinance reports cash dividends paid as a negative number
        for field in ["Cash Dividends Paid", "Common Stock Dividend Paid"]:
            if field in qcf.index and col_date in qcf.columns:
                val = qcf.loc[field, col_date]
                if pd.notna(val) and val != 0:
                    dividends_paid = abs(float(val))
                    # Get shares outstanding from balance sheet
                    shares = None
                    if qbs is not None and not qbs.empty:
                        for sf in ["Share Issued", "Ordinary Shares Number"]:
                            if sf in qbs.index and col_date in qbs.columns:
                                s = qbs.loc[sf, col_date]
                                if pd.notna(s) and s > 0:
                                    shares = float(s)
                                    break
                    if shares and shares > 0:
                        return dividends_paid / shares
        return None

    # ── Prices ───────────────────────────────────────────────────

    def _get_existing_price_dates(self, tickers: list[str]) -> set[tuple[str, str]]:
        """Return set of (ticker, date) already in BQ.

        phase-75.9 (data-bq-01): re-raises on query failure instead of
        swallowing to an empty set. A silently-empty dedup set would make
        ingest_prices insert every row again, producing duplicate
        (ticker,date) bars that distort features/MTM/Sharpe downstream. A
        genuinely empty *result* (first-run / cold table) is unaffected --
        that still returns set() via the normal return path below, so
        run_full_ingestion's cold-start insert-all behavior is unchanged.
        """
        table = self._table("historical_prices")
        query = f"""
            SELECT DISTINCT ticker, date
            FROM `{table}`
            WHERE ticker IN UNNEST(@tickers)
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers[:100]),
        ])
        try:
            rows = self.client.query(query, job_config=job_config).result(timeout=30)
            return {(r["ticker"], r["date"]) for r in rows}
        except Exception as e:
            logger.error(f"Dedup check failed for historical_prices (fail-closed, aborting batch): {e}")
            raise

    def ingest_prices(self, tickers: list[str], start_date: str, end_date: str) -> int:
        """Download OHLCV from yfinance and store in BQ. Returns row count inserted."""
        table = self._table("historical_prices")
        now = datetime.now(timezone.utc).isoformat()
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
                    if data.columns.nlevels > 1:
                        # MultiIndex columns from group_by='ticker'
                        if ticker in data.columns.get_level_values(0):
                            ticker_df = data[ticker]
                        else:
                            continue
                    else:
                        # Flat columns (shouldn't happen with group_by='ticker' but handle it)
                        ticker_df = data

                    ticker_df = ticker_df.dropna(subset=["Close"])  # type: ignore[arg-type]

                    # phase-50.5 (B data-quality door): drop unambiguous bad intl
                    # bars before BQ ingestion. US -> no-op (byte-identical).
                    _mkt = ticker.split(":", 1)[0].upper() if ":" in ticker else "US"
                    from backend.tools.price_quality import validate_ohlcv
                    ticker_df, _dq = validate_ohlcv(ticker_df, market=_mkt, ticker=ticker)

                    for idx, row in ticker_df.iterrows():
                        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")  # type: ignore[arg-type]
                        if (ticker, date_str) in existing:
                            continue
                        # Extract market from ticker namespace (e.g., "US:AAPL" → "US", "AAPL" → "US")
                        market = "US"
                        clean_ticker = ticker
                        if ":" in ticker:
                            market, clean_ticker = ticker.split(":", 1)
                        rows.append({
                            "ticker": clean_ticker,
                            "date": date_str,
                            "market": market,
                            "currency": markets.get_market_config(market)["currency"],  # phase-50.1: US->USD, EU->EUR, KR->KRW (was a USD-only stub)
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
        """phase-75.9 (data-bq-01): same fail-closed re-raise as
        _get_existing_price_dates -- see that docstring for the
        empty-result-vs-exception distinction this preserves."""
        table = self._table("historical_fundamentals")
        query = f"""
            SELECT DISTINCT ticker, report_date
            FROM `{table}`
            WHERE ticker IN UNNEST(@tickers)
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers[:100]),
        ])
        try:
            rows = self.client.query(query, job_config=job_config).result(timeout=30)
            return {(r["ticker"], r["report_date"]) for r in rows}
        except Exception as e:
            logger.error(f"Dedup check failed for historical_fundamentals (fail-closed, aborting batch): {e}")
            raise

    def ingest_fundamentals(self, tickers: list[str]) -> int:
        """Download quarterly financials from yfinance and store in BQ."""
        table = self._table("historical_fundamentals")
        now = datetime.now(timezone.utc).isoformat()
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

                    # Extract market from ticker (same logic as prices)
                    market = "US"
                    clean_ticker = ticker
                    if ":" in ticker:
                        market, clean_ticker = ticker.split(":", 1)
                    
                    rows.append({
                        "ticker": clean_ticker,
                        "market": market,
                        "report_date": report_date,
                        "filing_date": report_date,  # Approximation; true filing date not available from yfinance
                        "total_revenue": _get(qf, "Total Revenue"),
                        "net_income": _get(qf, "Net Income"),
                        "total_debt": _get(qbs, "Total Debt") or _get(qbs, "Long Term Debt"),
                        "total_equity": _get(qbs, "Total Equity Gross Minority Interest") or _get(qbs, "Stockholders Equity"),
                        "total_assets": _get(qbs, "Total Assets"),
                        "operating_cash_flow": _get(qcf, "Operating Cash Flow") if qcf is not None else None,
                        "shares_outstanding": _get(qbs, "Share Issued") or _get(qbs, "Ordinary Shares Number"),
                        "dividends_per_share": self._compute_dividends_per_share(qcf, qbs, col_date),
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
        """Dedupe key set for historical_macro.

        phase-82.0: FAIL-CLOSED. This previously caught bare Exception and
        returned an empty set, which is the worst possible failure mode for a
        dedupe check: an empty set makes EVERY fetched observation look new, so
        a transient BQ error during a backfill silently duplicates the whole
        table. Mirrors the fail-closed contract already used by
        `_get_existing_price_dates` (see that method) -- log at ERROR and
        re-raise so the caller aborts the batch instead of double-writing.
        """
        table = self._table("historical_macro")
        query = f"SELECT DISTINCT series_id, date FROM `{table}`"
        try:
            rows = self.client.query(query).result(timeout=30)
            return {(r["series_id"], r["date"]) for r in rows}
        except Exception as e:
            logger.error(
                f"Dedup check failed for historical_macro "
                f"(fail-closed, aborting batch): {e}"
            )
            raise

    def _resolve_macro_end_date(self, today: Optional[str] = None) -> str:
        """FRED `observation_end` for macro ingestion.

        phase-82.0 ROOT CAUSE: this used to be `settings.backtest_end_date`
        ("2025-12-31"), threaded in from api/backtest.py. That constant IS the
        value that historical_macro froze at -- every ingest dutifully asked
        FRED for observations ending on the backtest cap, inserted zero rows,
        and returned success. A macro feed's end date must track wall-clock,
        never the backtest window. `macro_ingest_end_date` overrides only when
        an operator wants a pinned, reproducible backfill.
        """
        pinned = (getattr(self.settings, "macro_ingest_end_date", "") or "").strip()
        if pinned:
            return pinned
        return today or datetime.now(timezone.utc).date().isoformat()

    def ingest_macro(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        fred_api_key: str = "",
    ) -> int:
        """Download FRED macro series and store in BQ.

        phase-82.0: `end_date` is now OPTIONAL and defaults to today via
        `_resolve_macro_end_date`. Callers must NOT pass
        `settings.backtest_end_date` -- see `_resolve_macro_end_date` for why
        that coupling froze the table at 2025-12-31.

        Every row is stamped with `realtime_start` (the vintage: the date on
        which this observation became visible to us). Without it the table is
        an un-attributable mosaic and macro-conditioned backtests inherit a
        publication-lag look-ahead -- e.g. a GDP row dated 2026-04-01 that FRED
        did not publish until 2026-07-30 is otherwise indistinguishable from
        one known on its observation date.
        """
        if not fred_api_key:
            logger.warning("FRED API key not configured, skipping macro ingestion")
            self._write_macro_receipt(0, "skipped_no_api_key", end_date or "")
            return 0

        end_date = self._resolve_macro_end_date(end_date)
        table = self._table("historical_macro")
        now = datetime.now(timezone.utc).isoformat()
        # Vintage stamp: the date we first observed these values.
        vintage = datetime.now(timezone.utc).date().isoformat()
        existing = self._get_existing_macro()
        total_inserted = 0
        failed_series: list[str] = []

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
                        "market": "US",  # FRED is US macro data
                        "date": date_str,
                        "value": float(val),
                        "ingested_at": now,
                        # phase-82.0 vintage: when this value became visible.
                        "realtime_start": vintage,
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
                failed_series.append(series_id)

        # phase-82.0 run-receipt. `MAX(ingested_at)` only advances when rows are
        # actually inserted, so append+dedupe makes a HEALTHY no-op run (nothing
        # new published yet) byte-indistinguishable from a job that never ran at
        # all. That ambiguity is precisely why a never-scheduled feed sat
        # unnoticed for months. The receipt records the attempt itself.
        outcome = "ok" if not failed_series else f"partial_failed={','.join(failed_series)}"
        self._write_macro_receipt(total_inserted, outcome, end_date)

        logger.info(
            f"Ingested {total_inserted} macro rows "
            f"(observation_end={end_date}, outcome={outcome})"
        )
        return total_inserted

    # phase-82.0 cycle-3 (Q/A CONDITIONAL finding 2): the receipts path is a
    # module-level override point so tests can redirect it. Previously the
    # suite appended to the REAL operational ledger -- the Q/A measured it
    # growing 13 -> 37 lines during one evaluation, including forged
    # {"outcome":"ok","rows_inserted":1} records byte-shaped like a genuine
    # ingest. That erodes the exact distinguishability criterion 5 exists to
    # create: a ledger any pytest run can write "ok" into is not evidence.
    _receipts_dir_override: Optional[Path] = None

    def _receipts_dir(self) -> Path:
        if self._receipts_dir_override is not None:
            return Path(self._receipts_dir_override)
        return Path(__file__).resolve().parents[2] / "handoff" / "logs"

    def _write_macro_receipt(self, inserted: int, outcome: str, end_date: str) -> None:
        """Append a macro-ingestion run receipt. Fail-OPEN by design.

        Deliberately fail-open: a receipt is observability, and losing one must
        never abort a successful ingest. This is the opposite call from
        `_get_existing_macro` (fail-CLOSED) because the blast radii differ --
        a lost receipt costs visibility, a lost dedupe set corrupts the table.
        """
        try:
            path = self._receipts_dir()
            path.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "job": "macro_ingest",
                "rows_inserted": inserted,
                "outcome": outcome,
                "observation_end": end_date,
            }
            with open(path / "macro_ingest_receipts.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:  # pragma: no cover - observability only
            logger.warning("macro run-receipt write fail-open: %r", exc)

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
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info(f"Starting full ingestion: {len(tickers)} tickers, {start_date} -> {end_date}")

        # Auto-create BQ tables if they don't exist
        self._ensure_tables_exist()

        prices = self.ingest_prices(tickers, start_date, end_date)
        fundamentals = self.ingest_fundamentals(tickers)
        # phase-82.0: deliberately does NOT forward `end_date` here. That param
        # carries settings.backtest_end_date ("2025-12-31"), and passing it made
        # every macro ingest a silent zero-row no-op. Passing None lets
        # _resolve_macro_end_date track wall-clock. Prices/fundamentals keep the
        # backtest window; only the macro feed is severed from it.
        macro = self.ingest_macro(start_date, None, fred_api_key)

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
