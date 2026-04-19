#!/usr/bin/env python
"""phase-6.8 end-to-end smoketest for the News & Sentiment Cron.

Serial pipeline: fetch news -> dedup (phase-6.4) -> BQ write news_articles
-> score sentiment (phase-6.5) -> BQ write news_sentiment -> calendar fetch
(phase-6.6) -> BQ write calendar_events -> flush api_call_log + llm_call_log
(phase-6.7) -> emit JSON summary + audit JSONL heartbeat.

Usage:
    # dry-run (default): use StubSource, do not hit live APIs
    python scripts/smoketest/phase6_e2e.py --dry-run

    # backfill: use live sources (finnhub, benzinga, alpaca)
    python scripts/smoketest/phase6_e2e.py --backfill --sources finnhub,benzinga,alpaca

    # explicit source selection
    python scripts/smoketest/phase6_e2e.py --sources stub --dry-run

Exit codes:
    0 = pipeline completed (even if BQ writes returned 0 due to auth/BQ absence;
        the smoketest validates code paths, not infra)
    1 = uncaught Python exception escaping the fail-open boundary

Outputs:
    - JSON pass/fail summary to stdout
    - Audit JSONL row appended to `handoff/audit/phase6_smoketest.jsonl`
    - Optional Slack heartbeat via `settings.slack_webhook_url` if configured

Per research brief (`handoff/archive/phase-6.8/research_brief.md`): at-least-
once semantics; repeat runs may duplicate rows (event_id/article_id); consumers
SELECT DISTINCT. No MERGE, no Storage Write API at <200 rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Make `backend` importable when invoked as `python scripts/smoketest/phase6_e2e.py`
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("phase6_e2e")


_AUDIT_JSONL = _ROOT / "handoff" / "audit" / "phase6_smoketest.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_audit(record: dict) -> None:
    try:
        _AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # pragma: no cover -- fail-open
        logger.debug("audit write fail-open err=%r", exc)


def _slack_heartbeat(summary: dict) -> bool:
    """POST the summary to slack_webhook_url if configured. Never raises."""
    try:
        from backend.config.settings import get_settings

        s = get_settings()
        url = getattr(s, "slack_webhook_url", "") or ""
        if not url:
            return False
        import requests

        payload = {"text": f"phase-6.8 smoketest: {json.dumps(summary)}"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as exc:
        logger.debug("slack heartbeat fail-open err=%r", exc)
        return False


def _run_pipeline(
    sources: list[str],
    dry_run: bool,
    backfill: bool,
    days_forward: int,
) -> dict:
    """Serial pipeline. Returns a summary dict regardless of per-stage failures."""
    summary: dict = {
        "ok": True,
        "started_at": _now_iso(),
        "dry_run": dry_run,
        "backfill": backfill,
        "sources": sources,
        "stages": {},
        "errors": [],
    }
    stages = summary["stages"]

    # --- Stage 1: fetch news ---
    try:
        from backend.news.fetcher import run_once as fetch_news

        # Live fetchers require dry_run=False to actually hit BQ after fetch;
        # in the smoketest's dry-run mode we still exercise the fetch path
        # with dry_run=True (no BQ write in run_once itself) and then rely
        # on bq_writer for an explicit write attempt.
        fetch_report = fetch_news(source_names=sources, dry_run=True)
        stages["news_fetch"] = {
            "ok": True,
            "n_articles": fetch_report.n_articles,
            "per_source": dict(fetch_report.per_source_counts),
            "errors": list(fetch_report.errors),
            "n_deduped": fetch_report.n_deduped,
        }
    except Exception as exc:
        stages["news_fetch"] = {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        summary["errors"].append(f"news_fetch: {exc!r}")
        fetch_report = None

    # --- Stage 2: write news_articles to BQ ---
    try:
        from backend.news.bq_writer import write_news_articles

        articles = getattr(fetch_report, "articles", []) or []
        if dry_run and not backfill:
            # In dry-run mode we still call the writer to exercise the code
            # path; it will fail-open and return 0 if BQ/auth is absent.
            inserted = write_news_articles(articles)
        else:
            inserted = write_news_articles(articles)
        stages["news_articles_insert"] = {"ok": True, "rows_inserted": inserted}
    except Exception as exc:
        stages["news_articles_insert"] = {"ok": False, "error": repr(exc)}
        summary["errors"].append(f"news_articles_insert: {exc!r}")

    # --- Stage 3: score sentiment ---
    sentiment_results: list = []
    try:
        from backend.news.sentiment import score_ladder

        articles = getattr(fetch_report, "articles", []) or []
        for a in articles:
            try:
                sentiment_results.append(score_ladder(a))
            except Exception as exc:
                stages.setdefault("sentiment_errors", []).append(repr(exc))
        stages["sentiment_score"] = {
            "ok": True,
            "n_scored": len(sentiment_results),
            "by_tier": _count_by_tier(sentiment_results),
        }
    except Exception as exc:
        stages["sentiment_score"] = {"ok": False, "error": repr(exc)}
        summary["errors"].append(f"sentiment_score: {exc!r}")

    # --- Stage 4: write news_sentiment to BQ ---
    try:
        from backend.news.bq_writer import write_news_sentiment

        inserted = write_news_sentiment(sentiment_results)
        stages["news_sentiment_insert"] = {"ok": True, "rows_inserted": inserted}
    except Exception as exc:
        stages["news_sentiment_insert"] = {"ok": False, "error": repr(exc)}
        summary["errors"].append(f"news_sentiment_insert: {exc!r}")

    # --- Stage 5: fetch calendar events ---
    cal_events: list = []
    try:
        from backend.calendar.watcher import run_once as fetch_calendar

        cal_report = fetch_calendar(days_forward=days_forward, days_backward=0)
        cal_events = list(cal_report.events or [])
        stages["calendar_fetch"] = {
            "ok": True,
            "n_events": cal_report.n_events,
            "by_type": dict(cal_report.by_type),
            "by_source": dict(cal_report.by_source),
            "errors": list(cal_report.errors),
        }
    except Exception as exc:
        stages["calendar_fetch"] = {"ok": False, "error": repr(exc)}
        summary["errors"].append(f"calendar_fetch: {exc!r}")

    # --- Stage 6: write calendar_events to BQ ---
    try:
        from backend.news.bq_writer import write_calendar_events

        inserted = write_calendar_events(cal_events)
        stages["calendar_events_insert"] = {"ok": True, "rows_inserted": inserted}
    except Exception as exc:
        stages["calendar_events_insert"] = {"ok": False, "error": repr(exc)}
        summary["errors"].append(f"calendar_events_insert: {exc!r}")

    # --- Stage 7: flush observability buffers ---
    try:
        from backend.services.observability import flush, flush_llm

        flushed_api = flush()
        flushed_llm = flush_llm()
        stages["observability_flush"] = {
            "ok": True,
            "api_call_log_rows": flushed_api,
            "llm_call_log_rows": flushed_llm,
        }
    except Exception as exc:
        stages["observability_flush"] = {"ok": False, "error": repr(exc)}
        summary["errors"].append(f"observability_flush: {exc!r}")

    # --- Stage 8: Slack heartbeat ---
    try:
        heartbeat_ok = _slack_heartbeat(
            {
                "dry_run": dry_run,
                "n_articles": stages.get("news_fetch", {}).get("n_articles", 0),
                "n_events": stages.get("calendar_fetch", {}).get("n_events", 0),
                "errors": len(summary["errors"]),
            }
        )
        stages["slack_heartbeat"] = {"ok": True, "sent": heartbeat_ok}
    except Exception as exc:
        stages["slack_heartbeat"] = {"ok": False, "error": repr(exc)}
        # non-fatal

    summary["finished_at"] = _now_iso()
    summary["ok"] = len(summary["errors"]) == 0
    return summary


def _count_by_tier(results: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        m = getattr(r, "scorer_model", "unknown")
        out[m] = out.get(m, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "phase-6.8 end-to-end smoketest. Default: dry-run using StubSource. "
            "Use --backfill with --sources finnhub,benzinga,alpaca for a 24h live run."
        )
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Fail-open everywhere; do not require BQ auth (default: True).",
    )
    ap.add_argument(
        "--backfill",
        action="store_true",
        default=False,
        help="Use live news sources (overrides --dry-run for the fetch stage).",
    )
    ap.add_argument(
        "--sources",
        default="stub",
        help="Comma-separated source names (default: stub; for backfill: finnhub,benzinga,alpaca).",
    )
    ap.add_argument(
        "--days-forward",
        type=int,
        default=7,
        help="Calendar window forward-days (default: 7; --backfill can pass 30).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.backfill and sources == ["stub"]:
        sources = ["finnhub", "benzinga", "alpaca"]

    try:
        summary = _run_pipeline(
            sources=sources,
            dry_run=args.dry_run and not args.backfill,
            backfill=args.backfill,
            days_forward=args.days_forward,
        )
    except Exception as exc:
        # Unrecoverable: escape the fail-open boundary. Exit 1.
        summary = {
            "ok": False,
            "fatal_exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(summary, default=str, indent=2))
        _write_audit(summary)
        return 1

    _write_audit(summary)
    print(json.dumps(summary, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
