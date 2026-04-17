"""phase-5.5 step 5.5.3: Gap analysis.

Compares the current provider scoring (scripts/audit/score_current_state.py
output) against the alt-data categories surveyed in the 5.5.2 lit review,
and emits a structured gap list.

Each gap carries:
- data_class      : one of the 10 taxonomy classes from phase-5.5-research.md
- current_provider_or_null : which provider covers this today (None if no coverage)
- severity        : "low" | "medium" | "high" | "critical"
- rationale       : short why-this-severity
- recommended_next_step : concrete add/keep/replace action

Usage:
    python3 scripts/audit/gap_analysis.py \\
        --current backend/data_audit/current_state.json \\
        --literature handoff/phase-5.5-research.md \\
        --output backend/data_audit/gaps.json

Exit 0 on success.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---- Taxonomy -> current-provider map --------------------------------
# Hard-coded per CLAUDE.md + 5.5.2 lit review; a new dev adding a class
# must update this table. Drift flagged as build failure by tests.

DATA_CLASSES: list[dict] = [
    {
        "data_class": "news_and_sentiment",
        "current_provider": "social_sentiment",  # AV social_sentiment only
        "severity": "high",
        "rationale": (
            "Only AV Social Sentiment feeds NLP today; no RavenPack / "
            "Bloomberg / FinGPT-style transformer pipeline. NLP "
            "sentiment via Vertex is embeddings-only, not a dedicated "
            "news sentiment model."
        ),
        "recommended_next_step": "add: build phase-6 News & Sentiment Cron (proposal already on masterplan)",
    },
    {
        "data_class": "institutional_filings_13F_13D_form4",
        "current_provider": "sec_edgar",
        "severity": "medium",
        "rationale": (
            "sec_edgar covers Form 4 insider trades only. No 13F/13D "
            "(hedge-fund holdings), no QuiverQuant congressional trades, "
            "no WhaleWisdom-style aggregation. Missing ~80% of the "
            "institutional-filing alpha surface."
        ),
        "recommended_next_step": "add: expand sec_edgar tool to pull 13F + 13D; evaluate QuiverQuant free tier",
    },
    {
        "data_class": "short_interest_and_options_flow",
        "current_provider": "yfinance_options",
        "severity": "medium",
        "rationale": (
            "options_flow gives open-interest + IV skew via yfinance; no "
            "FINRA short-interest, no unusual-options-activity detection, "
            "no put/call imbalance signal."
        ),
        "recommended_next_step": "add: FINRA short-interest bi-weekly ingest + unusual-options detector on existing options_flow data",
    },
    {
        "data_class": "retail_social_wsb_stocktwits_x",
        "current_provider": None,
        "severity": "medium",
        "rationale": (
            "Zero coverage. WSB + Stocktwits are published free signal "
            "sources; X firehose is paid (deprioritise). Missing the "
            "2021-style retail-squeeze signal entirely."
        ),
        "recommended_next_step": "add: phase-7 PRAW Reddit scraper + Stocktwits public RSS ingest",
    },
    {
        "data_class": "search_interest",
        "current_provider": "google_trends",
        "severity": "low",
        "rationale": (
            "google_trends via pytrends (archived) + 24h cache covers it; "
            "follow-up: add Wikipedia pageviews as second signal for "
            "ensemble robustness (Wikipedia has no 429 / ToS risk)."
        ),
        "recommended_next_step": "keep: google_trends; add Wikipedia pageviews as backup (free; zero auth)",
    },
    {
        "data_class": "economic_nowcasting",
        "current_provider": "fred",
        "severity": "low",
        "rationale": (
            "FRED covers macro time series well, but we do not ingest the "
            "NY Fed / Atlanta Fed GDPNow nowcasts (higher signal density "
            "during regime transitions)."
        ),
        "recommended_next_step": "add: NY Fed nowcast + Atlanta Fed GDPNow weekly ingest (public URLs, free)",
    },
    {
        "data_class": "satellite_and_geospatial",
        "current_provider": None,
        "severity": "low",
        "rationale": (
            "No coverage. Orbital Insight / SpaceKnow are paid + pricey; "
            "ROI questionable at our capital scale. Defer until "
            "multi-million AUM."
        ),
        "recommended_next_step": "skip: revisit when live AUM crosses the paid-tier cost-justification threshold",
    },
    {
        "data_class": "patent_activity",
        "current_provider": "google_patents_bq",
        "severity": "low",
        "rationale": (
            "Just replaced PatentsView with BQ public dataset; covers "
            "full US grants; free; low-latency enough for the signal."
        ),
        "recommended_next_step": "keep: current BQ public dataset implementation",
    },
    {
        "data_class": "alt_credit_and_spending",
        "current_provider": None,
        "severity": "medium",
        "rationale": (
            "No coverage. Second Measure / Earnest Analytics are "
            "expensive ($10k+/yr). Cheap proxy: company-filed 10-Q "
            "revenue deltas already in sec_edgar + yfinance -- "
            "not real-time but defensible."
        ),
        "recommended_next_step": "defer paid; add: build a 10-Q revenue-surprise signal from existing sec_edgar data",
    },
    {
        "data_class": "ai_frontier_timeseries_foundation_models",
        "current_provider": None,
        "severity": "high",
        "rationale": (
            "No time-series foundation models in use. FinGPT / TimesFM / "
            "Moirai / Chronos-2 all landed 2024-2026 and show zero-shot "
            "forecasting superior to classical ARIMA/GBM on many horizons. "
            "Phase-8 proposal on masterplan is the right home."
        ),
        "recommended_next_step": "add: phase-8 Transformer Signals -- pilot Moirai or Chronos-2 in backtest",
    },
]


def _parse_literature(md: str) -> dict:
    """Extract light signals from the lit review to cross-check severity.

    Today this is a presence-check: each data_class must be mentioned
    somewhere in the document so the gap analysis is tied to the same
    evidence base. Not a strict parser -- just a guardrail so future
    edits don't let the two artifacts drift apart.
    """
    mentioned = {}
    needles = {
        "news_and_sentiment": r"RavenPack|FinBERT|FinGPT|news sentiment",
        "institutional_filings_13F_13D_form4": r"13F|13D|Form 4|insider trad",
        "short_interest_and_options_flow": r"short interest|options flow|FINRA",
        "retail_social_wsb_stocktwits_x": r"WSB|wallstreet ?bets|Stocktwits|X firehose",
        "search_interest": r"google trends|wikipedia pageview|search interest",
        "economic_nowcasting": r"nowcast|GDPNow|Federal Reserve",
        "satellite_and_geospatial": r"satellite|Orbital Insight|SpaceKnow|geospatial",
        "patent_activity": r"patent|USPTO|Google Patents",
        "alt_credit_and_spending": r"Second Measure|Earnest Analytics|credit card",
        "ai_frontier_timeseries_foundation_models": r"TimesFM|Moirai|Chronos|foundation model|transformer",
    }
    lower = md.lower()
    for klass, needle in needles.items():
        mentioned[klass] = bool(re.search(needle, md, flags=re.IGNORECASE))
    return mentioned


def _validate_current(current: dict) -> None:
    if not current.get("providers"):
        raise RuntimeError("current_state.json has no providers block")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--literature", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cur_path = Path(args.current)
    lit_path = Path(args.literature)
    if not cur_path.exists():
        print(json.dumps({"error": f"current state not found: {args.current}"}), file=sys.stderr)
        return 2
    if not lit_path.exists():
        print(json.dumps({"error": f"literature not found: {args.literature}"}), file=sys.stderr)
        return 2

    current = json.loads(cur_path.read_text(encoding="utf-8"))
    _validate_current(current)
    lit_md = lit_path.read_text(encoding="utf-8")
    lit_mentions = _parse_literature(lit_md)

    providers = current.get("providers", {})
    gaps: list[dict] = []
    for entry in DATA_CLASSES:
        entry_out = dict(entry)
        # Attach the scored provider's aggregate_pct if we have coverage.
        prov = entry.get("current_provider")
        if prov and prov in providers:
            entry_out["current_provider_score_pct"] = providers[prov]["aggregate_pct"]
        else:
            entry_out["current_provider_score_pct"] = None
        # Flag if the literature doesn't mention this class (drift warning).
        entry_out["literature_mentioned"] = lit_mentions.get(entry["data_class"], False)
        gaps.append(entry_out)

    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    gaps.sort(key=lambda g: -severity_rank.get(g["severity"], 0))

    out = {
        "schema_version": 1,
        "total_gaps": len(gaps),
        "by_severity": {
            s: sum(1 for g in gaps if g["severity"] == s)
            for s in ("critical", "high", "medium", "low")
        },
        "literature_drift": [k for k, v in lit_mentions.items() if not v],
        "gaps": gaps,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out_path),
        "total_gaps": out["total_gaps"],
        "by_severity": out["by_severity"],
        "literature_drift": out["literature_drift"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
