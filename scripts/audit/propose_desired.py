"""phase-5.5 step 5.5.4: Desired-state proposal.

Reads backend/data_audit/gaps.json (from 5.5.3) and emits a desired-
state proposal: for each data_class, a concrete primary vendor +
fallback with cost, effort, alpha-tier, and license note. Rows are
sorted by (severity desc, cost asc) so the cheapest-high-value adds
surface first.

Usage:
    python3 scripts/audit/propose_desired.py \\
        --gaps backend/data_audit/gaps.json \\
        --output backend/data_audit/desired.json

Exit 0 on success.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Per-class proposal table. Each row carries the REQUIRED fields
# (vendor, fallback, cost_usd_month, effort_days, alpha_tier,
# license_note) plus advisory metadata. Alpha tiers: "S" (high-conviction,
# literature-backed), "A", "B", "C" (speculative). Cost 0 = free.

PROPOSALS: dict[str, dict] = {
    "news_and_sentiment": {
        "vendor": "finbert_via_huggingface",
        "fallback": "claude_haiku_chat_summarise",
        "cost_usd_month": 0,
        "effort_days": 5,
        "alpha_tier": "S",
        "license_note": "FinBERT Apache-2.0; Haiku metered (Peder approval for scale)",
        "integration_target": "phase-6 News & Sentiment Cron",
    },
    "institutional_filings_13F_13D_form4": {
        "vendor": "sec_edgar_expanded",
        "fallback": "quiverquant_free_tier",
        "cost_usd_month": 0,
        "effort_days": 3,
        "alpha_tier": "A",
        "license_note": "SEC EDGAR public domain; QuiverQuant free tier for derived congressional trades",
        "integration_target": "phase-7 Alt-Data Scraping",
    },
    "short_interest_and_options_flow": {
        "vendor": "finra_short_interest_bulk",
        "fallback": "yfinance_options_unusual_detector",
        "cost_usd_month": 0,
        "effort_days": 4,
        "alpha_tier": "A",
        "license_note": "FINRA bulk download free; public domain",
        "integration_target": "phase-7 Alt-Data Scraping",
    },
    "retail_social_wsb_stocktwits_x": {
        "vendor": "praw_reddit_wsb",
        "fallback": "stocktwits_public_rss",
        "cost_usd_month": 0,
        "effort_days": 4,
        "alpha_tier": "B",
        "license_note": "PRAW free + Reddit API ToS-OK; Stocktwits RSS free",
        "integration_target": "phase-7 Alt-Data Scraping",
    },
    "search_interest": {
        "vendor": "google_trends_cached",
        "fallback": "wikimedia_pageviews",
        "cost_usd_month": 0,
        "effort_days": 1,
        "alpha_tier": "B",
        "license_note": "google_trends ToS non-commercial-grey; Wikimedia CC0 explicit",
        "integration_target": "keep; add wikimedia as second signal",
    },
    "economic_nowcasting": {
        "vendor": "fred_macro",
        "fallback": "ny_fed_nowcast + atlanta_fed_gdpnow",
        "cost_usd_month": 0,
        "effort_days": 2,
        "alpha_tier": "B",
        "license_note": "FRED + Federal Reserve nowcasts public domain",
        "integration_target": "phase-9 Data Refresh Cron",
    },
    "satellite_and_geospatial": {
        "vendor": "DEFER",
        "fallback": None,
        "cost_usd_month": 0,
        "effort_days": 0,
        "alpha_tier": "C",
        "license_note": "Orbital Insight / SpaceKnow paid; revisit at >$2M AUM",
        "integration_target": "out of scope until post-go-live",
    },
    "patent_activity": {
        "vendor": "google_patents_bq",
        "fallback": "uspto_patentsview_v2_when_keys_reopen",
        "cost_usd_month": 0,
        "effort_days": 0,
        "alpha_tier": "B",
        "license_note": "BQ public dataset; USPTO gov public domain",
        "integration_target": "already shipped (Cycle 38)",
    },
    "alt_credit_and_spending": {
        "vendor": "sec_10q_revenue_surprise",
        "fallback": "second_measure_paid (needs Peder approval)",
        "cost_usd_month": 0,
        "effort_days": 5,
        "alpha_tier": "B",
        "license_note": "SEC 10-Q public domain; Second Measure $10k+/yr requires approval",
        "integration_target": "phase-7 Alt-Data Scraping (cheap proxy first)",
    },
    "ai_frontier_timeseries_foundation_models": {
        "vendor": "chronos_2_via_amazon_sagemaker",
        "fallback": "moirai_moe_via_huggingface_transformers",
        "cost_usd_month": 50,  # SageMaker inference hours for pilot
        "effort_days": 10,
        "alpha_tier": "S",
        "license_note": "Chronos-2 Apache-2.0; Moirai-MoE Apache-2.0; SageMaker metered (Peder approval)",
        "integration_target": "phase-8 Transformer Signals",
    },
}

REQUIRED_FIELDS = (
    "vendor", "fallback", "cost_usd_month", "effort_days",
    "alpha_tier", "license_note",
)
SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gaps_path = Path(args.gaps)
    if not gaps_path.exists():
        print(json.dumps({"error": f"gaps file not found: {args.gaps}"}), file=sys.stderr)
        return 2
    gaps_doc = json.loads(gaps_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    missing_proposals: list[str] = []
    for gap in gaps_doc.get("gaps", []):
        klass = gap["data_class"]
        if klass not in PROPOSALS:
            missing_proposals.append(klass)
            continue
        prop = dict(PROPOSALS[klass])
        missing_fields = [f for f in REQUIRED_FIELDS if f not in prop]
        if missing_fields:
            raise RuntimeError(f"proposal for {klass!r} missing fields {missing_fields}")
        prop["data_class"] = klass
        prop["severity"] = gap["severity"]
        prop["current_provider"] = gap.get("current_provider")
        prop["rationale"] = gap.get("rationale")
        rows.append(prop)

    if missing_proposals:
        print(json.dumps({"error": "gaps lack proposals",
                          "missing": missing_proposals}),
              file=sys.stderr)
        return 2

    rows.sort(
        key=lambda r: (-SEVERITY_RANK.get(r["severity"], 0),
                       r["cost_usd_month"]),
    )

    total_cost = sum(r["cost_usd_month"] for r in rows)
    total_effort = sum(r["effort_days"] for r in rows)

    out = {
        "schema_version": 1,
        "total_entries": len(rows),
        "total_cost_usd_month": total_cost,
        "total_effort_days": total_effort,
        "by_alpha_tier": {
            t: sum(1 for r in rows if r["alpha_tier"] == t)
            for t in ("S", "A", "B", "C")
        },
        "entries": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out_path),
        "entries": out["total_entries"],
        "total_cost_usd_month": total_cost,
        "total_effort_days": total_effort,
        "by_alpha_tier": out["by_alpha_tier"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
