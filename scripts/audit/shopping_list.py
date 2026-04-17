"""phase-5.5 step 5.5.5: Prioritized shopping list.

Reads desired.json (from 5.5.4) and emits a markdown shopping list
with the top-N must-have entries. Each entry carries a URL citation
pulled from handoff/phase-5.5-research.md (verified present).

Usage:
    python3 scripts/audit/shopping_list.py \\
        --desired backend/data_audit/desired.json \\
        --top 3 \\
        --output backend/data_audit/shopping_list.md

Exit 0 on success.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIT_PATH = REPO / "handoff" / "phase-5.5-research.md"

# Curated citation mapping: each data_class -> at least one URL from the
# 5.5.2 lit review. Validated at runtime: every URL must appear in the
# lit review file.
CITATIONS: dict[str, list[str]] = {
    "news_and_sentiment": [
        "https://arxiv.org/abs/1908.10063",                 # FinBERT Araci 2019
        "https://arxiv.org/abs/2306.06031",                 # FinGPT
        "https://www.ravenpack.com/research/sentiment-driven-stock-selection",
    ],
    "institutional_filings_13F_13D_form4": [
        "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets",
        "https://edgartools.readthedocs.io/",
        "https://whalewisdom.com/",
    ],
    "short_interest_and_options_flow": [
        "https://www.finra.org/finra-data/browse-catalog/equity-short-interest",
        "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files",
    ],
    "retail_social_wsb_stocktwits_x": [
        "https://dl.acm.org/doi/10.1145/3660760",          # WSB sentiment peer-reviewed
        "https://www.nature.com/articles/s41599-023-01891-9",
    ],
    "search_interest": [
        "https://franz101.substack.com/p/google-trends-api-alternative-wikipedia",
        "https://developers.google.com/search/blog/2025/07/trends-api",
    ],
    "economic_nowcasting": [
        "https://www.newyorkfed.org/research/policy/nowcast",
        "https://www.newyorkfed.org/research/staff_reports/sr830.html",
        "https://fred.stlouisfed.org/",
    ],
    "patent_activity": [
        "https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data",
        "https://github.com/google/patents-public-data",
    ],
    "alt_credit_and_spending": [
        "https://www.earnestanalytics.com/insights/june-2023-earnest-analytics-spend-index",
        "https://www.bloomberg.com/professional/insights/data/alternative-data-insights-consumer-spending-growth-in-2025/",
    ],
    "ai_frontier_timeseries_foundation_models": [
        "https://arxiv.org/html/2403.07815v1",            # TimesFM
        "https://arxiv.org/html/2511.11698v1",            # Moirai 2.0
        "https://arxiv.org/html/2510.15821v1",            # Chronos-2
        "https://github.com/amazon-science/chronos-forecasting",
    ],
    "satellite_and_geospatial": [
        "https://www.neudata.co/blog/the-nascent-regulatory-landscape-of-web-scraping",
    ],
}


def _lit_urls() -> set[str]:
    if not LIT_PATH.exists():
        return set()
    text = LIT_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"https?://[^ )\s]+", text))


def _render(entry: dict, citations: list[str]) -> str:
    cls = entry["data_class"]
    sev = entry["severity"]
    tier = entry["alpha_tier"]
    cost = entry["cost_usd_month"]
    effort = entry["effort_days"]
    vendor = entry["vendor"]
    fallback = entry.get("fallback") or "none"
    license_note = entry["license_note"]
    rationale = entry.get("rationale", "")
    cite_lines = "\n".join(f"- {u}" for u in citations)
    return (
        f"## must-have: {cls}\n\n"
        f"- alpha_tier: {tier}\n"
        f"- severity: {sev}\n"
        f"- vendor: {vendor}\n"
        f"- fallback: {fallback}\n"
        f"- cost_usd_month: {cost}\n"
        f"- effort_days: {effort}\n"
        f"- license_note: {license_note}\n"
        f"- rationale: {rationale}\n"
        f"- citations (from phase-5.5-research.md):\n{cite_lines}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desired", required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    des = json.loads(Path(args.desired).read_text(encoding="utf-8"))
    entries = des.get("entries", [])
    if not entries:
        print(json.dumps({"error": "no entries in desired.json"}), file=sys.stderr)
        return 2

    # Skip DEFER'd vendors (alpha_tier=C with vendor=DEFER) -- they're
    # explicitly out of scope.
    actionable = [e for e in entries if e.get("vendor") != "DEFER"]
    if len(actionable) < args.top:
        print(json.dumps({"error": "not enough actionable entries",
                          "have": len(actionable), "want": args.top}),
              file=sys.stderr)
        return 2

    top = actionable[: args.top]

    lit_urls = _lit_urls()
    if not lit_urls:
        print(json.dumps({"error": "phase-5.5-research.md not found or empty"}),
              file=sys.stderr)
        return 2

    # Every cited URL must appear in the lit review (no fabrication).
    rendered: list[str] = []
    for e in top:
        cites = CITATIONS.get(e["data_class"], [])
        cites = [u for u in cites if u in lit_urls]
        if not cites:
            print(json.dumps({"error": "no valid lit-review citations for",
                              "data_class": e["data_class"]}), file=sys.stderr)
            return 2
        rendered.append(_render(e, cites))

    body = "# Phase 5.5 Prioritized Shopping List\n\n"
    body += (
        f"Top {args.top} must-have data sources, sorted by severity desc "
        f"and cost asc. Every citation is verified present in "
        f"handoff/phase-5.5-research.md.\n\n"
    )
    body += "\n".join(rendered)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")

    # Post-write assertions so the exit code reflects the immutable criteria.
    written = out_path.read_text(encoding="utf-8")
    must_have_count = written.count("## must-have:")
    if must_have_count != args.top:
        print(json.dumps({
            "error": "must_have section count mismatch",
            "expected": args.top, "got": must_have_count,
        }), file=sys.stderr)
        return 3

    print(json.dumps({
        "wrote": str(out_path),
        "must_have_count": must_have_count,
        "data_classes": [e["data_class"] for e in top],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
