"""phase-5.5 step 5.5.1: Current-state scoring.

Reads the inventory produced by scripts/audit/data_sources.py and
assigns each provider a 1-5 score on five axes (cost, freshness,
coverage, spof, license) per the rubric documented in
handoff/current/contract.md.

Scores are hard-coded per provider so a reviewer can audit them in
code review (drift from reality surfaces as a diff). When a new
provider is added, the script errors loudly if its rubric entry is
missing.

Usage:
    python3 scripts/audit/score_current_state.py \\
        --input  backend/data_audit/inventory.json \\
        --output backend/data_audit/current_state.json

Exit 0 on success, non-zero on internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---- Rubric (see contract.md for definitions) --------------------------

RUBRIC: dict[str, dict[str, int]] = {
    "yfinance":         {"cost": 5, "freshness": 3, "coverage": 5, "spof": 2, "license": 2},  # free but ToS non-commercial, single provider
    "alphavantage":     {"cost": 5, "freshness": 3, "coverage": 4, "spof": 2, "license": 4},  # free tier 25/day; paid SaaS
    "fred":             {"cost": 5, "freshness": 3, "coverage": 3, "spof": 3, "license": 5},  # US gov public domain
    "google_trends":    {"cost": 5, "freshness": 3, "coverage": 4, "spof": 1, "license": 2},  # pytrends archived + non-commercial ToS
    "api_ninjas":       {"cost": 4, "freshness": 2, "coverage": 3, "spof": 2, "license": 4},  # paid SaaS, limited earnings coverage
    "vertex_ai":        {"cost": 3, "freshness": 4, "coverage": 5, "spof": 3, "license": 4},  # Vertex AI; per-token paid; GCP regional
    "yfinance_options": {"cost": 5, "freshness": 3, "coverage": 4, "spof": 2, "license": 2},  # shares yfinance ToS issue
    "google_patents_bq":{"cost": 5, "freshness": 2, "coverage": 5, "spof": 3, "license": 5},  # BQ free tier; public domain patent data
    "quant_model":      {"cost": 5, "freshness": 4, "coverage": 5, "spof": 5, "license": 5},  # local compute on our MDA cache
    "sec_edgar":        {"cost": 5, "freshness": 2, "coverage": 5, "spof": 4, "license": 5},  # US gov; full US public companies
    "sector_analysis":  {"cost": 5, "freshness": 3, "coverage": 3, "spof": 2, "license": 2},  # yfinance-backed; 11 sector ETFs
    "social_sentiment": {"cost": 5, "freshness": 4, "coverage": 4, "spof": 2, "license": 4},  # alpha vantage paid SaaS
    "anomaly_detector": {"cost": 5, "freshness": 5, "coverage": 5, "spof": 5, "license": 5},  # local compute
    "monte_carlo":      {"cost": 5, "freshness": 5, "coverage": 5, "spof": 5, "license": 5},  # local compute
    "gemini":           {"cost": 3, "freshness": 5, "coverage": 5, "spof": 3, "license": 4},  # Vertex AI Gemini; metered
    "anthropic":        {"cost": 2, "freshness": 5, "coverage": 5, "spof": 3, "license": 4},  # Claude; metered; pricier
    "openai":           {"cost": 2, "freshness": 5, "coverage": 5, "spof": 3, "license": 4},  # OpenAI; metered
}

REQUIRED_FIELDS = ("cost", "freshness", "coverage", "spof", "license")


def _validate_rubric() -> None:
    for name, row in RUBRIC.items():
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise RuntimeError(f"rubric entry {name!r} missing fields: {missing}")
        for f in REQUIRED_FIELDS:
            v = row[f]
            if not (isinstance(v, int) and 1 <= v <= 5):
                raise RuntimeError(f"rubric entry {name!r}.{f}={v!r} out of 1-5 range")


def score(inventory: dict) -> dict:
    _validate_rubric()
    out: dict = {
        "schema_version": 1,
        "rubric_doc": "handoff/current/contract.md (Cycle 45)",
        "providers": {},
    }
    missing_in_rubric: list[str] = []
    for name, meta in inventory.get("providers", {}).items():
        if name not in RUBRIC:
            missing_in_rubric.append(name)
            continue
        row = dict(RUBRIC[name])
        row["aggregate"] = round(sum(row[f] for f in REQUIRED_FIELDS) / len(REQUIRED_FIELDS), 2)
        row["aggregate_pct"] = int(row["aggregate"] * 20)  # 0-100
        row["kind"] = meta.get("kind")
        row["summary"] = meta.get("summary", "")
        out["providers"][name] = row

    if missing_in_rubric:
        raise RuntimeError(
            f"providers present in inventory but absent from RUBRIC: "
            f"{missing_in_rubric}. Update scripts/audit/score_current_state.py."
        )

    total = len(out["providers"])
    out["total_providers"] = total
    out["avg_aggregate_pct"] = (
        round(sum(p["aggregate_pct"] for p in out["providers"].values()) / total, 1)
        if total else 0
    )
    # Flag providers that score < 60% -- candidates for replacement.
    out["at_risk_providers"] = sorted(
        [n for n, p in out["providers"].items() if p["aggregate_pct"] < 60]
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inv_path = Path(args.input)
    if not inv_path.exists():
        print(json.dumps({"error": f"inventory not found at {args.input}"}),
              file=sys.stderr)
        return 2
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))

    try:
        scored = score(inventory)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    # Brief stdout summary.
    print(json.dumps({
        "wrote": str(out_path),
        "providers_scored": scored["total_providers"],
        "avg_pct": scored["avg_aggregate_pct"],
        "at_risk": scored["at_risk_providers"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
