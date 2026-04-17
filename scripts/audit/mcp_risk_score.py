"""phase-3.5 step 3.5.2: Security + risk scoring for MCP candidates.

Reads handoff/mcp_candidates.csv (from 3.5.1), applies a 4-axis
rubric (license, secret surface, rate-limit exposure, maintainer
health), and emits handoff/mcp_risk_scores.json with a risk_band
per candidate and a pending_peder_approval tag for anything paid.

Usage:
    python scripts/audit/mcp_risk_score.py

Exit 0 on success.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Per-server overrides where the CSV alone can't tell us the truth.
# - `paid`: requires Peder approval; tag is appended regardless of band.
# - `needs_secret`: server calls a paid upstream or holds a long-lived key.
# - `rate_limit_upstream`: server can burn an upstream budget if looped.
OVERRIDES: dict[str, dict] = {
    "alpaca-mcp":      {"paid": False, "needs_secret": True,  "rate_limit_upstream": True},
    "sec-edgar-mcp":   {"paid": False, "needs_secret": False, "rate_limit_upstream": False},
    "fmp-mcp":         {"paid": False, "needs_secret": True,  "rate_limit_upstream": True},
    "yfinance-mcp":    {"paid": False, "needs_secret": False, "rate_limit_upstream": True},
    "openbb-mcp":      {"paid": False, "needs_secret": False, "rate_limit_upstream": False},
    "cloudflare-mcp":  {"paid": True,  "needs_secret": True,  "rate_limit_upstream": True},
    "sentry-mcp":      {"paid": True,  "needs_secret": True,  "rate_limit_upstream": False},
    "linear-mcp":      {"paid": True,  "needs_secret": True,  "rate_limit_upstream": False},
    "exa-mcp":         {"paid": True,  "needs_secret": True,  "rate_limit_upstream": True},
    "polygon-mcp":     {"paid": True,  "needs_secret": True,  "rate_limit_upstream": True},
    "brave-search":    {"paid": False, "needs_secret": True,  "rate_limit_upstream": True},
    "github-mcp":      {"paid": False, "needs_secret": True,  "rate_limit_upstream": True},
    "slack-mcp":       {"paid": False, "needs_secret": True,  "rate_limit_upstream": False},
    "gdrive-mcp":      {"paid": False, "needs_secret": True,  "rate_limit_upstream": False},
    "genai-toolbox":   {"paid": False, "needs_secret": True,  "rate_limit_upstream": True},
}

# License risk: lower is safer for our commercial use.
LICENSE_RISK: dict[str, int] = {
    "MIT":        1,
    "Apache-2.0": 1,
    "BSD-3-Clause": 1,
    "CC0-1.0":    1,
    "AGPL-3.0":   3,  # viral; isolation required
    "GPL-3.0":    3,
    "Proprietary": 4,
    "UNKNOWN":    4,
}


def _score_one(row: dict) -> dict:
    name = row["name"]
    lic = row.get("license", "UNKNOWN") or "UNKNOWN"
    lic_risk = LICENSE_RISK.get(lic, LICENSE_RISK["UNKNOWN"])
    over = OVERRIDES.get(name, {})
    secret_risk = 3 if over.get("needs_secret") else 1
    rate_risk = 3 if over.get("rate_limit_upstream") else 1
    try:
        age = int(row.get("last_commit_age_days", "9999"))
    except ValueError:
        age = 9999
    maint_risk = 1 if age < 30 else 2 if age < 90 else 3
    total = lic_risk + secret_risk + rate_risk + maint_risk
    if total <= 4:
        band = "low"
    elif total <= 7:
        band = "medium"
    else:
        band = "high"
    return {
        "name": name,
        "repo": row.get("repo"),
        "category": row.get("category"),
        "license": lic,
        "last_commit_age_days": age,
        "axes": {
            "license_risk": lic_risk,
            "secret_surface": secret_risk,
            "rate_limit_exposure": rate_risk,
            "maintainer_health": maint_risk,
        },
        "total_risk": total,
        "risk_band": band,
        "paid": bool(over.get("paid")),
        "pending_peder_approval": bool(over.get("paid")),
        "notes": "AGPL isolation required" if lic == "AGPL-3.0" else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="handoff/mcp_candidates.csv")
    ap.add_argument("--output", default="handoff/mcp_risk_scores.json")
    args = ap.parse_args()

    in_path  = REPO / args.input
    out_path = REPO / args.output
    if not in_path.exists():
        print(json.dumps({"error": f"candidates csv not found: {args.input}"}),
              file=sys.stderr)
        return 2

    scores: dict[str, dict] = {}
    with in_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["name"]] = _score_one(row)

    # Immutable criteria assertions
    missing_band = [n for n, v in scores.items() if "risk_band" not in v]
    paid_missing_tag = [n for n, v in scores.items()
                         if v["paid"] and not v["pending_peder_approval"]]
    if missing_band or paid_missing_tag:
        print(json.dumps({
            "error": "criterion_failure",
            "missing_risk_band": missing_band,
            "paid_missing_approval_tag": paid_missing_tag,
        }), file=sys.stderr)
        return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    by_band = {b: sum(1 for v in scores.values() if v["risk_band"] == b)
               for b in ("low", "medium", "high")}
    paid_count = sum(1 for v in scores.values() if v["paid"])
    print(json.dumps({
        "wrote": str(out_path),
        "total": len(scores),
        "by_band": by_band,
        "paid_pending_approval": paid_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
