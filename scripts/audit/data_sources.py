"""phase-5.5 step 5.5.0: Automated provider inventory.

Static (ast + grep) walker that enumerates every external data provider
the backend pulls from. Emits JSON to stdout in --dry-run mode. Never
prints secret values -- only env-var KEY NAMES and file paths.

Usage:
    python3 scripts/audit/data_sources.py --dry-run

Exit 0 on success, non-zero on internal error.

Design:
- Each provider has a fixed record in PROVIDERS below with its source
  file, required env-var keys, and a short summary.
- At runtime we (a) confirm the source file exists, (b) count call-
  sites via a ripgrep shell-out (fast, no Python parsing), (c) sanity-
  check no literal secrets appear in our own output.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 15 providers declared statically; runtime layer confirms + counts.
PROVIDERS: list[dict] = [
    {"name": "yfinance",         "kind": "data", "source": "backend/tools/yfinance_tool.py",     "env_vars": [],                                                 "summary": "OHLCV + fundamentals + options chain (no auth)"},
    {"name": "alphavantage",     "kind": "data", "source": "backend/tools/alphavantage.py",      "env_vars": ["ALPHAVANTAGE_API_KEY"],                            "summary": "News + market intel (50 headlines; 5 req/min)"},
    {"name": "fred",             "kind": "data", "source": "backend/tools/fred_data.py",         "env_vars": ["FRED_API_KEY"],                                     "summary": "Federal Reserve macro series (7 series)"},
    {"name": "google_trends",    "kind": "data", "source": "backend/tools/alt_data.py",          "env_vars": [],                                                 "summary": "Search interest via pytrends (rate-limited; 24h cache)"},
    {"name": "api_ninjas",       "kind": "data", "source": "backend/tools/earnings_tone.py",     "env_vars": ["API_NINJAS_KEY"],                                   "summary": "Earnings call transcript tone"},
    {"name": "vertex_ai",        "kind": "data", "source": "backend/tools/nlp_sentiment.py",     "env_vars": ["GCP_PROJECT_ID"],                                   "summary": "text-embedding-005 for NLP sentiment"},
    {"name": "yfinance_options", "kind": "data", "source": "backend/tools/options_flow.py",      "env_vars": [],                                                 "summary": "Options open-interest + IV skew via yfinance"},
    {"name": "google_patents_bq","kind": "data", "source": "backend/tools/patent_tracker.py",    "env_vars": ["GCP_PROJECT_ID"],                                   "summary": "Google Patents public BQ dataset (replaced PatentsView)"},
    {"name": "quant_model",      "kind": "data", "source": "backend/tools/quant_model.py",       "env_vars": [],                                                 "summary": "MDA cache + live yfinance features"},
    {"name": "sec_edgar",        "kind": "data", "source": "backend/tools/sec_insider.py",       "env_vars": [],                                                 "summary": "Form 4 insider trades (custom User-Agent)"},
    {"name": "sector_analysis",  "kind": "data", "source": "backend/tools/sector_analysis.py",   "env_vars": [],                                                 "summary": "11 SPDR ETF sector mapping via yfinance"},
    {"name": "social_sentiment", "kind": "data", "source": "backend/tools/social_sentiment.py",  "env_vars": ["ALPHAVANTAGE_API_KEY"],                             "summary": "Alpha Vantage social sentiment aggregation"},
    {"name": "anomaly_detector", "kind": "data", "source": "backend/tools/anomaly_detector.py",  "env_vars": [],                                                 "summary": "Multi-dim z-score on price/volume (local compute)"},
    {"name": "monte_carlo",      "kind": "data", "source": "backend/tools/monte_carlo.py",       "env_vars": [],                                                 "summary": "1000-path GBM scenario simulation (local compute)"},
    {"name": "gemini",           "kind": "llm",  "source": "backend/agents/llm_client.py",       "env_vars": ["GCP_PROJECT_ID"],                                   "summary": "Vertex AI Gemini via llm_client.make_client"},
    {"name": "anthropic",        "kind": "llm",  "source": "backend/agents/llm_client.py",       "env_vars": ["ANTHROPIC_API_KEY"],                                "summary": "Claude via llm_client.make_client"},
    {"name": "openai",           "kind": "llm",  "source": "backend/agents/llm_client.py",       "env_vars": ["OPENAI_API_KEY", "GITHUB_TOKEN"],                   "summary": "OpenAI / GitHub Models via llm_client.make_client"},
]

# Any token that looks like a literal secret: reject output if present.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),     # OpenAI / Anthropic / Slack user
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack bot tokens
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),  # Google API key
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),    # GitHub personal token
    re.compile(r"-----BEGIN PRIVATE KEY-----"),
]


def _count_call_sites(name: str, source: str) -> int:
    """Best-effort: count imports / usages of this provider across backend/.

    Uses ripgrep if available, otherwise falls back to Python glob+regex.
    """
    stem = Path(source).stem
    rg = shutil.which("rg")
    if rg:
        try:
            out = subprocess.check_output(
                [rg, "-l", "-i", stem, str(REPO / "backend")],
                text=True, timeout=5,
            )
            return len([l for l in out.splitlines() if l.strip()])
        except subprocess.CalledProcessError:
            return 0
        except Exception:
            pass
    # stdlib fallback
    count = 0
    for py in (REPO / "backend").rglob("*.py"):
        try:
            if stem in py.read_text(encoding="utf-8", errors="replace"):
                count += 1
        except Exception:
            continue
    return count


def _build_inventory() -> dict:
    providers_out: dict[str, dict] = {}
    for p in PROVIDERS:
        full = REPO / p["source"]
        providers_out[p["name"]] = {
            "kind": p["kind"],
            "source_file": p["source"],
            "source_exists": full.exists(),
            "call_site_files": _count_call_sites(p["name"], p["source"]) if full.exists() else 0,
            "env_vars": list(p["env_vars"]),
            "summary": p["summary"],
        }
    return {
        "schema_version": 1,
        "total_providers": len(providers_out),
        "data_providers": sum(1 for p in PROVIDERS if p["kind"] == "data"),
        "llm_providers": sum(1 for p in PROVIDERS if p["kind"] == "llm"),
        "providers": providers_out,
    }


def _assert_no_secrets(blob: str) -> list[str]:
    hits: list[str] = []
    for pat in SECRET_PATTERNS:
        m = pat.search(blob)
        if m:
            hits.append(pat.pattern)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print JSON to stdout; do not write files")
    ap.add_argument("--output", default="handoff/data_sources_inventory.json")
    args = ap.parse_args()

    inventory = _build_inventory()
    blob = json.dumps(inventory, indent=2)

    hits = _assert_no_secrets(blob)
    if hits:
        print(json.dumps({"error": "secret_leak_detected", "patterns": hits}),
              file=sys.stderr)
        return 2

    if args.dry_run:
        print(blob)
        return 0

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(blob + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "total": inventory["total_providers"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
