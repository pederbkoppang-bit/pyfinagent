"""phase-66.3: token-derived metered LLM spend (dollars actually billed).

Why this module exists: `llm_call_log.session_cost_usd` is a CUMULATIVE
per-cycle nominal gauge (autonomous_loop._session_cost, stamped on every row
by api_call_log.log_llm_call's lazy-fill), not a per-call cost. Row-summing
it -- what the away sentinel did through 2026-07-07 -- over-counts roughly
quadratically and manufactured the 06-17/06-18 "metered breach" ($16.51/$42.00
reported vs ~$1 nominal). There is NO per-call billed-cost column, so the
truthful metered figure is DERIVED FROM TOKENS x a pinned price table over
metered providers only (FOCUS-spec billed-cost discipline: never report
estimated/gauge values as billed).

Billing classes (criterion-2 documented choice: provider/agent FILTERING, no
schema migration):
- METERED (counted): provider 'gemini' (Vertex/AI Studio per-token billing)
  and any 'anthropic'/'openai' row NOT belonging to the flat-fee rail
  (direct API calls, e.g. haiku ticket agents).
- FLAT-FEE / CREDIT (excluded): agent LIKE 'cc_rail%' (Claude Code CLI rail
  -- Max-subscription flat fee; since 2026-06-15 headless usage draws the
  Agent SDK monthly credit, still not per-token API billing) and provider
  'claude-code' (the CLI wrapper itself).
- Unpriced metered models are counted as $0 and surfaced FAIL-VISIBLE in
  warnings (never silently priced).

CLI: `python scripts/away_ops/metered_spend.py [--date YYYY-MM-DD]` -> JSON.
The sentinel imports compute_for_date(); tests import the pure helpers.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

# USD per 1M tokens (input, output). Pinned 2026-07-07; sources:
# - gemini-2.5-flash: ai.google.dev/pricing ($0.30 in / $2.50 out, text)
# - claude-sonnet-4-6: $3 / $15 (platform.claude.com pricing)
# - claude-opus-4-7 / 4-8: $5 / $25
# - claude-haiku-4-5: $1 / $5
# - claude-fable-5: $10 / $50
# - gemini-2.5-pro: $1.25 / $10
# Claude cache tokens: cache_read = 0.1x input price; cache_creation = 1.25x
# input price (5-min ephemeral) -- applied for anthropic-provider rows.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

FLAT_FEE_PROVIDERS = {"claude-code"}
RAIL_AGENT_PREFIX = "cc_rail"


def _price_for(model: str) -> tuple[float, float] | None:
    """Longest-prefix match so dated ids (claude-haiku-4-5-20251001) price."""
    best = None
    for prefix, p in PRICES_PER_MTOK.items():
        if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), p)
    return best[1] if best else None


def is_flat_fee(row: dict[str, Any]) -> bool:
    agent = row.get("agent") or ""
    provider = row.get("provider") or ""
    return provider in FLAT_FEE_PROVIDERS or agent.startswith(RAIL_AGENT_PREFIX)


def compute_metered(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pure: rows -> {metered_llm_usd, rail_failures, warnings, unpriced_models}."""
    usd = 0.0
    rail_failures = 0
    unpriced: dict[str, int] = {}
    for r in rows:
        agent = r.get("agent") or ""
        if agent.startswith(RAIL_AGENT_PREFIX) and not r.get("ok", True):
            rail_failures += 1
        if is_flat_fee(r):
            continue
        in_tok = int(r.get("input_tok") or 0)
        out_tok = int(r.get("output_tok") or 0)
        cache_create = int(r.get("cache_creation_tok") or 0)
        cache_read = int(r.get("cache_read_tok") or 0)
        if not (in_tok or out_tok or cache_create or cache_read):
            continue  # nothing moved, nothing billed
        model = r.get("model") or ""
        price = _price_for(model)
        if price is None:
            unpriced[model] = unpriced.get(model, 0) + 1
            continue
        in_p, out_p = price
        usd += (in_tok * in_p + out_tok * out_p) / 1_000_000.0
        if (r.get("provider") or "") == "anthropic":
            usd += (cache_read * in_p * 0.1 + cache_create * in_p * 1.25) / 1_000_000.0
    warnings = []
    if unpriced:
        warnings.append(
            "unpriced metered models counted as $0 (fix PRICES_PER_MTOK): "
            + ", ".join(f"{m} x{n}" for m, n in sorted(unpriced.items()))
        )
    return {
        "metered_llm_usd": round(usd, 4),
        "rail_failures": rail_failures,
        "warnings": warnings,
        "unpriced_models": sorted(unpriced),
    }


def fetch_rows(date: str | None = None) -> list[dict[str, Any]]:
    """BQ rows for one UTC date (default: today). Bounded, read-only."""
    from google.cloud import bigquery

    client = bigquery.Client(project="sunny-might-477607-p8")
    where = "DATE(ts) = CURRENT_DATE()" if date is None else "DATE(ts) = @d"
    cfg = bigquery.QueryJobConfig(use_query_cache=True)
    if date is not None:
        cfg.query_parameters = [bigquery.ScalarQueryParameter("d", "DATE", date)]
    sql = f"""
        SELECT provider, model, agent, ok, input_tok, output_tok,
               cache_creation_tok, cache_read_tok
        FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
        WHERE {where}
    """
    return [dict(r) for r in client.query(sql, job_config=cfg).result(timeout=25)]


def compute_for_date(date: str | None = None) -> dict[str, Any]:
    out = compute_metered(fetch_rows(date))
    out["date"] = date or "today"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=None, help="UTC date YYYY-MM-DD (default today)")
    print(json.dumps(compute_for_date(ap.parse_args().date), indent=2))
