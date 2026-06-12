#!/usr/bin/env bash
# sentinel.sh -- goal-away-ops phase-62.4: guardrail/budget sentinel.
#
# Run by run_away_session.sh pre-flight (62.3) before EVERY away session.
# Exit 0 = healthy; exit 1 = GATE BREACH (metered_budget | flags_match_tokens);
# exit 2 = INFRA (metered_source_unavailable). Any non-zero downgrades the
# session to digest-only (fail-closed) -- the distinct codes let the digest
# distinguish tamper from outage.
#
# METERED SOURCE (pinned): sunny-might-477607-p8.pyfinagent_data.llm_call_log
# -- SUM(cost_usd) WHERE DATE(ts)=CURRENT_DATE() (UTC, matches partitioning;
# never shell date arithmetic) AND the rail/provider is METERED (everything
# except the flat-fee claude_code CLI rail). This figure is a LOWER BOUND on
# invoice truth: grounding/per-request charges and provider-internal retry
# tokens are not in the log (Braintrust 2026; 58.1 ledger corroboration).
#
# BASELINE (pinned constant, NOT a rolling mean): baseline_usd=8.00/day --
# the 58.1-ledger daily class ($5-8) for the existing Gemini pipeline + the
# operator-approved $25 58.1 window, both EXEMPT from the $0 away decision.
# A 14-day rolling mean (~$0.006) would false-trip the first legitimate full
# cycle. Re-pin only with an operator token.
#
# TEST OVERRIDES (research-mandated -- a synthetic prod BQ row would sit in
# the streaming buffer ~30 min, un-DML-able, inflating the real metered
# figure all day = self-DoS): SENTINEL_TEST_METERED_USD=<x> substitutes the
# metered figure (test paths exit non-zero; an override can only TRIP gates,
# never pass a real breach); SENTINEL_ENV_FILE=<path> substitutes the .env
# for the flag-reconciliation gate; SENTINEL_TEST_BQ_FAIL=1 forces the infra
# path. The sentinel is READ-ONLY everywhere (a script, not a Claude tool
# call -- the 62.0 hook gates agent writes; program reads are safe).

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
cd "$REPO" || exit 2

"$REPO/.venv/bin/python" - <<'PYEOF'
import json, os, re, sys
from pathlib import Path

REPO = Path("/Users/ford/.openclaw/workspace/pyfinagent")
BASELINE_USD = 8.00  # pinned; see header
report = {
    "metered_llm_usd_today": None,
    "baseline_usd": BASELINE_USD,
    "kill_switch_paused": "unknown",
    "flags_match_tokens": None,
    "ok": False,
    "gates_failed": [],
    "warnings": [],
}

# ── metered spend today (BQ; LOWER BOUND -- see header) ────────────────
test_metered = os.environ.get("SENTINEL_TEST_METERED_USD")
if os.environ.get("SENTINEL_TEST_BQ_FAIL") == "1":
    report["gates_failed"].append("metered_source_unavailable")
    report["warnings"].append("forced by SENTINEL_TEST_BQ_FAIL")
elif test_metered is not None:
    report["metered_llm_usd_today"] = float(test_metered)
    report["warnings"].append("metered figure from SENTINEL_TEST_METERED_USD (test override)")
else:
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project="sunny-might-477607-p8")
        # Schema (verified 2026-06-12): provider/model/session_cost_usd; no
        # rail column. Flat-fee claude_code-rail rows log session_cost_usd=0
        # by design (60.4 writer), so a plain SUM IS the metered figure --
        # any metered call (gemini, anthropic API) carries real cost.
        sql = """
            SELECT COALESCE(SUM(COALESCE(session_cost_usd, 0)), 0) AS usd
            FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
            WHERE DATE(ts) = CURRENT_DATE()
        """
        rows = list(client.query(sql, job_config=bigquery.QueryJobConfig(
            use_query_cache=True)).result(timeout=25))
        report["metered_llm_usd_today"] = round(float(rows[0].usd or 0.0), 4)
    except Exception as e:
        report["gates_failed"].append("metered_source_unavailable")
        report["warnings"].append(f"BQ error: {type(e).__name__}: {str(e)[:160]}")

if (report["metered_llm_usd_today"] is not None
        and report["metered_llm_usd_today"] > BASELINE_USD):
    report["gates_failed"].append("metered_budget")

# ── flag-vs-token reconciliation ───────────────────────────────────────
try:
    env_path = Path(os.environ.get("SENTINEL_ENV_FILE", REPO / "backend" / ".env"))
    baseline = json.loads((REPO / "scripts/away_ops/flag_baseline.json").read_text())
    grandfathered = baseline.get("grandfathered", {})
    exempt = set(baseline.get("ops_flags_exempt", []))

    tokens_text = ""
    tok_path = REPO / "handoff" / "operator_tokens.jsonl"
    if tok_path.exists():
        tokens_text = tok_path.read_text(encoding="utf-8")

    mismatches = []
    flag_re = re.compile(r"^(PAPER_[A-Z0-9_]+)\s*=\s*(true|True|1)\s*$", re.M)
    for m in flag_re.finditer(env_path.read_text(encoding="utf-8")):
        key = m.group(1)
        if key in exempt:
            continue
        if grandfathered.get(key, "").lower() in ("true", "1"):
            continue
        # token line must mention the key (KNOWN_TOKEN_ENV_MAP application
        # writes the env name into the live_check; the jsonl carries the
        # human key -- accept either the env name or its registered human key)
        human_keys = []
        try:
            sys.path.insert(0, str(REPO))
            from backend.slack_bot.operator_tokens import KNOWN_TOKEN_ENV_MAP
            human_keys = [k for k, v in KNOWN_TOKEN_ENV_MAP.items() if v == key]
        except Exception:
            pass
        hit = key in tokens_text or any(hk in tokens_text for hk in human_keys)
        if not hit:
            mismatches.append(key)
    report["flags_match_tokens"] = not mismatches
    if mismatches:
        report["gates_failed"].append("flags_match_tokens")
        report["warnings"].append("unauthorized true flags: " + ",".join(mismatches))
except Exception as e:
    report["flags_match_tokens"] = None
    report["gates_failed"].append("flags_reconciliation_unavailable")
    report["warnings"].append(f"flag check error: {type(e).__name__}: {str(e)[:160]}")

# ── kill-switch (REPORT-ONLY, never a gate) ────────────────────────────
try:
    import urllib.request
    with urllib.request.urlopen("http://localhost:8000/api/paper-trading/kill-switch",
                                timeout=8) as r:
        report["kill_switch_paused"] = bool(json.load(r).get("paused"))
except Exception:
    try:
        last = "unknown"
        for line in (REPO / "handoff" / "kill_switch_audit.jsonl").read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "pause":
                last = True
            elif e.get("event") == "resume":
                last = False
        report["kill_switch_paused"] = last
    except Exception:
        pass

report["ok"] = not report["gates_failed"]
print(json.dumps(report))

if not report["gates_failed"]:
    sys.exit(0)
infra = {"metered_source_unavailable", "flags_reconciliation_unavailable"}
sys.exit(2 if set(report["gates_failed"]) <= infra else 1)
PYEOF
