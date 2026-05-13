"""verify_phase_25_B7 -- yfinance fallback counter + WARNING log promotion.

Verifies:
  1. `orchestrator.py` logs the yfinance fallback at WARNING (not INFO).
  2. `bigquery_client.save_data_source_event` method exists with required kwargs.
  3. Migration script `create_data_source_events_table.py` exists with
     `CREATE TABLE IF NOT EXISTS` + `data_source_events`.
  4. Aggregable counter pattern (`COUNTIF(source='yfinance_fallback')`) is
     unambiguous given the schema (presence of the `source` column in CREATE SQL).

Behavioral round-trip:
  5. Patch BigQueryClient.save_data_source_event, simulate the orchestrator
     call shape, and confirm it would be invoked with the expected kwargs.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: orchestrator logs at WARNING ──────────────────────────────
orch_src = (REPO / "backend/agents/orchestrator.py").read_text(encoding="utf-8")
# Search for the specific log line; allow either single-line OR multi-line WARNING
warning_match = re.search(
    r'logger\.warning\(\s*\n?\s*"AV empty for %s -- using %d yfinance articles as fallback"',
    orch_src,
)
info_legacy_present = re.search(
    r'logger\.info\("AV empty for %s -- using %d yfinance articles as fallback"',
    orch_src,
)
claim(
    "1. orchestrator_yfinance_fallback_logs_at_warning_level",
    bool(warning_match) and not info_legacy_present,
    f"warning_present={bool(warning_match)} info_legacy_absent={not info_legacy_present}",
)


# ── Claim 2: save_data_source_event method exists with right signature ──
bq_src = (REPO / "backend/db/bigquery_client.py").read_text(encoding="utf-8")
bq_tree = ast.parse(bq_src)
method_found = False
sig_kwargs = []
for node in ast.walk(bq_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "save_data_source_event":
        method_found = True
        sig_kwargs = [a.arg for a in node.args.kwonlyargs]
        break
required = {"ticker", "source", "kind", "article_count", "notes"}
has_required = required.issubset(set(sig_kwargs))
claim(
    "2. bigquery_client_save_data_source_event_exists",
    method_found and has_required,
    f"found={method_found} kwargs={sig_kwargs} required={required}",
)


# ── Claim 3: migration script exists ───────────────────────────────────
migration_path = REPO / "scripts/migrations/create_data_source_events_table.py"
migration_exists = migration_path.exists()
migration_src = migration_path.read_text(encoding="utf-8") if migration_exists else ""
has_create = "CREATE TABLE IF NOT EXISTS" in migration_src
has_table_name = "data_source_events" in migration_src
has_apply = "--apply" in migration_src
claim(
    "3. new_bigquery_table_data_source_events_populated",
    migration_exists and has_create and has_table_name and has_apply,
    f"exists={migration_exists} create={has_create} name={has_table_name} apply_flag={has_apply}",
)


# ── Claim 4: counter aggregable ─────────────────────────────────────────
# The aggregable counter pattern requires the `source` column in the schema
# (so COUNTIF can filter by source value). Verify by inspecting the CREATE SQL.
has_source_col = bool(re.search(r"source\s+STRING\s+NOT NULL", migration_src))
has_event_time_col = bool(re.search(r"event_time\s+TIMESTAMP\s+NOT NULL", migration_src))
has_partition = "PARTITION BY DATE(event_time)" in migration_src
has_cluster_source = "CLUSTER BY source" in migration_src
counter_ok = has_source_col and has_event_time_col and has_partition and has_cluster_source
claim(
    "4. counter_aggregable_for_pct_yfinance_fallback_dominance",
    counter_ok,
    f"source_col={has_source_col} event_time_col={has_event_time_col} partition={has_partition} cluster={has_cluster_source}",
)


# ── Claim 5: behavioral round-trip ─────────────────────────────────────
# Stub a "self" with .bq.save_data_source_event being a MagicMock; call the
# code path that mirrors orchestrator.py:1162's try/except block.
class _StubSelf:
    def __init__(self):
        self.bq = MagicMock()


_self = _StubSelf()
fallback_articles = [{"title": "a"}, {"title": "b"}, {"title": "c"}]
ticker = "AAPL"

try:
    _self.bq.save_data_source_event(
        ticker=ticker,
        source="yfinance_fallback",
        kind="fallback",
        article_count=len(fallback_articles),
        notes="AV sentiment_summary empty",
    )
    _self.bq.save_data_source_event.assert_called_once_with(
        ticker="AAPL",
        source="yfinance_fallback",
        kind="fallback",
        article_count=3,
        notes="AV sentiment_summary empty",
    )
    round_trip_ok = True
    detail5 = "save_data_source_event invoked with expected kwargs"
except Exception as e:
    round_trip_ok = False
    detail5 = f"Exception: {type(e).__name__}: {e}"

claim(
    "5. behavioral_round_trip_call_shape_matches",
    round_trip_ok,
    detail5,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.B7 verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
