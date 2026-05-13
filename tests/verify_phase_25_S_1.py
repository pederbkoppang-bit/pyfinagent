"""verify_phase_25_S_1 -- Per-call ticker tagging in llm_call_log + cost_tracker.

Verifies:
  1. Migration script exists with ALTER TABLE ADD COLUMN IF NOT EXISTS ticker.
  2. `log_llm_call` accepts `ticker` kwarg + persists it in the row dict.
  3. `AgentCostEntry` has `ticker` field; `CostTracker.record(...)` accepts ticker kwarg.
  4. `_generate_with_retry` plucks `_ticker` from generation_config + passes to ct.record.
  5. `ClaudeClient.generate_content` plucks `_ticker` from generation_config + passes to log_llm_call.
  6. Behavioral round-trip: build CostTracker, call record(ticker="AAPL"); entry carries the ticker.
  7. Behavioral round-trip: call log_llm_call(ticker="MSFT"); buffered row carries the ticker.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("RAG_DATA_STORE_ID", "test-store")

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: migration script ──────────────────────────────────────────
mig_path = REPO / "scripts/migrations/add_ticker_to_llm_call_log.py"
mig_exists = mig_path.exists()
mig_src = mig_path.read_text(encoding="utf-8") if mig_exists else ""
has_alter = "ALTER TABLE" in mig_src and "ADD COLUMN IF NOT EXISTS ticker STRING" in mig_src
has_apply = "--apply" in mig_src
claim(
    "1. migration_script_adds_ticker_column_to_llm_call_log",
    mig_exists and has_alter and has_apply,
    f"exists={mig_exists} alter={has_alter} apply_flag={has_apply}",
)


# ── Claim 2: log_llm_call signature + row dict ────────────────────────
api_src = (REPO / "backend/services/observability/api_call_log.py").read_text(encoding="utf-8")
api_tree = ast.parse(api_src)
log_fn = None
for node in ast.walk(api_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "log_llm_call":
        log_fn = node
        break

has_ticker_kwarg = False
if log_fn:
    arg_names = [a.arg for a in log_fn.args.args]
    has_ticker_kwarg = "ticker" in arg_names

has_row_field = '"ticker": ticker' in api_src

claim(
    "2. log_llm_call_persists_ticker_in_row_dict",
    has_ticker_kwarg and has_row_field,
    f"ticker_kwarg={has_ticker_kwarg} row_field={has_row_field}",
)


# ── Claim 3: AgentCostEntry + record ───────────────────────────────────
ct_src = (REPO / "backend/agents/cost_tracker.py").read_text(encoding="utf-8")
has_entry_field = bool(re.search(r"ticker:\s*Optional\[str\]\s*=\s*None", ct_src))
has_record_kwarg = bool(re.search(r"ticker:\s*Optional\[str\]\s*=\s*None\s*,?\s*\)", ct_src))
claim(
    "3. cost_tracker_record_accepts_ticker_kwarg",
    has_entry_field and has_record_kwarg,
    f"entry_field={has_entry_field} record_kwarg={has_record_kwarg}",
)


# ── Claim 4: _generate_with_retry plucks _ticker + passes to ct.record ─
orch_src = (REPO / "backend/agents/orchestrator.py").read_text(encoding="utf-8")
has_pluck = 'call_ticker = generation_config.get("_ticker")' in orch_src
has_pass = "ticker=call_ticker" in orch_src
claim(
    "4. generate_with_retry_propagates_ticker_from_generation_config",
    has_pluck and has_pass,
    f"pluck={has_pluck} pass_to_record={has_pass}",
)


# ── Claim 5: ClaudeClient.generate_content passes _ticker to log_llm_call ─
llm_src = (REPO / "backend/agents/llm_client.py").read_text(encoding="utf-8")
has_claude_ticker = bool(
    re.search(r'ticker=config\.get\("_ticker"\)', llm_src)
)
claim(
    "5. claude_client_generate_content_passes_ticker_to_log_llm_call",
    has_claude_ticker,
    "ClaudeClient passes ticker= to log_llm_call" if has_claude_ticker else "missing",
)


# ── Claim 6: behavioral CostTracker.record carries ticker ──────────────
try:
    from backend.agents.cost_tracker import CostTracker

    class _StubUsage:
        prompt_token_count = 1000
        candidates_token_count = 500
        total_token_count = 1500
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

    class _StubResponse:
        usage_metadata = _StubUsage()

    tracker = CostTracker()
    entry = tracker.record(
        agent_name="Insider",
        model="claude-sonnet-4-6",
        response=_StubResponse(),
        ticker="AAPL",
    )
    rt6_ok = entry is not None and entry.ticker == "AAPL"
    rt6_detail = f"entry.ticker={getattr(entry, 'ticker', '<missing>')}"
except Exception as e:
    rt6_ok = False
    rt6_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "6. behavioral_record_carries_ticker_to_entry",
    rt6_ok,
    rt6_detail,
)


# ── Claim 7: behavioral log_llm_call buffer row carries ticker ─────────
try:
    from backend.services.observability import api_call_log

    api_call_log._llm_buffer.clear()
    api_call_log.log_llm_call(
        provider="anthropic",
        model="claude-sonnet-4-6",
        agent="Insider",
        latency_ms=10.0,
        input_tok=100,
        output_tok=50,
        ticker="MSFT",
    )
    last = api_call_log._llm_buffer[-1] if api_call_log._llm_buffer else None
    rt7_ok = last is not None and last.get("ticker") == "MSFT"
    rt7_detail = f"buffered_row.ticker={last.get('ticker') if last else '<empty>'}"
    api_call_log._llm_buffer.clear()
except Exception as e:
    rt7_ok = False
    rt7_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "7. behavioral_log_llm_call_buffer_row_carries_ticker",
    rt7_ok,
    rt7_detail,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.S.1 verification ===\n")
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
