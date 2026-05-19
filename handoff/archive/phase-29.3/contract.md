# Contract — phase-29.3 (Register 4 in-app FastMCP servers in .mcp.json)

**Step ID:** phase-29.3
**Date:** 2026-05-19
**Author:** Main (overnight execution)
**Tier:** complex

---

## Research-gate summary

| Metric | Value |
|---|---|
| Sources read in full | 7 |
| Snippet-only | 10 |
| URLs collected | 17 |
| Recency scan + frontier-sync | DONE |
| `gate_passed` | true |
| Free-only verdict | **ADOPT all 4** |

**Brief:** `handoff/current/research_brief.md`.

**Headline findings:**
1. All 4 FastMCP servers (`backend/agents/mcp_servers/{backtest,data,risk,signals}_server.py`) have `if __name__ == "__main__": mcp.run()` blocks (file:line confirmed in brief §1 finding #5). Pattern A invocation: `command: <venv python> args: [<server.py>]` with `PYTHONPATH` env.
2. **FREE-only verified** — no paid external SaaS keys; BQ uses ADC (Application Default Credentials) and degrades gracefully (`_CACHE_AVAILABLE = False` on init failure).
3. **`alwaysLoad` decisions:** `risk` + `data` = TRUE (high-frequency, graceful-degradation); `backtest` + `signals` = FALSE (rare-use; signals is 1887 lines, startup cost concern).
4. **Risk-server gate chain confirmed at file:line:** kill_switch (`risk_server.py:179`) → pbo_check (`:186-198`) → projected_max_dd_pct (`:201-213`).
5. **PYTHONPATH pitfall:** servers import `from backend.backtest...`; PYTHONPATH MUST point at project root or imports fail silently to stub mode.

---

## Verbatim immutable success criteria

1. `mcp_json_has_pyfinagent_backtest_entry` — jq `.mcpServers."pyfinagent-backtest"` non-null
2. `mcp_json_has_pyfinagent_data_entry` — jq non-null
3. `mcp_json_has_pyfinagent_risk_entry` — jq non-null
4. `mcp_json_has_pyfinagent_signals_entry` — jq non-null
5. `mcp_json_valid_after_edit` — `python3 -c "import json; json.load(open('.mcp.json'))"` exit 0
6. `pythonpath_set_in_all_4_entries` — each entry's env has `PYTHONPATH` to project root
7. `alwaysLoad_true_on_risk_and_data_only` — risk=true, data=true, backtest=false, signals=false (matches researcher's recommendation; high-freq + graceful-degrade = true)
8. `smoke_test_recipe_documented_in_experiment_results` — recipe block present

**Verification command:**
```bash
python3 -c "import json; json.load(open('.mcp.json'))" && \
jq -e '.mcpServers | (."pyfinagent-backtest" and ."pyfinagent-data" and ."pyfinagent-risk" and ."pyfinagent-signals")' .mcp.json && \
jq -e '[.mcpServers["pyfinagent-backtest","pyfinagent-data","pyfinagent-risk","pyfinagent-signals"] | .env.PYTHONPATH] | all(. == "/Users/ford/.openclaw/workspace/pyfinagent")' .mcp.json && \
jq -e '.mcpServers."pyfinagent-risk".alwaysLoad == true' .mcp.json && \
jq -e '.mcpServers."pyfinagent-data".alwaysLoad == true' .mcp.json && \
jq -e '.mcpServers."pyfinagent-backtest".alwaysLoad == false' .mcp.json && \
jq -e '.mcpServers."pyfinagent-signals".alwaysLoad == false' .mcp.json && \
grep -q 'smoke test' handoff/current/experiment_results.md
```

**`verification.live_check`:** `"live_check_29.3.md captures (a) verbatim 4 new mcp.json entries, (b) Bash smoke-test for each server (start + immediate kill confirming the python process resolves the import and doesn't ImportError), (c) post-restart recipe for the operator to confirm /mcp panel shows all 4 attached + a sample evaluate_candidate() call via the risk server."`

---

## Plan

1. DONE — Researcher.
2. DONE — Contract.
3. NEXT — GENERATE:
   - EDIT 1: Add 4 entries to `.mcp.json` (per researcher's verbatim spec).
   - EDIT 2: Update masterplan 29.3 entry: name + audit_basis + verification fields.
   - EDIT 3: Run pre-flight smoke test (start each server in background, kill, confirm no ImportError on stderr).
   - EDIT 4: Write experiment_results.md (verbatim diff + smoke output + alwaysLoad rationale).
   - EDIT 5: Write live_check_29.3.md.
4. Spawn Q/A. Circuit breaker: 2 fresh-qa.
5. Log → flip → commit.

---

## Out of scope

- Adding any new tools to the 4 servers.
- Layer-2 agent code changes to USE the newly-attached MCPs (a separate phase).
- `signals_server` startup-time optimization (~1887 lines is large; live_check will note if it becomes a problem).
- `backend/.env.example` documentation of any new env vars (permission-blocked).
