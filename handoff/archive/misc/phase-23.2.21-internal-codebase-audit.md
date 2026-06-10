# Phase-23.2.21 Internal Codebase Audit — BigQuery MCP Pinning

**Date:** 2026-05-05
**Auditor:** Researcher agent (merged Explore role)
**Scope:** Is there any half-finished BQ MCP config? How does the Python code auth? What does the project expect?

---

## Files Inspected

| File | Lines read | Role | Status |
|------|-----------|------|--------|
| `.mcp.json` | all (10 lines) | Registered MCP servers | Has `alpaca` only — no BigQuery entry |
| `.claude/settings.json` | head 80 lines | Claude Code hooks + permissions | Has `mcp__bigquery__execute_sql` in `deny` list |
| `docs/MCP_ARCHITECTURE.md` | all (~120 lines) | Single source of truth for MCP | Documents BQ as "harness-injected" |
| `docs/audits/GAP_REPORT.md` | grep match | Audit gap tracking | Confirms `mcp__bigquery__execute_sql` in deny |
| `backend/db/bigquery_client.py` | lines 1–36 | Python BQ wrapper | Uses ADC fallback; SA key optional |
| `.claude/masterplan.json` | grep matches | Step verification criteria | Step has `bigquery_mcp_either_pinned_or_documented_as_ephemeral` check |

---

## Finding 1: No BQ MCP pinned in `.mcp.json`

`.mcp.json` (file: `.mcp.json`, all lines) contains only one entry:

```json
{
  "mcpServers": {
    "alpaca": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "alpaca-mcp-server==2.0.1", "alpaca-mcp-server"],
      "env": {
        "ALPACA_API_KEY_ID": "${ALPACA_API_KEY_ID:-}",
        "ALPACA_API_SECRET_KEY": "${ALPACA_API_SECRET_KEY:-}",
        "ALPACA_PAPER_TRADE": "true"
      }
    }
  }
}
```

The `alpaca` entry establishes the canonical shape for all future entries in this project:
- `"type": "stdio"` (not HTTP/SSE)
- `"command": "uvx"` (the project's preferred runner for external MCP servers)
- `"args": ["--from", "<pkg>==<version>", "<entrypoint>"]` for pinned-version installs
- `"env": { "KEY": "${ENV_VAR:-default}" }` for env-var interpolation with fallback

**Any new BQ MCP entry must mirror this exact shape.**

---

## Finding 2: `docs/MCP_ARCHITECTURE.md` documents BQ as "harness-injected" — aspirational

`docs/MCP_ARCHITECTURE.md` (line from "Harness-injected" section):

> "The BigQuery MCP server is **harness-injected**, not pinned in `.mcp.json`. See `CLAUDE.md::BigQuery Access (MCP)` for its tool surface and rules (read AND write against `sunny-might-477607-p8`). The fallback when MCP tools aren't present is the `bq` CLI or `google.cloud.bigquery` Python client."

Cross-checking `scripts/harness/` — the directory contains `run_harness.py`, `run_autonomous_loop.py`, `run_experiment.py`, etc. **None of these files contains any logic that injects a BQ MCP server.** The "harness-injected" claim in the docs is aspirational/historical; there is no actual injection mechanism. CLAUDE.md repeats the same claim verbatim. The net effect: the BQ MCP tools (`mcp__bigquery__*`) are only available in sessions where the claude.ai-hosted proxy MCP happened to authenticate (per-session OAuth friction). They are never reliably available.

---

## Finding 3: Python code uses ADC (with SA key fallback)

`backend/db/bigquery_client.py` lines 22–35:

```python
def __init__(self, settings: Settings):
    self.settings = settings
    credentials = None
    if settings.gcp_credentials_json:
        creds_info = json.loads(settings.gcp_credentials_json)
        credentials = service_account.Credentials.from_service_account_info(...)
    else:
        logger.warning("GCP_CREDENTIALS_JSON not set, falling back to Application Default Credentials")

    self.client = bigquery.Client(project=settings.gcp_project_id, credentials=credentials)
```

Auth priority: SA key JSON env var → ADC. In local development on Peder's Mac (no SA key set), the code falls through to ADC. The MCP server chosen must use the same auth path: **ADC-first, no per-session browser OAuth**.

---

## Finding 4: `settings.json` deny list blocks `mcp__bigquery__execute_sql`

`.claude/settings.json` permissions deny list (line ~68):

```json
"mcp__bigquery__execute_sql"
```

This is a deliberate safety guard — blocking the unrestricted write tool. Any pinned BQ MCP must account for this: the `execute_sql` (write) tool will be blocked by the pre-existing deny rule. Only read-only tools (`execute_sql_readonly`, `list_dataset_ids`, `list_table_ids`, `get_dataset_info`, `get_table_info`) will be callable by default. This is consistent with the project's `execute_sql_readonly` default preference documented in CLAUDE.md.

**Important implication:** if the chosen MCP server exposes tools named differently from `mcp__bigquery__execute_sql`, the deny rule may not fire. The server should expose a tool whose name maps to `mcp__bigquery__execute_sql` (i.e., the server-id in `.mcp.json` must be `bigquery` and the tool must be named `execute_sql`) for the deny rule to remain effective.

---

## Finding 5: Masterplan verification criterion

`.claude/masterplan.json` includes this verification check for the relevant step:

```python
"command": "python -c \"import json; mcp=json.load(open('.mcp.json')); deny=json.load(open('.claude/settings.json')).get('permissions',{}).get('deny',[]); assert 'bigquery' in str(mcp) or 'harness-injected' in open('CLAUDE.md').read(); assert any('bigquery__execute_sql' in r for r in deny)\""
```

This check passes if EITHER `bigquery` appears in `.mcp.json` OR `CLAUDE.md` still says "harness-injected". Since CLAUDE.md does say "harness-injected", the check currently passes vacuously. Pinning the BQ MCP in `.mcp.json` will make it pass on the first condition (and remains robust even if CLAUDE.md is updated).

---

## Finding 6: No launchd plist or wrapper script for BQ MCP injection

Grepped `scripts/`, `docs/`, `.claude/` for: `harness-inject`, `mcp_inject`, `bq_mcp`, `bigquery.*mcp`, `mcp.*bigquery`. No launchd plist, no shell wrapper, no Python bootstrap that launches a BQ MCP subprocess was found. The injection is entirely fictional/aspirational in the current codebase.

---

## Summary Table

| Claim | Evidence | file:line |
|-------|----------|-----------|
| `.mcp.json` has no BQ entry | Direct read | `.mcp.json:1-10` |
| Alpaca establishes `uvx --from pkg==ver` pattern | Direct read | `.mcp.json:4-5` |
| Env-var pattern is `${VAR:-}` | Direct read | `.mcp.json:8-10` |
| "Harness-injected" is aspirational — no script does it | Grep of scripts/ | `scripts/harness/` (no BQ MCP spawn in any file) |
| Python BQ client uses ADC when no SA key | Direct read | `backend/db/bigquery_client.py:25-35` |
| `mcp__bigquery__execute_sql` is in deny list | Grep + read | `.claude/settings.json:~68` |
| Masterplan check validates `bigquery in mcp.json OR harness-injected in CLAUDE.md` | Grep | `.claude/masterplan.json` |
| No launchd/plist/wrapper for BQ MCP | Negative grep | All of `scripts/`, `docs/`, `.claude/` |
