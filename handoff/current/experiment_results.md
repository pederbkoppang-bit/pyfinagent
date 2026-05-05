---
step: phase-23.2.21
cycle_date: 2026-05-05
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_21.py'
---

# Experiment Results — phase-23.2.21

## Hypothesis recap

User: "BQ MCP needs OAuth — fix this." The discoverable BQ MCP today
is the Claude.ai-hosted proxy at `bigquery.googleapis.com/mcp` which
demands per-session OAuth and ignores the user's ADC. `.mcp.json` had
no BQ MCP pinned. CLAUDE.md's "harness-injected" claim was aspirational —
no harness file actually injects one. Researcher recommendation:
pin LucasHild's `mcp-server-bigquery==0.3.2` (Feb 2026) via uvx,
mirroring the existing alpaca MCP shape; ADC fires automatically when
no `--key-file` is passed.

## What was changed

### Fix A — Pinned BQ MCP server in `.mcp.json`
New `bigquery` entry beside the existing `alpaca` server:
- `command: uvx`
- `args: ["--from", "mcp-server-bigquery==0.3.2", "mcp-server-bigquery", "--project", "sunny-might-477607-p8", "--location", "US"]`
- `env: {}` (no env vars; ADC fires from `~/.config/gcloud/application_default_credentials.json`)

### Fix B — `.claude/settings.json` deny list updated
- Removed obsolete `mcp__bigquery__execute_sql` (underscored — matches
  no real tool on LucasHild's server, was a dead rule giving a false
  sense of guard).
- Added `mcp__bigquery__execute-query` (hyphenated — LucasHild's actual
  tool name). Read-class tools (`list-tables`, `describe-table`)
  remain default-ask, ergonomic for inspection.

### Fix C — CLAUDE.md "BigQuery Access (MCP)" section rewritten
- Dropped the "harness-injected" myth.
- Replaced the imaginary tool list (`execute_sql_readonly`,
  `list_dataset_ids`, `get_dataset_info`, `get_table_info`,
  `list_table_ids`, `execute_sql`) with the actual three tools
  exposed by LucasHild: `list-tables`, `describe-table`, `execute-query`.
- Updated Rule 1 — there is no readonly variant; default to
  `list-tables` / `describe-table` for inspection, escalate to
  `execute-query` only when SQL is needed (deny rule prompts for
  approval per call).
- Added Rule 7 pointing to the smoke test script.

### Fix D — Smoke test script
`scripts/mcp_servers/smoke_test_bigquery_mcp.py` spawns the MCP server
via stdio, performs the MCP handshake (`initialize` ->
`notifications/initialized`), then `tools/list` and `tools/call
list-tables`. Asserts the response references the project AND
`pyfinagent_*` datasets. 30-second timeout. Exit 0 on success.

### Verifier
`tests/verify_phase_23_2_21.py` (5 checks, all atomic):
- `.mcp.json` has the pinned bigquery entry with correct args
- `.claude/settings.json` deny list includes the hyphenated rule and
  excludes the obsolete underscored one
- CLAUDE.md retired the harness-injected wording and lists actual tools
- Smoke test script exists with the pinned version
- **Smoke test runs end-to-end against live ADC and exits 0**

## Files modified / added

```
.mcp.json                                            -- pinned bigquery server
.claude/settings.json                                -- deny rule renamed (underscore -> hyphen)
CLAUDE.md                                            -- BigQuery Access (MCP) section rewritten
scripts/mcp_servers/smoke_test_bigquery_mcp.py       -- NEW, end-to-end MCP probe
tests/verify_phase_23_2_21.py                        -- NEW, 5-check verifier
handoff/current/contract.md                          -- updated for phase-23.2.21
handoff/current/phase-23.2.21-external-research.md   -- researcher output
handoff/current/phase-23.2.21-internal-codebase-audit.md -- researcher output
```

## Verification (verbatim output)

```
$ uvx --from mcp-server-bigquery==0.3.2 mcp-server-bigquery --help
(installs cleanly on Python 3.14, prints help with --project/--location/--key-file/--dataset/--timeout)

$ source .venv/bin/activate && python scripts/mcp_servers/smoke_test_bigquery_mcp.py
spawning: uvx --from mcp-server-bigquery==0.3.2 mcp-server-bigquery --project sunny-might-477607-p8 --location US
OK initialize -- server=bigquery
OK tools/list -- ['describe-table', 'execute-query', 'list-tables']
OK tools/call list-tables -- response references project + pyfinagent_*
(server stderr tail confirms: Found 6 datasets, Found 33 tables)

$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_21.py
OK .mcp.json
OK .claude/settings.json
OK CLAUDE.md
OK scripts/mcp_servers/smoke_test_bigquery_mcp.py
OK scripts/mcp_servers/smoke_test_bigquery_mcp.py -- end-to-end

phase-23.2.21 verification: ALL PASS (5/5)
```

## Research-gate evidence

Researcher (ac6de9d55afe579de) returned `gate_passed: true`:
- 7 sources read in full via WebFetch (Google BQ MCP docs, LucasHild
  GitHub, PyPI listing, ergut alternative, MCP Toolbox quickstart,
  Google Cloud blog on remote BQ MCP, Anthropic Claude Code MCP docs).
- 17 URLs collected; 10 in snippet-only.
- Recency scan 2024-2026; 3 query variants per topic.
- 6 internal files inspected.
- Concrete recommendation with three flagged caveats; all three
  addressed in this generate phase.

## Operator handoff (mandatory)

The pinned MCP server is loaded by the harness at session start.
**The user MUST `/clear` (or restart Claude Code) for the new BQ MCP
to attach to the next conversation.** Until then:
- The current session continues with whatever MCP tools are already
  in scope.
- The smoke test (`scripts/mcp_servers/smoke_test_bigquery_mcp.py`)
  proves the pinned MCP works standalone with ADC — independent of
  Claude Code attachment status.

After `/clear`, ToolSearch with query `bigquery` should return
`mcp__bigquery__list-tables`, `mcp__bigquery__describe-table`,
`mcp__bigquery__execute-query`. The Claude.ai-hosted OAuth-demanding
proxy should no longer be needed.

## Backwards compatibility

- The old deny rule matched no real tool; removing it has no behavioral
  effect. Adding the hyphenated rule is strictly more protective.
- `.mcp.json` change is additive; alpaca MCP is unaffected.
- CLAUDE.md changes are doc-only; no code path depends on the wording.
- The Python `bigquery.Client` fallback path remains valid for any
  session where the MCP fails to attach.

## Honest disclosures

- **The pinned MCP only loads after the user restarts Claude Code.**
  Until `/clear` or app restart, the deferred-tool list still shows
  the OAuth proxy. The smoke test confirms the pinned server works
  standalone via uvx + ADC, but Claude Code itself won't see it
  until it re-reads `.mcp.json`.
- **Tool surface is narrower than CLAUDE.md previously claimed.** No
  separate readonly variant; `execute-query` is dual-use and gated by
  the deny rule. There is no `list_dataset_ids` / `get_dataset_info`
  on this package — a future phase could write a thin wrapper if those
  primitives are missed, but `list-tables` + `describe-table` cover
  most inspection workflows.
- **Bare-date / RFC3339 string columns still need SAFE.TIMESTAMP wraps**
  in any SQL we write through `execute-query` — same pitfall as
  phase-23.2.20. The pin doesn't change SQL semantics.
- **Pin version drift.** `0.3.2` is current as of 2026-05-05. Bumping
  later requires re-running the smoke test on Python 3.14.
- **uvx PATH dependency.** Same as the existing alpaca MCP. If
  Claude Code is launched from the GUI on a fresh user account, uvx
  may not be on PATH and both MCPs will fail to attach. Already
  proven OK in this environment because alpaca attaches fine.
