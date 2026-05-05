---
step: phase-23.2.21
title: Pin ADC-backed BigQuery MCP server in .mcp.json
cycle_date: 2026-05-05
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_21.py'
research_brief: handoff/current/phase-23.2.21-external-research.md (also see phase-23.2.21-internal-codebase-audit.md)
---

# Contract — phase-23.2.21

## Hypothesis

User: "BQ MCP needs OAuth — fix this." Forensic state:
- `.mcp.json` only pins the alpaca MCP (no BigQuery server).
- The discoverable BQ MCP today is `mcp__claude_ai_Google_Cloud_BigQuery__*`,
  which is the Claude.ai-managed proxy at `bigquery.googleapis.com/mcp`.
  It demands per-session 3-legged OAuth and IGNORES the user's local
  ADC. That's friction the user doesn't want.
- User's local ADC is valid (`gcloud auth application-default
  print-access-token` returns a token; `~/.config/gcloud/application_default_credentials.json`
  exists; project = `sunny-might-477607-p8`).
- CLAUDE.md says the BQ MCP is "harness-injected" — that is aspirational;
  no harness file in this repo actually injects it.
- The existing deny rule `mcp__bigquery__execute_sql` (underscored) does
  not match any real tool name on the package we'll pin (LucasHild
  exposes hyphenated `execute-query`) — so it's a dead rule that's
  given a false sense of guard.

**Pinning LucasHild's `mcp-server-bigquery==0.3.2` (Feb 2026) via `uvx`
mirrors the existing alpaca MCP shape, uses ADC automatically when
`--key-file` is omitted, and exposes 3 tools: `execute-query`,
`list-tables`, `describe-table`.** Pre-flight smoke check passed on
Python 3.14: `uvx --from mcp-server-bigquery==0.3.2 mcp-server-bigquery
--help` prints help with the documented args.

## Research-gate summary

Researcher (ac6de9d55afe579de) returned `gate_passed: true`:
- 7 sources read in full (Google BQ MCP docs, LucasHild GitHub,
  PyPI listing, ergut alternative, MCP Toolbox quickstart, Google Cloud
  blog on remote BQ MCP, Anthropic Claude Code MCP docs).
- 17 URLs collected; 10 in snippet-only.
- Recency scan 2024-2026 — no new Python+ADC+uvx-compatible server
  has emerged that supersedes LucasHild; ergut went stale Apr 2025.
- 6 internal files inspected.
- Concrete recommendation with three flagged caveats: (1) deny-rule
  name mismatch, (2) Python 3.14 test-matrix gap (now resolved by
  smoke check), (3) tool surface narrower than CLAUDE.md claims.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `.mcp.json` contains a `bigquery` server entry pinning
   `mcp-server-bigquery==0.3.2` via `uvx`, with `--project
   sunny-might-477607-p8 --location US` as args, no env vars required.
2. `.claude/settings.json` deny list no longer contains the obsolete
   `mcp__bigquery__execute_sql` (it matches nothing real on LucasHild's
   server). Replaced with `mcp__bigquery__execute-query` so write-class
   SQL still requires explicit user approval. Read-class tools
   (`list-tables`, `describe-table`) remain default "ask".
3. CLAUDE.md "BigQuery Access (MCP)" section is updated to drop the
   "harness-injected" myth, list the actual tool names exposed by
   LucasHild (`execute-query`, `list-tables`, `describe-table`), and
   flag that `execute-query` covers BOTH read and write (no separate
   readonly variant — write-class queries deny by default per criterion 2).
4. A smoke-test script `scripts/mcp_servers/smoke_test_bigquery_mcp.py`
   exists and exits 0 when run against the user's ADC. The script
   spawns the MCP server via stdio, sends `initialize` + `tools/list`
   + `tools/call list-tables` MCP messages, and asserts the response
   contains at least one table from the `pyfinagent_data` dataset.
5. The pre-flight already-confirmed: `uvx --from mcp-server-bigquery==0.3.2
   mcp-server-bigquery --help` exits 0 on Python 3.14 (this run).
6. `python tests/verify_phase_23_2_21.py` exits 0 (asserts the .mcp.json
   server entry shape, settings.json deny entry, CLAUDE.md doc update,
   and smoke-test script presence + exit code).
7. The auto-changelog hook fires on commit (per CLAUDE.md rule).

## Plan steps

1. Add the `bigquery` server to `.mcp.json`. Keep the alpaca entry
   intact. Mirror its shape (type=stdio, command=uvx, args list, env
   object). LucasHild needs no env vars — ADC fires from the user's
   gcloud config when `--key-file` is omitted. Pin version `0.3.2`
   so the next `uvx` invocation deterministically resolves to the
   tested release.
2. Edit `.claude/settings.json`:
   - Remove the dead `mcp__bigquery__execute_sql` deny entry.
   - Add `mcp__bigquery__execute-query` to deny (any caller — including
     this assistant — must escalate for write SQL). `list-tables`
     and `describe-table` remain default-ask.
3. Update CLAUDE.md "BigQuery Access (MCP)" section:
   - Drop "harness-injected" claim. Replace with "pinned in `.mcp.json`
     via phase-23.2.21".
   - Replace the tool list (`execute_sql_readonly`, `list_dataset_ids`,
     `get_dataset_info`, `get_table_info`, `list_table_ids`,
     `execute_sql`) with the ACTUAL three tools LucasHild exposes
     (`execute-query`, `list-tables`, `describe-table`). Note that
     `execute-query` is the only SQL path and is dual-use.
   - Update Rule 1 ("Default to execute_sql_readonly") to reflect that
     LucasHild has no readonly variant; instead, default to
     `list-tables` / `describe-table` for inspection and reach for
     `execute-query` only when SQL is needed (which prompts for
     approval per the deny rule).
4. Add `scripts/mcp_servers/smoke_test_bigquery_mcp.py`. Spawns the
   server via subprocess+stdio, sends MCP `initialize` then
   `tools/list` then `tools/call name=list-tables`, and asserts
   `pyfinagent_data` is in the response. 30s timeout. Exit 0 on success.
5. Add `tests/verify_phase_23_2_21.py` that: (a) parses `.mcp.json` and
   asserts the bigquery entry shape; (b) parses `.claude/settings.json`
   and asserts the deny rules; (c) greps CLAUDE.md for the new tool
   names; (d) runs the smoke-test script and asserts exit 0.
6. Append `handoff/harness_log.md` AFTER Q/A PASS.
7. After commit + push, the user must restart Claude Code (`/clear`
   or app restart) for the new MCP to be loaded by the harness. This
   is in the operator-handoff section of experiment_results.md.

## Out of scope

- Custom MCP wrapper in `scripts/mcp_servers/` (Python). Only justified
  if no maintained option existed; LucasHild satisfies the requirement.
- Migration to Google's remote managed MCP (`bigquery.googleapis.com/mcp`).
  Per-session OAuth is exactly what we're avoiding.
- MCP Toolbox (Go binary). Operationally heavy for a single-developer
  local deployment.
- Backfill/audit of existing CLAUDE.md `bq` CLI fallback rule (rule 6
  in the BigQuery Access section). Falling back to the python client
  with `GCP_PROJECT_ID` from `backend/.env` remains valid.

## Backwards compatibility

- The old deny rule `mcp__bigquery__execute_sql` matched no real tool;
  removing it has no behavioral effect. Adding `execute-query` to
  deny is strictly more protective.
- CLAUDE.md changes are documentation-only; no code path depends on
  the wording.
- Adding a server to `.mcp.json` is additive — alpaca is unaffected.
- New MCP server only loads after the user restarts Claude Code; until
  then the existing session keeps working as before.

## References

- Researcher: `handoff/current/phase-23.2.21-external-research.md`,
  `handoff/current/phase-23.2.21-internal-codebase-audit.md`
- LucasHild/mcp-server-bigquery v0.3.2 on PyPI (Feb 7 2026)
- Anthropic Claude Code MCP docs — `.mcp.json` `"type":"stdio"` shape
- `backend/db/bigquery_client.py:33` — same ADC fall-through as the
  pinned MCP will use
