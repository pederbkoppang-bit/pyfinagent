# Phase-23.2.21 External Research Brief — BigQuery MCP Server for pyfinagent

**Date:** 2026-05-05
**Tier:** moderate
**Topic:** Which BigQuery MCP server to pin in `.mcp.json` for local ADC-based access on macOS

---

## RECOMMENDATION (read this first)

**Chosen package:** `mcp-server-bigquery` by LucasHild (PyPI: `mcp-server-bigquery`, version `0.3.2`)

**Why this one:**
- Python-native (same ecosystem as the rest of pyfinagent's backend)
- Uses `uvx` as the runner — directly mirrors the existing `alpaca` MCP pattern in `.mcp.json`
- Supports Application Default Credentials natively: when `--key-file` / `BIGQUERY_KEY_FILE` is absent, the server falls through to ADC (`~/.config/gcloud/application_default_credentials.json`)
- No per-session browser OAuth; no SA key file needed for local dev
- Requires Python >=3.13 — pyfinagent's venv is Python 3.14, which satisfies `>=3.13` (no reported incompatibility as of 2026-05-05)
- Latest release: v0.3.2, Feb 7 2026 — actively maintained
- MIT license

**Install command (for testing):**
```bash
uvx mcp-server-bigquery --project sunny-might-477607-p8 --location US
```

**Exact `.mcp.json` snippet to add under `"mcpServers"`:**
```json
"bigquery": {
  "type": "stdio",
  "command": "uvx",
  "args": [
    "--from", "mcp-server-bigquery==0.3.2",
    "mcp-server-bigquery",
    "--project", "sunny-might-477607-p8",
    "--location", "US"
  ],
  "env": {}
}
```

No env vars are required because the project ID and location are passed as CLI args. The server picks up ADC automatically from `~/.config/gcloud/application_default_credentials.json`. If a service-account key is ever needed, add `"BIGQUERY_KEY_FILE": "/path/to/key.json"` to `"env"`.

**Expected tools (server-id = `bigquery`):**
- `mcp__bigquery__execute-query` — runs arbitrary SQL (BigQuery dialect); includes DML unless blocked
- `mcp__bigquery__list-tables` — lists all accessible tables in the project
- `mcp__bigquery__describe-table` — returns schema for a specific table

**IMPORTANT — tool naming vs deny rule:** The LucasHild server exposes tools named `execute-query`, `list-tables`, `describe-table` (hyphenated, not underscored). The existing deny rule in `.claude/settings.json` is `mcp__bigquery__execute_sql` (underscored). These names do NOT match, so the deny rule will NOT fire on `execute-query`. The GENERATE phase must either:
  - Add `mcp__bigquery__execute-query` to the deny list alongside the existing `mcp__bigquery__execute_sql`, OR
  - Accept that write-capable queries are gated only by the server's own lack of a read-only mode (the server has no built-in read-only switch unlike the official Google remote MCP)

**Known limitations:**
1. Only 3 tools (`execute-query`, `list-tables`, `describe-table`) vs the richer 6-tool surface of Google's remote MCP (`execute_sql`, `execute_sql_readonly`, `list_dataset_ids`, `list_table_ids`, `get_dataset_info`, `get_table_info`). In particular: no `get_dataset_info` and no `execute_sql_readonly` (read-only enforcement is absent).
2. No built-in query-size / byte-processed limit. The 30-second timeout rule from CLAUDE.md must still be enforced at the caller level.
3. Python 3.14 is not explicitly listed in the package's test matrix (it requires `>=3.13`). No breakage reported as of 2026-05-05 but this is worth re-verifying if `uvx` pulls a fresh interpreter.
4. Stars: 125; forks: not listed. Maintained by a single developer. Supply-chain risk is present — pin to `==0.3.2` exactly, consistent with the project's phase-3.7.6 supply-chain policy.

**Runtime warning — PATH:** When Claude Code launches MCP stdio servers, it inherits the shell PATH from the process that launched it. `uvx` must be on that PATH. Confirm with `which uvx` in the terminal that starts Claude Code. If launched via the GUI, add the uv bin directory explicitly: typically `~/.local/bin` or `/usr/local/bin`. The alpaca MCP already works via `uvx`, so this PATH issue is already solved for this project.

---

## Alternative Considered and Rejected: Google's official remote BQ MCP

**Why rejected:** `bigquery.googleapis.com/mcp` is a remote HTTP server that requires 3-legged OAuth 2.0 per session (browser redirect). It does not support stdio transport. It ignores ADC. Per-session OAuth is exactly the friction the user wants to avoid (confirmed by main session context). The server is in public preview as of Jan 2026 and does not support DML via `execute_sql_readonly`. Verdict: wrong transport, wrong auth model for this local deployment.

**Why the ergut TypeScript server was not chosen:** `@ergut/mcp-bigquery-server` (Node.js) supports ADC and has slightly more stars (138 vs 125) but requires Node.js >=14 as a runtime. The project already uses `uvx` for the alpaca MCP and has a Python venv; adding a Node dependency diverges from the established pattern. Last commit: April 3, 2025 — over a year ago, which fails the supply-chain freshness check (phase-3.7.6: stale >90 days).

**Why MCP Toolbox for Databases was not chosen:** `googleapis/mcp-toolbox` is a Go binary download (not pip/uvx installable). It requires a `tools.yaml` config file alongside the binary and cannot be pinned in `.mcp.json` as a `uvx`-invocable package. Setup overhead is too high for a local single-developer deployment. Does support ADC natively and is officially maintained by Google.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.cloud.google.com/bigquery/docs/use-bigquery-mcp | 2026-05-05 | Official Google doc | WebFetch | Remote HTTP only; OAuth 2.0 only; no ADC; no stdio; read-only via `execute_sql_readonly` |
| https://github.com/LucasHild/mcp-server-bigquery | 2026-05-05 | OSS GitHub README | WebFetch | ADC supported natively when no `--key-file`; `uvx mcp-server-bigquery --project X --location Y`; Python >=3.13; v0.3.2 Feb 2026 |
| https://pypi.org/project/mcp-server-bigquery/ | 2026-05-05 | Package registry | WebFetch | Python >=3.13; v0.3.2 released Feb 7 2026; `uvx` install confirmed |
| https://github.com/ergut/mcp-bigquery-server | 2026-05-05 | OSS GitHub README | WebFetch | Node.js; ADC supported; read-only only; last commit Apr 3 2025 — stale |
| https://mcp-toolbox.dev/integrations/bigquery/samples/local_quickstart/ | 2026-05-05 | Official Google Toolbox doc | WebFetch | Go binary; `tools.yaml` config; ADC via `gcloud auth login --update-adc`; macOS arm64/amd64 binaries available; not pip/uvx installable |
| https://cloud.google.com/blog/products/data-analytics/using-the-fully-managed-remote-bigquery-mcp-server-to-build-data-ai-agents | 2026-05-05 | Google Cloud Blog | WebFetch | Remote HTTP MCP in preview Jan 2026; ADC used for local dev of *client* but MCP connection itself uses OAuth 2.0 browser flow; no stdio |
| https://code.claude.com/docs/en/mcp | 2026-05-05 | Official Anthropic Claude Code doc | WebFetch | `.mcp.json` `"type":"stdio"` format confirmed; `${VAR}` env interpolation; `--scope project` writes to `.mcp.json`; `--env KEY=value` flag; warning about SSE deprecation |

---

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.pondhouse-data.com/blog/mcp-server-bigquery | Blog post | Fetched via WebFetch but covered MCP Toolbox (Go binary), not LucasHild — lower relevance; key info extracted |
| https://github.com/LucasHild/mcp-server-bigquery/issues/2 | GitHub issue | Fetched in full; result: missing `--project` arg causes `ValueError: Project is required` — now documented as known pitfall |
| https://medium.com/@gilad437/configuring-mcp-to-automate-data-analysis-integrating-bigquery-and-claude-the-devops-way-f36d7a815669 | Medium blog | Fetched but recommends SA key file (not ADC); uses Smithery CLI wrapper — not relevant to ADC-first approach |
| https://github.com/googleapis/mcp-toolbox | GitHub | Snippet only; established as Go binary — ruled out |
| https://mcpservers.org/servers/LucasHild/mcp-server-bigquery | MCP directory | Snippet only; duplicates PyPI info |
| https://glama.ai/mcp/servers/@takuya0206/bigquery-mcp-server | MCP directory | Snippet only; third-party server, not evaluated |
| https://github.com/aicayzer/bigquery-mcp | GitHub | Snippet only; multi-project BQ server; Docker-focused; no uvx |
| https://lobehub.com/mcp/pvoo-bigquery-mcp | MCP directory | Snippet only; vector-search focused read-only server |
| https://github.com/pvoo/bigquery-mcp | GitHub | Snippet only; read-only + vector search; limited tools |
| https://composio.dev/toolkits/googlebigquery/framework/claude-code | Blog | Snippet only; Composio wrapper, adds another dependency layer |

---

## Recency scan (2024-2026)

**Queries run (3 variants per protocol):**
1. Current-year frontier: `"Google Cloud BigQuery MCP server github 2026"`
2. Last-2-year window: `"BigQuery MCP server application default credentials ADC authentication 2025"`
3. Year-less canonical: `"use bigquery MCP server google cloud official documentation"`
4. Supplementary: `"mcp-server-bigquery pypi npm actively maintained 2026"`, `"BigQuery MCP server model context protocol authentication 2024"`

**Findings from the 2024-2026 window:**

The MCP ecosystem for BigQuery has moved substantially in this period:
- Google's official remote BigQuery MCP server launched in **public preview, January 2026** (blog post confirmed via WebFetch). It is the most capable (6 tools, DML support, IAM-scoped) but uses remote HTTP + OAuth 2.0 — unsuitable for friction-free local use.
- `mcp-server-bigquery` by LucasHild released v0.3.2 on **February 7, 2026** — the most recent Python/uvx-compatible option.
- `@ergut/mcp-bigquery-server` (Node.js) last committed **April 3, 2025** — over a year of inactivity; fails the project's 90-day staleness check.
- `googleapis/mcp-toolbox` (formerly `genai-toolbox`) is actively maintained by Google in 2025-2026 but is a Go binary, not a pip/uvx package.
- No new Python+ADC+uvx BigQuery MCP server emerged between late 2024 and May 2026 that would supersede the LucasHild package.

**Conclusion:** No 2024-2026 finding supersedes the recommendation. The LucasHild package is the most recently updated, ADC-compatible, uvx-runnable option available.

---

## Key findings

1. **The official Google remote BQ MCP requires 3-legged OAuth 2.0 per session — not ADC.** It runs at `bigquery.googleapis.com/mcp` over HTTP; no stdio support. (Source: Google Cloud Docs, `docs.cloud.google.com/bigquery/docs/use-bigquery-mcp`, 2026-05-05)

2. **LucasHild `mcp-server-bigquery` supports ADC natively: no key file = ADC.** The server docs explicitly document `--key-file` as optional; omitting it triggers ADC. (Source: GitHub README, `github.com/LucasHild/mcp-server-bigquery`, 2026-05-05)

3. **`uvx mcp-server-bigquery --project X --location Y` is the zero-config local invocation.** Version pin `==0.3.2` should be applied via `--from mcp-server-bigquery==0.3.2` to match project supply-chain policy. (Source: PyPI + GitHub README, 2026-05-05)

4. **The Claude Code `.mcp.json` env-var interpolation syntax is `${VAR}` (no default fallback syntax needed for BQ — project/location are passed as args, not env vars).** The `${VAR:-}` pattern seen in the alpaca entry is for optional secrets; not required here. (Source: Anthropic Claude Code MCP docs, `code.claude.com/docs/en/mcp`, 2026-05-05)

5. **The tool names from LucasHild are hyphenated (`execute-query`), not underscored (`execute_sql`).** The existing `mcp__bigquery__execute_sql` deny rule will NOT match. A new deny entry `mcp__bigquery__execute-query` must be added in `.claude/settings.json` during the GENERATE phase if write protection is desired. (Source: GitHub README tools section + settings.json audit, 2026-05-05)

6. **The ergut Node.js server is stale (last commit Apr 3 2025) and read-only.** It fails the 90-day supply-chain freshness check from phase-3.7.6. (Source: GitHub repo, `github.com/ergut/mcp-bigquery-server`, 2026-05-05)

7. **MCP Toolbox for Databases (Google, Go binary) supports ADC and is actively maintained but cannot be invoked via `uvx`.** It requires a separate binary + `tools.yaml` config file. Suitable as a future upgrade path if the LucasHild server goes unmaintained. (Source: MCP Toolbox quickstart docs, `mcp-toolbox.dev/integrations/bigquery/samples/local_quickstart/`, 2026-05-05)

8. **The only known startup issue with LucasHild's server is a missing `--project` arg** (raises `ValueError: Project is required`). Mitigated by always passing `--project sunny-might-477607-p8` in the `.mcp.json` args. (Source: GitHub issue #2, `github.com/LucasHild/mcp-server-bigquery/issues/2`, 2026-05-05)

---

## Consensus vs debate

**Consensus:** ADC is the correct auth path for local development. All sources agree that `~/.config/gcloud/application_default_credentials.json` (from `gcloud auth application-default login`) is the right credential source for local-only deployments.

**Debate:** Whether to use LucasHild (uvx, Python, 3 tools) vs Google MCP Toolbox (Go binary, richer tool surface, ADC, actively maintained by Google). The Toolbox is more future-proof but significantly more operationally complex for a single-developer local deployment. The LucasHild server is the pragmatic choice given the existing `uvx` pattern.

**Open question not resolvable by research:** Will `uvx mcp-server-bigquery==0.3.2` resolve and install cleanly under Python 3.14 on macOS 25.4.0 (Darwin)? Python 3.14 satisfies `>=3.13` but hasn't been in the test matrix. The GENERATE phase should run `uvx mcp-server-bigquery==0.3.2 --project sunny-might-477607-p8 --location US --help` as a pre-flight check.

---

## Pitfalls (from literature and audit)

| Pitfall | Source |
|---------|--------|
| Remote BQ MCP (`bigquery.googleapis.com/mcp`) ignores ADC — uses OAuth 2.0 per-session browser flow | Google Cloud Docs, 2026-05-05 |
| Missing `--project` arg causes `ValueError: Project is required` at startup | GitHub issue #2, LucasHild repo, 2026-05-05 |
| `uvx` may not be on PATH when Claude Code is launched from GUI | Pondhouse-data blog + Claude Code docs, 2026-05-05 |
| `execute-query` tool name (hyphen) vs deny-rule `execute_sql` (underscore) — deny rule does not fire | Settings.json audit + GitHub README, 2026-05-05 |
| ergut server is stale (Apr 2025 last commit) — do not adopt | GitHub repo audit, 2026-05-05 |
| MCP Toolbox Go binary is not pip/uvx-installable — wrong runner for this project | MCP Toolbox docs, 2026-05-05 |
| No `execute_sql_readonly` tool in LucasHild server — read-only enforcement must be done via Claude settings deny list | GitHub README + tool comparison, 2026-05-05 |

---

## Application to pyfinagent

| External finding | Maps to |
|-----------------|---------|
| ADC-first auth | `backend/db/bigquery_client.py:25-35` — same pattern; MCP server must match |
| `uvx --from pkg==ver` invocation | `.mcp.json:4-5` alpaca entry — mirror exactly |
| Tool name `execute-query` (hyphen) | `.claude/settings.json:~68` deny list — must add `mcp__bigquery__execute-query` during GENERATE |
| `--project` required arg | Must be `sunny-might-477607-p8` in `.mcp.json` args |
| `--location` required arg | Must be `US` (primary dataset location per CLAUDE.md) |
| Supply-chain pin `==0.3.2` | Phase-3.7.6 policy documented in `docs/MCP_ARCHITECTURE.md` |
| Server-id must be `bigquery` | Masterplan check: `any('bigquery__execute_sql' in r for r in deny)` — server-id `bigquery` ensures `mcp__bigquery__*` tool namespace |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched: Google BQ MCP doc, LucasHild GitHub, PyPI, ergut GitHub, MCP Toolbox quickstart, Google Cloud Blog, Anthropic Claude Code docs)
- [x] 10+ unique URLs total (17 URLs collected across searches)
- [x] Recency scan (last 2 years) performed + reported (3-query variant discipline satisfied: 2026 frontier, 2025 window, year-less canonical)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see internal audit doc)

Soft checks:
- [x] Internal exploration covered every relevant module (`bigquery_client.py`, `.mcp.json`, `settings.json`, `MCP_ARCHITECTURE.md`, masterplan, all of `scripts/harness/`)
- [x] Contradictions / consensus noted (OAuth vs ADC debate; LucasHild vs Toolbox tradeoff)
- [x] All claims cited per-claim (not just listed in footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
