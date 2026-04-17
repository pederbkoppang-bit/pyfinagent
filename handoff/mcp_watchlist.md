# MCP Watchlist -- Developer-Workflow Servers

Phase-3.5 step 3.5.6 artifact. Drafted 2026-04-17.

These MCP servers are NOT adopted today but are tracked for potential
future adoption. Each entry carries an explicit `adopt_condition` --
the concrete trigger that would flip us from watchlist to adoption.

## Watchlist (alphabetical by name)

### 1. Playwright MCP (Microsoft)

- url: https://github.com/microsoft/playwright-mcp
- category: browser automation
- license: Apache-2.0
- adopt_condition: after phase-4.7 UI/UX audit needs per-commit
  automated UI smoketests (today we have phase-4.6.6 Playwright
  inside the harness, but not integrated as a standing MCP).

### 2. Sentry MCP

- url: https://github.com/getsentry/sentry-mcp
- category: observability
- license: Apache-2.0
- adopt_condition: after phase-4 go-live -- once we have live-capital
  incidents, Sentry's error triage MCP surface would let harness
  agents query recent errors as context.

### 3. Linear MCP

- url: https://github.com/linear/linear-mcp
- category: productivity (issues / planning)
- license: MIT
- adopt_condition: if pyfinagent switches task tracking from
  .claude/masterplan.json to Linear. Currently not planned.

### 4. GitHub MCP (official)

- url: https://github.com/github/github-mcp-server
- category: source control / CI
- license: MIT
- adopt_condition: once the harness needs to autonomously create or
  review PRs (phase-10.7 Meta-Evolution recursive prompt optimization
  requires PR-scoped edits).

### 5. Exa Search MCP

- url: https://github.com/exa-labs/exa-mcp-server
- category: web search
- license: MIT (server); requires Exa paid API (minimum 49 USD/mo)
- adopt_condition: after Peder approves the 49 USD/mo line item AND
  phase-6.5 Global Intelligence Directive ships. Exa would be the
  research-agent web-search backend.

### 6. Cloudflare MCP

- url: https://github.com/cloudflare/mcp-server-cloudflare
- category: infrastructure
- license: Apache-2.0 (requires Cloudflare paid plan for production)
- adopt_condition: if pyfinagent ever hosts a public frontend behind
  Cloudflare (not planned; currently Tailscale-only).

### 7. Brave Search MCP

- url: https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search
- category: web search (alternative to Exa)
- license: MIT (server); requires Brave API key, free tier
  2000 queries/month
- adopt_condition: after phase-6.5 Global Intelligence Directive
  ships and if Exa exceeds budget. Brave is the free fallback.

### 8. Puppeteer MCP

- url: https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer
- category: browser automation (alternative to Playwright)
- license: MIT
- adopt_condition: not planned -- Playwright is already adopted in
  phase-4.6.6 and covers the same surface.

### 9. Google Drive MCP

- url: https://github.com/modelcontextprotocol/servers/tree/main/src/gdrive
- category: productivity / document store
- license: MIT
- adopt_condition: if pyfinagent starts storing research notes or
  outcome summaries in Google Drive. Currently we use BQ + GCS +
  handoff/ markdown; no need.

### 10. Postgres MCP

- url: https://github.com/modelcontextprotocol/servers/tree/main/src/postgres
- category: database
- license: MIT
- adopt_condition: only if we migrate any hot OLTP state off SQLite
  (tickets.db) into Postgres. BigQuery remains the analytics store;
  no plan to introduce Postgres.

### 11. Memory MCP

- url: https://github.com/modelcontextprotocol/servers/tree/main/src/memory
- category: agent memory / key-value store
- license: MIT
- adopt_condition: if phase-10.7 Meta-Evolution Recursive Prompt
  Optimization needs a durable cross-session memory surface that
  outgrows BM25 memory in backend/agents/memory.py.

### 12. GenAI Toolbox for Databases (Google)

- url: https://github.com/googleapis/genai-toolbox
- category: data (BigQuery + Spanner)
- license: Apache-2.0
- adopt_condition: if we want Google-managed BigQuery MCP instead of
  the harness-injected one. Currently the BigQuery MCP already
  provided by the harness is sufficient (see CLAUDE.md
  "BigQuery Access (MCP)").

## Review cadence

This watchlist is revisited at every phase-boundary (phase-4 go-live,
phase-6 ship, phase-8.5 autoresearch ship). Items with satisfied
adopt_conditions get moved to phase-3.5 step 3.5.3 / 3.5.4 work
streams on a rolling basis.

## Out of scope (explicitly not watching)

- Paid enterprise MCPs (Snowflake, Databricks) -- no customer data.
- Paid financial MCPs beyond Alpaca / FMP / FRED -- Polygon.io and
  similar defer until live AUM > 100k per phase-5.5 shopping list.
- Social / comms MCPs beyond Slack -- Discord, Microsoft Teams, etc.
  have no pyfinagent use case today.
