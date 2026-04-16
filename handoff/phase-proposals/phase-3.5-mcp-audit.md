# Phase 3.5 - MCP Tool Audit & Adoption

Status: proposal (pending)
Owner: Peder / harness
Depends on: phase-4.5 (harness hardening must ship before new tool surface lands)
Gate: none (research + adoption plan; individual tool adoptions gated case-by-case)

## Goal

Audit the Model Context Protocol (MCP) surface available to pyfinagent today,
catalog the April 2026 MCP ecosystem for finance/trading + developer workflow,
and deliver a prioritized adoption plan that expands harness and paper-trading
capabilities without introducing new cost, secret-leak, or tool-sprawl risk.

The deliverable is a roadmap that:

1. Inventories the in-repo MCP surface (`.mcp.json`, `backend/mcp/`, and the
   BigQuery MCP server already wired into the harness per CLAUDE.md).
2. Maps the published MCP registry (registry.modelcontextprotocol.io, spec
   rev 2025-11-25) and identifies tools that plug into pyfinagent's existing
   stack (broker execution, filings, fundamentals, macro, news, dev workflow).
3. Scores each candidate on license, cost, maintainer health, secret surface,
   and rate-limit risk.
4. Recommends a top-5 "adopt now" list and a later-stage watchlist.
5. Defines A/B test gates so every new MCP tool proves itself against the
   existing Python client before replacing it.

## Success criteria

Each item maps to a verification.success_criteria flag in the proposed JSON.

1. `scripts/audit/mcp_inventory.py --json` emits an inventory of
   (a) every server referenced in `.mcp.json`, (b) every FastMCP server in
   `backend/mcp/`, (c) every MCP tool actually invoked from agent transcripts
   in `handoff/`, with zero secrets in output.
2. `handoff/phase-proposals/phase-3.5-mcp-audit.md` (this file) contains the
   current-state table, candidate table (>= 10 rows), per-candidate risk
   scoring, and a top-5 adoption ranking with A/B test plan per entry.
3. Every "adopt now" MCP has (a) an Apache/MIT/AGPL license declared in the
   candidate table, (b) a ToS link, (c) a fallback Python client already in
   `backend/` so removal is reversible without a code rewrite.
4. No paid MCP is auto-adopted - any monthly cost line item carries an
   explicit "pending Peder approval" tag per CLAUDE.md's LLM-cost rule
   (extended to data/tool vendor cost).
5. Phase 3 step 3.5 "Enrichment MCP Server" (the existing stub in
   `.claude/masterplan.json`) is either completed or explicitly superseded
   with rationale linked here (no orphaned stubs).
6. All adopt-now servers pass the MCP security checklist (Invariant Labs +
   Anthropic MCP security blog, 2025-11): no implicit write scope, secrets
   read from env not flags, rate-limit guard set, kill-switch documented.
7. A/B harness test (`scripts/harness/mcp_ab_test.py`) proves each newly
   adopted tool returns equivalent data to the existing Python client on >=
   20 ticker samples before being flipped to "default route".
8. >= 20 unique URLs cited in `## References`, >= 10 of them full reads.

## Step-by-step plan

Each step follows the harness protocol (RESEARCH -> PLAN -> GENERATE ->
EVALUATE -> LOG) and is harness-gated unless noted.

### Step 3.5.0 - MCP surface inventory (read-only)

Build `scripts/audit/mcp_inventory.py`. Walk:
- `.mcp.json` + user-level `~/.claude/` configs referenced in repo docs.
- `backend/mcp/` FastMCP servers (data, backtest, signals - stubs today).
- `handoff/transcripts/` for any `mcp__<server>__<tool>` call patterns.

Emit JSON: `{server_id, source_path, tools[], last_invoked_at, secret_env_keys[]}`.
No secret values in output; only env-var key names.

Verification command:
`source .venv/bin/activate && python scripts/audit/mcp_inventory.py --json > handoff/mcp_inventory.json && python -c "import json; d=json.load(open('handoff/mcp_inventory.json')); assert d['servers'], d"`

Success criteria:
- `mcp_inventory_json_valid`
- `no_secrets_in_output`
- `stub_servers_flagged` (data/backtest/signals from backend/mcp/)

### Step 3.5.1 - MCP registry crawl + candidate shortlist

Script `scripts/audit/mcp_registry_pull.py` hits
`registry.modelcontextprotocol.io` (and the community meta-list at
awesome-mcp-servers) and filters for categories: finance, trading, data,
research, developer-tools. Dumps a CSV with (name, maintainer, license,
stars, last_commit, install_cmd, tools_count, cost_tier).

Short-list produced by this step is the input to 3.5.2 scoring.

Verification command:
`python scripts/audit/mcp_registry_pull.py --output handoff/mcp_candidates.csv && python -c "import csv; rows=list(csv.DictReader(open('handoff/mcp_candidates.csv'))); assert len(rows) >= 20, len(rows)"`

Success criteria:
- `registry_pull_returns_20plus_candidates`
- `all_candidates_have_license_field`
- `all_candidates_have_last_commit_within_180_days`

### Step 3.5.2 - Security + risk scoring

For every shortlisted server, score on:
- License compatibility (Apache/MIT/BSD safe; AGPL flagged; proprietary
  needs review).
- Secret surface (does the server need a new API key? is it read-only?).
- Rate-limit exposure (does it hit a paid upstream that could blow our budget
  if an agent loops?).
- Maintainer health (stars, issues-response median, last commit).

Output: `handoff/mcp_risk_scores.json` keyed by server id.

Verification command:
`python scripts/audit/mcp_risk_score.py && python -c "import json; d=json.load(open('handoff/mcp_risk_scores.json')); assert all('risk_band' in v for v in d.values())"`

Success criteria:
- `risk_score_json_valid`
- `every_candidate_scored`
- `paid_servers_tagged_pending_peder_approval`

### Step 3.5.3 - Adopt-now wave 1: Alpaca MCP (broker + paper trading)

Install + register `alpaca-mcp-server` (Anthropic-listed, broker-grade,
80+ tools, free paper account). Wire into `.mcp.json` with env-var secret
pull, not flag-based.

Acceptance: harness can place a paper order via Alpaca MCP in dry-run mode
and read it back via the same server; results match the existing
`backend/services/paper_trading.py` shadow path.

Verification command:
`python scripts/harness/mcp_ab_test.py --server alpaca --samples 20 && python -c "import json; d=json.load(open('handoff/mcp_ab_test_alpaca.json')); assert d['parity_rate'] >= 0.95"`

Success criteria:
- `alpaca_mcp_registered_in_mcp_json`
- `alpaca_paper_order_placed_and_read`
- `parity_rate_vs_existing_client_ge_95_percent`
- `no_live_orders_placed_during_test`

### Step 3.5.4 - Adopt-now wave 2: SEC EDGAR + FMP + FRED MCPs

Three read-only servers, all free, all with permissive licenses:
- SEC EDGAR MCP (stefanoamorelli, AGPL-3.0 - legal review required before
  production merge; isolate to research agent only).
- Financial Modeling Prep MCP (Apache-2.0, 253 tools, requires FMP key
  which we already hold; no new spend).
- FRED MCP (free, macro data; replaces bespoke `fredapi` calls).

A/B test each vs the existing Python client on 20 symbols; flip to default
route only after parity_rate >= 0.95 and p95 latency within 1.5x.

Verification command:
`python scripts/harness/mcp_ab_test.py --server edgar,fmp,fred --samples 20`

Success criteria:
- `edgar_parity_ge_95`
- `fmp_parity_ge_95`
- `fred_parity_ge_95`
- `agpl_isolation_documented` (EDGAR read path only, never bundled into a
  derivative artifact)
- `p95_latency_within_1_5x_existing`

### Step 3.5.5 - Complete / supersede existing phase-3.5 MCP stubs

The current `.claude/masterplan.json` has an Enrichment MCP Server step
under phase-3. Decision point:
- Option A: finish it (wrap the Gemini enrichment agent behind FastMCP).
- Option B: retire it in favor of vendor MCPs from 3.5.3 + 3.5.4 where
  overlap exists.

This step writes `handoff/phase-3.5-stub-decision.md` with the chosen
direction, then either lands the implementation or removes the stub with
a link to this phase as the replacement.

Verification command:
`test -f handoff/phase-3.5-stub-decision.md && grep -Eq '^Decision: (finish|retire)' handoff/phase-3.5-stub-decision.md`

Success criteria:
- `decision_documented`
- `if_retire_then_stub_removed_from_masterplan`
- `if_finish_then_enrichment_mcp_tests_green`

### Step 3.5.6 - Developer-workflow MCPs (later-stage watchlist)

Research-only step. Catalog dev-side MCPs (GitHub MCP, Sentry MCP, Linear
MCP, Playwright MCP, Exa search $49/mo) with proposed wire-in timeline.
No adoption here; output is a dated watchlist we revisit after phase-4
ships.

Verification command:
`test -f handoff/mcp_watchlist.md && python -c "import re; t=open('handoff/mcp_watchlist.md').read(); assert len(re.findall(r'https?://', t)) >= 10"`

Success criteria:
- `watchlist_file_exists`
- `ge_10_urls_cited`
- `every_entry_has_adopt_condition` (e.g., "after phase-6 ships")

### Step 3.5.7 - Ongoing MCP health cron

Append an APScheduler job to the existing harness config that, weekly,
hits `registry.modelcontextprotocol.io` for version bumps + security
advisories on adopted servers and posts a summary to Slack (#harness).
Implementation: reuse the `backend/services/sla_monitor.py` scheduler
pattern; no new scheduler process.

Verification command:
`python -c "from backend.services.mcp_health_cron import check_once; r = check_once(); assert 'servers' in r and isinstance(r['servers'], list)"`

Success criteria:
- `cron_job_registered_with_apscheduler`
- `slack_post_on_critical_advisory`
- `no_new_scheduler_process`

## Research findings

Findings compiled from the April 2026 MCP ecosystem. Full URLs in
`## References`. Key observations:

### Current pyfinagent MCP surface (April 2026)

- `.mcp.json` references the Slack MCP only; BigQuery MCP is injected by
  the harness environment per CLAUDE.md "BigQuery Access (MCP)" section -
  so two live servers in practice.
- `backend/mcp/` holds three FastMCP stubs (data, backtest, signals) that
  were scaffolded during phase-3 but never registered - zero active routes.
- `.claude/masterplan.json` phase-3 contains step 3.5 "Enrichment MCP
  Server" marked pending; it predates the broader phase-3.5 audit scope
  proposed here.

### Ecosystem state (spec rev 2025-11-25)

- MCP is now the de facto agent-tool protocol; Anthropic's official
  registry lists 120+ servers across data, dev-tools, productivity, and
  finance-specific categories.
- Finance-specific servers with production-grade maintainers:
  - Alpaca MCP - broker execution + market data + paper trading, 80+ tools.
  - SEC EDGAR MCP (stefanoamorelli) - 251 GitHub stars, AGPL-3.0.
  - Financial Modeling Prep MCP - 253 tools, Apache-2.0, covers 13F,
    congressional trades, insider transactions.
  - FRED MCP - macroeconomic series.
  - Polygon.io MCP - market data, paid.
  - Interactive Brokers MCP - community fork, execution.
  - yfinance MCP community wrapper - duplicates our existing path; low
    marginal value.
  - financekit (OpenBB) MCP - research-oriented, Apache-2.0.
- Dev workflow: Playwright MCP, Sentry MCP, Linear MCP, GitHub MCP, Exa
  search MCP - largely orthogonal to trading signals but relevant for the
  harness "observe frontend in browser" and "ticket-to-backtest" loops.

### Risks the literature flags (Invariant Labs + Anthropic security, Nov 2025)

- **Tool sprawl**: more tools == bigger prompt == higher hallucination rate.
  Cap at the top-5 that earn their keep.
- **Secret leaks**: MCP servers can exfiltrate env vars if maintainer is
  compromised. Rotate keys + pin versions.
- **Rate-limit amplification**: an agent loop that hits a paid MCP can
  burn budget in minutes. Add rate-limit guards and a global kill-switch.
- **License ambiguity**: AGPL servers (EDGAR) force downstream AGPL if
  their output is bundled into a derivative work. Read-path-only use
  avoids the taint in most legal reads, but isolate to be safe.

### Adoption ranking (top-5 adopt now + watchlist)

| Rank | Server | Cost | License | ROI vs existing |
|------|--------|------|---------|-----------------|
| 1 | Alpaca MCP | Free (paper) | Apache-2.0 | Replaces bespoke broker glue; 80+ tools; paper trading native |
| 2 | FMP MCP | Free w/ key we hold | Apache-2.0 | Adds 13F + congressional trades (otherwise phase-7 scope) |
| 3 | SEC EDGAR MCP | Free | AGPL-3.0 | Replaces hand-rolled EDGAR pulls; read-path-isolated |
| 4 | FRED MCP | Free | MIT | Replaces `fredapi` direct calls; consistent MCP invocation pattern |
| 5 | financekit (OpenBB) MCP | Free | Apache-2.0 | Research-agent boost; complements phase-2 step 2.10 Karpathy loop |

Watchlist (reassess after phase-4 live): Polygon.io (paid), IB MCP,
Exa MCP ($49/mo), Playwright MCP, GitHub MCP, Sentry MCP, Linear MCP.

### Anti-patterns

- Registering every registry entry indiscriminately ("just in case").
- Relying on a single community fork without a license and maintainer
  check.
- Letting an MCP write to prod tables without a Python client fallback.
- Hiding MCP secrets in flag args instead of env vars (shell history
  leak).

## Proposed masterplan.json snippet

```json
{
  "id": "phase-3.5",
  "name": "MCP Tool Audit & Adoption",
  "status": "pending",
  "depends_on": ["phase-4.5"],
  "gate": null,
  "steps": [
    {
      "id": "3.5.0",
      "name": "MCP surface inventory (read-only)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "source .venv/bin/activate && python scripts/audit/mcp_inventory.py --json > handoff/mcp_inventory.json && python -c \"import json; d=json.load(open('handoff/mcp_inventory.json')); assert d['servers'], d\"",
        "success_criteria": ["mcp_inventory_json_valid", "no_secrets_in_output", "stub_servers_flagged"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.1",
      "name": "MCP registry crawl + candidate shortlist",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/audit/mcp_registry_pull.py --output handoff/mcp_candidates.csv && python -c \"import csv; rows=list(csv.DictReader(open('handoff/mcp_candidates.csv'))); assert len(rows) >= 20, len(rows)\"",
        "success_criteria": ["registry_pull_returns_20plus_candidates", "all_candidates_have_license_field", "all_candidates_have_last_commit_within_180_days"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.2",
      "name": "Security + risk scoring",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/audit/mcp_risk_score.py && python -c \"import json; d=json.load(open('handoff/mcp_risk_scores.json')); assert all('risk_band' in v for v in d.values())\"",
        "success_criteria": ["risk_score_json_valid", "every_candidate_scored", "paid_servers_tagged_pending_peder_approval"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.3",
      "name": "Adopt-now wave 1: Alpaca MCP (broker + paper)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/mcp_ab_test.py --server alpaca --samples 20 && python -c \"import json; d=json.load(open('handoff/mcp_ab_test_alpaca.json')); assert d['parity_rate'] >= 0.95\"",
        "success_criteria": ["alpaca_mcp_registered_in_mcp_json", "alpaca_paper_order_placed_and_read", "parity_rate_vs_existing_client_ge_95_percent", "no_live_orders_placed_during_test"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.4",
      "name": "Adopt-now wave 2: SEC EDGAR + FMP + FRED MCPs",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/mcp_ab_test.py --server edgar,fmp,fred --samples 20",
        "success_criteria": ["edgar_parity_ge_95", "fmp_parity_ge_95", "fred_parity_ge_95", "agpl_isolation_documented", "p95_latency_within_1_5x_existing"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.5",
      "name": "Complete or supersede phase-3 Enrichment MCP stub",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "test -f handoff/phase-3.5-stub-decision.md && grep -Eq '^Decision: (finish|retire)' handoff/phase-3.5-stub-decision.md",
        "success_criteria": ["decision_documented", "if_retire_then_stub_removed_from_masterplan", "if_finish_then_enrichment_mcp_tests_green"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.6",
      "name": "Developer-workflow MCP watchlist",
      "status": "pending",
      "harness_required": false,
      "verification": {
        "command": "test -f handoff/mcp_watchlist.md && python -c \"import re; t=open('handoff/mcp_watchlist.md').read(); assert len(re.findall(r'https?://', t)) >= 10\"",
        "success_criteria": ["watchlist_file_exists", "ge_10_urls_cited", "every_entry_has_adopt_condition"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "3.5.7",
      "name": "Ongoing MCP health cron",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python -c \"from backend.services.mcp_health_cron import check_once; r = check_once(); assert 'servers' in r and isinstance(r['servers'], list)\"",
        "success_criteria": ["cron_job_registered_with_apscheduler", "slack_post_on_critical_advisory", "no_new_scheduler_process"]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## Implementation notes

Files to create:

- `scripts/audit/mcp_inventory.py` - read-only walker over `.mcp.json`,
  `backend/mcp/`, and handoff transcripts.
- `scripts/audit/mcp_registry_pull.py` - thin wrapper over the MCP registry
  API + awesome-mcp-servers scrape, outputs CSV.
- `scripts/audit/mcp_risk_score.py` - applies the scoring rubric from
  step 3.5.2 to the CSV.
- `scripts/harness/mcp_ab_test.py` - generic A/B tester: given a server
  id and sample count, runs equivalent queries against the MCP and the
  legacy Python client, emits parity/latency JSON.
- `backend/services/mcp_health_cron.py` - APScheduler job reusing the
  pattern in `backend/services/sla_monitor.py`.
- `handoff/mcp_watchlist.md` - dev-workflow MCP roadmap.
- `handoff/phase-3.5-stub-decision.md` - phase-3 Enrichment MCP
  finish-or-retire decision log.

Files to modify:

- `.mcp.json` - add Alpaca, FMP, FRED, EDGAR (isolated) server entries.
  Secrets via `${ENV_VAR}` templating, never literal.
- `.claude/masterplan.json` - insert phase-3.5 block; update
  phase-3.5 "Enrichment MCP Server" disposition per step 3.5.5 outcome.
- `CLAUDE.md` "BigQuery Access (MCP)" section - extend to "MCP Access"
  with the new adopt-now servers listed alongside BigQuery rules.

Estimated effort: ~4 engineer-days end-to-end (1 day audit + 1 day
Alpaca A/B + 1 day EDGAR/FMP/FRED A/B + 1 day cron + watchlist + docs).

Cost: zero incremental monthly spend for the top-5 (all free tiers or
existing keys). Paid servers (Polygon, Exa) explicitly deferred to
watchlist pending Peder approval.

Rollout risk: low. Every adopted server has an existing Python client
fallback. The A/B test gate (parity_rate >= 0.95) prevents a silent
regression from replacing a working path. The health cron catches
upstream advisories without us having to poll manually.

## References

Access dates in YYYY-MM-DD.

1. https://modelcontextprotocol.io/specification/2025-11-25 - MCP spec rev, 2026-04-16 (full read)
2. https://registry.modelcontextprotocol.io/ - Official registry home, 2026-04-16
3. https://github.com/modelcontextprotocol/servers - Reference servers, 2026-04-16 (full read)
4. https://www.anthropic.com/news/model-context-protocol - Anthropic MCP launch post, 2026-04-16
5. https://github.com/alpacahq/alpaca-mcp-server - Alpaca MCP, 2026-04-16 (full read)
6. https://alpaca.markets/docs/api-references/trading-api/ - Alpaca API docs, 2026-04-16
7. https://github.com/stefanoamorelli/sec-edgar-mcp - EDGAR MCP, 2026-04-16 (full read)
8. https://www.sec.gov/edgar/sec-api-documentation - EDGAR API, 2026-04-16
9. https://github.com/financialmodelingprep/fmp-mcp-server - FMP MCP, 2026-04-16 (full read)
10. https://site.financialmodelingprep.com/developer/docs - FMP API, 2026-04-16
11. https://github.com/stlouis-fed/fred-mcp - FRED MCP, 2026-04-16
12. https://fred.stlouisfed.org/docs/api/fred/ - FRED API, 2026-04-16
13. https://github.com/OpenBB-finance/OpenBBTerminal - OpenBB MCP parent, 2026-04-16 (full read)
14. https://github.com/punkpeye/awesome-mcp-servers - Awesome MCP list, 2026-04-16 (full read)
15. https://invariantlabs.ai/blog/mcp-security-notification - MCP security, 2026-03-18 (full read)
16. https://www.anthropic.com/engineering/building-effective-agents - Effective agents, 2026-04-16 (full read)
17. https://www.anthropic.com/engineering/built-multi-agent-research-system - Multi-agent research, 2026-04-16 (full read)
18. https://github.com/microsoft/playwright-mcp - Playwright MCP, 2026-04-16
19. https://github.com/getsentry/sentry-mcp - Sentry MCP, 2026-04-16
20. https://github.com/linear/linear-mcp - Linear MCP, 2026-04-16
21. https://github.com/github/github-mcp-server - GitHub MCP, 2026-04-16
22. https://docs.exa.ai/mcp - Exa MCP docs, 2026-04-16
23. https://polygon.io/docs/mcp - Polygon MCP docs, 2026-04-16
24. https://github.com/ranaroussi/yfinance - yfinance (existing client we keep as fallback), 2026-04-16
25. https://cloud.google.com/blog/products/ai-machine-learning/model-context-protocol-bigquery - BigQuery MCP on GCP, 2026-04-16
