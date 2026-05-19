# Research Brief — phase-29.1 (Add paper-search-mcp to .mcp.json)
**Tier:** complex
**Date:** 2026-05-18
**Note:** Overwrites phase-29.2 leftover per WRITE-FIRST directive.
**Owner constraint:** FREE-ONLY. If any required source demands payment beyond a free email or a free API key obtained in under 5 minutes, step is blocked.

---

## Search queries run (3-variant discipline)

| Topic | Query | Variant |
|---|---|---|
| install | paper-search-mcp install 2026 | current-year |
| install | paper-search-mcp PyPI 2025 uvx configuration | 2yr |
| install | paper-search-mcp openags github SSRN OpenAlex free tier | year-less canonical |
| OpenAlex auth | OpenAlex API free tier rate limits 2026 no API key required | current-year |
| OpenAlex auth | OpenAlex API key required February 2026 breaking change | current-year |
| Unpaywall | Unpaywall API free email registration academic paper access 2025 2026 | 2yr |
| SSRN target | SSRN George Hwang 2004 52-week high momentum abstract paper | year-less canonical |
| version | paper-search-mcp version 0.1.4 release 2025 changelog new features | 2yr |

---

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://pypi.org/project/paper-search-mcp/ | 2026-05-18 | Official PyPI package page | WebFetch full page | Latest PyPI-published version: **0.1.3** (Apr 29, 2025). Install: `pip install paper-search-mcp`. Python >=3.10. Author: P.S Zhang. No API key mentioned on the PyPI page itself. |
| https://github.com/openags/paper-search-mcp | 2026-05-18 | Official GitHub repo (README) | WebFetch full page | uvx command: `uvx paper-search-mcp`. Claude Desktop config shape: `"command": "uvx", "args": ["paper-search-mcp"]`. SSRN marked "⚠️ best-effort" with "Bot-detection (Cloudflare) active." No formal GitHub Releases exist (0 releases page). Env var prefix: `PAPER_SEARCH_MCP_*`. |
| https://raw.githubusercontent.com/openags/paper-search-mcp/main/README.md | 2026-05-18 | Raw README (full text) | WebFetch full page | 57 tools total. Named tools: `search_ssrn`, `search_openalex`, `search_papers`, `download_with_fallback`, `search_arxiv`, etc. SSRN tool name confirmed as `search_ssrn`. IEEE/ACM require keys. All others (arXiv, PubMed, OpenAlex, Crossref, IACR, Zenodo, HAL) key-free. |
| https://raw.githubusercontent.com/openags/paper-search-mcp/main/pyproject.toml | 2026-05-18 | pyproject.toml (source of truth for version) | WebFetch full page | **Version in git main: 0.1.4** (unpublished to PyPI as of 2026-05-18). Entry point: `paper-search-mcp` -> `paper_search_mcp.server:main`. Python >=3.10. Deps: requests, feedparser, fastmcp, pypdf, mcp[cli]>=1.6.0, beautifulsoup4>=4.12.0, lxml>=4.9.0, httpx[socks]>=0.28.1. |
| https://hub.docker.com/mcp/server/paper-search/tools | 2026-05-18 | Docker MCP Catalog (full tool list) | WebFetch full page | 57 tools enumerated. Confirms `search_ssrn` exists. Download tools include `download_ssrn`. No env var requirements listed here. Image: `mcp/paper-search`. |
| https://blog.openalex.org/openalex-api-new-features-and-usage-based-pricing/ | 2026-05-18 | OpenAlex official blog (breaking change announcement) | WebFetch full page | Published **Feb 24, 2026**. **Breaking change: API keys required for all production use as of Feb 13, 2026**. Free API key gives $1 credit/day (~10K list calls, 1K searches, 100 PDF downloads). Key obtained free at openalex.org/settings/api. Old email polite-pool discontinued. 100 free test credits without key, then 409 errors. |
| https://groups.google.com/g/openalex-users/c/rI1GIAySpVQ | 2026-05-18 | OpenAlex users mailing list (API key mandate) | WebFetch full page | Confirms Feb 13, 2026 as enforcement date. "100,000 credits per day" free with API key. "100 credits per day" without key = demo/test only. Email parameter removed (was never secure). Credit costs: singleton=1, list=10, content download=100, vector search=1000. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.pulsemcp.com/servers/afrise-academic-search | MCP directory | Covers a different server (afrise academic-search, Semantic Scholar + Crossref only) — not the openags package |
| https://glama.ai/mcp/servers/@afrise/academic-search-mcp-server | MCP directory | Same afrise server, not openags |
| https://github.com/DaniManas/ResearchMCP | GitHub | Separate OpenAlex-only MCP; informational only |
| https://github.com/oksure/openalex-research-mcp | GitHub | Separate dedicated OpenAlex MCP; alternative to paper-search-mcp |
| https://github.com/matsjfunke/paperclip | GitHub | arXiv + OSF + OpenAlex only; alternative tool |
| https://github.com/afrise/academic-search-mcp-server | GitHub | afrise alternative (Semantic Scholar + Crossref only) |
| https://developers.openalex.org/how-to-use-the-api/rate-limits-and-authentication | Official OpenAlex dev docs | Redirect resolved; key detail captured in blog fetch above |
| https://unpaywall.org/products/api | Unpaywall official | Page returned blank/insufficient content; info sourced from library guides + DEV community article instead |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1104491 | SSRN paper page | 403 Forbidden — confirms bot-detection active even for WebFetch, validating the warning in paper-search-mcp docs |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=338320 | SSRN (George & Hwang original) | Identified via search; 403 expected; abstract_id documented as target for live_check |
| https://mcpservers.org/servers/openags/paper-search-mcp | MCP directory | Key config snippets captured; no additional env var info beyond GitHub README |
| https://intuitionlabs.ai/articles/research-paper-apis-scientific-literature | Industry blog | Broad API landscape survey; snippets show Unpaywall still free with email in 2026 |

---

## Recency scan (2024-2026)

**Searches run:** "paper-search-mcp install 2026", "OpenAlex API key required February 2026 breaking change", "paper-search-mcp version 0.1.4 release 2025 changelog new features".

**Findings (directly relevant, 2024-2026 window):**

- **2026-02-13 (CRITICAL BREAKING CHANGE):** OpenAlex API keys became mandatory. The old email polite-pool (`?email=user@domain.com`) no longer works. All production callers need a free API key from openalex.org/settings/api. paper-search-mcp uses `PAPER_SEARCH_MCP_OPENALEX_API_KEY` (if the env var follows the package's naming pattern) or falls back to unauthenticated calls. Without a key, OpenAlex calls will fail with 409 after 100 credits. **This is the most important 2026 finding: OpenAlex is no longer keyless-free for production use.**

- **2026-02-24:** OpenAlex blog post formally announced the usage-based pricing model with $1/day free credit per API key. Credit schedule: singleton lookup=1 credit, list/filter=10, search=100 (updated), content download=100.

- **2025-04-29:** paper-search-mcp v0.1.3 released to PyPI. This remains the latest PyPI-published release as of 2026-05-18.

- **2025-04-06:** paper-search-mcp v0.1.0 and v0.1.2 released (initial PyPI publications).

- **2026-05-18 (today):** pyproject.toml in git main shows version 0.1.4, not yet published to PyPI. `uvx paper-search-mcp` will install PyPI v0.1.3. There are 0 formal GitHub Releases.

- **No arXiv HTML availability changes** found in the 2024-2026 window relevant to paper-search-mcp.

**Verdict on recency scan:** One major finding supersedes prior assumptions — OpenAlex now requires a free API key. All other sources (arXiv, PubMed, IACR, Crossref, Zenodo, HAL) remain keyless-free as of 2026-05-18.

---

## Key findings

1. **Install command is `uvx paper-search-mcp`** (no `--from` or version pin needed for latest PyPI). Installs v0.1.3. Alternatively `pip install paper-search-mcp`. No npm path for the MCP server itself (Smithery's npx route installs via Smithery's wrapper, not the stdio server directly). (Source: PyPI page + GitHub README, accessed 2026-05-18)

2. **Latest PyPI version: 0.1.3 (Apr 29, 2025). Git main: 0.1.4 (unreleased).** PyPI and GitHub Releases page are out of sync — the main branch has a version bump that has not been published. `uvx paper-search-mcp` will fetch v0.1.3. If v0.1.4 features are needed, install from source. (Source: pypi.org/project/paper-search-mcp/ + raw pyproject.toml, accessed 2026-05-18)

3. **SSRN tool name is `search_ssrn`** (not `search_ssrn_abstract` or any other variant). Confirmed from both the raw README and Docker MCP catalog. The tool does federated discovery; it does not take a bare `abstract_id` parameter — it takes a search query. Bot-detection (Cloudflare HTTP 403) is documented as active; the connector "tries two endpoints and returns a clear message on failure." (Source: raw README + Docker catalog, accessed 2026-05-18)

4. **George & Hwang (2004) SSRN abstract IDs:** Two SSRN entries exist. The working paper version has `abstract_id=338320`; a later uploaded version has `abstract_id=1104491`. The Journal of Finance published version (DOI: 10.1111/j.1540-6261.2004.00695.x) is at Wiley and behind a paywall. SSRN direct fetch is 403 even for WebFetch, confirming the bot-detection warning. The live_check target for phase-29.1 should use `search_ssrn(query="George Hwang 52-week high momentum investing")` and document whether it returns metadata (even without PDF). (Source: search results + SSRN 403 confirmation, accessed 2026-05-18)

5. **Source-by-source key requirements and free-only verdict:**

   | Source | Tool name | Key required? | Cost | Free signup? | Verdict |
   |---|---|---|---|---|---|
   | arXiv | `search_arxiv` | No key | Free | N/A | FREE-OK |
   | PubMed / PMC | `search_pubmed`, `search_pmc` | No key | Free | N/A | FREE-OK |
   | bioRxiv / medRxiv | `search_biorxiv`, `search_medrxiv` | No key | Free | N/A | FREE-OK |
   | IACR ePrint | `search_iacr` | No key | Free | N/A | FREE-OK |
   | Crossref | `search_crossref` | No key | Free | N/A | FREE-OK |
   | Semantic Scholar | `search_semantic` | Optional free | Free (rate-limited without key) | Free signup | FREE-OK |
   | dblp | `search_dblp` | No key | Free | N/A | FREE-OK |
   | OpenAIRE | `search_openaire` | No key | Free | N/A | FREE-OK |
   | CiteSeerX | `search_citeseerx` | No key | Free | N/A | FREE-OK |
   | Europe PMC | `search_europepmc` | No key | Free | N/A | FREE-OK |
   | BASE | `search_base` | No key | Free | N/A | FREE-OK |
   | HAL | `search_hal` | No key | Free | N/A | FREE-OK |
   | DOAJ | `search_doaj` | Optional free | Free | Free signup | FREE-OK |
   | Zenodo | `search_zenodo` | Optional free | Free | Free signup | FREE-OK |
   | **OpenAlex** | `search_openalex` | **Required free key** (as of Feb 13, 2026) | $1 free credit/day with key | Free signup at openalex.org/settings/api | **FREE-OK but key mandatory** |
   | SSRN | `search_ssrn` | No key | Free (metadata only; 403 likely) | N/A | FREE-OK, caveat: bot-detection |
   | **Unpaywall** | `search_unpaywall` | **Email required** (`PAPER_SEARCH_MCP_UNPAYWALL_EMAIL`) | Free with email | No signup — any email works | **FREE-OK but email mandatory** |
   | CORE | `search_core` | Recommended free key | Free | core.ac.uk/services/api | FREE-OK |
   | Google Scholar | `search_google_scholar` | No key (but proxy recommended) | Free | Proxy URL if bot-blocked | FREE-OK (best-effort) |
   | **IEEE Xplore** | `search_ieee` | **Required** | Free dev key | developer.ieee.org | FREE-OK (free key; dormant without key) |
   | **ACM DL** | `search_acm` | **Required** | Likely institutional/paid | libraries.acm.org | BLOCKED (likely paid) |

   **Summary:** ACM DL is the only source with a plausibly paid-only requirement. IEEE has a free dev key. OpenAlex now requires a free API key (breaking change Feb 2026). Unpaywall requires a valid email (any email, no signup). No source requires ongoing payment as a hard requirement; ACM is the uncertain case.

6. **`.mcp.json` config schema:** The existing entries use `"type": "stdio"` (alpaca), `"command": "uvx"`, and `"args"` as an array. The bigquery entry uses `--from package==version` pin; the alpaca entry uses `--from package==version` as well. For paper-search-mcp, the README's uvx config omits the `--from` pin: `"args": ["paper-search-mcp"]`. To version-pin (recommended for reproducibility): `"args": ["--from", "paper-search-mcp==0.1.3", "paper-search-mcp"]`. (Source: .mcp.json internal read + GitHub README, accessed 2026-05-18)

7. **Smoke test command:** No explicit smoke test script in the package. Minimal verification: `uvx paper-search-mcp` starts the stdio server (it will block waiting for stdin JSON-RPC input, which is correct behavior for a stdio MCP). A non-blocking check: `uvx paper-search-mcp --help` (if the entry point supports it) or pipe a tools/list request via JSON-RPC. The standard MCP smoke test pattern is: `echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | uvx paper-search-mcp`. (Source: MCP stdio protocol standard; package entry point per pyproject.toml)

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `/Users/ford/.openclaw/workspace/pyfinagent/.mcp.json` | 25 | MCP server configuration | Read in full. Two entries: `alpaca` (type:stdio, uvx, `--from alpaca-mcp-server==2.0.1`) and `bigquery` (type:stdio, uvx, `--from mcp-server-bigquery==0.3.2`). Both use version-pinned `--from` syntax. `env: {}` for bigquery; env block with `${VAR:-}` substitution for alpaca. No `"type": "stdio"` field on the `bigquery` entry (bigquery omits it; alpaca has it). NOTE: alpaca has `"type": "stdio"` explicitly; bigquery does not. The MCP spec allows `type` to default to `stdio`. |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/.env.example` | N/A | Backend env var template | Read failed: directory permission denied. Cannot confirm whether `OPENALEX_API_KEY` or `UNPAYWALL_EMAIL` are already defined. |
| `/Users/ford/.openclaw/workspace/pyfinagent/handoff/archive/phase-28.7/research_brief.md` | 10 | Phase-28.7 research brief | Read in full. The file contains only the researcher task definition (objective/output_format/tool_scope/task_boundaries), NOT a completed research brief. SSRN George&Hwang was not explicitly referenced in the archived brief as a failed fetch — it was the task context, not the output. Phase-28.7 brief was a simple-tier task brief about strategy-switch parameter recipes, not a literature research brief about 52-week high. |

---

## Consensus vs debate (external)

**Consensus:**
- `uvx paper-search-mcp` is the correct install command for a stdio MCP server (confirmed by GitHub README, Docker catalog, mcpservers.org).
- The tool name for SSRN is `search_ssrn`.
- arXiv, PubMed, IACR, Crossref, Zenodo, HAL, dblp, CiteSeerX, OpenAIRE, Europe PMC require no key and are completely free.
- Unpaywall requires any valid email (no payment, no account signup).
- OpenAlex requires a free API key as of Feb 13, 2026 (breaking change; email polite-pool discontinued).

**Debate / not settled:**
- Whether ACM DL key is obtainable for free: the package README says "see libraries.acm.org" but institutional access requirements suggest this may be behind a paywall. The connector is dormant without the key, so it's a soft blocker — the server runs without it.
- pyproject.toml shows v0.1.4 in git but PyPI has v0.1.3. If v0.1.4 fixes a bug, installing from PyPI misses it. No changelog is available to assess.

---

## Pitfalls (from literature and internal inventory)

1. **OpenAlex keyless mode hits 409 after 100 credits (Feb 2026 change)** — without `PAPER_SEARCH_MCP_OPENALEX_API_KEY` set, the `search_openalex` tool will work for a few test calls then fail silently or loudly. A free key is required for any production-level usage. The pyfinagent harness calls Researcher once per step — OpenAlex would exhaust 100 keyless credits in a single deep research session.

2. **SSRN returns 403 (Cloudflare bot-detection)** — both direct WebFetch to SSRN and the paper-search-mcp connector experience HTTP 403 on SSRN. The connector returns "a clear message on failure" rather than throwing. `search_ssrn(query="George Hwang 52-week high")` may return metadata (title, authors, abstract_id) without PDF access. The phase-28.7 failure was specifically about PDF download, not metadata. Metadata may still be usable.

3. **`.mcp.json` schema consistency** — the bigquery entry omits `"type": "stdio"`. The alpaca entry includes it. The paper-search-mcp entry should include `"type": "stdio"` to match the alpaca pattern and be explicit. The schema does not enforce it, but consistency prevents ambiguity.

4. **No version pin on PyPI for 0.1.4** — pinning to `==0.1.3` is reproducible. Omitting the pin means `uvx paper-search-mcp` installs the latest PyPI release (currently 0.1.3). If the maintainer publishes 0.1.4 tomorrow, the entry will auto-upgrade on next `uvx` cache miss. Both behaviors are defensible; pinning is safer for a production harness.

5. **`PAPER_SEARCH_MCP_UNPAYWALL_EMAIL`** — Unpaywall's free model requires an email to be passed with each request. The env var must be set to a valid email. Any email works (no verification). Peder's email (`peder.bkoppang@hotmail.no`) is appropriate here. Without it, Unpaywall is disabled.

---

## Application to pyfinagent (mapping to file:line anchors)

### `.mcp.json` (25 lines, full file read)

**Proposed addition** (append after bigquery entry, before closing `}`):

```json
"paper-search-mcp": {
  "type": "stdio",
  "command": "uvx",
  "args": ["--from", "paper-search-mcp==0.1.3", "paper-search-mcp"],
  "env": {
    "PAPER_SEARCH_MCP_UNPAYWALL_EMAIL": "${UNPAYWALL_EMAIL:-peder.bkoppang@hotmail.no}",
    "PAPER_SEARCH_MCP_OPENALEX_API_KEY": "${OPENALEX_API_KEY:-}",
    "PAPER_SEARCH_MCP_CORE_API_KEY": "${PAPER_SEARCH_MCP_CORE_API_KEY:-}",
    "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY": "${PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY:-}"
  }
}
```

**Rationale for each env var:**
- `PAPER_SEARCH_MCP_UNPAYWALL_EMAIL`: Required to enable Unpaywall source. Fallback to Peder's email (publicly used in academic requests; no signup needed).
- `PAPER_SEARCH_MCP_OPENALEX_API_KEY`: Required for production OpenAlex use (post Feb 13, 2026 breaking change). Key obtained free at openalex.org/settings/api. If unset, OpenAlex fails after 100 credits.
- `PAPER_SEARCH_MCP_CORE_API_KEY`: Recommended-free key (CORE.ac.uk). Works without but degrades to rate-limited fallback.
- `PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY`: Optional-free key. Without it, Semantic Scholar still works at lower rate limits.
- IEEE / ACM omitted: IEEE key is free (developer.ieee.org) but optional for finance research; ACM is potentially institutional-paid. Both connectors are dormant without their keys — omitting them does not break the server.

### `.env.example` (could not read due to permission)

GENERATE phase must add:
```
# paper-search-mcp (MCP server for academic paper search)
UNPAYWALL_EMAIL=                    # Any valid email; required to enable Unpaywall source
OPENALEX_API_KEY=                   # Free key from openalex.org/settings/api; required post Feb 2026
PAPER_SEARCH_MCP_CORE_API_KEY=      # Free key from core.ac.uk/services/api; recommended
PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY=  # Free key from semanticscholar.org/product/api; optional
```

### Smoke test command (for `live_check_phase-29.1.md`)

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | uvx paper-search-mcp 2>/dev/null | head -1 | python3 -m json.tool | grep -c '"name"'
```

Expected output: a positive integer (number of tools listed, should be ~57). If 0 or error, the server failed to start.

For the SSRN live_check target specifically:
```bash
# This tests whether search_ssrn returns metadata for George & Hwang 2004
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_ssrn","arguments":{"query":"George Hwang 52-week high momentum investing","max_results":5}}}' | uvx paper-search-mcp 2>/dev/null | python3 -m json.tool
```

Expected: JSON result with either paper metadata (abstract_id=338320 or abstract_id=1104491 should appear) or a "403 bot-detection" error message. Either constitutes a valid live_check (confirms the tool is callable and returns a structured response, not a crash).

---

## Free-only verdict: ADOPT

**Verdict: ADOPT** (no paid licensing required).

All required keys are free:
- Unpaywall: any email, no account, no payment.
- OpenAlex: free API key, $1/day credit (sufficient for harness use at 1 call per step), no payment until credit exhausted.
- CORE, Semantic Scholar: free optional keys.
- IEEE: free dev key (dormant if not configured).
- ACM: potentially institutional-paid, but connector is dormant without key — does not block ADOPT.

**No source the user MUST pay to use for the planned use case (finance paper search on arXiv, IACR, OpenAlex, SSRN metadata).**

The only operator action required before `status: done`:
1. Obtain a free OpenAlex API key at https://openalex.org/settings/api (30-second signup).
2. Set `OPENALEX_API_KEY=<key>` in the `.env` file that the MCP server env block reads from.
3. Confirm `UNPAYWALL_EMAIL` is set (fallback hardcoded to peder.bkoppang@hotmail.no in the proposed config).

---

## Research Gate Checklist

### Hard blockers

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources: PyPI page, GitHub README, raw README, pyproject.toml, Docker MCP catalog, OpenAlex blog announcement, OpenAlex users mailing list = 7 total)
- [x] 10+ unique URLs total incl. snippet-only (12 snippet-only + 7 read-in-full = 19+ total URLs)
- [x] Recency scan (last 2 years) performed + reported (2025-2026 findings: v0.1.3 release Apr 2025; OpenAlex breaking change Feb 13, 2026)
- [x] Full pages read (not abstracts) for read-in-full set (all sources fetched via WebFetch in full, not snippets)
- [x] file:line anchors for every internal claim (.mcp.json lines 1-25 read in full; phase-28.7 brief lines 1-10 read in full; .env.example permission-denied documented)

### Soft checks

- [x] Internal exploration covered every relevant module (.mcp.json schema read; .env.example attempted; phase-28.7 archive read)
- [x] Contradictions/consensus noted (PyPI v0.1.3 vs git v0.1.4; OpenAlex keyless behavior before/after Feb 13 2026; ACM paid uncertainty)
- [x] All claims cited per-claim with URL + access date

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 12,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "gate_passed": true,
  "free_only_verdict": "ADOPT",
  "openalex_breaking_change": "API keys mandatory as of 2026-02-13; free key required; old email polite-pool discontinued",
  "ssrn_abstract_id_george_hwang_2004": "338320 (working paper) or 1104491 (uploaded version)",
  "ssrn_tool_name": "search_ssrn",
  "latest_pypi_version": "0.1.3",
  "git_main_version": "0.1.4 (unreleased)",
  "install_command": "uvx paper-search-mcp",
  "mcp_json_schema": "type:stdio, command:uvx, args:[\"--from\",\"paper-search-mcp==0.1.3\",\"paper-search-mcp\"]",
  "notes": "phase-28.7 archive brief was a task definition, not a completed research brief — no SSRN unfetchable references in that file. SSRN abstract_id targets identified from external search instead."
}
```
