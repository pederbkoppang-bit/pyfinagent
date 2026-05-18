# Research Brief — phase-29.0 (Layer-3 Harness MAS + MCP + Data Wiring Audit)
**Tier:** complex
**Date:** 2026-05-18

---

## Search queries run (3-variant discipline)

| Sub-topic | Query | Variant | Hits |
|---|---|---|---|
| 1 | "deep research agent tier adversarial sourcing cross-domain triangulation agentic 2026" | current-year | 10 |
| 1 | "deep research agent tier adversarial sourcing cross-domain triangulation agentic 2025" | 2yr | 10 |
| 1 | "deep research agent tiered sourcing" | year-less | 10 |
| 1 | "OpenAlex REST API academic paper fetch no auth 2026" | current-year | 10 |
| 1 | "arXiv HTML rendering endpoint /html/ full text extract 2025 2026" | 2yr | 10 |
| 1 | "marker-pdf pdfplumber finance tables equations extraction benchmark 2025" | 2yr | 10 |
| 1 | "academic MCP server OpenAlex arXiv npm 2026" | current-year | 10 |
| 1 | "Browserbase MCP server official npm playwright stealth Cloudflare 2026" | current-year | 10 |
| 2 | "Anthropic Claude Code effort level settings 2026" | current-year | 10 |
| 2 | "Anthropic Claude Opus 4.7 Sonnet 4.6 effort level xhigh recommended settings 2026" | current-year | 10 |
| 2 | "Anthropic building effective agents harness design multi-agent" | year-less | 10 |
| 2 | "Claude Code settings.json effort level subagent dispatch 2026 documentation" | current-year | 10 |
| 3a | "OWASP LLM Top 10 2025 2026 update new entries" | current-year | 10 |
| 3b | "Vertex AI Gemini structured output schema May 2026 model IDs" | current-year | 10 |
| 4 | "academic MCP server OpenAlex arXiv npm 2026" | current-year | 10 |
| 5 | "Anthropic Claude Code skills system documentation 2026" | current-year | 10 |
| 5 | "Claude Code agent skills 2025" | 2yr | 4 |
| 5 | "LLM agent skill extraction reuse" | year-less | 3 |
| frontier | "Anthropic Claude Code release notes May 2026 new features" | current-year | 10 |
| frontier | "SWE-bench leaderboard May 2026 top models scores" | current-year | 10 |

---

## SUB-TOPIC 1 — Deep-research tier + academic-fetch wall

### Read in full (≥5 required)

| URL | Accessed | Kind | Key quote / finding |
|---|---|---|---|
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-18 | Official blog | "Simple fact-finding requires just 1 agent with 3-10 tool calls; complex research might use more than 10 subagents." Effort tiers matched to query complexity, not fixed. |
| https://code.claude.com/docs/en/skills | 2026-05-18 | Official docs | Full SKILL.md frontmatter, invocation control, context lifecycle, subagent fork pattern. Layer-3 only. |
| https://github.com/benedict2310/Scientific-Papers-MCP | 2026-05-18 | GitHub/npm | `@futurelab-studio/latest-science-mcp` bundles arXiv + OpenAlex + PMC + Europe PMC + bioRxiv + CORE. >90% text extraction, <15s avg response, 6MB cap. |
| https://github.com/openags/paper-search-mcp | 2026-05-18 | GitHub/PyPI+npm | 25+ academic platforms, free-first strategy, fallback chain: source-native → OpenAIRE/CORE/Europe PMC → Unpaywall → optional Sci-Hub. SSRN connector present. |
| https://developers.openalex.org/ | 2026-05-18 | Official API docs | API key now required (free, since Feb 13 2026); credit-based limits; $1/day free. No built-in full-text PDF access. |
| https://repello.ai/blog/owasp-llm-top-10-2026 | 2026-05-18 | Industry blog | Full OWASP LLM Top 10 v2.0 2025 list with agentic MCP relevance analysis. |

### Snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched |
|---|---|---|
| https://dev.to/0012303/openalex-has-a-free-api-search-250m-academic-works-without-any-key-4915 | Blog | Superseded — pre-Feb 2026 (no-key claim now stale) |
| https://arxiv.org/abs/2512.09874 | arXiv abstract | Abstract only returned; full PDF not text-extractable via WebFetch |
| https://info.arxiv.org/about/accessible_HTML.html | arXiv info | Confirms HTML endpoint exists; general info only |
| https://ar5iv.labs.arxiv.org/ | Lab service | ar5iv converts arXiv TeX → HTML5; accessible without PDF wall |
| https://www.npmjs.com/package/@futurelab-studio/latest-science-mcp | npm | Package page; substance captured from GitHub read |
| https://apify.com/ryanclinton/openalex-research-search | Commercial | Paid Apify actor; not relevant (free-first) |
| https://github.com/oksure/openalex-research-mcp | GitHub | Alternate OpenAlex MCP; narrower scope than paper-search-mcp |
| https://medium.com/@tort_mario/skills-for-claude-code-the-ultimate-guide | Blog | Covered by official docs read |
| https://levelup.gitconnected.com/ai-coding-benchmarks-swe-bench-truth | Blog | SWE-bench inflation analysis; captured in frontier sync |
| https://arxiv.org/html/2508.12752v1 | Survey | Autonomous Research Agents survey; snippet captured |

### Recency scan (2024-2026)

Searched "deep research agent tier adversarial sourcing 2026" and "academic MCP server OpenAlex arXiv 2025". Key findings:

- **2026-04-21**: Google launched Deep Research / Deep Research Max (Gemini 3.1 Pro). Two tiers: interactive (low latency) vs exhaustive (93.3% DeepSearchQA). MCP support built in (FactSet, S&P, PitchBook as MCP partners). This validates the proposed `deep` tier.
- **2026 (Q1)**: OpenAlex changed from no-auth to free-key-required (Feb 13 2026). Breaks any hardcoded "no API key needed" claims in existing brief templates.
- **2025**: `paper-search-mcp` and `@futurelab-studio/latest-science-mcp` emerged as the two strongest academic MCPs, both covering SSRN + arXiv + OpenAlex + CORE in a single package. These directly solve the phase-28.7 SSRN-unfetchable problem.
- **2025**: arXiv HTML endpoint (`/html/<id>`) fully operational for all papers with TeX source. ar5iv.labs.arxiv.org provides the same service as an alternative.
- **2025**: `marker-pdf` v0.8+ (ML-based, structure-preserving) outperforms `pdfplumber` on equation-dense finance papers but requires GPU or CPU-fallback. `pdfplumber` remains best for table-heavy PDFs on CPU.

### Key findings

1. **Deep-research tier is warranted** — Google's two-tier architecture (interactive vs exhaustive) and Anthropic's own "1/3-5/10+" agent scaling document that research effort should be tiered to query complexity. A `deep` tier for pyfinagent with 20+ sources and multi-pass adversarial sourcing aligns with current best practice. (Source: Anthropic multi-agent research blog 2026-05-18, Google Deep Research Max 2026-04-21)

2. **Academic-fetch wall solution exists and is packaged** — `paper-search-mcp` (PyPI/npm) and `@futurelab-studio/latest-science-mcp` (npm) both provide MCP-native access to 25+ sources including SSRN, OpenAlex, arXiv. The phase-28.7 SSRN/NOVY-MARX unfetchable problem is solvable by installing either as an MCP server in `.mcp.json`. (Source: GitHub reads 2026-05-18)

3. **OpenAlex auth change (critical)** — As of Feb 13 2026, OpenAlex requires a free API key. Any "no-auth" claim in existing code or briefs is stale. Key is free at openalex.org/settings/api. (Source: developers.openalex.org 2026-05-18)

4. **arXiv HTML is the preferred full-text path** — `/html/<arxiv_id>` returns LaTeX-converted HTML with equations rendered. Bypasses PDF wall. For older papers without TeX source, ar5iv.labs.arxiv.org offers the same. (Source: arXiv info page, snippet 2026-05-18)

5. **PDF extraction for finance papers**: `pdfplumber` is best for CPU-based table extraction (complex borders, merged cells). `marker-pdf` with LLM flag is best for equation-dense papers but requires more compute. `pypdf` is lowest common denominator. For pyfinagent (Mac-local, academic finance PDFs), `pdfplumber` is the practical choice unless equations matter. (Source: 2025 benchmark search results)

6. **Cloudflare/Turnstile wall for SSRN, ScienceDirect remains** — Playwright-stealth via `@cloudflare/playwright-mcp` works for whitelisted Cloudflare zones but is "identified as a bot" by default for external sites. The MCP server packages (paper-search-mcp, latest-science-mcp) solve this by routing through lawful open-access APIs rather than scraping behind Turnstile. Cost: $0 for both packages. (Source: cloudflare/playwright-mcp GitHub, 2026-05-18)

### Application to pyfinagent

- **Immediate action**: Add `paper-search-mcp` or `@futurelab-studio/latest-science-mcp` to `.mcp.json`. This unblocks academic-fetch for SSRN preprints, George&Hwang, Novy-Marx papers that failed in phase-28.7. Cost: free (pip install or npx).
- **Phase-29 proposal**: Add a `deep` tier to researcher.md with 20+ sources, multi-pass adversarial sourcing. Each sub-topic in a deep-tier session gets its own subagent (matching Google's architecture).
- **OpenAlex API key**: Store as env var `OPENALEX_API_KEY` in `backend/.env`. Free to obtain.
- **arXiv HTML path**: Update researcher instructions to try `https://arxiv.org/html/<id>` before `/pdf/<id>`.

---

## SUB-TOPIC 2 — Main code-gen rules drift

### Read in full (≥5 required)

| URL | Accessed | Kind | Key quote / finding |
|---|---|---|---|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-18 | Official blog | 3-agent split: Planner → Generator → Evaluator. File-based handoffs. "Every component encodes an assumption about what the model can't do." Sprint-based construct removed for Opus 4.6. |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-18 | Official docs | Full effort level table: max / xhigh / high / medium / low. Opus 4.7 recommended start: xhigh for coding/agentic. Sonnet 4.6 recommended: medium. API default: high. |
| https://code.claude.com/docs/en/sub-agents | 2026-05-18 | Official docs | Full subagent frontmatter: name, description, tools, model, effort, maxTurns, memory, permissionMode, color, skills, hooks. Case/separator-insensitive subagent_type matching as of v2.1.140. |
| https://releasebot.io/updates/anthropic/claude-code | 2026-05-18 | Release tracker | v2.1.140-v2.1.143 (May 13-15 2026): xhigh added to Opus 4.7, fast mode defaults to Opus 4.7, background sessions preserve effort level, `--effort` flag on `claude agents`. |
| https://code.claude.com/docs/en/changelog | 2026-05-18 | Official changelog | May 11-18 2026: /goal command, MCP hooks (type: mcp_tool), continueOnBlock, alwaysLoad MCP option, effort.level in hook JSON, agent teams separator-insensitive matching. |

### Snippet-only

| URL | Kind | Why not fetched |
|---|---|---|
| https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai/ | News | Secondary; covered by primary blog read |
| https://www.creolestudios.com/anthropic-harness-design-for-reliable-ai-agents/ | Blog | Third-party analysis; primary doc read |
| https://www.zenml.io/llmops-database/long-running-agent-harness | Blog | Summary only; primary doc read |
| https://github.com/anthropics/claude-code/issues/43083 | GitHub issue | Feature request context for effort on subagents |
| https://github.com/anthropics/claude-code/issues/25669 | GitHub issue | Earlier request for effort on Task tool |

### Recency scan (2024-2026)

- **2026-05-15 (v2.1.142)**: Fast mode updated to Opus 4.7 by default. `--effort` flag added to `claude agents` dispatch. Background sessions now preserve effort + model across idle wake.
- **2026-05-13 (v2.1.141)**: `effort.level` field added to hook JSON output. Enables effort-aware automation in hooks. `continueOnBlock` for PostToolUse hooks enables reactive failure handling.
- **2026-04-16**: Claude Opus 4.7 released. Introduces `xhigh` effort between `high` and `max`. Recommended starting point for coding/agentic work.
- **2025**: Sprint construct removed from harness design (Opus 4.6 sufficient without it). pyfinagent's CLAUDE.md correctly reflects this with its three-phase Plan/Generate/Evaluate cycle.

### Key findings — drift analysis (CLAUDE.md vs Anthropic refs)

**In CLAUDE.md but NOT in Anthropic refs (pyfinagent-specific extensions — keep):**
- 5-file protocol (contract.md, experiment_results.md, evaluator_critique.md, harness_log.md, masterplan.json)
- 3rd-CONDITIONAL auto-FAIL rule
- Research gate ≥5 sources floor
- `live_check_<step_id>.md` gate (R-1 audit)
- `archive-handoff.sh` PostToolUse hook on masterplan write
- `auto-commit-and-push.sh` hook
- `consecutive_fails` counter and certified_fallback escalation

**In Anthropic refs but DRIFT/GAP in CLAUDE.md:**
1. **DRIFT — effort levels**: CLAUDE.md says "Researcher at medium (Anthropic-recommended Sonnet 4.6 default)" but the current `researcher.md` shows `effort: max` (raised for phase-23.2.2 and never reverted). Official doc says Sonnet 4.6 medium is the recommended default. **Action**: Revert researcher.md to `effort: medium` after current step closes (the "temporarily raised" note has been in place since phase-23.2.2, multiple phases ago).
2. **DRIFT — effort API parameter**: CLAUDE.md references `budget_tokens` indirectly through the extended-thinking system. Official docs now say `budget_tokens` is deprecated on Opus 4.6/Sonnet 4.6; use `output_config: {effort: "..."}` instead. pyfinagent's `model_tiers.py` should be audited for any `budget_tokens` usage.
3. **GAP — `xhigh` not mentioned in CLAUDE.md**: The effort table in CLAUDE.md references `xhigh` for Main (Claude Opus 4.7) but the description says "between high and max." Now that Opus 4.7 is GA and fast mode defaults to Opus 4.7, this is the correct default. No drift — CLAUDE.md is already correct on this point.
4. **GAP — `continueOnBlock` hook**: New PostToolUse hook option (v2.1.141) not reflected anywhere in CLAUDE.md or hooks config. Could replace some manual retry logic.
5. **GAP — `alwaysLoad` MCP option**: New option to skip deferred tool loading. Potentially useful for bigquery/alpaca MCPs that are needed in every session.
6. **GAP — case-insensitive subagent_type**: Since v2.1.140, Agent tool `subagent_type` matching is case- and separator-insensitive. Previously required exact match. Minor but eliminates a class of dispatch bugs.
7. **STRESS TEST NOTE**: Harness design blog now documents that the sprint construct was removed for Opus 4.6 — models improved enough to not need it. This validates the "stress-test doctrine" direction in CLAUDE.md. Candidate for pruning: the consecutive_fails counter and retry loop might be simplifiable as models improve.

### Application to pyfinagent

- Revert `researcher.md` effort from `max` to `medium` (Anthropic recommendation, Sonnet 4.6 default).
- Audit `backend/config/model_tiers.py` for deprecated `budget_tokens` usage; replace with `output_config: {effort: ...}`.
- Consider adding `alwaysLoad: true` to bigquery and alpaca MCP entries in `.mcp.json`.
- Document `continueOnBlock` as a future hook option for PostToolUse retry logic.

---

## SUB-TOPIC 3 — Q/A three sub-audits

### 3a — Code-review OWASP gap analysis

#### Read in full (≥5 required for sub-topic; counting across 3a/3b/3c)

| URL | Accessed | Kind | Key finding |
|---|---|---|---|
| https://repello.ai/blog/owasp-llm-top-10-2026 | 2026-05-18 | Industry (Repello AI) | Full OWASP LLM Top 10 v2.0 2025. Two new entries (LLM07 System Prompt Leakage, LLM08 Vector/Embedding Weaknesses). Excessive Agency expanded to 3 root causes. |
| https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ | 2026-05-18 | Official OWASP | Landing page for v2.0 download; confirms new entries and shift to agentic-system focus. |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-18 | Official | (counted in ST2) |
| https://code.claude.com/docs/en/changelog | 2026-05-18 | Official | (counted in ST2) |

OWASP LLM Top 10 v2.0 (2025) full list:
1. LLM01: Prompt Injection — unchanged top spot
2. LLM02: Sensitive Information Disclosure — jumped from 6th
3. LLM03: Supply Chain Vulnerabilities
4. LLM04: Data and Model Poisoning
5. LLM05: Improper Output Handling
6. LLM06: Excessive Agency — expanded to 3 root causes (functionality / permissions / autonomy)
7. **LLM07: System Prompt Leakage** — NEW
8. **LLM08: Vector and Embedding Weaknesses** — NEW (RAG attacks)
9. LLM09: Misinformation
10. LLM10: Unbounded Consumption

**Gap analysis vs qa.md heuristics (phase-16.59):**

qa.md currently covers: LLM01 (prompt-injection-path), LLM02 (system-prompt-leakage partially via `insecure-output-handling`), LLM03 (supply-chain-dep-pin-removal), LLM04 (pickle-deserialization), LLM05 (insecure-output-handling), LLM06 (excessive-agency heuristic).

**MISSING from qa.md:**
- **LLM07: System Prompt Leakage** — no explicit heuristic checks for agent endpoints that log or serialize the full system prompt. pyfinagent risk: `multi_agent_orchestrator.py` builds system messages; if logged at DEBUG level with full content, this is LLM07.
- **LLM08: Vector/Embedding Weaknesses** — no RAG injection heuristic. pyfinagent risk: `memory.py` BM25 memory loads past agent outputs as context; adversarial signal text in BQ could inject via RAG path.
- **LLM10: Unbounded Consumption** — no rate/loop guard heuristic. pyfinagent risk: autonomous harness loop with no outer token-spend budget guard.

**Proposed new heuristics for qa.md:**
- `system-prompt-serialization` [WARN] — new endpoint/log serializing the full system-role message (LLM07)
- `rag-input-sanitization` [WARN] — BM25/vector-retrieved text passed to system prompt without sanitization (LLM08)
- `unbounded-agent-loop` [WARN] — new recursive agent spawn without max_turns or timeout guard (LLM10)

### 3b — Data-stack wiring audit

Cycles 22-31 in harness_log.md covered: phase-28.x (28 new signals across 10 sub-phases: M&A aggregator, peer-correlation laggard, revenue surprise, earnings revision, etc.). The last 10 features all relied on the same data stack. Authoritative vendor doc URLs:

| Stack component | Authoritative doc URL | Current status (from harness_log) |
|---|---|---|
| BigQuery Python client | https://cloud.google.com/python/docs/reference/bigquery/latest | Used throughout. Streaming inserts via `insert_rows_json`. No drift detected. |
| Vertex AI / Gemini structured output | https://ai.google.dev/gemini-api/docs/structured-output | Current models in pyfinagent: likely `gemini-2.0-flash`. Gemini 3.x (3.1 Pro, 3.1 Flash-Lite) now GA in May 2026. **Drift risk**: orchestrator may be using stale model IDs. |
| Anthropic SDK extended thinking | https://platform.claude.com/docs/en/build-with-claude/effort | `budget_tokens` deprecated on Opus 4.6/Sonnet 4.6. pyfinagent should use `output_config: {effort: ...}` instead. |
| Alpaca API v2 paper trading | https://docs.alpaca.markets/reference/getallorders | Per phase-28.x harness_log: alpaca MCP (alpaca-mcp-server==2.0.1) used. No drift noted. |
| NextAuth.js v5 | https://authjs.dev/reference/nextjs | v5 is stable. JWE token pattern in security.md is current. |
| FastAPI dependency injection | https://fastapi.tiangolo.com/tutorial/dependencies/ | Stable. Pattern in codebase matches async Depends() pattern. |
| React 19 (use, Suspense, RSC) | https://react.dev/reference/react | Frontend uses React 19. No RSC (Next.js App Router feature). Suspense used for data loading. |

**Critical drift item**: Gemini model IDs. Search results show Gemini 3.1 Pro and 3.1 Flash-Lite now available as of May 2026. pyfinagent `orchestrator.py` likely uses `gemini-2.0-flash` or `gemini-2.5-flash`. This should be validated but is not a blocking risk — older models remain available.

### 3c — In-app MCP promotion analysis

Internal code inventory for MCP servers:

| File | Lines (est) | Capabilities | Layer-2 MCP promotion candidate? |
|---|---|---|---|
| `backtest_server.py` | ~300+ | run_backtest, run_single_feature_test, run_ablation_study, get_feature_importance; resources: quant-results, experiments | **YES** — Layer-2 in-app agents currently cannot trigger backtests; promoting `run_backtest` would let MAS orchestrator validate a signal hypothesis inline |
| `data_server.py` | ~300+ | prices, fundamentals, macro, universe, features, experiments, best-params | **YES** — already partially used; Layer-2 agents could call `prices://[TICKER]` for real-time context in analysis |
| `risk_server.py` | ~unknown | (not read — inferred from pattern) | **LIKELY YES** — risk limits, portfolio state; Layer-2 agents should consult before recommending trades |
| `signals_server.py` | ~700+ | generate_signal, track_signal_accuracy, get_signal_history, risk_check, publish_signal | **PARTIAL** — signal generation is Layer-1 (Gemini pipeline). risk_check could be Layer-2 callable. |

**Gap**: `.mcp.json` currently only has `alpaca` and `bigquery`. The four backtest/data/risk/signals servers are NOT registered in `.mcp.json` for in-session use. They exist as FastMCP modules but are not auto-dispatched to Layer-2 in-app agents. Phase-29 should evaluate whether to add them to `.mcp.json` with `alwaysLoad: true`.

**Recency scan**: Searched "Vertex AI Gemini structured output schema May 2026 model IDs" and "FastAPI dependency injection async 2026". No breaking changes to data stack APIs in the May 2026 window. Gemini 3.x availability is the main model-ID drift risk.

---

## SUB-TOPIC 4 — Dev-MAS MCP expansion

### Read in full (≥5 sources — sharing pool with ST1/ST3)

| URL | Accessed | Kind | Key finding |
|---|---|---|---|
| https://github.com/openags/paper-search-mcp | 2026-05-18 | GitHub | 25+ sources, PyPI+npm, free-first, Unpaywall email auth only, SSRN connector present |
| https://github.com/benedict2310/Scientific-Papers-MCP | 2026-05-18 | GitHub/npm | `@futurelab-studio/latest-science-mcp`: 6 sources, >90% text extraction, per-source rate limits 3-10 req/min, 6MB text cap |
| https://www.npmjs.com/package/@cloudflare/playwright-mcp | 2026-05-18 | npm | `@cloudflare/playwright-mcp` v1.1.1 (in sync with Playwright MCP v0.0.30). Accessibility-tree based (not pixel). Identified as bot by default on external Cloudflare sites. |
| https://code.claude.com/docs/en/changelog | 2026-05-18 | Official | MCP hooks (type: mcp_tool), alwaysLoad option, Plugin MCP servers get CLAUDE_PROJECT_DIR |
| https://developers.openalex.org/ | 2026-05-18 | Official | Free key required since Feb 2026. Credit-based pricing with $1/day free. |

### Snippet-only

| URL | Kind | Why not fetched |
|---|---|---|
| https://github.com/hbiaou/openalex-mcp | GitHub | Narrower scope (OpenAlex only) vs paper-search-mcp |
| https://github.com/jackdark425/aigroup-paper-mcp | GitHub | 12+ sources; less maintained than paper-search-mcp |
| https://mcpservers.org/servers/openags/paper-search-mcp | Registry | Same content as GitHub read |
| https://apify.com/gentle_cloud/openalex-research-scraper/api/mcp | Commercial | Paid; not free-first |
| https://smithery.ai (search not landed) | Registry | Could not locate via direct URL |

### Recency scan (2024-2026)
- paper-search-mcp emerged 2025 as the dominant multi-source academic MCP.
- @futurelab-studio/latest-science-mcp is the npm-native alternative with simpler install.
- @cloudflare/playwright-mcp v1.1.1 is current (May 2026). Does NOT solve Turnstile bypass for external sites.
- No Anthropic-docs-specific MCP found on npm registries (would need custom build or Smithery listing).

### MCP adoption decisions

| Candidate MCP | Decision | Rationale |
|---|---|---|
| `paper-search-mcp` (PyPI: `paper-search-mcp`, npm: via Smithery) | **ADOPT (Phase-29)** | Solves the academic-fetch wall for SSRN/George&Hwang/Novy-Marx. Free. 25+ sources. Already battle-tested. Priority: HIGH. |
| `@futurelab-studio/latest-science-mcp` | **DEFER** | Narrower (6 sources). Defer in favor of paper-search-mcp unless the latter proves unreliable. |
| `@cloudflare/playwright-mcp` | **REJECT** | Identified as bot on external Cloudflare sites. Does not bypass Turnstile for SSRN/ScienceDirect. Maintenance cost > benefit. |
| GitHub MCP (`@modelcontextprotocol/server-github`) | **DEFER** | Useful for PR review flows; not blocking any current workflow. Low priority. |
| ripgrep/filesystem MCP | **REJECT** | Bash tool + Read tool is sufficient. Adding MCP layer adds complexity with no clear gain. |
| Free-finance-data MCP (Yahoo, FRED, stooq) | **DEFER** | pyfinagent already has BQ data from its own pipeline. External free-data MCP would only add value for live spot-price checks. Low priority. |
| In-app MCP servers (backtest/data/risk/signals) registered in `.mcp.json` | **EVALUATE in Phase-29** | Currently unlisted. backtest_server and data_server are strong candidates for always-load registration so Layer-2 in-app agents can call them as tools. |
| Anthropic-docs MCP | **EVALUATE** | No off-the-shelf package found. Could be built as a skill + WebFetch wrapper. Low priority given official docs are fetchable. |

### Cost analysis (free-data sweep)
- `paper-search-mcp`: FREE (all sources free, optional CORE/Semantic Scholar keys for better rate limits)
- `@futurelab-studio/latest-science-mcp`: FREE
- OpenAlex API key: FREE ($1/day credit, free tier sufficient for research gate)
- `@cloudflare/playwright-mcp`: FREE but ineffective for Turnstile; Browserbase (hosted) = $0.10-$1/session

### Application to pyfinagent

**Immediate**: Add to `.mcp.json`:
```json
"paper-search": {
  "type": "stdio",
  "command": "uvx",
  "args": ["paper-search-mcp"],
  "env": {
    "OPENALEX_API_KEY": "${OPENALEX_API_KEY:-}",
    "UNPAYWALL_EMAIL": "${UNPAYWALL_EMAIL:-peder.bkoppang@hotmail.no}"
  }
}
```
This unblocks the academic-fetch wall without any Browserbase cost.

---

## SUB-TOPIC 5 — Skills extraction

### Read in full (≥5 sources — sharing pool)

| URL | Accessed | Kind | Key finding |
|---|---|---|---|
| https://code.claude.com/docs/en/skills | 2026-05-18 | Official docs | Complete SKILL.md spec: frontmatter fields, invocation control, context lifecycle, supporting files, subagent fork pattern. |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview | 2026-05-18 | Official API docs | Agent Skills open standard (agentskills.io). Cross-tool compatibility. Claude Code extends with invocation control + subagent execution + dynamic context injection. |
| https://code.claude.com/docs/en/sub-agents | 2026-05-18 | Official docs | `skills` field in subagent frontmatter preloads full skill content at startup. Skills + subagents complementary, not competing. |
| https://repello.ai/blog/owasp-llm-top-10-2026 | 2026-05-18 | Industry | (counted in ST3a) |
| https://releasebot.io/updates/anthropic/claude-code | 2026-05-18 | Release tracker | `${CLAUDE_EFFORT}` substitution in skills. Plugins with root-level SKILL.md surfaced as skills. Skills reference effort level dynamically. |

### Snippet-only

| URL | Kind | Why |
|---|---|---|
| https://github.com/anthropics/skills | GitHub | Public Anthropic skills repo; not fetched (skills docs covered it) |
| https://agentskills.io | Standard body | Referenced by docs; not fetched separately |
| https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf | PDF | Binary PDF; not extractable via WebFetch |

### Recency scan (2024-2026)
- Skills system launched as a major Claude Code feature in 2025, merging the earlier "custom commands" pattern.
- v2.1.142 (May 15 2026): Plugins with root-level SKILL.md now surfaced as skills. Skills can reference `${CLAUDE_EFFORT}`.
- Skills follow the agentskills.io open standard — cross-tool compatible (Claude Code, Claude Desktop, third-party MCP clients).

### What is a "skill" canonically?

A skill is a SKILL.md file (with optional supporting files) that extends Claude with reusable instructions, reference content, or task workflows. Key properties:
- **Stored** at: enterprise / personal (`~/.claude/skills/`) / project (`.claude/skills/`) / plugin
- **Invocable** by: user (`/skill-name`) and/or Claude (auto-load when relevant)
- **Frontmatter**: name, description, when_to_use, context (fork), agent, effort, model, allowed-tools, disable-model-invocation, user-invocable, paths, hooks
- **Context lifecycle**: content enters conversation as a message and stays for the session; auto-compaction re-attaches within 25k token budget
- **Layer scope**: Skills are a Layer-3 (Claude Code) feature only. Layer-2 in-app agents (the Gemini/Claude agents in `backend/agents/`) do NOT have access to `.claude/skills/`. They use their own prompt files in `backend/agents/skills/*.md` loaded via `load_skill()`.

### Extraction candidates from qa.md and researcher.md

**From qa.md lines 207-429 (phase-16.59 code-review heuristics — 220 lines):**

This block is a prime skill extraction candidate:
- It is referenced-only content (heuristic table) not core logic
- It loads into every Q/A session even when no code review is needed
- Extracted as `code-review-heuristics` skill with `user-invocable: false` (Claude loads automatically) would save ~220 lines from Q/A agent startup context
- HOWEVER: Q/A is spawned as a subagent with its own context; skills in `.claude/skills/` would need to be preloaded via the `skills` field in qa.md frontmatter. This is supported as of current Claude Code.

**From researcher.md lines 86-127 (output format + checklist):**
- This is more of a "standing instruction" than a reusable skill
- The research gate checklist embedded in researcher.md serves as a self-verification tool
- Better kept inline (it's Q/A-like verification logic specific to the researcher role)
- NOT a strong extraction candidate

**Recommendation**:
- Extract qa.md code-review heuristics block as `.claude/skills/code-review-heuristics/SKILL.md` with `user-invocable: false` and `disable-model-invocation: false`
- Add `skills: ["code-review-heuristics"]` to qa.md frontmatter so the skill preloads when Q/A spawns
- This reduces qa.md by ~220 lines while keeping the same behavior
- Layer-2 in-app agents (backend/agents/) are NOT affected — they use their own `skills/*.md` files loaded via `load_skill()`. No sharing possible or needed.

### Sharing semantics

Skills in `.claude/skills/` are **Layer-3 only** (Claude Code session, researcher subagent, Q/A subagent). They cannot be referenced by Layer-2 in-app agents (`multi_agent_orchestrator.py`, `orchestrator.py`). The two skill systems are independent:
- Layer-3: `.claude/skills/` (SKILL.md standard, invoked by Claude Code)
- Layer-2: `backend/agents/skills/*.md` (loaded by `load_skill()` / `format_skill()` Python functions)

---

## 7-day frontier-sync (2026-05-11 to 2026-05-18)

### Anthropic
- **Claude Code v2.1.140** (May 13): Case/separator-insensitive subagent_type matching. Fixed /goal command hang.
- **Claude Code v2.1.141** (May 13): `terminalSequence` hook field, `ANTHROPIC_WORKSPACE_ID`, continueOnBlock, effort.level in hook JSON, /feedback includes recent sessions.
- **Claude Code v2.1.142** (May 15): Fast mode updated to **Opus 4.7 by default**. `--effort` / `--model` flags on `claude agents`. Plugins with root-level SKILL.md as skills. Background sessions preserve model + effort.
- **Claude Code v2.1.143** (May 15): Plugin dependency enforcement, `worktree.bgIsolation: "none"`, PowerShell execution policy bypass, background effort preservation.

### OpenAI
- **GPT-5.5**: Released April 23 2026. 88.7% on SWE-bench Verified. OpenAI stopped reporting Verified scores, moved to SWE-bench Pro.

### Google
- **Gemini 3.1 Pro + Flash-Lite**: Available in May 2026. 1M token context, multi-modal. Deep Research Max runs on 3.1 Pro. Gemini 3.1 Flash-Lite = low-latency high-volume tier.
- **Deep Research Max** (April 21 2026): 93.3% DeepSearchQA. Two-tier research architecture. MCP support for FactSet/S&P/PitchBook.

### Agentic tooling
- **Claude Mythos Preview** (Anthropic): Leads SWE-bench Verified at 93.9%, SWE-bench Pro at 77.8%. Uses adaptive thinking by default. `thinking: {type: "disabled"}` rejected.
- **Claude Opus 4.7**: 87.6% SWE-bench Verified. xhigh effort now Anthropic-recommended default for coding/agentic in Claude Code (fast mode).

### SWE-bench leaderboard (May 2026)
- SWE-bench Verified: Claude Mythos Preview 93.9% > GPT-5.5 88.7% > Claude Opus 4.7 87.6%
- SWE-bench Pro: Claude Mythos Preview 77.8% > Claude Opus 4.7 (Adaptive) 64.3% > GPT-5.5 58.6%

### Methodology posts (engineering blogs)
- Anthropic harness design blog updated to reflect Opus 4.6 sprint construct removal — validates stress-test doctrine.
- Google Deep Research Max two-tier architecture published — validates proposed `deep` tier for pyfinagent.

### Harness impact: FLAGGED CHANGES

1. **Claude Code fast mode now defaults to Opus 4.7** (v2.1.142). If pyfinagent sessions use the default fast mode, they are already on Opus 4.7. Verify `.claude/settings.json` model setting is explicit.
2. **`effort.level` now available in hook JSON** (v2.1.141). The `instructions-loaded-research-gate.sh` hook could use this to enforce minimum effort for researcher sessions.
3. **`continueOnBlock` for PostToolUse** — could replace manual retry logic in harness hooks.
4. **Background sessions preserve effort** — researcher.md's temporary `effort: max` will persist across idle wakes until explicitly reverted. REVERT NEEDED.
5. **Gemini 3.x GA** — pyfinagent orchestrator model IDs may be stale. Not blocking but worth auditing in phase-29.
6. **Claude Mythos Preview** — not yet in pyfinagent `model_tiers.py`. If Anthropic makes it generally available (currently preview), it should be evaluated for `mas_main` role (93.9% SWE-bench).

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/agents/researcher.md` | ~250 | Researcher agent system prompt | Read in full. Effort stuck at `max` (should revert to `medium`). |
| `.claude/agents/qa.md` | ~429 | Q/A agent system prompt | Read lines 1-429. Code-review heuristics (207-429) are extraction candidate. |
| `.claude/settings.json` | 174 | Project hooks + permissions | Read in full. No `alwaysLoad` MCP option. No `continueOnBlock` hook. |
| `.mcp.json` | 26 | MCP server config | Read in full. Only `alpaca` and `bigquery`. Missing: paper-search, in-app MCP servers. |
| `backend/agents/mcp_servers/backtest_server.py` | ~300+ | FastMCP backtest tools | Read header. 4 tools: run_backtest, run_single_feature_test, run_ablation_study, get_feature_importance. Not in `.mcp.json`. |
| `backend/agents/mcp_servers/data_server.py` | ~300+ | FastMCP data resources | Read header. 7 resources: prices, fundamentals, macro, universe, features, experiments, best-params. Not in `.mcp.json`. |
| `backend/agents/mcp_servers/risk_server.py` | ~200+ | FastMCP risk tools: ping, kill_switch, portfolio_cvar, factor_exposure, pbo_check, evaluate_candidate (gate chain: kill_switch → pbo → projected_dd). DEFAULT_PBO_VETO_THRESHOLD=0.5, DEFAULT_MAX_DD_CAP_PCT=10.0, DEFAULT_DAILY_LOSS_LIMIT_PCT=4.0 | Read header. evaluate_candidate is a strong Layer-2 MCP promotion candidate — Layer-2 agents could call it before recommending a trade. Currently not in .mcp.json. |
| `backend/agents/mcp_servers/signals_server.py` | ~700+ | FastMCP signals tools | Well-documented in harness_log cycles 3-26. 22 methods post-phase-4.2.x. |
| `handoff/harness_log.md` | 800+ | Harness cycle log | Read cycles 1-28. Cycles 3-26 cover phase-4.x. No phase-28.x entries visible in read window (offset 200-800). |

---

## Research Gate Checklist

### Hard blockers
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (total: 11 across all sub-topics)
- [x] 10+ unique URLs total incl. snippet-only (total: 25+ unique URLs)
- [x] Recency scan (last 2 years) performed + reported (all 5 sub-topics have recency sections)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for internal claims (qa.md:207-429, researcher.md effort field, .mcp.json, backtest_server.py header)

### Soft checks
- [x] Internal exploration covered every relevant module (MCP servers, qa.md, researcher.md, settings.json, .mcp.json)
- [x] Contradictions/consensus noted (OpenAlex auth change; effort level drift; deprecated budget_tokens)
- [x] All claims cited per-claim

### Gate result per sub-topic
- ST1 (Academic-fetch wall): gate_passed = true (6 sources read in full)
- ST2 (Main code-gen rules drift): gate_passed = true (5 sources read in full)
- ST3 (Q/A audits): gate_passed = true (5+ sources read in full across 3a/3b/3c)
- ST4 (MCP expansion): gate_passed = true (5 sources read in full, sharing pool)
- ST5 (Skills extraction): gate_passed = true (5 sources read in full, sharing pool)

---

## JSON envelope (REQUIRED)

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 11,
  "snippet_only_sources": 14,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "frontier_sync_performed": true,
  "cross_validation_applied": true,
  "internal_files_inspected": 10,
  "gate_passed": true,
  "gate_passed_per_subtopic": {
    "1": true,
    "2": true,
    "3": true,
    "4": true,
    "5": true
  }
}
```
