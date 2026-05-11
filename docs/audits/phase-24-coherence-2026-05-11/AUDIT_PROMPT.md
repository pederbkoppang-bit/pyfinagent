# phase-24 Audit Prompt — Researcher Brief Template

**Purpose.** This file is the single source of truth for the "comprehensive
audit prompt" that owner asked for in phase-24. Future Main sessions
spawn the `researcher` subagent with the relevant section below as the
prompt body, substituting `<vendor>` and `<callsite_list>` as noted.

The brief is intentionally identical across vendors so cross-vendor
comparisons are apples-to-apples. The research-gate floor (5 sources
read in full via WebFetch, three-query-variant discipline, mandatory
last-2-year recency scan) applies to every section per
`.claude/rules/research-gate.md`.

---

## Section A — Vendor alignment audit (used by steps 24.8, 24.9, 24.10)

> You are the Researcher subagent. Audit the pyfinagent integration with
> **<vendor>** against current **<vendor>** documentation.
>
> **Floor (mandatory).** Fetch and read in full at least 5 sources via
> `WebFetch`. Search snippets do NOT count. Source-quality hierarchy:
> peer-reviewed > official docs > authoritative blogs > practitioner >
> community.
>
> **Search-query discipline (mandatory).** Run at least three query
> variants per topic: current-year frontier (append `2026`), last-2-year
> window (`2025`/`2024`), and year-less canonical (bare topic). Surface
> all three in the brief.
>
> **Recency scan (mandatory).** Dedicated "Recency scan (last 2 years)"
> section reporting N new findings or "no relevant new findings".
>
> **Internal-codebase leg.** Walk the file:line callsite list provided
> by Main below. Grep for the vendor SDK import, list every callsite,
> and assess each against the canonical pattern from the docs.
>
> **Callsite list (supplied by Main).** `<callsite_list>` —
> e.g., `backend/agents/llm_client.py:692-1050`,
> `backend/agents/orchestrator.py:<lines>`,
> `backend/agents/multi_agent_orchestrator.py:<lines>`,
> `backend/agents/cost_tracker.py:26-35`.
>
> **Feature checklist (for `<vendor>` = Anthropic).** Score each
> callsite × column:
> - Model IDs current? (Opus 4.7 `claude-opus-4-7`, Sonnet 4.6
>   `claude-sonnet-4-6`, Haiku 4.5 `claude-haiku-4-5-20251001`; no
>   `claude-opus-4`, `claude-sonnet-4`, `claude-opus-4-1` in runtime
>   paths).
> - Prompt caching enabled where system prompts >2-4K tokens?
>   (`cache_control: {type: ephemeral, ttl: 1h}`).
> - Extended thinking used where reasoning would improve output?
>   (note: temp=1 required when thinking active).
> - Tool-use loops well-formed (max iterations, `tool_choice`,
>   error-on-stop-reason handling)?
> - Files API used for re-read documents (SEC 10-Ks)?
> - Batch API used for async-tolerant calls (sentiment scoring)?
> - Citations enabled where appropriate (note: incompatible with
>   structured output).
> - Structured output (`response_format.type=json_schema`) used where
>   schema is known?
>
> **Feature checklist (for `<vendor>` = Google Gemini).** Score each
> callsite × column:
> - Model IDs current? (no `gemini-1.5-pro` in production paths).
> - Google Search Grounding configured on Gemini-locked agents?
> - Structured output via `response_schema` consistent across the 28
>   skill prompts?
> - Token budgets match the CLAUDE.md spec (Enrichment 1024, Debate
>   1536, Synthesis 4096)?
> - Function-calling idiom used correctly per the latest Vertex docs?
> - Safety filter overrides documented where applied?
>
> **Output format (mandatory).** Markdown table — one row per callsite,
> one column per feature, value is `green` / `yellow` / `red`. Red
> rows require a follow-up ticket id in `notes`. End with:
> ```json
> {"tier":"complex","external_sources_read_in_full":N,
>  "snippet_only_sources":N,"urls_collected":N,
>  "recency_scan_performed":true,"internal_files_inspected":N,
>  "gate_passed":true}
> ```
> `gate_passed: true` iff `external_sources_read_in_full >= 5` AND
> `recency_scan_performed == true` AND every red row has a ticket id.

---

## Section B — MCP inventory audit (used by step 24.11)

> You are the Researcher subagent. Audit the pyfinagent Model Context
> Protocol (MCP) configuration and propose a final inventory.
>
> **Floor + query discipline + recency scan**: same as Section A.
>
> **Inputs.**
> - `.mcp.json` — currently lists `alpaca` (`alpaca-mcp-server==2.0.1`)
>   and `bigquery` (`mcp-server-bigquery==0.3.2`).
> - `.claude/settings.json` deny-lists for `mcp__alpaca__*` write
>   actions and `mcp__bigquery__execute-query`.
> - Existing Python clients to compare against:
>   - `backend/alt_data/features.py` — Polygon, yfinance, AlphaVantage
>   - `backend/alt_data/news_adapters.py` — Finnhub, Benzinga, Alpaca
>     news
>   - `backend/agents/llm_client.py` — multi-provider LLM routing
>
> **Candidates to evaluate** (one row per candidate in the output):
> Polygon MCP, Finnhub MCP, NewsAPI MCP, fetch MCP, sequential-thinking
> MCP, git MCP, filesystem MCP, Brave-search MCP, time/calendar MCP,
> SQLite/Postgres MCP, Puppeteer/Playwright MCP, plus any new MCPs
> surfaced by the literature scan.
>
> **For each candidate, decide.** Add / Defer / Reject. Cite the
> duplicating Python module (`backend/alt_data/features.py:<line>` etc.)
> when rejecting. Cite the marginal value when proposing to add.
>
> **Output format (mandatory).** Two markdown tables:
> 1. **Attached** — `alpaca`, `bigquery`, plus any newly added — with
>    a one-line "last successful tool call" entry.
> 2. **Rejected / deferred candidates** — one row per candidate:
>    name, what-it-would-do, duplicating-Python-module-file-line,
>    decision, rationale.
>
> Conclude with: "Final phase-24 recommendation: <no additions | add X,
> Y, Z with rationale>."
>
> End with the same JSON envelope as Section A.

---

## Section C — Internal code audit (used by every other step)

> You are the Researcher subagent. The internal-codebase leg of your
> dual mandate (external literature + internal code) requires you to
> walk the file:line list Main provides, read each file's relevant
> excerpts, and answer the four questions below.
>
> **For each callsite Main supplies, answer.**
> 1. What does this code do today? (one paragraph)
> 2. What invariant must hold? (one bullet list)
> 3. Where would a change to this code propagate? (one bullet list of
>    importers + their files)
> 4. Is there an existing utility we should reuse before writing new
>    code? (cite file:line)
>
> **Floor.** At least 5 internal files inspected with `Read` (not
> Grep). Reading-via-grep does not count.
>
> **Output format.** One subsection per callsite. End with the JSON
> envelope from Section A (with `internal_files_inspected` matching
> the file count).

---

## How Main uses this file

1. At the start of each phase-24 step's PLAN phase, Main spawns
   `researcher` with **one of** Section A / B / C as the prompt body
   (Section A for steps 24.8/24.9/24.10, Section B for 24.11,
   Section C for the others — most steps need Section C plus a
   topic-specific brief). The substitution table:

   | Step | Section | `<vendor>` | `<callsite_list>` source |
   |---|---|---|---|
   | 24.0 / 24.1 | C | n/a | snapshot.json paths |
   | 24.2 | C | n/a | autonomous_loop.py + paper_trader.py |
   | 24.3 | C | n/a | portfolio_manager.py + settings.py |
   | 24.4 | C | n/a | autonomous_loop.py:561-616 |
   | 24.5 | C | n/a | autonomous_loop.py step 5.5 + risk_limits |
   | 24.5b | A + C | Anthropic | cost_tracker.py, orchestrator.py, MAS |
   | 24.6 / 24.7 | C | n/a | main.py lifespan + cron files |
   | 24.8 | A (both vendors) | Anthropic + Google | skills/*.md |
   | 24.9 | A | Anthropic | all `anthropic.Anthropic` callsites |
   | 24.10 | A | Google | all Vertex/Gemini callsites |
   | 24.11 | B | n/a | .mcp.json + alt_data clients |
   | 24.12 | C | n/a | frontend/src/app/* + api.ts |
   | 24.13 | C | n/a | all newly-touched files |

2. The researcher's output is filed at
   `handoff/current/research_brief.md`. The archive hook snapshots it
   to `handoff/archive/phase-24.<step>/research_brief.md` on step
   completion.

3. The step's `contract.md` cites the brief by file path; the Q/A
   subagent's LLM-judgment leg checks for that citation per
   `.claude/agents/qa.md`.

---

## North-star alignment line

Every researcher brief must end with a "North-star line" — one
sentence stating how the finding shifts **Net System Alpha = Profit -
(Risk Exposure + Compute Burn)**. Example: "Wiring stop-loss execution
reduces tail-risk drawdown by capping per-position loss at the
operator's configured percent; expected impact: reduces realized
Risk Exposure term, no change to Profit or Compute Burn."

This line is what closes the loop between the academic / vendor-doc
literature scan and the owner's stated goal. No brief is `gate_passed:
true` without it.
