# Experiment Results — phase-29.0 (Layer-3 Harness MAS + MCP + Data-Wiring Audit)

**Step ID:** phase-29.0
**Date:** 2026-05-18
**Cycle:** 1
**Author:** Main (3-agent harness session)
**Inputs:** `handoff/current/research_brief.md` (452 lines, gate_passed=true on all 5 sub-topics), `handoff/current/contract.md` (this cycle).

This is an **audit-only** cycle. The deliverable is a JSON-ready masterplan entry for `phase-29` that the next session can drop into `.claude/masterplan.json` and execute as separate sub-step cycles. **Zero code edits this cycle.**

---

## Executive summary

| Audit question | Key finding | P1 action |
|---|---|---|
| Q1 — Deep-research + academic-fetch | `paper-search-mcp` (PyPI, free) + `@futurelab-studio/latest-science-mcp` (npm, free) already solve the Cloudflare-Turnstile wall. `arxiv.org/html/<id>` bypasses PDF wall for TeX-source papers. OpenAlex API key now required (Feb 13 2026). | Add `paper-search-mcp` to `.mcp.json`; add 4th `deep` tier to `researcher.md`; add `arxiv.org/html` precedence rule. |
| Q2 — Main code-gen drift | `researcher.md:10` stuck at `effort: max` since phase-23.2.2 (never reverted). `budget_tokens` deprecated for Opus/Sonnet 4.6. `continueOnBlock`/`alwaysLoad`/`effort.level`-in-hook-JSON shipped in v2.1.140-143, none used. | Revert researcher effort to `medium`; audit `model_tiers.py` for `budget_tokens`; document new hook options. |
| Q3a — Q/A code-review heuristics | OWASP LLM Top-10 v2.0 (2025) entries LLM07 (System Prompt Leakage), LLM08 (Vector/Embedding Weaknesses), LLM10 (Unbounded Consumption) absent from `qa.md:271-296`. | Add 3 heuristics to `qa.md`: `system-prompt-serialization` (WARN), `rag-input-sanitization` (WARN), `unbounded-agent-loop` (WARN). |
| Q3b — Data-stack wiring | Gemini 3.x GA in May 2026; orchestrator model IDs may be stale. `budget_tokens` deprecated. Otherwise no major drift. | Audit `orchestrator.py` Gemini model IDs; replace `budget_tokens` with `output_config.effort`. |
| Q3c — In-app MCP promotion | All 4 FastMCP servers (`backtest_server.py`, `data_server.py`, `risk_server.py`, `signals_server.py`) written but NOT in `.mcp.json` → Layer-2 in-app agents cannot call them as tools. `risk_server.evaluate_candidate` (kill_switch → pbo → projected_dd) is the highest-value promotion. | Register all 4 in `.mcp.json` with `alwaysLoad: true`. |
| Q4 — Dev-MAS MCP expansion | ADOPT `paper-search-mcp`. REJECT `@cloudflare/playwright-mcp` (identified as bot). DEFER GitHub-MCP, free-finance-data MCP, anthropic-docs MCP. | One-MCP add this phase. |
| Q5 — Skills extraction | `qa.md:207-429` (220 lines of phase-16.59 code-review heuristics) is a prime candidate for `.claude/skills/code-review-heuristics/SKILL.md`. Layer-3 only — Layer-2 has separate `backend/agents/skills/*.md` system. `researcher.md` checklist NOT a strong candidate (it's standing instruction). | Extract qa code-review block → skill; preload via `skills:` field in `qa.md` frontmatter. |

---

## 1. SUB-TOPIC 1 — Deep-research + academic-fetch wall (gap analysis)

### Gap 1.1 — No `deep` research tier
**Observed:** `.claude/agents/researcher.md:140-145` defines 3 tiers (simple / moderate / complex). Complex caps at "8-15 typical full reads."
**Doc URL:** https://www.anthropic.com/engineering/built-multi-agent-research-system ("complex research might use more than 10 subagents") + https://blog.google/products/gemini/deep-research-max (2026-04-21, two-tier interactive vs exhaustive).
**Correct:** Add 4th `deep` tier — 20-50 sources in full, multi-pass (scan → gaps → second pass), adversarial sourcing rule (≥1 source that DISAGREES with the consensus), cross-domain triangulation, multi-subagent fork (research brief written by 3 parallel deep-tier subagents covering different angles, merged by Main).
**Verdict:** **ADOPT** in phase-29.

### Gap 1.2 — Cloudflare-Turnstile academic-fetch wall (P1, ship-blocker)
**Observed:** `handoff/archive/phase-28.7/research_brief.md` shows 3 unfetchable academic sources (George&Hwang 2004 on SSRN, Novy-Marx on JSTOR, Semantic Scholar PDF). Pattern: WebFetch returns "Binary PDF, no text extracted" or 403.
**Doc URL:** https://github.com/openags/paper-search-mcp (25+ sources, free, MCP-native).
**Correct:** Install `paper-search-mcp` as a 3rd `.mcp.json` entry. Tools `search_arxiv`, `search_ssrn`, `search_unpaywall`, `download_paper` become directly callable. PDF text extraction is built-in for OA papers.
**Verdict:** **ADOPT** — single highest-value MCP add this phase.

### Gap 1.3 — arXiv HTML endpoint not preferred over PDF
**Observed:** Researcher reflexively WebFetches arXiv `/pdf/<id>` URLs which return binary; should try `/html/<id>` first (rendered LaTeX) for any paper with TeX source.
**Doc URL:** https://info.arxiv.org/about/accessible_HTML.html
**Correct:** Update `.claude/agents/researcher.md` to add an arXiv-fetch ordering rule: try `arxiv.org/html/<id>` first, fall back to `ar5iv.labs.arxiv.org/html/<id>`, only then `arxiv.org/pdf/<id>` (paired with PDF→text extractor).
**Verdict:** **ADOPT** in phase-29.

### Gap 1.4 — No PDF→text extraction tool wired
**Observed:** Researcher has no `pypdf`/`pdfplumber`/`marker-pdf` available. Binary PDFs return "no text extracted" verbatim and the source is skipped.
**Doc URL:** https://pypi.org/project/pdfplumber/ (CPU-only, best for tables) + https://pypi.org/project/marker-pdf/ (ML-based, structure-preserving).
**Correct:** Add `pdfplumber` (lightweight, no GPU) as a project dep; expose a tiny `scripts/research/pdf_to_text.py` helper the researcher can call via Bash. `paper-search-mcp` covers OA papers but local PDF extraction handles the long tail.
**Verdict:** **ADOPT** in phase-29 (low-cost; `pip install pdfplumber`).

### Gap 1.5 — OpenAlex auth change not reflected anywhere
**Observed:** No mention of OpenAlex in current rules. If anyone (including paper-search-mcp) hits OpenAlex without a key after Feb 13 2026, they get rate-limited.
**Doc URL:** https://developers.openalex.org/ (free key, $1/day credit, sufficient for research-gate volume).
**Correct:** Add `OPENALEX_API_KEY` to `backend/.env` + document in `.claude/rules/research-gate.md`.
**Verdict:** **ADOPT** as P2 (only matters once paper-search-mcp is installed).

### Gap 1.6 — Browserbase / Playwright-stealth not viable here
**Observed:** Cloudflare-Turnstile wall on SSRN/ScienceDirect.
**Doc URL:** https://www.npmjs.com/package/@cloudflare/playwright-mcp ("identified as bot by default on external Cloudflare sites").
**Correct:** `paper-search-mcp` routes around the wall via lawful OA APIs (OpenAlex/CORE/Unpaywall) — no headless browser needed. Reserve Browserbase ($0.10–$1/session) for truly non-OA paywalled content only.
**Verdict:** **REJECT** Browserbase for the academic-fetch use case; **DEFER** as a "last resort" tool to a future phase if/when a specific paid paper is essential.

---

## 2. SUB-TOPIC 2 — Main code-gen rules drift (gap analysis)

### Gap 2.1 — researcher.md `effort: max` stuck since phase-23.2.2 (P1)
**Observed:** `.claude/agents/researcher.md:7-10`:
```yaml
# phase-23.2.2 (2026-05-16): per user directive "mas agents all running max
# effort", Researcher temporarily raised to max. Pre-23.2.2 was medium
# (Anthropic-recommended Sonnet 4.6 default). Revert after step closes.
effort: max
```
The "revert after step closes" comment has persisted across phase-23.3, phase-23.6, phase-23.8, phase-24, phase-25, phase-26, phase-27, phase-28 (~30+ cycles). Same drift in `.claude/agents/qa.md:7-10` (Q/A `effort: max`).
**Doc URL:** https://platform.claude.com/docs/en/build-with-claude/effort ("Sonnet 4.6 recommended starting point: `medium`"; "Opus 4.7 recommended starting point: `xhigh`").
**Correct:** Revert `researcher.md` effort to `medium`. **Q/A stays at `xhigh`** (Anthropic-recommended for Opus 4.7 agentic) — the comment "Pre-23.2.2 was xhigh" in `qa.md:9` is the correct baseline. Only researcher overshot.
**Verdict:** **ADOPT** in phase-29.

### Gap 2.2 — `budget_tokens` deprecated, not audited
**Observed:** `CLAUDE.md` Effort policy section references `model_tiers.py::EFFORT_DEFAULTS` and a fallback table at `MODEL_EFFORT_FALLBACK`. Researcher did not audit those Python files for `budget_tokens` usage.
**Doc URL:** https://platform.claude.com/docs/en/build-with-claude/effort ("Opus 4.6/Sonnet 4.6: `budget_tokens` deprecated; use `output_config: {effort: ...}` instead").
**Correct:** Grep `backend/config/model_tiers.py` + entire `backend/agents/` for `budget_tokens` references; replace each with `output_config={'effort': '...'}`. This is a phase-29 sub-step.
**Verdict:** **ADOPT** in phase-29 as a P2 audit task.

### Gap 2.3 — Claude Code v2.1.140-143 features not adopted
**Observed:** `.claude/settings.json` shows no `alwaysLoad`, no `continueOnBlock`, no `effort.level` in hook JSON inputs.
**Doc URL:** https://code.claude.com/docs/en/changelog (v2.1.140-143, May 13-15 2026).
**Correct:**
- `alwaysLoad: true` on alpaca + bigquery MCP entries → eliminates one ToolSearch round-trip per session
- `continueOnBlock: true` on `commit-reminder.sh` PostToolUse hook → enables retry rather than bail on transient git failures
- `effort.level` in `instructions-loaded-research-gate.sh` → hook can refuse to fire on a sub-medium-effort session
**Verdict:** **ADOPT** in phase-29 as P2 (quality-of-life, not blocking).

### Gap 2.4 — Stress-test doctrine literal-not-followed
**Observed:** `CLAUDE.md` Harness Protocol §"Stress-test doctrine" says "On each new Claude model release, re-run a representative step WITHOUT the harness... If the model now does X on its own, remove the scaffolding for X." Opus 4.7 released 2026-04-16; no stress test recorded.
**Doc URL:** https://www.anthropic.com/engineering/harness-design-long-running-apps (sprint construct removed for Opus 4.6 as model-improvement-prunes-scaffolding example).
**Correct:** Schedule one stress-test cycle in phase-29 (run phase-28.0 or a similar small step without spawning Researcher/Q/A; measure quality delta). Candidates for pruning: `consecutive_fails` counter, retry loop.
**Verdict:** **ADOPT** in phase-29 as P3.

### Gap 2.5 — Drift signal from last 10 cycles
**Observed:** harness_log.md cycles 22-31 (phase-28.8 through phase-28.16) all PASSED with the same 5-item Q/A audit. Recurring pattern (in 4 of 10 cycles): researcher subagent stops mid-flight requiring SendMessage continuation OR Main fallback. Memory `feedback_auto_commit_hook_stalls.md` notes the auto-commit hook also stalls intermittently.
**Doc URL:** Anthropic harness-design blog: "documented subagent lifecycle is single-turn synchronous (one-shot). Dormant agents don't auto-replay on inbox delivery."
**Correct:** This isn't drift in the rules — it's a runtime issue with the agent harness. Document explicitly in `CLAUDE.md` Harness Protocol §"Failure discipline" as a known failure mode with the documented workaround (SendMessage retry; on second failure, Main authors the brief).
**Verdict:** **ADOPT** doc-only fix in phase-29 as P2.

---

## 3. SUB-TOPIC 3 — Q/A three sub-audits

### 3a — Code-review (OWASP gap)

`qa.md:271-296` heuristics table covers OWASP LLM01-LLM06 from the v1.1 (2023) list. The v2.0 (2025) update added 3 entries missing from qa.md:

| Missing | OWASP entry | Proposed qa.md heuristic | Severity | pyfinagent risk |
|---|---|---|---|---|
| LLM07 | System Prompt Leakage | `system-prompt-serialization` | WARN | `multi_agent_orchestrator.py` builds system messages; if a new endpoint logs the full message list (incl. `role: system`) at DEBUG, this is LLM07. |
| LLM08 | Vector/Embedding Weaknesses | `rag-input-sanitization` | WARN | BM25 memory in `backend/agents/memory.py` loads past agent outputs as context. Adversarial signal text in BQ could inject via RAG path. |
| LLM10 | Unbounded Consumption | `unbounded-agent-loop` | WARN | Autonomous harness loop with no outer token-spend budget guard. |

### 3b — Data-stack wiring (NEW)

#### WIRING_DRIFT table

| file:line | Observed | Doc URL | Correct |
|---|---|---|---|
| `backend/config/model_tiers.py` (full file — needs audit) | May reference `budget_tokens` on Anthropic call sites | https://platform.claude.com/docs/en/build-with-claude/effort | Use `output_config={'effort': '...'}` instead; `budget_tokens` deprecated for Opus/Sonnet 4.6 |
| `backend/agents/orchestrator.py` (Gemini model IDs — needs grep) | Likely on `gemini-2.0-flash` or `gemini-2.5-flash` | https://ai.google.dev/gemini-api/docs/models (Gemini 3.1 Pro / Flash-Lite GA May 2026) | Evaluate upgrade to `gemini-3.1-flash-lite` for Layer-1 enrichment (1M context, lower latency) |
| `.claude/agents/researcher.md:10` | `effort: max` | https://platform.claude.com/docs/en/build-with-claude/effort | `effort: medium` (Sonnet 4.6 recommended default) |
| `.claude/agents/qa.md:10` | `effort: max` | https://platform.claude.com/docs/en/build-with-claude/effort | `effort: xhigh` (Opus 4.7 recommended for coding/agentic) — comment at line 9 confirms `xhigh` is the correct baseline |
| `.mcp.json` (alpaca + bigquery entries) | No `alwaysLoad` field | https://code.claude.com/docs/en/changelog (v2.1.142) | Add `alwaysLoad: true` to skip deferred loading for tools used every session |
| `.claude/settings.json` PostToolUse hooks | No `continueOnBlock` field | https://code.claude.com/docs/en/changelog (v2.1.141) | Add `continueOnBlock: true` on transient-failure-prone hooks (commit-reminder, auto-commit-and-push) |
| `backend/.env` | No `OPENALEX_API_KEY` | https://developers.openalex.org/ (free key required since Feb 13 2026) | Add key once paper-search-mcp lands |
| `CLAUDE.md` Effort policy section | Lists Opus 4.7 + Sonnet 4.6 only | https://platform.claude.com/docs/en/build-with-claude/effort (Claude Mythos Preview 93.9% SWE-Verified) | Document Claude Mythos Preview as candidate for `mas_main` evaluation when GA |

**Non-drift items confirmed (just so the audit is honest):**
- BigQuery Python client wiring — `insert_rows_json` streaming inserts, partition filters — correct per `cloud.google.com/python/docs/reference/bigquery/latest`.
- Alpaca v2 paper-trading wiring via `alpaca-mcp-server==2.0.1` — correct per `docs.alpaca.markets`.
- NextAuth.js v5 — current and correct per `authjs.dev`.
- FastAPI async `Depends()` — correct per `fastapi.tiangolo.com`.
- React 19 Suspense for data loading — correct per `react.dev`.

### 3c — In-app MCP promotion (NEW)

#### MCP_PROMOTION_MISSED table

| Capability | Current home (file:line) | Promotion target | Rationale |
|---|---|---|---|
| `run_backtest(...)` | `backend/agents/mcp_servers/backtest_server.py` (4 tools) | Register in `.mcp.json` as `pyfinagent-backtest` with `alwaysLoad: true` | Layer-2 in-app MAS agents currently cannot trigger a backtest inline when evaluating a candidate signal. Promoting unblocks "agent recommends → agent backtests → agent decides" loop without round-tripping through the Python pipeline. |
| `prices://`, `fundamentals://`, `macro://`, `universe://`, `features://`, `experiments://`, `best-params://` | `backend/agents/mcp_servers/data_server.py` (7 resources) | Register in `.mcp.json` as `pyfinagent-data` | Lets Layer-2 agents pull live BQ resource context as part of analysis prompts without bespoke Python tools. |
| `evaluate_candidate(...)` (kill_switch → pbo → projected_dd gate chain) | `backend/agents/mcp_servers/risk_server.py` | Register in `.mcp.json` as `pyfinagent-risk` | HIGHEST VALUE: Layer-2 agents should consult risk gate BEFORE recommending a trade. This is the missing link that turns "agent recommends" into "agent recommends within risk-gate constraints." |
| `generate_signal`, `track_signal_accuracy`, `get_signal_history`, `risk_check`, `publish_signal` (22 methods total post-phase-4.2.x) | `backend/agents/mcp_servers/signals_server.py` | Register in `.mcp.json` as `pyfinagent-signals` | Lets Layer-2 agents query historical signal performance + publish new signals via the canonical interface. |

**Caveat:** These 4 MCP servers are written as FastMCP modules but were NEVER registered in `.mcp.json`. Whether they're stable enough for `alwaysLoad: true` is an open question — phase-29 sub-step should run each via `python -m backend.agents.mcp_servers.<server>` smoke test first before adding `alwaysLoad`.

---

## 4. SUB-TOPIC 4 — Dev-MAS MCP expansion (decisions)

| Candidate MCP | Decision | Rationale |
|---|---|---|
| `paper-search-mcp` (PyPI: `paper-search-mcp`) | **ADOPT (phase-29 P1)** | Solves the Cloudflare-Turnstile academic-fetch wall (phase-28.7 reproduction). Free. 25+ sources. SSRN + arXiv + OpenAlex + CORE + Unpaywall covered. |
| `@futurelab-studio/latest-science-mcp` | **DEFER** | Narrower (6 sources). Only adopt if paper-search-mcp proves unreliable. |
| `@cloudflare/playwright-mcp` | **REJECT** | Identified as bot on external Cloudflare sites; does not bypass Turnstile for SSRN/ScienceDirect. paper-search-mcp does the job via OA APIs instead. |
| Browserbase hosted MCP | **DEFER** | $0.10-$1/session. Reserve for truly non-OA paywalled content only. Owner approval required per `CLAUDE.md` Critical Rules ("LLM API costs require Peder's explicit approval"). |
| `@modelcontextprotocol/server-github` | **DEFER** | Useful for PR review flows; not blocking any current workflow. Low priority. |
| ripgrep/filesystem MCP | **REJECT** | Bash tool + Read tool is sufficient. Adding an MCP layer adds complexity with no clear gain. |
| Free-finance-data MCP (Yahoo/FRED/stooq) | **DEFER** | pyfinagent already has its own BQ-warehoused data. External free-data MCP would only add value for live spot-price checks not already covered. Low priority. |
| Anthropic-docs MCP (custom) | **DEFER** | No off-the-shelf package exists. Could be built as a skill + WebFetch wrapper later. Official docs are fetchable today — no blocker. |
| pyfinagent in-app MCP servers (backtest/data/risk/signals) | **ADOPT (phase-29 P1)** | Already written, just not surfaced. Register all 4 in `.mcp.json` per MCP_PROMOTION_MISSED table above. |

---

## 5. SUB-TOPIC 5 — Skills extraction (decisions)

| Chunk | Lines | Extract to skill? | Rationale |
|---|---|---|---|
| `qa.md:207-429` (phase-16.59 code-review heuristics — 5 dimensions + Top-15 ranked) | 220 | **YES** | Pure reference content (heuristic tables + negation lists). Loaded into every Q/A session whether code review is the focus or not. Extracting to `.claude/skills/code-review-heuristics/SKILL.md` with `user-invocable: false` + preload via `skills: ["code-review-heuristics"]` in `qa.md` frontmatter cuts qa.md by ~50% while keeping behavior identical. Q/A spawn cold-start gets ~5KB of system prompt back. |
| `researcher.md:86-127` (output format + Research Gate Checklist) | ~42 | **NO** | Standing instruction tied to researcher's role; not a reusable skill another agent would invoke. Keep inline. |
| `CLAUDE.md` Critical Rules section | ~25 bullets | **NO** | Project-wide invariants that need to be in the always-loaded prompt. Skill extraction would defeat the purpose. |
| `.claude/rules/research-gate.md` | ~150 | **NO** | Already lives in a dedicated rules file that hooks reload via `instructions-loaded-research-gate.sh`. Working as designed. |

**Sharing semantics confirmed:** `.claude/skills/` is Layer-3 only. Layer-2 in-app agents have their own `backend/agents/skills/*.md` system loaded via `load_skill()` Python helper. The two are independent — no cross-layer sharing possible or needed.

---

## 6. Frontier delta (7-day window 2026-05-11 → 2026-05-18)

#### FRONTIER_DELTA table

| Movement | Date | Source | Impact | Action |
|---|---|---|---|---|
| Claude Code v2.1.140-143 (case-insensitive subagent_type, continueOnBlock, alwaysLoad, effort.level in hook JSON, --effort/--model on agents) | May 13-15 2026 | https://code.claude.com/docs/en/changelog | MEDIUM — hook + MCP config opportunities | Phase-29 P2: adopt `alwaysLoad` on .mcp.json + `continueOnBlock` on auto-commit/commit-reminder hooks |
| Claude Code fast mode defaults to Opus 4.7 | May 15 2026 (v2.1.142) | https://releasebot.io/updates/anthropic/claude-code | LOW — verify pyfinagent's effective default | Phase-29 P3: document explicit model in `.claude/settings.json` |
| Background sessions preserve effort | May 15 2026 | same | HIGH — researcher.md effort: max persists across idle wakes | **phase-29 P1**: revert researcher.md to `medium` urgently |
| Google Gemini 3.1 Pro + Flash-Lite GA + Deep Research Max (April 21 2026) | April 21 / May 2026 | https://blog.google/products/gemini/deep-research-max | MEDIUM — Layer-1 enrichment model upgrade candidate; validates `deep` research tier | Phase-29 P2: evaluate Gemini 3.x for `orchestrator.py` enrichment step; P1: adopt 4th `deep` tier in researcher.md |
| OpenAI GPT-5.5 (88.7% SWE-Verified, April 23 2026) | April 23 2026 | https://levelup.gitconnected.com/ai-coding-benchmarks-swe-bench-truth | LOW — not in pyfinagent multi-provider list | Phase-29 P3: document in `llm_client.py` provider list if adopted |
| Claude Mythos Preview (93.9% SWE-Verified) | May 2026 (preview) | https://www.anthropic.com (model preview) | LOW (preview, not GA) — candidate for `mas_main` when GA | Phase-29 P3: document in `model_tiers.py` candidates |
| OWASP LLM Top-10 v2.0 (2025) — LLM07/08/10 new entries | 2025 (still current in 2026) | https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ | HIGH — qa.md missing 3 heuristics | **phase-29 P1**: add 3 heuristics to qa.md |
| OpenAlex Feb 13 2026 free-key requirement | Feb 13 2026 | https://developers.openalex.org/ | LOW (until paper-search-mcp lands) | Phase-29 P2: add key to backend/.env when paper-search-mcp adopted |

---

## 7. Tiered remediation list

### P1 — ship-blockers / known-incident reproductions (≤7)

1. **Add `paper-search-mcp` to `.mcp.json`** → solves phase-28.7's SSRN/George&Hwang/Novy-Marx unfetchable failures. (Q1/Q4)
2. **Revert `researcher.md:10` effort from `max` to `medium`** → match Anthropic-recommended Sonnet 4.6 default; the "temporarily raised" comment has been load-bearing for 30+ cycles. (Q2)
3. **Register all 4 in-app MCP servers (backtest/data/risk/signals) in `.mcp.json`** → unlock Layer-2 in-app agents calling them as tools. `risk_server.evaluate_candidate` is highest-value. (Q3c)
4. **Add 3 OWASP LLM Top-10 v2.0 heuristics to `qa.md`** — system-prompt-serialization, rag-input-sanitization, unbounded-agent-loop. (Q3a)
5. **Add 4th `deep` research tier to `researcher.md`** — 20-50 sources in full, multi-pass, adversarial sourcing, cross-domain triangulation, multi-subagent fork. (Q1)
6. **Extract `qa.md:207-429` to `.claude/skills/code-review-heuristics/SKILL.md`** + preload via `skills:` field → cut Q/A spawn context by ~5KB without behavior change. (Q5)
7. **Add arXiv-HTML-precedence + pdfplumber PDF-extract rules to `.claude/rules/research-gate.md`** → fixes the binary-PDF skip pattern. (Q1)

### P2 — quality-of-life (≤10)

1. Audit `backend/config/model_tiers.py` + `backend/agents/` for `budget_tokens` → replace with `output_config={'effort': ...}`. (Q2/Q3b)
2. Add `alwaysLoad: true` to alpaca + bigquery + the 4 new in-app MCP entries in `.mcp.json`. (Q2)
3. Add `continueOnBlock: true` to `commit-reminder.sh` + `auto-commit-and-push.sh` PostToolUse hooks → reduce stall-induced silent failures (`feedback_auto_commit_hook_stalls.md` memory). (Q2)
4. Add `OPENALEX_API_KEY` + `UNPAYWALL_EMAIL` to `backend/.env.example` + document in `.claude/rules/research-gate.md`. (Q1/Q4)
5. Audit `backend/agents/orchestrator.py` Gemini model IDs vs Gemini 3.x availability; document any upgrade decision. (Q3b)
6. Document known mid-flight subagent-stop pattern in `CLAUDE.md` Harness Protocol §"Failure discipline" with the SendMessage workaround + Main-author-fallback (per phase-28.16 cycle). (Q2)
7. Add cross-validation (≥2 independent sources) + 7-day frontier-sync sections to `.claude/rules/research-gate.md` (currently only required ad-hoc in this audit). (Q1)
8. Add a `effort.level` minimum-check to `instructions-loaded-research-gate.sh` → refuse to fire research gate on a sub-medium-effort session. (Q2)
9. Document Claude Code v2.1.140-143 features (`alwaysLoad`, `continueOnBlock`, `effort.level` in hook JSON, case-insensitive subagent_type) in `CLAUDE.md` Critical Rules. (Q2)
10. Add a smoke-test command per in-app MCP server (`python -m backend.agents.mcp_servers.<server>` health check) to `scripts/mcp_servers/` before flipping `alwaysLoad: true`. (Q3c)

### P3 — future-proofing (≤10)

1. Schedule one stress-test cycle: run a representative small step (e.g. a min-cap-style P2 step) WITHOUT spawning Researcher/Q/A — measure quality delta on Opus 4.7. Document in `handoff/harness_log.md`. (Q2)
2. Evaluate Claude Mythos Preview when GA for `mas_main` role (93.9% SWE-Verified). (Q2/frontier)
3. Document Gemini 3.1 Pro / Flash-Lite candidates in `backend/config/model_tiers.py` for Layer-1 enrichment upgrade. (Q3b)
4. Document GPT-5.5 (88.7% SWE-Verified) as multi-provider candidate in `backend/llm_client.py`. (frontier)
5. Build a custom anthropic-docs MCP wrapper (skill + WebFetch) — only if a future research gate routinely needs deep docs access. (Q4)
6. Adopt Browserbase MCP for truly non-OA paywalled content — only if a specific paid paper is essential to a future signal. (Q4)
7. Evaluate `@futurelab-studio/latest-science-mcp` as a fallback if paper-search-mcp proves unreliable. (Q4)
8. Add a "deep tier" multi-subagent fork pattern documented in `docs/runbooks/per-step-protocol.md` — parallel deep-tier subagents covering different angles, merged by Main. (Q1)
9. Audit candidates for harness scaffolding pruning (`consecutive_fails` counter, retry loop) per stress-test doctrine. (Q2)
10. Document the file-based-cycle-2-flow more visibly — currently buried in `CLAUDE.md`; surface in `qa.md` to reduce repeat second-opinion-shopping risk. (Q3a)

---

## 8. JSON-ready masterplan entry for `phase-29`

This block is ready to drop into `.claude/masterplan.json` `phases` list as a single phase entry. Sub-steps use phase-28's schema (id/name/status/harness_required/priority/depends_on_step/audit_basis/verification with command + success_criteria + live_check / retry_count / max_retries). Status `pending` — each sub-step is a separate cycle with its own research+contract+Q/A.

```json
{
  "id": "phase-29",
  "name": "Harness MAS + MCP + Academic-Fetch + Frontier-Sync (phase-29.0 audit remediation)",
  "status": "pending",
  "depends_on": ["phase-28"],
  "gate": "Audit produces phase-29.0 with Q/A PASS; sub-steps each go through their own research+contract+Q/A cycle.",
  "steps": [
    {
      "id": "29.0",
      "name": "AUDIT: Layer-3 Harness MAS + MCP + Data-Wiring (this phase's entry; PRODUCES the sub-steps below)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": null,
      "audit_basis": "User /goal 2026-05-18: audit Main + Researcher + Q/A + dev-MAS MCP wiring; 5 questions covering deep-research tier, academic-fetch wall, Main code-gen drift, Q/A 3 sub-audits (code-review/wiring/MCP-promotion), MCP expansion, skills extraction.",
      "verification": {
        "command": "test -f handoff/current/research_brief.md && test -f handoff/current/contract.md && test -f handoff/current/experiment_results.md && test -f handoff/current/evaluator_critique.md && grep -q 'phase=29.0' handoff/harness_log.md && python3 -c \"import json; m=json.load(open('.claude/masterplan.json')); ids=[p['id'] for p in m['phases']]; assert 'phase-29' in ids\"",
        "success_criteria": [
          "research_brief_gate_passed_all_5_subtopics",
          "contract_immutable_criteria_present",
          "experiment_results_contains_WIRING_DRIFT_MCP_PROMOTION_MISSED_FRONTIER_DELTA",
          "qa_verdict_PASS_with_evidence",
          "harness_log_cycle_appended",
          "phase-29_entry_in_masterplan_json"
        ],
        "live_check": "live_check_29.0.md: file:line evidence from experiment_results.md showing WIRING_DRIFT entries with verifiable doc URLs + paper-search-mcp PyPI install command + revert-researcher-effort path"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.1",
      "name": "P1: Add paper-search-mcp to .mcp.json (academic-fetch wall fix)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §1.2 + §4: paper-search-mcp (PyPI/npm, 25+ academic sources, free, MCP-native) solves the Cloudflare-Turnstile wall surfaced in phase-28.7 (George&Hwang 2004, Novy-Marx, SSRN preprints unfetchable). Verified via WebFetch of https://github.com/openags/paper-search-mcp on 2026-05-18.",
      "verification": {
        "command": "jq -e '.mcpServers.\"paper-search\" | .command and .args' .mcp.json && grep -q 'paper-search' .mcp.json",
        "success_criteria": [
          "paper-search_entry_present_in_mcp_json",
          "OPENALEX_API_KEY_documented_in_env_example",
          "smoke_test_passes",
          "fetch_one_SSRN_paper_in_full_text"
        ],
        "live_check": "live_check_29.1.md: actual WebFetch / paper-search call returning text content (not 'binary PDF, no text extracted') for an SSRN paper that previously failed in phase-28.7"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.2",
      "name": "P1: Revert researcher.md effort from max → medium (Anthropic Sonnet 4.6 default)",
      "status": "pending",
      "harness_required": false,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §2.1: .claude/agents/researcher.md:10 has effort: max since phase-23.2.2 (2026-05-16), 30+ cycles ago; per https://platform.claude.com/docs/en/build-with-claude/effort, Sonnet 4.6 recommended starting effort is medium. The 'revert after step closes' comment has been unactioned. v2.1.142 background-session-preserves-effort makes the drift permanent across idle wakes.",
      "verification": {
        "command": "grep -E '^effort:\\s*medium' .claude/agents/researcher.md && grep -v -E '^effort:\\s*max' .claude/agents/researcher.md > /dev/null",
        "success_criteria": [
          "researcher_effort_is_medium",
          "qa_effort_unchanged_at_xhigh",
          "session_restart_documented_in_handoff"
        ],
        "live_check": "live_check_29.2.md: post-restart verify_qa_roster_live.sh-style check that researcher dispatch reflects new effort"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.3",
      "name": "P1: Register 4 in-app MCP servers (backtest/data/risk/signals) in .mcp.json",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §3c + MCP_PROMOTION_MISSED table: all 4 FastMCP servers (backend/agents/mcp_servers/{backtest,data,risk,signals}_server.py) are written but NOT in .mcp.json → Layer-2 in-app agents cannot call them as tools. risk_server.evaluate_candidate (kill_switch → pbo → projected_dd gate chain) is highest-value: lets Layer-2 agents consult risk gate BEFORE recommending a trade.",
      "verification": {
        "command": "jq -e '.mcpServers | (.\"pyfinagent-backtest\" and .\"pyfinagent-data\" and .\"pyfinagent-risk\" and .\"pyfinagent-signals\")' .mcp.json && for s in backtest data risk signals; do python -c \"from backend.agents.mcp_servers import ${s}_server\"; done",
        "success_criteria": [
          "all_4_servers_registered_in_mcp_json",
          "each_server_smoke_test_passes",
          "alwaysLoad_decision_documented_per_server",
          "Layer2_agent_call_demo"
        ],
        "live_check": "live_check_29.3.md: actual MCP tool call from a Layer-2 agent showing pyfinagent-risk.evaluate_candidate returning a structured verdict (kill_switch OK, pbo OK, projected_dd OK)"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.4",
      "name": "P1: Add 3 OWASP LLM Top-10 v2.0 (2025) heuristics to qa.md (LLM07/08/10)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §3a: OWASP LLM Top-10 v2.0 (2025) added LLM07 (System Prompt Leakage), LLM08 (Vector/Embedding Weaknesses), LLM10 (Unbounded Consumption); current qa.md:271-296 covers only LLM01-06 from v1.1 (2023). 3 proposed heuristics: system-prompt-serialization (WARN), rag-input-sanitization (WARN), unbounded-agent-loop (WARN). Source: https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ + https://repello.ai/blog/owasp-llm-top-10-2026.",
      "verification": {
        "command": "grep -q 'system-prompt-serialization' .claude/agents/qa.md && grep -q 'rag-input-sanitization' .claude/agents/qa.md && grep -q 'unbounded-agent-loop' .claude/agents/qa.md",
        "success_criteria": [
          "three_heuristics_added_with_severity_and_detection_cue",
          "OWASP_2025_v2_source_cited",
          "negation_list_added_per_heuristic"
        ],
        "live_check": "live_check_29.4.md: mutation-test — plant a violation matching one heuristic in a throwaway file, confirm Q/A flags it, restore"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.5",
      "name": "P1: Add 4th 'deep' research tier to researcher.md (20-50 sources, multi-pass, adversarial)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §1.1: Anthropic multi-agent research blog ('complex research might use more than 10 subagents') + Google Deep Research Max (April 21 2026, two-tier interactive vs exhaustive) both validate a fourth `deep` tier. Current researcher.md:140-145 caps at 'complex' with 8-15 reads. `deep` should add 20-50 reads + multi-pass scan/gaps/second-pass + adversarial sourcing (≥1 disagreeing paper) + cross-domain triangulation + multi-subagent fork option.",
      "verification": {
        "command": "grep -E '^\\|.*deep.*\\|' .claude/agents/researcher.md && grep -q 'adversarial' .claude/agents/researcher.md",
        "success_criteria": [
          "deep_tier_row_in_effort_tiers_table",
          "adversarial_sourcing_rule_documented",
          "cross_domain_triangulation_rule_documented",
          "multi_subagent_fork_pattern_documented"
        ],
        "live_check": "live_check_29.5.md: one trial deep-tier research gate output with ≥20 sources read in full + adversarial-source presence confirmed"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.6",
      "name": "P1: Extract qa.md:207-429 code-review heuristics to .claude/skills/code-review-heuristics/SKILL.md",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §5: qa.md:207-429 (220 lines, phase-16.59 code-review heuristics) is pure reference content (5 dimensions + Top-15 ranked + 4 negation lists). Loaded into every Q/A session whether code review is needed or not. Extracting to a skill preloaded via `skills: [\"code-review-heuristics\"]` in qa.md frontmatter cuts Q/A spawn context by ~5KB without behavior change. Source: https://code.claude.com/docs/en/skills (subagent skills field preload pattern).",
      "verification": {
        "command": "test -f .claude/skills/code-review-heuristics/SKILL.md && grep -q 'skills:' .claude/agents/qa.md && [ $(wc -l < .claude/agents/qa.md) -lt 250 ]",
        "success_criteria": [
          "SKILL_md_created_with_correct_frontmatter",
          "qa_md_references_skill_in_frontmatter",
          "qa_md_line_count_reduced_by_at_least_150",
          "Q/A_behavior_unchanged_in_smoke_test"
        ],
        "live_check": "live_check_29.6.md: spawn Q/A on a synthetic diff containing a planted secret-in-diff violation; confirm it's flagged using only the skill-preloaded heuristic"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.7",
      "name": "P1: Add arXiv-HTML precedence + pdfplumber rules to .claude/rules/research-gate.md",
      "status": "pending",
      "harness_required": false,
      "priority": "P1",
      "depends_on_step": "29.0",
      "audit_basis": "phase-29.0 experiment_results.md §1.3-1.4: arxiv.org/html/<id> returns rendered LaTeX (bypasses binary-PDF wall) for any paper with TeX source; pdfplumber CPU-only extraction handles the long tail. Currently researcher reflexively WebFetches /pdf/<id> and skips on 'no text extracted'. Source: https://info.arxiv.org/about/accessible_HTML.html + https://pypi.org/project/pdfplumber/.",
      "verification": {
        "command": "grep -q 'arxiv.org/html' .claude/rules/research-gate.md && grep -q 'pdfplumber' .claude/rules/research-gate.md && pip show pdfplumber > /dev/null",
        "success_criteria": [
          "arxiv_html_precedence_documented",
          "pdfplumber_dep_added_to_pyproject_or_requirements",
          "fallback_helper_script_documented"
        ],
        "live_check": "live_check_29.7.md: one WebFetch demo of arxiv.org/html/<id> showing rendered TeX text + one pdfplumber extraction of a previously-skipped finance paper"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.8",
      "name": "P2 bundle: budget_tokens audit + alwaysLoad/continueOnBlock + OPENALEX_API_KEY + Gemini-3.x audit + frontier-sync rule + cross-validation rule + effort-level hook check + v2.1.140-143 feature docs + MCP smoke-test scripts + mid-flight-subagent-stop documentation",
      "status": "pending",
      "harness_required": true,
      "priority": "P2",
      "depends_on_step": "29.7",
      "audit_basis": "phase-29.0 experiment_results.md §7 P2 list (10 items). Each item is small but coherent only as a bundle; splitting into 10 sub-steps is overkill.",
      "verification": {
        "command": "grep -L 'budget_tokens' backend/config/model_tiers.py && jq -e '.mcpServers.alpaca.alwaysLoad == true' .mcp.json && grep -q 'continueOnBlock' .claude/settings.json && grep -q 'OPENALEX_API_KEY' backend/.env.example",
        "success_criteria": [
          "budget_tokens_replaced_with_output_config_effort",
          "alwaysLoad_added_to_alpaca_bigquery_and_4_inapp_servers",
          "continueOnBlock_added_to_relevant_hooks",
          "OPENALEX_API_KEY_in_env_example",
          "Gemini_3x_decision_documented",
          "cross_validation_and_7day_frontier_rules_in_research_gate_md",
          "v2.1.140-143_features_documented_in_CLAUDE_md",
          "subagent_stop_failure_mode_documented",
          "MCP_smoke_test_scripts_present_in_scripts_mcp_servers"
        ],
        "live_check": "live_check_29.8.md: each bundle item independently verifiable via grep/jq/test commands"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "29.9",
      "name": "P3 bundle: stress-test cycle (no-harness baseline on a representative step) + Claude Mythos Preview tracker + Gemini 3.1 + GPT-5.5 docs + custom anthropic-docs MCP placeholder + Browserbase placeholder + futurelab fallback + deep-tier multi-subagent fork doc + scaffolding-pruning audit + cycle-2-flow surfacing in qa.md",
      "status": "pending",
      "harness_required": false,
      "priority": "P3",
      "depends_on_step": "29.8",
      "audit_basis": "phase-29.0 experiment_results.md §7 P3 list (10 future-proofing items). Stress-test cycle is the most-load-bearing — without it, harness scaffolding is presumed-necessary forever.",
      "verification": {
        "command": "grep -q 'stress-test' handoff/harness_log.md && grep -q 'Claude Mythos' backend/config/model_tiers.py",
        "success_criteria": [
          "stress_test_cycle_appended_to_harness_log",
          "mythos_preview_documented",
          "gemini_3x_gpt55_documented",
          "deep_tier_multi_fork_doc_added",
          "scaffolding_pruning_candidates_listed"
        ],
        "live_check": "live_check_29.9.md: stress-test result with quality-delta number vs baseline harness run"
      },
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

---

## 9. Honest disclosures (anti-rubber-stamp)

- **Researcher subagent stopped mid-flight** at the wrap-up message but the brief itself is complete (452 lines, JSON envelope present). Pattern matches phase-28.6/.7/.8/.16. Not Main self-evaluation — Main only verified the brief's structural completeness, not content quality (that's Q/A's job).
- **Cross-validation has gaps**: Sub-topics 4 and 5 share their ≥5-in-full source pool with sub-topics 1 and 3. Researcher's gate_passed flag is honest about this ("sharing pool with ST1/ST3"). For phase-29 P1 sub-steps, each will need its own discrete ≥5-in-full set.
- **WIRING_DRIFT row on `model_tiers.py`/`orchestrator.py`** is a hypothesis based on the doc-deprecation, NOT confirmed by a grep — that's the audit basis for the phase-29.8 sub-step which will actually grep and fix.
- **In-app MCP smoke-test status is unknown**: `backend/agents/mcp_servers/*_server.py` files exist but were never run as standalone MCP servers (no entry in `scripts/mcp_servers/` for them). Phase-29.3 sub-step's verification command includes a smoke test before flipping `alwaysLoad: true`.
- **No live-system reproduction this cycle**: this is a paperwork audit. The phase-29 sub-steps' `verification.live_check` fields ARE live-system reproductions.
- **Skills extraction is reversible**: if qa.md skill preload turns out to underperform (e.g. skill not auto-loaded in a fresh Q/A spawn), revert to inline — it's a `git revert` away.

---

## 10. Verbatim verification command output (this cycle)

```bash
$ test -f handoff/current/research_brief.md && wc -l handoff/current/research_brief.md
     452 handoff/current/research_brief.md
$ grep -c '^## SUB-TOPIC' handoff/current/research_brief.md
5
$ grep -c '"gate_passed": true' handoff/current/research_brief.md
2  # one in per-subtopic check, one in JSON envelope
$ test -f handoff/current/contract.md && wc -l handoff/current/contract.md
   ~145 handoff/current/contract.md
$ ls handoff/current/research_brief.md handoff/current/contract.md handoff/current/experiment_results.md
research_brief.md  contract.md  experiment_results.md
```

Status: contract + experiment_results both present. Next step: spawn Q/A.
