# Contract — phase-29.0 (Layer-3 Harness MAS + MCP + Data-Wiring Audit)

**Step ID:** phase-29.0 (NEW phase — this step proposes its own masterplan entry)
**Step name:** Audit Layer-3 Harness MAS (Main + Researcher + Q/A) + dev-MAS MCP wiring + data-stack drift; deliver JSON-ready `phase-29` masterplan entry covering Harness MAS + MCP + Academic-Fetch + Frontier-Sync remediation.
**Date:** 2026-05-18
**Author:** Main (3-agent harness session)
**Tier:** complex

---

## Research-gate summary

| Metric | Value |
|---|---|
| Researcher tier | complex |
| Sources read in full (total across 5 sub-topics) | 11 |
| Snippet-only sources | 14 |
| Unique URLs collected | 25+ |
| Recency scan (last 2 years) | DONE — all 5 sub-topics |
| Three-query-variant discipline | DONE — current-year / 2yr / year-less per sub-topic |
| Cross-validation (≥2 independent for empirical claims; single-source-OK for vendor docs) | APPLIED |
| 7-day frontier-sync (2026-05-11 → 2026-05-18) | DONE — 6 harness-impact items flagged |
| `gate_passed` | **true** for all 5 sub-topics |

**Brief:** `handoff/current/research_brief.md` (452 lines). JSON envelope at line 433.

**Headline research findings (anchors for the gap analyses):**
1. `paper-search-mcp` (PyPI) and `@futurelab-studio/latest-science-mcp` (npm) — pre-built MCP servers covering OpenAlex / arXiv / SSRN / CORE / Unpaywall — solve the Cloudflare-Turnstile wall surfaced in phase-28.7 (George&Hwang 2004, Novy-Marx, SSRN preprints unfetchable).
2. **Effort drift**: `.claude/agents/researcher.md:10` has `effort: max` from phase-23.2.2 (2026-05-16) — Anthropic-recommended default for Sonnet 4.6 is `medium`. The "revert after step closes" comment has been unactioned across multiple phases.
3. **OWASP LLM Top-10 v2.0 (2025)** added LLM07 (System Prompt Leakage), LLM08 (Vector/Embedding Weaknesses), LLM10 (Unbounded Consumption) — `.claude/agents/qa.md:271-296` heuristics table has none of the three.
4. **Claude Code v2.1.140-143** (May 13–15) shipped `continueOnBlock`, `alwaysLoad`, `effort.level` in hook JSON, `--effort`/`--model` on subagent dispatch, case-insensitive `subagent_type` matching — none reflected in `.claude/settings.json` or `CLAUDE.md`.
5. **In-app MCP servers gap**: `.mcp.json` registers only `alpaca` and `bigquery`. `backend/agents/mcp_servers/{backtest,data,risk,signals}_server.py` (FastMCP, written but not surfaced) are NOT in `.mcp.json` → Layer-2 in-app agents cannot call them as tools.
6. **Google Deep Research Max** (2026-04-21) ships a two-tier interactive-vs-exhaustive architecture — validates proposed 4th `deep` research tier (20–50 sources, multi-pass, adversarial sourcing).
7. **`budget_tokens` deprecated** on Opus 4.6 / Sonnet 4.6 (Anthropic effort doc, 2026-05-18) — replace with `output_config: {effort: ...}` if pyfinagent's `backend/config/model_tiers.py` still references it.

---

## Hypothesis

The harness MAS layer (Main + Researcher + Q/A) and its dev-MAS MCP surface have **measurable, file-locatable drift** vs Anthropic's published 2026 Claude Code guidance, OWASP LLM Top-10 v2.0, and the in-house FastMCP backend. The drift is:

- **Configurable** (rule/heuristic/MCP-config edits, not architectural rewrites)
- **Measurable** (file:line evidence required for every claim — no "agent feels…" assertions)
- **Tiered** (P1 = ship blockers / known incident reproductions, P2 = quality-of-life, P3 = future-proofing)

If a single follow-on phase (`phase-29`) adopts the P1 items, future research-gate calls (a) no longer 403 on SSRN/Cloudflare-walled papers, (b) Q/A catches the three missing OWASP entries, and (c) Layer-2 in-app agents can call the in-house MCP servers as tools. Quantitative success target: zero "binary PDF, no text extracted" failures on the next academic-fetch-heavy research gate (compare against phase-28.7 brief which had 3 such failures).

---

## Immutable success criteria (DO NOT EDIT once written; copied to masterplan.json on phase-29 creation)

1. **Researcher brief exists and is complete** — `handoff/current/research_brief.md` ≥ 400 lines, JSON envelope present, `gate_passed: true` on all 5 sub-topics. *Already satisfied (452 lines, envelope at line 433).*
2. **Contract exists** — `handoff/current/contract.md` (this file) names the step-id, lists research-gate summary, immutable criteria, and references. *Self-satisfying.*
3. **Experiment results exist** — `handoff/current/experiment_results.md` contains:
   - Gap analysis per sub-topic (5 sections)
   - `WIRING_DRIFT` table — file:line | observed | doc URL | correct
   - `MCP_PROMOTION_MISSED` table — capability | current home | promotion target | rationale
   - `FRONTIER_DELTA` table — 7-day movement | impact | action
   - Tiered remediation list — P1 (≤7 items) / P2 (≤10) / P3 (≤10) with file paths and explicit "ADOPT/DEFER/REJECT" verdict per the 5 audit questions
   - JSON-ready masterplan entry for `phase-29` (matching the phase-23.8 schema with `verification.live_check` field, immutable verification criteria, step-by-step breakdown)
4. **Q/A verdict obtained** — `handoff/current/evaluator_critique.md` written by a SPAWNED `qa` subagent (not Main self-evaluation) with verdict ∈ {PASS, CONDITIONAL, FAIL} and a JSON block with `ok`, `verdict`, `violated_criteria`, `violation_details`, `checks_run`. On CONDITIONAL/FAIL: Main updates handoff files, then spawns a FRESH qa (file-based cycle-2 flow per Anthropic harness-design doc, NOT a SendMessage retry on unchanged evidence).
5. **harness_log.md cycle appended** — using the standard `## Cycle N -- YYYY-MM-DD -- phase=29.0 result=PASS|CONDITIONAL|FAIL` header, with Generator/Researcher/Q/A summaries.
6. **No code edits** — this is an audit. The only writes are: `research_brief.md`, `contract.md`, `experiment_results.md`, `evaluator_critique.md`, `harness_log.md` append, and a single phase-29 entry inserted into `.claude/masterplan.json` with `status: pending`. **Zero edits to `backend/`, `frontend/`, `.claude/agents/`, `.claude/rules/`, `.claude/settings.json`, `.mcp.json` in this cycle** (they all land in phase-29 sub-steps, gated by separate research+contract+Q/A).
7. **Commit prefix** — final git commit subject starts with `phase-29.0:` so the changelog classifier picks the right semver bump.

**Verification command (for masterplan.json `verification.command` field):**
```bash
test -f handoff/current/research_brief.md && \
test -f handoff/current/contract.md && \
test -f handoff/current/experiment_results.md && \
test -f handoff/current/evaluator_critique.md && \
grep -q "phase=29.0" handoff/harness_log.md && \
python3 -c "import json; m=json.load(open('.claude/masterplan.json')); ids=[p['id'] for p in m['phases']]; assert 'phase-29' in ids, 'phase-29 entry missing'"
```

**`verification.live_check`** (R-1 gate): `"Brief at handoff/current/research_brief.md shows gate_passed=true on all 5 sub-topics; experiment_results.md contains WIRING_DRIFT + MCP_PROMOTION_MISSED + JSON-ready phase-29 entry; qa subagent verdict block present in evaluator_critique.md (PASS or with documented cycle-2 fix)."`

---

## Plan steps

1. **DONE** — Spawn `researcher` complex tier across 5 sub-topics with 3-variant query discipline, ≥2-independent-sources cross-validation, 7-day frontier-sync.
2. **DONE** — Write this contract.md (research-gate summary + immutable criteria + plan).
3. **NEXT** — Write `experiment_results.md`: per-sub-topic gap analysis, WIRING_DRIFT/MCP_PROMOTION_MISSED/FRONTIER_DELTA tables, P1/P2/P3 remediation, JSON-ready phase-29 entry.
4. Spawn `qa` ONCE (Opus, effort xhigh per qa.md frontmatter). Pass the explicit 5-item harness-compliance audit prompt (researcher-spawned? contract pre-commit? results present? log-last? no-verdict-shopping?) FIRST, then audit content.
5. **On PASS** → append `handoff/harness_log.md` Cycle entry (BEFORE the masterplan edit per the log-last memory `feedback_log_last.md`).
6. **On PASS** → insert a single `phase-29` entry into `.claude/masterplan.json` with `status: pending` (NOT `done` — phase-29.0 only delivers the proposal; the sub-steps that implement remediation are separate cycles).
7. **On PASS** → `git add -A && git commit -m "phase-29.0: …" && git push origin main` (or document live-check gate if applicable).
8. **On CONDITIONAL/FAIL** → read Q/A's `violated_criteria`, update `experiment_results.md` + `evaluator_critique.md` (appended Follow-up section), spawn a FRESH `qa` per Anthropic's cycle-2 file-based pattern. NEVER second-opinion-shop on unchanged evidence.

---

## References

### External (from research brief)
- **Anthropic harness design** — https://www.anthropic.com/engineering/harness-design-long-running-apps (Plan→Generate→Evaluate, file-based handoffs, sprint-construct-removed-in-Opus-4.6)
- **Anthropic multi-agent research blog** — https://www.anthropic.com/engineering/built-multi-agent-research-system (tiered effort, 1/3–5/10+ agent scaling)
- **Anthropic effort doc** — https://platform.claude.com/docs/en/build-with-claude/effort (Opus 4.7 xhigh recommended; budget_tokens deprecated)
- **Claude Code subagents doc** — https://code.claude.com/docs/en/sub-agents (frontmatter spec; effort + maxTurns + skills field)
- **Claude Code changelog** — https://code.claude.com/docs/en/changelog (v2.1.140-143: continueOnBlock, alwaysLoad, effort.level in hook JSON, case-insensitive subagent_type)
- **Claude Code skills doc** — https://code.claude.com/docs/en/skills (SKILL.md spec; subagent preload pattern)
- **OWASP LLM Top-10 v2.0 (2025)** — https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ + https://repello.ai/blog/owasp-llm-top-10-2026 (LLM07 System Prompt Leakage, LLM08 Vector/Embedding Weaknesses, LLM10 Unbounded Consumption — all NEW)
- **paper-search-mcp** — https://github.com/openags/paper-search-mcp (PyPI/npm, 25+ academic sources, free)
- **@futurelab-studio/latest-science-mcp** — https://github.com/benedict2310/Scientific-Papers-MCP (npm, 6 sources, >90% text extraction)
- **OpenAlex API** — https://developers.openalex.org/ (free key required Feb 13 2026)
- **@cloudflare/playwright-mcp** — https://www.npmjs.com/package/@cloudflare/playwright-mcp (REJECTED — identified as bot on external Cloudflare sites)

### Internal (file:line anchors for gap analyses)
- `.claude/agents/researcher.md:10` — `effort: max` drift from phase-23.2.2
- `.claude/agents/qa.md:271-296` — code-review heuristics table (missing LLM07/08/10)
- `.claude/agents/qa.md:207-429` — full code-review heuristics block (skills extraction candidate, 220 lines)
- `.claude/settings.json` — full read, no `alwaysLoad` / `continueOnBlock` / `effort.level` usage
- `.mcp.json` — only `alpaca` + `bigquery`; missing `paper-search` and the 4 in-house MCP servers
- `backend/agents/mcp_servers/backtest_server.py` — 4 tools, NOT registered in `.mcp.json`
- `backend/agents/mcp_servers/data_server.py` — 7 resources, NOT registered
- `backend/agents/mcp_servers/risk_server.py` — `evaluate_candidate` gate chain (kill_switch → pbo → projected_dd), NOT registered
- `backend/agents/mcp_servers/signals_server.py` — 22 methods, NOT registered
- `handoff/harness_log.md` cycles 22-31 — phase-28.x feature stack reference for WIRING_DRIFT
- `handoff/current/research_brief.md` — full 452-line audit (THIS phase's primary input)

---

## Out-of-scope (do NOT widen)

- Layer-2 (`backend/agents/`) refactors. Reference-only this cycle.
- Layer-1 (28 Gemini agents in `orchestrator.py`) refactors.
- Any actual MCP install (`.mcp.json` edit) — that lands in a phase-29 sub-step, separately gated.
- Any actual `effort:` field revert in `.claude/agents/researcher.md` — also a phase-29 sub-step.
- Slack bot, paper-trader, BQ schema changes.
