# Research Brief — phase-43.0 Production-Ready DoD Audit

**Tier:** moderate | **Cycle:** 12 | **Date:** 2026-05-28 | **Author:** Researcher subagent (this session)

## 1. Headline

Expected GENERATE-phase result under tonight's audit: **8 of 14 DoD criteria PASS today
(+ 3 candidates that may flip if GENERATE finds the live evidence)**, down from "8 + DoD-1
calendar-pending = effectively 9 of 14" claimed in `production_ready_audit_2026-05-23.md`.
**DoD-1 (cron health) is FAIL today** — the autoresearch cron has had 9 consecutive ERROR
days (2026-05-20 → 2026-05-28); today's failure mode is `ModuleNotFoundError: No module
named 'langchain_huggingface'` (`handoff/autoresearch/2026-05-28-ERROR-topic08.md`), a
DIFFERENT error than phase-39.1's "anthropic:" prefix fix targeted. phase-39.1 is still
`status: pending` in masterplan.json:12313. **DoD-14 (OWASP) status is contested**: prior
audit claims PASS but SKILL.md only explicitly names 7 of 10 LLM categories (LLM01-03, 06-08,
10); LLM04 (Data and Model Poisoning), LLM05 (Improper Output Handling — covered by
"insecure-output-handling" heuristic but not LLM05-tagged), and LLM09 (Misinformation) are
NOT explicitly named. **Net verdict expected: NOT_PRODUCTION_READY.**

## 2. Sources Read in Full (>=5 required)

| # | URL | Source-tier | Accessed | One-line takeaway |
|---|-----|-------------|----------|-------------------|
| 1 | [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) | Official docs (Tier 2) | 2026-05-28 | "Communication was handled via files" — per-cycle observability requires every action to land durably on disk before downstream agents proceed. Confirms our DoD-6 (learn-loop) + DoD-9 (cycle completion) framings. |
| 2 | [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | Official docs (Tier 2) | 2026-05-28 | "Each new session begins with no memory of what came before" — solved via "claude-progress.txt file that keeps a log" + git commits. Validates our `handoff/cycle_history.jsonl` approach for DoD-9 + DoD-13. Article does NOT define explicit production-readiness criteria (so DoD-9's "5 consecutive cycles" is our project-specific gate, not from this source). |
| 3 | [Anthropic — Managed Agents](https://www.anthropic.com/engineering/managed-agents) | Official docs (Tier 2) | 2026-05-28 | "Session log sits outside the harness, nothing in the harness needs to survive a crash" + `wake(sessionId)` + `getSession(id)` — canonical restart-survivability pattern. Validates DoD-13 (`cycle_lock.py` + `clean_stale_lock` in `main.py` lifespan). |
| 4 | [OWASP — LLM Top-10 (2025) genai.owasp.org](https://genai.owasp.org/llm-top-10/) | Authoritative consortium (Tier 1 equivalent — primary source) | 2026-05-28 | Official label is **"OWASP Top 10 for LLM Applications 2025"** (released March 12, 2025) — NOT "v2.0". No v2.1 or v2026 published as of today. Categories: LLM01:2025 Prompt Injection, LLM02 Sensitive Info Disclosure, LLM03 Supply Chain, LLM04 Data & Model Poisoning, LLM05 Improper Output Handling, LLM06 Excessive Agency, LLM07 System Prompt Leakage, LLM08 Vector & Embedding Weaknesses, LLM09 Misinformation, LLM10 Unbounded Consumption. |
| 5 | [Galileo — 8 Production Readiness Checklists for AI Agents](https://galileo.ai/blog/production-readiness-checklist-ai-agent-reliability) | Authoritative blog / vendor (Tier 3) | 2026-05-28 | 8-item checklist: (1) Architectural robustness, (2) Load + stress testing, (3) Failure scenario planning, (4) Rollback + recovery, (5) Monitoring + observability, (6) Operational capacity planning, (7) Risk mitigation + audit trails, (8) Continuous post-mortems. Our 14-criterion DoD maps cleanly to items 1+4+5+7+8; we're THINNER on items 2 (no formal load test) + 6 (no compute-capacity gate). Not a 2025-era gap-closing addition — pyfinagent is local/single-user. |
| 6 | [Arthur AI — Checklist to Launch a Production-Ready AI Agent](https://www.arthur.ai/blog/checklist-to-launch-a-production-ready-ai-agent) | Authoritative blog / vendor (Tier 3) | 2026-05-28 | 6-item checklist: observability + tracing, prompt management + versioning, continuous evaluations, supervised evals against fixed datasets, guardrails (intercept bad inputs/outputs), discovery + governance. Our DoD-6/7/13 cover observability + structured-output; "prompt management" maps to `.claude/skills/*.md` versioning via git; backtest fixtures cover supervised-evals. No new gap surfaced. |

## 3. Snippet-only Sources

| URL | Source-tier | One-line summary |
|-----|-------------|------------------|
| [Anthropic Three-Agent Harness — InfoQ](https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai/) | Tech-press (Tier 3) | Restates the planner/generator/evaluator pattern; no new criteria. |
| [Anthropic / cwc-long-running-agents repo](https://github.com/anthropics/cwc-long-running-agents) | Official docs (Tier 2) | Reference implementation; mirrors the article principles. |
| [FMSB — AI in Trading practitioner spotlight (Feb 2026)](https://www.fmsb.com/wp-content/uploads/2026/02/FMSB-AI-in-Trading_Final_12.02.26_FINAL.pdf) | Industry practitioner (Tier 4) | 660KB PDF, binary-not-text-extractable via WebFetch — flagged as a gap; would warrant `pdfplumber` extraction in a future deep-tier session. |
| [SEC Rule 15c3-5 Market Access Rule (SEC Compliance Guide)](https://www.sec.gov/files/rules/final/2010/34-63241-secg.htm) | Regulator (Tier 1) | "Risk controls that prevent entry of orders exceeding appropriate pre-set credit or capital thresholds" — pyfinagent's `paper_max_positions` + sector cap + stop-loss-always-set fulfill the spirit; not a registered broker-dealer so 15c3-5 itself N/A. |
| [FINRA RN 15-09 (algo trading effective practices)](https://www.finra.org/sites/default/files/notice_doc_file_ref/Notice_Regulatory_15-09.pdf) | Regulator (Tier 1) | 5 areas: risk assessment, code dev, testing, trading systems, compliance — already cited by SKILL.md kill-switch-reachability rule. |
| [arXiv:2509.16707 — "Increase Alpha: Performance and Risk of an AI-Driven Trading Framework"](https://arxiv.org/html/2509.16707v1) | Peer-reviewed (Tier 1) | AI quant frameworks must report Sharpe + drawdown + paper-trading parity; supports DoD-2 (Sharpe-match). |
| [arXiv:2412.20138v7 — TauricResearch TradingAgents framework](https://arxiv.org/pdf/2412.20138) | Peer-reviewed (Tier 1) | Multi-agent LLM trading framework; references Bailey & Lopez de Prado DSR (already on our docket). |
| [arXiv:2101.07217 — "Great Strategy or Fooling Yourself"](https://ar5iv.labs.arxiv.org/html/2101.07217) | Peer-reviewed (Tier 1) | STSE evaluation methodology — minimum-backtest-period requirement to avoid in-sample-inflated Sharpe; supports our walk-forward baseline. |
| [Anthropic Multi-Agent Research System blog](https://www.anthropic.com/engineering/built-multi-agent-research-system) | Official docs (Tier 2) | "Lead researcher synthesizes...decides whether more research is needed" — confirms 3-agent harness pattern; matches our Main + Researcher + Q/A architecture. |
| [HKUDS/AI-Trader — fully-automated agent-native trading](https://github.com/HKUDS/AI-Trader) | Community (Tier 5) | Open-source LLM-driven trader; not authoritative for production readiness criteria. |
| [OWASP Top 10 for LLM Applications (project page)](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Tier 1 | Confirms 2025 is the current revision; same 10 categories. |
| [Cloud9Infosystems — Agentic AI Readiness Checklist 2026](https://cloud9infosystems.com/agentic-ai-readiness-checklist-2026/) | Tier 4 consultancy | Restates the standard 8-checkpoint framing; no new criteria beyond Galileo/Arthur. |
| [VARRD — MCP for Trading guide](https://www.varrd.com/guides/mcp-trading.html) | Tier 4 vendor | Catalogs production MCP tools for quant research; relevant to DoD-11 audit footprint but not a readiness gate. |

## 4. Recency Scan (last 2 years, 2024–2026)

Result: **2 new findings worth flagging; none invalidate the existing 14-criterion DoD.**

1. **OWASP LLM Top 10 evolved 2023 v1.1 → 2025**: LLM07 (System Prompt Leakage), LLM08
   (Vector and Embedding Weaknesses), LLM10 (Unbounded Consumption) were ADDED in the 2025
   revision. SKILL.md correctly added all three but mislabels them "v2.0" rather than the
   official "2025". **Not invalidating** — coverage exists for 7 of 10. **Cosmetic terminology drift +
   genuine LLM04 / LLM05 / LLM09 explicit-naming gap.** (Source #4 above.)
2. **Anthropic harness-design literature (April 2026 InfoQ + Anthropic blog posts)**:
   confirms our Main + Researcher + Q/A structure is canonical; introduces "context resets" +
   "Managed Agents" pattern. Both already implemented in pyfinagent (per-cycle handoff
   archive + `cycle_lock.py`). **No supersession.** (Sources #1, #2, #3 above.)

No 2024-2026 literature found that would add a NEW DoD-criterion to our 14-item list (e.g.,
"LLM token-budget cap" or "Model-card publication") that would be silently FAILed.

## 5. Search Queries Run (3-variant composition)

| Topic | Year-locked current | Last-2-year window | Year-less canonical |
|-------|--------------------|--------------------|---------------------|
| Production readiness for AI quant | `"production readiness checklist autonomous trading system AI quant 2026"` | `"autonomous AI agent production readiness checklist 2025 observability learn-loop kill switch"` | `"production readiness checklist autonomous trading minimum bar"` |
| Harness ops health | `"Anthropic harness design long-running applications restart-survivable cycle observability 2026"` | (folded into year-locked query above) | `"software production readiness checklist definition of done go-live criteria"` |
| OWASP LLM v2.0 / v2025 | `"OWASP Top 10 LLM Applications 2025 vs 2026 version LLM01-LLM10 current revision"` | (folded into year-locked query above) | `"algorithmic trading production deployment risk controls FINRA SR 11-7 SEC 15c3-5 LLM autonomous"` |

Year-less canonical queries surfaced FINRA RN 15-09 + SEC 15c3-5 + Google SRE production-
readiness origin — all confirm the standard checklist shape; no novel criteria.

## 6. Internal File/Table/Command Map (per DoD-N)

| DoD | Criterion (verbatim from master_roadmap §6) | Source-of-truth | Evidence command for GENERATE | Last-known status (2026-05-25 audit) | Today's expected status |
|-----|---------------------------------------------|-----------------|-------------------------------|--------------------------------------|--------------------------|
| **DoD-1** | All cron jobs have last-run within SLA (`launchctl list \| grep pyfinagent` shows 0 jobs with last-exit != 0 OR last-run > 2 days ago) | `launchctl list \| grep pyfinagent` output + `handoff/autoresearch/2026-05-28-ERROR-topic08.md` | `launchctl list \| grep pyfinagent` then `ls handoff/autoresearch/ \| tail -10` | PASS (source-fixed) + CALENDAR-PENDING | **FAIL** — today (2026-05-28) emits ERROR `ModuleNotFoundError: No module named 'langchain_huggingface'` (different from phase-39.1's "anthropic:" prefix fix). 9 consecutive ERROR days (2026-05-20..28). `launchctl list \| grep pyfinagent.autoresearch` shows last-exit = 1. phase-39.1 still `status: pending` (masterplan.json:12313). |
| **DoD-2** | Sharpe and P&L match between backtest and paper-trading within 0.01 | `backend/services/perf_metrics.py:186` `compute_sharpe_gap()` + `backend/api/paper_trading.py:719` reconciliation route + walk-forward backtest results JSON under `backend/backtest/experiments/results/` | `curl -s http://localhost:8000/api/paper-trading/reconciliation` AND pull `compute_sharpe_gap(window_days=30)` value | UNKNOWN | **UNKNOWN** — no live-cycle Sharpe comparison artifact in current handoff. Requires GENERATE to execute the BQ query + emit a comparison row. Note: there is NO walk-forward results file in `backend/backtest/experiments/results/` (only signal_generation evidence + paper_runtime_evidence). DoD-2 may be **structurally** unverifiable today without rerunning the walk-forward suite. **Risk flag for GENERATE.** |
| **DoD-3** | Kill-switch hysteresis tested (pause -> wait 2h no breach -> auto-resume fires + Slack alert) | `backend/services/kill_switch.py:275` `check_auto_resume()` + `backend/tests/test_phase_38_1_kill_switch_auto_resume.py` (9 tests) | `pytest backend/tests/test_phase_38_1_kill_switch_auto_resume.py -v` | PASS | **PASS** — `check_auto_resume()` shipped phase-38.1 cycle 58 at `kill_switch.py:275-345`; default-OFF via `kill_switch_auto_resume_enabled` flag; AUTO_RESUME_TRIGGER_AT_SEC=7200; AUTO_RESUME_ALERT_AT_SEC=3600; idempotency state at lines 52-86. |
| **DoD-4** | Test coverage >70% per layer — REPLACED by TIERED policy 2026-05-25 (`docs/coverage_tier_overrides.md`) | `docs/coverage_tier_overrides.md` + `pytest --cov backend/services/...` per-module | `pytest --cov=backend/services/kill_switch --cov=backend/services/cycle_lock --cov=backend/services/factor_correlation --cov=backend/services/factor_loadings --cov=backend/services/paper_trader --cov=backend/services/portfolio_manager --cov=backend/services/perf_metrics --cov=backend/services/cycle_health --cov-report=term-missing` | PASS (tiered, all Tier-1 STRICT) | **PASS (tiered, status-quo)** — `coverage_tier_overrides.md` records 79.1% paper_trader, 81.2% portfolio_manager, 81.2% perf_metrics, 92% kill_switch, 84% cycle_lock, 85% factor_correlation, 78% factor_loadings, 72% cycle_health. **Honest dual-interpretation footnote**: the LITERAL ">70% per layer" still FAILs at the broad-layer level (services 26%, agents 22%, api 33%). GENERATE should re-state both the literal-FAIL + operational-PASS dual interpretation to avoid drift. |
| **DoD-5** | 0 Unknown bands in Data Freshness dashboard (`GET /api/paper-trading/freshness` returns no `band='Unknown'` rows) | `backend/api/paper_trading.py:445` `/freshness` (canonical) + `backend/api/observability_api.py:25` `/freshness` alias + `backend/services/cycle_health.py::compute_freshness` | `curl -s http://localhost:8000/api/paper-trading/freshness \| python -c "import json,sys; d=json.load(sys.stdin); print([r for r in d.get('rows',[]) if r.get('band')=='Unknown'])"` | UNKNOWN | **UNKNOWN** — endpoint exists + dual-route alias confirmed; band logic in `cycle_health.py` was NOT grep-hit (no string "Unknown" found in compute_freshness). GENERATE must (a) confirm the band schema by reading `cycle_health.py::compute_freshness` source AND (b) probe the live endpoint while backend is running. **Flag for GENERATE: must probe localhost:8000.** |
| **DoD-6** | Learn-loop alive in production (`outcome_tracking` has >=10 rows from autonomous; `agent_memories` has >=5 lessons loaded on next-cycle startup) | `backend/services/autonomous_loop.py:1961-2011` (phase-35.1 fallback writer SHIPPED) + BQ `financial_reports.outcome_tracking` row count + BQ `financial_reports.agent_memories` row count | `mcp__bigquery__execute-query` (requires user approval): `SELECT COUNT(*) FROM financial_reports.outcome_tracking WHERE cycle_id IS NOT NULL` AND `SELECT COUNT(*) FROM financial_reports.agent_memories` | FAIL (2026-05-23 audit; BQ-probe shows 0 rows in both tables) | **UNKNOWN — drift candidate**. phase-35.1 SHIPPED the fallback writer at `autonomous_loop.py:1975-2011` ("phase-35.1: fallback outcome_tracking row written..."). 6+ completed cycles since 2026-05-22 (cycle_history.jsonl). GENERATE should run the BQ COUNT queries to verify. If still 0, the writer path may not be hitting (e.g., no real stop-loss closes since the writer landed). |
| **DoD-7** | Risk Judge structured-output succeeds >95% (`grep -c "Risk Judge returned invalid JSON" backend.log` for last 24h / total Risk-Judge invocations <= 0.05) | `backend/agents/orchestrator.py:107-122` `_THINKING_RISK_JUDGE_CONFIG` (response_schema=RiskJudgeVerdict SHIPPED) + `backend/agents/schemas.py:117` `RiskJudgeVerdict` Pydantic model + `backend/agents/risk_debate.py:43-49` alt config + BQ `pyfinagent_data.llm_call_log` for production count | `grep -c "Risk Judge returned invalid JSON" /tmp/pyfinagent_backend.log` (where backend log lives via launchd) OR `mcp__bigquery__execute-query`: `SELECT COUNT(*) FROM pyfinagent_data.llm_call_log WHERE agent='risk_judge' AND DATE(ts) >= CURRENT_DATE() - 1` plus a sibling JSON-success-rate query | FAIL (80% fallback rate 2026-05-22) | **UNKNOWN — drift candidate**. phase-37.1 SHIPPED `response_mime_type="application/json"` + `response_schema=RiskJudgeVerdict` at `orchestrator.py:115-116` AND `risk_debate.py:48-49`. Telemetry-wrapper gap from phase-35.2 may still mean LLM call log doesn't capture autonomous-loop Risk-Judge invocations. GENERATE must check both: schema shipped (CONFIRMED) + production fallback rate (LIVE-CHECK needed). |
| **DoD-8** | Profit-protection BLOCK closed (OPEN-2 scale-out wiring) | `backend/services/paper_trader.py:523-637` `check_scale_out_fires()` + `_persist_scale_out_levels()` | `grep -n "check_scale_out_fires\|_persist_scale_out_levels" backend/services/paper_trader.py` | PASS | **PASS** — `check_scale_out_fires` defined at `paper_trader.py:530`; idempotency via `scale_out_levels_hit` column at `:560`; gated by `settings.paper_scale_out_enabled` (default False). Confirmed wired. |
| **DoD-9** | 5 consecutive cron cycles complete (`cycle_history.jsonl` tail shows 5 in a row with `status='completed'`) | `handoff/cycle_history.jsonl` | `python -c "import json; lines=[json.loads(l) for l in open('handoff/cycle_history.jsonl')]; recent=[l for l in lines if l.get('status') in ('completed','timeout','error','halted')][-10:]; print([l['cycle_id']+':' + l['status'] for l in recent])"` | UNKNOWN (1 in row 2026-05-22) | **PASS — drift candidate**. cycle_history.jsonl tail shows completed sequence since 2026-05-22 morning timeout: `dc3f6cf1`, `c7801712`, `4f8fdca6`, `c870fdab`, `0aead72b`, `387f1648` = **6 consecutive `completed` rows** (NO timeout/error since 2026-05-22T05:30 cycle `021ed63e`). **NOTE:** there are 7 orphan "started" rows that were SUPERSEDED by their own completion rows (not unique cycles). GENERATE must verify the orphan-handling logic + count UNIQUE cycle_ids with terminal status='completed' in order. |
| **DoD-10** | Source defaults match production env values (`model_tiers.py:62` returns `gemini-2.5-pro`; `settings.py` `deep_think_model` Field default = `gemini-2.5-pro`) | `backend/config/model_tiers.py:66` + `backend/config/settings.py:30` | `grep -n "gemini_deep_think\|deep_think_model" backend/config/model_tiers.py backend/config/settings.py` | PASS | **PASS** — `model_tiers.py:66` `"gemini_deep_think": "gemini-2.5-pro"`; `settings.py:30` `deep_think_model: str = Field("gemini-2.5-pro", ...)`. Confirmed aligned to production. |
| **DoD-11** | All audit P1/P2/P3 findings accounted for (0 silent drops; grep this roadmap + masterplan for each finding-id) | `handoff/current/master_roadmap_to_production.md` §2 (OPEN-1..33) + `.claude/masterplan.json` audit_basis fields | `grep -c "OPEN-" handoff/current/master_roadmap_to_production.md` then `for id in OPEN-{1..33}; do grep -q "$id" .claude/masterplan.json \|\| echo "MISSING: $id"; done` | PASS | **PASS — likely** (status-quo). master_roadmap §2 lists OPEN-1..33; masterplan.json references OPEN-1..29 across audit_basis fields. GENERATE should grep-verify no silent drops. |
| **DoD-12** | ASCII-only loggers (`python scripts/qa/ascii_logger_check.py` exits 0) | `scripts/qa/ascii_logger_check.py` | `python scripts/qa/ascii_logger_check.py` | PASS (528 files, 1761 calls, 0 violations) | **PASS** — script confirmed at `scripts/qa/ascii_logger_check.py`. GENERATE should re-run to confirm 0 new violations introduced since 2026-05-23. |
| **DoD-13** | Restart-survivable cycle state (kill backend mid-cycle; restart; next cycle starts cleanly) | `backend/services/cycle_lock.py` + `backend/services/autonomous_loop.py:142-173` (cycle_lock.acquire wiring) + `backend/main.py` lifespan calling `clean_stale_lock` | `grep -n "cycle_lock\|clean_stale_lock" backend/services/autonomous_loop.py backend/services/cycle_lock.py backend/main.py` | PASS | **PASS** — `_cycle_lock_acquire` imported at `autonomous_loop.py:150`; `_running` in-process flag at `autonomous_loop.py:78` kept for UI status only (line 148 comment); cycle_lock release at `:1151-1159`. Confirmed wired. |
| **DoD-14** | OWASP LLM Top-10 v2.0 compliance (`.claude/skills/code-review-trading-domain/SKILL.md` covers LLM01-LLM10; no open findings) | `.claude/skills/code-review-trading-domain/SKILL.md` | `grep -E "LLM0[1-9]\|LLM10" .claude/skills/code-review-trading-domain/SKILL.md` | PASS | **CONTESTED** — SKILL.md explicitly names: LLM01 (line 59, 79), LLM02 (line 56, 78), LLM03 (line 69, 82), LLM06 (line 65, 88), LLM07 (line 85), LLM08 (line 86), LLM10 (line 87) = **7 of 10**. **GAP:** LLM04 (Data and Model Poisoning) — N/A argued: pyfinagent doesn't train models, but FINE-TUNE is implicitly N/A and could be flagged; LLM05 (Improper Output Handling) — line 81 `insecure-output-handling` BLOCK is present but NOT explicitly tagged "LLM05:2025"; LLM09 (Misinformation) — NOT covered. **Verdict guidance for GENERATE:** "PASS in spirit" is defensible if SKILL.md adds explicit LLM04/05/09 tags + N/A justification for LLM04 (or a misinformation guardrail for LLM09). Without that, **the literal "covers LLM01-LLM10" criterion FAILs 3 of 10**. **Cosmetic:** SKILL.md line 100 + 230 refer to "v2.0 (2025)" — official label is "OWASP Top 10 for LLM Applications 2025". |

## 7. Drift Candidates Since 2026-05-22

Since the 2026-05-23 audit recorded "8 of 14 PASS + DoD-1 calendar-pending = effectively 9
of 14", these criteria may have FLIPPED state (one direction or the other):

| DoD | Why flagged | Evidence pointer | Recommend GENERATE re-test? |
|-----|-------------|------------------|------------------------------|
| **DoD-1** | **FLIPPED PASS → FAIL.** phase-39.1 fix worked, but new dep error surfaced. autoresearch has ERRORed 9 consecutive days. | `handoff/autoresearch/2026-05-28-ERROR-topic08.md` + `handoff/autoresearch/2026-05-2{0..7}-ERROR-topic*.md`; `launchctl list \| grep pyfinagent.autoresearch` shows last-exit=1 | **YES — confirm via `launchctl list` + ls of autoresearch errors today.** |
| **DoD-6** | **MAY HAVE FLIPPED FAIL → PASS.** phase-35.1 SHIPPED the writer at `autonomous_loop.py:1961-2011`; 6 completed cycles since 2026-05-22. | `autonomous_loop.py:1975` "phase-35.1: fallback outcome_tracking row written..."; `cycle_history.jsonl` recent tail | **YES — BQ COUNT(*) on `financial_reports.outcome_tracking` + `financial_reports.agent_memories` (requires user approval on `mcp__bigquery__execute-query`).** |
| **DoD-7** | **MAY HAVE FLIPPED FAIL → PASS.** phase-37.1 SHIPPED `response_schema=RiskJudgeVerdict` at `orchestrator.py:116`. | `grep -c "Risk Judge returned invalid JSON"` against the live backend log (path varies); BQ `llm_call_log` count | **YES — but blocked on phase-35.2 telemetry-wrapper fix not landing; LLM call log may not capture autonomous-loop Risk-Judge invocations.** |
| **DoD-9** | **MAY HAVE FLIPPED UNKNOWN → PASS.** 6 consecutive completed cycles since 2026-05-22 morning timeout. | `cycle_history.jsonl` tail: `dc3f6cf1`, `c7801712`, `4f8fdca6`, `c870fdab`, `0aead72b`, `387f1648` | **YES — verify orphan-handling logic; count UNIQUE cycle_ids with terminal status='completed' in chronological order.** |
| **DoD-14** | **MAY HAVE FLIPPED PASS → CONTESTED.** SKILL.md only explicitly tags 7 of 10 LLM categories. | `.claude/skills/code-review-trading-domain/SKILL.md` lines 56-100, 213, 230 | **YES — either GENERATE adds the missing tags (LLM04 + LLM05 + LLM09) OR the audit accepts "in spirit" with a documented dual-interpretation.** |

**Net drift impact (best-case):** 6 of 14 → 9 or 10 of 14 PASS (DoD-6 + 7 + 9 flip + DoD-14
relabeled "PASS in spirit"). DoD-1 flips back to FAIL. **Verdict NOT_PRODUCTION_READY
remains.**

## 8. Risks / Unknowns

- **DoD-2 structurally unverifiable today**: no walk-forward backtest results file under
  `backend/backtest/experiments/results/` carries a paper-trading Sharpe comparison. The
  reconciliation route (`/api/paper-trading/reconciliation`) exists but requires the backend
  to be running. **Risk:** GENERATE may need to either run a walk-forward backtest (1+ hr) OR
  accept UNKNOWN for DoD-2 with a documented "needs live-cycle window" follow-up. Per CLAUDE.md
  per-step protocol, "UNKNOWN" cannot count as PASS for the 43.0 gate.
- **DoD-4 budget**: full coverage run (`pytest --cov backend`) can exceed the 55s Q/A budget.
  Per `coverage_tier_overrides.md`, the audit should accept the module-level coverage numbers
  already recorded (last measured cycle 55) and NOT re-run the full suite during the gate
  audit. GENERATE should explicitly cite the tier policy doc as the evidence.
- **DoD-5 requires backend running**: `curl localhost:8000/api/paper-trading/freshness` only
  works if the backend launchd job is alive. Per `launchctl list` snapshot, backend is alive
  (PID 1257) — GENERATE can probe directly.
- **DoD-6 + DoD-7 require BQ MCP approval**: each `mcp__bigquery__execute-query` triggers a
  user-approval prompt (`.claude/settings.json` deny rule). Bundle the queries to minimize
  approval friction.
- **DoD-14 contested**: the prior audit (2026-05-23) claimed PASS but a literal grep against
  SKILL.md surfaces 3 missing LLM-category tags. GENERATE has two options: (a) update
  SKILL.md to explicitly tag LLM04 + LLM05 + LLM09 (the simpler path — adds 3 lines + 1
  N/A justification for LLM04), (b) accept "PASS in spirit" with documented dual-
  interpretation. Path (a) is preferable; path (b) drifts toward coverage-theater.
- **DoD-1 phase-39.1 is still pending**: the autoresearch fix targeted the `anthropic:`
  prefix issue, but TODAY's failure mode (2026-05-28) is `ModuleNotFoundError:
  langchain_huggingface`. This is a NEW upstream issue introduced by the gpt-researcher
  dependency. **The fix is owner-gated AND has expanded scope**: phase-39.1 must now
  also include the missing dependency pin/install. GENERATE should document this in the
  audit deliverable as a "phase-39.1 widening required" finding.
- **3rd-CONDITIONAL auto-FAIL not in force here** — this is a phase-43.0 audit; the per-step
  protocol's 3rd-CONDITIONAL-auto-FAIL rule (CLAUDE.md "Failure discipline F1") applies to a
  single step-id across cycles, not to the per-DoD verdict count.
- **FMSB PDF binary not extracted**: source #3 snippet-only is binary; would warrant
  `pdfplumber` extraction in a future `deep`-tier session to validate Tier-1 industry
  authority on AI-in-trading minimum bar.

## 9. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 13,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```

**Files inspected internally (12):**
1. `handoff/current/production_ready_audit_2026-05-23.md`
2. `handoff/current/closure_roadmap.md`
3. `handoff/current/master_roadmap_to_production.md` (§2 + §6)
4. `.claude/masterplan.json` (phase-39.1, phase-43.0, OPEN-N audit_basis)
5. `backend/services/kill_switch.py:52-345`
6. `backend/services/paper_trader.py:523-637` (scale-out)
7. `backend/services/autonomous_loop.py:78-2011` (cycle_lock + learn-loop writer)
8. `backend/services/cycle_lock.py:1-69`
9. `backend/services/perf_metrics.py:174-261`
10. `backend/agents/orchestrator.py:84-122` + `backend/agents/risk_debate.py:43-49` + `backend/agents/schemas.py:117`
11. `backend/config/model_tiers.py:62-66` + `backend/config/settings.py:30`
12. `.claude/skills/code-review-trading-domain/SKILL.md:50-160, 213, 230`
13. `scripts/qa/ascii_logger_check.py:1-30`
14. `handoff/cycle_history.jsonl` (terminal-status tail)
15. `handoff/autoresearch/2026-05-28-ERROR-topic08.md`
16. `docs/coverage_tier_overrides.md`

Gate passes: 6 sources read in full (>=5 floor), recency scan completed with explicit
findings, 3-variant queries documented, 16 internal files mapped to DoD criteria. **All
hard-blocker checklist items satisfied per `.claude/rules/research-gate.md`.**
