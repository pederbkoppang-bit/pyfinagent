# Phase-75 audit register -- full-stack code-quality audit vs docs + best practices

**Step:** 75.0  **Date:** 2026-07-19  **Session:** Fable 5 + ultracode (operator override: ALL agents on `claude-fable-5`, effort max, session-scoped; `.claude/agents/*.md` Opus pins untouched).

**Workflow:** `phase75-fullstack-audit` run `wf_03d6e7c4-fda` (245 agents, ~6.76M subagent tokens, 1858 tool calls; resumed once through a mid-run session-limit reset -- cached agents replayed, 40 lost verifiers + gap round + synthesis ran live).

## Pipeline stats

| stage | count |
|---|---|
| role finders (read-only Explore, Fable/max) | 14 |
| raw findings | 145 |
| after JS key-dedupe + fuzzy cluster | 136 |
| gap-round probes (completeness critic) | 6 |
| gap-round confirmed | 58 |
| **CONFIRMED (survived adversarial verify; P1 double-refuted)** | **184** |
| refuted / duplicate (killed at verify) | 16 |
| severity mix | P0:0 P1:20 P2:100 P3:64 |

No P0-severity findings (no single confirmed live-money-loss / proven-RCE defect). The 20 P1s cluster into control-plane authorization + rail-integrity gaps that the synthesis escalated to P0 *step priority* (bundled-risk urgency), which the independent step-reviewer approved.

Verification methodology (from research_brief_75.0.md, grounded in SWR-Bench precision crisis + Refute-or-Promote + Agent Audit): every finding carries repo-relative file:line + <=12 lines verbatim evidence + a citable doc/best-practice basis; each ran >=1 adversarial verifier (default stance: WRONG until proven); every P1 ran a second maximally-skeptical refuter; duplicates vs phase-69/70/71/72/73/74 registers killed at verify.

## Confirmed P1 findings (20) -- the security/rail-integrity core

| id | file:line | title |
|---|---|---|
| pysvc-01 | `backend/slack_bot/streaming_integration.py:177` | Slack assistant streaming calls coroutine AsyncWebClient.chat_stream un-awaited -- both streaming paths raise AttributeError |
| frontend-01 | `frontend/src/lib/hooks/useEventSource.ts:91` | /agents MAS observability page structurally cannot authenticate: SSE opened with withCredentials:false and raw credential-less fetches against auth-re |
| frontend-02 | `frontend/src/lib/api.ts:114` | 401 handler hard-redirects to /login with no already-on-login guard while the root-mounted LivePortfolioProvider polls authed endpoints on /login — re |
| data-bq-01 | `backend/backtest/data_ingestion.py:91` | Ingest dedup guards swallow all exceptions, silently re-inserting duplicate historical bars |
| security-01 | `backend/main.py:412` | Unauthenticated POST mutates HITL champion-challenger approval gate |
| sre-ops-01 | `scripts/away_ops/healthcheck.sh:248` | Sole log-rotation + host-health authority (away-watchdog) is dead; backend.log at 84MB past its 50MB threshold, all other service logs have no rotatio |
| llmeng-01 | `backend/agents/claude_code_client.py:536` | Live CC rail silently drops schema enforcement: Pydantic response_schema never reaches --json-schema and no system prompt is ever passed |
| deps-01 | `backend/requirements.txt:2` | No Python lockfile: floors have silently drifted majors (pandas 3.x, yfinance 1.x) and pip-audit never scans the real runtime env |
| deps-02 | `backend/backtest/markets.py:12` | Runtime code imports exchange_calendars and gpt-researcher, which exist only in the venv and are not declared anywhere |
| api-design-01 | `backend/api/mas_events.py:146` | GET /api/mas/dashboard runs sync subprocess + sync httpx inline in async def (up to ~35s event-loop stall) |
| gap1-01 | `backend/slack_bot/commands.py:338` | Any user's white_check_mark reaction on ANY message in #ford-approvals executes git push origin main |
| gap1-02 | `backend/slack_bot/scheduler.py:963` | P0 kill-switch iMessage pager logs 'sent' without checking imsg exit code; operator phone hardcoded in source |
| gap1-04 | `backend/slack_bot/self_update.py:444` | Deploy plane has zero caller authorization: text-only API, historical wiring ran it for every assistant message |
| gap2-01 | `frontend/src/middleware.ts:24` | Middleware auth gate keyed only on Google creds — passkey-only (or any non-Google) config disables all route protection (fail-open) |
| gap3-02 | `backend/governance/limits.yaml:30` | Immutable governance risk limits have zero enforcement consumers; live kill-switch runs 2x looser on a runtime-mutable setting |
| gap4-01 | `backend/agents/mcp_servers/signals_server.py:1259` | MCP publish path risk-checks against a fabricated $10K empty portfolio: PaperTrader.get_portfolio() does not exist |
| gap5-01 | `backend/agents/skills/quant_model_agent.md:78` | quant_model_agent prompt truncated by loader: agent receives NO data and NO instructions |
| gap5-03 | `backend/agents/skills/critic_agent.md:108` | Critic capped at 2048 tokens but must echo the full 4096-token report; truncation silently disables the hallucination gate |
| gap5-02 | `backend/agents/skills/synthesis_agent.md:182` | Phase-4.14.26 and phase-26.3 prompt sections are appended outside the extracted template region and never reach any model |
| gap6-01 | `scripts/risk/gauntlet.py:134` | Gauntlet runner has no live mode: promotion gate can only ever be authorized by fabricated stub returns |

## Remediation steps installed (phase-75.1 .. 75.16, all `pending`)

Independent adversarial step-review verdict: **13 approved as-is** (`75.1, 75.3, 75.4, 75.6, 75.7, 75.8, 75.9, 75.10, 75.11, 75.12, 75.13, 75.14, 75.15`); **3 approved with required revisions applied** (75.2 queue-conflict with 74.2, 75.5 retry-ownership deconfliction, 75.16 vacuous-assert tightening); **missing (uncovered confirmed P1): none**.

| step | prio | executor | covers | title |
|---|---|---|---|---|
| 75.1 | P0 | sonnet | 6 | S1 -- backend auth surface fail-closed |
| 75.2 | P0 | opus | 9 | S2 -- Slack control-plane authorization + dead-plane removal |
| 75.3 | P0 | opus | 8 | S3 -- MCP signals/data/backtest servers: fabricated-state removal + fail-closed publish pa |
| 75.4 | P0 | sonnet | 5 | S4 -- skill-prompt delivery integrity: loader truncation, undelivered sections, critic cap |
| 75.5 | P0 | sonnet | 7 | S5 -- live CC-rail schema enforcement, metered-bypass guards, model-retirement + telemetry |
| 75.6 | P0 | opus | 5 | S6 -- frontend auth fail-closed + session hardening |
| 75.7 | P0 | sonnet | 4 | S7 -- Slack assistant streaming await-correctness + P0 pager integrity |
| 75.8 | P0 | opus | 3 | S8 -- promotion-gate stub-fabrication refusal + governance-limits divergence observability |
| 75.9 | P1 | sonnet | 8 | S9 -- BigQuery fail-closed dedup, parameterization, 30s-timeout sweep, cost guard |
| 75.10 | P1 | sonnet | 12 | S10 -- event-loop hygiene sweep: to_thread the blocking money/API paths, get_running_loop  |
| 75.11 | P1 | sonnet | 7 | S11 -- SRE hardening: always-on log rotation, single service authority, unattended-wrapper |
| 75.12 | P1 | sonnet | 7 | S12 -- frontend data-plane: SSE/fetch auth transport, login reload-loop, dead charts, poll |
| 75.13 | P1 | sonnet | 4 | S13 -- Python dependency integrity: lockfile, undeclared runtime deps, dead + implicit dec |
| 75.14 | P1 | opus | 5 | S14 -- prompt-contract reconciliation, injection fencing, fact-ledger provenance, risk-jud |
| 75.15 | P2 | sonnet | 7 | S15 -- CI gates made real: advisory flip, requires_live migration, lock-count re-audit, co |
| 75.16 | P2 | sonnet | 9 | S16 -- Cloud Functions + Docker deploy-surface retirement/hardening + script bootstrap rep |

### Applied revisions (verbatim reviewer requirement -> action)
- **75.2** -- Queue conflict: this step deletes backend/slack_bot/assistant_handler.py, but the already-pending step 74.2 explicitly plans to repoint 'the assistant_handler.py:446 haiku call' (verified real: model=
  - *Required:* In the same commit, reconcile the pending queue: edit masterplan step 74.2 to re-anchor its second pilot call-site to the live assistant path (streaming_integration/direct_responder) or explicitly mar
  - *Applied:* see step name/vcmd revision markers + 74.2 re-anchor (75.2), degraded-flag/no-double-retry rescope (75.5), `error_message`-inclusive assert (75.16).
- **75.5** -- Retry-ownership collision in (c): the shared-parse-helper 'max_tokens -> one retry at higher budget' lands on top of TWO other retry seams — ClaudeClient already carries the phase-4.14.4 MF-26/27 stop
  - *Required:* Rescope (c) to stop_reason EXPOSURE on LLMResponse plus an explicit degraded flag at the shared parse helpers, with a no-double-retry guard and a deconfliction note in the contract naming exactly ONE 
  - *Applied:* see step name/vcmd revision markers + 74.2 re-anchor (75.2), degraded-flag/no-double-retry rescope (75.5), `error_message`-inclusive assert (75.16).
- **75.16** -- The gap6-04 leg of the verification command is vacuous: functions/quant/main.py builds error_message (containing traceback.format_exc()) on its own assignment line at :252 and yields it at :255, so no
  - *Required:* Tighten the quant assert so it fails pre-fix: extend the line filter to lines containing 'yield' OR 'error_message' (the :252 assignment currently contains format_exc, so the check fails today and pas
  - *Applied:* see step name/vcmd revision markers + 74.2 re-anchor (75.2), degraded-flag/no-double-retry rescope (75.5), `error_message`-inclusive assert (75.16).

## Gap-round probes (completeness critic -> targeted round-2 finders)
- **Slack-bot operator control plane bodies (deploy-from-Slack, away-ops tokens, HITL governance)** -- backend/slack_bot/self_update.py (467 ln), operator_tokens.py (191), governance.py (315), app_home.py (562), direct_resp
- **NextAuth v5 authentication stack — the sole wall in front of every finding above** -- frontend/src/lib/auth.ts, auth.config.ts, prisma.ts; frontend/src/app/api/auth/[...nextauth]/route.ts; frontend/src/app/
- **Layer-4 scheduled self-tuning packages audited at import-edge only** -- backend/meta_evolution/{cron,cron_allocator,provider_rebalancer,archetype_library,alpha_velocity}.py (987 ln — the LIVE 
- **Layer-2 in-app MAS decision modules and MCP-server bodies (grep-swept, never read)** -- backend/agents/{debate,risk_debate,bias_detector,conflict_detector,info_gap,rag_agent_runtime,agent_definitions,memory,c
- **Prompt-content layer and the prompt-patch pipeline that mutates it** -- backend/agents/skills/*.md (28 live prompts + _legacy_phase_26_4/), backend/config/prompts.py (1,058 ln), backend/intel/
- **Non-backend Python operational tier: BQ migrations, drills, and Cloud Function bodies** -- scripts/migrations/*.py (32 files), scripts/go_live_drills/ (34), scripts/audit/ (28), scripts/{risk,qa,debug,ablation,h

## Refuted / duplicate (killed at verify -- NOT queued)

| id | severity | reason (truncated) |
|---|---|---|
| arch-03 | P2 | Evidence verified verbatim (perf_metrics.py:549 negative-only ddof=1 Sortino vs metrics/sortino.py LPM_2 with DTB3 MAR; gauntlet.py:59 rf=0.0 vs analytics rf=0. |
| arch-10 | P3 | Core claim "zero production OR test importers" is false: tests/test_deduplication.py:20, tests/test_ingestion.py:17, and tests/test_end_to_end.py:22 all import  |
| frontend-08 | P3 | Evidence quote matches KillSwitchShortcut.tsx:17-24, but the core failure scenario is neutralized by the backend: POST /api/paper-trading/flatten-all (backend/a |
| data-bq-04 | P2 | Evidence fully verified: migrate_backtest_data.py:29-31/46/65 declare date/report_date as STRING and lines 93-94 create the tables with no TimePartitioning or c |
| security-02 | P1 | Quoted evidence is real (assistant_handler.py:239 calls handle_deploy_command with no operator check; self_update.py runs git pull/pkill/crontab with zero inter |
| sre-ops-03 | P1 | The malformed string is real (doubled sk-ant-oat01- prefix + embedded newline confirmed in all 4 plists via plutil raw extract, wc -l == 2), but the P1 causal c |
| sre-ops-10 | P2 | Core claim is false: the finder's grep was broken (no -E, unquoted --include glob errors in zsh), producing a spurious "0 hits". The money-path daily trading-cy |
| llmeng-07 | P2 | Quoted evidence exists verbatim, but the finding's central claim is refuted: both sites use direct-Anthropic ClaudeClient (never Gemini), and llm_client.py:1360 |
| llmeng-09 | P3 | Evidence verified verbatim (evaluator_agent.py:420-425 slice parse with conservative-FAIL fallback; planner_agent.py:180-197 slice parse with silent empty-propo |
| deps-10 | P3 | Evidence verified (package.json:35 next ^15.0.0, :55 eslint-config-next ^16.2.4; lockfile 15.5.12 vs 16.2.4), but the skew is intentional and documented, not a  |
| gap2-07 | P2 | Core claim ("no other passkey-registration surface in the app / no in-app way to create an Authenticator") is factually wrong: Sidebar.tsx:305-313 implements re |
| gap2-08 | P3 | Evidence verbatim at auth.ts:15 (flag) and :19 (suppression), and next-auth is ^5.0.0-beta.30 — facts accurate. But not a defect in context: .claude/rules/front |
| gap3-07 | P3 | Evidence quotes are accurate (lambda: None at line 30, except-pass fail-open, unconditional _registered=True), but the claim fails in context. (a) No production |
| gap3-10 | P3 | Evidence is accurate (tuples discarded after len(); Sobol d=1 over flattened index; no production caller), but the behavior is intentional per the phase-10.3 co |
| gap4-02 | P2 | Defect is real and verified verbatim in code (orchestrator.py:2216 wrong key, schemas.py:40 'action' field, bias_detector.py:119/153-155 dead gates, plus the ca |
| gap6-02 | P2 | Evidence fully verified: tca_report.py:119-121 unconditionally unlinks TCA_LOG_PATH, line 123 always seeds 70 fills at 2-9 bps (line 74), data_source (line 170) |

## Dropped low-value (P3 polish, not queued): 78
See `confirmed_findings.json::dropped_low_value` for the full list with per-item rationale. These remain in the confirmed set (evidence preserved) but the synthesis judged them below the queue bar.

## Full data
`handoff/current/audit_phase75/confirmed_findings.json` -- all 184 confirmed (with evidence + basis + verify_reason), 16 refuted, 78 dropped, step-review verdict.
