# Contract — Step 55.2

**Step id:** 55.2 — Ops incidents + agent-quality audit (away week 2026-06-01 → 2026-06-10)
**Date:** 2026-06-10
**Phase:** phase-55 (review-only; $0; NO fixes; NO LLM trading-cycle spend)
**Researcher gate:** PASSED — `handoff/current/research_brief_55.2.md` (tier=complex, 6 external sources read in full, 18 URLs, recency scan done, 11 internal modules audited; envelope `gate_passed: true`)

## Research-gate summary

Internal (brief §A): the "Missing API key for provider anthropic" string exists in NO repo/venv code — it is emitted by the `claude` CLI binary; `claude_code_client.py:163-170` deliberately scrubs ANTHROPIC_API_KEY from the subprocess env (phase-38.13.1) to force `~/.claude/` OAuth (Max flat-fee rail), so an expired OAuth session during the away week yields the provider-named error; do NOT un-scrub (billing guard at `llm_client.py:1967`). The `approval_approve` button (`governance.py:166-175`) has NO registered handler (dead control); typing "Approve" is ingested by `commands.py:185 handle_any_message` → LLM answer path → fails CLOSED on trade execution (the autonomous loop is independent; loss of oversight, not unintended trades). Watchdog "ReadTimeouts" = httpx digest probes (`scheduler.py:399/421`) timing out while the 18:00:00-UTC trading cycle (62-65 min on 06-04/05 per cycle_history.jsonl) starves the single-process event loop; every watchdog FAIL is "(1/3)" — never reached kickstart; backend never down. The 0.0/10 day: `analysis_results` 05-27 carries a parallel block of 10 rows `final_score=0.0, recommendation="HOLD"` (UPPERCASE = degraded fallback path; lowercase "Hold" = real path); digest `formatters.py:37` defaults score to 0 and published silently. CORRECTED PREMISES vs the masterplan text: `llm_call_log` HAS cycle_id+ticker columns (they are NULL); agent labels are NOT debate roles (99% NULL; labels are skill/tool tags when set); `strategy_decisions` is the rotation log (no per-ticker score). `llm_call_log` UNDER-COUNTS: zero rows for the 06-04/05/08/09 cycles because `claude_code_client.py` never calls `log_llm_call` (only GeminiClient `llm_client.py:1066` + ClaudeClient `:1736` do). Authoritative skill evidence = `paper_trades.signals` (Quant/SignalStack/Trader/RiskJudge taxonomy; every away-week BUY shows SignalStack "conviction 10.00; fallback (LLM unavailable)") + `analysis_results` (away-week full_report_json has 3 keys; insider/patent/sentiment/social empty on ALL rows). SNDK flip direction in the data is 5.0-HOLD (06-05) → 7.0-BUY (06-08), the OPPOSITE of the masterplan's "7.0→5.0" — reproduce and report the corrected direction. External (brief §B): arXiv:2603.27539 (financial-MAS evaluation taxonomy; coordination primacy; transaction-cost neglect; RAG timestamp look-ahead), OTel GenAI semantic conventions (execute_tool spans = the missing instrumentation), TianPan calibration (confident-when-wrong; ECE), Microsoft Agent Framework HITL (canonical fail-closed approval), AlignX HITL anti-patterns, Glasserman & Lin 2309.17322 (sentiment look-ahead two forms; entity masking).

## Hypothesis

The away week's ops incidents share one root condition — the Claude Code OAuth rail was down — producing (a) the approve-flow error, (c) the silent 0.0/10 degraded scoring, and the SignalStack static-fallback BUYs; the watchdog pattern is unrelated event-loop starvation; and the per-skill audit will show the lite screener path ran with enrichment skills silent, all decidable from stored BQ artifacts + logs at $0.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 55.2)

1. "incident triage with root cause or honest bounding for each: (a) the Slack 'Approve' -> 'Missing API key for provider anthropic' error (2026-06-01 x2) traced to file:line, with an explicit fail-open-vs-closed security determination for the approval path; (b) the nightly watchdog ReadTimeout pattern (~20:05-20:50 CEST on 05-27/05-28/06-04) characterized from logs with its trigger identified or honestly bounded; (c) the 05-28 all-analyses-0.0/10 day explained via pyfinagent_data.llm_call_log + strategy_decisions; each incident gets a severity and a stable finding ID for phase-56, or a WONTFIX rationale"

2. "a per-skill firing audit over 2026-06-01..2026-06-10 with an explicit skill -> evidence-source map (llm_call_log has NO cycle_id column and its agent labels are caller/debate roles like 'Bull R1/2', NOT skill names -- so map the observed agent-label taxonomy, join cycles via strategy_decisions.cycle_id or ts-bucketing, and audit deterministic non-LLM skills via their stored output artifacts/signals instead), explicitly covering rag, earnings_tone, insider, patent, news/social vs the lite-mode skip list (deep_dive, devil's-advocate, risk-assessment, multi-round debate), with the orchestrator.py:1491-2069 code paths cited; gaps between code expectation and observed firing are findings; the audit also reports realized away-week LLM burn (llm_call_log cost/token columns) against realized away-week P&L -- the N* Profit-minus-Burn reconciliation"

3. "a reasoning-quality spot-check of >=3 stored analyses from the away week (the agent rationale behind at least one whipsaw trade among MU/000660.KS/DELL included), assessing (a) whether the cited skills' outputs actually informed the decision, (b) rationale robustness -- factual claims trace to point-in-time sources, flagging narrative/hallucinated support, and (c) epistemic calibration -- scores express uncertainty honestly (the 05-28 all-0.0/10 day is the canonical miscalibration case)"

4. "signal stability is quantified across the week (count of day-over-day BUY/HOLD/SELL action flips and mean |delta-score| per ticker; the SNDK 7.0-BUY -> 5.0-HOLD flip reproduced from stored data and attributed new-information vs model-noise where the logged inputs allow), a one-paragraph look-ahead/temporal-sanitation assessment states whether the lite-path signals (news/social/RAG fact ledger) are point-in-time clean, AND scripts/harness/paper_execution_parity.py + scripts/risk/tca_report.py outputs are included or their failure honestly reported; NO fix work, NO LLM trading-cycle spend"

**Verification command (immutable):** `cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.2-ops-skill-audit.md && test -f handoff/current/live_check_55.2.md`

**Premise-correction note (immutability preserved):** criteria 2 and 4 embed factual premises that live data contradicts (llm_call_log HAS a cycle_id column — it is NULL-valued; the SNDK flip direction is 5.0→7.0, not 7.0→5.0). The criteria text is NOT edited; the deliverable satisfies each criterion's operative requirements (taxonomy map, ts-bucketing join, stored-artifact audit, flip reproduction) and reports the premise corrections explicitly as findings — honest reporting per the goal's evidence-first doctrine.

## Plan (ordered per brief GENERATE plan)

1. Incident triage with stable IDs + severity: F-A1 approve-flow OAuth-rail root cause (verify `~/.claude` OAuth/CLI auth state WITHOUT printing secrets; confirm env var present-but-scrubbed); F-A2 fail-closed determination + dead button; F-C watchdog starvation (log excerpts + cycle_history timestamps); F-D silent 0.0/10 (BQ rows, uppercase-HOLD tell, 05-27 vs 05-28 dating); F-E llm_call_log observability gap.
2. Per-skill firing audit: re-run + embed the away-window llm_call_log query (fire-count table), full-history agent-label taxonomy, paper_trades.signals 4-agent taxonomy, analysis_results enrichment-column audit (insider/patent/sentiment/social); cite orchestrator.py:1491-2069 lite-skip paths; answer the operator's skills question; flag the Lite-checkbox UI/runtime desync (Playwright capture from 55.1 referenced).
3. Mode determination: lite-vs-full from full_report_json key-shape + SignalStack fallback string.
4. Signal-stability table: per-ticker day-over-day action flips + mean |Δscore| from analysis_results; SNDK reproduction (corrected direction) + MU flat-7.0; new-info vs noise attribution.
5. Reasoning spot-check ≥3 analyses (MU 06-08 whipsaw + 000660.KS + DELL): skill-output→decision linkage, point-in-time traceability, calibration verdicts.
6. Burn vs P&L: $0.40 metered (16 rows) + unmetered-Claude-rail caveat; cross-check 55.1's total_analysis_cost=$5.05/36d; N* reconciliation against away-week realized P&L (−$132 churn / −2.26% week).
7. Look-ahead paragraph: away-week trades carry no news/sentiment look-ahead (skills DOWN ≠ proven clean); standing RAG timestamp risk + Glasserman-Lin entity-masking mitigation.
8. Include paper_execution_parity.py + tca_report.py outputs (rerun; report failures verbatim — parity expected to FAIL on client_order_id reuse as in 55.1).
9. Write `55.2-ops-skill-audit.md` + `live_check_55.2.md` (fire-count table, incident evidence excerpts, signal-stability table).

## Constraints

- $0; bounded BQ reads (date filters + LIMIT, 30s); review-only — NO fixes; no code outside handoff/.
- NEVER print secret values (key prefixes, token contents); report only is_set/validity booleans.
- Do NOT propose un-scrubbing ANTHROPIC_API_KEY as a fix direction (billing guard rationale documented).

## References

- handoff/current/research_brief_55.2.md (researcher, 2026-06-10, gate_passed: true)
- handoff/current/goal_post_away_review.md; handoff/archive/phase-55.1/55.1-away-week-postmortem.md (B-breaks staged for 55.3)
- Brief sources: arXiv:2603.27539 (html), OTel GenAI observability blog 2026, TianPan calibration 2026, MS Agent Framework HITL, AlignX HITL, Glasserman & Lin arXiv:2309.17322
- Code anchors: claude_code_client.py:163-170,188-197; llm_client.py:1066,1736,1967; governance.py:165-175; commands.py:185; scheduler.py:399,421,435,466; formatters.py:37; orchestrator.py:1491-2069,2050; assistant_handler.py:~344; scripts/launchd/backend_watchdog.sh; direct_responder.py:33-46
