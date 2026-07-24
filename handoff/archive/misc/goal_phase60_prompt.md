# Goal -- Phase-60: 59.3 re-audit remediation (operator-approved install, minus 60.5)

Set by operator 2026-06-11. Decision recorded verbatim: "Install minus 60.5 (Recommended)" -- install phase-60 steps 60.1-60.4 from handoff/current/phase-60-draft-UNAPPROVED.json (Main-authored, properly gated; supersedes the reverted ad349f57 install). 60.5 (score-hysteresis) is DROPPED: 55.3 ruled it same-family as the 53.1-rejected band = auto-FAIL. A minimum-holding-period lever MAY be proposed in a future phase ONLY if the post-60.2 replay still shows material churn.

## North star
(masterplan.json::goal) Maximize Net System Alpha = Profit - (Risk Exposure + Compute Burn). This goal's lens: the 59.3 blinded audit proved the live engine decides on a silently-degraded lite path while a churn mechanism bleeds money -- fix the machine that makes the money before judging the money.

## Scope -- in order (each step = FULL harness cycle)
1. Install phase-60 (60.1-60.4 only; commit quotes the operator decision).
2. 60.1 [P0] Deep-pipeline restoration + honest-degradation alarm (AW-4): replace retired gemini-2.0-flash pins (live smoke-proven model), KR tickers first-class or honestly skipped+alerted, fallback-rate alarm (default 50%) wired beside the 56.2 guard, lite-vs-full provenance operator-visible.
3. 60.2 [P0] Churn-engine fix (AW-5): conviction-0.0 sentinel eliminated behind flag (default OFF), re-eval/stamp mismatch closed, swap delta re-derived for the 1-10 scale; ON-vs-OFF $0 replay incl. away-week window; promotion = operator decision.
4. 60.3 [P1] Non-USD decision-input integrity (AW-9): KRW prices/caps converted or currency-labeled in BOTH lite prompts, implausible-input pre-check acts IN CODE, KR staleness labeled; US prompts byte-identical (test).
5. 60.4 [P1] Observability residuals: CC-rail llm_call_log writer, ingestion-silence + ticket-failure alarms, yfinance off the event loop + busy-aware watchdog, cost-budget enforce-or-respec (operator-gated), calendar_events fix-or-disable, meta-scorer fallback surfaced, API-key redaction in logs.

## Founding principles (non-negotiable)
- Full harness per step: researcher FIRST (>=5 sources in full, recency scan) -> contract.md (criteria verbatim) -> GENERATE -> ONE fresh Q/A -> harness_log.md append -> masterplan flip. No self-evaluation; no verdict-shopping.
- Every change cites an AW finding ID (59.3-harness-free-output.md; cites = snapshot 70a8242b).
- DO-NO-HARM: US momentum core byte-identical unless a flag is ON; NO live flag flips inside this goal EXCEPT the retired-model-pin repair (restores the intended full pipeline the $25 window approval contemplated -- disclose burn-rate impact in the 58.1 spend ledger).
- 53.1 + 55.3 rulings BINDING: no hysteresis/no-trade band in any renamed form.
- Evidence: BQ MCP rows in every live_check; Playwright capture for any UI claim.
- Fable-5 roster (59.1): run scripts/qa/verify_qa_roster_live.sh once at session start.

## Parallel duty -- 58.1 window stewardship
The $25 live window is RUNNING. Append spend-ledger rows + DoD-2/5/6/7/9 evidence to live_check_58.1.md as cycles land. 58.1 closes on window evidence, not inside this goal.

## Operator-gated (ask, never assume)
LLM spend beyond the $25 window; pip installs; BQ DROP / unqualified DELETE / backfills; launchctl changes; any flag promotion (60.2/60.4 decisions recorded verbatim).

## Done-definition (HARD STOP)
60.1-60.4 all PASS + phase-60 flipped done + cycle_block_summary.md refreshed with a crisp operator ask list (incl. pending flag-promotion decisions). Write the summary and stop.

## Stop conditions
SOFT STOP: 12 cycles OR an operator-blocking gate -> summary + crisp ask. Check git log after any background-agent notification (ad349f57 lesson).
