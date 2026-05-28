# Active Goal — pyfinAgent production-ready + max money on Opus 4.8

Set by operator 2026-05-28. Full approval; run autonomously to HARD STOP (max 12 cycles).
Mirror of auto-memory `project_goal_prod_ready_push`.

## North star
Maximize Net System Alpha = Profit - (Risk + Compute). Get the app trading safely and
making money, reach production-readiness, and exploit Opus 4.8 fully.

## Diagnosed root cause (2026-05-28 diagnostic workflow)
The app emits BUY/SELL recs (Layer-1 Gemini) but executes ZERO trades because the
autonomous paper loop (`decide_trades`) is a SEPARATE path with an EMPTY candidate set,
fed by `historical_prices` 52 days stale: `daily_price_refresh` writes to the WRONG table
(`pyfinagent_data.price_snapshots` instead of `financial_reports.historical_prices`), and
the slack-bot scheduler jobs never fire. The cockpit "GATE 0/5" is the go-live PROMOTION
gate (correctly red), NOT the trade blocker. `cost_tracker.py` is missing
`claude-opus-4-8` pricing (~50x mis-cost, regression from the 4.7->4.8 bump 8ecc9efe).

## Priorities (top first, money-impact-per-effort)
1. [MONEY] Restore `historical_prices` freshness (fix destination + scheduler firing + backfill).
2. [MONEY] First autonomous trade end-to-end (fix empty `new_candidates`; roll `sod_date`; >=1 real trade).
3. [OPUS-4.8] Add `claude-opus-4-8` to `cost_tracker.py` + `/claude-api` sweep; audit per-agent `max_tokens` at xhigh.
4. [PROD] Close paper-vs-backtest Sharpe gap (re-baseline on fresh prices + cost-model parity); fix 60% maxDD / -5.72 Sharpe mismark.
5. [MONEY/NORTH-STAR] Dynamic strategy rotation (per-strategy DSR; weekly promotion of highest earner).
6. [PROD] Learn-loop evidence (outcome_tracking on first sell-close); 5 consecutive clean cron cycles.
7. [PROD/UX] Best-in-class operator control surface: full UI control + ONE consistent layout/design/animation across ALL pages.
8. [HYGIENE] autoresearch `langchain_huggingface` (owner-gated pip), `sod_date` daily roll, never_run crons.
9. [OPUS-4.8] context-editing/memory-tool + mid-conversation system messages (defer to API-key migration).

## Harness mode (non-negotiable)
Every step: spawn `researcher` (gate, >=5 sources in full, never skip) -> write `contract.md`
-> GENERATE -> `experiment_results.md` -> FRESH `qa` (no self-eval) -> append
`harness_log.md` (last) -> flip masterplan status. CONDITIONAL/FAIL -> fix + update files +
fresh `qa` (no verdict-shopping). No emojis. Local-only. Verify live (curl/BQ) before PASS.

## Constraints
Claude-Code test env now (Max flat-fee) -> Anthropic API key later. LLM API spend + pip
installs + DROP/DELETE BQ need operator approval; otherwise full approval to proceed +
commit/push to main.

## Stop conditions
- HARD STOP (success): live `/run-now` shows `n_trades>=1` + fresh `paper_trades` row dated
  today, on FRESH `historical_prices`; priorities 3-7 shipped; phase-43.0 in-session DoDs
  closed -> write `handoff/current/cycle_block_summary.md` and stop.
- SOFT STOP: 12 cycles elapsed OR a blocker needing operator -> write summary + crisp ask.

## Cycle ledger (this run)
- Cycle 1 = phase-47.1 "Restore historical_prices freshness". DONE (PASS, committed d33e7197, pushed). Band red->green; 5-month gap filled; daily refresh rewired + UTC crons + catch-up-on-start.
- Cycle 2 = phase-47.2 "First autonomous trade end-to-end". PARKED (operator LLM-spend gate; backend reloaded w/ rotation code; free daily-cron fallback tmrw 14:00 UTC). Real cause = per-sector count cap blocks all buys without rotation.
- Cycle 3 = phase-47.3 "Opus 4.8 cost_tracker pricing". DONE (PASS, 7f4d39cd, pushed). claude-opus-4-8 added to MODEL_PRICING + settings_api (was ~50x under-costed).
- Cycle 4 = phase-47.4 "Sharpe/maxDD metric integrity". DONE (PASS, e3a6b4c7, pushed). Chronological-sort fix: cockpit Sharpe -5.72->+5.42, gate maxDD 60%->5.31%, max_dd_within_tolerance False->True.
- Cycle 5 = phase-47.5 "UX foundation (design-system enforcement layer)". DONE (PASS, 79c39d41, pushed). NEW design-tokens.ts + ui/Button + ui/StatusBadge; EmptyState/DataTable frontend.md fixes. Additive; isolated npm build green.
- SOFT STOP declared (condition b) 2026-05-29 after 5 cycles (4 done+pushed, 1 parked). Every remaining item is operator-gated: money path (47.2->5->6) on LLM-spend approval; remaining UX (W3-W8) on visual verification of authed pages (NextAuth wall, frontend.md rule 5). See cycle_block_summary.md. Research REFUTED the diagnostic: real cause = `decide_trades` blocks ALL buys on per-sector COUNT cap (`portfolio_manager.py:264-271`; book 7 Tech + 1 Industrials, candidates all Tech semis, max_per_sector=2) with no rotation; sod_date already wired (no fix). Plan: restart backend to load committed swap-rotation code (69c710ec) -> real cycle rotates weakest Tech holding into top candidate (Fix A); B1 swap-robustness / B2 cap 2->3 as escalation. VALIDATION cycle incurs Gemini LLM cost -> OPERATOR-GATED (or daily cron tomorrow 14:00 runs it free with the now-loaded rotation code).
