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
- Cycle 1 = phase-44.1 "Restore historical_prices freshness" (prerequisite to trading). IN PROGRESS.
