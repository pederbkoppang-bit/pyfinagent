# Alpaca MCP — Scope-3 (live) prerequisites

This document is the feeder checklist for **BLOCKER-4 (task #46)
Paper→Live execution transition**. Phase-17's scope-3 is NOT a live
flip — it is the preconditions that must hold before BLOCKER-4 can run.

## Where the live flip actually happens

- Not here. Not in phase-17. Not in any masterplan phase other than
  BLOCKER-4.
- BLOCKER-4 acceptance criterion #6 is the immutable gate: **Peder
  types `I accept live-capital operation` in the session. No autonomous
  agent may flip this switch.**

## Preconditions for BLOCKER-4 to proceed

1. **Phase-17 scope-1 + scope-2 all done**
   - [x] 17.1 research gate (brief at `handoff/current/alpaca-mcp-research-brief.md`)
   - [x] 17.2 traditional PK paper keys in backend/.env, ALPACA_PAPER_TRADE=true
   - [x] 17.3 smoketest mcp__alpaca*__get_account_info returned PA3VQZZLAKE2 ACTIVE
   - [ ] 17.4 researcher subagent calls Alpaca MCP (needs session restart; marked in-progress)
   - [x] 17.5 paper_trader wired through ExecutionRouter (bq_sim default)
   - [x] 17.6 5 paper orders submitted to Alpaca paper + canceled (drift gate deferred to post-open cycle)
   - [x] 17.7 max_notional_usd clamp + rollback runbook

2. **Rollback runbook rehearsed**
   - [ ] `docs/runbooks/alpaca-mcp-rollback.md` walked through manually
         at least once. Cancel-all + close-all verified working on a
         synthetic open-order.

3. **Kill-switch drill green under EXECUTION_BACKEND=alpaca_paper**
   - [ ] Pause/resume audit trail captured with Alpaca backend active.
   - [ ] After pause, subsequent `submit_order` calls raise (or no-op)
         instead of landing on Alpaca.

4. **Drift measurement < 2% over 10 trading days of paper operation**
   - [ ] Compare bq_sim fill vs alpaca_paper fill per trade.
   - [ ] 10 distinct trading days; mean absolute drift < 2%.
   - [ ] No single-trade drift > 5%.

5. **BQ reconciliation**
   - [ ] `financial_reports.paper_trades` rows with `source=alpaca_paper`
         match Alpaca's `/v2/account/activities` rows 1:1 over a 10-day
         window. No orphans either direction.

6. **Credential provisioning for live**
   - [ ] User generates a LIVE Alpaca API key at
         https://app.alpaca.markets/account → "View API Keys".
   - [ ] Live key starts with `AK` OR `PKLIVE*` (both are live-capital
         prefixes per Alpaca docs).
   - [ ] Key is set in backend/.env under a DIFFERENT name
         (e.g., `ALPACA_LIVE_API_KEY_ID`) — the paper key stays in
         `ALPACA_API_KEY_ID`. Code in phase-BLOCKER-4 selects which
         at runtime based on `EXECUTION_BACKEND=alpaca_live` vs
         `alpaca_paper`.

7. **Physical kill-switch rehearsal**
   - [ ] Peder performs a dry-run `/v2/orders DELETE` + `/v2/positions DELETE`
         on the paper account to practice the flat-everything motion.
   - [ ] Kill-switch audit trail shows the rehearsal in
         `handoff/kill_switch_audit.jsonl`.

8. **$100 test position**
   - [ ] First live order is a single buy of a low-volatility name
         (e.g., MSFT or SPY) for ~$100 notional.
   - [ ] Verified in the live Alpaca dashboard.
   - [ ] Verified in BQ `paper_trades` with `source=alpaca_live`.
   - [ ] Explicit CRITICAL-level log entry: `AUDIT: first_live_order_placed`.
   - [ ] Peder types approval before the order is submitted.

## What BLOCKER-4 owns (not phase-17)

- The actual flip of `EXECUTION_BACKEND=alpaca_live`.
- The live-key env-var rotation.
- Disabling `_refuse_live_keys()` for the specific `AK`/`PKLIVE`-prefix
  flow (or relaxing it to accept live keys under a new
  `ALLOW_LIVE_CAPITAL=1` env that requires typed owner consent).
- The $100 first-live-order rehearsal.
- Peder's typed acceptance recorded in handoff.

## Explicit non-requirements

- We do NOT need Alpaca Options Trading permissions for BLOCKER-4
  (equities only at go-live).
- We do NOT need Crypto permissions (explicitly out of scope).
- We do NOT need Alpaca Pro market data subscription — the free
  snapshot endpoint is sufficient for the clamp's pre-submit price
  lookup.

## Sign-off

When all 8 preconditions above are `[x]`, phase-17 is unambiguously
done and BLOCKER-4 is unblocked. Peder may proceed to BLOCKER-4 at
their discretion — no autonomous agent may.
