# Sprint Contract — phase-5 crypto removal (scope-reduction)

**Step id:** phase-5-crypto-removal (meta-action)
**Cycle:** 1 **Date:** 2026-04-19 **Tier:** simple

## Research-gate summary

Closure-style brief (no new external sources; builds on 2026-04-19 23:45 UTC phase-5-restructure research which already had 7 sources in full). `gate_passed: true`. Brief at `handoff/current/phase-5-crypto-removal-research-brief.md`.

## Hypothesis

Pure scope-subtraction from the 15-step phase-5 plan. Drop step 5.5 (Crypto Market Integration) entirely. Re-chain dependencies. Re-scope steps 5.10 (drop BITO/IBIT crypto-ETF tickers), 5.11 (replace crypto_vol input with equity-vol proxy), 5.12 (drop crypto_equity_spillover signal), 5.13 (drop asset_classes=['equity','crypto'] test), 5.14 (drop enable_crypto_trading flag), 5.15 (drop crypto portion of e2e gate). Archive the dropped step under `phase-5.archived_dropped_steps` for auditability. Phase still has 14 active sub-steps.

## Plan

1. Apply a single Python edit to `.claude/masterplan.json`:
   - Archive `phase-5.steps[4]` (id 5.5) into `phase-5.archived_dropped_steps` with `dropped_reason: "owner directive 2026-04-19 -- crypto is not a market we will pursue"`.
   - Remove id 5.5 from the active `steps` list.
   - Re-chain `depends_on` on 5.6, 5.10, 5.11, 5.13, 5.14 to drop the 5.5 reference.
   - Re-scope 5.10: remove "BITO, IBIT" from the success_criteria + rename from "Expanded ETF Universe (thematic + levered + intl)" dropping the crypto implication; keep thematic/levered/international tickers.
   - Re-scope 5.11 command + criteria: replace `crypto_vol=0.80` / `crypto_vol=1.2` / `crypto_vol=0.3` with `vvix=95` / `vvix=140` / `vvix=75`; drop `crypto_vol_30d` column from the DDL in the step's scope text; leave the underlying `cross_market_regime` intent intact.
   - Re-scope 5.12: drop `crypto_equity_spillover` from the listed signals; keep `fx_carry` + `yield_curve_rotation`. Drop that function from the verification command; replace with a check on `fx_carry(...)`.
   - Re-scope 5.13: replace `asset_classes=['equity','crypto']` with `asset_classes=['equity','fx']` throughout.
   - Re-scope 5.14: drop `enable_crypto_trading` from success_criteria + verification; keep `enable_fx_trading` + `enable_futures_trading`.
   - Re-scope 5.15: drop crypto lines from criteria; keep equity + FX sanity checks.
   - Update `path_decision.summary` to note the crypto-removal.
   - Update `_comments` field.
   - Leave `open_issues` unchanged (EODHD budget, IBKR infra, market priority, CFTC still apply).
2. Verify via Python read.
3. Write experiment-results with exact diff.
4. Spawn Q/A.
5. On PASS: append cycle block to `harness_log.md`.

## Immutable criteria (authored for this meta-action)

- `step_5_5_archived_not_active` — `phase-5.steps` no longer contains id "5.5"; the dropped step is preserved under `phase-5.archived_dropped_steps`.
- `deps_no_longer_reference_5_5` — no step in `phase-5.steps[*].depends_on` contains "5.5".
- `no_crypto_references_in_active_steps` — grep on JSON-serialized active `phase-5.steps` returns no match for `crypto`, `BTC`, `ETH`, `BITO`, `IBIT` (case-insensitive) except in docstrings/comments that explicitly say "no crypto".
- `step_count_14` — `len(phase-5.steps) == 14`.
- `path_decision_note_crypto_removed` — `phase-5.path_decision.summary` explicitly mentions the crypto removal.
- `json_valid` — masterplan.json parses cleanly.

## Out of scope

- No code deletion (no code was ever written for crypto — the sub-steps were just plan entries).
- No removal of Alpaca crypto capabilities from Alpaca itself (we don't own that).
- No change to phase-7 or any other phase.

## References

- `handoff/current/phase-5-crypto-removal-research-brief.md`
- `handoff/current/phase-5-restructure-contract.md` (predecessor)
- `.claude/masterplan.json` → phase-5 (currently last, 15 steps)
