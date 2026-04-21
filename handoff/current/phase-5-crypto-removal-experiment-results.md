# Experiment Results — phase-5 crypto removal

**Action:** meta-action on `.claude/masterplan.json`. **Cycle:** 1. **Date:** 2026-04-19.

## What changed

Only `.claude/masterplan.json` was edited. No code.

### Step count

- Before: 15 active steps (5.1–5.15).
- After: **14 active steps** (5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 5.14, 5.15). Step ids NOT renumbered so existing references remain valid.
- Archived: 1 step at `phase-5.archived_dropped_steps[0]` (id 5.5 "Crypto Market Integration") with `dropped_reason: "owner directive 2026-04-19 -- crypto is not a market we will pursue"`.

### Dependencies re-chained

- 5.6 (Options): was `[5.4, 5.5]` → `[5.4]`.
- 5.10 (ETF Universe): was `[5.5]` → `[5.2]`.
- 5.11 (Cross-Market Regime): was `[5.5, 5.7]` → `[5.7]`.
- 5.13 (Multi-Asset Backtest): was `[5.4, 5.5, 5.11]` → `[5.4, 5.11]`.
- 5.14 (Autonomous Loop): was `[5.5, 5.7, 5.8, 5.12, 5.13]` → `[5.7, 5.8, 5.12, 5.13]`.
- DAG still acyclic; 5.1 and 5.4 zero-dep; 5.15 remains sink.

### Step re-scopes (active crypto code paths stripped)

- **5.2** — EODHD test changed from `get_ohlcv('BTC-USD','crypto',...)` to `get_ohlcv('EQNR.OSL','equity',...)` (international equity still needs EODHD).
- **5.3** — "Multi-Asset BQ Schema Extension" → "Multi-Asset BQ Schema Extension (FX + futures)". `crypto_candles` table dropped from DDL scope; kept `fx_ohlcv` + `futures_ohlcv`.
- **5.4** — Risk Engine test changed from `compute_position_size('BTC-USD','crypto',...)` to `compute_position_size('EUR_USD','fx',...)`. Renamed to "Multi-Asset Risk Engine Extension (equity + options + FX + futures)".
- **5.10** — Name changed to "Expanded ETF Universe (thematic + levered + international, no crypto)". Command asserts `BITO` + `IBIT` are explicitly NOT in the ticker set.
- **5.11** — Replaced `crypto_vol` input with `vvix` (VIX-of-VIX, equity vol-of-vol proxy). Name now "Cross-Market Regime Detection (equity + FX + rates)". Dropped `crypto_vol_30d` column from the in-scope DDL.
- **5.12** — Dropped `crypto_equity_spillover` signal from the 3-signal list. Now documents "crypto_equity_spillover signal REMOVED". Kept `fx_carry` + `yield_curve_rotation`.
- **5.13** — `asset_classes=['equity','crypto']` → `asset_classes=['equity','fx']`. Renamed "Multi-Asset Backtest Engine Extension (equity + FX)".
- **5.14** — Dropped `enable_crypto_trading` settings flag. Verification command now uses `ENABLE_FX_TRADING=true`. Renamed "Multi-Market Autonomous Loop Integration (equity + FX + futures)".
- **5.15** — Dropped crypto portion of e2e test. 30-day backtest now `asset_classes=['equity','fx']`. Criterion added: "No crypto references in the e2e test".

### Metadata updates

- `path_decision.summary` rewritten to reflect the crypto removal and 14-step count.
- `path_decision.crypto_removed_at` timestamp added.
- `_comments` field rewritten (3 entries, including one documenting the non-renumbering convention).
- `open_issues` unchanged (EODHD budget, IBKR infra, market priority, CFTC still apply to FX/futures/international).
- Archived step preserved at `phase-5.archived_dropped_steps[0]`.

## Verbatim verifier output

```
$ python3 scripts/applier.py  # (inline, via bash)
step count: 14
step ids: ['5.1', '5.2', '5.3', '5.4', '5.6', '5.7', '5.8', '5.9', '5.10', '5.11', '5.12', '5.13', '5.14', '5.15']
archived_dropped_steps count: 1
archived dropped id: 5.5
dropped reason: owner directive 2026-04-19 -- crypto is not a market we will pursue
steps still depending on 5.5: 0

# After active-path cleanup:
mentions of 'crypto': 8         # all in exclusion/documentation contexts ("NO crypto_vol input", "crypto REMOVED", "excluded per owner directive")
mentions of 'btc': 0
mentions of 'eth': 0
mentions of 'bito': 2           # only in the 5.10 exclusion criterion
mentions of 'ibit': 2           # only in the 5.10 exclusion criterion
```

The remaining `crypto`/`BITO`/`IBIT` mentions are deliberate audit-trail strings that document what was removed. No active step has crypto as an input, output, or tested code path.

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `step_5_5_archived_not_active` | PASS | id 5.5 not in `phase-5.steps`; preserved under `archived_dropped_steps[0]` with drop reason. |
| 2 | `deps_no_longer_reference_5_5` | PASS | 0 steps have "5.5" in `depends_on`. |
| 3 | `no_crypto_references_in_active_steps` | PASS (with documented exception) | 0 active `BTC/ETH` references; remaining `crypto`/`BITO`/`IBIT` strings are all in exclusion-criteria / documentation-of-removal contexts (confirmed by context grep). |
| 4 | `step_count_14` | PASS | `len(phase-5.steps) == 14`. |
| 5 | `path_decision_note_crypto_removed` | PASS | `path_decision.summary` explicitly mentions the crypto removal + `path_decision.crypto_removed_at` timestamp. |
| 6 | `json_valid` | PASS | Python round-trip OK. |

## Caveats

1. **Ids not renumbered.** Step id "5.5" is now gone but "5.6"..."5.15" are unchanged. This preserves existing traceability in Q/A critiques, handoff_log, and open_issues; it's called out in the phase's `_comments` field.
2. **EODHD still in scope.** Step 5.2 still adds EODHD as a data provider because international equities (5.9) need it. Crypto removal does NOT remove EODHD (which was dual-purpose: crypto + international).
3. **`asset_class='crypto'` literal** does appear in 5.3's archived ancestor and 5.2's superseded test, but NOT in the active steps. Verified by grep.
4. **Archived legacy** from the original 3-placeholder state still present under `phase-5.archived_legacy_steps` (from the 23:45 UTC restructure); the dropped 5.5 is a separate archive under `archived_dropped_steps`.

## Pre-Q/A self-check

- JSON round-trip OK.
- 14 active steps; no id conflicts; no missing verification.
- No active step has a crypto code path.
- Archive entry for 5.5 with reason + timestamp.
- Masterplan still parses; phase-5 still the last phase in `mp["phases"]`.
- `harness_log.md` NOT yet appended.
