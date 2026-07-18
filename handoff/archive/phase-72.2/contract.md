# Contract — phase-72.2: P2 measurement integrity

**Step id:** 72.2 (phase-72, depends_on 72.1 = done/PASS @080f93c1)
**Session role:** Fable 5 + ultracode, AUDIT + RESEARCH ONLY. No product code, no .env, no flag flips, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_1ae448f7-007` (opus/max, tier=moderate): 7 external sources read in full (CFA GIPS 2026, Kitces, TWR canonical, Meradia/Cube/LiveFlow/AllInvestView industry), 22 URLs, recency scan, 10 internal files, every claim file:line-anchored. Brief: `handoff/current/research_brief_72.2.md`. Root causes already pinned at the code/git level:

1. **Benchmark discontinuity = deploy artifact**: commit `320b7dbb` (phase-38.7, 2026-05-22 23:57:41) switched `_get_benchmark_return`'s SPY anchor from `inception_date` to the later `first_funded_date` (`paper_trader.py:1283-1313`; anchor from `bigquery_client.py:1050-1079`). SPY rose between the two anchors → benchmark dropped 14.97→4.76 at the first post-deploy cycle (05-26). `save_daily_snapshot` MERGEs only today's row (`bigquery_client.py:993-1037`) so the stored series permanently mixes pre-fix and post-fix rows — ~+10pp of alpha is this artifact. Daily benchmark movement while NAV is frozen is correct-by-design (SPY moves at 100% cash).
2. **FX**: the 1.0-parity sell fallback was **closed by phase-69.1 (~07-11)** (`paper_trader.py:414-430` now blocks; `fx_rates.py:105-114` last-known-good) — but that landed AFTER the 10 in-window KR trades (.KS/.KQ in `paper_trades`, which has NO currency column). Still-LIVE bugs: `paper_round_trips.py:109` computes realized_pnl_usd from raw LOCAL prices (sums KRW+EUR+USD as dollars); non-US add-on avg_entry mixes USD-cost/local-shares when `paper_avg_entry_fx_fix_enabled` OFF (settings.py:455); `_usd_value_asof:199-225` degrades to today's rate (look-ahead FX).
3. **Snapshots**: `nav: None` is HARDCODED in `paper_metrics_v2.py:195` (permanent writer defect, separate API trigger). The daily snapshot is Step-8 (`autonomous_loop.py:1535-1544`) inside the main try; terminal `except TimeoutError`(:1639)/`except Exception`(:1644)/`finally`(:1662) return with NO fail-safe snapshot → any mid-cycle raise/timeout cascade (P0: 305 rail timeouts, breaker OPEN 07-10/13/14/15) drops the day's row.
4. Bonus divergence: `perf_metrics.compute_sharpe_from_snapshots:116` is flow-BLIND (ignores `external_flow_today`) and feeds the go-live gate, vs the canonical GIPS `_nav_to_returns`.
5. External consensus: benchmark must be same-basis/same-period as the portfolio; FX must use transaction-date spot from one stored source, never parity; never sum mixed-currency P&L.

## Hypothesis

The measurement layer overstates alpha by ~10pp (benchmark deploy artifact), may misstate realized P&L on KR round trips (local-raw realized computation; possible pre-69.1 parity bookings), and silently drops NAV observability exactly on the bad days (no fail-safe snapshot). Honest alpha re-derivation + a KR FX re-valuation + dropped-day census will bound the true measurement error, and 4-5 executor-tagged fix steps close the layer.

## Immutable success criteria (verbatim from .claude/masterplan.json step 72.2)

- "Benchmark computation located (file:line) and the 05-23->05-26 discontinuity mechanically explained with data evidence; honest re-derived alpha stated with method"
- "FX-1.0 mis-booking check on all in-window KR round trips + NAV with verdict clean/dirty and evidence; if dirty, corrected P&L bounds stated"
- "Snapshot-gap root cause identified; measurement fix steps appended to masterplan pending + executor-tagged; no product code edited"

verification.command: `bash -c 'test -f handoff/current/money_diagnosis_72.md && grep -Eqi "benchmark" handoff/current/money_diagnosis_72.md && grep -Eqi "FX" handoff/current/money_diagnosis_72.md && grep -Eqi "snapshot" handoff/current/money_diagnosis_72.md'`

## Plan

1. GENERATE — quantitative forensics Workflow (read-only, .venv python + yfinance for SPY/KRW histories, bounded BQ SELECTs): (a) honest-alpha re-derivation on a single consistent anchor (both variants: inception-anchored and funded-anchored) at key dates (05-15, 05-29, 06-03, 07-16); (b) KR FX re-valuation — every .KS/.KQ trade + round trip: booked total_value/realized vs recomputed shares×price×transaction-date-FX; verdict clean/dirty + bounds; test whether `paper_round_trips` stored realized values used raw local prices; (c) dropped-snapshot census — cycle-ran days (llm_call_log/log evidence) vs snapshot rows since 05-01 + the 07-17 cycle's terminal path in backend.log; (d) adversarial verify (barrier).
2. Update `money_diagnosis_72.md` §P2 with verdicts + honest alpha; add decision-sheet P2 note (measurement claims quarantine until fixes land).
3. Append executor-tagged measurement fix steps (benchmark history rebuild migration + single-anchor policy; round-trips FX fix; metrics-v2 nav writer; fail-safe snapshot in finally; flow-blind Sharpe fix — grouped sensibly, each with immutable live_check).
4. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 114) → flip 72.2 done.

## References

- `handoff/current/research_brief_72.2.md` (envelope + sources + measurement_seams file:line inventory)
- `handoff/current/money_recon_2026-07-18.md`, `money_diagnosis_72.md` (P0/P1 verified)
- git `320b7dbb` (phase-38.7); phase-69 register #18 (FX parity); GIPS/TWR + multi-currency sources per brief
