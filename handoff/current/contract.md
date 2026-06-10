# Contract — Step 56.1

**Step id:** 56.1 — FX/value/fee data-correctness fix
**Date:** 2026-06-10
**Phase:** phase-56 (fixes; every change cites a 55.x finding ID; do-no-harm on the US momentum core)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=moderate, 6 external sources read in full, 13 URLs, recency scan; 11 internal files audited; envelope `gate_passed: true`)

## Research-gate summary

Backend (F-2): the fix is 3 lines — BUY `paper_trader.py:265` `total_value` × `_local_to_usd` (already in scope, defined :208, used :299; the BUY fee at :266 is ALREADY USD); SELL `:413-414` `total_value` + `transaction_cost` × `_l2u` (defined :370) at row-build ONLY — upstream `net_proceeds`/`sell_value` must NOT be touched (cash credit :485 and round-trip P&L :440 are already correct; converting upstream would double-convert). Consumer audit: NO reader expects LOCAL — `perf_metrics.py:406-471` turnover, Slack formatters, frontend trades cells all assume USD (the fix also corrects a ~1500x turnover-inflation bug in perf metrics). Frontend (F-1): the goal-multimarket-ux `mvUsd` pattern (`positions/page.tsx:66-76`) already remediated positions table/donut slices/exposure subtotals; the TWO surviving sites are `useLiveNav.ts:34-39` (the root — cascades to NAV card, donut center, exposure denominator, Home/sovereign tiles) and `RiskMonitorCard` `cockpit-helpers.tsx:301-302`; recommend extracting `mvUsd` to a shared helper. `types.ts:653-654` confirms `market_value` is USD. `trades-columns.tsx:10-12` comment becomes TRUE post-backend-fix (annotate the pre-fix-rows caveat pending backfill). F-12 VS-KOSPI: backend fetches no ^KS11 (grep-confirmed); true per-market excess is phase-57-adjacent → strengthen the honest disclosure. F-2 backfill: operator-gated dry-run-default migration with explicit per-trade_id USD values (derivable from 55.1 §2.1/§4) + GIPS tier-3/4 disclosure. Tests: add KRW fixtures to `backend/tests/test_phase_50_2_multicurrency.py` (matched by `-k fx`); verification command currently exits 0 (24 passed; none of the 16 env-coupled failures land in the selection); fail-pre/pass-post via magnitude guard (KRW 248,000-scale vs USD ~164) + US byte-identity do-no-harm test. External: Fowler Money pattern, MDN Intl.NumberFormat, Stripe currencies, characterization-test pattern (Feathers), CFA/GIPS error correction; recency scan 2025-2026: complementary, none contradict.

## Hypothesis

Persisting USD at the trade-row build sites (3 lines) plus routing the two surviving frontend sites through stored-USD `market_value` eliminates every KR display corruption (NAV card sane, fee sane), provable by a KRW fixture test that fails pre-fix, with the US path byte-identical.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 56.1)

1. "total_value and SELL transaction_cost are persisted in USD for non-USD markets (the paper_trader.py:265 and :386-414 paths), covered by a unit test with a KRW fixture that FAILS on the pre-fix code and PASSES on the fixed code (regression-proof); all four FX conversion points (trade recording, mark-to-market, cash ledger, fees) are verified consistent post-fix"

2. "the NAV-discrepancy root cause identified by 55.1 (finding ID cited) is fixed or, if it is data-only (no code defect), the correction path is specified; the live /paper-trading UI shows sane Value/Fee/NAV/Cash for KR rows, evidenced by a Playwright capture in live_check_56.1.md; the trades-columns.tsx:11 comment and the VS-KOSPI handling are corrected per the 55.1 verdict (true index excess via ^KS11, or keeping/strengthening the already-disclosed tooltip limitation)"

3. "correction/backfill of historical corrupted BQ rows is executed ONLY as an operator-approved migration script under scripts/migrations/ (destructive ops are operator-gated); any executed restatement carries a persistent disclosure note (what changed, when, why) plus a materiality classification, GIPS-style; if the operator declines, the corrupted rows are flagged (not silently kept) and the audit-trail caveat is documented"

4. "every change in this step cites a 55.x finding ID; fixing anything WITHOUT a finding ID is a FAIL; the finding-ID -> fix mapping is recorded in live_check_56.1.md; the US momentum core paths are untouched (do-no-harm)"

**Verification command (immutable):** `cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'fx or paper_trader or krw' -q && test -f handoff/current/live_check_56.1.md`

## Plan (finding-ID-driven; test-first for the regression proof)

1. **Tests FIRST (F-2):** add KRW fixture tests to `backend/tests/test_phase_50_2_multicurrency.py` (BUY total_value USD; SELL total_value+fee USD; US byte-identity). Run → capture the verbatim PRE-FIX FAILURE output.
2. **Backend fix (F-2):** `paper_trader.py:265` `* _local_to_usd`; `:413-414` `* _l2u`. Row-build only. Re-run tests → PASS output.
3. **Frontend fixes (F-1):** shared `mvUsd` helper; `useLiveNav.ts` positionsValue via stored-USD market_value scaled where live tick available (US) / stored market_value (non-US); `RiskMonitorCard` concentration via market_value. (F-12) VS-KOSPI: per-market card label changed to honest "<MKT> holdings" + tooltip kept/strengthened. (F-13) stale "$10K" subtitle → neutral text. (F-2-display) `trades-columns.tsx:10-12` comment annotated with the pre-fix-rows caveat pending backfill.
4. **Migration (F-2 backfill):** `scripts/migrations/backfill_56_1_kr_trade_values.py` — dry-run default, `--execute` flag, per-trade_id explicit USD values, disclosure note + GIPS materiality classification written into the script header + live_check; NOT executed (operator-gated). Until approved, the corrupted rows are FLAGGED via the documented caveat (trades-columns comment + live_check audit-trail section), not silently kept.
5. **Verify:** full verification command; `cd frontend && npm run build` (or tsc) green; restart skip-auth :3100 → Playwright capture of sane NAV/Cash/Value/Fee for KR rows; four-FX-point consistency statement (trade recording now USD; mark-to-market/cash/fee-debit unchanged-correct per 55.1).
6. **live_check_56.1.md:** finding-ID → fix map (F-1, F-2, F-12, F-13), pre-fix FAIL + post-fix PASS test output verbatim, Playwright captures, migration dry-run output, do-no-harm evidence (US byte-identity test + untouched momentum-core paths).
7. experiment_results.md → fresh Q/A → harness_log → flip.

## Constraints

- Minimal diffs at the audited sites only; every changed file cites its finding ID in the live_check map.
- Do NOT touch `net_proceeds`/`sell_value`/cash/round-trip paths (already correct — double-conversion hazard).
- US momentum core byte-identical (multiply-by-1.0 identity + byte-identity test).
- Backfill NOT executed; no live flag flips; no LLM trading-cycle spend.
- Frontend conventions: navy/slate palette untouched (no visual redesign), no emojis, build must pass.

## References

- handoff/current/research_brief.md (researcher 56.1, gate_passed: true)
- Findings: handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md §1 (F-1, F-2, F-12, F-13); 55.1 §2.1/§4 (per-row USD re-derivations for the migration); 55.1 live_check B3 (row evidence)
- Code anchors: paper_trader.py:208,:265,:266,:299,:370,:413-414,:440,:485; useLiveNav.ts:24-51; cockpit-helpers.tsx:196-235,:300-305; positions/page.tsx:66-76; trades-columns.tsx:10-12; types.ts:653-654; layout.tsx:336; backend/tests/test_phase_50_2_multicurrency.py
- External: Fowler Money pattern; MDN Intl.NumberFormat; Stripe currency docs; Feathers characterization tests; CFA/GIPS error correction
