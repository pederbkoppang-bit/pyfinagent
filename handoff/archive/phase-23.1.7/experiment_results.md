---
step: phase-23.1.7
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'see contract.md front-matter'
---

# Experiment Results — phase-23.1.7

## What was built

Three coordinated edits close all 3 gaps blocking future-learning rationale capture, with **zero BQ migrations required**. Every trade row's `signals` JSON column now contains the full per-trade audit trail: Quant metrics → Signal Stack overlays → Trader's actual reason → Risk Judge's actual reasoning.

## Files modified

| File | Change |
|---|---|
| `backend/services/signal_attribution.py` | (1) Trader fallback chain extended with `full_report.analysis.reason` so the lite-Claude analyzer's reason surfaces; (2) Risk fallback chain adds `risk_assessment.reason` (lite-shape key); (3) NEW `extract_quant_signals(candidate)` produces 0-2 rows from a screener candidate dict — Quant signal (momentum/RSI/vol/sector/composite_score) + SignalStack signal (regime + PEAD + conviction + news + sector_event); (4) NEW `extract_all_signals(analysis, candidate=None)` wrapper inserts Quant/SignalStack BEFORE Trader; (5) `group_signals_for_drawer` adds `quant: []` and `signal_stack: []` keys with appropriate routing |
| `backend/services/portfolio_manager.py` | `decide_trades` accepts new `candidates_by_ticker: dict[str, dict] \| None = None`; buy-side now calls `extract_all_signals(analysis, candidate=screener_candidate)` so each BUY trade record carries the full screener-input audit trail. Sell-side unchanged (no candidate available) |
| `backend/services/autonomous_loop.py` | Step 6 builds `candidates_by_ticker` from the ranked candidate list and passes to `decide_trades` |
| `frontend/src/components/AgentRationaleDrawer.tsx` | `Rationale.tree` interface adds optional `quant?: Signal[]` and `signal_stack?: Signal[]`; drawer renders new "Quant" and "Signal Stack" `<Layer>`s between Analyst and Trader (drawer ordering: Analyst → Debate → Quant → Signal Stack → Trader → Risk Judge) |
| `tests/services/test_signal_attribution.py` | NEW (20 tests covering: lite-shape Trader extraction, full-shape Trader takes priority, Risk reason key fallback, Quant signal field formatting, SignalStack overlay routing, extract_all_signals ordering invariants, drawer JSON shape matches TS interface) |

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "from backend.services.signal_attribution import extract_signals_from_analysis, extract_quant_signals, extract_all_signals, group_signals_for_drawer; lite={'recommendation':'BUY','final_score':7,'risk_assessment':{'reason':'Strong momentum + reasonable valuation'},'full_report':{'analysis':{'reason':'Q1 beat with margin expansion'}}}; cand={'ticker':'ON','sector':'Technology','momentum_1m':4.2,'momentum_3m':11.8,'rsi_14':58.3,'composite_score':8.45,'conviction_score':8,'conviction_reason':'strong momentum + positive PEAD'}; sigs=extract_all_signals(lite, candidate=cand); agents={s['agent'] for s in sigs}; assert {'Quant','SignalStack','Trader','RiskJudge'} <= agents, f'Missing agents: {agents}'; trader=next(s for s in sigs if s['agent']=='Trader'); assert 'Q1 beat' in trader['rationale'], f'Trader rationale didnt extract Claude reason: {trader[\"rationale\"]}'; risk=next(s for s in sigs if s['agent']=='RiskJudge'); assert 'Strong momentum' in risk['rationale'], f'Risk didnt extract reason field: {risk[\"rationale\"]}'; tree=group_signals_for_drawer(sigs); assert 'quant' in tree and 'signal_stack' in tree; print('ok agents=' + str(sorted(agents)) + ' tree_keys=' + str(sorted(tree.keys())))"
ok agents=['Quant', 'RiskJudge', 'SignalStack', 'Trader'] tree_keys=['analyst', 'debate', 'quant', 'risk', 'signal_stack', 'trader']
exit=0
```

The command synthesizes a lite-Claude analysis dict + a screener candidate with all overlays, calls `extract_all_signals`, asserts:
- All 4 expected agents present (Quant, SignalStack, Trader, RiskJudge)
- Trader rationale contains "Q1 beat" — proving the new full_report.analysis.reason path works
- Risk rationale contains "Strong momentum" — proving the new risk_assessment.reason path works
- group_signals_for_drawer produces the new `quant` + `signal_stack` keys

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/ tests/api/test_settings_api_signal_stack.py -v --no-header -q
collected 115 items
tests/services/test_macro_regime.py ............         [ 10%]
tests/services/test_meta_scorer.py ..............        [ 22%]
tests/services/test_news_screen.py .....................  [ 40%]
tests/services/test_pead_signal.py ..................    [ 56%]
tests/services/test_sector_calendars.py ................  [ 70%]
tests/services/test_signal_attribution.py ....................  [ 87%]
tests/api/test_settings_api_signal_stack.py ..............  [100%]
============================== 115 passed in 0.34s ==============================
```

20 new + 95 existing = 115/115 tests pass. Zero regression across all 7 cycles (phase-23.1.1 through phase-23.1.7).

## Frontend type-check

```
$ cd frontend && npx tsc --noEmit
(silent — 0 errors)
```

Drawer renders the two new `<Layer>` components with `?? []` defaults so old trade rows (without quant/signal_stack tree keys) gracefully render empty layers (which then return null since `items.length === 0`).

## What this enables for future learning

After this cycle, every BUY trade record's `signals` JSON has shape:

```json
[
  {"agent": "Quant", "role": "screener",
   "rationale": "1m momentum +4.2%; 3m momentum +11.8%; RSI14 58.3; ann_vol 0.28; sector Technology; composite_score 8.450",
   "weight": 8.45},
  {"agent": "SignalStack", "role": "overlay",
   "rationale": "regime:risk_on; pead:positive_surprise; conviction 8.00; strong momentum + positive PEAD; news:earnings_beat; ...",
   "weight": 8.0},
  {"agent": "Trader", "role": "decision",
   "rationale": "Q1 beat consensus by 12% with margin expansion and raised guidance; momentum and PEAD aligned positively",
   "weight": 7.0},
  {"agent": "RiskJudge", "role": "gate",
   "rationale": "Strong momentum + reasonable valuation; standard 10% position size",
   "weight": 10.0}
]
```

Future-learning SQL:
```sql
SELECT ticker,
       JSON_EXTRACT_SCALAR(s, '$.rationale') AS rationale,
       JSON_EXTRACT_SCALAR(s, '$.agent') AS agent
FROM `paper_trades`,
     UNNEST(JSON_EXTRACT_ARRAY(signals)) AS s
WHERE created_at > '2026-04-27' AND action = 'BUY';
```
returns one row per (trade, agent), enabling pattern-matching like:
- "Did regime:risk_off trades underperform vs risk_on?"
- "Did news:earnings_beat catalysts predict +20% returns?"
- "Did high-conviction (≥8) meta-scorer picks beat low-conviction?"
- "Did the trader's reasoning contain 'guidance raised' actually raise alpha?"

The outcome_tracker → agent_memories BM25 retrieval loop now has REAL textual context to reflect on, not just "Recommendation: BUY".

## Out of scope (per contract)

- New `paper_trading_analyses` BQ table (Phase 2 — needs operator --apply migration)
- ALTER TABLE paper_trades adding columns (Phase 2 — same)
- `outcome_tracker.evaluate_recommendation` lite-fallback path (Phase 2 — depends on new BQ table)
- `build_situation_description` extension (Phase 2 — depends on new fields being available in BQ)
- E2E test of the drawer rendering the new layers (no claude-in-chrome MCP)

## Honest disclosure

The slimmer "no migration" scope means the new structured rationale lives in the existing `signals` JSON STRING column, not in dedicated columns. Future-learning queries pay a small cost for `JSON_EXTRACT_*` calls but the data is fully queryable. A Phase-2 follow-up can promote the high-value JSON fields to dedicated columns once the operator has run the migration script.

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit
3. Tomorrow's morning cycle will be the first run that captures full rationale into BQ
