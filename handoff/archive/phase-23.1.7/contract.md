---
step: phase-23.1.7
title: Capture full agent rationale + signal stack into paper_trades.signals JSON for future learning
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "from backend.services.signal_attribution import extract_signals_from_analysis, extract_quant_signals, extract_all_signals, group_signals_for_drawer; lite={\"recommendation\":\"BUY\",\"final_score\":7,\"risk_assessment\":{\"reason\":\"Strong momentum + reasonable valuation\"},\"full_report\":{\"analysis\":{\"reason\":\"Q1 beat with margin expansion\"}}}; cand={\"ticker\":\"ON\",\"sector\":\"Technology\",\"momentum_1m\":4.2,\"momentum_3m\":11.8,\"rsi_14\":58.3,\"composite_score\":8.45,\"conviction_score\":8,\"conviction_reason\":\"strong momentum + positive PEAD\"}; sigs=extract_all_signals(lite, candidate=cand); agents={s[\"agent\"] for s in sigs}; assert {\"Quant\",\"SignalStack\",\"Trader\",\"RiskJudge\"} <= agents, f\"Missing agents: {agents}\"; trader=next(s for s in sigs if s[\"agent\"]==\"Trader\"); assert \"Q1 beat\" in trader[\"rationale\"], f\"Trader rationale didnt extract Claude reason: {trader[\"rationale\"]}\"; risk=next(s for s in sigs if s[\"agent\"]==\"RiskJudge\"); assert \"Strong momentum\" in risk[\"rationale\"], f\"Risk didnt extract reason field: {risk[\"rationale\"]}\"; tree=group_signals_for_drawer(sigs); assert \"quant\" in tree and \"signal_stack\" in tree; print(\"ok agents=\" + str(sorted(agents)) + \" tree_keys=\" + str(sorted(tree.keys())))"'
research_brief: handoff/current/phase-23.1.7-research-brief.md
---

# Contract — phase-23.1.7

## Hypothesis

Three small backend edits (no BQ migration) capture every signal that drove a trade decision into the existing `paper_trades.signals` JSON column, giving the outcome_tracker / agent_memories learning loop real context to reflect on. The Agent Rationale drawer renders the new layers automatically once the signal-attribution extractor produces them.

## Plan

1. **`backend/services/signal_attribution.py`** — three changes:
   - **Trader fallback chain**: add `analysis.get("full_report", {}).get("analysis", {}).get("reason")` so the lite-Claude analyzer's "reason" field appears as the Trader's rationale (instead of falling through to literal "Recommendation: BUY").
   - **Risk fallback chain**: add `risk.get("reason")` so the lite shape `risk_assessment={"reason": "..."}` produces a populated RiskJudge signal.
   - **NEW `extract_quant_signals(candidate)`**: produces 0-2 new signal rows from a screener candidate dict — one "Quant" Analyst-tier signal (momentum_1m/3m/6m, RSI, vol, sector, composite_score), one "SignalStack" overlay signal (conviction_score, conviction_reason, news_event_type, news_rationale, source tag).
   - **NEW `extract_all_signals(analysis, candidate=None)`**: wrapper that combines `extract_signals_from_analysis(analysis)` + `extract_quant_signals(candidate)`, inserting the Quant/SignalStack rows BEFORE the Trader row so the drawer ordering is Analyst → Quant → SignalStack → Trader → Risk.
   - **`group_signals_for_drawer` extension**: add `quant: []` and `signal_stack: []` keys to the output tree; route by `agent == "Quant"` / `"SignalStack"` or `role == "screener"` / `"overlay"`.

2. **`backend/services/portfolio_manager.py`** — change `extract_signals_from_analysis(analysis)` calls to `extract_all_signals(analysis, candidate=...)` where the candidate dict is available. For the buy path (`build_buy_candidates`), the candidate IS available (we just iterated over it). For the sell path (existing positions), candidate is None — old behavior preserved.

3. **Wire candidate dict through autonomous_loop** — when iterating analyze_tickers, attach the corresponding candidate dict to the analysis payload so `portfolio_manager` can pass it to `extract_all_signals`. Cleanest: pass `candidates_by_ticker` as a parallel dict alongside `candidate_analyses` to `decide_trades`.

4. **Frontend `AgentRationaleDrawer.tsx`** — add `quant: Signal[]` and `signal_stack: Signal[]` to the `Rationale.tree` interface. Add two new `<Layer title="Quant" .../>` and `<Layer title="Signal Stack" .../>` calls between the existing Analyst and Trader layers. Backend `group_signals_for_drawer` change is already paired with the `signal_attribution.py` edit.

5. **Tests** at `tests/services/test_signal_attribution.py`:
   - Trader extracts `full_report.analysis.reason` when present (lite path)
   - Trader still falls back to "Recommendation: <REC>" when no reason is anywhere
   - Risk extracts `risk_assessment.reason` (new)
   - `extract_quant_signals` produces Quant + SignalStack signals from a typical candidate dict
   - `extract_quant_signals` returns empty list when no signals are populated
   - `extract_all_signals` ordering: Analyst → Quant → SignalStack → Trader → Risk
   - `group_signals_for_drawer` routes Quant + SignalStack into the right buckets

## Out of scope

- New `paper_trading_analyses` BQ table (Phase 2 — needs migration --apply by operator)
- ALTER TABLE paper_trades adding columns (Phase 2 — same reason)
- `outcome_tracker.evaluate_recommendation` fallback to read lite analyses (depends on new BQ table — Phase 2)
- `build_situation_description` extension for richer BM25 context (Phase 2 — depends on new table fields being available)
- Backtest validation that the new context actually improves reflection quality (separate cycle)

## Why "no BQ migration" still solves the user's question

The `paper_trades.signals` column is `STRING` (JSON-serialized list). It's already wired through `bigquery_client.py::save_paper_trade`. After this cycle, every trade row's signals JSON will contain:
- An Analyst signal (when the full Gemini path ran)
- A Quant signal (momentum, RSI, vol, sector, composite_score)
- A SignalStack signal (conviction + news + sector_event tags from cycles 1-5)
- A Trader signal (Claude's actual reason)
- A RiskJudge signal (risk reasoning)

Future-learning SQL: `SELECT ticker, JSON_EXTRACT_ARRAY(signals, '$') AS signals FROM paper_trades WHERE created_at > '...';` returns the full structured rationale. The outcome_tracker can be enhanced in Phase 2 to consume this without any schema migration.

## Files modified

- `backend/services/signal_attribution.py` — 3 fallback-chain extensions + 2 new functions + group routing
- `backend/services/portfolio_manager.py` — switch to `extract_all_signals` with candidate
- `backend/services/autonomous_loop.py` — thread `candidates_by_ticker` through to `decide_trades`
- `frontend/src/components/AgentRationaleDrawer.tsx` — render Quant + Signal Stack layers
- `tests/services/test_signal_attribution.py` — NEW (10+ tests)

## Verification

The front-matter command builds a synthetic lite-analysis dict + screener candidate, calls `extract_all_signals`, and asserts:
- All 4 expected agents present (Quant, SignalStack, Trader, RiskJudge)
- Trader rationale extracts the actual Claude reason ("Q1 beat...")
- Risk rationale extracts the lite-shape `reason` key ("Strong momentum...")
- `group_signals_for_drawer` produces the new tree keys

This proves all three gaps are closed without needing real BQ writes.

## References

- `handoff/current/phase-23.1.7-research-brief.md` — full brief (627 lines, 5 sources read in full, gate_passed: true)
- `backend/services/signal_attribution.py` — current 146-line file
- `backend/services/autonomous_loop.py:466-578` — `_run_claude_analysis` return shape
- `backend/services/portfolio_manager.py:90-201` — buy-side `decide_trades`
- `frontend/src/components/AgentRationaleDrawer.tsx` — drawer with 4 fixed layers
