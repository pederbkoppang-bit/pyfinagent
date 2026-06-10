# Live-check placeholder -- phase-25.S.1

**Step:** 25.S.1 -- Per-call ticker tagging in llm_call_log + cost_tracker for exact per-ticker attribution
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "After running migration --apply and a Claude-mode analysis, BQ llm_call_log table has non-NULL ticker for new rows and can be GROUP BY ticker"

## Pre-deployment evidence
- 7/7 verifier PASS.
- 2 LIVE behavioral round-trips:
  - `CostTracker.record(ticker="AAPL")` -> entry.ticker == "AAPL".
  - `log_llm_call(ticker="MSFT")` -> buffered row carries ticker == "MSFT".
- AST clean on all 5 touched .py files.

## Post-deployment operator workflow
1. Pull main + run the migration once (idempotent):
   ```
   git pull origin main
   source .venv/bin/activate
   python scripts/migrations/add_ticker_to_llm_call_log.py --apply
   ```
   Expected: `APPLIED: sunny-might-477607-p8.pyfinagent_data.llm_call_log now has ticker column`
2. Restart backend:
   ```
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
3. Run a Claude-mode analysis WITH ticker threaded (note: caller-adoption
   is the 25.S.1.1 follow-up; until then `_ticker` is None and the column
   stays NULL on new rows). To exercise the wire end-to-end manually:
   ```
   python -c "
   from backend.agents.cost_tracker import CostTracker
   class _U: prompt_token_count=100; candidates_token_count=50; total_token_count=150; cache_creation_input_tokens=0; cache_read_input_tokens=0
   class _R: usage_metadata=_U()
   ct = CostTracker()
   e = ct.record('Insider', 'claude-sonnet-4-6', _R(), ticker='NVDA')
   print('entry.ticker =', e.ticker)
   print('total cost =', ct.total_cost)
   "
   ```
4. Verify the BQ column exists:
   ```sql
   SELECT column_name, data_type, is_nullable
   FROM `sunny-might-477607-p8.pyfinagent_data.INFORMATION_SCHEMA.COLUMNS`
   WHERE table_name = 'llm_call_log' AND column_name = 'ticker';
   ```
5. Once 25.S.1.1 lands and callers thread `_ticker`, the per-ticker query unlocks:
   ```sql
   SELECT ticker,
          COUNT(*) AS calls,
          SUM(input_tok) AS total_input_tok
   FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
   WHERE DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
     AND ticker IS NOT NULL
   GROUP BY ticker ORDER BY total_input_tok DESC;
   ```

## North-star calculus
This unlocks exact per-ticker `profit_per_llm_dollar`. The meta-evolution
layer can use this to auto-prune unprofitable tickers (north-star: "shift
strategy to whichever is making the most money" rendered at the ticker
level instead of just strategy level). Compounds with 25.C9.1 + 25.D9.1
cost-reduction so the cost-denominator is both granular AND cheap.

## Closes audit basis
25.S follow-up RESOLVED (the deferred per-call ticker tagging). The
proportional-by-trade-count cost split in 25.S becomes optional; exact
per-call attribution is now possible once callers adopt the side-channel.

**Audit anchor for next bucket:** 25.S.1.1 (caller adoption -- thread `_ticker` in run_*_agent calls), 25.S.2 (Gemini instrumentation), 25.D9.2 (cache_control on doc block), 25.C9.2 (batch hot-path refactor).
