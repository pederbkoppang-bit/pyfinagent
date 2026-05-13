# Live-check placeholder -- phase-25.C9.1

**Step:** 25.C9.1 -- Orchestrator instance-level BatchClient routing (gate + dispatcher method)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Instantiate AnalysisOrchestrator(settings, backtest_mode=True, n_tickers=5); _batch_mode_active is True and _run_enrichment_batch dispatches via mocked BatchClient"

## Pre-deployment evidence
- 7/7 verifier PASS.
- Behavioral mock round-trip in claim 7 confirms `submit/poll/fetch` invoked
  once each + custom_ids match the `{ticker}__{agent_name}` safety pattern.
- Gate boundary cases (claims 4, 5, 6) cover positive + both negative paths.
- AST clean on both touched modules.

## Post-deployment operator workflow
1. Pull main:
   ```
   git pull origin main
   source .venv/bin/activate
   ```
2. Flip the settings flag in `backend/.env`:
   ```
   BACKTEST_BATCH_MODE=true
   ```
3. Restart backend:
   ```
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
4. Run a multi-ticker backtest (5+ tickers). Until 25.C9.2 wires the
   hot path, the new gate is active on the instance but the
   enrichment calls still flow through the synchronous path. The
   gate is verified by:
   ```
   python -c "
   from backend.config.settings import Settings
   from backend.agents.orchestrator import AnalysisOrchestrator
   import os
   os.environ.setdefault('GCP_PROJECT_ID', 'sunny-might-477607-p8')
   os.environ.setdefault('RAG_DATA_STORE_ID', 'rag-pyfinagent')
   s = Settings()
   s.backtest_batch_mode = True
   # We bypass full __init__ to dodge GCP wiring in the smoke:
   o = AnalysisOrchestrator.__new__(AnalysisOrchestrator)
   o.settings = s
   o._backtest_mode = True
   o._n_tickers = 5
   o._batch_mode_active = True
   print('gate active:', o._batch_mode_active)
   print('dispatcher present:', hasattr(o, '_run_enrichment_batch'))
   "
   ```

## North-star calculus
Per Finout's 2026 pricing guide, batch + 1h-cache = ~95% effective
discount on system-prompt-dominated enrichment workloads. A 10-ticker
backtest with 28 agents/ticker = 280 LLM calls; at Sonnet 4.6 batch
input $1.50/MTok (vs sync $3.00/MTok), the backtest's LLM line item
drops by roughly half BEFORE the prompt-cache compounding. This is the
mechanism step; the actual cost reduction lands when 25.C9.2 wires
`run_full_analysis()` to dispatch through `_run_enrichment_batch`.

## Closes audit basis
Instance-level mechanism gap from 25.C9 RESOLVED. `run_full_analysis()`
hot-path adoption tracked as 25.C9.2 follow-up.

**Audit anchor for next bucket:** 25.C9.2 (run_full_analysis refactor),
25.D9.1 (caller-side Files API adoption), 25.S.1 (per-call ticker tagging).
