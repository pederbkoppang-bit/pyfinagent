# Experiment Results — phase-8.5 / 8.5.4 REMEDIATION v1

Immutable `test -f && grep -q 'sharpe.*dsr.*pbo.*max_dd.*profit_factor.*cost.*realized_pnl'` → exit 0. Column count = 11 (not 12). Seed baseline values trace to `optimizer_best.json` (Sharpe=1.1704633..., DSR=0.9525811...). 3/3 literal success_criteria PASS. Regression 152/1 unchanged.

Researcher brief: 5 sources in full (DSR canonical paper, Wikipedia, Balaena Quant, QuantConnect docs, llm-quant GitHub). Column-count correction from 12→11 disclosed.
