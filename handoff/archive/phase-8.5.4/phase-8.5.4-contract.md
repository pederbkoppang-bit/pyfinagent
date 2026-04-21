# Sprint Contract — phase-8.5 / 8.5.4 (Evaluator + multi-metric results.tsv)

**Step id:** 8.5.4 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Hypothesis
Ship `backend/autoresearch/results.tsv` with a stable 12-column header containing all 7 required metrics, plus one seed row (MDA baseline) so downstream readers don't error on empty files.

## Immutable criterion
- `test -f backend/autoresearch/results.tsv && head -1 ... | grep -q 'sharpe.*dsr.*pbo.*max_dd.*profit_factor.*cost.*realized_pnl'`

## Plan
1. Write TSV with header + seed row.
2. Verify command.
3. Regression.
4. Q/A. Log. Flip.

## Out of scope
- No evaluator module (TSV-write scaffold ships; evaluator integration is phase-8.5.5/8.5.7).
- ASCII-only.
