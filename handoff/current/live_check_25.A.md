# Live-check placeholder -- phase-25.A

**Step:** 25.A -- Decouple RiskJudge with independent LLM call in lite path
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "BQ paper_trades signals column shows distinct trader_rationale vs risk_rationale text on next cycle"

## Pre-deployment evidence
- 10/10 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_A.py`)
- Behavioral round-trip (claim 6) executes `_run_claude_analysis(TEST, settings_mock)` with mocked anthropic + yfinance and asserts:
  - exactly 2 `client.messages.create` calls (trader + risk judge)
  - `risk_assessment.reasoning` is distinct from `analysis.reason`
  - `recommended_position_pct > 0`
- Behavioral fallback (claim 8) covers malformed risk JSON -> `APPROVE_REDUCED` default w/ position_pct > 0.
- Consumer bridge (claim 9): `signal_attribution.extract_signals_from_analysis` over the new return shape emits a `RiskJudge` row with `weight > 0` and rationale != trader rationale; `lite_path` flag NOT set (the `is_lite_dup` cosmetic patch auto-resolves to False, freeing 25.B to remove it).
- Backend AST clean for `autonomous_loop.py`.

## Post-deployment operator workflow (capture-after-next-cycle)
1. Trigger the next autonomous cycle (or wait for the scheduled daily run):
   ```
   curl -s -X POST http://localhost:8000/api/paper-trading/run-now \
     -H "Authorization: Bearer $TOKEN"
   ```
2. Query the most recent paper_trades row's `signals` column:
   ```sql
   SELECT ticker, signals
   FROM `sunny-might-477607-p8.pyfinagent_data.paper_trades`
   ORDER BY created_at DESC
   LIMIT 1;
   ```
3. The `signals` array must contain BOTH:
   - a `Trader` row with the trader's rationale,
   - a `RiskJudge` row with a DIFFERENT rationale + `weight > 0` (the `recommended_position_pct`).
4. Verify the cosmetic placeholder string is NOT present:
   - Bad (pre-fix): "Lite-path: Risk Judge inherited Trader's reasoning; no independent risk debate ran for this analysis."
   - Good (post-fix): substantive risk-axis reasoning text.

## Cost note
Two Anthropic Sonnet calls per ticker now. Estimated marginal cost ~$0.003/ticker; total per ticker ~$0.004 vs the existing $0.01 accounting field. No budget configuration change required.

## Downstream
Unblocks 25.B (remove the cosmetic aliasing patch at `signal_attribution.py:131-154` -- now inert by data).

**Audit anchor for next bucket:** 25.B (cosmetic-patch removal) -- depends on 25.A done.
