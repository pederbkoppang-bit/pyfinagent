# Live-check placeholder -- phase-25.D6

**Step:** 25.D6 -- Planner plateau-detection lock-file enforcement
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Trigger 10 consecutive discards; confirm lock file created and next run blocked"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_D6.py`)
- 6 behavioral round-trips covering: lock-write shape, 409 guard with detail payload, clear-lock + audit JSONL, cleared-sentinel handling, corrupt-JSON fail-open, 404 when no lock.
- Backend AST clean for both touched files.

## Post-deployment operator workflow

### To verify lock fires (synthetic stress)
1. Restart backend so the new threshold + helpers are loaded:
   ```
   source .venv/bin/activate && python -m uvicorn backend.main:app --reload --port 8000
   ```
2. Trigger 10 bad experiments. Easiest path: run the optimizer with `use_llm=false` for a short max_iterations against a degenerate strategy slice. Once 10 consecutive iterations all discard / dsr_reject / crash, the lock fires.
3. Verify the lock file:
   ```
   cat handoff/locks/optimizer_plateau.lock
   ```
   Expected shape:
   ```json
   {
     "created_at": "2026-05-13T...",
     "trigger": "plateau_10_discards",
     "consecutive_discards": 10,
     "run_id": "<run_id>",
     "cleared_at": null
   }
   ```
4. Try to start another run:
   ```
   curl -s -X POST "http://localhost:8000/api/backtest/optimize" \
     -H "Authorization: Bearer $TOKEN" \
     -H "content-type: application/json" \
     -d '{"max_iterations": 10, "use_llm": false}'
   ```
   Expected: `409` with body:
   ```json
   {
     "detail": {
       "error": "PlateauLockPresent",
       "message": "Optimizer halted after 10 consecutive discards (run_id=...). Strategy switch required. Call DELETE /api/backtest/optimize/lock to acknowledge and resume.",
       "lock": { ... }
     }
   }
   ```

### To clear (operator acknowledgment)
```
curl -s -X DELETE "http://localhost:8000/api/backtest/optimize/lock" \
  -H "Authorization: Bearer $TOKEN" | jq .
```
Expected response: `{"status": "cleared", "lock": {..., "cleared_at": "..."}}`.

Audit row appended to `handoff/audit/optimizer_plateau_audit.jsonl`:
```
{"created_at": "...", "trigger": "plateau_10_discards", "consecutive_discards": 10, "run_id": "...", "cleared_at": "..."}
```

After clearing, the next `POST /api/backtest/optimize` succeeds again.

## Closes audit basis
phase-24.6 F-5 RESOLVED. The 62-experiment plateau cannot recur silently:
- After N=10 consecutive discards/dsr_rejects/crashes, the loop halts.
- The lock-file survives backend restarts (file-based, not in-memory).
- Operators must explicitly acknowledge via DELETE before optimization can resume.

## Threshold rationale
N=10 matches:
- Keras `ReduceLROnPlateau` default patience=10.
- Optax patience range 5-15.
- The live-check requirement ("trigger 10 consecutive discards").
- Second tier above the existing `think_harder >= 5` softer signal at `quant_optimizer.py:205`.

**Audit anchor for next bucket:** 25.A10 (Alpaca MCP smoke test) OR 25.B (cosmetic-patch cleanup).
