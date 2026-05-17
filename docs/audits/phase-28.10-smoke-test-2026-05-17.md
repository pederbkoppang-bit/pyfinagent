# phase-28.10 Smoke Test — 2026-05-17

**Step:** phase-28.10 (Opportunistic insider buying signal — CMP classifier)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/insider_signal_screen.py').read()); from backend.services.insider_signal_screen import fetch_insider_signals; print('importable')" && grep -q 'insider_signal_screen_enabled' backend/config/settings.py
importable
```

Exit 0. **PASS.**

## Test 2: CMP classifier synthetic smoke

10-trade fixture across 3 insiders:
- Insider A (4yr history, May trades every year): 2026-05-17 → routine ✓
- Insider B (4yr history, never May): 2026-05-17 → opportunistic ✓
- Insider C (<3yr history): all trades → unknown (cold-start guard) ✓

Boost tiers at $500K (moderate, 1.04) / $2M (strong, 1.07) thresholds — confirmed at 5 aggregate values.

Apply identity paths (missing-ticker, empty dict, None signals) all PASS.

## Test 3: Q/A subagent verdict (19 deterministic checks, 21 unit assertions)

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "checks_run": 19
}
```

Q/A explicitly verified:
- CMP rule no-gaps (T1c — only 2 of 3 prior years → False)
- Cold-start <3y guard (T2a, T3d, T3e)
- BUY-only filter (sells excluded — CMP 82bps is buy-side)
- Default-OFF discipline
- Cost-bounding (2*top_n cap, Sem(3), 0.5s throttle)
- Honest deferral of live SEC fetch

## Stack traces / failures

None.

## Conclusion

CMP insider classifier implemented end-to-end, unit-tested at 100% public-surface coverage, Q/A-verified with 21 assertions. Live SEC fetch deferred (rate-limited; underlying `get_insider_trades` is production-tested). Default OFF.

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.10.md`, `phase-28.10-research-brief.md`
- `docs/design/phase-28.10-insider-signal.md`
- `backend/services/insider_signal_screen.py`, `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`
