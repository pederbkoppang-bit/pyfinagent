# Contract -- FOLLOW-UP: Risk Judge observability + brief envelope (task #47)

## Research gate

- Researcher spawn: 2026-04-24. Brief at `handoff/current/observability-patch-research-brief.md`.
- JSON envelope: tier=simple, external_sources_read_in_full=5 (floor 5), urls_collected=12, recency_scan_performed=true, internal_files_inspected=6, gate_passed=true.
- Internal confirmation: (a) `backend/services/portfolio_manager.py:153-158` stores `risk_judge_decision` on the candidate but never logs it; (b) `portfolio_manager.py:176-177` silently `continue`s on `buy_amount < 50`; (c) `handoff/current/virtual-fund-readiness-research-brief.md` ends at line 97 with no JSON envelope.
- Style precedent: `autonomous_loop.py:248` logger.warning("Dropping BUY for {ticker}: price=...") landed in BLOCKER-1 -- same class of silent-drop guard, match style.

## Hypothesis

If the Monday autonomous cycle produces zero trades again, today we have no way to distinguish (a) Claude's new prompt still returns HOLD too often vs (b) Claude returns BUY but Risk Judge REJECTs it vs (c) Risk Judge approves but position size falls below $50 minimum. The fix is two well-placed log lines that make each of those failure modes observable in the backend log.

## Planned change (MINIMUM scope)

### 1. `backend/services/portfolio_manager.py` -- two log lines

**Gap (a):** inside the candidate-build loop, right after `buy_candidates.append(...)` at line 158, log when Risk Judge returned anything other than APPROVE_FULL:

```python
decision = risk_assessment.get("decision", "") or ""
if decision and decision != "APPROVE_FULL":
    logger.info(
        "buy_candidate risk_judge decision=%s ticker=%s position_pct=%s final_score=%s",
        decision, ticker, position_pct, round(float(final_score or 0), 3),
    )
```

**Gap (b):** replace the silent `if buy_amount < 50: continue` at line 176-177 with a logged skip:

```python
if buy_amount < 50:
    logger.warning(
        "Skipping BUY %s: buy_amount=%.2f below $50 minimum (nav=%.2f position_pct=%s available_cash=%.2f)",
        cand["ticker"], buy_amount, nav, position_pct, available_cash,
    )
    continue
```

### 2. `handoff/current/virtual-fund-readiness-research-brief.md`

Append the mandatory JSON envelope per `.claude/rules/research-gate.md`. The brief content is already good; the envelope was omitted. Exact content:

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 5,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```

## NOT in scope this cycle

- Changing the REJECT -> `position_pct=0` -> 10% default-fallback behavior at `portfolio_manager.py:171`. That's an actual semantic bug (REJECT is silently overridden) but fixing it changes trading behavior. Observability first; semantic fix is a separate cycle.
- Changing the $50 minimum position threshold.
- Restarting the backend (already restarted post-BLOCKER-1; will pick up these log lines on the NEXT restart or next cycle — we'll restart after commit).

## Immutable success criteria

1. `grep -c "buy_candidate risk_judge decision" backend/services/portfolio_manager.py` >= 1.
2. `grep -c "below \$50 minimum" backend/services/portfolio_manager.py` >= 1 (the log message literal).
3. `grep -c "gate_passed" handoff/current/virtual-fund-readiness-research-brief.md` >= 1.
4. `python -c "import ast; ast.parse(open('backend/services/portfolio_manager.py').read())"` exits 0.
5. `python -c "from backend.services import portfolio_manager; print('ok')"` prints ok.
6. Drill re-run: `python scripts/go_live_drills/zero_orders_drill.py` still prints PASS (no regression in the synthetic BUY path).
7. A synthetic test that invokes `decide_trades` with (a) Risk Judge decision=REJECT and (b) nav small enough that buy_amount < 50 produces at least one log record each (captured via caplog/records).

## Verification command (Q/A reproduces)

```bash
source .venv/bin/activate
grep -c "buy_candidate risk_judge decision" backend/services/portfolio_manager.py
grep -c "below \$50 minimum" backend/services/portfolio_manager.py
grep -c "gate_passed" handoff/current/virtual-fund-readiness-research-brief.md
python -c "import ast; ast.parse(open('backend/services/portfolio_manager.py').read())" && echo SYNTAX_OK
python -c "from backend.services import portfolio_manager; print('IMPORT_OK')"
python scripts/go_live_drills/zero_orders_drill.py
# Synthetic log-capture test -- see experiment_results for one-shot Python.
```

## References

- `handoff/current/observability-patch-research-brief.md` (research deliverable)
- `backend/services/portfolio_manager.py` (target file)
- `handoff/current/virtual-fund-readiness-research-brief.md` (envelope append target)
- `backend/services/autonomous_loop.py:248` (style precedent from BLOCKER-1)
- `backend/agents/schemas.py:118` (Risk Judge decision enum)
