# FOLLOW-UP Risk Judge observability + brief envelope -- Experiment results

## What was built

Two pre-Monday observability patches before the $10K virtual fund cycle
fires at 14:00 local:

1. `backend/services/portfolio_manager.py` -- two `logger.*` lines that
   turn previously-silent gate events into structured log records
   Monday's cycle can be diagnosed against.
2. `handoff/current/virtual-fund-readiness-research-brief.md` --
   appended the mandatory research-gate JSON envelope (protocol
   compliance per `.claude/rules/research-gate.md`).

## Why

Q/A on the virtual-fund-readiness cycle flagged three possible
explanations if Monday's paper cycle still produces 0 trades:
(a) Claude still returns HOLD too often, (b) Claude returns BUY but
Risk Judge REJECTs, (c) Risk Judge approves but position size falls
below the $50 minimum. Today only (a) is observable (the BLOCKER-1
drop-log). (b) and (c) are silent. These two log lines close that gap.

## Files changed

1. `backend/services/portfolio_manager.py` (two inserts, same file):
   - After `buy_candidates.append(...)`: `logger.info("buy_candidate
     risk_judge decision=%s ticker=%s ...")` -- fires on any Risk Judge
     decision that is not `APPROVE_FULL` (REJECT / APPROVE_REDUCED /
     APPROVE_HEDGED).
   - Before `if buy_amount < 50: continue`: `logger.warning(
     "Skipping BUY %s: buy_amount=%.2f below $50 minimum ...")`.
2. `handoff/current/virtual-fund-readiness-research-brief.md` (append
   JSON envelope after the checklist).

## Verification command output (verbatim)

```
$ grep -c "buy_candidate risk_judge decision" backend/services/portfolio_manager.py
1
$ grep -c 'below \$50 minimum' backend/services/portfolio_manager.py
1
$ grep -c "gate_passed" handoff/current/virtual-fund-readiness-research-brief.md
1
$ python -c "import ast; ast.parse(open('backend/services/portfolio_manager.py').read())" && echo SYNTAX_OK
SYNTAX_OK
$ python -c "from backend.services import portfolio_manager; print('IMPORT_OK')"
IMPORT_OK

$ python scripts/go_live_drills/zero_orders_drill.py
step1: decide_trades emitted BUY for AAPL amount=$1000.00
step2: paper_trades row written: ticker=AAPL action=BUY qty=5.128205 price=195.0
PASS

$ <C7 synthetic log-capture test>
reject_log_captured=True
below50_log_captured=True
C7_PASS
```

All 7 contract criteria green including the synthetic log-capture
test that proves the two new log lines actually emit under the
two failure modes.

## Success-criteria coverage

| # | Criterion | Evidence |
|---|---|---|
| 1 | `buy_candidate risk_judge decision` line present | PASS (count 1) |
| 2 | `below $50 minimum` line present | PASS (count 1) |
| 3 | readiness brief has `gate_passed` envelope | PASS (count 1) |
| 4 | `ast.parse(portfolio_manager.py)` exits 0 | PASS |
| 5 | `portfolio_manager` module imports clean | PASS |
| 6 | zero_orders drill no-regression | PASS (still prints PASS) |
| 7 | Synthetic log-capture: REJECT + below-$50 both logged | PASS |

## Scope discipline

- Did NOT change the REJECT -> `position_pct=0` -> 10% default-fallback
  at `portfolio_manager.py:171`. That is a semantic bug (REJECT is
  silently overridden) tracked as a separate cycle; observability
  first, semantic fix second.
- Did NOT change the $50 minimum threshold.
- Did NOT restart the backend automatically -- the user will want to
  restart manually after commit so the log lines are loaded before
  Monday's cycle.

## Post-cycle action user should take before Monday

After Main commits + pushes, the user should run:
```
launchctl stop com.pyfinagent.backend && launchctl start com.pyfinagent.backend
ps -eo pid,lstart,command | grep -E "uvicorn.*backend\.main" | grep -v grep
```
and confirm the new PID's lstart is AFTER the commit timestamp.
Then Monday's cycle will log structured records for any REJECT or
sub-$50 skip, making the behavior fully observable.
