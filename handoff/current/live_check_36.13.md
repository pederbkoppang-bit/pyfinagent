# live_check — masterplan step 36.13

Immutable requirement: *"Verbatim test output for the reproduce-then-fix pair on BOTH the paused and
the disarmed case, plus the re-derived call-site grep output showing every non-test `execute_buy`
caller and its disposition."*

## (a) PAUSED — reproduce, then fix

PRE-FIX (gate reverted in memory to the pre-36.13 shape — no kill-switch reference in `execute_buy`):
```
>       assert result is None, (
E       AssertionError: a paused kill switch must refuse the order -- pre-fix this returned a trade
E       assert {'action': 'BUY', 'analysis_id': '', 'created_at': '...', 'price': 100.0, ...} is None
FAILED ...::test_phase_36_13_a_paused_book_refuses_a_buy
```
POST-FIX: returns `None`, `insert_paper_trade` call count **0**, `buy_rejections[0].reason ==
"kill_switch_paused"`, and the ERROR log carries `REFUSING BUY`.

## (b) DISARMED / baselines lost — reproduce, then fix

PRE-FIX:
```
>       assert result is None
E       AssertionError: assert {'action': 'BUY', ...} is None
FAILED ...::test_phase_36_13_lost_baselines_refuse_a_buy
2 failed, 8 deselected in 0.87s
```
POST-FIX: refused with `kill_switch_baselines_lost`, naming the actual snapshot values
(`sod_nav=None, peak_nav=None`). Book NOT paused in this case, so the refusal is attributable to the
baselines and not to the pause.

## (c) Fail-closed, and the case that must NOT refuse

- Unreadable state → refused, `kill_switch_unreadable` (criterion 6).
- **Stale-but-present anchor → NOT refused.** `evaluate_breach` reports `armed: False` for an anchor
  merely from yesterday; keying the gate on that would refuse every BUY each morning until the daily
  roll — the money-path regression phase-36.9's cycle-1 evaluator caught. Pinned by
  `test_phase_36_9..._a_STALE_anchor_does_NOT_refuse_a_buy`, which asserts the premise is real by
  calling `_sod_date_is_stale` before relying on it.

## (d) Re-derived call-site inventory

Full capture: `handoff/current/captures_36.13/call_site_inventory.txt`.

```
$ grep -rn '\.execute_buy(' --include='*.py' backend scripts | grep -v /tests/
backend/agents/mcp_servers/signals_server.py:444     -> GATED (the bypass this step closes)
backend/services/autonomous_loop.py:236              -> GATED (already behind the cycle check)
scripts/smoketest_stages_5_through_13.py:225         -> DELIBERATELY BYPASSING (injected state)
scripts/go_live_drills/zero_orders_drill.py:127      -> DELIBERATELY BYPASSING (injected state)

$ grep -rn 'kill_switch_state=' --include='*.py' backend | grep -v /tests/
(none -- production code cannot inject)
```

## (e) The drills, against a HEAD baseline

I broke both, caught it, and fixed it — recorded because the failure is the useful part.

| | HEAD (worktree) | this branch |
|---|---|---|
| `zero_orders_drill.py` | PASS | **PASS** |
| smoketest Stage 8 | PASS | **PASS** (was FAIL mid-step — my stub lacked baselines) |
| smoketest overall | 8/9, Stage 12 FAIL | **8/9, Stage 12 FAIL** (pre-existing) |

Stage 12's failure was established as pre-existing by running the smoketest in a `git worktree` at
HEAD, not by assumption and not by `git stash` (active hooks).

## Mutation matrix — 9 killed / 0 survived at baseline `203 passed`

`handoff/current/captures_36.13/mutation_matrix.txt`. Licenses *"these 9 were killed at this
baseline"* and nothing more. First run showed 2 survivors: one was a genuinely missing guard (the
drill injection), the other was an artifact of scoping the matrix to a single file — at immutable
scope it is killed by 14 tests in 36.9 + 36.12.

## Verification

```
$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or signals_server'
203 passed, 1 skipped, 2103 deselected
```

## Do-no-harm

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` at every measurement point,
including both live script runs and the full matrix. `:8000` GET-only, never restarted or POSTed to.
`:3000` never driven. No threshold in the diff. No peak reset. The drills ran against their own
stubbed BigQuery; the live book was never touched.

**NOT LIVE:** needs a backend restart the operator has not authorized.
