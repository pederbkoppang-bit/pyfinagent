# live_check 75.3 -- verbatim evidence (2026-07-20)

## Immutable verification command -- exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_mcp_truth.py -q
...........................                                              [100%]
28 passed in 2.28s
```

## git diff --stat (change surface)

```
 backend/agents/mcp_servers/backtest_server.py |  33 ++--
 backend/agents/mcp_servers/data_server.py     |  85 ++++++----
 backend/agents/mcp_servers/signals_server.py  | 219 +++++++++++++++++---------
 3 files changed, 222 insertions(+), 115 deletions(-)
?? backend/tests/test_phase_75_mcp_truth.py   (new, 28 tests)
```

## No-regression evidence

```
$ .venv/bin/python -m pytest tests/test_mcp_servers.py -q
..........                                                               [100%]
10 passed in 321.69s (0:05:21)
```

## Lint gate (HEAD-baseline separated)

```
$ uvx ruff check --select F821,F401,F811 backend/agents/mcp_servers/{signals,data,backtest}_server.py backend/tests/test_phase_75_mcp_truth.py
F401 `pathlib.Path` imported but unused --> backend/agents/mcp_servers/signals_server.py:29:21

$ git show HEAD:backend/agents/mcp_servers/signals_server.py | uvx ruff check --select F821,F401,F811 --stdin-filename backend/agents/mcp_servers/signals_server.py -
F401 `pathlib.Path` imported but unused --> backend/agents/mcp_servers/signals_server.py:29:21
  => PRE-EXISTING, identical at HEAD. data_server / backtest_server / new test file: clean.
```

## BOUNDARY proof -- risk thresholds byte-untouched

`test_thresholds_are_unchanged` pins the live values read back from
get_risk_constraints():

```
max_position_pct     ==   5.0
max_position_usd     == 1000.0
max_drawdown_pct     == -15.0
drawdown_warning_pct ==  -5.0
drawdown_derisk_pct  == -10.0
```

## Per-criterion test evidence (all offline: no BQ, no network)

- c1 fabricated portfolio: source no longer contains `paper_trader.get_portfolio(`;
  PaperTrader really lacks that method (premise pinned); real NAV/cash/positions flow
  through (25,000 / 5,000 / AAPL keyed from the LIST); BOTH degraded paths return
  stub:true with total_value == cash == 0.0; publish_signal on a degraded snapshot
  returns published=false + degraded reason AND `trader.execute_buy.assert_not_called()`.
- c2 risk gates: BUY with unresolved price -> allowed=False, conflicts contains
  'unknown_price'; SELL with unresolved price is NOT trapped by it (exits stay open);
  a -16% drawdown book blocks BUYs; explicit size_usd=999,999 clamps exactly to
  min(equity*pct, max_usd); all five thresholds pinned unchanged.
- c3 dedup: a remembered rejection replays published=False (never inverted); after
  forced eviction the entry is simply absent so nothing can be synthesized; seen-state
  and outcome evict together (one OrderedDict, no unbounded set); source no longer
  contains the `resp["published"] = True` synthesis.
- c4 stub provenance: emit_candidates block carries `"stub": True` +
  PENDING_IMPLEMENTATION while RETAINING `"dsr": dsr` and `n = max(5, int(n))`
  (phase-3.7 A/B consumer contract, additive change only); the dead real-DSR branch
  name is gone from source; publish_signal rejects a stub-marked signal with
  reason == 'stub_provenance'.
- c5 data_server: no end-of-2025 literal anywhere in the file; get_macro over a fake
  dict-shaped cache returns the requested series' {date,value} (the old dict-iteration
  bug would return an empty list here); get_prices/get_fundamentals cutoffs both equal
  date.today().isoformat().
- c6 backtest_server + SecretStr: a real BacktestResult dataclass yields status=PASS
  with sharpe/return_pct/max_drawdown_pct/total_trades extracted (was ALWAYS "ERROR"
  before); no fabricated `dsr` key; a regression test pins that the four spec-named
  fields genuinely do not exist on the dataclass; SecretStr proven truthy and
  str()-masking, with unwrap_secret returning the real token.

## $0 / no-live-impact confirmation

No LLM call added, removed or repointed. The signals MCP server is not on the live
money path (the autonomous loop calls PaperTrader.execute_buy/execute_sell directly;
no live service imports SignalsServer), so the new fail-closed refusals cannot block
live trading. Operator may re-run scripts/harness/mcp_ab_test.py and the go_live drills
at convenience.

## Cycle-2: mutation evidence for the two new behavioral guards

```
MUTATION 1 -- revert ONLY the unwrap call site (import left in place, so the
              old source-scan guard "unwrap_secret" in SIGNALS_SRC stays TRUE):
  $ grep -c unwrap_secret backend/agents/mcp_servers/signals_server.py
  1                                        <-- import still there
  $ pytest backend/tests/test_phase_75_mcp_truth.py -q
  FAILED test_secretstr_token_reaches_webclient_unwrapped_end_to_end
  1 failed, 26 passed                      <-- CAUGHT

MUTATION 2 -- reinstate the synthesized published=True on cache miss, REWORDED
              as resp.update({"published": True, ...}):
  $ grep -c 'resp\["published"\] = True' backend/agents/mcp_servers/signals_server.py
  0                                        <-- source-scan guard EVADED (stays green)
  $ pytest backend/tests/test_phase_75_mcp_truth.py -q
  FAILED test_evicted_refire_of_a_rejection_reports_published_false_end_to_end
  FAILED test_a_freed_up_rejection_can_be_retried_end_to_end
  2 failed, 25 passed                      <-- CAUGHT by behavior, not by grep

Both mutations reverted. Final: 27 passed in 1.68s.
Production diff unchanged this cycle (222 insertions / 115 deletions across the
3 servers, identical to cycle 1) -- the fix was entirely in the test suite.
```

## Cycle-3: mutation evidence for the emit_candidates behavioral guard

```
MUTATION -- keep the '"stub": True' literal in the candidate dict, then
            `for _c in candidates: _c.pop("stub", None)` immediately before return:
  $ grep -c '"stub": True' backend/agents/mcp_servers/signals_server.py
  3                                    <-- literal present; old source-scan guard GREEN
  $ pytest backend/tests/test_phase_75_mcp_truth.py -q
  FAILED test_emit_candidates_really_emits_stub_marked_candidates
  1 failed, 27 passed                  <-- CAUGHT by the emitted payload, not by grep

Mutation reverted. Final: 28 passed in 2.28s.
Production diff STILL unchanged from cycle 1: 222 insertions / 115 deletions
across the 3 servers. Cycles 2 and 3 were test-only.
```
