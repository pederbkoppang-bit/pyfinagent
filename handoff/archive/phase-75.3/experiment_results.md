# Experiment Results — Step 75.3: Audit75 S3 — MCP servers: fabricated-state removal + fail-closed publish path

- **Date:** 2026-07-20 · **Executor:** Opus 4.8
- **Findings closed:** gap4-01 (P1), gap4-04, gap4-05, gap4-06, gap4-07, gap4-08, gap4-09, security-05
- **Boundary honored:** paper-only; **no risk threshold edited** (a test pins all five current values); no BQ, no network in tests.

## Two spec corrections (verified by me before implementing)

The step spec would not have worked as written. Both corrections were flagged by the research gate and then **re-verified by me against live code** rather than taken on trust:

1. **Paths.** Spec says `scripts/mcp_servers/` — no such files. Real location is `backend/agents/mcp_servers/`, and `meta_coordinator.py` is under `backend/agents/`, not `backend/autoresearch/`. Confirmed by `ls`. Spec line numbers are all correct.
2. **`BacktestResult` field names.** Four of the five names in the spec do not exist. Confirmed by reading the dataclass at `backend/backtest/backtest_engine.py:112-123`: there is no `dsr`, `return_pct`, `max_drawdown_pct`, or `num_trades`. The real names are `aggregate_sharpe`, `aggregate_return_pct`, `aggregate_alpha_pct`, `aggregate_max_drawdown_pct`, `aggregate_hit_rate`, `total_trades`, `windows`. **DSR is not produced by the engine at all**, so the key is dropped rather than emitted as `0.0` — fabricating a DSR is precisely the failure class this step removes. A regression test (`test_backtest_result_lacks_the_fields_the_spec_named`) pins this so the correction cannot be quietly undone. No immutable criterion names these fields, so honoring reality amends nothing.

## What changed

```
 backend/agents/mcp_servers/backtest_server.py |  33 ++--
 backend/agents/mcp_servers/data_server.py     |  85 ++++++----
 backend/agents/mcp_servers/signals_server.py  | 219 +++++++++++++++++---------
 + backend/tests/test_phase_75_mcp_truth.py (new, 28 tests)
```

1. **gap4-01 (P1) — the fabricated portfolio.** `PaperTrader` has no `get_portfolio` (verified: only `get_or_create_portfolio:95` and `get_positions:115`), so every call raised `AttributeError` into a bare `except` that returned a plausible **$10,000 book** — which then sized and risk-checked real paper trades. Now uses the real API, mapping `total_nav`→`total_value`, `current_cash`→`cash`, and converting the **list** from `get_positions()` into a dict keyed by ticker. Both degraded paths are marked `stub: true` **and zeroed**: a zeroed book makes `size_position` return 0.0 via its own `equity<=0` guard and fails every notional gate closed, so the refusal is defense-in-depth rather than a single check. `publish_signal` refuses a degraded snapshot with `published:false, reason:degraded_portfolio`.
2. **gap4-05 — two inert gates.** An unresolved price left `proposed_notional` at 0.0, so the per-ticker, total-exposure **and** cash gates all passed trivially — a BUY of unknown size cleared every size limit. Now a BUY with an unresolvable unit price is rejected `unknown_price` before those gates. SELLs are deliberately exempt: they reduce exposure and are already bounded by the position check, so a missing mark must not trap an exit. Separately, `risk_check` read `current_drawdown_pct` from a key `get_portfolio` never set, leaving the drawdown breaker **permanently inert**; publish step 5 now wires `track_drawdown` in — note the key rename (`drawdown_pct` → `current_drawdown_pct`), which is the whole fix. Explicit `size_usd` is clamped to the hard cap instead of returning unbounded.
3. **gap4-06 — dedup inverting outcomes.** Root cause was a set/dict eviction asymmetry: an unbounded `_seen_signal_ids` set paired with a 50-entry FIFO response cache. After eviction the set still said "seen" while the cache missed, and the miss branch **synthesized `published=True`** — reporting a trade that may never have happened, for signals that had actually been *rejected*. Replaced with one `OrderedDict` as the single source of truth, so "seen" and "what happened" evict together. A cache miss now falls through and is processed normally. This also fixes an unbounded-set memory leak.
4. **gap4-04 — synthetic candidates.** Every `emit_candidates` candidate now carries `stub: true` + `PENDING_IMPLEMENTATION`, and `publish_signal` rejects stub provenance. The unreachable "real DSR" branch is deleted (the returns-by-variant dict was always empty and never written, so its guard could never be true). **Consumer contract preserved additively:** `scripts/harness/mcp_ab_test.py:470-490` asserts ≥5 candidates each carrying a `dsr` key, tied to a phase-3.7 immutable criterion — the count and the `dsr` field are untouched; `stub`/`reason` are added alongside. A test pins this.
5. **security-05 — SecretStr.** A non-empty `SecretStr` is truthy, so the `if not slack_token` guard never caught it, and passing the wrapper to `WebClient` yields `**********` rather than an error — a silent auth failure. Now unwrapped via `unwrap_secret`, with a local import preserving the lazy-import invariant above Step 7.
6. **gap4-07/08 — data_server.** `cached_macro` returns a **dict keyed by series_id**, so the old loop iterated *strings* and `item.get()` raised `AttributeError` into a swallow — meaning `macro://` returned empty **whenever data actually existed**. Now iterates `.items()` and filters to the requested series. The three hardcoded end-of-2025 cutoffs are `date.today()`-derived using the idiom already present in the same file; responses echo the effective range and the previously parsed-then-discarded `market` prefix.
7. **gap4-09 — backtest_server.** `.get()` on a dataclass raised `AttributeError` **after** the full walk-forward had run, so the tool always reported ERROR despite doing all the work. Now uses attribute access mirroring `meta_coordinator.run_proxy_validation`, passes the bq wrapper (the engine normalizes it), and drops the never-read `timeout_seconds`.

## Verification

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_mcp_truth.py -q
28 passed in 2.28s          # the step's immutable verification command, exit 0

$ .venv/bin/python -m pytest tests/test_mcp_servers.py -q
10 passed in 321.69s        # pre-existing MCP suite, no regression

$ uvx ruff check --select F821,F401,F811 <3 servers + new test>
signals_server.py:29 F401 `pathlib.Path` unused  -- PRE-EXISTING (reproduces at HEAD via
                                                    git show HEAD:<f> | ruff --stdin-filename)
                                                    all other files clean
```

## Mocking discipline (why these shipped undetected)

`tests/test_mcp_servers.py` contains **zero mocks** and asserts envelope shape rather than outcome — `assert "status" in result` passes when `status == "ERROR"`, and `assert result["series"] == "VIX"` passes on the AttributeError path. That is how a tool that *always* returned ERROR sat green. The new suite therefore uses `create_autospec(PaperTrader, instance=True)`, so calling a method the real class lacks raises rather than auto-vivifying — a bare `Mock()` would happily serve `.get_portfolio()` and let gap4-01 regress — and constructs a **real** `BacktestResult` instance.

## Honesty note

Three separate times this phase I have written a literal into a comment that the criterion's own source scan then flagged (the CGNAT pattern in 75.1, and here both the date cutoff and the dead-branch name). Each was caught by my own verification run before Q/A, but the pattern is worth naming: **when a criterion scans source for the absence of a string, that string must not survive in explanatory prose either.**

## Blast radius

The signals MCP server is **not on the live money path** — the autonomous loop calls `PaperTrader.execute_buy/execute_sell` directly and no live service imports `SignalsServer`, so fail-closed refusals cannot stop live trading. Consumers that should be re-run by the operator when convenient: `scripts/harness/mcp_ab_test.py` and the go_live drills (`position_limits:198` pins the strict-`>` boundary — thresholds are byte-identical; `first_week_monitoring:219` sets `server._peak_equity` directly, a seam the drawdown wiring preserves; `kill_switch:131`; `slack_signals_e2e` S9-S14 AST checks).

## Cycle-2 addendum (post Q/A wf_fcf4f363-339 CONDITIONAL)

The Q/A found the production logic correct on every criterion but capped the verdict
because two of my guards were proxies a realistic regression would evade: criterion 6
rested on `"unwrap_secret" in SIGNALS_SRC` (a string present on BOTH the import line and
the call site, so reverting only the call site kept it green), and criterion 3's two
behavioral halves were never driven through `publish_signal` at all.

Fixed test-only and additively (no production diff): a WebClient-observing test for the
SecretStr boundary, and two end-to-end dedup tests through `publish_signal`. Each was
**mutation-tested** to confirm it fails on the exact regression described -- including a
*reworded* `published=True` synthesis that provably evades the source scan (grep count 0,
old guard green) while failing both new behavioral tests. Suite 24 -> 27.

This is the third distinct self-inflicted issue of this phase and the most instructive:
the step's premise is that shape-asserting tests are why these bugs shipped, and I wrote
shape-asserting tests for the step's own criteria. Verifying that a guard *fails when the
thing it guards is broken* is now part of how I write regression suites, not an optional
extra.
