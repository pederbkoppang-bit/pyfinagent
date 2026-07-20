# Contract — Step 75.3: Audit75 S3 — MCP servers: fabricated-state removal + fail-closed publish path

- **Step id:** 75.3 (phase-75, P0, executor: opus-4.8/xhigh)
- **Date:** 2026-07-20
- **Boundary:** paper-only; kill-switch / DSR / PBO gate **THRESHOLDS byte-untouched** — plumbing and fail-closed additions only.
- **Findings remediated:** gap4-01 (P1), gap4-04, gap4-05, gap4-06, gap4-07, gap4-08, gap4-09, security-05 (P2)

## Research-gate summary

Gate PASSED (moderate tier, `handoff/current/research_brief_75.3.md`): 7 sources read in full (MCP spec, Stripe idempotency, MITRE CWE-636, Pydantic, CPython dataclasses, CPython unittest.mock, arXiv:2607.11098 AgentCheck), 27 URLs, 3-variant discipline, recency scan, 18 internal files line-anchored.

External consensus is one-directional: **mark degraded state, refuse the privileged action, never substitute a plausible default.** CWE-636 names this exact anti-pattern ("fail functional" instead of "fail safe"). The MCP spec puts business-logic errors in the result via `isError` rather than raising. Stripe replays the **true** terminal outcome (including 500s) and never synthesizes success, and deliberately leaves validation failures unpersisted so they stay retryable. AgentCheck measures that agent failures are "silent, confident use of incorrect tool outputs rather than crashes" — which is precisely this step's fault class.

### TWO SPEC CORRECTIONS (verified by me independently before planning)

1. **Paths.** The step spec says `scripts/mcp_servers/` — those files do not exist there. Real location is **`backend/agents/mcp_servers/{signals,data,backtest}_server.py`**, and `meta_coordinator.py` is under **`backend/agents/`**, not `backend/autoresearch/`. All spec LINE numbers are correct against the real files.
2. **`BacktestResult` field names.** Four of the five names in the spec do not exist. Verified at `backend/backtest/backtest_engine.py:112-123`: the dataclass has `aggregate_sharpe`, `aggregate_return_pct`, `aggregate_alpha_pct`, `aggregate_max_drawdown_pct`, `aggregate_hit_rate`, `total_trades`, `windows`, `nav_history`, `all_trades`. There is **no `dsr`, no `return_pct`, no `max_drawdown_pct`, no `num_trades`**. DSR is not produced by the engine at all — so the key is dropped rather than emitted as a fabricated `0.0` (emitting 0.0 would be the very fault this step exists to remove). The immutable criteria never name these fields, so honoring reality does not amend any criterion.

Both corrections re-verified by me against live code (`ls`, dataclass read, `grep` for `def get_portfolio`) rather than taken on trust — twice this phase I have shipped an unverified claim, so parity/shape claims now get checked before use.

## Hypothesis

Every one of these findings is the same bug wearing different clothes: **a failure path that returns a confident, plausible value instead of refusing.** Replacing each with a marked-degraded, fail-closed result removes the fabricated state without touching a single risk threshold.

## Immutable success criteria (verbatim from .claude/masterplan.json step 75.3)

1. New backend/tests/test_phase_75_mcp_truth.py passes offline and asserts AT MINIMUM: (1) get_portfolio with a stub PaperTrader exposing get_or_create_portfolio/get_positions returns the stub's real NAV and positions, and signals_server source no longer contains 'paper_trader.get_portfolio('; (2) with a degraded (stub:true) snapshot publish_signal returns published=false with a degraded reason and books no trade
2. Test asserts a BUY whose unit price resolves to 0.0 is rejected 'unknown_price'; a -16% drawdown book blocks BUYs; an oversized explicit size_usd is clamped to the hard cap -- with all EXISTING threshold constants unchanged (test pins current values)
3. Test asserts a previously-rejected signal_id re-fired after cache eviction reports published=false (never a synthesized published=true), and a freed-up rejection can be retried
4. Test asserts every emit_candidates candidate carries stub:true and publish_signal rejects stub provenance; the compute_dsr_real dead branch is gone (source scan)
5. Test asserts get_macro returns the requested series' {date,value} entries from a fake _macro_full (dict iteration fixed) and that no '2025-12-31' literal remains in data_server.py; get_prices/get_fundamentals default end date is computed from date.today()
6. Test asserts backtest_server extracts metrics from a mocked BacktestResult dataclass without AttributeError, and that a SecretStr slack token reaches WebClient as its unwrapped plain-string value

## Confirmed root causes (each verified against live code)

- **gap4-01 (P1, live not latent).** `PaperTrader` has no `get_portfolio` — confirmed by enumerating its defs (`get_or_create_portfolio:95`, `get_positions:115`). So `signals_server.py:1259` raises `AttributeError` → bare `except` at `:1268` → fabricated $10K book → consumed at `:361` by `size_position` (`:988` reads `total_value`) and `risk_check` (`:396`). Note `get_positions()` returns a **list**, so the fix must build `{p['ticker']: p for p in ...}`.
- **gap4-05, with a second inert gate.** A `0.0` unit price makes `proposed_notional` 0.0, so per-ticker (`:871`), total-exposure (`:880`) and cash (`:890`) gates all pass trivially — the code comment at `:819-821` admits it. Separately `current_dd` is read at `:798` from a `current_drawdown_pct` key that `get_portfolio` never sets, so the drawdown breaker at `:896-901` is **permanently inert**. `track_drawdown` already exists at `:1237` and must be wired into publish step 5.
- **gap4-06 is a set/dict eviction asymmetry.** `_seen_signal_ids` (set, `:79`) is never evicted; `_recent_responses` (`:80`) is capped at 50 (`:279-282`). So `:334` stays True forever while `:335` misses, falling into `:343-348` which sets `published=True` — inverting the outcome for previously **rejected** signals. Also an unbounded-set memory leak. Fix: one `OrderedDict` as single source of truth so both evict together.
- **gap4-04 dead branch proven statically unreachable.** `returns_by_variant` is `{}` at `:1835` and never written, so `dsr_source` can never become `compute_dsr_real` and `:1853-1858` is dead.
- **gap4-07 is worse than "iterates wrong".** `cache.cached_macro` returns a dict keyed by series_id, so `data_server.py:177` iterating it yields **str** keys and `item.get()` raises `AttributeError` → caught at `:187` → `macro://` returns empty **whenever data exists**.
- **gap4-09.** `backtest_server.py:119` calls `.get()` on a dataclass → `AttributeError` → except at `:133` → the tool runs the FULL walk-forward and then **always** returns ERROR.
- **security-05.** A non-empty `SecretStr` is truthy, so the `if not slack_token` guard at `:459` does not catch it, and `str()` yields `**********`.

## Plan steps

1. **gap4-01** — `get_portfolio` uses `get_or_create_portfolio()` + `get_positions()` (list→dict by ticker), mapping `total_nav`→`total_value`, `current_cash`→`cash`. BOTH degraded paths (no-backend `:1249`, exception `:1268`) return `stub: true` **and zeroed `total_value`/`cash`** per the brief's defense-in-depth recommendation: a zeroed book makes `size_position` return 0.0 via its own `equity<=0` guard and makes every BUY gate fail closed even if the explicit refusal were bypassed. `publish_signal` refuses on `stub` with `published:false, reason: degraded_portfolio`.
2. **gap4-05** — `risk_check` rejects `unknown_price` when a BUY's unit price is `<= 0`, BEFORE the notional gates; wire `track_drawdown` into publish step 5 so `current_drawdown_pct` is actually populated; clamp explicit `size_usd` to the hard cap. **No threshold constant is edited** — tests pin current values.
3. **gap4-06** — single `OrderedDict` keyed by signal_id holding the true terminal outcome; evict both together; replay the true prior outcome; do NOT persist rejections permanently (Stripe's retryable-validation-failure rule).
4. **gap4-04** — every `emit_candidates` candidate carries `stub: true` + `PENDING_IMPLEMENTATION`; delete the unreachable branch; `publish_signal` rejects stub provenance. **CONSUMER CONSTRAINT:** `scripts/harness/mcp_ab_test.py:470-490` asserts ≥5 candidates each carrying a `dsr` key, tied to a phase-3.7 immutable criterion — add `stub:true` **additively**; do NOT drop the count or the `dsr` field.
5. **security-05** — `unwrap_secret` before `WebClient`, local import inside the function to preserve the lazy-import invariant at `:450-452`.
6. **gap4-07/08** — `get_macro` iterates `.items()` and filters to the requested series; the three hardcoded `2025-12-31` cutoffs (`:95`, `:142`, `:172`) become `date.today()`-derived using the idiom already present in the same file at `:251-252`; responses carry the effective range; drop the parsed-but-unused `market` prefix.
7. **gap4-09** — attribute access mirroring `meta_coordinator.py:306-311` with the **real** field names; drop the DSR key (engine does not produce it) rather than fabricating 0.0; pass the bq wrapper (BacktestEngine normalizes at `:176-177`); delete the never-read `timeout_seconds`.
8. **Tests** — `backend/tests/test_phase_75_mcp_truth.py`, offline. **Mocking discipline is load-bearing:** the existing `tests/test_mcp_servers.py` has zero mocks and asserts envelope shape rather than outcome (`assert "status" in result` passes when status == ERROR), which is *why* all of this shipped. New tests MUST use `create_autospec(PaperTrader, instance=True)` — a bare `Mock()` would happily serve `.get_portfolio()` and let gap4-01 regress — and return a real `BacktestResult` instance.

## Blast radius (fail-closed is safe here)

The signals MCP server is **not on the live money path**: the autonomous loop calls `PaperTrader.execute_buy/execute_sell` directly and no live service imports `SignalsServer`. Consumers to re-run: `scripts/harness/mcp_ab_test.py` and four go_live_drills — `position_limits:198` pins the strict-`>` boundary at `:874` (thresholds must stay byte-identical), `first_week_monitoring:219` sets `server._peak_equity` directly (the drawdown wiring must not break that seam), `kill_switch:131`, and `slack_signals_e2e` S9-S14 AST checks.

## References

- `handoff/current/research_brief_75.3.md` — 7 sources, 2 spec corrections, blast radius, mocking-discipline finding
- `handoff/current/audit_phase75/register.md` — gap4-01/04/05/06/07/08/09, security-05
- `.claude/masterplan.json` step 75.3 (immutable criteria + verification command)
