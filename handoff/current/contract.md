# Contract: Phase 4.4.4.2 -- Position-Limit Drill (BUY rejection on breach)

## Goal

Land executable evidence for the Phase 4.4.4.2 Go-Live Checklist item:
`risk_check` rejects a BUY that would push per-ticker exposure past 10% or
total past 100%, and rejects a BUY when the daily trade cap of 5 is reached.

Ship a standalone stdlib-only drill at `scripts/go_live_drills/
position_limits_test.py` mirroring the Cycle 9 `kill_switch_test.py`
shape. Then flip `docs/GO_LIVE_CHECKLIST.md` item 4.4.4.2 from `[ ]` to
`[x]` with a one-line `**Evidence**:` note under the bullet pointing at
the drill script + command + commit.

## Phase

PLAN -> GENERATE -> EVALUATE -> LOG. RESEARCH gate WAIVED: pure
verification drill against the already-pinned `SignalsServer.risk_check`
interface. Underlying method is stable at `signals_server.py:723`. All
three limit literals are already evidence-locked by Phase 4.4.4.4
(commit 9b0e943). No new research surface.

## Scope (strict)

- `scripts/go_live_drills/position_limits_test.py` -- NEW file
  (non-`.py`-package, stdlib-only, no venv, no GCP, no network).
- `docs/GO_LIVE_CHECKLIST.md` -- 2 lines touched on item 4.4.4.2:
  `[ ]` -> `[x]` and appended `**Evidence**:` line.

Nothing else. In particular:
- ZERO edits to `backend/agents/mcp_servers/signals_server.py`.
- ZERO edits to any other `.py` file.
- ZERO edits to `scripts/go_live_drills/kill_switch_test.py` (the
  4.4.4.1 drill remains byte-identical to commit `68a19c1`).
- ZERO new imports outside stdlib.
- ZERO touches to any other checklist item's checkbox or evidence.

## Success Criteria

### A. Scope discipline

- **SC1**: Exactly 2 files touched: `scripts/go_live_drills/position_limits_test.py`
  (added) and `docs/GO_LIVE_CHECKLIST.md` (modified).
- **SC2**: Zero `.py` files under `backend/` touched.
- **SC3**: `kill_switch_test.py` byte-identical to `git show 68a19c1:scripts/go_live_drills/kill_switch_test.py`.
- **SC4**: `position_limits_test.py` imports only `importlib.util`, `sys`,
  `pathlib`. No third-party imports. Zero non-ASCII bytes in the file.

### B. Drill structure

- **SC5**: Drill loads `signals_server.py` via `importlib.util` file-path
  loader, mirroring the 4.4.4.1 pattern (bypasses package `__init__` to
  avoid pulling FastAPI / GCP deps).
- **SC6**: Drill constructs `SignalsServer()` in stub mode (no GCP creds).
- **SC7**: Drill runs a pre-drill sanity check confirming all 4 limit
  keys match the 4.4.4.4 literals: `max_exposure_per_ticker_pct == 10.0`,
  `max_total_exposure_pct == 100.0`, `max_drawdown_pct == -15.0`,
  `max_daily_trades == 5`. Any drift fails the drill loudly.
- **SC8**: Drill defines >= 6 scenario functions covering the three
  thresholds named in the 4.4.4.2 HOW recipe:
    1. Per-ticker 10% breach -> BUY blocked (`max_exposure_per_ticker`)
    2. Per-ticker 10% at-boundary (exact 10.00%) -> BUY allowed (strict `>`)
    3. Per-ticker breach via existing-position aggregation -> BUY blocked
    4. Total exposure 100% breach -> BUY blocked (`max_total_exposure`)
    5. Daily trade count at 5 -> BUY blocked (`max_daily_trades`)
    6. Daily trade count at 4 -> BUY allowed

### C. Drill behavior

- **SC9**: Running `python scripts/go_live_drills/position_limits_test.py`
  exits with code 0 and prints `DRILL PASS: 6/6 position-limit scenarios
  verified against SignalsServer.risk_check` (or equivalent wording
  with count 6 and method name).
- **SC10**: Every scenario uses `assert` on both `resp["allowed"]` and
  the expected conflict string. No silent pass-throughs.
- **SC11**: S1 (per-ticker breach) response has `allowed is False` AND
  `"max_exposure_per_ticker" in resp["conflicts"]`.
- **SC12**: S4 (total-exposure breach) response has `allowed is False`
  AND `"max_total_exposure" in resp["conflicts"]`.
- **SC13**: S5 (daily trade cap breach) response has `allowed is False`
  AND `"max_daily_trades" in resp["conflicts"]`.
- **SC14**: S2 (exact 10% boundary) demonstrates the strict-greater
  semantics of `risk_check` line 872 (`if ticker_pct > max_per_ticker_pct`):
  10.00% is allowed, any hair above blocks.

### D. Checklist evidence

- **SC15**: `docs/GO_LIVE_CHECKLIST.md` section 4.4.4.2 bullet flipped
  from `- [ ]` to `- [x]`. All other checkbox states in the file
  preserved (4.4.4.1 `[x]`, 4.4.4.4 `[x]`, all others `[ ]`).
- **SC16**: A new `- **Evidence**:` line is appended immediately under
  the 4.4.4.2 bullet, naming the drill path, the execution date
  (2026-04-15), the scenario pass count (6/6), and the commit hash
  that landed the drill.
- **SC17**: Section header `### 4.4.4.2 Position limits tested: submit
  oversized position -> verify rejection` unchanged.
- **SC18**: No other 4.4.4.x section touched.

### E. Global invariants

- **SC19**: `python -c "import ast; ast.parse(open('scripts/go_live_drills/position_limits_test.py').read())"` clean.
- **SC20**: `python -m py_compile scripts/go_live_drills/position_limits_test.py` clean.

## Adversarial Probes (for QA)

1. **Boundary-inclusive trap**: if the drill treats the 10% boundary as
   exclusive (blocks at exactly 10%), it would disagree with
   `risk_check`'s strict `>` semantics and pass only because the server
   disagreed silently. SC14 pins the boundary at exactly 10.00% ->
   allowed.
2. **Helper price disagreement**: the drill must set `proposed_trade`
   with explicit `price`, not rely on existing-position `last_price`
   fallback. The price must be nonzero or the cash check swallows the
   test (`proposed_notional = 0`, concentration check trivially zero).
   SC11 scenario uses explicit `price=100.0` on all trades.
3. **Trades_today shape**: `risk_check` accepts both list and int.
   Daily-trade scenarios must use a list of length 5 (not an int 5)
   so the drill exercises the list branch, which is the production
   shape from `_append_signal_history`.
4. **Cash exhaustion hides the real conflict**: if cash < proposed
   notional, risk_check returns `insufficient_cash` BEFORE checking
   the drawdown/per-ticker/total branches. All BUY scenarios must set
   `cash` >= `proposed_notional` so the test exercises the intended
   hard branch, not the cash branch.
5. **Total-exposure aggregation trap**: S4 must set existing positions
   worth 9500 + proposed BUY 600 = 10100 > 10000 total_value. Must also
   confirm the proposed ticker is a DIFFERENT symbol from the existing
   positions, otherwise per-ticker fires first.
6. **Daily-trade order-of-evaluation trap**: `risk_check` checks
   `trades_today_count >= max_daily_trades` BEFORE per-ticker /
   total checks (line 846). So S5 must set trades_today=list-of-5
   and a trivially-sized trade (1 share @ $100) so it exercises the
   daily-cap branch specifically, not the exposure branches.
7. **Non-ASCII slip**: file must be byte-ASCII to respect the
   security.md logger rule even though this script does not log via
   `logger.*()` calls. Defense-in-depth.
8. **Drill must be re-runnable**: it must not mutate any state that
   would persist across runs (no file writes, no BQ calls, no GCP
   calls). `SignalsServer` stub mode guarantees this, but the drill
   must not touch `server.publish_signal` or `_append_signal_history`.

## Budget

- Lines added to `position_limits_test.py`: <= 220
- Lines touched in `docs/GO_LIVE_CHECKLIST.md`: 2 (one flip, one append)
- Total diff: <= 225 lines added, 1 line deleted (the checkbox flip
  counts as a delete+add).
- No Slack API cost (just status post).
- No LLM cost (self-eval justified or single qa-evaluator Opus
  spawn).

## DO NOT

- Do not edit `signals_server.py`.
- Do not refactor `kill_switch_test.py` for code reuse. Copy-paste the
  loader helper is intentional -- each drill is an independent,
  copy-pasteable artifact future-Ford can run in isolation.
- Do not touch `masterplan.json`. Phase 4.4 remains `pending` until
  every one of its 27 items is `[x]`.
- Do not manually update `CHANGELOG.md` (PostToolUse hook).
