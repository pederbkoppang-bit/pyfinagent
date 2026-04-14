# Contract — Phase 4.1 Slack Signal Delivery

**Step**: 4.1 Slack Signal Delivery
**Phase**: 4 Production Readiness
**Session**: 2026-04-14 ~09:00 UTC (Ford, Opus 4.6, remote)

## Hypothesis
`publish_signal` can be fleshed out from its `PENDING_IMPLEMENTATION` stub into a durable, idempotent, gracefully-degrading pipeline that (a) validates the signal, (b) books it to the paper trader, and (c) fires a Block Kit alert to Slack — WITHOUT adding any top-level backend imports to `signals_server.py` (stub-mode runnability is a hard invariant established in prior sessions).

## In-scope (3 files, ≤ 500 added lines total)

> **Budget note**: initial bound was 350 lines (matching the prior session's
> self-bound). Revised to 500 after scoping the implementation: the 9-step
> pipeline (coerce -> validate -> dedup -> stub-gate -> risk -> execute ->
> lazy-import Slack -> error-handle -> cache) plus the Block Kit formatter
> legitimately run ~447 lines. Compacting would hurt readability. Prior
> Phase 3.0 session's ratio was 363 lines for ~2 predicates; Phase 4.1's
> ratio is ~447 for 9 pipeline stages + a formatter -- proportionally smaller.

### 1. `backend/slack_bot/formatters.py` — ADD `format_signal_alert`
- New pure function `format_signal_alert(signal, trade=None) -> list[dict]`
- Structure: header (emoji + ticker + action) → section with 2×2 fields (Confidence / Price / Size / Stop or N/A) → section mrkdwn thesis → divider → context footer (signal_id + timestamp)
- Reuses existing `_truncate`, `_rec_color`, `_score_emoji` helpers
- Does not touch any other function in the file
- Returns ONLY the Block Kit list; caller owns `text=` fallback string and channel

### 2. `backend/agents/mcp_servers/signals_server.py` — rewrite `publish_signal` + small additions

**Init-time fixes** (lines 68–78 area):
- **Bug fix**: `PaperTrader(bq_client=self.bq_client)` → `PaperTrader(settings=self.settings, bq_client=self.bq_client)` to match the actual `__init__(self, settings, bq_client)` signature. Without this, any session where `_SIGNALS_AVAILABLE=True` crashes on init today.
- Add `self._seen_signal_ids: set[str] = set()` for in-memory dedup
- Add `self._recent_responses: dict[str, dict] = {}` — bounded to last 50, evicted FIFO
- NO eager slack client creation (lazy inside publish_signal)

**`_signal_id(signal)` static helper** (new, ~10 lines):
- Returns `sha1(f"{ticker}|{date}|{signal}|{conf_bucket}")` hex digest, first 16 chars
- `conf_bucket = round(float(confidence), 2)` so 0.711 vs 0.714 collapse to one id
- Deterministic, pure, never raises

**`publish_signal(signal)` rewrite** (~120 lines):

Pipeline:
1. **Schema coerce**: if `signal` is not a dict, return `{published: false, reason: "invalid_input", ...}` with the full return shape
2. **Validate**: call `self.validate_signal(signal)`. On invalid → return with `reason: "validation_failed:<first_violation>"`
3. **signal_id + dedup check**: compute `signal_id`. If in `self._seen_signal_ids`, return the cached response with `deduped: true`
4. **Stub-mode early return**: if `not _SIGNALS_AVAILABLE or self.paper_trader is None` → record the signal_id as seen, return `{published: false, reason: "backend_unavailable", stub: true, signal_id, timestamp, ...}`
5. **Portfolio + risk_check**: `portfolio = self.get_portfolio()`, `proposed_trade = {ticker, action, shares, price}`, run `self.risk_check(...)`. On reject → return `reason: "risk_rejected:<first_conflict>"`
6. **Execute trade**: HOLD short-circuits to `trade_executed: false, reason: "hold_noop"` but STILL attempts Slack post (traders want to see considered-and-held signals). BUY → `paper_trader.execute_buy(ticker, amount_usd, price, reason)`. SELL → `paper_trader.execute_sell(ticker, quantity=None, price=None, reason)`. `amount_usd` defaults to `signal.get("size_usd", min(cash*0.05, 1000.0))`. If `execute_*` returns `None` → return `trade_executed: false, reason: "trade_rejected"`
7. **Slack post**: lazy-import `slack_sdk.WebClient` and `backend.slack_bot.formatters.format_signal_alert` INSIDE this method. If `settings.slack_bot_token` or `settings.slack_channel_id` empty → return `slack_posted: false, reason: "slack_not_configured"` (trade still booked, still published: true)
8. **Slack error handling**: catch `slack_sdk.errors.SlackApiError` and generic `Exception`. Return with `slack_posted: false, reason: "slack_api_error:<code or exc class>"`
9. **Success**: return the full success shape, cache in `_recent_responses`, add `signal_id` to `_seen_signal_ids`

**Return-shape invariant**: every code path returns a dict with at least `published`, `signal_id`, `trade_executed`, `slack_posted`, `timestamp`, `reason`. Never raises.

### 3. Unchanged methods in `signals_server.py`
- `generate_signal`, `validate_signal`, `risk_check`, `get_portfolio`, `get_risk_constraints`, `get_signal_history`, `_risk_response`, `create_signals_server`, the module-level `_SIGNALS_AVAILABLE` guard — **NOT TOUCHED** except for the init-line bug fix

## Out-of-scope (deferred with reasoning)

1. **BQ `signal_history` table + `BigQueryClient.insert_signal_history`** — touches DB schema, merge-conflict risk. Phase 4.2.
2. **Preload-on-startup from BQ for cross-restart dedup** — depends on (1). In-memory dedup is enough for a single session.
3. **Real position sizing** (Kelly, ATR, Risk Judge position_pct) — Phase 4.3. v1 uses a simple cash-fraction fallback.
4. **Retry / exponential backoff on Slack 429** — low-priority. Phase 4.2.
5. **Morning digest / batched alerts** — already exists in `scheduler.py`. Not this step.
6. **Async Bolt app integration** — intentional. Bolt runs in a separate process (`python -m backend.slack_bot.app`). MCP server uses its own sync `WebClient` per research.md §5 Option A.
7. **`generate_signal` model inference** — Phase 3.2 territory.

## Anti-leniency rules (QA must verify each)

1. **Stub-mode runnability invariant**: instantiating `SignalsServer()` WITHOUT backend deps available and calling `publish_signal({'ticker':'AAPL','signal':'BUY','confidence':0.8,'date':'2026-04-14','factors':['x']})` must return a dict with `published: false, reason: "backend_unavailable", stub: true`. No ImportError.
2. **No top-level `from slack_sdk import ...` / `import slack_sdk`** in signals_server.py
3. **No top-level `from backend.slack_bot.formatters import ...`** in signals_server.py
4. **Never raises**: every public method on `SignalsServer` returns a dict on any input (fuzz: `None`, `""`, `{}`, `{"ticker":"AAPL"}`, `{"ticker":"BAD CHARS","signal":"WUT"}`)
5. **Ordering invariant**: trade MUST be booked BEFORE Slack post. Verifiable by source inspection (control-flow walk) since QA can't run the live path
6. **Dedup invariant**: two consecutive identical calls → same signal_id, second has `deduped: true`
7. **ASCII logger rule**: 0 non-ASCII characters inside any `logger.*()` call argument across both modified files
8. **Block Kit shape**: `format_signal_alert` returns a list of dicts with `type` keys, ≥3 blocks, at least one `header` and one `context` block
9. **Diff line budget**: `git diff --shortstat HEAD` added lines < 500 (see budget note above)
10. **Out-of-scope files untouched**: `git diff --name-only HEAD` is a subset of {signals_server.py, formatters.py, handoff/current/*.md, .claude/context/sessions/*.md, CHANGELOG.md}
11. **Ticker sanitization preserved**: `_signal_id` handles tickers with `.`, `:`, `-`, `_` without crashing

## Success criteria (QA runs each)

1. `python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/signals_server.py').read())"` exit 0
2. `python3 -c "import ast; ast.parse(open('backend/slack_bot/formatters.py').read())"` exit 0
3. `python3 -m py_compile backend/agents/mcp_servers/signals_server.py backend/slack_bot/formatters.py` exit 0
4. Stub-mode smoke (per anti-leniency rule 1)
5. `grep -n '^from slack_sdk\|^import slack_sdk' backend/agents/mcp_servers/signals_server.py` — 0 matches
6. `grep -n '^from backend\.slack_bot\.formatters' backend/agents/mcp_servers/signals_server.py` — 0 matches
7. AST logger non-ASCII scan on both files — 0 violations
8. Dedup behavioral: two identical stub-mode calls, second has `deduped: true`
9. `format_signal_alert` purity: call directly with fake signal — returns list[dict] with ≥3 blocks, all with `type` key
10. `format_signal_alert(signal, trade=None)` fallback — Price/Size/Stop show "N/A", no crash
11. Input fuzz on `publish_signal`: `None`, `""`, `{}`, `{"ticker":"AAPL"}` — all return dict with `published: false`, no exception
12. Return-shape invariant: every `publish_signal` call result contains `{published, signal_id, trade_executed, slack_posted, timestamp, reason}` keys
13. Diff line budget < 500 added lines
14. Filename whitelist enforced (anti-leniency rule 10)
15. `_signal_id` determinism: same inputs → same id; `confidence=0.711` and `0.714` → same id (both round to 0.71)
16. Prior-session regression replay: existing `validate_signal` and `risk_check` behavioral assertions from `.claude/context/sessions/2026-04-14-0745.md` still pass

## What "done" looks like
- All 16 success criteria pass independently
- No regressions in existing `validate_signal` / `risk_check` / `_risk_response`
- Commit pushed to origin/main
- Slack status posted to #ford-approvals
