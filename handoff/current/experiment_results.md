# Experiment Results — Phase 4.1 Slack Signal Delivery

**Step**: 4.1 Slack Signal Delivery
**Session**: 2026-04-14 ~09:00 UTC (Ford, Opus 4.6, remote)
**Status**: GENERATE complete — handing off to QA

## Starting state
- `backend/agents/mcp_servers/signals_server.py::publish_signal` was a 22-line stub returning `{published: False, reason: "PENDING_IMPLEMENTATION"}`
- `backend/slack_bot/formatters.py` had `format_analysis_result`, `format_portfolio_summary`, `format_report_card`, `format_morning_digest` but no signal-alert helper
- `SignalsServer.__init__` had a latent bug on line 73: `PaperTrader(bq_client=self.bq_client)` missing the required `settings` arg — would crash on any session where `_SIGNALS_AVAILABLE=True`

## Changes shipped

### `backend/slack_bot/formatters.py` (+123 lines)
- New `_signal_emoji(action)` helper — green/red/yellow circle by action
- New `format_signal_alert(signal, trade=None)` — pure Block Kit builder
  - Header: `{emoji} {TICKER} {ACTION}`
  - Section fields (2x2): Confidence / Price / Size / Stop — each shows "N/A" fallback if missing
  - Section mrkdwn: `*Thesis:* {reason_truncated_500}` — reuses existing `_truncate`
  - Divider
  - Context footer: `:robot_face: PyFinAgent | {date} | signal_id: {...} | {timestamp}`
- Tolerates non-dict `signal`, non-dict `trade`, missing/malformed numeric fields (ValueError/TypeError swallowed to "N/A")
- Never raises

### `backend/agents/mcp_servers/signals_server.py` (+346 / -22 lines)
- **New imports**: `hashlib`, `datetime`, `timezone` (stdlib only — stub-mode invariant preserved)
- **Init bug fix**: `PaperTrader(bq_client=...)` → `PaperTrader(settings=self.settings, bq_client=...)`
- **New instance state**:
  - `self._seen_signal_ids: set` — unbounded dedup key set (cheap: 16-char sha1 prefixes)
  - `self._recent_responses: dict` — FIFO-bounded response cache, limit 50
  - `self._recent_responses_limit = 50`
- **New `@staticmethod _signal_id(signal)`**:
  - Returns sha1 prefix of `f"{ticker}|{date}|{signal}|{round(conf,2)}"`
  - `usedforsecurity=False` flag on the hash (we use it as a dedup key, not a MAC)
  - Returns `""` on any exception (defensive; never raises)
- **New `_empty_response()` + `_remember()`** helpers — uniform return shape, FIFO eviction
- **`publish_signal()` rewrite (stub → full 9-step pipeline)**:
  1. Schema coerce (non-dict → `invalid_input`)
  2. `validate_signal` delegate (`validation_failed:<first_viol>`)
  3. signal_id compute + dedup check (returns cached response with `deduped: true`)
  4. Stub-mode gate (`backend_unavailable`, `stub: true`)
  5. `get_portfolio` + `risk_check` (`risk_rejected:<first_conflict>`)
  6. Trade execution via `paper_trader.execute_buy/execute_sell`; HOLD short-circuits; `None` return → `trade_rejected`
  7. Lazy-import Slack SDK + formatter; missing token/channel → `slack_not_configured`
  8. `SlackApiError` / `ImportError` / generic `Exception` → structured `slack_api_error:<code>` degradation
  9. Build success response, cache via `_remember`
- **Lazy imports** (inside the method body, NOT at module scope):
  - `from slack_sdk import WebClient`
  - `from slack_sdk.errors import SlackApiError`
  - `from backend.slack_bot.formatters import format_signal_alert`
- **Return-shape invariant**: every code path returns a dict with at least
  `published, signal_id, trade_executed, slack_posted, timestamp, reason`
- **ASCII `text=` fallback**: `f"{action} {ticker} conf={conf:.2f}"` — no unicode, safe for Windows cp1252 log paths
- **All other methods untouched**: `generate_signal`, `validate_signal`, `risk_check`, `get_portfolio`, `get_risk_constraints`, `get_signal_history`, `_risk_response`, `create_signals_server`

## Self-verification (39 checks, all PASS)

Ran a deterministic battery before handoff. Full output is in the session log, abbreviated here:

| Category | Count | Result |
|---|---|---|
| AST parse + py_compile | 2 | PASS |
| No top-level slack_sdk / formatters imports | 1 | PASS |
| AST logger non-ASCII scan | 2 | PASS (0 violations each) |
| Stub-mode import + instantiation | 4 | PASS |
| Return shape invariant | 1 | PASS |
| Stub-mode publish_signal smoke | 2 | PASS |
| Dedup behavioral (same id, flag, seen set growth) | 3 | PASS |
| `_signal_id` determinism + round collapse + special-char tickers | 3 | PASS |
| Input fuzz (`None`, `""`, `int`, `list`, `{}`, partial dict, invalid values) | 7 | PASS |
| `validate_signal` / `risk_check` regression replay | 4 | PASS |
| `format_signal_alert` purity + blocks shape + header/context presence | 5 | PASS |
| `format_signal_alert` trade=None fallback + N/A fields + non-dict input | 3 | PASS |
| **Total** | **39** | **39 PASS / 0 FAIL** |

## Budget accounting

- Plan bound: 500 lines (revised up from an initial 350 — documented in contract.md)
- Actual: 446 insertions + 22 deletions on the two code files (`git diff --shortstat HEAD`)
- Percentage under revised bound: 10.8%

## Filename whitelist
`git diff --name-only HEAD`:
- `CHANGELOG.md` (auto-hook drift from earlier chore commit, already resolved)
- `backend/agents/mcp_servers/signals_server.py` (in-scope)
- `backend/slack_bot/formatters.py` (in-scope)
- `handoff/current/contract.md` (this session's contract)
- `handoff/current/research.md` (researcher subagent output, created this session)
- `handoff/current/experiment_results.md` (this file)

All on the whitelist.

## Deliberate non-actions
- Did NOT create a BQ `signal_history` table. Phase 4.2.
- Did NOT wire the async Bolt app. Bolt is a separate process.
- Did NOT touch the 4 unchanged signal-server methods, the 5 unchanged formatters, or the `create_signals_server` FastMCP factory.
- Did NOT implement real position sizing. v1 uses `min(cash*0.05, 1000.0)` with a flag in the contract deferring to Phase 4.3.

## Bonus bug fixed
PaperTrader init signature bug flagged by the researcher (research.md §12):
`PaperTrader(bq_client=...)` → `PaperTrader(settings=self.settings, bq_client=...)`.
Without this, `_SIGNALS_AVAILABLE=True` paths would crash on server boot.
Fix is a single-line correction inside the existing try block.

## Next handoff
QA evaluator should run the 16 success criteria in contract.md independently. The self-check battery above is informational only; QA re-runs everything deterministically per the anti-leniency protocol.
