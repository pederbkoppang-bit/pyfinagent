# Research: Phase 4.1 — Slack Signal Delivery (`publish_signal`)

Scope: fleshing out `backend/agents/mcp_servers/signals_server.py::publish_signal`
to (a) record a trade via `PaperTrader`, (b) post a Block Kit alert to Slack,
(c) degrade gracefully when backend deps are missing, and (d) audit everything.

Research Gate status: 7/7 categories covered, 15 unique URLs collected, 2 pages
read in full (apxml MCP error handling, slackapi/bolt-python#564 issue). Several
authoritative docs (api.slack.com, modelcontextprotocol.io, quantconnect.com)
returned HTTP 403 to WebFetch but their content was recovered from WebSearch
result summaries. Notes at the bottom flag the gap.

---

## 1. Slack Block Kit for trading alerts

Canonical Block Kit pattern for alerts is header -> section(fields) ->
section(mrkdwn summary) -> divider -> context(footer). This matches what
`backend/slack_bot/formatters.py::format_analysis_result` already does and is
the recommended shape across every source I found.

Key takeaways:
- Header block = the "subject line". One short sentence. Use it to carry the
  ticker and action (`AAPL  BUY signal`). Knock's "deep dive" post is explicit:
  header = subject line equivalent.
- Section `fields` arrays are perfect for 2-column compact KV pairs
  (Confidence / Price / Size / Stop). Max 10 fields, 2000 chars each, but
  practical limit is 4-6 for scanability.
- Context block at the bottom for metadata the reader only glances at:
  timestamp, signal_id, "pyFinAgent v0.4.1". This is where we put the audit
  trail hooks.
- Divider block between the "what" (action, confidence) and the "why" (reason,
  factors). This is the standard APM/alert pattern cited by MagicBell and
  Knock.
- 3000-char section text limit (already enforced by `_truncate()` in our
  formatters.py at line 9). Reuse that helper.
- Block Kit messages still need a top-level `text` fallback for mobile push
  notifications and for screen readers -- the header block alone is not
  enough. `text="AAPL BUY @ 0.72 conf"` is the push-preview string.

Sources:
- https://knock.app/blog/taking-a-deep-dive-into-slack-block-kit
- https://docs.slack.dev/block-kit/
- https://api.slack.com/block-kit
- https://api.slack.com/messaging/composing/layouts
- https://www.magicbell.com/blog/slack-blocks
- https://knock.app/blog/the-guide-to-designing-slack-notifications

## 2. Slack idempotency / dedup

The slackapi/bolt-python#564 thread (read in full) confirms that Slack does NOT
provide server-side dedup for `chat.postMessage` -- `client_msg_id` is for
*user* messages, not bot posts, and the Events API `event_id` only helps with
incoming event dedup. Outgoing dedup is 100% client responsibility.

Practical patterns:
- App-level `signal_id` = deterministic hash of
  `(ticker, date, signal_type, confidence_bucket)`. Store a small in-memory
  `set[str]` of seen ids inside the `SignalsServer` instance, optionally
  persisted to BQ `signal_history` for cross-restart dedup.
- Slack's own guidance when you hit duplicate posts is to check for
  `X-Slack-Retry-Num` and short-circuit retries, but since we initiate the
  post, we own dedup.
- Return the existing `slack_ts` on a dedup hit instead of re-posting.
  Treat this as success (`published: true, deduped: true`) rather than error.
- Two-key dedup is cleaner than one: `(signal_id, date)` so we can post a
  fresh signal tomorrow without false collisions.

Sources:
- https://github.com/slackapi/bolt-python/issues/564
- https://docs.slack.dev/tools/bolt-python/concepts/message-sending/
- https://github.com/slackapi/bolt-python/issues/693 (duplicate chat.postMessage symptom)
- https://api.slack.com/messaging/sending

## 3. Paper trading signal-to-execution patterns

QuantConnect's Algorithm Framework is the gold-standard reference. Flow:

```
Universe -> Alpha (Insights) -> PortfolioConstruction (PortfolioTargets)
         -> RiskManagement (adjusted PortfolioTargets) -> Execution (fills)
```

Crucially: *broadcast/notification is orthogonal* to execution. QC's
`Notify.Email` / webhook path is called from within `OnOrderEvent`, AFTER
the execution model has filled (or tried to fill) the order, not before.
This gives us our ordering: **validate -> risk_check -> execute_paper_trade
-> on_success_post_to_slack**, never the reverse. If Slack fails we still
kept the trade (it's durable in BQ); if the trade fails we must NOT post.

Other references:
- Freqtrade's `notify_enter`/`notify_exit` hooks are fired from the
  `ExitCheck`/`EntryCheck` -> `execute_entry` path, after the virtual order
  is booked.
- Backtrader's `notify_order()` and `notify_trade()` likewise hang off the
  broker callback, not the strategy `next()` tick.

Implications for our publish_signal:
1. Idempotent precheck (signal_id seen?)
2. `validate_signal` (already exists)
3. `risk_check` (already exists)
4. `paper_trader.execute_buy/execute_sell` -- this is the "execution model"
5. If trade record returned -> build Block Kit -> post to Slack
6. Persist `signal_history` row with both `trade_id` and `slack_ts`
7. On partial failure (trade ok, Slack fails) still return
   `published: true, slack_posted: false` -- the trade is the source of truth.

Sources:
- https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/overview
- https://www.quantconnect.com/docs/v1/algorithm-framework/execution
- https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/key-concepts
- https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/notifications
- https://github.com/QuantConnect/Lean (Execution model code)

## 4. MCP tool graceful degradation

Read the apxml.com MCP error handling course page in detail. Consensus across
three sources (apxml, mcpcat.io, gofastmcp.com) is unambiguous:

**MCP tools SHOULD return structured error-shape results, NOT raise
exceptions that bubble to the client.**

The FastMCP / mcp-python-sdk pattern is:

```python
CallToolResult(
    content=[TextContent(type="text", text="Paper trader unavailable")],
    isError=True
)
```

With the lower-level `@mcp.tool` decorator + pydantic return models, the
equivalent is returning a dict with `{"error": "...", "ok": false}` and
letting FastMCP wrap it. Raising an unhandled exception terminates the JSON-RPC
conversation which breaks the LLM's recovery path.

For our `publish_signal`, the existing stub pattern is correct:
`{"published": False, "reason": "PENDING_IMPLEMENTATION"}`.

Graceful degradation ladder we should implement:
1. `_SIGNALS_AVAILABLE=False` -> return `{"published": False,
   "reason": "backend_unavailable", "stub": True}` and log at INFO.
2. Slack token missing -> execute trade, skip post, return
   `{"published": True, "trade_executed": True, "slack_posted": False,
     "reason": "slack_not_configured"}`.
3. Slack API error (network, 500, rate limit) -> same shape but
   `reason: "slack_api_error: <code>"`; include the trade_id so the caller
   still has an audit record.
4. Trade fails (insufficient cash, risk reject) -> return
   `{"published": False, "trade_executed": False, "reason": "..."}`.
   No Slack post, no partial state.

Sources:
- https://apxml.com/courses/getting-started-model-context-protocol/chapter-3-implementing-tools-and-logic/error-handling-reporting
- https://mcpcat.io/guides/error-handling-custom-mcp-servers/
- https://gofastmcp.com/clients/tools
- https://modelcontextprotocol.io/docs/concepts/tools
- https://modelcontextprotocol.info/docs/best-practices/
- https://github.com/modelcontextprotocol/python-sdk

## 5. Slack async posting from sync context

This is the stickiest piece. Our MCP server is a plain sync class;
`slack_bolt.async_app.AsyncApp` (used in `backend/slack_bot/app.py`) uses
`AsyncWebClient`. Four options, ranked:

### Option A (recommended): use `slack_sdk.WebClient` (sync) directly
Don't share a client with the Bolt app. Build a fresh `WebClient(token=...)`
inside `SignalsServer.__init__` and call `client.chat_postMessage(...)`
synchronously. No asyncio gymnastics. Slack SDK split into sync/async
exactly so this pattern works -- per the engineering blog and issue #633.

Pros: zero event-loop coupling, safe from any caller (sync test, MCP server,
FastAPI sync endpoint, Celery worker).
Cons: a second Slack connection pool. Negligible.

### Option B: `asyncio.run(AsyncWebClient.chat_postMessage(...))`
Works if the caller is a pure sync context with NO running event loop.
Fails with `RuntimeError: asyncio.run() cannot be called from a running
event loop` if invoked from inside a FastAPI async handler or from the
Bolt app itself.

### Option C: `asyncio.run_coroutine_threadsafe(coro, loop)`
Requires an already-running loop in another thread and a handle to it.
Overkill for our setup; couples the MCP server to the bot lifecycle.

### Option D: `anyio.from_thread.run()`
Clean but adds a dep and we don't use anyio elsewhere.

**Decision: Option A. Use sync `WebClient`.** The Bolt async app in
`backend/slack_bot/app.py` is a *separate* process already
(`python -m backend.slack_bot.app`). The MCP server runs in the backend
process. They must not share a client anyway.

Sources:
- https://docs.slack.dev/tools/python-slack-sdk/reference/web/async_client.html
- https://github.com/slackapi/python-slack-sdk/issues/633
- https://slack.engineering/rewriting-the-slack-python-sdk/
- https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-5.html
- https://docs.python.org/3/library/asyncio-task.html (run_coroutine_threadsafe)
- https://death.andgravity.com/asyncio-bridge

## 6. Signal alert UX — less is more

Consensus from Smashing / Toptal / Coyle / Reteno trading-alert guidance:

- Cap at 3-5 key fields in the "above the fold" portion. Our header + 4-field
  section fits exactly.
- Must-include: ticker, action (BUY/SELL/HOLD), confidence, 1-line thesis,
  timestamp.
- Nice-to-have: entry price, suggested position size, stop loss level.
- Omit: full factor list, raw model scores, multi-paragraph rationale. Put
  those behind a "View full analysis" button linking to the FastAPI
  `/analysis/{id}` page if we have it, else the signal_id in a context block.
- Alert fatigue is the #1 cited trading-alert failure mode. Tradefundrr and
  Smashing both recommend 3-5 alerts per session max. Our existing
  `PaperTrader` already rate-limits via `max_daily_trades` (5 in risk_check).
  Reuse that as the dedup gate -- don't post a 6th alert.
- Personalization nearly 4x open rate. Not actionable at phase 4.1 but note
  for phase 4.2.
- Push preview (the `text=` fallback) should be the single most important
  sentence: `"BUY AAPL @ 0.87 conf -- earnings momentum"` is better than
  `"pyFinAgent alert"`.

Proposed minimal field set (maps straight to Block Kit):

| Block | Content |
|-------|---------|
| header (plain_text) | `{EMOJI} {TICKER} {ACTION}` e.g. ":green_circle: AAPL BUY" |
| section fields (2x2) | Confidence `0.87`, Price `$178.43`, Size `$5,000 (2.4%)`, Stop `$172.10` |
| section mrkdwn | `*Thesis:* {reason, truncated 500}` |
| context | `signal_id  2026-04-14 10:32 ET  trade_id:...` |

Sources:
- https://www.smashingmagazine.com/2025/07/design-guidelines-better-notifications-ux/
- https://www.toptal.com/designers/ux/notification-design
- https://coyleandrew.medium.com/design-better-alerts-2e2ee238afde
- https://tradefundrr.com/setting-alerts-on-trading-platforms/
- https://uxcam.com/blog/push-notification-guide/
- https://hybridsolutions.com/blog/mobile-ux-best-practices-for-trading-apps/

## 7. Delivery audit / signal_history

tradesignal.tech, Slack's own audit-logs docs, and general fintech compliance
guidance converge on the same minimum schema for a durable signal ledger:

Required fields:
- `signal_id` (PK, deterministic hash)
- `ticker`
- `signal` (BUY/SELL/HOLD)
- `confidence`
- `generated_at` (ISO8601 UTC, millisecond precision)
- `published_at` (nullable -- set on Slack success)
- `slack_ts` (nullable -- Slack's own message ts, doubles as idempotency key)
- `slack_channel`
- `trade_id` (nullable -- FK to paper_trades)
- `trade_executed` (bool)
- `error` (nullable text)
- `attempt_count` (for retry semantics)

Why it matters even for paper trading:
1. Post-hoc attribution: "did we post about TSLA before or after the move?"
   Without a durable timestamp distinct from the trade timestamp you can't
   answer this.
2. Dedup across restarts: in-memory `set` gets wiped on backend reload.
3. Regulatory muscle memory: when pyFinAgent goes to a real brokerage we need
   this ledger already in place; retrofitting audit logging is a classic
   post-launch bug source.
4. Outcome tracking: `backend/services/outcome_tracker.py` reads past
   recommendations; extending its query to `signal_history` lets us measure
   whether "alerted" signals outperform "silently traded" signals.

Storage: BigQuery table `signal_history` (new). Reuse the existing
`BigQueryClient.upsert_*` pattern. Millisecond timestamps via `datetime.now(
timezone.utc).isoformat(timespec="milliseconds")`. Index hint: PARTITION BY
DATE(generated_at), CLUSTER BY ticker.

Sources:
- https://tradesignal.tech/blog/trading-automation-1/how-can-you-log-monitor-and-troubleshoot-live-automated-signals-20
- https://api.slack.com/admins/audit-logs
- https://docs.slack.dev/reference/audit-logs-api/methods-actions-reference/

---

## Design implications (for contract.md)

1. **Use sync `slack_sdk.WebClient`, not AsyncWebClient.** Instantiate once in
   `SignalsServer.__init__` guarded by `settings.slack_bot_token`. If the
   token is missing the client is `None` and we degrade path (b) below.
2. **Pipeline inside `publish_signal`:** dedup-check -> validate_signal ->
   risk_check -> paper_trader.execute_buy/execute_sell -> build blocks ->
   post to Slack -> write signal_history row. Any failure after step 4
   still returns `published: true` for the trade half but flags
   `slack_posted: false`.
3. **Ordering is non-negotiable:** never post to Slack before the trade is
   booked. The trade is the source of truth; the alert is a side-effect.
4. **Dedup key:** `signal_id = sha1(f"{ticker}|{date}|{signal}|{round(conf,2)}")`.
   Persist to in-memory set on the server instance + (ideally) BQ
   `signal_history`. Return `deduped: true` + existing `slack_ts` on hit.
5. **Block Kit shape:** header + 2x2 fields (conf/price/size/stop) + section
   thesis + context footer. Always pass `text=` fallback for push preview.
   Reuse `_truncate`, `_score_emoji`, `_rec_color` from formatters.py -- add
   a new `format_signal_alert(signal, trade)` helper there rather than
   re-implementing in the MCP server. Keep formatting in one place.
6. **Graceful degradation ladder** (4 rungs, each a distinct `reason` string):
   `backend_unavailable` -> `slack_not_configured` -> `slack_api_error:<code>`
   -> `trade_rejected:<reason>`. All return a dict; none raise.
7. **Return shape** (superset of current stub, backwards-compatible):
   ```python
   {
     "published": bool,
     "signal_id": str,
     "deduped": bool,
     "trade_executed": bool,
     "trade_id": str | "",
     "slack_posted": bool,
     "slack_ts": str | "",
     "slack_channel": str | "",
     "timestamp": str,  # ISO8601 UTC ms
     "reason": str,
   }
   ```
8. **`signal_history` table**: new BQ table, schema above. If the table
   doesn't exist, log and skip -- don't block posting. Add a
   `BigQueryClient.insert_signal_history(row)` method.
9. **Top-level `text=` fallback** must be ASCII-only (our security rule:
   ASCII-only logger messages; same principle applies to the fallback string
   because it also hits cp1252 stdout in some log paths). Use
   `"BUY AAPL 0.87 -- earnings momentum"` not fancy arrows.
10. **Rate limiting**: honor `max_daily_trades=5` from `risk_check`. Don't
    implement a separate alert rate limiter; the risk check already gates us.
11. **PaperTrader init bug flag:** current signals_server line 73 does
    `PaperTrader(bq_client=self.bq_client)` but `PaperTrader.__init__`
    requires `(settings, bq_client)`. Fix this as part of 4.1.
12. **Idempotency on reload**: on startup, preload the last 24h of signal_ids
    from `signal_history` into the in-memory set. Prevents post-crash
    duplicate posts.

---

## Gaps / unread sources

Three primary docs returned HTTP 403 to WebFetch and I relied on WebSearch
result summaries instead of the full page text:
- api.slack.com/methods/chat.postMessage -- full response schema
- modelcontextprotocol.io/docs/concepts/tools -- canonical CallToolResult shape
- quantconnect.com/docs/.../algorithm-framework/overview -- full flow diagram

The apxml and bolt-python#564 pages were read in full and corroborate the
implications above. The QuantConnect flow is well-covered by the search
result excerpts (which quote the docs verbatim). Slack chat.postMessage is
familiar enough from our existing `backend/slack_bot/formatters.py` and
`commands.py` usage that the gap is low-risk.

Research Gate checklist:
- [x] 3+ authoritative sources (15+ collected, 6+ authoritative)
- [x] 10+ unique URLs (15)
- [x] Full papers read (apxml MCP errors, bolt-python#564)
- [x] All claims cited with URLs
- [x] 7/7 categories covered
- [x] Consensus + pitfalls noted (see sections 4, 5, 6)
