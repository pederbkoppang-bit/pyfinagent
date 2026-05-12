---
step: 25.1
slug: wire-check-stop-losses-autonomous-loop
tier: moderate
cycle_date: 2026-05-12
---

## Research: Wire `check_stop_losses()` into the autonomous daily loop with auto-sell

### Search queries run (three-variant discipline)

1. **Current-year frontier**: `systematic stop-loss execution autonomous trading loop 2026`
2. **Last-2-year window**: `Alpaca broker order state machine paper trading stop loss execution 2025`
3. **Year-less canonical**: `idempotent stop-loss execution duplicate sell prevention trading system` / `stop loss trading system design position existence natural idempotency guard`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://arxiv.org/html/2604.27150 | 2026-05-12 | paper | WebFetch | "exit design matters meaningfully: stronger configurations improve risk-adjusted performance and generally favor tighter loss limits"; applies exit rules as "a unifying policy across the swarm" |
| https://docs.alpaca.markets/us/docs/orders-at-alpaca | 2026-05-12 | official doc | WebFetch | Full Alpaca order state machine: new → partially_filled → filled; stop orders require price-threshold activation before converting to market order — key for understanding the paper_trader.py ExecutionRouter path |
| https://blog.traderspost.io/article/stop-loss-strategies-algorithmic-trading | 2026-05-12 | authoritative blog | WebFetch | "Position state as natural guard: query current position size before executing. If position already closed/reduced, skip stop." — natural idempotency pattern that matches paper_trader.execute_sell's own guard at line 233 |
| https://python-statemachine.readthedocs.io/en/latest/async.html | 2026-05-12 | official doc | WebFetch | Async state machine locking: "only one coroutine acquires the processing lock at a time" — run-to-completion model; directly applicable to asyncio.to_thread wrapping stop-loss execution in autonomous_loop.py |
| https://thearchitectsnotebook.substack.com/p/advanced-idempotency-in-system-design | 2026-05-12 | authoritative blog | WebFetch | "Position existence checks function as natural guards — verifying a position exists before executing prevents duplicate fills. Client-order-ID strategies ensure each sell order carries a unique identifier." |
| https://docs.alpaca.markets/us/docs/paper-trading | 2026-05-12 | official doc | WebFetch | Paper trading fill mechanics — orders filled when marketable; stop orders not present in pyfinagent's execution path (BQ-sim is the primary mode), confirming position-existence is the effective guard |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://dzone.com/articles/designing-stop-loss-ai-automated-trading | blog | HTTP 403 returned |
| https://www.ainvest.com/news/idempotency-keys-prevent-duplicate-trades-digital-finance-2508/ | blog | Empty page returned |
| https://medium.com/javarevisited/idempotency-in-distributed-systems-preventing-duplicate-operations-85ce4468d161 | blog | Behind Medium paywall |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | official doc | Fetched but redirected to software-agent context, not trading-loop specifics; file-based handoff pattern directly quoted in CLAUDE.md |
| https://www.quantifiedstrategies.com/automated-trading-stategies/ | blog | Snippet only; general survey |
| https://brokerchooser.com/invest-long-term/risk-management/stop-loss-order-alpaca-trading | blog | Snippet only; summarizes Alpaca stop-order UI |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on stop-loss execution in autonomous trading loops and idempotent order management. Result: found 1 directly relevant new paper (arXiv 2604.27150, April 2026) that confirms tighter stop-loss limits improve risk-adjusted performance in autonomous trading agent swarms and recommends applying stop checks as a unifying policy across the portfolio — consistent with the proposed Step 5.6 placement. No new architecture supersedes the position-existence-as-natural-guard pattern; it remains the canonical approach in both 2025 and 2026 practitioner literature.

---

### Key findings

1. **Exit rules before entry decisions, always** — arXiv 2604.27150 documents exit rules as "a unifying policy across the swarm" applied before new-entry decisions. This validates placing Step 5.6 (stop check) between mark_to_market (Step 5) and decide_trades (Step 6). (Source: arXiv 2604.27150, https://arxiv.org/html/2604.27150)

2. **Position existence is the natural idempotency guard** — `execute_sell()` at paper_trader.py:233 calls `self.get_position(ticker)` and returns `None` immediately if the position does not exist. A second call for the same ticker after the first sell has already deleted the position row will simply return `None` harmlessly. No external deduplication key is needed. (Source: https://thearchitectsnotebook.substack.com/p/advanced-idempotency-in-system-design, confirmed by code reading at paper_trader.py:233-236)

3. **`execute_sell()` accepts `quantity=None` to close whole position** — docstring at paper_trader.py:232: "If quantity is None, sells entire position." The stop-loss case should pass `quantity=None` (or equivalently omit the argument) to guarantee a full close regardless of partial fills from earlier in the same cycle. (Source: paper_trader.py:224-232)

4. **`check_stop_losses()` returns only tickers, not quantities** — paper_trader.py:414-423 returns `list[str]` of triggered tickers, not position objects. The caller must fetch each position's current price separately OR pass `quantity=None` to `execute_sell()` and let it fetch the full position. The cleanest pattern is to iterate triggered tickers and call `execute_sell(ticker, quantity=None, reason="stop_loss_trigger")`. (Source: paper_trader.py:414-423)

5. **Async wrapper is mandatory** — all `execute_sell`/`execute_buy` calls in autonomous_loop.py use `await asyncio.to_thread(trader.execute_sell, ...)` (autonomous_loop.py:399-406) to avoid blocking the event loop on BQ + yfinance I/O. Step 5.6 must follow the same pattern. (Source: autonomous_loop.py:399-406)

6. **`portfolio_manager.py:82-88` already has a stop-loss check** — but it is inside `decide_trades()`, which runs at Step 6. The portfolio_manager check is triggered only when a holding_analysis exists for that ticker; if no fresh analysis is available, the stop is silently bypassed. Step 5.6 fires unconditionally, catching stops independently of analysis coverage. (Source: portfolio_manager.py:82-89)

7. **Duplicate-sell risk between Step 5.6 and Step 6** — portfolio_manager.py:82-88 may attempt a SELL for the same ticker at Step 6 after Step 5.6 has already sold it. Because execute_sell checks position existence first (paper_trader.py:233-236), the Step 6 attempt returns `None` harmlessly and emits a warning log. No accounting corruption. (Source: paper_trader.py:233-236)

8. **`closed_tickers` list at Step 7 feeds the learning step at Step 9** — autonomous_loop.py:455-460 calls `_learn_from_closed_trades(closed_tickers, ...)`. Stop-loss sells should be appended to `closed_tickers` so the learning step processes them. (Source: autonomous_loop.py:455-460)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/paper_trader.py` | 414-423 | `check_stop_losses()` definition — returns list of triggered tickers | Zero callers; the bug |
| `backend/services/paper_trader.py` | 224-232 | `execute_sell()` signature — `quantity=None` closes full position | Active, used by decide_trades path |
| `backend/services/paper_trader.py` | 233-236 | Position-existence guard in execute_sell — natural idempotency | Active |
| `backend/services/autonomous_loop.py` | 302-307 | Step 5: mark_to_market | Active |
| `backend/services/autonomous_loop.py` | 309-330 | Step 5.5: kill-switch evaluation; returns early if triggered | Active |
| `backend/services/autonomous_loop.py` | 332-386 | Step 6: decide_trades — includes sector enrichment preamble | Active |
| `backend/services/autonomous_loop.py` | 388-437 | Step 7: execute trades loop — sells first, then buys | Active |
| `backend/services/autonomous_loop.py` | 391-409 | Sell execution pattern: `await asyncio.to_thread(trader.execute_sell, ...)` | Model for Step 5.6 |
| `backend/services/autonomous_loop.py` | 455-460 | Step 9: learn from closed_tickers | Active; must receive stop-loss tickers |
| `backend/services/autonomous_loop.py` | 489-499 | `summary.update()` at Done — shape of the final summary dict | Active |
| `backend/services/portfolio_manager.py` | 82-89 | Stop-loss check inside decide_trades — conditional on analysis coverage | Existing partial guard; NOT a replacement for Step 5.6 |
| `tests/_phase_24_helpers.py` | 1-270 | Phase24Verifier helper class — model for phase-25 verifiers | Active; pattern to copy |
| `tests/verify_phase_24_0.py` | 1-57 | Minimal verifier wrapping Phase24Verifier | Active; pattern to copy |

---

### Consensus vs debate

**Consensus:** Stop checks must run before new entry decisions. Position-existence is the canonical guard against duplicate fills. The `asyncio.to_thread` wrapper is non-negotiable for blocking BQ/yfinance calls in the async loop.

**Debate / open question:** Whether the portfolio_manager.py:82-89 check should be removed after Step 5.6 is wired. The safest approach for this cycle is to leave it in place — redundant stop checks with a position-existence guard are harmless, and removing the portfolio_manager check is a separate scope concern. This should be flagged as a follow-up cleanup.

---

### Pitfalls (from literature and code reading)

1. **Not passing `quantity=None`** — passing `pos.get("quantity")` from a stale in-memory object after mark_to_market could pass a wrong quantity if the position was partially filled earlier. Safest: pass `quantity=None` to let execute_sell fetch fresh state.
2. **Not appending to `closed_tickers`** — if stop-loss sells are not appended, Step 9 (learning) and the final summary's `closed_tickers` key will miss them.
3. **Not wrapping in `asyncio.to_thread`** — calling `trader.check_stop_losses()` and `trader.execute_sell()` synchronously blocks the event loop during BQ reads.
4. **Placing Step 5.6 AFTER Step 6** — would allow decide_trades to queue another SELL for the same ticker, causing a redundant order attempt (harmless but noisy).
5. **`summary["stop_loss_triggered"]` key absent** — the Slack cycle-completion summary (future phase-25.N) will need this key; initialize it to `[]` before the loop so it is always present even when no stops fire.

---

### Application to pyfinagent — exact insertion point

**Exact insertion line:** `autonomous_loop.py:332` — the comment `# ── Step 6: Decide trades ──`.

Step 5.6 must be inserted **between line 330** (the `return summary` of the kill-switch early-exit block) **and line 332** (the Step 6 comment). Inserting at line 332 means the new block runs only when the kill-switch has NOT halted the cycle (correct — we do not want stop-loss sells bypassing the kill-switch).

**Insertion block (pseudocode):**

```python
# ── Step 5.6: Stop-loss enforcement ─────────────────────
logger.info("Paper trading: Step 5.6 -- Checking stop losses")
summary["steps"].append("stop_loss_check")
summary["stop_loss_triggered"] = []
triggered_tickers = await asyncio.to_thread(trader.check_stop_losses)
for ticker in triggered_tickers:
    sl_trade = await asyncio.to_thread(
        trader.execute_sell,
        ticker,
        None,   # quantity=None -> close full position
        None,   # price=None -> fetch live price inside execute_sell
        "stop_loss_trigger",
        None,   # signals=None
    )
    if sl_trade:
        trades_executed += 1
        closed_tickers.append(ticker)
        summary["stop_loss_triggered"].append(ticker)
        logger.info("Stop-loss sell executed: %s", ticker)
```

Note: `closed_tickers` is initialized at autonomous_loop.py:391. Step 5.6 appends to it; Step 9 (learning) then picks up all closed tickers including stop-loss sells.

---

### Verifier design sketch — `tests/verify_phase_25_1.py`

Model on `tests/verify_phase_24_0.py` + `tests/_phase_24_helpers.py`. No shared helper exists yet for phase-25; this verifier should be self-contained (stdlib only) following the same pattern.

**Required assertions (immutable from masterplan):**

1. **`check_stop_losses_has_caller`** — `grep -rn "check_stop_losses" backend/services/autonomous_loop.py` returns at least one match. Implementation: read file text, assert `"check_stop_losses"` substring present.

2. **`stop_loss_trigger_reason_present`** — file text contains the string `"stop_loss_trigger"` in autonomous_loop.py (i.e., the reason argument is passed correctly).

3. **`stop_loss_triggered_in_summary`** — file text contains `"stop_loss_triggered"` as a dict key initialization in the loop body.

4. **`step_56_placed_before_step_6`** — assert the line offset of `"Step 5.6"` is less than the line offset of `"Step 6:"` in autonomous_loop.py.

5. **`unit_test_stop_loss_sells_on_breach`** — presence of a test function that mocks a position with `current_price <= stop_loss_price` and asserts `execute_sell` is called. Can be a standalone `def test_stop_loss_triggered()` in the same file (or a separate `tests/test_stop_loss_step56.py`).

6. **`syntax_clean`** — `python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"` exits 0.

**Verification command (for masterplan):**
```
source .venv/bin/activate && python tests/verify_phase_25_1.py
```

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 read in full)
- [x] 10+ unique URLs total (incl. snippet-only) — 12 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted (portfolio_manager duplicate-check concern)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 6,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
