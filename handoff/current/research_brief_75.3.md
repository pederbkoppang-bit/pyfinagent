# Research Brief -- masterplan step 75.3

**Step:** Audit75 S3 -- MCP signals/data/backtest servers: fabricated-state removal + fail-closed publish path
**Tier:** moderate (caller-specified)
**Researcher:** Layer-3 Researcher (merged external + internal)
**Date:** 2026-07-20
**Status:** COMPLETE -- gate PASSED

---

## 0. BLOCKER FOR THE IMPLEMENTER: the step spec's file paths are wrong

The step spec (and the caller's internal-audit list) names `scripts/mcp_servers/signals_server.py`,
`scripts/mcp_servers/data_server.py`, `scripts/mcp_servers/backtest_server.py`, and
`backend/autoresearch/meta_coordinator.py`. **None of those paths exist.** The real paths are:

| Step spec says | Reality (verified) |
|---|---|
| `scripts/mcp_servers/signals_server.py` | `backend/agents/mcp_servers/signals_server.py` (1887 lines) |
| `scripts/mcp_servers/data_server.py` | `backend/agents/mcp_servers/data_server.py` (478 lines) |
| `scripts/mcp_servers/backtest_server.py` | `backend/agents/mcp_servers/backtest_server.py` (407 lines) |
| `backend/autoresearch/meta_coordinator.py` | `backend/agents/meta_coordinator.py` |

`scripts/mcp_servers/` exists but holds only smoke-test/reconcile scripts (`smoke_test_alpaca_mcp.py`,
`smoke_test_bigquery_mcp.py`, `smoke_test_playwright_mcp.py`, `reconcile_alpaca_deny_list.py`).
The **line numbers in the step spec are correct** and match the real files exactly -- only the
directory is wrong. `.mcp.json:44,55,66,77` launches the real servers from
`backend/agents/mcp_servers/`.

---

## 1. Search queries run (3-variant discipline)

| # | Variant | Query | Outcome |
|---|---------|-------|---------|
| 1 | year-less canonical | `idempotency key cache eviction replay stored response API design` | 10 URLs; led to Stripe official docs |
| 2 | current-year (2026) | `MCP server security fail-closed degraded state LLM agent tool 2026` | 10 URLs; led to arXiv AgentCheck 2607.11098 |
| 3 | last-2-year (2025) | `defensive programming trading system stale data provenance flag synthetic data 2025` | 8 URLs; mostly off-target (see Recency scan) |

Plus direct fetches of canonical official docs (MCP spec, MITRE CWE, Pydantic, CPython stdlib).

---

## 2. Read in full (7; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://modelcontextprotocol.io/specification/2025-06-18/server/tools | 2026-07-20 | Official spec (tier 2) | WebFetch, full page | Tool **execution** errors -- explicitly including "**Business logic errors**" -- MUST be reported in the RESULT with `isError: true`, NOT as JSON-RPC protocol errors. Servers **MUST** "Validate all tool inputs" and "Sanitize tool outputs". With an `outputSchema`, "Servers **MUST** provide structured results that conform to this schema." |
| 2 | https://docs.stripe.com/api/idempotent_requests | 2026-07-20 | Official docs (tier 2) | WebFetch, full page | Canonical idempotency semantics: saves "the resulting status code and body of the first request ... **regardless of whether it succeeds or fails**"; replays "the same result, **including `500` errors**". After pruning (>=24h): "We **generate a new request** if a key is reused after the original is pruned." Results are NOT saved when params fail validation or on concurrent conflict -- "You can retry these requests." |
| 3 | https://cwe.mitre.org/data/definitions/636.html | 2026-07-20 | Official standards body (tier 2) | WebFetch, full page | CWE-636 Not Failing Securely ('Failing Open'): "When the product encounters an error condition or failure, its design requires it to fall back to a state that is less secure than other options that are available." Root cause is choosing "fail functional" over "fail safe" to preserve operational continuity. Parent: CWE-755 (Improper Handling of Exceptional Conditions), CWE-657 (Violation of Secure Design Principles). |
| 4 | https://pydantic.dev/docs/validation/latest/api/pydantic/types/ | 2026-07-20 | Official docs (tier 2) | WebFetch, full page (after 301 from docs.pydantic.dev) | "When the secret value is nonempty, it is displayed as `'**********'` instead of the underlying value in calls to `repr()` and `str()`." Accessor is `get_secret_value()`. Confirms the `str(SecretStr)` footgun. |
| 5 | https://docs.python.org/3/library/dataclasses.html | 2026-07-20 | Official docs (tier 2) | WebFetch, full page | Dataclass instances are **normal class instances accessed via attributes, not dictionaries** -- no `.get()`, no subscripting, no mapping protocol. `asdict(obj)` converts "to a dict of its fields, as `name: value` pairs" and "raises `TypeError` if _obj_ is not a dataclass instance"; `fields()` returns the `Field` tuple. |
| 6 | https://docs.python.org/3/library/unittest.mock.html | 2026-07-20 | Official docs (tier 2) | WebFetch, full page | "Mocks are callable and create attributes as new mocks when you access them" -- so an unspecced mock answers methods the real object does not have. `spec` -> "Accessing any attribute not in this list will raise an `AttributeError`". `autospec=True` -> "Methods and functions being mocked will have their arguments checked and will raise a `TypeError` if they are called with the wrong signature." |
| 7 | https://arxiv.org/html/2607.11098 (AgentCheck) | 2026-07-20 | arXiv preprint (tier 1) | WebFetch, full HTML | Fault-injection workbench for LLM agents over MCP. 12 fault types incl. **B1 stale data** and **B4 silent empty**. Headline: "The failures are usually silent, confident use of incorrect tool outputs rather than crashes." Category B (data quality) is the worst category for every agent tested -- best agent 29/40, weakest 13/40. Retry mitigations fixed timeouts (3/10 -> 10/10) but "stale-data faults remain near 3-4 of 10 regardless of the mitigation", requiring architectural fixes like "freshness-aware retrieval" or "timestamp checks". |

## 3. Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://zuplo.com/learning-center/implementing-idempotency-keys-in-rest-apis-a-complete-guide | Vendor blog | Superseded by Stripe official docs (#2) |
| https://oneuptime.com/blog/post/2026-01-30-idempotency-implementation/view | Blog | Community tier; Stripe canonical preferred |
| https://thecodeforge.io/system-design/idempotency-api-design/ | Blog | Community tier |
| https://thearchitectsnotebook.substack.com/p/advanced-idempotency-in-system-design | Newsletter | Community tier |
| https://medium.com/@mathildaduku/how-to-design-idempotent-apis-safely-what-to-cache-and-what-to-ignore-feb93a16fc00 | Medium | Community tier; useful snippet: do not cache transient 5xx |
| https://dev.to/leonardkachi/idempotency-keys-your-apis-safety-net-against-chaos-j1b | dev.to | Community tier |
| https://dev.to/aliasgarmk/idempotency-in-payment-apis-not-optional-not-later-3b2j | dev.to | Community tier |
| https://www.xflowpay.com/blog/idempotency-key | Vendor blog | Community tier |
| https://ieeexplore.ieee.org/document/11394790/ (MCP-Secure) | IEEE paper | Paywalled; abstract only -- host-side enforcement wrapper, read-only defaults, approval gating |
| https://pipelab.org/blog/state-of-mcp-security-2026/ | Industry report | Snippet: >50 tracked MCP vulns, 13 critical |
| https://authzed.com/blog/timeline-mcp-breaches | Industry blog | Snippet only |
| https://www.fiddler.ai/blog/mcp-agent-observability | Vendor blog | Snippet: "Your MCP Agent Is Failing Silently" -- corroborates #7 |
| https://konghq.com/blog/engineering/mcp-tool-governance-security-meets-context-efficiency | Vendor blog | Snippet only |
| https://dataintellect.com/blog/stale-data-measuring-what-isnt-there/ | Industry practitioner | Snippet: measuring stale data in trading systems |
| https://arxiv.org/pdf/2602.23784 (TradeFM) | arXiv | Off-topic (generative market simulation, not fail-closed) |
| https://arxiv.org/pdf/2504.02486 | arXiv | Off-topic (data curation for scientific discovery) |
| https://dev.to/aws-heroes/mcp-tool-design-why-your-ai-agent-is-failing-and-how-to-fix-it-40fc | dev.to | Community tier |
| https://aaif.io/blog/mcp-is-growing-up/ | Blog | Community tier |
| https://www.mayerbrown.com/en/insights/publications/2026/07/synthetic-data-as-a-deal-asset... | Law firm | Legal/IP framing of synthetic data, not engineering |
| https://media.defense.gov/2025/Jan/29/2003634788/-1/-1/0/CSI-CONTENT-CREDENTIALS.PDF | NSA/CISA | Content provenance (C2PA) -- adjacent but media-integrity scoped |

**Total unique URLs collected: 27** (7 read in full + 20 snippet-only).

---

## 4. Recency scan (last 2 years, 2024-2026)

**Performed.** Result: **one materially new finding**, plus a negative result worth recording.

1. **NEW (2026), directly on point -- arXiv:2607.11098 "AgentCheck"** (read in full, #7). This is the
   first source found that empirically measures what happens when an MCP tool degrades. Its
   central finding is the exact failure mode this step remediates: agents do not crash on bad
   tool data, they *silently and confidently proceed*. Its fault taxonomy gives us names for our
   own bugs -- **B4 "silent empty"** is `get_macro` returning `{"data": []}`; **B1 "stale data"** is
   the `2025-12-31` hardcoded cutoff; **A4 "schema drift"** is the `BacktestResult`-as-dict bug.
   Its measured mitigation result is a warning for us: retries fix transport faults but
   **stale-data faults stay broken** unless you add explicit timestamp/freshness checks. That
   directly justifies the step's requirement that the response carry the *effective date range*
   rather than just serving whatever the cutoff produced.

2. **NEGATIVE result on the 2025 window.** The 2025-scoped query for "stale data provenance /
   synthetic data flags in trading systems" surfaced almost entirely *legal and IP* literature
   (synthetic data as a trade-secret / M&A diligence asset, EU AI Act training-data provenance)
   plus generative market-simulation papers (TradeFM). There is **no 2025 engineering literature
   that supersedes the canonical sources** on fail-closed design or idempotency. The canonical
   sources remain authoritative: CWE-636 (fail-open weakness) and Stripe's idempotency contract
   are unchanged and uncontradicted by anything in the window.

3. **MCP spec revision.** The current spec revision fetched is `2025-06-18`. The `isError` /
   protocol-error split and the `outputSchema` MUST-conform rule are present in it; nothing in
   the 2026 security literature proposes changing that contract -- the 2026 work (MCP-Secure,
   AgentCheck) layers *enforcement and testing* on top of it rather than revising it.

---

## 5. Key findings (external) and the rule each one gives us

1. **Fail-open is a named, classified weakness -- not a style preference.** "When the product
   encounters an error condition or failure, its design requires it to fall back to a state that
   is less secure than other options" (CWE-636). Its stated root cause is choosing "fail
   functional" to preserve operation. `get_portfolio`'s `except -> $10K book` is textbook
   CWE-636 with CWE-755 as parent.
   -> **Rule: an exception handler that invents a value is a defect. Return an explicitly
   degraded marker and let the caller refuse.**

2. **MCP already has the right channel for a refusal.** Execution and "business logic errors"
   belong in the result with `isError: true`; only unknown-tool/invalid-args are protocol errors.
   Servers **MUST** "Sanitize tool outputs" (MCP spec).
   -> **Rule: a degraded publish must return a well-formed result that says `published:false`
   with a reason -- not raise, and not silently succeed.** This is compatible with the server's
   existing "never raises, always returns the `_empty_response` shape" invariant
   (`signals_server.py:300-301`).

3. **The canonical idempotency contract stores the TRUE terminal outcome, including failures,
   and RE-EXECUTES after eviction.** Stripe: saves status+body "regardless of whether it
   succeeds or fails"; replays "the same result, including `500` errors"; and after pruning
   "We generate a new request if a key is reused."
   -> **Rule: never synthesize a terminal outcome for a key whose stored outcome is gone.
   On cache miss, re-execute (or refuse) -- do not assume success.**

4. **Stripe deliberately does NOT persist validation failures**, precisely so they stay
   retryable ("If incoming parameters fail validation ... we don't save the idempotent result
   ... You can retry these requests.").
   -> **Rule: rejections must not be permanently deduped** -- exactly what gap4-06 requires.

5. **Dataclasses have no mapping interface.** `.get()` on a `BacktestResult` raises
   `AttributeError`, and if that lands in a broad `except Exception` the tool reports a generic
   ERROR forever while the expensive work still ran.
   -> **Rule: mirror the attribute-access pattern already proven in-repo; if a dict is genuinely
   wanted, use `dataclasses.asdict()`.**

6. **Unspecced mocks are why this class of bug survives testing.** "Mocks ... create attributes
   as new mocks when you access them" -- so `Mock().get_portfolio()` succeeds in a test while the
   real `PaperTrader` raises. `spec`/`spec_set` raise `AttributeError` for non-existent
   attributes; `autospec=True` additionally raises `TypeError` on wrong signatures.
   -> **Rule: every PaperTrader/engine/cache mock in the new tests MUST be
   `create_autospec(PaperTrader, instance=True)` (or `spec_set=`).** A plain `Mock()` would let
   gap4-01 regress silently.

7. **Silent acceptance is the dominant agent failure mode, and stale data resists retry
   mitigation** (AgentCheck: "silent, confident use of incorrect tool outputs rather than
   crashes"; B1 stale "remain near 3-4 of 10 regardless of the mitigation").
   -> **Rule: staleness must be surfaced in-band (effective date range in the response), not
   just fixed at the source.** A consumer that cannot see the as-of date cannot detect drift.

### Consensus vs debate

Consensus is strong and one-directional across all seven sources: on error, **degrade visibly and
refuse the privileged action**; never substitute a plausible default. The only genuine tension is
*availability vs safety* -- CWE-636 names "fail functional" as the tempting anti-pattern, and the
`$10K` fallback is exactly that trade taken in the wrong direction for a code path that sizes
orders. For a **paper-trading** surface the safety side wins unambiguously; the step's
BOUNDARY (paper-only, thresholds byte-untouched) keeps the blast radius contained.

---

## 6. Internal code inventory (all anchors verified this session)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/mcp_servers/signals_server.py` | 1887 | Signals MCP server (6 tools + 3 resources) | 5 findings (gap4-01/04/05/06, security-05) |
| `backend/agents/mcp_servers/data_server.py` | 478 | Data MCP server | 2 findings (gap4-07/08) |
| `backend/agents/mcp_servers/backtest_server.py` | 407 | Backtest MCP server | 1 finding (gap4-09) |
| `backend/services/paper_trader.py` | 1250+ | Real trading API | Reference (correct signatures) |
| `backend/backtest/backtest_engine.py` | 1167 | `BacktestResult` dataclass @ :111-123 | Reference |
| `backend/agents/meta_coordinator.py` | -- | `run_proxy_validation` @ :276 | Reference (CORRECT pattern) |
| `backend/agents/llm_client.py` | -- | `unwrap_secret` @ :32 | Reference (the fix helper) |
| `backend/backtest/cache.py` | -- | `cached_macro/prices/fundamentals` @ :380/:282/:331 | Reference (return types) |
| `tests/test_mcp_servers.py` | -- | Existing MCP tests | **Vacuous -- see 6.9** |
| `.mcp.json` | 44-77 | Server launch config | Reference |

### 6.1 gap4-01 (P1) -- fabricated portfolio -> real sizing

`signals_server.py:1245-1276`. `get_portfolio()` calls `self.paper_trader.get_portfolio()` at
**:1259**. **That method does not exist on `PaperTrader`** -- verified by enumerating every `def`
in `backend/services/paper_trader.py`; the class (`:66`) exposes `get_or_create_portfolio` (`:95`),
`get_positions` (`:115`), `get_position` (`:118`), `execute_buy` (`:123`), `execute_sell` (`:391`),
`mark_to_market` (`:555`) etc., and **no `get_portfolio`**.

Causal chain (this is a live defect, not a latent one):
`AttributeError` at :1259 -> caught by the bare `except Exception` at **:1268** -> returns
`{"total_value": 10000.0, "positions": {}, "cash": 10000.0, "error": ...}` (**:1270-1276**) ->
consumed by `publish_signal` step 5 at **:361** -> `size_position(signal, portfolio)` at **:366**
reads `total_value` as equity (**:988**) -> `risk_check(portfolio, ...)` at **:396** reads
`total_value`/`cash`/`positions` from the same fabricated dict. So **every** notional, cash and
concentration gate is computed against a $10K empty book instead of the real one.

The `_SIGNALS_AVAILABLE`/no-trader branch at **:1249-1255** fabricates the same $10K with **no
error key and no stub marker** -- indistinguishable from a real snapshot.

**Correct replacement shapes (verified):**
- `get_or_create_portfolio() -> dict` (`paper_trader.py:95-113`) with keys
  `portfolio_id, starting_capital, current_cash, total_nav, total_pnl_pct,
  benchmark_return_pct, inception_date, updated_at`.
  So the mapping is **`total_nav` -> `total_value`** and **`current_cash` -> `cash`** (matches the
  step spec).
- `get_positions() -> list[dict]` (`paper_trader.py:115-116`, delegates to
  `self.bq.get_paper_positions()`). **It returns a LIST, not a dict** -- the server's contract
  wants `positions` keyed by ticker, so the fix must build
  `{p["ticker"]: p for p in get_positions()}`. Do not assume a dict.
- Note `risk_check` reads `pos.get("price")` and `pos.get("shares")` (`:861-862`, `:829`, `:839`),
  so the per-position dicts must expose those keys (or be normalized) or the concentration maths
  silently reads 0.

### 6.2 gap4-05 (P2) -- zero-price notional bypass + a drawdown key nobody supplies

`signals_server.py:816-832`. Price resolution is `explicit -> position record -> 0.0`, and the
comment at **:819-821** admits it: "else 0.0 (concentration check still runs; cash check passes
trivially -- documented degraded mode)". Then **:832** `proposed_notional = unit_price * shares_int`.

With `unit_price == 0.0`, `proposed_notional == 0.0`, so on a BUY:
- per-ticker concentration (**:871-877**) -> adds 0 -> passes
- total exposure (**:880-886**) -> adds 0 -> passes
- cash floor (**:890-894**) -> `0.0 > cash` is False -> passes

All three financial gates pass **trivially**. The step's fix (reject with `unknown_price`) is
correct and must be placed **before** those gates.

`current_dd` is read at **:798** as `portfolio.get("current_drawdown_pct", 0.0)`. The
`get_portfolio()` return shape (**:1261-1267**) **never sets that key**, so the drawdown circuit
breaker at **:896-901** always compares `0.0 <= -15.0` -> False -> **permanently inert**. The
drawdown machinery already exists on the server (`track_drawdown`, returning
`{peak, equity, drawdown_pct, tier, kill_switch}` at **:1237-1243**, with `_peak_equity` state);
step 5 must call it and inject `current_drawdown_pct` into the portfolio dict before `risk_check`.

`size_position`'s explicit override at **:978-985** returns `explicit_val` **unclamped** -- any
caller-supplied `size_usd` bypasses both the 5%-of-equity arm and the `max_position_usd: 1000.0`
absolute cap from `get_risk_constraints()` (**:1294-1295**). Clamp to
`min(explicit_val, hard_cap)` where `hard_cap = min(equity * max_position_pct/100,
max_position_usd)`.

### 6.3 gap4-06 (P2) -- dedup fabricates success after FIFO eviction

State: `_seen_signal_ids: set` (**:79**), `_recent_responses: Dict` (**:80**),
`_recent_responses_limit = 50` (**:81**). `_remember` (**:271-282**) adds to the set and the dict,
then evicts the oldest dict entry past 50 (**:279-282**) -- **but never removes from the set.**

That asymmetry is the bug. At **:334** `signal_id in self._seen_signal_ids` stays True forever,
while `self._recent_responses.get(signal_id)` (**:335**) returns None after eviction, falling into
**:343-348**:

```python
# Cache miss but seen -- synthesize a minimal dedup response.
resp = self._empty_response(signal_id=signal_id)
resp["published"] = True      # <-- fabricated terminal outcome
```

So a signal that was **rejected** (`risk_rejected:*` at :400-403, `trade_rejected` at :421-424 /
:438-441, or `backend_unavailable` at :354-358 -- all of which call `_remember`) is reported as
`published: True` once it ages out of the 50-entry window. This inverts the true outcome, and it is
the precise opposite of the Stripe contract (replay the true result including failures;
re-execute after pruning).

Secondary: `_seen_signal_ids` is an **unbounded set** -- a slow memory leak on a long-lived stdio
server, and the mechanism that keeps the fabrication reachable indefinitely.

### 6.4 gap4-04 (P2) -- emit_candidates fabricates, and the "real DSR" branch is unreachable

`signals_server.py:1808-1874`. `base_signals = ["BUY","SELL","BUY","HOLD","BUY"]` (**:1840**),
`base_factors` (**:1841-1847**), `confidence = round(0.60 + 0.05*(i%7), 3)` (**:1865**),
`dsr = round(0.92 + 0.01*(i%8), 3)` (**:1858/:1860**) -- entirely synthetic, presented in the same
shape as a real candidate.

Dead branch proof: `returns_by_variant` is initialized `{}` at **:1835** and never written (the
`TODO(phase-3.7.4)` at :1834 says so). `dsr_source` can only become `"compute_dsr_real"` inside
`if compute_dsr_fn and returns_by_variant:` (**:1836-1837**), which requires a truthy dict.
Therefore the `if dsr_source == "compute_dsr_real":` branch at **:1853-1858** is **statically
unreachable**. Delete :1853-1858 (and the now-dead `compute_dsr_fn` plumbing at :1822-1829) or
wire real returns -- the step says delete.

**CONSUMER CONSTRAINT (do not break this):** `scripts/harness/mcp_ab_test.py:484-490` calls
`emit_candidates("AAPL", n=5)` and computes `candidates_per_call` plus
`dsr_annotated = all("dsr" in c for c in cands)`; the in-file comment (**:470-472**) states this
exists "so the immutable verification can assert >= 5". **The fix must keep returning >=5
candidates, each still carrying a `dsr` key.** Add `stub: true` + `reason:
PENDING_IMPLEMENTATION` additively; do not remove the count or the `dsr` field.

### 6.5 security-05 (P2) -- SecretStr into WebClient

`signals_server.py:457` reads `slack_token = getattr(self.settings, "slack_bot_token", "")` and
**:477** passes it straight to `WebClient(token=slack_token)`. If the setting is typed `SecretStr`,
the SDK receives the wrapper; per the Pydantic docs a non-empty `SecretStr` stringifies to
`'**********'`, so the auth header is garbage -- and because a non-empty `SecretStr` is truthy,
the `if not slack_token` guard at **:459** does **not** catch it. This is the exact phase-51.1
failure mode recorded in project memory (SecretStr killed 4 alpha overlays).

Fix helper already exists: `backend/agents/llm_client.py:32 def unwrap_secret(v) -> str`.
Established call convention (mirror it verbatim) -- `llm_client.py:1953-1956`:
```python
anthropic_key = unwrap_secret(getattr(settings, "anthropic_api_key", ""))
```
and `backend/services/meta_scorer.py:202-203` does a local import inside the function, which is the
right idiom here too since the Slack block is deliberately lazy-imported (**:450-452** "Hard
invariant: nothing above this line imports slack_sdk"). `unwrap_secret` is a no-op for `str`,
returns `""` for `None` (proven by `backend/tests/test_phase_51_1_secretstr.py:25-33`), so it is
safe regardless of the setting's declared type.

### 6.6 gap4-07 (P2) -- get_macro iterates a dict as a list and ignores `series`

`backend/backtest/cache.py:380` declares `def cached_macro(cutoff_date: str) -> dict` and builds
`result[series_id] = entry` (**:389-397**) -- a **dict keyed by series_id**.

`data_server.py:172` calls it, then **:177-182** does `for item in macro_data: ... item.get("date")`.
Iterating a dict yields its **keys (str)**, and `str` has no `.get`, so this raises
`AttributeError` on the first iteration for any non-empty macro payload -> caught at **:187** ->
returns `{"series": series, "data": [], "error": "'str' object has no attribute 'get'"}`.

Consequence: the `macro://` surface returns **empty data whenever data actually exists**, and looks
merely "empty" to a caller that ignores `error`. This is AgentCheck's **B4 "silent empty"** exactly.
Additionally the `series` argument is accepted and echoed but **never used to filter** -- the fix
must select the requested series from the dict.

### 6.7 gap4-08 (P2) -- hardcoded 2025-12-31 cutoffs + dead market prefix

- `data_server.py:95` -- `cache.cached_prices(ticker, "2023-01-01", "2025-12-31")`
- `data_server.py:142` -- `cache.cached_fundamentals(ticker, "2025-12-31")`
- `data_server.py:172` -- `cache.cached_macro("2025-12-31")`

Today is 2026-07-20, so this surface serves ~7-month-stale data with no as-of marker (AgentCheck
**B1 stale data** -- the fault class its own mitigations could not fix without explicit timestamp
checks).

**The correct idiom already exists in the same file**: `get_features` (**:231-252**) declares
`cutoff_date: Optional[str] = None` and does `if cutoff_date is None: cutoff_date =
date.today().isoformat()` (**:251-252**). Mirror that for prices/fundamentals/macro and echo the
effective range in the response (the step's requirement, and independently justified by finding 7).

Dead code: **:89-92** parses `market` out of `"NO:EQNR"`-style tickers, and **:136-139** repeats it,
but `market` is then **never used** -- not in the cache call, not in the response. Either thread it
through or drop it (the step allows either).

### 6.8 gap4-09 (P2) -- BacktestResult dataclass treated as a dict; **the step spec's field list is also wrong**

`backtest_server.py:117` gets `result = engine.run_backtest()`, then **:119** does
`result.get("full_period", {})`. `BacktestResult` is a **dataclass** (`backtest_engine.py:111`), so
`.get` raises `AttributeError` -> caught at **:133** -> `{"status": "ERROR"}`. The tool therefore
**runs the entire walk-forward backtest and then always reports ERROR**.

**Actual `BacktestResult` fields (`backtest_engine.py:112-123`) -- verified verbatim:**
```
windows: list[WindowResult]      aggregate_sharpe: float
aggregate_return_pct: float      aggregate_alpha_pct: float
aggregate_max_drawdown_pct: float  aggregate_hit_rate: float
total_trades: int                feature_importance_mdi: dict[str, float]
feature_importance_mda: dict[str, float]   nav_history: list[dict]
strategy_params: dict            all_trades: list[dict]
```

**Warning for the implementer:** the step spec instructs mapping
"`aggregate_sharpe, dsr, return_pct, max_drawdown_pct, num_trades`". Only `aggregate_sharpe`
exists. There is **no `dsr` field, no `return_pct`, no `max_drawdown_pct`, no `num_trades`, and no
`run_id`** on `BacktestResult`. The real names are `aggregate_return_pct`,
`aggregate_max_drawdown_pct`, `total_trades`. **DSR is not produced by the engine at all** -- either
drop the `dsr` key from the tool response or compute it explicitly (e.g. via
`backend/services/perf_metrics.compute_dsr`, the same helper `emit_candidates` references at
:1825); do not emit a fabricated 0.0, which would recreate the very class of bug this step removes.
The `run_id` at :123 has the same problem.

**Canonical pattern to mirror** -- `backend/agents/meta_coordinator.py:306-311` (attribute access):
```python
result = engine.run_backtest()
if result and result.aggregate_sharpe is not None:
    ...
    return result.aggregate_sharpe
```

`bq_client` arg: `backtest_server.py:109` passes `bq_client=self.bq_client.client` (the raw
google-cloud client) while `self.bq_client` is a `BigQueryClient` wrapper (**:29, :60**).
`BacktestEngine.__init__` already normalizes -- `backtest_engine.py:176-177`:
`if hasattr(bq_client, 'client'): bq_client = bq_client.client`. So passing the **wrapper**
(`self.bq_client`) is the type-consistent call and is what the engine expects; the current
pre-unwrapping works only by accident of that guard. Aligning it is safe.

`timeout_seconds`: set at **:54** (`self.timeout_seconds = 30`) and **never read anywhere** -- the
only other occurrence is the misleading comment "Run backtest with timeout" at **:115**. Enforce it
or delete it (the step allows either); note a real walk-forward backtest will vastly exceed 30s, so
enforcing it naively would break the tool -- **prefer deletion, or raise it to a realistic value**
with the comment corrected.

### 6.9 Why every one of these shipped: the existing tests assert envelopes, not outcomes

`tests/test_mcp_servers.py` contains **zero mocks** -- `grep` for `spec=|autospec|Mock()` returns
nothing. Its assertions are structurally incapable of catching any finding above:

```python
def test_get_macro(self):
    result = DataServer().get_macro("VIX")
    assert result["series"] == "VIX"        # passes on the AttributeError path (gap4-07)

def test_run_backtest(self):
    result = BacktestServer().run_backtest({"holding_days": 90})
    assert "status" in result               # passes when status == "ERROR" (gap4-09)
```

They assert the *shape of the envelope* while the payload is an error. The new tests must assert
**outcomes and provenance** (`published is False`, `reason == "degraded_portfolio"`,
`stub is True`, `allowed is False`, `conflicts == ["unknown_price"]`), not key presence.

### 6.10 Blast radius of a fail-closed refusal (asked for explicitly)

**The signals MCP server is NOT on the live money path.** The autonomous trading loop calls
`PaperTrader.execute_buy/execute_sell` directly; `SignalsServer` is not imported by any live
service. Its consumers are:

| Consumer | Path | Impact of fail-closed |
|---|---|---|
| Claude Code sessions | `.mcp.json:68` (`alwaysLoad: false`) | Interactive only |
| A/B harness | `scripts/harness/mcp_ab_test.py:327, 470-490` | **Must re-run** -- asserts >=5 candidates + `dsr` key (see 6.4) |
| Position-limits drill | `scripts/go_live_drills/position_limits_test.py:198` | Passes explicit `price=$100`, so the `unknown_price` gate does not fire; but it pins boundary semantics ("strict `>` at line 872") -- **thresholds must stay byte-untouched**, per the step BOUNDARY |
| Kill-switch drill | `scripts/go_live_drills/kill_switch_test.py:131` | Re-run |
| First-week monitoring drill | `scripts/go_live_drills/first_week_monitoring_test.py:219` | Sets `server._peak_equity = 10000.0` directly -- **the drawdown wiring in 6.2 must not break this seam** |
| Slack e2e drill | `scripts/go_live_drills/slack_signals_e2e_test.py:24-29` | AST-level checks (S9-S14) on `publish_signal` -- verifies it *calls* `chat_postMessage` and degrades to `slack_not_configured`. The `unwrap_secret` change is AST-compatible; **re-run to confirm** |
| Unit tests | `tests/test_mcp_servers.py`, `tests/test_mcp_integration.py` | Will need updating (see 6.9) |

**Net: no live-money path breaks.** The real risk is drills/harness expectations, all listed above.

---

## 7. Application to pyfinagent -- implementation guidance

1. **Two distinct degraded states, both marked.** Do not collapse them.
   `no-backend` (`:1249`) and `exception` (`:1268`) must BOTH return
   `{"stub": true, "degraded": true, "reason": ..., "total_value": 0.0, "cash": 0.0,
   "positions": {}}`. Use **0.0, not 10000.0** -- a zeroed book makes `size_position` return 0.0
   naturally (its `equity <= 0.0 -> return 0.0` guard at **:991-997**) and makes every BUY gate
   fail closed even if the explicit refusal is somehow bypassed. Defense in depth.

2. **Refuse at the publish seam, not just the data seam.** In step 5 (**:361**), after fetching
   the snapshot, check the marker and return `published: false, reason: "degraded_portfolio"`
   *before* sizing. Reuse `_empty_response(signal_id=...)` so the return-shape invariant
   (**:310**) holds, and `_remember` the refusal so the true outcome is what replays.

3. **Ordering matters in `risk_check`.** Insert the `unknown_price` rejection immediately after
   price resolution (**:827-831**) and **before** `proposed_notional` is computed at **:832**.
   Return via `_risk_response(False, 0.0, max_per_ticker_pct, ["unknown_price"], ...)` to keep the
   uniform shape (**:911-928**). Restrict it to BUY (a SELL is de-risking and is already gated by
   `insufficient_position` at :842).

4. **Dedup: store the terminal outcome, replay it truthfully.** Keep one authoritative record per
   `signal_id`: `{"published": bool, "reason": str, ...}`. Then:
   - hit -> replay the stored outcome verbatim + `deduped: true` (already correct at :336-342)
   - **miss-but-seen -> do NOT fabricate.** Either re-execute, or return
     `published: false, reason: "dedup_state_evicted"`. Never `published = True`.
   - **do not persist rejections into the permanent seen-set** (Stripe: validation failures stay
     retryable). Simplest correct fix: make the set and the response cache evict **together**
     (single `OrderedDict` as the one source of truth), which also fixes the unbounded-set leak.

5. **Provenance is a first-class field, and the gate must read it.** `stub: true` +
   `reason: PENDING_IMPLEMENTATION` on every `emit_candidates` candidate, and `publish_signal`
   must reject stub-provenance signals. Marking without enforcing is what let gap4-04 persist --
   the `dsr_source` label was already honest ("placeholder_...") and nothing consumed it.

6. **Dates: use the existing in-file idiom** (`data_server.py:251-252`), make cutoffs
   `Optional[str] = None`, and echo the effective range in the response.

7. **Tests: `create_autospec(PaperTrader, instance=True)`.** This is the single highest-value
   testing decision in the step -- a bare `Mock()` would have happily served
   `.get_portfolio()` and let gap4-01 regress. Mock `BacktestEngine.run_backtest` to return a real
   `BacktestResult(...)` instance (not a dict) so the attribute mapping is genuinely exercised.
   All offline: no BQ, no network, no Slack.

### Pitfalls (from the literature + this codebase)

- **Do not raise out of an MCP tool for a business-logic refusal** -- the spec puts those in the
  result (`isError`/structured refusal), and this server's own invariant says "never raises"
  (**:300-301**).
- **Do not let `except Exception` invent a value** -- CWE-636/CWE-755. Catch, mark, refuse.
- **Do not cache a transient failure as a terminal outcome** -- Stripe excludes validation
  failures and concurrent conflicts from the idempotent record precisely to keep them retryable.
- **Do not "fix" staleness only at the source** -- AgentCheck shows stale-data faults survive
  retry mitigations; surface the as-of date in-band.
- **Do not follow the step spec's `BacktestResult` field list verbatim** (see 6.8) -- 4 of its 5
  named fields do not exist, and `dsr` is not produced by the engine at all.
- **Do not change any threshold** -- `get_risk_constraints()` (**:1286-1300**) and the strict `>`
  comparisons at :874/:883 are pinned by `position_limits_test.py` and by the step BOUNDARY.

---

## 8. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**: 6 official-docs/standards tier + 1 arXiv preprint)
- [x] 10+ unique URLs total (**27**: 7 full + 20 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (section 4; 1 new finding + 1 negative result)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the spec, plus the consumers and the existing tests
- [x] Contradictions / consensus noted (section 5)
- [x] Claims cited per-claim with URL + access date
- [x] Source-quality hierarchy enforced (no community-tier source in the read-in-full set)

---

## 9. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 20,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All 8 findings (gap4-01 P1, gap4-04/05/06/07/08/09 + security-05 P2) reproduced with verified file:line anchors. Two spec corrections block naive implementation: the servers live in backend/agents/mcp_servers/ not scripts/mcp_servers/, and 4 of 5 BacktestResult field names in the step spec do not exist (no dsr/return_pct/max_drawdown_pct/num_trades; real names are aggregate_return_pct, aggregate_max_drawdown_pct, total_trades). PaperTrader.get_portfolio() confirmed nonexistent, so the except at :1268 fabricates a $10K book that really does size and risk-check trades. External consensus (CWE-636, MCP spec isError, Stripe idempotency, AgentCheck arXiv:2607.11098) is one-directional: mark degraded state, refuse the privileged action, replay true terminal outcomes, never re-execute-as-success after eviction. Tests must use create_autospec(PaperTrader) -- unspecced mocks are why gap4-01 shipped. Signals MCP is NOT on the live money path; consumers to re-run are mcp_ab_test.py (asserts >=5 candidates with dsr) and 4 go_live_drills.",
  "brief_path": "handoff/current/research_brief_75.3.md",
  "gate_passed": true
}
```
