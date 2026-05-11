## Research: phase-23.5.7.1 — Fix format_evening_digest KeyError on dict-shaped trades_today

Effort tier: simple (caller-specified). All external-source floors apply unchanged.

---

### Queries run (three-variant discipline)

1. Current-year frontier: "Python defensive coercion dict vs list API response isinstance guard 2026"
2. Last-2-year window: "Python API response envelope unwrapping defensive pattern isinstance dict list 2025"
3. Year-less canonical: "Python KeyError slice(None, 10, None) dict slicing not supported" + "REST API response envelope vs flat list design"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|---------|------|-------------|---------------------|
| https://realpython.com/ref/builtin-exceptions/keyerror/ | 2026-05-09 | Official doc | WebFetch | "KeyError is a built-in exception that occurs when you try to access a missing key in a dictionary." Confirms slice object is not a valid dict key; slicing a dict raises KeyError with the slice as the missing key. |
| https://docs.slack.dev/reference/block-kit/blocks/section-block | 2026-05-09 | Official doc | WebFetch | Section block requires either a `text` object or `fields` array. `text` max 3000 chars, min 1. `mrkdwn` text with "*Today's Trades:* No trades executed today." is fully valid as an empty-state. |
| https://hector-reyesaleman.medium.com/api-response-wrapper-patterns-49d846578ac0 | 2026-05-09 | Authoritative blog | WebFetch | Five patterns: Envelope, JSend, RFC 7807, JSON:API, Pagination wrappers. Dict envelope recommended for collections: "you should never leave an array as the root container of your response." Flat list advantages: simpler for internal microservices. Defensive client handling implied: check HTTP status codes alongside wrapper fields. |
| https://betterstack.com/community/guides/scaling-python/python-pattern-matching/ | 2026-05-09 | Authoritative blog | WebFetch | Pattern matching (Python 3.10+) cleanly handles dict vs list shapes: `case list(l)` vs `case dict(d)`. Compared to `isinstance`: "The pattern-matching version provides a consistent structure that clearly separates the type checking from the value processing." Both approaches are valid; isinstance is simpler for a one-liner guard. |
| https://docs.slack.dev/reference/block-kit/blocks/ | 2026-05-09 | Official doc | WebFetch | Block Kit overview confirms section blocks are a valid building block for empty-state messages. No minimum block count required in a message. |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.positioniseverything.net/invalid-slice-error/ | Blog | Covered by Real Python doc + internal log evidence already definitive |
| https://discuss.huggingface.co/t/keyerror-invalid-key-slice-0-1000-none-please-first-select-a-split/9089 | Forum | Same root cause, dataset-library context not relevant |
| https://myappapi.com/blog/api-design-best-practices-2025 | Blog | hectoraleman Medium article is more specific to envelope patterns |
| https://dev.to/kaushikcoderpy/pydantic-data-validation-border-control-for-python-apis-2026-49p1 | Blog | Pydantic validation not applicable to this caller-side fix |
| https://stackoverflow.blog/2020/03/02/best-practices-for-rest-api-design/ | Blog | Fetched; no direct envelope vs flat-list guidance found |
| https://discuss.python.org/t/facing-trouble-when-slicing-dictionaries-in-a-list/21200 | Forum | Snippet confirms dict slicing raises KeyError |
| https://www.quora.com/Can-you-slice-a-dictionary-in-Python | Forum | Confirmed: slicing a dict produces KeyError with the slice object as key |
| https://peps.python.org/pep-0622/ | PEP | Structural pattern matching PEP -- betterstack article covers the relevant parts |

---

### Recency scan (2024-2026)

Searched for 2025-2026 literature on Python defensive isinstance guard, API response envelope unwrapping, Slack Block Kit empty state. Result: no new findings that supersede the canonical approach. The 2025-2026 community consensus has shifted toward Python 3.10+ structural pattern matching (match/case) as a more readable alternative to isinstance chains, but isinstance remains the simpler, backwards-compatible choice for a one-liner guard. The Slack Block Kit section-block specification is unchanged. No 2024-2026 paper or official doc recommends anything that contradicts Option A or Option B as described below.

---

### Key findings

1. `/api/paper-trading/trades?limit=10` returns a dict envelope `{"trades": [...], "count": N}` -- NOT a flat list. Source: `backend/api/paper_trading.py:226` (`result = {"trades": trades, "count": len(trades)}`).
2. `format_evening_digest` receives `trades_data` directly from `trades_res.json()` (scheduler.py:251). `trades_data` is therefore always a dict when status==200. The fallback `[]` (list) only fires on non-200.
3. `for t in trades_today[:10]:` at `formatters.py:376` -- slicing a dict raises `KeyError: slice(None, 10, None)` because Python converts `d[a:b]` to `d.__getitem__(slice(a,b,None))`, and dicts do not implement `__getitem__` for slice keys. (Source: Real Python KeyError doc; confirmed by `handoff/logs/slack_bot.log:341`.)
4. `format_morning_digest(portfolio_data, reports_data)` -- `reports_data` comes from `GET /api/reports/?limit=5`. The `/reports/` list endpoint (`backend/api/reports.py:28-40`) has `response_model=list[ReportSummary]` and returns a bare list. Therefore `format_morning_digest` is NOT susceptible to the same dict-envelope bug -- its `recent_reports` parameter receives a flat list.
5. The existing test suite (`tests/slack_bot/test_digest_url_semantics.py`) passes `_fake_response(200, [])` (a bare list) as the trades response, which silently masks the dict-envelope mismatch. The tests pass but do not exercise the real API shape.
6. `format_evening_digest` already handles the empty-trades case correctly (lines 387-392): it renders "*Today's Trades:* No trades executed today." -- a valid Slack section block with mrkdwn text satisfying min-length-1. (Source: Slack Block Kit section block doc.)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/formatters.py` | 354-400 | `format_evening_digest` -- builds Block Kit blocks for evening digest | Bug at line 376: `trades_today[:10]` assumes list, raises KeyError when dict |
| `backend/slack_bot/formatters.py` | 309-351 | `format_morning_digest` -- builds Block Kit blocks for morning digest | NOT affected: `recent_reports` comes from a flat-list API endpoint |
| `backend/slack_bot/scheduler.py` | 241-263 | `_send_evening_digest` -- fetches portfolio + trades, calls formatter | Bug source: `trades_data = trades_res.json()` returns dict; no coercion before passing |
| `backend/api/paper_trading.py` | 215-228 | `GET /api/paper-trading/trades` | Returns `{"trades": [...], "count": N}` -- dict envelope |
| `backend/api/reports.py` | 28-40 | `GET /api/reports/` | Returns flat `list[ReportSummary]` -- no envelope |
| `tests/slack_bot/test_digest_url_semantics.py` | 106-132 | Evening digest URL regression tests | Passes `[]` (bare list) as trades body -- masks the real shape bug; must be updated |

---

### Consensus vs debate (external)

The literature (Betterstack pattern matching guide, hectoraleman Medium) notes that dict envelopes are preferred for collection endpoints to enable future pagination metadata, but internal-microservice flat lists are also common. pyfinagent uses a dict envelope for trades (`{"trades": [...], "count": N}`) and a flat list for reports. This mixed convention is the root cause of the mismatch. Both Option A (coerce in formatter) and Option B (coerce in scheduler) fix the immediate crash; the literature slightly favors Option B (keep formatter strictly typed, normalize at the caller boundary) but both are acceptable.

### Pitfalls (from literature)

- The pattern `if isinstance(x, dict): x = x.get("trades", [])` is safe and idiomatic. The `.get()` default of `[]` ensures the formatter always receives a list even if the key is missing or renamed. (Real Python KeyError doc; Python isinstance built-in.)
- Python 3.10+ `match/case` is more readable but adds a language-version dependency and is heavier for a one-liner guard. For this single-callsite fix, `isinstance` is the canonical choice.
- The existing test that passes `_fake_response(200, [])` for trades MUST be updated to `_fake_response(200, {"trades": [], "count": 0})` to catch future regressions. Failing to update the test defeats the fix.

### Application to pyfinagent (mapping external findings to file:line anchors)

- **Root cause:** `paper_trading.py:226` returns a dict; `scheduler.py:251` passes it raw to `format_evening_digest`; `formatters.py:376` slices it -- crash.
- **Option A fix location:** `formatters.py:374` -- before the `if trades_today:` guard, add `if isinstance(trades_today, dict): trades_today = trades_today.get("trades", [])`. This makes the formatter robust regardless of caller.
- **Option B fix location:** `scheduler.py:251` -- change `trades_data = trades_res.json() if ...` to extract the list: `raw = trades_res.json() if ... else []; trades_data = raw.get("trades", []) if isinstance(raw, dict) else raw`. This keeps the formatter strictly typed.
- **Test update required:** `test_digest_url_semantics.py:110` -- change `_fake_response(200, [])` to `_fake_response(200, {"trades": [], "count": 0})` in the evening digest tests.
- `format_morning_digest` is not affected; no change needed there.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched: Real Python KeyError, Slack section-block reference, Slack blocks overview, BetterStack pattern matching, hectoraleman Medium API envelope)
- [x] 10+ unique URLs total (incl. snippet-only) -- 13 unique URLs collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (formatters, scheduler, paper_trading API, reports API, test file)
- [x] Contradictions / consensus noted (mixed API convention is root cause; Option A vs B tradeoff documented)
- [x] All claims cited per-claim

---

### Three answers required by caller

**Answer 1: What shape does `/api/paper-trading/trades?limit=10` actually return?**

Dict envelope: `{"trades": [...], "count": N}`. Source: `backend/api/paper_trading.py:226` -- `result = {"trades": trades, "count": len(trades)}`. Confirmed by the `KeyError: slice(None, 10, None)` traceback in `handoff/logs/slack_bot.log:341`.

**Answer 2: Is `format_morning_digest` (reports_data) susceptible to the same bug?**

No. `GET /api/reports/?limit=5` has `response_model=list[ReportSummary]` (`backend/api/reports.py:28`) and returns a bare Python list. `format_morning_digest` at `formatters.py:333` does `for r in recent_reports[:5]:` which works correctly on a list. No fix needed for the morning digest.

**Answer 3: Recommended fix shape**

**Option B** -- defensive coerce in `_send_evening_digest` (`scheduler.py`) before calling `format_evening_digest`. This keeps formatters strictly typed (their function signatures say `list`; they should receive a list) and isolates the envelope-unwrapping logic at the HTTP boundary where it belongs.

Concretely, change `scheduler.py:251`:
```python
# Before
trades_data = trades_res.json() if trades_res.status_code == 200 else []

# After
_raw = trades_res.json() if trades_res.status_code == 200 else []
trades_data = _raw.get("trades", []) if isinstance(_raw, dict) else _raw
```

Additionally update `test_digest_url_semantics.py:110` to pass `_fake_response(200, {"trades": [], "count": 0})` so the test exercises the real API shape.

Option A (coerce inside `format_evening_digest`) is also valid and arguably more defensive against future callers passing wrong shapes, but it hides the envelope contract from the formatter's type signature. Option C (standardize the API to return flat list) is out of scope.

Rationale: API envelope pattern literature (hectoraleman Medium) recommends dict envelopes for collections; the fix should normalize at the caller boundary, not inside the formatter. Python isinstance guard is the idiomatic one-liner. (BetterStack pattern matching guide; Real Python KeyError doc.)
