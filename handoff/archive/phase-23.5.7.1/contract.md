---
step: phase-23.5.7.1
title: Fix format_evening_digest KeyError on dict-shaped trades_today
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 tests/verify_phase_23_5_7_1.py'
research_brief: handoff/current/phase-23.5.7.1-research-brief.md
---

# Contract — phase-23.5.7.1

## Hypothesis

`/api/paper-trading/trades?limit=10` returns a dict envelope
`{"trades": [...], "count": N}` (confirmed at
`backend/api/paper_trading.py:226`:
`result = {"trades": trades, "count": len(trades)}`). Post-23.5.3.1
the httpx call lands and `_send_evening_digest` (`scheduler.py:251`)
assigns the dict to `trades_data`, then passes it to
`format_evening_digest`. Inside the formatter at
`formatters.py:376`, the code does `trades_today[:10]` — Python
treats `dict[:10]` as `dict.__getitem__(slice(None,10,None))`,
which raises `KeyError: slice(None, 10, None)`.

The minimal-blast-radius fix is **Option B** per researcher:
defensive coerce at the HTTP boundary in `_send_evening_digest`,
NOT inside the formatter. Keeps the formatter strictly typed
(`trades_today: list`) and unwraps the envelope at the call site.

The exact edit at `scheduler.py:251`:
```python
# Before
trades_data = trades_res.json() if trades_res.status_code == 200 else []

# After
_raw = trades_res.json() if trades_res.status_code == 200 else []
trades_data = _raw.get("trades", []) if isinstance(_raw, dict) else _raw
```

Additionally, update the URL-semantics test for evening_digest
(`tests/slack_bot/test_digest_url_semantics.py:110`) to use the
actual API response shape `{"trades": [], "count": 0}` instead
of a bare empty list — so future regressions in the envelope
unwrapping are caught.

## Research-gate summary

`researcher` agent `a73a8f62656a7419b` ran tier=simple and
returned `gate_passed: true` with:
- 5 external sources fetched in full (≥5 floor): Real Python
  KeyError reference, Slack Block Kit section + blocks overview,
  API Response Wrapper Patterns, BetterStack pattern-matching.
- 8 snippet-only + 5 read-in-full = 13 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 6 internal files inspected.

Brief: `handoff/current/phase-23.5.7.1-research-brief.md`.

**Three explicit answers:**
1. **API shape:** dict envelope `{"trades": [...], "count": N}`
   (`backend/api/paper_trading.py:226`).
2. **`format_morning_digest` is NOT susceptible** — `GET /api/reports/`
   has `response_model=list[ReportSummary]` (`reports.py:28`) and
   returns a bare list. No fix needed for the morning path.
3. **Option B (defensive coerce in caller, not formatter)** —
   normalize at the HTTP boundary; keeps the formatter strictly
   typed; minimal blast radius.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied from `.claude/masterplan.json::23.5.7.1.verification`:

```
python3 tests/verify_phase_23_5_7_1.py
```

The verifier exits 0 only when:
1. `_send_evening_digest` body in `scheduler.py` contains the
   defensive `isinstance(_raw, dict)` coercion.
2. `format_evening_digest` body is unchanged (no fix in the
   formatter — the fix lives upstream).
3. New unit tests `tests/slack_bot/test_evening_digest_envelope_coerce.py`
   pass:
   - dict envelope with trades → list passed to formatter
   - dict envelope with empty trades → empty list to formatter
   - bare list (legacy fallback) → unchanged passthrough
   - status != 200 → empty list (existing behavior preserved)
4. Existing tests in `tests/slack_bot/test_digest_url_semantics.py`
   still pass (the test for evening_digest is updated to use the
   real envelope shape; morning_digest test unchanged).

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Edit `backend/slack_bot/scheduler.py:_send_evening_digest`
      at the line currently reading
      `trades_data = trades_res.json() if trades_res.status_code == 200 else []`.
      Insert the 2-line defensive coerce per the contract above.
   b. Update `tests/slack_bot/test_digest_url_semantics.py` —
      change the evening_digest test's `trades_resp` fake to use
      `{"trades": [], "count": 0}` instead of `[]`, so the test
      exercises the new envelope-unwrap path AND catches future
      regressions.
   c. Add `tests/slack_bot/test_evening_digest_envelope_coerce.py`
      with 4 explicit tests covering:
      - dict envelope (typical) → unwraps to list
      - dict envelope empty trades → unwraps to []
      - bare list passthrough (legacy)
      - status != 200 → empty list
   d. Add `tests/verify_phase_23_5_7_1.py` — 4-check verifier:
      - source-grep: `_send_evening_digest` contains `isinstance(_raw, dict)`.
      - source-grep: `format_evening_digest` body unchanged
        (`trades_today[:10]` slice still present, formatter still
        assumes list-shaped).
      - new unit tests pass.
      - existing `test_digest_url_semantics.py` tests still pass.
   e. Restart slack-bot daemon to deploy the fix before the next
      evening-digest fire.
   f. Re-run sibling verifiers (23.5.1 ... 23.5.7).
   g. Write `experiment_results.md`.
4. **EVALUATE phase:** spawn fresh `qa` agent with explicit Write-
   step skeleton. 5-item harness audit FIRST, deterministic re-
   verification, LLM judgment.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip 23.5.7.1
   status only after the log append.

## Anti-patterns guarded (≥3)

1. **Fixing inside `format_evening_digest`** — would couple the
   formatter to envelope semantics; researcher recommends Option
   B (boundary coercion) on the call-site rule.
2. **Refactoring the trades API endpoint** to return a flat list —
   Option C, out of scope. Several other call sites likely depend
   on the count.
3. **Loosening the formatter's type expectations** — formatter
   signature declares `trades_today: list`; preserve that contract.
4. **Adding pattern-matching (Python 3.10+ `match/case`)** — when
   `isinstance` is the simpler idiomatic one-liner per BetterStack
   and Real Python.
5. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Refactoring `/api/paper-trading/trades` endpoint shape.
- Touching `format_morning_digest` (researcher confirmed it is
  not susceptible — reports endpoint returns a bare list).
- The 6 sibling phase-9 jobs.
- The 6 launchd jobs.

## Backwards compatibility

- The defensive coerce preserves bare-list inputs unchanged
  (legacy fallback).
- `format_evening_digest` unchanged — its existing tests still
  pass.
- The URL-semantics test's evening_digest case is updated to the
  realistic envelope shape; morning case unchanged.

## Risk

- **Other callers of `/api/paper-trading/trades`** may have
  silently relied on the dict envelope. Coverage check: the
  researcher's grep didn't find any other callers from the slack-
  bot side; only `_send_evening_digest`. Out-of-scope to audit
  the frontend / other backend callers in this step.
- **`response_model` for the trades endpoint** — out of scope to
  pin a Pydantic model in this step; could be a follow-up
  hardening task.

## References

- Research brief:
  `handoff/current/phase-23.5.7.1-research-brief.md` (researcher
  `a73a8f62656a7419b`, 5 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.7.1.verification`.
- Files to edit:
  - `backend/slack_bot/scheduler.py` (line ~251 — `_send_evening_digest`).
  - `tests/slack_bot/test_digest_url_semantics.py` (evening case
    update).
- New files:
  - `tests/slack_bot/test_evening_digest_envelope_coerce.py` (4 tests).
  - `tests/verify_phase_23_5_7_1.py` (4-check verifier).
- API source (envelope confirmed):
  `backend/api/paper_trading.py:226`.
- Real Python KeyError reference:
  https://realpython.com/ref/builtin-exceptions/keyerror/
- Slack Block Kit section block:
  https://docs.slack.dev/reference/block-kit/blocks/section-block
- API Response Wrapper Patterns (hector-reyesaleman.medium.com):
  https://hector-reyesaleman.medium.com/api-response-wrapper-patterns-49d846578ac0
