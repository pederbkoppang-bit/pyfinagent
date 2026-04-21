# Q/A Critique -- phase-6.3 Streaming adapters (Finnhub, Benzinga, Alpaca)

**Verdict: PASS**

## Harness-protocol audit (5/5)
1. researcher_63 spawn present: `phase-6.3-research-brief.md` exists (mtime 08:41), contract cites `researcher_63 (tier=moderate) gate_passed=true`.
2. Contract PRE-commit: contract mtime 08:42 < experiment-results 08:44. OK.
3. `experiment-results.md` present with verbatim verification.
4. Masterplan `phase-6.3.status = pending`; no log entry yet -- log-last order correct.
5. Fresh Q/A (qa_63) -- this run.

## Deterministic checks (7/7)
- 4 new files exist (`__init__`, `finnhub`, `benzinga`, `alpaca`).
- Registry: `sorted(get_sources().keys()) == ['alpaca','benzinga','finnhub','stub']`.
- Protocol conformance: all three expose `.name` + callable `.fetch()`.
- Settings: 4 new fields present in `backend/config/settings.py:61-64`, all default `""`.
- `run_once(['finnhub','benzinga','alpaca'], dry_run=True)` -> counts all 0, `errors=[]`, no raise.
- Syntax OK on all 6 touched files.
- phase-6.2 regression: `python -m backend.news.fetcher` exits 0 with `phase-6.2 smoke: OK` and stub count=3.

## Code-review spot checks
- finnhub: `httpx.Client` sync, `token=<key>` query param, `category=general`, Unix int -> ISO via `datetime.fromtimestamp(..., tz=utc)`. OK.
- benzinga: `Authorization: token <key>` header, `stocks[0].ticker` unwrap with dict/str fallback, channels -> categories. OK.
- alpaca: `Apca-Api-Key-Id` + `Apca-Api-Secret-Key` headers, `{"news": [...]}` envelope unwrapped, `symbols[0]` as ticker. OK.

## LLM judgment
Real adapters (not stubs): httpx wired, field maps match documented provider shapes per research brief, graceful degrade via early return on empty keys / non-200 / exception. Scope honesty: experiment-results correctly flags "no live API calls this cycle" -- acceptable since keys empty in dev. Benzinga file content matches Benzinga adapter (no leftover typo).

**checks_run:** harness_audit, syntax, registry, protocol, settings, graceful_degrade, regression, code_review.
