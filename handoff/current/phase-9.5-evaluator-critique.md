# Q/A Critique — phase-9.5 (hourly signal warmup) — REMEDIATION v1

**Verdict: PASS**  **qa_id:** qa_95_remediation_v1  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_95_remediation_v1",
  "violated_criteria": [],
  "checks_run": ["protocol_audit_5item", "ast_parse", "pytest_3of3", "file_existence", "mtime_ordering", "verbatim_quote", "spot_read_line21_line31", "mutation_resistance_line22", "brief_source_authenticity", "carry_forward_defensibility", "contract_code_alignment"],
  "reason": "All deterministic checks pass (ast OK, pytest 3/3, handoff files present, contract mtime 18:30 <= results mtime 18:31). Research gate: 5 sources in full, 11 URLs, three-variant queries, recency scan, gate_passed=true. Mutation-resistance: changing line 22 fallback to always-{} would fail test_cache_backend_is_injectable. Carry-forwards defensibly deferred. Cycle v1 on fresh evidence."
}
```

## Protocol audit (5/5 PASS)

1. Researcher brief: 5 sources in full, 11 URLs, three-variant queries, recency scan, gate_passed=true.
2. Contract mtime ≤ results mtime.
3. Verbatim verification in results.
4. Log-last: no 9.5 log append yet.
5. Cycle v1 on fresh MAS evidence.

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse | exit 0 |
| pytest -q | 3 passed, exit 0 |
| Handoff files | all 3 present |
| Line 21 `IdempotencyKey.hourly(...)` + line 31 `cache[ticker] = fn(ticker)` | matches brief pattern |
| Mutation: line 22 → always `{}` | `test_cache_backend_is_injectable` breaks (KeyError on `my_cache["X"]`) |

## LLM judgment

5 real sources — oneuptime (cache warming + invalidation 2026), Gupta Medium (dict vs Redis Apr-2026), pandas_market_calendars PyPI, hellointerview. Three-variant query discipline visible. Recency scan reports Gupta as new finding.

Carry-forwards (TTL, market-hours gating, Redis upgrade path, watchlist prioritization) defensibly deferred with single-process APScheduler rationale. The `compute_signal_fn` no-op fallback at line 29 is flagged for phase-9.6 or follow-up hardening.

DI surface on all four seams (`watchlist`, `compute_signal_fn`, `cache_backend`, `store`, `iso_hour`) matches contract. Code-contract alignment tight.

Cleared for log append + masterplan status confirmation.
