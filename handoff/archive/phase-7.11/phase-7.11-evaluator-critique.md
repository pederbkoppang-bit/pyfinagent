# Q/A Evaluator Critique -- phase-7 / 7.11 (Shared scraper infrastructure)

**Verdict:** PASS
**Reviewer id:** qa_711_v1
**Cycle:** 1
**Date:** 2026-04-20

## 5-item harness-protocol audit

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawned before contract; `gate_passed: true` | PASS | `phase-7.11-research-brief.md` mtime 00:36:44 < contract 00:37:57. 8 sources read in full, 18 URLs, three-variant queries (`2026` frontier, `2025/2024` last-2y, year-less canonical), recency scan present, internal file:line anchors for all 8 ingesters. JSON envelope `external_sources_read_in_full=8, gate_passed=true`. |
| 2 | Contract pre-dates experiment results | PASS | contract 00:37:57 < experiment-results 00:39:48. Immutable criteria copied verbatim to contract. |
| 3 | `experiment_results.md` contains verbatim verification output | PASS | ast.parse `SYNTAX OK`, bq ls row shown, `GREP EXIT=0`, pytest `152 passed, 1 skipped`. |
| 4 | Log-last discipline: `harness_log.md` last block is prior step (7.10), NOT yet 7.11 | PASS | `tail -60 harness_log.md` -> last block is `Cycle -- 2026-04-20 00:35 UTC -- phase=7.10 result=PASS`. 7.11 not yet appended (correct pre-Q/A state). |
| 5 | First Q/A on 7.11 (no prior critique to second-opinion-shop) | PASS | No `phase-7.11-evaluator-critique*.md` existed prior to this file. |

## Deterministic checks A-H

| # | Check | Command | Result |
|---|-------|---------|--------|
| A | AST parse | `python -c "import ast; ast.parse(open('backend/alt_data/http.py').read())"` | SYNTAX OK |
| B | Audit table live | `bq ls --project_id=sunny-might-477607-p8 pyfinagent_data \| grep scraper_audit_log` | exit 0; row shows `DAY (field: ts) \| source, status_code` |
| C | Schema conformance | `bq show --schema sunny-might-477607-p8:pyfinagent_data.scraper_audit_log` | 11 columns in declared order: request_id (REQUIRED STRING), source (REQUIRED STRING), url (REQUIRED STRING), method STRING, status_code INTEGER, latency_ms FLOAT, user_agent STRING, ip_hash STRING, ts (REQUIRED TIMESTAMP), bytes_returned INTEGER, error STRING. PARTITION BY DATE(ts); CLUSTER BY source, status_code. Matches compliance doc Sec. 6.1 exactly. |
| D | Regression suite | `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | 152 passed, 1 skipped (unchanged from 7.10 baseline). |
| E | File existence + ASCII decode | `ls backend/alt_data/http.py`; `open(...).read().decode('ascii')` | exists (14184 bytes), ASCII OK. |
| F | Scope honesty: only new file + handoff artifacts | `ls backend/alt_data/` | http.py is the sole new module; mtimes on congress.py/f13.py/finra_short.py/etf_flows.py/twitter.py/reddit_wsb.py/google_trends.py/hiring.py are from earlier 7.x cycles (22:28-00:38). Contract out-of-scope respected: no existing ingester refactored. |
| G | Bounded backoff math | Read `_jittered_backoff` (L200-203) | `cap = base_s * max_multiplier`; `exp = base_s * (2**attempt)`; `return min(cap, exp) + random.uniform(0.0, 1.0)`. Hard cap `base * 8` at `max_multiplier=8`. At SEC base=60s: worst-case single sleep 480s + [0,1)s jitter. At FINRA base=5s: worst-case 40s + jitter. Unbounded `60 * 2^attempt` from f13.py/finra_short.py is eliminated. **adv_73_cdn_403 resolved.** |
| H | FINRA preset correctness | Read `SOURCE_PRESETS["finra.cdn"]` (L78-81) | `RateLimit(per_second_cap=0.5, backoff_403_base_s=5.0)`; confirmed the 5-second base required to close adv_73. |

## LLM judgment

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Circuit breaker: sliding window + fail-rate + 4xx excluded | PASS | `_cb_window: collections.deque[bool] = collections.deque(maxlen=circuit_window)` (L236, default 20). `_record_cb` (L251-260) appends; when `len == maxlen` AND `fail_rate > threshold` (default 0.5) sets `_cb_open_until = now + 60s`. 403 path (L350-358) explicitly does NOT call `_record_cb(False)` (comment: "Do NOT record as circuit failure: 403 is a rate-limit signal"). Non-200 non-retry path (L370-371) gates CB recording on `< 400 or >= 500`, so 4xx don't poison the window. Matches oneuptime 2026 pattern cited in brief. |
| 2 | Correlation ID per `.get()` call | PASS | `request_id = uuid.uuid4().hex[:16]` (L306) generated once per call, passed to every `_audit_row` call (success, 403-exhausted, exception-exhausted). Maps to the MS Engineering Playbook rule ("assign as early as possible at request boundary"). |
| 3 | Full jitter on every backoff sleep | PASS | `_jittered_backoff` is the only backoff path used for both 403 (L350-355) and 5xx (L359-367). `random.uniform(0.0, 1.0)` is additive, matches AWS `full_jitter` canonical formula cited in the brief. |
| 4 | `_audit_write` is streaming insert only (adv_71 resolved) | PASS | L178 docstring: *"Append ONE row to scraper_audit_log. Streaming insert only; never MERGE."* Implementation uses `client.insert_rows_json(table_ref, [row])` (L186). No MERGE / UPDATE path anywhere in the file. |
| 5 | `requests` not `httpx` (per brief decision) | PASS | Grep `httpx` on http.py -> no hits. Only `import requests` lazily at L299 inside `.get()`. Matches brief finding #7: "Keep `requests` (not httpx) for phase-7.11 -- all 8 ingesters already import requests; httpx is the right long-term upgrade but switching transports in this phase adds scope without benefit." |
| 6 | No test file required (out-of-scope per contract) | PASS | Contract Out-of-Scope item: *"No tests -- the smoketest pattern for a synchronous HTTP helper belongs in phase-7.12 or a future cleanup."* Scope honesty preserved; regression suite still green. |
| 7 | Fail-open discipline | PASS | `ensure_audit_table` (L163-175), `_audit_write` (L178-193), `_get_bq_client` (L150-160) all catch broad `Exception`, log WARNING with `%r`, return `False` / `None`. ASCII logger messages (`--`, `->`) per security.md. |
| 8 | ASCII-only & Python 3.14 typing | PASS | `open('backend/alt_data/http.py','rb').read().decode('ascii')` succeeds. `collections.deque[bool]` annotation is valid at runtime on 3.14 (noted caveat in experiment-results sec "Known caveats"). |

## Advisories resolved

- **adv_73_cdn_403** -- bounded backoff via `min(base * 2**attempt, base * backoff_max_multiplier)` with `max_mult=8`; FINRA preset `backoff_403_base_s=5.0`. Worst-case 403 sleep on FINRA is 40s + jitter (was unbounded `60 * 2^attempt`). Verified in `_jittered_backoff` and `SOURCE_PRESETS["finra.cdn"]`.
- **adv_71_docstring_merge** -- `_audit_write` docstring explicit: streaming insert only; implementation uses `insert_rows_json`; no MERGE path exists. Audit-log is append-only.

## Scope / honesty

Experiment-results "Known caveats" discloses five real limitations up-front:
existing 8 ingesters NOT migrated (explicitly out-of-scope per contract),
no tests, ip_hash disabled (future work), fixed 60s CB cooldown, 3.14
typing-annotation caveat. No overclaim.

## Mutation-resistance sanity

The immutable criteria are shallow (ast.parse + bq ls grep). A deeper
mutation test is not required at this tier since the regression suite
exercises every call site importing backend.alt_data, and the audit
schema is verified by live `bq show` against the compliance doc.
No ingester currently imports `http.py` (by contract), so a full
integration mutation test belongs in phase-7.12 when the first
ingester is migrated.

## Immutable criteria

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `python -c "import ast; ast.parse(open('backend/alt_data/http.py').read())"` | PASS |
| 2 | `bq ls pyfinagent_data \| grep -q scraper_audit_log` | PASS (exit 0) |

## Decision

**PASS** (qa_711_v1). All 5/5 protocol-audit items, 8/8 deterministic
checks A-H, 8/8 LLM-judgment criteria, 2/2 immutable criteria satisfied.
Both advisories (adv_73_cdn_403, adv_71_docstring_merge) resolved with
verified code + schema evidence. Regression green (152 passed, 1
skipped). Orchestrator may proceed: append the 7.11 block to
`handoff/harness_log.md`, then flip `phase-7 / 7.11` pending -> done.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 2 immutable criteria met; 5/5 protocol audit, 8/8 deterministic A-H, 8/8 LLM judgment. Bounded backoff verified (min(base*2^attempt, base*8)+jitter); FINRA preset backoff_403_base_s=5.0; circuit breaker deque(maxlen=20) with 4xx excluded; streaming-insert-only audit_write; requests not httpx; audit table schema matches compliance Sec 6.1 (11 cols, DATE(ts) partition, source+status_code cluster). Regression 152 passed / 1 skipped.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "protocol_audit_5item",
    "syntax_ast_parse",
    "bq_ls_grep",
    "bq_show_schema_11cols",
    "pytest_regression",
    "ascii_decode",
    "scope_no_ingester_modified",
    "bounded_backoff_math",
    "finra_preset_5s",
    "circuit_breaker_deque20_fail_rate_gt_half",
    "4xx_excluded_from_cb",
    "correlation_id_uuid4_hex16_per_get",
    "full_jitter_uniform_0_1",
    "audit_write_streaming_insert_only",
    "no_httpx_import",
    "fail_open_everywhere",
    "adv_73_resolved",
    "adv_71_resolved"
  ],
  "reviewer_id": "qa_711_v1"
}
```
