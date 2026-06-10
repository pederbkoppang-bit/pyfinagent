---
step: phase-10.8.1
title: Wire log_slot_usage into thursday_batch / friday_promotion / monthly_champion_challenger / rollback
tier: simple
researcher: researcher (merged Explore role)
date: 2026-04-21
---

## Research: phase-10.8.1 -- log_slot_usage wiring

### 1. Executive summary

`log_slot_usage` in `backend/autoresearch/slot_accounting.py` is a complete, fail-open BQ sink
with a `bq_insert_fn` dependency-injection parameter already in its signature. The four target
routines (`trigger_thursday_batch`, `run_friday_promotion`, `run_monthly_sortino_gate`,
`auto_demote_on_dd_breach`) each expose branching paths (already_fired, success, ledger_write_failed,
no_breach, etc.) that need a single post-state-write call. The correct strategy is to insert one
`log_slot_usage(...)` call per exit path (not per branch) after any ledger write has completed, using
the `bq_insert_fn` DI parameter for test stubs -- matching the idiom already established in
`tests/autoresearch/test_slot_accounting.py`. The new test module
`tests/autoresearch/test_slot_usage_wiring.py` follows the same DI-capture pattern.

---

### 2. External sources

#### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://opentelemetry.io/docs/concepts/observability-primer/ | 2026-04-21 | Official doc | WebFetch | Three-pillar model (logs/metrics/traces); logs are immutable timestamped records -- confirms post-state-write logging as the canonical "event record" pattern |
| https://pytest-with-eric.com/mocking/mocking-vs-patching/ | 2026-04-21 | Authoritative blog | WebFetch | Distinguishes mock (fake object) vs patch (temporary replacement); confirms DI is preferred for distributed/shared infrastructure like telemetry; warns against overuse of patch creating overly-complex tests |
| https://rednafi.com/python/patch-with-pytest-fixture/ | 2026-04-21 | Authoritative blog | WebFetch | Recommends pytest fixtures wrapping patch() for DRY; demonstrates `patch.object` for module-level callables; confirms DI (injected callable) is simpler when the function already accepts a callable param |
| https://docs.python.org/3/library/unittest.mock.html | 2026-04-21 | Official doc (Python 3) | WebFetch | "Patch where it is looked up, not defined"; documents `side_effect`, `assert_called_once_with`, `call_args_list` for verifying telemetry calls; DI removes the lookup-location problem entirely |
| https://betterstack.com/community/guides/observability/opentelemetry-best-practices/ | 2026-04-21 | Authoritative blog | WebFetch | "Implement circuit breakers in your telemetry pipeline to prevent telemetry collection from impacting application availability"; confirms fail-open / non-blocking telemetry is a hard requirement |
| https://opentelemetry.io/blog/2025/ai-agent-observability/ | 2026-04-21 | Official doc (OTel 2025) | WebFetch | Establishes semantic conventions for AI agent observability; "baked-in" vs "external" instrumentation trade-off -- baked-in (DI callable) wins for library code that must remain portable |
| https://oneuptime.com/blog/post/2026-02-06-monitor-kubernetes-cronjobs-jobs-opentelemetry/view | 2026-04-21 | Authoritative blog | WebFetch | "The number one issue with Job telemetry is data that never makes it out" -- confirms synchronous flush/call after state write is correct; fire-and-forget is a risk for batch jobs |
| https://arxiv.org/html/2510.02991v1 | 2026-04-21 | Peer-reviewed preprint | WebFetch | Cloud-native observability pattern catalog (11 patterns); Audit Logging pattern maps directly to slot_accounting's per-event BQ row; confirms append-only, timestamped, structured rows are the established approach |

#### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://juliofalbo.medium.com/effective-logging-strategies-for-better-observability-and-debugging-4b90decefdf1 | Blog | Fetched but no actionable content on no-op/fail-open specifics |
| https://betterprogramming.pub/testing-in-python-dependency-injection-vs-mocking-5e542783cb20 | Blog | Redirect to authenticated Medium wall |
| http://mauveweb.co.uk/posts/2014/09/every-mock-patch-is-a-little-smell.html | Blog | Certificate expired; accessible via search snippet only |
| https://medium.com/@bhagyarana80/mock-anything-in-python-pytest-unittest-mock-deep-dive-for-real-world-testing-d4ed26f65649 | Blog | Search snippet only |
| https://neuralception.com/python-better-unittest-pytest/ | Blog | Fetched; no DI vs patch comparison for telemetry specifically |
| https://www.dash0.com/guides/logging-best-practices | Guide | Fetched; no fail-open or idempotent-path specifics |
| https://pytest-with-eric.com/mocking/pytest-monkeypatch/ | Blog | Fetched; no DI-vs-monkeypatch comparison for telemetry |
| https://calmops.com/devops/opentelemetry-observability-2026-complete-guide/ | Blog | Search snippet only; general OTel survey |
| https://rootly.com/sre/top-10-observability-tools-2026-boost-reliability | Blog | Search snippet only; tool survey not patterns |
| https://dev.to/cronmonitor/how-to-monitor-cron-jobs-in-2026-a-complete-guide-28g9 | Blog | Search snippet only |

---

### 3. Recency scan (2024-2026)

Queries run:
1. `observability logging patterns cron routines post-state-write telemetry 2026` (frontier)
2. `Python unittest.mock.patch vs dependency injection logging telemetry pytest 2025` (last-2y)
3. `idempotent function telemetry observability no-op path logging best practices` (year-less canonical)

Result: The 2025-2026 window produced two relevant findings:
- OpenTelemetry's 2025 AI agent observability post establishes that "baked-in" instrumentation via
  injectable callables is the preferred pattern for library code -- directly supporting the DI approach
  already used in `slot_accounting.py`.
- The 2026 cron/Kubernetes OTel article's "force_flush before exit" guidance maps to: log_slot_usage
  must be called synchronously on the same call stack as the ledger write, not deferred, to avoid
  data loss. Both confirm the existing fail-open design is correct.

No finding in the 2024-2026 window contradicts or supersedes the canonical approach.

---

### 4. Internal code audit

#### Files inspected

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/slot_accounting.py` | 150 | BQ telemetry sink | Complete; DI-ready |
| `backend/autoresearch/thursday_batch.py` | 149 | Phase-10.3 Thursday routine | No log_slot_usage call yet |
| `backend/autoresearch/friday_promotion.py` | 171 | Phase-10.4 Friday routine | No log_slot_usage call yet |
| `backend/autoresearch/monthly_champion_challenger.py` | 311 | Phase-10.6 monthly gate | No log_slot_usage call yet |
| `backend/autoresearch/rollback.py` | 171 | Phase-10.7 rollback kill-switch | No log_slot_usage call yet |
| `tests/autoresearch/test_slot_accounting.py` | 196 | Existing slot_accounting tests | DI capture pattern established |
| `tests/autoresearch/test_thursday_batch.py` | 76 | Thursday tests | tmp_path + no mock pattern |
| `tests/autoresearch/test_friday_promotion.py` | 40+ | Friday tests | tmp_path + seed helper |
| `tests/autoresearch/test_rollback.py` | 60+ | Rollback tests | tmp_path + datetime injection |

#### `log_slot_usage` signature (slot_accounting.py:30-40)

```python
def log_slot_usage(
    *,
    week_iso: str,
    slot_id: str,
    routine: str,
    result: dict[str, Any],
    phase: str = "phase-10",
    bq_insert_fn: Callable[[str, list[dict[str, Any]]], bool] | None = None,
    table: str = _DEFAULT_TABLE,
    now: datetime | None = None,
) -> dict[str, Any]:
```

All parameters are keyword-only. `bq_insert_fn=None` defaults to the real BQ client -- passing a
capture closure in tests is the established pattern (see `test_slot_accounting.py:24-30`).

#### Branch analysis per routine

**`trigger_thursday_batch` (thursday_batch.py:32-91)**

Branch map:
1. `already_fired=True` path -- line 56-69: returns early after detecting existing ledger row with
   `thu_batch_id`. No ledger write occurs. Question: log or not?
2. Fresh-fired path -- line 75-91: `weekly_ledger.append_row(...)` at line 75; returns result dict
   with `batch_id`, `candidates_kicked`, `already_fired=False`.

Ledger write can fail (`ok=False` at line 83) but the routine still returns the result dict --
fail-open behavior already in place. The `log_slot_usage` call belongs after the append attempt
(post-state-write, regardless of `ok`).

**`run_friday_promotion` (friday_promotion.py:30-148)**

Branch map:
1. Fail-closed path -- line 62-73: `no_thursday_batch_on_ledger` -- no ledger read/write attempted.
   Returns `error="no_thursday_batch_on_ledger"`.
2. Already-fired path -- line 76-91: detects `fri_promoted_ids` already set. Returns `already_fired=True`.
3. Success path -- line 121-148: `weekly_ledger.append_row(...)` at line 121; on `ok=False` returns
   `error="ledger_write_failed"` at line 135; on `ok=True` returns the full result.

`log_slot_usage` belongs: after every exit point except possibly the fail-closed no-thursday path
(callers should still know a no-op fired). The masterplan criteria says "on the success path" but
auditing all non-error paths is safer for observability.

**`run_monthly_sortino_gate` (monthly_champion_challenger.py:43-198)**

Branch map (entry point only; `fired=False` if not-last-friday returns early):
1. `not_last_trading_friday` -- line 84-87: returns without doing any work. `fired=False`.
2. Prior pending not expired -- line 107-112: returns `approval_pending=True` without re-evaluating.
3. Prior approved -- line 113-117: returns `approved=True`.
4. Short challenger returns -- line 120-122: returns early, no state write.
5. Sortino/PBO/DD gate failure -- lines 145-165: each returns early without state write.
6. Gate pass -- line 168-197: writes state to JSON + fires optional slack_fn. `gate_pass=True`,
   `approval_pending=True`.

`week_iso` is NOT a direct parameter; must be derived from `eval_date`:
```python
import datetime as _dt
year, weeknum, _ = eval_date.isocalendar()
week_iso = f"{year}-W{weeknum:02d}"
```

`log_slot_usage(slot_id='monthly_gate', ...)` fires on the `gate_pass=True` branch and on the
`not_last_trading_friday` path is safely skipped (fired=False, no slot consumed). The masterplan
criteria does not restrict to gate-pass-only, so firing on all "slot was considered" paths is valid.

**`auto_demote_on_dd_breach` (rollback.py:32-116)**

Branch map:
1. `no_breach` -- line 64-65: `abs(dd) <= threshold`, returns early with `demoted=False`.
2. `already_demoted` -- line 71-74: state shows prior demotion, returns early.
3. `auto_demoted` -- lines 76-115: writes JSONL audit + state JSON + optional ledger notes. Returns
   `demoted=True`.

`week_iso` is an optional param (line 39: `week_iso: str | None = None`). When provided, the caller
passes it through. The masterplan criteria says "week_iso set correctly" -- so the wiring call must
use the `week_iso` argument from the function signature directly.

`log_slot_usage(slot_id='rollback', ...)` fires on both the `auto_demoted` path and the `no_breach`/
`already_demoted` paths if observability coverage of no-ops is desired. Minimum: fire on `auto_demoted`.

---

### 5. Design recommendation

#### Q1: Post-state-write vs always-fire on idempotent paths

**Recommendation: always-fire (fire on every exit path, including already_fired and no_breach).**

Rationale: the slot_accounting table is an audit log, not a counter. Logging a no-op (already_fired,
no_breach) is valuable -- it tells the operator "the routine ran and correctly determined no work was
needed." This matches the Audit Logging pattern from the cloud-native pattern catalog (arxiv 2510.02991)
and avoids silent-skip confusion. The fail-open design in `log_slot_usage` means the logging call will
not throw even if BQ is down.

Exception: `run_monthly_sortino_gate` when `fired=False` (not-last-friday) -- this is genuinely "never
ran", not "ran and was a no-op". Skip on `not_last_trading_friday`.

#### Q2: DI vs monkeypatch for tests

**Recommendation: use the existing `bq_insert_fn` DI parameter. Do NOT use `unittest.mock.patch`.**

The `log_slot_usage` function already exposes `bq_insert_fn` for test injection (same design as BQ
client injection throughout the codebase). The existing `test_slot_accounting.py` uses a
`_capture_factory()` closure returning `(store, capture)`. Use the same pattern.

For wiring tests, the routines themselves do not yet accept a `log_fn` param -- two choices:
1. Add `log_fn: Callable | None = None` to each routine and pass through to `log_slot_usage`.
2. Use `monkeypatch.setattr('backend.autoresearch.thursday_batch.log_slot_usage', capture)` in tests.

**Recommend option 1 (add `log_fn` param to each routine).** This matches the existing codebase idiom
(`bq_insert_fn`, `slack_fn`, `bq_query_fn`, `now` are all injected). It avoids the "where to patch"
namespace problem documented in the Python unittest.mock docs. It keeps tests fast and namespace-safe.

The signature addition per routine is minimal:
```python
log_fn: Callable[..., Any] | None = None,
```
and the call:
```python
from backend.autoresearch.slot_accounting import log_slot_usage as _log_slot_usage
_log_fn = log_fn or _log_slot_usage
_log_fn(week_iso=week_iso, slot_id='thu_batch', ...)
```

#### Q3: No-op path telemetry

**Yes, fire for already_fired and already_demoted.** Do not fire for not_last_trading_friday (slot never
considered). The `result` dict for no-op paths should include `already_fired=True` / `decision=already_demoted`
so the BQ row's `status` field (derived from `result.get("status")`) is meaningful.

#### Exact `log_slot_usage(...)` call per routine

**trigger_thursday_batch** (after line 90, before `return`):
```python
# Both branches (already_fired and freshly_fired) call log_fn
log_fn(
    week_iso=week_iso,
    slot_id="thu_batch",
    phase="phase-10",
    routine="trigger_thursday_batch",
    result={
        "batch_id": <batch_id from result dict>,
        "candidates_kicked": <candidates_kicked from result dict>,
        "already_fired": <bool>,
        "status": "already_fired" if already_fired else "ok",
    },
)
```

Since the routine has two return points (lines 62-69 for already_fired and lines 84-91 for fresh),
the cleanest implementation is to compute the result dict first, call log_fn, then return.

**run_friday_promotion** (success + already_fired paths; optionally fail-closed too):
```python
log_fn(
    week_iso=week_iso,
    slot_id="fri_promotion",
    phase="phase-10",
    routine="run_friday_promotion",
    result={
        "promoted_ids": promoted_ids,
        "rejected_count": len(rejected_ids),
        "already_fired": already_fired,
        "status": "already_fired" if already_fired else ("ledger_write_failed" if error == "ledger_write_failed" else "ok"),
    },
)
```

**run_monthly_sortino_gate** (gate_pass=True branch + already_fired/prior_pending paths; skip not_last_trading_friday):
```python
year, weeknum, _ = eval_date.isocalendar()
week_iso = f"{year}-W{weeknum:02d}"
log_fn(
    week_iso=week_iso,
    slot_id="monthly_gate",
    phase="phase-10",
    routine="run_monthly_sortino_gate",
    result={
        "gate_pass": result["gate_pass"],
        "approval_pending": result["approval_pending"],
        "sortino_delta": result.get("sortino_delta"),
        "month": result["month"],
        "status": "gate_pass" if result["gate_pass"] else "gate_fail",
        "reason": result.get("reason"),
    },
)
```

**auto_demote_on_dd_breach** (all paths including no_breach, already_demoted, auto_demoted):
```python
log_fn(
    week_iso=week_iso or "unknown",
    slot_id="rollback",
    phase="phase-10",
    routine="auto_demote_on_dd_breach",
    result={
        "demoted": result["demoted"],
        "decision": result["decision"],
        "challenger_id": challenger_id,
        "dd": dd,
        "threshold": threshold,
        "status": result["decision"],
    },
)
```

Note: `week_iso` may be None in rollback. Using `week_iso or "unknown"` is safe; BQ row will have
`week_iso="unknown"` which is queryable and clearly distinct from ISO week strings.

#### Test strategy for `tests/autoresearch/test_slot_usage_wiring.py`

Pattern: use `_capture_factory()` from the slot_accounting test idiom. Pass the capture closure as
`log_fn=capture` to each routine. Assert on `store` after the call.

```python
def _capture_factory():
    store: list = []
    def capture(**kwargs):
        store.append(kwargs)
        return {"inserted": True, "row_id": "test-row-id"}
    return store, capture

def test_thursday_batch_logs_slot(tmp_path):
    store, capture = _capture_factory()
    r = trigger_thursday_batch("2026-W17", ledger_path=tmp_path/"l.tsv", log_fn=capture)
    assert len(store) == 1
    assert store[0]["slot_id"] == "thu_batch"
    assert store[0]["week_iso"] == "2026-W17"
    assert store[0]["routine"] == "trigger_thursday_batch"
    assert "batch_id" in store[0]["result"]

def test_thursday_batch_already_fired_still_logs(tmp_path):
    store, capture = _capture_factory()
    lpath = tmp_path / "l.tsv"
    trigger_thursday_batch("2026-W17", ledger_path=lpath, log_fn=capture)
    trigger_thursday_batch("2026-W17", ledger_path=lpath, log_fn=capture)
    assert len(store) == 2
    assert store[1]["result"]["already_fired"] is True
```

Parametrize with `@pytest.mark.parametrize` for the four slot_ids to satisfy the masterplan
verification criterion: "captured log_slot_usage calls include all 4 slot_ids".

---

### 6. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched; 5 substantive)
- [x] 10+ unique URLs total (incl. snippet-only): 18 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (5 source files + 3 test files)
- [x] DI vs patch consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
