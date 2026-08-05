# Contract -- masterplan step 82.13

**Step id:** 82.13 (phase-82, P1, harness_required: true) | **Date:** 2026-08-05 | **Cycle:** 1

---

## 1. Research gate summary

**Brief:** `handoff/current/research_brief_82.13.md` | **Envelope:** `gate_passed: true`
-- 6 sources read in full, 18 snippet-only, 24 URLs, recency scan performed, 12 internal
files inspected, tier `moderate`.

### Findings that change the design (each re-measured by Main before adoption)

**(a) `preload_macro` has FIVE return paths and `0` is AMBIGUOUS.** Confirmed at
`backend/backtest/cache.py:333`:

| Line | Returns | Means |
|---|---|---|
| `:345` | **positive** cached total | already warm -- fires on EVERY optimizer iteration under `skip_cache_clear=True` |
| `:356` | `0` | table returned no rows |
| `:418` | `0` | **REFUSAL** -- could not parse a usable date |
| `:435` | `0` | **REFUSAL** -- a series is past its SLA |
| `:461` | positive | success |

So `if not preload_macro(): abort` is a trap: it cannot tell "refused" from "empty
table", and it would misfire on the warm path. **The only honest predicate for "do we
have macro" is `_macro_full` non-empty; the only honest predicate for "was it refused" is
a status the refusing branch sets itself.**

**(b) The consequence path is real** -- `cache.py:603-641`, one 30s-timeout BQ query per
distinct cutoff, driven from `historical_data.py:48` (`get_point_in_time_macro`).

**(c) THE "~40-MINUTE HANG" IS UNMEASURED FOLKLORE.** The brief traced it to
`CLAUDE.md:30` and found nothing measurable behind it. **This contract will not repeat
that number as measured**, and no artifact for this step may present it as evidence. The
defect stands on its own: an unbounded number of 30s queries where zero were intended.

**(d) `BacktestResult` (`:119-132`) has no availability field, and adding one is safe** --
the report dict is schemaless, `api/backtest.py:1059` already adds keys post-hoc, and the
TS interface is structural.

**(e) AST census: 967 files, 15 `preload_*` call sites, 8 discarded / 7 used.**

**(f) No test monkeypatches `preload_macro` on the engine path** -- so a new guard here
cannot collide with an existing fixture.

---

## 2. Hypothesis

The defect is not the discarded `int` -- it is that **no caller can distinguish a refusal
from an empty table**, so the engine cannot act on one. Giving the refusing branches a
structured status, having the engine read it, and recording macro availability on the
result converts a silent degradation into a labelled one. Crucially, a "macro-free mode"
that leaves the per-cutoff fallback armed does **not** fix the consequence -- it still
issues one query per cutoff -- so the degraded mode must actually **disable** the
fallback.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. `a fixture in which preload_macro returns 0 causes the backtest engine to surface the refusal explicitly rather than proceeding silently, asserted by a test that fails against the current discard-the-return behaviour`
2. `the run result records whether macro features were available, so a macro-free run cannot be mistaken for a normal one when its numbers are read later`
3. `every preload_* call site is enumerated from the source and each is classified as handling or discarding its return value, with file:line, and the enumeration is asserted non-empty`
4. `a fixture in which preload_macro succeeds is unaffected, so the guard cannot pass by always aborting`

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_13_preload_refusal_handling.py -q`

---

## 4. Plan

### 4.1 `backend/backtest/cache.py` -- make the refusal legible

- Module-level `_macro_status: dict` recorded on **every** return path with an explicit
  `outcome` (`"warm"` / `"empty"` / `"refused_unparseable"` / `"refused_stale"` /
  `"loaded"`), plus detail (stale series names, row counts).
- Public `macro_load_status() -> dict` and `reset_macro_status()` (test seam).
- **The `int` return contract is UNCHANGED**, so none of the other 14 call sites move.
- `MACRO_UNAVAILABLE` sentinel so a degraded run can **suppress the per-cutoff fallback**
  in `cached_macro` rather than silently issuing one query per cutoff. Without this the
  degraded mode does not fix the consequence.

### 4.2 `backend/backtest/backtest_engine.py` -- act on it

At the preload block (`:315-317`): read `macro_load_status()`. On a refusal outcome,
surface it explicitly via the existing `_report_progress` hook **and** a WARNING, set
`macro_available=False` + `macro_status` on the result, and put the cache into
macro-free mode so the fallback cannot fire. A normal load is untouched.

### 4.3 `BacktestResult` -- criterion 2

Add `macro_available: bool = True` and `macro_status: dict`. Flow through
`generate_report` so a degraded run is labelled where its numbers are read.

### 4.4 Tests -- `backend/tests/test_phase_82_13_preload_refusal_handling.py`

- **C1:** fixture where `preload_macro` refuses -> engine surfaces it. Must FAIL against
  the discard-the-return behaviour (verified by mutation, not asserted).
- **C2:** result carries `macro_available=False` + the reason; a normal run carries `True`.
- **C3:** **AST** enumeration of every `preload_*` call site (`ast.Call`, not grep --
  grep cannot tell a call from a mention), asserted **non-empty**, each classified
  handling/discarding with file:line.
- **C4:** success fixture unaffected -- no abort, no degraded flag, fallback still armed.
- **Ambiguity guard:** a `0` from the EMPTY-table path must not be reported as a refusal,
  and the warm path (positive return) must not trip anything.

### 4.5 Queued, not absorbed

The brief found two stale-anchor/docstring defects inside `cache.py` (`:51` cites a line
that has moved; the `:337` docstring is false on 3 of its 5 paths). Own step.

### 4.6 Out of scope

No change to macro SLAs, to `preload_prices`/`preload_fundamentals` (82.40 owns their
missing gates), or to any live position. The other 7 discarding call sites are
**classified** here, not fixed.

---

## 5. Files expected to change

`backend/backtest/cache.py`, `backend/backtest/backtest_engine.py`,
`backend/tests/test_phase_82_13_preload_refusal_handling.py` (NEW), `.claude/masterplan.json`.

---

## 6. References

`handoff/current/research_brief_82.13.md`; CWE-252 Unchecked Return Value; Google SRE
Book on graceful degradation (a degraded run must be labelled); ML experiment-metadata
prior art (model cards / MLflow run tags) on not pooling missing-feature runs with
complete ones.
