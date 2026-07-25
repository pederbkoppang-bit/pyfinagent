# Experiment Results — phase-80.1

**Step:** `80.1` (P0) — `GET /api/signals/{ticker}` returned HTTP 500 for every ticker.
Date 2026-07-25. Contract: `handoff/current/contract_80.1.md`.
Gate: `handoff/current/research_brief_80.1.md` (`gate_passed: true`, 7 sources in full,
22 URLs, 16 internal files).

---

## 1. What was built — **Layer A only**

A recursive non-finite→`None` sanitiser at the **API response boundary**. Nothing under
`backend/tools/` was touched.

**Why the boundary and not the source** — the single finding that set this step's scope:
`sector_analysis.get_sector_analysis` and `quant_model.get_quant_model_signal` are called
by **both** the API (`signals.py:94`, `:98`) **and the Layer-1 trading pipeline**
(`orchestrator.py:1261`, `:1271-1273` → `:1992`, `:2014` →
`backend/tasks/analysis.py:176`, `:297-298` → BigQuery). A sanitiser in `backend/api/`
**cannot** move the live book — the orchestrator never traverses Starlette. A `dropna` in
`backend/tools/` **would** change trading inputs. So the source fix is deferred to 80.27,
where it can be judged against trading criteria instead of a 200-vs-500 criterion.

**Why not the obvious alternatives** (all measured/read, not assumed):
- `allow_nan=True` → emits a bare `NaN` token. RFC 8259 §6: *"Numeric values that cannot
  be represented in the grammar below (such as Infinity and NaN) are not permitted."* The
  browser's `JSON.parse` throws. That relocates the break; it does not fix it.
- Pydantic `response_model` → Pydantic's own default for `ser_json_inf_nan` **is** `'null'`
  (measured: `TypeAdapter(dict).dump_json({"a": nan})` → `b'{"a":null}'`), **but FastAPI
  never consults model config on this path** (`serialize_response` → `jsonable_encoder` →
  `JSONResponse.render`). fastapi#11821.
- `ORJSONResponse` → works, but adds an uninstalled compiled dependency.
- **A yfinance flag → does not exist.** `keepna=False` is already the default and its mask
  (`scrapers/history.py:495-499`) is `.all(axis=1)`: a row is dropped only if *every*
  price+volume column is NaN-or-zero. The placeholder row has a **real non-zero Volume**,
  so it survives. Post-hoc handling is the only mechanism available.

### Files

| File | Δ | What |
|---|---|---|
| `backend/api/_json_safe.py` | **new** | `sanitize_non_finite()` + `NaNSafeJSONResponse` |
| `backend/api/signals.py` | +~20 | import + `default_response_class=NaNSafeJSONResponse` on the signals router **only** (13 routes, all JSON) |
| `backend/tests/test_phase_80_1_signals_nan_serialisation.py` | **new**, 14 tests | unit + endpoint level |

`isinstance(v, float) and not math.isfinite(v)` → `None`. Ordering matters:
`math.isfinite(None)` **raises TypeError**, so the isinstance check must come first.
`np.float64` is a `float` subclass so numpy scalars are caught; `np.isfinite` and `pd.isna`
are deliberately **not** used (the former raises on non-numeric input, the latter returns
an array for list input). `bool` subclasses `int`, not `float`, so booleans pass through.

**Scoped to the signals router, never app-wide** — an app-wide `default_response_class`
would silently re-encode ~70 unrelated routes.

---

## 2. Verification output (verbatim)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_80_1_signals_nan_serialisation.py -q
..............                                                           [100%]
14 passed, 40 warnings in 2.44s
```

Immutable verification command, live (see `live_check_80.1.md` §A for the rig):

```
BEFORE  operator :8000 (pre-fix)  -> HTTP 500  (18.958060s)
AFTER   :8001 rig (80.1 code)     -> 200
```

---

## 3. Mutation matrix — **the 5 mutations run were all killed**

> **Cycle-2 correction (Q/A finding 2).** This heading previously read
> "**5/5 guards held, 0 vacuous**". The second half was a forbidden
> suite-level claim: a mutation matrix licenses only *"these N mutations were
> killed"*, never *"this suite contains no vacuous guard"*
> (Goodenough-Gerhart; `feedback_measure_dont_assert_claims`). Q/A falsified
> it directly by finding a vacuous test in this very file — see §3.1.

Driver `scratchpad/mutate_80_1.py`; each mutation applied to the real file, guard run,
file restored from an in-memory snapshot (never `git stash`).

| # | Mutation | File | Result |
|---|---|---|---|
| N1 | Remove the sanitiser entirely (router back to plain `JSONResponse`) | `signals.py` | **FAILED as required** |
| N2 | Sanitiser returns `0.0` instead of `None` (the silent-wrong-number regression) | `_json_safe.py` | **FAILED as required** |
| N3 | Sanitiser **DROPS** the key instead of nulling it | `_json_safe.py` | **FAILED as required** |
| **N4** | **FIXTURE mutation** — sector tool returns a finite float instead of NaN | *the test file* | **FAILED as required** |
| N5 | Sanitiser stops recursing into lists (nested `quant_model` leak returns) | `_json_safe.py` | **FAILED as required** |

N2 and N3 are the two that matter for the step's **additional** criterion: a test asserting
only "no 500" would survive both. N4 is the class this project has shipped broken five
times — if the fixture can't represent the failure, every assertion above it is vacuous.

### 3.1 CYCLE 2 — a vacuous test I shipped, found by Q/A

`test_the_fixture_really_carries_a_non_finite_float` asserted
`not math.isfinite(float("nan"))` and `math.isfinite(2.5)`. Those are **library facts** —
true regardless of what the fixture does. So the test **PASSED under the very fixture
mutation (N4) it claimed to guard**, and its `from backend.tools import sector_analysis`
was dead (`# noqa: F401`). This is the *"library-fact assertion posing as a fixture pin"*
shape recorded verbatim in `feedback_mutation_test_guards_and_fixtures` — I shipped the
exact anti-pattern that memory exists to prevent, in a step where I had already written a
correct behavioural guard two lines away.

Replaced with a pin that binds to the actual subject via `inspect.signature`:

```python
default = inspect.signature(_install_fake_tools).parameters["sector_1mo"].default
assert isinstance(default, float)
assert not math.isfinite(default)
```

**Proof it is no longer vacuous** — the same N4 mutation, before and after:

```
BEFORE (old test):  N4 applied -> test PASSED        <- vacuous
AFTER  (new test):  N4 applied -> 1 failed
  backend/tests/..._nan_serialisation.py:276: AssertionError
Restored; suite 14 passed.
```

**Honest accounting:** N4 was still killed by a genuine behavioural guard
(`test_signals_endpoint_returns_200_with_a_nan_sector_return`), so the fixture was never
actually unpinned — the defect was in the *record*, not the fix. That is precisely why the
"0 vacuous" claim was the real error: it asserted a property of the whole suite that one
mutation run cannot establish.

**A separate vacuity I caught myself while writing the fixture:** my first draft guarded the
monkeypatching with `if hasattr(mod, fn)`. Every one of my 12 assumed function names was
wrong (the real ones are `get_insider_trades`, `get_options_flow`, `get_patent_data`,
`get_macro_indicators` on `fred_data` not `fred_macro`, `get_anomaly_scan`,
`get_monte_carlo_simulation`, …). With `hasattr`, the patches would have silently no-op'd
and the tests would have hit the **real network tools** — passing for the wrong reason.
It now `assert`s each attribute exists, so a future rename fails loudly.

---

## 4. Criteria → evidence

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | `GET /api/signals/AAPL` returns 200 (not 500) | live_check §B: 500 → 200, re-measured against live `:8000` for the "before" | **MET** |
| 2 | all 12 signal keys present | live_check §C: `present 12/12; missing=NONE` | **MET** |
| 3 | NaN rendered as null/None, NOT dropped and NOT 500 | live_check §D: `"1mo"` present with value `null`; mutations N2 (0.0) and N3 (dropped) both kill the suite | **MET** |
| 4 | regression test + MUTATION-TESTED | 14 tests; §3 matrix, N1 reverts the sanitiser and the suite goes red | **MET** |
| add | assert the period key is ABSENT-or-None, not merely "no 500" | the endpoint test asserts `"1mo" in returns` **AND** `returns["1mo"] is None`; N2/N3 prove each half can fail | **MET** |

---

## 5. Scope honesty

- **Inert on `:8000` until the operator restarts** (`79.55` is an open RESTART BLOCKER).
  Same disclosure as 80.2. On the un-restarted process the endpoint still 500s — that is
  the "before" control in live_check §B, not a stale quote.
- **Layer B (`dropna` at `quant_model.py:63`) was NOT done**, deliberately. It is a
  correct one-word fix and the temptation was real, but it changes `quant_model.score` for
  live tickers on the next analysis cycle. Deferred to 80.27. If it is ever done it must
  use row-wise `hist.dropna(subset=["Close"])` so `closes`/`volumes` stay index-aligned —
  the column-independent form at `anomaly_detector.py:66-69` is the very defect queued as
  **80.31**.
- **80.27 is NOT fixed and its evidence is intact by design** — live_check §F measures
  **31 non-finite floats** still in the raw tool output (independently reproducing the
  audit's count), `signal: 'NEUTRAL'` from an all-NaN input, and a summary string that
  literally reads `+nan%`.

### A framing correction I am recording rather than burying

I began this step assuming the 500 was an alarm protecting the trading path, and that
silencing it was therefore dangerous. **That was wrong**, and the research corrected it:
the trading path never renders JSON, so it never 500'd. The endpoint has been dead while
the pipeline quietly produced NEUTRALs. Fixing the display removes an alarm that was only
ever protecting the UI.

This does not make the conflation safe — it means **80.27 was always the real defect** and
this step must not be allowed to look like its fix. In practice the outcome is better than
neutral: live_check §G shows the repaired page now renders *"3M return: +nan% vs sector
+nan% vs S&P +nan%. Signal: NEUTRAL"* and *"Quant model score: nan → NEUTRAL. MDA source:
backtest"* **on screen**. The defect moved from invisible-behind-a-500 to visible.

### Queued, not silently fixed

1. **`nlp_sentiment.py:161`** — `np.mean([])` is neutralised by a `max/min` clamp into a
   silently wrong **`1.0`**. A correctness smell, not a 500; needs its own step.
2. **The analysis-report poll route** — flagged by the researcher as a HIGH-PRIORITY
   CANDIDATE that embeds the same `quant_model` dict and may 500 identically.
   **NOT VERIFIED** — no HTTP request was issued against it. Flagged, not asserted.
3. **Frontend `null` rendering** — a null signal value should render as an explicit "data
   unavailable" state, not blank or zero, or "green" still hides missing data.

### Membership rule for the NaN-leak sweep (stated, not asserted)

From the brief: *under `backend/tools|api/` AND produces a float from
arithmetic/aggregation AND can reach a returned dict AND has no `dropna`/`isfinite` guard
between.* Applying it: **only the two known sites leak.** The other 10 tools in the gather
all guard their denominators (`max(x,1)` / `if total == 0`), read individually.
`backend/api/*.py` has zero arithmetic hits. Excluded by rule step 3 (not on a response
path, recorded so the exclusion is auditable if that changes): `screener.py`,
`price_quality.py`.

---

## 6. DO-NO-HARM

| Item | Status |
|---|---|
| Live paper-trading book | **Untouched.** Nothing under `backend/tools/`, `orchestrator.py`, `tasks/analysis.py`, or `config/prompts.py` — all four on the trading path |
| Structural argument | The orchestrator never traverses Starlette, so an API-layer response class is *incapable* of changing a trading input |
| `.env` / flags / optimizer | No edit, no flip, no run; `historical_macro` FROZEN |
| Kill-switch / stops / sector caps / DSR / PBO | Not in the diff |
| `default_response_class` | Signals router only — **never app-wide** |
| `allow_nan` | **Not** used (would emit invalid JSON) |
| `_safe()`'s `{"signal": "ERROR", ...}` at `signals.py:66` | Left alone — the documented tool-failure shape, and **80.27 depends on it** |
| Second trading loop | Prevented by `--lifespan off` on the verification rig |
| Operator `:8000` / `:3000` | pid 70791 unchanged; `/` 302, `/login` 200 |
| `tsconfig.json` / `next-env.d.ts` | Restored, md5s back to baseline, `git status` clean |

## 7. Tier ledger

| Phase | Role | Model / effort | Why |
|---|---|---|---|
| RESEARCH | Agent-tool `researcher` | **T3** Opus 5 / max | The scope question (shared code with the trading pipeline) was the whole risk; worth depth |
| GENERATE | Main | **T3** Opus 5 / xhigh | Bounded, well-specified fix once the boundary decision was made |
| EVALUATE | fresh Q/A | **T3** Opus 5 / max | Independent verdict on a P0 |

**Fable (T4) not spent** — this step changes no trading logic. Quota reserved for `80.27`.
