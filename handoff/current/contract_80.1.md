# Contract — phase-80.1

**Step id:** `80.1` (phase-80, priority **P0**, `harness_required: true`)
**Title:** *[P0 — FEATURE FULLY DEAD] `GET /api/signals/{ticker}` returns HTTP 500 for
EVERY ticker, so the entire Signals page cannot fetch.* This is the operator-reported bug
in the 2026-07-25 screenshot (*"Network error calling /api/signals/AAPL: Load failed"*).

Date 2026-07-25 | Wave 1 (the NaN family) of the masterplan drain | **Tier T3** (Opus 5
`xhigh`). Fable/T4 is reserved for `80.27`, the step that touches live decisions.

---

## 1. Research gate (PASSED — ran BEFORE this contract)

`handoff/current/research_brief_80.1.md`, Agent-tool `researcher` (Opus 5 / max),
written incrementally. Envelope: **`gate_passed: true`**,
`external_sources_read_in_full: 7` (floor 5), `urls_collected: 22` (floor 10),
`recency_scan_performed: true`, `internal_files_inspected: 16`.

Findings that decided the design:

- **The 500 is unavoidable at the framework level.** starlette 1.0.0
  `responses.py:194-201` hardcodes `json.dumps(..., allow_nan=False)`, and FastAPI's
  `jsonable_encoder` passes `np.float64` NaN through untouched (measured). So *any*
  non-finite float in a response dict is a guaranteed 500.
- **`_safe()` provably cannot catch it** (`signals.py:57-66`, read not assumed): the
  `ValueError` is raised during ASGI **rendering**, after the route has already returned.
- **Both leak sites confirmed at the cited lines.** `sector_analysis.py:34`
  (`_compute_return`, with `is not None` misused as a *numeric-validity* test at `:55`,
  `:71`, `:79`, `:88`; SPY fetched for every ticker at `:74`, which is why the failure is
  ticker-universal) and `quant_model.py:63` (missing `.dropna()`, unlike siblings
  `anomaly_detector.py:66` and `monte_carlo.py:40`).
- **yfinance has NO flag to suppress the forming-session row.** `keepna=False` is already
  the default, and its mask at `scrapers/history.py:495-499` uses `.all(axis=1)` — the
  placeholder row *escapes* it because Volume is a real int64. Post-hoc dropping is the
  only mechanism. (This kills the "just pass a parameter" option outright.)
- **Rejected alternatives, with reasons:** `allow_nan=True` emits bare `NaN` — **invalid
  JSON**, so it breaks `JSON.parse` in the browser instead of fixing anything; a Pydantic
  `response_model` does **not** work because FastAPI ignores `ser_json_inf_nan`
  (fastapi#11821); an app-wide `ORJSONResponse` adds a dependency and silently re-encodes
  ~70 routes.
- **No existing sanitiser to reuse** (grep-confirmed). The only precedent,
  `backend/slack_bot/formatters.py:663`, returns `0.0` — **copying it would violate
  criterion 3.**

### The finding that sets the scope of this step

> `sector_analysis.get_sector_analysis` and `quant_model.get_quant_model_signal` are
> **SHARED with the Layer-1 trading pipeline** at `orchestrator.py:1261` and `:1271-1273`
> — so a fix inside `backend/tools/` changes trading inputs (80.27), while a fix in
> `backend/api/` cannot. **Do Layer A only.**

---

## 2. Hypothesis

A recursive non-finite→`None` sanitiser applied at the **API response boundary** makes
every signals sub-endpoint serialisable without touching a single line that the trading
pipeline executes. The orchestrator never traverses Starlette, so this change is
*structurally* incapable of moving the paper-trading book.

---

## 3. Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

> 1. GET /api/signals/AAPL returns HTTP 200 (not 500) with the backend running
> 2. The response body parses as JSON and every one of the 12 signal keys is present (insider, options, social_sentiment, patent, earnings_tone, fred_macro, alt_data, sector, nlp_sentiment, anomalies, monte_carlo, quant_model)
> 3. A ticker whose sector returns produce NaN still yields 200 -- the NaN is rendered as null/None, NOT dropped silently and NOT 500
> 4. A regression test asserts the NaN case: construct a payload containing float('nan') and assert the endpoint/serialiser returns 200. MUTATION-TEST IT -- revert the sanitiser and confirm the test FAILS (a guard that cannot fail does not count; see feedback_mutation_test_guards_and_fixtures)

**Immutable verification command** (verbatim):

```
curl -s -m 120 -o /dev/null -w '%{http_code}\n' http://localhost:8000/api/signals/AAPL
```

**Immutable `live_check`** (verbatim): `handoff/current/live_check_80.1.md`: verbatim curl
output showing 200 + the sector block with its 1mo value, plus the pytest output for the
NaN regression test.

**ADDITIONAL criterion carried in the step body** (verbatim):

> the mutation test must assert the period key is ABSENT (or None) rather than merely that
> no 500 occurs -- a test asserting only 'no 500' would still pass if NaN were replaced by
> 0.0, which is a silent-wrong-number regression, not a fix.

---

## 4. Plan — **Layer A only**

1. **New `backend/api/_json_safe.py`**: `sanitize_non_finite(obj)` — a recursive walker
   over `dict` / `list` / `tuple` mapping any non-finite float to `None`, plus a
   `NaNSafeJSONResponse(JSONResponse)` that sanitises in `render()`.
   - Predicate `isinstance(v, float) and not math.isfinite(v)`. `np.float64` **is** a
     `float` subclass, so this catches it; the brief measured the traps that rule out
     `np.isfinite` (raises on non-numeric) and `pd.isna` (elementwise on arrays).
   - **Must PRESERVE the key** (criterion 3: "NOT dropped silently"). A
     `{k: v for ... if isfinite}` comprehension would pass a naive "no 500" test and fail
     criterion 3 — it will be mutation-tested precisely for this.
   - Must recurse into nested dicts AND lists (`quant_model.data.features` is a nested
     dict; `top_factors` is a list of dicts).
2. **`backend/api/signals.py`**: set `default_response_class=NaNSafeJSONResponse` on the
   **signals router only** (`:29`, 13 routes, all JSON — verified none returns a
   `StreamingResponse`/`FileResponse`). This satisfies the step's "fix it at the response
   boundary so EVERY signal sub-endpoint is covered, not just sector". It is explicitly
   **not** app-wide — the brief's live-book checklist forbids that.
3. **Tests** — `backend/tests/test_phase_80_1_signals_nan_serialisation.py`:
   - endpoint-level via `authed_test_client(app)` with the 12 tool functions
     monkeypatched (no network), the fake `sector` tool returning
     `{"stock_returns": {"1mo": float("nan")}}` — reproducing the exact reported chain
     (`dict item "1mo"` → `"stock_returns"` → `"sector"`);
   - assert **all three**: `200`; all 12 keys present; and
     `body["sector"]["stock_returns"]["1mo"] is None` **AND**
     `"1mo" in body["sector"]["stock_returns"]`;
   - unit tests on the sanitiser for `inf`, `-inf`, `np.float64('nan')`, nested lists,
     tuples, and non-float types passing through untouched.
4. **Mutation matrix — all four, including the FIXTURE mutation:**

   | # | Mutation | Expected |
   |---|---|---|
   | N1 | Remove the sanitiser entirely | test FAILS (500) |
   | N2 | Sanitiser returns `0.0` instead of `None` | test FAILS (assertion 3) |
   | N3 | Sanitiser DROPS the key instead of nulling it | test FAILS (the `in` check) |
   | N4 | **FIXTURE** mutated to return a finite float instead of NaN | test must FAIL — else the fixture cannot represent the failure |

   N4 is the class this project has shipped broken five times
   (`feedback_mutation_test_guards_and_fixtures`).
5. **live_check** on the isolated `:8001` rig (`--lifespan off`), same discipline as 80.2 —
   the operator's `:8000` stays un-restarted while `79.55` is open.

---

## 5. Explicitly OUT of scope

- **Layer B — `dropna` at `quant_model.py:63`.** It is a genuinely correct one-word fix
  and the temptation is real, but it **changes numbers the trading pipeline consumes** and
  would alter `quant_model.score` for live tickers on the next analysis cycle. Deferred to
  **80.27**, where it can be judged against trading criteria instead of a 200-vs-500
  criterion. If it is ever done, it must use the row-wise `hist.dropna(subset=["Close"])`
  form so `closes`/`volumes` stay index-aligned — the column-independent form at
  `anomaly_detector.py:66-69` is the very defect queued as **80.31**.
- **The NEUTRAL-laundering chain** (`quant_model._classify_signal` falling through to
  `NEUTRAL` on NaN; `sector_analysis.py:136-153` likewise; and
  `backend/config/prompts.py:1111` doing `json.dumps(...)` with stdlib
  `allow_nan=True`, handing the Gemini agent a prompt containing the literal token `NaN`
  — a **third** laundering surface the brief found). All of it is 80.27.

**A framing correction I am recording rather than burying** (brief §"where I'd push
back"): I had assumed the 500 was an alarm protecting the trading path. It was not — the
trading path never renders JSON and so never 500s. The endpoint has been down while the
pipeline quietly produced NEUTRALs. Fixing the display removes an alarm that was only ever
protecting the UI. That does **not** make this step safe to conflate with 80.27; it means
80.27 was always the real defect and this step must not be allowed to look like its fix.

**Non-blocking expectation:** the frontend should render a `null` signal value as an
explicit "data unavailable" state, not as blank or zero — otherwise "green" still hides
missing data. Queued, not done here.

---

## 6. DO-NO-HARM

- **Structurally cannot move the book:** nothing under `backend/tools/`,
  `backend/agents/orchestrator.py`, `backend/tasks/analysis.py`, or
  `backend/config/prompts.py` is touched — all four are on the trading path. The
  orchestrator does not traverse Starlette.
- No `.env` edit, no flag flip, no optimizer run, `historical_macro` FROZEN.
  Kill-switch / stops / sector caps / DSR / PBO byte-untouched.
- `default_response_class` scoped to the signals router, **never app-wide**.
- `allow_nan=True` is **not** used (it produces invalid JSON).
- `_safe()`'s `{"signal": "ERROR", ...}` fallback at `signals.py:66` is left alone — it is
  the documented tool-failure shape from `.claude/rules/backend-tools.md` and **80.27
  depends on it**.
- **80.27's evidence is deliberately left intact:** the sanitiser sits at the far edge of
  the pipe, so a Q/A or operator investigating 80.27 will still see NaN in the tool
  outputs and the literal `NaN` token in the LLM prompt.
- `git add -An` before the flip.

## 7. Evidence to produce

`experiment_results_80.1.md` (files + verbatim output + mutation matrix + tier ledger) ·
`live_check_80.1.md` · `evaluator_critique.md` (Q/A verdict, transcribed verbatim) ·
`harness_log.md` append **before** the flip.
