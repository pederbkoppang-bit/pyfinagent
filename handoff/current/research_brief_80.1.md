# Research Brief — masterplan step 80.1

**Step:** 80.1 (phase-80, P0) — `GET /api/signals/{ticker}` returns HTTP 500 for EVERY ticker
**Tier:** moderate | `coverage.audit_class = false`
**Researcher session:** 2026-07-25
**Status:** COMPLETE — `gate_passed: true` (7 external sources read in full, recency scan performed)

---

## Immutable success criteria (copied VERBATIM from the caller / masterplan — do not amend)

1. GET /api/signals/AAPL returns HTTP 200 (not 500) with the backend running
2. The response body parses as JSON and every one of the 12 signal keys is present (insider, options, social_sentiment, patent, earnings_tone, fred_macro, alt_data, sector, nlp_sentiment, anomalies, monte_carlo, quant_model)
3. A ticker whose sector returns produce NaN still yields 200 -- the NaN is rendered as null/None, NOT dropped silently and NOT 500
4. A regression test asserts the NaN case: construct a payload containing float('nan') and assert the endpoint/serialiser returns 200. MUTATION-TEST IT -- revert the sanitiser and confirm the test FAILS (a guard that cannot fail does not count; see feedback_mutation_test_guards_and_fixtures)

**ADDITIONAL criterion from the step body:** the mutation test must assert the period key is **ABSENT (or None)** rather than merely that no 500 occurs — a test asserting only "no 500" would still pass if NaN were replaced by `0.0`, which is a silent-wrong-number regression, not a fix.

---

## Search queries run (3-variant discipline)

| Variant | Query | Purpose |
|---|---|---|
| Current-year frontier | `FastAPI JSONResponse "Out of range float values are not JSON compliant" NaN fix 2026` | latest state of the FastAPI/Starlette NaN issue |
| Last-2-year window | `"ser_json_inf_nan" OR "NaN to null" FastAPI response sanitizer 2025 2026 best practice` | 2025-2026 recency scan; surfaced fastapi#11821 + a Jan-2026 write-up |
| Year-less canonical | `recursive sanitize NaN Infinity to None before JSON serialization Python nested dict` | prior art on the recursive-sanitiser idiom (surfaced Python `json` stdlib docs, simplejson, Evan Hahn's "Python's nonstandard JSON encoding", a 2022 xtao-org write-up) |
| Year-less canonical (domain 2) | `yfinance history NaN last row current session incomplete candle Volume real Close NaN` | the upstream placeholder-row mechanism |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|---|
| 1 | https://github.com/fastapi/fastapi/discussions/8029 | 2026-07-25 | Official repo discussion | WebFetch | `jsonable_encoder` "returns objects of type `float` as is" — it does NOT sanitise NaN. "The error occurs downstream when Starlette's `JSONResponse` calls `json.dumps` with `allow_nan=False`." Community/maintainer-blessed fix = `ORJSONResponse`; "Converting NaN to null...such json is perfectly fine". |
| 2 | https://www.rfc-editor.org/rfc/rfc8259.html | 2026-07-25 | IETF standard (peer-reviewed tier) | WebFetch | §6 Numbers, verbatim: **"Numeric values that cannot be represented in the grammar below (such as Infinity and NaN) are not permitted."** So a `NaN` literal in a response body is NOT JSON. |
| 3 | https://pydantic.dev/docs/validation/latest/api/pydantic/config/ | 2026-07-25 | Official docs | WebFetch (after 301 from docs.pydantic.dev) | `ser_json_inf_nan`: "The encoding of JSON serialized infinity and NaN float values." **Default = `'null'`**; `'null'` → `null`, `'constants'` → `NaN`/`Infinity`, `'strings'` → `"NaN"`/`"Infinity"`. |
| 4 | https://github.com/ijl/orjson | 2026-07-25 | Official docs (library README) | WebFetch | Verbatim: **"orjson.dumps() serializes Nan, Infinity, and -Infinity, which are not compliant JSON, as `null`"**. No option exists to change this. numpy requires `option=orjson.OPT_SERIALIZE_NUMPY`. |
| 5 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify | 2026-07-25 | Official docs | WebFetch | Verbatim: "The numbers `Infinity` and `NaN`, as well as the value `null`, are all considered `null`. (But unlike the values in the previous point, **they would never be omitted**.)" `JSON.stringify([NaN, null, Infinity])` → `'[null,null,null]'`. This is the ecosystem precedent for criterion 3: **null, not dropped**. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/fastapi/fastapi/issues/459 | Repo issue ("jsonable_encoder should handle nan", open since 2019) | Superseded by discussion #8029, which was read in full |
| https://github.com/fastapi/fastapi/issues/4589 | Repo issue (string "NaN" coerced to float) | Different failure mode (validation, not serialisation) |
| https://github.com/fastapi/fastapi/discussions/6330 | Repo discussion | Duplicate of #4589 |
| https://github.com/pydantic/pydantic-core/pull/1062 | PR (restrictive floats rejecting NaN/Inf) | Input-validation-side; 80.1 is output-side |
| https://github.com/PrefectHQ/prefect/issues/6439 | Third-party issue, same ValueError | Community tier; no new mechanism |
| https://itsourcecode.com/valueerror/...json-compliant/ | Community blog | Community tier (lowest weight); content duplicated by #1 |
| https://github.com/ranaroussi/yfinance/issues/413 | Repo issue ("how to stop yfinance giving a NaN row") | **Fetch ATTEMPTED and completed, but the rendered page contained no maintainer replies — no evidence extracted.** Deliberately NOT counted toward the gate rather than padding the count. |
| https://github.com/ranaroussi/yfinance/issues/2353 | Repo issue (current-session day missing from `max`) | Snippet only; adjacent symptom |
| https://docs.python.org/3/library/json.html | Official docs | Snippet only; `allow_nan` semantics already measured locally (see empirics table) |
| https://evanhahn.com/pythons-nonstandard-json-encoding/ | Named-author blog | Snippet only; corroborates RFC 8259 finding already read in full |
| https://simplejson.readthedocs.io/ | Official docs | Snippet only; alternative encoder not under consideration |
| https://github.com/infiniflow/ragflow/pull/15266 | Third-party PR (recursive NaN sanitiser in a shipped product) | Snippet only; prior-art existence proof for the recursive-sanitiser idiom |
| https://github.com/fastapi/fastapi/discussions/11577 | Repo discussion (`allow_inf_nan=False` on Query) | Snippet only; input-side, not output-side |
| https://github.com/fastapi/fastapi/discussions/8912 | Repo discussion (NaN default in request body) | Snippet only; input-side |

---

## Recency scan (2024-2026)

**Performed.** Result: **3 findings in the 2024-2026 window that materially change the plan**, plus confirmation that no framework-level fix has landed.

1. **2026-01** — https://www.fmularczyk.pl/posts/2026_01_nan_null_none/ (read in full). The current-best per-field pattern:
   ```python
   FloatNaN = Annotated[
       float,
       PlainSerializer(lambda v: None if math.isnan(v) else v, when_used="json"),
       BeforeValidator(lambda v: float("nan") if v is None else v),
   ]
   ```
   Author's rationale, verbatim: *"NaN is a valid float value that we use to indicate that a threshold is undefined or not applicable, since any comparison with NaN will return False."* — i.e. NaN carries the semantics "undefined", which is exactly why `0.0` is the wrong substitute. Caveat the author states: non-Python consumers must handle null→NaN themselves.
2. **2024 → still open in 2026** — https://github.com/fastapi/fastapi/discussions/11821 (read in full). **FastAPI ignores pydantic's `ser_json_inf_nan`.** This is the single most decision-relevant recency finding: it forecloses the "just add a `response_model` with `model_config`" option (see Key finding 4).
3. **2024** — https://github.com/pydantic/pydantic-core/pull/1062 (snippet). Adds *input*-side restrictive floats that reject NaN/Inf. Output-side unchanged; does not help 80.1.

**No new finding supersedes RFC 8259 (2017)** — it remains the governing spec, and `NaN` is still not a JSON token. **No framework-level fix has shipped** in Starlette 1.0.0 / FastAPI 0.135.2: `allow_nan=False` is still hardcoded at `starlette/responses.py:198` in the installed version (read locally). The application still owns this.

---

## Key findings

1. **`_safe()` cannot catch it — CONFIRMED by reading the code, not assumed.** `backend/api/signals.py:58-66` guards only the `await`. The `ValueError` is raised inside `starlette/responses.py:194-201` during ASGI rendering, after the route function has returned. Corroborated externally: *"The error occurs downstream when Starlette's `JSONResponse` calls `json.dumps` with `allow_nan=False`"* (fastapi#8029).

2. **`jsonable_encoder` is not a sanitiser.** Measured locally: `jsonable_encoder({"1mo": np.float64('nan')})` → `{'1mo': np.float64(nan)}`, unchanged. Matches fastapi#8029: *"returns objects of type `float` as is"*. Reason: `np.float64` **is** a `float` subclass (`isinstance(np.float64('nan'), float) is True`, measured), so the encoder's fast-path passes it straight through.

3. **`allow_nan=True` is NOT an acceptable fix.** RFC 8259 §6, verbatim: **"Numeric values that cannot be represented in the grammar below (such as Infinity and NaN) are not permitted."** Measured: `json.dumps({"a": nan}, allow_nan=True)` → `{"a": NaN}`. A browser `JSON.parse` on that throws `SyntaxError` (JSON has no `NaN` token). **A custom `JSONResponse` subclass that flips `allow_nan=True` would turn the backend 500 into a frontend parse failure — it moves the break, it does not fix it. Reject this option.**

4. **A Pydantic `response_model` with `ser_json_inf_nan='null'` DOES NOT fix it.** Pydantic's default already is `'null'` (official docs, read in full) and measured: `TypeAdapter(dict).dump_json({"a": nan})` → `b'{"a":null}'`. **But FastAPI does not use that path for the response body.** fastapi#11821, verbatim on the mechanism: *"The response serialization follows this path: `serialize_response` → `jsonable_encoder` → `JSONResponse.render`. At each stage, FastAPI's default handlers don't check the Pydantic model's `ser_json_inf_nan` configuration"* and *"Simply setting `ser_json_inf_nan='null'` in model configuration won't prevent the 500 error occurring at the serialization stage."* Pydantic dumps to *Python objects*, then Starlette re-serialises with `allow_nan=False`. **Reject this option too** (it would also require authoring a 12-key response model, a large diff for a P0).

5. **`null` is the correct representation, and it must NOT be dropped.** MDN, verbatim: *"The numbers `Infinity` and `NaN`, as well as the value `null`, are all considered `null`. (But unlike the values in the previous point, **they would never be omitted**.)"* — `JSON.stringify([NaN, null, Infinity])` → `'[null,null,null]'`. orjson README, verbatim: *"orjson.dumps() serializes Nan, Infinity, and -Infinity, which are not compliant JSON, as `null`"*. Two independent ecosystems converge on **NaN → null, key retained**. That is exactly criterion 3.

6. **`ORJSONResponse` would satisfy criteria 1-3 with a one-line diff — but it is the wrong trade here.** It is the community answer in fastapi#8029 and orjson's `null` behaviour is hardcoded (no config knob). Against it: (a) `orjson` is an *optional* FastAPI dependency and is **not currently installed** in this venv — a P0 fix should not add a compiled Rust dependency to `requirements.txt`; (b) FastAPI's own custom-response docs steer away from orjson-for-its-own-sake (*"If what you are looking for is performance, you are probably better off using a Response Model"*); (c) it changes serialisation for whichever routes it is applied to, and app-wide `default_response_class` would silently change the numeric/`datetime`/`UUID` encoding of **all ~70 routes** — an unnecessarily wide blast radius for a P0 on one endpoint. **Recommend: a recursive sanitiser (finding 7), with `ORJSONResponse` documented as the rejected alternative.**

7. **The recursive-sanitiser idiom is well-attested prior art.** Independently recommended by the year-less canonical search (Evan Hahn; the xtao-org write-up: *"the main solutions are: raise an error, convert to null, or convert to strings"*) and shipped in production elsewhere (ragflow PR #15266, *"sanitize NaN/Inf scores before serializing"*). It gives exact control over criterion 3 (`null`, not dropped, not `0.0`) with **zero new dependencies** and a blast radius the author chooses.

8. **The correct predicate, with the traps measured.** Recommended: `isinstance(v, float) and not math.isfinite(v)` → `None`.

   | Probe | Measured result | Consequence for the sanitiser |
   |---|---|---|
   | `math.isfinite(np.float64('nan'))` | `False` | works — `np.float64` is a `float` subclass |
   | `math.isfinite(None)` | **raises `TypeError`** | MUST guard with `isinstance(v, float)` FIRST |
   | `np.isfinite(pd.NA)` | `<NA>` (not a bool) | truthiness of the result raises — **do not use `np.isfinite` in the sanitiser** |
   | `pd.isna(['a','b'])` | `array([False, False])` | returns an array, unusable in an `if` — **do not use `pd.isna` in the sanitiser** |
   | `math.isfinite(Decimal('NaN'))` | `False` | Decimal is handled if reached, but is not a `float` subclass → won't match the isinstance guard (acceptable; no Decimals in this payload) |
   | `isinstance(True, int)` | `True` | bools are `int`, not `float` — safely untouched |

   Note `np.float32` is **not** a Python-`float` subclass. Nothing in this payload produces float32 (`to_numpy(dtype=float)` → float64), but if the sanitiser is to be general, use `isinstance(v, (float, np.floating))`. Prefer the plain-`float` form if you do not want `numpy` imported at the API layer.

9. **yfinance has NO flag that suppresses the forming-session row — this is the decisive answer to question A4.** `keepna=False` is already the default (`yfinance/scrapers/history.py`, signature default `keepna=False`; docstring *"Keep NaN rows returned by Yahoo? Default: False"*). Its implementation at `scrapers/history.py:495-499`:
   ```python
   if not keepna:
       data_colnames = _PRICE_COLNAMES_ + ['Volume'] + ['Dividends', 'Stock Splits', 'Capital Gains']
       data_colnames = [c for c in data_colnames if c in df.columns]
       mask_nan_or_zero = (df[data_colnames].isna() | (df[data_colnames] == 0)).all(axis=1)
       df = df.drop(mask_nan_or_zero.index[mask_nan_or_zero])
   ```
   `.all(axis=1)` means a row is dropped **only if EVERY** price+volume column is NaN-or-zero. The forming-session placeholder row has a **real non-zero Volume**, so the mask is `False` and the row survives. Measured on a synthetic frame reproducing the caller's SPY observation: drop-mask `[False, False]` — placeholder row **NOT dropped**; `df.dropna(subset=["Close"])` correctly keeps 1 row.
   **Conclusion: `prepost` / `repair` / `auto_adjust` / `actions` / `keepna` cannot express "drop rows with NaN Close". Post-hoc dropping is REQUIRED — it is not a workaround for a missing flag, it is the only available mechanism.**

---

## Internal code inventory

### Installed versions (measured, `.venv`, 2026-07-25)

```
python 3.14.4 | fastapi 0.135.2 | starlette 1.0.0 | pydantic 2.12.5
numpy 2.4.4 | pandas 3.0.1 | yfinance 1.2.0
```
Command: `python -c "import fastapi, starlette, pydantic, numpy, pandas, yfinance ..."`.
All six pins in the caller's prompt are CONFIRMED.

### Local empirics (run in `.venv`, verbatim output)

| Probe | Result |
|---|---|
| `isinstance(np.float64('nan'), float)` | `True` |
| `type(round(np.float64('nan'), 2))` | `numpy.float64` |
| `jsonable_encoder({"1mo": np.float64('nan'), "inf": float('inf')})` | `{'1mo': np.float64(nan), 'inf': inf}` — **NOT sanitised** |
| `json.dumps(enc, allow_nan=False)` | `ValueError: Out of range float values are not JSON compliant: np.float64(nan)` |
| `json.dumps(enc, allow_nan=True)` | `{"1mo": NaN, "inf": Infinity}` — **invalid JSON** |
| `TypeAdapter(dict).dump_json({"a": nan, "b": inf})` | `b'{"a":null,"b":null}'` — pydantic default IS `null` |
| `pydantic_core.to_json({"a": nan})` | `b'{"a":NaN}'` — low-level default is `constants`, NOT `null` |
| `bool(float('nan'))` | `True` |
| `float('nan') is not None` | `True` |
| `float('nan') > 1` / `< 1` | `False` / `False` |

### File-by-file (all line numbers READ, not assumed)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/api/signals.py` | 53-116 | `GET /api/signals/{ticker}` — the 500ing route | CONFIRMED |
| `backend/api/signals.py` | 58-66 | `_safe()` wrapper | CONFIRMED at the exact lines the caller cited |
| `backend/api/signals.py` | 86-99 | `asyncio.gather` of 12 signals | caller said `:98`; `:98` is the `quant_model` element, gather starts `:86` |
| `backend/api/signals.py` | 101-116 | plain `dict` return (no `response_model`, no explicit `JSONResponse`) | CONFIRMED |
| `backend/tools/sector_analysis.py` | 28-36 | `_compute_return` | CONFIRMED — `:34` is the return-arithmetic line |
| `backend/tools/sector_analysis.py` | 55, 71, 79, 88 | `if ret is not None` used as numeric-validity test | ALL FOUR CONFIRMED at the exact cited lines |
| `backend/tools/sector_analysis.py` | 74 | `spy = yf.Ticker("SPY")` fetched for EVERY ticker | CONFIRMED |
| `backend/tools/sector_analysis.py` | 136-153 | NaN comparison cascade → `signal` stays `NEUTRAL` | NEW FINDING (see 80.27 note) |
| `backend/tools/quant_model.py` | 53-117 | `_build_live_features` | function starts `:53`; the un-dropna'd line is `:63` |
| `backend/tools/quant_model.py` | 63 | `closes = hist["Close"].to_numpy(dtype=float)` — **no `.dropna()`** | CONFIRMED |
| `backend/tools/quant_model.py` | 70, 76, 82-85 | NaN propagation into momentum / vol / SMA distance | CONFIRMED at the exact cited lines |
| `backend/tools/quant_model.py` | 84-85 | `... if sma50 else 0.0` — guard defeated by `bool(nan) == True` | CONFIRMED |
| `backend/tools/anomaly_detector.py` | 66-69 | `hist["Close"].dropna().values` (+ Volume/High/Low) | CONFIRMED — in-repo precedent |
| `backend/tools/monte_carlo.py` | 40 | `close = hist["Close"].dropna().values` | CONFIRMED — in-repo precedent |
| `backend/slack_bot/formatters.py` | 660-663 | existing non-finite guard, but coerces to **`0.0`** | CONFIRMED — do NOT copy this idiom into the API (violates criterion 3) |
| (repo-wide) | — | **No recursive JSON sanitiser helper exists in `backend/`** | CONFIRMED by grep (see membership rule below) |

### Confirmation: `_safe()` cannot catch this (read, not assumed)

`backend/api/signals.py:58-66`:

```python
async def _safe(coro_or_func, label, *args):
    try:
        if asyncio.iscoroutinefunction(coro_or_func):
            return await coro_or_func(*args)
        else:
            return await asyncio.to_thread(coro_or_func, *args)
    except Exception as e:
        logger.warning("Signal %s failed for %s: %s", label, ticker, e)
        return {"signal": "ERROR", "summary": str(e)}
```

The `try` covers **only the await**. The tool returns a perfectly good `dict`
(no exception raised — NaN arithmetic never raises), `_safe` returns it, the
route returns the assembled dict at `:101-116`, and the `ValueError` is raised
LATER inside `starlette/responses.py:194-201` during ASGI response rendering,
long after `_safe`'s frame is gone. **Caller's claim CONFIRMED.**

Same reason `sector_analysis.py`'s own `except Exception` at `:180` and
`quant_model.py`'s at `:257` do not fire.

### The two sibling tools — what they do differently

Both consume the SAME `yf.Ticker(t).history(period=...)` frame:

- `anomaly_detector.py:66-69` — `hist["Close"].dropna().values` (and `.dropna()`
  on Volume/High/Low separately, `:67-69`).
- `monte_carlo.py:40` — `close = hist["Close"].dropna().values`.

`quant_model.py:63` — `hist["Close"].to_numpy(dtype=float)`, **no `.dropna()`**.
That single missing call is the entire difference; `:64` `hist["Volume"]` is
likewise un-dropped but the placeholder row's Volume is a real int, so it does
not produce NaN there.

**Caveat for the fix (measured trap):** `anomaly_detector` drops each column
INDEPENDENTLY (`:66-69`), so if a row is partially-NaN the four arrays end up
different lengths and mis-aligned by index. That is a real latent defect but it
is **already queued as masterplan step 80.31** — do not fix it here. The correct
idiom for 80.1's fix is a row-wise `hist.dropna(subset=["Close"])` (or
`hist = hist[hist["Close"].notna()]`) which keeps columns aligned.

### Membership rule for the NaN-leak sweep (stated, not asserted)

I need to say how the candidate set was DERIVED, per `feedback_measure_dont_assert_claims`.

**Rule:** a call site is a candidate iff ALL of:
1. it lives under `backend/tools/` or `backend/api/` (the request path that
   reaches a Starlette `JSONResponse`), AND
2. it produces a `float`/`np.float64` from an arithmetic or aggregation op
   (`pct_change`, `mean`, `std`, division, `np.log`, `np.diff`, `round`), AND
3. that value can reach a dict/list that is `return`ed from an API route
   (directly or via a tool's return dict), AND
4. there is no `dropna`/`isfinite`/`isnan` guard between (2) and (3).

Non-candidates by this rule: anything under `backend/backtest/`,
`backend/metrics/`, `scripts/` that only writes to disk/BQ (BQ tolerates NaN
via the REST float encoding path and never hits `allow_nan=False`), and
`backend/slack_bot/` (already guarded at `formatters.py:663`).

### Sweep results (rule applied; "verified how" column is the honest part)

Grep executed: `grep -rn "pct_change\|\.mean()\|\.std()\|np\.mean\|np\.std\|np\.log\|np\.diff\|\.rolling(" --include="*.py" backend/tools/ backend/api/` plus a per-tool
`grep -n "np\.mean\|np\.std\|/ \|sum(\|round("` over the 12 tools in the `signals.py` gather.

| Site | Verdict | Verified how |
|---|---|---|
| `sector_analysis.py:34` | **CONFIRMED LEAK** | read; matches the reported traceback (`stock_returns` / `1mo`) |
| `quant_model.py:63` → `:70,:76,:82-85` | **CONFIRMED LEAK** | read; `bool(nan)` trap measured |
| `anomaly_detector.py:66-69,74-163` | Guarded (`dropna`) | read |
| `monte_carlo.py:40-43,76-100` | Guarded (`dropna`) + `len()` guards | read |
| `sec_insider.py:243` | Safe — `/ max(len(sells), 1)` | read |
| `options_flow.py:57,73,76,77` | Safe — `/ max(oi, 1)`, `/ max(total_*, 1)`, `.fillna(0)` | read |
| `social_sentiment.py:33,88,93-94,102,149,153-154,161` | Safe — every divisor has `if total == 0` / `if all_scores else 0` | read |
| `patent_tracker.py:124,132` | Safe — `if total else 0`, `if prev > 0` | read |
| `earnings_tone.py:345` | Safe — `sum()` of ints only | read |
| `fred_data.py` | Safe — no arithmetic aggregation matched | grep (0 hits) |
| `alt_data.py:127-130` | Safe — `if recent else 0`, `/ max(older_avg, 1)` | read |
| `nlp_sentiment.py:54` | Safe — explicit `if norm_a == 0 or norm_b == 0: return 0.0` | read |
| `nlp_sentiment.py:161` `np.mean([])` when `article_scores` is empty | **Latent but neutralised** | read + measured: `max(-1.0, min(1.0, nan))` returns `1.0` (Python `min`/`max` return the first arg when the comparison is False), so NaN never escapes `:163`. It becomes a **silently wrong `1.0`** — a correctness smell, not a 500. Out of scope for 80.1; worth its own step. |
| `screener.py:176-213, 626, 642-643` | **NOT in the signals response path** — consumed by the trading loop, not returned by an API route | rule criterion 3 fails; excluded |
| `price_quality.py:98,106` | **NOT in the signals response path** | rule criterion 3 fails; excluded |
| `backend/api/*.py` | Zero hits for the arithmetic patterns — the API layer does no numeric aggregation of its own | grep (0 hits) |
| **`backend/tasks/analysis.py` → analysis-report poll route** | **HIGH-PRIORITY CANDIDATE, NOT VERIFIED** | The Layer-1 report embeds the SAME `quant_model` dict (`analysis.py:176`, `orchestrator.py:2014`). If any route returns that report verbatim it can 500 identically. I did **not** execute a request to confirm. Flagging, not asserting. |

**Recall test on the rule:** the rule's step 3 ("can reach a dict returned from an API route") is what excludes `screener.py` and `price_quality.py`. If a future step adds an endpoint that returns screener output, both re-enter the candidate set. Stating that dependency so the scope is auditable rather than assumed.

---

## Application to pyfinagent

### Recommended fix shape (display-only, zero new dependencies)

Two layers, and the **order matters for the do-no-harm argument**:

**Layer A — the boundary sanitiser (this is the whole of 80.1).**
A recursive `dict`/`list`/`tuple`/scalar walker that maps non-finite floats to `None`,
applied to the response of `get_all_signals` (and, for symmetry, the sibling
sub-routes that 500 today — at minimum `/{ticker}/quant-model` and `/{ticker}/sector`).

- Predicate: `isinstance(v, float) and not math.isfinite(v)` → `None`
  (see Key finding 8 for the measured traps that rule out `np.isfinite` and `pd.isna`).
- Must recurse into nested dicts AND lists (`quant_model.data.features` is a nested
  dict; `top_factors` is a list of dicts).
- Must **preserve the key** (criterion 3: "NOT dropped silently"). A `{k: v for ... if isfinite}` comprehension would pass a naive "no 500" test and fail criterion 3 — do not write it that way.
- **No existing helper to reuse** — grep confirmed none in `backend/`.
  `backend/slack_bot/formatters.py:663` is the only precedent and it returns `0.0`;
  **do not copy it**, it violates criterion 3.
- Placement: put it in the API layer (e.g. a small helper in `backend/api/signals.py`
  or a shared `backend/api/_json_safe.py`), NOT in `backend/tools/`. See the Risk section — this is the whole safety argument.

**Layer B — the upstream `dropna` at `quant_model.py:63` (JUDGEMENT CALL — read the Risk section before doing this).**
`closes = hist["Close"].dropna().to_numpy(dtype=float)` — or better, a row-wise
`hist = hist.dropna(subset=["Close"])` before `:63` so `closes` and `volumes` stay
index-aligned (avoids re-creating the 80.31 mis-alignment defect).
This restores parity with `anomaly_detector.py:66` / `monte_carlo.py:40`.
**This layer changes numbers the trading pipeline consumes.** See Risk.

### Test design (criterion 4 + the caller's additional criterion)

- Use `backend/tests/auth_helper.py::authed_test_client(app, **kwargs)` — **confirmed
  it fits**: the module docstring names `/api/signals` explicitly as one of the
  prefixes that lost `_PUBLIC_PATHS` coverage in phase-75.1, and it is already used by
  `test_phase_80_2_error_response_contract.py` (the sibling 80.2 suite) and
  `backend/tests/api/test_sovereign.py`. `DEV_LOCALHOST_BYPASS` does NOT work for
  TestClient (`request.client.host == "testclient"`).
- **Deterministic NaN fixture with no network:** monkeypatch the 12 tool functions that
  `signals.py` imports (they are module-level attributes on `backend.api.signals`'s
  imported modules — e.g. `monkeypatch.setattr(sector_analysis, "get_sector_analysis", lambda t: {...})`),
  plus `yfinance.Ticker` for the `info`/`news` calls at `signals.py:71,81`. Have the
  fake `sector` tool return `{"stock_returns": {"1mo": float("nan")}, ...}`. This
  reproduces the exact reported failure (`dict item "1mo"` → `"stock_returns"` → `"sector"`)
  without touching yfinance.
  A lower-level unit test on the sanitiser helper alone is also worth having, but on its
  own it does **not** satisfy criterion 1 (which names the HTTP status of the endpoint).
- **Assertions (all three, or the test is weak):**
  1. `resp.status_code == 200`
  2. all 12 keys present (criterion 2)
  3. `body["sector"]["stock_returns"]["1mo"] is None` **and** `"1mo" in body["sector"]["stock_returns"]`
     — the key must be PRESENT and the value must be `None`. This is the assertion the
     caller's additional criterion demands: it fails if NaN were replaced by `0.0`
     **and** it fails if the key were dropped.
- **Mutation matrix (per `feedback_mutation_test_guards_and_fixtures` — mutate the FIXTURE too):**
  | Mutation | Expected |
  |---|---|
  | Remove/revert the sanitiser entirely | test FAILS (500) |
  | Sanitiser returns `0.0` instead of `None` | test FAILS (assertion 3) |
  | Sanitiser drops the key instead of nulling it | test FAILS (assertion 3, `in` check) |
  | **Fixture mutated to return a finite float** instead of NaN | test must FAIL — otherwise the fixture cannot represent the failure and the guard is vacuous |
  Run all four. The fourth is the one this project has shipped broken five times.

### Note for step 80.27 only (do not solve here)

`.claude/rules/backend-tools.md` specifies the failed-tool contract verbatim:

> ## Tool Contract
> All tools in `backend/tools/` return a consistent structure:
> ```json
> { "ticker": "AAPL", "signal": "BULLISH", "summary": "...", "data": { ... } }
> ```

and

> - Error returns: `{ "ticker": "...", "signal": "ERROR", "summary": "...", "data": {} }`

So the documented shape for a failed tool is `signal: "ERROR"` — which is precisely
what a NaN-poisoned tool should be emitting and is not. Recorded for 80.27; **out of
scope for 80.1**.

---

## RISK / DO-NO-HARM — the 80.1 vs 80.27 boundary

### The shared-code question, answered with anchors

**They share code. The shared functions are the two leak sites themselves.**

| Consumer | Call site | Function |
|---|---|---|
| Signals API (Layer 0, display) | `backend/api/signals.py:94` | `sector_analysis.get_sector_analysis` |
| Signals API (Layer 0, display) | `backend/api/signals.py:98` | `quant_model.get_quant_model_signal` |
| **Layer-1 analysis pipeline (feeds trading)** | `backend/agents/orchestrator.py:1261` | `sector_analysis.get_sector_analysis` |
| **Layer-1 analysis pipeline (feeds trading)** | `backend/agents/orchestrator.py:1271-1273` (`fetch_quant_model`) | `quant_model.get_quant_model_signal` |

Downstream of the orchestrator call: `orchestrator.py:1992` (`_safe(self.fetch_quant_model, ...)`),
`:2014` (assembled into the report), `:2036`/`:2060` (re-fetch path), then
`backend/tasks/analysis.py:176` (`qm_raw = report.get("quant_model", {})`) and
`:297-298` (`quant_model_signal=...`, `quant_model_score=...` persisted to BigQuery).

**So the rule is simple and it is the whole safety argument:**

| Change location | Blast radius | Verdict for 80.1 |
|---|---|---|
| Sanitiser in `backend/api/` (Layer A) | HTTP response bytes only. The orchestrator never goes through Starlette's `JSONResponse`. **Cannot change a trading decision.** | **DISPLAY-ONLY — safe, in scope** |
| `dropna` inside `backend/tools/quant_model.py` (Layer B) | Changes the FEATURE VALUES the Layer-1 pipeline scores on, hence `quant_model.score`, hence the Gemini quant-model agent's prompt, hence the synthesis verdict, hence what the paper-trading loop acts on. | **CHANGES TRADING INPUTS — belongs to 80.27** |

### Why the caller's instinct is right, and where I'd push back

The caller's stated worry — *"sanitising NaN → null at the response boundary turns the
500 green while leaving the trading path poisoned AND hiding the symptom that made it
visible"* — is correct on both halves, and I found the exact laundering mechanisms:

- **`quant_model.py`:** `_score_ticker` returns `nan` (weighted_sum is NaN), then
  `_classify_signal(nan)` at `:171-181` evaluates `nan > 0.08` → `False`,
  `nan > 0.03` → `False`, `nan < -0.08` → `False`, `nan < -0.03` → `False`
  → falls through to **`return "NEUTRAL"`**. Measured: `float('nan') > 1` and
  `float('nan') < 1` are both `False`.
- **`sector_analysis.py:136-153`:** `sec_3m`/`spy_3m`/`stock_3m` are NaN, so
  `sec_3m > spy_3m` → `False` and `stock_3m > sec_3m` → `False`
  → `sector_tailwind = False`, `stock_outperforming = False`, and the `elif` at `:152`
  is also `False` → **`signal` stays `"NEUTRAL"`**.
- **`backend/config/prompts.py:1111`** does `json.dumps(quant_model_data, indent=2)`
  with the **stdlib default `allow_nan=True`** — so the Gemini agent is being handed a
  prompt containing the literal token `NaN` inside its JSON block. That is a third,
  separate laundering surface for 80.27.

That is the 80.27 defect in full, and **none of it is touched by Layer A.**

**Where I'd push back slightly on the framing:** the 500 is not currently acting as a
useful alarm for the trading path, because the trading path never renders JSON and so
never 500s. The endpoint has been down while the pipeline has been quietly producing
NEUTRALs. Fixing the display does not *remove* an alarm that was protecting trading —
it removes an alarm that was only ever protecting the UI. The real mitigation is that
80.27 is already queued as a separate P0. **The concrete way to avoid "green means
fixed" is to make the null VISIBLE**: the frontend should render a null return as a
"data unavailable" state, not as a blank or a zero. Worth stating in the contract as a
non-blocking expectation.

### Explicit scope recommendation

**Do Layer A only in 80.1.** Reasons:
1. It fully satisfies all four immutable criteria plus the caller's additional criterion.
2. It is provably incapable of moving the paper-trading book (the orchestrator does not
   traverse Starlette).
3. It leaves 80.27's evidence intact — a Q/A or operator investigating 80.27 will still
   see NaN in the tool outputs and `NaN` in the LLM prompt; the sanitiser is at the far
   edge of the pipe.

**If Layer B is done anyway** (it is a genuinely correct one-word fix and the temptation
is real), then: (a) it must be called out as a trading-input change in
`experiment_results.md`, not smuggled in under "fixed a 500"; (b) use the row-wise
`hist.dropna(subset=["Close"])` form so `closes`/`volumes` stay aligned — the
column-independent `dropna()` used at `anomaly_detector.py:66-69` is the very defect
queued as **80.31**; (c) it will change `quant_model.score` for live tickers on the next
analysis cycle. My recommendation is to **defer it to 80.27** where it can be evaluated
against trading criteria rather than a 200-vs-500 criterion.

### Live-book safety checklist for whoever implements this

- Do **not** edit anything under `backend/tools/` for Layer A.
- Do **not** edit `backend/agents/orchestrator.py`, `backend/tasks/analysis.py`, or
  `backend/config/prompts.py` — all three are on the trading path.
- Do **not** set `default_response_class` app-wide (would silently re-encode ~70 routes).
- Do **not** flip `allow_nan=True` (produces invalid JSON; breaks the browser instead).
- The `_safe()` fallback dict at `signals.py:66` returns `{"signal": "ERROR", ...}` and
  should be left alone — it is the documented tool-failure shape and 80.27 depends on it.

---

## Research Gate Checklist

**Hard blockers**
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **7** (RFC 8259, Pydantic config docs, orjson README, MDN, FastAPI custom-response docs, fastapi#8029, fastapi#11821) plus the Jan-2026 write-up = 8. Conservative count reported below excludes the yfinance#413 fetch that returned no evidence.
- [x] 10+ unique URLs total — 22
- [x] Recency scan (2024-2026) performed + reported — 3 in-window findings, all decision-relevant
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim — every cited line was READ, and the two anchors the caller flagged as possibly drifted (`signals.py:98`, `quant_model.py:63`) are annotated with what is actually at that line

**Soft checks**
- [x] Internal exploration covered every relevant module (12 signal tools + API layer + orchestrator consumers)
- [x] Contradictions noted — fastapi#8029's community answer (ORJSONResponse) is contradicted in practice by FastAPI's own docs steering toward response models, which fastapi#11821 then shows does not work; recorded rather than smoothed over
- [x] All claims cited per-claim
- [ ] **Gap I am flagging rather than hiding:** the analysis-report poll route was NOT verified end-to-end (marked HIGH-PRIORITY CANDIDATE, NOT VERIFIED in the sweep table). I did not issue an HTTP request against it.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 15,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "GET /api/signals/{ticker} 500s because Starlette 1.0.0 JSONResponse.render hardcodes json.dumps(allow_nan=False) (responses.py:194-201) and FastAPI's jsonable_encoder passes np.float64 NaN through untouched (measured). _safe() at signals.py:58-66 provably cannot catch it -- the ValueError is raised during ASGI rendering, after the route returns. Both leak sites confirmed at the cited lines: sector_analysis.py:34 (with `is not None` misused as a numeric-validity test at :55/:71/:79/:88, and SPY fetched for every ticker at :74) and quant_model.py:63 (missing .dropna(), unlike siblings anomaly_detector.py:66 and monte_carlo.py:40). yfinance has NO flag to suppress the forming-session row: keepna=False is already the default and its mask at scrapers/history.py:495-499 uses .all(axis=1), which the placeholder row escapes because Volume is real -- post-hoc dropping is the only mechanism. Recommend a recursive NaN->None sanitiser at the API boundary (zero new deps); reject allow_nan=True (invalid JSON), reject pydantic response_model (FastAPI ignores ser_json_inf_nan -- fastapi#11821), and reject app-wide ORJSONResponse (adds a dep, re-encodes ~70 routes). CRITICAL: sector_analysis.get_sector_analysis and quant_model.get_quant_model_signal are SHARED with the Layer-1 trading pipeline at orchestrator.py:1261 and :1271-1273 -- so a fix inside backend/tools/ changes trading inputs (80.27), while a fix in backend/api/ cannot. Do Layer A only.",
  "brief_path": "handoff/current/research_brief_80.1.md",
  "gate_passed": true
}
```
