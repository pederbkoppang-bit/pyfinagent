# Research Brief -- masterplan step 82.13

**Step:** 82.13 (P1) -- backtest engine discards `preload_macro()`'s return; 82.0 armed the
staleness gate so the `return 0` refusal path is now REACHABLE for the first time. Fix the
CONSEQUENCE (silent degradation into the per-cutoff BQ slow path), not the likelihood.

**Tier:** moderate. **Audit-class:** false. **Researcher:** Layer-3 (external + internal).
**Started:** 2026-08-05. **Status:** IN PROGRESS (write-first; this file grows incrementally).

---

## 0. Scope restated (one line)

Design a way for `run_backtest`-family code to (a) detect a macro-preload refusal, (b) either
abort loudly or run in an explicitly-labelled macro-free mode, and (c) record data availability
in the run result -- plus an AST-derived enumeration of every `preload_*` call site.

---

## 1. Search queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| Q1 | `CWE-252 unchecked return value Python static analysis must_use errcheck` | **YEAR-LESS canonical** |
| Q2 | `fail fast versus fail safe degraded mode must be labelled data pipeline design` | **YEAR-LESS canonical** |
| Q3 | `silent data quality degradation machine learning pipeline detection 2025 2026` | last-2-year window |
| Q4 | `machine learning experiment metadata record feature availability degraded run provenance 2026` | current-year frontier |

All three variants present. The year-less passes are what surfaced CWE-252 (1990s-era prior art)
and the fail-fast/graceful-degradation body of work -- a 2026-locked query returns only
blog-tier "data observability" vendor content.

---

## 2. Sources READ IN FULL via WebFetch (6 -- counts toward the gate)

| # | URL | Tier | Accessed | What it establishes |
|---|-----|------|----------|---------------------|
| S1 | https://cwe.mitre.org/data/definitions/252.html | Official standard (MITRE) | 2026-08-05 | CWE-252 *Unchecked Return Value*. Verbatim: "The product does not check the return value from a method or function, which can prevent it from detecting unexpected states and conditions." Root cause is two assumptions -- "this function call can never fail" and "it doesn't matter if this function call fails." Mitigation, verbatim: "Check the results of all functions that return a value and verify that the value is expected"; "Ensure that you account for all possible return values from the function"; and, design-side, "When designing a function, make sure you return a value or throw an exception in case of an error." Class is **Not Language-Specific**; parent CWE-754; member of CWE-389 (Error Conditions, Return Values, Status Codes). Detection by SAST rated effectiveness **High** -- for languages where the tooling exists. |
| S2 | https://sre.google/sre-book/addressing-cascading-failures/ | Official docs (Google SRE Book) | 2026-08-05 | Distinguishes the two options this step chooses between. Verbatim: "Graceful degradation takes the concept of load shedding one step further by reducing the amount of work that needs to be performed" (e.g. "search a subset of data ... or use a less-accurate ranking algorithm"), vs the fail-fast rule "When overloaded at either the frontend or backend layers, fail early and cheaply." Crucially for criterion 2, the degraded mode is required to be **observable**: "Monitor and alert when too many servers enter these modes." **Honest caveat:** the chapter does NOT state that a degraded result must be labelled *in its own output* -- it mandates operator-side monitoring only. The in-result-labelling argument comes from S3/S4/S6, not from SRE. |
| S3 | https://ar5iv.labs.arxiv.org/html/1810.03993 | Peer-reviewed (Mitchell et al., FAT* 2019) | 2026-08-05 | *Model Cards for Model Reporting*. The canonical argument that a performance number is meaningless without the conditions that produced it: "there are no standardized documentation procedures to communicate the performance characteristics of trained machine learning (ML) and artificial intelligence (AI) models," and "systematic errors were only exposed after models were put into use." Prescribes explicit **Training Data / Evaluation Data / Factors / Caveats and Recommendations** sections and advocates performance "to be broken down by ... domain-relevant conditions." **Honest caveat:** it does not literally forbid pooling runs from different data conditions; it argues for disaggregation, which is the weaker-but-sufficient form of the claim. |
| S4 | https://ar5iv.labs.arxiv.org/html/1803.09010 | Peer-reviewed (Gebru et al., CACM 2021) | 2026-08-05 | *Datasheets for Datasets*. Direct support for criterion 2, verbatim: "**Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable).**" And on the consumer side: "For dataset consumers, the primary objective is to ensure they have the information they need to make informed decisions about using a dataset." The electronics analogy -- "we propose that every dataset be accompanied with a datasheet that documents its motivation, composition, collection process, recommended uses" -- is exactly the shape of a `data_availability` block on a BacktestResult. |
| S5 | https://doc.rust-lang.org/reference/attributes/diagnostics.html | Official docs (Rust Reference) | 2026-08-05 | `#[must_use]`: "The `must_use` attribute marks a value that should be used." When "the expression of an expression statement is a call expression ... whose function operand is a function to which the attribute is applied, the use triggers the `unused_must_use` lint." Opt-out is explicit: "Using a `let` statement or destructuring assignment with a pattern of `_` when a must-used value is purposely discarded is idiomatic." This is precisely the machinery Python lacks -- and it also gives the *shape* of the mitigation: make discarding **explicit and visible** at the call site. |
| S6 | https://mlflow.org/docs/latest/ml/tracking/ | Official docs (MLflow) | 2026-08-05 | Experiment-tracking prior art for recording data context on a run. A run "records metadata (various information about your run such as metrics, parameters, start and end times) and artifacts". Dataset context is first-class: "MLflow offers the ability to track datasets that are associated with model training events ... stored through the use of the `mlflow.log_input()` API," enabling users to "Filter metrics based on specific datasets for fair model comparison." I.e. the industry-standard answer to "don't compare a degraded run with a normal one" is *tag the run with its data context*, not *refuse to run*. |

**Failed fetches (attempted, counted as URLs collected, NOT as read-in-full):**
`https://best.openssf.org/Secure-Coding-Guide-for-Python/CWE-703/CWE-252/` -> HTTP 404;
`https://doc.rust-lang.org/std/attr.must_use.html` -> HTTP 404 (recovered via the Rust
Reference, S5).

---

## 3. Snippet-only sources (context; does NOT count toward gate)

| URL | Tier | Why not read in full |
|-----|------|----------------------|
| https://arxiv.org/abs/2506.06147 (Stream DaQ, 2025-06-06, cs.DB) | Preprint | Only the **abstract page** was fetched; per `.claude/rules/research-gate.md` an abstract-only fetch may NOT be counted as read-in-full. Establishes the 2025 framing that quality issues "propagate silently through continuous pipelines" and proposes "quality meta-streams for real-time pipeline awareness". |
| https://best.openssf.org/Secure-Coding-Guide-for-Python/CWE-703/CWE-252/ | Official docs | 404 on fetch; search snippet establishes the Python-specific framing: return values matter "when they may be used as an alternative to raising exceptions, such as with `str.find()`, which returns -1 instead of raising a ValueError". That is *exactly* `preload_macro`'s `return 0`. |
| https://doc.rust-lang.org/std/attr.must_use.html | Official docs | 404; superseded by S5. |
| https://docs.veracode.com/updates/r/c_all_static | Vendor docs | SAST coverage note only. |
| https://www.mathworks.com/help/bugfinder/ref/cwe252.html | Vendor docs | C/C++-scoped. |
| https://vulnerabilityhistory.org/tags/cwe-252 | Community | Case index, no new normative content. |
| https://designgurus.substack.com/p/when-to-fail-fast-vs-degrade-gracefully | Community/blog | Low tier; the SRE Book (S2) covers the same ground authoritatively. |
| https://www.databricks.com/blog/data-pipeline-best-practices | Vendor blog | Generic. |
| https://ijcttjournal.org/2025/Volume-73%20Issue-4/IJCTT-V73I4P120.pdf | Low-tier journal | Error-handling survey; not load-bearing. |
| https://www.dqlabs.ai/blog/data-pipeline-monitoring-and-anomaly-detection/ | Vendor blog | Recency-scan evidence only. |
| https://arxiv.org/pdf/2602.06594 | Preprint | ML practitioners' data-quality views under EU regulation; adjacent, not on-point. |
| https://arxiv.org/pdf/2312.06254 (Modyn) | Preprint | Pipeline orchestration; adjacent. |
| https://dl.acm.org/doi/abs/10.1145/3595360.3595859 (MLflow2PROV) | Peer-reviewed | Provenance extraction; paywalled abstract. |
| https://www.sciencedirect.com/science/article/pii/S0306437924001534 | Peer-reviewed | End-to-end ML provenance; paywalled. |
| https://www.amazon.science/publications/automatically-tracking-metadata-and-provenance-of-machine-learning-experiments | Industry research | Establishes the "artifact metadata store" pattern; MLflow (S6) is the concrete instance. |
| https://arxiv.org/pdf/2507.01078 (yProv4ML) | Preprint | 2025 provenance-tracking library. |
| https://arxiv.org/pdf/2210.11831 | Preprint (survey) | ML lifecycle artifact management survey. |
| https://ckaestne.medium.com/versioning-provenance-and-reproducibility-in-production-machine-learning-355c48665005 | Authoritative blog (CMU faculty) | Course-notes level restatement of S3/S4/S6. |

**URLs collected: 24** (6 read in full + 18 snippet-only).

---

## 4. Recency scan (last 2 years, 2024-2026) -- PERFORMED

Queries Q3 and Q4 above were scoped to 2025/2026. Result: **the 2024-2026 window produced
NO finding that supersedes the canonical sources; it produced 2 findings that COMPLEMENT them,
and 1 negative finding that matters for this step.**

1. **Complementary (2025).** *Stream DaQ* (arXiv:2506.06147, 2025-06-06) frames the problem in
   the same terms this step uses -- quality issues that "propagate silently through continuous
   pipelines feeding analytics and AI models" -- and its remedy is a **quality meta-stream**
   emitted alongside the data. That is the streaming analogue of a `data_availability` block on
   a batch result. It confirms the 2018-2019 model-card/datasheet argument is still the live
   consensus, restated for pipelines.
2. **Complementary (2024-2026).** The provenance literature (MLflow2PROV, yProv4ML 2025,
   ScienceDirect 2024 end-to-end ML provenance) is converging on the criticism that
   "ML artifact management systems like MLflow still have rudimentary provenance capabilities
   ... data preprocessing and feature transformation steps are often not reflected in
   provenance." Relevance: recording *which features were available* is precisely the gap the
   2024-2026 work says tooling still misses -- so pyfinagent doing it in-result is defensible,
   not over-engineering.
3. **Negative finding (matters).** No 2024-2026 source found any change to the Python
   ecosystem's ability to statically catch a discarded return. There is still **no
   `[[nodiscard]]` / `#[must_use]` equivalent in Python**: PEP 484/mypy/pyright do not warn on a
   discarded non-`None` return, and `ruff`/`flake8` have no general unchecked-return rule
   (only narrow cases like `B015` useless-comparison). Go's `errcheck` and Rust's
   `#[must_use]` (S5) have no counterpart. **Consequence for the contract: the guard must be a
   TEST, not a linter.** That is the single most decision-relevant recency result.

---

## 5. Internal findings (every file:line RE-DERIVED 2026-08-05)

### 5.1 The three preload calls -- CONFIRMED at :315/:316/:317

Measured: `Read backend/backtest/backtest_engine.py` offset 255 limit 110 (2026-08-05).

```
315  cache.preload_prices(universe_tickers + [_benchmark], global_start, global_end)
316  cache.preload_fundamentals(universe_tickers)
317  cache.preload_macro()
```

- **Enclosing function:** `BacktestEngine.run_backtest(self, universe_tickers=None,
  skip_cache_clear=False) -> BacktestResult`, defined at `backend/backtest/backtest_engine.py:275`.
  Docstring at :280-288.
- **Control flow before the preloads:** `:289` universe default -> `:295`
  `self._auto_ingest_if_needed(universe_tickers)` -> `:299` `self.trader.full_reset()` ->
  `:301` `windows = self.scheduler.generate_windows()` -> `:307`
  `self._report_progress("preloading", ...)` -> `:308-314` date math + benchmark lookup ->
  `:315-317` the three preloads.
- **Can it raise?** YES, and it already does: `preload_macro` opens with
  `assert _bq_client is not None` (`cache.py:339`), and the BQ call at `cache.py:352` is
  **not** wrapped in try/except (contrast `cached_macro`'s fallback query at `cache.py:633-637`,
  which IS wrapped). So `run_backtest` is already a function that propagates exceptions out of
  the preloading step. **There is no try/except around :315-317.** An explicit `raise` at :317
  is therefore consistent with the existing contract, not a new failure mode.
- **Progress hook available at that point:** YES --
  `self._report_progress(step, detail="", **kwargs)` is defined at
  `backend/backtest/backtest_engine.py:1103` and is already called for the `"preloading"` step at
  `:307`. A refusal can be surfaced through it (e.g. a second `_report_progress("preloading",
  "macro unavailable -- running macro-free", macro_available=False)`), because it accepts
  arbitrary `**kwargs` that flow into the emitted dict.
- **Per-window failures are already swallowed:** `:334-346` wraps `self._run_window(...)` in
  `try/except Exception` and only `logger.error`s. That is the same silent-degradation idiom
  this step is fixing, one level down -- worth noting so the fix is not undone by that handler
  (it is OUTSIDE the preload block, so an exception raised at :317 is NOT caught by it).

### 5.2 `preload_macro` return paths -- EXACT list (the naive `if not preload_macro()` trap)

Measured: `Read backend/backtest/cache.py` offset 320 limit 160 (2026-08-05). Signature at
`cache.py:333`, `def preload_macro() -> int`.

| # | Line | Return | Meaning | Is it a refusal? |
|---|------|--------|---------|------------------|
| R1 | `cache.py:345` | `return total` (a **POSITIVE** int, `sum(len(v) for v in _macro_full.values())`) | ALREADY WARM -- early return, skips BQ entirely, `_macro_full` is populated | NO -- macro IS available |
| R2 | `cache.py:356` | `return 0` | `preload_macro: 0 rows returned` -- the BQ table is EMPTY | NO data (not a refusal, but macro is unavailable) |
| R3 | `cache.py:418` | `return 0` | **REFUSAL** -- "could not parse a usable date from any of N rows"; fail-closed on an unevaluable date column | YES |
| R4 | `cache.py:435` | `return 0` | **REFUSAL** -- "stale data, refusing to cache -- N of M series past their per-series SLA" (this is the 82.0-armed gate) | YES |
| R5 | `cache.py:461` | `return total_rows` (positive) | SUCCESS -- `_macro_full` populated | NO |
| E1 | `cache.py:339` | `AssertionError` | cache not initialised | (raises, not returns) |
| E2 | `cache.py:352` | propagates BQ exception | query failure / 60s timeout -- **NOT** caught | (raises, not returns) |

**CRITICAL, exactly as the caller suspected:** a positive return means "macro is in
`_macro_full`", including R1 where **zero new rows were loaded**. A `0` return conflates R2
("no data") with R3/R4 ("refused"). So:

- `if not preload_macro(): abort` -- CORRECT on the availability question by luck (R1 returns
  positive, R2/R3/R4 all mean `_macro_full` is empty) but WRONG on the diagnostic: it cannot
  tell "table empty" from "gate refused", and a future R-path that returns 0 while still
  populating the cache would break it silently.
- The ONLY invariant that actually holds for every path is: **`_macro_full` is non-empty iff
  macro is available.** R1/R5 populate it; R2/R3/R4 leave it empty. That is the honest predicate.
  Recommend a dedicated accessor in `cache.py` (e.g. `macro_is_loaded() -> bool` /
  `macro_status() -> dict`) rather than re-deriving availability from an `int`. See §6.

### 5.3 The consequence path -- CONFIRMED, and the ~40-minute figure is FOLKLORE

Measured: `Read backend/backtest/cache.py` offset 560 limit 90 + `grep -rn "cached_macro"
--include="*.py" .` (2026-08-05).

- `cached_macro(cutoff_date)` at `cache.py:569`. Fast path `if _macro_full:` at `cache.py:576`,
  returns from memory (`:601`). **Fallback at `cache.py:603-641`**: `_cache_stats["misses"] += 1`,
  then a parameterised per-cutoff BQ query (`:618-628`) with `.result(timeout=30)` at `:634`.
  Confirmed: **empty `_macro_full` => one BQ query per distinct cutoff_date**.
- Memoised per cutoff by `_macro_cache` (`:571-573`, `:640`) -- so it is one query per DISTINCT
  cutoff, not per call.
- **The driver:** `backend/backtest/historical_data.py:48` -> `return cache.cached_macro(cutoff_date)`.
  Other non-backtest caller: `backend/agents/mcp_servers/data_server.py:185`.
- **The ~40-minute number: NOT MEASURED ANYWHERE IN THE REPO.** It appears as prose in
  `CLAUDE.md` ("Always call `cache.preload_macro()` or backtests hang after ~40min") and is
  re-quoted in a code comment at `backend/backtest/cache.py:53`. I found no benchmark, test,
  log artefact, or timing harness that produces it. **Do NOT restate it as measured** -- the
  defensible statement is "one uncached BQ round-trip (30s timeout) per distinct cutoff date,
  unbounded by the number of walk-forward decision dates". That is the real, checkable harm.

### 5.4 `preload_*` call-site enumeration (criterion 3) -- AST-derived, 15 sites

Measured by an `ast.walk` over every `.py` file in the repo (script:
`scratchpad/ast_preload.py`; excludes `.venv`, `node_modules`, `.git`, `.next`,
`site-packages`). **967 files parsed, 15 call sites.** Classification = "DISCARDED" iff the
`ast.Call` node's direct parent is an `ast.Expr` (statement-expression => return value dropped).

| File:line | Uses return? | Call |
|---|---|---|
| `backend/backtest/backtest_engine.py:315` | **DISCARDED** | `cache.preload_prices(universe_tickers + [_benchmark], global_start, global_end)` |
| `backend/backtest/backtest_engine.py:316` | **DISCARDED** | `cache.preload_fundamentals(universe_tickers)` |
| `backend/backtest/backtest_engine.py:317` | **DISCARDED** | `cache.preload_macro()` |
| `backend/tests/test_phase_82_0_macro_ingestion.py:212` | USED (Assign) | `cache_mod.preload_macro()` |
| `backend/tests/test_phase_82_0_macro_ingestion.py:239` | USED (Assign) | `cache_mod.preload_macro()` |
| `backend/tests/test_phase_82_0_macro_ingestion.py:266` | USED (Assign) | `cache_mod.preload_macro()` |
| `backend/tests/test_phase_82_0_macro_ingestion.py:286` | USED (Compare) | `cache_mod.preload_macro()` |
| `backend/tests/test_phase_82_12_string_column_guards.py:514` | USED (Return) | `bq_cache.preload_macro()` |
| `backend/tests/test_phase_82_15_macro_point_in_time.py:242` | **DISCARDED** | `cache.preload_macro()` |
| `backend/tests/test_phase_82_15_macro_point_in_time.py:272` | **DISCARDED** | `cache.preload_macro()` |
| `scripts/diag_label_pin.py:25` | **DISCARDED** | `cache.preload_prices(tickers + ['SPY'], '2020-06-01', '2023-12-31')` |
| `scripts/diag_label_pin.py:26` | **DISCARDED** | `cache.preload_fundamentals(tickers)` |
| `scripts/diag_label_pin.py:27` | **DISCARDED** | `cache.preload_macro()` |
| `tests/verify_phase_25_D7.py:100` | USED (Assign) | `cache_mod.preload_macro()` |
| `tests/verify_phase_25_D7.py:130` | USED (Assign) | `cache_mod.preload_macro()` |

Totals: **8 DISCARDED / 7 USED**. Production (non-test) discards: 6 -- the three engine calls
plus all three in `scripts/diag_label_pin.py`. **Note for criterion 3:** the enumeration must be
AST-derived and asserted NON-EMPTY; a plain `grep` cannot distinguish a call from a mention
(e.g. `cache.py:53` and `settings.py:252` mention `cached_macro` in prose, and
`backend/tests/test_phase_75_mcp_truth.py:278` *defines* a shadowing `cached_macro`). The
script above is a working reference implementation for the test.

---

### 5.5 Result metadata (criterion 2) -- where a `data_availability` field must land

Measured: `Read backend/backtest/backtest_engine.py` offset 96 limit 120; `grep -rn
"BacktestResult\|asdict(" --include="*.py" backend/ scripts/`; `sed -n '735,800p'
backend/backtest/analytics.py`; `sed -n '20,50p' backend/backtest/result_store.py`;
`grep -n "generate_report(" ...`.

- **`BacktestResult` is at `backend/backtest/backtest_engine.py:119-132`** (11 fields:
  `windows`, `aggregate_sharpe`, `aggregate_return_pct`, `aggregate_alpha_pct`,
  `aggregate_max_drawdown_pct`, `aggregate_hit_rate`, `total_trades`,
  `feature_importance_mdi`, `feature_importance_mda`, `nav_history`, `strategy_params`,
  `all_trades`). CONFIRMED: **no data-availability field**. Every field has a default, so
  appending one more defaulted field is source-compatible with every existing construction.
- **Constructed at `backtest_engine.py:319`** -- `BacktestResult(strategy_params=self._strategy_params)`
  (keyword-only, so field ORDER is not load-bearing for that call site).
- **Flow:** `BacktestResult` -> `analytics.generate_report(result, num_trials=1,
  baselines=None) -> dict` (`backend/backtest/analytics.py:741`). Report dict top-level keys
  measured: `analytics`, `per_window`, `feature_importance`, `equity_curve`, `nav_history`,
  `strategy_params`, `baselines` (+ `trades`/`trade_statistics` per the backtest rules doc).
- **generate_report has 18 call sites** (`grep -rn "generate_report("`): `api/backtest.py:1058`,
  `quant_optimizer.py:205` + `:267`, `autoresearch/strategy_backtest_adapter.py:162`, and 8
  scripts under `scripts/harness/` + `scripts/ablation/run_ablation.py:170`. All treat the
  return as an opaque dict -- **adding a key breaks none of them.**
- **Persistence is schemaless:** `result_store.save_result(run_id, report: dict)` at
  `backend/backtest/result_store.py:23` does `path.write_text(json.dumps(report, default=str))`
  at `:39`. A new key round-trips for free (and `default=str` means even a `date` survives).
- **The API already mutates the report post-hoc** -- `backend/api/backtest.py:1059-1060`:
  `report["run_id"] = run_id` then `report["config"] = {...}`, saved at `:1073`. **That is the
  in-repo precedent for adding a top-level report key.** Follow it.
- **Frontend:** `frontend/src/lib/types.ts:892` `export interface BacktestResults` (with a
  nested `analytics: {...}` at `:895`). TypeScript structural typing means an EXTRA runtime key
  is not an error -- nothing breaks if the key is absent from the interface. Adding it to the
  interface is only needed if the UI is to DISPLAY it. **Scope call for the contract:** the
  criteria say "the run result records" -- persisting it in the report JSON satisfies that;
  UI rendering is optional and should be an explicit choice, not an accident.
- **Other `BacktestResult` readers that must tolerate a new field:**
  `backend/agents/mcp_servers/backtest_server.py:119` (reads fields by name -- safe);
  `backend/backtest/quant_optimizer.py:656` (`_extract_top_features`, reads `feature_importance_mda`
  -- safe). **Trap:** `backend/tests/test_phase_75_mcp_truth.py:371-375` asserts
  `not hasattr(r, absent)` for `("dsr", "return_pct", "max_drawdown_pct", "num_trades")` --
  do NOT name the new field any of those four, or that test flips.
- No `asdict(BacktestResult)` call exists (the `asdict` hits in the grep are on other
  dataclasses: `autonomous_loop.py:54`, `agents/trace.py:92`, `news/bq_writer.py:147`, etc.),
  so there is no BQ-schema round-trip to break.

### 5.6 Existing precedent for a "degraded" flag -- THERE IS NONE

Measured: `grep -rn "degraded\|macro_available\|data_available\|_available\|coverage\|warnings"
--include="*.py" backend/backtest/`. **Exactly one hit, and it is a prose comment**
(`backend/backtest/data_ingestion.py:25`). There is **no existing shape to follow** inside
`backend/backtest/`. Nearest in-repo analogues are outside this package (e.g. the API's
post-hoc `report["config"]` block at `api/backtest.py:1060`). So the implementer must invent
the shape -- recommend keeping it minimal and mirroring the `report["config"]` idiom.

### 5.7 Tests that would break if the return started being checked -- NONE (measured)

Measured: `grep -rn "preload_macro|preload_prices|preload_fundamentals" --include="*.py"
backend/tests/ tests/` filtered for monkeypatch/setattr/lambda/def.

- **Zero tests monkeypatch `cache.preload_macro` on the engine path.** The two hits
  (`backend/tests/test_phase_82_12_string_column_guards.py:107` and `:129`) are *local helper
  functions named* `preload_macro(rows)` -- fixture builders, not patches of the cache module.
- The 4 direct callers in `backend/tests/test_phase_82_0_macro_ingestion.py` (`:212`, `:239`,
  `:266`, `:286`) call `cache_mod.preload_macro()` and already ASSERT ON THE RETURN -- they
  test the refusal semantics directly and are unaffected by an engine-side change.
- `backend/tests/test_phase_82_15_macro_point_in_time.py:242` and `:272` DISCARD the return but
  are cache-level tests, not engine-level; unaffected.
- `dev/t_backtest_mock.py:130-144` monkey-assigns `cache_mod.cached_macro = mock_cached_macro`
  -- it stubs the *accessor*, not the preloader, so an engine-side abort WOULD fire under that
  mock unless it also stubs the availability check. **Flag this file to the implementer**: it is
  a dev harness, not part of the pytest suite, but it will break loudly (which is arguably
  correct) if the guard is added naively.

### 5.8 Test-fixture feasibility for criteria 1 and 4

Measured: `Read backend/backtest/backtest_engine.py` offset 255/1530; `sed -n '320,400p'
backend/tests/test_phase_75_mcp_truth.py`.

A cheap, hermetic engine-level fixture IS reachable:
- `_auto_ingest_if_needed` (`backtest_engine.py:1530-1547`) is wrapped in
  `except Exception as e: logger.warning("Auto-ingest check failed (non-fatal): %s")` -- it
  cannot abort a test.
- Stubbing `self.scheduler.generate_windows()` to return `[]` makes `run_backtest` reach
  `:315-317` and then fall straight through the window loop -- so the whole test is the preload
  block plus trivial aggregation. No BQ, no ML.
- Precedent for `__new__`-based construction + attribute injection is
  `backend/tests/test_phase_75_mcp_truth.py:357` (`srv = bs.BacktestServer.__new__(bs.BacktestServer)`);
  the same trick avoids `BacktestEngine.__init__`'s `cache.init_cache(...)` at
  `backtest_engine.py:189`.
- **Criterion 1 mutation-check:** with `preload_macro` stubbed to return `0` AND
  `cache._macro_full` left empty, TODAY's code proceeds silently -> the test asserting an
  explicit refusal FAILS on current `main`. That is the required failing-first property.
- **Criterion 4 mutation-check:** with `preload_macro` stubbed to return a positive int AND
  `_macro_full` populated, the run must complete normally and record availability = true. A
  guard that always aborts fails this. **Both fixtures must set `_macro_full` consistently with
  the return value** -- see the trap in §6.

### 5.9 A SECOND silent-degradation site on the same path (out of scope, but file it)

`backend/backtest/historical_data.py:269` guards the macro features with a bare `if macro:` --
so when `cached_macro` returns `{}` (empty table, timed-out fallback query at `cache.py:636`,
or a point-in-time filter that excludes everything) the macro features are **silently omitted
from the feature vector** and the model simply trains without them. The 82.15 comment at
`cache.py:75-79` says exactly this: "which would blank all six macro features across the entire
2018-2025 backtest window -- silently, because `historical_data.py` guards on `if macro:` and
simply never sets the [features]". **This is the same defect class one layer down and it is NOT
covered by 82.13's criteria.** Per the standing "queue discovered defects in the masterplan"
rule, it should get its own step rather than be smuggled into this one.

---

## 6. Recommendation for the contract

### 6.1 The design (what to build)

**A. Add a truthful availability accessor to `cache.py` -- do not re-derive from the `int`.**
The only invariant that holds across all five return paths (§5.2) is *`_macro_full` non-empty
iff macro is available*. Add next to `get_preloaded_tickers()` (`cache.py:466`):

- `macro_is_loaded() -> bool` -- `return bool(_macro_full)`; and/or
- `macro_status() -> dict` -- `{"loaded": bool, "series": int, "rows": int}`.

Then at `backtest_engine.py:317`, **stop discarding**: capture the int for the log line, but
branch on the accessor. This is the CWE-252 mitigation "account for all possible return values"
(S1) applied honestly: the int is a *diagnostic*, the accessor is the *decision*.

**B. Choose ONE of the two allowed outcomes -- and make it configurable, defaulting to
FAIL-FAST.** The criteria permit either. The literature splits cleanly:
- *Abort* is right when the degraded path is slower and worse than not running (S2:
  "When overloaded at either the frontend or backend layers, fail early and cheaply").
  Here the degraded path is an unbounded sequence of 30s BQ round-trips producing a
  macro-blind model -- so **fail-fast is the correct default**.
- *Run labelled* is right when a partial answer has value (S2 graceful degradation), and is
  only acceptable if the result is labelled (S3/S4/S6).

Recommended: raise a dedicated exception (e.g. `MacroUnavailableError`) at `:317` whose message
names the refusal reason, gated by a settings flag (`backtest_allow_macro_free_run: bool =
False`) that switches to the labelled-degraded mode. Both branches set the metadata in A/C, so
criterion 2 holds in both.

**C. Record availability on the result AND in the report (criterion 2).**
- New defaulted field on `BacktestResult` (`backtest_engine.py:119-132`), e.g.
  `data_availability: dict = field(default_factory=dict)` populated as
  `{"macro": bool, "macro_reason": str, "macro_rows": int, "prices": ..., "fundamentals": ...}`.
  **Do not name it `dsr`/`return_pct`/`max_drawdown_pct`/`num_trades`** (§5.5 trap).
- Surface it in `generate_report` (`analytics.py:741`) as a top-level `"data_availability"` key
  AND (recommended) a boolean inside `report["analytics"]` so that any consumer reading only
  `["analytics"]` -- which is what `strategy_backtest_adapter.py:162` and
  `strategy_candidate_producer.py:28` do -- cannot miss it. This is the model-card /
  datasheet argument (S3, S4) and the MLflow "filter metrics based on specific datasets for
  fair model comparison" argument (S6) applied to `optimizer_best.json` / `quant_results.tsv`.
- Free via `result_store.save_result`'s `json.dumps(report, default=str)`
  (`result_store.py:39`); no frontend change required (`types.ts:892` is structural).

**D. Criterion 3 -- AST enumeration, asserted non-empty.** Ship a test that walks every repo
`.py` with `ast.parse` + `ast.walk`, collects `ast.Call` nodes whose func name starts with
`preload_`, and classifies DISCARDED iff the direct parent node is `ast.Expr`. Assert the
collection is non-empty, then assert the specific classification for the engine's three sites.
A working implementation produced the 15-row table in §5.4; reuse it. **A grep-based version
does not satisfy the criterion** -- `cache.py:53` and `settings.py:252` contain the string in
prose, and `backend/tests/test_phase_75_mcp_truth.py:278` *defines* a shadowing function.

**E. Fix the lying docstring.** `cache.py:337` says "Returns the total number of rows loaded."
That is false on 3 of 5 paths (R1 returns the cached total, R3/R4 return 0 as a *refusal*).
Rewrite it to enumerate the return contract -- CWE-252's design-side mitigation is "When
designing a function, make sure you return a value or throw an exception in case of an error"
(S1); the minimum here is that the contract be *documented*.

### 6.2 Traps to avoid

1. **`if not cache.preload_macro(): abort` conflates R2/R3/R4** and would silently break if a
   future path returns 0 with a populated cache. Branch on availability, not on the int (§5.2).
2. **The already-warm path returns a POSITIVE int without loading anything** (`cache.py:345`).
   The optimizer runs with `skip_cache_clear=True`, so this fires on **every iteration after the
   first**. A guard keyed to "rows loaded THIS call > 0" would abort every optimizer iteration
   but the first. This is the highest-probability implementation bug in the step.
3. **`clear_cache()` (`cache.py:193-203`) clears `_macro_full`**, so availability is per-run
   state, not process state. Compute it right after the preload block, not lazily later.
4. **Don't put the abort inside the per-window `try/except Exception`** at
   `backtest_engine.py:334-346` -- that handler swallows exceptions into `logger.error` and
   would recreate the exact silent-degradation this step removes. The check belongs at
   `:317-318`, before the loop.
5. **A guard that always aborts passes criterion 1 and fails criterion 4.** The success fixture
   must set `_macro_full` non-empty; if it only stubs the return int, an accessor-based guard
   will still abort and the implementer will "fix" it in the wrong direction.
6. **`dev/t_backtest_mock.py:130-144`** stubs `cached_macro` but not the preloader -- expect it
   to start failing. Decide deliberately whether to update it.
7. **Don't widen scope to `preload_prices`/`preload_fundamentals`** (also discarded, §5.4) or to
   the `if macro:` guard at `historical_data.py:269` (§5.9). File those; don't smuggle them in.
   The criteria are macro-specific.
8. **Do not restate "~40 minutes" as a measured figure** anywhere in the contract,
   `experiment_results.md`, or a code comment. It is unmeasured (§5.3).

### 6.3 Where the step description / repo is WRONG or STALE

| Claim | Verdict |
|---|---|
| "backtest_engine.py:317 calls cache.preload_macro()" | **CORRECT** as of 2026-08-05 (§5.1). |
| "the engine ... CLAUDE.md's documented ~40-minute backtest hang" | **STALE/FOLKLORE.** `CLAUDE.md:30` is the sole origin; re-quoted at `cache.py:53` and `handoff/harness_log.md:11866`. **No measurement exists in the repo.** The defensible claim is "one uncached 30s-timeout BQ round-trip per distinct cutoff date". |
| **`backend/backtest/cache.py:51` cites "backtest_engine.py:308 DISCARDS preload_macro's return value"** | **STALE ANCHOR, introduced by 82.0 itself.** The real line is **:317** (`:308` is the `global_start` date computation). Fix this comment in the same step -- it is the exact class of stale anchor that has burned this session. |
| **`backend/backtest/cache.py:337` docstring: "Returns the total number of rows loaded."** | **WRONG on 3 of 5 return paths** (§5.2 R1/R3/R4). This is arguably the *root* of the whole defect: the call site discarded a value the docstring described as a harmless statistic. |
| "82.0 ... widened the daily-series SLA 5 -> 12 days; measured headroom had been ONE day" | **CONFIRMED** verbatim at `cache.py:44-56` (`DGS10: 12`, `T10Y2Y: 12`; comment records "DGS10 newest 2026-07-30 = 4d" against a 5-day bound). |
| "The only signal is a WARNING log" | **CONFIRMED**: `logger.warning` at `cache.py:411` (unparseable dates) and `cache.py:428` (stale series). Nothing else is emitted. |
| "cached_macro falls through to a per-cutoff-date BQ query, one per distinct cutoff" | **CONFIRMED**: fallback at `cache.py:603-641`, memoised per cutoff by `_macro_cache` (`:571`, `:640`). Driver is `historical_data.py:48`. |
| "There is NO field for data availability [on BacktestResult]" | **CONFIRMED** (`backtest_engine.py:119-132`), and there is no precedent anywhere in `backend/backtest/` to copy (§5.6). |

---

## 7. Research Gate checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6** (S1-S6; 2 official
      standards/docs bodies, 2 peer-reviewed papers, 2 official vendor docs). Two further URLs
      were attempted and 404'd; they are recorded as snippet-only, not counted.
- [x] 10+ unique URLs total -- **24**.
- [x] Recency scan (last 2 years) performed + reported -- §4, including a decision-relevant
      NEGATIVE finding (Python still has no `must_use`/`nodiscard`, so the guard must be a test).
- [x] Full pages read (not abstracts) for the read-in-full set. arXiv:2506.06147 was
      abstract-only and is therefore in the snippet-only table, NOT the gate count.
- [x] file:line anchors for every internal claim, each with the command that measured it.

Soft checks:
- [x] Internal exploration covered the engine, the cache, the report/persistence path, the
      frontend type, the test suite, and the AST call-site census.
- [x] Contradictions noted -- S2 (Google SRE) does NOT support in-result labelling; that claim
      rests on S3/S4/S6 and is flagged as such rather than over-claimed.
- [x] Claims cited per-claim.

Known gaps (declared, not padded):
- The **~40-minute** hang could NOT be measured; no timing artefact exists in the repo. Stated
  as folklore, not fact.
- Two source fetches 404'd (OpenSSF Python CWE-252 page; `doc.rust-lang.org/std/attr.must_use.html`).
  The Rust claim was recovered from the Rust Reference (S5); the OpenSSF Python framing is
  carried only at snippet strength and is marked as such.

---

## 8. Gate envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 18,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "82.13 confirmed at backend/backtest/backtest_engine.py:317 (run_backtest, no try/except around the preload block, _report_progress hook available at :307/:1103). preload_macro (cache.py:333) has FIVE return paths: :345 returns a POSITIVE cached total when already warm (fires on every optimizer iteration under skip_cache_clear=True), :356 returns 0 for an empty table, :418 and :435 return 0 as REFUSALS, :461 returns rows loaded -- so `if not preload_macro()` is a trap; the only honest predicate is `_macro_full` non-empty. The consequence path is confirmed (cache.py:603-641, one 30s BQ query per distinct cutoff, driven by historical_data.py:48) but the ~40-minute figure is UNMEASURED folklore from CLAUDE.md:30. AST census: 967 files, 15 preload_* call sites, 8 discarded / 7 used. BacktestResult (:119-132) has no availability field; adding one is safe (report dict is schemaless, api/backtest.py:1059 already adds keys post-hoc, TS interface is structural). No test monkeypatches preload_macro on the engine path. Two stale-anchor/docstring defects found in cache.py itself (:51 cites :308; :337 docstring is false on 3 of 5 paths).",
  "brief_path": "handoff/current/research_brief_82.13.md",
  "gate_passed": true
}
```
