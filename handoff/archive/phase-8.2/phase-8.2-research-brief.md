---
step: phase-8.2
title: Chronos-Bolt shadow-logged feature pilot
date: 2026-04-19
tier: moderate
---

## Research: Chronos-Bolt shadow-logged feature pilot (phase-8.2)

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://github.com/amazon-science/chronos-forecasting | 2026-04-19 | code/doc | WebFetch | Canonical install: `pip install chronos-forecasting`; supports Chronos-Bolt and Chronos-2 families |
| https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py | 2026-04-19 | code | WebFetch | `ChronosBoltPipeline.predict(inputs, prediction_length)` returns tensor shape `[batch, num_quantiles, pred_len]`; `predict_quantiles` is the quantile variant |
| https://huggingface.co/amazon/chronos-bolt-small | 2026-04-19 | doc | WebFetch | Import: `from chronos import BaseChronosPipeline`; `pipeline.predict(context=torch.tensor(series), prediction_length=12)` returns `[num_series, num_quantiles, pred_len]`; small = 48M params |
| https://arxiv.org/html/2511.18578v1 | 2026-04-19 | paper | WebFetch | Re(Visiting) TSFMs in Finance (Rahimikia 2025): zero-shot Chronos(large) R^2=-1.37%, directional acc ~51%; zero-shot TimesFM(500M) R^2=-2.80%, acc ~50%; fine-tuned Chronos outperforms TimesFM |
| https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html | 2026-04-19 | doc | WebFetch | AutoGluon route wraps Chronos-Bolt via TimeSeriesPredictor; not suited for thin standalone inference; `chronos-forecasting` pip package is the right choice for a lightweight client |
| https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/ | 2026-04-19 | blog | WebFetch | AWS official: Chronos-Bolt(Base) 250x faster, 20x more memory-efficient than Chronos-T5 same size; four variants: Tiny(9M), Mini(21M), Small(48M), Base(205M) |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://huggingface.co/amazon/chronos-bolt-base | doc | Search snippet; small confirmed as lighter and preferred for shadow pilot |
| https://huggingface.co/autogluon/chronos-bolt-small | doc | autogluon-namespace mirror of amazon/chronos-bolt-small; redundant |
| https://huggingface.co/autogluon/chronos-bolt-base | doc | autogluon-namespace mirror; redundant |
| https://pypi.org/project/chronos-forecasting/1.5.1/ | doc | PyPI page failed to load |
| https://github.com/amazon-science/chronos-forecasting/releases | doc | Snippet only; version history not needed for API design |
| https://galileo.ai/blog/amazon-chronos-ai-time-series-forecasting-guide | blog | Overview blog; no new API specifics beyond HF card |
| https://paperswithbacktest.com/course/timesfm-vs-chronos-vs-moirai | blog | Snippet: no Chronos-Bolt vs TimesFM head-to-head numbers available |
| https://arxiv.org/html/2507.07296v1 | paper | Snippet only; multivariate focus, no Chronos-Bolt-specific results |

### Recency scan (2024-2026)

Searched for 2024-2026 literature on Chronos-Bolt inference, Chronos-Bolt vs TimesFM equity forecasting, and chronos-forecasting package updates. Key findings:

- Chronos-Bolt released November 2024 (HuggingFace); Chronos-2 followed October 2025 with 90%+ win-rate over Chronos-Bolt.
- Rahimikia (arXiv 2511.18578, November 2025): zero-shot Chronos modestly outperforms TimesFM on directional accuracy for equity returns (~51% vs ~50%); neither dominates ensemble models in raw R^2.
- No 2026 paper compares Chronos-Bolt specifically against TimesFM 2.5; the 2025 paper uses Chronos-T5 large, not Bolt.
- `chronos-forecasting` PyPI reached v1.5.1 (2025); no breaking API change to `BaseChronosPipeline.predict` observed.

---

### Key findings

1. **Canonical install: `pip install chronos-forecasting`** -- the `chronos-forecasting` package is the lightweight standalone path. AutoGluon is the heavy enterprise integration; overkill for a shadow pilot client. (Source: amazon-science/chronos-forecasting README, HF card)

2. **Forecast API shape**: `BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")` then `.predict(context=torch.tensor(series, dtype=torch.float32), prediction_length=N)` returns tensor `[1, num_quantiles, N]`. Point forecast = median quantile = `result[0, result.shape[1]//2, :].tolist()`. (Source: chronos_bolt.py, HF card)

3. **Recommended `_MODEL_NAME = "amazon/chronos-bolt-small"`** (48M params). Tiny(9M) and Mini(21M) exist but are less accurate; Base(205M) is 4x heavier with marginal gain for a shadow pilot. Small matches the phase-8.1 precedent of choosing the mid-tier variant. (Source: AWS blog, HF model cards)

4. **StubModel shape for tests**: the real pipeline returns a torch tensor; stub should return an object where `result[0, result.shape[1]//2, :]` yields a float-iterable. A numpy array of shape `(1, 9, horizon)` is the cleanest stub. (Source: chronos_bolt.py predict signature)

5. **Lazy import discipline**: import `chronos` (for `BaseChronosPipeline`) inside `_get_model`; import `torch` inside the same try-block (torch is a chronos dependency but absent in CI). Match timesfm_client.py fail-open pattern. (Source: timesfm_client.py internal, lines 60-78)

6. **Equity return signal context**: zero-shot Chronos-T5(large) gets ~51% directional accuracy vs TimesFM ~50% (Rahimikia 2025). Marginal advantage, consistent with shadow-only designation. Neither is production-ready zero-shot; both need fine-tuning to be competitive with ensemble models. (Source: arXiv 2511.18578)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/models/timesfm_client.py` | 207 | Phase-8.1 TimesFM shadow client -- the mirror template | Current, complete |
| `backend/models/__init__.py` | ~1 | Package init | Present (empty or minimal) |
| `tests/models/test_timesfm_client.py` | 129 | 10 tests mirroring pattern for 8.2 | Current, all patterns reusable |
| `tests/models/__init__.py` | ~1 | Test package init | Present |

Key patterns from `timesfm_client.py` to mirror exactly:

- `_MODEL_NAME`, `_SHADOW_TABLE`, `_DEFAULT_CONTEXT_LENGTH`, `_DEFAULT_HORIZON_LENGTH` module-level constants (lines 25-28)
- `__init__` stores `self._model: Any = None` for lazy init (line 54)
- `_get_model()` wraps the import in a `try/except Exception` returning `None` on failure (lines 56-78)
- `forecast()` guards on `len(series) < 2` and `h <= 0` before touching the model (lines 88-95)
- `forecast_batch()` filters to `clean` list, preserves all tickers with `setdefault(t, [])` (lines 126-146)
- `shadow_log()` mirrors column set: `model_name`, `ticker`, `as_of_date`, `horizon`, `forecast_values`, `observed_values`, `logged_at` (lines 192-195)
- All logger messages ASCII-only, no Unicode (security rule)

Divergence point for Chronos-Bolt vs TimesFM: the model call.

TimesFM: `point, _ = model.forecast(horizon=h, inputs=[np.asarray(series)])` -- returns `(point_array, quantile_array)`.

Chronos-Bolt: `result = pipeline.predict(context=torch.tensor(...), prediction_length=h)` -- returns tensor `[1, num_quantiles, h]`; extract median with `result[0, result.shape[1]//2, :].tolist()`. Requires `torch` import inside the method body (lazy).

---

### Consensus vs debate

Consensus: `pip install chronos-forecasting` is the right package for standalone inference (not autogluon). Both HF cards, the README, and the AWS blog confirm this.

Debate: `amazon/chronos-bolt-small` vs `amazon/chronos-bolt-tiny` for a shadow pilot. Tiny is 9M and marginally faster; Small is 48M and noticeably more accurate. Phase-8.1 chose the mid-tier model (TimesFM 2.5 200M) -- same discipline applies here, so Small is recommended.

Note: Chronos-2 (October 2025) now outperforms Chronos-Bolt by 90%+. A future phase-8.x could upgrade the model name without changing client API (both use `BaseChronosPipeline`).

---

### Pitfalls (from literature and source)

1. **torch dependency**: `chronos-forecasting` depends on PyTorch; must be lazy-imported like numpy in timesfm_client.py. If torch is absent, fail open.
2. **Tensor vs numpy input**: `pipeline.predict` expects `torch.tensor`, not a numpy array. Wrap the float list: `torch.tensor(series, dtype=torch.float32)`.
3. **Quantile index for point forecast**: predict returns `[batch, num_quantiles, pred_len]`. `num_quantiles` varies by model (typically 9 for official models). Use `shape[1]//2` not a hardcoded index.
4. **Python version**: same 3.14 constraint as TimesFM applies; lazy imports + fail-open is mandatory since CI does not install chronos.
5. **Chronos-2 vs Chronos-Bolt import**: Chronos-2 uses `Chronos2Pipeline`; Chronos-Bolt uses `BaseChronosPipeline` (or `ChronosBoltPipeline`). Using `BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small")` is confirmed correct.

---

### Application to pyfinagent

| Design decision | Mapping | file:line anchor |
|----------------|---------|-----------------|
| `_MODEL_NAME = "amazon/chronos-bolt-small"` | 48M params, mid-tier, mirrors 8.1 discipline | New file mirrors timesfm_client.py:25 |
| Lazy `from chronos import BaseChronosPipeline` | Absent in Python 3.14 CI venv; fail-open | Mirror timesfm_client.py:60 |
| Lazy `import torch` | Dependency of chronos; same absent-in-CI risk | Mirror timesfm_client.py:97 |
| `predict(context=torch.tensor(series, dtype=torch.float32), prediction_length=h)` | Exact Bolt API call | Mirror timesfm_client.py:102 |
| Median quantile extraction: `result[0, result.shape[1]//2, :].tolist()` | `predict` returns `[1, num_quantiles, h]` | Replaces timesfm_client.py:102-103 |
| `shadow_log` writes to same `ts_forecast_shadow_log` table | `model_name` column distinguishes Bolt vs TimesFM entries | Mirror timesfm_client.py:149-203 |
| StubModel in tests returns `np.zeros((1, 9, horizon))` | 9 quantiles, horizon cols; median = index 4 | Mirror test_timesfm_client.py:60-65 |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (14 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (timesfm_client.py + test file read in full)
- [x] Contradictions / consensus noted (autogluon vs chronos-forecasting package debate resolved)
- [x] All claims cited per-claim

---

### Three-variant query log

1. Current-year frontier: `amazon chronos-bolt forecasting model GitHub amazon-science` (2026 session)
2. Last-2-year window: `amazon chronos-bolt-small chronos-bolt-base parameters inference speed 2024 2025`
3. Year-less canonical: `chronos-forecasting vs autogluon timeseries chronos-bolt which package standalone inference`
4. Equity comparison: `TimesFM vs Chronos-Bolt equity return forecasting benchmark comparison 2025`
5. Install path: `chronos-bolt 2025 autogluon chronos-forecasting package install`

---

### Summary (< 150 words)

**(a) Canonical install**: `pip install chronos-forecasting`. Not autogluon -- autogluon wraps Chronos-Bolt inside TimeSeriesPredictor and is too heavy for a thin shadow client. The `chronos-forecasting` package provides `BaseChronosPipeline` directly.

**(b) Forecast API shape**: `from chronos import BaseChronosPipeline`; `pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")`; `result = pipeline.predict(context=torch.tensor(series, dtype=torch.float32), prediction_length=horizon)` returns tensor `[1, num_quantiles, horizon]`. Point forecast: `result[0, result.shape[1]//2, :].tolist()`. Both `chronos` and `torch` must be lazy-imported with fail-open.

**(c) Recommended `_MODEL_NAME`**: `"amazon/chronos-bolt-small"` (48M params). Mirrors phase-8.1's mid-tier discipline. Tiny(9M) too weak; Base(205M) too heavy for a shadow pilot.

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/phase-8.2-research-brief.md",
  "gate_passed": true
}
```
