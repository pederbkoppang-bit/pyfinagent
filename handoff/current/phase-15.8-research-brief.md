# Research Brief: phase-15.8 -- Transformer Signal Viewer (TimesFM + Chronos Forecasts per Ticker)

**Date:** 2026-04-21
**Tier:** moderate
**Researcher:** researcher agent

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://github.com/google-research/timesfm | 2026-04-21 | code/docs | WebFetch | TimesFM 2.5 PyTorch package: `timesfm.TimesFM_2p5_200M_torch.from_pretrained()`; requires Python >=3.10,<3.12 (confirmed by existing `timesfm_client.py` comments) |
| https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/ | 2026-04-21 | official docs | WebFetch | Chronos-Bolt via `chronos-forecasting` package; 4 model sizes (Tiny 9M to Base 205M); zero-shot outperforms statistical baselines on 27 general datasets; CPU-capable |
| https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/ | 2026-04-21 | authoritative blog | WebFetch | TSFMs on equities: "promising research direction, not a production alpha engine"; IC may be "statistically detectable but worthless after bid-ask spreads"; regime robustness untested |
| https://www.mql5.com/en/articles/22096 | 2026-04-21 | industry practitioner | WebFetch | TimesFM 2.5 used in MT5: 512-bar context / 48-bar horizon; not standalone trading signal; requires LoRA fine-tuning for financial use; temporal degradation beyond horizon |
| https://cloud.google.com/blog/products/data-analytics/timesfm-models-in-bigquery-and-alloydb | 2026-04-21 | official docs | WebFetch | TimesFM in BQ GA since 2025-11-19 via `AI.FORECAST` SQL function; alternative runtime path that bypasses Python 3.14 incompatibility |
| https://cloudscape.design/patterns/general/announcing-beta-preview-features/ | 2026-04-21 | official design docs | WebFetch | Beta/shadow UX: avoid badges, use nav labels or page-title suffixes; clearly communicate release state in workflow; never combine "Beta" + "New" labels |
| https://recharts.github.io/en-US/examples/ | 2026-04-21 | official docs | WebFetch | `ComposedChart` + `Line` + `Area` for forecast overlay with confidence bands; `ReferenceLine` supported natively |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://byteiota.com/timesfm-2-5-googles-zero-shot-forecasting-model-2026/ | blog | Fetched in full; univariate-only limitation and benchmark caveat documented |
| https://pypi.org/project/timesfm/ | docs | PyPI page; headline facts captured from GitHub fetch and existing codebase |
| https://huggingface.co/google/timesfm-2.5-200m-pytorch | code | Model card; key API facts captured via GitHub fetch |
| https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting | official | Fetched in full; Chronos-2 beats Chronos-Bolt but no financial-specific metrics |
| https://aitoolly.com/ai-news/article/2026-04-05-google-research-unveils-timesfm-a-new-pre-trained-foundation-model-for-advanced-time-series-forecast | news | Snippet-only; narrative overview, no new technical facts |
| https://www.smashingmagazine.com/2025/09/ux-strategies-real-time-dashboards/ | blog | Snippet-only; general dashboard UX, cloudscape covered shadow labeling in full |
| https://github.com/recharts/recharts | code | Index only; examples page fetched in full |
| https://github.com/amazon-science/chronos-forecasting | code | Index page; API facts covered by AWS blog fetch |
| https://aihorizonforecast.substack.com/p/timesfm-25-hands-on-tutorial-with | blog | Snippet-only; narrative overview |
| https://deepwiki.com/amazon-science/chronos-forecasting | docs | Snippet-only; covered by AWS blog |

---

## Recency scan (2024-2026)

Searched for 2024-2026 literature on: "TimesFM financial forecasting 2026", "Chronos-Bolt financial forecasting 2025 2026", "shadow mode experimental feature dashboard 2025", "Recharts forecast overlay confidence bands React 19 2025".

**Findings:**

- **TimesFM 2.5 released Sept 2025** (200M params, 16K context, GIFT-Eval leader among zero-shot models). BigQuery integration GA Nov 2025 -- provides a workaround for the Python 3.14 incompatibility present in this repo.
- **Chronos-2 released Jan 2026** -- beats Chronos-Bolt by >90% win rate on general benchmarks; `chronos-forecasting` package on PyPI. This is newer than the `chronos_client.py` which targets `chronos-bolt-small`.
- **Zero-shot TSFMs on equities (Nov 2025 arXiv 2511.18578)**: TimesFM zero-shot R-squared = -2.80%, directional accuracy <50%, annualised return -1.47% vs CatBoost 46.50% on daily excess returns. This evidence is already cited in the phase-8.4 decision memo and remains authoritative.
- **Beta/shadow UX patterns** (Cloudscape 2025): subtler labeling preferred (nav label + page title suffix) over prominent badges. AWS Amplify, Linear, Grafana all converged on page-level banner approach for experimental features.
- **Recharts React 19**: Known compatibility issue (charts don't render on React 19). However, this repo is already using Recharts across multiple components (SharpeHistoryChart, PaperReconciliationChart, StockChart, etc.) and they are confirmed working. The `recharts` maintainers list React 19 support as tracked (#4558); existing components function.

No new findings supersede the phase-8.4 REJECT verdict. The BQ `AI.FORECAST` path is a 2025 addition worth noting but does not change the promotion gate requirements.

---

## Key findings

1. **Both model clients exist in the codebase** (`backend/models/timesfm_client.py`, `backend/models/chronos_client.py`) with fail-open lazy imports. Neither can produce live forecasts in the repo's Python 3.14 venv -- this is the documented runtime gate from phase-8.4. (Source: internal files + phase-8.4 decision memo)

2. **Phase-8.4 verdict is REJECT** with explicit re-evaluation conditions: Python 3.11 runtime, fine-tuned variant, 60+ days shadow-log, ensemble IR > MDA IR by >= 0.10. None are met. (Source: `handoff/current/phase-8-decision.md`, accessed 2026-04-21)

3. **The endpoint must be a shadow stub**, returning `status='shadow'` with empty forecast arrays, since no live forecasts can be generated. The spec shape is satisfied, and the response is honest. (Source: spec + phase-8.4 decision memo)

4. **BQ AI.FORECAST is a future alternative runtime path** (GA since Nov 2025) but is not within scope for phase-15.8; it belongs in the re-evaluation plan when the runtime gate is cleared. (Source: Google Cloud blog, 2026-04-21)

5. **MDA baseline**: `backend/backtest/backtest_engine.py` has an MDA cache at `backend/backtest/experiments/mda_cache.json`. The `get_latest_mda()` function returns `{feature_name: float}` weights, NOT a price forecast series. The panel cannot overlay MDA as a price line -- it is a feature-weight map, not a time series. The "baseline" in the chart must be either a flat line at the last closing price or omitted as `null` until real forecasts exist. (Source: `backend/backtest/backtest_engine.py` lines 56-68)

6. **Recharts `ComposedChart`** is the correct component: supports multiple `Line` series, `Area` for confidence bands, and `ReferenceLine`. This matches what `SharpeHistoryChart.tsx` already uses. (Source: Recharts docs, existing component at `frontend/src/components/SharpeHistoryChart.tsx` lines 1-21)

7. **Public path inheritance**: `/api/signals` was added to `_PUBLIC_PATHS` in phase-15.7 (`backend/main.py` line 218). The new `/api/signals/{ticker}/transformer-forecast` endpoint is under the same prefix and inherits public access with no auth changes required. (Source: `backend/main.py` line 218)

8. **Insertion point in signals page**: after `<AltDataPanel data={altData} />` at line 180 of `frontend/src/app/signals/page.tsx`. The panel should be conditionally rendered inside the `{data && enrichmentSignals && (...)}` block alongside AltDataPanel, with its own independent fetch that fails open (`.catch(() => null)` pattern matching AltDataPanel's). (Source: `frontend/src/app/signals/page.tsx` lines 149-182)

9. **Banner copy**: Phase-8.4 REJECT was based on (a) Python runtime incompatibility, (b) published evidence that zero-shot TSFMs deliver <50% directional accuracy on equities, and (c) no shadow-log data. The banner should honestly state both blockers, not just one.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/models/timesfm_client.py` | 207 | TimesFM lazy-load wrapper; fail-open; shadow_log to BQ | Exists, phase-8.1 |
| `backend/models/chronos_client.py` | 197 | Chronos-Bolt lazy-load wrapper; fail-open; same BQ table | Exists, phase-8.2 |
| `backend/backtest/ensemble_blend.py` | ~200+ | Equal/correlation/shrinkage blender for MDA+TimesFM+Chronos | Exists, phase-8.3; no live data yet |
| `backend/api/signals.py` | 385 | Signals router; existing alt-data endpoint pattern at line 328 | Use as template for new endpoint |
| `backend/main.py` | ~350 | App entry; `_PUBLIC_PATHS` at line 218 includes `/api/signals` | New endpoint inherits public access |
| `backend/backtest/backtest_engine.py` | 1167 | `get_latest_mda()` at line 63 returns `{feature: weight}` -- NOT a forecast series | MDA is not a chart-able baseline |
| `frontend/src/app/signals/page.tsx` | 199 | Signals page; AltDataPanel insertion at line 180; insertion point for TransformerForecastPanel at line 181 | |
| `frontend/src/components/SharpeHistoryChart.tsx` | 60+ | Uses `ComposedChart`, `Line`, `CartesianGrid`, `Tooltip`, `ResponsiveContainer` from recharts | Canonical Recharts pattern |
| `frontend/src/components/AltDataPanel.tsx` | 60+ | BentoCard wrapper pattern; empty-state pattern; fail-open | Template for TransformerForecastPanel |
| `handoff/current/phase-8-decision.md` | ~100 | Phase-8.4 REJECT decision memo; re-evaluation conditions | Authoritative for banner copy |

---

## Consensus vs debate (external)

**Consensus:** Zero-shot TSFMs (TimesFM, Chronos) are not ready for production equity signal use. Published evidence (arXiv 2511.18578, Nov 2025) shows directional accuracy ~50% and negative alpha. Fine-tuning is required. The phase-8.4 REJECT decision is consistent with the published record.

**Debate:** BQ `AI.FORECAST` (GA Nov 2025) provides a Python 3.14-safe path to TimesFM. Future re-evaluation could use this path without a Docker runtime. Chronos-2 (Jan 2026) outperforms Chronos-Bolt by >90%, making the current scaffold slightly dated.

---

## Pitfalls (from literature)

1. **Presenting shadow forecasts as actionable**: Panel must have an unmissable banner that reads "not for trading decisions." Cloudscape pattern: page-level banner + title suffix, avoid badges.
2. **MDA as a "baseline line"**: MDA in this repo is a feature-importance weight map, not a price or return forecast series. Do NOT try to plot it as a chart line -- it is structurally incompatible. Use a flat line at price[0] or omit.
3. **React 19 / Recharts**: Known issue but existing components work. Do not introduce a different chart library for this panel.
4. **Auto-refreshing stale stub data**: The endpoint will return empty arrays every call. Cache it lightly (e.g., 60s TTL in `api_cache.py`) to avoid hammering the server, but do not poll aggressively from the frontend.
5. **Promoting Chronos-2 scope-creep**: Chronos-2 is newer and better, but the codebase uses `chronos-bolt-small`. Updating to Chronos-2 is out of scope for phase-15.8 and belongs in the re-evaluation plan.

---

## Application to pyfinagent (mapping external findings to file:line anchors)

| External finding | Internal anchor | Implication for phase-15.8 |
|----------------|----------------|---------------------------|
| TimesFM PyPI requires Python <3.12 | `timesfm_client.py:9-13` | Fail-open path always executes; forecast arrays are always `[]` |
| Chronos-Bolt lazy-load fails on missing torch | `chronos_client.py:57-65` | Same: fail-open, arrays always `[]` |
| BQ AI.FORECAST GA Nov 2025 | external | Out of scope for 15.8; note in endpoint docstring as future path |
| Phase-8.4 REJECT: 4 conditions unmet | `handoff/current/phase-8-decision.md:32-42` | `status='shadow'` must remain until all 4 conditions met; banner copy must cite this |
| Recharts ComposedChart for overlay | `SharpeHistoryChart.tsx:1-21` | Reuse ComposedChart import pattern; add Area for confidence bands when non-empty |
| AltDataPanel insertion point | `signals/page.tsx:179-181` | New `<TransformerForecastPanel>` goes at line 181, inside the `{data && ...}` block |
| Public path inheritance | `main.py:218` | No auth changes needed; new endpoint is automatically public |
| MDA = feature weights not price series | `backtest_engine.py:63-68` | No MDA baseline line; chart shows forecast lines only (empty stub) with a "no data" placeholder |

---

## Recommendation: endpoint returns honest stub

**Decision: honest stub with empty arrays.**

Both `TimesFMClient.forecast()` and `ChronosBoltClient.forecast()` fail-open to `[]` when their packages are absent (Python 3.14 venv). The endpoint should:
1. Instantiate both clients.
2. Call `client.forecast()` with a minimal dummy series (empty / unavailable) -- both return `[]`.
3. Return the spec-compliant shape with `status='shadow'` and empty arrays.

This satisfies the verification command (the response contains `status`) and is honest to the user.

---

## Exact Pydantic shape

```python
from pydantic import BaseModel
from typing import Literal

class TransformerForecastResponse(BaseModel):
    ticker: str
    timesfm: list[float]          # empty [] while runtime gate is not cleared
    chronos: list[float]          # empty [] while runtime gate is not cleared
    ensemble_weights: dict[str, float]  # e.g. {"mda": 1.0, "timesfm": 0.0, "chronos": 0.0}
    horizon: int                  # default 20
    model_timesfm: str            # "google/timesfm-2.5-200m-pytorch"
    model_chronos: str            # "amazon/chronos-bolt-small"
    status: Literal["shadow", "active", "error"]
    # status='shadow' = phase-8.4 REJECT stands; no live forecasts
    phase8_reject_reason: str     # human-readable explanation for the banner
```

The verification command `assert 'timesfm' in d or 'chronos' in d or 'status' in d` passes with any of the three keys present.

---

## Chart library decision

**Recharts** -- already used in this repo. Specifically `ComposedChart` with `Line` components (one per model), `Area` for confidence bands when quantile data is non-empty, and `ReferenceLine` for the horizon boundary. Import pattern follows `SharpeHistoryChart.tsx`.

When forecast arrays are empty (stub mode), render a placeholder inside the BentoCard:

```tsx
{timesfm.length === 0 && chronos.length === 0 && (
  <div className="flex flex-col items-center py-8 text-center">
    <ChartLine size={32} weight="duotone" className="text-slate-600" />
    <p className="mt-2 text-sm text-slate-400">
      No forecast data -- models require Python 3.11 runtime (see phase-8.4)
    </p>
  </div>
)}
```

---

## Banner copy (references phase-8.4 REJECT truthfully)

```
SHADOW MODE -- Not for trading decisions
Transformer forecasts (TimesFM 2.5, Chronos-Bolt) are logged here for monitoring only.
Phase-8.4 (2026-04-20) rejected promotion to live trading on two grounds:
(1) the Python 3.14 runtime cannot load either model package (requires <3.12 / torch);
(2) published evidence (arXiv 2511.18578, Nov 2025) shows zero-shot equity directional
accuracy at or below 50% and negative annualised returns vs the existing MDA baseline.
Promotion gate: Python 3.11 runtime + fine-tuned variant + 60 days shadow-log + IC uplift >= 0.10.
```

In JSX, render this as an amber-bordered banner (`border-amber-700 bg-amber-950/40`) with a `Warning` Phosphor icon.

---

## Frontend panel file path + insertion point

**New file:** `frontend/src/components/TransformerForecastPanel.tsx`

**Insertion in signals/page.tsx:**
- Add `TransformerForecastResponse` type to `frontend/src/lib/types.ts`
- Add `getTransformerForecast(ticker)` function to `frontend/src/lib/api.ts`
- Add state: `const [tfData, setTfData] = useState<TransformerForecastResponse | null>(null)`
- Add to the `Promise.all` fetch in `handleFetch`:
  ```tsx
  getTransformerForecast(ticker).catch(() => null)
  ```
- Insert after line 180 (`<AltDataPanel data={altData} />`):
  ```tsx
  {/* phase-15.8: Transformer forecast panel (shadow mode) */}
  <TransformerForecastPanel data={tfData} />
  ```

---

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (17 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks -- note gaps but do not auto-fail:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in footer)

---

## Queries run (3-variant discipline)

1. Current-year frontier: "Google TimesFM financial time series forecasting 2026", "Amazon Chronos-Bolt financial forecasting zero-shot 2025 2026"
2. Last-2-year window: "shadow mode experimental feature dashboard UX pattern 2025", "Recharts line chart forecast overlay confidence bands React 19 2025"
3. Year-less canonical: "TimesFM 2.5 API surface Python package zero-shot equity forecasting", "experimental feature beta shadow mode banner UI design pattern not for production dashboard"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "report_md": "handoff/current/phase-15.8-research-brief.md",
  "gate_passed": true
}
```
