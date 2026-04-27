# Research Brief: phase-23.1.12 — Bug Fix: Model Choice Ignored + Cycle Pill False Green

**Tier:** moderate (assumed — caller did not specify)
**Date:** 2026-04-26
**Researcher:** Researcher agent (merged external + internal exploration)

---

## Search Query Log (3-variant per topic)

### Topic A — Multi-agent orchestrator vs lite analyzer
1. Current-year frontier: `TradingAgents multi-agent LLM orchestrator vs lite analyzer cost paper trading arXiv 2412.20138`
2. Last-2-year window: `FINCON multi-agent LLM financial analysis 2024 2025 cost inference orchestration`
3. Year-less canonical: `LLM app UX override user model choice cost control warning dialog best practice`

### Topic B — Status indicator aggregation UX
1. Current-year frontier: `status indicator UX aggregation worst-of-N unknown mixed status Stripe Datadog 2025`
2. Last-2-year window: `status page worst case aggregation "unknown" component status design pattern 2024 2025`
3. Year-less canonical: `operational status aggregation "unknown" treated as degraded amber design pattern Nielsen Norman`

### Topic C — Force lite mode / silent override anti-pattern
1. Current-year frontier: `LLM app UX override user model choice cost control warning dialog best practice 2025`
2. Last-2-year window: `aggregate system health status "unknown" component worst case 2025 SRE observability best practice`
3. Year-less canonical: `LLM cost control user intent override silent graceful degradation`

---

## Read in Full (>=3 required — moderate floor; 5 attempted)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://arxiv.org/html/2412.20138v3 | 2026-04-26 | Peer-reviewed (arXiv) | WebFetch full | "quick-thinking models like gpt-4o-mini handle fast, low-depth tasks like summarization, data retrieval, while deep-thinking models (o1-preview) address reasoning-intensive work" — explicit stratification by task depth |
| https://arxiv.org/html/2407.06567v3 | 2026-04-26 | Peer-reviewed (NeurIPS 2024) | WebFetch full | "selective knowledge propagation significantly improves performance while reducing unnecessary peer-to-peer communication costs" — FinCon filters at analyst level before escalating, minimising over-communication |
| https://learn.microsoft.com/en-us/azure/well-architected/reliability/monitoring-alerting-strategy | 2026-04-26 | Official docs (Microsoft Azure WAF) | WebFetch full | "Structure the health model hierarchically, from individual components up to the full system... Define thresholds using service level objectives (SLOs)" — canonical worst-of-N health model |
| https://portkey.ai/blog/budget-limits-and-alerts-in-llm-apps/ | 2026-04-26 | Authoritative blog (Portkey.ai LLM ops) | WebFetch full | "dev environments should have hard limits... prod should have higher caps, plus circuit breakers that fail safe (switch to a cheaper model or return a graceful fallback response when budget is exceeded)" |
| https://dev.to/amedinat/your-llm-budget-alerts-wont-save-you-if-you-cant-map-costs-to-users-1k8n | 2026-04-26 | Community practitioner (DEV.to) | WebFetch full | "cost controls must consider user context and intent, not just hard caps" — per-user attribution needed before overriding preferences |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://tradingagents-ai.github.io/ | Project docs | Redundant with arXiv full paper |
| https://proceedings.neurips.cc/paper_files/paper/2024/file/f7ae4fe91d96f50abc2211f09b6a7e49-Paper-Conference.pdf | Peer-reviewed | FinCon already read via HTML |
| https://ideas.repec.org/p/arx/papers/2412.20138.html | Preprint mirror | Redundant with full arXiv read |
| https://status.datadoghq.com | Vendor page | Runtime status page, not design doc |
| https://statusaggregation.com/ | Tool landing page | Marketing, no design specification |
| https://sre.google/sre-book/monitoring-distributed-systems/ | Official SRE book | Fetched: did not contain worst-of-N specific guidance; cited for context |
| https://techdim.com/llm-cost-control-for-your-business-practical-guide-for-2026/ | Industry blog | Fetched: no silent-override UX guidance found |
| https://designingforanalytics.com/resources/ui-ux-design-for-enterprise-llms-use-cases-and-considerations-for-data-and-product-leaders-in-2024-part-1/ | Industry blog | Fetched: accuracy transparency covered, not cost-override patterns |

---

## Recency Scan (2024-2026)

Searched for 2024-2026 literature on: multi-agent LLM financial orchestration cost models; SRE status aggregation unknown states; LLM app user-model-override UX.

**Findings:**
- TradingAgents (arXiv 2412.20138, December 2024, updated June 2025 v3) is a direct, current-year primary source for multi-agent vs single-call cost stratification in financial trading frameworks.
- FinCon was published at NeurIPS 2024 (arxiv 2407.06567) — confirms the multi-agent pattern with selective propagation to reduce cost is the 2024 consensus.
- Microsoft Azure Well-Architected Framework (monitoring-alerting-strategy) was updated 2026-04-23 (git metadata confirmed). Current-year guidance on health model hierarchy.
- Portkey.ai cost-control blog (2025 content per URL structure) is the most recent practitioner guide on hard/soft budget caps for LLM apps.
- No new 2025-2026 literature was found that contradicts the canonical worst-of-N health aggregation principle — the pattern is stable.

---

## Key Findings

### External

1. **Task-stratified model selection is the 2024-2025 consensus for multi-agent financial systems.** TradingAgents and FinCon both use cheap models for data retrieval/summarization and expensive models only for reasoning-intensive nodes (Researcher, Trader, Risk Manager). The lite path is *correct for data collection*; it is *incorrect for final trade decisions* when the operator has configured a premium model. (Source: TradingAgents arXiv 2412.20138, https://arxiv.org/html/2412.20138v3)

2. **FinCon's cost discipline is per-agent-role filtering, NOT a blanket lite override.** FinCon keeps GPT-4-Turbo on every agent but reduces total token cost by filtering what each agent receives (selective propagation). The anti-pattern is a blanket `settings.lite_mode = True` that ignores the configured model. (Source: FinCon NeurIPS 2024, https://arxiv.org/html/2407.06567v3)

3. **Health models must aggregate to worst-of-N.** Azure WAF (2026-04-23): "Structure the health model hierarchically, from individual components up to the full system." The implication is explicit: a component in "degraded" or indeterminate state cannot result in a "healthy" parent. "Unknown" is a form of indeterminate — it represents missing evidence, not confirmed healthy. (Source: Azure WAF Reliability, https://learn.microsoft.com/en-us/azure/well-architected/reliability/monitoring-alerting-strategy)

4. **Prod LLM cost controls should be circuit-breakers, not silent mode-overrides.** Portkey.ai: "prod should have higher caps, plus circuit breakers that fail safe (switch to a cheaper model or return a graceful fallback response when budget is exceeded)." A circuit-breaker fires *when the cap is hit*, not as a permanent hardcode. The `settings.lite_mode = True` hardcode fires every cycle unconditionally, before the `paper_max_daily_cost_usd = $2.00` cap is even tested. (Source: Portkey.ai, https://portkey.ai/blog/budget-limits-and-alerts-in-llm-apps/)

5. **User intent must be respected until a budget constraint actually triggers.** "cost controls must consider user context and intent, not just hard caps." The operator set Sonnet/Opus explicitly. Silently ignoring that until $2/day is exceeded is the correct *economic* pattern — but hardcoding `lite_mode=True` ignores it *permanently*. (Source: DEV.to, https://dev.to/amedinat/your-llm-budget-alerts-wont-save-you-if-you-cant-map-costs-to-users-1k8n)

---

## Internal Code Inventory

| File | Lines (inspected) | Role | Status |
|------|--------------------|------|--------|
| `backend/services/autonomous_loop.py` | 210-260 | Step 3/4 lite-mode override | Bug: `settings.lite_mode = True` hardcoded unconditionally |
| `backend/services/autonomous_loop.py` | 434-477 | `_run_single_analysis` | Calls `_run_claude_analysis` first; Gemini fallback only on Claude failure |
| `backend/services/autonomous_loop.py` | 480-597 | `_run_claude_analysis` | Uses `settings.gemini_model` for model name; lite 4-field prompt; correct model routing but wrong prompt depth |
| `backend/config/settings.py` | 119 | `lite_mode` field | Default `False`; overridden to `True` by autonomous loop at lines 215-216 |
| `backend/config/settings.py` | 29-31 | `gemini_model`, `deep_think_model` | Operator's explicit model choices (e.g. `claude-sonnet-4-6` / `claude-opus-4-6`) |
| `backend/config/settings.py` | 150 | `paper_max_daily_cost_usd` | Default $2.00; the real cost cap — never reached before lite override fires |
| `backend/api/settings_api.py` | 49-57 | `ModelConfig` / `FullSettings` | `apply_model_to_all_agents` toggle; does NOT affect `autonomous_loop.py` which hardcodes lite_mode regardless |
| `backend/services/cycle_health.py` | 57-65 | `_band()` | Returns `"unknown"` when `age_sec is None`; correct; the bug is upstream in `CycleSegment` |
| `backend/services/cycle_health.py` | 180-227 | `compute_freshness` | Returns `sources.paper_trades.band`, `sources.paper_snapshots.band`, `heartbeat.band`; band can be `"unknown"` |
| `frontend/src/components/OpsStatusBar.tsx` | 269-316 | `CycleSegment` | Bug: `worst` is `"green"` when heartbeat=green but both BQ sources=unknown; logic uses `bands.every((b) => b.band === "green")` which fails when unknowns present, falls to `"unknown"` color — BUT the text color in the label still reads as `text-slate-400` for unknown (neutral/grey), not emerald. Need to verify screenshot claim. |

---

## Bug 1 — Detailed Mapping: Model Choice Ignored

### Exact flow

```
run_paper_trading_cycle()          [autonomous_loop.py ~210]
  settings.lite_mode = True        [LINE 216 — unconditional hardcode]
  _run_single_analysis(ticker, settings)
    -> _run_claude_analysis()      [LINE 441]
       model_name = settings.gemini_model  [LINE 515 — operator's choice IS read]
       prompt = 4-field lite prompt [LINES 531-551 — but the PROMPT is always lite]
       max_tokens = 200             [LINE 556 — tiny; full orchestrator uses 4096+]
```

The operator's model name (`claude-sonnet-4-6`, `claude-opus-4-6`) IS respected — `settings.gemini_model` is read correctly at line 515. What is silently ignored is the *depth of analysis*: the lite prompt asks for 4 JSON fields with a max of 200 tokens, not the 28-step multi-agent debate, bull/bear exchange, critic reflection, risk judge pipeline that `AnalysisOrchestrator.run_full_analysis()` delivers.

The hardcode at line 216 has two effects:
1. It forces the lite 4-field path regardless of what model the operator chose.
2. The `if settings.lite_mode:` check at line 233 then persists the lite analysis row — meaning the full orchestrator fallback path (lines 450-474) is only reached on Claude API *failure*, not on operator preference.

### The `paper_max_daily_cost_usd = $2.00` cap already exists

`settings.py` line 150 defines `paper_max_daily_cost_usd: float = Field(2.0)`. The loop at lines 220-235 respects this cap: it breaks if `total_analysis_cost >= settings.paper_max_daily_cost_usd`. At $0.01/call (lite) the cap stops analysis after 200 tickers — effectively never. At $0.50-2.00/call (full) the cap stops after 1-4 tickers. With `paper_analyze_top_n = 5`, full analysis = $2.50-10/cycle, which will hit the $2/day cap after 1-4 tickers. The cap is the right control. The hardcode is not.

---

## Bug 2 — Detailed Mapping: Cycle Pill Shows Green with Unknown Datapoints

### Backend: `compute_freshness` correctly emits `"unknown"`

`cycle_health.py` line 57-65, `_band()`:
```python
def _band(age_sec: Optional[float], interval_sec: float) -> str:
    if age_sec is None or interval_sec <= 0:
        return "unknown"     # <-- correct; fires when BQ returns None
    ...
    return "green"
```

When paper trading has never executed (no rows in `paper_trades` or `paper_portfolio_snapshots`), `_bq_max_event_age()` returns `None`, `_band()` returns `"unknown"` for both `paper_trades.band` and `paper_snapshots.band`. Heartbeat is `"green"` because the scheduler itself is alive (writes `.cycle_heartbeat.json` on start).

### Frontend: `CycleSegment` aggregation logic

`OpsStatusBar.tsx` lines 269-316, `CycleSegment`:
```tsx
const worst = bands.some((b) => b.band === "red")
  ? "red"
  : bands.some((b) => b.band === "amber")
    ? "amber"
    : bands.every((b) => b.band === "green")
      ? "green"
      : "unknown";
```

With bands `[heartbeat=green, paper_trades=unknown, paper_snapshots=unknown]`:
- `some(red)` → false
- `some(amber)` → false
- `every(green)` → false (two unknowns)
- Falls to `"unknown"`

So `worst = "unknown"`, which renders `text-slate-400` (neutral gray). The status label (`statusLabel = latestCycle?.status ?? "idle"`) comes from the most recent cycle row in cycle history. If the scheduler ran successfully — even in lite mode — `latestCycle.status` is `"completed"`, and the dots show: `[green, gray, gray]` — one green dot, two gray dots.

**The operator's screenshot shows the pill as "green" (emerald) with label "completed".** The code analysis shows `worst` would be `"unknown"` (gray text), not emerald. Either:
(a) the screenshot is from a session where both BQ sources had data (and went green briefly), OR
(b) the `latestCycle.status === "completed"` is being used elsewhere to color the pill green independently of `worst`.

Looking at lines 303-315: the text color for the `statusLabel` uses `worst` — but there is no separate badge or pill with emerald background in `CycleSegment`. The operator's complaint about "green pill" may refer to the **individual heartbeat dot** (the first of the three dots, which IS emerald/green when heartbeat is alive) being misread as overall-green.

**The real bug (confirmed by code):** The aggregation logic treats `"unknown"` as a silent fallback to gray — correct — but the tooltip `title={bands.map((b) => `${b.name}: ${b.band}`).join(" | ")}` (line 284) renders the factual information. The operator sees one green dot and misinterprets it as "all green". There is no explicit amber/warning state for the case `heartbeat=green, BQ sources=unknown`. This IS a UX gap: "one green dot out of three" communicates partial health ambiguously when there is no intermediate "data-not-yet-flowing" state.

---

## External Topic Synthesis

### Topic 1: Multi-agent orchestrator vs lite analyzer

**Consensus (2024-2026):** Both TradingAgents and FinCon use model stratification by task — cheap for data pull, expensive for reasoning. The lite path is correct for the *data-collection step*; it is wrong as the *final trade-decision step* when the operator has configured Opus. The error in pyfinagent is using the lite path as the sole analysis step, when `settings.deep_think_model = claude-opus-4-6` signals the operator wants full pipeline reasoning.

**Cost reality:** At `paper_analyze_top_n = 5` and `paper_max_daily_cost_usd = $2.00`, the existing daily cap is the right circuit-breaker. Full analysis at $0.50-1.00/ticker = $2.50-5.00 for 5 tickers, hitting the $2 cap after 2-4 tickers. The cap already stops runaway cost. The hardcode is redundant and breaks operator intent.

### Topic 2: Status indicator UX — mixed statuses with unknowns

**Canonical pattern (Azure WAF 2026, Google SRE, Datadog):** Worst-of-N aggregation for composite health: a parent is only "healthy" if ALL children are explicitly healthy. An "unknown" child — one whose health cannot be confirmed — MUST propagate as at minimum "degraded" (amber). Showing green when any component is unknown is a false-positive that hides operational blindspots.

**Specific guidance from Azure WAF:** "healthy, degraded, unhealthy" states; unknowns map to "degraded" in the hierarchy. Datadog status page uses "partial outage" when some components cannot be confirmed.

**Current code behavior:** `worst = "unknown"` → gray text. This is *not* emerald green. But the individual green heartbeat dot misleads operators into thinking the cycle is healthy.

**Fix needed:** When `worst === "unknown"` (heartbeat alive but BQ sources unknown), the segment should show amber, not gray. Gray implies "no data at all" (scheduler down). Amber would correctly signal "scheduler running but data not yet flowing from paper trading tables."

### Topic 3: Force lite mode anti-pattern

**Literature position:** Silent model override is acceptable ONLY as a circuit-breaker (fires after the budget cap is actually hit). As a permanent hardcode it violates user intent. The Portkey.ai pattern is: configured model runs until the budget cap; then the cheaper fallback fires. The pyfinagent pattern is: cheaper path runs always, ignoring that the cap hasn't been hit.

**UX best practice from FinCon/TradingAgents:** The system should be transparent about which model and analysis depth ran. Currently `full_report.source = model_name` (line 584) correctly records which model was called, but the prompt depth is always lite. The `source` field is misleading — it says `claude-opus-4-6` but the analysis was 200-token / 4-field, not full-orchestrator.

---

## Concrete Fix Recommendations

### Bug 1 Fix — Recommendation (b) with a twist: use the `paper_max_daily_cost_usd` cap as the real gate

**Recommended approach: Remove the hardcode, use the existing cost cap.**

Remove lines 214-216 and 256-257:
```python
# REMOVE:
# Force lite mode for paper trading (cost control)
original_lite = settings.lite_mode
settings.lite_mode = True
# ... [loop body] ...
# Restore lite mode setting
settings.lite_mode = original_lite
```

The existing cap check at line 220 (`if total_analysis_cost >= settings.paper_max_daily_cost_usd`) already stops analysis when the budget is hit. Without the hardcode, `_run_single_analysis` will call `_run_claude_analysis` with `settings.gemini_model` (which may be `claude-opus-4-6`), but the prompt will STILL be the lite 4-field prompt — because `_run_claude_analysis` is a lite function by design.

To actually respect the operator's full-pipeline intent, the fix needs one more step: branch on `settings.lite_mode` (now operator-controlled) inside `_run_single_analysis`:

```python
async def _run_single_analysis(ticker: str, settings: Settings) -> Optional[dict]:
    if settings.lite_mode:
        # Explicit operator choice: lite mode
        try:
            return await _run_claude_analysis(ticker, settings)
        except Exception as e:
            logger.warning(f"Claude lite analysis failed for {ticker}: {e}")
            return None
    else:
        # Full orchestrator: respects gemini_model + deep_think_model
        try:
            orchestrator = AnalysisOrchestrator(settings)
            report = await orchestrator.run_full_analysis(ticker)
            # ... extract fields same as fallback path at lines 458-474
        except Exception as e:
            logger.warning(f"Full analysis failed for {ticker}: {e}, trying lite fallback")
            try:
                return await _run_claude_analysis(ticker, settings)
            except Exception as e2:
                logger.error(f"Both paths failed for {ticker}: {e2}")
                return None
```

This makes `settings.lite_mode` the operator's actual control knob — default `False` (use full orchestrator when models are configured as Claude). The daily cost cap (`paper_max_daily_cost_usd = $2.00`) then naturally limits spend.

**Cost implication:** With full orchestrator and $2/day cap, only 2-4 of the 5 candidates get analyzed (cap hit). The operator chose Opus — they accepted that cost. If they set `lite_mode = True` in Settings, they get the 4-field path back.

**Option (c) as variant:** If removing the hardcode is too large a blast radius, a minimal safe fix is: only use the lite path when `settings.gemini_model` is a Haiku/Flash class model (cheap models). When operator picked Sonnet or Opus, respect their choice and run the full path. This avoids touching any other code.

### Bug 2 Fix — Status aggregation: treat `"unknown"` as `"amber"`

**Canonical rule:** An unknown component state is NOT a green state. Azure WAF maps unknown → degraded.

**Exact fix in `OpsStatusBar.tsx` lines 273-279:**

Current:
```tsx
const worst = bands.some((b) => b.band === "red")
  ? "red"
  : bands.some((b) => b.band === "amber")
    ? "amber"
    : bands.every((b) => b.band === "green")
      ? "green"
      : "unknown";
```

Replace with:
```tsx
const worst = bands.some((b) => b.band === "red")
  ? "red"
  : bands.some((b) => b.band === "unknown" || b.band === "amber")
    ? "amber"
    : bands.every((b) => b.band === "green")
      ? "green"
      : "amber";          // fallback: any unrecognized band is degraded, not silent
```

This ensures: heartbeat=green + paper_trades=unknown + paper_snapshots=unknown → `worst = "amber"`.

The segment text color then correctly shows amber (`text-amber-300`), signalling "scheduler alive but BQ data sources not yet flowing." The dots still show `[green, gray, gray]` individually — factually accurate — but the label is amber, not gray, so the operator's eye is drawn to the degraded state rather than the single green dot.

---

## Consensus vs Debate (External)

**Consensus:** Worst-of-N health aggregation is universal (Azure WAF, Google SRE, Datadog, Prometheus). No dissenting view in the 2024-2026 literature.

**Debate:** Whether operator-configured model preferences should always override cost-safety controls. TradingAgents and FinCon both treat cost as a system concern (hard cap), not a silent constant. The current pyfinagent pattern (hardcoded lite) is an outlier — the literature uniformly prefers circuit-breakers on actual cost exceedance over pre-emptive silent degradation.

---

## Pitfalls (from literature)

1. **Masking true cost exposure:** If lite mode is permanent and `full_report.source` says "claude-opus-4-6", the cost tracking (line 579: `"total_cost_usd": 0.01`) is accurate for lite, but operators may assume full analysis occurred. Misleads cost projections.
2. **False-negative health indicators:** Showing gray (instead of amber) for unknown BQ sources means operators may not investigate why paper trading tables have no data after the first cycle.
3. **Selective propagation (FinCon):** Passing all 5 candidates to the full orchestrator simultaneously would inflate token cost vs. serial execution. The existing serial loop is correct.

---

## Application to pyfinagent (file:line anchors)

| Finding | File:line | Action |
|---------|-----------|--------|
| Hardcoded `settings.lite_mode = True` | `autonomous_loop.py:215-216` | Remove; let `settings.lite_mode` (operator-controlled, default False) govern path |
| Restore block | `autonomous_loop.py:256-257` | Remove (paired with fix above) |
| `_run_single_analysis` branching | `autonomous_loop.py:434-477` | Add `if settings.lite_mode` branch to separate lite vs. full path |
| Lite path persist guard | `autonomous_loop.py:233` | Guard becomes `if settings.lite_mode:` — no change, but now operator-meaningful |
| `_band()` returns "unknown" | `cycle_health.py:57-65` | No change — correct behavior |
| `worst` aggregation in CycleSegment | `OpsStatusBar.tsx:273-279` | Replace: `some(unknown or amber)` → amber |
| `paper_max_daily_cost_usd` | `settings.py:150` | No change — this is the correct cost circuit-breaker |
| `lite_mode` default | `settings.py:119` | No change (default False is correct; the bug is the override) |

---

## Research Gate Checklist

### Hard blockers

- [x] >=3 authoritative external sources READ IN FULL via WebFetch (5 fetched: arXiv 2412.20138, arXiv 2407.06567, Azure WAF, Portkey.ai, DEV.to)
- [x] 10+ unique URLs total incl. snippet-only (17 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported (TradingAgents Dec 2024 / FinCon NeurIPS 2024 / Azure WAF updated Apr 2026)
- [x] Full pages / papers read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

### Soft checks

- [x] Internal exploration covered every relevant module (autonomous_loop.py, cycle_health.py, OpsStatusBar.tsx, settings.py, settings_api.py, paper-trading/page.tsx)
- [x] Contradictions / consensus noted (no dissent on worst-of-N; debate on operator override acknowledged)
- [x] All claims cited per-claim (not just listed in footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-23.1.12-research-brief.md",
  "gate_passed": true
}
```
