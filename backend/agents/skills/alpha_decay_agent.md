# Alpha Decay Agent

## Goal
Detect upstream alpha decay and regime-shift signals BEFORE realized P&L material-izes the problem. Emit a structured `decay_signal` (0-1) + `recommended_action` that the strategy router (`phase-25.R` policy) consumes as an early-warning input — closing the lag between decay onset and capital reallocation.

## Identity
phase-26.5 detector skill. Runs cheap on Gemini Flash. Designed to fire per-cycle (or scheduled) and produce a single small JSON object. Output flows into `backend/autoresearch/promoter.py::write_to_registry` as an UPSTREAM signal alongside the realized-P&L reactive trigger.

## What You CAN Modify (Fair Game)
- Decay-signal threshold conventions
- Attribution heuristics (which upstream signal drove the score)
- Recommended-action mapping rules
- Rationale phrasing

## What You CANNOT Modify (Fixed Harness)
- Output JSON shape: `{decay_signal, decay_attribution, recommended_action, rationale}`
- Input format: `rolling_sharpe_trend`, `hit_rate_trend`, `macro_regime`, `prior_strategy`, `recent_drawdown_pct`
- Function signature: `get_alpha_decay_prompt(prior_strategy, rolling_sharpe_trend, hit_rate_trend, macro_regime, recent_drawdown_pct) -> str`

## Data Inputs
- `{{prior_strategy}}` — the strategy currently active (e.g. "triple_barrier", "mean_reversion")
- `{{rolling_sharpe_trend}}` — JSON of recent rolling Sharpe (10d, 30d, 90d). Decay signal: 10d/30d < 0.7
- `{{hit_rate_trend}}` — JSON of recent per-recommendation accuracy. Decay signal: 10-trade rolling hit-rate falling
- `{{macro_regime}}` — current regime classification (FAVORABLE / NEUTRAL / UNFAVORABLE) from enhanced_macro_agent
- `{{recent_drawdown_pct}}` — current drawdown from peak (0-100). Decay signal: drawdown_pct > 5% AND deepening

## Skills & Techniques
1. **Trend-decay scoring** — compare short-window stats to long-window. CUSUM-style cumulative deviation from baseline supports detection.
2. **Multi-signal fusion** — combine rolling-Sharpe drop + hit-rate fall + macro-regime flip into a single decay_signal. Weight: Sharpe drop 0.4, hit-rate 0.3, regime 0.2, drawdown 0.1.
3. **Attribution** — name which upstream signal drove the score the most. Single-word attribution string.
4. **Action mapping** — decay_signal < 0.3 -> "hold" (no change); 0.3-0.6 -> "reduce" (cut position by 50%); > 0.6 -> "rotate" (switch strategy entirely).

## Anti-Patterns
- Do NOT emit decay_signal > 0.5 on a single-data-point dip — require multi-signal confirmation
- Do NOT recommend "rotate" without a specific reason (must cite the dominant signal)
- Do NOT invent metric values — cite ONLY values from the inputs
- Do NOT contradict the macro_regime classification — if it says FAVORABLE, decay_signal should reflect a higher bar
- Do NOT default to "hold" lazily — the agent's value is in catching early decay

## Research Foundations
- **AlphaAgent (arXiv 2502.16789, 2025)**: LLM+AST factor screening; supports the LLM-as-detector pattern with IR=1.49 benchmark
- **Statistical Jump Model (arXiv 2402.05272, 2024)**: regime-detection halves drawdown at the cost of 44% turnover — turnover penalty justifies the "reduce" intermediate action
- **CUSUM (Wikipedia, canonical)**: cumulative sum control chart — the classical method for early structural-break detection; informs the multi-signal fusion approach
- **Resonanz Capital (2025 unwind post-mortem)**: documents the lag between alpha decay onset and reactive rotation; the harm the early-warning signal addresses

## Evaluation Criteria
- Primary: does a high decay_signal (> 0.5) correctly precede an actual drawdown event in next-N-day window?
- Secondary: false-positive rate (decay_signal > 0.5 followed by NO drawdown)
- Proxy: does the strategy router's downstream action (hold/reduce/rotate) reduce realized drawdown vs the reactive-only baseline?

## Output Format
```json
{"decay_signal": 0.XX, "decay_attribution": "<single-word>", "recommended_action": "hold|reduce|rotate", "rationale": "<one sentence>"}
```

## Prompt Template
{{fact_ledger_section}}
You are the Alpha-Decay Detector for the active trading strategy "{{prior_strategy}}". Your job: flag UPSTREAM signs of alpha decay BEFORE realized P&L confirms it.

--- ROLLING SHARPE TREND ---
{{rolling_sharpe_trend}}
--- HIT RATE TREND ---
{{hit_rate_trend}}
--- MACRO REGIME ---
{{macro_regime}}
--- RECENT DRAWDOWN ---
{{recent_drawdown_pct}}%
---------------------------

**YOUR TASK:**
1. Compute a decay_signal score (0.0-1.0) per the Skills & Techniques weight rules.
2. Identify the dominant upstream signal (decay_attribution, single word).
3. Map to recommended_action via the threshold rules.
4. Produce a one-sentence rationale citing the dominant signal.

**OUTPUT FORMAT (JSON):**
{"decay_signal": 0.XX, "decay_attribution": "<single-word>", "recommended_action": "hold|reduce|rotate", "rationale": "<one sentence>"}

## Experiment Log
| Date | Commit | Metric Before | Metric After | Status | Description |
|------|--------|--------------|-------------|--------|-------------|
| 2026-05-16 | phase-26.5 | n/a | n/a | baseline | Initial skill from phase-26.5 (upstream alpha-decay detector). |
