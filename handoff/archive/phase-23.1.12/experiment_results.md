---
step: phase-23.1.12
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_12.py'
---

# Experiment Results — phase-23.1.12

## Two operator-reported bugs, both fixed

### Bug 1 — Operator picks Sonnet/Opus, app silently runs lite-Claude

**Root cause:** `backend/services/autonomous_loop.py` Step 3 had a hardcoded `settings.lite_mode = True` (and corresponding restore) that overrode the operator's Settings choice. Comment said "Force lite mode for paper trading (cost control)". TradingAgents and FinCon literature explicitly warn against silent degradation of operator-configured behavior; the canonical pattern is **cap-as-circuit-breaker** (the `paper_max_daily_cost_usd` cap was already in place).

**Fix:**
- Removed the hardcoded mutation in Step 3 (and the restore at end of Step 4).
- Refactored `_run_single_analysis` to branch on `settings.lite_mode`:
  - `lite_mode=True` → `_run_claude_analysis` (lite 4-field path, cheap fast)
  - `lite_mode=False` → `AnalysisOrchestrator(settings)` with the operator's `gemini_model` + `deep_think_model` (FULL pipeline). Falls back to lite Claude if orchestrator fails.
- Lite return dict now carries a `"_path": "lite"` marker so the cycle's persist-to-`analysis_results` guard can correctly identify which path produced the analysis (full path's own `bq.save_report` writes its row inside the orchestrator).
- Added a `lite_mode` toggle to the Manage tab → Trading Settings card so the operator can opt back into lite mode if they want speed/cost over depth.

### Bug 2 — Cycle pill GREEN despite `paper_trades: unknown` and `paper_snapshots: unknown`

**Root cause:** `frontend/src/components/OpsStatusBar.tsx` worst-of-N aggregator at line 273-279 had `unknown` falling through to a separate `"unknown"` band, so a single-green `heartbeat` made the pill render as green even though two of three sources were unknown.

**Fix:** Per Google SRE / Azure WAF convention, collapse `unknown ⇒ amber` in the aggregator. Updated logic:
```tsx
const worst = bands.some((b) => b.band === "red")
  ? "red"
  : bands.some((b) => b.band === "amber" || b.band === "unknown")  // <— was just "amber"
    ? "amber"
    : bands.every((b) => b.band === "green")
      ? "green"
      : "amber";  // <— was "unknown"
```

Now: `[green, unknown, unknown] → "amber"`. The dots stay individually accurate; the label correctly draws the operator's eye to the degraded state.

## Files modified

| File | Change |
|---|---|
| `backend/services/autonomous_loop.py` | Removed `settings.lite_mode = True` and restore in Step 3/4. Refactored `_run_single_analysis` to branch on `settings.lite_mode` (full orchestrator with operator's models when False, lite Claude when True; lite as last-resort fallback either way). Added `"_path": "lite"` marker to lite return dict. Updated 2 persist-guard call sites from `if settings.lite_mode:` to `if analysis.get("_path") == "lite":`. |
| `frontend/src/components/OpsStatusBar.tsx` | Worst-of-N aggregator: `unknown ⇒ amber`. |
| `frontend/src/app/paper-trading/page.tsx` | NEW lite_mode toggle in Manage tab → Trading Settings card (full-width row at top of grid). Operator-visible knob with explanatory hint. |
| `tests/services/test_run_single_analysis_branch.py` | NEW (6 tests covering both branches, fallback paths, both-fail scenarios). |
| `tests/verify_phase_23_1_12.py` | NEW immutable verification script (regex-checks autonomous_loop.py + OpsStatusBar.tsx). |

## Verbatim verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_12.py
ok lite_mode override removed + branch path correct + _path marker + OpsStatusBar amber-on-unknown
exit=0
```

The script regex-asserts:
1. No active `settings.lite_mode = True` assignment (allowed inside backticks/comments)
2. No active `settings.lite_mode = original_lite` restore
3. `_run_single_analysis` body contains `if settings.lite_mode:` AND `AnalysisOrchestrator(settings)` (with operator's settings, NOT a Gemini-fallback override)
4. Lite return dict includes `"_path": "lite"` marker
5. `OpsStatusBar.tsx` worst-of-N collapses `unknown` into the amber clause

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/api/ tests/services/ -v --no-header -q
collected 160 items
... 160 passed in 3.74s
```

6 new + 154 prior = 160/160 tests pass. Zero regression across all 12 phase-23.1 cycles.

Frontend tsc: silent (0 errors).

## What changes for the operator at 09:30 ET tomorrow

| Scenario | Before this cycle | After this cycle |
|---|---|---|
| Operator has `gemini_model=claude-sonnet-4-6` and `lite_mode=False` | Cycle silently runs lite 4-field Claude prompt despite the model choice | Cycle runs FULL `AnalysisOrchestrator` with Sonnet 4.6 + Opus 4.6 — debate / bull / bear / risk-judge / bias all populated in `analysis_results` |
| Operator has `lite_mode=True` (explicit opt-in) | Cycle runs lite Claude (correct) | Cycle runs lite Claude (unchanged) |
| Heartbeat green + paper_trades unknown + paper_snapshots unknown | Pill shows GREEN (misleading) | Pill shows AMBER (correct degraded state) |
| Operator wants to flip lite_mode | Edit `backend/.env` + restart | Toggle in Manage tab → Trading Settings |

## Cost-and-budget honesty

The cap (`paper_max_daily_cost_usd: $2.00`) is now the real circuit breaker:
- Lite mode: ~$0.01/ticker × 5 candidates = $0.05/day → cap rarely hit
- Full mode: ~$0.50-2.00/ticker × 5 candidates = $2.50-10/day → cap hit after 1-2 candidates
- The cap was already in the code (line 220); previously it wasn't load-bearing because lite mode was forced. Now it's the ACTUAL guarantee that operator-chosen Opus doesn't bankrupt the daily budget.

Operator who picked Opus accepted that 1-2 candidates per cycle is what their budget supports. The remaining candidates get skipped with a clear log line: `"Daily cost cap ($2.0) reached, stopping analysis"`.

## Out of scope (per contract; Phase-2 follow-ups)

- Cost-aware model fallback: "if Opus exceeds cap, downgrade to Sonnet for the remaining candidates" (Phase 2)
- Async parallel analysis: current 1-ticker-at-a-time loop limits throughput (Phase 2)
- Per-ticker model selection: operator could want Opus for biotech, Haiku for utilities (Phase 2)
- New `cycle_health` source rows for `paper_trades` + `paper_snapshots` so they're not perpetually `unknown` (Phase 2 — the underlying compute_freshness helper just doesn't read those rows yet)

## Honest disclosure

- **Existing 11 paper-trading trades** booked under the old hardcoded lite path won't retroactively become full-pipeline reports. From tomorrow forward (assuming operator keeps `lite_mode=False`), every analyzed candidate gets the FULL treatment.
- **Cost surprise risk:** if operator has `lite_mode=False` AND `gemini_model=claude-opus-4-7` AND `paper_max_daily_cost_usd=$5.00`, that's ~3-5 full analyses per cycle = $1.50-10/cycle. The Manage tab now exposes both knobs (model selector via main Settings page; cap via Manage); operator's choice is fully transparent.
- **The `paper_trades` / `paper_snapshots` "unknown" status** is a real underlying issue — `compute_freshness` doesn't read those data sources yet. The pill correctly shows degraded; the underlying gap stays Phase 2.

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → restart backend + frontend
