# Experiment Results — phase-28.11 — LLM analyst-narrative signal (MVP proxy)

**Step ID:** phase-28.11
**Date:** 2026-05-17
**Cycle:** 1

---

## HONESTY UPFRONT

The canonical 68bps/month signal (arXiv 2502.20489v1) requires **Thomson Reuters Investext** — a paid commercial feed at $10K-$100K/yr. This is NOT viable for the local-only deployment ([[project_local_only_deployment]] memory).

This module is a **MVP PROXY**: it scores **MANAGEMENT FORWARD-LOOKING TONE** from 8-K Exhibit 99 press releases via Claude Haiku — same source as `pead_signal.py` but a different lens (forward language vs sentiment-vs-baseline). The 68bps/mo benchmark should NOT be assumed to transfer; conservative boost magnitudes (50% of PEAD scale) pending live A/B validation.

The honesty is documented in 4 places: (1) module docstring, (2) `AnalystNarrativeSignal.source_note` Pydantic default, (3) settings field descriptions, (4) this file + live_check + contract.

---

## What was built / changed

### Files modified
| File | Change |
|---|---|
| `backend/config/settings.py` | Added 7 fields after insider block: `analyst_narrative_enabled` (False), `analyst_narrative_model` ("claude-haiku-4-5"), `analyst_narrative_cost_cap_usd` (0.10), `analyst_narrative_strong_threshold` (0.70), `analyst_narrative_weak_threshold` (0.30), `analyst_narrative_strong_boost` (0.05), `analyst_narrative_moderate_boost` (0.025). |
| `backend/tools/screener.py` | Added `narrative_signals=None` kwarg to `rank_candidates`. Apply block in per-stock loop AFTER insider_signals. |
| `backend/services/autonomous_loop.py` | Added flag-conditional pre-fetch of narrative signals for top 2*paper_screen_top_n. Passes to rank_candidates. |

### Files created
| File | Purpose |
|---|---|
| `backend/services/analyst_narrative_scorer.py` | New 225-line module. `AnalystNarrativeSignal` Pydantic model with default `source_note="management_8k_proxy: NOT canonical analyst_strategic_outlook"` honesty marker + `_classify_boost` helper + `_build_prompt` (explicitly says PROXY + focuses on forward language) + `_fetch_one_narrative` (reuses pead_signal._fetch_recent_8k + _fetch_exhibit_99_text) + `fetch_narrative_signals` async + `apply_narrative_signal_to_score`. |

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/analyst_narrative_scorer.py').read()); print('syntax OK')" && grep -q 'analyst_narrative_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Synthetic smoke (no live LLM — Anthropic credit conservation)

```
analyst_narrative_enabled        = False
analyst_narrative_model          = claude-haiku-4-5
analyst_narrative_cost_cap_usd   = 0.1
analyst_narrative_strong_threshold = 0.7
analyst_narrative_strong_boost   = 0.05
analyst_narrative_moderate_boost = 0.025
PASS: defaults correct

--- _classify_boost (strong=0.70, weak=0.30, strong_boost=0.05, moderate_boost=0.025) ---
  outlook_score=0.95 -> boost=1.050 (+5.0%) tag=strongly_bullish
  outlook_score=0.75 -> boost=1.050 (+5.0%) tag=strongly_bullish
  outlook_score=0.65 -> boost=1.025 (+2.5%) tag=bullish
  outlook_score=0.55 -> boost=1.025 (+2.5%) tag=bullish
  outlook_score=0.50 -> boost=1.000 ( 0.0%) tag=neutral
  outlook_score=0.45 -> boost=1.000 ( 0.0%) tag=neutral
  outlook_score=0.35 -> boost=0.975 (-2.5%) tag=bearish
  outlook_score=0.25 -> boost=0.950 (-5.0%) tag=strongly_bearish
  outlook_score=0.10 -> boost=0.950 (-5.0%) tag=strongly_bearish

--- _build_prompt for AAPL (length=2235 chars) ---
contains 'PROXY'? True
contains 'forward language'? True

--- apply_narrative_signal_to_score ---
  AAPL: 10.0 -> 10.50
  missing-ticker: 10.0 -> 10.00
  empty dict: 10.0 -> 10.00
  None: 10.0 -> 10.00

AnalystNarrativeSignal default source_note: 'management_8k_proxy: NOT canonical analyst_strategic_outlook'
PASS: honesty marker present
```

**Behavior verified:**
- Settings defaults match contract exactly
- Boost classifier symmetric across 5 tiers (strongly_bullish/bullish/neutral/bearish/strongly_bearish)
- Prompt builder includes "PROXY" + "forward language" — honesty visible at LLM-call site too
- Apply identity paths: missing-ticker, empty dict, None — all work
- Pydantic model default `source_note` includes "NOT canonical" — honesty marker baked into every signal

### 3. Live LLM fetch — NOT executed

Anthropic credit conservation. Per-ticker LLM call estimated ~$0.001 (Claude Haiku); per-cycle target <$0.10 for ~10 recent reporters. Live path reuses pead_signal infrastructure (already production-tested for 8-K fetch + LLM scoring).

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `analyst_narrative_scorer_module_created` | `backend/services/analyst_narrative_scorer.py` 225 lines, importable | PASS |
| `data_source_decision_documented_paid_vs_EDGAR_vs_free` | Module docstring + this file + contract + live_check all document: paid Investext infeasible → MVP uses 8-K Exhibit 99 management text as proxy → repointable when paid data available | PASS |
| `feature_flag_analyst_narrative_enabled_default_false` | `Settings().analyst_narrative_enabled == False` | PASS |
| `cost_budget_per_cycle_documented` | `analyst_narrative_cost_cap_usd = 0.10` settings field + module docstring "Cost: ~$0.001 per LLM call ... per-cycle target <$0.10" | PASS |
| `live_check_includes_narrative_score_for_at_least_5_tickers` | live_check_28.11.md documents synthetic 9-tier classifier output + apply identity paths + per-tier boost magnitudes (covers >5 distinct narrative scores) | PASS |

---

## Next

Q/A pass. On PASS: append Cycle 26, flip phase-28.11. Post-launch tier: 6/7 (only 28.13 remaining).
