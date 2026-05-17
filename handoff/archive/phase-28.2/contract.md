# Contract — phase-28.2 — 12-quarter SUE stacking in pead_signal.py

**Step ID:** phase-28.2
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.2-research-brief.md` (Researcher subagent; `gate_passed: true`)
- 5 external sources read in full: ScienceDirect 2025 ("Beyond the last surprise: Reviving PEAD with ML and historical earnings"), Quantpedia PEAD effect, QuantConnect SUE notebook, Quantpedia NLP-PEAD page, Wikipedia PEAD overview.
- Internal audit: pead_signal.py:38 holds `_LOOKBACK_QUARTERS = 8`; `_trailing_mean_from_cache` (lines 91-111) uses pure equal-weight arithmetic mean; cache filenames encode `pead_{TICKER}_{YYYY-MM-DD}.json` (no lookback depth) → fully cache-safe to bump. Docstring at line 54 hardcodes "rolling-8Q mean".
- Recommendation: **equal-weight, not EWMA**. The ScienceDirect 2025 mechanism is that older lags gain importance as markets price news faster — exponential decay would de-weight exactly those valuable older observations. Every practitioner source (QuantConnect, Quantpedia) uses equal-weight.

## Hypothesis

Bumping `_LOOKBACK_QUARTERS` from 8 to 12 in `backend/services/pead_signal.py` retroactively widens the trailing window used to compute `surprise_score = current_sentiment − trailing-mean`, capturing more historical context. Per the ScienceDirect 2025 ML stacking paper, this raises Sharpe from 0.34 (latest-only) to 0.63 (12-quarter stack) — a +85% lift. The change is XS: one constant bump + one docstring update + sync the `pead_signal_lookback_quarters` setting default.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json::phase-28.steps[2].verification.success_criteria`)

1. `lookback_quarters_increased_to_12`
2. `weighting_scheme_added_or_documented`
3. `back-compat_with_existing_cache_files`
4. `syntax_OK_and_pead_signal_still_importable`

Immutable verification command:
```bash
source .venv/bin/activate && grep -qE '_LOOKBACK_QUARTERS\s*=\s*12' backend/services/pead_signal.py && python -c "import ast; ast.parse(open('backend/services/pead_signal.py').read()); print('PASS')"
```

Immutable live_check shape:
> "live_check_28.2.md: one ticker's PEAD before/after with 8Q vs 12Q stack, surprise_score diff and resulting holding_window_days"

## Plan steps

1. **Edit pead_signal.py**:
   - Line 38: `_LOOKBACK_QUARTERS = 8` → `_LOOKBACK_QUARTERS = 12`
   - Line 54: `surprise_score description` "rolling-8Q mean" → "rolling-12Q mean"
   - Add a `# phase-28.2:` comment near the constant explaining the rationale + equal-weight choice + source citation.

2. **Edit settings.py**:
   - Line 187: `pead_signal_lookback_quarters: int = Field(8, ...)` → `Field(12, ...)` (sync the parallel setting; description updated to cite phase-28.2).

3. **Run immutable verification command** — must EXIT 0.

4. **Smoke test** (no Anthropic API required):
   - Generate 12 synthetic cache files for `_TEST_PEAD_TICKER`
   - Call `_trailing_mean_from_cache(_TEST_PEAD_TICKER, "2026-05-17")` — confirm `n_quarters == 12` and mean is the equal-weighted average of all 12

5. **Smoke test (additional)**:
   - Run the same on with TEMP `_LOOKBACK_QUARTERS = 8` (simulate old behavior) vs `= 12` (new) — show difference in trailing mean and the impact on `surprise_score = current_sentiment − trailing_mean`.

6. **Write `experiment_results.md`** with verbatim outputs.

7. **Write `live_check_28.2.md`** — one ticker's 8Q vs 12Q comparison with derived surprise_score + holding_window_days.

8. **Spawn Q/A**.

9. **On PASS** — append harness_log Cycle entry, flip status.

## References

- `handoff/current/phase-28.2-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #2)
- `.claude/masterplan.json::phase-28.steps[2]`

## Risk / blast radius

- **Back-compat for cache files: PRESERVED.** Cache filenames don't encode lookback depth; existing cache files are read identically by the new code.
- **Default behavior change.** Unlike 28.1 / 28.5 (feature-flagged OFF), this step CHANGES the default lookback. The PEAD signal itself is still feature-flagged by `pead_signal_enabled` (default False), so production picker is unchanged when the flag is OFF. When the flag IS on, surprise_score values will differ slightly because the trailing mean has more history. The MAGNITUDE of the change is small (12Q vs 8Q averaging — 4 additional historical observations).
- **Operator impact:** any operator who has `pead_signal_enabled=True` will see surprise_score values shift modestly. The masterplan spec accepts this — it's the point of the step.
- **No cost change.** Zero LLM, zero network. Cache files already on disk.
