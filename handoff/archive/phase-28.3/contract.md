# Contract — phase-28.3 — GPR-triggered energy-sector tilt in macro_regime.py

**Step ID:** phase-28.3
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.3-research-brief.md` (Researcher subagent; `gate_passed: true`)
- 7 external sources read in full: policyuncertainty.com GPR page, Dario Caldara author page, ERL GPR vs supply/demand oil shocks, ERL geopolitical risk oil price bubbles, PMC GPR contagion across strategic sectors, KPMG 2025 top energy risks, ECB Economic Bulletin GPR & oil prices.
- 17 URLs collected, 10 snippet-only.
- Internal audit: macro_regime.py has `compute_macro_regime` (line 172+) producing `MacroRegimeOutput` with `sector_hints.overweight`/`underweight` (Pydantic). The cleanest extension is a post-LLM `_apply_gpr_tilt()` helper that injects "XLE" into `sector_hints.overweight` (deduped) when the latest GPR-Acts exceeds threshold.

## Hypothesis

The Caldara-Iacoviello GPR-Acts index — text-mined newspaper counts of GEOPOLITICAL EVENTS that already occurred (vs GPR-Threats which is forward) — spikes around military conflicts and major incidents. Since the US became a net oil exporter (late 2010s), GPR-Acts spikes ASYMMETRICALLY benefit US energy majors (XOM, CVX, COP, OXY) and the XLE sector ETF. Per the IMF GFSR 2025 + Caldara-Iacoviello AER 2022, this is the cleanest documented mechanism for the oil-majors reference case.

Implementation: add a thin GPR-Acts fetcher + post-LLM `sector_hints.overweight` injection of "XLE" when latest GPR-Acts exceeds a 90th-percentile threshold (calibrated from the rolling 5-year history; default threshold = 0.90 as a quantile, NOT an absolute value). Feature-flagged default OFF; identity behavior when OFF.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json::phase-28.steps[3].verification.success_criteria`)

1. `gpr_index_fetcher_implemented_with_caching`
2. `sector_tilt_branch_added_to_macro_regime`
3. `threshold_documented_in_audit_basis`
4. `live_check_shows_XLE_overweight_when_gpr_above_threshold`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'gpr|geopolitical' backend/services/macro_regime.py
```

Immutable live_check shape:
> "live_check_28.3.md: one cycle log showing GPR-Acts value + threshold + resulting sector_hints.overweight contents"

## Plan steps

1. **Settings additions** (after analyst_revisions block):
   - `gpr_signal_enabled: bool = Field(False, ...)` — opt-in
   - `gpr_signal_quantile: float = Field(0.90, ...)` — threshold as quantile of historical
   - `gpr_signal_cache_hours: int = Field(24, ...)` — cache TTL (matches publication cadence)
   - `gpr_signal_sector_etfs: str = Field("XLE", ...)` — comma-sep list of ETFs to overweight

2. **macro_regime.py additions**:
   - Module-level constants: `_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"`, `_GPR_CACHE_DIR`, `_GPR_CACHE_TTL_HOURS`
   - New helper `async def _fetch_gpr_acts() -> Optional[tuple[float, float, str]]` returning (current_GPRA, threshold_value, last_date) or None
   - Helper reads from cache when fresh; otherwise downloads .xls (tries .xls then .csv fallback); parses with pandas, extracts `GPRA` column, computes 90th-percentile threshold from trailing 60 months (5y).
   - Inside `compute_macro_regime()`, AFTER the LLM returns, when `settings.gpr_signal_enabled` is True: call `_fetch_gpr_acts()`; if `current_GPRA > threshold`, inject the configured ETFs into `parsed.sector_hints.overweight` (deduped, preserving order).
   - Add "phase-28.3" comment block explaining the asymmetry rationale + Caldara-Iacoviello + IMF GFSR citations + the 90th-percentile threshold choice.

3. **Run immutable verification** — must EXIT 0.

4. **Smoke test**:
   - Direct fetch: `_fetch_gpr_acts()` returns a recent GPRA value + computed threshold
   - Apply test: `_apply_gpr_tilt` injects "XLE" when above threshold; no-op when below

5. **Write `experiment_results.md`** with verbatim outputs.

6. **Write `live_check_28.3.md`** — GPR-Acts value, threshold, sector_hints before/after.

7. **Spawn Q/A**.

8. **On PASS** — append harness_log Cycle entry, flip status.

## References

- `handoff/current/phase-28.3-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #3 + Caldara-Iacoviello citation in Section C)
- `.claude/masterplan.json::phase-28.steps[3]`

## Risk / blast radius

- **Default OFF** — `gpr_signal_enabled = False` keeps current behavior.
- **No interference with FRED/Claude regime call** — GPR tilt is post-LLM, deterministic injection. If LLM call fails, no GPR tilt happens (the fallback path returns regime='unknown' and skips GPR).
- **Cost** — single Excel download per 24h cache window (~few-hundred-KB). Zero LLM cost on the GPR side.
- **License** — CC-BY 4.0 (per Researcher); document attribution in the file's module docstring.
- **External dependency risk** — matteoiacoviello.com could fail. Helper returns None and tilt is skipped.
