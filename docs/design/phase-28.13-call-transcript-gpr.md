# phase-28.13 — Design: Earnings-call NLP for firm-level GPR exposure

**Step:** phase-28.13 (Candidate Picker Expansion — last post-launch item)
**Date:** 2026-05-17
**Effort:** L (new 220-line module + honesty marker chain across 9+ surfaces)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## HONESTY UPFRONT

Fed 2025 (FEDS Note "Measuring Geopolitical Risk Exposure Across Industries"; 240K+ transcripts; R²=0.23) demonstrated CONTEMPORANEOUS relationship only — **NO forward predictability**. This is a **DEFENSIVE RISK FILTER**, NOT an alpha source. Honesty marker in Pydantic default `source_note`.

## Interface

```python
class GprExposureSignal(BaseModel):
    ticker: str
    exposure_tier: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    key_phrases: list[str]
    rationale: str
    source_note: str = "defensive_filter_only_per_Fed_2025_R2_0.23_contemporaneous_no_forward_alpha"

async def fetch_gpr_exposure_signals(tickers, model="claude-haiku-4-5", bucket_name="") -> dict[str, GprExposureSignal]: ...
def apply_gpr_exposure_to_score(base, ticker, sector, signals, exempt_sectors_csv, high_penalty) -> float: ...
```

## Penalty logic

- exposure_tier=HIGH AND sector NOT in exempt list → multiplier 0.97 (−3%)
- All other cases → identity

Exempt sectors (default `"Industrials,Energy"`) BENEFIT from elevated GPR per phase-28.3 US-as-net-exporter asymmetry:
- Industrials = defense contractors (LMT, NOC, RTX, GD)
- Energy = oil majors (XOM, CVX, COP, OXY)

Penalizing them would invert the documented mechanism.

## Infrastructure reuse

Reuses `earnings_tone.get_earnings_tone(ticker)` — Yahoo Finance scraping + GCS caching. No new API key. Same ClaudeClient as pead_signal + analyst_narrative_scorer.

## Feature flag

`call_transcript_gpr_enabled = False`.

## Honesty marker chain (6+ surfaces)

1. Module docstring HONESTY section at top
2. `GprExposureSignal.source_note` Pydantic default (travels WITH signal)
3. 4 settings field descriptions cite Fed 2025 + DEFENSIVE FILTER
4. `_build_prompt` body: "NOTE: this signal is used as a DEFENSIVE FILTER... be accurate, not generous"
5. Log format: "DEFENSIVE FILTER per Fed 2025 -- no forward alpha"
6. contract.md + experiment_results.md + live_check_28.13.md

## Test plan

5 immutable criteria evidenced. Q/A: 23 deterministic checks PASS. 1 NOTE on U+2014 em-dash in logger format — post-fixed to `--` per security.md ASCII-only rule.

## References

- `handoff/current/phase-28.13-research-brief.md`
- `.claude/masterplan.json::phase-28.steps[13]`
