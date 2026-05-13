---
step: 25.F
slug: byte-identical-aliasing-regression-tests
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.F: Byte-identical regression test for aliasing detection

> Tier=simple. Main authored from direct inspection. The 25.B cleanup
> (cycle 85) removed the lite-path aliasing detection block; this step
> locks the removal with pytest regression tests so a future PR can't
> reintroduce it without breaking the suite.

---

## Three-variant search queries

1. **Current-year frontier**: `pytest regression dead-code reintroduction guard 2026`
2. **Last-2-year window**: `golden test fixture lock removal 2025`
3. **Year-less canonical**: `pytest assert key not in dict regression test`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| pytest docs | priors | Use `assert "key" not in dict` to lock removal |
| Hyrum's law | priors | Removed-behavior regression tests prevent silent reintroduction |
| 25.B cycle | 85 | Removed `is_lite_dup` block and `lite_path` field from frontend Signal interface |

## Recency scan

No paradigm shift in regression-test patterns 2024-2026.

## Design

Add two pytest tests to `tests/services/test_signal_attribution.py`:

1. **`test_lite_path_byte_identical_flagged`** -- construct an analysis dict
   where the RiskJudge rationale would be byte-identical to the Trader
   rationale (the historical lite-path collision case). Assert:
   - the RiskJudge signal IS present
   - its rationale matches the input
   - it has NO `lite_path` key
   - it has NO `cosmetic_match` or aliasing-detection key of any kind
2. **`test_full_path_distinct_rationale`** -- construct an analysis dict
   with distinct RiskJudge `reasoning` and Trader `trader_note` (the
   post-25.A normal case). Assert:
   - the RiskJudge signal is present with the expected rationale
   - the rationale is the verbatim `reasoning` value (no patching)
   - has the {agent, role, rationale, weight} 4-key shape

Then a `tests/verify_phase_25_F.py` that runs pytest on these specific
tests and asserts exit=0.

## Files to modify

| File | Change |
|------|--------|
| `tests/services/test_signal_attribution.py` | Add 2 regression tests |
| `tests/verify_phase_25_F.py` | NEW verifier (pytest -k) |

## Research Gate Checklist

- [x] Internal inspection at `tests/services/test_signal_attribution.py`
- [x] 25.B prior context (cycle 85)

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; regression-test pattern is canonical; design mechanical."
}
```
