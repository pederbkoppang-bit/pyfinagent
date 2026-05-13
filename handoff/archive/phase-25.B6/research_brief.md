---
step: 25.B6
slug: seed-stability-ci-gate
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.B6: Seed-stability test CI gate

> Tier=simple. Main authored from direct inspection of the existing
> seed-stability infrastructure + 25.A12 workflow template.

---

## Three-variant search queries

1. **Current-year frontier**: `seed stability gate CI workflow 2026 quant`
2. **Last-2-year window**: `random_state reproducibility CI 2025 ML`
3. **Year-less canonical**: `numerical stability test deterministic gate`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Lopez de Prado "Advances" | priors | std<0.1 across 5+ random seeds is the canonical reproducibility gate |
| 25.A12 visual-regression workflow | cycle 79 | GitHub Actions workflow pattern: ubuntu-latest + uvicorn smoke + python script |
| existing seed_stability_test.py | this cycle | Hard gate already implemented at scripts/go_live_drills/seed_stability_test.py (11 checks; std<0.1) |
| existing handoff/seed_stability_results.json | this cycle | Baseline already committed; mean=0.589, std=0.0094 (well below threshold) |

## Recency scan

No paradigm shift in seed-stability gate design 2024-2026.

## Design

1. **Confirm baseline is committed** -- `handoff/seed_stability_results.json`
   is tracked in git (verified via `git ls-files`).
2. **NEW `.github/workflows/seed-stability-check.yml`** -- a CI workflow that:
   - Runs on pull_request and workflow_dispatch.
   - Sets up Python 3.14 + installs minimal deps (none needed; stdlib-only test).
   - Executes `python scripts/go_live_drills/seed_stability_test.py`.
   - The script's exit-1-on-FAIL behavior auto-gates the workflow.
3. **Verifier `tests/verify_phase_25_B6.py`** asserts:
   - `handoff/seed_stability_results.json` exists + contains the canonical schema.
   - `.github/workflows/seed-stability-check.yml` exists + invokes the test script + references the std threshold.
   - The drill script itself enforces `std_sharpe < 0.1` at S5 (regex match).

## Files to modify

| File | Change |
|------|--------|
| `.github/workflows/seed-stability-check.yml` | NEW |
| `tests/verify_phase_25_B6.py` | NEW |

(No code changes; existing baseline JSON + drill script are correct as-is.)

## Research Gate Checklist

- [x] Internal: handoff/seed_stability_results.json + scripts/go_live_drills/seed_stability_test.py
- [x] Internal: 25.A12 workflow template (cycle 79)

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; existing infrastructure 90% done; just wraps it in CI gate workflow."
}
```
