---
step: phase-16.16
title: Backend correctness re-verification (pytest+AST+migration+health)
cycle_date: 2026-04-25
harness_required: true
retrospective: false
forward_cycle: true
parent_phase: phase-16 (Full-application end-to-end UAT)
---

# Sprint Contract -- phase-16.16

## Research-gate summary

Source: `handoff/current/phase-16.16-research-brief.md` (172 lines).

JSON envelope (verbatim):
```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.16-research-brief.md",
  "gate_passed": true
}
```

Floor met: 6/5 sources in-full, 16/10 URLs, recency scan present.

## Hypothesis

The backend is in a known-good state today (2026-04-25). Pytest from
repo root will pass cleanly because the `backend/calendar/` stdlib-shadow
is mitigated by repo-root invocation. AST audit across all
`backend/**/*.py` will be clean (no syntax drift since 1122a021).
The strategy_deployments BQ migration --verify will continue to pass
(verified live earlier today). `/api/health` will return 200.

## Success Criteria (verbatim from .claude/masterplan.json step 16.16)

Verification command (immutable):
```
source .venv/bin/activate && python -m pytest backend/tests/ -q && python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('backend/**/*.py', recursive=True) if '__pycache__' not in f]; print('syntax_ok')" && python scripts/migrations/create_strategy_deployments_view.py --verify && curl -sS http://127.0.0.1:8000/api/health
```

- pytest_all_pass
- ast_clean_all_backend_py
- bq_migration_verify_pass
- backend_health_200

## Plan steps

1. Run the immutable verification command verbatim (chained with `&&`).
2. Capture stdout + exit code per stage.
3. If any stage fails, isolate the regression -- don't paper over.
4. Spawn Q/A with the evidence pack.

## What Q/A must audit

1. Verification command was run verbatim (no edits to flags/paths)
2. All 4 success criteria met or honestly disclosed if not
3. No code changes this cycle (read-only verification)
4. Brief gate envelope is truthful (sources real, not hallucinated)
5. Research-gate file exists at `phase-16.16-research-brief.md` (step-specific, not rolling)

## References

- `handoff/current/phase-16.16-research-brief.md`
- `backend/main.py` -- `/api/health` route
- `backend/tests/` -- 21 test files
- `scripts/migrations/create_strategy_deployments_view.py`
- `CLAUDE.md` -- harness protocol
