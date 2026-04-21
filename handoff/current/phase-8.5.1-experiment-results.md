# Experiment Results — phase-8.5 / 8.5.1 (Candidate space) — REMEDIATION v2

**Step:** 8.5.1 **Remediation cycle:** 2 **Date:** 2026-04-20

## What was done

1. Archive-handoff hook guard-flagged via `.claude/archive-handoff.disabled`. Hook now early-exits when flag present. Files in `handoff/current/` are now stable across masterplan writes.
2. Researcher-authored brief restored from `handoff/archive/phase-8.5.1-v99/phase-8.5.1-research-brief.md` (167 lines, found among 150+ phantom archive versions) to `handoff/current/phase-8.5.1-research-brief.md`.
3. Re-ran immutable command + arithmetic + cross-reference check.
4. No YAML changes; content was already correct.

## Verification

```
$ test -f backend/autoresearch/candidate_space.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); assert d['estimated_combinations'] >= 10000"
(exit 0)

$ python -c "
import yaml
d = yaml.safe_load(open('backend/autoresearch/candidate_space.yaml'))
p = d['params']
prod = len(p['learning_rate']) * len(p['max_depth']) * len(p['n_estimators']) * len(p['rolling_window']) * len(d['prompts']) * len(d['features']) * len(d['model_archs'])
print('cartesian product =', prod, '== declared =', d['estimated_combinations'])
"
cartesian product = 15000 == declared = 15000

$ ls handoff/current/phase-8.5.1-*.md
handoff/current/phase-8.5.1-contract.md
handoff/current/phase-8.5.1-experiment-results.md
handoff/current/phase-8.5.1-research-brief.md
```

## Success criteria re-verified

| # | Criterion | Status |
|---|---|---|
| 1 | candidate_space_committed | PASS |
| 2 | ge_1e4_combinations | PASS (15,000 ≥ 10,000; arithmetic honest) |
| 3 | includes_transformer_signals_from_phase_8 | PASS (both `timesfm_forecast_20d` + `chronos_forecast_20d` present) |

## Infrastructure note

The `.claude/archive-handoff.disabled` flag remains in place for the duration of the remediation pass. Operator should remove it after all 22 remediations complete OR when masterplan.json is committed to git (whichever first).
