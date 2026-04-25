---
step: phase-16.38
title: Pre-flight masterplan verifier + SIPDO global-confirmation gate (#29, #55)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - scripts/meta/preflight_verify_masterplan.py (CLI)
  - backend/meta_evolution/directive_rewriter.py (+ should_apply_globally)
  - tests/meta_evolution/test_sipdo_global_confirm.py
---

# Sprint Contract -- phase-16.38

## Research-gate summary

`handoff/current/phase-16.38-research-brief.md`. tier=moderate, 6 in-full,
16 URLs, recency scan present, gate_passed=true. 7 internal files inspected.

## Bundled scope

| # | Task | Surface |
|---|------|---------|
| #29 | Pre-flight script (diff verification commands vs live code) | new ~180 LOC CLI |
| #55 | SIPDO global-confirmation gate | new ~50 LOC pure function + ~120 LOC tests |

## Concrete plan

### #29: scripts/meta/preflight_verify_masterplan.py

Mirrors `validate_cron_budget.py` CLI conventions exactly: positional
`path`, `--quiet` flag, `_check(label, ok, detail, *, quiet)` helper,
exit codes `0=pass / 1=broken refs / 2=fs error`.

Algorithm:
1. Walk `phases[].steps[]` in masterplan.json
2. For each step, extract `verification` field. Handle BOTH shapes:
   - String: `verification: "python -c '...'"`
   - Object: `verification: {command: "...", success_criteria: [...]}`
3. Strip `source .venv/bin/activate &&` prefix
4. Tokenize with `shlex.split(cmd, posix=True)`; wrap in try/except
   ValueError → emit WARN, not BROKEN
5. For each token:
   - **Path-like:** contains `/` OR has suffix in `{.py, .yaml, .json,
     .md, .sh, .tsv, .csv, .mjs}` AND does not start with `-` →
     check `Path(token).exists()` (relative to repo root)
   - **Import string:** if token contains `from X.Y import` or
     `import X.Y` → extract module name, call
     `importlib.util.find_spec(module)` → None means broken
6. Report `[BROKEN] step=X.Y: <detail>` to stderr, exit 1 if any.

### #55: should_apply_globally() in directive_rewriter.py

Append after `rewrite_directive()` (line 327), before `persist_version`.
3 new module-level constants near line 42:

```python
MIN_CONFIRMATIONS_FOR_GLOBAL_APPLY = 3
MIN_PREFIX_OVERLAP_RATIO = 0.80
MIN_PASS_RATE_FOR_GLOBAL = 0.67
```

Function:

```python
def should_apply_globally(
    recent_versions: list[DirectiveVersion],
    recent_qa_verdicts: list[str],
) -> bool:
    """phase-16.38 (#55) SIPDO global-confirmation gate.

    Pure function. Returns True iff:
    1. len(recent_versions) >= MIN_CONFIRMATIONS_FOR_GLOBAL_APPLY (3+)
    2. All versions have is_acceptable() == True (above floor)
    3. Pairwise SequenceMatcher ratio >= MIN_PREFIX_OVERLAP_RATIO (0.80)
       across all pairs (convergence check; SIPDO reconfirmation pattern)
    4. Verdict-weighted pass-rate >= MIN_PASS_RATE_FOR_GLOBAL (0.67)
       PASS=1.0, CONDITIONAL=0.5, FAIL=0.0
    """
```

Pure: no I/O, no BQ, no file writes. Caller (orchestrator) decides
whether to surface the proposal to HITL.

### Test file

`tests/meta_evolution/test_sipdo_global_confirm.py` (~120 LOC, 8 tests):
1. `test_below_min_confirmations_returns_false`
2. `test_unacceptable_version_in_set_returns_false`
3. `test_diverging_versions_below_overlap_returns_false`
4. `test_converging_versions_above_overlap_returns_true`
5. `test_pass_rate_below_floor_returns_false`
6. `test_all_pass_verdicts_returns_true`
7. `test_conditional_verdicts_weighted_correctly`
8. `test_constants_are_pinned` (regression guard for the 3 thresholds)

## Success Criteria (verbatim, immutable)

```
python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json --quiet && \
python -m pytest tests/meta_evolution/test_directive_rewriter.py tests/meta_evolution/test_sipdo_global_confirm.py -v
```

Plus:
- `preflight_script_exists`: file at `scripts/meta/preflight_verify_masterplan.py`.
- `preflight_runs_clean_or_reports`: exit 0 (clean) or exit 1 with
  meaningful broken-ref report. (Not failing on existing broken refs
  in masterplan is fine — the script is for FUTURE catches; existing
  broken refs are pre-existing tech debt, not regressions caused by
  this cycle.)
- `should_apply_globally_imports`: function importable + callable.
- `sipdo_constants_added`: 3 constants present at module level.
- `tests_pass`: 8/8 SIPDO tests + 8/8 existing directive_rewriter
  tests = 16/16 total.
- `no_other_regressions`: previous 55-test sweep still PASS.

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0
   (note: pre-flight script MAY exit 1 on existing broken refs;
   that's a known starting condition. The contract permits exit 1
   PROVIDED the broken-ref report is well-formed; Q/A should
   distinguish "pre-flight is broken" from "masterplan has known
   pre-existing broken refs").
2. Pre-flight script handles BOTH verification shapes (string + object).
3. Pre-flight script strips `source .venv/bin/activate &&` prefix.
4. Pre-flight script wraps shlex.split in try/except.
5. `should_apply_globally` is PURE (no I/O imports beyond stdlib
   `difflib`).
6. 3 new constants pinned in `test_constants_are_pinned`.
7. SequenceMatcher used for convergence (NOT a custom prefix-len
   algorithm).
8. Verdict weighting matches contract (PASS=1.0, CONDITIONAL=0.5,
   FAIL=0.0).
9. 16/16 directive_rewriter + sipdo tests PASS.
10. No mutation to existing `rewrite_directive()` body or
    `DirectiveVersion` dataclass.
