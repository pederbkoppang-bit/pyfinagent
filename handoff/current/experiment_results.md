# Experiment results -- Step 75.8 (promotion-gate stub-fabrication refusal + governance divergence)

Date: 2026-07-23. Executor: Main (opus-tier step; session model Fable 5).
Every figure below was MEASURED before this document was written
(run-then-write); the commands and verbatim tails are in
`handoff/current/live_check_75.8.md`.

## What was built

**(a) gap6-01 -- gauntlet refuses to fabricate** (`scripts/risk/gauntlet.py`)
- Guard (i): `run()` raises `NotImplementedError` when `dry_run=False`,
  BEFORE any RNG/report work. Live CLI evidence: exit 1, traceback, and
  `handoff/gauntlet/baseline/report.json` untouched (mtime still Jun 10).
- Guard (ii), defense-in-depth: report writing factored into
  `_write_report()`, which raises `RuntimeError` unless
  `report.get("dry_run") is True` (explicit raise, not `assert`, so it
  survives `python -O`).
- Module docstring updated to document dry-run as the only implemented mode.

**(b) gap6-01 consumer side + gap6-10 -- promotion_gate**
(`scripts/risk/promotion_gate.py`)
- Stub-fingerprint rejection after report load: blocked (exit 1) when the
  non-skipped regime list is non-empty AND every entry has
  `bt_drawdown == drawdown` exactly. Skipped regimes filtered (as the
  evaluator does); the empty list is NOT fingerprinted (`all([])` trap
  guarded explicitly).
- BOTH `optimizer_best.json` writers (allocation-stage init at the former
  :150 and the gauntlet SHA-256 stamp at the former :165) now guarded
  behind `if not args.dry_run`, with the dry-run branch printing the
  would-be mutation (terraform-plan idiom). The post-write re-read is
  guarded for the fresh-deploy no-file case.
- Module docstring rewritten: --dry-run is now documented (truthfully) as
  strictly no-write.

**(c) gap3-02 -- governance divergence observability**
- NEW `backend/governance/divergence.py`: pure `compute_divergence()`
  (via the sanctioned lru-cached `limits_schema.load()`; normalizes
  governed fractions x100; `math.isclose` so float noise cannot
  manufacture a divergence) + never-raising `log_divergence_warnings()`.
- Wired into `backend/main.py` lifespan immediately after the existing
  limits-loader block, inside its own try/except, WARNING-only.
- Live measurement with CURRENT repo values: daily_loss_kill_switch
  settings 4.0% vs governed 2.0% -> divergent=true;
  trailing_dd_kill_switch 10.0% vs 10.0% -> divergent=false (the
  naive fraction-vs-percent comparison would have false-positived this).
- `handoff/current/governance_limits_divergence_75.md` written: all six
  limits.yaml entries vs measured runtime counterparts + drafted operator
  token GOV-LIMITS-DECIDE (advisory recommendation: BIND-SETTINGS).

## Files changed (measured: `git diff --stat HEAD` + `git status --short`)

Code (4 modified + 2 new):
```
 .claude/masterplan.json        |  20 ++++++++  (75.8.1 queued, status pending)
 backend/main.py                |  12 ++++-    (lifespan wiring + 1 pre-existing F401 removed)
 scripts/risk/gauntlet.py       |  41 +++++++++++++++--
 scripts/risk/promotion_gate.py | 101 +++++++++++++++++++++++++++++++----------
 NEW backend/governance/divergence.py
 NEW backend/tests/test_phase_75_promotion_gate.py
```
Handoff artifacts: contract_75.8.md, research_brief_75.8.md,
governance_limits_divergence_75.md (+ rolling contract.md /
research_brief.md syncs, and later this file + the critique + live_check).

**Criterion-5 boundary check (by file list): ZERO edits** to
`backend/backtest/gauntlet/evaluator.py` (thresholds), any kill-switch
enforcement code, DSR/PBO constants, or `backend/governance/limits.yaml`
-- none of those files appears in the diff.

Also in the working tree but NOT step changes (runtime-daemon appends
from the live backend, disclosed for diff honesty):
`handoff/.cycle_heartbeat.json`, `handoff/away_ops/auth_probe_last.json`,
`handoff/cycle_history.jsonl`, `handoff/kill_switch_audit.jsonl`, plus
the hook-managed `handoff/audit/*.jsonl` + `.claude/.archive-baseline.json`
and the untracked `handoff/archive/phase-75.7/` snapshot from the prior
step's close.

## Verification (immutable command)

```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py -q
-> 20 passed in 0.10s, exit 0
```

## Mutation matrix -- 11 applied, 11 killed, 0 survivors (measured)

Scripted (`scratchpad/mutation_matrix_75_8.py`): each mutation asserts
exactly-one substitution applied, runs the suite, restores byte-exact.

| # | Mutation | Killed |
|---|---|---|
| M1 | drop gauntlet NotImplementedError guard | yes |
| M2 | drop `_write_report` refusal | yes |
| M3 | drop stub-fingerprint rejection | yes |
| M4 | drop skipped-filter (fingerprint over ALL regimes) | yes |
| M5 | drop empty-list guard (`all([])` trap) | yes |
| M6 | unguard allocation-init writer | yes |
| M7 | unguard gauntlet-stamp writer | yes |
| M8 | drop unit normalization (x100) in divergence | yes |
| M10 | stub the settings read (claim-vacuity check) | yes |
| M11 | gut fingerprint equality to a tautology | yes |
| M12 | unwire the lifespan divergence call | yes |

M9 (fixture mutation -- one regime with bt != dd must NOT fingerprint)
is IN-SUITE as `test_single_divergent_regime_defeats_the_fingerprint`.
Anti-fixture-divorce: `test_real_gauntlet_dry_run_report_is_rejected_end_to_end`
feeds ACTUAL stub output (not a hand-built fixture) through the gate.
Per the cycle-131 rule this matrix licenses only "these 11 mutations were
killed", not a global no-vacuous-guards claim.

## Lint + syntax (measured)

- `uvx ruff check --select F821,F401,F811 <explicit 5-file list>` ->
  "All checks passed!", exit 0. One F401 was found first
  (`backend/main.py:346` `import asyncio as aio_lib`), PROVEN
  pre-existing (ruff flags the `git show HEAD:` copy too) and dead (zero
  `aio_lib` uses), removed per the 75.5 touched-file precedent.
- `ast.parse` OK on all four touched/new python files (criterion 6).

## Regression check (measured, 3 runs + worktree)

- Full `backend/tests/` suite, run twice in this tree:
  **10 failed / 1325 passed / 12 skipped / 5 xfailed / 1 xpassed** --
  IDENTICAL 10-test fail set both runs.
- The same 10, run in ISOLATION in this tree: **0 failed** -> the set is
  order/state-dependent under the full suite, not deterministic breaks.
- Symmetric-difference discipline: "failing in my tree but not at a clean
  HEAD worktree" = EMPTY (the HEAD worktree fails a SUPERSET, 13, of the
  same live-environment families: runtime-log freshness, 57.1
  reject-binding, 60.x flags, portfolio_swap -- cycle 133 documented this
  standing red set at 13).
- None of the 10 imports gauntlet/promotion_gate/divergence; the one file
  mentioning `backend/main.py` only source-scans for an untouched string
  (its passing test), while its failing test reads the RUNTIME log.
- Conclusion: zero regressions attributable to this diff.

## Out of scope -> queued (not prose-disclosed only)

- **75.8.1** (queued in masterplan this step, status pending): the SECOND
  gauntlet-report consumer `backend/autonomous_harness.py::promote_strategy`
  (:258-289) has no fingerprint/dry_run-label guard; step text specifies a
  shared predicate module both consumers import. From research correction
  #3 (wf_26a12896-e0c).

## Not verified live

- No backend restart performed: the lifespan WARNING will first appear on
  the next operator restart (the running process predates this change).
  The helper's behavior is proven by direct invocation + caplog tests.
- No live gauntlet/backtest run (historical_macro frozen; boundary).
- No UI surface touched.
