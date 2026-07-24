# Experiment Results (DRAFT) -- Step 75.15: CI gates made real

**Executor**: Sonnet-4.6/high GENERATE delegate. CI-config + test edits ONLY --
no production code touched, no `.env` edits, no server started, no
masterplan/harness_log/commit/push. Boundary re-verified at handoff (below).

## 0. Two corrections to the contract's diagnosis (measured, not assumed)

The contract's A/B/C classification of the 10-red worklist was RIGHT on 8
of 10 items and WRONG on 2, both discovered by actually running the
failing tests with the fix applied and checking cause, not just symptom:

1. **`test_phase_23_2_15::test_phase_23_2_15_known_pass_scripts_still_pass`**
   was contracted as category-A "shells verify scripts, env-sensitive --
   mark or fix." Diagnosis: with `.venv` PATH sourced (mirroring CI), 5 of
   6 previously-"env-sensitive" sub-scripts already pass; only
   `verify_phase_23_1_22.py` still failed, which shells out to
   `tests/api/test_pause_resume_timeout.py`. That file patches
   `backend.api.paper_trading.BigQueryClient` (the class), but
   `resume_trading`/`get_kill_switch_state` call `get_bq_client()` (an
   `@lru_cache`-memoized singleton factory added `phase-75.9`,
   `backend/db/bigquery_client.py:1116`) -- the patch target went stale
   post-75.9 and became a no-op, letting the real cached BQ client's live
   data leak through (NAV 23896.61 instead of the mocked hang; `resp`
   without a `status_code` instead of 503). **This is test-mock drift, not
   a production regression** -- the production code already implements the
   documented 503 + NAV-to-0.0 degraded-mode behavior correctly (verified
   by reading `backend/api/paper_trading.py:480-569`). Fixed the patch
   target in `tests/api/test_pause_resume_timeout.py` (2 call sites) to
   `backend.api.paper_trading.get_bq_client`. Result:
   `verify_phase_23_1_22.py` now exits 0; `test_phase_23_2_15` needs **no
   requires_live marker at all** -- fully CI-safe as-is.
2. **`test_portfolio_swap::test_swap_framework_fills_zero_buy_gap`** was
   contracted as category-C "genuine drift/logic -- FIX, phase-70
   non-atomic-swap area." Diagnosis: `_make_settings()` in the test never
   sets `paper_swap_churn_fix_enabled`, so `Settings(**base)` falls back to
   whatever the environment provides for that field -- and the operator's
   untracked local `.env` has it `True` (code default is `False`,
   `backend/config/settings.py:350`). With `churn_fix_on=True` the swap
   denominator clamps to `max(abs(holding_score), 1.0)`; with it `False`
   (the CI/code default) the denominator is `max(abs(holding_score),
   0.01)`, changing which candidate clears the 25% delta bar. Forcing
   `PAPER_SWAP_CHURN_FIX_ENABLED=false` makes this test (and the other 3
   in the file) pass with **zero code changes**. **This is a THIRD
   instance of the exact operator-`.env`-pollution root cause already
   identified for `test_phase_57_1`/`test_phase_60_3`, not genuine
   logic/expectation drift.** No production or test fix needed; folded
   into the same CI-equivalent-env disclosure as the other 3.

Net effect: **zero requires_live marks were needed for either of these
two files** (down from the contracted 1 mark for 23.2.15's failure mode),
and the CI-equivalent local baseline needs a **third** env override
(`PAPER_SWAP_CHURN_FIX_ENABLED=false`), not just the two named in the
contract.

**No production bug was found or queued.** The one candidate that looked
like it might be a production regression (kill-switch degraded-mode via
`verify_phase_23_1_22.py`) was diagnosed as test-mock drift and fixed
within the test-edit boundary -- see item 1 above. No QUEUE-CANDIDATE
section follows because there is nothing to queue.

## 1. Worklist resolution (10 reds -> action + rationale)

| # | Test | Contract classification | Actual resolution |
|---|------|--------------------------|--------------------|
| 1 | `test_phase_23_2_10::watchdog_log_present_and_fresh` | A: mark requires_live | **Marked.** Live watchdog log exists on operator machine but is 1016h stale (process not currently running) -- genuine live-machine state; gitignored so CI would skip anyway, but the mark is correct regardless. |
| 2 | `test_phase_23_2_6::backend_log_has_skipping_buy_evidence` | A: mark requires_live | **Marked.** Live backend.log + newest rotation archive both have zero "Skipping BUY" occurrences today (an older 2026-06-12 archive has 56, not checked by the fallback) -- genuine live-log-state dependency. |
| 3 | `test_phase_23_2_9::backend_log_has_prewarm_evidence` | A: mark requires_live | **Marked.** Prewarm line only fires on backend boot; the live backend.log has zero occurrences in its current unrotated window (no recent restart) -- genuine live-uptime-state dependency. |
| 4 | `test_phase_23_2_15::known_pass_scripts_still_pass` | A: mark or fix (env-sensitive) | **Fixed, not marked** (see section 0.1). Root cause was a stale test mock-patch target unrelated to CI/live state; now fully green everywhere. |
| 5 | `test_phase_57_1::reject_binding_main_path_off_emits_on_blocks` | B: do not mark, .env pollution | **Confirmed, no change.** `paper_risk_judge_reject_binding` defaults False in code; operator `.env` has it True. |
| 6 | `test_phase_57_1::reject_binding_swap_path_off_emits_on_blocks` | B: same | **Confirmed, no change.** |
| 7 | `test_phase_57_1::off_identity_prompts_are_verbatim_constants` | B: same | **Confirmed, no change.** |
| 8 | `test_phase_60_3::flag_defaults_off` | B: do not mark, .env pollution | **Confirmed, no change.** `paper_data_integrity_enabled` defaults False in code; operator `.env` has it True. |
| 9 | `test_60_1::claude_code_rail_declares_latency_profile` | C: fix (genuine drift) | **Fixed.** `phase-61.2` (`claude_code_client.py:487`) moved `recommended_step_timeout` to a per-instance override; the CLASS attribute (`:478`, still 150) was left as a stale doc-only default and the test kept reading it, comparing `150 > 150 = False`. Test now reads the instance attribute (`client.recommended_step_timeout`), which is what `orchestrator.py:399`'s `getattr(model, ...)` actually consults. |
| 10 | `test_portfolio_swap::test_swap_framework_fills_zero_buy_gap` | C: fix (genuine drift) | **Reclassified to B, no change** (see section 0.2). A third operator-`.env`-pollution instance (`paper_swap_churn_fix_enabled`), not logic drift. |

Summary: **3 marked requires_live, 2 fixed (test-only), 5 confirmed as
category-B `.env` pollution requiring no change** (up from the
contracted 4 -- see correction 0.2).

## 2. e2e-smoke.yml changes (leg a + f)

- `continue-on-error: true` removed from the job (the lane is now
  enforcing).
- `--ignore=...` 6-file list replaced with
  `python -m pytest backend/tests/ -q -m "not requires_live"`, run with
  `PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false
  PAPER_SWAP_CHURN_FIX_ENABLED=false` set as step-level `env:` (all 3
  default False in code; a fresh CI checkout has no `.env` so this is
  already CI's natural state -- set explicitly so the lane is
  self-documenting).
- `npm run test` (-> `scripts/run-test.mjs` -> `vitest run`, jsdom, no
  server) added to the frontend step, after `npx tsc --noEmit` and before
  `npm run build`.
- Of the step-named 5 previously-unmarked/ignored files: `test_phase_23_2_10`
  is now marked; `test_phase_23_2_14`/`16`, `test_agent_map_live_model`,
  `test_rainbow_canary` were already green and are un-ignored by the
  switch to marker-based selection (no `--ignore` list to un-ignore them
  from anymore).

### CI-equivalent green tail (verbatim, CI env)

```
$ PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false PAPER_SWAP_CHURN_FIX_ENABLED=false \
  .venv/bin/python -m pytest backend/tests/ -q -m "not requires_live"
........................................................................ [ 92%]
.......................................................................s [ 97%]
...................                                                      [100%]
1467 passed, 2 skipped, 15 deselected, 5 xfailed, 1 xpassed, 1 warning in 119.36s (0:01:59)
```

Collection: `1475/1490 tests collected (15 deselected)` under
`-m "not requires_live"` (measured via `--collect-only`; the 1490/1475
totals include the 16 new tests in `test_phase_75_ci_gates.py` added this
step, none of which carry `requires_live` -- the 15-deselected count is
unchanged from before this step's 3 new marks were added on top of the
pre-existing 12).

### Raw local run (no env overrides) -- the operator-`.env` delta, disclosed

```
$ .venv/bin/python -m pytest backend/tests/ -q -m "not requires_live"
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
FAILED backend/tests/test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
FAILED backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
5 failed, 1446 passed, 2 skipped, 15 deselected, 5 xfailed, 1 xpassed, 1 warning in 107.24s (0:01:47)
```

All 5 raw-local reds are the operator's untracked `.env` setting 3 flags
(`paper_risk_judge_reject_binding`, `paper_data_integrity_enabled`,
`paper_swap_churn_fix_enabled`) to non-default values; none reproduce on
a fresh CI checkout (no `.env` present) or with the 3 overrides forced.

## 3. Leg (b): lock-count guard -- verified no-op

```
$ .venv/bin/python -m pytest backend/tests/test_phase_23_2_14_no_reentrant_locks.py -q
.....                                                                    [100%]
5 passed in 0.11s
```

`EXPECTED_LOCK_COUNT=18` already matches the measured count (phase-75.10
bumped it 2026-07-23). Confirmed collected under the new
`-m "not requires_live"` selection (no `requires_live` mark on this file).
No code change made -- consistent with the contract's own "no-op"
determination.

## 4. Leg (c): coverage_tier_check.py + refreshed doc

`scripts/qa/coverage_tier_check.py` (new, 150 lines): parses the Tier-1
STRICT + Tier-1 EXTENDED section headers of
`docs/coverage_tier_overrides.md` for their numeric bar (both currently
75%) and every backtick-quoted `backend/...py` module row under each
header -- zero hardcoded bars. Compares against a `coverage json`
report's per-file `percent_covered`. Exit 0 all-pass / 1 any-below-bar /
2 malformed-input (never a silent pass on zero parsed modules or a
missing report).

### Refreshed measurements (2026-07-24, full non-live suite, CI-equivalent env)

| Module | 2026-05-25 (stale) | 2026-07-24 (measured) | Bar | Status |
|---|---|---|---|---|
| kill_switch.py | 89% | 88.2% | 75% | PASS |
| cycle_lock.py | 84% | 83.0% | 75% | PASS |
| factor_correlation.py | 85% | 85.1% | 75% | PASS |
| factor_loadings.py | 78% | 78.1% | 75% | PASS |
| paper_trader.py | 79.1% | 78.3% | 75% | PASS (-0.8pp drift) |
| portfolio_manager.py | 81.2% | 83.7% | 75% | PASS (+2.5pp) |
| perf_metrics.py | 81.2% | 84.8% | 75% | PASS (+3.6pp) |

```
$ .venv/bin/python scripts/qa/coverage_tier_check.py --coverage-json /tmp/coverage_75_15.json
Tier-1 coverage check -- 7 module(s) gated (bars sourced from docs/coverage_tier_overrides.md)
  PASS  backend/services/cycle_lock.py: 83.0% (bar 75%)
  PASS  backend/services/factor_correlation.py: 85.1% (bar 75%)
  PASS  backend/services/factor_loadings.py: 78.1% (bar 75%)
  PASS  backend/services/kill_switch.py: 88.2% (bar 75%)
  PASS  backend/services/paper_trader.py: 78.3% (bar 75%)
  PASS  backend/services/perf_metrics.py: 84.8% (bar 75%)
  PASS  backend/services/portfolio_manager.py: 83.7% (bar 75%)

PASS -- all Tier-1 modules at or above bar
```
Exit 0.

### Can-fail proof (mutation: bar raised to 99% in a tmp copy of the doc, real coverage numbers unchanged)

```
FAIL  backend/services/paper_trader.py: 78.3% (bar 99%)
FAIL  backend/services/perf_metrics.py: 84.8% (bar 99%)
FAIL  backend/services/portfolio_manager.py: 83.7% (bar 99%)

FAIL -- Tier-1 module(s) below bar:
  - backend/services/paper_trader.py: 78.3% < bar 99%
  - backend/services/perf_metrics.py: 84.8% < bar 99%
  - backend/services/portfolio_manager.py: 83.7% < bar 99%
```
Exit 1. STRICT-section modules (kill_switch etc, unaffected by the
EXTENDED-only mutation) correctly stayed PASS -- confirms the parser
scopes bars per-section, not globally.

### Error-path proofs

- Missing coverage json -> `ERROR: coverage json report not found ...`, exit 2.
- Missing doc -> `ERROR: coverage tier doc not found ...`, exit 2.

Wired into new `.github/workflows/coverage-tier-check.yml` (nightly
06:42 UTC + `workflow_dispatch`, NOT on every PR/push -- disclosed
deviation from "wire into a workflow leg": chose a **new small workflow**
rather than adding a job to `e2e-smoke.yml`, because the full `--cov` run
adds real wall-clock time on top of the already-run credential-free
suite, and coverage drift is a slow-moving nightly-cadence signal, not a
per-commit one).

`docs/coverage_tier_overrides.md` updated: all 7 numbers refreshed to
2026-07-24 measurements, new section "3a. Enforcement (phase-75.15,
qa-tests-04)" describing the runner, new audit-trail row.

## 5. Legs (d)/(e): seed-stability wording + visual-regression baseline gate

**Leg (d)**: `seed-stability-check.yml` header reworded to drop the "A
failing drill blocks the PR" reproducibility overclaim; replaced with an
honest statement that the lane re-validates a FROZEN baseline's own
arithmetic/shape and structurally cannot catch a live code regression in
today's actual seed-to-seed spread. Verified: `'blocks the PR' not in s`
now True (also added `run_seed_stability` mentions, satisfying the OR
clause redundantly).

**Leg (e)** (not covered by the immutable command -- done anyway per the
contract): `visual-regression.yml` gained a "Check baseline presence"
step (`hashFiles`-style `find ... -name '*.png'` check) whose output
gates the actual `npx playwright test` comparison run. When zero PNGs are
committed (confirmed: 0 today under
`frontend/tests/visual-regression/snapshots/chromium/`), a new "Skip (no
baselines committed yet)" step runs instead, so push/PR triggers stay
wired (no operator re-enable step needed later) without ever producing a
guaranteed-red run. Per GitHub's own docs (cited in the research brief):
"Successful check statuses are success, skipped, and neutral."

## 6. Leg (f): vitest full-suite baseline (frontend, run once)

```
$ cd frontend && npx vitest run
 Test Files  30 passed (30)
      Tests  201 passed (201)
   Duration  5.13s
```
No server, no `next dev`, no `:3000` touched (`run-test.mjs` ->
`spawn(vitest, ...)`, `jsdom` environment per `vitest.config.ts`).

## 7. Leg (g): npm-audit.yml + current npm audit result (disclosed, not hidden)

New `.github/workflows/npm-audit.yml` mirrors `pip-audit.yml`: weekly cron
(Mon 07:15 UTC, offset from pip-audit's 07:00) + push/PR on
`frontend/package-lock.json` + `workflow_dispatch`; `npm ci` then
`npm audit --audit-level=high` in `frontend/`; artifact-on-failure; no
`npm audit fix` anywhere in an executable line.

**Current result, run locally once, read-only (2026-07-24):**

```
$ cd frontend && npm audit --audit-level=high
...
42 vulnerabilities (1 low, 19 moderate, 19 high, 3 critical)
```
Real exit code: **1** (confirmed via a clean invocation, not through a
pipe that would mask it). High/critical transitives: `tmp` (path
traversal), `undici` (6 advisories, TLS/HTTP), `vite` (2 advisories),
`ws` (3 advisories). **This lane WILL be red on its first CI run.** Per
the contract's explicit instruction, this is disclosed as a finding, not
hidden by raising `--audit-level`. No `npm audit fix` was run (would
mutate `package-lock.json`, out of scope + forbidden).

## 8. New guard file + mutation matrix

`backend/tests/test_phase_75_ci_gates.py` (new, 317 lines, 16 tests):
CI-lane text/shape assertions for every leg, reading workflow files
directly (no skip-on-missing guard) so a wrong path hard-fails.

### Mutation matrix (scripted, scratchpad `mutation_matrix_75_15.py`, exactly-once + byte-restore)

| # | Mutation | Target | Expected | First pass | Final (after fix) |
|---|---|---|---|---|---|
| M1 | revert `continue-on-error` to true | e2e-smoke.yml | FAIL | **KILLED** | KILLED |
| M2 | drop the `-m` selection | e2e-smoke.yml | FAIL | **SURVIVED** | **KILLED** (test fixed) |
| M3 | un-mark one requires_live test | test_phase_23_2_10 | FAIL | **KILLED** | KILLED |
| M4 | point checker at missing json | coverage_tier_check.py | ERROR (exit 2) | **KILLED** | KILLED |
| M5 | bar-above-current (tmp doc copy) | coverage_tier_overrides.md | FAIL (exit 1) | **KILLED** | KILLED |
| M6 | remove npm-audit audit step | npm-audit.yml | FAIL | **SURVIVED** | **KILLED** (test fixed) |
| M7 (stub) | yaml-assert test reads wrong path | test_phase_75_ci_gates.py | hard-fail, not skip-green | **KILLED** | KILLED |

**Final: 7/7 KILLED.** Two honest first-pass survivors, both real gaps in
my own guard, not false alarms: `test_e2e_smoke_uses_requires_live_marker_not_ignore_list`
and `test_npm_audit_workflow_exists_and_shaped` originally did
whole-file substring checks, and my own explanatory header comments in
each workflow happened to mention the same substrings in prose ("the
backend pytest lane now selects tests via `-m "not requires_live"`" /
"then `npm audit --audit-level=high`") -- so deleting the ACTUAL run
step still left the guard's substring "present" via the comment. Fixed
both to check the actual non-comment `run:` line specifically; re-ran the
full matrix and all 7 killed cleanly with byte-identical restoration
verified after each mutation (asserted inside the mutation script itself).

## 9. Verification (rc discipline)

- (i) Immutable verification command verbatim: **exit 0**.
- (ii) CI-equivalent green tail: 1467 passed, 0 failed (section 2).
- (iii) Full backend suite raw (no overrides): 5 failed / 1446 passed --
  exactly the disclosed operator-`.env` delta (section 2), a reduction
  from the original 10-red baseline by the 3 marks + 2 fixes.
- (iv) `ruff check --select F821,F401,F811` over the touched-file scope:
  **1 finding**, `F401` unused `dataclasses.dataclass` import at
  `backend/tests/test_phase_23_2_6_sector_cap_emit.py:25` -- confirmed
  PRE-EXISTING via `git show HEAD:...` (present before this step's edit,
  which only touched lines 227+). Not fixed (out of scope for a
  CI-config+test-edit step to drive-by-fix unrelated lint debt);
  disclosed here rather than silently left for someone to discover later.
- (v) `python -c "import yaml; yaml.safe_load(...)"` on all 5 touched/new
  workflow files: all valid. No `.sh` files touched this step (`bash -n`
  N/A).
- (vi) `curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/login`
  (read-only): **200** -- operator's dev server observed running and
  untouched; nothing in this step started a server.

## 10. Deviations named

1. Worklist corrections in section 0 (2 items reclassified vs the
   contract, with evidence).
2. Leg (c) workflow placement: new standalone `coverage-tier-check.yml`
   rather than a job inside an existing workflow (disclosed in section 4).
3. `docs/coverage_tier_overrides.md` numbers moved from "(post-phase-43.0.2)"
   dated labels to "(2026-07-24)" for the 3 EXTENDED modules, since this
   step's measurement supersedes that snapshot; STRICT section labels
   left as plain percentages (no date label existed there to begin with).
4. `test_phase_75_ci_gates.py`'s collection-count pin
   (`1475/1490 tests collected (15 deselected)`) is itself sensitive to
   future test additions to `backend/tests/` -- documented in the test's
   own docstring as a phase-75.15 baseline snapshot, not a permanent
   invariant.
5. Pre-existing F401 (section 9-iv), disclosed, not fixed.

## 11. Files touched

```
 .github/workflows/e2e-smoke.yml                    | 52 ++++++++++++++--------
 .github/workflows/seed-stability-check.yml         | 14 +++++-
 .github/workflows/visual-regression.yml            | 28 +++++++++++-
 backend/tests/test_phase_23_2_10_watchdog_no_fire_7d.py | 15 ++++++-
 backend/tests/test_phase_23_2_6_sector_cap_emit.py | 16 ++++++-
 backend/tests/test_phase_23_2_9_ticker_meta_latency.py | 14 +++++-
 backend/tests/test_phase_60_1_deep_pipeline.py     | 11 ++++-
 docs/coverage_tier_overrides.md                    | 40 ++++++++++++-----
 tests/api/test_pause_resume_timeout.py             | 13 +++++-
 9 files changed, 163 insertions(+), 40 deletions(-)

 .github/workflows/coverage-tier-check.yml   (new, 62 lines)
 .github/workflows/npm-audit.yml             (new, 73 lines)
 backend/tests/test_phase_75_ci_gates.py     (new, 317 lines)
 scripts/qa/coverage_tier_check.py           (new, 150 lines)
```

No production code touched. No `.env` edits. No server started. No
masterplan/harness_log edits. No commits/pushes.
