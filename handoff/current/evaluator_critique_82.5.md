# Evaluator Critique — phase 82.5 (Exit-quality tiles: ratio-aggregation blowup)

**Evaluator:** Layer-3 Q/A (single agent, merged qa-evaluator + harness-verifier)
**Date:** 2026-08-04
**Launch:** Agent-tool `qa` subagent (fallback rail; Workflow rail not used for this spawn)
**Cycle:** 1 (no prior Q/A verdict exists for 82.5)
**Verdict:** **CONDITIONAL**

Immutable verification command:
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_5_exit_quality_metrics.py -q`

Immutable criteria (verbatim from `.claude/masterplan.json`, re-read by me, byte-compared
against the contract's §2 transcription — **identical**):
1. a fixture round-trip with MFE == 0 does not produce an unbounded or sign-flipped capture value in the aggregate
2. a fixture round-trip with MAE == 0 is handled explicitly rather than silently dropped from the edge-ratio aggregate
3. the reported aggregate for both tiles is robust to a single extreme outlier: adding one round-trip with mfe/mae = 1e-4 to a fixture of ordinary trades moves the reported value by less than 20 percent
4. a test pins the pre-fix behaviour by asserting the OLD mean formula would have returned a value with magnitude greater than 40 on the committed real-data fixture, so the guard cannot silently regress

---

## 0. Harness-compliance audit (5 items, run FIRST)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawned BEFORE the contract | **PASS** | `handoff/current/research_brief_82.5.md` (491 lines, mtime 19:19:17) + envelope `handoff/current/qa_returns/ws1xvyys5.output.json`: `gate_passed=true`, `external_sources_read_in_full=7` (≥5 floor), `urls_collected=26` (≥10), `recency_scan_performed=true`, `internal_files_inspected=14`. Ran on the **Workflow rail** (task `ws1xvyys5`), satisfying `feedback_both_dev_mas_agents_use_workflows`. Contract §1 and §8 cite it. |
| 2 | Contract written BEFORE GENERATE | **PASS** | mtimes, measured: research 19:19:17 → contract 19:22:27 → fixture 19:23:21 → code 19:31:38/19:32:15 → test 19:32:08 → experiment_results 19:33:02. Strictly ordered. |
| 3 | `experiment_results.md` present, with file list + verbatim command output | **PASS** | `handoff/current/experiment_results_82.5.md`, 133 lines, §8 verbatim pytest block, §9 mutation matrix, §10 collateral disclosure, §11 scope honesty. |
| 4 | LOG-LAST (harness_log append not yet done, status still `pending`) | **PASS** | `grep -c 'phase=82.5' handoff/harness_log.md` → **0**. Last entry is `## Cycle 1139 -- 2026-08-04 -- phase=82.7`. Masterplan `82.5.status = "pending"`, `retry_count = 0`. Correct order. |
| 5 | No verdict-shopping / 3rd-CONDITIONAL rule | **N/A — PASS** | Zero prior `82.5` entries in `handoff/harness_log.md`; `retry_count=0`. This is the first Q/A on this step, so the 3rd-consecutive-CONDITIONAL auto-FAIL does not trigger. |

---

## 1. Deterministic checks (run by me, in my shell, not replayed from the handoff)

### 1.1 Immutable verification command — exit 0

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_5_exit_quality_metrics.py -q
................                                                         [100%]
16 passed, 1 warning in 1.91s
IMMUTABLE_EXIT=0
```

(Exit code captured with a bare redirect. Note: `${PIPESTATUS[0]}` is a **bash**-ism —
this shell is zsh, where it expands empty. First attempt silently produced `EXIT=`.)

### 1.2 Python lint gate — scope DERIVED from git, green on the change

`git diff --name-only HEAD -- '*.py'` → 5 files (non-empty guard satisfied):
`backend/api/paper_trading.py`, `backend/services/paper_round_trips.py`,
`backend/services/paper_trader.py`, `backend/services/perf_metrics.py`,
`backend/tests/test_dod4_tier1_coverage_investment.py`.

**Vacuity shape #9 struck again, on my own first attempt.** The canonical
`uvx ruff check ... $FILES` line from `qa.md` §1a **silently linted ZERO files** here and
printed `All checks passed! exit=0` — because zsh does *not* word-split unquoted
parameter expansions, so the newline-joined list was passed as ONE filename
(`warning: Failed to lint …: No such file or directory`). Re-run with
`| tr '\n' '\0' | xargs -0`:

```
ruff --select F821,F401,F811  ->  Found 4 errors, ruff_exit=1
  F401 threading / datetime.datetime / datetime.timezone / pytest
       in backend/tests/test_dod4_tier1_coverage_investment.py
```

All 4 are **PRE-EXISTING**: I extracted `git show HEAD:backend/tests/test_dod4_tier1_coverage_investment.py`
and linted it — byte-identical 4 F401s, `head_exit=1`. Not introduced by 82.5.

The new **untracked** test file is outside `git diff --name-only HEAD` and had to be added
by hand; linting it separately yields **1 new F401**: `import statistics as st` at
`backend/tests/test_phase_82_5_exit_quality_metrics.py:23` is unused. Cosmetic (NOTE-level).

### 1.3 Frontend lint + typecheck (diff touches `frontend/**`)

```
$ npx tsc --noEmit                                        -> tsc_exit=0
$ npx eslint .                                            -> exit 1, 26 errors, 58 warnings
$ npx eslint src/components/MfeMaeScatter.tsx src/lib/types.ts src/lib/api.ts
                                                          -> 0 errors, 2 warnings, exit 0
```
All 26 repo-wide errors are in pre-existing **build-artifact** directories
(`.next-audit-36-12/`, `.next-functional/` webpack bundles), enumerated via
`eslint . -f json`. **Zero** are in source, and zero in any file this step changed.
Gate is green on the scope the change defines.

### 1.4 Backend runtime smoke (diff touches `backend/**`)

```
import backend.services.perf_metrics        -> OK (MIN_MFE_PCT = 1.0)
import backend.services.paper_round_trips   -> OK
import backend.services.paper_trader        -> OK
import backend.api.paper_trading            -> OK
curl /api/health                            -> 200
```

---

## 2. Attack A — is criterion 3 satisfied only by the 1.0pp floor?

**Answer: the criterion's own verbatim test, YES. The suite's coverage of criterion 3, NO —
and Main disclosed this himself (§9) rather than being caught at it.**

I ran my **own** 7-mutant matrix via an in-memory source-patching pytest plugin (repo
working tree never touched; every mutation asserts its target exists exactly once before
replacing, so a no-op `str.replace` cannot masquerade as a kill):

| Mutant | What it breaks | Result |
|---|---|---|
| CONTROL | — | 16 passed |
| **M_MEAN_CAPTURE** | capture headline → arithmetic mean | **2 failed** |
| **M_MEAN_EDGE** | edge headline → arithmetic mean | **1 failed** |
| **M_NO_FLOOR** | `MIN_MFE_PCT 1.0 → 0.0` | **3 failed** |
| **M_UNDEF_ZERO** | undefined capture → fabricated `0.0` | **5 failed** |
| **M_DROP_MAE_ZERO** | `mae==0` → `None` (dropped again) | **4 failed** |
| **M_OLD_MFE_GUARD** | delete the floor clause entirely, restore bare `mfe > 0` | **3 failed** |
| RESTORED | — | 16 passed |

Every mutant is killed. Two of them (M_MEAN_EDGE, M_OLD_MFE_GUARD) are mine, not in
Main's matrix, and both die too.

**The criterion-3 finding, measured:** under `M_MEAN_CAPTURE`,
`test_one_extreme_outlier_moves_each_headline_by_under_20_percent` — the test that
implements criterion 3 **verbatim** (poison at `mfe/mae = 1e-4`) — **PASSES**. The 1.0pp
floor excludes that row before the estimator ever sees it, so that guard is
estimator-agnostic. Confirmed by execution, not inferred.

What kills `M_MEAN_CAPTURE` is `test_the_median_itself_is_load_bearing_not_just_the_floor`,
and I named the killing assertion rather than crediting the test as a whole
(vacuity shape #11):

```
backend/tests/test_phase_82_5_exit_quality_metrics.py:201: AssertionError
>       assert drift < 0.20, f"capture median moved {drift:.1%}"
E       AssertionError: capture median moved 4766.7%
E       assert 47.66666666666667 < 0.2
```

That is the criterion-3 *shape* (<20% drift on a single added round-trip) with a poison
row **above** the floor (`mfe=1.5`), where the floor cannot help. So criterion 3 has two
independent, separately-killable guards: the floor (killed by M_NO_FLOOR / M_OLD_MFE_GUARD)
and the estimator (killed by M_MEAN_CAPTURE).

**Is the added test itself vacuous? No.** It carries its own anti-vacuity legs, and I
verified each is meaningful rather than decorative:
- `assert after["capture_n_defined"] == before["capture_n_defined"] + 1` — proves the
  poison row is genuinely *admitted*, not floor-filtered (this is the leg that would have
  exposed the test if the poison had been below 1.0pp).
- `assert abs(mean_after - mean_before)/abs(mean_before) > 1.0` — proves the poison is
  potent enough to discriminate between the two estimators. Without it the test could
  pass on a poison a mean also survives.

NOTE (non-blocking): the poison uses `realized_pnl_pct = -900.0`, which is not reachable
for a long-only equity round-trip (floor −100%). It is a synthetic stress input, not a
realistic one. I checked it is not load-bearing: a realistic poison (`mfe=1.5`,
`pnl=-15`) still moves a mean ~80% (>20%) while leaving the median inside 20%, so the
test would discriminate at realistic magnitudes too.

---

## 3. Attack B — is the fixture load-bearing, genuine, and un-softened?

### 3.1 Every number reproduces from the raw inputs (re-derived, not read)

| Claim | Re-derived by me | Match |
|---|---|---|
| legacy mean capture −42.0785 | −42.078488 (stored col) / **−42.091829 (re-derived `pnl/mfe if mfe>0 else 0`)** | ✔ both magnitude > 40 |
| legacy mean edge 86.9218 over 26 rows | 86.921752 over **26** rows | ✔ |
| 8 rows `mfe == 0` | 8 | ✔ |
| 6 rows `mae == 0` | 6 | ✔ |
| worst capture −1269.5726 (000660.KS) | −1269.5726 stored / −1270.0 re-derived | ✔ |
| max edge 1483.3388 (INTC) | 1483.3388 | ✔ |
| new capture median 0.6304, n 20/12 | 0.6304, 20/12 | ✔ |
| new edge median 3.0900, 6 × +inf | 3.09, 6 | ✔ |
| secondary 0.6446 / 3.9636 | 0.6446 / 3.9636 | ✔ |

**No row is hand-edited.** The stored `capture_ratio_legacy` column agrees with the
formula re-derived from `realized_pnl_pct / mfe_pct` on **31 of 32** rows to <0.02; the
single divergence is the 000660.KS row (−1269.5726 stored vs −1270.0 re-derived), which
is exactly what storing 4-dp-rounded inputs (`mfe_pct: 0.0001`) does to a
division-by-near-zero. That is the signature of genuine data, not of tuning.

### 3.2 Independent corroboration that the fixture IS production data

I did not take Main's word that it came from BigQuery. The **live** pre-fix backend still
serving on :8000 returns, for the real book:

```
$ curl -s http://127.0.0.1:8000/api/paper-trading/mfe-mae-scatter
"edge_ratio": 86.9218,  "avg_capture_ratio": -42.0785,
"mfe_p75": 28.3553,     "n_points": 32,  "n_leakers": 0
```

Byte-identical to the values the fixture reproduces. The fixture is the real book.

### 3.3 Would criterion 4 FAIL on a softened fixture? Yes — 5 of 5 softening mutants kill it

| Fixture mutant | old_mean | criterion-4 guard |
|---|---|---|
| CONTROL | −42.0785 | PASS |
| S1 blowup row softened to −30 | −3.3418 | **FAIL** |
| S2 blowup row deleted | −2.4819 | **FAIL** |
| S3 all legacy values halved | −21.0392 | **FAIL** |
| S4 legacy winsorized at −100 | −5.5293 | **FAIL** |
| E1 tiny-MAE rows widened to −0.5 | n=32, mean 34.5467 | **FAIL** (edge pin) |

I additionally ran two fixture-substitution mutants through the *real* suite (via a plugin
that rebinds the module's `FIXTURE` path and asserts the attribute exists first, so an
override that did nothing would raise rather than pass):
- **F1** — raw `mfe_pct`/`realized_pnl_pct` of the blowup row changed while the stored
  `capture_ratio_legacy` is left stale → **2 failed** (`test_the_new_aggregate_is_sane…`,
  `test_the_old_edge_mean_also_blows_up…`).
- **F2** — fixture truncated to 24 rows → **9 errors** (the module fixture's
  `assert len(rows) == 32` fires).

### 3.4 Credentials / PII

Read the fixture in full (225 lines). Schema is exactly
`{ticker, mfe_pct, mae_pct, realized_pnl_pct, capture_ratio_legacy}`. No keys, no
timestamps, no account identifiers, no position sizes, no NAV. Tickers only. Clean.

---

## 4. Attack C — consumer-contract break on `float | None`

I enumerated consumers by grep across `*.py`, `*.ts`, `*.tsx`, `*.sql` (excluding
`.venv`/`node_modules`) rather than trusting the disclosure.

- **BQ writes** — `capture_ratio` is declared `FLOAT64` (no `REQUIRED` mode) in
  `scripts/migrations/add_round_trip_schema.py:59,80`, i.e. NULLABLE. `None` is a legal
  write. `_safe_save_trade`'s schema-error prune path is unaffected.
- **BQ reads** — nothing reads the stored `capture_ratio` back. `pair_round_trips`
  recomputes it from `mfe_pct`/`realized_pnl_pct`. Verified.
- **Optimizer / promoter / meta_evolution / Slack bot / backtest / SQL** — **zero**
  occurrences of `capture_ratio` or `edge_ratio`. The
  `tests/meta_evolution/test_alpha_velocity.py:130` hit is a generic `components` dict,
  not this metric.
- **Go-live gate** — `paper_go_live_gate.py` uses only `len(round_trips)`; the pairing
  loop is untouched, so no promotion boolean moves.
- **Frontend `MfeMaeScatter.tsx` / `types.ts`** — updated correctly; both render paths
  discriminate on `=== null` and never multiply a null.
- **`npx tsc --noEmit` → exit 0.**

### ✖ FINDING C-1 (WARN): a contract-named change site was skipped, and the disclosure says otherwise

`handoff/current/contract_82.5.md` §4 explicitly lists the change site
`frontend/…/types.ts:766 + api.ts:524 -> number | null`. **`api.ts` was not changed.**

```
frontend/src/lib/api.ts:524      avg_capture_ratio: number;
frontend/src/lib/api.ts:537      capture_ratio: number;
```

Both belong to `getPaperRoundTrips()`, whose backend now genuinely returns `null` for
both (`paper_round_trips.summarize` → `avg_capture_ratio: None`; `pair_round_trips` →
per-trip `capture_ratio: None`). `experiment_results.md` §11 states *"frontend types and
the tooltip were updated"* — that is true of `types.ts` and false of `api.ts`, which the
contract named in the same breath. This is the
`feedback_gate_scope_and_disclosure_completeness` shape: naming the files you changed is
not the same as covering the files the contract said to change.

**Blast radius today: nil.** `getPaperRoundTrips` is declared but **never called** anywhere
in `frontend/src` (grep: one definition, zero call sites), which is why `tsc --noEmit`
still exits 0. It is a false type declaration, not a live crash.

---

## 5. Attack D — the three-copy de-duplication, verified BY EXECUTION

I did not read the source to confirm this. I built 32 synthetic BUY/SELL trade pairs whose
`pair_round_trips` output reproduces the committed fixture, then drove **both real code
paths** on the same rows — `paper_round_trips.summarize()` (the `/round-trips` +
`/performance` surface) and `paper_trading.get_mfe_mae_scatter()` (the endpoint coroutine,
with a stubbed BQ client):

```
summarize()                 avg_capture_ratio = 0.6304  ratio_of_sums = 0.6446  n 20/12  floor 1.0
get_mfe_mae_scatter()       avg_capture_ratio = 0.6304  ratio_of_sums = 0.6446  n 20/12  floor 1.0
                            edge_ratio = 3.09  edge_ratio_of_sums = 3.9636  edge_n_infinite = 6
                            aggregation = "median"  null capture points = 12  n_points = 32
PARITY: avg_capture_ratio equal? True   ratio_of_sums equal? True
        n_defined equal? True           min_mfe_pct equal? True
```

**The de-duplication claim is TRUE.** I also reproduced the "grep returns nothing" claim:
no `realized_pnl_pct / mfe`, `pnl_pct / mfe_pct`, `/ mfe_pct if`, or `/ mfe if` survives
in non-test backend code.

### ✖ FINDING D-1 (WARN): INV5's regression guard is a source scan (vacuity shape #2)

`test_performance_and_scatter_surfaces_share_one_definition` enforces the single-definition
property with `assert "realized_pnl_pct / mfe" not in inspect.getsource(mod)`. That is a
literal string scan, defeated by whitespace. Demonstrated:

```
a re-inlined copy written as 'realized_pnl_pct/mfe'         caught? False
                             'realized_pnl_pct /mfe'        caught? False
                             '(realized_pnl_pct) / (mfe)'   caught? False
                             'realized_pnl_pct * (1.0/mfe)' caught? False
```

The second leg (`"aggregate_exit_quality" in getsource(...)`) is also a source scan and
cannot observe whether the two surfaces actually *agree*. The property is real — I proved
it by execution above — but the suite's guard for it is weak. Non-blocking because INV5 is
a contract-authored invariant, **not** one of the four immutable criteria, and no immutable
criterion depends on it. Fix shape: assert the two paths return equal aggregates on a
shared row list (i.e. codify the parity run above), instead of scanning source text.

---

## 6. Attack E — the pre-existing failure and the changed test

### 6.1 `test_paper_trader_execute_buy_average_up_recomputes_avg_entry` is genuinely pre-existing — verified, and Main's stated evidence is not the actual mechanism

I did not accept "verified against the failure lists captured during 82.7". I extracted the
HEAD copies of `perf_metrics.py`, `paper_round_trips.py`, `paper_trader.py` and the HEAD
copy of the test file, pre-registered the HEAD modules in `sys.modules` via a pytest
plugin, and ran the **pre-82.5 test file against pre-82.5 code**:

```
HEAD overlay:  1 failed, 71 passed
FAILED test_paper_trader_execute_buy_average_up_recomputes_avg_entry
current tree:  1 failed, 71 passed   (same test, same assertion)
```

**Root cause, measured** — not the one cited: the *live* kill switch is engaged.

```
$ curl -s http://127.0.0.1:8000/api/paper-trading/kill-switch
{"paused":true,"pause_reason":"manual", ...}
ERROR backend.services.paper_trader: kill_switch: REFUSING BUY AAPL ($600.00) --
      the kill switch is PAUSED (pause_reason='manual')
```

Same conclusion (pre-existing, not caused by 82.5), different evidence. Worth recording so
the next reader does not chase 82.7 artifacts.

### 6.2 Broader regression — zero failures attributable to 82.5

Full `backend/tests/` (2530 tests, 3m49s): **31 failed, 2487 passed, 12 skipped, 5 xfailed,
1 xpassed**. I did not average that away. I derived the subset of failing files that import
any changed module (12 files) and ran them under the HEAD-module overlay vs the current
tree:

```
CURRENT (with 82.5):  21 failed, 99 passed, 1 skipped
HEAD overlay:         21 failed, 99 passed, 1 skipped
diff of FAILED name sets: EMPTY (only the timing line differs)
```

The remaining 9 failures are in files that import **none** of the four changed modules
(`test_phase_23_2_13/15/4/6`, `test_phase_40_2_*`, `test_phase_75_17_*`,
`test_phase_75_prompt_contracts`, `test_phase_75_sre_ops`) — log-scraping, masterplan-diff
and operator-token checks, environment-driven. 21 + 9 + 1 (`test_dod4`) = 31, reconciled.

*(Method note for the next reader: my first attempt at this comparison silently ran ZERO
tests — zsh does not word-split `$FILES`. `pytest` reported `ERROR: file or directory not
found` with the whole string as one path. Had I greped only for `FAILED` I would have seen
two empty sets and declared "IDENTICAL FAILURE SETS". Vacuity shape #9, twice in one
session.)*

### 6.3 The changed test was changed for the right reason — ACCEPTED

`test_paper_trader_execute_sell_capture_ratio_zero_when_no_gain` asserted
`capture_ratio == 0.0` for `MFE == 0`. That assertion **encoded the defect**: it pinned the
fabricated zero that this step exists to remove. Renaming to `..._none_when_no_gain` and
asserting `is None` is a legitimate contract update, not a red-test-made-green:
- the production behaviour deliberately changed, and the old assertion was the *old*
  contract;
- the original intent (no NaN, no `ZeroDivisionError`) is still asserted — the call still
  executes and returns;
- the rename + docstring make the change auditable in `git log` rather than silent;
- `M_UNDEF_ZERO` (restore `0.0`) kills 5 tests, so the new assertion is load-bearing.

---

## 7. Criterion-by-criterion coverage map, each with its named killing mutation

| # | Covering evidence | Mutation that turns it red | Executed? |
|---|---|---|---|
| 1 | `test_mfe_zero_does_not_produce_an_unbounded_or_sign_flipped_capture` (8 real rows → `None`; median finite AND > 0), `test_an_undefined_capture_is_distinguishable_from_a_real_zero`, `test_an_all_undefined_fixture_reports_none_not_zero` | **M_UNDEF_ZERO** → 5 failed | ✔ by me |
| 2 | `test_mae_zero_is_handled_explicitly_and_not_dropped` (6 real rows → `+inf`, `n_defined+n_undefined == n_points == 32`), `test_edge_ratio_degenerate_cases_are_each_defined`, `test_an_infinite_median_is_reported_as_none_never_as_infinity` | **M_DROP_MAE_ZERO** → 4 failed | ✔ by me |
| 3 | `test_one_extreme_outlier_moves_each_headline_by_under_20_percent` (verbatim criterion) **+** `test_the_median_itself_is_load_bearing_not_just_the_floor` (poison above the floor) | **M_NO_FLOOR** / **M_OLD_MFE_GUARD** → 3 failed each (floor leg); **M_MEAN_CAPTURE** → 2 failed, killing assertion `:201 assert drift < 0.20` (estimator leg) | ✔ by me |
| 4 | `test_the_old_mean_formula_blows_up_on_the_committed_real_data` (\|−42.0785\| > 40, worst < −1000) + `test_the_old_edge_mean_also_blows_up_on_the_real_data` | fixture mutants **S1–S4, E1** all FAIL the guard; **F1/F2** substitutions fail the suite | ✔ by me |

All four criteria are COVERED. No `Missing_Assumption` on any criterion.

### ✖ FINDING #4-A (WARN): criterion 4's own test is one degree indirect

`test_the_old_mean_formula_blows_up_on_the_committed_real_data` averages the **stored**
`capture_ratio_legacy` column rather than **re-deriving** `realized_pnl_pct / mfe_pct if
mfe_pct > 0 else 0.0` from the raw inputs. Mutant **F1** (raw inputs edited, stored column
left stale) leaves *this* test green — it is caught only by two sibling tests. The criterion
says "the OLD mean formula would have returned"; averaging a persisted column is not the
same as executing the formula. Cheap fix: compute the legacy value in the test and assert
it agrees with the stored column (which it does — I verified, 31/32 exact, 1 row explained
by 4-dp input rounding).

---

## 8. The blockers

### ✖ BLOCKER 1 — `qa.md` §1c: no live UI capture for a UI-changing step

The diff changes **what the card renders**: `hint` text `mean(MFE / |MAE|)` →
`median(MFE / |MAE|)`; a new `"n/a"` branch when either headline is `null`; a new
`n=<capture_n_defined>` suffix on the capture hint; a null branch in the per-point tooltip.
`experiment_results.md` §2 and §5 make explicit UI claims. §1c is BINDING: such a step
**cannot receive PASS** without a live Playwright capture.

I attempted the capture myself, as §1c requires (browser tools are on my surface; loaded
via the deterministic `select:` form):

```
browser_navigate("http://localhost:3000/paper-trading/exit-quality")
  -> Page URL: http://localhost:3000/login   (NextAuth wall, no session in this profile)
```

The skip-auth :3100 instance is **down** (`curl :3100 -> 000`), and starting it is **Main's**
lifecycle responsibility, never mine (the 2026-07-17 :3000-outage class). So no capture
exists, from me or from Main. Per §1c this caps the verdict at CONDITIONAL with
`Missing_Assumption: live UI capture`.

### ✖ BLOCKER 2 — the running app does not have the fix, and §5 reads as though it does

Measured, not inferred:

```
$ ps -eo pid,lstart,command | grep uvicorn
654  tir. 28 jul. 18.39.22 2026  .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
$ stat -f "%Sm %N" backend/services/perf_metrics.py
Aug  4 19:32:15 2026 backend/services/perf_metrics.py
```

The backend process started **28 July**, seven days before the code was written, and runs
**without `--reload`**. The live endpoint confirms it is serving pre-fix code:

```
$ curl -s http://127.0.0.1:8000/api/paper-trading/mfe-mae-scatter
"edge_ratio": 86.9218,  "avg_capture_ratio": -42.0785
null capture rows: 0        (no `aggregation`, `min_mfe_pct`, `capture_n_defined` keys)
```

`experiment_results.md` §5 is headed **"What the tiles now report on that same real data"**
and lists `Avg capture : 63% (was -4208%)` / `Edge ratio : 3.09 (was 86.92)`. Those are
fixture-derived values. **What the tiles report right now is −4208% and 86.92.** The
present-tense framing asserts a live state that does not hold — the
`feedback_verify_own_completed_action_claims` shape. The arithmetic is not in question; the
claim about the running system is.

Remediation for both blockers is mechanical and does not touch the math:
1. Main restarts the backend (parent + child workers, zombie prevention per CLAUDE.md).
2. Re-curl `/api/paper-trading/mfe-mae-scatter`; expect `avg_capture_ratio: 0.6304`,
   `edge_ratio: 3.09`, `capture_n_defined: 20`, `capture_n_undefined: 12`,
   `edge_n_infinite: 6`, `aggregation: "median"`, 12 `null` capture points.
3. Main brings up the skip-auth :3100 instance; a **fresh** Q/A takes the
   `browser_navigate` + `browser_take_screenshot` capture of
   `/paper-trading/exit-quality` and writes `handoff/current/live_check_82.5.md`.
   (No `verification.live_check` field is set on step 82.5, so the auto-commit hook will
   not hold the commit — the gate here is `qa.md` §1c, not the hook.)
4. Reword §5 to say what it is (a fixture computation) or restate it against the restarted
   service.
5. Fix `api.ts:524` / `:537` to `number | null` (FINDING C-1) and correct the §11 claim.

---

## 9. Non-blocking findings (fix or file, but do not gate on them)

| # | Finding | Severity |
|---|---|---|
| C-1 | `frontend/src/lib/api.ts:524,537` still declare `avg_capture_ratio: number` / `capture_ratio: number` for a payload that now returns null; contract §4 named this site; §11 claims "frontend types … were updated". `getPaperRoundTrips` has zero call sites, so no runtime break today. | WARN |
| D-1 | INV5's guard is a literal source scan, defeated by whitespace (vacuity shape #2). The property itself I verified by execution. | WARN |
| #4-A | Criterion 4's test averages a stored column rather than re-executing the old formula (vacuity shape #5-adjacent). Caught by sibling tests, not by itself. | WARN |
| N-1 | `import statistics as st` unused in the new test file (ruff F401). | NOTE |
| N-2 | No `MfeMaeScatter.test.tsx` despite 18+ `*.test.tsx` component tests in the repo. The `null → "n/a"` render branch has **neither** a component test **nor** a live capture. | NOTE |
| N-3 | `PaperRoundTripSummary` in `types.ts` gained `edge_ratio_of_sums` / `edge_n_infinite`, but `summarize()` never emits either key. Optional, so no type error; the type over-promises. | NOTE |
| N-4 | Commit-scope hazard (`feedback_audit_the_commit_not_the_diff`): `git add -An` shows a stray **zero-byte** `threshold` file at the repo root (created 4 Aug 12:40, not this step's work) plus `handoff/current/phase83_research_raw/*` from another session. No foreign **source** would be swept, but `threshold` should not enter 82.5's commit. | NOTE |
| N-5 | Doc drift in `qa.md` itself: the §1a lint one-liner `uvx ruff check … $FILES` reports a **false green** under zsh (linted zero files, exit 0). Worth hardening to `| tr '\n' '\0' | xargs -0`, since the empty-set guard passes while the file set is one bogus path. | NOTE |

---

## 10. Adversarial worst-of-N-lenses (P1 money-path step)

- **correctness lens** — **PASS.** The Cauchy argument is sound and the implementation
  matches it. Both degeneracies get the treatment the contract specifies; the asymmetry
  (exclude for capture, rank `+inf` for edge) is correct and non-obvious. `_median_or_none`
  correctly refuses a non-finite headline (an even-length median straddling `+inf`
  evaluates to `inf` → `None`), and no `-inf`/`nan` path exists to produce a `nan` median.
  The `mfe > 0` / `mfe < floor` split is deliberately two statements after a real
  self-caught bug (§7) — the right call.
- **does-it-reproduce lens** — **PASS.** Immutable command exit 0; every quantitative claim
  in the handoff re-derived independently and matched; 7 code mutants + 5 fixture-softening
  mutants + 2 fixture-substitution mutants all killed; both aggregate paths executed and
  proven equal; regression set proven identical to HEAD.
- **scope-honesty lens** — **FAIL → CONDITIONAL.** §9 disclosing the criterion-3 floor
  problem unprompted is exactly the behaviour this gate wants, and §10/§11 disclose the
  collateral test change, the `MIN_MFE_PCT` judgement call, the misleading retained key
  name and the absence of a BQ backfill. But §5 states live-tile values that the live app
  does not produce, §11's "frontend types … were updated" is incomplete against the
  contract's own §4 list, and there is no UI evidence for a UI-changing diff.

`verdict = min(lens verdicts)` = **CONDITIONAL**.

---

## 11. Verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 4 immutable criteria are covered by non-vacuous guards -- verified by MY OWN 7-mutant code matrix, 5 fixture-softening mutants and 2 fixture-substitution mutants, all killed -- and the immutable command exits 0 (16 passed). The fixture is genuine production data: the pre-fix live endpoint independently returns the same -42.0785/86.9218 the fixture reproduces, and the stored legacy column agrees with the re-derived old formula on 31/32 rows. The three-copy de-duplication is real, proven by EXECUTING both aggregate paths on the same rows (both 0.6304 / 0.6446 / n 20-12). Zero regressions: the failing-test name set under a HEAD-module overlay is identical to the current tree (21/21), and the one flagged pre-existing failure reproduces on pre-82.5 code (root cause is the live kill switch being manually PAUSED, not the 82.7 artifact cited). CONDITIONAL, not PASS, on three fixable items: (1) qa.md 1c is BINDING and there is NO live UI capture for a diff that changes what the card renders -- I attempted the capture and :3000 redirects to /login while the skip-auth :3100 instance is down (Main's lifecycle); (2) the running backend (PID 654, started 28 Jul, no --reload) still serves the PRE-FIX -42.0785/86.9218, yet experiment_results 5 is headed 'What the tiles now report' and lists 63%/3.09 -- a present-tense claim about a live state that does not hold; (3) contract 4 named 'api.ts:524 -> number | null' and api.ts was not changed (still 'avg_capture_ratio: number', 'capture_ratio: number') while 11 claims 'frontend types ... were updated' -- zero runtime blast radius because getPaperRoundTrips has no call sites, but the type surface is false and the disclosure incomplete. On criterion 3 specifically: the VERBATIM criterion-3 test passes identically under a mean (the 1.0pp floor excludes the mfe=1e-4 poison), which Main disclosed himself in 9; criterion 3 is NOT floor-only in the suite because test_the_median_itself_is_load_bearing_not_just_the_floor dies under M_MEAN_CAPTURE at the assertion 'assert drift < 0.20' (line 201) with a poison ABOVE the floor, and that test carries its own two anti-vacuity legs.",
  "violated_criteria": [
    "Missing_Assumption: live UI capture",
    "Contradiction: experiment_results §5 states live tile values the running service does not produce",
    "Overgeneralization: contract-named change site api.ts:524/:537 skipped while §11 claims frontend types were updated"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "Q/A attempted browser_navigate('http://localhost:3000/paper-trading/exit-quality') to satisfy qa.md §1c for a diff that changes MfeMaeScatter.tsx render output (hint mean->median, new 'n/a' null branch, n= suffix, tooltip null branch)",
      "state": "navigation redirected to http://localhost:3000/login (NextAuth wall, no session in the evaluator's browser profile); skip-auth :3100 instance is DOWN (curl :3100 -> 000); starting it is Main's lifecycle responsibility, never the evaluator's; no capture exists from Main either; masterplan step 82.5 sets no verification.live_check field so the auto-commit hook will not hold the commit",
      "constraint": "qa.md §1c (BINDING, phase-59.2/75.20): a step whose contract, immutable criteria or diff makes UI claims CANNOT receive PASS without a LIVE Playwright capture; a missing or stale capture caps the verdict at CONDITIONAL"
    },
    {
      "violation_type": "Contradiction",
      "action": "experiment_results_82.5.md §5, headed 'What the tiles now report on that same real data', asserts 'Avg capture : 63% (was -4208%)' and 'Edge ratio : 3.09 (was 86.92)'",
      "state": "backend PID 654 started 2026-07-28 18:39:22 with no --reload; changed modules written 2026-08-04 19:31:38/19:32:15; live GET /api/paper-trading/mfe-mae-scatter returns edge_ratio 86.9218, avg_capture_ratio -42.0785, 0 null capture points, and none of the new keys (aggregation / min_mfe_pct / capture_n_defined). The tiles currently report -4208% and 86.92.",
      "constraint": "A present-tense claim about a running system must be verified against that system (feedback_verify_own_completed_action_claims); the values quoted are a fixture computation, not a live readout. Remedy: restart the backend, re-curl, and restate §5 or re-measure it."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "experiment_results_82.5.md §11 states 'frontend types and the tooltip were updated', while contract_82.5.md §4 enumerated the change sites as 'frontend types.ts:766 + api.ts:524 -> number | null'",
      "state": "frontend/src/lib/api.ts:524 still declares 'avg_capture_ratio: number;' and :537 'capture_ratio: number;' for the /round-trips payload, which paper_round_trips.summarize() and pair_round_trips now return as null. tsc --noEmit exits 0 only because getPaperRoundTrips has zero call sites in frontend/src (grep: 1 definition, 0 callers), so the false type is latent, not exploited.",
      "constraint": "Every change site the contract enumerates must be covered or its omission disclosed (feedback_gate_scope_and_disclosure_completeness): naming the files you changed is not describing the files the contract required."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item_audit",
    "immutable_verification_command_exit_0",
    "python_lint_gate_ruff_F821_F401_F811_scope_derived_from_git",
    "pre_existing_lint_baseline_vs_HEAD",
    "frontend_tsc_noemit_exit_0",
    "frontend_eslint_scoped_to_changed_files_exit_0",
    "backend_import_smoke_4_modules",
    "live_api_health_200",
    "live_api_mfe_mae_scatter_curl",
    "independent_mutation_matrix_7_code_mutants_all_killed",
    "killing_assertion_identified_by_line_number",
    "fixture_reproduction_re_derived_from_raw_inputs",
    "fixture_softening_mutants_S1_S4_E1_all_kill_criterion_4",
    "fixture_substitution_mutants_F1_F2_via_suite",
    "fixture_pii_credential_scan",
    "consumer_grep_py_ts_tsx_sql_bq_schema",
    "both_aggregate_paths_executed_and_compared",
    "source_scan_guard_weakness_demonstrated",
    "full_backend_regression_2530_tests",
    "HEAD_module_overlay_failure_set_symmetric_diff",
    "pre_existing_failure_root_caused_to_live_kill_switch",
    "playwright_live_ui_capture_ATTEMPTED_blocked_by_auth_wall",
    "harness_log_last_and_masterplan_status_check",
    "third_conditional_counter_check",
    "git_add_dry_run_commit_scope_audit"
  ]
}
```

**Note on the 3rd-CONDITIONAL rule:** this is the first Q/A verdict for step 82.5
(`retry_count: 0`, zero prior `phase=82.5` entries in `handoff/harness_log.md`), so the
auto-FAIL escalation does not apply. The correct next move is the documented cycle-2 flow:
Main fixes the three blockers, updates `experiment_results_82.5.md` and this critique's
follow-up section, and spawns a **fresh** Q/A on the changed evidence.

---
---

# CYCLE 2 — Fresh Q/A on CHANGED evidence

**Evaluator:** Layer-3 Q/A (fresh instance, Agent-tool `qa` rail)
**Date:** 2026-08-04
**Cycle:** 2
**Prior verdict:** CONDITIONAL (cycle 1, same file above)
**Evidence changed since cycle 1?** — to be measured below (this is the
verdict-shopping discriminator: *did the files change between spawns?*)

**Verdict:** **CONDITIONAL** (full JSON at §C2.12 below)
**Cycle-1 blockers C1/C2/C3:** all three genuinely CLOSED (verified independently, §C2.5/C2.6/C2.8)
**New blocker:** E-1 — the capture run silently mutated two tracked frontend config files (§C2.9)

## C2.0 — Work log (written incrementally; this file is the durable artifact)

### C2.1 Harness-compliance audit (5 items, run FIRST)

| # | Item | Result | Evidence (measured by me) |
|---|------|--------|---------------------------|
| 1 | Researcher spawned BEFORE the contract | **PASS** | `handoff/current/research_brief_82.5.md` mtime **19:19:17** < contract **19:22:27**. Envelope `handoff/current/qa_returns/ws1xvyys5.output.json` (Workflow rail — satisfies `feedback_both_dev_mas_agents_use_workflows`). |
| 2 | Contract written BEFORE GENERATE | **PASS** | contract 19:22:27 → fixture 19:23:21 → code 19:25:26/19:31:38/19:32:15 → test 19:32:08 → results 19:33:02 (cycle-1), amended 19:58:49 (cycle-2). Strictly ordered. |
| 3 | `experiment_results` present w/ file list + verbatim output | **PASS** | 212 lines incl. §8 verbatim pytest, §9 mutation matrix, §12 cycle-2 closure. |
| 4 | **LOG-LAST** | **PASS** | `grep -c 'phase=82.5' handoff/harness_log.md` → **0** (the only `82.5` hit is a `**Next**:` pointer at line 30538). Masterplan `82.5.status = "pending"`, `retry_count = 0`. Correct order — log and flip both still ahead. |
| 5 | No verdict-shopping / 3rd-CONDITIONAL | **PASS** | Prior verdict = CONDITIONAL (cycle 1, this file). **Evidence genuinely changed between spawns** — the discriminating test: `api.ts` 19:54:30, `types.ts` 19:55:02, capture PNG 19:57, `live_check_82.5.md` 19:58:19, `experiment_results` 19:58:49, all POST-dating the cycle-1 verdict. This is the documented cycle-2 fresh-respawn, not a verdict shop. Zero `result=CONDITIONAL` entries for 82.5 in `harness_log.md`, so this would be the 2nd verdict, not the 3rd — auto-FAIL escalation does not trigger. |

### C2.2 Deterministic checks (my shell, not replayed)

```
immutable cmd: 16 passed, 1 warning in 1.92s          IMMUTABLE_EXIT=0
backend imports (4 modules)                            OK, MIN_MFE_PCT = 1.0
GET /api/health                                        200
npx tsc --noEmit                                       tsc_exit=0  (0 lines output)
npx eslint <3 changed frontend files>                  0 errors, 2 warnings, exit 0
ruff F821,F401,F811 over 6 derived files               5 errors, exit 1
```

Lint scope **derived**, never typed: `git diff --name-only HEAD -- '*.py'` (5) **plus**
`git ls-files --others --exclude-standard -- '*.py'` (1) — the untracked new test file is
invisible to `git diff` and would otherwise have escaped the gate. Count asserted non-empty
(6) before reading the exit code. Split with `tr '\n' '\0' | xargs -0`, because the canonical
`qa.md` §1a one-liner silently lints ZERO files under zsh (vacuity shape #9).

Ruff triage: **4 of 5 are pre-existing** — I linted `git show HEAD:backend/tests/test_dod4_tier1_coverage_investment.py`
and got the byte-identical 4 F401s, and the actual 82.5 diff to that file is only the
capture-ratio test rename. **1 is new and still unfixed**: `import statistics as st` at
`test_phase_82_5_exit_quality_metrics.py:23`. Cycle 1 already filed this as N-1; it was not
addressed. NOTE-level (cosmetic, no behaviour).

*(`${PIPESTATUS[0]}` expands empty in zsh — my first `tsc` capture produced `tsc_exit=`.
Re-run with a plain redirect. Same trap cycle 1 hit; recording it so the next reader doesn't
trust a blank exit code.)*

### C2.3 Attack A — is criterion 3 satisfied only by the floor?

I ran **my own** 8-mutant matrix (in-memory source patching; repo working tree never
touched; every mutation asserts its target occurs **exactly once** before replacing, so a
no-op `str.replace` raises instead of passing silently).

| Mutant | What it breaks | Killed |
|---|---|---|
| CONTROL | — | 0 (16 passed) |
| QM1_CAPTURE_MEAN | capture headline → arithmetic mean | **2** |
| QM2_EDGE_MEAN | edge headline → arithmetic mean | **2** |
| QM3_FLOOR_OFF | `MIN_MFE_PCT 1.0 → 0.0` | **3** |
| QM4_FLOOR_CLAUSE_DELETED | delete the floor clause entirely | **3** |
| QM5_UNDEF_CAPTURE_ZERO | undefined capture → fabricated `0.0` | **4** |
| QM6_MAE_ZERO_DROPPED | `mae==0` → `None` (dropped again) | **4** |
| QM7_MEDIAN_ALLOWS_INF | `_median_or_none` returns a non-finite median | **1** |
| QM8_MEDIAN_UNSORTED_LOW | median → `min` (order-statistic swap) | **4** |

All 8 killed. QM7 and QM8 are mine and in neither Main's nor cycle-1's matrix; both die.

**The measured answer, and it goes further than cycle 1 or Main.** The verbatim criterion-3
test `test_one_extreme_outlier_moves_each_headline_by_under_20_percent` **SURVIVES BOTH**
mean mutants — QM1 *and* QM2. It loops over `("capture_median", "edge_median")`, and each
leg is estimator-agnostic **for a different reason**:

```
criterion-3 poison edge value = mfe/|mae| = 1e-4/1e-4 = 1.0

CAPTURE: n_defined 20 -> 20   poison ADMITTED? False   (excluded by the 1.0pp floor)
EDGE   : n_defined 20 -> 21   poison ADMITTED? True    (no floor on the edge path)

EDGE mean   before/after = 3.2355 / 3.1290   drift = 3.29%   <- passes <20%
EDGE median before/after = 3.3046 / 3.2759   drift = 0.87%   <- passes <20%
ordinary edge range 2.500 .. 3.718; the poison's value 1.0 is not an outlier at all
```

So on the **edge** tile the criterion's own poison specification is inert: `mfe/mae = 1e-4`
divides to exactly **1.0**, an unremarkable ratio. The floor has nothing to do with it. Main's
§12 "Criterion 3, on the record" attributes the whole survival to the floor — *"the VERBATIM
criterion-3 test does pass under a mean, because the 1.0pp floor excludes the mfe=1e-4
poison"* — which is correct for capture and **wrong for edge**, where the poison is admitted
and simply isn't extreme. See FINDING A-1.

**Is criterion 3 nonetheless covered?** Capture: **yes, genuinely** — QM1 is killed by
`test_the_median_itself_is_load_bearing_not_just_the_floor` (poison `mfe=1.5` *above* the
floor). Edge: QM2 *is* killed, but by `test_the_new_aggregate_is_sane_on_the_same_real_data`
(a value pin, `edge_median == 3.09`) and `test_an_infinite_median_is_reported_as_none_never_as_infinity`
(a non-finite guard — `mean([inf]*5) = inf`). **Neither is an outlier-robustness guard.**
No test anywhere poisons the edge aggregate with a genuinely extreme edge value. See FINDING A-2.

**Is `test_the_median_itself_is_load_bearing_not_just_the_floor` itself vacuous? No.** Its
three legs are each load-bearing and I checked them separately: the `n_defined == before+1`
leg proves the poison is admitted rather than floor-filtered (it is the leg that would expose
the test if the poison drifted below 1.0pp); the `compute_capture_ratio(...) < -500` leg pins
the poison's potency; and the closing `mean_after vs mean_before > 1.0` leg proves the row
discriminates between estimators. QM1's killing assertion is line **201**, `assert drift <
0.20` (`capture median moved 4766.7%`) — naming the assertion, not crediting the test whole
(vacuity shape #11).

### C2.4 Attack B — is the fixture genuine, un-softened, and load-bearing?

**Genuine: proven against the live book, not taken on trust.** I curled the restarted
`/api/paper-trading/mfe-mae-scatter` and compared its 32 `points` to the 32 fixture rows as a
**multiset** on `(ticker, mfe_pct, mae_pct, realized_pnl_pct)`:

```
round=4dp   fixture-only=1  live-only=1
    F: ('AMD', 2.1286, -11.3645, -11.316 )
    L: ('AMD', 2.1286, -11.3645, -11.3161)
round=3dp   fixture-only=0  live-only=0      <- IDENTICAL
round=2dp   fixture-only=0  live-only=0
```

The sole divergence is a 4th-decimal rounding on one AMD row. The fixture **is** the live
book. *(Method note: my first pass keyed on `(ticker, mfe_pct)` and reported a phantom MU
discrepancy — two MU rows share `mfe_pct=0.0`, so the dict collided. The multiset comparison
above is the correct derivation; recording the false start so the next reader doesn't trust
the first number.)*

Every criterion-4 figure re-derived by me from the RAW inputs:

| Claim | My re-derivation | Match |
|---|---|---|
| legacy mean capture −42.0785 | −42.0785 (stored) / −42.0918 (re-derived) | both \|·\| > 40 |
| legacy mean edge 86.9218 over 26 rows | 86.9218 over **26** | yes |
| 8 × `mfe==0`, 6 × `mae==0` | 8, 6 | yes |
| min capture −1269.5726 (000660.KS) | −1269.5726 | yes |
| max edge 1483.3388 (INTC) | 1483.3388 | yes |

**No hand-editing.** Stored `capture_ratio_legacy` agrees with the re-derived old formula on
**31 of 32** rows to <0.02; the one divergence is 000660.KS (−1269.5726 stored vs −1270.0
re-derived), exactly what 4-dp rounding of `mfe_pct: 0.0001` does to a near-zero division.
That is the signature of real data, not of tuning.

**Load-bearing: 6 of my own fixture mutants, all caught.**

| Fixture mutant | Killed | Criterion-4 guard |
|---|---|---|
| QF1 blowup row tamed (`mfe→5.0`) | 3 | **DIES** |
| QF2 blowup row deleted | 6 errors | **DIES** (module `assert len(rows)==32`) |
| QF3 legacy column clipped ±100 | 1 | **DIES** |
| QF4 `mfe==0` rows filled to 3.0 | 4 | **DIES** |
| QF5 `mae==0` rows widened to −1.5 | 3 | survives (correct — mae does not enter capture; the *edge* pin dies) |
| QF6 **only** the stored column softened, raw inputs left genuine | 1 | **DIES** |

QF6 is the one cycle-1 could not reach: it confirms criterion 4's test *is* sensitive to the
column it averages. Combined with cycle-1's F1 (raw inputs edited, stored column left stale →
test stays green), the picture is complete: the guard is blind in exactly one direction, and
I independently closed that hole by re-deriving the formula from raw inputs (31/32 agreement
above). Cycle-1's FINDING #4-A stands as a NOTE, not a blocker.

**Credentials / PII: clean.** 4223 bytes, five keys only
(`ticker, mfe_pct, mae_pct, realized_pnl_pct, capture_ratio_legacy`). Zero token-like
strings, zero dates, zero emails, zero account identifiers, zero position sizes, zero NAV.

### C2.5 Attack C — consumer-contract break on `float | None`

Consumers enumerated by repo-wide grep over `*.py`/`*.ts`/`*.tsx`/`*.sql`/`*.json`
(excluding `.venv`, `node_modules`, build dirs), not from the disclosure.

- **BQ writes** — `capture_ratio` is `FLOAT64` with no `REQUIRED` mode
  (`scripts/migrations/add_round_trip_schema.py:59,80`) → NULLABLE. `None` is a legal write.
  `paper_trader._ROUND_TRIP_FIELDS` prune path is value-agnostic.
- **BQ reads** — nothing reads the stored column back; `pair_round_trips` recomputes.
- **Optimizer / go-live gate / Slack bot / meta_evolution / backtest / SQL** — zero hits.
  (`tests/meta_evolution/test_alpha_velocity.py:130` is a generic `components` dict.)
- **Frontend** — `api.ts:529` `avg_capture_ratio: number | null` and `:547`
  `capture_ratio: number | null` are now correct (**cycle-1 C-1 CLOSED**). All three render
  paths in `MfeMaeScatter.tsx` discriminate on `=== null` before multiplying (`:130` edge,
  `:136-138` capture, `:197` tooltip). `getPaperRoundTrips` still has **zero call sites** and
  `round_trip_summary` is consumed by no component, so there is no unguarded render anywhere.
- **Type surface matches the payload exactly (cycle-1 N-3 CLOSED).** I diffed the declared
  key set against the LIVE `/round-trips` response: declared = emitted, **zero
  declared-but-not-emitted keys**.
- `npx tsc --noEmit` → **exit 0**, zero output.

### C2.6 Attack D — the three-copy de-duplication, verified BY EXECUTION

Not by reading source. I curled all three live surfaces on the same book:

```
/performance      -> round_trip_summary : 0.6304  0.6446  n 20/12  floor 1.0   (n_round_trips 32)
/mfe-mae-scatter  -> summary            : 0.6304  0.6446  n 20/12  floor 1.0   (n_points 32)
/round-trips      -> top level          : 0.6304  0.6446  n 20/12  floor 1.0
                                          + 12 of 32 per-trip capture_ratio = null
```

**Three-way parity. The de-duplication claim is TRUE.** *(My first read of `/round-trips`
reported all-None because I looked under a `summary` key that does not exist — the summary is
spread at the top level. Corrected above; recording the false start.)*

FINDING D-1 from cycle 1 (INV5's guard is a literal source scan defeated by whitespace) is
unchanged and still a WARN — the property is real, as just executed, but
`test_performance_and_scatter_surfaces_share_one_definition` guards it with
`assert "realized_pnl_pct / mfe" not in getsource(mod)`. Named fix: replace the string scan
with the parity assertion above.

### C2.7 Attack E — the pre-existing failure and the changed test

**Full `backend/tests/`: 32 failed, 2486 passed, 12 skipped, 5 xfailed, 1 xpassed (154s).**
Cycle 1 measured 31. I did not average the delta away — I measured it.

Derived the 20 failing files from the run (never hand-typed), then the 10 that import any
changed module, and ran that set twice: current tree vs a **HEAD-module overlay** (pre-82.5
`perf_metrics`, `paper_round_trips`, `paper_trader`, `paper_trading` installed into
`sys.modules`; the overlay asserts each HEAD blob actually DIFFERS from the working tree, so
an overlay that overlaid nothing raises instead of faking an "identical" result).

```
CURRENT tree :  19 failed, 169 passed
HEAD overlay :  20 failed, 168 passed        (overlay installs: 4 of 4)

SYMMETRIC DIFFERENCE
  only failing in CURRENT (caused by 82.5) :  <EMPTY>
  only failing under HEAD  (fixed by 82.5) :  test_paper_trader_execute_sell_capture_ratio_none_when_no_gain
```

**Zero regressions attributable to 82.5**, by symmetric difference rather than by count. The
single asymmetry points the right way: the renamed test fails against HEAD code (which returns
`0.0`) and passes against the fix — which is what a legitimate contract update looks like.

`test_paper_trader_execute_buy_average_up_recomputes_avg_entry` fails under **both** → genuinely
pre-existing, independently confirmed. Cycle-1's root cause holds: the live kill switch is
engaged (`{'paused': True, 'pause_reason': 'manual'}`), and the capture below shows
`KILL PAUSED` on the operator's own status bar.

The 32-vs-31 delta versus cycle 1 is **environmental, not code**: the failing set is dominated
by log-scraping and live-service tests (`test_phase_23_2_6_backend_log_has_skipping_buy_evidence`,
`test_phase_36_12_*`, `test_phase_23_2_4_pause_resume_..._live`) whose inputs moved when the
backend was restarted at 19:55 as part of this step's own remediation. The symmetric-difference
run above is the controlled measurement and it is empty.

**The changed test was changed for the right reason — ACCEPTED.**
`test_paper_trader_execute_sell_capture_ratio_zero_when_no_gain` asserted `capture_ratio == 0.0`
for `MFE == 0`; that assertion *encoded the defect*. The rename to `..._none_when_no_gain` plus
`assert result["capture_ratio"] is None` is a contract update, not a red-test-made-green: the
production behaviour deliberately changed, the original intent (no NaN, no `ZeroDivisionError`)
is still asserted because the call still executes and returns, the rename makes it auditable in
`git log`, and QM5_UNDEF_CAPTURE_ZERO (restore `0.0`) kills 4 tests, so the new assertion is
load-bearing. I verified the 82.5 diff to that file is *only* this rename + docstring.

### C2.8 Live UI gate (qa.md §1c)

**I attempted the capture myself, as §1c requires** (browser tools loaded via the deterministic
`select:` form):

```
browser_navigate("http://localhost:3000/paper-trading/exit-quality")
  -> Page URL: http://localhost:3000/login     (NextAuth wall, no session in this profile)
curl :3100 -> 000 (down)     curl :3000/paper-trading/exit-quality -> 302
```

The skip-auth :3100 instance was torn down after Main's capture, and standing it up is **Main's
lifecycle responsibility, never the evaluator's**. So this verdict rests on a **Main-produced
capture — the explicitly-degraded fallback under §1c, and I am saying so here as §1c requires.**

I did not accept it on trust. I read the PNG and independently corroborated every claim it makes
against the live API:

| Tile | PNG renders | Live API | Agree |
|---|---|---|---|
| EDGE RATIO | **3.09**, hint `median(MFE / \|MAE\|)` | `edge_ratio: 3.09` | yes |
| AVG CAPTURE | **63%**, hint `median(realized_pnl / MFE), n=20` | `avg_capture_ratio: 0.6304` → 63% | yes |
| ROUND-TRIPS | 32, "closed only" | `n_points: 32` | yes |
| LEAKERS | 0, `capture < 40% & MFE > P75` | `n_leakers: 0` | yes |

The hints do say **median**, not mean, so the estimator change is visible to the operator rather
than silently altering a number under an unchanged label; and `n=20` reaches the UI, so a reader
can see the headline covers 20 of 32 round-trips. Sidebar/status bar also corroborate independent
facts I measured separately (`KILL PAUSED`, `GATE NOT ELIGIBLE 2/5`). No emoji in the UI or in
any changed file (checked by codepoint scan). **Cycle-1 blockers C1 and C2 are genuinely closed.**

### C2.9 NEW FINDING E-1 — the live_check run silently rewrote two TRACKED files

Not disclosed anywhere in `experiment_results_82.5.md` or `live_check_82.5.md`, and not present
in cycle 1 (it did not exist yet — it is collateral of the cycle-1 remediation itself).

```
frontend/next-env.d.ts   19:56:35   -/// <reference path="./.next/types/routes.d.ts" />
                                    +/// <reference path="./.next-functional/types/routes.d.ts" />
frontend/tsconfig.json   19:56:35   + ".next-functional/types/**/*.ts"   (added to "include")
```

Both mtimes land inside the :3100 window `live_check_82.5.md` §A describes, and the edit
direction matches its own `PLAYWRIGHT_DIST_DIR=.next-functional` exactly — causally attributable
to this step's capture run. `next-env.d.ts` carries the literal comment *"This file should not be
edited"*.

Why it matters:
- **Both referenced dirs are GITIGNORED** (`frontend/.gitignore:3` → `.next-functional/`;
  `.gitignore:21` → `.next/`) and `.next-functional` is **entirely untracked** (`git ls-files` →
  0). Committing these two files persists a reference to a Playwright-only build directory that
  does not exist on a fresh checkout.
- **`git add -An` confirms both are staged for sweep**, so the auto-commit hook's `git add -A`
  will ship them under 82.5's name. This is `feedback_audit_the_commit_not_the_diff` exactly.
- **Both my `tsc --noEmit` green and cycle-1's were measured against the MUTATED tsconfig.** I
  checked the blast radius: `.next/types/routes.d.ts` and `.next-functional/types/routes.d.ts`
  are **byte-identical**, so the type surface is materially the same and the green is not fake —
  but that is luck, not design.
- Operator impact today: **none.** `:3000` is healthy (`/ → 302`, `/login → 200`), launchd
  `com.pyfinagent.frontend` pid 863 intact; these are TS declaration files and do not affect
  runtime.

Named fix (mechanical, pre-flip): `git checkout -- frontend/next-env.d.ts frontend/tsconfig.json`
before the commit, and disclose the `.next-functional` side effect in `live_check_82.5.md` §A so
the next capture run expects it.

Also still open from cycle 1 and **not addressed**: the ruff gate exits **1** on the derived
scope. Four F401s are pre-existing (proven byte-identical at HEAD); **one is new and introduced
by this step** — `import statistics as st` at `test_phase_82_5_exit_quality_metrics.py:23`, filed
as N-1 in cycle 1 and still present. Cosmetic, but it is a new lint error left standing after
being flagged.

### C2.10 Criterion-by-criterion coverage map (each with its named killing mutation)

| # | Covering evidence | Mutation that turns it red | Executed by me |
|---|---|---|---|
| 1 | `test_mfe_zero_does_not_produce_an_unbounded_or_sign_flipped_capture` (8 real rows → `None`; median finite AND > 0), `test_an_undefined_capture_is_distinguishable_from_a_real_zero`, `test_an_all_undefined_fixture_reports_none_not_zero` | **QM5_UNDEF_CAPTURE_ZERO** → 4 killed | yes |
| 2 | `test_mae_zero_is_handled_explicitly_and_not_dropped` (6 real rows → `+inf`; `n_defined+n_undefined == n_points == 32`), `test_edge_ratio_degenerate_cases_are_each_defined`, `test_an_infinite_median_is_reported_as_none_never_as_infinity` | **QM6_MAE_ZERO_DROPPED** → 4 killed; **QM7_MEDIAN_ALLOWS_INF** → 1 killed | yes |
| 3 | **capture:** `test_the_median_itself_is_load_bearing_not_just_the_floor` (poison ABOVE the floor) + the floor tests. **edge:** verbatim test only, and it is estimator-agnostic (FINDING A-2) | **QM1_CAPTURE_MEAN** → 2 killed, killing assertion `:201`; **QM3/QM4** floor → 3 each; **QM8** order-statistic → 4 | yes |
| 4 | `test_the_old_mean_formula_blows_up_on_the_committed_real_data` (\|−42.0785\| > 40, worst < −1000) + `test_the_old_edge_mean_also_blows_up_on_the_real_data` | fixture mutants **QF1, QF2, QF3, QF4, QF6** all kill the guard | yes |

All four criteria are **COVERED**. No `Missing_Assumption` on any criterion as written.
Criterion 3's *edge half* is covered only weakly — see FINDING A-2, WARN-level under the qa.md
§4c wiring because QM2 is still killed by two genuine (if differently-aimed) guards.

### C2.11 Adversarial worst-of-N-lenses (P1 money-path step)

- **correctness lens — PASS.** The Cauchy argument is sound and the code matches it. The
  asymmetric treatment (exclude for capture, rank `+inf` for edge) is correct and non-obvious.
  `_median_or_none` refuses a non-finite headline (QM7 proves that guard is live). The
  `mfe > 0` / `mfe < floor` split is deliberately two statements after a real self-caught bug.
  Three live surfaces agree. Leakage now requires a DEFINED capture, which is the right call and
  is disclosed.
- **does-it-reproduce lens — PASS.** Immutable command exit 0; 8 code mutants + 6 fixture
  mutants all killed; every quantitative claim re-derived independently and matched; fixture
  proven identical to the live book; regression symmetric difference empty.
- **scope-honesty lens — FAIL → CONDITIONAL.** The cycle-1 items were closed properly and §5 was
  corrected **in place** rather than left true-looking above a contradicting addendum, which is
  the right instinct. But: the capture run mutated two tracked files and neither artifact says so
  (E-1); §12's account of criterion 3 attributes the edge leg's survival to the floor, which is
  measurably not the cause (A-1); and a lint error the previous Q/A named is still there.

`verdict = min(lens verdicts)` = **CONDITIONAL**.

---

## C2.12 Verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 4 immutable criteria are COVERED by non-vacuous guards and the immutable command exits 0 (16 passed). Verified with MY OWN 8 code mutants (2 of them -- QM7 non-finite-median guard, QM8 order-statistic swap -- in neither Main's nor cycle-1's matrix) and 6 fixture mutants, every one killed, each asserting its target exists exactly once before replacing. The fixture is PROVEN genuine, not trusted: its 32 rows are an exact multiset match to the 32 live points from the restarted /mfe-mae-scatter at 3dp (one AMD row differs only at the 4th decimal), and the stored legacy column agrees with the re-derived old formula on 31/32 rows with the single divergence explained by 4-dp rounding of mfe=0.0001. Three-copy de-duplication proven BY EXECUTION: /performance, /mfe-mae-scatter and /round-trips all return 0.6304 / 0.6446 / n 20-12 / floor 1.0 on the same book. ZERO regressions by SYMMETRIC DIFFERENCE (not counts) against a HEAD-module overlay: nothing fails in the current tree that passes at HEAD; the only asymmetry is the renamed capture test failing against HEAD code, which is the correct direction. All three cycle-1 blockers are genuinely closed -- backend restarted (PID 62664 @ 19:55:37, live endpoint now serves the new values), api.ts:529/:547 now number|null with the declared key set matching the live payload exactly, and a live UI capture exists. CONDITIONAL, not PASS, on three items. (1) NEW, undisclosed: the live_check :3100 run silently rewrote two TRACKED files at 19:56:35 -- frontend/next-env.d.ts now references ./.next-functional/types/routes.d.ts instead of ./.next/, and frontend/tsconfig.json gained .next-functional/types/**/*.ts -- both pointing at GITIGNORED, entirely untracked build dirs, and git add -An confirms both will ship under 82.5's name; next-env.d.ts literally says 'This file should not be edited'. Blast radius is contained only because the two routes.d.ts files happen to be byte-identical, so the tsc green is real but lucky. (2) experiment_results section 12 attributes the verbatim criterion-3 test's survival under a mean entirely to the 1.0pp floor; measured, that is true for capture (poison excluded, n_defined 20->20) and FALSE for edge (poison ADMITTED, n_defined 20->21, but its value is mfe/mae = 1e-4/1e-4 = exactly 1.0, an ordinary ratio inside the base range 2.5-3.718, giving 3.29% drift under a mean) -- so the stated cause is wrong for half the test. (3) the ruff gate still exits 1 on the derived scope: 4 F401s are proven pre-existing byte-identical at HEAD, but one is NEW from this step (import statistics as st, test file line 23), filed as N-1 by cycle 1 and left unfixed. On criterion 3 specifically: it IS satisfied as written and is NOT floor-only for the capture tile, because test_the_median_itself_is_load_bearing_not_just_the_floor dies under QM1 at line 201 with a poison ABOVE the floor and carries three separately-verified anti-vacuity legs; but NO test anywhere poisons the EDGE headline with a genuinely extreme edge value, so the edge half of 'both tiles' rests on a value pin and a non-finite guard rather than a robustness guard.",
  "violated_criteria": [
    "Invalid_Precondition: the live_check capture run mutated two tracked frontend config files to point at gitignored build dirs, undisclosed, and staged to ship under this step",
    "Contradiction: experiment_results §12 attributes the criterion-3 edge leg's estimator-agnosticism to the 1.0pp floor, which is measurably not the cause",
    "Missing_Assumption: no guard demonstrates the EDGE headline is robust to an extreme outlier, though criterion 3 says 'both tiles'",
    "Threshold_Not_Met: ruff F821/F401/F811 gate exits 1 on the derived scope with one NEW error introduced by this step and flagged unfixed since cycle 1"
  ],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "The :3100 skip-auth dev server started for the live_check capture (LIGHTHOUSE_SKIP_AUTH=1 ... PLAYWRIGHT_DIST_DIR=.next-functional npx next dev --port 3100) rewrote frontend/next-env.d.ts and frontend/tsconfig.json at 19:56:35",
      "state": "next-env.d.ts: '/// <reference path=\"./.next/types/routes.d.ts\" />' -> './.next-functional/types/routes.d.ts'. tsconfig.json 'include' gained '.next-functional/types/**/*.ts'. Both .next/ and .next-functional/ are gitignored (frontend/.gitignore:3, .gitignore:21) and .next-functional is untracked (git ls-files -> 0 files). git add -An lists both, so the auto-commit hook's git add -A ships them under 82.5's name. Neither experiment_results_82.5.md nor live_check_82.5.md discloses the mutation. next-env.d.ts carries the comment 'This file should not be edited'. Both cycle-1's and my tsc --noEmit exit 0 were measured against the mutated tsconfig; blast radius is nil today only because .next/types/routes.d.ts and .next-functional/types/routes.d.ts are byte-identical, and operator :3000 is healthy (302 / login 200, launchd pid 863).",
      "constraint": "feedback_audit_the_commit_not_the_diff: run git add -An before every flip -- a foreign or collateral change must not ship under this step's name; and a step's own remediation must disclose the tracked-file side effects it caused. Fix: git checkout -- frontend/next-env.d.ts frontend/tsconfig.json before committing, and record the .next-functional side effect in live_check_82.5.md §A."
    },
    {
      "violation_type": "Contradiction",
      "action": "experiment_results_82.5.md §12 'Criterion 3, on the record' states: 'The VERBATIM criterion-3 test does pass under a mean, because the 1.0pp floor excludes the mfe=1e-4 poison before the estimator sees it.'",
      "state": "Measured by me: test_one_extreme_outlier_moves_each_headline_by_under_20_percent loops over BOTH ('capture_median','edge_median') and survives BOTH mean mutants. For capture the stated cause is right (poison excluded, capture_n_defined 20 -> 20). For EDGE the stated cause is wrong: there is no floor on the edge path, the poison IS admitted (edge_n_defined 20 -> 21), and it is simply not extreme -- mfe/|mae| = 1e-4/1e-4 = exactly 1.0, against an ordinary base range of 2.500..3.718, producing 3.29% drift under a mean and 0.87% under a median, both far inside the 20% bound.",
      "constraint": "qa.md §4b: every quantified or causal claim in the handoff must be reproducible; a stated cause that does not reproduce for half the assertion it explains is a Contradiction. Fix: restate §12 to name the two distinct reasons (floor for capture, inert poison specification for edge)."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Criterion 3 requires 'the reported aggregate for BOTH tiles is robust to a single extreme outlier'; the suite covers the capture tile with test_the_median_itself_is_load_bearing_not_just_the_floor but has no equivalent for the edge tile",
      "state": "QM2_EDGE_MEAN (edge headline -> arithmetic mean) is killed by exactly two tests: test_the_new_aggregate_is_sane_on_the_same_real_data (a value pin, edge_median == 3.09) and test_an_infinite_median_is_reported_as_none_never_as_infinity (a non-finite guard -- mean([inf]*5) = inf). Neither is an outlier-robustness guard. No test poisons the edge aggregate with a genuinely extreme edge value.",
      "constraint": "qa.md §4c: name the concrete mutation that makes each criterion's guard fail. Per the §4c verdict wiring this is WARN-level rather than blocking, because a vacuous guard alongside genuine behavioral guards is a named-fix finding, not sole-coverage vacuity. Fix: add an edge twin of the median-load-bearing test with a real edge outlier (e.g. mfe=50, mae=-0.01 -> edge 5000) and assert a mean fails it."
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "uvx ruff check --select F821,F401,F811 over the scope derived from git diff --name-only HEAD -- '*.py' PLUS git ls-files --others --exclude-standard -- '*.py' (6 files, count asserted non-empty before reading the exit code, split with tr/xargs -0 because zsh does not word-split)",
      "state": "Found 5 errors, ruff_exit=1. Four are pre-existing -- linting git show HEAD:backend/tests/test_dod4_tier1_coverage_investment.py reproduces the byte-identical 4 F401s, and 82.5's only diff to that file is the capture-ratio test rename. One is NEW and introduced by this step: F401 'import statistics as st' at backend/tests/test_phase_82_5_exit_quality_metrics.py:23, filed as N-1 by the cycle-1 Q/A and still unfixed.",
      "constraint": "qa.md §1a: non-zero ruff exit = FAIL, quoted verbatim. Cosmetic in effect (no behaviour), but it is a new error left standing after being named. Fix: delete the unused import."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item_audit",
    "evidence_changed_between_spawns_verified_by_mtime_not_verdict_shopping",
    "third_conditional_counter_check_harness_log_and_masterplan",
    "immutable_verification_command_exit_0",
    "python_lint_gate_ruff_scope_derived_from_git_diff_PLUS_untracked",
    "pre_existing_lint_baseline_proven_against_HEAD_blob",
    "frontend_tsc_noemit_exit_0_with_real_exit_capture",
    "frontend_eslint_scoped_to_changed_files_exit_0",
    "backend_import_smoke_4_modules",
    "live_api_health_200",
    "independent_mutation_matrix_8_code_mutants_all_killed",
    "mutation_targets_asserted_present_exactly_once_before_replace",
    "killing_assertion_identified_by_line_number",
    "criterion_3_estimator_agnosticism_measured_for_BOTH_tiles_separately",
    "anti_vacuity_legs_of_the_median_load_bearing_test_checked_individually",
    "fixture_multiset_compared_to_LIVE_production_rows_at_2_3_4dp",
    "fixture_numbers_re_derived_from_raw_inputs_not_stored_columns",
    "fixture_hand_edit_detector_stored_vs_rederived_31_of_32",
    "6_fixture_softening_mutants_all_caught",
    "fixture_pii_credential_scan",
    "consumer_grep_py_ts_tsx_sql_json_repo_wide",
    "bq_schema_nullability_confirmed",
    "frontend_null_render_paths_audited_all_three",
    "declared_vs_emitted_key_set_diffed_against_live_payload",
    "three_live_surfaces_curled_and_compared_for_parity",
    "full_backend_regression_2530_tests",
    "HEAD_module_overlay_symmetric_difference_empty",
    "head_overlay_asserts_blobs_differ_so_a_noop_overlay_raises",
    "pre_existing_failure_confirmed_failing_under_BOTH_trees",
    "kill_switch_state_confirmed_as_root_cause",
    "playwright_live_ui_capture_ATTEMPTED_blocked_by_auth_wall_degraded_fallback_disclosed",
    "main_produced_capture_png_read_and_cross_checked_against_live_api",
    "git_add_dry_run_commit_scope_audit",
    "undisclosed_tracked_file_mutation_detected_and_attributed_by_mtime",
    "gitignore_status_of_referenced_build_dirs_verified",
    "emoji_codepoint_scan_on_changed_files"
  ]
}
```

**3rd-CONDITIONAL rule:** does NOT trigger. `grep 'phase=82.5' handoff/harness_log.md` → **0**
entries, so zero logged `result=CONDITIONAL` for this step-id; masterplan `retry_count = 0`.
Cycle 1's CONDITIONAL lives only in this file, never in the log. By substance this is verdict
**#2**, not #3.

**Next move (documented cycle-2 flow, and all four items are mechanical — none touches the
math):**
1. `git checkout -- frontend/next-env.d.ts frontend/tsconfig.json`, and add the
   `.next-functional` side effect to `live_check_82.5.md` §A so the next capture run expects it.
2. Restate `experiment_results_82.5.md` §12 with the two distinct causes for criterion 3
   (floor for capture; inert poison specification for edge).
3. Delete `import statistics as st` from the new test file; re-run the ruff gate.
4. Optional but recommended, and worth its own queued step rather than scope creep here
   (`feedback_queue_discovered_defects_in_masterplan`): an edge-tile robustness test, and
   replacing INV5's source scan with the executed parity assertion (FINDING D-1).

Then spawn a FRESH Q/A on the changed evidence.

---
---

# CYCLE 3 — Fresh Q/A on CHANGED evidence

**Evaluator:** Layer-3 Q/A (fresh instance, Agent-tool `qa` rail)
**Date:** 2026-08-04
**Cycle:** 3
**Prior verdicts on this step-id:** CONDITIONAL (cycle 1), CONDITIONAL (cycle 2) — both in this file.
**Verdict:** IN PROGRESS — written incrementally below; final JSON at §C3.14.

## C3.0 — The 3rd-CONDITIONAL rule is LIVE this cycle (stated up front, before any finding)

`qa.md` Constraints: *"Before issuing a CONDITIONAL verdict, grep `handoff/harness_log.md`
for the current step-id. If there are already 2+ `result=CONDITIONAL` entries for this
step-id (i.e. this would be the third consecutive CONDITIONAL), return FAIL instead."*
CLAUDE.md F1: *"if a single step-id accumulates 3+ consecutive CONDITIONAL verdicts without
an intervening PASS or FAIL, the next Q/A pass MUST return FAIL."*

Both prior cycles resolved this by grepping `harness_log.md` and finding 0 — correct on the
letter, but the log is empty **by construction**: LOG-LAST means nothing is appended until
after a PASS, so the logged counter can never reach 2 while the step is still failing. Read
literally, the counter is unreachable and the escalation could never fire on any step. The
governing intent is explicit in CLAUDE.md — *"This prevents the harness from logging instead
of correcting"* — and the durable record of prior verdicts for this step-id is THIS FILE,
which carries two CONDITIONALs with no intervening PASS or FAIL.

**Therefore, binding on this cycle: if my substantive verdict is CONDITIONAL, I return FAIL.**
PASS remains fully available and is decided on the evidence.

## C3.1 — Session start state (measured)

```
$ git log --oneline -3
8c54f541 chore: auto-changelog hook entry for 2e040941
2e040941 fix(82.7): make credential redaction reachable from every entry point
4ecc41e6 chore: auto-changelog hook entry for 13cd9b86
```

### C3.2 Harness-compliance audit (5 items, run FIRST)

| # | Item | Result | Evidence measured by me |
|---|------|--------|--------------------------|
| 1 | Researcher spawned BEFORE the contract | **PASS** | `research_brief_82.5.md` mtime **19:19:17** < contract **19:22:27**. Envelope `handoff/current/qa_returns/ws1xvyys5.output.json`: `gate_passed=True`, `external_sources_read_in_full=7` (≥5 floor), `urls_collected=26` (≥10), `recency_scan_performed=True`, `internal_files_inspected=14`, `tier=moderate`. Workflow rail — satisfies `feedback_both_dev_mas_agents_use_workflows`. |
| 2 | Contract written BEFORE GENERATE | **PASS** | contract 19:22:27 → fixture 19:23:21 → code 19:25:26/19:31:38 → test 19:32:08 → results 19:33:02, amended 20:21:08. Strictly ordered. |
| 3 | `experiment_results` present w/ file list + verbatim output | **PASS** | 296 lines; §8 verbatim pytest, §9 mutation matrix, §12 cycle-2 closure, §13 cycle-3 closure. |
| 4 | **LOG-LAST** | **PASS** | `grep -c 'phase=82.5' handoff/harness_log.md` → **0** (only hit is a `**Next**:` pointer at line 30538). Masterplan `82.5.status="pending"`, `retry_count=0`. Last log entry is Cycle 1139 phase=82.7. Log and flip both still ahead — correct order. |
| 5 | Verdict-shopping / 3rd-CONDITIONAL | **PASS on shopping; rule ACKNOWLEDGED on the counter** | Evidence genuinely CHANGED between spawns — the discriminating test: `test_phase_82_5_*.py` 20:19:26, `perf_metrics.py` 20:19:43, `experiment_results` + `live_check` 20:21:08, all POST-dating the cycle-2 verdict. Documented cycle-2/3 fresh-respawn, not a shop. See §C3.0 for how I handle the counter. |

### C3.3 Deterministic checks (my shell, nothing replayed)

```
immutable cmd   17 passed, 1 warning in 2.35s              IMMUTABLE_EXIT=0
ruff F821,F401,F811 over a 6-file DERIVED scope            All checks passed!   ruff_exit=0
npx tsc --noEmit                                           tsc_exit=0, 0 lines
npx eslint <3 changed frontend files>                      0 errors, 2 warnings, exit 0
backend imports (4 changed modules)                        4/4 OK, MIN_MFE_PCT = 1.0
GET /api/health                                            200
```

Lint scope **derived, never typed**: `git diff --name-only HEAD -- '*.py'` **plus**
`git ls-files --others --exclude-standard -- '*.py'` (the new test file is untracked and
invisible to `git diff`). N=6 asserted non-empty BEFORE reading the exit code; split with
`tr '\n' '\0' | xargs -0` because zsh does not word-split an unquoted `$FILES` (vacuity
shape #9, which struck both prior cycles). **Cycle-2's ruff finding is genuinely closed:
5 errors → 0.**

### C3.4 The four cycle-2 items — all independently verified CLOSED

| Item | Verified how | Result |
|---|---|---|
| **D1** tracked files mutated by the capture run | `git status --short frontend/next-env.d.ts frontend/tsconfig.json` → EMPTY; file content back to `./.next/types/routes.d.ts`; `grep next-functional frontend/tsconfig.json` → none | **CLOSED**, and the side effect is now disclosed in `live_check_82.5.md` §A with the `globalTeardown` root cause |
| **D2** wrong causal claim on criterion 3 | Re-derived independently — see §C3.6 | **CLOSED**, and the corrected §12 is numerically exact |
| **D3** no edge-tile robustness guard | New `test_the_edge_headline_is_robust_to_an_extreme_outlier`; my QC2/QC12 kill it | **CLOSED** |
| **D4** ruff gate red | `All checks passed!` on the derived 6-file scope | **CLOSED** |

I also re-checked §13's own supporting story rather than accepting it: the claim that
"line 144 is a COMMENT" is true **of the HEAD revision** (line numbers shifted when the
3 import lines were removed):

```
$ git show HEAD:backend/tests/test_dod4_tier1_coverage_investment.py | sed -n '144p'
    # because snapshot() re-acquired the threading.Lock.
```
and the only other `threading` occurrence in HEAD is a test *name* at :142. Ruff was
right, Main's corrected check was right, and the write-up reproduces.

### C3.5 My own mutation matrix — 14 code mutants + 4 fixture mutants

In-memory source patching; the repo working tree was NEVER written. Every mutation
asserts its target occurs **exactly once** before replacing, and asserts the result
differs from the original, so a no-op `str.replace` raises instead of passing silently.
CONTROL asserted at exactly 17 passed before any mutant ran.

| Mutant | Killed | Killing tests |
|---|---|---|
| QC1 capture headline → mean *(caller-mandated)* | **2** | `..._median_itself_is_load_bearing...`, `..._new_aggregate_is_sane...` |
| QC2 edge headline → mean, isolated so it trips only robustness | **1** | `test_the_edge_headline_is_robust_to_an_extreme_outlier` |
| QC3 `_median_or_none` itself becomes a mean | **3** | both robustness tests + value pin |
| QC4 `mae==0` → `None` instead of `+inf` | **4** | |
| QC5 floor 1.0 → 0.5 (partial weakening) | **1** | `test_min_mfe_floor_is_surfaced_not_buried` |
| QC6 capture ratio inverted (`mfe/pnl`) | **3** | |
| QC7 `n_undefined` never incremented | **3** | INV2 counting tests |
| QC8 median allows a non-finite headline | **1** | |
| QC9 `+inf` edges silently dropped before the median | **1** | |
| QC10 undefined capture fabricated as `0.0` at the aggregate | **4** | |
| QC11 `mfe_sum_defined > 0` → `!= 0` | **0** | **EQUIVALENT MUTANT — see below** |
| QC12 edge headline → plain mean (cycle-2 QM2 shape) | **3** | incl. the new robustness test |
| QC13 floor fully removed (1.0 → 0.0) | **3** | |
| QC14 capture headline → plain mean, unguarded | **2** | |

**QC11 is an equivalent mutant, not a coverage gap, and I proved it by execution rather
than by reasoning**: `compute_capture_ratio` admits a row only when `mfe > 0` holds
unconditionally, so every value summed into `mfe_sum_defined` is strictly positive and
`> 0` is identical to `!= 0` for any non-empty set. 200,000 randomised trials over
`mfe ∈ [-50,50]`, `min_mfe_pct ∈ {0.0,0.5,1.0,2.0}` produced **0/200000** admitted rows
with `mfe <= 0`. Reporting it as a finding would have been a false positive.

**Fixture-softening mutants (criterion 4), all NEW:**

| Fixture mutant | Criterion-4 guard |
|---|---|
| QG1 blowup row retuned so the legacy mean is exactly **−39.9** (tightest possible softening) | **DIES** |
| QG2 every `mfe_pct` scaled 100x, legacy column re-derived consistently | **DIES** |
| QG3 the 8 `mfe==0` rows filled to 2.0 | criterion-1 + edge pins die |
| QG4 one `mae==0` row widened to −0.5 | criterion-2 + edge pins die |

Criterion 4 is genuinely fixture-load-bearing: the guard fails at a legacy mean of −39.9,
i.e. it discriminates at the exact threshold the criterion names.

### C3.6 Attack A — the caller's question: is criterion 3 satisfiable only by the floor?

**Measured answer, and it is sharper than "the floor".** The verbatim criterion-3 test
`test_one_extreme_outlier_moves_each_headline_by_under_20_percent` **survives all 14 code
mutants above — every single one.** It cannot be made to fail by reverting either
estimator, by removing or halving the floor, by inverting the ratio, by breaking the
counters, or by allowing a non-finite median.

Re-derived from the production code:

```
criterion-3 poison edge value = mfe/|mae| = 1e-4/1e-4 = exactly 1.0
CAPTURE  n_defined 20 -> 20   poison ADMITTED? False   (excluded by the 1.0pp floor)
EDGE     n_defined 20 -> 21   poison ADMITTED? True    (no floor on the edge path)
base edge range 2.500 .. 3.718 ; EDGE mean drift = 3.29% ; EDGE median drift = 0.87%
base capture values are ALL exactly 0.6, so the capture median cannot move at all
```

So the criterion's own poison is inert for **two different reasons** — excluded on the
capture leg, and simply not an outlier on the edge leg (1.0 sits below the ordinary
range). Main's corrected §12 states exactly this, and my 3.29% reproduces his 3.29% to
the digit.

**Is that a defect in the work? No — it is a property of the IMMUTABLE criterion**, whose
poison specification (`mfe/mae = 1e-4`) Main cannot change. What matters is whether the
step's coverage of criterion 3's *intent* is vacuous, and it is not:

- **capture leg** — `test_the_median_itself_is_load_bearing_not_just_the_floor` poisons
  ABOVE the floor (`mfe=1.5`) and dies under QC1/QC14 at `:200 assert drift < 0.20`
  ("capture median moved 4766.7%"). Naming the killing assertion, not crediting the test
  as a whole (vacuity shape #11).
- **edge leg** — the cycle-3 addition `test_the_edge_headline_is_robust_to_an_extreme_outlier`
  poisons with a genuine edge outlier (`mfe=50, mae=-0.01` → 5000) and dies under
  QC2/QC12/QC3.

**Neither supplementary test is itself vacuous.** I checked each leg separately:
`edge_n_defined == before+1` proves the poison is ADMITTED (the leg that would expose the
test if the poison drifted below a filter); `compute_edge_ratio(...) == 5000.0` pins its
value; and the closing mean-drift leg measures **7354.1%** against a required >100%, so
the row genuinely discriminates between estimators. Same three-leg structure verified on
the capture twin.

Main disclosed this weakness himself in §9 before any Q/A raised it, and again in §12/§13.
That is the behaviour this gate exists to reward.

### C3.7 Attack B — fixture genuineness

- **Reproduces the live book.** Current-tree `aggregate_exit_quality` run on the 32 rows
  the LIVE endpoint returns matches the live service on **9/9** fields
  (0.6304 / 0.6446 / 20 / 12 / 3.09 / 3.9636 / 6 / 1.0 / 32).
- **No hand-editing**, confirmed by the softening matrix above: every direction of
  softening I could construct kills a guard, including the −39.9 knife-edge.
- **PII/credentials: clean.** 4223 bytes, exactly five keys
  (`ticker, mfe_pct, mae_pct, realized_pnl_pct, capture_ratio_legacy`), 21 distinct
  tickers, **0** email / key-like / base64 / date / UUID matches.

### C3.8 Attack C — consumer contract on `float | None`

Enumerated by repo-wide grep over `*.py/*.ts/*.tsx/*.sql/*.json` (excluding
`.venv`, `node_modules`, build dirs), not from the disclosure.

- **BQ writes** — `capture_ratio` is `FLOAT64` with no `REQUIRED` mode
  (`scripts/migrations/add_round_trip_schema.py:59,80`) → NULLABLE; `None` is legal.
- **BQ reads** — nothing reads the stored column back; `pair_round_trips` recomputes.
- **Optimizer / go-live gate / Slack bot / meta_evolution / backtest / SQL** — zero hits.
- **Frontend** — `api.ts:529,532,547` and `types.ts:769,772` are now `number | null`
  (cycle-1 C-1 closed). **All three render paths in `MfeMaeScatter.tsx` discriminate on
  `=== null` before multiplying** (`:130` edge, `:136-138` capture, `:197` tooltip); the
  only other numeric renders (`n_points`, `n_leakers`, `leakage_threshold_capture`,
  `mfe_pct`, `mae_abs_pct`, `realized_pnl_pct`) are non-nullable in the payload.
- `npx tsc --noEmit` → exit 0.

### C3.9 Attack D — three-copy de-duplication, verified BY EXECUTION on the live service

```
/performance     -> round_trip_summary : 0.6304  0.6446  n 20/12  floor 1.0  (n_round_trips 32)
/mfe-mae-scatter -> summary            : 0.6304  0.6446  n 20/12  floor 1.0  (n_points 32)
                                          edge 3.09  edge_of_sums 3.9636  edge_n_infinite 6
/round-trips     -> top level          : 0.6304  0.6446  n 20/12  floor 1.0
                                          + 12 of 32 per-trip capture_ratio = null
```
Three-way parity. The claim is TRUE.

### C3.10 Attack E — regression, by SYMMETRIC DIFFERENCE not by counts

Full `backend/tests/`: **34 failed, 2485 passed, 12 skipped, 5 xfailed, 1 xpassed (151s)**
(cycle 1 saw 31, cycle 2 saw 32 — the set is dominated by live-service and log-scraping
tests whose inputs move with machine state). I did not average that away. I derived the 22
failing files from the run, kept the **11** that reference a changed module, and ran that
set twice — current tree vs a HEAD-module overlay that **asserts each HEAD blob differs
from the working tree (4/4) and asserts it installed**, so an overlay that overlaid
nothing raises instead of faking an identical result.

```
CURRENT tree : 20 failed, 173 passed, 1 skipped
HEAD overlay : 21 failed, 172 passed, 1 skipped

SYMMETRIC DIFFERENCE
  fails ONLY with 82.5 (CAUSED by 82.5) : <EMPTY>
  fails ONLY at HEAD  (FIXED by 82.5)   : test_paper_trader_execute_sell_capture_ratio_none_when_no_gain
  common (pre-existing)                 : 20
```

**Zero regressions attributable to 82.5.** The single asymmetry points the right way: the
renamed test fails against HEAD code (which returns `0.0`) and passes against the fix —
which is what a legitimate contract update looks like, not a red test made green. The
changed test's rewrite is correct: the old `assert capture_ratio == 0.0` **encoded the
defect**, the original intent (no NaN, no `ZeroDivisionError`) is still asserted because
the call still executes and returns, and QC10 (restore the fabricated zero) kills 4 tests
so the new assertion is load-bearing.

`test_paper_trader_execute_buy_average_up_recomputes_avg_entry` fails under BOTH trees →
genuinely pre-existing. Root cause confirmed live: `{"paused": true, "pause_reason":
"manual"}` on `/api/paper-trading/kill-switch`.

### C3.11 Live UI gate (qa.md §1c) — DEGRADED FALLBACK, disclosed as required

**I attempted the capture myself**, as §1c requires (browser tools loaded via the
deterministic `select:` form):

```
browser_navigate("http://localhost:3000/paper-trading")
  -> Page URL: http://localhost:3000/login      (NextAuth wall, no session in this profile)
curl :3100 -> 000 (down)   :3000/ -> 302   :3000/login -> 200   :3000/paper-trading -> 302
launchctl: com.pyfinagent.frontend pid 863 (real pid, not orphaned)
```

:3100 is down and standing it up is **Main's lifecycle responsibility, never the
evaluator's**. **This verdict therefore rests on a MAIN-PRODUCED capture — the
explicitly-degraded §1c fallback, and I am saying so here because §1c requires it.**

I did not accept it on trust. I read the PNG and corroborated every claim against the
live API:

| Tile | PNG renders | Live API | Agree |
|---|---|---|---|
| EDGE RATIO | **3.09**, hint `median(MFE / \|MAE\|)` | `edge_ratio: 3.09` | yes |
| AVG CAPTURE | **63%**, hint `median(realized_pnl / MFE), n=20` | `0.6304` → 63%, `capture_n_defined: 20` | yes |
| ROUND-TRIPS | 32, "closed only" | `n_points: 32` | yes |
| LEAKERS | 0, `capture < 40% & MFE > P75` | `n_leakers: 0`, `leakage_threshold_capture: 0.4` | yes |

Both hints say **median**, so the estimator change is visible to the operator rather than
silently altering a number under an unchanged label, and `n=20` reaches the UI. Sidebar
corroborates independent facts I measured separately (`KILL PAUSED`, `GATE NOT ELIGIBLE
2/5`). Proper-range emoji scan over all changed files: **0 hits** (an earlier scan of mine
flagged 100+ U+2500 box-drawing separators — those are pre-existing style, not emoji; I
corrected my own detector rather than reporting the false positives).

**Self-check on my own side effects (the cycle-2 E-1 class, applied to me):** my
`browser_navigate` wrote `.playwright-mcp/page-*.yml` and `console-*.log`. That directory
is gitignored (`.gitignore:71`) and `git status` shows nothing playwright-related. **My
evaluation added nothing to this step's commit.**

### C3.12 Findings

#### ✖ W-1 (WARN) — a contract-named change site skipped and undisclosed, for the SECOND time in this step

`contract_82.5.md` §4 enumerates the change sites and includes:

> `backend/tests/test_paper_trading_v2.py:235-243` (the existing test asserts key presence
> only and cannot fail on any value)

`git diff --name-only HEAD | grep -c test_paper_trading_v2` → **0**. The file is unchanged,
and neither `experiment_results_82.5.md` nor `live_check_82.5.md` mentions it at all. The
site is still exactly what the contract described:

```python
for key in ("edge_ratio", "avg_capture_ratio", "n_points", "n_leakers"):
    assert key in body["summary"]
```

A key-presence assertion that cannot fail on any VALUE — the contract itself identified it
as a vacuous guard (qa.md §4c shape #4), on the very endpoint this step rewrote. This is
the same `feedback_gate_scope_and_disclosure_completeness` class the cycle-1 Q/A caught on
`api.ts`; that instance was fixed, this second item on the same list was not.

**Why it is WARN and not blocking:** it guards no immutable criterion (all four are
covered by the new 17-test suite with named, executed killing mutations), it introduces
nothing false (the test passes and was equally vacuous before this step), and I verified
the endpoint's actual values live in §C3.9. Strengthening a pre-existing weak test is
also a legitimate candidate for its own queued step under
`feedback_queue_discovered_defects_in_masterplan` rather than scope creep here. What is
genuinely missing is one sentence of disclosure.

**Named fix:** either add a value-pinning assertion to `test_paper_trading_v2.py:235-243`,
or disclose in §11 that the contract-named site was deliberately deferred — and queue it.

#### ✖ W-2 (WARN) — the verbatim criterion-3 test is a guard that cannot fail

Survives all 14 code mutants (§C3.6). Not attributable to Main — the poison is fixed by
the immutable criterion text, Main disclosed the weakness unprompted, and added two
supplementary guards that DO die. Recorded so the next reader does not mistake that test
for the thing protecting criterion 3.
**Named fix:** none available inside 82.5 (criteria are immutable). If a future step
re-specifies this metric, specify the poison as an outlier *in the aggregated statistic*,
not as a raw input ratio.

#### ✖ W-3 (WARN, carried from cycles 1 and 2, still open) — INV5's guard is a literal source scan

`test_performance_and_scatter_surfaces_share_one_definition` asserts
`"realized_pnl_pct / mfe" not in inspect.getsource(mod)` — defeated by whitespace
(vacuity shape #2). The property itself is real; I proved it by executing all three live
surfaces (§C3.9). **Named fix:** replace the string scan with that parity assertion.
Worth its own queued step, as cycle 2 also recommended.

#### Notes (non-blocking)

- **N-1** — a zero-byte `threshold` file sits at the repo root and `git add -An` confirms
  it will ship under 82.5's name. Created 4 Aug 12:40, i.e. **before** this step's work
  began (19:19), so it is not 82.5's product. Flagged by cycle 1 as N-4 and still present.
  `feedback_audit_the_commit_not_the_diff`: remove it before the flip.
- **N-2** — `perf_metrics.py` mtime **20:19:43** post-dates both the backend restart
  (PID 62664 @ 19:55:37, no `--reload`) and the UI capture (19:57), so the live evidence
  was produced by a marginally earlier build. **I proved this is immaterial rather than
  assuming it:** the current tree's aggregate run on the live rows matches the live
  service on 9/9 fields (§C3.7). Worth noting only so the ordering is on the record.
- **N-3** — the §1c capture is Main-produced (§C3.11), the explicitly-degraded fallback,
  mitigated by independent corroboration against the live API.
- **N-4** — `git add -An` also sweeps `handoff/current/phase83_research_raw/*` and
  `.claude/agent-memory/researcher/project_pbo_level_and_dead_gate_82_27.md` from other
  sessions/steps, plus three `handoff/archive/phase-82.*/` snapshots. No foreign *source*
  file is swept; disclose or stage narrowly.

### C3.13 Adversarial worst-of-N-lenses (P1 money-path step)

- **correctness lens — PASS.** The Cauchy argument is sound and the implementation matches
  it. The asymmetric treatment (exclude for capture, rank `+inf` for edge) is correct and
  non-obvious; QC4 proves the `+inf` ranking is live, QC8 proves the non-finite refusal is
  live, QC7/QC10 prove the counters are load-bearing. The `mfe > 0` / `mfe < floor` split
  is deliberately two statements after a real self-caught bug (§7). Three live surfaces
  agree. Leakage now requires a DEFINED capture — the right call, and disclosed.
- **does-it-reproduce lens — PASS.** Immutable command exit 0 (17 passed); 14 code mutants
  (13 killed, 1 proven equivalent by execution) + 4 fixture mutants all behave as required;
  every quantitative claim in the handoff re-derived independently and matched, including
  Main's 3.29% to the digit and his HEAD-line-144 lint story; fixture proven to reproduce
  the live book on 9/9 fields; regression symmetric difference EMPTY.
- **scope-honesty lens — PASS with WARN.** §5 was corrected in place rather than left
  true-looking above an addendum; §12 now names the two distinct causes for criterion 3
  and is numerically exact; §13/D1 discloses the capture run's tracked-file side effect
  AND root-causes it to the skipped `globalTeardown`; §7 discloses a bug Main found in his
  own fix. The residual is W-1: one item on the contract's own change-site list was left
  undone and unmentioned.

`verdict = min(lens verdicts)` = **PASS**, with W-1/W-2/W-3 recorded as WARN-level
findings with named fixes.

## C3.14 Verdict

**PASS.** Stated plainly against the §C3.0 escalation rule: I am not issuing a third
CONDITIONAL, and I am not avoiding one by inflating a PASS. All four immutable criteria
are covered by guards I turned red myself with mutations neither Main nor the two prior
cycles ran; the immutable command exits 0; every gate (ruff, tsc, eslint, imports, live
API) is green; every cycle-1 and cycle-2 blocker is independently verified closed; and
zero regressions are attributable to the step by symmetric difference. The three residual
findings are WARN-level, none touches an immutable criterion, a money path, or a live
value, and each has a named fix that belongs in a queued step rather than a fourth cycle.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria are covered by NON-VACUOUS guards, each with a named killing mutation I executed myself, and the immutable command exits 0 (17 passed). My independent matrix ran 14 code mutants and 4 fixture mutants that neither Main nor the two prior Q/A cycles ran: 13 code mutants killed; the 14th (QC11, 'mfe_sum_defined > 0' -> '!= 0') I proved by execution to be an EQUIVALENT MUTANT rather than a coverage gap (200000 randomised trials returned 0 admitted rows with mfe <= 0, since compute_capture_ratio requires mfe > 0 unconditionally), so I did not report it as a finding. All 4 NEW fixture-softening mutants kill criterion 4's guard, including the knife-edge QG1 that retunes the blowup row to a legacy mean of exactly -39.9 -- so criterion 4 discriminates at the exact threshold it names. On the caller's criterion-3 question, measured rather than reasoned: the VERBATIM criterion-3 test survives ALL 14 code mutants -- it cannot be made to fail by reverting either estimator, removing or halving the floor, inverting the ratio, breaking the counters, or allowing a non-finite median. The cause is NOT only the floor: the poison is excluded by the 1.0pp floor on the CAPTURE leg (n_defined 20->20) but is ADMITTED on the EDGE leg (20->21) and is simply not extreme there (mfe/|mae| = 1e-4/1e-4 = exactly 1.0 against an ordinary base range 2.500..3.718, giving 3.29% mean drift -- reproducing Main's corrected figure to the digit). That is a property of the IMMUTABLE criterion's own poison specification, which Main cannot change, and he disclosed it unprompted before any Q/A raised it. Criterion 3's intent IS genuinely covered: the capture twin dies under QC1/QC14 at line 200 'assert drift < 0.20' (capture median moved 4766.7%) with a poison ABOVE the floor, and the cycle-3 addition test_the_edge_headline_is_robust_to_an_extreme_outlier dies under QC2/QC12/QC3 with a genuine edge outlier of 5000; I verified each test's anti-vacuity legs individually (admission leg, value pin, and a mean-drift leg measuring 7354.1% against a required >100%), so neither supplementary test is itself vacuous. The fixture is genuine: the current tree's aggregate on the LIVE endpoint's 32 rows matches the live service on 9 of 9 fields, and it carries no credentials or PII (5 keys, 21 tickers, zero email/key/base64/date/UUID matches). The three-copy de-duplication is proven BY EXECUTION -- /performance, /mfe-mae-scatter and /round-trips all return 0.6304 / 0.6446 / n 20-12 / floor 1.0 on the same book. ZERO regressions by SYMMETRIC DIFFERENCE against a HEAD-module overlay that asserts all 4 blobs differ and asserts it installed: nothing fails with 82.5 that passes at HEAD; the only asymmetry is the renamed capture test failing at HEAD, which is the correct direction for a contract update, and QC10 proves its new assertion is load-bearing. All four cycle-2 items are independently verified CLOSED (tracked files restored and the side effect now disclosed with its globalTeardown root cause; the causal claim corrected and numerically exact; the edge robustness guard added and killable; ruff 5 errors -> All checks passed on a 6-file derived scope), as are all three cycle-1 items. Three WARN-level findings remain, none blocking: W-1, contract section 4 named backend/tests/test_paper_trading_v2.py:235-243 as a change site and it was neither changed nor mentioned -- a key-presence assertion that cannot fail on any value, on the endpoint this step rewrote, and the second instance of the disclosure-completeness class in this step; W-2, the verbatim criterion-3 test is a guard that cannot fail, recorded so no future reader mistakes it for the thing protecting criterion 3; W-3, INV5's guard is still a literal source scan defeated by whitespace, though I proved the property itself by executing all three surfaces. None of the three touches an immutable criterion, a money path, or a live value, and each has a named fix better queued as its own step than spun into a fourth cycle. Live UI gate: I attempted the capture myself and :3000 redirected to /login while :3100 was down (Main's lifecycle), so this verdict rests on a MAIN-PRODUCED capture -- the explicitly-degraded qa.md 1c fallback, disclosed here as 1c requires and mitigated by corroborating all four tiles against the live API.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item_audit",
    "third_conditional_rule_acknowledged_and_reasoned_explicitly",
    "evidence_changed_between_spawns_verified_by_mtime_not_verdict_shopping",
    "immutable_verification_command_exit_0_17_passed",
    "python_lint_gate_ruff_scope_derived_from_git_diff_PLUS_untracked_nonempty_asserted",
    "frontend_tsc_noemit_exit_0",
    "frontend_eslint_scoped_to_changed_files_exit_0",
    "backend_import_smoke_4_modules",
    "live_api_health_200",
    "independent_mutation_matrix_14_code_mutants",
    "equivalent_mutant_proven_by_execution_not_reported_as_false_finding",
    "mutation_targets_asserted_present_exactly_once_before_replace",
    "control_asserted_at_exactly_17_before_any_mutant",
    "killing_assertion_identified_by_line_number",
    "criterion_3_measured_separately_for_BOTH_tiles",
    "verbatim_criterion_3_test_proven_unkillable_by_all_14_mutants",
    "anti_vacuity_legs_of_both_robustness_tests_checked_individually",
    "4_new_fixture_softening_mutants_incl_knife_edge_at_39.9",
    "fixture_reproduces_live_book_on_9_of_9_fields",
    "fixture_pii_credential_scan",
    "consumer_grep_py_ts_tsx_sql_json_repo_wide",
    "bq_schema_nullability_confirmed",
    "frontend_null_render_paths_audited_all_three",
    "three_live_surfaces_curled_and_compared_for_parity",
    "full_backend_regression_2534_tests",
    "HEAD_module_overlay_symmetric_difference_EMPTY",
    "overlay_asserts_blobs_differ_and_asserts_it_installed",
    "pre_existing_failure_confirmed_failing_under_BOTH_trees",
    "kill_switch_state_confirmed_as_root_cause",
    "changed_test_rewrite_audited_for_red_made_green",
    "cycle2_D1_tracked_file_restoration_verified",
    "cycle2_D4_lint_story_reproduced_against_HEAD_blob",
    "playwright_live_ui_capture_ATTEMPTED_blocked_degraded_fallback_disclosed",
    "main_produced_capture_png_read_and_cross_checked_against_live_api",
    "self_check_my_own_browser_side_effects_are_gitignored",
    "emoji_scan_corrected_after_my_own_false_positives",
    "git_add_dry_run_commit_scope_audit",
    "research_gate_envelope_verified",
    "harness_log_last_and_masterplan_status_check"
  ]
}
```

**Next move for Main:** append the `harness_log.md` cycle entry (LOG-LAST), then flip
`.claude/masterplan.json` 82.5 → `done`. Before the flip, remove the stray zero-byte
`threshold` file from the commit scope (N-1). Queue W-1 and W-3 as their own
research-gated masterplan steps per `feedback_queue_discovered_defects_in_masterplan`:
a value-pinning test for `/mfe-mae-scatter` (replacing the key-presence-only assertion at
`test_paper_trading_v2.py:235-243`), and replacement of INV5's source scan with the
executed cross-surface parity assertion.


