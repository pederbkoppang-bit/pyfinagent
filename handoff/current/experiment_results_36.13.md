# Experiment results — masterplan step 36.13

**[P0] `execute_buy` had no kill-switch gate, so the MCP signals path placed orders while the switch
was PAUSED or its baselines were lost.** CWE-424 alternate path, now closed at the choke point.

## What shipped

| File | Change |
|---|---|
| `backend/services/paper_trader.py` | NEW `_kill_switch_refusal_for_buy()` + the gate at the top of `execute_buy`; `__init__` gains `kill_switch_state=None` (the drill/test seam) which WARN-logs when non-default |
| `backend/services/kill_switch.py` | NEW module-level `baselines_present_in(snap)`; `evaluate_breach` now computes its `baselines_present` key through it, so there is ONE definition |
| `scripts/go_live_drills/zero_orders_drill.py` | injects `_DrillKillSwitchState()` (healthy, with baselines) |
| `scripts/smoketest_stages_5_through_13.py` | same, at both construction sites |
| `backend/tests/test_phase_36_13_kill_switch_execute_buy_gate.py` | NEW — 13 tests |

**Behaviours changed, stated as behaviours:**
1. A BUY is refused when the book is PAUSED, and when both loss baselines are unrestorable.
2. The refusal FAILS CLOSED — any exception reading kill-switch state refuses the order.
3. The refusal is observable: it joins `buy_rejections` (which the cycle summary already consumes at `autonomous_loop.py:236-1582`) with a reason distinguishing `kill_switch_paused` / `kill_switch_baselines_lost` / `kill_switch_unreadable`, plus an ERROR log.
4. **Staleness does NOT refuse a BUY.** The gate reads `baselines_present`, never `armed`.
5. `execute_sell` is deliberately ungated — the switch flattens *through* it.
6. Drills/smoketests supply their own state instead of bypassing a control; injection WARN-logs.

## The research gate changed the design — that is why it runs

I intended **(c)**: rename to `_execute_buy_unchecked` + a guarded wrapper. The gate produced **(a′)**
and was right on two counts I had wrong:
- **My escape-hatch premise was false.** Both "deliberate" callers already inject a stubbed BigQuery
  client. They need to *supply state*, not to bypass a safety control.
- **The rename is 19 call sites, not 4** (15 in tests). CWE-638 — the *parent* of the CWE-424 defect —
  names the remedy directly: *"create and use a single interface that performs the access checks."*

**The trap it found, which I converted from a reading to a measurement:** `kill_switch._state` is a
module singleton whose boot replay sets `_paused = True` from any `pause` row. Executed: a fresh
process with one pause row reports `is_paused() == True`. A naive `get_state().is_paused()` gate would
therefore have broken both scripts **only on days the book happened to be paused** — the worst possible
failure schedule, invisible to a green suite.

## Criteria 1 & 2 — the bypass reproduced BEFORE the fix

`handoff/current/captures_36.13/prefix_reproduction.txt`, from reverting the gate in memory:

```
PRE-FIX REPRODUCTION -- execute_buy with NO kill-switch gate
>       assert result is None, (
E       AssertionError: a paused kill switch must refuse the order -- pre-fix this returned a trade
E       assert {'action': 'BUY', 'analysis_id': '', 'created_at': '...', 'price': 100.0, ...} is None
>       assert result is None
E       AssertionError: assert {'action': 'BUY', ...} is None
2 failed, 8 deselected
```

A **paused** book returned a complete BUY trade dict. Same for lost baselines.

## Criterion 4 — call-site inventory, RE-DERIVED by command

`handoff/current/captures_36.13/call_site_inventory.txt` (verbatim commands + output). Note
`--include='*.py'` must be quoted under zsh or the glob is eaten.

| Non-test caller | Disposition |
|---|---|
| `signals_server.py:444` | **GATED** — the bypass this step closes |
| `autonomous_loop.py:236` | **GATED** — already behind the cycle check; now belt-and-braces |
| `zero_orders_drill.py:127` | **DELIBERATELY BYPASSING** via injected state, WARN-logged + AST-pinned |
| `smoketest_stages_5_through_13.py:225` | **DELIBERATELY BYPASSING**, same mechanism |

`grep -rn 'kill_switch_state=' --include='*.py' backend | grep -v /tests/` → **none**: production
code cannot inject.

Line-number corrections to the step's own text, measured: `autonomous_loop.py:236` (step said `:207`);
`paper_trader.py:1304` / `autonomous_loop.py:236` (step said `:1097` / `:1287`).

## Criterion 5 — the drills still work, established against a HEAD BASELINE

**I broke both scripts and caught it, which is the reportable part.**

- After wiring the gate through `baselines_present_in`, the drill failed:
  `FAIL: execute_buy returned None`, and smoketest **Stage 8 flipped PASS → FAIL**.
- Cause: my injected stub's `snapshot()` returned only the pause fields, so the gate read it as
  **lost history**. The stub was wired in correctly and still didn't work.
- I did not guess whether Stage 12 was mine. I built a `git worktree` at HEAD (never `git stash` —
  active hooks) and ran the smoketest there: **HEAD is 8/9 with Stage 12 already failing and Stage 8
  passing.** So Stage 12 is pre-existing and Stage 8 was mine.
- Fixed the stubs to carry positive baselines. Re-measured: drill **PASS**; smoketest **8/9, Stage 8
  PASS** — byte-identical to the HEAD baseline.

Guarded two ways, because "wired in" and "works" are different claims:
`test_..._the_drills_inject_their_own_kill_switch_state` (AST parse, not grep — a string match would
pass on a comment) and `test_..._drill_stub_state_actually_passes_the_gate` (runs the REAL predicate
against the stub imported from the script).

**My earlier claim "the drill PASSES" was true when I ran it and false by the time I finished the
step.** I re-ran it only because criterion 5 forced an explicit assertion. The lesson is the standing
one: run the proving check in the same turn as the claim, and re-run it after any later change to the
thing it proves.

## Criterion 8 — mutation matrix, one batch at the FINAL baseline

`handoff/current/captures_36.13/mutation_matrix.txt` — **9 mutants, 9 killed, 0 survived**, baseline
`203 passed, 1 skipped`.

**The first run reported 2 survivors and both were findings about my own work:**
- `M_drill_stops_injecting_state` **survived** — nothing pinned the drill's injection, which
  criterion 8 explicitly requires ("removing the bypass must fail the drill test"). Added the AST guard.
- The second survived only because my matrix ran a single test file. A matrix that cannot see the
  guards licenses nothing, so it now runs at immutable scope.

**CORRECTED in cycle 2, after a Q/A caught two claim-level defects in this very section:**
- My reporter picked the first stdout line containing `"failed"`, which matched a LOG line
  ("baseline-history probe failed: audit tree unreadable") rather than pytest's summary — so one row
  printed a warning where a count belonged. A row credited as KILLED on evidence that does not show a
  kill is exactly the "mis-attributed kill mechanism" shape. Reporter now matches
  `^\d+ (passed|failed)`; every row carries a real summary.
- More substantively: reverting `baselines_present = baselines_present_in(s)` to HEAD's inline
  `not (daily_baseline_missing or trailing_baseline_missing)` is an **EQUIVALENT MUTANT**.
  `kill_switch.py:768-769` define those two flags as the exact negations, so the inline form and the
  helper are the identical boolean and NO test can distinguish them. The refactor is unkillable by
  construction; my earlier "killed by 14 tests" described a *different* mutation
  (`baselines_present = True`) and should never have been attached to the refactor. Replaced with the
  mutation that pins the predicate's LOGIC — `baselines_present_in -> return True` — which fails
  **15 tests**, independently reproduced by the evaluator.

Final matrix: **9 killed, 0 survived** at baseline `203 passed`, every DETAIL a real pytest summary.
  A matrix that cannot see the guards licenses nothing, so the matrix now runs at immutable scope.

## Criterion 7 — no threshold touched

`git diff` contains no change to any limit value: `paper_daily_loss_limit_pct`,
`paper_trailing_dd_limit_pct`, DSR, PBO and sector caps are byte-untouched. This step adds a gate; it
never changes when the switch trips. The pause/resume API surface is unchanged.

## The half-wired seam I found and closed

The gate initially honoured the injected state for `is_paused()` but called `evaluate_breach()`, which
reads the module **singleton** — so injection governed the pause check and not the baseline check, and
a drill could still be refused by the real book's lost baselines. Fixed by adding
`baselines_present_in(snap)` and routing **both** `evaluate_breach` and the gate through it. Deliberately
not re-implemented at the call site: a hand-copied predicate that drifts is the exact defect class
phase-36.9 hit.

## Verification — measured at the final state

```
$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or signals_server'   # IMMUTABLE
203 passed, 1 skipped, 2103 deselected

$ python -m pytest backend/tests/test_phase_36_13_kill_switch_execute_buy_gate.py -q
13 passed

$ python scripts/go_live_drills/zero_orders_drill.py
step2: paper_trades row written: ticker=AAPL action=BUY qty=5.128205 price=195.0
PASS

$ python scripts/smoketest_stages_5_through_13.py
Stage 8 -- phase-25.6 HARD BLOCK synthesizes 8% stop: PASS
PASS: 8/9   (Stage 12 fails at HEAD too -- pre-existing, verified in a worktree)

$ FILES=$(git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py')
$ echo "$FILES" | xargs ruff check --select F821,F401,F811,E9
All checks passed!
# scope = the 5 changed/untracked .py files:
#   backend/services/kill_switch.py, backend/services/paper_trader.py,
#   scripts/go_live_drills/zero_orders_drill.py,
#   scripts/smoketest_stages_5_through_13.py,
#   backend/tests/test_phase_36_13_kill_switch_execute_buy_gate.py
```

## Out of scope → FILED as their own steps

**FILED, not merely disclosed.** A cycle-2 Q/A correctly pointed out that these carried a
"(to be filed)" label and no masterplan step existed — a future-tense promise is not a disposition,
and the operator's standing rule is that an out-of-scope defect gets its OWN research-gated step
written for an executor with no memory of the discovery.

- **36.22** — `signals_server.py` carries a **weaker duplicate** control: its own in-memory peak
  (`:88-89`, resets every restart) feeding `drawdown_circuit_breaker` at `:950`. It never observes
  PAUSED and has no `armed`/`baselines_present` equivalent. That duplicate is why this P0 gap looked
  covered. The step requires all three weaknesses REPRODUCED before any change.
- **36.23** — `scripts/go_live_drills/kill_switch_test.py` **does not test the kill switch**; its
  scenarios drive that duplicate. It is a GO-LIVE DRILL, so its name is the assurance an operator
  reads before real money moves, and that assurance is currently false. The step requires the
  executor to re-derive by grep rather than trust the description.

## Do-no-harm

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after every run,
including the full matrix and both script executions. `:8000` GET-only, never restarted or POSTed to.
`:3000` never driven. No threshold, stop, sector cap, DSR or PBO value in the diff. No peak reset.
The drills ran against their own stubbed BigQuery — the live book was never touched.

**NOT LIVE:** like 36.12, 36.8 and 36.9, this needs a backend restart the operator has not authorized.


## Cycle 2 -- two claim-level defects, both mine

Neither touched shipped behaviour; the evaluator independently mutation-proved the money path,
including two mutants I had not written (`injection_ignored_gate_reads_singleton`,
`refusal_computed_but_not_enforced`) and the fixture-side mutant `qa.md` 4c requires.

**The lint gate was green because I had narrowed its scope.** Over the true
`git diff --name-only HEAD` scope, ruff exits 1: F401 `portfolio_manager.TradeOrder` imported but
unused in `scripts/smoketest_stages_5_through_13.py` -- a file this step modifies. I verified it is
PRE-EXISTING (identical import at `HEAD:32`) and genuinely unused, then removed it: ruff's own
autofix, zero behaviour change, in a file already being touched. Both scripts re-run green
afterwards (drill PASS; smoketest 8/9 with Stage 8 PASS). The artifact now prints the real command
and its real 5-file argument list rather than the placeholder `<git-derived scope>`, which was
unreproducible by construction.

**Why this matters more than an unused import:** the defect was not the F401, it was that I wrote a
gate result I had not run over the scope I claimed. A gate is only green on the scope the CHANGE
defines, and that scope must be derived, printed, and reproducible.
