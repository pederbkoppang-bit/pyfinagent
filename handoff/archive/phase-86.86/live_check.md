# live_check -- phase-86.86 (D6)

**Step:** 86.86 (P1, LIVE MONEY) -- the lite risk-judge paths destroyed an
explicit 0% position verdict at the construction seam.
**Date:** 2026-08-15. **Cycle:** 1.
**Tree state at capture:** `main`, HEAD `cd148eea` + this step's working-tree
changes. `backend/services/autonomous_loop.py` sha256 `5b714a9e5f43...`.

> **NOT YET IN FORCE.** Per the standing operator instruction, backend restarts
> are batched to SESSION END. The running backend imported the pre-fix module,
> so this fix is **committed but not active** until the end-of-session restart.
> Every measurement below is from a fresh process, not from the running one.

---

## 0. Section index vs criteria

| criterion | section |
|---|---|
| 1 reproduction, both paths, zero vs silent indistinguishable | §1 |
| 2 seam fix, exactly one route to the default | §4, §5A |
| 3 class enumerated from source, rule written, per-member reasons | §2, §5A |
| 4 positive control 0.0 -> 0.0 while absent -> 3.0 | §5B |
| 5 both states of both flags (with the honesty note) | §5D, §6 |
| 6 every other behaviour change disclosed | §5C, §6 |
| 7 mutation-tested, control green first, byte-identical restore | §5E |
| 8 downstream consequence driven, not read | §5D |
| 9 no gate loosened, no flag promoted, no .env written | §8 |

---

## 1. PRE-FIX reproduction (criterion 1) -- captured BEFORE any edit

Command: `PYTHONPATH=. python scratchpad/repro_d6_prefix.py` against the shipped
tree, importing the real `_LITE_RISK_DEFAULT` and driving the real
`decide_trades`.

**(a) The seam alone**

```
judge emits 0.0  (explicit no-buy) -> persisted pct = 3.0
judge emits 3.0                    -> persisted pct = 3.0
judge silent (key absent)          -> persisted pct = 3.0
```

Rows 1 and 3 are **identical**. This is the criterion-1 requirement stated
precisely: it is not merely that `0.0` is wrong, it is that **an explicit zero
and a silent judge are indistinguishable after the line**, so no downstream
guard can recover the verdict.

**(b) Downstream, real `decide_trades`, NAV 23,997.71, all four flag combos**

```
shape_fix=False binding=False | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
shape_fix=False binding=True  | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
shape_fix=True  binding=False | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
shape_fix=True  binding=True  | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
```

**(c) Control -- the same inputs when the seam is honest**

```
shape_fix=False binding=False | true 0.0 -> no order
shape_fix=False binding=True  | true 0.0 -> no order
shape_fix=True  binding=False | true 0.0 -> no order
shape_fix=True  binding=True  | true 0.0 -> no order
```

$719.93 independently reproduces the figure the 86.74 cycle-7 Q/A reported.

**Both lite paths.** The Claude (`:3091-3094`) and Gemini (`:3337-3340`) blocks
were **byte-identical** pre-fix -- confirmed by the AST enumeration in §2, which
found the same five keys twice, once per path. The reproduction therefore
applies to both by construction, and §5A shows both now route through one
producer.

**A correction to the D6 brief's harm bound, stated because it makes the defect
WORSE, not better.** The brief bounded exposure with
`paper_risk_judge_reject_binding=True`. The reproduction above uses
`decision="APPROVE_REDUCED"` -- a non-REJECT decision paired with a 0.0 pct,
which is exposure case (a) in the brief -- and the BUY fires **with binding ON**.
So the live-harm bound is not "an .env line"; with a non-REJECT decision that
.env line is irrelevant.

**Two further measured absurdities in the same expression**

```
'0'  (string zero)   -> 0.0     <-- the STRING zero SURVIVED
0.0  (float zero)    -> 3.0     <-- the FLOAT zero DIED
'high' (garbage str) -> RAISES ValueError
''   (empty string)  -> 3.0
False                -> 3.0
```

The falsy test precedes `float()`, so the *serialisation* of the same number
decided whether it survived.

---

## 2. The CLASS, enumerated FROM SOURCE (criterion 3)

**Enumeration rule, written down as the criterion requires.** A member of the
`or _LITE_RISK_DEFAULT[...]` family is **DECISION-INVERTING** iff its falsy
trigger value is (i) in the judge's legitimate emitted domain, (ii) semantically
*distinct from absence*, and (iii) substituting the default changes what
`decide_trades` returns. **Clause (iii) is settled by driving `decide_trades`,
never by reading it.**

**Pre-fix enumeration (AST walk, `ast.BoolOp`/`ast.Or` with a
`_LITE_RISK_DEFAULT` subscript operand): 10 sites, 5 keys x 2 lite paths.**

```
line  3084  key='reasoning'                 line  3332  key='reasoning'
line  3086  key='decision'                  line  3334  key='decision'
line  3093  key='recommended_position_pct'  line  3339  key='recommended_position_pct'
line  3095  key='risk_level'                line  3341  key='risk_level'
line  3096  key='risk_limits'               line  3342  key='risk_limits'

distinct keys: ['decision', 'reasoning', 'recommended_position_pct', 'risk_level', 'risk_limits']
positive control -- known members the scan FAILED to find: none
```

**Why an AST walk and not a grep:** the docstrings in this repo quote the very
idiom being hunted, so a text scan matches its own documentation. An AST walk
sees expressions, not prose. §5A shows the negative control that proves this.

**Per-member classification, each measured:**

| key | falsy trigger | measured effect on `decide_trades` | class |
|---|---|---|---|
| `recommended_position_pct` | `0.0` | `no order` -> `BUY $719.93` | **DECISION-INVERTING** |
| `decision` | `""` | `""` -> BUY $719.93; `APPROVE_REDUCED` -> BUY $719.93 -- **identical**, only exact `REJECT` blocks (driven with binding ON and OFF) | audit-fabricating; **latent** fail-open |
| `risk_level` | `""` | `""` / `MODERATE` / `EXTREME` all -> BUY $719.93; appears **0 times** in `portfolio_manager.py` | audit-fabricating |
| `reasoning` | `""` | read by no decision; persisted as the `reason` summary column | audit-fabricating |
| `risk_limits` | `{}` | stop 90.0 with the substitution vs 92.0 without (100.0 entry) -- a stop is **installed** where none existed | protective, leave alone |

**This CORRECTS the research brief on two counts, by measurement.** The brief
classified `decision` as "HARMFUL (fail-open)" and `risk_level` as
"HARMFUL-MILD". Driven through the real `decide_trades`, **neither changes any
order today.** The brief's underlying observation is kept rather than dismissed:
the `decision` collapse *would* invert the moment the gate changes from "block
exact REJECT" to an allow-list, so it is recorded as a **latent** fail-open.

**Scope call, stated so the Q/A judges it rather than discovers it.** This step
fixed the decision-inverting member only. The three audit-fabricating members
are real -- the substituted `reasoning` literally reads *"risk-judge parse
failed; falling back to conservative default sizing"* when the parse SUCCEEDED
and only that field was blank, writing a false statement into a persisted audit
column -- and are filed as **masterplan step 86.87** with these measurements
attached. `risk_limits` is deliberately untouched because its substitution is
protective.

---

## 3. Files changed

| file | change |
|---|---|
| `backend/services/autonomous_loop.py` | `+_lite_position_pct`, `+_build_lite_risk_assessment`, both lite paths routed through the latter, import extended |
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | +21 tests (41 -> 62) driving the REAL producer and the REAL `decide_trades` |
| `scripts/qa/verify_lite_risk_seam_86_86.py` | NEW -- re-runnable AST class enumerator + seam guard |
| `scripts/qa/mutation_matrix_86_86.py` | NEW -- 6 producer mutation cells |
| `.claude/masterplan.json` | +86.86 (this step), +86.87 (queued sweep finding) |
| `handoff/current/contract_86.86.md`, `research_brief_86.86.md`, this file | handoff artifacts |

`scripts/qa/mutation_matrix_86_74.py` was **not** modified -- it is a different
step's artifact and its cells target a different subject (the consumer).

---

## 4. The fix

Both lite paths carried **byte-identical** copies of the `risk_assessment` dict
literal. They now call one module-level producer:

```python
"recommended_position_pct": _lite_position_pct(risk_dict, ticker),
```

and `_lite_position_pct` routes through `_resolve_position_pct` -- the **same**
three-state resolver the full path uses, not a second idiom:

* `SIZE` -> the judge's number, **0.0 included**
* `UNPARSEABLE` -> `0.0`, fail **closed and loud**
* `ABSENT` -> the 3.0 default, and **only** ABSENT reaches it

Extraction was not cosmetic: a dict literal buried in a 300-line async LLM
function **cannot be driven by a test**, and a mutation cell aimed at a site no
test executes is UNSCORABLE. The research gate flagged exactly this.

---

## 5. Verbatim command output

### 5A. AST class enumeration + seam checker (criteria 2, 3)

`python scripts/qa/verify_lite_risk_seam_86_86.py`

```
phase-86.86 -- lite risk-judge INGRESS seam checker
======================================================================
  PASS  control(+): scanner FOUND the idiom it hunts (['recommended_position_pct', 'risk_level'])
  PASS  control(-): prose/comments quoting the idiom did NOT register (AST, not grep)

  Enumerated `or _LITE_RISK_DEFAULT[...]` sites in autonomous_loop.py: 4
    line  2392  key='reasoning'
    line  2394  key='decision'
    line  2402  key='risk_level'
    line  2403  key='risk_limits'

  PASS  'recommended_position_pct' appears in ZERO `or _LITE_RISK_DEFAULT[...]` nodes (the decision-inverting member is gone from the class)
  PASS  remaining members are exactly the retained set ['decision', 'reasoning', 'risk_level', 'risk_limits']
  PASS  exactly ONE function can reach _LITE_RISK_DEFAULT['recommended_position_pct']: _lite_position_pct (at line(s) [2359, 2362])
  PASS  _build_lite_risk_assessment defined exactly once (line 2365)
  PASS  BOTH lite paths route through the one producer (call sites [3186, 3422])
  PASS  _lite_position_pct: 1 definition (line 2313), 1 call site (line 2401) -- no second parallel idiom
======================================================================
checks emitted: 8  (PASS 8 / FAIL 0)

RESULT: OK
exit=0
```

**Counted, not asserted: 8 checks emitted, 8 PASS, 0 FAIL.** The site count
drops 10 -> 4 because the two byte-identical copies became one producer; the
distinct-key count drops 5 -> 4 because `recommended_position_pct` left the
class entirely.

**The control is not built from the subject.** `control(+)` runs the scanner
against a synthetic module that *contains* the idiom and requires it to be
found; `control(-)` runs it against one that only *mentions* the idiom in a
comment and a docstring and requires it **not** to match. A scanner that cannot
find a member it is handed is reported as a FAILED gate, not a clean result.

> **CORRECTION (post-verdict, from Q/A finding N2 -- this REPLACES the reading
> the line above would otherwise invite).** The checker's message *"exactly ONE
> function can reach `_LITE_RISK_DEFAULT['recommended_position_pct']`"* is true
> **only under a subscript-read reading**, and I did not qualify it. The Q/A's
> own AST walk over every `ast.Name` reference to `_LITE_RISK_DEFAULT` finds
> **12 references, of which FOUR are whole-dict copies** --
> `dict(_LITE_RISK_DEFAULT)` at `autonomous_loop.py:3177, 3182` (Claude lite)
> and `:3411, 3416` (Gemini lite), in the no-JSON and exception handlers. Those
> carry the 3.0 into `risk_dict`, which then reaches the producer as
> **`SIZE 3.0`, not `ABSENT`**. They cannot destroy a zero (they are reachable
> only when the judge produced nothing at all) and they are byte-identical
> pre/post, which is why criterion 2 stands -- but the accurate statement is
> **"exactly one function performs a subscript READ of the pct default; four
> further sites copy the whole dict"**, not "exactly one place can reach it".
> Two consequences, both queued as step **86.88**: the checker's
> `<whole-dict>` branch (`verify_lite_risk_seam_86_86.py:65-66`) is **dead** --
> a `Call` is not a `BoolOp`, so it can never fire, i.e. a zero-assertion guard
> -- and a judge FAILURE now persists as `SIZE 3.0` rather than `ABSENT`, which
> is the same collapse shape one seam over.

### 5B. Positive control (criterion 4) -- driving the REAL producer

```
=== POSITIVE CONTROL (criterion 4) -- real producer ===
  judge emits 0.0 -> persisted pct = 0.0
  judge emits 3.0 -> persisted pct = 3.0
  judge silent    -> persisted pct = 3.0
  0.0 stays 0.0; absent stays 3.0; they are DISTINGUISHABLE.
```

`0.0 -> 0.0` **while** `absent -> 3.0`. Mapping both to 0.0 (or both to 3.0)
would swap one collapse for another; the suite asserts the *inequality*
directly (`test_zero_and_absent_are_DISTINGUISHABLE`) so a fix cannot pass by
satisfying the two halves independently, and mutation cell **D6-M2** injects
exactly that mirror collapse and is KILLED.

### 5C. Full disclosure table (criterion 6) -- REAL producer

```
  0.0              -> 0.0
  0 (int)          -> 0.0
  absent           -> 3.0
  null explicit    -> 3.0
  3.0              -> 3.0
  '0' string       -> 0.0
  'high' garbage   -> 0.0
  '' empty         -> 0.0
  False            -> 0.0
```

This matches the prediction written into `contract_86.86.md` §8 **before** the
change, row for row. Behaviour changes other than the falsy-zero repair:

| input | before | after | disclosure |
|---|---|---|---|
| `'high'` | **raised `ValueError`** | `0.0` + WARNING | **a raise became a value.** Called out explicitly because criterion 6 names undisclosed raise->value as a scope breach. It fails closed *and* loud; the WARNING is asserted by a test and defended by cell D6-M4 |
| `''` | `3.0` | `0.0` + WARNING | `float('')` raises -> UNPARSEABLE. A verdict that cannot be read is not evidence of safety |
| `False` | `3.0` | `0.0` | `float(False) == 0.0` -> SIZE 0.0 |
| `0` (int) | `3.0` | `0.0` | the same collapse as `0.0` |

`absent`, `null`, `3.0` and `'0'` are **unchanged**.

### 5D. Criterion 8 -- driven through the REAL `decide_trades`

```
  NAV = 23997.71
  shape_fix=False binding=False | 0.0=no order | 3.0=BUY $719.93 | absent=BUY $719.93
  shape_fix=False binding=True  | 0.0=no order | 3.0=BUY $719.93 | absent=BUY $719.93
  shape_fix=True  binding=False | 0.0=no order | 3.0=BUY $719.93 | absent=BUY $719.93
  shape_fix=True  binding=True  | 0.0=no order | 3.0=BUY $719.93 | absent=BUY $719.93

  PRE-FIX the same three columns were: BUY $719.93 | BUY $719.93 | BUY $719.93
```

The `3.0` and `absent` columns are **unchanged** -- proof the fix did not simply
suppress buying. `test_absent_verdict_still_buys_at_the_lite_default` is the
anti-vacuity guard for exactly that.

### 5E. Mutation matrix (criterion 7)

`python scripts/qa/mutation_matrix_86_86.py` -> **exit 0**

```
subject : backend/services/autonomous_loop.py  sha=5b714a9e5f43

=== CONTROL (must be GREEN before any cell is scorable) ===
  suite control       : GREEN
  seam-checker control: GREEN

  D6-M1: KILLED     (4 tests selected)  restore the falsy-zero `or _LITE_RISK_DEFAULT[...]` in the producer
  D6-M2: KILLED     (4 tests selected)  the mirror collapse: map ABSENT to 0.0 instead of the default
  D6-M3: KILLED     (5 tests selected)  let an UNPARSEABLE verdict reach the default instead of failing closed
  D6-M4: KILLED     (1 tests selected)  silence the UNPARSEABLE warning (fail closed but no longer loud)
  D6-M5: KILLED     (8 tests selected)  reintroduce the falsy-zero downstream of the producer, end-to-end
  SEAM-M1: KILLED     point the Gemini lite path off the shared producer

restore verified byte-identical for all 1 subject(s): True

cells scored: 6

RESULT: all 6 cells KILLED. This licenses exactly 'these 6 mutations were killed' -- no global claim.
```

**D6-M1 is the criterion-7 cell**: it restores
`or _LITE_RISK_DEFAULT["recommended_position_pct"]` verbatim. Both control legs
were observed **GREEN first**, every cell proved its `-k` selector selects a
non-zero number of tests (pytest exits 5 on an empty selection, which
`rc != 0` would otherwise score as a KILL), and the restore is verified by
sha256.

**"EACH fixed site", addressed directly.** There were two sites pre-fix; there
is one producer post-fix. **SEAM-M1** covers the second site by pointing the
Gemini lite path at its own copy of the old literal -- i.e. re-creating the
duplication -- and requires the AST checker to catch it. Without that cell,
"both paths route through one seam" would be an unproven claim.

### 5F. Immutable command

```
62 passed, 1 warning in 1.97s
```

**41 before this step, 62 after -- 21 new tests, counted from the two runs, not
estimated.**

---

## 6. The flag-state honesty note (criterion 5)

Criterion 5 asks for both states of **both** flags, and the tests are
parametrised over both. **What that does and does not prove:**

* `paper_risk_judge_reject_binding` has real production readers
  (`portfolio_manager.py:385`, and `autonomous_loop.py:1146, 2485, 2499`
  **post-fix** -- these were `:1139, 2384, 2398` when the contract was written
  and this step added +101 lines above them; corrected per Q/A finding N4, which
  confirmed the citations were accurate at the time of writing). Its two
  states carry information, and §5D exercises both.
* `paper_risk_judge_shape_fix_enabled` has **ZERO production readers.**
  Repo-wide grep returns only the `settings.py:350` definition, the
  `settings_api.py:283` env mapping, a docstring mention at
  `portfolio_manager.py:1116`, and test files. **Parametrising over it proves
  the code is INSENSITIVE to it; it does not exercise a gated branch, because
  there is none.** Said explicitly so a green matrix is not over-read as
  "both branches covered". The research gate's wording: *"do not hang the fix
  on it"* -- and this fix does not; it is unconditional.

Effective values read from `get_settings()` on 2026-08-15:
`paper_risk_judge_shape_fix_enabled = False`,
`paper_risk_judge_reject_binding = True`.

---

## 7. Regression check -- measured, not assumed

Full backend suite after the change: **21 failed, 3493 passed** (8m39s).

Causality established by reverting `autonomous_loop.py` to HEAD and re-running
**the identical 21 node ids**, with a sha256-verified byte-identical restore:

```
failures at HEAD:                 20
failures on re-run WITH the fix:  20
set difference (comm):            EMPTY -- identical failure sets
RESTORE VERIFIED BYTE-IDENTICAL
```

The first full-suite run showed one extra failure,
`test_phase_86_6_subprocess_channel.py::test_the_optin_IS_honoured_so_a_real_window_remains_possible`.
It **passes in isolation** and **did not reproduce** on the re-run. That test
shells out against `http://localhost:8000` -- the live backend -- so it is
environment/timing dependent. It is reported here rather than omitted; it is not
attributed to this change, and the evidence for that is the identical failure
set, not an assertion.

**Conclusion: zero regressions attributable to this step.** The 20 pre-existing
failures are already tracked -- `test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap`
as masterplan step **86.51**, and the pre-existing suite failures as **86.5**.
The swap failure was additionally confirmed pre-existing on its own, by the same
revert-and-restore method, before the full-suite comparison.

**The 21 node ids, ENUMERATED (added post-verdict per Q/A finding N5, which
correctly noted the artifact named only two of them and asked for the list):**

```
backend/tests/test_phase_23_2_10_watchdog_no_fire_7d.py::test_phase_23_2_10_watchdog_log_present_and_fresh
backend/tests/test_phase_23_2_13_governance_watcher.py::test_phase_23_2_13_backend_log_boot_pair_present
backend/tests/test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
backend/tests/test_phase_23_2_9_ticker_meta_latency.py::test_phase_23_2_9_backend_log_has_prewarm_evidence
backend/tests/test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit
backend/tests/test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
backend/tests/test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
backend/tests/test_phase_75_17_verification_paths.py::test_masterplan_diff_touches_only_the_ten_sibling_insertions
backend/tests/test_phase_75_17_verification_paths.py::test_sweep_over_live_masterplan_is_clean
backend/tests/test_phase_75_17_verification_paths.py::test_sweep_shape_census_matches_the_corrected_figures
backend/tests/test_phase_75_19_preflight_calibration.py::test_live_masterplan_is_currently_clean
backend/tests/test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token
backend/tests/test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted
backend/tests/test_phase_75_sre_ops.py::test_c6_no_launchctl_bootstrap_executed_in_ops_scripts
backend/tests/test_phase_82_39_outcome_rebuild_query.py::test_the_sweeps_recall_limit_is_recorded_not_assumed
backend/tests/test_phase_82_48_outcome_write_schema.py::test_the_fetch_supplies_every_field_the_write_REQUIRES
backend/tests/test_phase_82_48_outcome_write_schema.py::test_write_really_persists_into_bigquery
backend/tests/test_phase_86_6_subprocess_channel.py::test_the_optin_IS_honoured_so_a_real_window_remains_possible
backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
```

The Q/A corroborated the causality independently by a different method
(in-process HEAD injection over the 43 test files that import
`autonomous_loop`): 4 failures with the fix, the same 4 with HEAD injected, set
difference **EMPTY**. Its 4 are the three `test_phase_57_1_reject_binding.py`
cases plus `test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off`,
which it identifies as an operator-`.env` flag-state dependency rather than a
code failure. Two independent methods, same conclusion.

---

## 7b. Working-tree changes that are NOT mine (Q/A finding N6)

The Q/A flagged that `backend/api/sovereign_api.py` and five `frontend/src`
files are modified in the working tree, **all with mtime 2026-08-14 -- a day
before this step** -- and warned that the auto-commit hook's `git add -A` would
sweep them into this step's commit.

**They are a peer session's uncommitted work and I did not touch them.** They
are absent from the Files-changed table in §3 because they are not part of this
change. **Handling:** this step is committed with **explicit pathspecs only**,
and the masterplan flip is performed in a way that does not trigger the
`git add -A` hook, so the peer's files stay uncommitted and under their owner's
control. Verified after committing -- see §10.

---

## 8. Non-scope compliance (criterion 9)

- **No flag promoted, no `.env` written, no gate loosened.** The fix is
  unconditional and makes the system strictly *more* restrictive on the one
  input that changed a decision.
- **Paper only.** No live-book interaction; every measurement is an in-process
  call to `decide_trades` with a synthetic portfolio.
- **No manual cycle run.**
- **`decide_trades`, `_resolve_position_pct`, `_sizing_pct` and all other
  phase-86.74 code are untouched.** The consumer was correct; only the producer
  lied to it.
- **`risk_debate.py`, `orchestrator.py` and `_data_integrity_blocked_analysis`
  untouched** -- their zeros are written as plain literals and already survive.
- **Restart batched to session end.** Recorded above as NOT YET IN FORCE.

---

## 9. What I could not verify

1. **The fix is not active in the running backend.** It is verified in a fresh
   process only. Confirming it in the live process requires the end-of-session
   restart, which has not happened at the time of writing.
2. **No production lite judge has been observed emitting `0.0` in the wild
   during this step.** The defect is proven by driving the code, and by the fact
   that the codebase's own judges emit `recommended_position_pct: 0` on REJECT
   (`risk_debate.py:152`, `orchestrator.py:2415`). I did **not** query BigQuery
   for a historical lite row carrying a 0.0 verdict, so the *frequency* of live
   exposure is unmeasured -- only its mechanism and magnitude.
3. **`''`, `False` and garbage-string reachability from a real lite judge is
   unmeasured.** Those rows in the disclosure table are behaviour-under-input
   facts, not claims that a judge emits them.
4. **The pre-existing failing test `test_the_optin_IS_honoured...` was not root
   caused** -- only shown not to be attributable to this change.
