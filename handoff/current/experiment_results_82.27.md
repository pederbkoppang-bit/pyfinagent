# Experiment results -- phase-82.27

**Step**: RE-SPEC of 82.23 criterion 1 -- name the SWEEP-LEVEL producer, not `generate_report`.
**Contract**: `handoff/current/contract_82.27.md` | **Research**: `handoff/current/research_brief_82.27.md` (gate_passed=true, 6 read in full, 28 URLs, recency scan, 13 internal files)

## 1. What was built

### 1a. THE WIRING DEFECT -- the `min_pbo_trials` floor I shipped in 82.23 was DEAD

`backend/autoresearch/strategy_candidate_producer.py` rebuilt every candidate from a
hardcoded 5-key whitelist (`strategy`, `dsr`, `pbo`, `params`, `sharpe`). So `pbo`
reached `PromotionGate.evaluate`, but every field describing HOW it was computed was
discarded in transit -- including `pbo_n_trials`, which is the floor's only input.

Found by the 82.27 research gate, which ran the real modules rather than reasoning
about it. Reproduced by me on the same modules:

```
gate verdict WITH    pbo_n_trials=3 -> {'promoted': False, 'reason': 'pbo_trials_below_min:3<10'}
gate verdict WITHOUT pbo_n_trials   -> {'promoted': True,  'reason': None}
```

The second line was the live behaviour. A gate term whose input never arrives cannot
fail, and a term that cannot fail is indistinguishable from one that passes -- the
`feedback_operations_that_cannot_fail_loudly` class, this time in something I shipped
myself one step earlier.

Fix: forward the six provenance keys when present. Deliberately **forwarded, not
recomputed** -- this module never sees the PnL matrix, so it cannot re-derive them and
must not invent them. An absent key stays absent, which the gate treats as the legacy
no-N case (unchanged behaviour), never as a passing value.

### 1b. The MCP surface re-exposed the false-good 0.0

`backend/agents/mcp_servers/risk_server.py::pbo_check` called raw `compute_pbo`, which
returns **0.0 -- the best possible value -- at N<2 or T<S*2**. On a `pbo > threshold`
veto that sentinel does not merely fail to inform: it manufactures a clean pass. Now
routed through `compute_pbo_checked`, with a refusal reported as `ok=False` +
`isError=True`, and with the trial count and diversity travelling alongside every
accepted value.

**CORRECTION (cycle-1 Q/A [BLOCK]):** this paragraph originally claimed the refusal
was reported "so a caller cannot read it as within bounds". That was FALSE. The only
in-repo consumer -- `evaluate_candidate`, thirty lines below in the same file --
read ONLY `pbo_result['vetoed']` and ignored `ok`/`isError`, so a refusal produced a
composite verdict of `passed_all_gates`. Fixed in cycle 2; see section 7.

### 1c. The file drawer is now reported

`strategy_backtest_adapter.py` silently drops a variant's column when it raises. Bailey
et al.: *"Hiding trials will lead to an underestimation of the overfit."* Failures are
disproportionately the bad configurations, so dropping them biases PBO **down**, toward
promoting. `pbo_dropped_columns` is now emitted on all three return paths -- including
the two refusal paths, which is exactly where a reader most needs it. This does not
correct the bias; it makes it visible.

### 1d. New suite

`backend/tests/test_phase_82_27_pbo_sweep_producer.py`, 14 tests. Every guard EXECUTES
the code under test -- the 82.23 Q/A rejected `inspect.getsource` token scans as
satisfiable by a comment, and none are used here.

**Criterion 2 is behavioural, not a signature check**: it runs the REAL
`generate_report` against a stub result and asserts no `pbo` appears anywhere in the
returned structure, with a sanity assert that the report did produce metrics so the
test cannot pass by the report being empty.

## 2. Files changed

| File | Change |
|---|---|
| `backend/autoresearch/strategy_candidate_producer.py` | forward 6 PBO provenance keys through the candidate hop (additive) |
| `backend/agents/mcp_servers/risk_server.py` | `pbo_check` -> `compute_pbo_checked`; refusal reported as `ok=False`/`isError=True`; N + diversity returned |
| `backend/autoresearch/strategy_backtest_adapter.py` | compute + emit `pbo_dropped_columns` on all 3 return paths |
| `backend/tests/test_phase_82_27_pbo_sweep_producer.py` | NEW, 19 tests (14 at cycle 1) |

No other production code touched.

## 3. Verbatim verification output

> **CYCLE 1 -- SUPERSEDED BY SECTION 7.** The capture below is the genuine, unedited
> cycle-1 output (35 = 21 + 14). It is deliberately NOT rewritten to the current 40:
> a "verbatim" block that is edited to match a later run is a fabricated capture, which
> is the defect in `feedback_operations_that_cannot_fail_loudly`. The current
> measurement lives in section 7.

```
$ python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py backend/tests/test_phase_82_27_pbo_sweep_producer.py -q
...................................                                      [100%]
35 passed in 7.43s
exit=0
```

## 4. Mutation matrix -- production mutants, applied then reverted

> **CYCLE 1 -- SUPERSEDED BY MATRIX v3 IN SECTION 7.** CONTROL 14 was the tree at cycle
> 1; the suite has since grown to 19. Kept as the historical record, not as a claim
> about the delivered tree.

```
CONTROL                                       -> 14 passed
M1 revert the whitelist (the ORIGINAL defect) ->  2 failed 12 passed
M2 drop ONLY pbo_n_trials                     ->  2 failed 12 passed
M3 dropped-column count hardcoded to 0        ->  2 failed 12 passed
M4 MCP back to raw compute_pbo (false-good)   ->  2 failed 12 passed
RESTORED                                      -> 14 passed
```

M1 is the strongest evidence available: it re-introduces the exact defect that shipped,
and the suite catches it. M3 is the discrimination check -- a hardcoded 0 satisfies
"reports a count" but not "reports the RIGHT count", which is why
`test_dropped_column_count_is_zero_when_nothing_failed` exists beside it.

## 5. What the research gate corrected in me -- four times, all measured

1. **"16 call sites" is 15.** The 16th is `monkeypatch.setattr` -- a symbol
   substitution, not an invocation. Substance confirmed; count wrong. Acted on its
   better advice: criterion 2's companion asserts a **property** (no plural collection
   parameter) rather than a number, so it cannot drift when a script is added.
2. **This step needed a WIRING CHANGE, not only tests.** I had framed that as an open
   question; it measured the answer.
3. **`quant_optimizer` is not a sweep producer** and is now explicitly excluded, not
   left open: it retains only scalars (`:610`), and it is a guided search whose
   intermediate steps Bailey Sec 5.2 bans as CSCV columns. Two independent
   disqualifications.
4. **`compute_pbo` is at `analytics.py:276`, not `:184`.** These line numbers have
   moved twice this phase. Re-derived before use.

## 6. Scope honesty

- **Nothing schedules the rotation path.** `rotation_runner.py:36` and
  `strategy_candidate_producer.py:33` both still describe the weekly CRON as DEFERRED,
  and no rotation scheduling exists in `backend/services`, `backend/main.py` or
  `backend/slack_bot`. The hop this step repairs is real and closed but **currently
  unexercised in production**. This is a pre-emptive fix, not an outage repair, and it
  should not be described as a live-money save.
- **`_DEFAULT_K = 8` is BELOW the `min_pbo_trials = 10` floor** (adapter `:70` vs
  `gate.py`). The default sweep will now be REFUSED where it previously promoted. That
  is the correct behaviour and it is surfaced rather than accommodated: I did not raise
  K to make the gate pass, which would be tuning the test to the answer. Choosing K is
  a separate decision and is queued as **82.26**.
- The live `optimizer_best.json` stays mis-attributed; regenerating it needs an
  optimizer run gated on the frozen `historical_macro`. Unchanged from 82.22.
- 82.23 stays permanently `pending`, superseded in place, criteria untouched.
- The `risk_server` change is a behaviour change on an MCP tool: an undersized matrix
  that previously returned `{ok: True, pbo: 0.0, vetoed: False}` now returns
  `{ok: False, pbo: None, isError: True}`. Fail-CLOSED on a risk veto is deliberate,
  but it IS a contract change for any caller that only checked `vetoed`.

---

## 7. CYCLE 2 -- the four cycle-1 blockers, closed

Cycle 1 (task `wusnhzw5a`, verbatim in `evaluator_critique_82.27.md`) returned
CONDITIONAL. Criteria 1 and 2 MET and mutation-killed by the evaluator itself; it also
answered all four attacks I asked it to run, confirmed no source scan does load-bearing
work, mutated my FIXTURE (not just the code) to prove the type-aware `__getattr__` is
load-bearing -- a blanket `0.0` fallback turns 6 of 14 tests red -- and verified my
`_DEFAULT_K=8 < 10` claim as **stronger than I stated**: every strategy at default K is
now refused, not merely "would be".

### B1 -- criterion 3's second clause was unproven ON THE PRODUCER

The producer's under-length return was exactly
`['dsr','n_variants','n_windows','pbo_dropped_columns','sharpe']` -- **no reason key at
all**. The reason existed only as a `_log.warning` that no consumer and no test can
see. Worse, my one reason-asserting test targeted `risk_server.pbo_check` -- which my
own contract classifies as a CONSUMER, not the producer -- and drove it with T=40/N=1,
exercising the **N<2 branch, not the under-length branch the criterion names**. So I
tested the right idea on the wrong object via the wrong branch.

Fixed: both producer refusal paths now return `pbo_refused`. The candidate producer's
skip log also now prints it, since that is the line an operator reads when a strategy
silently stops being considered.

Measured:
```
under-length (T=19):  pbo=None  pbo_refused='matrix undersized/degenerate: need >=2
                                             columns and >=32 rows; got 12 usable variant(s)'
well-sized  (T=399):  pbo=0.4609  pbo_refused=None
```

### B2 -- the matrix named no mutant for criterion 2's guard

True. The cycle-1 Q/A supplied the missing mutant itself (patching `generate_report` to
add `analytics['pbo']=0.0`), and it KILLED the test -- so the guard was sound; the
matrix simply did not name it. Now M5.

### B3 -- lint gate red

`F401 json` and `F401 math` in `risk_server.py`. Pre-existing at HEAD (I re-derived
against `git show HEAD:`), both genuinely dead (`grep` for `json.`/`math.` returns
nothing), and in a file this diff already edits. Removed. Gate now `All checks passed!`.

### B4 -- THE COMPOSITE GATE SWALLOWED THE REFUSAL. This is the real one.

`evaluate_candidate` read only `pbo_result['vetoed']`. A refused PBO -- `ok=False`,
`pbo=None`, `isError=True` -- fell straight through as `passed_all_gates`. So the
fail-closed property I *claimed for this very step* was false at the only in-repo
consumer, in the same file, thirty lines below the function that produces the refusal.

I asked the evaluator to look for exactly this kind of caller. It found one. That is
the difference between asking a question and answering it myself: I wrote the
disclosure in section 6 naming the contract-change risk **generically**, which reads as
diligence while never checking whether such a consumer existed.

Not a regression -- the pre-change raw `compute_pbo` returned a false-good 0.0, which
also produced `vetoed=False`, so the composite outcome is unchanged. But "no worse than
before" is not the property being advertised. A refusal means the overfitting term
**could not be evaluated**, and on a risk veto that must block. Now:
`reason = f"pbo_unevaluable:{...}"`.

### Re-measured on the current tree

```
$ python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py backend/tests/test_phase_82_27_pbo_sweep_producer.py -q
40 passed
exit=0
```

### Mutation matrix v3 -- attribution MEASURED per row, not labelled

Cycle-2 Q/A [BLOCK]: v2 labelled M1/M2 "[criterion 1 wiring]", but they kill only the
candidate-producer HOP tests -- never `test_sweep_producer_emits_pbo_with_its_trial_
count_and_diversity`, which is criterion 1's actual guard. So no row named a mutation
for criterion 1's own guard, and two rows were mis-credited. That is vacuity shape #11
(mis-attributed kill mechanism), and it is **the same class as cycle-1's B2 finding,
which I fixed for criterion 2 only** rather than sweeping every criterion for it. The
guard itself was sound; the artifact was wrong about why.

Every row below was re-run with `pytest -rf` and the FAILING TEST NAMES captured from
the run. Nothing here is inferred from what a mutant "should" break. Each mutant is
`ast.parse`-validated first; a mutant that does not compile prints SYNTAX-BROKEN and is
not counted (see the note below).

**Scope: `backend/tests/test_phase_82_27_pbo_sweep_producer.py` alone (19 tests).**
Cycle-3 Q/A [NOTE 1]: on the immutable command's FULL scope (both suites) M5=3, M9=2 and
M10=3, because each also kills one genuine test in the 82.23 suite. Every named test
below is correct and the omission is CONSERVATIVE -- no row over-claims -- but the scope
belongs in the header rather than being inferred.

```
                                          failed  tests actually killed
CONTROL                                    0 (19 passed)
M1  revert provenance passthrough              2   hop, trials_floor
M2  hop drops only pbo_n_trials                2   hop, trials_floor
M3  dropped-count hardcoded 0                  2   dropped_columns_are_reported,
                                                   dropped_count_..._when_pbo_is_refused
M4  MCP back to raw compute_pbo                3   refusal_reason_..._mcp_surface,
                                                   mcp_surface_still_vetoes,
                                                   composite_gate_..._clean_pass
M5  generate_report emits a pbo                2   single_run_report_emits_no_pbo_anywhere,
                                                   no_generate_report_invocation_... (see caveat)
M6  composite gate ignores ok=                 1   composite_gate_..._clean_pass
M7  producer reports NO refusal reason         1   the_producer_itself_reports_why_it_refused
M8  refusal reason is a constant               1   the_producer_itself_reports_why_it_refused
M9  adapter drops pbo_column_corr_mean         1   sweep_producer_emits_pbo_with_..._diversity
M10 adapter drops pbo_n_trials                 2   sweep_producer_emits_pbo_with_..._diversity,
                                                   dropped_columns_are_reported
RESTORED                                       0
```

**Criterion -> the mutant that kills THAT criterion's own guard:**

| Criterion | Guard | Killed by |
|---|---|---|
| 1 sweep producer emits pbo + N + diversity | `test_sweep_producer_emits_pbo_with_its_trial_count_and_diversity` | **M9, M10** |
| 2 single-run path emits no pbo | `test_single_run_report_emits_no_pbo_anywhere` | **M5** |
| 3 under-length refuses AND reports why | `test_the_producer_itself_reports_why_it_refused` | **M7, M8** |
| 4 (this table) | -- | -- |

M1/M2 are re-labelled honestly: they cover the **candidate-producer hop** (the live
dead-floor defect), which is the step's headline repair but is NOT criterion 1's guard.

**CAVEAT ON M5, stated because attribution is exactly what was wrong before.** M5 kills
two tests, but only the FIRST is a genuine kill. I built M5 by renaming `generate_report`
to `_orig_generate_report` and adding a `def generate_report(*a, **k)` wrapper -- and
that wrapper's `*args` signature is itself enough to fail
`test_no_generate_report_invocation_receives_more_than_one_result`, whose assertion is
about the parameter list. That second failure is an artifact of HOW the mutant was
constructed, not evidence about the property. Criterion 2's real evidence is the first
kill alone.

**A note on how this matrix was run.** My first M7/M8 attempt spliced on `s.index(')}')`,
which matched inside `{len(results)}` and produced a syntactically broken file --
reported as "8 failed", which would have read as a strong kill. The harness now
`ast.parse`s every mutated file before running and prints SYNTAX-BROKEN instead of a
count. A mutant that does not compile tests nothing; counting it would have inflated the
evidence.

### Kill-switch safety

`test_composite_gate_*` needs an unpaused switch (this machine's real one is paused, so
without the patch the tests would never reach the PBO gate). The mock patches the READ
path only and never flips real state. Verified before and after the suite:
`paused=True reason='manual'` -- unchanged.

### Commit scope -- measured, not asserted

The cycle-1 Q/A measured that a flip would stage **23 paths, only 4 of them 82.27's**.
The other 19 are another session's in-flight phase-83 work (the masterplan re-plan,
`handoff/current/phase83_research_raw/`, a `Cycle 1137 phase=83 DESCOPE` log block,
`.claude/agent-memory/researcher/` files) plus the phase-82.22 archive and hook JSONLs.
82.27 will be committed with a deliberate `git add` of its own files only -- not
`git add -A`.
