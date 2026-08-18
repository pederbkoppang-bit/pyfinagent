# evaluator_critique -- step 86.116

## Verdict ledger

| cycle | verdict | run id | recorded |
|---|---|---|---|
| 1 | **CONDITIONAL** | `wf_6c5d3dfc-43a` | 2026-08-18T05:28:04Z |
| 2 | **CONDITIONAL** | `wf_10d2c895-28e` | 2026-08-18T05:51:37Z |
| 3 | **CONDITIONAL** | `wf_62e2fe3c-126` | 2026-08-18T06:12:00Z |

**PARKED.** `[C,C,C]` -- CLAUDE.md F1 forces the next verdict to FAIL regardless
of evidence. Budget is 4/5, so the BUDGET is not the constraint; the rule is.
See `escalation_86.116_third_conditional.md`.

---

## Cycle 1 -- CONDITIONAL -- `wf_6c5d3dfc-43a`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed. Main records the verdict and never authors it.*

**verdict:** `CONDITIONAL` | **ok:** `false`

**reason:**

> The PRODUCT fix is correct, minimal, complete and mutation-proven, and I reproduced every
> headline number myself; two fixable EVIDENCE defects cap the verdict. Reproduced: immutable
> command exit=0 ("parses"); 12/12 tests pass; ruff F821/F401/F811 clean over a git-DERIVED
> 5-file scope (non-empty guard asserted first); the author's matrix re-run gives control GREEN
> then 6/6 KILLED with collected-count parity, named-test-must-fail and a byte-identical sha256
> restore (I verified 9f5f1d67... before and after, 0 MUTANT markers); the full suite re-run by
> me gives 19 failed / 3634 passed vs the author's 20/3633 -- the delta is exactly the one
> line-pin test they fixed, the remaining 19 are exactly their enumerated 18 pre-existing plus
> the ordering artifact, which I confirmed passes alone. Independently derived from BigQuery:
> AVB 2026 raw=159 distinct=155, drop_duplicates()=159 vs ~index.duplicated()=155, and AKAM
> 2025 where BOTH give 250 -- the step's method claim AND its own counter-example disclosure
> both hold; the close-gap distribution behind cell E1 (394,719 keys, p50 0.0000%, p99
> 0.0000%, max 0.9326%, zero above 2%) reproduces exactly even though its derivation is not
> shipped. Call-site completeness verified independently: _prices_full has exactly ONE
> assignment (:293) and _prices_cache ONE (:641), both deduped; no SQL-level DISTINCT in either
> read query; all four audit_basis consumers route through cached_prices. Zero DML in the
> commit. Harness compliance clean on all 5 items; research gate 6 sources / 25 URLs / recency
> scan / COMPLETE envelope, including three ADVERSARIAL sources. CAPPING FINDINGS: (1)
> criterion 6's only quantified mechanism credits a DOCUMENTED DEAD KEY --
> vol_barrier_multiplier has zero readers repo-wide and is listed by name in
> backend/autoresearch/rotation_runner.py:58-69 `_DEAD_KEYS` ("NO engine reader (reverted in
> 9fbd9cd6)"); the literal `barriers = daily_vol x multiplier` exists only as a COMMENT at
> quant_optimizer.py:213, and backtest_engine._compute_triple_barrier_label:1088 uses fixed
> tp_pct/sl_pct with no volatility term -- while the LIVE amplifier the step misses is
> backtest_trader.py:89 `vol_scale = min(self.target_vol / stock_vol, 3.0)`, fed by
> annualized_volatility (historical_data.py:128 -> backtest_engine.py:1251 -> :1257 ->
> backtest_trader.py:146), which at the measured 0.7995 vol ratio inflates position size
> ~1.25x; the conclusion is right and understated, the credited mechanism is not wired. (2) The
> first re-runnable check the live_check advertises, `python scripts/qa/verify_86_116.py`, now
> ABORTS ("INVARIANT FAILED: no_dedup_existed_before_this_step ... {'drop_duplicates': 2,
> 'index\\.duplicated': 1, 'is_unique': 2}") because :281 defaults --base-rev to HEAD and :286
> runs that probe first, so criteria 1/2/5/6 evidence never prints; `--base-rev 34a56b03`
> restores all 27 invariants and every number byte-for-byte. Both fixes are edits to evidence,
> not to production code.

**violated_criteria:** `criterion_6_gate_effect_mechanism_credits_a_dead_key`,
`live_check_rerunnable_evidence_command_aborts`,
`criterion_7_read_path_fixtures_cannot_see_a_method_swap`

### violation_details

**1. Unjustified_Inference.** `vol_barrier_multiplier` has **zero readers
repo-wide**: the 7 non-86.116 hits are a search-space bound
(`quant_optimizer.py:214`), the write into `engine._strategy_params` (`:715-716`),
a cache-key membership list (`:738`), a docstring plus an explicit `_DEAD_KEYS`
entry (`rotation_runner.py:21-24, :58-69` -- *"NO engine reader (reverted in
9fbd9cd6) ... nothing reads them today"*), and a `0.0` default
(`archetype_library.py:111`). The cited formula exists only as a **COMMENT** at
`quant_optimizer.py:213`. `_compute_triple_barrier_label` (`:1066-1103`) computes
`tp_price = entry_price * (1 + self.tp_pct/100 + round_trip_cost_pct)` -- fixed
percentage barriers, **no volatility term**. `features['daily_volatility']` is
written once and read nowhere. The comment at `:718` names
`_compute_vol_target_scale` as another reader; **that function does not exist
anywhere in the repo**. The LIVE mechanism, unmentioned by the step:
`historical_data.py:53 cached_prices -> :128 annualized_volatility ->
backtest_engine.py:1251 -> :1257 signal dict -> backtest_trader.py:146-147 ->
:89 vol_scale = min(self.target_vol / stock_vol, 3.0)`, i.e. inverse-volatility
position sizing, which at the measured 0.7995 ratio inflates size ~1.25x.
Mechanism (a) (features -> candidate selection) **IS real and verified end to
end**, and no threshold was moved.

**2. Invalid_Precondition.** `python scripts/qa/verify_86_116.py` aborts
immediately; `:281` defaults `--base-rev` to `HEAD` and `no_dedup_before()` is
`main()`'s FIRST call (`:286`), so census, driven-harm, parity-oracle and gate
sections never print. `--offline` shares the defect. *"HEAD was the pre-fix tree
only while the fix was uncommitted; commit 539f16eb made it the post-fix tree, so
the printed line 'pre-fix revision under test: HEAD' is now false."* With
`--base-rev 34a56b03` all 27 invariants hold and the output matches the
live_check **byte-for-byte**. Hand-checked: the ten year-bucket key counts sum to
exactly 1,152,607 and the dup counts to exactly 706,875.

**3. Missing_Assumption.** Independent cells on scratch copies (repo sha256
verified unchanged, 0 MUTANT markers): mutating `_dedupe_index` to value-keyed
`drop_duplicates()` and driving BOTH read paths with the shipped `_fake_rows`
fixture leaves **both read-path tests green** (`unique=True len=6` on each,
identical to control), because the fixture's duplicates are byte-identical
`dict(r)` copies. With `close + 1e-9` -- the shape 394,719 of 706,875 real
duplicated keys have -- the same mutant leaks (`unique=False len=12` on both). So
the method choice the step itself calls *"THE METHOD IS THE FINDING"* is pinned
**only** by the helper-level `test_value_keyed_dedup_is_insufficient`.

---

## Main's response -- cycle 2 (all three fixed)

**All three accepted; none disputed.** Finding 1 is one I have on record as a
recurring error of mine -- *a correct observation can credit the wrong
mechanism*. My conclusion (volatility corruption reaches the gates) was right and,
as the evaluator says, **understated**; the mechanism I named was not wired.

- **The dead key is out.** `criterion 6` now credits the **live** chain,
  `annualized_volatility -> signal dict -> size_position -> vol_scale =
  min(target_vol / stock_vol, 3.0)`, and reports **position-size inflation
  1.2508x** (= 1/0.7995) against the 3.0 cap, with `sqrt(2)` = 1.4142x as the
  bound under full duplication. The artifact states the direction explicitly
  because it is counter-intuitive: `stock_vol` is in the **denominator**, so an
  understated volatility makes positions **larger** -- the backtest was taking
  more risk than its own vol-targeting believed.
- **Two tripwires keep the correction honest**: one asserts
  `vol_barrier_multiplier` is still listed in `_DEAD_KEYS`, the other that the
  triple-barrier label still has no volatility term. If either changes, the
  section must be re-derived rather than trusted. *(A first attempt at the
  tripwire was a grep-with-filters and it fired on the setter's own `if ... in
  params:` guard -- too brittle, so it was replaced with the repo's own
  authoritative statement.)*
- **The re-runnable command works from any checkout.** `--base-rev` is pinned to
  `539f16eb~1` instead of `HEAD`. A default that silently expires the moment the
  work is committed is a defect, not a convenience.
- **The fixture can now represent the failure.** `_fake_rows` produces duplicates
  differing by `1e-9`, so a value-keyed mutant leaks through **both read paths**
  -- new cells **M3b** and **M3c** confirm it, and a new test
  (`test_the_read_path_fixture_would_defeat_a_value_keyed_dedup`) asserts the
  precondition directly, so a future edit restoring byte-identical twins fails
  loudly rather than silently disarming two tests.

Matrix **8/8 KILLED**, control GREEN first at 13 collected, SHA-256 restore
verified. Evidence script **31 invariants**. No production code changed in this
cycle -- every fix was to evidence, exactly as the evaluator characterised them.

---

## Cycle 2 -- CONDITIONAL -- `wf_10d2c895-28e`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed.*

**verdict:** `CONDITIONAL` | **ok:** `false` | **harness_compliance_ok:** `true`

**reason (the capping half; the confirmatory half is in the run record):**

> The PRODUCT fix is correct, minimal, complete and mutation-proven, all three cycle-1
> findings are genuinely fixed, and I reproduced every headline number independently -- one new
> capping finding, in a guard cycle 2 itself added. [...] the FULL suite (not a -k selection)
> gives 19 failed / 3635 passed, which is exactly 86.118's enumerated 18 plus the declared
> ordering artifact [...] and the three independent runs reconcile arithmetically (author
> 3633/20 pre-pin-fix, cycle-1 Q/A 3634/19, mine 3635/19, the +1 being precisely the one test
> cycle 2 added). [...] CAPPING FINDING: the cycle-2 tripwire
> triple_barrier_label_has_no_volatility_term (verify_86_116.py:319-325) cannot fire for the
> condition it names. Executed, not argued: adding a real volatility term to
> _compute_triple_barrier_label while retaining self.tp_pct -- the shape any vol-scaled barrier
> would take -- leaves the guard GREEN (clause A False, clause B True), and it fires only when
> self.tp_pct is ALSO renamed away, i.e. it detects a RENAME. The `or "self.tp_pct" in
> label_src` clause is unconditionally true on the control and disarms the half that works.
> Criterion 7 says mutation-test EVERY NEW GUARD; the two tripwires are new guards, are not
> cells in mutation_86_116.py, and the one Main's own comment calls "the claim that actually
> matters" survives. **Notably the recorded cycle-2 lesson was that the first tripwire was too
> BRITTLE -- the remedy for brittleness introduced the vacuity.** FIX: drop the escape clause,
> bound the slice to the function instead of a fixed 2600 chars (the function is 1,788, so it
> overshoots 812 into _compute_sample_weights), and add matrix cells that mutate both
> tripwires. T1 is NOT vacuous and I checked it: the literal `"vol_barrier_multiplier",` occurs
> exactly once, at rotation_runner.py:64 inside _DEAD_KEYS, and the :22 docstring mention does
> not satisfy it.

**violated_criteria:**
`criterion_7_cycle2_tripwire_not_mutation_tested_and_survives_its_own_subject`

**violation -- Circular_Reasoning.** Mutated `_compute_triple_barrier_label` in
memory to add a real volatility term and evaluated the guard expression:
`CONTROL A=True B=True GUARD=True`; `MUTANT V1 (vol term added, self.tp_pct
retained) A=False B=True GUARD=True **SURVIVED**`; `V2 (inline) SURVIVED`; `V3
(vol term added AND self.tp_pct renamed) fires`. *"Clause A alone would have
fired on V1/V2; the clause `or "self.tp_pct" in label_src` is already True on the
unmutated tree and overrides it, so the guard detects a RENAME rather than a
volatility term."* Severity WARN/capping rather than BLOCK, because it is *"a
durability guard on a narrative correction rather than the evidence for an
immutable criterion -- criterion 6 requires the gate effect be REPORTED, and that
report is correct and was independently verified."*

---

## Main's response -- cycle 3

**Accepted, and the evaluator's sharpest line is the one I want on the record:**
*the remedy for brittleness introduced the vacuity.* Cycle 2's tripwire v1 was a
grep-with-filters that fired on the setter's own guard; I replaced it with a
text check carrying an `or` escape clause, and that clause was **true on the
unmutated tree**, so it short-circuited the half that worked. I reproduced the
exact result before changing anything: `MUTANT vol added -> A=False B=True
GUARD=True <- SURVIVES`, and measured the function at **1753** chars against a
2600-char window (my number differs slightly from the evaluator's 1,788/812; I
report mine and the direction is identical).

**v3 has no escape clause and reads the AST.** `_volatility_identifiers()`
collects every identifier the function actually *references* -- `Name`,
`Attribute`, `arg` -- and rejects any containing `vol`. Verified against the
evaluator's own mutants: `V1 (daily_vol, vol_mult)` **KILLED**, `V2
(self.daily_volatility)` **KILLED**, and a comment-only mention correctly does
**not** fire, so it is precise rather than merely sensitive. The slice is bounded
by `ast`, not a character count.

**Both tripwires are now mutation-tested, and that required a second target.**
The matrix ran only against `cache.py`, which is exactly how cycle 2 shipped an
untested guard. It now carries a second target (`verify_86_116.py`, exercised via
`--offline` so no cell depends on BigQuery) with its own control, its own
SHA-256 restore, and cells T1/T2/T3.

**The first attempt at those cells SURVIVED, and that was the real lesson.**
Disarming `_ok(name, EXPR)` to `True` cannot be detected by that assertion --
the same result 86.59 produced. So the rules are now named predicates
(`_declares_dead_key`, `_volatility_identifiers`) backed by a fixture of
known-bad inputs they must reject, and an **incomplete** fixture is reported as
a fixture failure rather than crashing with `KeyError` (which had scored T3
UNSCORABLE rather than as a kill).

Matrix **11/11 KILLED** across both targets, 0 SURVIVED, 0 UNSCORABLE. Evidence
**33 invariants**. `backend/backtest/cache.py` is byte-identical to what cycles
1 and 2 evaluated (`9f5f1d67...`) -- **no production code has changed since the
original fix.**

---

## Cycle 3 -- CONDITIONAL -- `wf_62e2fe3c-126`

*Transcribed verbatim from the captured return in the same turn it landed.*

**verdict:** `CONDITIONAL` | **ok:** `false`

**violated_criteria:** `criterion_6_cap_guard_cannot_fire_for_the_saturation_it_names`,
`criterion_1_identity_assertion_credited_with_drift_protection_it_lacks`,
`scope_honesty_no_restart_pending_is_false_for_the_in_process_api_path`

**1. Circular_Reasoning.** *"`cap_is_accounted_for` asserts `size_inflation <
3.0`, but the 3.0 cap binds on target_vol/stock_vol, not on the inflation ratio;
with max multiplicity 2 the inflation is bounded by sqrt(2)=1.4142 -- a bound the
same function PRINTS -- so it cannot fail."* Driven with saturating inputs:
`vol 0.020/0.025 -> ALL 8 gate invariants PASS, script reports 1.2500x, TRUE
1.0000x`; `0.040/0.060 -> reports 1.5000x, TRUE 1.2000x`. *"The script takes
--ticker, so the false negative is reachable."*

**2. Overgeneralization.** *"Over the grouping SUM(n)==total_rows and
COUNT(*)==keys, so SUM(n-1)==total_rows-keys is an ALGEBRAIC IDENTITY that cannot
detect a normalisation error."* Criterion 1 is met independently by the printed
rule and the per-share labels; only the prose claim was wrong.

**3. Invalid_Precondition.** *"'no restart pending' is FALSE for the in-process
API path: uvicorn pid 41635 started 2026-08-17 15:57:16 (etime 16:10, no
--reload) while cache.py landed 07:55, backtest_engine.py:25 imports cache at
module level, and backend/api/backtest.py:1008 runs engine.run_backtest inside
that process -- so every API-triggered backtest still reads duplicated frames.
[...] The justification is wrong: preload_prices is called per run, but from the
already-imported module object."*

**NOTE-level:** `_volatility_identifiers` rejects only `vol`-named identifiers,
so `sigma`, `self.daily_sigma`, an inline `.std()` and `_atr_width` all MISS --
*"though it is NOT vacuous (I reproduced V1/V2 and T2 killing it)"*. The
"positive control finds this step's own fix" line is prose-only. The live_check
full-suite block is the pre-pin-fix `20/3633` capture, unrelabelled.

---

## Main's response -- cycle 3

**All three accepted; finding 3 is the one that mattered and I verified it
myself before changing anything**: pid 41635, started 2026-08-17T15:57:16Z, up
16h14m, no `--reload`; `backtest_engine.py:25` module-level import confirmed; a
*fresh* import carries `_dedupe_index` while that process cannot. My
justification -- "the backtest loads it per run" -- was wrong in exactly the way
this project has a standing rule about: `preload_prices` runs per call, **from
the already-imported module object**. Committed is not in force. The claim is
corrected in `experiment_results` **and in the next-session goal file**, which
had carried "Restart: none pending" to the operator.

**Finding 1 fixed and the replacement is proven to fire**: the guard now asserts
`vol_scale` is unsaturated on **both** sides (`target_vol / vol_ann < 3.0`,
`target_vol = 0.15` from `backtest_trader.py:54`). On the real control it passes
(vol_scale 0.4487 / 0.3587 -- matching the evaluator's independently computed
figures); on **both** of the evaluator's saturating cases it **FIRES**.

**Finding 2 fixed by renaming rather than defending**: the assertion is now
`census_sql_is_internally_consistent` and claims only what an identity can claim,
and the false prose is replaced by an explicit correction.

**The NOTE is carried forward, not silently absorbed**: `_volatility_identifiers`
would miss a `sigma`-named term. It is not vacuous, but it is narrower than its
name suggests, and that belongs in the escalation rather than in a fix made
during EVALUATE.

---

## Cycle-3 Follow-up -- what Main changed in response

All three cycle-3 violations are addressed. Each was verified by driving the
system, and the capture below was regenerated from ONE run rather than edited.

### V1 -- `cap_is_accounted_for` could not fire (the BLOCKING one)

The cycle-3 Q/A was exactly right: the guard asserted `size_inflation < 3.0`
while the 3.0 cap binds on `target_vol/stock_vol`, and with max multiplicity 2
the inflation is bounded by sqrt(2)=1.4142 -- a bound the same function PRINTS.
It could not fail on any input, and `--ticker` made the false negative
reachable.

**The replacement had TWO further problems, both found by driving it rather
than by reading it, and both fixed:**

1. **A DEAD CLAUSE.** `scale_pre < 3.0 and scale_post < 3.0` -- but
   `scale = target_vol/vol` is decreasing in vol, and the guard two lines above
   already establishes `a["vol_ann"] < b["vol_ann"]`, so `scale_pre > scale_post`
   ALWAYS and the first clause IMPLIES the second. Measured: `a=0.04` fires the
   guard; every attempt to saturate the POST side alone either saturates the pre
   side too (`a=0.045 b=0.05`) or trips `pre_fix_volatility_is_the_lower_one`
   first (`a=0.20 b=0.05`). **No fixture can falsify clause two**, so it was
   decoration inside an `and`. Replaced with `max(scale_pre, scale_post) < 3.0`
   -- identical semantics, one clause, no unfalsifiable half.

2. **UNREACHABLE OFFLINE.** The guard lived inside `gate_effect()`, which needs
   BigQuery-derived volatilities -- so `--offline` never reached it and the
   matrix could not exercise it. **That is precisely how the vacuous version
   shipped**, and `check_mechanism_tripwires()`'s own docstring already names the
   failure: *"a guard that can only be reached through a network call is a guard
   that will not be mutation-tested."*

   `gate_effect()` takes a plain dict, so the tripwires now drive **the real
   function** with synthetic volatilities -- not an extracted helper, because a
   guard tested through a helper is not tested at the seam that uses it. That
   needed a recursion break (`gate_effect` calls the tripwires, which now call
   it), observed first as a `RecursionError` while writing it and recorded in
   the code comment.

**Paired negative AND positive**, because "it raised" is not "it
discriminated":

- `gate_guard_rejects_saturating_inputs` -- the cycle-3 Q/A's exact attack
  (`a=0.04`, the input that made the script report 1.2500x while the TRUE
  inflation was 1.0000x) must RAISE.
- `gate_guard_accepts_unsaturated_inputs` -- an ordinary input (scales 0.75 /
  0.60) must NOT raise, so a guard that fires on everything is caught too.

Both red states watched before shipping, each on the correct named guard:

```
guard forced always-TRUE  -> INVARIANT FAILED: gate_guard_rejects_saturating_inputs
guard forced always-FALSE -> INVARIANT FAILED: gate_guard_accepts_unsaturated_inputs
```

Offline invariants **19 -> 29**; full run **43 invariants hold**. Cells
**T4/T5/T6** added (always-true, always-false, and the fixture neutered).

### V2 -- the algebraic-identity overclaim

Already corrected in the shipped code: the guard is renamed
`census_sql_is_internally_consistent` and its comment states plainly that over
the grouping `SUM(n-1) == total_rows - keys` is an ALGEBRAIC IDENTITY which
**cannot** detect a normalisation error. It is kept only because it would catch a
malformed edit to `CENSUS_SQL`, and the name no longer overclaims.

### V3 -- "no restart pending" was false for the in-process API path

**Now true, and measured rather than asserted.** `cache.py` landed
2026-08-18T07:06:16+02:00; the running backend is **pid 89340, started
08.26.53** -- after the fix -- so the module the live process imported carries
`_dedupe_index`. Verified from the RUNNING process, not from the file:

```
$ pgrep -f "uvicorn backend.main:app"  -> 89340
$ ps -o pid,lstart,etime -p 89340      -> 89340  tir. 18 aug. 08.26.53 2026  05:14:10
$ git log -1 --format=%cI 539f16eb -- backend/backtest/cache.py
                                       -> 2026-08-18T07:06:16+02:00
```

### Post-fix state, regenerated from one run

```
control -> rc=0  collected=13  GREEN
tripwire control (verify_86_116.py --offline) -> rc=0 GREEN
KILLED 14 / 14   SURVIVED 0   UNSCORABLE 0   EQUIVALENT-BY-DESIGN 1 (not scored)
restore verified: cache.py 9f5f1d6798833281... verify_86_116.py 9fcf8806b56755ef...
```

`python scripts/qa/verify_86_116.py` -> **OK: all 43 invariants hold**, and the
report now PRINTS the saturation state rather than leaving it to inference:
`vol_scale pre / post: 0.4487 / 0.3587 (cap 3.0 -- UNSATURATED, so the inflation
above is the real one)`.

Pre-spawn class sweep (`scripts/qa/pre_spawn_gate.py 86.116`): **CLEAN** -- no
spliced capture, no guard asserting a truthy literal, no prose contradicting its
own captures, no SURVIVED or UNSCORABLE cell.
