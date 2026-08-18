STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.74
WRITTEN: 2026-08-14T15:38:03Z
COMPLETED: 2026-08-14T15:52:00Z

Scope: COMMIT 76ac89ee ONLY. Not a re-grade of my CONDITIONAL on 9d14291e+a541f10c.
All mutations done by sys.modules injection -- tree never written. Post-run md5s
confirm portfolio_manager.py and the test file unchanged.

## Q1 -- is the defect real? RE-DERIVED, YES
In-memory mutant = the shipped source with the new floor block excised
(src[:i]+src[j:]), driven through the REAL `_compute_swap_candidates`:

  CONTROL (as shipped)   0% REJECT -> []
  CONTROL                3% legit  -> [('SELL','OLD',None), ('BUY','NEW',300.0)]
  CONTROL                absent    -> [('SELL','OLD',None), ('BUY','NEW',1000.0)]
  MUTANT (floor removed) 0% REJECT -> [('SELL','OLD',None), ('BUY','NEW',0.0)]
  MUTANT                 3% legit  -> [('SELL','OLD',None), ('BUY','NEW',300.0)]
  MUTANT                 absent    -> [('SELL','OLD',None), ('BUY','NEW',1000.0)]

Exactly as described: a real SELL of the displaced holding paired with a $0.00
no-op BUY, net -1 position, from a 0% REJECT. Not taken on Main's word.

## Q2 -- tightening only? YES
Suppression set is exactly {nav*pct/100 < 50}. 3% of $10k = $300 fires
unchanged; ABSENT -> 10% = $1000 fires unchanged. PARITY, not a new restriction:
`decide_trades` already carries an identical `if buy_amount < 50` at
portfolio_manager.py:536, so the swap path now matches the main buy path.
ATOMIC path unchanged: the pre-existing floor at :932 runs AFTER
`min(buy_amount, available_cash+freed)`, and min() only lowers, so the new
early check at :919 can never reject anything the later one would have passed.
Found no swap that should fire and is now suppressed.

## Q3 -- C2 on the swap path? YES, behaviourally
Primary mutation cell (floor removed entirely) KILLS
test_swap_path_zero_pct_verdict_emits_no_buy (1 failed, 2 passed). The guard is
genuine on the BUY half -- not a source scan.

## Q4 -- regressions? NONE NEW, independently verified
Target file: 40 passed. The four adjacent failures re-run against the PARENT
module a541f10c injected into sys.modules (parenthood asserted by
`'cycle-3' not in parent`): the SAME four fail. Mechanism also unrelated --
test_swap_framework_fills_zero_buy_gap fails on "Expected 2 swap SELLs, got 1"
with its swap BUYs at $1000.00, twenty times the $50 floor. Lint F821/F401/F811
over `git diff --name-only 76ac89ee^ 76ac89ee -- '*.py'` (2 files, non-empty
asserted): clean, exit 0.

## Q5 -- over-configured? NO -- and the stated REASON is wrong
The four explicit values match the PRODUCTION Settings defaults exactly:
paper_swap_enabled True (settings.py:368), paper_swap_min_delta_pct 25.0 (:372),
paper_swap_max_per_cycle 2 (:378), paper_atomic_swap_enabled False (:493). The
scenario is production-representative; this is STRONGER evidence of live
reachability than Main's own framing.
BUT: "paper_swap_max_per_cycle defaults to 0 and short-circuits the whole
function" is false about production. The 0 is the getattr FALLBACK at
portfolio_manager.py:719 for an attribute ABSENT from the test's SimpleNamespace
stub (`_settings()` at test:18-32 never sets it). The claim appears in the `_run`
docstring AND the commit message as a production default. A future reader could
conclude the swap path is dark by default; it is LIVE by default. Wrong
provenance on a correct configuration, on a money path.

## FINDING THAT CAPS -- the orphan SELL has NO guard
The commit's own claim is "emitting neither leg, so the SELL cannot orphan".
That property is not asserted anywhere. Mutant: remove the early floor and
suppress ONLY the BUY append (`if buy_amount < 50: continue` inserted between the
SELL append and the BUY append) -- i.e. the exact net -1 harm.
RESULT: 3 passed, 37 deselected, rc 0. SURVIVES.
`test_swap_path_zero_pct_verdict_emits_no_buy` filters `o.action == "BUY"`, so
the half that causes the loss is unmeasured. Named one-line fix: assert the whole
returned list is empty (`assert self._run(0.0, SIZE) == []`), not just its BUY
subset. Shipped code is correct -- this is absent coverage of the named harm, not
a wrong fix, which is why it caps rather than fails.

VERDICT COMPUTED: CONDITIONAL.
