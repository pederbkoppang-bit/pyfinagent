# ESCALATION -- step 86.120 -- THIRD CONSECUTIVE CONDITIONAL -- OPERATOR DECISION

**Sequence:** `[CONDITIONAL, CONDITIONAL, CONDITIONAL]`
(`wf_a285260f-0dd`, `wf_e985d5f3-94f`, `wf_ea3f6587-cc2`)
**Attempts used:** 3 of 5 -- **the budget is NOT the binding constraint.**
**Binding constraint:** CLAUDE.md F1 3rd-CONDITIONAL rule. The next Q/A pass
**must return FAIL regardless of evidence.**

## Why I parked instead of spawning a fourth

A fourth spawn cannot return PASS. Spending an attempt to obtain a verdict the
rule has already determined would burn tokens for no information. The standing
instruction (and this project's own recorded precedent -- 86.108, 86.110,
86.116, 86.59 all parked on this exact rule) is to park rather than iterate,
so the step is `pending` and the fix is committed.

**This is the fifth step to hit this rule this session.** The distinction
86.59's own escalation drew between itself and 86.108/86.110 applies here, and
86.120 sits on the 86.108/86.110 side of that line:

> 86.59 found a **real, blocking, reproducible defect** on every one of its
> three cycles. 86.108/86.110 parked with **every criterion MET**.

86.120 is the 86.108/86.110 shape. **Zero product defects were found across all
three cycles, by three independent evaluators, using progressively deeper
mutation searches (11 cells -> 15 cells -> 13 NEW cells each time).** The
production file's sha256 (`76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1`)
is byte-identical across every cycle -- not one line of shipped code changed
since Cycle 1. Every finding was a test-coverage gap on a genuinely new branch
no prior cycle had examined, each closed with a small (~5-30 line), additive,
mutation-verified test.

## What each cycle actually found -- a widening search, not a repeated miss

| cycle | new guard found uncovered | proof | fix size |
|---|---|---|---|
| 1 | 3 guards: the production call that persists a cooldown (`cooldown_record_hit`), the call that clears it on success (`cooldown_clear_on_success`), and the tz-fallback bug fixed during GENERATE | deleting each survived 27/27 -- every existing test pre-seeded state directly instead of driving a real failure through the production entry point | 3 new tests, ~90 lines |
| 2 | 1 guard: `cooldown_status()`'s except-branch safety claim ("a corrupt record fails toward SAFE") | inverting `active = True` to `active = False` survived 30/30 -- Cycle 1's own code review had RELIED on this exact property to wave through 7 broad `except` blocks, untested | 1 new test, ~20 lines |
| 3 | 1 guard: `classify_limit_failure`'s JSON `result`-field extraction | deleting it survived 31/31, AND created a measured false-positive risk (a successful call whose result merely mentions limit-shaped text elsewhere would engage a real multi-hour cooldown) | 1 new test, ~30 lines |

No cycle repeated a prior cycle's finding. No cycle found a product bug. Each
mutation matrix was run independently, in-memory, against the real file, with
its own green control first -- the three evaluators did not share matrices.

## State at the park -- the residual IS fixed, it is just unevaluated

The Cycle-3 blocker is closed and verified by execution, the same way Cycles 1
and 2 were:

- `test_classify_extracts_the_result_field_not_the_raw_envelope` added,
  asserting both consequences Q/A measured (the operator-facing message
  degrading to a raw JSON blob, and the false-positive misclassification);
- mutation-verified against the real file: green control (32/32), the exact
  deletion Q/A specified applied, the one intended test going red (1
  failed/31 passed) with **the failure output itself reproducing the
  false-positive** (`kind='session'` derived from raw JSON text), then a
  `diff`-verified byte-identical sha256 restore;
- full transcript and the two smaller NOTE-level prose fixes from Cycle 2
  (a misleading test comment, a mis-attributed mutation-provenance claim) are
  recorded in `experiment_results_86.120.md` and `live_check_86.120.md`.

**No Q/A has seen this fix.** A green local re-run is a self-check, not a
verdict -- exactly the distinction this project's own harness protocol exists
to enforce.

## The product, which survives the park

Independently re-derived by three separate evaluators, not just asserted: the
CLI's three documented limit messages (session/weekly/Opus) are correctly
classified from the untruncated envelope; a classified hit persists a
cooldown to disk that survives both `rail_guard_reset()` and a simulated
restart; the rail is skipped before any subprocess spawn across multiple
cycles; the cooldown self-clears on a real success and bounds itself via a
new Settings field; the existing phase-66.1 generic breaker is unchanged for
non-limit failures; at least one always-on signal agent (pead_signal's exact
`make_client()` call shape) respects the cooldown; and `make_client()`'s
existing $0-metered routing-breach guard is unweakened -- reproduced live by
the Cycle-3 evaluator by driving the real `llm_client.py:2198` fallthrough.

**Two things filed rather than absorbed, discovered while investigating this
step's neighborhood, not by it:**

1. **Step 86.121** (P2) -- the nightly autoresearch's Semantic Scholar
   retriever is rate-limited every night, invisible to the existing
   arXiv-scoped health check (86.80).
2. **Step 86.122** (P1) -- `debate.py`/`risk_debate.py`'s Moderator/Risk-Judge
   calls lack the phase-61.2 retry-on-empty-cc-rail-response guard that
   `orchestrator.py`'s Critic path already has, confirmed live-firing on a
   real DELL report (silent HOLD/0.5 fallback, zero retry, zero degraded
   flag).

## What the operator can decide

1. **Authorise one attempt** knowing it returns FAIL by rule, purely to reset
   the counter, then a fifth to grade the Cycle-3 fix on its merits. Costs 2
   of the 2 remaining attempts.
2. **Accept on the evaluators' own findings** that the product is correct on
   every cycle, nothing has shipped to production, and the residual is a
   test-only addition -- and flip. Main cannot do this without a PASS, which
   is why the step sits at `pending`.
3. **Leave parked.** The fix is committed, mutation-verified, and
   re-runnable; the cooldown mechanism does not need to be live this instant
   to be correct -- the analyst simply keeps using today's per-cycle generic
   breaker (phase-66.1, unmodified) until this closes.

**Nothing shipped, so a park costs the answer, not the engine.** Zero
production files changed across all three cycles -- only tests and prose.
