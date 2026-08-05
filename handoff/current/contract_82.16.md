# Contract -- masterplan step 82.16

**Step id:** 82.16 (phase-82, P1, harness_required: true) | **Date:** 2026-08-05 | **Cycle:** 1

---

## 1. Research gate summary

**Brief:** `handoff/current/research_brief_82.16.md` | **Envelope:** `gate_passed: true`
-- 9 sources read in full, 33 snippet-only, 42 URLs, recency scan performed, 13 internal
files inspected, tier `complex`.

### Measured findings that shape the design

**(a) The defect is CONFIRMED, and Main verified the headline instance directly.**
`_compute_quality_momentum_label` (`backtest_engine.py:1190`) calls
`build_feature_vector(ticker, entry_date)` and classifies on `momentum_6m` and
`quality_score` -- both entry-date features, both in `_NUMERIC_FEATURES`. No forward
price is fetched. `_compute_factor_label` (`:1274`) is the same shape. The other five
methods do read post-entry data.

**(b) Measured on the committed 82.2 fixture (880 rows):**

| Strategy | labelled | changed under post-entry mutation |
|---|---|---|
| `quality_momentum` | 880 | **0** |
| `factor_model` | **0** | 0 |
| `mean_reversion` | -- | 119 |
| `meta_label` (aliases triple_barrier) | 375 | passes |

**(c) THE TRAP THAT WOULD MAKE THIS STEP WORTHLESS.** `factor_model`'s `0 changed` is
**VACUOUS** -- it returns `None` everywhere on that fixture (which omits
`momentum_12m`), so "no label changed" is indistinguishable from "no label existed". A
mutation test alone would report it identically to a genuinely forward-looking strategy
that happened to produce no rows. **The test must therefore assert per-strategy non-None
COVERAGE as a precondition**, or criterion 1 passes for free on any strategy that
silently returns None.

**(d) `mean_reversion` changes on only 119 of 880 rows**, which is why criterion 1's "at
least one fixture row" wording is correct and must not be tightened to "most rows".

**(e) `meta_label` aliases `_compute_triple_barrier_label` and passes on its own** --
no special-casing needed, and none will be added.

**(f) This is CIRCULAR ANALYSIS (Kriegeskorte et al. 2009), not look-ahead bias.** The
mirror image: the label contains no future content at all, so the model learns a
deterministic function of its own inputs. **A shuffled-label permutation test would PASS
this defect** -- worth stating because that is the reflex check and it does not work here.

**(g) Bailey & Lopez de Prado: a non-comparable candidate must be REMOVED from the trial
pool, not left in with a caveat.** That settles fix-vs-remove (§2).

**(h) Trap inventory:** an empty parametrize collects 0 tests and reports green (exactly
why criterion 3 exists); `_engine()` in the 82.2 fixture lacks `tp_pct`/`sl_pct` and
raises `AttributeError` on 2 of 8 strategies; removing a name without patching
`quant_optimizer.py:68 AVAILABLE_STRATEGIES` leaves the optimizer logging a strategy that
never ran; step 82.17 will make `archetype_library` raise at import.

---

## 2. Hypothesis, and the fix-vs-remove decision

The criteria permit either making the two labels forward-looking or removing them.
**This step REMOVES them**, for reasons that are measured rather than preferential:

1. **Making them forward-looking is not a fix, it is a new strategy.** A label with new
   semantics needs its own research validation; shipping it under the same name would
   silently change what every historical `quality_momentum` row in `quant_results.tsv`
   means.
2. **`quality_momentum` would remain untrainable anyway.** It depends on `quality_score`,
   and step 82.21 measured that `historical_fundamentals` has **zero rows before
   2024-06-30** -- ~81% of the standard 2018-2025 window. A forward-looking rewrite would
   still have nothing to train on for four fifths of the sample.
3. **The literature is explicit** (§1g): remove the non-comparable candidate from the
   pool.

Removal is to a **named, documented `NON_COMPARABLE_STRATEGIES` mapping** -- the methods
and their history are preserved and the reason is recorded in code, so this is a
demotion out of the selection pool, not a deletion.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. `a test asserts, for EVERY name in STRATEGY_REGISTRY, that its label method consults data strictly after entry_date -- implemented by invoking the method with a price series whose post-entry values are mutated and asserting the returned label changes for at least one fixture row`
2. `any strategy whose label does not change under post-entry price mutation is either fixed to be forward-looking or removed from STRATEGY_REGISTRY, and the test enumerates the registry rather than a hand-written list so a newly added strategy cannot skip the check`
3. `the enumeration is asserted non-empty, so the guard cannot pass vacuously on an empty registry`
4. `a fixture confirms the guard can fail: a deliberately non-forward stub label registered in a temporary copy of the registry is detected`

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_16_label_forward_information.py -q`

---

## 4. Plan

### 4.1 `backend/backtest/backtest_engine.py`

- Move `quality_momentum` and `factor_model` out of `STRATEGY_REGISTRY` into
  `NON_COMPARABLE_STRATEGIES`, with an in-code explanation of *why* (circular analysis,
  measured 0/880 label change, the 82.21 fundamentals interaction).
- The two methods themselves are **kept**, so nothing that references them by name breaks
  and the demotion is reversible if either is ever given a real forward label.

### 4.2 `backend/backtest/quant_optimizer.py`

Derive `AVAILABLE_STRATEGIES` from `STRATEGY_REGISTRY` instead of restating the names, so
the optimizer cannot select a demoted strategy. Per §1h, leaving the hardcoded list would
have the optimizer log a strategy that never ran -- the exact silent-wrongness this step
is about. `"blend"` is not a registry key and is preserved explicitly.

### 4.3 Tests -- `backend/tests/test_phase_82_16_label_forward_information.py`

- **C1+C2:** parametrized over `STRATEGY_REGISTRY.keys()` (the registry itself, not a
  list), mutating post-entry closes and asserting the label set changes on >= 1 row.
- **THE COVERAGE PRECONDITION (§1c):** each strategy must also produce >= 1 non-None
  label on the fixture. Without this, a strategy that returns None everywhere passes
  criterion 1 for free -- which is precisely how `factor_model` measured 0/880.
- **C3:** assert `STRATEGY_REGISTRY` is non-empty *and* that the parametrization is
  non-empty, because an empty parametrize collects zero tests and reports green.
- **C4:** register a deliberately non-forward stub in a **temporary copy** of the
  registry and assert the guard flags it.
- **Demotion guards:** assert the two names are absent from `STRATEGY_REGISTRY`, present
  in `NON_COMPARABLE_STRATEGIES` with a recorded reason, and unreachable from
  `AVAILABLE_STRATEGIES`.
- Fixture reuse from `test_phase_82_2_candidate_strategies.py`, with the `tp_pct`/`sl_pct`
  gap in `_engine()` fixed (§1h) rather than worked around.

### 4.4 Out of scope

No new label semantics; no change to the five forward-looking strategies; no change to
`optimizer_best.json` or historical `quant_results.tsv` rows (their meaning is preserved
precisely BY not redefining the names); 82.17 owns `IMPLEMENTED_STRATEGY_IDS`.

---

## 5. Files expected to change

`backend/backtest/backtest_engine.py`, `backend/backtest/quant_optimizer.py`,
`backend/tests/test_phase_82_16_label_forward_information.py` (NEW),
`.claude/masterplan.json`.

---

## 6. References

`handoff/current/research_brief_82.16.md`; Lopez de Prado, *Advances in Financial Machine
Learning* Ch.3 (triple-barrier, meta-labelling); Kriegeskorte et al. 2009 (circular
analysis); Bailey & Lopez de Prado (trial-pool composition and the deflated Sharpe).
