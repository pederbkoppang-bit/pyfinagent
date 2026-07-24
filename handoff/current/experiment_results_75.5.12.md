# Experiment results — Step 75.5.12 (P1: bare `cc_rail` spend-discriminator gap)

Date: 2026-07-25 | Cycle: 159 | Execution: Main (Opus 5) GENERATE

## What was changed (2 files, both inside the stated boundary)

### 1. `backend/services/observability/spend.py` — the predicate (:218)

```diff
-              AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')
+              AND (agent IS NULL
+                   OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
```

**Why this form and not the one the step text suggests.** The step proposes
`NOT (agent = 'cc_rail' OR agent LIKE 'cc_rail:%')`. It is logically identical, but it
drops the literal substring `NOT LIKE 'cc_rail:%'` that the test's `FakeBQClient` keys
on at `test_phase_75_5_1_spend_metric.py:85`. Adopting it would have **silently
neutered the existing shape-2 guard**: the suite would have stayed green while
testing strictly less. The research gate caught this; GENERATE followed the gate rather
than the step text, and the criteria are untouched either way.

Exact `!=` rather than a `cc_rail%` prefix is deliberate and is what criterion 1's
"not a prefix-wildcard that could swallow unrelated agents" asks for — a prefix would
also exclude a future metered agent named `cc_railway`, under-counting real spend
(a false-negative on the budget breaker, the more dangerous direction).

> **CORRECTION (Q/A cycle-1, 2026-07-25) — the justification above was WRONG; the
> decision was right for a different and stronger reason.**
>
> I claimed adopting the step text's form "would silently neuter the existing shape-2
> guard — the test would keep passing while no longer testing anything." **That is
> refuted by execution.** I asserted a behavioral counterfactual without running it;
> the Q/A ran it, and so did I afterwards:
>
> ```
> === MSTEP: the masterplan step text's OWN suggested predicate ===
>   2 failed, 11 passed
>   reds: ['test_cc_rail_rows_contribute_zero_both_shapes',
>          'test_bare_cc_rail_shape_contributes_zero']
> ```
>
> The *mechanism* half was right — the substring `NOT LIKE 'cc_rail:%'` that the fake
> keys on does drop out — but the *consequence* is inverted: with that branch not
> firing, the rail row is INCLUDED in the aggregate, so the assertion fails **loudly**.
> Nothing is silently neutered.
>
> **The actual reason to decline the step text's form** (supplied by the Q/A, then
> verified here): taken literally, without re-adding an `agent IS NULL` guard, it
> **drops every NULL-agent row** via SQL three-valued logic. ANSI oracle:
>
> ```
> STEP TEXT: NOT (agent='cc_rail' OR agent LIKE 'cc_rail:%')   kept: ['cc_railway', 'synthesis']
> MINE:      (agent IS NULL OR (agent!='cc_rail' AND ...))     kept: [None, 'cc_railway', 'synthesis']
> ```
>
> And NULL is the **common** metered case, not an edge case: `llm_client.py:1127` logs
> `agent=config.get("_role")`, and `"_role"` is set in exactly two places repo-wide
> (`autonomous_loop.py:2722`, `:2762`). Measured blast radius from this cycle's own 30d
> query: **226 Gemini calls / 232,090 tokens** plus 3 haiku calls carry a NULL agent and
> would have been silently dropped from metered spend — an **under-count**, i.e. the
> breaker opens LATE. That is the more dangerous direction, and it is why the
> De-Morgan'd form is strictly safer.
>
> Recorded rather than quietly rewritten: the original claim is the kind of unverified
> assertion this project's `feedback_measure_dont_assert_claims` memory exists to catch,
> and it was load-bearing — it was the sole stated basis for deviating from an explicit
> instruction on a P1 money step.


### 2. `backend/services/observability/spend.py` — docstring invariant 1

Now enumerates all **three** shapes the rail actually produces, each with its writer
anchor, plus why the bare shape dominates and why the exclusion uses exact equality.
The shape-set was **derived** by enumerating every `log_llm_call(` writer in `backend/`
and `scripts/` (criterion 2 says "all shapes actually produced", so it could not be
copied from the step text): there is no fourth shape.

### 3. `backend/tests/test_phase_75_5_1_spend_metric.py` — 2 new tests + fake upgrades

- `test_bare_cc_rail_shape_contributes_zero` — shape 3 fixture; red before the fix.
- `test_cc_railway_is_not_swallowed_by_the_bare_cc_rail_exclusion` — the over-match
  guard criterion 1 demands.
- `FakeBQClient` learned three new predicate branches, each keyed on the SQL text the
  way every existing branch is: the bare-shape exclusion, a prefix-form branch (so M2
  is *detectable* rather than merely unrecognized), and SQL three-valued-logic for
  NULL agents (so M4 is detectable — see below).

The existing `test_cc_rail_rows_contribute_zero_both_shapes` was left **untouched on
purpose**, so criterion 3's "existing both-shapes tests stay green" remains an
independent signal rather than something I edited into agreement.

## Verification (verbatim)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py -q
13 passed in 0.91s
exit=0
```
11 pre-existing + 2 new.

## Criteria status

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Excludes bare `cc_rail` exactly, not a prefix wildcard, proven by a new fixture | MET | live_check §3, §5: new bare-shape fixture + M1; the `cc_railway` fixture + M2 prove the "not a prefix-wildcard" half |
| 2 | Docstring names all shapes actually produced | MET | live_check §4 — three shapes, derived from the writers, with anchors |
| 3 | MUTATION: revert → bare-shape test red; existing both-shapes tests stay green | MET | live_check §5 M1: exactly one red, the both-shapes test green |

## The one thing that did NOT work first time (recorded, not smoothed over)

**M4 initially did not kill.** The research gate predicted the fake never evaluates
`agent IS NULL`; the first matrix run confirmed it — deleting the `agent IS NULL OR`
guard left all 13 tests green. That guard was therefore untested, and a guard that
cannot fail does not count. Rather than report it as covered or defer it, the fake now
models three-valued logic (with a NULL agent, `agent != 'cc_rail'` evaluates to NULL,
not TRUE, so BigQuery would DROP those metered rows and under-count spend). M4 now
kills. Both the before and after states are in live_check §5.

## Boundaries held

- Only `spend.py` + its test file changed. **No flag flipped** —
  `cost_budget_use_llm_spend_enabled` stays at its default; this step makes the metric
  correct so the operator can flip it safely later.
- Two adjacent seams the gate found are **deliberately not bundled**, to be queued
  rather than silently fixed: `scripts/away_ops/metered_spend.py:69`
  (`startswith("cc_rail")` — already catches the bare shape but carries the
  `cc_railway` over-match) and `scripts/diagnostics/funnel_report.py:96`
  (`LIKE 'cc_rail%'`, opposite polarity, diagnostic-only).
- The gate's finding that `_` is itself a LIKE wildcard — so `'ccXrail:synthesis' LIKE
  'cc_rail:%'` is True — has **zero production impact** (no such agent names exist,
  verified against the 30d shape census) and is likewise left for a queued P3 rather
  than bundled here.
