# Contract — Step 75.5.12 (P1: fetch_llm_spend misses the BARE `cc_rail` agent shape)

Date: 2026-07-25 | Cycle: 159 | Executor: Main (Opus 5) | Research gate: **PASSED**

## Research-gate summary

`handoff/current/research_brief_75.5.12.md` — tier `simple`, envelope
`{"external_sources_read_in_full":8,"urls_collected":22,"recency_scan_performed":true,`
`"internal_files_inspected":9,"gate_passed":true}`. 3-variant search discipline visible;
recency scan confirms **no change to BigQuery `LIKE`/`NOT LIKE`/NULL semantics in the
2024–2026 window**, so the canonical docs remain current.

Four load-bearing findings, each of which changes what GENERATE does:

1. **Use the De-Morgan'd form, NOT the predicate the step text suggests.**
   The step proposes `NOT (agent = 'cc_rail' OR agent LIKE 'cc_rail:%')`. That is
   logically identical but **drops the literal substring `NOT LIKE 'cc_rail:%'`**,
   which the test's `FakeBQClient` keys on at
   `test_phase_75_5_1_spend_metric.py:85` (`if "NOT LIKE 'cc_rail:%'" in sql ...`).
   Adopting the step's wording would **silently neuter the existing shape-2 guard** —
   the test would keep passing while no longer testing anything. Required form:
   ```sql
   AND (agent IS NULL
        OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
   ```

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

2. **The true shape-set is exactly THREE, derived from code rather than from the step
   text** (criterion 2 says "all shapes actually produced", so this had to be derived):
   (a) `provider='claude-code'` with an arbitrary agent (`autonomous_loop.py:2299`);
   (b) `cc_rail:<role>` and (c) bare `cc_rail`, both from the ternary at
   `claude_code_client.py:504`. No fourth shape exists.
   **Why (c) dominates**: `orchestrator.py:826-835` sets only `_ticker`, never `_role`
   — `"_role"` is written in exactly two places repo-wide
   (`autonomous_loop.py:2722`, `:2762`) — so every Layer-1 pipeline rail call takes
   the else-branch.
3. **The fixture CAN genuinely fail** (this is the anti-pattern the project keeps
   hitting): `FakeBQClient` (:58-103) evaluates predicates **parsed from the production
   SQL text**, not a mocked return value and not a source scan. A bare-`cc_rail` row
   passes all three of today's predicates, so the new assertion is red pre-fix. The
   fake must learn the new clause **keyed on the SQL string**, or mutation M3 becomes
   undetectable.
4. **M4 warning**: the fake never evaluates `agent IS NULL`, so deleting the
   `agent IS NULL OR` guard may NOT turn `test_agent_none_rows_are_included` red.
   This must be **reproduced, not assumed**, before claiming the NULL guard is covered.

## Hypothesis

Widening the exclusion to the bare shape stops ~2,549 flat-fee rail calls (measured
below) from being priced at full API rates the moment the operator flips
`cost_budget_use_llm_spend_enabled` — the exact phantom-trip failure mode 75.5.1 exists
to prevent, hiding inside its own discriminator.

## Measurement (Main, 2026-07-25, 30d `llm_call_log`)

```
anthropic  claude-sonnet-4-6  cc_rail              2192 calls  4,370,458 tok
anthropic  claude-opus-4-7    cc_rail               357 calls    500,651 tok
anthropic  claude-sonnet-4-6  cc_rail:drill_66_1      7 calls          0 tok
```
The bare shape outnumbers the colon shape ~364:1. (The step text's 2,241/4.1M is the
2026-07-24 window; the window has slid — the numbers above are this cycle's own
measurement, reported as such.)

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. fetch_llm_spend excludes bare agent='cc_rail' rows exactly (not a prefix-wildcard
   that could swallow unrelated agents), proven by a new test row fixture
2. The docstring inventory names all shapes actually produced (bare 'cc_rail',
   'cc_rail:<agent>', provider='claude-code')
3. MUTATION: revert the clause -> the bare-shape fixture test goes red; existing
   both-shapes tests stay green

Immutable verification command:
```
.venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py -q
```

## Plan

1. `spend.py:218` → the De-Morgan'd predicate above (preserves the substring the
   existing guard keys on; satisfies criterion 1's "not a prefix-wildcard" by using
   exact `!=` for the bare shape, so `cc_railway` is NOT swallowed).
2. Docstring invariant 1 → name all THREE shapes.
3. Teach `FakeBQClient` the new clause, keyed on the SQL string
   (`if "agent != 'cc_rail'" in sql and r["agent"] == "cc_rail": continue`).
4. New fixture rows: bare `cc_rail` (must be excluded) **and** `cc_railway` (must be
   INCLUDED — this is the over-match guard criterion 1 explicitly demands).
5. **Mutation matrix**:
   - M1 revert the bare clause → bare-shape test RED, existing both-shapes tests GREEN.
   - M2 replace the exact `!=` with a prefix wildcard (`NOT LIKE 'cc_rail%'`) → the
     `cc_railway` over-match test RED (proves criterion 1's "not a prefix-wildcard").
   - M3 **FIXTURE/STUB**: neuter the fake's new clause → the bare-shape test must go
     RED (proves the fixture is load-bearing, not self-satisfying).
   - M4 delete the `agent IS NULL OR` guard → **reproduce first**; if the NULL test
     does NOT go red, say so plainly and record the gap rather than claiming coverage.
6. Q/A on changed evidence → `harness_log.md` append → masterplan flip (log LAST).

## Boundaries

- `backend/services/observability/spend.py` + `backend/tests/test_phase_75_5_1_spend_metric.py` only.
- **No flag is flipped.** `cost_budget_use_llm_spend_enabled` stays at its default; this
  step makes the metric correct so the operator *can* flip it later.
- Two adjacent seams found by the gate are **out of boundary and deliberately not
  touched**, to be queued rather than bundled:
  `scripts/away_ops/metered_spend.py:69` (`startswith("cc_rail")` — catches the bare
  shape already but carries the `cc_railway` over-match) and
  `scripts/diagnostics/funnel_report.py:96` (`LIKE 'cc_rail%'`, opposite polarity,
  diagnostic-only).
- The gate's `_`-is-a-wildcard finding (`'ccXrail:synthesis' LIKE 'cc_rail:%'` → True)
  has **zero production impact** (no such agent names exist, verified against the 30d
  shape census) and is queued as **75.5.13**, together with the two adjacent seams
  above -- that step also re-decides the STARTS_WITH/wildcard-free question, whose
  original rejection rationale was refuted by execution.

## References

- `handoff/current/research_brief_75.5.12.md` (8 sources read in full, 22 URLs)
- `backend/agents/claude_code_client.py:504` (the ternary producing shapes b/c)
- `backend/agents/orchestrator.py:826-835` (why the bare shape dominates)
- `backend/tests/test_phase_75_5_1_spend_metric.py:58-103` (the SQL-text-parsing fake)
- `feedback_mutation_test_guards_and_fixtures` — mutate the STUB too.
