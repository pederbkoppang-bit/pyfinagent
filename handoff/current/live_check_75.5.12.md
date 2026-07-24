# live_check 75.5.12 — verbatim evidence (2026-07-25)

## 1. The measurement (30d `llm_call_log`, this cycle's own window)

```sql
SELECT provider, model, agent, COUNT(*) AS calls, SUM(input_tok + output_tok) AS tokens
FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY provider, model, agent ORDER BY calls DESC
```

```
anthropic  claude-sonnet-4-6  cc_rail              2192 calls   4,370,458 tok
anthropic  claude-opus-4-7    cc_rail               357 calls     500,651 tok
anthropic  claude-sonnet-4-6  cc_rail:drill_66_1      7 calls           0 tok
```

**2,549 bare-`cc_rail` calls / ~4.87M flat-fee tokens** would have been priced at full
API rates on the next flag flip, versus **7** rows in the colon shape the old predicate
actually caught — a ~364:1 ratio. The step text's 2,241/4.1M figure is the 2026-07-24
window; the window has slid and the numbers above are this cycle's measurement,
reported as its own.

## 2. The defect, before

`backend/services/observability/spend.py:218` read:

```sql
AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')
```

The colon is **required** by that pattern, so `agent = 'cc_rail'` does not match it and
those rows passed the metered-only filter.

Root cause of the bare shape's dominance (derived from the writers, not assumed):
`backend/agents/claude_code_client.py:504` tags `agent=f'cc_rail:{agent}' if agent else
'cc_rail'`, and `backend/agents/orchestrator.py:826-835` sets only `_ticker`, never
`_role` — `"_role"` is written in exactly two places repo-wide
(`autonomous_loop.py:2722`, `:2762`). So **every Layer-1 pipeline rail call takes the
else-branch.**

## 3. The fix

```sql
AND (agent IS NULL
     OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
```

De-Morgan'd form, chosen over the step text's suggested
`NOT (agent = 'cc_rail' OR agent LIKE 'cc_rail:%')` for a concrete reason: the
suggested form is logically identical but **drops the literal substring
`NOT LIKE 'cc_rail:%'`**, which the test's `FakeBQClient` keys on at
`test_phase_75_5_1_spend_metric.py:85`. Adopting it would have silently neutered the
existing shape-2 guard — the suite would stay green while testing less.

Exact `!=` (not a `cc_rail%` prefix) satisfies criterion 1's "not a prefix-wildcard
that could swallow unrelated agents".

## 4. Docstring — THREE shapes, as actually produced

Invariant 1 now enumerates all three with their writer anchors: (a)
`provider='claude-code'` (`autonomous_loop.py:2299`), (b) `cc_rail:<role>` and (c) bare
`cc_rail` (both `claude_code_client.py:504`), plus why (c) dominates and why the
exclusion uses exact equality. Derived by enumerating every `log_llm_call(` writer in
`backend/` and `scripts/` — **there is no fourth shape.**

## 5. Mutation matrix — all four kill (criterion 3)

| # | Mutation | Target | Result |
|---|----------|--------|--------|
| M1 | revert to the pre-75.5.12 predicate | bare-shape test | `1 failed, 12 passed` → `test_bare_cc_rail_shape_contributes_zero` **RED**; the existing both-shapes test stayed **GREEN** (criterion 3 exactly) |
| M2 | exact `!=` → `NOT LIKE 'cc_rail%'` prefix | over-match test | `1 failed, 12 passed` → `test_cc_railway_is_not_swallowed_by_the_bare_cc_rail_exclusion` **RED** |
| M3 | **STUB**: neuter the fake's own bare-shape clause | bare-shape test | `1 failed, 12 passed` → **RED** (the fixture is load-bearing, not self-satisfying) |
| M4 | delete the `agent IS NULL OR` guard | NULL-agent test | `1 failed, 12 passed` → `test_agent_none_rows_are_included` **RED** |

```
=== BASELINE ===      13 passed in 0.91s
=== POST-REVERT ===   13 passed in 0.92s  reds: []
SHA identical: True   (spend.py + test_phase_75_5_1_spend_metric.py)
```

**M4 required a fix to become a real mutation, and that is worth recording.** The
research gate predicted the fake never evaluates `agent IS NULL`, and the first matrix
run confirmed it: deleting the NULL guard left all 13 tests **green**. A guard that
cannot fail does not count, so the fake now models SQL three-valued logic — with a NULL
agent, `agent != 'cc_rail'` evaluates to NULL rather than TRUE, so BigQuery would DROP
those metered rows and under-count real spend. With that modeled, M4 kills. Reported
here as a fixed gap rather than as coverage that was there all along.

## 6. Immutable verification command (verbatim)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py -q
13 passed in 0.91s
exit=0
```

11 pre-existing + 2 new. **No flag was flipped**:
`cost_budget_use_llm_spend_enabled` remains at its default. This step makes the metric
correct so the operator *can* flip it safely later.
