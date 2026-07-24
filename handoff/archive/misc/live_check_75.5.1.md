# live_check_75.5.1 — LLM-spend metric for the $25/day budget guard (arm a, DARK)

All output verbatim from live runs 2026-07-24. **No live BQ billing query was run**
(per the live_check constraint) — every result below is from the offline suite with
the SQL-semantics-aware fake client.

## 1. Verification command — exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py -q
...........                                                              [100%]
11 passed in 1.41s
```

Regression sweep (the 75.5 rail suite pins the arch-04 seam + hard-block + consumer
resolution — my `_check_cost_budget` edit is inside its pinned surface):

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py backend/tests/test_phase_75_llm_rail.py -q
53 passed, 1 warning in 5.72s
```

(One cycle-internal catch: the first import shape broke the 75.5 pin
`test_consumers_resolve_fetch_spend_from_observability` — fixed by splitting the
import so the pinned literal survives; the failure and fix happened before Q/A.)

## 2. ON-vs-OFF: the breaker's trip point is UNCHANGED with the flag OFF

`test_flag_off_is_byte_identical_to_bq_source` drives the REAL
`llm_client._check_cost_budget` (cache reset between runs, env escape hatch cleared)
with sentinel sources:

- flag **OFF** + BQ metric says (9999, 9999) + LLM metric says (0, 0) → **trips** (as today)
- flag **OFF** + BQ metric says (0, 0) + LLM metric says (9999, 9999) → **does not trip**
  — the LLM metric is ignored even when it screams: $0 trip-point diff vs today.
- flag **ON** inverts both (test_flag_on_reads_the_llm_metric).
- `test_flag_default_is_off` pins `Settings.model_fields[...].default is False`.

## 3. The metered-only crux (phantom free tokens cannot trip the breaker)

`test_cc_rail_rows_contribute_zero_both_shapes`: 600M flat-fee CC-rail tokens in BOTH
row shapes (`provider='claude-code'` AND `provider='anthropic'`+`agent='cc_rail:x'`)
alongside one small metered Gemini row → the computed spend equals the metered row
alone. The fake BQ client applies ONLY the predicates present in the SQL text, so
this test fails if the production SQL loses its exclusions (proven by mutation S3),
and the fake's own discriminating power is self-tested
(`test_fake_client_honors_filter_absence`, proven by mutation S5).

## 4. Fail-open + arch-04 seam regression (criterion 3)

`test_fail_open_returns_zero_and_fires_degradation_seam`: a raising BQ client →
`fetch_llm_spend()` returns (0.0, 0.0) AND `spend_guard_status()` shows
`degraded_count==1, alerted==True` with the error recorded — the SAME
`_record_degradation` seam as `fetch_spend`, so the 75.5 alert-on-transition
behavior covers both metrics.

## 5. Mutation matrix — 6 mutations, 6 killed, 0 survivors

Runner: scratchpad `run_mutations_75_5_1.py`; verbatim log `mutation_matrix_75_5_1.txt`.
Full table in `experiment_results.md` (criterion 4 requires it there).

```
SUMMARY: 6 mutations, 6 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
```

## 6. Lint (scope derived from git after the last edit)

New test + spend.py: only findings are 3× BLE001 (blind `except Exception`) — the
DOCUMENTED fail-open idiom, 2 of 3 pre-existing at HEAD (proven by linting
`git show HEAD:...spend.py`: same class), the third on the new function matching the
established pattern. `__init__.py`: same 4 finding classes before and after (proven
against HEAD baseline) — zero new classes. qa.md's F821/F401/F811 gate: clean.

## 7. git diff --stat (this step's edits)

```
 backend/agents/llm_client.py                     (flag-routed source selection in _check_cost_budget)
 backend/config/settings.py                       (+cost_budget_use_llm_spend_enabled, default False)
 backend/services/observability/__init__.py       (+fetch_llm_spend export)
 backend/services/observability/spend.py          (+_price_llm_tokens +fetch_llm_spend; docstring: 3 invariants)
 backend/tests/test_phase_75_5_1_spend_metric.py  (new, 11 tests)
 .claude/masterplan.json                          (75.5.1 in_progress; +75.5.11 queued)
```

## 8. Process disclosure

A stray `git stash -q` inside a Main diagnostic command briefly stashed the entire
uncommitted GENERATE (the exact `feedback_no_git_stash_with_active_hooks` hazard).
Recovered surgically via `git checkout stash@{0} -- <every stashed file>` +
`git stash drop`; the full suite (53/53), imports, contract, and masterplan status
were re-verified identical after restoration (section 1 output is post-recovery).
