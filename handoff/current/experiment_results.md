# Experiment results — Step 75.5.1 (the cost-budget guard gets the LLM-spend metric its name promises)

Date: 2026-07-24. Execution model: opus-tagged P1 MONEY-ADJACENT step → Main (Fable 5)
GENERATE; Researcher gate opus/max (wf_9cece795-b16, PASSED, 6 read-in-full, pricing
externally validated). **Arm (a) chosen** per the research verdict; shipped **DARK**
(flag default OFF, byte-identical trip point until the operator flips).

## What was built

1. **`fetch_llm_spend()`** in `backend/services/observability/spend.py`: (daily, monthly)
   METERED LLM spend — one month-window BQ query over `llm_call_log` (column-pruned,
   `timeout=30`, GROUP BY model with a same-day split), priced in Python via
   `_price_llm_tokens` against the LIVE imported `cost_tracker.MODEL_PRICING` +
   `_DEFAULT_PRICING` with the cache-aware formula (read 0.1×, write 2.0×) ported from
   cost_tracker.py:198-206. Module docstring now carries the THREE invariants:
   metered-only (CC-rail excluded — both row shapes), raw-tokens-×-pricing (never stored
   dollars; session_cost_usd is a gauge; token counts are invariant across the 75.5
   cache-cost fix), cache-aware (the sovereign_api cache-blind variant under-counts).
2. **Metered-only SQL filter** (the crux): `WHERE ... AND ok AND provider !=
   'claude-code' AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')` — flat-fee CC-rail
   tokens are FREE on the Max rail; pricing them at API rates would falsely trip the
   $25 breaker and halt trading (the session_cost_usd-staircase phantom class).
3. **Fail-open through the arch-04 seam**: any exception → (0.0, 0.0) via the SAME
   `_record_degradation` (counter + one-shot P2 alert) as `fetch_spend`.
4. **Flag routing** in `llm_client._check_cost_budget`:
   `cost_budget_use_llm_spend_enabled` (settings.py, default **False**) selects
   `fetch_llm_spend` vs `fetch_spend`. Nothing else in the hot path changed (cache TTL,
   caps, trip logic, fail-open, env escape hatch untouched). Import split keeps the 75.5
   pinned literal `from backend.services.observability import fetch_spend` intact.
5. **New offline suite** `backend/tests/test_phase_75_5_1_spend_metric.py` (11 tests):
   real-pricing-table assertions with expected values re-derived INLINE (never via the
   production helper — anti-tautology), cache-token pricing, CC-rail zero-contribution
   (both shapes), failed-call exclusion, agent-NULL inclusion, daily/monthly window
   split, fake-client self-test (the stub CAN represent a filter-less query), fail-open
   + seam regression, flag OFF/ON routing against the real breaker, flag-default pin.
6. **Queued 75.5.11** (research-gated, sonnet-tagged): the DISCOVERED caps disagreement —
   hard-block enforces $25/$300 from settings while the tile + Slack watcher hardcode
   $5/$50 (operator-facing numbers 5× off the real halt point). Per
   feedback_queue_discovered_defects_in_masterplan, queued not folded in.

## Files changed

`backend/services/observability/spend.py`, `backend/services/observability/__init__.py`,
`backend/config/settings.py`, `backend/agents/llm_client.py`,
`backend/tests/test_phase_75_5_1_spend_metric.py` (new),
`.claude/masterplan.json` (75.5.1 → in_progress; +75.5.11),
`handoff/current/{contract.md, research_brief_75.5.1.md, live_check_75.5.1.md,
experiment_results.md}`.

## Verbatim verification output

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py -q
...........                                                              [100%]
11 passed in 1.41s
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py backend/tests/test_phase_75_llm_rail.py -q
53 passed, 1 warning in 5.72s
```

## Mutation matrix (immutable criterion 4 + qa.md §4c) — 6 mutations, 6 killed

Runner + verbatim log in scratchpad (`run_mutations_75_5_1.py`,
`mutation_matrix_75_5_1.txt`); summary line verbatim:
`SUMMARY: 6 mutations, 6 killed, survivors: NONE` + `post-restore sanity: pytest exit 0`.

| # | Mutation (applied to real code, executed) | Killed by |
|---|---|---|
| S1 | flag-ON branch swapped back to the BQ metric (criterion-4 required) | `test_flag_on_reads_the_llm_metric` |
| S2 | flag gate removed — always the LLM metric (criterion-4 required) | `test_flag_off_is_byte_identical_to_bq_source` |
| S3 | CC-rail exclusion DROPPED from the SQL (the phantom-spend crux) | `test_cc_rail_rows_contribute_zero_both_shapes` |
| S4 | cache-read discount broken (0.1× → 1.0×) | `test_metered_rows_priced_against_real_pricing_table` + `test_cache_tokens_are_priced_not_ignored` |
| S5 | **STUB**: fake client filters CC-rail unconditionally (SQL-sensitivity neutered — would mask S3) | `test_fake_client_honors_filter_absence` |
| S6 | **FIXTURE/expected**: test's inline cache multiplier drifted to match a hypothetical wrong prod | `test_metered_rows_priced_against_real_pricing_table` |

## Honest disclosures

- Cycle-internal regression caught pre-Q/A: my first import shape broke the 75.5 pin
  `test_consumers_resolve_fetch_spend_from_observability`; fixed by import split, both
  suites green after.
- Process incident: a stray `git stash -q` in a diagnostic command stashed the entire
  uncommitted GENERATE mid-cycle (the codified `feedback_no_git_stash_with_active_hooks`
  hazard, hit by Main itself). Surgical recovery via `git checkout stash@{0} -- <files>`
  + drop; all 53 tests, imports, contract, and masterplan state re-verified identical
  post-recovery. Recommend the operator consider a `Bash(git stash*)` deny rule to make
  the memory mechanical.
- Known conservative biases of the new metric (documented, acceptable for a breaker):
  no `is_batch` column in llm_call_log → batched rows priced at full rate (over-count →
  trips EARLY = safe); advisor blended-model calls priced at the row's single model.
- Lint: 3× BLE001 blind-except on spend.py are the documented fail-open idiom
  (2 pre-existing at HEAD, proven); `__init__.py` finding classes unchanged vs HEAD.
