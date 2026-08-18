# experiment_results -- step 86.108

**GENERATE complete for all six criteria.** Full verbatim evidence lives in
`handoff/current/live_check_86.108.md`; this file is the build record.

## What was built

### New files

| File | Purpose |
|---|---|
| `backend/agents/parse_failure_ledger.py` | The countable-record layer. Turns an unparseable agent output from a log sentence into a record carrying agent / kind / site / **model_name** / **rail** / **rail_basis** / ticker. Thread-safe, process-local, bounded ring + monotonic counters. |
| `backend/config/gated_flags.py` | Derives the operator-gated-flag population from a stated rule and reports each flag's `in_force` value beside what `backend/.env` says, with a computed `divergent`. |
| `backend/tests/test_phase_86_108_parse_failure_ledger.py` | 37 tests. **36 drive the REAL function or the REAL route handler; one does not** -- `test_every_parse_call_site_forwards_a_model_name` is a static AST completeness check that executes no production code, and is labelled as such where it lives. |
| `scripts/qa/mutation_86_108.py` | 19-cell mutation matrix with a strict scoring rule (see below). |
| `scripts/qa/era_rail_86_108.py` | The era-bucketed rail split, with the non-derivability of a per-event split stated in the output rather than in a footnote. |

### Modified files

| File | Change |
|---|---|
| `backend/agents/debate.py` | `_parse_json` feeds the ledger on both the syntactic and the wrong-shape branch. Legacy warning kept verbatim. |
| `backend/agents/risk_debate.py` | Same, at `risk_debate.py:_parse_json`. |
| `backend/agents/orchestrator.py` | `_parse_json_with_fallback` feeds the ledger. |
| `backend/agents/llm_parse.py` | `parse_llm_json` records `truncated` and `parse_failed` as DIFFERENT kinds, plus the non-dict branch. **It has ZERO production callers today** -- it is 75.5's shared helper and its rewiring is masterplan step 75.5.5 -- so it is wired for surface uniformity, NOT as one of the three sites that actually produce the 2,859. The `_parse_llm_json` hits elsewhere in the repo are a different, private function in `meta_evolution/directive_rewriter.py`. |
| `backend/api/observability_api.py` | `GET /api/observability/parse-failures` -- read-only, uncached. |
| `backend/api/settings_api.py` | `GET /api/settings/flags` -- read-only, uncached, no `response_model` filter. |

**`backend/services/autonomous_loop.py` was deliberately NOT touched** -- a peer
session holds uncommitted work there, and the existing analysis-level
`_parse_failed` marker already works. The new ledger sits at the four emit
sites, one layer below, so the two do not collide.

## Verbatim verification output

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/agents/orchestrator.py\").read()); print(\"parses\")"'
parses
EXIT=0

$ .venv/bin/python -m pytest backend/tests/test_phase_86_108_parse_failure_ledger.py -q
37 passed in 2.06s

$ { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u
  (13 files -- DERIVED, not hand-typed; includes the peer session's
   backend/services/autonomous_loop.py and backend/api/sovereign_api.py,
   which this step does not own)

$ uvx ruff check --select F821,F401,F811 --no-cache --output-format=concise $(cat scope)
backend/agents/debate.py:16:20: F401 [*] `typing.Callable` imported but unused
Found 1 error.
RUFF_EXIT=1

$ git show HEAD:backend/agents/debate.py > /tmp/debate_head.py && uvx ruff check ... /tmp/debate_head.py
/tmp/debate_head.py:16:20: F401 [*] `typing.Callable` imported but unused
Found 1 error.        <- PRE-EXISTING: reproduces on the HEAD copy
  -> PRE-EXISTING and NOT owned by this step; queued below. The scope is
     DERIVED from git diff, never hand-typed.

$ python -m pytest backend/tests/ -q -p no:cacheprovider -k "debate or llm_parse or parse or orchestrat or settings or observab or 75_5 or 70_4 or 72_0_2"
567 passed, 3143 deselected, 1 warning in 5.22s
  -> ZERO failures. The sweep previously carried one, and the explanation
     that used to sit here -- "test_phase_40_2 pins effortLevel=='xhigh' while
     the operator moved it to 'max'" -- is now OBSOLETE: step 86.118 repaired
     that test (commit 1bf26bf8) to assert the documented `max`, and it is
     still selected by this -k pattern. Consequence stated rather than left
     implicit: masterplan step 86.112, filed by THIS step to fix that test, is
     now MOOT and should be closed or re-scoped.

$ python scripts/qa/mutation_86_108.py
CONTROL rc=0  collected=37
KILLED=19/19  SURVIVORS=none  UNSCORABLE=none
RESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.

$ python scripts/qa/era_rail_86_108.py
ROTATED ONLY         2859   <- reproduces the filed 2,859
INCL. LIVE           2874

$ python scripts/qa/era_rail_86_108.py --sql   # executable; re-ran it in BigQuery
                                               # job_LpeztBcfqtV1hDhaVs9Po2RZgQAe
                                               # reproduced RAIL_MIX row for row
```

## Artifact shape

- A ledger record -> `{ts, agent, kind, site, model_name, rail, rail_basis,
  detail, ticker}`.
- `GET /api/observability/parse-failures` -> `{records_seen, records_retained,
  evicted, reconciles, recorder_errors, by_agent_kind, by_site, by_rail,
  recent[], process:{pid}}`. `records_seen` is a **counter**;
  `records_retained` is a **gauge** and is labelled as one in the payload.
- `GET /api/settings/flags[?only=a,b]` -> `{population_rule, count,
  population_total, divergent[], divergent_count, requested_but_unknown[],
  env_file, pid, flags:{name:{in_force, env_file, env_file_present,
  divergent}}}`.

## The three findings that changed the shape of the work

1. **Criterion 1's per-event rail split is not derivable**, and the reasons are
   measured (no rail field on any line; no `request_id` to join on;
   `llm_call_log.ok` is 2xx-shaped and blind to an invalid body). Delivered an
   era bucket and said so. The prospective fix -- recording the model on every
   new event and deriving the rail FROM THAT MODEL -- ships in the same step.
2. **The dark-flag population is 168, not the 7 the step named.** Deriving it
   from a stated rule rather than listing it is what makes the endpoint
   drift-proof AND secret-safe: the rule admits only bool/int/float, so no
   `str`/`SecretStr` field can enter the payload by construction.
3. **The evidence argues against the obvious remedy.** The Moderator's config
   declares `response_schema: ModeratorConsensus` and the census still counts
   Moderator failures (359 of the 2,859 rotated corpus; 368 of the 2,874
   including the live log), so "the schema makes this unreachable" is refuted
   in-repo. Combined with the measured constraint tax and death-loop literature,
   the remedy chosen is observability, not suppression. No schema change, no
   retry loop.

## Deviations from the contract, stated for the evaluator

- **Contract P5 said "make the dead `_FIELD_TO_ENV` rows reachable". I did
  not.** Reachability means adding those flags to `SettingsUpdate`, i.e. a UI
  **write** path for dark flags -- a promotion surface. Criterion 4 asks only
  for a read-only route. Filed as ASK-1 for the operator.
- **Contract P2 anticipated an era bucket keyed on `paper_use_claude_code_route`
  over time.** No history of that flag's value exists (`.env` is not tracked),
  so the era key is the log-rotation window and the rail evidence is
  `llm_call_log`'s per-window provider mix. Same idea, an available key.

## NOT YET IN FORCE

Backend **pid 41635** started 2026-08-17 13:57:16Z, before these edits.
`/api/settings/flags` and `/api/observability/parse-failures` both return
**404** on the running process while `/api/observability/latency` returns
**200** (positive control). Restart is batched to session end per the standing
rule. Nothing else in this step needs a restart.

## Cycle 2 -- response to the CONDITIONAL (`wf_f0fc7207-486`)

Five findings, all fixed. **The first was a real product defect in this step's
own code**, and the evaluator found it by executing a mutation this step's
matrix did not contain.

1. **`current_rail()` inherited the rail from a flag instead of measuring it.**
   The client enters the CC rail on `model_name.startswith("claude-") AND
   paper_use_claude_code_route`; reading the flag alone stamped every
   Gemini-served failure `claude_code`, and Gemini traffic outnumbers
   claude-code-tagged traffic ~20x, so the misattribution was the common case.
   Replaced by `resolve_rail(model_name) -> (rail, basis)` mirroring the real
   predicate. `model_name` is threaded through all four emit sites and their
   eight call sites by `_effective_model_name`, which reuses the client's own
   `model_name or model.model_name` resolution. Records now carry `model_name`,
   `rail` and `rail_basis`; three cases resolve to an explicit `unknown` with a
   stated reason rather than a guess (no model in scope; settings unreadable;
   `paper_rail_failforward_enabled` able to substitute Vertex-Gemini).
2. **The rail guard was a set-membership assertion**, so an inverted
   attribution survived. Assertions now check the value; added a 5-cell truth
   table, the discriminating `test_resolve_rail_disagrees_with_the_flag_only_rule`,
   three `unknown`-basis tests, an emit-site threading test, and mutation cells
   **M13** (inverted mapping) and **M14** (revert to flag-only) -- both KILLED.
3. **Ruff F401** (`sys` unused in the matrix runner) removed; gate exits 0.
4. **`era_rail_86_108.py`'s re-derivation path was fiction** -- a prose
   placeholder in `RAIL_QUERY` and a `--refresh-help` flag that did not exist.
   `--sql` now prints executable SQL, re-run in BigQuery to reproduce RAIL_MIX.
5. **"368 Moderator" carried no population qualifier.** Both figures now ship
   with their denominators (359/2,859 rotated-only; 368/2,874 incl. live).

Files changed in cycle 2: `backend/agents/parse_failure_ledger.py`,
`backend/agents/debate.py`, `backend/agents/risk_debate.py`,
`backend/agents/orchestrator.py`, `backend/agents/llm_parse.py`,
`backend/tests/test_phase_86_108_parse_failure_ledger.py`,
`scripts/qa/mutation_86_108.py`, `scripts/qa/era_rail_86_108.py`,
`handoff/current/live_check_86.108.md`. No new files; no production behaviour
beyond the ledger fields changed; `backend/.env` still untouched.

## Cycle 3 -- response to the second CONDITIONAL (`wf_a49d2d57-3e1`)

The evaluator confirmed all five cycle-1 findings CLOSED by execution and
raised three more.

1. **My cycle-2 fix relocated the defect one seam upstream and guarded only the
   old seam.** `resolve_rail` was thoroughly guarded; the value feeding it --
   `_effective_model_name` at the call site, new cycle-2 code -- was not. A
   mutant hardcoding it to `"claude-opus-4-8"` SURVIVED 29/29, reinstating the
   original misattribution one call frame upstream of the fix. Closed with
   three guards of different kinds, because none alone covers the seam:
   **behavioural** drivers that run the REAL `run_debate` / `run_risk_debate`
   with a fake client under an ON CC-route flag and assert each record carries
   the client's Gemini name (covers the 6 debate/risk_debate sites); **unit**
   tests for `_effective_model_name` and the new `_client_model_name`; and an
   **AST completeness** guard requiring every `_parse_json*` call site to pass
   `model_name=` AND requiring that argument not to be a literal (covers all 9
   sites, including one that does not exist yet). The orchestrator's three
   inline `getattr` expressions became the named `_client_model_name`. Cells
   **M15** (the exact surviving mutant), **M16** and **M17** are KILLED; the
   matrix is 17/17.
2. **The ruff block was a hand-assembled 11-file scope that omitted the one
   file with a finding**, under a "Verbatim verification output" heading, with
   an elided argument list that was unreproducible by construction.
   Regenerated over a scope DERIVED from `git diff`; it now exits 1 and names
   the pre-existing `backend/agents/debate.py:16` finding.
3. **The regression sweep figure was the cycle-1 capture** (543) pasted into a
   cycle-2 document (552 at the time). Re-run; the block now carries the
   current 560.

Prose corrections it caught: the threading is through **nine** call sites, not
eight, and the orchestrator's three never used `_effective_model_name`.

Files changed in cycle 3: `backend/agents/orchestrator.py` (extracted
`_client_model_name`), `backend/tests/test_phase_86_108_parse_failure_ledger.py`
(+8 tests), `scripts/qa/mutation_86_108.py` (+3 cells),
`handoff/current/live_check_86.108.md`. No production behaviour changed beyond
naming an existing expression; `backend/.env` still untouched.

## Cycle 4 -- closing the third CONDITIONAL, then PARKING

Verdict `wf_95c6d117-784` recorded **all six immutable criteria MET and the
product sound under 28 executed mutation cells**. Five residual findings, none
of them a product defect; all five closed:

1. The AST completeness guard was a **blacklist** (rejected `ast.Constant`
   only), so `_client_model_name(None)` and `... or "claude-opus-4-8"` both
   survived. Converted to a **whitelist** of accepted shapes
   (`_accepted_model_name_arg`); cells **M18**/**M19** reproduce both
   survivors, both KILLED. Matrix **19/19**.
2. The regression figure was regenerated in this file but left stale in
   `live_check`, which claimed otherwise. Both now carry the measured **560**.
3. "Queued as a defect" was prose. Now real masterplan steps: **86.112**
   (stale `effortLevel` test), **86.113** (pre-existing `debate.py` F401),
   **86.114** (the fabricated `APPROVE_REDUCED at 3% NAV`, was ASK-2).
4. "Every one of the 37 tests drives the REAL function" corrected to 36 of 37.
5. `parse_llm_json`'s **zero production callers** disclosed, with the coverage
   consequence stated.

**Then PARKED, not re-spawned.** The verdict sequence is
`[CONDITIONAL, CONDITIONAL, CONDITIONAL]`, so CLAUDE.md's 3rd-CONDITIONAL rule
requires the next pass to return FAIL regardless of evidence. Per the standing
rule -- all criteria MET + starved => PARK + escalation file, never iterate --
the step is parked at `status: pending` and the decision is the operator's.
See `handoff/current/escalation_86.108_third_conditional.md`.
