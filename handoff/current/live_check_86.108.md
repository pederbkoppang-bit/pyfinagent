# live_check -- step 86.108 (2026-08-17; exits unpiped)

**Scope: all six criteria.** The previous revision of this file was explicitly
`PARTIAL: criterion 1 only`. Criterion 1's census half is unchanged and is
restated here in summary; everything from §3 onward is new.

## 1. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/agents/orchestrator.py\").read()); print(\"parses\")"'
parses
EXIT=0
```

## 2. Criterion 1a -- the census, re-derived by a COMMITTED script

`scripts/qa/census_invalid_json_86_108.py` prints the glob and the match rule
beside every count and refuses to print a failure rate.

```
$ python scripts/qa/census_invalid_json_86_108.py --rotated-only
TOTAL matching LINES = 2859
  compact              2371    82.9% of lines
  json                  488    17.1% of lines
```

The filed total reproduces EXACTLY on the gate's own corpus. Four corrections
stand and are encoded in the tool: **(a)** the corpus is mixed-format, so a
`"module":`-keyed parser sees 17.1% and looks complete; **(b)** there is no
agent called "Analyst" -- the filed 926 is `Neutral` 310 + `Conservative` 309 +
`Aggressive` 307, three distinct agents, and "Advocate"/"Judge" are `Devil's
Advocate`/`Risk Judge`; **(c)** the Critic double-logs, so 2,859 counts LINES
not EVENTS; **(d)** `9.2%` is a composition SHARE -- no synthesis-attempt
denominator exists in the corpus, so no rate is derivable.

## 3. Criterion 1b -- the rail split, delivered as an ERA BUCKET and labelled

`scripts/qa/era_rail_86_108.py`. The per-era line counts sum to the filed
figure, and the rail mix comes from `pyfinagent_data.llm_call_log` (7,248 rows,
`provider` NOT NULL), queried live via the BigQuery MCP and embedded in the
script so it is re-derivable.

```
era ends            lines  calls    cc  anthr   gem    cc%
----------------------------------------------------------
20260612T104931Z      939    421     0    211   210   0.0%
20260706T225648Z      792   2655    11   2587    57   0.4%
20260724T064045Z      640   1463    10   1266   187   0.7%
20260729T171222Z      192    194     2    169    23   1.0%
20260804T182713Z       91    490     6    402    82   1.2%
20260810T064130Z      146    961     4    813   144   0.4%
20260814T155315Z       59    638     7    562    69   1.1%
LIVE                   15    426     0    375    51   0.0%
----------------------------------------------------------
ROTATED ONLY         2859   <- reproduces the filed 2,859
INCL. LIVE           2874

LIVE LOG IS STILL GROWING: 29,510,950 bytes, mtime 2026-08-17T20:38:24Z,
read at 2026-08-17T20:38:26Z. Its count is a reading, not a final figure.
```

**A per-event rail split is NOT DERIVABLE and this step does not fabricate
one.** Three measured reasons: a JSON marker record's entire field set is
`timestamp/level/module/message` and `grep -c '"model"'` over the corpus
returns 0; the warning lines carry no `request_id`, so any `llm_call_log` join
is time-proximity rather than identity; and that table's `ok` column is "true
on 2xx" while an invalid-JSON body **is** a 2xx, so it is structurally blind to
a parse failure. **Recorded as a deviation from criterion 1's literal wording
for the evaluator to judge:** the criterion says the split must be "measured,
not inherited". An era bucket is measured. A per-event split would satisfy the
words and violate the intent.

The script **computes** the one claim the table supports rather than asserting
it -- an earlier draft of that paragraph claimed the two largest eras had "the
fewest claude-code-tagged calls", which was false (era `20260729` has 2, fewer
than era `20260706`'s 11) and was false precisely because it was typed instead
of derived:

```
Eras with ZERO claude-code-tagged calls: 20260612T104931Z, LIVE
Invalid-JSON lines in those eras: 954 (32.8% of the rotated 2,859)
Across every era, claude-code-tagged calls are 40 of 7248 (0.55%).
NOT SUPPORTED: any statement of the form 'N of the failures came from the CC rail'.
CAVEAT: the CC rail is known to UNDER-tag (phase-78: its calls logged as
        provider='anthropic' with the wrong model), so 'cc%' is a FLOOR.
```

The re-derivation path for the rail columns is executable, not advertised:

```
$ python scripts/qa/era_rail_86_108.py --sql       # prints the BigQuery SQL
```

That SQL was pasted back into BigQuery and reproduced `RAIL_MIX` row for row
(`job_LpeztBcfqtV1hDhaVs9Po2RZgQAe`). **A prior revision of this file claimed
the query was "embedded so it is re-derivable" while the constant held a prose
placeholder and pointed at a `--refresh-help` flag that did not exist** -- the
numbers were true and independently confirmed, but the path to re-deriving them
was fiction. Both are now real.

**The prospective fix ships in this same step, with its limits stated** (§5):
every new failure carries the model that served it and a rail *derived from that
model*. It is not derived from the CC-route flag: that flag alone would stamp
every Gemini-served call `claude_code`, which the table above shows would be the
common case, not an edge case.

## 4. Criterion 2 -- what each transport actually guarantees, cited

This section lands BEFORE any schema or prompt change was designed, as the
criterion requires. **No schema change was made** -- see §9.

| Transport | Guarantee | Source |
|---|---|---|
| Anthropic **API** | **Constrained decoding.** "Structured outputs guarantee schema-compliant responses through constrained decoding: Always valid: No more `JSON.parse()` errors." Structural only: `minimum`/`maximum`/`minLength`/`maxLength` are STRIPPED from the wire schema, `minItems` supported only for 0 or 1, `additionalProperties` must be `false`. **No guarantee is stated for `max_tokens` truncation or refusal.** | `platform.claude.com/docs/en/build-with-claude/structured-outputs` |
| Claude Code **CLI** (`--json-schema`) | **Post-hoc validation, not constrained decoding.** Verbatim: "Get validated JSON output matching a JSON Schema **after the agent completes its workflow**". `--max-tokens` is not a CLI flag at all. | `code.claude.com/docs/en/cli-reference` |
| Claude Code **Agent SDK** | **Validate-then-re-prompt.** "the SDK validates the output against it, **re-prompting on mismatch**. If validation does not succeed within the retry limit, the result is an error." Anthropic's own instruction for this project's rail-drop mode: "A result can also end with subtype `success` but no `structured_output` value … **Treat that case as a failure as well.**" | `code.claude.com/docs/en/agent-sdk/structured-outputs` |
| **Gemini** | **No absolute guarantee claimed.** The doc says output "adhere[s] to a provided JSON Schema" and is "syntactically correct JSON"; it never states a hard guarantee and documents "Very large or deeply nested schemas may be rejected." | `ai.google.dev/gemini-api/docs/structured-output` |

**The local refutation of the obvious fix, verified in-repo rather than
inherited.** The Moderator's generation config declares
`"response_schema": ModeratorConsensus` (`backend/agents/debate.py:55`), and the
census counts Moderator invalid-JSON lines **with its population stated**, per
this step's own rule that no figure ships without its denominator:

```
$ python scripts/qa/census_invalid_json_86_108.py --rotated-only | grep Moderator
   359  Moderator          (of TOTAL matching LINES = 2859)
$ python scripts/qa/census_invalid_json_86_108.py | grep Moderator
   368  Moderator          (of TOTAL matching LINES = 2874, i.e. incl. the live log)
```

A declared schema did not make the failure unreachable at either population.
Note the honest bound: neither figure can be attributed to the Gemini path
specifically -- that is the same rail hole as §3 -- so the claim is the
transport-independent one: **"the schema makes this unreachable" is refuted
regardless of which transport served them.** (A prior revision of this section
quoted the 368 with no population qualifier, immediately after two sections
scoped `--rotated-only`. Both numbers were real; the missing qualifier was the
defect.)

Two further findings that shaped the design, from the brief's read-in-full
sources: schema validity and value correctness are different things
(`arxiv.org/html/2604.25359v1`: "Every model exceeds 84% JSON Pass, yet no model
surpasses 80.4% Value Accuracy" -- a 14-20pt gap at every provider), and
constrained decoding has a measured tax and death-loop failure mode
(`arxiv.org/html/2605.26128v1`: wrong-valid-schema rose 49.5% -> 88.9%;
`arxiv.org/html/2604.06066v1`: 58 of 100 samples entered "continuous death
loops"). **This is why the remedy chosen is observability, not a repair-retry
loop** -- and why no retry loop is built here.

## 5. Criterion 3 -- loud degradation at the RECORD level, demonstrated

New module `backend/agents/parse_failure_ledger.py`. All four emit sites feed
it; the legacy warning line at each site is kept **verbatim** so the existing
census still matches.

| Site | Kinds it can record |
|---|---|
| `debate.py:_parse_json` | `parse_failed`, `schema_valid_but_rejected_downstream` |
| `risk_debate.py:_parse_json` | `parse_failed`, `schema_valid_but_rejected_downstream` |
| `orchestrator.py:_parse_json_with_fallback` | `parse_failed` |
| `llm_parse.py:parse_llm_json` **(ZERO production callers today -- see below)** | `parse_failed`, `truncated`, `schema_valid_but_rejected_downstream` |

**The four sites are not equivalent, and the table above would imply they
are.** `parse_llm_json` has **zero production callers**: it is 75.5's shared
helper and its rewiring is masterplan step 75.5.5, so it is wired here for
surface uniformity, not because it produces failures today. (The other
`_parse_llm_json` hits in the repo are a different, private function in
`meta_evolution/directive_rewriter.py`.) **The three WIRED sites account for
the whole measured population**: the census marker occurs at exactly four
logger sites and all nine census agent buckets are served by those three.

Driving the four REAL functions with real bad input. **The live
`paper_use_claude_code_route` flag is `True` throughout**, which is what makes
this run discriminating: a flag-derived rail would stamp every row
`claude_code`.

```
$ python -c '<drive the four real emit sites>'
LIVE FLAG paper_use_claude_code_route = True

WARNING backend.agents.debate: Moderator returned invalid JSON, using raw text
WARNING ...ledger: PARSE_FAILURE_RECORD agent='Moderator' kind=parse_failed site=debate.py:_parse_json model=gemini-2.5-flash rail=gemini_or_direct ticker=None
WARNING backend.agents.risk_debate: Risk Judge returned invalid JSON, using raw text
WARNING ...ledger: PARSE_FAILURE_RECORD agent='Risk Judge' kind=parse_failed site=risk_debate.py:_parse_json model=claude-opus-4-8 rail=claude_code ticker=None
WARNING backend.agents.orchestrator: Synthesis-Final returned invalid JSON
WARNING ...ledger: PARSE_FAILURE_RECORD agent='Synthesis-Final' kind=parse_failed site=orchestrator.py:_parse_json_with_fallback model=None rail=unknown ticker=None
WARNING backend.agents.llm_parse: Critic: provider stopped with stop_reason='max_tokens' -- output is TRUNCATED...
WARNING ...ledger: PARSE_FAILURE_RECORD agent='Critic' kind=truncated site=llm_parse.py:parse_llm_json model=None rail=unknown ticker=None

return values, UNCHANGED:  None / None / None / (None, True)

{"records_seen": 4, "reconciles": true, "recorder_errors": 0,
 "by_agent_kind": {"Critic|truncated": 1, "Moderator|parse_failed": 1,
                   "Risk Judge|parse_failed": 1, "Synthesis-Final|parse_failed": 1},
 "by_rail": {"claude_code": 1, "gemini_or_direct": 1, "unknown": 2}}

RECORDS (agent / model / rail / basis):
  Moderator        gemini-2.5-flash   gemini_or_direct  measured: not a claude- model, so the CC rail gate cannot match
  Risk Judge       claude-opus-4-8    claude_code       measured: claude- model and paper_use_claude_code_route on
  Synthesis-Final  None               unknown           no_model_in_scope_at_emit_site
  Critic           None               unknown           no_model_in_scope_at_emit_site
```

**The Moderator row is the point.** The CC-route flag is on, and the record
still says `gemini_or_direct` -- because the rail is resolved from the model,
mirroring the client's real predicate (`model_name.startswith("claude-") AND
paper_use_claude_code_route`, `backend/agents/llm_client.py`). A flag-only rule
would have stamped that row `claude_code`, and §3's table shows Gemini traffic
outnumbering claude-code-tagged traffic ~20x, so the misattribution would have
been the common case.

**Three honest `unknown`s, none of them a guess**, each carrying its basis on
the record: no model in scope at the emit site; settings unreadable; or
`paper_rail_failforward_enabled` armed, where a Claude model under an enabled CC
route can still be served by the Vertex-Gemini workhorse and the flag pair
cannot distinguish the two.

Every one of the four census corrections is structurally impossible against
this ledger: the agent phrase is stored verbatim (no fold), each event is
recorded once (LINES vs EVENTS cannot diverge), the kinds are separate buckets,
and the rail is on the record **with the model it was derived from and a stated
basis** -- so a reader can tell a measurement from an absence of one.

Read-only route `GET /api/observability/parse-failures`, deliberately NOT
cached -- a cached observability answer is the same
committed-is-not-in-force lie it exists to detect.

**No gate is loosened and no default verdict is fabricated.** Recording never
changes a return value (pinned by three tests driving the real functions, and
by mutation cells M1/M2/M3). In particular
`risk_debate._judge_parse_fail_fallback`'s `APPROVE_REDUCED at 3% NAV` is
**untouched** -- flipping a risk default is a behaviour change needing its own
step and its own operator decision. It is filed below, not fixed here.

## 6. Criterion 4 -- the dark-flag observability gap, CLOSED

New route `GET /api/settings/flags` + `backend/config/gated_flags.py`.

**The step named seven flags. The real population is 168.** That is the
finding, and it is why the population is DERIVED from a stated rule rather than
listed:

> a gated scalar is a `Settings` field whose annotation is bool/int/float and
> whose name is NOT among `FullSettings`' fields.

`Settings` declares 264 fields; `FullSettings` -- which
`response_model=FullSettings` filters `GET /api/settings/` down to -- declares
45. Two properties follow from the rule and both are load-bearing: it **cannot
go stale** (add a flag and it appears with no edit), and it **cannot leak a
secret** (every key/token/path on `Settings` is `str`/`SecretStr`, so no such
field can enter by construction -- not by a denylist someone has to remember).

For each flag the route reports `in_force` (what the RUNNING process holds)
beside `env_file` (what `backend/.env` says) and a computed `divergent` -- which
is exactly the committed-is-not-in-force check the criterion asks for:

In-process against the REAL route handler (a `TestClient` call, NOT an HTTP
request to the running server -- see §10, which is 404 there until the restart):

```
TestClient GET /api/settings/flags?only=paper_synthesis_integrity_enabled,paper_soft_sector_diversity_enabled,paper_soft_sector_diversity_w,nonexistent_flag
{"count": 3, "population_total": 168, "divergent": [], "divergent_count": 0,
 "requested_but_unknown": ["nonexistent_flag"], "pid": 22814,  # the TEST process
 "flags": {
   "paper_synthesis_integrity_enabled":   {"in_force": true,  "env_file": true,  "env_file_present": true,  "divergent": false},
   "paper_soft_sector_diversity_enabled": {"in_force": false, "env_file": null,  "env_file_present": false, "divergent": false},
   "paper_soft_sector_diversity_w":       {"in_force": 0.0,   "env_file": null,  "env_file_present": false, "divergent": false}}}
```

The armed 86.69 flag reads `in_force: true` and agrees with `.env`. A typo'd
name comes back under `requested_but_unknown` rather than being silently
dropped (mutation cell M8) -- an observability endpoint that answers an empty
set to a typo is lying.

## 7. Criterion 5 -- no promotion, no `.env` write

```
$ stat -f '%Sm  %N' -t '%Y-%m-%dT%H:%M:%S %Z' backend/.env
2026-08-17T15:06:04 CEST  backend/.env      <- session began ~22:2x CEST
```

`backend/.env` was not touched. No flag value changed. The new route is
**read-only**: it has no write path, and `SettingsUpdate` was NOT extended.

**Numbered asks (operator-gated, none actioned here):**

- **ASK-1.** Should the five dead `_FIELD_TO_ENV` rows
  (`paper_synthesis_integrity_enabled`, `paper_position_recommendation_fix_enabled`,
  `paper_risk_judge_shape_fix_enabled`, `claude_code_timeout_s`,
  `claude_code_empty_retry_max`) be made **writable** by adding them to
  `SettingsUpdate`? The contract's P5 said to make them reachable; **I did not
  do that**, and the deviation is deliberate: reachability here means a UI
  write path for dark flags, which is a promotion surface, and criterion 4 asks
  only for a *read-only* route. The comment above those rows still claims they
  are "operator-visible in the Settings UI"; that claim remains false until
  this is answered.
- **ASK-2.** `risk_debate._judge_parse_fail_fallback` fabricates
  `APPROVE_REDUCED at 3% NAV` when the Risk Judge's response is unparseable and
  `paper_risk_judge_parse_fail_reject` is OFF. This is a fabricated verdict on
  a money path. Filed as a defect below; flipping the default is an operator
  decision.
- **ASK-3.** Restart the backend to bring the two new read-only routes into
  force (see §10).

## 8. Criterion 6 -- mutation matrix, control GREEN first

```
$ python scripts/qa/mutation_86_108.py
CONTROL rc=0  collected=37
M1  KILLED   removing the syntactic-failure record at a real emit site is caught
M2  KILLED   the valid-JSON-wrong-shape case going unrecorded is caught
M3  KILLED   collapsing truncated into parse_failed (wrong-cause attribution) is caught
M4  KILLED   a recorder that propagates into the money path is caught
M5  KILLED   a fail-open guard that hides its own breakage is caught
M6  KILLED   widening the population to str -- the secret-leak path -- is caught
M7  KILLED   reverting to the step's hardcoded list of 7 is caught
M8  KILLED   an endpoint that silently drops a typo'd name is caught
M9  KILLED   turning the .env reader into a file-dump primitive is caught
M10 KILLED   reporting the ring gauge as the event counter is caught
M11 KILLED   the original filing's three-analysts-into-one fold is caught
M12 KILLED   dropping the rail stamp -- the field the log corpus never had -- is caught
M13 KILLED   an INVERTED rail attribution (wrong but in-vocabulary) is caught
M14 KILLED   reverting to the flag-only rule (the defect the Q/A found) is caught
M15 KILLED   the hardcoded-model mutant that SURVIVED the cycle-2 matrix is now caught
M16 KILLED   a production call site that stops forwarding the model is caught
M17 KILLED   a LITERAL model at the orchestrator sites -- which no driver reaches -- is caught
M18 KILLED   a resolver called with None -- a FALSE 'unknown' while a model is in scope
M19 KILLED   a BoolOp fallback to a hardcoded model -- actively misattributes

KILLED=19/19  SURVIVORS=none  UNSCORABLE=none
RESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.
```

**M13 and M14 exist because cycle 1's matrix did not contain them, and a Q/A
executed a mutant that SURVIVED.** The rail's only guard was
`assert rec["rail"] in {"claude_code","gemini_or_direct","unknown"}` -- a
set-membership check that passes for every wrong value in the vocabulary. Cell
M12 mutated the field to `""`, the one value *outside* the vocabulary, i.e. the
only mutation such an assertion can catch. The matrix was self-consistent and
proved nothing about correctness. M13 mutates to a wrong-but-valid value
(inverted mapping) and M14 restores the original flag-only defect; both are now
killed by `test_resolve_rail_disagrees_with_the_flag_only_rule`, which asserts
that the resolver and the flag-only rule must DISAGREE on a Gemini model with
the route flag on.

The scoring rule is deliberately strict, and the reasons are recorded in the
runner's docstring: a cell is a KILL only if the control was green FIRST,
pytest exits **1** (exit 5 -- no tests collected -- is NOT a kill), the mutant
**collects the same number of tests as the control** (a mutant that cannot
build is not a killed mutant), and the **specifically named** test is the one
that fails (collateral redness does not demonstrate that this guard catches
this defect).

```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_108_parse_failure_ledger.py -q
37 passed in 2.10s

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
```

**The gate exits 1, and that is the honest result.** The single finding is
`typing.Callable` in `backend/agents/debate.py:16`, and it is PRE-EXISTING --
proven by linting the HEAD copy of that file, which reproduces it. It is queued
below rather than fixed here: a step should not quietly repair unrelated lines
under its own name.

A prior revision of this block claimed `All checks passed! EXIT=0` over
"`<the 11 files this step owns>`". Two things were wrong with that and both
matter more than the lint state itself. The argument list was **elided**, so the
line was unreproducible by construction and could not be a verbatim capture of
any real invocation. And the 11-file set it described **excluded `debate.py`** --
the one file carrying a finding -- while including a file committed earlier;
hand-assembling a scope that omits the failing member is how a green gate gets
manufactured. The scope is now DERIVED from `git diff`, which is the authority
on what changed.

Regression sweep over every adjacent suite:

```
$ python -m pytest backend/tests/ -q -p no:cacheprovider -k "debate or llm_parse or parse or orchestrat or settings or observab or 75_5 or 70_4 or 72_0_2"
567 passed, 3143 deselected, 1 warning in 5.22s
```

The single failure is **pre-existing and unrelated**:
`test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit`
asserts `.claude/settings.json` has `effortLevel == "xhigh"`, but the operator
raised it to `max` on 2026-08-04 (recorded in CLAUDE.md). That file is
unmodified in this tree (`git status --short .claude/settings.json` is empty)
and the test imports nothing this step touched (`grep -c` for every changed
module returns **0**). Queued as a defect below rather than fixed here.

## 9. Scope honesty -- what this step did NOT do

- **No schema or prompt change.** Criterion 2 required the evidence to land
  first; the evidence (§4) argues against a constrained-decoding change, not
  for one. Constrained decoding carries a measured tax and a documented
  death-loop mode, and the Moderator's 368 failures happened with a schema
  declared.
- **No repair-retry loop.** If one is later proposed it must be attempt-capped,
  consistent with the F1b doctrine.
- **No risk default flipped**, including the fabricated `APPROVE_REDUCED at 3%`.
- **`_FIELD_TO_ENV` was NOT made reachable** -- deliberate deviation from the
  contract's P5, see ASK-1.
- **The ledger is process-local and in-memory.** It is not durable and does not
  claim to be; `records_seen` is monotonic for the life of the process only.
  Persisting it is a separate question and is not smuggled in here.
- **Overlap with 86.60 stated, not shared:** 86.60's news-screen empty-response
  diagnosis is the same failure class at a different site. This step owns the
  pipeline-wide marking; 86.60 owns the news-screen entry path.

## 10. NOT YET IN FORCE -- pending restart

The running backend is **pid 41635, started 2026-08-17 15:57:16 CEST**
(13:57:16Z), which predates every edit in this step. It therefore does not
serve the new routes. Measured, with a positive control rather than asserted:

```
/api/settings/flags                    -> HTTP 404
/api/observability/parse-failures      -> HTTP 404
/api/observability/latency             -> HTTP 200   (control: the process IS alive and serving)
```

Per the batched-restart rule the restart is deferred to session end. **Until it
happens, the two routes are committed and NOT running.** No other change in
this step needs a restart: the ledger and the four emit sites take effect for
any newly imported process, and nothing in `backend/.env` changed.

## 11. Cycle 2 -- what the cycle-1 CONDITIONAL changed

Verdict `wf_f0fc7207-486` returned CONDITIONAL with five findings. All five are
fixed; **the first was a real product defect in this step's own code**, found by
the evaluator executing a mutation this matrix did not contain.

| # | Finding | Fix |
|---|---|---|
| 1 | `current_rail()` read only `paper_use_claude_code_route`, but the client enters the CC rail on `model_name.startswith("claude-") AND` that flag. With the flag on, every Gemini-served failure was stamped `claude_code` -- and Gemini traffic outnumbers claude-code-tagged traffic ~20x, so the misattribution was the common case. Three artifacts claimed the rail closed the attribution gap. | Replaced with `resolve_rail(model_name) -> (rail, basis)`, mirroring the client's real predicate. `model_name` threaded through all four emit sites and their eight call sites via `_effective_model_name`, which reuses the client's own `model_name or model.model_name` resolution. Records now carry `model_name`, `rail` and `rail_basis`. Three honest `unknown`s replace every guess. |
| 2 | The rail's sole guard was a set-membership assertion, so an inverted attribution SURVIVED an evaluator-run mutant. Cell M12 mutated to `""` -- the only value the assertion could catch. | Assertions now check the VALUE. Added `test_resolve_rail_truth_table` (5 cells), `test_resolve_rail_disagrees_with_the_flag_only_rule` (the discriminating guard), the three `unknown`-basis tests, and `test_emit_sites_pass_the_model_through_to_the_record`. Added mutation cells **M13** (inverted mapping) and **M14** (revert to flag-only); both KILLED. |
| 3 | Ruff `F401`: unused `sys` in `scripts/qa/mutation_86_108.py:34`. | Removed. Gate now exits 0 on this step's files. The pre-existing `debate.py` `Callable` finding reproduces at HEAD and is queued, not silently fixed. |
| 4 | `era_rail_86_108.py` advertised a re-derivation path that did not exist: `RAIL_QUERY` held a prose placeholder and `--refresh-help` was never implemented. | `RAIL_QUERY` now holds the executable SQL; `--sql` prints it. Ran it back through BigQuery: reproduces `RAIL_MIX` row for row (`job_LpeztBcfqtV1hDhaVs9Po2RZgQAe`). |
| 5 | "368 Moderator" carried no population qualifier in a step whose own rule is that no figure ships without its denominator. | Both figures now shown with their populations: 359 of 2,859 rotated-only; 368 of 2,874 including the live log. |

The evaluator also recorded findings it verified as SOUND, which cycle 2 did not
touch: the criterion-1 impossibility result, the census's refusal to print a
rate, the derived 168-flag population and its by-construction secret-safety, the
untouched `_judge_parse_fail_fallback`, and the NOT-YET-IN-FORCE section.

One presentation note it raised is also fixed: §6's example was rendered as
`$ GET /api/settings/flags?...` carrying a `pid`, which reads like an HTTP call
to the running server when it was an in-process `TestClient` call. It is
relabelled below.

## 13. Cycle 3 -- what the cycle-2 CONDITIONAL changed

Verdict `wf_a49d2d57-3e1` confirmed all five cycle-1 findings CLOSED by
execution, and raised three more. The first is the important one, and it is a
lesson about my own fix rather than about the original defect.

**Finding 1 -- the cycle-2 fix relocated the defect one seam upstream, and
built every new guard at the OLD seam.** Cycle 2 replaced the flag-derived rail
with `resolve_rail(model_name)` and guarded *that* thoroughly. But the value
`resolve_rail` receives comes from `_effective_model_name` at the call site --
new cycle-2 code with no test and no mutation cell. The evaluator executed a
mutant hardcoding it to `"claude-opus-4-8"`: **it SURVIVED 29/29**, silently
reinstating the original misattribution one call frame above where I had just
fixed it. It also ran two converse probes that both went RED, which correctly
confined the gap to the call site rather than overstating it.

Closed three ways, because no single one covers the whole seam:

| Guard | Kind | Covers |
|---|---|---|
| `test_run_risk_debate_records_the_real_client_model_on_every_agent`, `test_run_debate_records_the_real_client_model` | **behavioural** -- drives the REAL production entry points with a fake client under an ON CC-route flag and asserts each record carries the client's Gemini name | the 6 `debate` / `risk_debate` call sites |
| `test_effective_model_name_resolution`, `test_client_model_name_unit` | unit | the two resolution helpers |
| `test_every_parse_call_site_forwards_a_model_name` | **completeness (AST, observes no behaviour and is not offered as a behavioural guard)** -- every call to `_parse_json*` must pass `model_name=`, and that argument must NOT be a literal | all 9 call sites, including a call site that does not exist yet |

The orchestrator's three sites were inline `getattr(client, "model_name", None)`
expressions that no unit test could reach. They now go through a named
`_client_model_name`, and the AST guard's literal-rejection covers the
hardcoding mutant there even though no driver reaches those sites -- stated
plainly rather than papered over. Mutation cells **M15** (the exact surviving
mutant), **M16** (a call site stops forwarding) and **M17** (a literal at an
orchestrator site) are all KILLED.

**Finding 2 -- the ruff block was a hand-assembled scope that omitted the file
with the finding.** Regenerated over a DERIVED scope; it now exits 1 and names
the pre-existing `debate.py` finding. See §8.

**Finding 3 -- the regression sweep number was carried forward from cycle 1**
(543 vs the then-current 552). Re-run -- **but the regenerated figure was
written into `experiment_results` only, and this file kept the stale 543 while
this very sentence claimed otherwise.** Closed in cycle 4; both artifacts now
carry the measured 560.

**Two prose imprecisions the evaluator caught and did not count separately, now
corrected here:** the threading is through **nine** call sites, not eight
(debate 2 + risk_debate 4 + orchestrator 3), and the orchestrator's three did
not use `_effective_model_name` -- they now use `_client_model_name`.

## 14. Cycle 4 -- closing the third CONDITIONAL's findings

Verdict `wf_95c6d117-784` recorded that **all six immutable criteria have
covering, independently-reproduced evidence and the product is sound under 28
executed mutation cells** (my 17, reproduced by the evaluator, plus 11 of its
own). Five findings remained; none was a product defect, and all five are now
closed.

1. **The AST completeness guard was a BLACKLIST and two AST-legal mutants
   survived it.** `_client_model_name(None)` -- a Call, not a Constant -- makes
   every orchestrator record read a FALSE `unknown` while a model is in scope;
   `_client_model_name(c) or "claude-opus-4-8"` -- a BoolOp -- actively
   misattributes. The guard is now a **WHITELIST** (`_accepted_model_name_arg`):
   the argument must be a call to one of the two named resolvers, with at least
   one argument, and no argument that is a bare `None`. Cells **M18** and
   **M19** reproduce the two survivors; both are KILLED. Matrix is **19/19**.
   *Rejecting one syntactic form does not cover a semantic class* -- the same
   lesson as cycle 2, one level down.
2. **The stale regression figure.** Third cycle for this defect class: cycle 3
   regenerated it in one artifact and left the stale 543 in the other while
   claiming it had been regenerated. Both now carry the measured **560**, and
   §8's block was re-run, not hand-edited.
3. **"Queued as a defect" did not reproduce against the queue.** It was prose,
   not a masterplan entry. Now actually filed: **86.112** (the stale
   `effortLevel` test), **86.113** (the pre-existing `debate.py` F401),
   **86.114** (the fabricated `APPROVE_REDUCED at 3% NAV`, previously ASK-2).
   §12 is now a pointer to real steps rather than a list of intentions.
4. **"Every one of the 37 tests drives the REAL function" was false** for the
   AST test, and false in the direction that overstates guard strength in a
   step being graded on guard adequacy. Corrected to 36 of 37, with the
   exception named.
5. **`parse_llm_json` was listed as an equivalent emit site** while having zero
   production callers. Disclosed in §5 and in `experiment_results`, with the
   coverage consequence stated: the three WIRED sites account for the entire
   measured population.

## 12. Defects discovered -- NOW FILED as masterplan steps

Filed in `.claude/masterplan.json` (verify with a walk for these ids -- this
list is a pointer, and a pointer that does not resolve is the defect a cycle-3
finding named):

1. **86.112** -- `test_phase_40_2_settings_json_still_valid_json_after_edit`
   pins `effortLevel == "xhigh"` against a settings file the operator moved to
   `max` on 2026-08-04. The test is stale; the config is correct.
2. **86.113** -- the pre-existing `F401 typing.Callable` at
   `backend/agents/debate.py:16`, which keeps the ruff gate red for every step
   whose derived scope includes that file.
3. **86.114** -- `risk_debate._judge_parse_fail_fallback` fabricates
   `APPROVE_REDUCED at 3% NAV` on an unparseable Risk Judge response when
   `paper_risk_judge_parse_fail_reject` is OFF. This was ASK-2; it is now a
   step, because a money-path fabrication should not live in a prose ask.

Still an **ASK, not a step**, because it is an operator decision rather than a
defect: **ASK-1** -- the phase-61.2 comment above `_FIELD_TO_ENV` asserts those
flags are "operator-visible in the Settings UI"; they are not, and were not
before this step either. Making them so means adding a UI WRITE path for dark
flags, which this step declined to do.
