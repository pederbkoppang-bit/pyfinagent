# Experiment Results -- masterplan step 75.5

**Step**: 75.5 -- Audit75 S5, LLM rail: schema enforcement, metered-bypass guards,
model retirement, cost correctness
**Cycle**: 7 (verdict history: CONDITIONAL x2, **FAIL**, CONDITIONAL x2, **FAIL**; retry_count=2 of 3 -- see §6c-§6f) | **Date**: 2026-07-20 | **Priority**: P0
**Contract**: `handoff/current/contract.md` | **Research**: `research_brief_75.5.md`
(gate `wf_0cea9f6a-482`, PASSED)

---

## 1. Verbatim verification command output

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_llm_rail.py -q

41 passed, 1 warning in 4.78s
EXIT=0
```

(Cycle 1 was 40 passed. Cycle 2 replaces one comment-satisfiable source scan with **two**
behavioural tests -- see §6.)

0 skipped, 0 xfailed. The warning is the pre-existing `google/genai`
`_UnionGenericAlias` `DeprecationWarning`, unrelated to this step.

---

## 2. What changed

### (e) llmeng-10 -- THE MONEY FIX
`backend/agents/cost_tracker.py`: `regular_input = max(0, input_tokens - cache_read -
cache_creation)` -> `regular_input = input_tokens`.

Anthropic's prompt-caching doc, verbatim: *"The `input_tokens` field represents only the
tokens that come after the last cache breakpoint in your request - not all the input
tokens you sent"*, and *"total_input_tokens = cache_read_input_tokens +
cache_creation_input_tokens + input_tokens"*. `input_tokens` **already excludes** both
cache buckets, so subtracting them again charged for fewer uncached tokens than were
billed. The `max(0, ...)` clamp floored the result instead of going negative -- which is
exactly why this never surfaced.

**Measured against the real `CostTracker` and the real pricing table:**

```
input=1000 uncached, cache_read=5000, model=claude-opus-4-8 (input $5.0/Mtok)
  OLD (double-subtracting, clamped): $0.002500
  NEW (this fix):                    $0.007500
  expected:                          $0.007500
  under-report eliminated:           66.7%
```

**The direction was the single highest-risk thing in this step** -- getting it backwards
would corrupt cost accounting rather than repair it. The research gate quoted the doc
verbatim and the adversarial leg independently re-derived it before I touched the line.
`anthropic==0.96.0` is an exact pin, so there is no SDK version skew.

**Incidental-safety hazard, now pinned.** `record()` is provider-polymorphic: it reads
Gemini's `prompt_token_count` but Anthropic's cache field names -- and **Gemini's
semantics are the opposite** (its `prompt_token_count` INCLUDES cached tokens).
`regular_input = input_tokens` is correct today *only* because `GeminiClient` never
populates the cache fields. That safety was incidental, not asserted; a test now pins it,
so a future Gemini caching change fails loudly instead of silently over-counting.

### (a) llmeng-01 -- CC-rail schema enforcement
`claude_code_client.py`: the `isinstance(dict)` gate dropped Pydantic model **classes**,
so `--json-schema` was **dead code on the entire Layer-1 pipeline path** (all 9
`response_schema` values there are classes) while the sibling direct-API client enforced
the schema. Added class handling via `model_json_schema()` +
`_ensure_additional_properties_false`.

**The dict branch is preserved, not replaced** -- the step text called Pydantic classes
"the ONLY schema shape the pipeline passes", and the adversarial leg **REFUTED** that:
six production services pass pre-cleaned dicts (`meta_scorer.py:232`,
`news_screen.py:288`, `pead_signal.py:292`, `macro_regime.py:525`,
`analyst_narrative_scorer.py:161`, `call_transcript_gpr.py:139`). Replacing the branch
would have broken all six. A test guards the dict path explicitly.

`$defs`/`$ref` need no flattening (Anthropic documents them as supported); measured 4 of
6 schemas emit `$defs` and **zero** carry an unsupported keyword. Conversion failure
fails **open** (degrade to the pre-fix unconstrained path) but **loudly**.

### (b) llmeng-03 -- advisor_call spend + routing guards
A live metered call site that skipped both rails. Added `_check_cost_budget()` and the
routing-breach guard (pattern copied from `make_client`), and replaced the raw
`os.getenv` client build with `unwrap_secret(getattr(settings, "anthropic_api_key", ""))`.

**SecretStr trap avoided deliberately**: a non-empty `SecretStr` is *truthy*, so
`settings.x or ""` returns the **wrapper**, not the string -- that exact bug silently
disabled 4 alpha overlays for ~3 weeks (auto-memory `project_secretstr_dead_overlays`).
`unwrap_secret` is the sanctioned idiom.

### (c) llmeng-04 -- stop_reason, degraded, and retry ownership
`LLMResponse` gains `stop_reason`, `degraded`, and `is_truncated()`. Populated on **all
four** client paths (Claude, Gemini, Claude Code, OpenAI). Normalization policy: store the
provider-native string verbatim (Claude lowercase, Gemini UPPERCASE) and case-fold in one
helper, so no call site hand-rolls `== "max_tokens"` and silently misses the Gemini form.

**Criterion 3 says "the shared JSON-parse helper" -- there was no such thing.** Several
independent truncation-blind parsers existed instead. Created `backend/agents/llm_parse.py`
as that helper.

> **NO COUNT IS ASSERTED HERE, AND THAT IS THE FIX.** Cycles 1-5 said "three". Cycle 6
> corrected it to "four". Cycle 6's Q/A then showed **the population is at least six**,
> and that my "corrected" number was not a measurement either -- it just moved the digit.
> Two further live members: `agent_definitions.py:353::parse_llm_classification` (on parse
> failure **silently returns a default `AgentType.MAIN` routing**) and
> `evaluator_agent.py:412::_parse_evaluation_response` (returns a conservative FAIL
> verdict). Both take a bare str off a model reply and cannot tell truncation from
> malformation -- the exact property my docstring claimed to enumerate.
>
> **The defect was never arithmetic.** It was asserting a *bounded* count for a population
> whose boundary I never defined operationally. "Takes a str and json-loads it" sweeps in
> GCS loaders, JWE decrypt and BigQuery readers, so it is the wrong discriminator;
> LLM-provenance is the right one. Fixing that requires a census, not a bigger number.
>
> **AND I DID IT AGAIN INSIDE THE FIX (cycle 8).** The cycle-7 version of this passage
> asserted "~36 str->json functions" -- a number I never measured, written into the very
> paragraph explaining why unmeasured numbers must not be asserted. Cycle 7's Q/A executed
> the stated rule and got **17**; I executed it and got **20**. Neither is 36, and neither
> matches the other -- because the rule never said whether `json_io.loads` counts, whether
> the parameter must be annotated, or how `self` shifts the position. **Two disciplined
> measurements disagreeing is itself the proof that the boundary was never operational.**
> The digit is now gone from production source, from here, and from step 75.5.8; the
> argument never needed it.
>
> `llm_parse.py`'s docstring now **asserts no total**, lists the six known sites as
> explicitly "NOT A COMPLETE ENUMERATION", explains why no number appears, and points at
> the census step. Step 75.5.5 is rescoped to rewiring the **named** sites and is now
> forbidden from claiming completeness; the census is queued as **75.5.8**.

**It never retries.** The contract names the single owner verbatim, and so does the
module docstring:
> `ClaudeClient.generate_content`'s phase-4.14.4 MF-26/27 `stop_reason` dispatch,
> `llm_client.py:~1656-1681`, single-shot at `min(max_tokens*2, 8192)`.

The step text's premise of "a single owner" was **already false**: a second, pre-existing
owner lives in the Layer-2 MAS loop (`multi_agent_orchestrator.py:1363-1394`, bounded at
32768). They sit on different paths; unifying them is explicitly **out of scope** and
queued. Knowing both exist is why no third was added.

The nastiest case is covered: truncated JSON that still **parses** (a closed object
missing later fields) is marked degraded. A helper that only flagged parse *failures*
would wave that through.

### (d) llmeng-06 -- model pins + retirement tripwire
Created `GEMINI_DEEP_THINK` (it did **not** exist -- only `GEMINI_WORKHORSE`; deep-think
lived inside a dict) and routed the 5 named pins through the constants. Added
`gemini_retirement_warning()`, date-injectable so it is testable without freezing the
clock. Gemini 2.5 shutdown **2026-10-16** confirmed official
(`ai.google.dev/gemini-api/docs/deprecations`); warns from 2026-09-15.

**No tier pin VALUE changed** -- the step's boundary ("literals become constants only")
is respected; `deep_think_model` still resolves to the operator's `.env` override.

**Scope note**: the step said "the 5 hardcoded pins". The adversarial leg counted **13**
behavioural pins. Criterion 4 is scoped to the 5 named files, so it stays satisfiable as
written -- I fixed those 5 and **queued the other 8** rather than silently widening scope.

**Criterion 4 read strictly.** After routing the pins, 4 `gemini-2.5` occurrences
remained in those files as **docstring/description prose**. The criterion says "zero
`gemini-2.5` literals", and I did not reinterpret an immutable criterion to suit my
implementation -- I reworded the prose too. All 5 files now scan **0**.

### (f) llmeng-11 -- OpenAI telemetry
`OpenAIClient.generate_content` was the only provider path writing no `llm_call_log` row.
Added the retrofit matching the GeminiClient shape, plus the `_t0` latency timer it
lacked. **`provider` is conditional** (`github_models` vs `openai`) because one class
serves both, routed by `base_url` -- a hardcoded value would mis-attribute every GitHub
Models call. Fail-open.

### (g) arch-04 -- public fetch_spend + degradation counter
Promoted the implementation into `backend/services/observability/spend.py` as public
`fetch_spend()`; repointed `llm_client`, `cost_budget_api`, and the Slack job.
Fail-open preserved, but a **degradation counter + one-shot P2 alert** added, because the
prior behavior returned `(0.0, 0.0)` on failure -- **indistinguishable from "no spend"**,
so an outage silently *disabled* the budget guard.

**Back-compat alias retained deliberately**, not out of politeness:
`tests/slack_bot/test_scheduler_wiring_phase991.py:150` monkeypatches that exact
attribute and lives **outside** `backend/tests/`, i.e. outside this step's verification
command -- removing the name would have broken it **silently**. I verified the
monkeypatch myself rather than relying on the brief (the adversarial leg flagged it had
not independently confirmed it). That file: **9 passed**.

---

## 3. Mutation matrix -- 16/16 killed, 0 survived

```
baseline: 40 passed [0 skipped]

M1   KILLED  revert the isinstance gate so a Pydantic CLASS is dropped
M2   KILLED  emit the schema WITHOUT additionalProperties:false
M3   KILLED  remove _check_cost_budget from advisor_call
M4   KILLED  remove the routing-breach guard from advisor_call
M5   KILLED  restore the raw os.getenv client build in advisor_call
M6a  KILLED  drop stop_reason from the CLAUDE return
M6b  KILLED  drop stop_reason from the GEMINI return
M6c  KILLED  drop stop_reason from the CLAUDE CODE return
M7   KILLED  make the parse helper issue its own retry (double-retry)
M8   KILLED  restore a gemini-2.5 literal in ONE of the 5 files
M9   KILLED  make the retirement warning ALWAYS fire (negative control)
M10  KILLED  MONEY: revert regular_input to the double-subtracting form
M11  KILLED  GeminiClient starts populating cache fields (incidental-safety guard)
M12  KILLED  remove log_llm_call from OpenAIClient
M13  KILLED  point a consumer back at the private _default_fetch_spend
M14  KILLED  HARNESS: stub CostTracker so the money test cannot see real math

16/16 mutations killed; 0 survived
post-matrix restore check: 40 passed [0 skipped]
```

**SCOPED CLAIM, stated precisely**: these 16 mutations were killed. That is **not** the
same as "this suite contains no vacuous guards" -- cycle 131 is the standing proof that
the global claim cannot be inferred from a passing matrix.

### M14 initially SURVIVED -- a real finding in my own work

On the first run, **M14 survived**: replacing `CostTracker` with a stub returning the
expected number left the suite green while the money math went entirely unverified. That
is the same shape as 75.4's M10, one module over, and it is a genuine vacuous-guard
finding -- *not* the invalid "delete the assertion's input" shape I retired as N3 in
75.4, because it substitutes the object under test rather than removing the assertion.

Fixed two ways, both needed:
1. **Parametrized** the money test over 4 token mixes (cache-heavy, cache-write, fully
   cached). A stub returning one hardcoded number cannot satisfy all four.
2. **Anti-stub clause**: assert `type(tracker).__module__ == "backend.agents.cost_tracker"`.

M14 now kills 4 tests.

**I also removed a `pytest.skip` I had written** into the OpenAI telemetry test before
running the matrix. It was passing, so the skip never fired -- but a latent
`try/except -> skip` is a guard that cannot fail, which is precisely the cycle-131 lesson.

---

## 4. Regression + honesty

- **Full backend suite**: **12 failed / 1290 passed** (see the composition table below).

  **CORRECTED IN CYCLE 4 -- twice over.** Cycles 1-3 said *"13 failed ... the 13 are the
  project's standing live-environment red set (stale backend.log, no recent BQ writes,
  backend not running), unchanged and unrelated."* The cycle-3 Q/A **measured** that and
  it was **false**: the set contained pure in-process static assertions with no
  live-environment component at all. I had asserted a *composition* I never enumerated --
  the same measure-don't-assert defect as the count, one level up. It is now enumerated:

  | # | test | category | at HEAD? |
  |---|---|---|---|
  | 1 | `23_2_10 watchdog_log_present_and_fresh` | live artifact (log 931h stale) | yes |
  | 2 | `23_2_12 lite_proxy_in_last_7d` | live BQ | yes |
  | 3 | `23_2_12 analysis_results_has_recent_writes` | live BQ | yes |
  | 4 | `23_2_6 backend_log_has_skipping_buy_evidence` | live artifact | yes |
  | 5 | `23_2_9 backend_log_has_prewarm_evidence` | live artifact | yes |
  | 6 | `57_1 reject_binding_main_path_off` | operator dark-flag state | yes |
  | 7 | `57_1 reject_binding_swap_path_off` | operator dark-flag state | yes |
  | 8 | `57_1 off_identity_prompts_are_verbatim` | operator dark-flag state | yes |
  | 9 | `60_3 flag_defaults_off` | operator dark-flag state | yes |
  | 10 | `23_2_15 known_pass_scripts_still_pass` | **static** -- 6 verify scripts red | yes |
  | 11 | `60_1 claude_code_rail_declares_latency_profile` | **static** -- `assert 150 > 150` | yes |
  | 12 | `test_portfolio_swap zero_buy_gap` | **static/logic** -- expects 2 SELLs, gets 1 | yes |

  So the honest split is **5 live-environment, 4 operator-gated dark flags, 3 static**.
  All 12 reproduce at a clean HEAD worktree (the 7 non-live ones verified directly this
  cycle: `7 failed, 42 passed` at HEAD, identical identities). **Zero introduced by this
  step.**

  **The count dropped 13 -> 12 because this step FIXED one of them** -- see the lock-roster
  entry below.

- **Lock roster guard (phase-23.2.14) -- a real defect this step caused, and its cause.**
  The cycle-3 Q/A found that `(g)/arch-04` added a new `threading.Lock()`
  (`_DEGRADED_LOCK`, `spend.py:39`) while `EXPECTED_LOCK_COUNT` stayed at 15, and that
  guard's own docstring requires *"an explicit phase-23.2.14 re-audit + this bump"* in the
  same commit. I had not done it, and **my truncated regression capture is what hid it**.

  Re-audited properly and measured: **17 sites**, not 15.
  - **16th** `_RAIL_GUARD_LOCK` (`claude_code_client.py:103`) -- **pre-existing, not mine**:
    added 2026-07-07 by phase-66.1 (`27d40df5`) without the required bump, i.e. this guard
    had been RED for ~2 weeks.
  - **17th** `_DEGRADED_LOCK` -- **mine**, from this step.

  Both audited against the phase-23.2.14 re-entrancy criteria and both **CLEAN**: each
  mutates state under the lock, copies a flag out, and performs its
  `raise_cron_alert_sync` call **outside** the `with` block -- the canonical
  non-re-entrant shape. `EXPECTED_LOCK_COUNT` bumped 15 -> 17 with that audit recorded in
  the test file. The guard now **passes**.

  **The lesson is the important part**: the pre-existing redness *masked* my new lock. A
  guard that is already failing cannot detect the next drift -- it silently stops being a
  guard. That is why "13 pre-existing failures, all unrelated" was not just imprecise but
  dangerous: it is the sentence that let a real defect through three review cycles.

- **Lint gate** (`ruff --select F821,F401,F811` over `$(git diff --name-only HEAD -- '*.py')`,
  a **derived** scope, not a typed list): **exit 1, 2 F401 findings** as of cycle 8, both
  in `backend/autonomous_loop.py` (`BacktestEngine` :409, `EvaluationVerdict` :436) -- both
  inside `try/except ImportError` availability probes, so removing them needs the
  `find_spec` judgment call and is deferred to **75.5.6**.

  > **ELEVENTH-INSTANCE CORRECTION (cycle 8, from the root-cause research workflow).** For
  > cycles 2-7 this bullet asserted *"3 pre-existing F401s"*. The forensic agent measured
  > the derived scope and found **four**, not three -- the fourth being `import pytest`
  > unused at `backend/tests/test_phase_23_2_14_no_reentrant_locks.py:20`, a file this step
  > edited (the lock-roster bump), which never appeared in the hand-typed list. Same defect
  > class as instance #2, live in the shipped artifacts for six cycles. Fixed by
  > **measurement, not assertion**: I removed the two `[*]`-fixable genuinely-dead imports
  > in files this step already touched (`pytest` here, `typing.Any` in `rag_agent_runtime`),
  > re-ran the gate over the git-derived scope, and it now reports **2** -- both the
  > `find_spec` probes. Every number in this bullet was produced by running the command,
  > not typed. I did also introduce one F401 myself earlier (`import os`, orphaned by the
  > arch-04 move) and removed it in cycle 1.

  **CORRECTION (cycle 2).** Cycle 1 stated this as *"over all 10 touched files: All checks
  passed."* That was **false**. I ran the gate over a 10-file list that **omitted the four
  pin files** (`evaluator_agent`, `rag_agent_runtime`, `skill_modification_review`,
  `autonomous_loop`) and then reported the clean result as covering everything. The Q/A
  reproduced the real command and got exit 1. Nothing was hidden by it -- the findings are
  pre-existing -- but the claim did not reproduce as written, which is exactly the
  `feedback_measure_dont_assert_claims` failure: a result asserted while one command from
  verification. Queued as **75.5.6**.
- **Out-of-scope test**: `tests/slack_bot/test_scheduler_wiring_phase991.py` -- 9 passed.

### Not verified live
**No live LLM call was made** (metered spend needs owner approval) and **no live BQ spend
query** was executed -- `fetch_spend` was exercised only through its failure path. A
**backend restart** is required for the running process to pick any of this up. No UI
surface is touched, so no Playwright capture applies.

### MONEY IMPACT -- read before the next cost dashboard review
The (e) fix moves **reported** cost **upward** on every cached Anthropic call (it was
under-reporting by up to 66.7%). Cost dashboards will step up and the $25/day budget
guard may trip **sooner** than before. That is the correction working, not a regression --
the money was always being spent; it just was not being counted.

---

## 6. Cycle-2 record -- what the Q/A caught and how it was fixed

Cycle-1 Q/A (`wf_f4a5526b-e6a`) returned **CONDITIONAL** with 2 blockers. **It was right
on both.** Verdict transcribed verbatim in `handoff/current/evaluator_critique.md`.

**First, the thing that mattered most: it independently CONFIRMED the money fix.** It
fetched the Anthropic prompt-caching doc itself, cross-checked the `claude-api` skill
reference, re-derived the arithmetic from scratch ($0.002500 -> $0.007500, 66.7%), and
verified the `anthropic==0.96.0` pin in both `requirements.txt` and the installed
environment. The direction is correct, not backwards. It also re-ran 15 mutations of its
own via `sys.modules` injection (touching zero repo files) -- all KILLED -- and confirmed
criteria 1, 2, 4, 5, 6 MET on behavioural evidence.

Both blockers were **test-strength and reporting-honesty only**. **No production code
changed in cycle 2.**

### Blocker 1 -- the 17th vacuous guard (it found the one I asked it to hunt for)
Criterion 3 requires `stop_reason` *"populated by the Claude, Gemini, and CC client
paths"*. Only the **CC** limb was behavioural. The Claude and Gemini limbs rested on
`test_stop_reason_field_is_wired_on_the_claude_and_gemini_return_sites`, a **disk source
scan** the Q/A proved comment-satisfiable: replacing the real wiring with
`stop_reason=None,  # was: stop_reason=getattr(response, "stop_reason", None)` left the
assertion **green** while the production behaviour was destroyed.

It graded this CONDITIONAL rather than FAIL because it verified the production wiring is
genuinely correct (`llm_client.py:1890` Claude, `:1146` Gemini) -- a weak guard over
working code, not broken code.

**Fix**: deleted the scan; added two BEHAVIOURAL tests driving the real `ClaudeClient` and
`GeminiClient` against mocked SDK responses, mirroring the CC-path test that already
worked. **I then re-ran the Q/A's own two mutants -- both now KILLED**, so the fix is
proven rather than asserted.

### Blocker 2 -- my lint claim did not reproduce
Corrected in §4 with the honest numbers and the root cause.

### Cycle-2 mutation results
```
baseline: 41 passed [0 skipped]

QA-C   KILLED (1 failed)   Claude: stop_reason=None + a comment carrying the old text
QA-G   KILLED (1 failed)   Gemini: stop_reason=None + a comment carrying the old text

2/2 of the Q/A's comment-satisfiable mutants KILLED
```
Combined with cycle 1: **18 specific mutations killed, 0 survived.** Scoped precisely --
that is *not* a claim that this suite has no vacuous guards. Cycle 1 of this very step is
the standing proof that the global claim cannot be inferred: I ran a 16/16 matrix and the
17th guard was still there, because I never mutated it.

### The pattern worth naming
Three of my last four self-inflicted defects are the same shape: **a check that reports
success over a narrower surface than the one I claimed.** A source scan that "covers" a
behaviour it cannot observe; a ruff invocation that "covers" 10 files while omitting 4; a
mutation matrix that "covers" a suite while never touching one guard. In each case the
green result was real -- the *scope* of the claim was not.

**And a fourth, while writing this very section**: my cycle-2 edit to §1 used a plain
`str.replace()` whose anchor omitted the `cd ` prefix, so it silently no-matched and left
the stale `40 passed` in place. I only caught it because I re-read the file. Same root
cause as the 75.4 instance, same fix: **assert every substitution**. A `.replace()` that
matches nothing is indistinguishable from one that worked.

**And a FIFTH, which I did not catch -- the cycle-2 Q/A did.** Two stale figures survived
in §4 (`1288 passed`, `+40`) *inside the very document announcing the fix and naming the
discipline*. I had corrected §1's count and not swept the rest. The correct figures,
measured: **1289 passed, +41**. That is the honest count: **five** instances, not four.

The lesson underneath all five is narrower than "be careful": **when a number changes,
the unit of work is the SWEEP, not the edit.** I kept fixing the instance I was looking at
and never enumerated every place the number appeared. §4 of this cycle was written by
grepping every numeric claim in both documents and checking each against a fresh
measurement -- lines 20/162/182 legitimately still read `40` because they are historical
references to the cycle-1 matrix, which really did run at 40 tests.

---

## 6b. Cycle-3 record -- the two blockers cycle 2 found

The cycle-2 Q/A (`wf_a96a2ef9-119`) returned **CONDITIONAL** and **verified both cycle-1
blockers genuinely fixed** -- it re-ran its own comment mutants (QA-C, QA-G: both KILLED),
and then probed the new behavioural tests for the 75.2.1 fixture-divorce failure mode by
destroying the *derivation* (`_finish_reason = getattr(_fr,"name",...)` -> None) rather
than the assignment. The Gemini test still died, proving the hand-rolled `SimpleNamespace`
mock drives the real production path rather than a parallel fiction. It ran 11 further
mutants of its own; all KILLED. Verdict transcribed verbatim in `evaluator_critique.md`.

Both new blockers were **reporting integrity**. **No production code changed in cycle 3.**

### Blocker A -- two stale figures, in the section announcing the fix
Covered in §4 and §6. Fifth instance of the pattern. Corrected by **sweeping** every
numeric claim in both documents against a fresh measurement rather than editing the one
figure I happened to be looking at.

### Blocker B -- a "verbatim" capture that had been hand-edited
`live_check_75.5.md` §1 is labelled *"Verification command, verbatim"*, but its pytest
progress line carried **40 dots above a `41 passed` summary** -- internally impossible for
a single real run. Cause: in cycle 2 I regex-replaced only the *summary* line and left the
progress line from cycle 1. The underlying command genuinely passed, so no result was
misrepresented -- but the artifact the operator is meant to audit was not the verbatim
output it claimed to be, which defeats the entire purpose of the live_check gate
(CLAUDE.md: it exists to convert *"the agent claimed PASS"* into *"an artifact an operator
can audit"*).

**The fix is not a better edit -- it is to never edit a capture.**
`live_check_75.5.md` is now produced by a generator that runs each command and emits its
stdout verbatim; no block is hand-written. Verified internally consistent: the progress
line now has **41 dots** under a **41 passed** summary. The one filter applied (dropping
the unrelated urllib3 `RequestsDependencyWarning` env lines) is disclosed in the document
itself rather than silently applied.

---

## 6c. Cycle-4 record -- the FAIL, and the real defect underneath it

The cycle-3 Q/A (`wf_6d4dac30-eb7`) returned **FAIL**. Two things about that verdict:

**The rule fired as designed.** Its own grading of the two blockers was CONDITIONAL, but
with two prior consecutive CONDITIONALs and no intervening PASS/FAIL, the
3rd-CONDITIONAL auto-FAIL rule (`per-step-protocol.md` §4 / CLAUDE.md F1) required FAIL.
I had explicitly instructed it not to soften findings to avoid that outcome and not to
inflate anything to reach it; it stated plainly that it did neither. **This is the harness
working, not misfiring** -- three cycles of the same defect class is exactly the signal
the rule exists to escalate. `retry_count` 0 -> 1; status remains `pending`.

**And it found a real defect, not just a reporting one.** That distinction matters:

### The lock-roster defect -- and why three review cycles missed it
`(g)/arch-04` added a new `threading.Lock()` (`_DEGRADED_LOCK`, `spend.py:39`). The
phase-23.2.14 guard's docstring requires *"an explicit re-audit + this bump"* in the same
commit. I did neither.

It survived three cycles because of a **compound failure**:
1. The guard was **already red** (HEAD measured 16 against `EXPECTED_LOCK_COUNT = 15`,
   because phase-66.1 added `_RAIL_GUARD_LOCK` on 2026-07-07 without bumping it). So my
   new lock produced **no status change** -- red before, red after.
2. My regression capture was **truncated**, so the operator could not see which tests were
   red.
3. My prose asserted the reds were *"the standing live-environment red set ... unchanged
   and unrelated"* -- a **composition I never measured**, which was false on both counts.

Each alone was survivable. Together they made a real defect invisible. **This is the
strongest argument yet for why the reporting defects were never cosmetic**: the sloppy
sentence was load-bearing.

**Fixed**: both extra locks audited against the phase-23.2.14 re-entrancy criteria (both
CLEAN -- state mutated under the lock, flag copied out, alerting performed outside the
`with` block), `EXPECTED_LOCK_COUNT` bumped 15 -> 17 with that audit recorded in the test
file. The guard **passes** again, so it can detect the next drift. Full-suite reds drop
13 -> 12 as a direct result.

### The other two blockers
- **Red-set composition**: now enumerated in a 12-row table in §4 with each test
  categorised (5 live-environment, 4 operator dark-flag, 3 static) and each verified
  present at a clean HEAD worktree. Zero introduced.
- **live_check verbatim over-claim**: the cycle-3 note said every block was verbatim with
  one named filter, while three blocks were truncated, reformatted, or concatenated. The
  file is now emitted **unmodified** -- warnings and the complete 12-line failure list
  included -- and the one block that genuinely composes two runs is **labelled as a
  composition** instead of hidden under a single `$` prompt.

### What I actually got wrong, stated once
Six instances across two steps of one defect: **claiming a scope I had not measured.** A
scan that "covers" behaviour it cannot observe; a lint run that "covers" 10 files while
omitting 4; a matrix that "covers" a suite while never touching one guard; a count carried
forward instead of re-measured; a capture labelled verbatim that was edited; and a failure
set described by a category I never enumerated. Every green result was real. The *scope of
the claim* never was. The engineering has been confirmed correct at every cycle -- the
money fix independently re-derived twice, and **no production code changed in cycles 2, 3,
or 4** except the lock-roster bump that cycle 3 correctly demanded.

---

## 6d. Cycle-5 record -- the seventh instance

The cycle-4 Q/A (`wf_899ffbc7-1a6`) returned **CONDITIONAL** with exactly one blocker, and
**verified every cycle-3 fix independently rather than trusting it**: it re-counted the
lock sites with the guard's own regex (17, enumerated), traced `_RAIL_GUARD_LOCK` to
commit `27d40df5` (2026-07-07, phase-66.1) confirming the 16th is not mine, and **audited
both locks for re-entrancy itself** -- following `raise_cron_alert_sync` through to
`AlertDeduper.should_fire` to check it cannot re-enter, and confirming `alerting.py` has
zero references to `spend`, so no A->B->A cycle exists. It also confirmed the counter had
reset (so it graded on merits, not under the auto-FAIL rule), and endorsed the reasoning
for clearing the lock guard -- with one caveat that became the blocker.

**The blocker: §5 claimed seven defects were "queued as their own steps". None existed.**

I wrote the section, listed `75.5.1`-`75.5.7`, cited the operator rule that forbids
prose-only disclosure -- and never created the steps. The previous step (75.4) *did*
create real `75.4.1`-`75.4.7`, so the convention was established and simply not executed.
The Q/A caught it by walking the masterplan for each id; I verified the same way before
fixing, and again after.

Now genuinely queued: 7 steps, `pending`, `harness_required`, 4 criteria each, verified
present with no drift to any existing step's `verification` block.

### Why this one is the sharpest of the seven
The first six were miscounted or over-scoped *descriptions of real work*. This was a
description of work that **did not exist** — and it would have destroyed the artifacts:
the archive hook moves `experiment_results.md` on status flip, so the seven defects,
including a money-guard measuring the wrong metric, would have been archived out of the
active queue with no trace in the plan.

It also punctures my own cycle-4 reasoning: I justified clearing the lock guard partly on
"and I queued the process gap as 75.5.7". Half that justification was unsupported at the
time I made it. The Q/A noticed and said so.

### The pattern, seventh instance, final form
Every one of the seven is the same move: **stating a scope or a state I had not
verified.** A scan covering behaviour it cannot see; a lint run covering 10 files of 14; a
matrix covering a suite minus one guard; a count carried forward; a capture labelled
verbatim that was edited; a failure set described by an unmeasured category; and now a
queue that was never written. The engineering was correct every time. The claims about it
were not, and the last one was load-bearing enough that a Q/A had to go count rows in a
JSON file to catch it.

**No production code changed in cycle 5.**

---

## 6e. Cycle-6 record -- the eighth instance, in production source

The cycle-5 Q/A (`wf_dc64f22e-ee7`) returned **CONDITIONAL** and independently confirmed
the cycle-4 fix was real: it walked the masterplan with `jq`, verified all 7 ids exist as
executable steps (not placeholders), checked each was self-contained for a memoryless
executor, and re-derived the diff itself (`+133/-0`, 7 added, 0 removed, **no
pre-existing verification block changed**). It also re-derived the money math from
scratch a third time and re-confirmed every criterion behaviourally.

Two blockers, both verified by me before fixing:

### Blocker 1 -- "three duplicated parse helpers" was wrong. There are FOUR.
Measured: `directive_rewriter.py:212::_parse_llm_json`, same truncation-blind
`text: str` signature, consumed live by `directive_review.py:36`,
`directive_rewriter.py` itself, and `skill_modification_review.py:186` -- **a file this
step edited**.

**This is the first of the eight to reach production source.** The count came from the
research brief and was carried, unmeasured, into `llm_parse.py`'s docstring, this
document, and a masterplan step's success criteria. The others were disposable prose; a
docstring is what the next engineer reads and believes.

Corrected in all three places; 75.5.5 widened to four sites with its enumeration criterion
strengthened to require a **programmatic scan** rather than a hand-written list, so the
next person cannot repeat my mistake by trusting the number.

### Blocker 2 -- the registration gap, in the module this step created
`fetch_spend` / `spend_guard_status` / `reset_spend_guard_status` were re-exported from
`observability/__init__.py` but **omitted from `__all__`** -- measured: `len(__all__)`
still 15, `import *` did not bind them, while all 15 other re-exports *were* listed. The
`# noqa: F401` I had written was ruff reporting exactly this, and I suppressed it.

arch-04's entire purpose was to give the money guard a **public** home, so leaving it out
of the public surface undercut the fix. And it is the same added-without-registering class
as the lock roster — the one I queued **75.5.7** to generalise — reappearing in the module
I had just written. Registered; `__all__` is now 18; the `F401` suppression is gone.

### Eight instances, one move
Every one: **stating a scope or state I had not measured.** The eighth is the most
instructive because it shows the pattern is not about carelessness with prose — it
propagates into code, into criteria, and into the next step's definition of done. The
research gate said "three"; six artifacts and five review cycles repeated it; nobody
measured until a Q/A did.

**No production behaviour changed in cycle 6** -- one docstring and one `__all__`.

---

## 6f. Cycle-7 record -- FAIL #2, and stopping the count instead of moving it

The cycle-6 Q/A (`wf_e67de0a0-674`) returned **FAIL**. Its own grading of the blockers was
CONDITIONAL; it was the third consecutive CONDITIONAL after the cycle-3 FAIL reset, so the
auto-FAIL rule required FAIL. It stated it neither softened findings to avoid that nor
inflated anything to reach it, and both blockers reproduce from two grep commands.
**`retry_count` 1 -> 2 of max_retries 3.**

### The ninth instance -- my correction was not a measurement
Cycle 6 "fixed" the parse-helper count 3 -> 4. Cycle 6's Q/A proved **that was wrong too**:
under the docstring's own stated definition the population is at least six. I verified
both new members myself before accepting it.

### And the strengthening measure I added was circular
In cycle 6 I amended 75.5.5 to require proving completeness "by a programmatic scan for any
remaining `def _parse_*json*(text: str` shaped helper". I ran that exact shape this cycle:
**it matches 3 of the 4 members I had just enumerated.** It misses
`orchestrator.py:309::_parse_json_with_fallback` because that parameter is named
`json_string`, not `text` -- and it is structurally blind to both newly-found members,
whose names contain no `json` token at all.

I had drawn a regex around the answers I already knew and then asserted it proved the
unknown tail. That is worse than the original error: it is a completeness proof that
cannot discover anything it does not already contain.

### The fix: stop asserting the count
Three cycles of moving a number is the signal that the number was the wrong deliverable.

- `llm_parse.py`'s docstring now **asserts no total**. It lists the six known sites under
  the heading "NOT A COMPLETE ENUMERATION", explains in-source why no number appears, and
  instructs the next reader not to replace that wording with a number unless they have run
  the census.
- **75.5.5 rescoped** to rewiring the *named* sites -- bounded, executable work -- with an
  explicit criterion that it must **not** claim completeness, plus per-site behavioural
  tests pinning the two domain-object failure modes (silent `AgentType.MAIN` default;
  conservative FAIL verdict).
- **75.5.8 queued**: the behavioural census, whose discriminator must be **LLM-provenance**
  rather than function names, and which must pass a **self-test** proving it rediscovers
  every already-known member. A scan that cannot find its own known members fails that
  criterion by construction -- the exact trap I fell into.

### Nine instances, and what they actually share
Not carelessness with prose. Every one was a **claim about a set** whose membership rule I
never wrote down: files covered by a lint run, tests covered by a matrix, failures in a
category, steps in a queue, helpers in a population. When the membership rule is unstated,
"I checked" and "I checked the part I happened to look at" are indistinguishable -- to me
most of all. The durable fix is not more care. It is: **state the membership rule first,
make the check derive from it, and if the rule cannot be stated, do not assert the count.**

**No production behaviour changed in cycle 7** -- one docstring, two masterplan steps.

---

## 5. Discovered defects -> queued as their own steps

**VERIFIED PRESENT in `.claude/masterplan.json`** (cycle 5): ids `75.5.1`-`75.5.7`, all
`status: pending`, `harness_required: true`, 4 immutable success criteria each, executor-
tagged, with a `live_check` per step. Confirmed by walking the file after writing, and by
a diff against `git show HEAD:` showing 7 NEW ids, 0 removed, and **no `verification`
block changed on any pre-existing step**.

> **Cycle-4 correction.** Cycles 1-4 carried this exact section heading and list while
> **none of these steps existed** -- I wrote the prose and never created them. The cycle-4
> Q/A walked the masterplan and found zero `75.5.x` ids. That is the **seventh** instance
> of this step's recurring defect, and the sharpest: not a miscounted number but *a claim
> that work had been done which had not been done at all*. It also had teeth -- on status
> flip the `archive-handoff` hook moves this file to `handoff/archive/phase-75.5/`, and
> seven real defects would have gone with it, including **75.5.1** (the $25/day "LLM"
> budget guard actually measures BigQuery spend) and **75.5.2** (8 more `gemini-2.5` pins
> against a 2026-10-16 shutdown). The operator rule is explicit that a discovered defect
> gets its own step, *"never just a prose disclosure"* -- and prose is exactly what this
> was.

Per `feedback_queue_discovered_defects_in_masterplan`, each is research-gated and written
for an executor with no memory of this discovery:

1. **75.5.1** -- `fetch_spend` measures **BigQuery** spend
   (`INFORMATION_SCHEMA.JOBS_BY_PROJECT.total_bytes_billed` at $6.25/TiB), but
   `settings.cost_budget_daily_usd` is documented as the *"Daily LLM-spend cap"* and
   CLAUDE.md calls the $25/day cap the LLM circuit breaker. **The guard does not measure
   what its name says.** (g) preserved this behavior deliberately; reconciling it is its
   own decision.
2. **75.5.2** -- the 8 remaining `gemini-2.5` behavioural pins outside criterion 4's five
   files (`directive_review.py:159`, `directive_rewriter.py:202`, `news/sentiment.py:81`,
   `harness_memory.py:322,:503`, `services/autonomous_loop.py:2648,:2663`,
   `api/agent_map.py:132`).
3. **75.5.3** -- schema-keyword guard: a future `Field(ge=/le=)` would make the CC rail
   start 400-ing silently (Anthropic rejects `minimum`/`maxLength`/recursive schemas).
4. **75.5.4** -- unify or formally document the **two** max_tokens retry owners.
5. **75.5.5** -- route the **NAMED** duplicated JSON-parse helpers through
   `llm_parse.parse_llm_json`. **NO total count is asserted** (cycles 5-8 proved "three"
   and "four" both wrong; see 75.5.5's own scope note and 75.5.8's census). The named set
   is the six in `llm_parse.py`'s docstring: `debate.py:122`, `risk_debate.py:118`,
   `orchestrator.py:309`, `directive_rewriter.py:212`, `agent_definitions.py:353`,
   `evaluator_agent.py:412`. This step created the shared helper but deliberately did
   **not** rewire the live paths.
6. **75.5.7** -- phase-66.1 added `_RAIL_GUARD_LOCK` (2026-07-07, `27d40df5`) without the
   phase-23.2.14 re-audit + count bump its guard requires, leaving that guard RED for ~2
   weeks and therefore unable to detect further drift. phase-75.5 cleared the count as
   part of its own required re-audit, but the process gap that let a lock land unaudited
   is unaddressed: consider making the guard's failure message name the offending commit,
   or wiring it into the pre-commit path so it cannot go quietly red.
7. **75.5.6** -- clear the remaining F401s. **MEASURED (cycle 8), not asserted**: after
   removing the two `[*]`-fixable dead imports in files this step already touched
   (`pytest` in the lock test, `typing.Any` in `rag_agent_runtime`), the derived-scope gate
   reports **2** -- `autonomous_loop.py:409` (`BacktestEngine`) and `:436`
   (`EvaluationVerdict`), both inside `try/except ImportError` probes where ruff suggests
   `find_spec`. 75.5.6 is the `find_spec` judgment call on those two.
8. **75.5.8** -- behavioural CENSUS of truncation-blind LLM-JSON parsers (queued cycle 7):
   the population count must be *measured* by an LLM-provenance discriminator with a
   known-member recall self-test, never asserted.
