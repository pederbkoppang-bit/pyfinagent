# Experiment Results -- masterplan step 75.4

**Step**: 75.4 -- Audit75 S4, skill-prompt delivery integrity
**Cycle**: 2 (cycle-1 Q/A returned CONDITIONAL; blockers fixed -- see §7)
**Date**: 2026-07-20
**Contract**: `handoff/current/contract.md`
**Research brief**: `handoff/current/research_brief_75.4.md` (gate `wf_10cb9956-68a`, PASSED)

---

## 1. Verbatim verification command output

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_skill_delivery.py -q

27 passed, 1 warning in 1.97s
EXIT=0
```

(Cycle 1 was 25 passed / 0 skipped. Cycle 2 is **27 passed / 0 skipped** -- the two added
tests are BEHAVIORAL, driving the real `run_synthesis_pipeline`; see §7.)

(The single warning is a pre-existing `DeprecationWarning` from `google/genai/types.py`
about `_UnionGenericAlias` in Python 3.14 -- unrelated to this step.)

---

## 2. What was built / changed

### (a) gap5-01 -- loader truncation, `quant_model_agent.md`
`## Quant Model Data` and `## Instructions` demoted to `### `. These H2s sat *inside*
the `## Prompt Template` region, so `load_skill`'s `(?=^## |\Z)` terminator cut the
delivered prompt at the first of them.

**Measured, not asserted** -- delivered template length for `quant_model_agent`:

| stage | delivered chars | `{{quant_model_data}}` present |
|---|---|---|
| before this step | **190** (of 7532 raw) | **no** |
| after (a) alone | 908 | yes |
| after (a)+(b) | **2739** | yes |

The agent was being told "Interpret the MDA-weighted quant model factor score" and then
handed no score at all -- the builder's `quant_model_data=json.dumps(...)`
(`prompts.py:1034`) had no placeholder to land in and was silently discarded.

**Class check**: a runtime sweep of all 29 skill files confirms `quant_model_agent` was
the *only* file with an unintended mid-template H2 and the only one with a lost
placeholder. The test `test_no_skill_has_an_unintended_h2_inside_its_template_region`
now enforces that the class stays empty.

### (b) gap5-02 -- never-delivered sections relocated
Relocated INSIDE the template region, heading demoted `## ` -> `### `, body byte-identical:

- `Uncertainty Permission` + `Empty-bracket retraction format` -- **8 files**
- `Code Execution Tasks` -- **3 files**

Delivered-length deltas (criterion 6 / risk R6 quantification):

| file | delivered before | delivered after | canaries |
|---|---|---|---|
| bias_detector | 411 | 1223 | OK |
| critic_agent | 2136 | 3046 | OK |
| deep_dive_agent | 595 | 1505 | OK |
| enhanced_macro_agent | 967 | 1895 | OK |
| moderator_agent | 1624 | 2534 | OK |
| quant_model_agent | 908 | 2739 | OK |
| risk_judge | 1438 | 2348 | OK |
| scenario_agent | 1002 | 2737 | OK |
| synthesis_agent | 5792 | 6702 | OK |

**Criterion 6 -- byte-identity proof (measured, not claimed)**: a line-multiset
comparison of every skill file before vs after, ignoring heading lines, reports
**0 non-blank body drift** -- no non-blank line was added, removed, or altered in any
file, and heading TEXT (level-stripped) is identical in all 9 files. Only heading
levels and section positions changed.

**Correction (cycle 2, Q/A wf_8d493697-c73)**: cycle 1 stated this as the absolute "No
non-heading line was added, removed, or altered in any file." That was **overstated**.
The Q/A re-derived the diff from `git show HEAD` and found exactly **one blank
separator line consumed per file in 8 of the 9 files** (`bias_detector`, `critic_agent`,
`deep_dive_agent`, `moderator_agent`, `quant_model_agent`, `risk_judge`,
`scenario_agent`, `synthesis_agent`; `enhanced_macro_agent` lost none). Blank lines are
non-heading lines, so the absolute wording was wrong. I re-measured independently and
confirm the Q/A's count exactly -- **8 blank lines total**. The substance of criterion 6
is unaffected: no content was lost, only a blank separator absorbed by the relocation.

`quant_strategy.md` was deliberately **NOT** touched (contract §1b.3): it has no
`## Prompt Template` at all and is read whole at `quant_optimizer.py:488`, so its
phase-26.3 section already reaches the model. Relocating it would change what the
optimizer sees. A test asserts it still carries its `## Code Execution Tasks` H2.

### (c) gap5-03 -- critic output budget + the fail-open branch
- `_CRITIC_STRUCTURED_CONFIG.max_output_tokens`: **2048 -> 6144**.
- `_THINKING_CRITIC_CONFIG.max_output_tokens`: **2048 -> 6144**. *Not in the step text* --
  found during research. It is defined but **never referenced** anywhere in the tree;
  raised in lockstep and annotated so it cannot silently reintroduce the truncation if
  ever adopted.
- The `treating as PASS with draft` branch is **gone**. Replaced by the existing in-repo
  idiom (`llm_client.py:1656-1684`): retry once at `min(6144*2, 8192)`, and if the
  verdict is still unparseable, log a distinct warning and set `critic_degraded = True`
  so the report proceeds **flagged** rather than silently blessed.
- `critic_degraded` is attached to **all four** value-return paths of
  `run_synthesis_pipeline` (verified structurally by AST, and by mutation -- stripping it
  from any single path now fails the suite).
- **DISCLOSURE (cycle 2)**: `critic_degraded` currently has **no consumer**. A grep
  across all `.py`/`.ts`/`.tsx` finds nothing that reads it, so the flag is *write-only*
  today and carries **no behavioral effect** -- it is observability groundwork. The
  behavioral win of (c) is the retry plus the removal of the silent PASS upgrade, not
  the flag. Wiring a consumer (surface it in the report UI / alert on it) is queued as
  **75.4.5**.

Chose the "raise the cap" arm of criterion 4's OR, per Anthropic's documented remedy for
truncation ("retry with a higher max_tokens"); the patch-semantics arm is queued as its
own step (§6). 6144 = 1.5 x the 4096 `SynthesisReport` the critic is instructed to echo,
per Anthropic's "max_tokens at least 1.5-2x your expected output size".

### (d) gap5-06 -- enrichment cap made provider-independent
`_skill_gen_config` now always returns `max_output_tokens: 1024` -- on the file-id path
*and* the two no-file-id paths (it previously returned `None`). Return type narrowed
`dict | None` -> `dict`.

New module constant `_ENRICHMENT_MAX_OUTPUT_TOKENS = 1024` is the single source of truth,
referenced by both the Gemini bundle `base_config` and the helper, so the two rails
cannot drift.

**Gemini behavior is unchanged** and this is measured, not assumed: `llm_client.py:968-970`
merges `bundle.base_config` via `setdefault`, and both `_general_vertex` and
`_quant_exec_vertex` already carried `max_output_tokens: 1024`. I independently
re-confirmed the `_quant_exec_vertex` half at `orchestrator.py:586-590`. This corrects
risk R2b in the research brief, which claimed (d) would newly cap 12 Gemini agents --
the adversarial verification leg **REFUTED** that, and I verified the refutation.
The real gap was Claude-rail-only: `llm_client.py:1348` fell back to its own 2048 default,
silently double the documented cap.

### (e) gap5-10 -- stem typo + startup assertion
`orchestrator.py`: `_skill_gen_config("sector_agent")` -> `_skill_gen_config("sector_analysis_agent")`.

The bug was **fail-open, not a crash**: the file-id lookup simply missed, the helper
returned `None`, and the Sector agent silently lost the phase-25.D9 Files-API token
saving (~98% skill-body reduction) forever. Its *prompt* was always correct, because the
builder at `prompts.py:457` already used the right stem. A silent cost regression that
nothing surfaced.

Added `_SKILL_GEN_STEMS` (12 stems) + `_assert_skill_stems_exist()`, called at import.
A stem with no backing `.md` now raises at startup instead of degrading silently.

### (f) `format_skill` kwarg-mismatch warning
Warns when a caller passes a kwarg with no matching `{{placeholder}}` -- i.e. the runtime
value is silently discarded, exactly the gap5-01 failure mode.

Compared against the **extracted** template, never the raw file (contract §1b.10): a
raw-file comparison emits false positives on `moderator_agent` and `risk_judge`, whose
`## Data Inputs` prose documents bare `{{debate_history}}` / `{{past_memory}}` /
`{{devils_advocate}}` tokens that are documentation, not placeholders. Only this
direction is checked; the inverse would fire on intentionally-empty `*_section` kwargs.

### Consumer updated -- `tests/verify_phase_25_D9_1.py`
Criterion 5 legitimately changes the `_skill_gen_config` contract, which **breaks three
live assertions** in the phase-25.D9.1 verifier (claims 3, 4, 5). Updated in the same
commit with an explicit amendment note in its docstring. Re-run output:

```
=== phase-25.D9.1 verification ===

[PASS] 1. orchestrator_has_skill_gen_config_helper
        -> _skill_gen_config method present
[PASS] 2. enrichment_agents_pass_generation_config_with_skill_file_id
        -> call_sites=12 (expected >=11)
[PASS] 3. helper_returns_enrichment_cap_only_when_skill_file_ids_empty_gemini_fallback
        -> got {'max_output_tokens': 1024}
[PASS] 4. helper_returns_skill_file_id_dict_for_mapped_stem
        -> got {'max_output_tokens': 1024, 'skill_file_id': 'file_xyz_123'}
[PASS] 5. helper_returns_enrichment_cap_only_for_unmapped_stem_no_keyerror
        -> got {'max_output_tokens': 1024}

ALL 5 CLAIMS PASS
```

The stale docstring at the helper ("The `None` return is the safe fallback") was rewritten.

---

## 3. File list

| file | change |
|---|---|
| `backend/agents/orchestrator.py` | (c) critic budget x2 + retry/degraded branch; (d) `_ENRICHMENT_MAX_OUTPUT_TOKENS` + helper returns cap on both paths; (e) stem fix + `_SKILL_GEN_STEMS` + import-time assertion |
| `backend/config/prompts.py` | (f) `format_skill` kwarg-mismatch warning |
| `backend/agents/skills/quant_model_agent.md` | (a) 2 headings demoted; (b) 3 sections relocated |
| `backend/agents/skills/{bias_detector,critic_agent,deep_dive_agent,moderator_agent,risk_judge,synthesis_agent}.md` | (b) 2 sections relocated each |
| `backend/agents/skills/scenario_agent.md` | (b) 3 sections relocated |
| `backend/agents/skills/enhanced_macro_agent.md` | (b) 1 section relocated |
| `tests/verify_phase_25_D9_1.py` | consumer of the changed `_skill_gen_config` contract |
| `backend/tests/test_phase_75_skill_delivery.py` | **NEW** -- 27 tests (25 in cycle 1, +2 behavioral in cycle 2) |

`git diff --stat` (source only; excludes unrelated dirty runtime artifacts
`mda_cache.json`, `handoff/audit/*.jsonl`, `.archive-baseline.json`):

```
 backend/agents/orchestrator.py                | 141 +++++++++++++++++++++++---
 backend/agents/skills/bias_detector.md        |   5 +-
 backend/agents/skills/critic_agent.md         |  15 ++-
 backend/agents/skills/deep_dive_agent.md      |  15 ++-
 backend/agents/skills/enhanced_macro_agent.md |  12 +--
 backend/agents/skills/moderator_agent.md      |  15 ++-
 backend/agents/skills/quant_model_agent.md    |  21 ++--
 backend/agents/skills/risk_judge.md           |  15 ++-
 backend/agents/skills/scenario_agent.md       |  17 ++--
 backend/agents/skills/synthesis_agent.md      |  15 ++-
 backend/config/prompts.py                     |  28 ++++-
 tests/verify_phase_25_D9_1.py                 |  24 +++--
```

---

## 4. Mutation matrix (cycle 1) -- 12/12 listed mutations killed

> **The cycle-1 heading of this section read "12/12 killed, 0 vacuous guards". The
> second clause was RETRACTED in cycle 2 -- see the retraction below and §7.**

Per the phase-75 durable rule (harness_log Cycle 130): *a guard that cannot fail does not
count; mutate the fixture too, and mutate first the guard you catch yourself defending.*
Each mutation was applied to a backup-protected file, the suite run, and the file
restored. Script: scratchpad `mutate.py`.

```
baseline: 25 passed, 1 warning in 2.05s

M1   KILLED (4 failed)   restore '## Quant Model Data' H2 (re-truncates the template)
M2   KILLED (3 failed)   restore '## Instructions' H2
M3   KILLED (1 failed)   move ONE file's (risk_judge) Uncertainty section outside the region
M4   KILLED (1 failed)   move ONE file's (scenario_agent) Code Execution section outside
M5   KILLED (1 failed)   revert _CRITIC_STRUCTURED_CONFIG to 2048
M5b  KILLED (1 failed)   revert the dead twin _THINKING_CRITIC_CONFIG to 2048
M6   KILLED (1 failed)   reinstate the fail-open literal 'treating as PASS with draft'
M7   KILLED (2 failed)   revert the sector stem to 'sector_agent'
M8   KILLED (1 failed)   remove the format_skill kwarg-mismatch warning
M9   KILLED (1 failed)   remove max_output_tokens from the NO-file-id return ONLY
M10  KILLED (1 failed)   HARNESS MUTATION: swap the real load_skill for a string stub
M11  KILLED (1 failed)   remove the import-time stem-exists assertion call

12/12 mutations killed; 0 survived
post-matrix restore check: 25 passed, 1 warning in 2.02s
```

**RETRACTION (cycle 2).** The cycle-1 headline "12/12 mutations killed; **0 vacuous
guards**" was **not measured** -- the second clause was *inferred* from the first. A
mutation matrix licenses only "these 12 mutations were killed", never the global "this
suite contains 0 vacuous guards". The Q/A proved the difference by finding a **6th
vacuous guard** the matrix never attempted -- and it was the guard protecting my own
§2(c) claim. See §7 for the full cycle-2 record. The only claim cycle 1 supports is:
**the 12 listed mutations were killed.** Nothing more.

Notes on the two the research flagged as highest-risk-of-being-skipped:
- **M9** killed -> criterion 5's "both paths" clause is real, not satisfied by the easy
  file-id path alone.
- **M10** killed -> criterion 1's anti-stub clause is real. This mutates the **test
  harness itself**, which is the level that escaped detection in 75.2.1.
- **M3 / M4** each killed exactly **1** test -> the per-file parametrization is genuinely
  per-file, not an `any()`-shaped assertion that would pass with 7 of 8 files broken.

I added **M5b** myself (not in the contract's matrix) because the dead-twin config was my
own addition and therefore the guard I was most likely to be defending.

**A guard I got wrong and the tests caught**: my first draft asserted one shared body
phrase across all 8 uncertainty files and all 3 code-execution files. It failed on
`bias_detector`, `enhanced_macro_agent` and `scenario_agent` -- their section bodies
genuinely differ. Rather than weaken the assertion to the heading alone, I measured the
real per-file body text and pinned each file to its own verbatim line. A single shared
canary would have silently under-covered whatever it did not match.

---

## 5. Honest scope + regression disclosure

### The `bias_detector.md` fix is COSMETIC -- not a behavioral win
`bias_detector.md` is an **orphan**: no `load_skill("bias_detector")` caller exists
anywhere. Production bias detection is deterministic Python
(`backend/agents/bias_detector.py`, imported at `orchestrator.py:38`). Fixing it
satisfies criterion 2 but has **zero** production effect. Stated here so it is not
mistaken for one of the real deliveries.

### Full-suite regression check -- measured against a clean HEAD worktree
`.venv/bin/python -m pytest backend/tests/ -q` on the working tree: **13 failed, 1246
passed**. I did **not** assume these were pre-existing. Measured:

- A detached worktree at HEAD (no 75.4 changes), with `backend/.env` and the same live
  `backend.log` / `handoff/logs/backend-watchdog.log` symlinked in, reproduces **10** of
  them identically.
- The 3 that differed are all **live-runtime-log evidence** tests, and all 3 fail on
  artifact staleness, not code:
  - `test_phase_23_2_10_watchdog...` -- *"watchdog log stale: latest entry 2026-06-11 is
    931.0h old (max 24h)"*. Reproduces identically at HEAD with the same log linked.
  - `test_phase_23_2_9_...prewarm_evidence` -- reads `backend.log`. Reproduces identically
    at HEAD with the same log linked.
  - `test_phase_23_2_6_...skipping_buy_evidence` -- counts `"Skipping BUY"` lines in the
    historical `backend.log`; found 0. In the bare worktree it *skips* ("backend.log
    freshly rotated and no archive found") rather than running, which is why the raw
    counts differed. My diff cannot retroactively alter historical log content, and the
    backend has not been restarted on this code.
- `grep` confirms none of the 3 imports or references `orchestrator`, `prompts`,
  `load_skill`, `format_skill`, `_skill_gen_config`, or any skill file.

**Net: 0 regressions introduced by this step.** The 13 failures are the project's
standing live-environment red set (stale logs / no recent BQ writes / backend not
running), unchanged by 75.4.

### Runtime import smoke
```
orchestrator imports OK; startup stem assertion ran
  _CRITIC_STRUCTURED_CONFIG max_output_tokens = 6144
  _THINKING_CRITIC_CONFIG  max_output_tokens = 6144
  _ENRICHMENT_MAX_OUTPUT_TOKENS = 1024
  registered stems = 12
```

### Not verified live
The Layer-1 analysis pipeline was **not** run end-to-end against a live LLM -- that costs
metered spend and needs owner approval, and this step's criteria are all offline. The
delivered-prompt improvements are proven through the real `load_skill()`, which is the
exact function the production builders call, but **no live model response was observed**.
A backend restart is required for the running process to pick up these changes.
No UI surface is touched by this step, so no Playwright capture applies.

---

## 7. Cycle-2 record -- what the Q/A caught and how it was fixed

The cycle-1 Q/A (`wf_8d493697-c73`) returned **CONDITIONAL**. It verified all 6 immutable
criteria as MET and found **zero unintended production change** -- but it found a **6th
instance of this project's repeat vacuous-guard defect**, and it was in the guard
protecting my own §2(c) claim. Verdict transcribed verbatim in
`handoff/current/evaluator_critique.md`.

**It was right on every count.** No criterion was violated and the shipped behavior was
already correct -- this was weak-guard-not-broken-feature. All four fixes are test-only
or documentation; **no production code changed in cycle 2**.

### Blocker 1 -- the vacuous guard (Unjustified_Inference)
`test_critic_degraded_flag_is_present_on_every_return_path` asserted
`target.count("critic_degraded") >= 5` AND `len(dict_returns) >= 4`, **never binding a
return path to the flag**. The Q/A produced **5 surviving mutants**: stripping the flag
from one, two, or three of the four return paths all SURVIVED, and so did a maximally
vacuous mutant carrying the flag on **zero** return paths with comment-only mentions.

**Fix**: replaced with a structural per-return-path AST check
(`_returns_missing_critic_degraded`). For a dict-literal return the key must be in the
literal; for `return <name>` the enclosing statement list must contain a preceding
`<name>["critic_degraded"] = ...`. **I re-ran all 5 of the Q/A's own mutants -- all now
KILLED.**

### Blocker 2 -- substring guards satisfiable by a comment (Missing_Assumption)
`assert "Critic-Retry" in src` and `assert "critic_degraded" in src` were bare source
scans. The Q/A deleted the **entire 16-line retry block**, left a `# Critic-Retry`
comment, and the suite stayed GREEN; it also restored fail-open semantics under a
reworded log message and stayed GREEN.

**Fix**: two new **behavioral** tests drive the real `run_synthesis_pipeline` with a
stubbed LLM. `test_unparseable_critic_triggers_exactly_one_retry_then_flags_degraded`
asserts a second `Critic-Retry` call actually happens and `critic_degraded is True` on
the returned dict; `test_parseable_critic_verdict_does_not_retry_and_is_not_flagged` is
the negative control. Both of the Q/A's mutants are now KILLED.

**A second-order defect I introduced and then caught**: my first behavioral draft wrapped
the pipeline call in `try/except -> pytest.skip`. Both tests **SKIPPED** (the stub lacked
`settings.max_synthesis_iterations`). A skipped test is itself a guard that cannot fail --
the exact defect being fixed. I supplied a real settings/client stub and **removed the
skip wrappers entirely**, so future breakage goes RED instead of hiding. 0 skipped.

### Blocker 3 -- the falsified headline (Overgeneralization)
Retracted in §4. "0 vacuous guards" was inferred, not measured.

### Blocker 4 -- overstatements (Overgeneralization)
Criterion-6 blank-line wording corrected in §2(b) after independent re-measurement (8
blank lines, exactly the 8 files the Q/A named). `critic_degraded` write-only status
disclosed in §2(c).

### Cycle-2 mutation matrix -- 10/10 killed, 0 survived

Every QA* mutant below is one the **Q/A itself** produced as a SURVIVOR against my
cycle-1 guards. Reproducing the evaluator's own mutants is the only way to *prove* the
fix rather than assert it.

```
baseline: 27 passed, 1 warning [skipped=0]

--- Q/A's surviving mutants against the OLD count-based guard ---
QA1    KILLED (1 failed)   remove critic_degraded from ONE corrected_report return
QA2    KILLED (1 failed)   remove critic_degraded from BOTH corrected_report returns
QA3    KILLED (3 failed)   remove critic_degraded from the final_data return
QA4    KILLED (1 failed)   remove critic_degraded from the error-dict return
QA5    KILLED (3 failed)   flag on ZERO return paths + comment-only mentions (max vacuity)

--- Q/A's surviving mutants against the OLD substring guards ---
QA6    KILLED (1 failed)   delete the ENTIRE retry block, keep a '# Critic-Retry' comment
QA7    KILLED (1 failed)   restore fail-OPEN semantics under a reworded log message

--- new-guard self-mutations (the guards I just wrote) ---
N1     KILLED (1 failed)   make the retry fire TWICE instead of once
N2     KILLED (2 failed)   flag degraded on the HAPPY path too (breaks the negative control)
N3b    KILLED (2 failed)   HARNESS+PROD: blind the AST helper to `return <name>` AND strip that flag

10/10 mutations killed; 0 survived
post-matrix restore check: 27 passed [skipped=0]
```

**A mutation of mine that was invalid, disclosed rather than quietly dropped**: my first
N3 replaced `missing = _returns_missing_critic_degraded(target)` with `missing = []`. It
"survived" -- but that is not a vacuous-guard finding, it is a **meaningless mutant**:
deleting an assertion's input trivially survives against *any* suite. A valid harness
mutation keeps the assertion and breaks the **fixture**, as M10 did. N3b is the correct
shape -- blind the AST helper to `return <name>` paths *and* strip the flag from one such
path, so only a helper that genuinely inspects Name returns catches it. It KILLED.

**Scoped claim, stated precisely**: across cycles 1 and 2, **22 specific mutations were
killed and 0 survived**. That is not the same as "this suite has no vacuous guards" --
cycle 1 is the standing proof that such a claim cannot be inferred from a passing matrix.

---

## 6. Out-of-scope defects found -- queued as their own steps

Per the standing rule (`feedback_queue_discovered_defects_in_masterplan`), each is queued
as a research-gated masterplan step written for an executor with no memory of this
discovery, **not** left as a prose disclosure:

1. **75.4.1** -- critic patch/corrected-fields-only semantics (the arm not taken).
2. **75.4.2** -- `skill_optimizer` post-write delivery-invariant check. `apply_modification`
   (`skill_optimizer.py:397-459`) autonomously rewrites skill files, guarded only by an
   occurs-exactly-once check; the phase-71.4 independent review gate is flag-gated DARK.
   An autonomous run can re-break the headings this step just fixed. Relocating content
   *into* the template region newly exposes it to the auto-optimizer.
3. **75.4.3** -- `quant_strategy.md` has no `## Prompt Template`, so `load_skill` raises
   `ValueError` on it. Harmless today (read whole by the optimizer) but a live landmine
   for any code that globs `SKILLS_DIR`.
4. **75.4.4** -- `bias_detector.md` orphan disposition: delete the file or wire it up.
5. **75.4.5** -- wire a consumer for `critic_degraded` (surface it on the report / alert
   on it). Found in cycle 2: the flag is currently write-only, so a degraded critic run
   is recorded but nothing acts on it.
6. **75.4.7** -- extend the behavioral critic tests to cover the two `corrected_report`
   return paths and the error-dict return. Found by the cycle-2 Q/A: an `if False:`
   unreachable-attachment mutant on a `corrected_report` return survives the whole suite,
   because static AST cannot reason about reachability and the behavioral leg exercises
   only the `final_data` return. A known limit of the technique, not a cycle-1-style
   categorical vacuity -- every realistic regression shape is killed.
7. **75.4.6** -- `orchestrator.py` lacks `from __future__ import annotations` and carries
   3 pre-existing ruff findings (F401 `generate_reflection` unused at :48; F821 undefined
   `Any` at :1009 and :1010). Byte-identical to HEAD, so **not** introduced by this step,
   but the Q/A notes they survive only because Python 3.14 defers annotation evaluation
   (PEP 649) and would still break `typing.get_type_hints()` on those two functions.
