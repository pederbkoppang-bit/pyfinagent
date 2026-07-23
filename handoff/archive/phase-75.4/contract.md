# Contract -- masterplan step 75.4

**Step id**: `75.4`
**Name**: Audit75 S4 -- skill-prompt delivery integrity: loader truncation, undelivered sections, critic cap
**Phase**: phase-75 (Full-stack code-quality audit vs official docs + best practices)
**Priority**: P0
**Executor tag**: `[executor: sonnet-4.6/high]` (executed this session)
**Cycle**: 1
**Date**: 2026-07-20

---

## 1. Research-gate summary

**Gate**: PASSED. Workflow run `wf_10cb9956-68a`, two legs (researcher + independent
adversarial verifier).

**Envelope**: `tier=moderate`, `external_sources_read_in_full=5`,
`snippet_only_sources=15`, `urls_collected=20`, `recency_scan_performed=true`,
`internal_files_inspected=44`, `gate_passed=true`. Not audit-class (bounded,
enumerated defect list -- `coverage.dry` not required).

**Brief**: `handoff/current/research_brief_75.4.md` (write-first honored -- skeleton
created within the first few tool calls, appended incrementally).

### 1a. Claim verification (adversarial leg -- independent re-derivation)

Every factual claim in the step text was re-derived from the current files by a
second agent that read the brief only *after* forming its own view. Verdicts:

| Finding | Verdict | Note |
|---|---|---|
| gap5-01 | **CONFIRMED (exact)** | Loader executed, not inferred: 190 chars, `{{quant_model_data}}` absent. Lines :78/:81 exact. Sole affected file across all 29 skills. |
| gap5-02 | **PARTIAL** | Counts (8 / 3) exact. Three mechanism corrections -- see §1b. |
| gap5-03 | **PARTIAL** | Cap, literal string and line range all exact. The "4096-token report" *premise* is not sourced from `critic_agent.md` -- restated in §1b. |
| gap5-06 | **CONFIRMED** | Both `_skill_gen_config` returns omit the cap; `ClaudeClient` default 2048 at `llm_client.py:1348`. |
| gap5-10 | **CONFIRMED (exact)** | `sector_agent` at `orchestrator.py:1267` is the only occurrence; no such `.md` exists. |
| item (f) | **CONFIRMED** | `format_skill` at `prompts.py:207` has no logging. |

### 1b. Corrections the research forced into this plan

1. **"sit AFTER `## Experiment Log`" is false for `bias_detector.md`** -- that file has
   no Experiment Log heading at all (`bias_detector.md:6,18,28`). The correct universal
   is "sit after the first H2 that terminates the extracted region."
2. **`bias_detector.md` is an ORPHAN** -- no `load_skill("bias_detector")` caller exists;
   production bias detection is deterministic Python (`backend/agents/bias_detector.py`,
   imported at `orchestrator.py:38`). Fixing it satisfies criterion 2 but has **zero**
   production effect. It will **not** be reported as a behavioral win.
3. **A 4th file carries the phase-26.3 section**: `quant_strategy.md:241`. It is out of
   the `load_skill` path (that file has no `## Prompt Template`, so `load_skill` raises
   `ValueError`) and is read whole at `quant_optimizer.py:488` -- its section **is**
   already delivered. **It must NOT be touched.** Criterion 2's "3 files" is correct as
   written for the `load_skill`-delivered set.
4. **gap5-03 premise restated.** There is no `4096` or `token` string anywhere in
   `critic_agent.md`. The real chain is: the *delivered* critic template instructs
   "Always include the corrected_report field with the full report JSON";
   `CriticVerdict.corrected_report` is `Optional[SynthesisReport]` (`schemas.py:66`);
   `SynthesisReport` is budgeted at 4096 (`_SYNTHESIS_STRUCTURED_CONFIG`,
   `orchestrator.py:85`). A 2048 critic budget cannot fit the echo it demands.
5. **NEW, not in the step text**: `_THINKING_CRITIC_CONFIG` (`orchestrator.py:97-103`)
   carries the same `max_output_tokens: 2048` and is **defined but never referenced**
   anywhere in the tree. Fixing only `_CRITIC_STRUCTURED_CONFIG` leaves a misleading
   dead 2048 twin for the next grep to misread. In scope for this step.
6. **NEW**: `load_skill("quant_strategy")` raises `ValueError` (no `## Prompt Template`).
   Harmless in production, but a live landmine for any test that globs `SKILLS_DIR`.
   The new test must exclude it **assertively**, never via a `try/except` swallow.
7. **`_skill_gen_config` cap is Claude-path-only.** The brief's risk R2b (claiming (d)
   would newly cap 12 Gemini agents) was **REFUTED** by the adversarial leg:
   `llm_client.py:968-970` merges `bundle.base_config` via `setdefault`, and both
   `_general_vertex` and `_quant_exec_vertex` already carry
   `_enrichment_config = {..., "max_output_tokens": 1024}` (`orchestrator.py:530,566-568,586-590`).
   I independently re-confirmed the `_quant_exec_vertex` half. **There is no Gemini
   output-length reduction to disclose.**
8. **gap5-10 severity sharpened**: the bad stem is **fail-open, not a crash**.
   `fid_map.get("sector_agent")` misses -> `return None` -> the inline-skill path runs.
   The Sector agent's prompt is correct today; what is silently lost is only the
   phase-25.D9 Files-API token saving for that one agent. A silent **cost** regression --
   which is exactly why a startup stem-exists assertion is the right remedy.
9. **Criterion 5 breaks three live assertions** in `tests/verify_phase_25_D9_1.py`
   (claims 3, 4, 5 at :68-95). This is a legitimate contract change to a prior phase's
   verifier; it MUST be updated in the same commit and disclosed, or a previously-green
   verifier silently goes red. The stale docstring at `orchestrator.py:919-921` ("The
   `None` return is the safe fallback") must be rewritten too.
10. **item (f) implementation constraint**: the check MUST compare kwargs against the
    **extracted** template, never the raw file. A raw-file diff produces false positives
    on `moderator_agent` and `risk_judge`, whose `## Data Inputs` prose documents bare
    `{{debate_history}}` / `{{past_memory}}` / `{{devils_advocate}}` tokens that are not
    real placeholders. Only the kwarg-with-no-placeholder direction is in scope; the
    inverse direction would fire on intentionally-empty conditional sections.

### 1c. External basis for the (c) design decision

The step offers two arms for gap5-03. **This contract takes the "raise to >= 6144" arm**
and treats the branch fix as non-optional regardless.

- Anthropic official docs (read in full): *"Treat max_tokens truncation as a retriable
  error for structured outputs, not as a valid response to parse"* and *"Do NOT treat
  truncated responses as success."* The structured-outputs doc adds that on truncation
  the response is HTTP 200 but *"may be incomplete and NOT match your schema."* This
  directly condemns the `treating as PASS with draft` branch, which is **fail-OPEN**: the
  quality gate disappears rather than fails.
- The `>= 6144` figure in criterion 4 is doc-sanctioned, not arbitrary: Anthropic's
  sizing rule is *"max_tokens at least 1.5-2x your expected output size"*; expected
  output is a full `SynthesisReport` budgeted at 4096, and 1.5 x 4096 = 6144.
- **Why not patch semantics**: the delta/patch recommendation in the structured-outputs
  doc is motivated by grammar-compilation cost and schema-complexity limits, **not** by
  truncation. Taking that arm would require changing `CriticVerdict.corrected_report`
  (`schemas.py:66`), rewriting `critic_agent.md:66/71/103/108`, and adding merge logic at
  two separate accept paths (`orchestrator.py:1535-1538` and `:1544-1547`) on the live
  synthesis path -- versus a one-integer edit. Patch semantics will be **queued as its
  own masterplan step**, per the standing rule that out-of-scope defects get their own
  research-gated step rather than a prose disclosure.
- **In-repo prior art for the branch fix**: `llm_client.py:1653-1684` already handles
  `stop_reason == "max_tokens"` by retrying once with `min(max_tokens*2, 8192)` at :1664.
  The critic parse-fail branch will mirror this existing idiom rather than invent a new one.

---

## 2. Hypothesis

Five independent defects cause skill-prompt content to silently fail to reach the model,
and one causes a quality gate to silently fail open. Each is a *silent* failure -- nothing
crashes, nothing logs an error, and the pipeline reports success. Repairing the loader
inputs (a, b), the output budgets (c, d), the stem resolution (e), and adding a
kwarg-mismatch warning (f) restores the delivered prompt to what the skill files were
written to say, and converts the critic's fail-open branch into a fail-flagged one.

The load-bearing risk is **not** the fix -- it is that the fix's tests are canary-substring
assertions, the exact shape that produced five consecutive vacuous-guard incidents in
75.2/75.2.1/75.3. The mutation matrix in §5 is therefore a first-class deliverable, not a
formality.

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

> 1. New backend/tests/test_phase_75_skill_delivery.py passes offline and asserts via the REAL load_skill() (not string stubs): load_skill('quant_model_agent') output contains '{{quant_model_data}}' and at least one line of the former Instructions block
> 2. Test asserts the 'Uncertainty Permission' canary phrase appears in load_skill() output for all 8 phase-4.14.26 files and a 'Code Execution' canary appears for the 3 phase-26.3 files -- proving the sections now live inside the extracted region
> 3. Test asserts format_skill emits a warning (caplog) when passed a kwarg with no matching placeholder, and that the sector call site uses stem 'sector_analysis_agent' with a startup stem-exists assertion in place
> 4. Critic output budget: test asserts _CRITIC_STRUCTURED_CONFIG max_output_tokens >= 6144 OR (patch-semantics: critic skill instructs corrected-fields-only AND the parse site merges patches); the literal log string 'treating as PASS with draft' no longer exists in orchestrator.py
> 5. Enrichment cap: test asserts _skill_gen_config returns a config carrying max_output_tokens=1024 on both the file-id and no-file-id paths
> 6. No skill .md loses content: every relocated section's text is byte-identical, only heading levels/positions change (diff summary in experiment_results.md)

**Verification command** (immutable):

```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_skill_delivery.py -q
```

**live_check** (immutable): `handoff/current/live_check_75.4.md` -- verbatim output of the
verification command (exit 0) + `git diff --stat` proving the change surface; ON-vs-OFF $0
diff for any flag-gated live-loop behavior; Playwright/curl capture for UI-touching parts.
Findings covered: gap5-01, gap5-02, gap5-03, gap5-06, gap5-10.

*No criterion is amended. Criterion 4's OR is resolved to its first arm (see §1c);
criterion 2's "3 files" is satisfied as written (see §1b.3).*

---

## 4. Plan

### (a) gap5-01 -- loader truncation in `quant_model_agent.md`
Demote `## Quant Model Data` (:78) and `## Instructions` (:81) to `### `. Body text
byte-identical. Result: the full template ships with `{{quant_model_data}}` present.

### (b) gap5-02 -- undelivered sections
Relocate, **byte-identical body text**, into the `## Prompt Template` region:
- `Uncertainty Permission` + `Empty-bracket retraction format` -- 8 files:
  `synthesis_agent`, `critic_agent`, `moderator_agent`, `risk_judge`, `deep_dive_agent`,
  `scenario_agent`, `quant_model_agent`, `bias_detector`.
- `Code Execution Tasks` -- 3 files: `enhanced_macro_agent`, `quant_model_agent`,
  `scenario_agent`.
- **`quant_strategy.md` is explicitly NOT touched** (§1b.3).
Headings demoted to `### ` on relocation so they cannot re-terminate the region.

### (c) gap5-03 -- critic budget + fail-open branch
- `_CRITIC_STRUCTURED_CONFIG.max_output_tokens`: 2048 -> **6144**.
- `_THINKING_CRITIC_CONFIG` (dead twin, §1b.5): raise to 6144 **and** annotate as
  currently-unreferenced, so the two configs cannot drift.
- Replace the `treating as PASS with draft` branch with the in-repo idiom
  (`llm_client.py:1656-1684`): retry once at a raised budget; on continued parse failure
  log a distinct warning and set an explicit `critic_degraded` flag so the report proceeds
  **flagged**, never silently blessed. The literal string is removed.

### (d) gap5-06 -- enrichment cap provider-independence
`_skill_gen_config` returns `max_output_tokens: 1024` on **both** paths (file-id and
no-file-id). Gemini no-op (§1b.7); real fix on the Claude rail (`llm_client.py:1348`
default 2048 -> honors 1024). Update `tests/verify_phase_25_D9_1.py` claims 3/4/5 and the
stale docstring at `orchestrator.py:919-921` in the same commit (§1b.9).

### (e) gap5-10 -- stem typo + startup assertion
`orchestrator.py:1267`: `"sector_agent"` -> `"sector_analysis_agent"`. Add a startup
assertion that every stem passed to `_skill_gen_config` resolves to a real file in
`backend/agents/skills/`. All 12 call sites enumerated and checked.

### (f) `format_skill` kwarg-mismatch warning
Log a warning when a kwarg has no matching `{{placeholder}}` in the **extracted** template
(§1b.10). Blast radius is small -- `format_skill` has no callers outside `prompts.py`.

### Test
New `backend/tests/test_phase_75_skill_delivery.py`, offline, asserting through the
**real** `load_skill()` -- never a string stub. Excludes `quant_strategy.md` and
`SKILL_TEMPLATE.md` **assertively** (explicit named exclusion + an assertion that the
exclusion list is exactly what is expected), never a `try/except` swallow.

---

## 5. Mutation matrix (MANDATORY -- a guard that cannot fail does not count)

Grounded in arXiv 2301.12284's kill-criterion: *an assertion no mutant falsifies is by
definition weak.* Per the standing durable rule from 75.2.1 -- **mutate the fixture too,
not just the code under test, and mutate first the guard you catch yourself defending.**

| id | Mutation | Must fail |
|---|---|---|
| M1 | Restore `## Quant Model Data` to H2 in `quant_model_agent.md` | criterion 1 test |
| M2 | Restore `## Instructions` to H2 | criterion 1 test |
| M3 | Move **ONE** file's `Uncertainty Permission` section back out (not all 8) | criterion 2 test -- proves the assertion is not `any()`-shaped |
| M4 | Move **ONE** file's `Code Execution` section back out (not all 3) | criterion 2 test |
| M5 | Revert `_CRITIC_STRUCTURED_CONFIG` to 2048 | criterion 4 test |
| M6 | Reinstate the literal `treating as PASS with draft` string | criterion 4 test |
| M7 | Revert the stem to `"sector_agent"` | criterion 3 test |
| M8 | Remove the `format_skill` warning | criterion 3 caplog test |
| M9 | Remove `max_output_tokens` from the **NO-file-id** return only | criterion 5 test -- proves BOTH paths are covered, not just the easy one |
| M10 | Swap the real `load_skill` for a string stub in the test | the test must break -- proves criterion 1's anti-stub clause is real, i.e. mutating the **harness**, not the product |
| M11 | Remove the startup stem-exists assertion | criterion 3 test |

M9 and M10 are flagged by the research as the two highest-risk-of-being-skipped. M10 in
particular mutates the test harness itself -- the exact level that escaped detection in
75.2.1.

---

## 6. Risks (carried from research, adversarially filtered)

- **R1 (HIGH)** -- `skill_optimizer.apply_modification` (`skill_optimizer.py:397-459`)
  rewrites skill files autonomously via `old_text` -> `new_text` replacement, guarded only
  by an occurs-exactly-once check (:409-421); the phase-71.4 independent review gate
  (:429-442) is flag-gated and **DARK**. An autonomous run can re-break the headings.
  Relocating content *into* the template region newly exposes it to the auto-optimizer.
  Mitigation: the post-write `load_skill()` validation already exists at :455-462; this
  step's test makes a regression detectable. Full invariant-check hardening is out of
  scope -> queue as its own step.
- **R2 (HIGH, measured)** -- criterion 5 breaks three live assertions in
  `tests/verify_phase_25_D9_1.py`. Must be updated in the same commit and disclosed.
- **R2b -- REFUTED**, do not act on it. See §1b.7.
- **R4 (MEDIUM)** -- `load_skill("quant_strategy")` raises; any naive glob-over-all-skills
  loop crashes. Assertive exclusion required.
- **R5 (MEDIUM)** -- vacuous-guard relapse. The specific trap: asserting a canary against
  `Path.read_text()` rather than `load_skill()` output proves **nothing**, because the
  phrase is already in all 8 files today and always was. §5 M3/M4/M10 exist to kill this.
- **R6 (LOW)** -- relocation increases every affected delivered prompt. Quantify new
  delivered lengths in `experiment_results.md`.
- **R7 (LOW)** -- the `bias_detector.md` fix is cosmetic (orphan file). Must not be
  reported as a behavioral improvement.

## 7. Out of scope -> queue as their own steps

Per the standing rule (auto-memory `feedback_queue_discovered_defects_in_masterplan`), each
gets its own research-gated masterplan step, written for an executor with no memory of this
discovery -- never a prose-only disclosure:
1. Critic patch/corrected-fields-only semantics (the arm not taken in §1c).
2. `skill_optimizer` post-write delivery-invariant check (R1).
3. `quant_strategy.md` missing `## Prompt Template` / `load_skill` contract mismatch (§1b.6).
4. `bias_detector.md` orphan-file disposition -- delete or wire up (§1b.2).

## 8. References

- `handoff/current/research_brief_75.4.md` (research gate, `wf_10cb9956-68a`)
- Anthropic docs: max_tokens / truncation handling; structured outputs & constrained decoding
- arXiv 2301.12284 (mutation-testing kill criterion)
- In-repo: `llm_client.py:1653-1684` (retry-on-truncation idiom); `prompts.py:976-982`
  (phase-32.3 inverse-direction regression); `.claude/rules/backend-agents.md` (documented
  output caps)
- `handoff/harness_log.md` Cycle 130 -- "THE LESSON, FINALLY GENERALIZED" (vacuous guards)
