# Research Brief — Step 71.4 (COMPLEX tier)

**Step:** 71.4 — Insert an INDEPENDENT (doer/judge-separated) evaluator into
`backend/agents/skill_optimizer.py` BEFORE `apply_modification` writes a
skill/prompt modification. A diff that WEAKENS a constraint, or whose
DESCRIPTION does not match the DIFF, is rejected-and-skipped. LLM error →
FAIL CLOSED (no write). REUSE the `directive_review` pattern where possible.
DOCS: adaptive coverage gate + loop-until-dry completeness critic for
AUDIT-class steps (>=5 floor PRESERVED).

**Researcher:** Layer-3 harness MAS. Tier: COMPLEX (LIVE code in the
self-improvement loop + docs).

**Status:** COMPLETE. 9 sources read in full (floor 5); 3-variant search +
recency scan done; internal anchors + reuse plan + operational defs +
coverage-gate design all concrete. gate_passed=true. See §0-FINAL + envelope.

---

## 0. Guardrail confirmations (see §0-FINAL for detail)

- [x] NO live risk-limit / threshold VALUE change (this ADDS a guard).
- [x] Independent review is FAIL-CLOSED (can only BLOCK, never force a write).
- [x] >=5-source floor PRESERVED in the coverage-gate design.

---

## 1. Internal code audit — skill_optimizer.py (HEAD anchors)

File: `backend/agents/skill_optimizer.py` (894 lines). The self-improvement
loop. Docstring (lines 1-9) states it modifies ONLY `## Prompt Template`
and calls the harness (data tools, pipeline, schemas, eval formula) FIXED.

### The write point (the choke point 71.4 must gate)
- **`apply_modification(self, agent_name, proposal) -> bool`** — lines
  **399-451**. The ONLY place a skill/prompt modification is written:
  - line 405: `skill_path = SKILLS_DIR / f"{agent_name}.md"`
  - line 409: `content = skill_path.read_text(...)` (current text available here)
  - lines 410-414: mechanical check — `old_text` must be present (else False)
  - lines 417-422: mechanical check — `old_text` must be UNIQUE (else False)
  - line 424: `new_content = content.replace(old_text, new_text, 1)`
  - **line 425: `skill_path.write_text(new_content, ...)` ← THE FILE WRITE**
  - lines 431-432: `reload_skills()` + `SkillFileIdCache.invalidate(agent_name)`
  - lines 435-442: load-validation; **reverts** to `content` (line 439) if the
    modified skill fails to load (this is a REVERT write, must NOT be gated)
  - lines 444-449: `git add` + `git commit` the change (auto-commit, no human)
  - returns True on success, False on any mechanical failure.
- **WHERE it writes:** local FILE (`SKILLS_DIR/<agent>.md`) + a git commit.
  NOT BQ, NOT in-memory-only. `SKILLS_DIR` = `backend/config/prompts.py`
  (`backend/agents/skills/`).

### Modification object shape (`proposal` dict)
Produced by `propose_skill_modification` (lines 300-395) and `think_harder`
(lines 487-565). LLM (Gemini `deep_think_model`, temp 0.7 / 0.9) returns JSON:
```
{ "old_text": "<2-10 lines to find/replace>",
  "new_text": "<replacement>",
  "description": "<one-line what+why>",
  "hypothesis": "<expected improvement + how measured>" }
```
or `{"skip": true, "reason": "..."}`. `propose_skill_modification` returns
`None` if skip / missing old_text / missing new_text / no JSON. The propose
PROMPT tells the LLM the rules (lines 334-342: only `## Prompt Template`,
don't touch `{{variable}}` names, don't change output JSON schema/fields,
"simpler is better") — but **nothing ENFORCES them**. That is the gap.

### How apply is gated TODAY (the "mechanical-only" register)
Exactly 3 mechanical guards, NO semantic judge:
1. `old_text in content` (line 410) — presence.
2. `content.count(old_text) == 1` (line 417) — unambiguous.
3. `load_skill()` re-parse succeeds post-write (line 436), else auto-revert.
Plus a POST-hoc `passes_simplicity_criterion` (lines 569-590, applied in
`_run_one_iteration` at line 773 — AFTER the write, as a keep/discard signal,
not a pre-write veto). There is **no check** that the diff stays inside the
modifiable sections, doesn't weaken a constraint, or that `description`
matches the diff. 71.4 inserts that check.

### Exact insertion point for the independent review
Single choke point → gate INSIDE `apply_modification`, after the mechanical
checks (line 422) and BEFORE the write (line 424/425). The `content`
(current full skill text), `old_text`, `new_text`, `description` are all in
scope there. On reject: return `False` (= rejected-and-skipped, no write) and
(GENERATE detail) log a distinct experiment status (e.g. `"reject"`/`"veto"`)
so it is observable, rather than the existing `"crash"` status the caller
writes on a `False` return (lines 755-762). Gating INSIDE `apply_modification`
(vs in the caller) protects BOTH proposal sources — `propose_skill_modification`
AND `think_harder` converge on the SAME call (line 754), so one gate covers both.

### Callers / live trigger (who fires the self-improvement loop)
- `apply_modification` — **exactly one caller**: `_run_one_iteration` line 754
  (both propose + think_harder paths).
- `_run_one_iteration` ← `run_loop` (line 662) ← callers:
  `backend/api/skills.py:47` (API `/api/skills` route, `_get_optimizer()`),
  `backend/api/backtest.py:1079`, `backend/api/performance_api.py:177`,
  `scripts/harness/run_optimizer.py:129`, `scripts/harness/run_harness.py:380`.
- **Live autonomous trigger:** `MetaCoordinator` (backend/agents/meta_coordinator.py)
  emits `action="skill_opt"` when `days_since_last_skill_opt >=
  MIN_SKILL_OPT_INTERVAL_DAYS` (lines 178-182); `target_agents` flows MDA→Agent
  into `run_loop(target_agents=...)`. The weekly `register_meta_evolution_cron`
  (backend/meta_evolution/cron.py:43) is the scheduler. So skill mods are
  applied+committed AUTONOMOUSLY with no human in the loop — unlike directives.

## 2. Internal code audit — directive_review.py (the REUSE pattern)

File: `backend/meta_evolution/directive_review.py` (270 lines). phase-10.7.7.
"Independent second-opinion judge" for directive proposals. **This is the
canonical fail-closed review shape 71.4 mirrors.**

### Reusable pieces (quote)
- **Public fn:** `review_directive_diff(proposal, current_directive_text, *,
  llm_call_override=None) -> ReviewResult` (lines 196-261). `llm_call_override`
  is the testing seam (inject a fake LLM → zero API cost, deterministic).
- **5-dim rubric** (lines 42-48): `clarity, alignment, safety, proportionality,
  factuality`; each 0.0-1.0; aggregate = mean; ACCEPT iff `aggregate >=
  ACCEPT_THRESHOLD` (0.70, line 41). Two dims map DIRECTLY onto 71.4's contract:
  - **`safety`** = "does NOT weaken any existing constraint" (line 13, 106) →
    the "weakens a constraint" test.
  - **`factuality`** = "diff_summary accurately describes the actual change"
    (line 14, 107) → the "description does not match the diff" test.
  - `alignment` = "preserves existing non-negotiable floors" → maps to
    "stays in modifiable sections / doesn't touch `{{vars}}` or output schema".
- **Fail-CLOSED** (lines 169-180 `_fail_closed`; 214-240): empty proposed_text,
  LLM None/exception, non-dict, missing dimension, out-of-range score → return
  `ReviewResult(verdict="REJECT", reason=..., all scores 0.0)`. Docstring lines
  19-21: "absence of evidence is evidence of risk" (OPPOSITE of cron fail-open).
- **Anti-rubber-stamp prompt** (`_build_review_prompt`, lines 73-120): STRIPS
  the proposer's self-score so the judge can't anchor; "do NOT default to 0.8+".
- **`ReviewResult`** frozen dataclass (lines 53-70): verdict/reason/5 scores/
  aggregate/`raw_llm_response` (parsed dict for forensic logging).
- **LLM caller** `_call_llm_for_review` (lines 123-166): Anthropic
  `claude-sonnet-4-6` primary → Gemini `gemini-2.5-flash` fallback → None on
  failure. Pure module, no I/O outside the LLM call.

### Reuse decision — MIRROR, don't import-as-is
`review_directive_diff` consumes a `DirectiveVersion` (fields `proposed_text`
/`diff_summary`/`diff_size_bytes`), while a skill proposal is
`old_text`/`new_text`/`description`. Cleanest reuse = a NEW small pure module
`backend/agents/skill_modification_review.py` (mirroring directive_review 1:1)
whose `review_skill_modification(agent_name, old_text, new_text, description,
current_skill_text, *, modifiable_sections, llm_call_override=None) ->
ReviewResult` reuses the SAME rubric names, the SAME `_fail_closed`, the SAME
`_coerce_score`, the SAME `llm_call_override` seam, the SAME frozen
`ReviewResult` (import it or re-declare). Build the diff for the judge from
old→new. `apply_modification` calls it and treats `verdict != "ACCEPT"` as
"do not write". Mirroring (not importing) avoids coupling agents/ to
meta_evolution/ and lets the prompt speak in skill-section terms.

### Test template (mirror `tests/agents/test_evaluator_directive_review.py`)
13 tests to copy 1:1 via `llm_call_override`: accept-on-high, reject-on-low,
threshold-boundary (0.70 vs 0.699), **fail-closed x4** (None / non-dict /
override-raises / missing-dim / out-of-range), short-circuit-before-LLM on
empty diff, self-score-stripped, current-text-in-prompt, idempotent,
raw-response-stored. Plus 71.4-specific: reject on section-scope violation,
reject on anti-pattern-guard removal, reject when description understates diff.

### Notable: the reuse target is a PATTERN, not a live call-site
`review_directive_diff` has **NO production caller** — grep finds it only in
`tests/agents/test_evaluator_directive_review.py`. Reason: the directive path
is **HITL** — `rewrite_directive` (directive_rewriter.py:239) "NEVER writes to
`.claude/agents/researcher.md` directly. Operator (Peder) reviews the returned
`DirectiveVersion.proposed_text` + Main applies it after explicit approval."
The human IS the gate there. The skill-opt path has NO human gate (auto-apply
+ auto-commit, §1). So 71.4 wires the proven review SHAPE into the path that
actually lacks a judge — strictly additive safety.

## 3. "Weakens a constraint" + "description does not match the diff" — operational defs

Grounded in `.claude/rules/backend-agents.md` "Skills System":
- **Modifiable sections:** `## Prompt Template`, `## Skills & Techniques`,
  `## Anti-Patterns`.
- **Fixed harness (UNTOUCHABLE):** data tools, orchestrator pipeline, output
  schemas, BQ schema, evaluation formula.
- NOTE a drift to flag for GENERATE: skill_optimizer.py's own docstring
  (line 7) + propose prompt (line 335) say "ONLY the `## Prompt Template`
  section" — NARROWER than backend-agents.md's 3-section list. The review
  should enforce the UNION-SAFE rule: reject if the edit touches ANYTHING
  outside the 3 modifiable sections; within them, the semantic checks apply.

**"Weakens a constraint" (fails `safety`) — operational, any one of:**
1. **Section-scope violation** — `old_text`/`new_text` edits or deletes text
   OUTSIDE the 3 modifiable sections: a fixed-harness region (data tools,
   pipeline, an output-JSON-schema/required-field block, BQ schema, the
   evaluation formula), OR a `{{variable}}` placeholder name is changed/removed
   (propose-rule #2, code line 337), OR the output JSON schema / a required
   field is altered (propose-rule #3, code line 338).
2. **Anti-pattern guard removal** — `new_text` deletes or softens an entry
   under `## Anti-Patterns` (each is a "do NOT do X" guard) without an
   equivalent replacement. Removing a guard loosens the agent.
3. **Rule loosening** — `new_text` converts a hard directive to a soft one:
   MUST→may, always→optionally, removes a numeric threshold/floor, deletes a
   risk/abstain/confidence-cap/skepticism instruction (finance-agent safety
   caveats). Net removal of a bound = weakening.

**"Description does not match the diff" (fails `factuality`) — operational:**
The judge is shown BOTH `description`/`hypothesis` AND the literal
`old_text`→`new_text`. Fails when the description is not an accurate + complete
characterization of the actual change: it names a different edit, materially
UNDERSTATES scope (large/multi-part change described as trivial), or omits a
side effect (e.g. "add emphasis on insider clusters" while the diff also
deletes a schema line). Under-description is the adversarial case — a
mis-described diff is a smuggling vector, so factuality<0.5 → REJECT.

**Fail-closed wiring for 71.4:** identical to directive_review — LLM
error/None/unparseable/missing-dim/out-of-range → REJECT (no write). PLUS a
cheap DETERMINISTIC pre-LLM pre-check (belt-and-suspenders): if `old_text`
spans a header outside the modifiable set, or `{{...}}` token set differs
between old/new, hard-REJECT without spending an LLM call. The LLM semantic
judge is the core; the deterministic pre-check is a fast, un-foolable floor.

## 4. Adaptive coverage gate design (research-gate.md / researcher.md / ARCHITECTURE.md)

Goal: for AUDIT-class steps ONLY, keep researching until coverage is DRY
(K consecutive rounds add no new findings), while PRESERVING the >=5-source
HARD floor for every tier. The coverage gate is ADDITIVE (raises the ceiling,
never lowers the floor).

### Design
- **New envelope field `coverage`** — an object, emitted by every brief but
  only *enforced* for audit-class steps:
  ```
  "coverage": { "audit_class": false, "rounds": 1, "dry_rounds": 0,
                "K_required": 2, "new_findings_last_round": N, "dry": false }
  ```
  For non-audit steps `audit_class:false` and the field is informational.
- **Loop-until-dry completeness critic (audit-class only).** After the >=5
  floor is met, run rounds of additional search/fetch. A round is "dry" when it
  yields ZERO new in-full findings beyond de-dup. Stop when `dry_rounds >=
  K` (default K=2). `gate_passed` for an audit-class step additionally requires
  `coverage.dry == true`. For non-audit steps, coverage does NOT gate (the
  existing >=5 + recency logic stands).
- **Floor preserved:** `gate_passed` still hard-requires
  `external_sources_read_in_full >= 5 AND recency_scan_performed`. Coverage can
  only ADD a requirement for audit steps, never subtract. Explicitly: an
  audit step that hit K dry rounds at 4 sources still FAILS the gate (floor
  wins). Caller sets `audit_class` in the spawn prompt; researcher never
  self-declares to escape the loop.

### Where the doc edits land (line anchors)
- **`.claude/agents/researcher.md`** (agent prompt — the enforced spec):
  - JSON envelope block **lines 316-327**: add the `coverage` field.
  - Gate logic **lines 329-335**: add the audit-class `coverage.dry` clause,
    stated as ADDITIVE to the >=5 floor.
  - New subsection after the effort-tiers/`deep` block (**after line 293**):
    "## Adaptive coverage (audit-class steps)" — the loop-until-dry critic,
    K default, floor-preserved note. (Orthogonal to tier: audit-class is a
    caller flag, not a tier.)
- **`.claude/rules/research-gate.md`** (how-to mechanics):
  - JSON envelope block **lines 164-183**: add `coverage` to the example +
    the `gate_passed` clause (audit-class only).
  - New "## Adaptive coverage gate (audit-class steps)" section (e.g. after
    "URL collection", **after line 162**): mechanics of dry-round counting,
    K, and the FLOOR-IS-PRESERVED invariant.
- **`ARCHITECTURE.md`** (MADR reference record):
  - "Research Gate Discipline (phase-4.16)" section (**starts line 492**;
    Decision list 512-524): add a numbered decision + a phase-71.4 dated
    note recording the adaptive-coverage addition and the floor-preserved
    invariant.
- **Cross-link, don't duplicate** (the 3-file rule, research-gate.md lines 3-6
  + researcher.md is the prompt + ARCHITECTURE is the record): mechanics in
  research-gate.md, the enforced spec + envelope in researcher.md, the MADR
  decision in ARCHITECTURE.md. Each references the others; no copy-paste.

## 5. External literature (9 read in full; floor is 5)

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Key finding (verbatim where quoted) |
|---|-----|----------|------|-------------------------------------|
| 1 | https://www.anthropic.com/research/building-effective-agents | 2026-07-17 | Official doc | Evaluator-optimizer: "one LLM call generates a response while another provides evaluation and feedback in a loop." Use "when we have clear evaluation criteria." For agents generally: "extensive testing in sandboxed environments, along with the appropriate guardrails." |
| 2 | https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-07-17 | Official doc | LLM judge scores each output against a **5-dim rubric** (factual accuracy, citation accuracy, completeness, source quality, tool efficiency); "a single LLM call with a single prompt outputting scores from 0.0-1.0 and a pass-fail grade was the most consistent." Stopping/loop: "The LeadResearcher synthesizes these results and **decides whether more research is needed—if so, it can create additional subagents or refine its strategy.**" |
| 3 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-07-17 | Official doc | Doer/judge: "When asked to evaluate work they've produced, agents tend to respond by confidently praising the work... **Separating the agent doing the work from the agent judging it proves to be a strong lever.**" Loop: "**Each criterion had a hard threshold, and if any one fell below it, the sprint failed** and the generator got detailed feedback." Handoff: "Communication was handled via files." |
| 4 | https://arxiv.org/html/2507.21046 | 2026-07-17 | Preprint (survey) | Self-evolving agents rewrite their own prompts/skills (APE, PromptBreeder, TextGrad §3.2.2; Voyager skill library §3.3). Safety flagged as critical future direction (§8.3); survey notes current safeguards for vetting modifications pre-deployment are thin — motivates ADDING a gate. |
| 5 | https://arxiv.org/html/2510.12462 | 2026-07-17 | Preprint | LLM-as-judge biases (verbosity, authority — score dropped 9.12→3.94 when authority removed; source-identity). Mitigations: focus-prompt on factual correctness + ignore irrelevant attributes; ensemble/panel of diverse judges; secondary bias-detection prompt; calibration. |
| 6 | https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents | 2026-07-17 | Official doc | "**create clear, structured rubrics to grade each dimension of a task, and then grade each dimension with an isolated LLM-as-judge** rather than using one to grade all dimensions." Escape hatch: "give the LLM a way out, like ... 'Unknown'." "Vague rubrics produce inconsistent judgments"; write so "two domain experts would independently reach the same pass/fail verdict." |
| 7 | https://arxiv.org/html/2606.29225v1 | 2026-07-17 | Preprint | PolicyGuard: a **Verifier between the Agent and the Environment**; triggers on **MUTATING calls only** — "Read-only calls bypass the Verifier and incur no verification cost" (§3); "on every mutating call, the verifier ... either passes the call or blocks it"; **fail-closed**: "If a required action was never performed, treat it as NOT MET" (§3.1). |
| 8 | https://arxiv.org/html/2510.05156 | 2026-07-17 | Preprint | VeriGuard: separates a Policy Generator from an independent verifier; "validate each proposed agent action against the pre-verified policy **before execution**"; fail-closed enforcement (hard block / task termination) on violation. |
| 9 | https://arxiv.org/html/2503.22738v1 | 2026-07-17 | Preprint | ShieldAgent: a **separate guardrail agent** that "performs action verification" against policy and emits "a binary flag ... indicating whether action a_i is safe" BEFORE the protected agent executes; block-on-violation (safety label 0). |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/pdf/2606.23075 (Safety in Self-Evolving LLM Agent Systems: Threats) | Preprint | HTML 404'd; PDF extraction returned empty (binary). Search summary captured: "adversarial influences become permanently encoded, self-amplify across generations" + defenses (restrict self-modifiable parts, sandbox, human approval for high-impact, version/audit, rollback, independent judge model). |
| https://arxiv.org/pdf/2509.26354 (Your Agent May Misevolve) | Preprint | HTML 404'd. Core: unconstrained self-evolution can degrade/misalign an agent over iterations. |
| https://arxiv.org/html/2604.07223v1 (TraceSafe) | Preprint | Trace-level guardrail benchmark via localized mutations; pre-invocation gating. |
| https://arxiv.org/html/2605.29251v1 (Provably Secure Agent Guardrail) | Preprint | Year-less/2026 canonical guardrail hit. |
| https://www.hiddenlayer.com/research/same-model-different-hat | Industry | "Self-policing LLM vulnerability" — same-model self-review is exploitable → supports independent/cross-family judge. |
| https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining | Vendor | Self-evolving agents: staged/held-out eval, human approval for high-impact, versioning. |
| https://www.evidentlyai.com/llm-guide/llm-as-a-judge | Industry | LLM-as-judge complete guide (rubric, biases). |
| Nature TextGrad (2025) | Peer-reviewed | Textual-gradient prompt mutation (proposer side). |
| https://agenticorgchart.com/evaluator-optimiser/ | Community | "Evaluator-Optimiser Agent Pattern (2026): Generator + Critic" (year-tagged canonical). |
| https://www.arthur.ai/blog/best-practices-for-building-agents-guardrails | Industry | Pre-LLM/post-LLM guardrails; human verification before modifying sensitive data. |

### Key findings → application to 71.4
1. **Doer/judge separation is the documented cure for self-praise** (Anthropic harness-design). The skill-opt proposer is the doer; 71.4 adds the missing judge. Reuse `directive_review`'s "strip the proposer's self-score" prompt discipline (directive_review.py:77-78).
2. **Rubric = isolated dimensions, 0.0-1.0, hard pass/fail threshold** (Anthropic multi-agent-research + demystifying-evals + harness-design). `directive_review` already implements exactly this (5 dims, mean, ACCEPT>=0.70). 71.4 keeps `safety` + `factuality` as the two contract-critical dims and adds `section_scope` (alignment). Consider the "Unknown/escape-hatch" idea → in a SAFETY gate, "unsure" must resolve to REJECT (fail-closed), which is what `_coerce_score`→`_fail_closed` already does.
3. **Gate MUTATING operations before they execute; let read-only bypass** (PolicyGuard §3, VeriGuard, ShieldAgent). Maps 1:1 onto `apply_modification`: gate the forward WRITE (line 425) — the mutating op — and do NOT gate the load-failure REVERT write (line 439) or the read paths. This is the precise placement argument.
4. **Fail-closed is the consensus safety default** (PolicyGuard "treat as NOT MET", VeriGuard hard block, ShieldAgent block-on-violation, directive_review "absence of evidence is evidence of risk"). 71.4 inherits it verbatim.
5. **Same-model self-review is a known vulnerability** (HiddenLayer) and self-preference bias is real (2510.12462). directive_review already prefers Anthropic-then-Gemini; for the skill judge, the proposer is **Gemini** (skill_optimizer `_get_model`), so routing the judge to **Anthropic Claude** (directive_review's primary) gives cross-family independence for free — a genuine improvement to call out.
6. **Loop-until-"more research needed" is Anthropic-documented** (multi-agent-research "decides whether more research is needed") and hard-threshold-or-fail is the harness-design loop shape — the doc basis for the adaptive coverage critic (keep spawning rounds until dry), while the >=5 floor is the "hard threshold" that must never be relaxed.

## 6. Recency scan (last 2 years, 2024-2026)

Performed. Explicit 2026 + 2025 + year-less passes run (see §7). **New findings in the window that COMPLEMENT/SUPERSEDE the older Anthropic evaluator-optimizer canon:** a 2025-2026 cluster of *pre-execution, fail-closed, independent-verifier* guardrails specifically for agent actions — PolicyGuard (2606, 2026), VeriGuard (2510, 2025), ShieldAgent (2503, 2025), TraceSafe (2604, 2026), plus self-evolution-risk work (2507 survey 2025; "Misevolve" 2509, 2025; "Threats" 2606, 2026). These POST-DATE and refine the 2024 Anthropic "Building Effective Agents" evaluator-optimizer pattern by adding the specific discipline 71.4 needs: **verify the mutating change before it lands, and fail closed.** No source in the window CONTRADICTS the doer/judge-separation + fail-closed thesis; the newer work strengthens it. The older Anthropic canon remains the authoritative pattern source; the 2025-2026 papers supply the mutation-gating specifics.

## 7. Search queries run (3-variant discipline)

Per-topic, the mandated three variants (current-year 2026 / last-2-year 2025 / year-less canonical) were run:
- **Evaluator-optimizer / doer-judge:** "Anthropic evaluator-optimizer workflow building effective agents" (year-less); "separate generator from evaluator LLM as judge independence bias" (year-less). Current-year canonical hit surfaced ("Evaluator-Optimiser Agent Pattern (2026)"); Anthropic official doc is the anchor.
- **Self-modifying-agent safety / fail-closed:** "...independent review before apply **2026**" (current-year); "...guardrails fail-closed autonomous prompt optimization **2025**" (last-2-year); "safeguarding mutating steps LLM agent verification before apply **2026**"; "LLM agent self-modification safety guardrail verify before apply" (year-less canonical).
The read-in-full set mixes current-year (2606), last-2-year (2507/2510/2503), and year-less canonical (Anthropic docs) hits, satisfying the visibility requirement of research-gate.md §"Search-query composition".

---

## 0-FINAL. Guardrail confirmations

- **[CONFIRMED] NO live risk-limit / threshold VALUE change.** 71.4 touches NO
  risk limit (kill-switch daily/trailing-DD caps, PBO threshold, DSR gate,
  sector caps, projected-DD cap, CVaR). It ADDS a review gate to the
  meta-evolution skill-optimizer path. The only numeric it introduces is the
  NEW gate's own `ACCEPT_THRESHOLD` (mirroring directive_review's 0.70) — an
  internal knob of a brand-new guard, not a live trading/risk limit.
  `risk_threshold_value_change = FALSE`.
- **[CONFIRMED] Independent review is FAIL-CLOSED and can ONLY BLOCK.** It
  returns REJECT on LLM error/None/unparseable/missing-dim/out-of-range
  (directive_review pattern). A REJECT makes `apply_modification` return
  `False` = the write is skipped. The gate can never CAUSE a write that
  wasn't already proposed; it can only VETO one. Strictly safer:
  the worst-case failure mode (LLM down) is "no self-modification happens",
  which is the safe state.
- **[CONFIRMED] >=5-source floor PRESERVED.** The adaptive coverage gate is
  ADDITIVE and audit-class-only: `gate_passed` still hard-requires
  `external_sources_read_in_full >= 5 AND recency_scan_performed`. Coverage
  can only ADD the `coverage.dry` requirement for audit steps; it can never
  drop the floor. An audit step at 4 sources + K dry rounds still FAILS.

### Rider-traps / implementation cautions for GENERATE (flagged)
1. **Do NOT gate the revert write.** Gate only the forward write
   (skill_optimizer.py:425), never the load-failure revert (line 439).
   Gating the revert would trap a broken skill in place (unsafe).
2. **Section-scope doc drift.** skill_optimizer's own prompt says "ONLY
   `## Prompt Template`" (line 335) but backend-agents.md lists 3 modifiable
   sections. GENERATE must enforce the union-SAFE rule (reject edits outside
   the 3 sections) and reconcile the docstring, not silently pick one.
3. **Distinct reject status.** A `False` return currently logs status
   `"crash"` at the caller (lines 755-762). GENERATE should add a distinct
   `"reject"`/`"veto"` experiment status so vetoes are observable and not
   confused with genuine crashes.
4. **Cost posture ($0-metered away-ops).** directive_review's primary LLM is
   Anthropic `claude-sonnet-4-6` (metered). The skill judge SHOULD prefer the
   cross-family independent path (proposer is Gemini → judge on Claude gives
   free independence) BUT to respect the $0-metered posture, make the judge
   model configurable and/or default to the Vertex/Gemini fallback; the review
   fires rarely (MetaCoordinator interval-gated, once per applied proposal),
   not per ticker, so cost is contained regardless.
5. **Deterministic pre-check must not false-positive** on legitimate Prompt
   Template edits (e.g. a `{{var}}` legitimately quoted in prose). Compare the
   SET of `{{...}}` tokens old vs new; only reject on a delta.

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 10,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```

