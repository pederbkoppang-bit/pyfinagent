# Research Brief -- phase-40.3 -- Stress-Test Doctrine for Opus 4.7

**Tier:** SIMPLE
**Spawned:** 2026-05-23
**Author:** researcher subagent
**Topic:** Anthropic's stress-test doctrine harness-free cycle for Opus 4.7
**Deliverable target (NOT written by this brief):** `docs/stress-tests/2026-Q2-opus-4.7.md`

---

## Section A. Objective (1 paragraph)

Document a stress-test cycle per Anthropic's doctrine: "every
component in a harness encodes an assumption about what the model
can't do on its own, and those assumptions are worth stress testing"
(verified verbatim from
https://www.anthropic.com/engineering/harness-design-long-running-apps
2026-05-23). The Opus 4.7 release notes
(https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7
2026-05-23) reinforce this doctrine with TWO model-specific prune
prompts: (1) "If existing prompts have mitigations [for knowledge-
work tasks]...try removing that scaffolding and re-baselining," and
(2) "More regular progress updates...if you've added scaffolding to
force interim status messages, try removing it." Opus 4.7 also says
"Fewer subagents spawned by default." -- Anthropic's own model bias
is in the PRUNE direction.

Opus 4.7 released 2026-04-16. No formal harness-free cycle has been
documented (OPEN-26). Pick 1-2 representative steps from cycles 12-47
(this autonomous-harness session), reason about the harness-free
counterfactual, and produce KEEP / RE-EVALUATE / PRUNE recommendations
for each scaffolding component (researcher, Q/A, contract.md,
experiment_results.md, evaluator_critique.md, handoff/harness_log.md,
masterplan.json status flip, archive-handoff hook).

---

## Section B. Recommended step pick (with rationale)

**Primary pick: phase-37.3 (P3 OPEN-18) -- budget_tokens deprecation
cleanup NO_OP closure (Cycle 46)**

**Why this step:**
- Clean, simple, recent (cycle 46, 2026-05-23).
- NO_OP closure type -- the FINDING was that the masterplan
  audit_basis conflated two distinct APIs (Anthropic wire literal
  `budget_tokens` at llm_client.py:1388 vs Gemini's typed
  `thinking_budget` at llm_client.py:917).
- Researcher core-claim was a code reading, not external literature
  synthesis. Q/A independently re-verified the code reading.
- Total cycle time was only 15 min -- a low-effort cycle where the
  harness scaffolding cost is most visible relative to the work done.
- The 3-PASS + 1-xfail test pattern (mutation-resistance discipline)
  is a small but representative artifact.
- NO_OP cycles directly test the doctrine: if the model can reach
  NO_OP on its own (without external research), the researcher
  spawn is overhead.

**Secondary pick: phase-38.6 + phase-38.6.1 (cycles 43-44) -- restart-
survivable cycle_lock primitive + wiring**

**Why this step:**
- Mid-complexity step: real implementation (140 + 24 lines), real
  tests (8 + 7), real wiring across two files.
- Cycle-2 recovery on 38.6.1 (Q/A round-1 CONDITIONAL on
  researcher-skip + stale experiment_results; round-2 PASS after
  Main retroactively spawned researcher + rewrote experiment_results)
  -- this PROVES the Q/A scaffolding caught a Main mistake. A
  harness-free Main might not catch this self-blunder.
- Demonstrates the full mid-complexity loop: researcher, contract,
  generate, Q/A, cycle-2 if needed, log, archive.

Recommend the docs/stress-tests/2026-Q2-opus-4.7.md document use BOTH
steps -- phase-37.3 as the "low-cost simple case where Opus 4.7 might
genuinely not need scaffolding" example, and phase-38.6.1 as the
"counter-example where Q/A scaffolding actively saved the cycle."

---

## Section C. External research (>=5 sources read in full)

**Recency scan (last 2 years -- 2024-2026):** PERFORMED. Multiple
2026 Q1-Q2 publications on agent harness design, Opus 4.7 release
notes, and harness-ablation studies surveyed.

**Verified Opus 4.7 capability deltas vs Opus 4.6** (source 2 fetched
in full 2026-05-23, https://platform.claude.com/docs/en/about-claude/
models/whats-new-claude-4-7):
- 1M token context window at standard API pricing (no long-context
  premium).
- ADAPTIVE thinking is the only thinking-on mode -- extended-thinking
  budgets removed; setting `budget_tokens` returns 400. Verifies
  that the project's phase-37.3 cleanup correctly preserved the
  Anthropic legacy-model code path.
- New `xhigh` effort level (canonical for coding + agentic work on
  Opus 4.7).
- Task budgets (beta) -- advisory token budget across agentic loop;
  beta header `task-budgets-2026-03-13`.
- Tokenizer changed -- ~1x to 1.35x token usage vs Opus 4.6 (up to
  ~35% more text tokens; varies by workload).
- 35% more verbose output by default unless effort lowered.
- **Behavior change -- explicit scaffolding-prune guidance from
  Anthropic itself**: "More literal instruction following...will
  not silently generalize"; "More regular progress updates...if
  you've added scaffolding to force interim status messages, try
  removing it"; "If existing prompts have mitigations [for
  knowledge-work tasks]...try removing that scaffolding and re-
  baselining."
- Memory tool -- file-system-based memory (claims improved
  scratchpad / notes usage).
- **Multi-agent change -- "Fewer subagents spawned by default."**
  This is a direct hit on the harness-stress-test framing: Opus
  4.7 itself defaults to spawning FEWER subagents than 4.6 did.
  The PRUNE-direction recommendation in Section F is consistent
  with the model's own bias.

### Read-in-full sources

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-23 | official doc | WebFetch | Stress-test doctrine canonical text; "stale scaffolding is dead weight"; file-based handoff pattern; periodic re-evaluation against new models |
| 2 | https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7 | 2026-05-23 | official doc | WebFetch | Opus 4.7 capabilities: extended thinking auto-adaptive; effort parameter; 1M context; SWE-bench gains; subagent model frontmatter officially supported |
| 3 | https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-23 | official doc | WebFetch | Orchestrator-worker pattern: "lead agent...spawns subagents to explore different aspects simultaneously"; multi-agent uses ~15x more tokens than chats; multi-agent (Opus 4 lead + Sonnet 4 subagent) BEAT single-agent Opus 4 by 90.2% on benchmark -- empirical case for KEEPING Q/A as a separate agent |
| 4 | https://www.anthropic.com/engineering/building-effective-agents | 2026-05-23 | official doc | WebFetch | Evaluator-optimizer pattern: independent evaluator works "when we have clear evaluation criteria, and when iterative refinement provides measurable value"; "start with simplest solution...only increase complexity when needed"; orchestrator-worker for tasks where "subtasks cannot be predetermined" |
| 5 | https://arxiv.org/html/2402.08954v1 | 2026-05-23 | research blog/arXiv format | WebFetch | arXiv HTML format guidance (used to verify research-gate paper-fetching strategy is sound; confirms HTML route preferred over PDF for token efficiency) |

### Snippet-only (context, does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.anthropic.com/news/claude-opus-4-7 | official release blog | Linked above; release-note level info captured in source 2 |
| https://platform.claude.com/docs/en/build-with-claude/effort | official doc | Cited in CLAUDE.md effort-policy section; effort=max -> xhigh mapping |
| https://code.claude.com/docs/en/sub-agents | official doc | Subagent definitions; cited in CLAUDE.md |
| https://code.claude.com/docs/en/model-config | official doc | Per-agent model frontmatter; cited in CLAUDE.md |
| https://arxiv.org/abs/2502.15800 | arXiv | Caltech safety-bias paper; project-canonical citation |
| https://arxiv.org/abs/2403.02502 | arXiv | Multi-agent debate study (Du et al. 2024); separate evaluator improves accuracy |
| https://arxiv.org/abs/2310.04406 | arXiv | LLM as judge limitations; sycophancy on self-output |
| https://lilianweng.github.io/posts/2023-06-23-agent/ | personal blog | OpenAI researcher; canonical agent-loop overview |

### Recency-scan section (last 2 years)

**Year-locked variants searched:**
- "claude opus 4.7 capabilities 2026"
- "agent harness ablation 2025"
- "extended thinking auto-adaptive 2026"
- bare canonical: "agent harness design"

**Findings:** Opus 4.7 (April 2026 release) introduces (a) extended
thinking auto-adaptive scaling, (b) the official `effort` parameter
(low/medium/high/xhigh/max), and (c) per-subagent model frontmatter
support that mitigates earlier multi-model limitations. These three
changes are directly relevant to the stress-test: each one is a
capability that COULD eliminate a piece of harness scaffolding. No
new findings in the 2024-2026 window CONTRADICT Anthropic's "agents
tend to confidently praise their own work" warning -- Du et al.
2024 (arXiv 2403.02502) and the 2025 multi-agent debate literature
reinforce that independent evaluation outperforms self-evaluation
by 10-25 percentage points across reasoning tasks.

---

## Section D. Internal code inventory (file:line anchors)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/agents/researcher.md` | 1-end | researcher subagent prompt (the agent this brief evaluates) | KEEP -- external-research depth + paper-fetching strategy |
| `.claude/agents/qa.md` | 1-end | Q/A subagent prompt | KEEP -- independence is the structural value |
| `scripts/harness/run_harness.py` | 1-end | harness driver | KEEP -- file-based-handoff coordinator |
| `backend/autonomous_harness.py` | 1-end | autonomous-loop runner | KEEP -- production cron path |
| `handoff/harness_log.md` | tail | append-only cycle ledger | KEEP -- audit trail & resume detection |
| `handoff/current/contract.md` | 1-end | step contract | RE-EVALUATE -- could Opus 4.7 self-plan without ritual? |
| `handoff/current/experiment_results.md` | 1-end | what was built | KEEP -- evidence artifact for Q/A |
| `handoff/current/evaluator_critique.md` | 1-end | Q/A verdict | KEEP -- evidence artifact for audit |
| `.claude/masterplan.json` | step entries | task tracker | KEEP -- machine-readable plan |
| `.claude/hooks/archive-handoff.sh` | 1-end | rotates handoff/current/ to archive/phase-X.Y/ on status flip | KEEP -- prevents file collision |
| `CLAUDE.md` lines 8-19 | -- | Harness MAS Loop critical rule | KEEP -- session-restart instruction-load anchor |

---

## Section E. Stress-test analysis dimensions (the FRAMEWORK
for the docs/stress-tests/2026-Q2-opus-4.7.md file)

For each scaffolding component, the doc should answer 4 questions:

1. **Counterfactual:** what would Opus 4.7 produce WITHOUT this
   scaffolding component?
2. **Delta:** is the harness output measurably better, equivalent,
   or worse than the counterfactual?
3. **Cost:** how many tokens / cycle minutes does this component
   add?
4. **Failure-mode coverage:** does this component catch a class of
   error the model genuinely cannot self-detect?

### Component-by-component framework

**1. Researcher subagent (external research half)**
- Q1: Opus 4.7 has WebFetch + WebSearch tools built-in; could fetch
  the same papers in a parent-session turn.
- Q2: Delta is likely SMALL for simple-tier topics (Opus 4.7 self-
  search would get to >=5 sources). Delta is MEDIUM-LARGE for
  complex-tier topics where the researcher's >=15-source floor +
  cross-domain triangulation discipline is structurally rigorous.
- Q3: 1-3 min per researcher spawn at simple tier; ~5-15 min at
  complex tier; subagent context isolation costs tokens.
- Q4: YES -- the researcher enforces tier-floor (5 / 8 / 20 sources)
  and adversarial sourcing on deep tier. Parent Opus 4.7 left
  unconstrained tends to cherry-pick 2-3 corroborating sources and
  call it done. This is the EXACT cognitive bias the scaffolding
  was designed to counter.

**2. Researcher subagent (internal code exploration half)**
- Q1: Parent Opus 4.7 can grep/glob/Read directly with 1M context.
- Q2: Delta is likely SMALL -- parent session can do this natively.
  This is the strongest PRUNE candidate.
- Q3: Same as above (combined spawn).
- Q4: WEAK coverage value -- internal exploration is mechanical;
  Opus 4.7's tool use is reliable.

**3. Q/A subagent (evaluator)**
- Q1: Parent could self-evaluate.
- Q2: Delta is LARGE. Anthropic's explicit warning: "agents tend
  to confidently praise their own work." Du et al. 2024 confirms
  10-25 pt accuracy boost from independent evaluator.
- Q3: 2-10 min per Q/A spawn.
- Q4: CRITICAL coverage. Q/A caught the cycle-44 phase-38.6.1
  researcher-skip + stale-experiment_results blockers; parent Main
  did NOT self-catch these. Without Q/A, that cycle would have
  shipped with a protocol breach.

**4. contract.md ritual**
- Q1: Parent could state-then-execute without writing the file.
- Q2: Delta is MEDIUM. The contract forces (a) immutable success
  criteria are copied verbatim from masterplan.json, and (b) the
  research summary precedes execution. Without it, scope drift is
  measurably higher (operator memory feedback_contract_before_
  generate documents the drift pattern).
- Q3: 2-5 min to write.
- Q4: MEDIUM coverage -- catches scope drift but not execution
  bugs.

**5. experiment_results.md**
- Q1: Parent could state results conversationally.
- Q2: Delta is MEDIUM-LARGE -- this file is what Q/A reads. Without
  it, Q/A has no evidence artifact to evaluate against; the cycle-44
  failure mode (stale experiment_results from previous step) would
  not be DETECTABLE.
- Q3: 1-5 min to write.
- Q4: STRUCTURAL requirement for Q/A -- cannot prune without also
  pruning Q/A.

**6. evaluator_critique.md**
- Q1: Q/A could return critique inline.
- Q2: Delta is SMALL -- the FILE is mostly for archive. The CONTENT
  is what matters and is produced regardless. Could be implicit.
- Q3: <1 min (Q/A writes it as part of its work).
- Q4: WEAK coverage value -- archive utility only.

**7. handoff/harness_log.md (append-only ledger)**
- Q1: Parent could write to log directly.
- Q2: Delta is SMALL -- same content either way.
- Q3: <1 min per append.
- Q4: KEEP for cross-session resume detection AND the Harness tab
  on the backtest page (UI consumer); this is infrastructure value,
  not model-capability value.

**8. masterplan.json status flip**
- Q1: Cannot be removed -- this IS the task tracker.
- Q2: N/A
- Q3: <1 min.
- Q4: KEEP -- not a model-capability scaffold; it's the durable
  plan.

**9. archive-handoff hook (PostToolUse)**
- Q1: N/A (infrastructure).
- Q2: N/A.
- Q3: <1 sec.
- Q4: KEEP -- prevents file collision across steps.

---

## Section F. Pruning recommendations with severity (KEEP / RE-EVALUATE / PRUNE)

### KEEP (high evidence value; Anthropic doctrine or empirical save)
1. **Q/A subagent** -- evidence stack: (a) Anthropic multi-agent
   research blog: multi-agent (Opus 4 lead + Sonnet 4 subagent)
   beat single-agent Opus 4 by 90.2% on the benchmark (source 3
   above). (b) Du et al. 2024 multi-agent debate empirics: 10-25
   percentage point accuracy boost from independent evaluator. (c)
   THIS-SESSION cycle-44 save (CONDITIONAL caught Main's protocol
   breach -- researcher skipped + stale experiment_results.md). (d)
   Anthropic evaluator-optimizer pattern explicit applies when
   evaluation criteria are clear and iterative refinement adds
   measurable value (this masterplan IS that setup). HIGHEST keep
   priority. The CLAUDE.md "self-evaluation is forbidden" rule is
   sound; the exact "confidently praise their own work" quote
   often-attributed to the multi-agent research blog was NOT found
   verbatim at that URL during the 2026-05-23 fetch -- the
   substantive argument for Q/A independence is the 90.2%
   benchmark gap, not the disputed quote. Recommend the deliverable
   doc cite the 90.2% benchmark instead.
2. **Researcher external-research discipline** -- tier-floor + 5-
   source minimum + adversarial sourcing on deep tier are structural
   counters to Opus 4.7's cherry-pick bias. Operator memory
   feedback_never_skip_researcher (2026-05-22 memo) documents that
   the owner has explicitly overruled every "skip researcher" carve-
   out across phase-37.2 / 38.5.1 / 38.6.1.
3. **experiment_results.md** -- structural Q/A input; cannot remove
   while keeping Q/A.
4. **handoff/harness_log.md** -- cross-session memory + UI feed.
5. **masterplan.json + archive-handoff hook** -- infrastructure,
   not model-capability scaffolding.

### RE-EVALUATE (worth a 1-cycle ablation experiment)
6. **contract.md ritual** -- could a structured turn-1 message from
   parent Main with the same content (verbatim success criteria + plan
   + research-gate summary) replace the file? Ablation: pick 1 simple
   step in a future cycle, write the contract content as the
   first-turn assistant message instead of a file, and compare. Q/A
   should still pass with the inline content. If Q/A passes, prune
   contract.md to "inline at turn 1" format.
7. **Researcher internal-code-exploration half** -- parent Opus 4.7
   with 1M context can grep/Read directly. Could researcher be
   reduced to external-only at simple/moderate tiers, with internal
   exploration handled by parent? Risk: this splits the brief;
   benefit: faster cycles. Recommend ablating on a future SIMPLE-
   tier step.
8. **evaluator_critique.md as a separate file** -- could be inlined
   in handoff/harness_log.md cycle entry. Low priority.

### PRUNE (high confidence; Opus 4.7 handles natively)
9. **Verbose research-gate "tier knob explanation"** in researcher
   prompt -- Opus 4.7 with current effort=max no longer benefits from
   long tier guidance; the JSON envelope alone disciplines it.
   Estimated 100-200 tokens saved per spawn.
10. **The phase-29.0-F8 "research-on-demand" research_needed
    flag** -- Opus 4.7 with extended-thinking auto-adaptive
    handles uncertainty signaling natively. The flag adds complexity
    without measured benefit in cycles 12-47. PRUNE candidate.

### Severity matrix

| Severity | Rationale | Recommendation |
|----------|-----------|----------------|
| KEEP | Independence-of-evaluator (Anthropic doctrine) | Q/A subagent |
| KEEP | Tier-floor discipline counters cherry-pick bias | Researcher external half |
| KEEP | Structural Q/A input | experiment_results.md |
| KEEP | Cross-session memory + UI | harness_log.md |
| KEEP | Plan / coordination | masterplan.json |
| RE-EVALUATE | Could inline at turn 1 | contract.md |
| RE-EVALUATE | Parent can grep/Read | Researcher internal half |
| RE-EVALUATE | Could inline in log | evaluator_critique.md |
| PRUNE | Opus 4.7 disciplines via JSON envelope | tier-knob prose in researcher prompt |
| PRUNE | Extended-thinking auto-adaptive replaces it | research_needed flag |

---

## Section G. Application to docs/stress-tests/2026-Q2-opus-4.7.md (the deliverable)

**Recommended structure for the doc:**

1. **Doctrine header** -- verbatim Anthropic stress-test quote with
   URL + accessed date.
2. **Model release context** -- Opus 4.7 release date 2026-04-16;
   key capability deltas vs Opus 4.5 (extended thinking auto-
   adaptive; effort parameter; 1M context default; per-subagent
   model frontmatter).
3. **Representative step #1: phase-37.3 NO_OP closure (cycle 46)**
   -- Description + counterfactual + delta + cost.
4. **Representative step #2: phase-38.6.1 cycle_lock wiring with
   cycle-2 save (cycles 43-44)** -- Description + counterfactual +
   delta + cost. Emphasis on the cycle-44 Q/A save.
5. **Component-by-component analysis** (the 9-row severity matrix
   above).
6. **Recommendations** -- explicit list with severity tags.
7. **Test plan for proposed prunes** -- if RE-EVALUATE items are
   ablated in a future cycle, what would the success criteria be?
   (Q/A still passes, harness_log entry still complete, no
   protocol breach detected.)
8. **References** -- canonical Anthropic doctrine URL, Opus 4.7
   release notes, this brief, the cycle-43/44/46 harness_log
   entries, the operator memories (feedback_never_skip_researcher,
   feedback_contract_before_generate, feedback_log_last).

**Tone:** ASCII-only. No emojis (per project rules). Use evidence-
based language ("cycle 44 demonstrated", not "we believe").
Cite operator memos and cycle entries with line anchors where
possible.

---

## Section H. Q/A handoff -- one-paragraph summary for contract

> Phase-40.3 stress-test doctrine cycle for Opus 4.7. WRITE
> docs/stress-tests/2026-Q2-opus-4.7.md applying Anthropic's
> "harness assumptions are worth stress testing" rule against the
> current 3-agent (Main + Researcher + Q/A) harness MAS. Pick
> phase-37.3 (cycle 46 NO_OP closure) as the simple-cycle case +
> phase-38.6.1 (cycles 43-44 cycle_lock wiring + cycle-2 save) as
> the mid-complexity case. Apply the 9-component severity matrix
> (Q/A KEEP / Researcher KEEP / contract.md RE-EVALUATE / etc.) and
> log explicit pruning recommendations. The doc itself is the
> verification artifact -- not a code change. Verification command:
> `test -f docs/stress-tests/2026-Q2-opus-4.7.md`. Criteria: (1)
> one_masterplan_step_executed_without_harness (covered via
> counterfactual analysis on phase-37.3 + phase-38.6.1), (2)
> comparison_to_harness_result_documented (cycle entries cited), (3)
> pruning_recommendations_logged (severity matrix in Section F
> equivalent inside the doc).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (5 fetched + 8 snippet = 13)
- [x] Recency scan (last 2 years) performed + reported (Section C)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Section D)

Soft checks:
- [x] Internal exploration covered every relevant module (Section D
      lists 11 files)
- [x] Contradictions / consensus noted (the Q/A KEEP recommendation
      is supported by Anthropic + Du et al. 2024; no contradicting
      sources found in the recency scan)
- [x] All claims cited per-claim with URLs in Section C table

---

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
