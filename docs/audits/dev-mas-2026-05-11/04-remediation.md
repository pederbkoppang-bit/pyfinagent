# Phase 4 — Remediation

Audit date: 2026-05-11. Scope: dev MAS only. **No application-code
fixes.** Every recommendation traces to a finding in Phases 1-3.

Phase 1 → `01-roster.md` (roster, namespace collisions).
Phase 2 → `02-per-agent.md` (per-agent findings with doc + code
citations).
Phase 3 → `03-symptoms.md` (symptom traces; systemic verification
pattern).

---

## Architectural recommendations (ranked by expected impact)

### R-1 — Promote the "live data + UI check" from convention to hook-enforced gate

**Problem**: Phase 3's systemic pattern. All three symptoms show
verification-stage failure: unit tests pass on synthetic inputs, no
step opens production data / UI to confirm the operator-visible
behavior. Tied to findings Q-5 (Anti-rubber-stamp not hook-enforced),
H-2 (TaskCompleted hook duplicates Q/A but with weaker rubric).

**Proposed change**: Add a per-step `verification.live_check` field
to `.claude/masterplan.json` (new optional field, NOT a change to
existing immutable verification criteria). When set, the
`auto-commit-and-push.sh` hook refuses to push unless a file at
`handoff/current/live_check_<step_id>.md` exists and contains a
verbatim block matching the field's expected shape (e.g., a curl
output, a BQ query result, a screenshot path). Q/A's deterministic
leg reads this file as part of its existing
"existing_results_check" step (`qa.md:85-92`).

**Doc citation supporting the change**: HARNESS-DOC verbatim — "agents
tend to respond by confidently praising the work — even when, to a
human observer, the quality is obviously mediocre" — implies the
evaluator needs evidence the agent cannot fake. A file-based artifact
of curl/BQ output is harder to fake than a prose claim.

**Expected effect on symptoms**:
- Symptom 1: live_check would require a verbatim
  `paper_trader.create_trade` row with `reason='stop_loss'` after
  forcing a synthetic price drop. Currently no step asserts this.
- Symptom 3: live_check on the 23.2.A-fix step would require a BQ
  sample of NEW paper_trades.signals showing the relabel + lite_path
  flag. The current cycle (commit ad9d773c) included unit tests but
  no BQ sample.
- Symptom 2: live_check would require the operator to open the
  drawer on a recent BUY and report what they see. This converts the
  "scoping gap" into a recurring forcing function for the doc-fix
  conversation.

### R-2 — Eliminate the hook-agent / Q/A redundancy

**Problem**: Phase 2 findings H-1, H-2. TaskCompleted hook spawns an
agent with unconstrained tool access (defaults to settings.json
`bypassPermissions`), running an inlined prompt that overlaps with
qa.md's deterministic leg. It can return "ok:true" before qa.md's
rigorous LLM-judgment leg runs.

**Proposed change**: Choose one:

- **Option A — delete the TaskCompleted hook**. qa.md is the
  documented evaluator (`.claude/agents/qa.md:3` "MUST BE USED in
  every EVALUATE phase"). The hook adds noise + cost. Most
  conservative.
- **Option B — promote the TaskCompleted hook to a tools-restricted
  subagent file**. Move the inlined prompt at `settings.json:67`
  into `.claude/agents/task_completed.md` with a frontmatter:
  `tools: Read, Bash` (no Write/Edit), `model: haiku` (cheaper),
  `permissionMode: plan`. Define explicit boundaries vs qa.md (e.g.,
  "your job is fast-checks only; qa.md owns the LLM-judgment
  verdict").

**Doc citation**: SUBAGENT-DOC verbatim — "Each subagent runs in its
own context window with a custom system prompt, **specific tool
access**, and independent permissions." The current hook violates
"specific tool access".

**Expected effect**: Verdicts emitted by the hook will no longer
contradict Q/A. Fewer Bash calls with full project-write permissions.

### R-3 — Rename to break namespace collisions

**Problem**: Phase 1 + Phase 2 finding F-1 / C-2. The labels "Main",
"Researcher", "Q/A" each have **two distinct agents** (one Layer-3
subagent, one Layer-2 in-app role) that share the same name. The
audit prompt itself opens with the user's confusion: "Ford ≟
MultiAgentOrchestrator?"

**Proposed change** (no schema-changing renames; just labels):

- **In-app Layer-2 (`agent_definitions.py`)** — rename for clarity:
  - `Ford (Main Agent)` → `Ford (Slack Orchestrator)`.
  - `Researcher` (in-app) → `Slack Researcher` or `Live Researcher`.
  - `Analyst (Q&A Agent)` is already disambiguated by the "Analyst"
    label; keep it.
- **Layer-3 harness MAS subagents** — keep names (`researcher.md`,
  `qa.md`) and CLAUDE.md's "Main = this Claude Code session".
- **CLAUDE.md "exactly 3 agents" rule** — clarify scope:
  "**The Harness MAS layer is exactly 3 agents.**" (drop the
  bare-noun "MAS"). Add one sentence: "The in-app Slack/iMessage MAS
  in `backend/agents/multi_agent_orchestrator.py` is a separate
  4-agent system (Communication + Ford + Analyst + Live Researcher);
  do not conflate the two."

**Doc citation**: SUBAGENT-DOC verbatim — "Specialize behavior with
focused system prompts for specific domains" implies names should
disambiguate domain.

**Expected effect**: Operator queries like "what did Researcher
say?" become unambiguous; log scanners can filter by layer.

### R-4 — Hardcoded thresholds → runtime config

**Problem**: Phase 2 finding P-1. `planner_agent.py:23-31` embeds
META_PLAN as code (Sharpe >1.2, trades <50/mo, sector <30%, 2× cost
stress). These drift with the project and cannot be tuned without a
deploy.

**Proposed change**: Move META_PLAN values into
`backend/backtest/experiments/optimizer_best.json` (or a sibling
`meta_plan.json`). PlannerAgent reads on each call. Add Q/A check:
"Sharpe target in plan must equal current best Sharpe + delta_target
or be flagged."

**Doc citation**: HARNESS-DOC stress-test doctrine verbatim — "every
component in a harness encodes an assumption about what the model
can't do on its own, and those assumptions are worth stress testing."
Hardcoded thresholds **resist** stress testing.

**Expected effect**: The planner uses current empirical floors, not
six-month-old aspirational targets.

### R-5 — Apply DirectiveReview's pattern to Q/A

**Problem**: Phase 2 finding D-3. DirectiveReview already implements
the "evaluator never sees proposer's self-grade" + "fail-CLOSED on
LLM error" patterns cleanly
(`backend/meta_evolution/directive_review.py:77-82, 17-21`). Q/A does
NOT have a "strip Main's self-grade" mechanic (there's no `judge_score`
on Main's output to strip — yet) and Q/A returns
`{ok:true, reason:"loop prevention"}` if `stop_hook_active`
(`qa.md:188-189`), which is fail-OPEN.

**Proposed change**:

- Add a `self_grade` field to `experiment_results.md` template
  (Main optionally writes a quick self-grade). Q/A's prompt is
  modified to strip this field before judgment (parallel to
  `directive_review.py::_build_review_prompt`'s strip behavior).
- Change `qa.md:188-189` to fail-CLOSED (`{ok:false, verdict:"FAIL",
  reason:"loop_prevention_did_not_run"}`) — match
  `directive_review.py:17-21`. The current fail-open makes Q/A's
  evaluator independence weaker than DirectiveReview's.

**Doc citation**: HARNESS-DOC verbatim — "Separating the generator
from the evaluator is the strongest lever" (paraphrased in
`evaluator_agent.py:6` "Separating generation from evaluation is
the strongest lever").

**Expected effect**: Q/A's independence is architectural, not just
prompt-stated. Failure modes are explicit.

### R-6 — Delete deprecated stubs

> **CLOSURE (phase-23.8.3, 2026-05-11)**: this recommendation
> **superseded** by header-correction. The audit took the
> "DEPRECATED — Phase 4 stub" file headers at face value, but
> cycle 37's research gate (phase-23.8.0) found both files are
> LIVE with active importers (`autonomous_loop.py:19,50,462-488,896-897`
> for `meta_coordinator.py`; `scripts/risk/phase4_9_redteam.py:58`
> for `autonomous_harness.py`). Deleting them would break paper
> trading + FINRA-15-09 negative tests. Cycle 40 (phase-23.8.3)
> closed R-6 by correcting the misleading headers so future
> auditors see the live status. A future delete cycle would need
> to refactor the importers first. The original audit text below
> is preserved as a historical record.

**Problem**: Phase 2 finding C-A7. `backend/autonomous_harness.py`
and `backend/agents/meta_coordinator.py` are explicit
`DEPRECATED — Phase 4 stub` files left on disk.

**Proposed change**: Delete both files. Update `_inventory.json` to
remove `meta_coordinator` and `autonomous_harness` if listed.

**Doc citation**: HARNESS-DOC verbatim — "Stale scaffolding is dead
weight — prune it." (paraphrased; the actual quote is "**As models
improve, remove dead-weight scaffolding that no longer provides
lift.**")

**Expected effect**: Reader of `backend/agents/*.py` no longer has
two ghost files to ignore.

### R-7 — Doc reconciliation for "28 Layer-1 skills" framing

**Problem**: Phase 3 Symptom 2's SCOPING_GAP root cause. CLAUDE.md
and ARCHITECTURE.md describe "28 Layer-1 skills". The rationale
drawer aggregates them into 5 layers, dropping to 3 in lite mode.
There is no single doc paragraph reconciling the two.

**Proposed change** (docs-only, no code): Add a paragraph to
`ARCHITECTURE.md` under the "Layer 1" section that explicitly maps
the 28 skills → 5 rationale-drawer layers (Analyst / Bull / Bear /
Trader / Risk) → 3-row lite output. Add an info banner to the
rationale drawer when in lite mode ("Lite mode active — only 3 of
the 28 Layer-1 agents are summarized here").

**Doc citation**: EFFECTIVE-DOC verbatim — "Success in the LLM space
isn't about building the most sophisticated system. It's about
building the right system **for your needs**." If the lite path is
the right system, the UI should make that explicit.

**Expected effect**: Operator expectation aligns with implementation
without changing the implementation.

---

## Proposed masterplan step (NOT applied; proposal only)

Drafted in the project's existing schema (matching the 23.6.4 +
2.13 patterns from `.claude/masterplan.json`). Verification criteria
are objective (a script could check them) and would be IMMUTABLE
once committed.

```json
{
  "id": "23.8.0",
  "name": "Dev-MAS harness compliance — live-check gate + namespace cleanup + DirectiveReview-pattern adoption",
  "status": "pending",
  "harness_required": true,
  "priority": "P1",
  "estimated_minutes": 240,
  "verification": {
    "command": "source .venv/bin/activate && python3 scripts/qa/verify_dev_mas_compliance.py",
    "success_criteria": [
      "live_check_field_supported_by_auto_commit_hook",
      "task_completed_hook_either_deleted_or_promoted_to_subagent_md_file_with_tool_list",
      "claude_md_three_agent_rule_scoped_to_layer3_explicitly",
      "agent_definitions_ford_renamed_or_disambiguated",
      "planner_meta_plan_thresholds_read_from_runtime_config_not_hardcoded",
      "qa_md_stop_hook_active_branch_returns_fail_not_pass",
      "deprecated_autonomous_harness_py_and_meta_coordinator_py_deleted",
      "architecture_md_has_28_to_5_to_3_drawer_mapping_paragraph",
      "research_brief_cites_harness_design_post_and_multi_agent_research_post",
      "qa_verdict_includes_bq_sample_paste_for_new_paper_trade_post_fix"
    ]
  },
  "contract": "handoff/current/contract.md",
  "research_gate_brief": "Research must cover: (a) Anthropic harness-design + multi-agent research blog posts (the two foundational refs already cited in CLAUDE.md); (b) verify the 'stale scaffolding is dead weight' quote and re-evaluate which Layer-2 / hook agents qualify as scaffolding given the current Opus 4.7 model tier (per HARNESS-DOC stress-test doctrine); (c) verify Claude Code subagent docs at https://code.claude.com/docs/en/sub-agents on tool-list scoping for hook-spawned subagents; (d) internal code audit of backend/services/portfolio_manager.py to confirm Read A/B/C of Symptom 1 from the audit. Min 5 sources read in full + recency scan per .claude/rules/research-gate.md. Files expected to inspect: .claude/settings.json (hook blocks), .claude/agents/qa.md (line 188-189), backend/agents/agent_definitions.py (lines 177-220, 270-315), backend/agents/planner_agent.py (lines 23-31), CLAUDE.md (the 'exactly 3 agents' rule), ARCHITECTURE.md (Layer 1 paragraph).",
  "files_expected_to_change": [
    ".claude/settings.json (hook blocks)",
    ".claude/agents/task_completed.md (NEW if Option B chosen) or NULL if Option A",
    ".claude/agents/qa.md (line 188-189 fail-mode + line 192 second-opinion text vs CLAUDE.md drift)",
    "backend/agents/agent_definitions.py (rename Ford label; rename in-app Researcher label)",
    "backend/agents/planner_agent.py (META_PLAN -> config read)",
    "backend/backtest/experiments/meta_plan.json (NEW, runtime config)",
    "backend/autonomous_harness.py (DELETE)",
    "backend/agents/meta_coordinator.py (DELETE)",
    "CLAUDE.md (scope the 'exactly 3 agents' rule)",
    "ARCHITECTURE.md (NEW paragraph: 28 -> 5 -> 3 drawer mapping)",
    "scripts/qa/verify_dev_mas_compliance.py (NEW, 10-claim source-level assertion script)",
    ".claude/hooks/auto-commit-and-push.sh (NEW live_check gate)"
  ],
  "rollback_note": "Single-commit revert. The auto-commit-and-push.sh hook live_check gate is the only piece with operational risk — if a false-positive blocks a legitimate push, the operator can manually `git push origin main` (the gate's failure mode is exit 0 with a log line, per the existing hook discipline)."
}
```

**Why this proposed step satisfies the per-step contract**:

- **id** chosen as `23.8.0` (new sub-phase under phase-23, the
  current active phase per masterplan).
- **harness_required: true** — this step touches CLAUDE.md and
  `.claude/agents/qa.md`, which are dev-MAS critical files; the
  full harness cycle (researcher + qa) must run.
- **verification.command** is one bash invocation a script can
  exit 0/1 on. The 10 success criteria are immutable and each
  corresponds to a recommendation R-1 through R-7 above.
- **research_gate_brief** explicitly cites the harness-design +
  multi-agent posts + sub-agents doc + research-gate rule — meeting
  the project's research-gate doctrine.
- **rollback_note** describes single-commit revert safety, matching
  the project's per-step auto-push commit pattern (one step = one
  commit).

---

## What to stop doing

Items from Phase 2 where the dev MAS declares a principle but does
not enforce it. For each: enforce or retire (per HARNESS-DOC's
stress-test doctrine — "those assumptions are worth stress testing").

### Stop: pretending the research gate is enforced when it is behavioral

**Declared principle** (CLAUDE.md): "**Research Gate is mandatory** —
no step proceeds to GENERATE without deep research"; "every cycle
spawns researcher first".

**Reality** (auto-memory `feedback_research_gate.md`,
`phase-4.10` audit reference): "Main slips on 7 of 9 phase-4.8
cycles."

**Doc-grounded recommendation**: Either (a) **enforce** via a hook
that fails the auto-commit if `handoff/current/contract.md`'s
"references" section lacks ≥5 distinct `https://` URLs OR (b)
**retire** the "mandatory" framing and downgrade to "Researcher SHOULD
be spawned when X" with explicit X criteria. SUBAGENT-DOC: tool-list
boundaries are enforceable; prose rules are not. The doctrine
mismatch is what produces the slippage. Recommendation: option (a)
— enforce. See R-1 above for the hook design.

### Stop: claiming "Q/A is the sole independent evaluator" while two evaluators run

**Declared principle** (CLAUDE.md): "Q/A is the sole independent
verification agent for the pyfinagent masterplan system."

**Reality** (`.claude/settings.json:64-71`): TaskCompleted hook
spawns an unnamed evaluator with no tool restrictions, running its
own inlined cross-verification prompt.

**Doc-grounded recommendation**: per SUBAGENT-DOC "specific tool
access", either delete the TaskCompleted hook (R-2 Option A) or
promote it to a tools-restricted subagent file with a non-overlapping
charter (R-2 Option B). The current state — two evaluators with
overlapping scope and one having unconstrained Write access — is
worse than either alternative.

### Stop: framing the harness MAS as "exactly 3 agents" without naming the scope

**Declared principle** (CLAUDE.md): "**The MAS is exactly 3 agents:
Main + Researcher + Q/A.**"

**Reality** (this audit, Phase 1): the dev MAS has 15-16 agents
across Layer 2, 3, 4 and hook-driven categories.

**Doc-grounded recommendation**: per HARNESS-DOC stress-test doctrine,
the "3 agents" rule encodes an architectural assumption that needs
to be scoped. Rewrite to "**The Harness MAS layer (Layer 3) is
exactly 3 agents.**" Add a sentence pointing at
`backend/agents/_inventory.json` as the canonical
roster for the broader dev MAS. See R-3 above.

### Stop: shipping unit tests as "verification complete"

**Declared principle** (qa.md:33-99): "Verification order
(deterministic FIRST) ... 1. Deterministic checks ... 2. Existing
results check ... 3. Harness dry-run ... 4. LLM judgment (last
resort)."

**Reality** (Symptoms 1 + 3 in Phase 3): unit tests pass; live BQ
sampling / live UI sampling / live-pipeline reproduction does NOT
occur as part of the Q/A cycle. The phase-23.2.A-fix shipped on
2026-05-04 with three new unit tests and no BQ sample of a
post-fix paper trade.

**Doc-grounded recommendation**: HARNESS-DOC verbatim — "the
generator has something concrete to iterate against." A unit test
on synthetic input is concrete; a live BQ row is more concrete.
Promote the `live_check` field from R-1 to a Q/A required check
when the step's diff touches user-visible UI or persisted data.

### Stop: leaving deprecated stubs on disk

**Declared principle** (file headers of `autonomous_harness.py` +
`meta_coordinator.py`): "DEPRECATED — Phase 4 stub. Not part of the
active MAS architecture."

**Reality**: both files have been on disk since at least phase 4 (per
`handoff/archive/phase-audit-2.10-4.14.20-research-brief.md:49`,
"meta_coordinator.py is a DEPRECATED stub"). They are visible to any
reader who lists `backend/agents/*.py` or `backend/*.py` and can
mislead a fresh contributor.

**Doc-grounded recommendation**: HARNESS-DOC verbatim — "remove
dead-weight scaffolding." Delete. R-6 above.

---

## Self-bias check (Phase 4)

1. **Pro-architectural-fix bias.** I'm proposing seven architectural
   recommendations; the natural pull is to recommend MORE complexity
   (new hooks, new fields, new subagent files). EFFECTIVE-DOC
   verbatim: "**add complexity only when it demonstrably improves
   outcomes**." Counter-check: R-2 Option A is "DELETE the
   TaskCompleted hook"; R-6 is "DELETE deprecated stubs". Two of the
   seven recommendations REMOVE components. The bias correction is
   visible.
2. **Pro-Anthropic-doc bias.** Every recommendation cites
   HARNESS-DOC / MULTI-DOC / EFFECTIVE-DOC / SUBAGENT-DOC.
   Counter-check: I deliberately did NOT cite the Anthropic blogs
   for R-4 (hardcoded thresholds) since that finding is grounded in
   the project's own `RESEARCH.md` Bailey & López de Prado DSR
   refs and the **stress-test doctrine**, which is a doctrine I
   could be applying too broadly. The "rebuild for runtime config"
   recommendation is also a standard software-engineering call,
   not a uniquely Anthropic one.
3. **Pro-fix-Q/A bias.** I am Main (Layer 3). Recommending fixes to
   Q/A makes Main's life easier (clearer guardrails, fewer
   pull-up-by-the-bootstraps). Counter-check: R-5 makes Q/A
   STRICTER (fail-closed on loop-prevention; strip self-grade) —
   that constrains Main, not Q/A. The bias correction is in the
   direction that DOESN'T flatter Main.
4. **Charity vs realism on "what to stop doing".** Phrasing
   "stop doing X" is more confrontational than "consider doing Y."
   I went confrontational because the audit prompt explicitly says
   "Anti-leniency. When evidence is mixed, default to the harsher
   reading."

## Done criteria check

- [x] Every recommendation traces to a specific finding in Phases
  1-3. R-1 → Q-5, H-2, Phase 3 systemic pattern; R-2 → H-1, H-2;
  R-3 → F-1, C-2, Phase 1 namespace section; R-4 → P-1; R-5 → D-3;
  R-6 → C-A7; R-7 → Phase 3 Symptom 2 scoping gap.
- [x] The proposed masterplan step's verification criteria are
  objective (a script could check them) and copied immutable into
  the snippet. 10 criteria, each phrased as a single grep/diff/file
  assertion.
- [x] "What to stop doing" lists at least the items where Phase 2
  found declared-but-unenforced principles. Four items: research
  gate, Q/A sole-evaluator framing, "exactly 3 agents", "unit tests
  as verification complete". Plus the deprecated-stubs item.
