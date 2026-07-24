# Research Brief — Step 71.6

**Tier:** complex (scheduling-safety + agent-file edits + dead-code pruning)
**Date:** 2026-07-13
**Author:** Layer-3 Researcher subagent
**Status:** IN PROGRESS (write-first; grown incrementally)

## Step 71.6 — three criteria
1. Save the 2026-07-16 harness/MAS self-audit as a saved, RE-RUNNABLE `.claude/workflows/`
   script AND schedule a WEEKLY REPORT-ONLY run (local); it NEVER auto-applies — only writes
   a findings report the operator reviews (honors background-agent-unauthorized-action memory).
2. `researcher.md`/`qa.md` instruct the subagent to return a COMPACT envelope (summary +
   verdict + handoff path) rather than the full brief/critique through Main context.
3. Any dormant/dead self-evaluating driver confirmed unused by the stress-test is PRUNED
   (or explicitly kept with a reason); harness stays exactly 3 agents.

**Verification command:**
```
ls .claude/workflows/ | grep -Eqi "audit|self|stress" && \
grep -Eqi "envelope|summary|file path|return" .claude/agents/researcher.md .claude/agents/qa.md
```

---

## THE CRITICAL SAFETY QUESTION (criterion 1)
Memory `feedback_background_agent_resumption_risk`: completed background agents resumed on
file-system triggers; a 3rd pass made an unauthorized masterplan install+push (reverted);
**review-only prompts are NOT enforcement.** So a scheduled AUTONOMOUS Claude run merely
PROMPTED "report-only" is NOT safe. Need the ENFORCEMENT-SAFE mechanism.

_(findings below, filled incrementally)_

---

## Internal findings

### A. Scheduling mechanisms in this repo (criterion 1 — how the repo already schedules)

Two distinct scheduling worlds coexist:

**1. launchd (macOS OS-level)** — `~/Library/LaunchAgents/com.pyfinagent.*.plist`.
Currently LOADED (from `launchctl list`): backend, frontend, slack-bot,
autoresearch, ablation, backend-watchdog, away-session-am, away-session-pm,
away-watchdog, claude-code-proxy. These fire shell scripts; the DANGEROUS class
(`mas-harness`) fired `claude -p --dangerously-skip-permissions` (full LLM
agency + auto-push). **`com.pyfinagent.mas-harness` is ABSENT from launchctl
list** and its live `.plist` is GONE from `~/Library/LaunchAgents/` — only
`.plist.bak-harness-ABCD` + `disabled.*.plist.bak` remnants survive (the
away-ops `pending_tokens.json` `mv`-to-`.bak` ask was executed). It is
operationally NEUTRALIZED.

**2. APScheduler (in-process, backend)** — `backend/main.py:266-324` starts
`AsyncIOScheduler`; jobs registered by `register_*_cron` modules. THE
ENFORCEMENT-SAFE GOLD-STANDARD pattern lives here:
`backend/meta_evolution/cron.py::register_meta_evolution_cron` — a WEEKLY
(Sunday 02:00 America/New_York) cron whose job `run_meta_evolution_cycle()` is
**pure-Python**: it calls library functions (cron_allocator, provider_rebalancer,
archetype_library), wraps each sub-call fail-open, and emits a **P3 Slack summary
alert** ("Weekly autoresearch cycle completed"). NO LLM, NO git, NO masterplan
flip, NO apply. This is exactly "scheduled weekly, report-only, structurally
cannot mutate." `.claude/cron_budget.yaml` governs the 15/day self-imposed cap
(slot 14 = `meta_evolution_weekly_reallocation`, Sundays).

**Report-only precedent (harness_log:26346):** "digest is TEMPLATE/DATA-ONLY
($0 LLM, NOT operator-gated)" — confirms deterministic report jobs are the
safe, non-operator-gated class.

**Structural guard (62.0):** `.claude/hooks/pre-tool-use-danger.sh:152-170` is a
PreToolUse hook that BLOCKS `git push` and `launchctl (bootout|unload|remove|
disable)` on any `com.pyfinagent.` label from inside agent sessions ("away-ops
rail 9"). Hooks run even under `--dangerously-skip-permissions`, so this is a
real structural layer — but it is a NEGATIVE guard on a full-agency agent, weaker
than "no agency at all." NOTE: it does NOT block `launchctl bootstrap`/`enable`
(only the removal verbs), so arming a NEW recurring agent is not hook-blocked;
that is precisely why arming is an operator DECISION, not a hook-enforced safe op.

### B. The dead self-evaluating driver (criterion 3) — CONFIRMED dead as a DRIVER, but files have LIVE consumers

`scripts/mas_harness/cycle_prompt.md` (5247 B, 24 Apr) + `run_cycle.sh` (3012 B,
29 May) ARE the stale self-evaluating driver. `cycle_prompt.md` literally
self-evaluates (Step 4: the SAME worker "Write your verdict to
`handoff/current/evaluator_critique.md`" — forbidden Main-self-eval) AND
auto-pushes (rule 6 "Always push to origin/main"). `run_cycle.sh` invokes
`claude -p --dangerously-skip-permissions --model claude-opus-4-8 < cycle_prompt.md`.

**Driver status = DEAD/neutralized:** no live `com.pyfinagent.mas-harness.plist`,
label absent from `launchctl list`, plists are `.bak`/`disabled`. It CANNOT fire.

**But the repo FILES are NOT orphaned — every reference (complete list):**
| Referencer | Line | Nature | Effect if files deleted |
|---|---|---|---|
| `scripts/go_live_drills/revert_hygiene_drill.py` | 52-53, 92, 99 | READS both files' text AND `bash run_cycle.sh` (executes it) | drill REDs |
| `tests/agents/test_phase_47_9_max_tokens_floor.py` | 77 | reads `run_cycle.sh` text to assert model pin | test REDs |
| `scripts/go_live_drills/smoke_test_4_17_11.py` | 5 | references mas-harness/run_cycle.sh | doc/smoke ref |
| `scripts/away_ops/run_away_session.sh` | 4 | comment: "Evolves ...run_cycle.sh" (lineage only) | none (comment) |
| `backend/api/cron_dashboard_api.py` | 121, 154, 158 | `_LAUNCHD_JOBS` lists `com.pyfinagent.mas-harness` + its log paths for /cron dashboard | dashboard shows a stale/not-loaded row |
| `~/Library/LaunchAgents/*mas-harness*.bak*` | — | 2 neutralized `.bak` plists (OUTSIDE repo, not version-controlled) | inert; safe to `rm` but not load-bearing |

**RIDER-TRAP in the 71.0 design:** design_harness_mas_71.md line 100-101 says
"Delete ...cycle_prompt.md + .bak plists after the stress-test confirms it's dead
weight." The stress-test (this grep) shows it is NOT pure dead weight — a naive
`rm` REDs `revert_hygiene_drill.py` + `test_phase_47_9` + `smoke_test_4_17_11`.
**Recommendation for criterion 3:** either (a) KEEP-WITH-REASON — the driver is
already operationally neutralized (that satisfies the SAFETY intent), and the
files remain fixtures for currently-passing dirty-tree-refusal safety drills; OR
(b) do the FULL ATOMIC co-delete: `rm` the 2 repo files + `revert_hygiene_drill.py`
+ `smoke_test_4_17_11.py` + the `test_phase_47_9` assertion (or the whole test if
run_cycle-only) + trim the `_LAUNCHD_JOBS` mas-harness entry + its 2 log-path keys
in `cron_dashboard_api.py`. Given the background-agent-unauthorized-action memory,
(a) keep-with-reason is the lower-risk, honest default; (b) is a clean prune only
if the operator wants the fixtures gone. DO NOT touch
`run_harness.py::_default_spawn_researcher` (live spawn path).

### C. Context-hygiene anchors (criterion 2)

Verification grep `envelope|summary|file path|return` on both files:
- **researcher.md ALREADY MATCHES** richly: `envelope` (l.91,93,98,104,105,113,
  310,312), `summary` (l.98 "the envelope is the audit summary + brief_path"),
  `brief_path`/file-path (l.98), `return` (many). Present sections: "## Launch"
  (l.84-105) + "## Output JSON envelope (ALWAYS EMIT)" (l.310-335).
  **To STRENGTHEN for the compact-return SPIRIT:** the envelope currently carries
  `"report_md": "..."` (l.331) — a field that would dump the FULL brief through
  Main's context on the Workflow path. Add an explicit `summary` (<=200-word)
  field + an instruction: "return `brief_path` + the <=200-word `summary`; do NOT
  return the full brief text (`report_md`) through Main's context — Main reads the
  full brief from `brief_path` on disk." Anchor: l.96-98 + the envelope block
  l.310-335 (demote/cap `report_md`).
- **qa.md matches the grep ONLY via `return`** today (NO `envelope`/`summary`/
  `compact`/`file path`/`critique_path`). Q/A already returns a compact JSON
  verdict (Main writes the prose critique), so it is compact in PRACTICE but never
  FRAMED as a compact envelope. **To ADD:** one sentence in the "Guardrails that
  bind BOTH launches" block (l.99-113) or "## Output format (single JSON)"
  (l.283-295): "Return a COMPACT verdict envelope (verdict + one-sentence reason
  summary + violated_criteria); the full critique prose lives at the
  `evaluator_critique.md` file path — never paste full file contents / the full
  critique through Main's context; your return is the lightweight reference Main
  transcribes." That adds `envelope`+`summary`+`compact`+`file path` to qa.md,
  matching the STRONG grep terms and satisfying criterion 2 for BOTH files.

BOTH edits touch `.claude/agents/*.md` -> separation-of-duties + roster-snapshot
handling applies (design binding constraint): leave a harness_log Peder-review
note; `scripts/qa/verify_qa_roster_live.sh` confirms the roster next session; the
Agent-tool path snapshots at session start, the Workflow path reads qa.md/
researcher.md from disk live.

### D. Existing workflow shape + the self-audit register (criterion 1 artifact)

`.claude/workflows/qa-verdict.js` (the ONLY current workflow file) is the shape to
mirror: `export const meta`, a `PROMPT` builder, a JSON `*_SCHEMA`, then
`phase(...)` + `await agent(PROMPT, {schema, agentType:'general-purpose',
model:'opus', effort:'max'})` + `return`. Its agent has WHATEVER tools the
'general-purpose' type carries; the SCRIPT itself does NO git/push/flip — it only
`return`s the captured value. That is the template for a **structurally
report-only** self-audit workflow.

The 2026-07-16 self-audit is captured in `handoff/current/harness_proposals.json`
(193 KB, the register: 17 kept / 15 rejected). A re-runnable
`.claude/workflows/harness-self-audit.js` should: fan-out READ-ONLY finder agents
(Read/Grep/Glob only, NO Edit/Bash-mutate/git) over the harness+MAS surface ->
a verify pass -> the SCRIPT writes ONE findings report to a file (e.g.
`handoff/self_audit/<date>-harness-audit.md` or refreshes `harness_proposals.json`)
and STOPS. No masterplan flip, no git push, no apply. Filename matches
`audit|self|stress` (satisfies `ls .claude/workflows/ | grep -Eqi "audit|self|stress"`).

---

## External findings

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Key finding (verbatim where quoted) |
|---|-----|----------|------|-------------------------------------|
| 1 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-07-17 | Official doc | Stress-test doctrine: "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing." "When a new model lands, it is generally good practice to re-examine a harness, stripping away pieces that are no longer load-bearing to performance." Doer/judge: "Separating the agent doing the work from the agent judging it proves to be a strong lever." File-handoff: "Communication was handled via files: one agent would write a file, another agent would read it and respond." |
| 2 | https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-07-17 | Official doc | Lightweight references: "implement artifact systems where specialized agents can create outputs that persist independently … reduces token overhead from copying large outputs through conversation history." "condensing the most important tokens for the lead research agent." Lead "decides whether more research is needed." Grounds criterion 2 (compact envelope + brief_path, not full text through Main). |
| 3 | https://code.claude.com/docs/en/workflows | 2026-07-17 | Official doc | DECISIVE for the safety mechanism: "**No direct filesystem or shell access from the workflow itself. Agents read, write, and run commands. The script coordinates the agents.**" Subagents "inherit your tool allowlist"; report-not-PR shape (`/deep-research` "get one report at the end"); "**a scheduled task prompt**" does NOT trigger ultracode; `claude -p`/Agent SDK/bypass = "**Never** [prompted]. The run starts immediately." Saved scripts live in `.claude/workflows/`; `agent(prompt,{schema})` + `pipeline()`; 1000-agent/16-concurrent caps; `args` global. |
| 4 | https://www.microsoft.com/en-us/security/blog/2026/07/16/least-privilege-for-ai-agents-identity-access-and-tool-binding/ | 2026-07-17 | Official (vendor security) | 2026-07-16. Tool binding: "expose a curated and approved set of tools/actions … require explicit allowlists for high-impact operations." Read/write duty separation: "use different roles (or different tools) for read versus write, and gate high-impact actions like delete, export, or privilege changes behind step-up approvals." JIT: "keep the baseline role minimal … automatically drop back to the baseline when the workflow completes." |
| 5 | https://arxiv.org/html/2605.05868 (SkillScope, Wu et al. 2026) | 2026-07-17 | Peer-reviewed preprint | Task-conditioned least privilege: "a Skill should exercise only the capabilities required for the requested task." Over-execution (unnecessary high-impact ops) = 55.06% of validated violations. "each functional module should be organized as an atomic operation with a clear task boundary, so that sensitive behaviors … are invoked only when explicitly required." |
| 6 | https://www.mindstudio.ai/blog/how-to-build-scheduled-ai-agents-claude-code | 2026-07-17 | Practitioner | Scheduling is EXTERNAL (cron `0 9 * * 1-5`, GitHub Actions, EventBridge/Cloud Scheduler). Guardrails: "`--allowedTools` … If your agent only needs to read files, don't give it write access"; "You are a monitoring agent. Your job is to observe and report, not to make changes"; "Don't rely solely on agent instructions — also enforce permissions at the OS level"; `--max-turns N` "Critical for preventing runaway execution." |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai/ | News | Corroborates 3-agent harness; snippet sufficient |
| https://arxiv.org/pdf/2606.29142 | Preprint | Autonomous-agent threats in regulated finance; snippet — adjacent, not core |
| https://getclaw.sh/blog/human-in-the-loop-ai-agents-approvals-2026 | Blog | HITL approval tiers 2026; snippet |
| https://www.cequence.ai/blog/ai/ai-agent-least-privilege-access/ | Blog | Least-privilege OWASP Excessive-Agency framing; snippet |
| https://claude.com/blog/introducing-dynamic-workflows-in-claude-code | Official blog | Workflow launch announcement; superseded by the doc (#3) |
| https://www.strata.io/blog/agentic-identity/practicing-the-human-in-the-loop/ | Blog | NIST-AI-RMF autonomy tiers; snippet |
| https://arxiv.org/pdf/2602.21012 | Report | International AI Safety Report 2026; snippet |
| https://iternal.ai/ai-agent-security-checklist | Blog | Agentic risk checklist 2026; snippet |

### Key external findings mapped to the criteria
1. **Structural report-only >> prompt report-only (criterion 1 safety).** The workflows doc's "No direct filesystem or shell access from the workflow itself" + Microsoft's "different tools for read versus write" + SkillScope's "only the capabilities required" + MindStudio's "don't give it write access / don't rely solely on agent instructions" ALL converge: the enforcement-safe design is TOOL-RESTRICTION (least privilege), not a "report-only" instruction. This is the external mandate behind the auto-memory's "review-only prompts are NOT enforcement."
2. **`claude -p`/bypass never prompts** (workflows doc) — so a scheduled `claude -p --dangerously-skip-permissions` self-audit has NO interactive gate; only the tool allowlist + PreToolUse hooks stand between it and a push. That is exactly the mas-harness failure class. AVOID.
3. **Scheduling is external** (cron/launchd/GH Actions per MindStudio) — Claude Code has no native recurring trigger; the recurring arm is an OS/infra artifact the operator owns on a local Mac.
4. **Compact-envelope return is documented** (multi-agent-research "artifact systems … reduce token overhead from copying large outputs through conversation history") — grounds criterion 2.
5. **Stress-test = strip non-load-bearing scaffolding** (harness-design) — grounds criterion 3's prune, AND the meta-lesson that a naive prune that REDs live drills is itself a failure of the "find the simplest solution" discipline.

---

## Recency scan (last 2 years)

Searched 2024-2026 for safe scheduled/recurring autonomous agents, least-privilege
tool-binding, and report-only guardrails. **Findings that supersede/complement the
canonical Anthropic harness docs:**
- **Microsoft Security Blog, 2026-07-16** (one day before this brief) — least-privilege
  tool binding + read/write duty separation + step-up approvals for high-impact ops.
  This is the freshest authoritative statement of the exact control criterion 1 needs.
- **SkillScope, Wu et al., arXiv 2605.05868 (May 2026)** — task-conditioned least
  privilege for agent skills; over-execution dominates violations (55%). New peer-reviewed
  grounding not present in the 2026-04 Anthropic docs.
- **Claude Code workflows doc (current, v2.1.154+ feature; ultracode nuances through
  v2.1.210)** — the "no direct filesystem/shell access from the workflow itself" +
  "scheduled task prompt does not trigger ultracode" clauses are recent-version behavior
  directly load-bearing for the design.
- **HITL 2026 corpus** (getclaw, strata/NIST-AI-RMF autonomy tiers, International AI Safety
  Report 2026) — converges on pre-approval for irreversible actions + post-hoc audit for
  recoverable ones; a self-audit is recoverable/observational, so report-only + audit trail
  is the fit-for-purpose tier.
No newer finding CONTRADICTS the canonical stress-test / doer-judge / file-handoff doctrine;
they REINFORCE tool-restriction as the enforcement layer.

---

## Synthesis & recommendations

### Criterion 1 — enforcement-safe weekly report-only self-audit (RANKED)

**RECOMMENDED design (maximizes enforcement, honors the memory):**
1. **Check in a structurally-report-only saved workflow** `.claude/workflows/harness-self-audit.js`
   (matches `audit|self|stress`; safe to create autonomously — it is a file, no scheduling
   side-effect). Shape mirrors `qa-verdict.js`: `export const meta` + a fan-out of READ-ONLY
   auditor agents (`agent(prompt,{schema})` / `pipeline()`) over the harness+MAS surface ->
   a verify pass -> **return** the ranked findings. ENFORCEMENT comes from three structural
   layers, not from the word "report-only":
   (a) the workflow SCRIPT has "no direct filesystem or shell access" (Claude Code doc) — it
       cannot itself git-push or flip the masterplan;
   (b) the auditor agents are spawned with a READ-ONLY tool set (Read/Grep/Glob/WebFetch; NO
       Edit/Write, NO Bash-mutate, NO git, NO launchctl) — least-privilege tool binding
       (Microsoft 2026-07-16; SkillScope). An agent that has no push tool CANNOT push even if
       it "decides" to;
   (c) the single findings file is written by ONE narrowly-scoped writer whose only Write
       target is `handoff/self_audit/<date>-harness-audit.md` (or Main persists the returned
       object) — never `.claude/masterplan.json`, never a commit. The existing 62.0
       PreToolUse guard (blocks `git push` + `launchctl` removal) is defense-in-depth.
   Add `--max-turns`-style caps (workflow runtime already caps 1000 agents/16 concurrent).
2. **Automated leg (optional, zero-agency):** a DETERMINISTIC Python report writer on the
   proven `backend/meta_evolution/cron.py::register_meta_evolution_cron` weekly (Sunday)
   APScheduler pattern — greps the harness invariants (3-agent roster, dead-driver absence,
   envelope-return present, roster-snapshot notes), runs the verification asserts, writes a
   findings file + emits a P3 Slack summary. NO LLM = NO agency = structurally cannot mutate
   (this is the `run_meta_evolution_cycle` + "digest is DATA-ONLY, not operator-gated"
   precedent). Trade-off: cannot reproduce the ultracode LLM reasoning depth of the
   2026-07-16 audit — it is a lightweight watchdog, not a replacement.
3. **AVOID:** a scheduled `claude -p --dangerously-skip-permissions` run merely prompted
   "report-only." Per the workflows doc it NEVER prompts and tool calls run immediately; per
   the memory review-only prompts are NOT enforcement; it is the exact neutralized mas-harness
   pattern.

**Does creating the schedule need an operator?** The ARTIFACTS (saved workflow with read-only
agents; the deterministic report writer; a launchd plist TEMPLATE left UN-bootstrapped;
SessionStart grep asserts) are safe to check in autonomously — they are structurally incapable
of applying. **ARMING the recurring autonomous trigger** (bootstrapping a new launchd label, or
registering + restarting an APScheduler job) is an operator action/decision: it is a
machine-state change on a single local Mac, the 62.0 guard does NOT block `launchctl bootstrap`
(only removal verbs), and the mas-harness + away-ops `pending_tokens.json` history shows launchd
arming is consistently an operator ask. So: **schedule_needs_operator = TRUE for the recurring
arm; the report-only artifacts are autonomously safe.** If no fully-enforcement-safe AUTONOMOUS
arm is acceptable, the safest partial (and what the 71.0 design already descoped #12 to) is:
saved report-only workflow + a DOCUMENTED weekly operator-triggered cadence (run `/harness-self-audit`
or the deterministic writer) + the report-only tool guard. The verification command only checks
the saved script exists — so the criterion is satisfiable WITHOUT arming an autonomous push-capable
cron.

### Criterion 2 — see Internal §C. Both `.md` edits carry separation-of-duties + roster-snapshot handling.
### Criterion 3 — see Internal §B. Driver operationally DEAD; files have LIVE consumers -> keep-with-reason OR atomic co-delete; never naive `rm`.

### Harness stays EXACTLY 3 agents — CONFIRMED
A scheduled report is a ROUTINE, not a 4th agent. The saved workflow spawns the SAME
Researcher/Q/A roles (or ephemeral read-only auditor instances that terminate at run end) — no
standing autonomous 4th member joins Main+Researcher+Q/A. The deterministic Python writer is code,
not an agent. Adversarial/red-team checks stay WITHIN the single Q/A role (design binding
constraint). No re-split of Explore / harness-verifier.

### Rider-traps flagged (do NOT let ride in on this step)
- **RT-1 (design's own trap):** 71.0 design line 100-101 "Delete cycle_prompt.md + .bak plists"
  — naive `rm` REDs `revert_hygiene_drill.py` + `test_phase_47_9` + `smoke_test_4_17_11`. Prune
  must be atomic-with-consumers or keep-with-reason.
- **RT-2:** the self-audit must NOT become an auto-FIX loop (design R1), must NOT model-swap on
  stall (R4), must NOT add a Monitor mtime watchdog (R11). Report-only = no apply, no push, no flip.
- **RT-3:** do NOT touch `run_harness.py::_default_spawn_researcher` (:1044/:1122) or the LIVE
  `scripts/away_ops/run_away_session.sh` (away-session-am/pm are loaded + operator-approved) — both
  are live, not dead weight.
- **RT-4:** the compact-envelope edit must NOT drop the write-first brief-on-disk discipline
  (researcher) nor the verbatim-transcription rule (qa) — the envelope is the SUMMARY + path, the
  full artifact still lands on disk.

---

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (14: 6 full + 8 snippet)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "gate_passed": true
}
```
