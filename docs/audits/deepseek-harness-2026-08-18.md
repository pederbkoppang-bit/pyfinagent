# DeepSeek Harness — research brief and adoption assessment

Date: 2026-08-18
Subject: `deepseek-ai/deepseek-harness` (`dsh`), released 2026-08-13, MIT.
Audited at commit `99f6f02fecdb7dff40c3fbc9470f5907c29f74ca`, version `0.1.0-rc.7`
(shallow clone taken 2026-08-18; `git log -1` read directly, not from a release page).

Question asked: *is there anything in the DeepSeek harness that could improve ours?*

Answer in one line: **adopt several of its mechanisms; do not adopt the software.**

---

## 0. What it actually is, and what it is not

`dsh` is a **coding-agent runtime** — a competitor to Claude Code, i.e. Layer 0
underneath this project, not a competitor to the pyfinagent Layer-3 harness
(`Plan → Generate → Evaluate` with file-based handoffs). It does not replace
`per-step-protocol.md`; it replaces the thing that *runs* it.

Architecture, from `docs/architecture.md`: everything is a plugin on a vendored
copy of [Cordis](https://github.com/cordiverse/cordis), including "the model
adapter, the tool registry, the session log, and the agent loop itself". There is
"no privileged core to patch". A running `dsh` is a plugin tree composed at boot
from ordered layers (bundles → profile patch → home patch → `--patch` overlay).

The comparison that matters for us is therefore **not** "should we switch",
it is "what did a well-resourced team building the same category of thing
independently arrive at, and where does that indict our design".

---

## 1. Genuinely new mechanisms — ranked by value to us

### A1. Progress-gated retry (highest value)

`packages/compaction/compaction-basic/src/index.ts:191-222`. On a context-overflow
failure the plugin snapshots `agent.session.surface.replaceGeneration`, attempts
compaction, and then:

```ts
if (signal.aborted
  || agent.session.surface.replaceGeneration <= generation) return next()  // NO retry
this.overflowRetries.set(agent, retries + 1)
return { kind: 'retry' }
```

A retry is admitted **only if durable state actually advanced**. If nothing
changed, the original error stands and is authoritative. There is *also* an
independent count cap (`maxOverflowRetries`). Two bounds, different jobs: one
says "you may not go around again without having changed something", the other
says "you may not go around forever even if you do".

Even the error path honours it — a model-free prune that landed before a later
summarisation threw still counts as progress, because the durable reduction is
real: "That durable reduction is sufficient retry proof; do not discard it just
because the optional second phase threw."

**Why this indicts us.** Our anti-verdict-shopping rule is the single strongest
piece of Layer-3 doctrine — CLAUDE.md forbids "spawning a fresh Q/A to overturn a
verdict on **unchanged evidence**" — and it is enforced by *nothing but Main's
honesty*. `scripts/harness/attempt_gate.py` counts attempts and has no notion of
evidence at all (verified: no hash/digest/evidence/unchanged logic in the file).
So the harness can currently distinguish "5 attempts" from "4 attempts" but
cannot distinguish a genuine fix-and-regrade from a re-submission of identical
evidence.

**Adoption sketch.** Add an evidence digest to the attempt ledger row that
`attempt_gate.py` already appends: hash `experiment_results.md` plus the files
the prior critique named, and deny a re-spawn whose digest is unchanged, with the
same escalation path as budget exhaustion. This has a second benefit — it fixes
the *converse* failure recorded in `project_third_conditional_rule_parks_converging_steps`:
a step that genuinely changed its evidence every round can be shown to be
converging rather than looping, so the 3rd-CONDITIONAL rule stops parking work
that was actually progressing.

### A2. A runtime invariant registry with ownership rules

`packages/runtime-diagnostics/invariants/README.md`. `ctx.invariants` is a
registry service; every workspace package publishes a `./invariant` companion
registered under its exact npm name. Three parts are the interesting ones:

1. **Independent re-derivation before commit.** The `dsh-goal` companion
   "maintains an independent fold of each attached session" and rejects malformed
   changes, discontinuous revisions, illegal transitions and timestamp
   regressions **"before the candidate event enters the durable log"**. The
   checker does not trust the producer; it recomputes.
2. **Assertions are not synthetic.** A companion installs a check *only* where
   the package owns "an observable event relationship or relevant mutable-data
   relationship". Confirming a method exists, a plugin name, or a pure-function
   result is explicitly declared a type/unit-test concern, not a runtime
   invariant.
3. **Explained absence is mandatory.** Where no plausible relationship exists the
   companion ships an *empty* installer with a leading `No runtime invariant:`
   comment saying why — and `pnpm run verify-package-invariants` rejects
   "unexplained empty installers" and "non-empty installers that omit or ignore
   the reporter".

**Why this indicts us.** We have 30+ ad-hoc checkers in `scripts/qa/`, most named
for the phase that produced them (`census_invalid_json_86_108.py`,
`derive_qa_rail_drop_table_86_31.py`, …). There is no registry, no ownership
mapping, no rule about when a check is warranted, and no way to tell "this area
has no invariant because none applies" from "nobody wrote one". Point 3 is the
part we lack entirely, and it is the part that makes coverage *legible*.

Point 1 is convergent validation of our best decision: `research-gate.js`
already recomputes `gate_passed` rather than trusting the agent's self-report,
and cross-checks claimed URLs against the brief on disk. DeepSeek arrived at the
same "don't trust the producer" principle independently, and generalised it.

### A3. A documentation word-budget gate

`scripts/verify-doc-budgets.ts` + `scripts/doc-budgets.manifest.json`. Standing
docs carry hard `wc -w` ceilings:

```json
{ "AGENTS.md": 1950, "docs/architecture.md": 2400,
  "docs/defensive-patterns.md": 550, "docs/testing.md": 1150, ... }
```

"Ceilings ratchet down with at least 5% headroom; raising one requires the
justification defined in `docs/AGENTS.md`." Missing files and invalid ceilings
fail the gate; `--list` reports current usage.

**Why this indicts us.** Their whole agent-facing rules file is budgeted at 1,950
words. Our `CLAUDE.md` is far past that and growing monotonically, because every
correction is *appended* rather than *applied*. Phase-86.75 measured the
consequence: four different research-source floors live in four instructed-reading
files, including two contradictory numbers inside `CLAUDE.md` itself. A ratcheting
budget is roughly 40 lines of script and would have made that drift impossible to
sustain.

### A4. "One home per fact" and current-state prose — the doctrine behind A3

Their `AGENTS.md` states: *"Comments and docs state complete contracts and
context, not reasoning transcripts. … Do not narrate control flow or tests,
**preserve review history**, or restate code."* Plus "current-state prose, one
physical line per paragraph, one home per fact", all enforced through
`pnpm run doc-sync`.

**Why this indicts us.** We already hold this rule —
`feedback_a_correction_must_replace_not_accompany` is in auto-memory — and
`CLAUDE.md` violates it in nearly every long bullet ("an earlier revision of this
paragraph claimed X — FALSE", "corrected phase-86.28: this sentence used to
say…"). Preserved review history is exactly what the budget in A3 is spent on.
A3 without A4 just forces deletion of the wrong text.

### A5. `--dump-config` — print what actually booted

`dsh --profile web --dump-config` prints the composed plugin tree the machine
actually boots, and "any row it prints can be replaced by a patch of your own".

**Why this indicts us.** Our single most-repeated operational defect class is
*documented-state versus live-state* divergence, and it has its own cluster in
auto-memory: `feedback_committed_is_not_in_force`,
`reference_named_workflow_script_snapshots` (a named Workflow dispatch ran a
script up to 8h36m stale across two commits), `reference_qa_md_deletions_are_inert_until_restart`,
and the standing batched-restart trap in CLAUDE.md ("never claim a config is live
because the file says so"). We have exactly one narrow probe for this —
`scripts/qa/verify_qa_roster_live.sh` — and no general "dump the live
composition" command.

### A6. Durable state, non-durable *authority*

`packages/goal/README.md`: the objective is event-sourced and survives resume,
fork and driver replacement, but **"Activation is never persisted. A fresh cache
and every `agent/session-start` edge disarm it even when replay finds an active
durable phase."** A continuation driver also disarms before unload or "after
durability uncertainty". Resuming restores the goal and its round count *without
initiating work*; only an explicit resume mutation re-arms it.

**Why this matters to us.** `feedback_background_agent_resumption_risk` records
that background agents can self-resume and escalate, and phase-84's auto-dream is
held DISARMED. DeepSeek makes born-disarmed a *structural* property of the state
machine rather than a flag someone has to remember to leave off. Same shape as
our researcher's born-inert `brief_status: INCOMPLETE` marker, applied to
continuation authority — which is the more dangerous thing to get wrong.

### A7. One `block` phase with a code, not many terminal states

Also `packages/goal/README.md`: "provider limits, configured budgets, execution
errors, and requests for human input all use this one durable phase rather than
multiplying lifecycle states". A block carries a policy-owned
lower-kebab-case code plus a normalised free-form explanation.

**Why this matters to us.** This is a direct, shipped answer to our **parked**
phase-86.25 outcome-vocabulary step (`project_phase86_25_outcome_vocabulary`).
Our verdict space has accreted PASS / CONDITIONAL / FAIL plus NO_VERDICT plus
rail-drop plus budget-denial, and the vocabulary sprawl is precisely why 86.25
stalled. Collapsing every abnormal termination into one phase + a code keeps the
verdict axis clean (three values, unchanged semantics) while making the
*reason* machine-readable.

### A8. Loop detection that cannot be laundered

`packages/guard/repeat-tool-reminder/README.md`. An advisory loop-breaker keyed on
`(tool name, canonical arguments)` with deep key-sorted canonicalisation. Two
rules are the sharp ones:

- **"Untracked calls are transparent to the chain."** An excluded bookkeeping tool
  neither increments nor resets the counter, so `grep X → todo_write → grep X`
  still counts as two consecutive `grep X`. Stated explicitly: "bookkeeping tools
  interleaved into a loop must not launder it."
- **"Denied calls count."** Detection sits on `tools/post-execute`, which runs
  even for calls a pre-execute listener denied — "a model hammering a denied call
  is exactly the loop worth breaking."

It also never vetoes: it injects an escalating advisory and leaves the decision
to the model.

**Why this matters to us.** Same insight as A1 approached from the other side. In
our loop, a Main that touches a doc between two otherwise-identical Q/A spawns
launders the repetition; and our budget denial is itself an event that should
count, not reset. Worth reading before implementing A1.

---

## 2. Convergent validation — things we already hold, independently rediscovered

No action required. Value is that our rules are not idiosyncratic; two teams
building long-running agent harnesses hit the same walls.

| DeepSeek statement | Our existing rule |
|---|---|
| "Verify the world, not the self-report … a keyword probe on the agent's own output lets a cheating agent pass" (`docs/testing.md`) | anti-rubber-stamp leg in `qa.md`; `feedback_measure_dont_assert_claims` |
| "A guard only guards if the regression actually fails it … introduce the regression, watch red, revert" (`docs/testing.md`) | `feedback_red_first_guards`, `feedback_zero_assertion_guard_passes_vacuously` |
| "Line coverage is necessary, never sufficient — it proves lines ran, not that the feature works as shipped" | `feedback_a_green_suite_can_be_blind` |
| Postmortem 0003: bare Vite returned HTTP 200 while the page was white — "HTTP readiness, build success, and a boot manifest are different facts" | `feedback_port_200_is_not_a_health_check` |
| "Test the real entry path … the published artifact" (postmortem 0001: 178 green tests, 100% coverage, product completely broken) | `feedback_drive_the_real_thing_for_behavioural_claims` |
| A rejected pre-step "still closes a durable turn that spent no step, so the log records the attempt" | `attempt_budget.py`'s central choice: increment on **attempt, not outcome** |
| "Misconfiguration fails loud at load … never silently skip a missing referent" | `feedback_fail_open_guards_hide_their_own_breakage` |

Their in-repo numbered postmortems (`docs/postmortem/0001`–`0004`) are worth
reading in full as a genre: each is explicitly about *why the process let it
through*, not the one-line fix, and each closes with the guardrails it motivated.
That is the same artifact our `handoff/` cycle produces, kept as a permanent
indexed series rather than archived per-phase.

---

## 3. Where our harness is ahead — do not import these

**Evaluator independence.** `packages/goal/README.md`, Known Limitations:

> **No independent evaluator** — the caller that records completion or blocking is
> authoritative; evaluator-backed certification is deferred to a separate policy layer.

The agent that says "done" is believed. Our entire Layer-3 is built on the
opposite rule — mandatory independent Q/A, self-evaluation forbidden, because
"agents tend to confidently praise their own work". Do not adopt their completion
model. If A7 (the `block` phase) is taken, take the vocabulary and leave the
authority model.

**Budget scope.** Their `maxGoalRounds` (default 256) is documented as
"Round-count budget only — does not meter tokens, currency, wall time, or
provider quotas". Our `attempt_budget.py` meters attempts (5) *and* tokens
(1.2M), both sourced from this repo's measured distribution. Ours is the stronger
bound.

---

## 4. Adoption verdict on the software itself: no

Not a close call, for four independent reasons.

**4.1 The sandbox is a write boundary, not a confidentiality boundary.**
Verified directly in source, not taken from press coverage —
`packages/sandbox/sandbox/README.md:39`: "the seam expresses no network, process,
syscall, device, or credential restrictions"; `sandbox-policy/README.md:68`:
"`SandboxMode` governs file effects; network and process policy are outside its
vocabulary, so no knob here restricts them"; and the Windows backend states it
plainly: "**Writes are restricted; reads, network, and process visibility are
not.**"

On this machine that means any command it runs can read anything the user account
can read — `backend/.env`, `~/.config/gcloud/application_default_credentials.json`,
Alpaca keys, Slack tokens — and egress is unrestricted. For a repo that holds
live brokerage credentials this is disqualifying on its own.

**4.2 Telemetry-off is a config default, not a structural boundary.** The seam
vocabulary is `full | feedback-only | disabled`
(`packages/session/session-telemetry/README.md`), deployment-selected, with
`full` meaning "every event is handed over as it happens". An independent audit
([magnus919](https://magnus919.com/2026/08/deepseek-harness-what-i-found-before-i-let-it-run/))
reached the same reading and additionally found a stable pseudonymous UUID on
DeepSeek API requests and a first-prompt titling call. `api.deepseek.com` is the
only DeepSeek host in non-test source.

**4.3 Maturity and provenance.** `0.1.0-rc.7`, developer preview, with the repo's
own warning that breaking changes are expected and `SESSION_FORMAT_VERSION` held
at `0` with "no compatibility promise". The same independent audit found the
audited source commit declared rc.5 while npm published rc.6, with no public tag
linking them and no npm provenance attestation.

**4.4 It would break the cost constraint anyway.** Running `dsh` means running
DeepSeek models on a metered API, off the flat-fee Max rail that currently makes
Layer-3 subagents free. That violates the standing `$0 metered` away-ops
constraint (`project_away_ops_plan`) — and it buys us a Layer-0 replacement we
did not need, since our harness's problems are all at Layer 3.

If it is ever worth hands-on evaluation, do it the way the independent auditor
did: a throwaway account on an isolated machine, no repo access, Code Mode off,
no MCP servers.

---

## 5. Recency scan (last 2 years)

The subject itself is five days old, so the scan is about the surrounding
frontier rather than superseding prior art.

- **[LangChain, *The Anatomy of an Agent Harness*](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)** — `Agent = Model + Harness`; the filesystem is "the most foundational harness primitive"; notes measurably different scores for the *same model* across harnesses on Terminal Bench 2.0, i.e. harness choice is itself a performance variable. Complements Anthropic's harness-design piece; does not supersede it.
- **[arXiv:2606.04017, *Neither Layer Alone: Epistemic Integrity Requires Hierarchical Joint Design for Long-Running AI Agents*](https://arxiv.org/abs/2606.04017)** — names "Interface Volatility": independently evolving model and harness layers silently alter the semantics of beliefs, capabilities and goals across their boundary. Argues evaluation "should derive from the contract itself", testing whether commitments persist across upgrades. This is a research-grade statement of our stress-test doctrine, and worth a dedicated read before the next model bump — it is the strongest external argument for why the doctrine exists.
- **[arXiv:2604.10352, ClawVM](https://arxiv.org/pdf/2604.10352)** — harness-managed virtual memory for stateful tool-using agents; relevant to compaction/spill, which is the one dsh subsystem we have no analogue for and probably do not need at Layer 3.
- No finding in the window supersedes Anthropic's harness-design guidance as our canonical reference; the DeepSeek release is best read as a second independent implementation of it.

---

## 6. Second pass — `.agents/`, the tree the first read missed entirely

The first pass read ~15 files. A second sweep measured the real scope (219 leaf
packages, 271 READMEs, 111 English docs, **1,404 Agent Notes**, 199,303 lines of
TypeScript) and found that the densest material is in `.agents/`, which the first
pass never opened. Four further mechanisms, all absent here:

### B1. A decision record with a `rejected/` tree and mandatory alternatives

`.agents/notes/README.md`. Path-encoded lifecycle and class
(`{lifecycle}/{class}/yyyy-mm-dd-topic.md`) over a gate-enforced closed class
set, four lifecycles (`proposed` / `implemented` / `rejected` / `archived`), and
an `## Alternatives considered` section that is **mandatory on every note**, for
a reason stated in one line: *"A decision recorded without what it beat invites
re-litigation — the failure Agent Notes exist to prevent."*

Two further rules. `implemented/` notes are kept **current with shipped
reality** — when code later moves a file or changes a default, the note updates
in the same change (facts only, never the decision) — and the gate **rejects
proposal-era headings** (`## Proposal`, `## Plan`, `## Acceptance criteria`) in
an implemented note. And archived notes are **permanently frozen** under an
append-only content manifest, *"never treated as authority for current
behavior"* — which matters here because `CLAUDE.md` and step artifacts do cite
archived handoff briefs as authority.

The `rejected/` tree is the highest-value part, and two of its 11 notes defend
mechanisms this project also depends on. **"Drop durable step boundaries"** was
rejected because a bare start record is the only durable evidence that *"a model
request began but produced no chunks before failing"* — which is precisely our
rail-drop signature. **"Truncate interrupted turns"** was rejected because *"a
single turn can contain substantial real work"*, while the rejected proposal's
own counter-argument is equally sharp: synthetic repair events *"invent events
that never happened"*, so a preserved tail must be **labelled** interrupted, never
silently promoted to real output.

### B2. `dsh-trim-cot-leakage` — a documented, *safe* method for our prose problem

`.agents/skills/dsh-trim-cot-leakage/SKILL.md` exists to remove exactly the prose
class that dominates `CLAUDE.md`: dead design-session citations *including phase
labels*, change narration ("used to", "no longer", "the old X"), and review
choreography ("rejected in review", round attributions). Its one test: *"could a
reader at HEAD, with no access to any session transcript, resolve every reference
and verify every claim?"*

What makes it adoptable rather than dangerous is the **"What is not leakage"**
section, whose keep-rules protect what we must not lose — *"Measured bounds …
the provenance word `measured` is load-bearing"*, *"Counterfactual-present
regression pins (without X, Y happens)"*, and suppression justifications — plus
named **overcorrection traps**: trims that *"flip an obligation into an
endorsement, promote a hypothetical to a shipped feature, delete a true fact, or
drop provenance."* A naive shortening of `CLAUDE.md` would destroy the measured
figures this harness runs on. This method preserves them and converts our
accompany-form corrections into counterfactual-present form.

`docs/AGENTS.md` supplies the surrounding structure: a tier taxonomy stating for
each document tier what does **not** belong there, and a slop checklist.

### B3. Enforcement-bypass as a standing review check

`.agents/skills/dsh-code-review/SKILL.md`: *"**Enforcement:** follow every denial
path to the operation that executes it; exercise direct and alternate callers
that can bypass schemas, prompts, facades, wrappers, or listener ordering."*

Applied here it names our attempt-gate hole from the outside: the gate denies on
the Workflow rail while the Agent-tool rail reaches the same operation ungated.
We found that only because the gate's author wrote the scope bound into the
docstring.

### B4. Attribution needs a correlation contract

`.agents/notes/rejected/simplification/2026-07-12-collapse-workflow-to-foreground-core.md`
rejects deleting their unused workflow-progress events — *"make it useful through
a consumer instead of deleting it"* — and diagnoses why they were unusable:
`WorkflowRunInfo` carries `{id, meta}` but **no parent agent, session, or
tool-call identity**, so *"a global ACP listener could not route an event to the
correct client session."* Their prescription is that any consumer *"starts from a
correlation contract that names the parent agent/session/tool call"*.

That is exactly our gap: the Workflow rail carries `args.step_id` and is gated;
the Agent-tool rail carries a free-text prompt and is not. Relatedly,
`packages/hooks/hook-protocol/README.md` defines `mergeHookOutputs(outputs)` →
**most-restrictive** outcome — a documented merge rule for concurrent hooks,
which we lack despite our hooks running in parallel.

---

## 7. Filed as phase-89

Seven steps appended to `.claude/masterplan.json`, all `pending`, all with
verification commands confirmed **red before the work** (a criterion that is
already green tests nothing). The phase gate forbids any of them changing verdict
semantics.

| Step | P | Subject | Source |
|---|---|---|---|
| 89.1 | P0 | Evidence digest in the attempt ledger — an attempt that changes nothing is denied | A1 |
| 89.2 | P1 | Budget + de-slop the 19,569-word instruction surface | A3, A4, B2 |
| 89.3 | P1 | Correlation contract — close the Agent-tool attribution hole | B4 |
| 89.4 | P2 | One blocked phase + reason code; unparks 86.25 | A7 |
| 89.5 | P2 | Rejected-decisions record with mandatory alternatives | B1 |
| 89.6 | P2 | `dump-live` — print what is actually in force | A5 |
| 89.7 | P3 | Enforcement-bypass check in `qa.md` | B3 |

Ordering constraints: 89.1 before 89.3 (both edit `attempt_gate.py`; do not run
concurrently). 89.2 before 89.7 (which adds text to a file 89.2 budgets).

**Deliberately not filed.** A2 (invariant registry) is design *input* to the
existing **87.8** ("amortize the verification apparatus… `scripts/qa/` grew from
6 files to 102 files, 34,082 lines"), not a separate step — its transferable part
is the ownership rule and the *explained-empty* companion, which makes coverage
legible. A6 (born-disarmed authority) and A8 (unlaunderable loop detection) are
reading assignments recorded in the notes of 89.1 and any future continuation
driver, not steps of their own.

## Sources

**Scope of the read, stated so it is not overclaimed.** The repo is 219 leaf
packages / 271 READMEs / 111 English docs / 1,404 Agent Notes / 199,303 lines of
TypeScript. What follows was read; the **516 implemented Agent Notes and the bulk
of the 199k LOC were not**. Findings above are therefore a floor on what the repo
contains, not a survey of it.

Primary (read in full):
- `deepseek-ai/deepseek-harness` @ `99f6f02` (v0.1.0-rc.7) — `docs/architecture.md`, `docs/agent-lifecycle.md`, `docs/testing.md`, `docs/defensive-patterns.md`, `docs/AGENTS.md`, `AGENTS.md`, all four `docs/postmortem/*`, `.agents/notes/README.md`, the complete `.agents/notes/rejected/` tree (11 notes) and `proposed/` index (25), `.agents/skills/{dsh-trim-cot-leakage,dsh-code-review,dsh-find-simplifications,dsh-pre-push-checks}/SKILL.md`, `packages/goal/README.md`, `packages/runtime-diagnostics/invariants/README.md`, `packages/guard/*/README.md`, `packages/sandbox/*/README.md`, `packages/subagent/subagent/README.md`, `packages/hooks/hook-protocol/README.md`, `packages/session/session-telemetry/README.md`, `packages/compaction/compaction-basic/src/index.ts`, `scripts/verify-doc-budgets.ts`, `scripts/doc-budgets.manifest.json`
- https://deepseek.com/harness/en/
- https://deepseek-harness.github.io/deepseek-harness/en/guide/quickstart
- https://magnus919.com/2026/08/deepseek-harness-what-i-found-before-i-let-it-run/
- https://www.langchain.com/blog/the-anatomy-of-an-agent-harness
- https://arxiv.org/abs/2606.04017

Secondary / snippet-only:
- https://github.com/cordiverse/cordis · https://thenewstack.io/deepseek-harness-open-source-plugins/ · https://www.marktechpost.com/2026/08/17/deepseek-ai-releases-deepseek-harness-in-developer-preview/ · https://venturebeat.com/technology/deepseek-harness-launches-as-open-source-rival-to-claude-code-alongside-v4-pro-on-api-with-higher-prices · https://www.theregister.com/ai-and-ml/2026/08/14/deepseeks-innovative-harness-treats-everything-as-a-plug-in/ · https://arxiv.org/pdf/2604.10352 · https://explainx.ai/blog/cordis-spatiotemporal-composability-explained-2026
