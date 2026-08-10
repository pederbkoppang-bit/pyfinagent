# Research Brief — step 86.31

**Tier:** complex (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Researcher:** Layer-3 Researcher, Workflow rail. **Started:** 2026-08-10.

## Objective

How should an automated EVALUATOR (the Layer-3 Q/A) persist its verdict without gaining the
ability to tamper with the work it is grading, and how should a partially-written verdict be
handled?

Sub-questions:
1. **Least-privilege write for an auditor** — prior art on exactly-one-writable-sink; append-only
   / WORM audit logs; separation of duties; capability security + confused deputy; failure modes of
   a single-path allowlist (traversal, symlink swap, TOCTOU, wrong-subject verdict).
2. **Partial / torn writes and completion markers** — commit markers, checksums/trailers, two-phase
   write + atomic rename, journal semantics. Making an incomplete record *unmistakable*, not merely
   detectable.
3. **Crash-derived evidence** — crash-only software, idempotent recovery; partial output as
   INFORMATION never as RESULT.
4. **Intermittent non-termination of LLM structured output** — 2025-2026 evidence on agents failing
   to emit the required final structured/tool call after a long tool-using session; prevalence,
   correlation with context length / tool-call count; mitigations that do NOT change the model.

Background (NOT the question): on this project the drop is measured as **intermittent** — a dropped
run at 174,664 tokens vs a completed run at 176,900 tokens; a context-reduction fix was tried and
falsified.

---

## Status log (write-first discipline)

- [x] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [x] Brief created (this file) before any web fetch.
- [x] Internal inventory (below)
- [x] External reads (12 read in full)
- [x] Recency scan (dedicated section, 7 queries, 6 findings)
- [x] Envelope (tail of this file)

---

## Internal code inventory (the Explore half)

| File | Lines read | Role | Status |
|---|---|---|---|
| `.claude/hooks/qa-write-guard.sh` | 1-87 (full) | PreToolUse hook, matcher `Write\|Edit`. THE enforcement point. | LIVE, wired |
| `.claude/workflows/qa-verdict.js` | 1-202 (full) | Q/A launch rail; `agentType:'qa'`; verdict = captured return. | LIVE |
| `.claude/workflows/research-gate.js` | 1-711 (full) | Contrasting rail; `agentType:'researcher'` (needs Write); 2-stage w/ artifact cross-check. | LIVE |
| `.claude/agents/qa.md` | 1-557 (full) | Q/A role; read-only guarantee; output schema. | LIVE |
| `.claude/settings.json` | `hooks` block (full) | Hook wiring. | LIVE |
| `CLAUDE.md` | Harness-protocol section | Single-Q/A rule, Workflow-first launch, no-auto-PASS. | LIVE |
| `.claude/rules/research-gate.md` | full | Research floors. | LIVE |
| `.claude/agents/researcher.md` | full | Researcher role + write-first. | LIVE |

### 1. What `qa-write-guard.sh` actually matches on and denies

Wiring: `.claude/settings.json` → `PreToolUse[1]` with `"matcher": "Write|Edit"` → one `command` hook
running `.claude/hooks/qa-write-guard.sh`. (There is a second, unmatched `PreToolUse[0]` entry running
`pre-tool-use-danger.sh` on every tool.) Per auto-memory `reference_claude_code_hooks_run_in_parallel`,
hooks under one matcher run in PARALLEL — no ordering may be assumed between the two PreToolUse entries.

The predicate, verbatim, `qa-write-guard.sh:65-71`:

```python
if agent_type == "qa" and tool_name in ("Write", "Edit"):
    norm = os.path.normpath(file_path.replace("\\", "/"))
    if MEMORY_DIR.rstrip("/") + "/" not in norm + "/":
        print("deny qa-write-outside-memory")
```
with `MEMORY_DIR = ".claude/agent-memory/qa/"` (`:63`). Deny → `exit 2` (`:81`); everything else `exit 0`.

Load-bearing properties, each with an anchor:
- **Identity is `agent_type == "qa"` ONLY** (`:65`). Main's own calls carry no `agent_type` and are always
  allowed (`:14-16` comment). The `researcher` agent type is therefore untouched — stated explicitly at
  `research-gate.js:46-48`.
- **The path test is a SUBSTRING test, not a prefix test** (`:69`: `... not in norm + "/"`). Any path
  anywhere on the filesystem that *contains* the segment `.claude/agent-memory/qa/` is allowed. There is no
  anchoring to `CLAUDE_PROJECT_DIR`, and no `realpath`.
- **`os.path.normpath` collapses `..` lexically but does NOT resolve symlinks** (`:67-68` comment claims
  only the traversal property, correctly). A symlink *inside* `.claude/agent-memory/qa/` pointing outside it
  passes the check — the classic link-following bypass.
- **FAIL-OPEN by design** (`:23-25`, `:38`, `:43`, `:75-76`): malformed JSON, a missing `python3`, or any
  exception yields `allow`. This is deliberate (never brick the session) and is the correct default for a
  *hook*, but it means the guard is an advisory control, not a capability boundary.
- **DISCLOSED GAP at `:18-20`:** "Write/Edit hooks do not intercept Bash subprocess writes; the Main-side
  post-verdict git-status cleanliness rule (per-step-protocol.md section 4) is the covering control." The Q/A
  DOES hold `Bash` (`qa.md:4`), so `bash -c 'echo ... > file'` is outside this hook entirely.
- Always-on shape log to `handoff/logs/qa_write_guard.log` (`:32`, `:55-61`).

### 2. Why the guard exists at all (the residual it was built to cover)

`qa.md:4` lists tools WITHOUT `Write`/`Edit`. But `qa.md:26` sets `memory: project`, and per
`qa-write-guard.sh:5-11` the upstream loader then **auto-injects Read/Write/Edit** so the subagent can curate
its memory files — past the frontmatter allowlist. `qa-verdict.js:183-192` states this is **probe-proven**
(`wf_9277ada4-390`) and that **`disallowedTools` is silently ignored**; it names the residual as
"queued as its own masterplan step" — i.e. this step. So:

> The Q/A **already has** `Write`/`Edit` at the runtime level. The read-only guarantee is enforced by a
> fail-open hook + prose (`qa.md:485-488`), not by the tool surface.

This inverts the naive framing of 86.31. The question is not "should we grant the Q/A a write?" — it already
has one. The question is **which single sink is allowlisted**, and whether widening the allowlist from one
directory to one directory + one file materially changes the tamper surface.

### 3. The contrast rail: why the Researcher survives a drop and the Q/A does not

`research-gate.js:42-48` is explicit that this is the ONE place the Q/A precedent deliberately does not carry:

> "qa-verdict.js pins agentType 'qa' to RESTRICT the surface (Q/A is read-only). The researcher legitimately
> NEEDS Write: write-first is non-negotiable, and a session that cannot clear the gate must still leave a
> partial brief on disk. agentType 'researcher' gets Write via its `memory: project` injection, and the
> qa-write-guard PreToolUse hook matches agent_type == 'qa' only, so it does not block this path."

Consequences that matter for the design:
- The researcher's deliverable is the **artifact on disk**; the envelope is only the audit summary
  (`researcher.md:85-89`). A dropped return still leaves a partial brief.
- The Q/A's deliverable **IS** the return value (`qa-verdict.js:9-12`, `qa.md:64-66`). A dropped return leaves
  NOTHING. That asymmetry is the whole of 86.31.
- The researcher rail does NOT trust the agent's self-report: `enforceGate` recomputes (`research-gate.js:349-508`)
  and a **stage-2 independent agent** (`:616-652`, `agentType:'Explore'`, `effort:'low'`) reads the brief. The
  producer never attests to its own artifact (`:304-305`). This is the pattern a Q/A write-sink must copy.
- `research-gate.js:305-317` already carries the discipline note that a *structural* check ("a section exists")
  must never be named as if it were *semantic* ("the scan was substantive"). Directly reusable for a verdict
  completion marker: `verdict_file_complete` would be structural only.

### 4. Where Main persists the verdict today (the tamper-relevant path)

- `qa-verdict.js:9-12` + `qa.md:80-83`: **Main transcribes the returned verdict VERBATIM** into
  `handoff/current/evaluator_critique.md`. Main is the scribe; Q/A never authors a file.
- `qa.md:442-454` (phase-71.3): Main ALSO persists `handoff/current/evaluator_critique.json` — the same
  object plus Main-injected `step_id` / `cycle_num`, read deterministically by the status-flip / live_check
  gate as `verdict == "PASS" AND ok == true`. **Q/A "stays read-only and never writes files."**
- So today there are exactly two verdict artifacts, both written by Main. Adding a Q/A-written sink creates a
  THIRD, and the design must say which one is authoritative and how they are reconciled.

### 5. The invariant that must survive this change (verbatim)

`CLAUDE.md` Harness Protocol, and mirrored at `qa.md:88-90` and `qa-verdict.js:11-12`:

> "An errored/empty return is **NO VERDICT, never PASS** → fall back to the Agent-tool path."

and `qa.md:499-504`: on `stop_hook_active`, exit verdict-NEUTRAL, "Never return ok:true from a
loop-prevention exit -- an evaluator must have no auto-PASS path".

A partially-written verdict file is the new way to violate this rule: if a truncated file can be read as a
verdict, the no-auto-PASS invariant is defeated through the artifact rather than through the return value.

### 6. Precedent already in-tree for "artifact exists ≠ artifact is valid"

`CLAUDE.md` `verification.live_check` bullet: `.claude/hooks/lib/live_check_gate.py` "currently only checks the
artifact FILE EXISTS, never its content — an empty file passes; hardening it to require non-empty + at least
one fenced block is queued (phase-75.5 follow-up)." That is the exact failure mode sub-question (2) is about,
already observed in this repo.

---


## Read in full (WebFetch, full page — counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://www.sqlite.org/atomiccommit.html | 2026-08-10 | Official docs (SQLite) | WebFetch, full | The commit is ONE observable fact; a partial journal is made UNROLLBACKABLE by construction (page-count starts at 0) |
| 2 | https://arxiv.org/html/2503.13657v2 | 2026-08-10 | Peer-reviewed/preprint (MAST) | WebFetch, full | FM-3.1 Premature termination = **7.82%** of failures; FC3 Task Verification = **21.30%**; verifiers themselves fail |
| 3 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | 2026-08-10 | Peer-reviewed (Saltzer & Schroeder 1975) | WebFetch, full | Least privilege, complete mediation, fail-safe defaults — the canonical wording |
| 4 | https://cwe.mitre.org/data/definitions/367.html | 2026-08-10 | Official (MITRE CWE) | WebFetch, full | TOCTOU: "the most basic advice ... is to not perform a check before the use" |
| 5 | https://lwn.net/Articles/457667/ | 2026-08-10 | Authoritative (LWN) | WebFetch, full | The 5-step atomic-replace recipe incl. the **directory fsync**; explicitly NO checksum/corruption discussion |

### Source 1 — SQLite, "Atomic Commit In SQLite" (official docs)

The single most transferable design in this brief. Verbatim:

> "After the database changes are all safely on the mass storage device, the rollback journal file is deleted.
> **This is the instant where the transaction commits.**"

> "The existence of a transaction depends on whether or not the rollback journal file exists and the deletion
> of a file appears to be an atomic operation from the point of view of a user-space process. Therefore, a
> transaction appears to be an atomic operation."

And on making a partial record **unmistakable rather than merely detectable** (§6.2):

> "SQLite records the number of pages in the rollback journal in the header of the rollback journal. **This
> number is initially zero.** So during an attempt to rollback an incomplete (and possibly corrupt) rollback
> journal, the process doing the rollback will see that the journal contains zero pages and will thus make no
> changes to the database."

> "Prior to a commit, the rollback journal is flushed to disk ... and **only then** is the page count in the
> header changed from zero to the number of pages."

Second line of defence (§6.2): "SQLite also uses a 32-bit checksum on every page of data in the rollback
journal. ... **If an incorrect checksum is seen, the rollback is abandoned.**" Rationale: "SQLite assumes that
the underlying filesystem can reorder write requests and that the page count can be burned into oxide first
even though its write request occurred last."

And the cheapest possible commit marker (§3.11 / PERSIST mode):

> "Overwriting the header of the journal with zeros is not atomic, but **if any part of the header is
> malformed the journal will not roll back.** Hence, one can say that the commit occurs as soon as the header
> is sufficiently changed to make it invalid. Typically this happens as soon as **the first byte of the header
> is zeroed**."

Design lesson for 86.31: the DEFAULT STATE OF A PARTIAL RECORD IS "INERT". SQLite does not write a full record
and then hope a reader notices truncation; it writes a record that is *incapable of being acted on* until a
final, small, atomic act flips it to actionable. Applied here: a verdict file must be born non-authoritative.

### Source 2 — MAST, "Why Do Multi-Agent LLM Systems Fail?" (arXiv:2503.13657v2)

The empirical anchor for sub-question (4). Taxonomy with measured frequencies:

- **FC1 Specification Issues — 41.77%**: FM-1.1 disobey task spec 10.98%, FM-1.2 disobey role spec 0.5%,
  FM-1.3 step repetition 17.14%, FM-1.4 loss of conversation history 3.33%,
  **FM-1.5 unaware of termination conditions 9.82%**.
- **FC2 Inter-Agent Misalignment — 36.94%**: FM-2.6 reasoning-action mismatch 13.98%, FM-2.2 fail to ask for
  clarification 11.65%, FM-2.3 task derailment 7.15%, others <4%.
- **FC3 Task Verification — 21.30%**: **FM-3.1 premature termination 7.82%**, FM-3.2 no or incomplete
  verification 6.82%, FM-3.3 incorrect verification 6.66%.

Method: 200+ traces across 7 MAS frameworks, 150+ human-annotated by 6 expert annotators (Grounded Theory),
>20 h annotation per annotator; inter-annotator Cohen's Kappa 0.24 → 0.92 → 0.84, 0.79 on unseen systems; LLM
annotator Kappa 0.77 / accuracy 94%.

Two findings that bear directly on this step:
1. **Verifiers themselves are a failure mode.** "Current verifiers often only perform superficial checks
   (e.g., missing comments or code compilation) and struggle to ensure deeper correctness." Recommendation:
   "multi-level checks assessing low-level correctness alongside high-level objectives."
2. **The fix is structural, not a bigger model.** "Many MAS failures arose from the challenges in
   organizational design and agent coordination rather than the limitations of individual agents" and
   "**Improvements in the base model capabilities will be insufficient to address the full MAST.**" Measured
   structural interventions: improved role specification +9.4% success (ChatDev); enhanced multi-level
   verification **+15.6 percentage points absolute**.

This is the literature's direct answer to "mitigations that do NOT involve changing the model", and it
independently corroborates the project's own measurement that the drop is model-agnostic.

### Source 3 — Saltzer & Schroeder (1975), "Basic Principles of Information Protection"

Verbatim principles that govern the guard's design:

- **Least privilege**: "Every program and every user of the system should operate using the least set of
  privileges necessary to complete the job."
- **Complete mediation**: "Every access to every object must be checked for authority."
- **Fail-safe defaults**: "Base access decisions on permission rather than exclusion." Rationale: "a design or
  implementation mistake in a mechanism that gives explicit permission tends to fail by refusing permission, a
  safe situation, since it will be quickly detected."
- **Economy of mechanism**: "Keep the design as simple and small as possible."
- **Separation of privilege**: a mechanism requiring two keys is "more robust and flexible than one that
  allows access to the presenter of only a single key."

Also relevant: the paper is skeptical of "proposals to gain performance by remembering the result of an
authority check", noting cached results "must be systematically updated".

**Tension to disclose, not paper over:** `qa-write-guard.sh` is FAIL-OPEN (`:23-25`, `:75-76`), which is the
*opposite* of fail-safe defaults; and it is not complete mediation because Bash writes bypass it entirely
(`:18-20`). The existing design consciously trades Saltzer-compliance for never bricking a session. Any
86.31 proposal that ADDS a permitted path must not silently widen that already-imperfect mediation.

### Source 4 — CWE-367, Time-of-check Time-of-use (MITRE)

> "The product checks the state of a resource before using that resource, but the resource's state can change
> between the check and the use in a way that invalidates the results of the check."

Mitigation, verbatim: "**The most basic advice for TOCTOU vulnerabilities is to not perform a check before the
use.**" Others: make check+use a single atomic operation; "Ensure that locking occurs before the check, as
opposed to afterwards"; recheck after use; minimise the race window. Likelihood of exploit: **Medium**.

Directly named children/peers: **CWE-363 Race Condition Enabling Link Following**, CWE-386 Symbolic Name not
Mapping to Correct Object; attack patterns **CAPEC-27 Leveraging Race Conditions via Symbolic Links** and
CAPEC-29. Demonstrative example 2 is exactly our shape: check permissions with `access()`, "an attacker
replaces the file with a symbolic link between these calls, causing the program to operate on an unauthorized
target file using root privileges."

**Mapped onto `qa-write-guard.sh`:** the hook is a *check* (`:65-71`) and the tool performs the *use*. That is
CWE-367 by construction, and it is unavoidable for any PreToolUse-hook design — the hook cannot perform the
write itself. The mitigations that DO apply are (a) shrink what a successful race can reach, and (b) make the
check operate on a name that cannot be re-pointed (see the symlink discussion below).

### Source 5 — LWN, "Ensuring data reaches disk" (Jeff Layton / LWN)

The canonical two-phase-write recipe, verbatim:

> "1. create a new temp file (on the same file system!)
> 2. write data to the temp file
> 3. fsync() the temp file
> 4. rename the temp file to the appropriate name
> 5. fsync() the containing directory"

Skipping step 3 "can result in the loss of existing data"; the sequence gives "an atomic update of the file,
so that other readers get one copy of the data or another."

**Explicit negative finding, and it matters:** the article contains NO discussion of checksums or
application-level integrity verification, and none of recovery from a partially written file. Atomic rename
solves *torn visibility* — a reader never sees half a file — but it says nothing about a file that was
*completely written and semantically incomplete* (the agent stopped mid-verdict and the harness renamed what
it had). That is precisely the pyfinagent hazard, and it is why rename-alone is insufficient here and SQLite's
"born inert + explicit commit marker" is the pattern that actually covers it.

---

### Source 6 — Candea & Fox, "Crash-Only Software", HotOS-IX 2003 (read in full)

Fetched `https://dslab.epfl.ch/pubs/crashonly.pdf`; WebFetch returned binary, so per
`.claude/rules/research-gate.md` step 3 the PDF was extracted with **pdfplumber** (33,105 chars, all 6 pages
incl. references). Verbatim:

> "Crash-only programs crash safely and recover quickly. There is only one way to stop such software — by
> crashing it — and only one way to bring it up — by initiating recovery."

> "It is impractical to build a system that is guaranteed to never crash ... Since crashes are unavoidable,
> software must be at least as well prepared for a crash as it is for a clean shutdown. But then — in the
> spirit of Occam's Razor — if software is crash-safe, why support additional, non-crash mechanisms for
> shutting down?"

The state-externalisation rule, which is the load-bearing one here:

> "All important non-volatile state is managed by dedicated state stores, leaving applications with just
> program logic. ... **Applications become stateless clients of the state stores**, which allows them to have
> simpler and faster recovery routines."

> "These state stores must also be crash-only, **otherwise the problem has just moved down one level.**"

On in-flight work — the direct answer to sub-question (3):

> "Requests are entirely self-describing, by making the state and context needed for their processing
> explicit. This allows a fresh instance of a rebooted component to pick up a request and continue from where
> the previous instance left off. Requests also carry information on whether they are idempotent, along with
> a time-to-live."

> "Recovering from a failed idempotent sub-operation entails simply reissuing it; for non-idempotent
> operations, the system can either roll them back, apply compensating operations, or tolerate the
> inconsistency resulting from a retry."

And the reliability argument for exercising the recovery path constantly:

> "Recovery code deals with exceptional situations, and must run flawlessly. Unfortunately, exceptional
> situations are difficult to handle, occur seldom, and are not trivial to simulate during development; this
> often leads to unreliable recovery code. In crash-only systems, however, **recovery code is exercised every
> time the system starts up**, which should ultimately improve its reliability."

Also relevant: progress counters live OUTSIDE the component because insider counters are less trustworthy —

> "Components themselves can also implement progress counters that more accurately reflect application
> semantics, but they are **less trustworthy, because they are inside the components**."

**Critical reading for 86.31.** Crash-only says: RE-RUN the unit of work, do not salvage its debris. Nowhere
does the paper promote a partial in-memory result to an authoritative one; the entire architecture is built so
that the retry is cheap enough that salvage is unnecessary. Applied here: a truncated verdict file is
*evidence that a Q/A ran and roughly where it got to* — a diagnostic — and the correct response is to re-spawn
a fresh Q/A, exactly as `qa.md:88-90` already prescribes for an empty return. The paper also supplies the
principle that the verdict store must be a **dedicated state store outside the evaluator**, and that a
progress/completion signal produced INSIDE the evaluator is the least trustworthy kind.

### Source 7 — "Constraint Tax in Open-Weight LLMs: Tool Calling Suppression Under Structured Output Constraints" (arXiv:2606.25605)

The most on-point paper for sub-question (4), and it **partially contradicts** the pyfinagent symptom — which
makes it the most useful source in the set.

Phenomenon: "multiple open-weight models cease invoking tools when Tool Calling and Structured Output
constraints are simultaneously enabled." Pipeline breaks selectively — Task Understanding OK, Tool Planning
OK, **Tool Execution FAILS**, Response Generation OK.

Measured (Table 7, tool-invocation rate T1 tools-only / T2 joint / T3 schema-only, and suppression rate SR):

| Model | T1 | T2 | T3 | SR |
|---|---|---|---|---|
| GPT-5.4-mini (closed) | 100% | 100% | 100% | **0%** |
| Qwen3.6-35B-A3B | 100% | 0% | 80-100% | 100% |
| Qwen3.5-122B-A10B | 100% | 0% | 100% | 100% |
| GPT-OSS-20B | 100% | 0% | 100% | 100% |
| Nemotron 3 Super | 100% | 0% | 100% | 100% |
| Qwen3.5-397B-A17B | 100% | 0% | 100% | 100% |
| Qwen3-VL-235B-Thinking | 100% | 0% | 100% | 100% |

Determinism: "5 test rounds independently under each condition"; "No valid tool call events were observed in
any evaluated session"; "Parser-level inspection confirms missing tool calls are genuine behavioral outcomes."
The phenomenon is reported as **deterministic, not intermittent**.

Confound controls: suppression "remained unchanged" across simple (1-3 field), medium (5-10) and production
(20+) schemas; "Results remained consistent" across `tool_choice` optional / prompted / `"required"`.
**No context-length or prompt-length correlation is reported** — the paper does not analyse it.

Mitigation — Transparent Two-Pass Execution (pass 1 tools ON / schema OFF; pass 2 tools OFF / schema ON with
results injected): tool-invocation 0% → 100%, JSON compliance 100% → 100%, end-to-end success **0% → 100%**,
avg tool calls/session 0 → 5-8. Cost: doubles inference rounds, repeated context transmission. Crucially:

> "This design does not modify model weights, training data, inference kernels, or serving frameworks.
> Therefore, it can be deployed directly within existing production Agent systems without requiring model
> retraining."

And weight-level fixes were tried and failed: fine-tuning variants (SFT, GRPO) "failed to eliminate
suppression."

**Why this is the key adversarial data point.** The measured shape is 0%/100% and deterministic on
open-weight models, and **0% suppression on the closed frontier model**. pyfinagent's symptom is neither:
it is intermittent, on a closed frontier model, on the schema-constrained return itself rather than on
intermediate tool calls. So this paper does NOT explain the pyfinagent drop, and a design that assumes it
does would be mis-aimed. What it *does* establish, robustly, is the architectural lesson: **when a
schema-constrained emission is the sole channel for a result, a single suppression event destroys the whole
result, and the fix is to decouple the channels rather than to change the model.**

### Source 8 — "When Web Agents Finish but Still Fail" (arXiv:2606.20724)

The single most decision-relevant finding for pyfinagent's falsified context-reduction fix.

Definition of FM2: "The agent emits a special token (`<final_answer>`) before exhausting at least 50% of the
round budget, with element-wise recall <0.6 on the gold answer."

Prevalence at 16k context, r=16 — Human 34/103 imperfect episodes (33%); Balanced 29/104 (28%); Synth-Heavy
39/100 (39%).

Verbatim, and this is the load-bearing sentence:

> "**Premature termination is unaffected by budget scaling: the agents terminate well before exhausting the
> round cap, regardless of context size.**"

Scaling 16k/r=8 → 16k/r=16 gives minimal improvement; 64k/r=16 shows no consistent gain. Reproducible trigger:
"List-valued questions ('list all XX,' 'every YY,' 'for each ZZ') where the gold answer contains >=4 items
spanning heterogeneous source pages." Worked case Q47: terminated in 5-7 rounds with 2/7 gold items while
`Rformat=1.0, Rprocess=1.0, Rcorrect=0.78` — i.e. **format and process credit masked the correctness
failure**, which is precisely the "a partial artifact scores as complete" hazard of sub-question (2).

Harness-level mitigations named: "**coverage-gated stopping**" and "**final-answer verification** rather than
simply increasing context, rounds, or synthetic data."

**Direct corroboration:** this is independent external evidence that context reduction is the WRONG lever for
a termination failure — matching the project's own falsified experiment. Do not spend the 86.31 design on
shrinking the Q/A prompt.

### Source 9 — HORIZON, "The Long-Horizon Task Mirage?" (arXiv:2604.11978v1)

7-category taxonomy — Environment Error [S], Instruction Error [S], Catastrophic Forgetting [L], False
Assumption [S], Planning Error [S], History Error Accumulation [L], Memory Limitation [L] ([L] =
predominantly long-horizon-specific). 3100+ failure trajectories from 700+ tasks across WebArena, AgentBench,
MAC-SQL, Isaac Sim; models GPT-5 variants and Claude-4-Sonnet; 40-trajectory human validation, kappa 0.61
(inter-annotator) / 0.84 (human-judge). Split: Process-level risks 72.5% vs Design-level 27.5%.

**Honest negative findings, recorded because they bound what can be claimed:** the paper does **not**
explicitly correlate failure rate with context-window utilisation, token count, turns or tool-call frequency
("the mechanism ... remains unanalyzed"), and does **not** systematically characterise intermittent vs
deterministic failure. It states "model scaling alone is unlikely to resolve the dominant failure mechanisms."

So: no source found quantifies a token-count threshold for emission failure. The pyfinagent measurement
(dropped at 174,664 vs completed at 176,900) is, as far as this survey found, **ahead of the published
literature**, and the correct posture is to design for an intermittent fault of unknown rate rather than to
look for a threshold to stay under.

### Source 10 — "Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures" (arXiv:2608.02645)

Atomicity for agent actions: a tool call is atomic only when "the response r faithfully encodes the state
transition from S to S'". Failure mode 3 is **Partial Success** — "The action is a compound operation. An
internal service error causes only a subset of effects to be applied."

The key warning for a completion-marker design:

> "A verifier that only checks [row.exists] will pass on partial success (Failure Mode 3). So the system must
> follow all parts of the formalizations."

i.e. **existence checks pass on partial success** — exactly the `live_check_gate.py` "an empty file passes"
defect already in this repo.

Mechanism used: idempotency keys `k = hash(agent_id, action_type, payload, timestamp_bucket)` so that
"repeated requests with the same key are guaranteed to produce at most one effect". Note the key binds the
**agent identity and the payload**, which is the countermeasure to "the auditor writing a verdict for a
DIFFERENT subject than the one it evaluated".

Measured (activate_customer): success 92→100 (low fault), 80→100 (medium), 64→100 (high, **+36 pts**);
duplicate actions 20→0 / 44→16 / 72→20. Ablation: baseline ~58% success, verify-only ~80%, duplicates 42%→20%.
Explicit limitation: "the wrapper cannot eliminate failures when the verifier itself observes outdated or
incomplete state." Two-phase commit and write-ahead logging are NOT used (checked; the paper does not discuss
them), and it does not address an agent failing to emit a final call.

### Source 11 — "Empirical Study for Structured Output Control in LLMs for SE" (arXiv:2606.09395)

Unconstrained prompting: CallNavi 42.5% combined syntax/structural errors (LLaMA-3.1-8B); BFCL v2 73.3%
syntax/structural across models; BigCodeBench <20% format failures. Grammar-constrained decoding
(Outlines/XGrammar) collapses syntax errors (LLaMA-3.1-8B on BFCL v2: 1,833 → 2) but **structural errors
persist** (~290-305 on CallNavi) and exact match barely moves (0.4883 → 0.5130). Template-based TTMG: 28 vs
1,833 syntax errors, but 56/38 structural and 185/544 value errors.

Central finding, verbatim: forcing syntactically valid outputs exposes that "**deeper structural and semantic
errors remain unsolved even when syntax is fully controlled**."

Negative finding recorded honestly: **empty/truncated output rates are NOT measured** in this paper, and
retry/two-pass loops are described but "not measured or recommended" — "the model is prompted to fix its
answer ... such loop may repeat until output parses correctly (or retry limit is hit)."

This is the schema-conformance-is-structural-only argument that `research-gate.js:19-24` already relies on,
now with primary numbers behind it.

### Source 12 — CWE-59, Improper Link Resolution Before File Access (MITRE)

> "The product attempts to access a file based on the filename, but it does not properly prevent that filename
> from identifying a link or shortcut that resolves to an unintended resource."

Consequence: "An attacker may be able to traverse the file system to unintended locations and read or
overwrite the contents of unexpected files. **If the files are used for a security mechanism then an attacker
may be able to bypass the mechanism.**" Mitigation: "Follow the principle of least privilege ... **Denying
access to a file can prevent an attacker from replacing that file with a link to a sensitive file.**"

Family: parent CWE-706 Use of Incorrectly-Resolved Name or Reference; variants CWE-61 UNIX symlink, CWE-62
UNIX hard link, CWE-64/65 Windows shortcut/hard link, CWE-1386 junction; can follow CWE-73 External Control of
File Name and **CWE-363 Race Condition Enabling Link Following**.

**Direct hit on `qa-write-guard.sh:67-69`:** `os.path.normpath` is a *lexical* canonicaliser. It defeats
CWE-22-style `../` traversal (correctly, and the code comment says exactly that and no more) but it does
nothing about CWE-59. A symlink placed at `.claude/agent-memory/qa/<name>` pointing anywhere passes the
substring test. Note the Q/A cannot create a symlink with `Write`, but it *can* with `Bash` (`ln -s`), and
Bash is outside the hook (`:18-20`). The two disclosed gaps compose.

---

## Recency scan (last 2 years, 2024-2026) — MANDATORY SECTION

**Queries run (three-variant discipline per `.claude/rules/research-gate.md`):**

| Variant | Query | Purpose |
|---|---|---|
| Year-less canonical | `append-only audit log write-once WORM separation of duties auditor least privilege` | prior art on auditor write privilege |
| Year-less canonical | `confused deputy problem capability-based security ambient authority least privilege` | Q1 founding literature |
| Year-less canonical | `crash-only software Candea Fox recovery idempotent partial output` | Q3 founding paper |
| Year-less canonical | `tamper-evident append-only log hash chain audit integrity design` | Q1/Q2 integrity mechanisms |
| Current-year 2026 | `LLM agent premature termination fails to emit final tool call long horizon 2026` | Q4 frontier |
| Last-2-year 2025 | `"structured outputs" LLM failure to return tool call end of turn 2025 mitigation retry` | Q4 recency |
| Current-year 2026 | `AI evaluator agent separation of duties tamper write access verdict artifact 2026` | Q1 applied to agents |

**Result: 6 new findings in the 2024-2026 window that COMPLEMENT (and in one case partially contradict) the
canonical sources.**

1. **arXiv:2606.20724 (2026)** — premature termination is **uncorrelated with context budget**. This
   SUPERSEDES the intuitive "shrink the prompt" remedy and independently corroborates pyfinagent's own
   falsified experiment. Highest-value recency finding in this brief.
2. **arXiv:2606.25605 (2026), "The Constraint Tax"** — schema constraints can suppress tool emission
   entirely; measured 100% suppression on 6 open-weight models, **0% on the closed frontier model**;
   fine-tuning did not fix it; a two-pass harness change took end-to-end success 0% → 100%. Partially
   CONTRADICTS the assumption that the pyfinagent drop is an instance of this class (see "Consensus vs
   debate").
3. **arXiv:2503.13657 MAST (2025-2026)** — first quantified taxonomy: premature termination 7.82%, task
   verification failures 21.30% overall; structural verification improvements worth **+15.6 pts absolute**;
   explicit statement that better base models will not close the gap.
4. **arXiv:2604.11978 HORIZON (2026)** — 3100+ trajectories; confirms no published context-length/emission
   correlation exists yet. A negative result, and it bounds what 86.31 may claim.
5. **arXiv:2608.02645 (2026)** — idempotency keys binding `agent_id + action_type + payload`; and the warning
   that an existence-only verifier "will pass on partial success".
6. **arXiv:2606.09395 (2026)** — primary numbers for "constrained decoding fixes syntax, not semantics."

**No new work in the window supersedes** Saltzer & Schroeder (1975), Hardy's confused deputy (1988),
Candea & Fox (2003), or the SQLite atomic-commit design. The 2026 agent-security material found
(Security Boulevard, insiderisk.io, iternal.ai) is community/vendor tier and adds no mechanism beyond
"give every agent a unique scoped identity, enforce least privilege, keep tamper-evident evidence" — which
is Saltzer restated. Recorded as snippet-only; it does not change the design.

---

## Internal inventory, part 2 — THE ESCALATION PATH (found late; most decision-relevant fact in this brief)

Files: `.claude/hooks/lib/verdict_gate.py` (7,728 bytes, read), `.claude/hooks/lib/live_check_gate.py`,
`scripts/housekeeping/verify_handoff_layout.py`, `.claude/hooks/archive-handoff.sh`,
`.claude/hooks/auto-commit-and-push.sh:202-208`.

**`evaluator_critique.json` is not an inert record — it is the INPUT TO AN AUTOMATED GATE.**
`auto-commit-and-push.sh:202-208` wires `VERDICT_GATE_HELPER=.claude/hooks/lib/verdict_gate.py`, and
`verdict_gate.py::gate_decision` (`:56-79`) returns:

```python
verdict = data.get("verdict")
if verdict is None: ...            # -> 'proceed' (fail-open)
ok = data.get("ok", True)          # :76  -- DEFAULTS TO TRUE WHEN ABSENT
if str(verdict).strip().upper() == "PASS" and bool(ok):
    return "passed"
return "hold"
```

and `resolve_verdict_source` (`:106-117`) searches, **in priority order**:

1. `handoff/current/evaluator_critique_<sid>.json`  ("current:per-step")  <-- HIGHEST PRIORITY
2. `handoff/current/evaluator_critique.json`        ("current:rolling")
3. `handoff/archive/phase-<sid>/evaluator_critique_<sid>.json` ("archive:step")

Meanwhile `scripts/housekeeping/verify_handoff_layout.py:42,49` declares
`ROLLING_KEEP_PREFIXES = ("evaluator_critique_",)` accepted when `name.endswith(".json")` — so
`handoff/current/evaluator_critique_86.31.json` is layout-LEGAL.

**Therefore: the obvious, natural, layout-legal name for a Q/A-written verdict sink is EXACTLY the
highest-priority input of the gate that authorises the auto-commit and push.** Granting the Q/A write access
to that name would not merely let the evaluator record its verdict — it would let the evaluator **authorise
its own step closure and push**. That is a textbook confused-deputy escalation (the hook is the deputy; the
Q/A supplies the name and content it acts on) and it is precisely the "tamper with the work it is grading"
risk the step names.

Two further facts that bear on the design:
- **`ok` defaults to TRUE when absent** (`verdict_gate.py:76`). A JSON object carrying `verdict: "PASS"` but
  missing `ok` gates as `passed`. A syntactically-truncated file fails `json.loads` and falls to `proceed`
  (fail-open, not passed), so truncation is currently survivable — but only by accident of JSON syntax, not
  by design. This is the local instance of arXiv:2608.02645's "a verifier that only checks [row.exists] will
  pass on partial success".
- **`live_check_gate.py:72`** is literally `return "passed" if artifact.exists() else "skip"` — existence-only,
  content never inspected. CLAUDE.md already discloses this ("an empty file passes"). Same lesson, second
  instance, already in-tree.
- **`archive-handoff.sh:146`** copies only `contract.md experiment_results.md evaluator_critique.md
  research.md research_brief.md`; the per-step `.json` files are NOT in that list (only `${sid}-*.md` /
  `phase-${sid}-*.md` are moved at `:160`). A new per-step sink would linger in `handoff/current/` unless the
  archive list is extended — a housekeeping item, not a blocker.

---

## Key findings (each cited)

1. **The Q/A already holds Write/Edit; only the allowlist is in question.** `memory: project` in `qa.md:26`
   makes the loader inject Read/Write/Edit past the `tools:` allowlist; `disallowedTools` is silently ignored
   (probe-proven, `qa-verdict.js:183-192`). The read-only guarantee is prose + a fail-open hook, not a
   capability boundary. (Internal: `qa-write-guard.sh:5-11`.)

2. **The sink must NOT be, or resolve to, `evaluator_critique*.json` in `handoff/current/`.** That name is the
   top-priority input of `verdict_gate.py` (`:106-117`), which the auto-commit/push hook consults
   (`auto-commit-and-push.sh:202-208`). Writing there = the evaluator authorising its own push. This is the
   confused deputy: "a computer program that is tricked by another program into misusing its authority"
   (https://cwe.mitre.org/data/definitions/59.html consequence text; Hardy 1988 via snippet).

3. **Least privilege says one sink; fail-safe defaults say deny-by-default; the current hook does neither
   perfectly.** "Every program ... should operate using the least set of privileges necessary" and "Base
   access decisions on permission rather than exclusion" — Saltzer & Schroeder 1975
   (https://web.mit.edu/Saltzer/www/publications/protection/Basic.html). `qa-write-guard.sh` is fail-OPEN
   (`:23-25`, `:75-76`) and is not complete mediation (Bash writes bypass it, `:18-20`). A new grant must
   therefore be backed by a **reader-side** rule, not only by the hook.

4. **A single-path allowlist has four named failure modes, and this implementation has three of them.**
   (a) *Substring not prefix* — `qa-write-guard.sh:69` tests `".claude/agent-memory/qa/" in norm + "/"`, so
   any path anywhere containing that segment passes; no anchoring to `CLAUDE_PROJECT_DIR`.
   (b) *Symlink swap* — `os.path.normpath` is lexical and does not resolve links; CWE-59 warns "If the files
   are used for a security mechanism then an attacker may be able to bypass the mechanism"
   (https://cwe.mitre.org/data/definitions/59.html). The Q/A cannot `ln -s` with `Write`, but it holds `Bash`.
   (c) *TOCTOU* — a PreToolUse hook is a check preceding a use by construction; CWE-367's own advice is "The
   most basic advice for TOCTOU vulnerabilities is to not perform a check before the use"
   (https://cwe.mitre.org/data/definitions/367.html), which a hook cannot honour. Mitigation available:
   shrink the blast radius (one FILE is a smaller target than one DIRECTORY) and re-verify after the fact.
   (d) *Wrong-subject verdict* — the countermeasure already exists in-tree: `research-gate.js:208-211` has
   the SCRIPT compute the artifact path from the caller's `step_id` and tell the agent the exact path, "so
   write-first and the artifact cross-check cannot refer to different files." Copy that verbatim; never let
   the agent name its own sink.

5. **Atomic rename solves torn VISIBILITY, not semantic incompleteness.** The LWN recipe
   (https://lwn.net/Articles/457667/) is temp -> write -> `fsync(temp)` -> `rename` -> `fsync(dir)`, and the
   article contains **no** discussion of checksums or partially-written-file recovery. A verdict that was
   completely flushed but stopped mid-thought renames perfectly and is still not a verdict.

6. **SQLite's answer is the transferable one: make a partial record INERT BY CONSTRUCTION, then commit with
   one small atomic act.** The journal's page count "is initially zero. So during an attempt to rollback an
   incomplete ... journal, the process doing the rollback will see that the journal contains zero pages and
   will thus make no changes"; and "the commit occurs as soon as the header is sufficiently changed to make
   it invalid. Typically this happens as soon as the first byte of the header is zeroed"
   (https://www.sqlite.org/atomiccommit.html). Plus a checksum as "a second line of defense" — "If an
   incorrect checksum is seen, the rollback is abandoned."

7. **Crash-only says re-run, do not salvage.** "There is only one way to stop such software—by crashing
   it—and only one way to bring it up—by initiating recovery"; state lives in dedicated stores, "otherwise
   the problem has just moved down one level"; and progress counters inside a component are "less
   trustworthy, because they are inside the components" (Candea & Fox 2003,
   https://dslab.epfl.ch/pubs/crashonly.pdf). A dropped Q/A is a crash; the recovery is a fresh Q/A, which is
   already `qa.md:88-90`.

8. **A re-spawn after a DROP is not verdict-shopping.** `qa.md:506-511`: "The distinguishing test: did the
   files change between spawns?" A dropped return produced **no verdict**, so there is no prior opinion to
   shop away from. This only stays true if a partial artifact is never itself readable as a verdict — which
   is an argument FOR strict inertness, not merely a nicety.

9. **Premature termination does not scale away with context.** "Premature termination is unaffected by budget
   scaling: the agents terminate well before exhausting the round cap, regardless of context size"
   (https://arxiv.org/html/2606.20724). 28-39% of imperfect episodes. Independently corroborates
   pyfinagent's own falsified context-reduction experiment. Recommended levers are harness-level:
   "coverage-gated stopping" and "final-answer verification".

10. **Quantified prevalence exists and the fix is structural.** MAST: premature termination **7.82%**,
    FC3 task-verification failures **21.30%**, and enhanced multi-level verification is worth **+15.6 points
    absolute**; "Improvements in the base model capabilities will be insufficient to address the full MAST"
    (https://arxiv.org/html/2503.13657v2).

11. **Decoupling the constrained channel from the work channel is the measured mitigation.** Two-pass
    execution took tool invocation 0% -> 100% and end-to-end success 0% -> 100% with "no ... model
    retraining" (https://arxiv.org/html/2606.25605v1). Fine-tuning (SFT, GRPO) "failed to eliminate
    suppression." The pyfinagent analogue of pass-1/pass-2 is exactly "artifact on disk" vs
    "schema-constrained return".

12. **Constrained decoding fixes syntax, not truth.** BFCL v2 syntax errors 1,833 -> 2 under grammar
    constraints while structural errors persisted and exact match moved 0.4883 -> 0.5130
    (https://arxiv.org/html/2606.09395). Reinforces `research-gate.js:19-24`.

13. **Bind the record to its subject with a content-derived key.** Idempotency key
    `k = hash(agent_id, action_type, payload, timestamp_bucket)` so "repeated requests with the same key are
    guaranteed to produce at most one effect" (https://arxiv.org/html/2608.02645). The verdict analogue is a
    key over `(step_id, evidence digest, criteria)` — which also detects a verdict written against a
    different tree state.

---

## Consensus vs debate

**Consensus across all sources.**
- One writable sink, computed by the caller, never named by the writer. (Saltzer least privilege;
  `research-gate.js:208-211` in-tree precedent.)
- Existence is not completeness. (SQLite; arXiv:2608.02645 "will pass on partial success";
  `live_check_gate.py:72` in-tree.)
- Fix termination/emission failures at the harness, not the model. (MAST; Constraint Tax; 2606.20724.)
- Recovery means re-run. (Candea & Fox.)

**Genuine debate / contradiction — recorded rather than smoothed.**
- **Is the pyfinagent drop an instance of "constraint tax"?** arXiv:2606.25605 measures 100% suppression on
  open-weight models and **0% on the closed frontier model**, deterministic across 5 rounds, with no
  context-length analysis. pyfinagent's symptom is intermittent, on a closed frontier model, at ~175K tokens.
  **The paper does not support attributing the pyfinagent drop to schema-vs-tools interference**, and a
  design premised on that attribution would be aimed at the wrong target. What survives is the architectural
  lesson (decouple channels), not the causal one.
- **Does any source quantify a token threshold for emission failure?** No. HORIZON explicitly leaves the
  mechanism "unanalyzed" and does not characterise intermittent vs deterministic failure
  (https://arxiv.org/html/2604.11978v1). 2606.20724 finds termination *independent* of budget. So the honest
  posture is: an intermittent fault of unmeasured rate; design for survivability, not for avoidance.
- **Fail-open vs fail-safe.** Saltzer says base decisions on permission (deny by default). Every gate in this
  repo (`qa-write-guard.sh`, `live_check_gate.py`, `verdict_gate.py:42-44`) is deliberately fail-OPEN so a
  helper error never breaks the masterplan Write. These are in direct tension. The resolution used elsewhere
  in-tree is `research-gate.js:450-454`: the *hook* may fail open, but the *decision function* fails CLOSED
  ("failing closed rather than trusting the self-report"). Any 86.31 design should place fail-closed on the
  consumer of the verdict, not on the hook.

---

## Pitfalls (from the literature and from in-tree precedent)

1. **Existence-only completion checks.** `live_check_gate.py:72`; arXiv:2608.02645. A marker must be a
   property of the CONTENT, not of the path.
2. **Structural checks named as if semantic.** `research-gate.js:305-317` already legislates this: a field
   that says "a section exists" must never be read as "the scan was substantive". A `verdict_complete` flag
   is structural only and must be named and documented so.
3. **Self-attestation.** The producer must not certify its own artifact — `research-gate.js:304-305`, and
   Candea & Fox on in-component progress counters being "less trustworthy". If a completion marker is
   written by the Q/A, an independent reader must corroborate it (the stage-2 pattern,
   `research-gate.js:616-652`).
4. **Fail-open defaults that lean permissive.** `verdict_gate.py:76` `ok = data.get("ok", True)`.
5. **Salvaging a partial into an authoritative result.** Crash-only forbids it in spirit; `qa.md:88-90`
   forbids it in letter ("NO auto-PASS on an errored/empty return").
6. **A `const: true` / unrepresentable-failure schema.** `research-gate.js:25-27` — "it makes an honest
   `false` unrepresentable, so the agent must either lie or fail to return." Do not add a
   `wrote_verdict_file: true` field that cannot be false.
7. **Assuming hook ordering.** Auto-memory `reference_claude_code_hooks_run_in_parallel`: hooks under one
   matcher run in parallel. A second `Write|Edit` PreToolUse entry cannot be sequenced against the existing
   one.
8. **Widening a matcher without widening mediation.** The hook covers `Write|Edit` only; `Bash` writes are
   out of scope by the hook's own disclosure (`:18-20`).
9. **Verdict-shopping by accident.** If a partial artifact IS readable as a verdict, a post-drop re-spawn
   stops being a recovery and becomes a second opinion on unchanged evidence (`qa.md:506-511`).

---

## Application to pyfinagent (external findings -> file:line anchors)

**A. The grant.** Do not name the sink `handoff/current/evaluator_critique*.json` — that is
`verdict_gate.py:106-117`'s highest-priority input and `auto-commit-and-push.sh:202-208` acts on it. A sink
under a namespace no automated gate reads (e.g. a Q/A-owned subdirectory, or a `.partial`-suffixed name that
`resolve_verdict_source` provably does not match) keeps the evaluator out of its own authorisation path.
Whatever is chosen, the check `resolve_verdict_source` performs must be re-derived and asserted, not assumed.

**B. The path must be computed by the launcher, not the agent.** Copy `research-gate.js:208-211` exactly:
the script derives the path from the caller's `step_id` and states it in the prompt, so the artifact the
agent writes and the artifact the gate later verifies cannot diverge. This closes failure mode (d)
structurally.

**C. Harden the allowlist predicate if it is extended** (`qa-write-guard.sh:67-69`): prefix-match against a
`realpath`-resolved `CLAUDE_PROJECT_DIR`-anchored root rather than substring-match a lexically-normalised
path (CWE-59 / CWE-367). Keep the hook fail-open (session safety) but do not let fail-open leak into the
verdict decision.

**D. Inertness by construction, per SQLite.** The strongest available shape given the constraints: the Q/A
writes only a NON-AUTHORITATIVE artifact, and an independent party performs the one small act that makes a
verdict authoritative. That mirrors the two-stage design already shipping at
`research-gate.js:594-666` (stage 1 produces, stage 2 independently verifies, `enforceGate` decides) and it
means no completion marker written by the evaluator is ever load-bearing on its own.

**E. Reader-side fail-closed.** Per `research-gate.js:450-454`, the consumer must treat an
unverifiable/incomplete artifact as a FAILED read, never as an absent one. Combined with `qa.md:88-90`, the
rule that must survive unchanged is: **an errored/empty/partial return is NO VERDICT, never PASS.** A partial
artifact can only ever (i) prove a Q/A ran, (ii) carry diagnostics forward into the next spawn's prompt, and
(iii) trigger a fresh Q/A. It may never move a step toward `done`.

**F. Do not spend the design on shrinking the prompt.** Two independent 2026 sources say budget scaling is
the wrong lever (https://arxiv.org/html/2606.20724 explicitly; https://arxiv.org/html/2604.11978v1 by
absence of any measured correlation), matching the project's own falsified experiment.

**G. Housekeeping.** `archive-handoff.sh:146` does not archive per-step `.json` artifacts; extend the list if
the sink is a `.json`, or the file lingers in `handoff/current/`.
`verify_handoff_layout.py:42-49` already permits `evaluator_critique_*.json` — which is why (A) is a real
collision risk rather than a theoretical one.

---

## Read in full — complete list (12; counts toward the gate)

| # | URL | Kind |
|---|---|---|
| 1 | https://www.sqlite.org/atomiccommit.html | Official docs |
| 2 | https://arxiv.org/html/2503.13657v2 | Preprint (MAST) |
| 3 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | Peer-reviewed (Proc. IEEE 1975) |
| 4 | https://cwe.mitre.org/data/definitions/367.html | Official (MITRE) |
| 5 | https://lwn.net/Articles/457667/ | Authoritative technical |
| 6 | https://dslab.epfl.ch/pubs/crashonly.pdf | Peer-reviewed (HotOS-IX 2003), pdfplumber-extracted |
| 7 | https://arxiv.org/html/2606.25605v1 | Preprint (Constraint Tax) |
| 8 | https://arxiv.org/html/2606.20724 | Preprint (web-agent termination) |
| 9 | https://arxiv.org/html/2604.11978v1 | Preprint (HORIZON) |
| 10 | https://arxiv.org/html/2608.02645 | Preprint (verified tool calls) |
| 11 | https://arxiv.org/html/2606.09395 | Preprint (structured-output control) |
| 12 | https://cwe.mitre.org/data/definitions/59.html | Official (MITRE) |

## Identified but snippet-only or failed (context; does NOT count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://cap-lore.com/CapTheory/ConfusedDeputy.html | Peer-reviewed (Hardy 1988) | WebFetch failed: "unable to verify the first certificate". Content covered via CWE-59/CWE-367 + Wikipedia snippet |
| https://css.csail.mit.edu/6.858/2014/readings/confused-deputy.pdf | Peer-reviewed mirror | HTTP 404 |
| https://web.mit.edu/Saltzer/www/publications/protection/ | Peer-reviewed (TOC page) | Returned table of contents only; the section page (#3 above) was read instead |
| https://en.wikipedia.org/wiki/Confused_deputy_problem | Community | Encyclopaedia; superseded by CWE + Saltzer |
| https://blog.acolyer.org/2016/02/16/capability-myths-demolished/ | Authoritative blog | Summary of Miller/Yee/Shapiro; primary sources preferred |
| https://www.beyondtrust.com/blog/entry/confused-deputy-problem | Vendor | Vendor tier |
| https://capisc.io/blog/confused-deputy-problem-coming-for-multi-agent-systems | Vendor | Vendor tier; no measurement |
| https://en.wikipedia.org/wiki/Crash-only_software | Community | Primary paper read instead |
| https://www.usenix.org/conference/hotos-ix/crash-only-software | Official | Abstract/landing page only |
| http://roc.cs.berkeley.edu/retreats/winter_03/slides/candea_crashonly.pdf | Slides | Slides, not the paper |
| https://research.cs.wisc.edu/areas/os/ReadingGroup/os-old/Papers/HotOSIX/Candea-CrashOnlySoftware.pdf | Mirror | Duplicate of #6 |
| https://www.semanticscholar.org/paper/Crash-Only-Software-Candea-Fox/118391e04c7552c637b84d22f08c6369bd3cd483 | Index | Metadata only |
| https://arxiv.org/abs/2606.22936 | Preprint | "When Agents Commit Too Soon" — premature *commitment* (epistemic), not emission failure; adjacent |
| https://arxiv.org/pdf/2605.25310 | Preprint | Tool-call dependency decoding; mechanistic, off-question |
| https://arxiv.org/pdf/2606.12882 | Preprint | HarnessBridge; harness controller, learned (model-side) |
| https://arxiv.org/pdf/2603.29848 | Preprint | AgentFixer; failure detection -> fix recommendation |
| https://arxiv.org/pdf/2510.14453 | Preprint | Natural Language Tools; alternative to schema tool calling |
| https://arxiv.org/pdf/2606.23003 | Preprint | VCT verifiable transcript for LLM conversations; adjacent to tamper-evidence |
| https://arxiv.org/pdf/2604.25200 | Preprint | Auditable AI-assisted grant evaluation |
| https://arxiv.org/pdf/2605.00065 | Preprint | Merkle-tree log integrity for IoT edge |
| https://www.usenix.org/legacy/event/sec09/tech/slides/crosby.pdf | Slides | Crosby & Wallach tamper-evident logging — slides only |
| https://arxiv.org/pdf/1808.06641 | Preprint | PDFS data feed; off-topic |
| https://arxiv.org/pdf/1807.00515 | Preprint | Automatic software repair bibliography; off-topic |
| https://arxiv.org/pdf/1212.1651 | Preprint | Search noise; off-topic |
| https://hoop.dev/blog/immutable-audit-logs-and-granular-database-roles-a-guide-to-enhanced-security-and-compliance | Vendor | Vendor tier |
| https://www.cyberhaven.com/infosec-essentials/what-is-audit-log | Vendor | Vendor tier |
| https://devsecopsschool.com/blog/worm-storage/ | Community | Low tier |
| https://mattermost.com/blog/compliance-by-design-18-tips-to-implement-tamper-proof-audit-logs/ | Vendor | Vendor tier |
| https://www.aptible.com/hipaa/audit-log-retention | Vendor | Compliance retention, not mechanism |
| https://www.accountablehq.com/post/hipaa-compliance-for-audit-logs-requirements-and-best-practices | Vendor | Compliance, not mechanism |
| https://censinet.com/perspectives/hipaa-audit-log-requirements-explained | Vendor | Compliance, not mechanism |
| https://blubanyan.com/best-practices-for-managing-audit-trails-in-erp/ | Vendor | Vendor tier |
| https://cefcore.com/blog/audit-trail-best-practices/ | Community | Low tier |
| https://tracehold.ai/blog/immutable-audit-log-hmac-hash-chain/ | Vendor | Vendor tier |
| https://www.designgurus.io/answers/detail/how-do-you-design-tamperevident-audit-logs-merkle-trees-hashing | Community | Low tier |
| https://github.com/AyoubTadlaoui/GoLogX | Code | Implementation, no measurement |
| https://www.emergentmind.com/topics/immutable-audit-log | Index | Aggregator |
| https://dev.to/robertatkinson3570/the-architecture-behind-tamper-proof-audit-logs-56ek | Community | Low tier |
| https://medium.com/swlh/capability-based-security-and-macaroons-aaa64fb9fc01 | Community | Low tier |
| https://medium.com/@sohail_saifii/the-capability-based-security-model-that-makes-privilege-escalation-impossible-8231d679b972 | Community | Low tier |
| https://amlalabs.com/blog/confused-deputy/ | Vendor | Vendor tier |
| https://securityboulevard.com/2026/03/separation-of-duties-for-ai-agent-workflows-explained/ | Industry | 2026; restates least privilege, no mechanism |
| https://www.insiderisk.io/research/agentic-ai-insider-risk-2026 | Industry | 2026; governance framing |
| https://iternal.ai/ai-agent-security-checklist | Vendor | 2026 checklist |
| https://www.kiteworks.com/cybersecurity-risk-management/ai-agent-security-incidents-2026/ | Vendor | Incident stats, not mechanism |
| https://kiteworks.substack.com/p/ai-agent-liability-section-4-executive-order | Industry | Legal, off-question |
| https://medium.com/@Indext_Data_Lab/ai-agent-audit-the-complete-2026-governance-and-compliance-guide-aa945b2d2f67 | Community | Low tier |
| https://futureagi.com/blog/ai-agent-failure-modes-2026/ | Vendor | 2026 taxonomy blog; MAST read instead |
| https://futureagi.com/blog/llm-function-calling-2025/ | Vendor | Vendor tier |
| https://changegamer.ai/resources/reliable-tool-calling | Vendor | Retry-loop advice, unmeasured |
| https://projectsupply.in/blog/structured-output-llm-2026 | Community | Low tier |
| https://grokipedia.com/page/Crash-only_software | Community | Low tier |
| https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea_html/ | Official mirror | Not attempted after pdfplumber succeeded |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **12** (11 via WebFetch; #6 via
      WebFetch-download + pdfplumber per `.claude/rules/research-gate.md` step 3, which the rule counts as a
      full read)
- [x] 10+ unique URLs total (incl. snippet-only) — see counts in the envelope, measured from this file
- [x] Recency scan (last 2 years) performed + reported — dedicated section above, 6 findings, 7 queries listed
- [x] Full papers / pages read (not abstracts) for the read-in-full set — no arxiv.org/pdf/ URL was
      WebFetched; `/html/` used throughout per the rule
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the scope, plus three the scope did not name
      (`verdict_gate.py`, `live_check_gate.py`, `verify_handoff_layout.py`) — the first of which turned out to
      carry the decisive fact
- [x] Contradictions / consensus noted — see "Consensus vs debate"; the Constraint-Tax attribution is
      explicitly REFUTED rather than adopted
- [x] All claims cited per-claim

**Disclosed gaps / honest limits:**
- Hardy's "The Confused Deputy" (1988) could not be fetched in full (TLS cert failure on cap-lore.com; MIT
  mirror 404). Its content is represented through CWE-59/CWE-367 and Saltzer, which are read in full. It is
  recorded as snippet-only and does NOT count toward the gate.
- No source found measures a token-count threshold for structured-output emission failure. The pyfinagent
  measurement appears to be ahead of the published literature; this brief therefore recommends designing for
  an intermittent fault of unknown rate rather than for a threshold.
- The Constraint-Tax paper's phenomenon is deterministic and open-weight-specific; it does NOT explain the
  pyfinagent symptom. Stated as a refutation, not smoothed into support.

## Envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 12,
  "snippet_only_sources": 52,
  "urls_collected": 64,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 4,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 1,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.31.md",
  "gate_passed": true
}
```
