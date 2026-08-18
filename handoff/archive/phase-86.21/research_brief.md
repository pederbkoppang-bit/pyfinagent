# Research Brief -- Step 86.21

**Tier:** moderate (caller-specified). **Audit-class:** NO (`coverage` reported for
information; `coverage.dry` not required). **Written first, grown incrementally.**

## Objective (restated)

Design a governance counter that constrains the party who supplies its own input.
Concrete instance: the 3rd-CONDITIONAL auto-FAIL escalation rule, whose prescribed
data source (`handoff/harness_log.md`) is written only at step CLOSE -- so for
exactly the in-flight steps the rule is meant to govern, the count reads **zero**.

Sub-questions: (1) fail-open vs fail-closed for a counter whose source may be
missing/unparseable; (2) the independence problem when the counted party supplies
the count; (3) robust parsing of durable append-only artifacts under format drift,
and why a silent zero is the worst return; (4) idempotent append-only ledger design
for per-cycle events + where to write it; (5) CI/release-gate precedent for counting
prior attempts of the same unit of work.

---

## PART A -- INTERNAL INVENTORY (all figures measured, not asserted)

### A1. The rule exists in THREE places and they DISAGREE

| Copy | Predicate | Reset semantics |
|---|---|---|
| `CLAUDE.md:358-364` | 3+ **consecutive** CONDITIONAL "without an intervening PASS or FAIL" | "Counter resets on PASS, FAIL, or new step-id" |
| `.claude/agents/qa.md:512-519` | "grep ... for the current step-id. If there are already **2+ `result=CONDITIONAL` entries**" | none stated |
| `docs/runbooks/per-step-protocol.md:238-240, :254-269` | "3 CONDITIONALs without intervening PASS/FAIL" (:239 drops the word *consecutive*; :261 restores it) | implied only |

qa.md is a **cumulative** count; CLAUDE.md is a **consecutive-run** count. The
parenthetical at `qa.md:514` ("i.e. this would be the third consecutive CONDITIONAL")
asserts an equivalence that holds only when no PASS/FAIL intervenes. **The two give
opposite answers on the only real case in the repo** (A3): at cycle 194, step 36.17
had cumulative CONDITIONAL = 2 -> qa.md says *return FAIL*; consecutive-run = 1
(FAILs at 191/192 reset it) -> CLAUDE.md says *CONDITIONAL is allowed*. Any fix must
pick one predicate and make all three files say it.

### A2. Confirmed: the log's write point is step CLOSE

`CLAUDE.md:223` puts the `harness_log.md` append in the **LOG** phase, last of the
five-file protocol; the Critical-Rules bullet requires it "BEFORE the status flip so
it's included in the auto-commit"; runbook `:278` repeats it. Auto-memory
`feedback_log_last` agrees. The artifact the counter reads is produced **after** the
verdict sequence it is meant to count.

### A3. The failure is already in git, with a written confession

Commit **a1b92d14** (2026-08-09 19:43), subject *"docs(36.17): log cycles 190-194 --
the counter could not see this step at all"*. Body, verbatim:

> harness_log.md had ZERO 36.17 rows across five Q/A cycles, so the grep-based
> 3rd-CONDITIONAL counter was blind to the entire step and every Q/A had to be told
> its own verdict history in the spawn prompt.

Five verdicts (190 CONDITIONAL, 191 FAIL, 192 FAIL, 193 CONDITIONAL, 194 CONDITIONAL)
were backfilled in ONE commit. During all five the prescribed grep returned **zero**,
and the compensating control was *the orchestrator telling the Q/A its own history* --
precisely the independence defect (B2). Current step: `grep -c "86.21"
handoff/harness_log.md` -> **0**. The counter is blind to 86.21 right now.

### A4. Parse fragility of the prescribed source -- MEASURED

`handoff/harness_log.md` = 32,308 lines. Re-runnable greps:

| Measurement | Value | Consequence |
|---|---|---|
| `^## Cycle` headers | 1189 | the assumed format |
| `^### Cycle` headers | 16 | heading-depth drift (addenda, roll-ups) |
| Cycle headers with **no** `phase=` token | **589 / 1205 = 48.9%** | step-id unextractable for half the log |
| ...same, last 200 headers only | 15 / 200 = 7.5% | drift is mostly historical but NOT gone |
| Lines matching `result=CONDITIONAL` | 26 | the counter's target population |
| ...on a `^#+ Cycle` header | 20 | strict-regex recall = **20/26 = 77%** |
| ...on a differently-shaped `^## phase-…` header | 6 | real records missed, e.g. `## phase-10.5.7 -- 2026-04-24 -- result=CONDITIONAL` |
| ...of those 6, carrying `phase=` | 1 | 5 carry the step-id only in prose heading text |
| `grep -c "36\.17"` (bare step-id, as qa.md prescribes) | 12 vs 6 real rows | **precision 0.50** |

The prescribed instruction is simultaneously **lossy** (misses 23% of CONDITIONALs)
and **noisy** (over-counts 2:1 on bare step-id).

### A5. Verdict-vocabulary drift -- the 86.20 failure class, again

`grep -oE "result=[A-Za-z_]+"` yields **19 distinct values**: `PASS` 743,
`CONDITIONAL` 26, `BLOCKED` 6, `FAIL` 4, `N` 3, `CERTIFIED_FALLBACK` 2, `Q` 2,
`SUPERSEDED` 2, and singletons `auth`, `CRITERION`, `DELIVERED`, `DESCOPE`,
`HEALTHY`, `OVERNIGHT_BLOCKED_NEEDS_BQ_MIGRATION`, `PASS_AFTER_RETRY`,
`PASS_WITH_FINDINGS`, `PENDING`, `true`.

Two distinct hazards: `PASS_AFTER_RETRY`/`PASS_WITH_FINDINGS` are PASS-class tokens
that `== "PASS"` will not reset on; and `CERTIFIED_FALLBACK`/`BLOCKED`/`SUPERSEDED`/
`PENDING` are none of PASS/FAIL/CONDITIONAL -- the reset rule at `CLAUDE.md:362` has
**no defined behaviour** for them. Same normalisation trap as auto-memory
`project_rec_vocabulary_86_20`.

### A6. The proposed alternative source has its OWN recall problem

`handoff/current/evaluator_critique_<id>.md`. Across `handoff/` (13,483 archived
critique files) the digit-folded name shapes are: `_NN.N.md` (57), `_NN.NN.md` (35),
`_NN.N_cycleN.json` (16), `_NN.N.json` (12), `_NN.N.N.md` (8), `_NN.N_passN.json` (5),
`_NN.N_cycleN.md` (4), `_NN.NN.json` (4), `_NN.NN_cycleN.json` (3), `_NNNN.N.md` (3,
the phase-4000 family), plus singletons `_audit.md`, `_final.md`, `_main.md`,
`_upgrade.md`, `_phaseN.md`, `_phases_N_N.md`, `_NN.N_cycleN_ERRORED.json` -- **>=17
shapes**. Also: `handoff/current/` still holds critiques for *closed* steps (36.7-36.13,
dated 26 Jul), so the directory is not scoped to the in-flight step; and **one file per
cycle is not an invariant** -- 36.17 ran six cycles but leaves a single
`evaluator_critique_36.17.md` (overwritten). Critique files are a *presence* signal
("a verdict existed"), not a reliable per-cycle counter.

### A7. Where a ledger could hook in -- and the hard runtime constraint

`.claude/workflows/qa-verdict.js` (201 lines) states at `:36` that the two workflow
scripts *"cannot share a module because the Workflow runtime forbids imports"*.
Auto-memory `reference_workflow_runtime_constraints` records the rest: **no `fs`, no
Node APIs**, only `export const meta`, return lands in `workflows/<run>.json`. So a
Workflow script **cannot append to a ledger itself**. Writers available: (a) the Q/A
agent -- BLOCKED, it has no `Write` tool (`qa.md:543`); (b) **Main**; (c) a hook.
House checker pattern: `scripts/qa/` (24 entries) -- `.mjs` precedent is
`verify_research_gate_workflow.mjs` / `verify_workflow_args_boundary.mjs`.

### A8. Internal file inventory

| File | Lines / anchor | Role | Status |
|---|---|---|---|
| `CLAUDE.md` | :354-368, :223, :378-379 | rule copy A + write-point | DRIFTED vs qa.md |
| `.claude/agents/qa.md` | :512-519, :543 | rule copy B (the grep); no Write tool | DRIFTED; wrong predicate |
| `docs/runbooks/per-step-protocol.md` | :238-240, :254-269, :278 | rule copy C | :239 drops "consecutive" |
| `handoff/harness_log.md` | 32,308 lines; 1205 cycle headers | prescribed source | close-time only; 48.9% lack `phase=` |
| `.claude/workflows/qa-verdict.js` | 201; :36 no-imports | Q/A launch rail | no fs -> cannot write a ledger |
| `.claude/workflows/research-gate.js` | 504 | sibling rail | precedent: recompute + cross-check disk |
| `handoff/current/evaluator_critique_*.{md,json}` | 30 in `current/`, 13,483 archived | proposed alt source | >=17 name shapes; not per-cycle |
| `scripts/qa/verify_research_gate_workflow.mjs` | -- | re-runnable-checker idiom | reusable |

---

## PART B -- EXTERNAL RESEARCH

### Search queries run (three-variant discipline)

- **Year-less canonical:** `fail-open vs fail-closed security control design when data
  source unavailable`; `self-reported compliance metrics independence problem
  attestation trustworthy evidence`; `GitHub Actions run_attempt counting workflow
  re-run attempts release gate retry limit`
- **Last-2-year / current-year:** `LLM self-evaluation bias self-assessment unreliable
  independent verifier arXiv 2025 2026`

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-09 | official (IETF/IAB RFC 9413) | WebFetch, full | §5.1: *"Choosing to generate fatal errors for unspecified conditions instead of attempting error recovery can ensure that faults receive attention."* Reframes the question: *"Do not ask 'how do I tolerate this error?' but rather 'how do we fix the underlying problem?'"* §4.1 names the *"pathological feedback cycle"* where tolerated errors *"become entrenched."* |
| https://slsa.dev/spec/v1.0/threats | 2026-08-09 | official spec (OpenSSF SLSA v1.0) | WebFetch, full | Build L2+: *"the trusted control plane generates all information that goes in the provenance"*; *"the worker reports the output artifacts but otherwise has no influence over the provenance."* Missing-provenance mitigation: *"Verifier requires provenance before accepting the package."* |
| https://kubernetes.io/docs/concepts/workloads/controllers/job/ | 2026-08-09 | official docs (Kubernetes) | WebFetch, full | `.status.failed` / `.status.succeeded` are maintained by the **Job controller, not the Pod**; only `RestartPolicy: Never|OnFailure` allowed *"to ensure the Job controller, not the kubelet, controls retry behavior"*; counters live on the Job object so they survive Pod restarts; `podFailurePolicy` decides whether a failure **counts** toward the limit. |
| https://martinfowler.com/eaaDev/EventSourcing.html | 2026-08-09 | authoritative blog (Fowler) | WebFetch, full | *"every change to the state of an application is captured in an event object, and that these event objects are themselves stored in the sequence they were applied"*; current state is *"purely derivable from the event log"*; replay powers *"dissolve if events are reconstructed post-hoc rather than preserved contemporaneously."* Gateways must be *"disabled during the replay processing."* |
| https://arxiv.org/html/2601.22548 | 2026-08-09 | peer-review-track preprint (arXiv:2601.22548v4, 2026-06-22) | WebFetch, arXiv HTML, full | **Qualifies** the self-bias narrative: *"evaluator uncertainty accounts for an average of 89.6% of measured self-preference"*; only *"51% of examples in previous findings retain statistical significance."* Residual real on subjective tasks. *"this paper does not dispute the existence of self-preference, but it does advise on where (not) to look."* |
| https://arxiv.org/html/2504.03846v2 | 2026-08-09 | preprint (arXiv:2504.03846v2, 2025) | WebFetch, arXiv HTML, full | Self-preference largely competence-legitimate (accuracy-vs-SPR r = 0.801/0.817/0.771) BUT *"stronger models ... show a greater tendency towards this harmful bias when they are incorrect"* -- overconfidence / **flawed introspection**, i.e. exactly the case where a self-supplied count would be wrong. |
| https://authzed.com/blog/fail-open | 2026-08-09 | engineering blog (AuthZed) | WebFetch, full | *"Ensuring that authorization systems default to a fail-closed state during failures is critical in preventing unauthorized access."* Notably **silent** on the "no data vs denied" distinction and on degraded modes -- a gap this step must fill itself. |

### Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://read.thecoder.cafe/p/fail-open-fail-closed | blog | community-tier; superseded by AuthZed |
| https://trainingcamp.com/glossary/fail-open/ | glossary | stub, no design guidance |
| https://trainingcamp.com/glossary/fail-close/ | glossary | stub |
| https://redeagle.tech/eaglepedia/fail-open-vs-fail-closed | wiki | community-tier |
| https://community.cisco.com/t5/security-knowledge-base/fail-open-amp-fail-close-explanation/ta-p/5012930 | community | community-tier |
| https://www.keysight.com/blogs/en/tech/nwvs/2020/05/20/fail-closed-fail-open-fail-safe-and-failover-abcs-of-network-visibility | vendor blog | network-appliance framing |
| https://www.zengrc.com/blog/role-of-self-attestation-in-compliance-benefits-challenges/ | industry blog | SLSA/NIST cover it authoritatively |
| https://www.mgocpa.com/perspective/soc-2-compliance-automation-software/ | industry | marketing framing |
| https://www.compliance.com/resources/independent-compliance-program-assessments-in-response-to-the-updated-guidance-by-oig-and-doj/ | industry | US-healthcare-regulatory specific |
| https://www.fluxforce.ai/controls/independent-testing | industry | vendor control catalogue |
| https://web.cs.wpi.edu/~guttman/pubs/icics_attestation.pdf | paper (PDF) | binary PDF; SLSA covers the applied question |
| https://arxiv.org/abs/2509.26600 | preprint | self-bias in LLM-generated benchmarks; adjacent, budget |
| https://arxiv.org/pdf/2508.06709 | preprint | statistical self-bias measurement; PDF |
| https://arxiv.org/pdf/2604.22891 | preprint | quantifying/mitigating judge self-preference; PDF |
| https://arxiv.org/abs/2508.21164 | preprint | label-induced bias in self/cross-eval |
| https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs | official docs | `run_attempt` semantics captured from snippet; K8s Job read in full instead |
| https://github.com/marketplace/actions/job-run-attempt | code/marketplace | community action |
| https://github.com/int128/rerun-workflows-action | code | community action |
| https://blog.logto.io/automatic-github-workflow-rerun | blog | community-tier |

**URLs collected: 26** (7 read in full + 19 snippet-only).

### Recency scan (2024-2026) -- MANDATORY, performed

Searched the 2025-2026 window for LLM self-evaluation / self-report reliability.
**Result: 2 new findings that QUALIFY but do not overturn the canonical independence
argument, and one of them sharpens it.**

1. arXiv:2601.22548v4 (2026-06-22) shows ~89.6% of previously-reported LLM
   self-preference is *evaluator uncertainty*, not narcissism. Taken alone this
   *weakens* a naive "the model will lie about its own count" framing.
2. arXiv:2504.03846v2 (2025) supplies the sharpening: self-preference is mostly
   legitimate competence, **but "stronger models ... show a greater tendency towards
   this harmful bias when they are incorrect"** -- failure of *introspection*, not
   malice. That is exactly the regime a 3rd-CONDITIONAL counter operates in.

**Net:** the case for an externally-supplied count does **not** rest on assuming a
dishonest agent. It rests on (a) the agent being *stateless across cycles* -- a fresh
Q/A has no memory of cycles 1-2 at all -- and (b) demonstrated poor self-error-
recognition. RFC 9413 (2023) and SLSA v1.0 (2023) remain the canonical sources; no
2024-2026 work supersedes them on the parsing or provenance questions.

---

## Key findings

1. **A missing-source counter must fail LOUD, not open and not silently closed.**
   RFC 9413 §5.1: *"Choosing to generate fatal errors for unspecified conditions
   instead of attempting error recovery can ensure that faults receive attention."*
   (https://www.rfc-editor.org/rfc/rfc9413.html, accessed 2026-08-09). Applied here:
   `count == 0` must be **three-valued** -- `0 (verified)`, `unknown (source absent)`,
   `unknown (source unparseable)` -- never collapsed to a single integer.

2. **A silent zero is worse than either failure mode** because it is
   *indistinguishable from the good state*. It fails open (no escalation) while
   *reporting compliance*. RFC 9413 §4.1's *"pathological feedback cycle"* is exactly
   what a1b92d14 records: the tolerated zero became the norm for five cycles.

3. **Non-falsifiability comes from WHO WRITES, not from honesty.** SLSA Build L2+:
   *"the trusted control plane generates all information that goes in the provenance
   ... the worker reports the output artifacts but otherwise has no influence"*
   (https://slsa.dev/spec/v1.0/threats). The Q/A may *consume* the count; it must not
   *produce* it. Corollary: the current compensating control -- Main pasting the
   history into the spawn prompt (a1b92d14) -- is Build-L1-grade at best, because the
   value is unattested and re-typed each cycle.

4. **The canonical shape of an attempt counter is: platform-maintained, attached to
   the unit of work, not to the attempt.** Kubernetes keeps `.status.failed` on the
   *Job*, and forbids `RestartPolicy: Always` specifically so *"the Job controller,
   not the kubelet, controls retry behavior"*
   (https://kubernetes.io/docs/concepts/workloads/controllers/job/). GitHub Actions
   does the same with `github.run_attempt`, injected by the platform and starting at
   1. `podFailurePolicy` supplies the missing piece for A5: **an explicit policy for
   which outcomes count** -- the pyfinagent analogue is deciding, in one place, what
   `CERTIFIED_FALLBACK` / `BLOCKED` / `PASS_WITH_FINDINGS` do to the run.

5. **Contemporaneous append is the whole point of an event log.** Fowler:
   *"every change to the state of an application is captured in an event object ...
   stored in the sequence they were applied"*, and the replay powers *"dissolve if
   events are reconstructed post-hoc"*
   (https://martinfowler.com/eaaDev/EventSourcing.html). `harness_log.md` is a
   **close-time narrative**, not an event log; the fix is to add the event log rather
   than to re-time the narrative.

6. **The independence argument survives the 2026 debunk.** arXiv:2601.22548 removes
   ~90% of the "narcissism" effect, but arXiv:2504.03846v2 keeps the part that
   matters: models are *worst at self-assessment precisely when they are wrong*. And
   the dominant defect here is not bias at all -- it is **statelessness**: a fresh Q/A
   per cycle cannot remember prior cycles, so the count must come from outside it.

## Consensus vs debate (external)

- **Consensus:** attestations/counters about a party should be produced by a trusted
  plane, not the party (SLSA); attempt counters belong on the unit of work and are
  platform-maintained (K8s, GH Actions); tolerating malformed input entrenches drift
  (RFC 9413); event logs must be written contemporaneously (Fowler).
- **Debate:** fail-open vs fail-closed has **no** universal answer -- AuthZed
  explicitly leaves it to the implementer (*"decide for yourself where to risk
  writing something fail-open or fail-closed"*), and safety-critical egress paths
  invert the security default. RFC 9413 supplies the tie-breaker for *this* case: the
  fault must receive attention, which neither a silent allow nor a silent block does.
- **Debate:** magnitude of LLM self-bias (2601.22548 vs 2504.03846 / 2509.26600).
  Resolved above -- our design must not depend on the contested magnitude.

## Pitfalls (from literature + measured internally)

1. **Silent zero** -- fails open while reporting success (RFC 9413 §4.1; measured:
   a1b92d14, five cycles).
2. **Tolerant parsing entrenches drift** -- a regex loosened to catch the 16 `###`
   headers legitimises further drift (RFC 9413 §4.2's "two stable states").
3. **Self-supplied evidence** -- SLSA "forge/tamper with provenance"; the prompt-
   pasted history is the current instance.
4. **Post-hoc reconstruction** -- Fowler; backfilling 190-194 restored the *narrative*
   but could never restore the *decision*.
5. **Uncounted outcome classes** -- 19 `result=` tokens vs a 3-token rule; K8s solves
   this with an explicit `podFailurePolicy`.
6. **Replay/duplicate writes** -- Fowler's gateway warning maps onto re-running a
   cycle: the ledger append must be idempotent on `(step_id, cycle_no)` or a re-run
   inflates the count and manufactures a FAIL.
7. **Two predicates, one name** -- A1: cumulative vs consecutive. Fixing the source
   without fixing the predicate leaves the escalation ambiguous.

## Application to pyfinagent (external finding -> internal anchor)

| Finding | Anchor | Implication for the contract |
|---|---|---|
| Fail loud, three-valued (RFC 9413 §5.1) | `qa.md:512-519` | The Q/A must never treat "no ledger row" as "0 CONDITIONALs". `unknown` should CAP the verdict (the `Missing_Assumption` idiom already at `qa.md:263`), not silently allow a 3rd CONDITIONAL. |
| Trusted plane writes, subject reads (SLSA L2+) | `qa.md:543` (Q/A has no Write) + `qa-verdict.js:36` (no fs) | The only viable writer is **Main or a hook**. That is *architecturally* the right answer, not a workaround: the counted party structurally cannot write its own count. |
| Counter on the unit of work (K8s `.status.failed`) | new ledger keyed by `step_id` | Key the ledger on `step_id`, append one row per cycle, and keep it independent of `harness_log.md`'s close-time narrative. |
| Explicit "does this count?" policy (`podFailurePolicy`) | `CLAUDE.md:362` reset rule vs A5's 19 tokens | The contract must enumerate the outcome vocabulary and state, for each, whether it increments, resets, or is inert. |
| Contemporaneous append (Fowler) | `CLAUDE.md:223` LOG phase | Do **not** move the `harness_log.md` append earlier -- that breaks the auto-commit ordering (`feedback_log_last`, `feedback_masterplan_status_flip_order`). Add a *separate* per-cycle ledger; `harness_log.md` stays the close-time narrative. |
| Idempotent replay (Fowler gateways) | -- | Append-only JSONL keyed `(step_id, cycle_no)`; a repeated append with the same key must be a no-op or detectably duplicate. |
| Append-only JSONL house convention | `.claude/rules/research-gate.md` "Handoff folder convention": `handoff/audit/` is *"Append-only JSONL audit streams"*, and `verify_handoff_layout.py` **forbids `*_audit.json*` at `handoff/` root** | A ledger at `handoff/audit/qa_verdict_ledger.jsonl` fits the existing partition and the layout verifier; it is NOT archived away by `archive-handoff.sh` on step close, unlike anything in `handoff/current/`. |
| Recompute, never trust the self-report | `research-gate.js` (504 lines) precedent | Same discipline: a checker that reads the ledger from disk and recomputes the escalation, rather than trusting a number in the verdict envelope. |

**Open design question for Main (not for me to decide):** whether the ledger row is
appended by Main *after* transcribing the verdict (simple, but Main is the party with
an incentive to keep steps alive) or by a `PostToolUse` hook on the
`evaluator_critique*` write (more independent, but hooks run in parallel under one
matcher -- see auto-memory `reference_claude_code_hooks_run_in_parallel` -- and
`PostToolUse` cannot block). SLSA's hierarchy favours the hook; the repo's measured
hook hazards favour Main plus a re-runnable `scripts/qa/` checker that can detect a
missing row after the fact.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total -- **26**
- [x] Recency scan (last 2 years) performed + reported -- 2 findings, both evaluated
- [x] Full pages read (not abstracts) for the read-in-full set (arXiv via
      `arxiv.org/html/`, never `/pdf/`)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (CLAUDE.md,
      qa.md, harness_log.md, evaluator_critique files, qa-verdict.js, scripts/qa) --
      **plus** an unrequested third rule copy found in
      `docs/runbooks/per-step-protocol.md`
- [x] Contradictions noted -- A1 (three-way rule drift); fail-open/closed has no
      universal answer; the 2026 self-bias debunk is recorded as a qualifier
- [x] Claims cited per-claim with URL + access date, or file:line
- [ ] **Gap:** I did not read `.claude/hooks/archive-handoff.sh` or
      `scripts/housekeeping/verify_handoff_layout.py` in full; the claim that a
      `handoff/audit/` ledger survives step close is inferred from the documented
      folder convention in `.claude/rules/research-gate.md`, not from the hook source.
      Main should verify before committing to that path.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 19,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.21.md",
  "gate_passed": true
}
```
