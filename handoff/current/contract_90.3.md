# Contract -- step 90.3

**Step:** 90.3 -- progress-gated retry, corrected: the digest must exempt rail drops and
must never be read as proof of convergence. **Priority:** P1.
**Contract written:** 2026-08-21. **Phase:** phase-90.

---

## 1. Research gate -- PASSED (enforced)

`Workflow({scriptPath: '.claude/workflows/research-gate.js'})`, run `wf_8f0a6091-2d0`,
2 agents, 207,296 tokens, 718s. Brief: `handoff/current/research_brief_90.3.md`
(38,271 chars). Enforced return: `gate_passed: true`, `self_report_disagreed: false`,
`violations: []`; 10 sources read in full, 34 URLs, recency scan performed, all 10 claimed
URLs present in the brief.

### The finding that reshapes this step, and it was found BEFORE any code existed

**THE DIGEST WOULD HAVE BEEN VACUOUS BY CONSTRUCTION — 89.1's defect through a different
door.** Criterion 1 derives the file set from *"the step's declared masterplan paths union
`git diff --name-only HEAD` union `git ls-files --others --exclude-standard`"*. Measured
today, that union resolves to exactly:

- `handoff/audit/attempt_budget_audit.jsonl` — **tracked, and written BY THE GATE ITSELF on
  every launch**
- `handoff/audit/pre_tool_use_audit.jsonl` — written on **every tool call**

**I verified this myself rather than taking it from the brief.** `git diff --name-only HEAD`
returns exactly those two files — and, right now, a third: `.claude/agent-memory/researcher/MEMORY.md`,
**written by the researcher during the very gate that found the hazard.** A third self-reference,
in the same union, discovered by running the command.

So unless the allowlist excludes `handoff/audit/`, **the digest advances on every launch
whether or not anything about the step changed.** That is precisely the failure this step
exists to correct 89.1 for: a signal that measures the instrument rather than the work.

**And "declared masterplan paths" has no source.** Measured by me across the whole plan:
**0 of 1222 steps carry a `files` or `paths` key.** One third of criterion 1's stated input
does not exist anywhere in the corpus it names.

### What the literature says, and it inverts the intuition

| Finding | Source | Consequence here |
|---|---|---|
| SHA-256 **duplicate_code** fires at 0–50.8% and **code_cycle** at 0.7–3.8%, while *semantic* **no_progress** fires at **44.6–84.6%** | CUDABeaver (arXiv 2607.00038) | A byte-digest is the **weakest** of the three published stagnation signals. It is worth building only because it is cheap and deterministic — not because it is the right instrument. This must be stated, not implied. |
| Optimal stopping triggers on an **absolute score**, explicitly **not** inter-iteration change | arXiv 2608.10729 | Direct support for criterion 5: a changed digest must never be read as progress. |
| A permanent failure is one that **"requires a change to your input"** | Temporal retry policies | A byte-identical relaunch **after a drop** is textbook-correct, because a drop produces nothing to change. Criterion 2 is not a carve-out; it is the standard. |
| Admission-control webhooks: fail-open vs fail-closed is a per-check decision | Kubernetes admission controllers | Criterion 4's split — a crash exits 0, an incomplete-inputs DENY does not — matches published practice. |

### Corrections to the step's own audit_basis, stated not smoothed

- **Denominators have drifted, and I re-derived them rather than adopting the brief's.**
  The basis says the verdict ledger holds 138 rows and the attempt ledger 92. Measured by
  me today: verdict ledger **146**; attempt ledger **118 rows = 114 `attempt` + 4
  `operator_extension`**. The brief reported "118 attempts"; the precise split is 114
  attempts within 118 rows, and the distinction matters because extension rows are not
  attempts. The *ratios* the basis argues from are unaffected; the quoted counts are stale.
- **90.1's backfill has ALREADY RUN.** `outcome` and `total_tokens` are live on the ledger
  (`attempt_gate.py:87-89`, `:243`) even though 90.1 is `pending`. **So 90.3's stated
  prerequisite is satisfied on disk**, and this step is not blocked by 90.1's flip.
- **Criterion 5's grep is unusable as written.** It returns **1086 lines across 111 files**,
  overwhelmingly `.bak` copies of the masterplan containing the criterion's own text. A
  grep that matches its own specification cannot go red for the right reason.
- `scripts/qa/mutation_matrix_90_3.py` does not exist — correct, and RED at filing by design.

---

## 2. Hypothesis

89.1's mechanism denies a relaunch whose evidence digest is unchanged. Measured, it **fires
on none of the real loops** (the cycle-2 flow mandates updating the artifacts, so the digest
advances every round) and **denies the doctrine-mandated post-drop retry** (14 of 16
NO_VERDICT rows were followed by a retry; 8 of the 11 affected steps later reached PASS).

A corrected digest is worth building **only** if it is honest about being the weakest of the
three published signals: it can detect an exact repeat and an A→B→A oscillation, and it can
detect nothing else. Its value is that it is deterministic and cheap, and that it **cannot
be read as evidence of progress** — criterion 5 is the load-bearing half of this step, not
criterion 1.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. the digest is computed over file CONTENT only, from a set derived from the step's declared masterplan paths union `git diff --name-only HEAD` union `git ls-files --others --exclude-standard`, restricted to a checked-in root allowlist, with .claude/agent-memory/qa/verdicts/ EXCLUDED and the file set NEVER inferred from critique prose; a cell that os.utime()s every input without changing bytes must still DENY
2. the check is SKIPPED when the step's most recent verdict-ledger row is NO_VERDICT, or when no verdict row postdates the previous attempt row -- proven by a fixture rebuilt from the real 75.11.4 pair (NO_VERDICT 19:39:57Z -> CONDITIONAL 19:58:38Z, zero repo commits between): a byte-identical relaunch after a drop must be ADMITTED
3. comparison is against the SET of all digests recorded for the step id and a row is appended on DENY as well as on allow, so an A->B->A oscillation DENIES on the third launch
4. inputs-incomplete is a DENY carrying its own machine reason, distinct from a gate crash; a crash still exits 0 but must append a row with digest null and an explicit unavailable status so the next launch has no comparable baseline, and a mutant turning the incomplete-inputs DENY into exit 0 must be KILLED
5. NO consumer may read a changed digest as evidence of progress or convergence, or use it to suppress the 3rd-CONDITIONAL rule -- asserted by a grep-based test over .claude/ and scripts/ that goes RED when such a consumer is added
6. red-first for the whole mechanism: control observed GREEN with the check disabled (byte-identical relaunch admitted), RED with it enabled, and scripts/harness/attempt_gate.py sha256-identical before and after the cell run
7. handoff/verdict_ledger.jsonl is sha256-identical before and after a denial -- a denial is not a verdict
8. the immutable command below cannot be satisfied by a comment: it drives the gate and asserts a behavioural outcome, not the presence of a token in the source

**Immutable verification command** (RED at filing — `mutation_matrix_90_3.py` does not exist):

```
python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_3.py --verify
```

---

## 4. Plan

### 4.1 The allowlist must EXCLUDE the instrument (criterion 1)

The root allowlist is the load-bearing part of criterion 1, not an afterthought. It admits
step-evidence roots (`handoff/current/`, `scripts/`, `backend/`, `frontend/`, `.claude/`)
and **excludes `handoff/audit/` and `handoff/logs/` explicitly**, because those are written
by the gate and by every tool call. A cell must prove the exclusion is load-bearing: with
`handoff/audit/` admitted, a byte-identical relaunch is ADMITTED (the digest moved on its
own); with it excluded, the same relaunch is DENIED.

"Declared masterplan paths" resolves to the empty set today, since no step carries such a
key. That is stated in the output rather than silently dropped, and the union is well-defined
without it.

`.claude/agent-memory/qa/verdicts/` is excluded per the criterion — the Q/A's write-first
record advances on every spawn, so including it would be the same vacuity by a third door.

### 4.2 Drop-exemption from the ledger, not from a heuristic (criterion 2)

Skip when the most recent verdict-ledger row for the step is `NO_VERDICT`, or when no verdict
row postdates the previous attempt row. Fixture rebuilt from the **real** 75.11.4 pair. This
is not a special case: Temporal's definition of a permanent failure — *"requires a change to
your input"* — makes a post-drop relaunch the standard-correct behaviour.

### 4.3 A SET, not a pairwise comparison (criterion 3)

Compare against every digest recorded for the step id, and append on DENY as well as on
allow, so A→B→A denies on the third launch. A cell drives exactly that sequence.

### 4.4 Three distinct outcomes, not two (criterion 4)

`incomplete inputs` → DENY with its own machine reason. `gate crash` → exit 0 (fail-open,
per the hook contract) **but append a row with `digest: null` and an explicit unavailable
status**, so the next launch has no comparable baseline and cannot be compared against a
phantom. A mutant turning the incomplete-inputs DENY into exit 0 must be KILLED.

### 4.5 Criterion 5 is the load-bearing half, and its grep must be scoped to be meaningful

As written the grep returns 1086 lines over 111 files, mostly `.bak` copies of the
criterion's own text — **a grep that matches its own specification cannot go red for the
right reason.** The test will scope to tracked, non-`.bak` sources under `.claude/` and
`scripts/`, state the scoping rule in its output, and prove it is not vacuous by adding a
synthetic consumer and observing RED. The criterion is satisfied *more* strictly by scoping,
and the gap between its literal wording and the executable form is stated.

### 4.6 Honesty about what this instrument is (criteria 5, 8)

The output must carry the measured comparison: a byte-digest is the **weakest** of the three
published stagnation signals (duplicate_code 0–50.8%, code_cycle 0.7–3.8%, semantic
no_progress 44.6–84.6%). It is built because it is deterministic and cheap. **It detects an
exact repeat and an oscillation, and nothing else.**

---

## 5. Explicitly NOT done here

- **No semantic progress measure.** Out of scope, and the criteria do not ask for one.
- **No flip of 89.1 to `superseded`.** That is an operator decision, queued in 90.3's own
  notes, and not taken by Main.
- **No change to verdict semantics**, and no consumer that reads the digest as progress —
  criterion 5 forbids exactly that.

## 6. References

- `handoff/current/research_brief_90.3.md` (gate PASSED, enforced, `wf_8f0a6091-2d0`).
- CUDABeaver, stagnation signals: https://arxiv.org/html/2607.00038
- Optimal stopping on absolute score: https://arxiv.org/html/2608.10729v1
- Temporal retry policies (permanent = requires a change to your input): https://docs.temporal.io/encyclopedia/retry-policies
- Kubernetes admission controllers (fail-open vs fail-closed): https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/
- Anthropic, harness design for long-running apps: https://www.anthropic.com/engineering/harness-design-long-running-apps
