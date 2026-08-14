# Contract — step 86.79

**Step id:** `86.79` (P1, `harness_required: true`)
**Title (masterplan, verbatim):** *records_retained COUNTS THE CURRENT SPAWN, not
prior spawns, and pruning can make the attempt counter UNDERCOUNT exactly past the
escalation threshold*
**Phase order:** RESEARCH (done, PASSED) → **PLAN (this file)** → GENERATE → EVALUATE → LOG
**Written:** before GENERATE. No code in `scripts/qa/qa_wip.py` has been modified at
the time this file is written; the only prior activity is read-only measurement
(`/private/tmp/.../scratchpad/repro_86_79.py`, which writes exclusively to temp dirs).

---

## 1. Research gate — PASSED

| field | value |
|---|---|
| run | `wf_267244ab-91e` (Workflow rail, `.claude/workflows/research-gate.js`) |
| brief | `handoff/current/research_brief_86.79.md` (25,389 chars, independently re-read by the script) |
| sources read in full | **10** (floor 5) |
| URLs collected | **25** (floor 10) |
| recency scan | performed — 4 queries, 3 findings |
| internal files inspected | 9 |
| `gate_passed` | **true**, recomputed by the script; self-report agreed; `violations: []` |
| audit-class | no (`coverage` informational) |

### What the research changed about my prior understanding

Three things, all of which move the plan:

1. **A THIRD off-by-one I had not found.** `qa_wip.py` documents `DEFAULT_KEEP = 3`
   as *"Current record + this many prior attempts"* (4 retained) while
   `prune_wip_records` does `records[keep:]`, retaining **3 total**. The comment and
   the arithmetic disagree inside the same module. (Brief F-I1, measured on a temp
   sink.)
2. **The defect has a name at the type level.** Prometheus, *Metric types*:
   a counter *"can only increase or be reset to zero on restart"*; **"Do not use a
   counter to expose a value that can decrease."** A count derived from a pruned
   retained set is a **gauge being read as a counter**. (Brief E4.)
3. **The remedy has canonical prior art: emit the loss.** Linux perf ring buffer —
   *"the kernel keeps how many records it lost and generates the `PERF_RECORD_LOST`
   records"*. Retained data and the count of dropped data are **separate records**.
   (Brief E5.) This is the shape the fix adopts, and I would not have chosen it
   unprompted.

Supporting, load-bearing for the *wording* of the fix:

- **E1 — a name is not a unit.** Temporal `MaximumAttempts` is **inclusive**
  (*"Setting the value to 1 means a single execution attempt and no retries"*);
  Step Functions `MaxAttempts` is **exclusive** (counts retries only). Two tier-2
  official docs, same word, off by one. ⇒ **write the unit next to the number.**
- **E6 — temporal coupling's remedy is structural, not documentary** (Seemann):
  misuse *"cannot be caught during compilation—only at runtime"*; you fix it by
  making the invalid state unrepresentable, **not** by writing the required order
  in a comment or a second file. ⇒ a `qa.md` sentence can never be the fix for the
  write-first coupling.
- **E7 — no framework surveyed offers a durable cross-session attempt count**
  (OpenAI Agents SDK: *"The turn count does not persist across separate runs"*;
  Step Functions resets on redrive; Temporal needs a separate Describe call). So
  pyfinagent's cross-session Layer-3 loop must derive its count from a durable
  artifact — which is exactly why the retention window is load-bearing.
- **Disclosed research gaps** (carried forward honestly): the LangGraph
  `GRAPH_RECURSION_LIMIT` page returned a bare redirect and was **not** read; and
  **no source names "counter saturation disables escalation" as a documented defect
  class** — the argument rests on Prometheus's type rule and the kernel's
  lost-record pattern, not on a paper about this exact failure.

---

## 2. Hypothesis

`records_retained` is a **gauge presented as a counter**, and three independent
defects follow from that one type error:

- **D1 (live).** It counts `len(records)` — *including the current spawn's own
  write-first record* — while `.claude/agents/qa.md` calls it *"the count of prior
  Q/A spawns"* **and** *"the attempt number"* in one sentence. Those two halves
  differ by exactly one; the second half is the true one.
- **D2 (live).** The number is correct **only because an ordering rule in a
  different file** (write-first, `qa.md`) guarantees the current record exists when
  the count is taken. `qa_wip.py` cannot observe whether that happened, and the
  failure direction is **OPEN** — a lower attempt number *suppresses* escalation.
- **D3 (latent).** `prune_wip_records(keep=3)` makes the number **saturate at 3**,
  so F1b's 5-attempt escalation becomes unreachable. **Latent, not live:** prune has
  **zero automatic callers** (enumerated, §5 C3). Any change that schedules pruning
  arms it silently.

**Fix thesis:** stop asking one integer to be three different things. Return
separately-named, unit-stated integers; make the count **refuse to be computed**
when it cannot be trusted (fail *closed*, never 0); and make the pruner **record
what it destroyed** so the count stays knowable across a lossy window.

---

## 3. Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

> 1. the off-by-one is REPRODUCED, not argued: drive qa_wip.report() for a step with a known number of records and show records_retained equals prior+1, with the line of qa_wip.py that produces it quoted
> 2. the write-first coupling is demonstrated: show that the attempt number is correct ONLY because the current spawn writes before reporting, by driving a report BEFORE any write-first file exists and showing the number differs
> 3. the pruning saturation is DEMONSTRATED in a scratch repo -- create 6 records, prune to keep=3, and show the derived attempt number reports 3 rather than 6 -- and separately confirm by enumeration whether prune is called automatically anywhere in the live tree, with the search command stated
> 4. whichever fix is chosen, the DOC and the CODE are made to agree, and the step states which one moved and why -- renaming the field, changing the description, or returning a separate prior_count are all acceptable, silently leaving them divergent is not
> 5. the escalation still fires correctly after the fix: drive the 3rd-consecutive-CONDITIONAL boundary and F1b's 5-attempt budget against the corrected number and show both trigger
> 6. verdict semantics are UNCHANGED and demonstrated: nothing in this change can turn a FAIL into a PASS, and a counter that cannot be computed must FAIL CLOSED rather than report 0
> 7. mutation-test the fix: revert it and show the check goes red, with the control observed GREEN first

**Immutable verification command** (run at baseline BEFORE any change, exit 0,
output `qa_wip-parses` — recorded so criterion 7's "control observed GREEN first"
has a pre-change anchor):

```
bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/qa/qa_wip.py\").read())" && echo qa_wip-parses'
```

**Live check artifact:** `handoff/current/live_check_86.79.md`.

---

## 4. The fix — design, with the evidence each part rests on

All changes are confined to **`scripts/qa/qa_wip.py`** plus new checker scripts.
**No `.claude/agents/qa.md` edit is made** — see §6.

### F1 — split the field, and state the unit at the point of use  *(E1, E6, brief (i))*

`report()` keeps `records_retained` **at its current value and name** (no silent
shift of a number the live rail already reads) and adds:

| new key | meaning |
|---|---|
| `attempt_number` | `int \| None` — this spawn's attempt, **INCLUSIVE of itself** (Temporal convention). A first attempt is `1`. |
| `prior_attempts` | `int \| None` — attempts strictly before this one. |
| `attempt_number_status` | `ok` / `no_spawn_identity` / `no_record_for_this_spawn` / `source_missing` |
| `attempt_number_is_lower_bound` | `bool` — true when retention loss means the number can only be a floor |
| `records_retained_unit` | a string carrying the unit **in the payload**, per E1 |

`records_retained_unit` is the E1 remedy applied literally: the reader gets the
unit next to the number, in the authoritative artifact, without having to know
what another file claims.

### F2 — make the write-first coupling unrepresentable, not documented  *(E6, F-I4)*

`attempt_number` is computed **only** when the current spawn is identified:
`spawned_at` was passed **and** a record belonging to it was found. Otherwise it is
`None` with `attempt_number_status = no_record_for_this_spawn`, and the guidance
says so. A reader can no longer silently read `N-1` as `N`, which is the exact
fail-open path D2 describes. `prior_attempts` is still reported in that case,
because it genuinely *is* knowable.

### F3 — the pruner records what it destroyed  *(E5, `PERF_RECORD_LOST`)*

`prune_wip_records` writes a durable, **monotonic** per-step high-water mark
**before** unlinking:

```
.claude/agent-memory/qa/verdicts/.attempt_lost_<sid>.json   {"lost": N, "updated": "<ISO>"}
```

- Dot-prefixed, so it is invisible to `list_wip_records`' `verdict_wip_*` globs and
  to `audit_memory.py`'s non-recursive top-level glob.
- **Never decreases** (`max(old, new)`), per Prometheus's counter rule (E4).
- `report()` adds it in: `prior_attempts = lost + retained_priors_before_current`.
- Written **before** the unlink, so a crash mid-prune over-counts (safe direction)
  rather than under-counts.

### F4 — fix `DEFAULT_KEEP`'s own off-by-one comment  *(F-I1)*

**The DOC moves, not the code.** `records[keep:]` is standard keep-N semantics and
matches the precedents the module already cites (k8s retaining exactly one prior,
journald); changing the arithmetic to match the comment would silently retain 4 and
alter live retention behaviour for no benefit. The comment is rewritten to state
the unit: `keep` is the **TOTAL** retained, **inclusive** of the current record.

### F5 — fail closed  *(criterion 6, brief (iv), copying `verdict_history_86_21.py:98-113`)*

Every path that cannot compute the number returns **`None`**, never `0`, and says
why. `records_retained` stays an honest raw file count (0 files retained really is
0 files); the *derived* numbers are the ones that refuse to guess.

### Explicitly NOT done

- **No `verdict` key, no verdict semantics touched.** `report()` still carries
  `is_verdict: false` and has nothing for a caller to scrape as a verdict. Nothing
  in this change can turn a FAIL into a PASS (criterion 6, asserted in §5 C6).
- **No wiring of `prune_wip_records`.** It stays uncalled. This step makes pruning
  *safe if wired*; wiring it is not in scope.
- **No `attempt_budget.py` wiring** — that is step **86.71**.
- **No `qa.md` edit** — §6.

---

## 5. Plan steps → criteria map

| # | Action | Criterion | Artifact |
|---|---|---|---|
| P0 | Baseline-run the immutable command; record GREEN | 7 (control) | `live_check` §0 |
| P1 | `scripts/qa/verify_counter_86_79.py` — drives `report()` with 2 priors + current, asserts `records_retained == priors + 1`, quotes the producing line **re-derived by grep, not hardcoded** | 1 | `live_check` §1 |
| P2 | Same checker: report **before** the current record is written vs after; assert the numbers differ and that post-fix `attempt_number` is `None` rather than wrong | 2 | `live_check` §2 |
| P3 | Same checker: scratch repo, 6 records → `prune(keep=3)` → assert pre-fix derived number is 3; assert post-fix `attempt_number` is **6** via the loss ledger. Re-run the prune-caller enumeration with the command printed | 3 | `live_check` §3 |
| P4 | Implement F1–F5 in `scripts/qa/qa_wip.py` | 4 | `experiment_results` |
| P5 | Drive the 3rd-consecutive-CONDITIONAL boundary and F1b's 5-attempt budget against the corrected number; assert both fire, **including through a prune** | 5 | `live_check` §5 |
| P6 | Assert `report()` has no `verdict` key, `is_verdict is False`; assert every uncomputable path returns `None` not `0`; replay `attempt_budget` exhaustion and assert it cannot produce PASS | 6 | `live_check` §6 |
| P7 | `scripts/qa/mutation_matrix_86_79.py` — revert each fix limb independently, assert the checker goes RED; control observed GREEN first | 7 | `live_check` §7 |
| P8 | Write `experiment_results_86.79.md` + `live_check_86.79.md`; spawn Q/A | — | — |

**Mutation cells planned** (each must be *discriminating* — the control answer and
the mutant's fail-safe answer must not coincide):

| cell | mutation | must go RED in |
|---|---|---|
| M1 | `attempt_number` computed without identity (drop the `no_record_for_this_spawn` guard) | P2 |
| M2 | `prune_wip_records` stops writing the loss ledger | P3, P5 |
| M3 | loss ledger allowed to decrease (`max()` → plain assignment) | P3 |
| M4 | uncomputable path returns `0` instead of `None` | P6 |
| M5 | `DEFAULT_KEEP` comment restored to the "current + 3 priors" wording | P7 (doc assertion) |

---

## 6. KNOWN BLOCKER — criterion 4 is split, and half of it is operator-gated

Criterion 4 requires the DOC and the CODE to agree. There are **two** doc/code
divergences, and they are not equally reachable:

| # | divergence | reachable here? |
|---|---|---|
| **4a** | `DEFAULT_KEEP`'s comment vs `records[keep:]` — both inside `qa_wip.py` | **YES** — fixed by F4; the doc moved, reason stated |
| **4b** | `.claude/agents/qa.md:622` — *"`records_retained` is the count of prior Q/A spawns on this step — the attempt number"* | **NO** — needs a `qa.md` edit |

**Why 4b is not simply done here.** `.claude/agents/qa.md` already carries **four
Main-authored edits awaiting operator review** under CLAUDE.md's separation-of-duties
rule. A fifth deepens a hold the operator owns, and the operator instruction for this
session is explicit: *"If a fix genuinely needs `qa.md`, stop and ask."* The
masterplan step's own notes say the same: *"prefer changing `qa_wip.py`, or hand it
to a fresh executor."*

**Mitigation shipped instead of the edit** — this is why the divergence is **not
silent**, which is what criterion 4 actually forbids:

1. `records_retained_unit` puts the correct unit **in the payload the Q/A reads**,
   at the point of use (E1's own remedy).
2. `attempt_number` / `prior_attempts` give the Q/A two correctly-named fields, so
   the stale sentence stops being load-bearing.
3. The exact one-line `qa.md` patch is written out for the operator in
   `handoff/current/qa_md_patch_86.79.md` — reviewed, not applied.

**I will ask the operator** which route to take for 4b before EVALUATE, and record
the answer. If no answer is given, the step is expected to be **CONDITIONAL on 4b**,
which is the honest outcome — not a PASS.

---

## 7. Risks

| risk | mitigation |
|---|---|
| Changing `records_retained` shifts a number the live rail reads, in the **lenient** direction | Its value and name are left **untouched**; only additive fields |
| The loss ledger is a new write in a directory the memory tooling reads | Dot-prefixed, one level down, outside both globs; asserted in P3 |
| Hand-deleted records are still undetectable | **Disclosed**, not papered over: `attempt_number_is_lower_bound` exists for it; prune is the only automated deleter and is now accounted |
| A mutation cell survives because control and fail-safe answers coincide | Each cell names the specific assertion it must flip; M2/M3 distinguished deliberately |
| Line numbers cited here go stale mid-cycle | Every anchor in the checkers is **grep-derived at runtime**, never hardcoded |

---

## 8. References

- `handoff/current/research_brief_86.79.md` — the gate brief (E1–E8, F-I1–F-I8)
- Prometheus, *Metric types* — https://prometheus.io/docs/concepts/metric_types/
- Linux kernel, *perf ring buffer* — https://docs.kernel.org/userspace-api/perf_ring_buffer.html
- Temporal, *Retry policies* — https://docs.temporal.io/encyclopedia/retry-policies
- AWS Step Functions, *Error handling* — https://docs.aws.amazon.com/step-functions/latest/dg/concepts-error-handling.html
- Google SRE Book, *Handling overload* — https://sre.google/sre-book/handling-overload/
- Mark Seemann, *Design Smell: Temporal Coupling* — https://blog.ploeh.dk/2011/05/24/DesignSmellTemporalCoupling/
- `CLAUDE.md` — F1 / F1b failure discipline; separation of duties on agent edits
- `scripts/qa/verdict_history_86_21.py` — the fail-closed model copied by F5
