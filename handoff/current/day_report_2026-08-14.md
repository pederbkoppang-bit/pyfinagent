# Day report — 2026-08-14

**Session goal:** *"fix the harness first so everything else later got proper fixed."*
**Steps flipped to `done`: ZERO.** Both worked steps remain `pending`. **No verdict was
issued by Main at any point.** Version unchanged at **v6.93.222** — correctly, since no
step closed.

---

## 1. The headline: the harness had TWO snapshot defects, and both were measured

Neither was in the plan. Both are the *"committed is NOT in force"* class, and together
they mean **a whole class of harness fix silently does nothing until the session
restarts.**

### A. `Workflow({name: ...})` runs a session-snapshotted SCRIPT

Caught **before** it produced a verdict. The first 86.78 evaluation was launched by name
and was executing a copy of `qa-verdict.js` predating the day's edits. Enumerated across
every named launch this session: **0 of 3 carried either of the day's two fixes**, while
the disk file carried both. The run was **stopped**, and the evaluation relaunched with
`scriptPath`, which does not snapshot.

### B. A `qa.md` DELETION is inert until restart — the decisive one

Found by a Q/A run that **dropped**, preserved only because write-first made it write
findings before dying, then **confirmed independently** by a probe spawn:

| | on disk now | in the live judge's system prompt |
|---|---|---|
| 5 pre-fix consequence markers | **0** | **5 of 5 present** |
| 3 post-fix markers | 1–4 | **0 of 3 present** |

`qa-verdict.js` launches `agentType: 'qa'`, so the judge's **system prompt is `qa.md` as
of session start**. Its STEP 0 also makes the judge *read* `qa.md` at runtime — which is
why the rail was believed live. Both are true, and the distinction is everything:

> **A runtime read can only ADD text. It cannot RETRACT text already in the system
> prompt.** An ADDITION is live immediately; a **DELETION is inert until restart.**

Phase-86.78 is entirely deletion-shaped. **This falsifies a claim written in both
`CLAUDE.md` and `qa-verdict.js:13-16`** — that a qa.md edit "is live immediately on this
path; only the Agent-tool roster snapshots at session start." False as written, and
falsified by that file's own `agentType: 'qa'`.

**The same trap is armed on the researcher** (`research-gate.js:710` →
`agentType: 'researcher'`). No live divergence today, because `researcher.md` was last
edited 08-10, before this session.

---

## 2. The rail drops — measured on both rails

**A drop returns nothing and costs full tokens.** Population rule: every
`*/subagents/workflows/wf_*/agent-*.jsonl`, classed by its first user record; a drop =
no `StructuredOutput` tool-call name anywhere in the transcript. Controls: both known
drops classify as drops, two known-good runs classify as not-drops.

| rail | runs | dropped | rate |
|---|---|---|---|
| qa | 374 | 39 | **10.4%** |
| researcher | 278 | 21 | **7.6%** |

*(A first pass reported 5.3% / 7.2% using a loose substring match that also hit prose.
Corrected, not left standing.)*

**Both of today's final Q/A runs dropped, uncovered — 369K tokens, no verdicts.** The
peer session committed a retry wrapper at 12:15; 86.79 launched at 12:10.

**The researcher rail is already ahead of the Q/A here:** it has carried drop handling
since phase-86.37 (08-10), and write-first leaves a complete brief on disk. The Q/A had
neither until today, which is why a drop there destroyed an entire evaluation.

---

## 3. What actually shipped

### 86.79 — the attempt counter was a GAUGE read as a COUNTER

Three independent defects, each **reproduced**, not argued:

| defect | measured |
|---|---|
| `records_retained` counts the CURRENT spawn | 2 priors → **3** |
| correct ONLY because write-first ran first | same spawn: **2** before its own write, **3** after |
| pruning SATURATES it | 6 true attempts → reports **3**; F1b's ceiling reads `3/5 CONTINUE` instead of `6/5 ESCALATE` |

A **third** off-by-one came from the research gate, not from me: `DEFAULT_KEEP`'s comment
promised 4 retained while the code delivers 3.

Fix, research-led — Prometheus's counter-vs-gauge rule (*"Do not use a counter to expose
a value that can decrease"*) and the Linux kernel's `PERF_RECORD_LOST` shape (*emit the
loss, don't hide it*): the field is split into unit-stated `attempt_number` /
`prior_attempts`; the number is computed **only** when this spawn's record is identified,
so the write-first coupling is unrepresentable rather than documented; the pruner records
what it destroys **before** unlinking, so a crash **over**-counts (escalates early);
every uncomputable path returns `None`, **never 0**. `records_retained` keeps its name
**and value** — no live number shifted.

**55 checks, 11/11 mutation cells killed.** Verdicts: CONDITIONAL, CONDITIONAL, drop.

### 86.78 — the judge was told its verdict's consequence before it graded

`arXiv 2604.15224` **resolves** (no fabrication): content held strictly constant, one
consequence sentence varied, 18,240 judgments → **LENIENT in 58 of 72 cells, p<0.001**,
peak −9.8pp. **Reward framing is as lenient as punishment framing.** `ERRJ = 0.000` — the
bias never surfaces in chain-of-thought, so it can never be audited by reading the
judge's notes.

**Mechanism correction:** a prior step credited self-preference / authority anchoring.
`arXiv 2603.04582` refutes that — *"explicitly stating that the action comes from the
monitor does not by itself induce self-attribution bias."* The mechanism is **stakes**.

Shipped: the rail prompt states no consequence; `enforceEscalation()` computes the
threshold at the post-`agent()` seam, pure, returning `escalation` **alongside** the
verdict and failing closed; both law-of-the-case safeguards implemented caller-side.

**51 checks, 13/13 cells.** Verdicts: STOPPED, CONDITIONAL, drop.

---

## 4. The Q/As found things I did not — every cycle

This is the harness working, and it is worth stating plainly.

| cycle | what it found that I had missed |
|---|---|
| 86.79 c1 | I enumerated **one member of a two-member class** — the same false claim was duplicated in `qa-verdict.js`, which was **never operator-gated** |
| 86.79 c2 | a **second** stale `qa.md` site, right after c1 corrected me for enumerating one member of a class; plus two surviving mutants and five stale numbers in my own artifact |
| 86.78 c1 | *"the counter is DUPLICATED, not MOVED"* — sharper than my own framing; plus three surviving mutants |
| 86.78 c2 | the system-prompt snapshot (§1B) — **the finding of the day**, carried out of a dying run |

**Three times, a mutant survived inside a guard written to close an earlier survivor** —
the remediation was narrower than the class each time.

### The fresh executor earned its keep

On the operator's instruction, a **fresh executor** applied the `qa.md` edits, so Main
did not author the agent file it is graded against. It found three things nobody asked
it to look for:

1. **The consequence reached the judge through a TOOL, not the prose** —
   `verdict_history_86_21.py` printed `auto-FAIL armed : True`. A prose scrub cannot
   close a channel that runs through a tool.
2. **After that was suppressed, the mode still printed `consecutive : N`** — an
   aggregate hands the judge the boundary's unit *and its own position in it*. It found
   this by **re-running the mode rather than trusting its own earlier reading**.
3. **The ADR filename named both units**, and `qa.md` cited the path.

It also caught **its own** first draft quoting the suppressed line inside the warning
forbidding it, and **pushed back on my directive** — my change-C brief told it to name
the threshold's shape, which is itself consequence information. It was right.

Asked for the residual channel count rather than a clean number, it returned **five, not
zero**, and ranked them.

---

## 5. What I could NOT verify

- **Neither step is closable from this session.** 86.78's fix is not in force (§1B);
  86.79's last cycle produced no verdict.
- **Every verdict collected today ran under the stale system prompt**, so all carry the
  leniency exposure. Direction matters: leniency makes a CONDITIONAL *conservative*
  evidence, not suspect. It is **PASSes** that would be in doubt — and none was issued.
- **Hand-deleted WIP records remain undetectable.** The loss ledger accounts only for
  the automated pruner.
- **The saturation defect is still LATENT** — `prune_wip_records` has no production
  caller. This step makes pruning *safe if wired*.
- **`attempt_budget.py` is still unwired** (86.71).
- **arXiv 2604.15224 is single-lab and under review.** No independent replication of the
  stakes result exists.
- **The mutation matrices license one claim each:** those specific mutations were killed.
  Three cells exist only because a Q/A found them surviving — direct evidence that an
  author's own matrix is not a completeness proof.

---

## 6. Owed to the operator

1. **ROTATE THE OAUTH TOKEN** — you said you would do it this session. Re-verify with the
   scan in `INCIDENT_2026-08-14_credential_exposure.md`; 5 files, still on `origin/main`,
   public since 08-08, and the repo has a fork so only rotation revokes.
2. **A session restart**, then `scripts/qa/verify_qa_roster_live.sh` — without it the
   86.78 scrub stays inert and no Q/A verdict on it can be trusted.
3. The `qa.md` review queue is now **6+ edits**, of which today's are **executor**-authored.

## 7. Queued, not lost

- `CLAUDE.md` + `qa-verdict.js:13-16` state a claim that is now measured FALSE (§1B).
- `verify_counter_86_79.py` does not guard criterion 4's members 4b/4c — both could be
  reverted with the gate green.
- A stale C4 assertion label ("un-applied"), true until cycle 4.
- `records_pruned_known` and the `ok`-path unit are unguarded.
- `verify_handoff_layout.py` did not flag `prompt_leak_redteam_audit.jsonl` at the
  `handoff/` root — *"a layout invariant checker may be blind"* is the real finding.
- A stakes-free re-grade of ambiguous **PASSes**, 86.68 among them.
