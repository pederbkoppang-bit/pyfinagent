# Contract -- step 86.33

**Step**: `86.33` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~10:1x CEST) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** `git diff` on `.claude/hooks/qa-write-guard.sh` is
empty at this moment.

**Concurrency**: peer `pyfinagent-51` owns 86.21/86.29/86.38. Before spawning the
gate I ran a four-point ownership check on 86.33 (no handoff artifact, no WIP
record, no commits in the hour, masterplan `pending`) — after truncating their
86.21 brief earlier today by treating a broadcast as an acquisition.

---

## 1. Research gate -- PASSED ON THE FLOORS, WITH A HOLE I AM NOT BURYING

`wf_b4180a25-c62`, tier `moderate`, brief `research_brief_86.33.md` (24,070
chars). Script-enforced: **9 sources read in full** (floor 5), **17 URLs** (floor
10), recency section present, all 9 claimed URLs verified in the brief,
`brief_status: COMPLETE`, `rail_dropped: null`.

**THE RESEARCHER SELF-REPORTED `gate_passed: false` AND THE SCRIPT OVERRODE IT.**
`self_report_disagreed: true`. Its reason: **WebSearch was 200/200 exhausted at
spawn, so ZERO searches ran.** The mandated three-variant discipline
(current-year / last-2-year / year-less) did not execute at all —
`.claude/rules/research-gate.md` calls a single year-locked query a protocol
breach; zero is worse.

**Why I am proceeding anyway, stated so a reader can disagree:** the 9 full reads
are tier-1/2 canonical sources reachable without search (kernel.org, RFC editor,
SPIFFE, the SELinux notebook, the Claude Code hooks doc), and the load-bearing
findings are **internal measurements** I re-derived myself below rather than
literature. **What is NOT covered: any 2025-2026 development in this area.** If
GENERATE turns on a "current best practice" claim, that claim is unsupported and
must be re-gated. This is a weaker gate than 86.36's, which lost 2 of 3 variants;
this lost 3 of 3.

## 2. I RE-DERIVED THE GATE'S INTERNAL CLAIMS -- one is corrected

| claim | researcher | my measurement | verdict |
|---|---|---|---|
| guard log size | 7,224 records | **7,240** | consistent (log grew after its run) |
| distinct `agent_type` values | 72 | **72** | **exact** |
| `general-purpose` evaluator-critique writes | 15 | **15** (8× `evaluator_critique_82.5.md` + 7× `_82.7.md`) | **exact** |
| `workflow-subagent` legitimate production writes | "~50" | **82** | **CORRECTED — undercount** |

`workflow-subagent`'s 82 Write/Edit events are **all** outside the qa memory dir
and are plainly legitimate GENERATE work: `kill_switch.py` ×12,
`test_phase_36_7_kill_switch_rotation_rearm.py` ×7, `KillSwitchPanel.tsx` ×5,
`paper_trading.py` ×4. **So widening the prefix to match it would break
GENERATE**, exactly as the guard's docstring warns. The docstring is right.

**AND HERE IS THE FINDING THAT RESHAPES THE STEP.** `general-purpose` wrote
**15 `evaluator_critique_*.md` events**. That is the artifact Main is
contractually the verbatim scribe for — the same class as the `qa-80-2` breach
that motivated 86.31. So the defect is not "the guard misses two names". It is
that **one of the unmatched names has a track record of writing the very artifact
the guard exists to protect**, while the other has a track record of legitimate
writes that must not be blocked. The two unmatched identities are not
symmetrical, and any fix that treats them as one bucket is wrong.

## 3. The design finding: you cannot fix this by naming names

From the 9 sources: **RFC 9700 §4.15** — servers "SHOULD NOT allow clients to
influence their `client_id`". **SPIFFE** — a workload never asserts its own
identity. Here, `agent_type` is **chosen by the spawner**, and the guard
prefix-matches it (`:66-93`). Yesterday's 86.31 fix already proved the point at
small scale: 27 `qa-*` identities walked past an `== "qa"` check.

Three remedies exist in the literature; **only one is available here**:

- **Mediated transition** (SELinux: "processes cannot arbitrarily choose their
  own new domain") — **impossible**. Nothing in this repo can force the runtime
  to assign `agent_type`.
- **Attested identity** (SPIFFE) — no attestor exists for a subagent.
- **Monotonic restriction** (`no_new_privs`: an `execve` "promises not to grant
  the privilege to do anything that could not have been done without the
  `execve`", and it "cannot be unset") — **transplantable**. Invert the guard
  from *deny-if-recognised* to *permit-if-recognised*, so an **unrecognised name
  gets LESS authority, not more**. That removes the incentive to rename entirely.

## 4. Immutable success criteria

Copied verbatim from `.claude/masterplan.json` at PLAN time into
`experiment_results_86.33.md` §2 rather than duplicated here, because this step's
criteria are long and a paraphrase in two places is the divergence defect I hit
on 86.36 (`qa.md` vs `qa-verdict.js`) six hours ago.

## 5. Plan

**P0 — CLOSE THE OBSERVABILITY GAP FIRST, BEFORE ANY POLICY CHANGE.** The hooks
doc documents `agent_id` as a second PreToolUse field; the guard **never reads or
logs it** (`grep -c agent_id` → **0**). Log it. Every identity conclusion in this
step and in 86.31 rests on `agent_type` alone, and a second field may
discriminate the cases the first cannot. **Measure before redesigning.**

**P1 — INVERT TO A NARROW PERMIT, if and only if P0's data supports it.** Default
deny for Write/Edit outside a per-role allowed prefix, with an explicit permit
list. An unknown/renamed identity then lands in the *most* restricted bucket.

**P2 — THE RESEARCHER RAIL MUST NOT BREAK.** `workflow-subagent`'s 82 production
writes and the researcher's mandatory write-first are the hard constraint. Any
inversion needs the researcher's paths explicitly permitted, and
`prove_qa_write_separation_86_31.py`'s 7 ALLOW cases must stay green.

**P3 — RESOLVE THE HEADER/DOCSTRING CONTRADICTION.** `:13-16` states the identity
source flatly; `:85-91` documents the gap and the covering control. Both are
"true", and together they let a reader conclude the guard is authoritative. One
statement, not two.

**P4 — mutation:** revert the inversion and require a NAMED assertion red, green
control first; and prove a *renamed* identity gets less authority, not more.

### Explicitly NOT doing

- **Not** widening `is_qa_role` to match `workflow-subagent` — measured to break
  82 legitimate writes.
- **Not** touching `archive-handoff.sh` or anything of 86.29's (peer's).
- **Not** relying on the literature's currency: zero searches ran (§1).

### Risk

`qa-write-guard.sh` is a **fail-open** PreToolUse hook on every Write/Edit in
both live sessions. A syntax error in its embedded python makes it allow
everything (measured 2026-08-10: one apostrophe). Every change must keep
`prove_qa_write_separation_86_31.py` at 15/15 and the liveness probe green.

## 6. References

- `handoff/current/research_brief_86.33.md` (gate `wf_b4180a25-c62`, floors met,
  **zero searches** — §1)
- RFC 9700 §4.15; SPIFFE concepts; SELinux domain transitions; `no_new_privs`;
  seccomp filter; `credentials(7)`; Claude Code hooks doc; confused-deputy
- `.claude/hooks/qa-write-guard.sh:13-16, 66-93`; `handoff/logs/qa_write_guard.log`
