# Goal — Masterplan drain, full-day AUTONOMOUS run (set 2026-08-07)

Supersedes goal_masterplan_drain_2026-07-26.md. The operator is AWAY all day:
run start to finish without them. NEVER ask the operator anything mid-run (no
AskUserQuestion — it blocks forever). A step needing an operator decision is
recorded in the ask list and SKIPPED, not waited on. When one step closes,
start the next immediately; never idle.

## Read first
1. This file. 2. CLAUDE.md (harness protocol, Fable budget rule, effort
policy). 3. .claude/rules/research-gate.md. 4. Auto-memory MEMORY.md — every
feedback_* lesson is binding.

## Measured scope (2026-08-07 — RE-DERIVE before trusting)
status=="pending": 316 steps / 27 phases. Priority {P0:25, P1:80, P2:139,
P3:53, P4:8, unset:11}. 53 are harness_required:false (operator actions,
mostly phase-79) → 263 executor steps. Naive "not done" = 345 (adds
deferred 15, blocked 1, merged 2, superseded 4, dropped 7) — state which
count you mean.

## Wave order (each step = FULL harness cycle)
1. P0 gates that block whole phases: 83.0, 83.0.1, 83.0.3, 83.1.1 (news
   corpus + timestamps + PBO tool + go/no-go arithmetic) and 82.23 (promotion
   gate's PBO term never computed). These decide whether phase-83 is buildable
   at all — before any phase-83 build step.
2. Remaining executor P0s, oldest first: 27.6*, 61.2/61.3, 62.1/62.2/62.7,
   63.3/63.4, 65.2/65.4, 68.1/68.3/68.5, 72.0.1/72.0.2. Re-read each first;
   rescope or close dropped-with-reason where stale.
3. Newest P1 tail: 84.1–84.4, 85.1–85.3, 4000.4/4000.5/4000.10, 81.0/81.3.
4. phase-83/82 P1s in dependency order (83.1 → 83.1.2 → 83.2 → 83.3 → 83.4 →
   83.5), then the phase-82 tail (42 pending — triage before executing).
5. P2+: an obsolete step is closed dropped-with-reason, not executed.

## Non-negotiable
- Full harness per step: researcher (≥5 sources read in full) → contract.md
  (criteria byte-for-byte; tier as a NAMED field) → GENERATE → ONE fresh Q/A
  on the Workflow rail (lean prompt; persist the verdict the same turn it
  returns) → harness_log append → flip. No self-eval; no verdict-shopping on
  unchanged evidence; 3rd consecutive CONDITIONAL auto-FAILs.
- DO-NO-HARM: paper only; historical_macro FROZEN; no optimizer runs;
  immutable gates (DSR≥0.95, PBO≤0.20) byte-untouched; nothing that moves the
  live book.
- $0 metered: Max rail only. No Fable roster repin; T4 per-invocation only,
  always with an Opus fallback; errored/empty return = NO VERDICT, never PASS.
- UI claims: Playwright capture on the isolated :3100 skip-auth rig, never the
  operator's :3000. Restore tsconfig.json + next-env.d.ts after.
- git add -An before every flip (the hook stages the whole tree); verify
  git log -1 in the same turn. Mutation-test every guard (mutate the
  production call site, not the helper). Measure, never assert — no claim
  about a set whose membership rule you didn't run.

## Done-definition (HARD STOP for the day)
Not all 316. The day closes green when: every executor P0 is PASS or
dropped/deferred with a recorded reason; the five wave-1 gate steps have
verdicts; ≥2 P1s from {84.x, 85.x, 4000.x} are closed; and the operator ask
list is batched in ONE file (handoff/current/operator_ask_2026-08-07.md) with
a recommendation per item — a recorded DECLINE closes a step as validly as
doing it.

## Stop conditions
SOFT: 12 cycles, or only operator-gated work remains → write the handoff
summary + regenerate this goal file with measured state carried forward for
the next session. HARD: any live-book move; any safety-gate change in the
less-conservative direction; metered spend; a Fable roster repin. Before
declaring a step operator-blocked, check whether a rig you own can measure or
fix it. After every background-agent notification: check git log and
re-verify the working tree before the next flip.
