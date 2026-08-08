---
name: kill-switch-deadlock-85-6
description: Step 85.6 research — the un-resumable book; two step premises corrected (wrong autonomous_loop.py, wrong latch date); the SOD roll has ONE trigger; resilience4j names this exact bug as a shipped default
metadata:
  type: project
---

Step 85.6 (P0: kill switch cannot be un-paused). Researched 2026-08-08. Brief:
`handoff/current/research_brief_85.6.md`.

**Fact / decision-shaping findings:**

- **TWO `autonomous_loop.py` files.** `backend/autonomous_loop.py` (620 lines) is
  the HARNESS Plan/Generate/Evaluate loop and has ZERO kill-switch references.
  The trading cycle is `backend/services/autonomous_loop.py`. Any step citing
  "autonomous_loop.py:<line>" above ~620 means the `services/` one. I have now
  seen a step definition get this wrong once.
- **The SOD roll has exactly ONE production trigger.** `update_sod_nav` is called
  from one place only (`paper_trader.py:1298`, gated by `sod_anchor_needs_reroll`
  at `:1297`), inside `check_and_enforce_kill_switch`, whose only production
  caller is `services/autonomous_loop.py:1375` — Step 5.5 of 10, behind the
  analysis phase at `:1148`. Everything else in a repo-wide grep is tests.
- **`_paused_at` is reset by EVERY pause row on replay** (`_load_from_audit:270`).
  The live audit file had 36 redundant `trigger:"manual"` pause rows since the
  last resume, several arriving the day I looked. So the phase-38.1 auto-resume
  hysteresis clock (`AUTO_RESUME_TRIGGER_AT_SEC=2h`) can never mature while they
  keep landing. Any design that "just enables auto-resume" looks correct and
  never fires. I did not identify the writer.
- **Audit-file replay IS the state store.** `handoff/kill_switch_audit.jsonl` is
  git-tracked; last `resume` and last `sod_snapshot` rows fully determine paused
  state and anchor freshness. Reading it with a 5-line Python histogram is the
  fastest way to establish live kill-switch state without touching the service.

**Why:** 85.6 asked whether the anchor can roll without the analysis phase
completing. Answer required knowing every writer of the anchor and every caller
of its container.

**How to apply:** before proposing ANY kill-switch change, grep the writer AND
the container's callers, and replay the audit JSONL rather than trusting a
snapshot endpoint. See [[project_kill_switch_36_9_armed_semantics]],
[[project_kill_switch_36_12_traps]], [[project_cycle_never_completes_85_4]].
