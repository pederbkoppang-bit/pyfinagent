---
name: kill-switch-ui-36-20
description: Kill-switch cockpit facts measured 2026-07-26 -- the stale-anchor false alarm, the fabricated 0.00%, the nav_invalid classification trap, and why "enable Resume" is wrong
metadata:
  type: project
---

Findings from the phase-36.20 research gate (frontend alarm-design step),
measured against the running code on 2026-07-26. Re-verify line numbers before
acting; the semantics below are the durable part.

**Only TWO frontend files read `armed`** -- `KillSwitchPanel.tsx` and
`OpsStatusBar.tsx` (`KillSegment`). Zero frontend reads of
`daily_baseline_stale`, `baselines_present`, `sod_date` or
`baseline_provenance`, all of which the endpoint already serves. No MCP server
or script reads `armed` either. The consumer census is genuinely that small.

**`daily_baseline_stale` and `daily_baseline_missing` are MUTUALLY EXCLUSIVE by
construction** -- `_sod_date_is_stale` returns False when `sod_nav` is absent
("reporting it as stale on top would be a second name for the same absence").
That is the clean client-side discriminator between the self-clearing overnight
state and genuine lost baselines; no new backend key is needed.

**THE CLASSIFICATION TRAP: `evaluate_breach`'s `nav_invalid` early return passes
through the computed `daily_baseline_stale` while both `*_missing` can be
false.** `GET /kill-switch` falls back to `... or 0.0` on a 5s BQ timeout, so a
timeout inside the stale window produces `armed:false, stale:true,
missing:false, missing:false` -- identical to the benign case while NOTHING can
be measured. Any "is this the benign stale state?" predicate MUST also exclude
`nav_invalid` / `nav_invalid_disarmed`, or it renders calm on a total
measurement failure.

**The em-dash guard misses the stale case, so the cockpit prints a fabricated
`0.00%`.** `daily_loss_pct` stays at its 0.0 initialiser whenever
`daily_leg_unevaluable` (missing OR stale), but both components branch the
em-dash on `daily_baseline_missing` alone. Guard must be
`missing || stale`. This errs REASSURING and is the more dangerous of the two
36.20 bugs.

**"Enable Resume" is the wrong instinct** -- `POST /resume` raises a
stale-specific **409** whose own text says "NO operator action is required".
Enabling the button just converts a badge into a failed click. But the UI's
tooltip ("loss baselines unrestorable") is FALSE in that state and directly
contradicts that 409. Also: Resume only RENDERS when `paused === true`, so on a
`paused:false` book the badge is the only visible harm.

**Amber is already taken twice** -- it is DISARMED's token AND the project's
degraded/unknown colour (`CycleSegment` worst-of-N, stale-poll segment). A new
non-alarm state must use sky (the project's established in-progress/info token),
never amber.

**`KillSwitchPanel.disarmed.test.tsx` booby-trap:** six assertions are
whole-container `not.toContain("ACTIVE")` / `not.toContain("DISARMED")`, so any
new label sharing those substrings breaks them and invites loosening. And
`expect(html).toContain("amber")` is a weak assertion any amber token satisfies.

**Why: the alarm-design literature is unanimous** (ISA-18.2 "requiring a
response"; AHRQ 72-99% false alarms and 566 FDA deaths; Google SRE; AWS
OPS08-BP04 rates the risk High) that a recurring non-actionable alarm trains
operators to ignore the real one. Kubernetes readiness-vs-liveness is the exact
architectural analogue and its docs carry a boxed Caution about conflating them.
The strongest counter -- Hexagon/ALI + SRE "eliminate, don't expand" -- targets
*suppression* and *paging-rule count*, not display states, and does not cover a
condition with partial safety significance.

**How to apply:** any future kill-switch UI work should derive states from the
already-served discriminators with strict `=== true`/`=== false`, keep `armed` a
strict boolean (both backend gates are `.get("armed", True)` = fail-OPEN, so a
tri-valued `armed` silently opens them), and treat "unknown is not healthy" as
binding -- rendering plain ACTIVE during the window re-creates phase-36.7.

Related: [[kill-switch-36-9-armed-semantics]], [[kill-switch-36-12-traps]],
[[fabricated-safe-80-36]].
