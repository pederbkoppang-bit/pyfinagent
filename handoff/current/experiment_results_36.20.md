# Experiment results — masterplan step 36.20

**[P1] The cockpit badge read DISARMED on a healthy book every morning** — plus a **fabricated
`0.00%`** the research gate found on the same panel, which is the more dangerous of the two.

## What shipped

| File | Change |
|---|---|
| `frontend/src/components/KillSwitchPanel.tsx` | new `reanchoring` predicate + `dailyUnevaluable`; RE-ANCHORING badge (sky); em-dash guard now branches on unevaluable; Resume reason text corrected; breach type extended with 4 optional keys |
| `frontend/src/components/OpsStatusBar.tsx` | same predicate + badge + colour + em-dash guard + type extension |
| `frontend/src/components/KillSwitchPanel.reanchoring.test.tsx` | NEW — 8 tests |

**Behaviours changed, stated as behaviours:**
1. A healthy book whose daily anchor is from an earlier UTC day now renders **RE-ANCHORING** (sky, non-alarm) instead of the DISARMED alarm — in **both** components.
2. Genuine absence still renders DISARMED with alarm styling. The two are distinguishable from an identical `armed: false`.
3. An unmeasurable NAV inside the stale window does **not** get the friendly badge.
4. A daily leg that cannot fire prints `—`, never a fabricated `0.00%`.
5. The card/icon alarm rollup no longer tints for the stale state — status, not alarm.
6. The Resume tooltip states the real cause; it no longer asserts the baselines were unrestorable.
7. `armed` remains a strict boolean on the wire; the new state is derived client-side.

## The research gate changed the design AND found a worse defect

`handoff/current/research_brief_36.20.md` — `gate_passed: true`, **9 sources read in full**, 40 URLs,
recency scan performed, 10 internal files inspected.

**It contradicted my instinct on criterion 3.** I intended Resume enabled for the stale state; the
server 409s on exactly that state, so enabling it converts a badge into a click that always fails.

**It found the `0.00%`.** `daily_loss_pct` keeps its `0.0` initialiser when the leg is skipped
(`kill_switch.py:781`, `:828-832`), but both components branched the em-dash on
`daily_baseline_missing` **alone** (`KillSwitchPanel.tsx:189`, `OpsStatusBar.tsx:320`). Verified on
the live endpoint: `daily_loss_pct: 0.0` served with `daily_baseline_stale: true`. This errs
*reassuring* — the direction phase-36.7 exists to prevent — and it is a consequence of my own 36.9
change, so it is in scope, not a discovery to punt.

**It found a classification trap.** `evaluate_breach`'s `nav_invalid` early return passes
`daily_baseline_stale` through while both `*_missing` stay false, and `GET /kill-switch` falls back to
`... or 0.0` on a 5s BQ timeout. Without an explicit `nav_invalid` exclusion, a genuine "we cannot
measure the book" state would have rendered as the friendly badge. Guarded, and mutation-proven (M2).

**The literature is unanimous.** ISA-18.2 defines an alarm as a condition *requiring a response* and
prescribes reclassification otherwise; AHRQ quantifies the harm of non-actionable alarms; Google SRE
and AWS OPS08-BP04 restate it for software; Kubernetes readiness-vs-liveness is the architectural
analogue. The counter-argument (Hexagon/ALI, Google SRE: *eliminate, don't expand*) is recorded and
rejected — rendering plain ACTIVE would assert coverage the daily leg does not have, which is
phase-36.7 in mirror image.

**Sky, not amber**, because amber is already DISARMED's token and this project's degraded colour.
Per WCAG 1.4.1 the state is carried by badge **text**, with colour as a secondary cue only.

## Verification

```
$ cd frontend && npx tsc --noEmit          # IMMUTABLE
(no output, exit 0)

$ npx vitest run src/components/KillSwitchPanel.reanchoring.test.tsx
Tests  8 passed (8)

$ npx vitest run src/components/KillSwitchPanel.disarmed.test.tsx   # the 36.7 guard
Tests  13 passed (13)

$ npx vitest run                            # whole suite
Test Files  37 passed (37)    Tests  268 passed (268)
```

Mutation matrix: **4 killed / 0 survived** at baseline `8 passed` — see `live_check_36.20.md` (e).

## What is NOT done, stated plainly

**The Playwright `:3100` capture was not taken.** Full reasoning in `live_check_36.20.md`, which
leads with it. In short: I broke the operator's `:3000` with that rig earlier in this session, and
the project's own frontend rule §5 explicitly permits "visual verification pending operator review"
as the alternative. This does **not** satisfy the immutable live_check as written, and the Q/A must
rule on whether the step can close without it. `:3000` verified at **200** afterwards; no dev server
was started at all.

**Criterion 3 was not implemented as literally worded** (Resume stays disabled). Reasoning in the
contract and live_check; the Q/A must rule.

## Out of scope → to be FILED before this step flips

- `sod_date` and `baseline_provenance` are served but read by **zero** frontend components, so
  phase-36.12's `baseline_provenance == "lost_history_anchor"` marker — "armed, but measuring from a
  fiction" — is currently invisible in the UI.
