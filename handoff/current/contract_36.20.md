# Contract — masterplan step 36.20

**[P1] The cockpit badge reads DISARMED on a healthy book every morning** — and, found by the
research gate, a **fabricated `0.00%`** is on that same panel right now.

Step id: `36.20` · Phase: PLAN · Date: 2026-07-26 · HEAD at contract time: `3227347a`

## Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. A stale-but-present daily anchor (`daily_baseline_stale=true`, `daily_baseline_missing=false`, `trailing_baseline_missing=false`) renders a visually distinct NON-alarm state in BOTH KillSwitchPanel and OpsStatusBar -- not the DISARMED badge
2. Genuine absence (`daily_baseline_missing` or `trailing_baseline_missing` true) still renders DISARMED with alarm styling -- a test proves the two states are distinguishable and FAILS if they are collapsed
3. The Resume button is NOT disabled for the stale-only state, and IS still disabled for genuine absence
4. No consumer encodes the third state as `armed: undefined`; both backend `.get("armed", True)` gates keep their fail-OPEN behaviour unchanged
5. MUTATION-TEST: collapsing the new state back into `armed === false` must fail the new test

Immutable command: `cd frontend && npx tsc --noEmit`
Immutable `live_check`: a Playwright capture from the isolated skip-auth `:3100` rig showing the
stale-anchor state rendering as the new non-alarm badge with Resume enabled, plus a second capture of
genuine absence still rendering DISARMED with Resume disabled.

## Research gate — it contradicted criterion 3, and I must say so up front

`handoff/current/research_brief_36.20.md` — `gate_passed: true`, **9 sources read in full** (floor 5),
40 URLs, recency scan performed, 10 internal files inspected.

**Four independent domains converge on the same rule: a condition requiring no operator response is
not an alarm.** ISA-18.2 defines an alarm as an abnormal condition *requiring a response*, and
prescribes reclassification to a recordable event otherwise. AHRQ quantifies the harm (72–99% of
clinical alarms false; staff "turn down the volume, ignore, or deactivate"; 566 FDA-reported deaths
2005–2008). Google SRE and AWS OPS08-BP04 restate it for software, AWS rating the risk **High**.
Kubernetes readiness-vs-liveness is the exact architectural analogue for *stale* (self-clearing) vs
*missing* (durable).

**CRITERION 3 IS FACTUALLY WRONG AND I WILL NOT IMPLEMENT IT AS WRITTEN.** It requires Resume to be
enabled for the stale-only state. But `paper_trading.py:612` returns a **409 on exactly that state** —
the staleness-specific refusal I added in step 36.13's cycle 3. Enabling the button would convert a
misleading badge into a button that always fails. The criteria are immutable and I am not amending
them; I am recording that criterion 3 cannot be satisfied as literally worded without re-opening a
closed step's server behaviour, and implementing the **intent** — the stale state must not be
presented as a durable fault — by fixing the button's *reason text* instead of its enabled state.
The Q/A must rule on this explicitly. If it judges the literal wording binding, this step FAILS and
the right fix is a follow-up step that changes the 409, not a UI change that lies about it.

Note also (measured): the Resume button is only *rendered* when `paused === true`, and the live book
is `paused: false`. So on the live system the **badge is the sole visible harm** today.

## What the gate found that the step did not know — a WORSE defect, live now

**A fabricated `0.00%` is on screen.** `daily_loss_pct` stays at its `0.0` initialiser whenever the
leg is unevaluable (`kill_switch.py:781`, `:828-832` — `daily_leg_unevaluable = missing OR stale`),
but both components branch the em-dash on `daily_baseline_missing` **alone**
(`KillSwitchPanel.tsx:189`, `OpsStatusBar.tsx:320`). So the panel currently prints
`daily 0.00% / 4%` for a leg that **cannot fire**. This errs *reassuring*, which makes it more
dangerous than the badge bug this step was filed for, and it is a direct consequence of my own 36.9
change. **In scope**: it is the same defect family (the UI conflating "unevaluable" with "healthy")
and criterion 2's distinguishability requirement is meaningless while the percentage lies.

**A classification trap the step's framing missed.** `evaluate_breach`'s `nav_invalid` early return
(`kill_switch.py:803-826`) passes the computed `daily_baseline_stale` through while both `*_missing`
stay false, and `GET /kill-switch` falls back to `... or 0.0` on a 5s BQ timeout
(`paper_trading.py:516`). So **a BQ timeout inside the stale window is indistinguishable from the
benign stale case** unless the new predicate also excludes `nav_invalid` / `nav_invalid_disarmed`.
Without that guard the new non-alarm badge would swallow a real "we cannot measure the book" state.

**The fail-open trap is confirmed** at `paper_trading.py:630` and `kill_switch.py:994`, both
`.get("armed", True)`. The design therefore keeps `armed` a strict boolean and derives the new state
**client-side**; neither gate is touched (criterion 4).

## Plan

**A fourth badge VALUE, classified as status and not alarm.** Derived client-side as:
`stale && !daily_baseline_missing && !trailing_baseline_missing && !nav_invalid && !nav_invalid_disarmed`.
Rendered **sky**, not amber — amber is already DISARMED's token *and* the project's degraded colour,
so reusing it would defeat the visual distinction criterion 1 demands. Removed from the `alarm`
rollup so the card and icon stay neutral. Per WCAG 1.4.1 the state is carried by **text plus icon**,
never colour alone.

**The em-dash guard is corrected** in both components to branch on *unevaluable* (missing OR stale),
so a leg that cannot fire shows `—` rather than a fabricated `0.00%`.

**Resume stays disabled** for the stale state, with truthful text matching the server's own 409
("the anchor is from <date>; the next cycle re-anchors it; no operator action required"), instead of
the current tooltips at `KillSwitchPanel.tsx:175/:224` which assert the baselines "could not be
restored" — a cause `paper_trading.py:618-628` explicitly refutes.

**Counter-argument, documented and rejected on the record.** Hexagon/ALI and Google SRE both argue
*eliminate alarms, don't expand the taxonomy*. Rejected here because the alternative — rendering plain
ACTIVE — would assert coverage the daily leg does not have. That is phase-36.7 in mirror image:
"unknown is not healthy" (`kill_switch.py:816`).

## Out of scope → to be FILED as their own steps before this one flips

("(to be filed)" is not a disposition — step 36.13 cycle 2 failed on exactly that.)
- `sod_date` and `baseline_provenance` are served but read by **zero** frontend components, so
  `baseline_provenance == "lost_history_anchor"` — phase-36.12's "armed, but measuring from a
  fiction" marker — is currently **invisible in the UI**.

## Do-no-harm

Frontend only; no backend behaviour change; `armed` semantics untouched so both fail-open gates are
byte-identical. Kill-switch limits, stops, sector caps, DSR, PBO untouched. **Never drive `:3000`** —
UI evidence only from the isolated skip-auth `:3100` rig with its own `PLAYWRIGHT_DIST_DIR`, and
`curl :3000/login` must return 200 afterwards (a second `next dev` broke it earlier in this session).
`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e`.

## References

- `handoff/current/research_brief_36.20.md`
- [ISA-18.2 via Yokogawa](https://www.yokogawa.com/us/library/resources/media-publications/implementing-alarm-management-per-the-ansi-isa-182-standard-control-engineering/) · [AHRQ Alarm Fatigue](https://www.ncbi.nlm.nih.gov/books/NBK555522/) · [Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/) · [AWS OPS08-BP04](https://docs.aws.amazon.com/wellarchitected/latest/framework/ops_workload_observability_create_alerts.html)
- [Kubernetes probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/) · [WCAG 1.4.1](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html) · [Red Hat badge a11y](https://ux.redhat.com/elements/badge/accessibility/) · [Hexagon/ALI](https://aliresources.hexagon.com/operations-maintenance/the-most-important-alarm-improvement-technique-in-existence)
- Internal: `KillSwitchPanel.tsx:137/:175/:189/:219/:224`, `OpsStatusBar.tsx:318/:320/:368`,
  `kill_switch.py:781/:803-826/:828-832/:994`, `paper_trading.py:516/:612/:618-628/:630`
