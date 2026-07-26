# Research Brief -- phase-36.20 (P1, frontend)

**Tier:** moderate (caller-specified). **Audit-class:** false.
**Question:** the kill-switch cockpit renders a DISARMED alarm on a healthy
book every day from 00:00 UTC until the first autonomous cycle rolls the
daily anchor, because both frontend consumers derive `disarmed =
breach.armed === false` and neither reads the new `daily_baseline_stale`
key introduced by step 36.9. This is an ALARM-DESIGN problem (a recurring,
self-clearing, non-actionable alarm), not a React problem.

**Status:** COMPLETE -- gate passed (9 sources read in full, 40 URLs, recency
scan performed). Written incrementally per write-first discipline.

---

## Queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| Q1 | `EEMUA 191 ISA 18.2 alarm management nuisance alarm operator action required definition` | year-less canonical |
| Q2 | `alarm management standing alarm stale alarm self-clearing condition status indication not an alarm` | year-less canonical |
| Q3 | `alarm fatigue clinical alarms non-actionable nurses ignore Joint Commission sentinel event alert 50 percentage false alarms` | year-less canonical |
| Q4 | `alert fatigue 2026 observability non-actionable alerts reduce noise actionable alerting best practice` | current-year (2026) |
| Q5 | `status badge design degraded state three-state health indicator design system accessibility` | year-less canonical |
| Q6 | `ISA 18.2 IEC 62682 2025 alarm rationalization revision alarm philosophy update` | last-2-year (2025) |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html | 2026-07-26 | Standard (W3C, Level A) | WebFetch, full | "Color is not used as the only visual means of conveying information, indicating an action, prompting a response, or distinguishing a visual element." Level **A**. Explicitly permits colour coding "if it is complemented by other visual indication". Failures F73/F81; techniques G182/G183/G205. |
| 2 | https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/ | 2026-07-26 | Official docs (CNCF) | WebFetch, full | The canonical two-class health split. Readiness = "recovering from temporary faults or overloads" -> remove from traffic, **does NOT restart**. Liveness = "when to restart a container ... catch a deadlock". "**Incorrect implementation of liveness probes can lead to cascading failures.** ... **Understand the difference between liveness and readiness probes.**" |
| 3 | https://www.ncbi.nlm.nih.gov/books/NBK555522/ | 2026-07-26 | Peer-reviewed / federal evidence review (AHRQ, *Making Healthcare Safer III*) | WebFetch, full | "Alarm fatigue occurs when clinicians experience high exposure to medical device alarms, causing alarm desensitization and leading to missed alarms or delayed response." False alarms **72%-99%**. Non-actionable = "an alarm system works as designed but signifies an event that is not clinically significant". Consequence: "staff doubt the reliability of alarms and as a result **turn down the volume, ignore, or deactivate** the alarms". FDA: **566 patient deaths** 2005-2008; TJC Sentinel Event Alert 2013 + 2014 National Patient Safety Goal. Remedy emphasis: adjust thresholds / discontinue unnecessary monitoring **rather than adding new alerts**. |
| 4 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-07-26 | Authoritative engineering (Google SRE book) | WebFetch, full | **[ADVERSARIAL to a new state]** "Every page should be actionable." "Every page response should require intelligence. If a page merely merits a robotic response, it shouldn't be a page." Noise makes operators "second-guess, skim, or even ignore incoming alerts, sometimes even ignoring a 'real' page that's masked by the noise." Simplicity rule: "The rules that catch real incidents most often should be as simple, predictable, and reliable as possible." Remove signals not exposed on a dashboard nor used by an alert. |
| 5 | https://aliresources.hexagon.com/operations-maintenance/the-most-important-alarm-improvement-technique-in-existence | 2026-07-26 | Industry practitioner (Hexagon/ALI, *Alarm Management Handbook* lineage) | WebFetch, full | **[ADVERSARIAL to a new state]** "Bad actor" resolution cuts alarm rates "by 60% to 80% or more". "The top 20 most frequent alarms usually comprise anywhere from 25% to 95% of the entire system load"; one system: 10 alarms = 96% of occurrences. Philosophy is **elimination, not expansion**: "at the end of the bad actor resolution step, there should be no suppressed alarms left"; rejects adding new alarm classes/priorities as the fix. |
| 6 | https://www.yokogawa.com/us/library/resources/media-publications/implementing-alarm-management-per-the-ansi-isa-182-standard-control-engineering/ | 2026-07-26 | Industry / standards summary (Control Engineering via Yokogawa) | WebFetch, full | The ISA-18.2 alarm definition verbatim: "An audible and/or visible means of indicating to the operator an equipment malfunction, process deviation, or abnormal condition **requiring a response**." Alarms are distinguished from messages/events "by their urgency requirement"; "Many nuisance alarms can be **reclassified as recordable events** rather than active alerts requiring operator intervention." |
| 7 | https://docs.aws.amazon.com/wellarchitected/latest/framework/ops_workload_observability_create_alerts.html | 2026-07-26 | Official docs (AWS Well-Architected OPS08-BP04) | WebFetch, full | Anti-pattern #1: "Setting up too many non-critical alerts, leading to alert fatigue." "**Reduce alert fatigue**: Minimize non-critical alerts. When teams are overwhelmed with numerous insignificant alerts, they can lose oversight of critical issues, which diminishes the overall effectiveness of the alert mechanism." Risk level if not established: **High**. |
| 8 | https://ux.redhat.com/elements/badge/accessibility/ | 2026-07-26 | Official design-system docs (Red Hat) | WebFetch, full | "Relying on color alone to communicate information causes barriers to access for many readers." "**In addition to indicating badge status via color, visible or visually-hidden text should be added manually for context.**" Badges must not be interactive / focusable. Cites WCAG SC 1.4.1. |
| 9 | https://www.processvue.com/resources/alarm-management-guidelines/ | 2026-07-26 | Industry (ProcessVue) | WebFetch, full -- **LOW YIELD, disclosed** | Page carried only framework-level text (alarms must be "Relevant ... Timely ... Prioritized"; 10-stage lifecycle; rationalization = alarms "necessary, unique, and actionable"). It did **not** carry the EEMUA-191 rate targets or the nuisance taxonomy I fetched it for. Recorded honestly rather than paraphrased into something stronger. |

**Failed / rejected fetch attempts (disclosed, not counted):**

| URL | Outcome |
|-----|---------|
| https://plcprogramming.io/blog/what-is-alarm-management | HTTP **429** Too Many Requests -- not retried inside budget |
| https://carbondesignsystem.com/patterns/status-indicator-pattern/ | Fetched but returned truncated -- no extractable guidance |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC4587624/ | Wrong article (a rat-pharmacology paper); my PMC ID was bad. Replaced by source #3, which is stronger. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.eemua.org/products/publications/digital/eemua-publication-191 | Standard (paywalled) | EEMUA 191 is a purchased publication; no free full text |
| https://www.isa.org/standards-and-publications/isa-18-series-of-standards | Standard (paywalled) | ANSI/ISA-18.2-2016 is a purchased standard |
| https://www.exida.com/Alarm-Management/Resources | Industry | Link hub, no primary content |
| https://www.empoweredautomation.com/alarm-management-standards-and-best-practices | Industry | Duplicate of #6's content at lower authority |
| https://assets.new.siemens.com/.../simatic-pcs-7-alarm-management.pdf | Vendor PDF | Binary PDF; superseded by #6 |
| https://www.a14m.uk/2018/08/stale-alarms/ | Industry blog | Snippet already gave the definition ("alarm ... never returns to a normal state within 24 hours") |
| https://techdocs.broadcom.com/.../persistent-and-stale-alarms.html | Vendor docs | Product-specific; snippet sufficient |
| https://www.sciencedirect.com/topics/engineering/alarm-condition | Reference | Snippet gave the self-clearing/condition-vs-alarm distinction |
| https://www.jointcommission.org (SEA 50) | Standard body | Superseded by #3, which quotes the same TJC/NPSG facts |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11036661/ | Peer-reviewed (2024) | Recency-scan hit; corroborates #3, adds no new mechanism |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC12321328/ | Peer-reviewed (2025) | Recency-scan hit; intervention study, same direction |
| https://nacns.org/wp-content/uploads/2016/11/AF-Introduction.pdf | Professional body | 85-99% non-actionable figure already captured |
| https://www.ncbi.nlm.nih.gov/.../NBK555522 refs | -- | -- |
| https://v10.carbondesignsystem.com/patterns/status-indicator-pattern/ | Design system | Snippet gave the colour/shape/symbol/text multi-channel rule |
| https://design-system.agriculture.gov.au/components/status-badge | Design system | Snippet only; corroborates Red Hat |
| https://designsystemproblems.com/accessibility-compliance/accessible-status-indicators/ | Blog | "at least three of four indicators" claim; lower tier |
| https://aws-observability.github.io/observability-best-practices/signals/alarms/ | Official docs | Superseded by #7 |
| https://oneuptime.com/blog/post/2026-02-20-monitoring-alerting-best-practices/view | Blog (2026) | Recency hit; "only alert on things that are actionable" |
| https://www.logicmonitor.com/blog/network-monitoring-avoid-alert-fatigue | Vendor blog | Same direction, lower tier |
| https://icinga.com/blog/alert-fatigue-monitoring/ | Vendor blog | Same direction |
| https://panther.com/blog/what-is-alert-fatigue | Vendor blog | Same direction |
| https://www.linkedin.com/pulse/alarm-management-standards-compliance-2026-whats-changed-trywc | Industry post (2026) | Recency hit for standards status; lowest tier, not load-bearing |
| https://www.isa.org/intech-home/2021/august-2021/departments/isa18-update-management-of-alarms | Standards body | ISA18 committee status; snippet sufficient |
| https://aliresources.hexagon.com/operations-maintenance/understanding-the-iec-62682-alarm-management-standard | Industry | Fetched; landing page only, white paper behind a PDF download |
| https://www.instrumentationblog.in/alarm-management-isa-18-2/ | Blog | Lower tier |
| https://www.myamericannurse.com/hear-hear-combating-alarm-fatigue/ | Professional press | Superseded by #3 |
| https://nurse.org/articles/alarm-fatigue-statistics-patient-safety/ | Trade press | 350 alarms/bed, 85-99% non-actionable |
| https://www.patientsafetysolutions.com/docs/May_2013_Joint_Commission_Sentinel_Event_Alert_Alarm_Safety.htm | Secondary | TJC SEA summary; 98 events / 80 deaths |

**Unique URLs collected: 40** (9 read in full + 3 failed attempts + 28 snippet-only).

## Recency scan (2024-2026)

Performed via Q4 (2026-scoped) and Q6 (2025-scoped), plus the 2024/2025 PMC
hits surfaced by Q3.

**Result: no new finding supersedes the canonical sources; two corroborate
and one clarifies standards status.**

1. **Standards status (Q6).** ANSI/ISA-18.2 remains at its **2016** revision;
   IEC 62682 (2014, updated 2022/2023) is aligned with it and with EEMUA 191.
   I found **no 2025 or 2026 revision** to the alarm definition or the
   rationalization test. The canonical "requiring a response" definition
   (source #6) is current, not stale.
2. **Observability practice 2026 (Q4).** AWS Well-Architected OPS08-BP04
   (source #7) and 2026 practitioner writing restate Google SRE's 2016
   actionability rule verbatim in substance ("only alert on things that are
   actionable"). Ten years on, the rule has not been revised or softened --
   it has been adopted into vendor best-practice frameworks. That *raises*
   confidence in source #4 rather than superseding it.
3. **Clinical alarm fatigue 2024-2025.** PMC11036661 (2024, ICU alarm fatigue
   and perceived stress) and PMC12321328 (2025, bed-exit alert management)
   are recent primary studies in the same direction as the AHRQ review; both
   report the same desensitization mechanism. No contradicting recent study
   surfaced.

Nothing in the 2024-2026 window argues that a recurring non-actionable
alarm is harmless, and nothing argues for adding alarm classes as the
remedy.

## Key findings

1. **A condition that requires no operator response is, by definition, not an
   alarm.** ISA-18.2 defines an alarm as "an audible and/or visible means of
   indicating to the operator an equipment malfunction, process deviation, or
   abnormal condition **requiring a response**" (Yokogawa/Control Engineering,
   https://www.yokogawa.com/us/library/resources/media-publications/implementing-alarm-management-per-the-ansi-isa-182-standard-control-engineering/,
   accessed 2026-07-26). The rationalization test is therefore "what must the
   operator DO?" -- and the standard's prescribed disposition for a condition
   with no required action is to **reclassify it as a recordable event**, not to
   annunciate it: "Many nuisance alarms can be reclassified as recordable events
   rather than active alerts requiring operator intervention" (ibid.). The
   pyfinagent daily-stale window has a literally-zero-action remedy -- the
   backend's own 409 text says "**NO operator action is required**"
   (`backend/api/paper_trading.py:624`) -- so under ISA-18.2 it fails the alarm
   test outright.

2. **Non-actionable alarms measurably destroy trust in the whole annunciator,
   and the harm is fatal-grade, not cosmetic.** The AHRQ evidence review reports
   false-alarm rates of **72%-99%** and the direct behavioural consequence:
   "staff doubt the reliability of alarms and as a result turn down the volume,
   ignore, or deactivate the alarms" -- with **566 FDA-reported patient deaths
   (2005-2008)**, a 2013 Joint Commission Sentinel Event Alert and a 2014
   National Patient Safety Goal as the result (https://www.ncbi.nlm.nih.gov/books/NBK555522/,
   accessed 2026-07-26). Google SRE states the same mechanism for software
   operators: noise makes people "second-guess, skim, or even ignore incoming
   alerts, sometimes even ignoring a 'real' page that's masked by the noise"
   (https://sre.google/sre-book/monitoring-distributed-systems/, accessed
   2026-07-26). A *daily, guaranteed* false DISARMED is the strongest possible
   trainer of that reflex: it teaches the operator that the DISARMED badge means
   "it's morning", which is precisely the reading that will get a genuine
   lost-baselines event ignored.

3. **The correct architectural shape is TWO fault classes, not one boolean --
   and the canonical implementation names them readiness and liveness.**
   Kubernetes separates a *temporary, self-clearing* condition (readiness ->
   stop routing traffic, **do not restart**; explicitly for "recovering from
   temporary faults or overloads") from a *durable, unrecoverable* one
   (liveness -> restart) and warns in a boxed Caution that conflating them
   causes "cascading failures" and that you must "understand the difference
   between liveness and readiness probes"
   (https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/,
   accessed 2026-07-26). Mapping is exact: `daily_baseline_stale` is a
   readiness-class condition (self-repairs at the next cycle's SOD roll);
   `daily_baseline_missing || trailing_baseline_missing` is a liveness-class
   condition (durable; only an operator restoring baselines from the archives,
   or a KS-PEAK-RESET token, repairs it). Kubernetes also shows the
   *third* member of the family -- the **startup probe**, which exists purely so
   a slow-initialising component is not misdiagnosed as dead. The pyfinagent
   00:00-UTC window is a startup/initialisation window in exactly that sense.

4. **Colour alone is never sufficient, and the fix is cheap.** WCAG 2.2 SC
   1.4.1 (Level **A**): "Color is not used as the only visual means of conveying
   information, indicating an action, prompting a response, or distinguishing a
   visual element" (https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html,
   accessed 2026-07-26) -- while explicitly *permitting* colour coding "if it is
   complemented by other visual indication". Red Hat's design system operationalises
   it for this exact component: "In addition to indicating badge status via
   color, visible or visually-hidden text should be added manually for context"
   (https://ux.redhat.com/elements/badge/accessibility/, accessed 2026-07-26).
   Both pyfinagent badges already carry a text label (`ACTIVE` / `DISARMED` /
   `PAUSED`), so today they PASS 1.4.1; the requirement on a new state is simply
   that it ship a distinct **word**, and ideally a distinct **icon shape**, not
   just a new hex value. Carbon's status-indicator pattern (snippet) makes the
   same point as colour + shape + symbol + text.

5. **[ADVERSARIAL] The alarm-management literature's own prescription is to
   REMOVE, never to ADD.** Hexagon/ALI's bad-actor technique -- "the top 20 most
   frequent alarms usually comprise anywhere from 25% to 95% of the entire system
   load", one system where 10 alarms were 96% of all occurrences, 60-80%
   reductions from fixing a handful -- is explicitly a *deletion* discipline, and
   it insists "at the end of the bad actor resolution step, there should be no
   suppressed alarms left"
   (https://aliresources.hexagon.com/operations-maintenance/the-most-important-alarm-improvement-technique-in-existence,
   accessed 2026-07-26). Google SRE reinforces it: "The rules that catch real
   incidents most often should be as simple, predictable, and reliable as
   possible." AWS Well-Architected lists "setting up too many non-critical
   alerts" as the **first** anti-pattern and rates the risk of ignoring the
   practice as **High**
   (https://docs.aws.amazon.com/wellarchitected/latest/framework/ops_workload_observability_create_alerts.html,
   accessed 2026-07-26). Taken literally, this body of work does NOT endorse
   "add a third badge state" -- it endorses "stop annunciating the non-actionable
   condition at all". That is the strongest counter-argument to the caller's
   instinct and it is answered in the RECOMMENDATION section, not dismissed.

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/components/KillSwitchPanel.tsx` | 315 | Kill-switch card (badge + Pause/Resume/Flatten + confirm modal) | READ IN FULL -- must change |
| `frontend/src/components/OpsStatusBar.tsx` | 505 | One-row operator status bar; `KillSegment` :298-396 | READ IN FULL -- must change |
| `frontend/src/components/KillSwitchPanel.disarmed.test.tsx` | 259 | Vitest pin on the 36.7/36.12 rendering, BOTH components | READ IN FULL -- must be EXTENDED, not weakened |
| `backend/services/kill_switch.py` | 1050+ | `evaluate_breach` :720-853, `baselines_present_in` :856, `_sod_date_is_stale` :876 | READ (relevant range) -- source of the payload |
| `backend/api/paper_trading.py` | 1300+ | `GET /kill-switch` :499-542, `POST /resume` :554-651 | READ (relevant range) -- serves the payload + owns the 409s |
| `backend/services/paper_trader.py` | -- | BUY gate :183-208 + :1233-1243 read `baselines_present`, NOT `armed` | grep-audited |
| `frontend/playwright.config.ts` | 140+ | :3100 skip-auth functional rig | READ (relevant range) |
| `.claude/rules/frontend.md` / `frontend-layout.md` | -- | design-token + status-bar constraints | READ IN FULL |

### A. Exact payload served today (measured shape, from source)

`backend/services/kill_switch.py::evaluate_breach` has TWO return shapes and
BOTH carry the new key:

* **`nav_invalid` early return** -- `kill_switch.py:803-826`; keys include
  `"daily_baseline_stale"` (:817), `"baselines_present"` (:824),
  `"armed": False` (:825), plus `nav_invalid: True` (:808) and
  `nav_invalid_disarmed: True` (:818).
* **normal return** -- `kill_switch.py:840-853`; `"daily_baseline_missing"`
  (:848), `"daily_baseline_stale"` (:849), `"trailing_baseline_missing"`
  (:850), `"baselines_present"` (:851), `"armed"` (:852).

CONFIRMED: `daily_baseline_stale` **is present in both return shapes**
(:817 and :849), so a frontend consumer never has to branch on which shape
it received. Same for `baselines_present` (:824 and :851).

The arming rule itself, `kill_switch.py:768-782`:

```python
daily_baseline_missing = not (sod is not None and sod > 0)
trailing_baseline_missing = not (peak is not None and peak > 0)
daily_baseline_stale = _sod_date_is_stale(s.get("sod_date"), sod)
daily_leg_unevaluable = daily_baseline_missing or daily_baseline_stale
armed = not (daily_leg_unevaluable or trailing_baseline_missing)
```

`_sod_date_is_stale` (`kill_switch.py:876-899`) returns **False** when
`sod_nav` is absent -- so `daily_baseline_stale` and `daily_baseline_missing`
are MUTUALLY EXCLUSIVE by construction (:893-894: "when `sod_nav` is absent
the leg is already unevaluable via `daily_baseline_missing`, and reporting it
as *stale* on top would be a second name for the same absence"). That is the
clean discriminator the frontend needs.

The endpoint that feeds both components is
`backend/api/paper_trading.py:499-542` (`GET /api/paper-trading/kill-switch`);
it passes `breach` straight through at :537 and also serves `sod_date` (:529)
and `baseline_provenance` (:535) -- neither of which any frontend component
reads today.

### B. Frontend consumer census (`grep -rn` over `frontend/src`, COMPLETE)

Reads of `armed`: **exactly 2 non-test call sites.**

| File:line | What it does |
|-----------|--------------|
| `KillSwitchPanel.tsx:25` | `armed?: boolean` in the local `KillSwitchState` interface |
| `KillSwitchPanel.tsx:137` | `const disarmed = breach.armed === false;` |
| `KillSwitchPanel.tsx:138` | `const alarm = paused \|\| breach.any_breached \|\| disarmed;` |
| `OpsStatusBar.tsx:39` | `armed?: boolean` in the local `KillSwitchState` type |
| `OpsStatusBar.tsx:318` | `const disarmed = kill.breach.armed === false;` |
| `OpsStatusBar.tsx:319` | `const alarm = kill.paused \|\| kill.breach.any_breached \|\| disarmed;` |

Reads of `daily_baseline_missing`: `KillSwitchPanel.tsx:26` (type),
`:189` (em-dash readout); `OpsStatusBar.tsx:40` (type), `:320` (em-dash).
Reads of `trailing_baseline_missing`: `KillSwitchPanel.tsx:27`, `:195`;
`OpsStatusBar.tsx:41`, `:323`.

Reads of **`daily_baseline_stale`: ZERO.**
Reads of **`baselines_present`: ZERO.**
Reads of `sod_date`: ZERO in these components (served but unused).
Reads of `baseline_provenance`: ZERO anywhere in `frontend/src`.

(The only other `armed` hits in `frontend/src` are unrelated string literals
in fixture data: `RedLineMonitor.test.tsx:35` `detail: "armed"` and
`StrategyDetail.test.tsx:36/111` `"first_week_armed"`.)

### C. Every use of `disarmed`, with the exact styling tokens

**`KillSwitchPanel.tsx`**

| Line | Use | Current token / text |
|------|-----|----------------------|
| :137 | derive | `breach.armed === false` |
| :138 | `alarm` rollup | feeds icon weight + icon colour |
| :145-149 | card container | `disarmed && !paused && !breach.any_breached` -> `"border-amber-500/40 bg-amber-950/30"`; else alarm -> `"border-rose-500/40 bg-rose-950/30"`; else `"border-navy-700 bg-navy-800/60"` |
| :153-157 | `IconWarning` | `size={16}`, `weight={alarm ? "fill" : "regular"}`, `className={alarm ? "text-rose-400" : "text-slate-500"}` -- **NOTE: this is keyed on `alarm`, not on `disarmed`, so a disarmed-but-not-paused card paints an amber container with a ROSE filled warning icon** (a pre-existing inconsistency with the amber badge at :167) |
| :161-169 | badge pill | base `"rounded-full px-2 py-0.5 text-[10px] font-semibold"`; paused -> `"bg-rose-500/20 text-rose-300"`; disarmed -> `"bg-amber-500/20 text-amber-300"`; else `"bg-emerald-500/20 text-emerald-300"` |
| :170-177 | badge `title=` | `"DISARMED: the loss baselines could not be restored, ... writes an audited baseline_anchor_on_lost_history row ..."` |
| :179 | badge text | `{paused ? "PAUSED" : disarmed ? "DISARMED" : "ACTIVE"}` |
| :219 | Resume `disabled` | `busy \|\| breach.any_breached \|\| disarmed` |
| :220-226 | Resume `title=` | disarmed -> `"Cannot resume: kill switch DISARMED (loss baselines unrestorable). ..."` |

There is **no `aria-live`, no `role="status"`, no `role="alert"`** anywhere in
`KillSwitchPanel.tsx`; the badge conveys state through text + colour only
(the text token is what currently satisfies WCAG 1.4.1 -- see external notes).

**`OpsStatusBar.tsx`** (`KillSegment`, :298-396)

| Line | Use | Current token / text |
|------|-----|----------------------|
| :318 | derive | `kill.breach.armed === false` |
| :319 | `alarm` rollup | feeds `IconWarning weight` at :331 |
| :329-339 | `IconWarning` | `size={14}`, `weight={alarm ? "fill" : "regular"}`; colour: paused/breached -> `"text-rose-400"`, disarmed -> `"text-amber-400"`, else `"text-slate-500"` (this one IS keyed on disarmed, unlike the panel) |
| :340-348 | badge pill | base `"rounded-full px-2 py-0.5 text-[10px] font-semibold"`; paused -> `"bg-rose-500/15 text-rose-300"`; disarmed -> `"bg-amber-500/15 text-amber-300"`; else `"bg-emerald-500/15 text-emerald-300"` (note `/15` here vs `/20` in the panel) |
| :349-353 | badge `title=` | `"DISARMED: loss baselines unrestorable, so neither breach leg can fire"` |
| :355 | badge text | `{kill.paused ? "PAUSED" : disarmed ? "DISARMED" : "ACTIVE"}` |
| :368 | Resume `disabled` | `busy !== null \|\| kill.breach.any_breached \|\| disarmed` |
| :369 | Resume a11y | `aria-label="Resume paper trading"` (no title, so **no reason is exposed at all** when it is disabled) |

Other design-system anchors in this file for a new state: the existing
"degraded" precedent is `CycleSegment` :420-426, which collapses `unknown`
into **amber** ("worst-of-N", phase-23.1.12), and the stale-poll segment at
:186-190 (`IconWarning ... text-amber-400` + `text-amber-300` text +
`data-testid="ops-stale-segment"`). Amber is therefore ALREADY the
project's "degraded / unknown / stale" colour -- and it is already taken by
DISARMED in both components.

### D. THE TRAP the caller flagged, re-derived and CONFIRMED

Both backend gates fail **OPEN** on a missing key:

* `backend/api/paper_trading.py:630` -- `if not breach.get("armed", True):`
  (comment at :597 states the intent explicitly: "`.get("armed", True)` fails
  OPEN on a dict that predates the key.")
* `backend/services/kill_switch.py:994` -- `if not breach.get("armed", True):`
  in the auto-resume hysteresis path; comment :992-993 "Fail-open on a missing
  key so an older/partial dict cannot wedge the hysteresis path."

CONSEQUENCE FOR THE DESIGN: a third state must NOT be encoded by making
`armed` absent/undefined/tri-valued. `armed` must stay a strict boolean and
keep its 36.9 meaning ("can this leg fire RIGHT NOW"). The third state must be
derived on the CLIENT from the ALREADY-SERVED discriminators
(`daily_baseline_stale` vs `daily_baseline_missing` /
`trailing_baseline_missing`), and any new backend key must be ADDITIVE.
Line numbers re-derived 2026-07-26; both are at the caller's stated
locations.

### E. A second, LOUDER defect in the same region: the tooltip LIES

`backend/api/paper_trading.py:612-629` already discriminates the two cases on
the RESUME path and returns a **stale-specific 409** whose text says, verbatim:

> "The baselines themselves are intact ... the trailing leg is
> date-independent and still armed. NO operator action is required: the daily
> start-of-day roll stamps today's anchor at the top of the next
> paper-trading cycle and this refusal clears itself."

Meanwhile the cockpit, on that exact same state, tells the operator
(`KillSwitchPanel.tsx:224`): "Cannot resume: kill switch DISARMED (**loss
baselines unrestorable**)" and (`:175`) "the loss baselines **could not be
restored**". Both are FALSE when `daily_baseline_stale=true` and both
`*_missing` are false -- `baselines_present` is true and the live book has
`sod_nav 23838.19 / peak_nav 24666.57`. The UI asserts a cause its own
backend explicitly refutes. That is the same defect class 36.9 and 36.12
fixed on the money path, now living on the read path.

### F. THE COUNTER-TRAP: the server still refuses RESUME while stale

The caller's instinct includes "Resume NOT disabled". **Verify before
adopting.** `paper_trading.py:612-629` raises **409 on a stale anchor** --
resume is refused by the server in exactly this state. So:

* Leaving Resume ENABLED would produce a click -> 409 -> `window.alert`
  (`OpsStatusBar.tsx:149`) or an error span (`KillSwitchPanel.tsx:241`). That
  is a worse operator experience than a disabled control with an honest
  reason, and it violates the components' own stated design ("mirrors the
  server-side 409", `KillSwitchPanel.tsx:217`).
* Note also that Resume is only RENDERED when `paused === true`
  (`KillSwitchPanel.tsx:213`, `OpsStatusBar.tsx:363`). The live state is
  `paused: false`, so on the daily-stale window the Resume button **is not on
  screen at all** -- the visible harm is the badge, not the button.

Therefore the correct disposition is: keep the control disabled (it mirrors a
real server refusal), but **fix the REASON text** and **stop classifying the
state as an alarm**. Any contract that promises "Resume NOT disabled" must
first change `paper_trading.py:612` -- which is a money-path change, out of
scope for a P1 frontend step.

### G. The existing test pin (`KillSwitchPanel.disarmed.test.tsx`)

Three fixtures, `:45-97`:

* `DISARMED` (:45-65) -- `sod_nav: null, sod_date: null, peak_nav: null`,
  `armed:false, daily_baseline_missing:true, trailing_baseline_missing:true`.
  This is **genuine absence**, NOT the stale case. It carries no
  `daily_baseline_stale` key at all.
* `ARMED_AT_HIGH_WATER` (:69-80) -- `sod_date: "2026-07-26"`, `armed:true`.
* `LEGACY_NO_MARKER` (:83-97) -- pre-36.7 backend, no marker keys.

Ten assertions across two `describe` blocks:

| Test (line) | Asserts |
|-------------|---------|
| :132 | `DISARMED` -> text contains "DISARMED", NOT "ACTIVE" |
| :139 | em-dash, never `0.00%`, when a leg has no baseline; still shows `4%` / `10%` |
| :150 | `innerHTML` contains `"amber"` and NOT `"bg-emerald-500/20"` |
| :157 | `ARMED_AT_HIGH_WATER` -> "ACTIVE", not "DISARMED", `0.00%` shown, `bg-emerald-500/20` present |
| :166 | `LEGACY_NO_MARKER` -> "ACTIVE" (absent key = today's behaviour) |
| :174 | disarmed+paused -> Resume `disabled === true`, `title` matches `/DISARMED/` |
| :185 | Resume tooltip contains `"blocks new orders"`, NOT `"re-anchors them"` / `"next cycle re-anchor"` (36.12 anti-revert pin) |
| :193 | badge tooltip starts with `"DISARMED:"`, contains `"blocks new orders"` + `"baseline_anchor_on_lost_history"`, NOT `"re-anchors them"` / `"Resume is blocked until the next cycle"` |
| :206 | armed+paused+healthy -> Resume enabled |
| :218-258 | the same five shapes against `OpsStatusBar` (incl. the SCOPED `[title^="Daily:"]` em-dash assertion at :231-234) |

**Weakening risk, named:** the assertions at `:136`, `:161`, `:170`, `:222`,
`:241`, `:249` are all `expect(txt).not.toContain("ACTIVE")` /
`.not.toContain("DISARMED")` on the WHOLE container text. If a new state
label contains either substring (e.g. "NOT ACTIVE", "DISARMED (STALE)")
these tests break or, worse, are "fixed" by loosening them. Choose a label
that shares no substring with "ACTIVE"/"DISARMED", and ADD stale-case
fixtures rather than editing the three existing ones -- every existing
fixture is a genuine-absence / healthy / legacy case and all ten assertions
remain correct under the recommended design.

Also note `:150`'s `expect(html).toContain("amber")` is a WEAK assertion
(any amber token anywhere satisfies it). A new amber-adjacent state must not
be allowed to satisfy the DISARMED test by accident -- the new tests should
assert the BADGE TEXT and the specific pill token, not a bare colour
substring.

### H. Project frontend constraints that bind the new badge

From `.claude/rules/frontend.md`:

* Phosphor icons only, imported from `@/lib/icons` -- never
  `@phosphor-icons/react` directly. **No emoji anywhere in the UI.**
* Dark-mode only; navy + slate palette, never Tailwind `zinc`. Colour coding
  convention is stated as "green=bullish, red=bearish, **amber=neutral**,
  gray=error/unavailable".
* Contrast targets on dark navy: `text-slate-300` >= 10:1 for secondary;
  `text-slate-400` is chrome-only, "NOT for risk-relevant numbers".
* Tailwind JIT-safe class strings: no template-built class names; use a
  static literal lookup map (canonical example
  `PortfolioAllocationDonut.tsx::DOT_BG_CLASS`).
* Every component needs error / loading / empty states (both components
  already have them: `KillSwitchPanel.tsx:94-124`, `OpsStatusBar.tsx:307-314`).
* Live-UI verification is mandatory for UI claims (Playwright MCP, never
  code-reading).

From `.claude/rules/frontend-layout.md` §4.5: the ops bar is ONE dense row of
labelled segments; a new state must fit the existing pill footprint
(`px-2 py-0.5 text-[10px]`) and must not add a second row or a new card.
§9 also pins "Pre-attentive attributes for status ... processed in <250ms"
(NNG / Cleveland & McGill) -- relevant to how many distinct status colours
the bar can carry.

### I. The :3100 skip-auth Playwright rig (DO NOT RUN -- documented only)

* Bypass seam: `frontend/src/middleware.ts:28-34` --
  `process.env.LIGHTHOUSE_SKIP_AUTH === "1"` skips the auth wall.
* Config: `frontend/playwright.config.ts:126-138`. The `webServer` set is
  SELECTED by `LIGHTHOUSE_SKIP_AUTH` (:126); the functional server runs
  `npx next dev --port 3100` (:129) with `url: "http://localhost:3100"`
  (:130), `LIGHTHOUSE_SKIP_AUTH: "1"` (:134) and
  **`PLAYWRIGHT_DIST_DIR: ".next-functional"` (:138)** so the :3100 server
  "never shares `.next` with the operator's :3000" (:137). The `functional`
  project (:93-100) has `baseURL: "http://localhost:3100"` and exists ONLY
  when `LIGHTHOUSE_SKIP_AUTH` is set (:86-93).
* Invocation (from `scripts/away_ops/prompt_pm.md:43`):
  `LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line`
  run from `frontend/`.
* The alternative capture path used by the Q/A live-UI gate is the manual
  one in `.claude/rules/frontend.md` ("Live-UI verification"):
  `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100`, capture
  with `mcp__playwright__browser_*`, then kill :3100 and verify :3000 still
  302s to `/login`.
* Standing hazard (auto-memory, 2026-07-17): a second `next dev` that shares
  `.next` or lets Playwright manage :3000 WILL break the operator's :3000.
  `PLAYWRIGHT_DIST_DIR=.next-functional` is the isolation that prevents it.
  NOT RUN in this session, per the caller's hard constraint.

### J. SECOND DEFECT FOUND, same components, same window (P1-grade)

The em-dash guard does **not** cover the stale case, so the cockpit currently
renders a **fabricated 0.00% for a leg that cannot fire**.

* `kill_switch.py:828-832`: `daily_loss_pct = 0.0` is the initialiser, and the
  computation is skipped `if not daily_leg_unevaluable` -- and
  `daily_leg_unevaluable = daily_baseline_missing or daily_baseline_stale`
  (:781). So on a **stale** anchor the payload carries
  `daily_loss_pct: 0.0` with `daily_baseline_missing: false`.
* `KillSwitchPanel.tsx:189` branches the em-dash on
  `breach.daily_baseline_missing` **only** -- false while stale -- so it prints
  `daily 0.00% / 4%`.
* `OpsStatusBar.tsx:320` has the identical bug (`0.0%`).

That is verbatim the failure this component's own comment
(`KillSwitchPanel.tsx:185-187`) claims to have eliminated: "an em-dash, never
`0.00%`, when the leg has no baseline -- 0.00% is a legitimate healthy reading
and must not be shown for a leg that cannot be measured." Today, every UTC
morning, the panel shows a green-looking 0.00% daily drawdown for a leg that
is switched off. **This is arguably worse than the badge bug** (the badge at
least errs loud; this errs reassuring) and it must be fixed in the same step.
The correct guard in both files is
`daily_baseline_missing || daily_baseline_stale`.

## Consensus vs debate (external)

**Strong consensus (5 independent sources, 3 domains):**

* "No required action -> not an alarm" -- ISA-18.2 (#6), Google SRE (#4),
  AWS (#7), AHRQ (#3), Hexagon (#5). Industrial process control, SRE,
  cloud-vendor guidance and clinical safety converge on the identical rule.
  This is cross-domain triangulation, not one field's folklore.
* Non-actionable alarms cause real alarms to be missed -- measured in the
  clinical literature (#3, 72-99% false, 566 deaths), asserted from operational
  experience in SRE (#4) and vendor frameworks (#7).
* Never colour alone -- W3C (#1, Level A) and Red Hat (#8), plus Carbon
  (snippet).

**Genuine debate -- what to DO about it:**

* **"Reclassify / demote"** camp: ISA-18.2 says nuisance alarms "can be
  reclassified as recordable events" (#6). This *supports* a non-alarm state.
* **"Eliminate, don't expand"** camp: Hexagon (#5) is explicit that the fix is
  removing bad actors and that no suppressed alarms should remain; Google SRE
  (#4) says keep the rule set "as simple, predictable, and reliable as
  possible". Read strictly, this camp opposes ANY new state and would prefer
  the condition disappear from the annunciator.
* **Not a real contradiction on inspection:** #5's target is *suppression*
  (hiding a condition that is still real) and #4's target is the *number of
  paging rules*, not the number of display states. Neither addresses a case
  where the underlying condition has partial-but-nonzero safety significance.
  #3's remedy list -- "adjust thresholds and discontinu[e] unnecessary
  monitoring rather than adding new alerts" -- is the closest true dissent, and
  it is the reason the recommendation below deliberately does NOT add a new
  alarm, only a new *non-alarm* status value.

## Pitfalls (from literature)

1. **Fixing the display while leaving the alarm classification intact.** AHRQ
   (#3) reports staff responding to nuisance alarms by disabling them; a
   recolour that still reads as "something is wrong every morning" trains the
   same reflex more slowly.
2. **Suppression instead of reclassification.** Hexagon (#5): "there should be
   no suppressed alarms left." Do not implement this as "hide the DISARMED
   badge before 09:00" -- a time-window suppression would also hide a genuine
   lost-baselines event in the same window.
3. **Adding a colour without adding a word.** WCAG 1.4.1 is Level A (#1);
   Red Hat (#8) requires visible or visually-hidden text. A new hex value with
   the same label is a conformance regression, not a fix.
4. **Solving it in the alerting rule rather than at the source.** Google SRE
   (#4): pages requiring a "robotic response" indicate a systemic problem to be
   escalated, not papered over.
5. **Conflating temporary with durable.** Kubernetes (#2) is a boxed Caution
   for a reason: treating a self-clearing state as a hard fault produces
   "cascading failures". The pyfinagent analogue is a daily P1-flavoured
   badge that an operator eventually stops reading.

## Application to pyfinagent

| External finding | pyfinagent anchor |
|---|---|
| Alarm = "requiring a response" (#6) | `paper_trading.py:624` already states "NO operator action is required" for this exact state -- the backend has already rationalized it; only the UI has not |
| Readiness vs liveness (#2) | `daily_baseline_stale` (self-clearing at the SOD roll) vs `daily_baseline_missing` / `trailing_baseline_missing` (durable, operator-repaired). Discriminators already served at `kill_switch.py:848-851`, both return shapes |
| Reclassify, don't suppress (#5, #6) | Keep the state VISIBLE and named; just stop painting it as an alarm. Do not add a time-window hide |
| Colour + word + shape (#1, #8) | New label token + `IconInfo` (already exported from `@/lib/icons`, used at `OpsStatusBar.tsx:5`) instead of `IconWarning weight="fill"` |
| Alert-fatigue harm is cumulative (#3, #4, #7) | The window recurs EVERY UTC day, unconditionally -- the highest-frequency possible nuisance class |
| "Unknown is not healthy" | `kill_switch.py:809-816` -- forbids the naive "just render ACTIVE" fix |

## RECOMMENDATION

**Adopt a fourth badge VALUE that is explicitly a STATUS INDICATION, not an
alarm -- derived client-side from keys the backend already serves. Do NOT add
a backend key, do NOT make `armed` tri-valued, and do NOT enable Resume.**

**R1 -- derive the state, with the `nav_invalid` guard.** In both components:

```ts
// keys to ADD to the local breach type: daily_baseline_stale?, nav_invalid?,
// nav_invalid_disarmed?  (all already served -- kill_switch.py:808/:817/:849)
const navUnmeasured =
  breach.nav_invalid === true || breach.nav_invalid_disarmed === true;
const dailyLegStaleOnly =
  breach.armed === false &&
  breach.daily_baseline_stale === true &&
  breach.daily_baseline_missing === false &&
  breach.trailing_baseline_missing === false &&
  !navUnmeasured;
const disarmed = breach.armed === false && !dailyLegStaleOnly;
```

* **The `navUnmeasured` guard is NOT optional and is a REAL reachable path.**
  `evaluate_breach`'s `nav_invalid` early return (`kill_switch.py:803-826`)
  passes through the *computed* `daily_baseline_stale` (:817) while both
  `*_missing` flags can be false. `GET /kill-switch` falls back to
  `... or 0.0` on a 5s BQ timeout (`paper_trading.py:516`), so a BQ timeout
  inside the stale window yields exactly `armed:false, stale:true,
  missing:false, missing:false` -- and without this guard the cockpit would
  render the calm new state while **nothing at all** could be measured. That is
  the phase-36.7 defect re-created a third time. Mutation-test it.
* Strict `=== true` / `=== false` throughout (phase-80.36 "discriminate on
  presence, never on value"). An older backend that omits
  `daily_baseline_stale` yields `dailyLegStaleOnly === false` -> DISARMED, i.e.
  today's louder behaviour, which is the correct fail direction.
* This satisfies the caller's CRITICAL TRAP: `armed` stays a strict boolean, so
  `paper_trading.py:630` and `kill_switch.py:994` (`.get("armed", True)`,
  fail-OPEN) are untouched and cannot be silently opened.

**R2 -- label.** `PARTIAL` (or `PARTIAL COVER`). It must share no substring
with `ACTIVE` or `DISARMED`, because six existing assertions are whole-container
`not.toContain("ACTIVE")` / `not.toContain("DISARMED")` (test `:136, :161,
:170, :222, :241, :249`). `RE-ANCHORING` is substring-safe but I advise against
it: phase-36.12 deliberately banned "wait for the automatic re-anchor" phrasing
in this component, and a label that *names* the re-anchor invites the banned
sentence back into the tooltip.

**R3 -- colour: sky, not amber.** Amber is already DISARMED's token in both
files (`KillSwitchPanel.tsx:167`, `OpsStatusBar.tsx:346`) AND the project's
"degraded / unknown / stale-poll" colour (`OpsStatusBar.tsx:188-189`,
`CycleSegment:420-426`). Reusing amber would collapse the exact distinction
this step exists to draw. Sky is the project's established
informational / in-progress token (active tab `bg-sky-500/10 text-sky-400`,
pulsing step dot `bg-sky-400`, confirm button `bg-sky-600`). Use
`bg-sky-500/20 text-sky-300` in `KillSwitchPanel` and `bg-sky-500/15
text-sky-300` in `OpsStatusBar` -- matching each file's existing opacity
convention (`/20` vs `/15`). Literal class strings only (Tailwind JIT rule).

**R4 -- de-alarm it.** Remove the new state from the `alarm` rollup
(`KillSwitchPanel.tsx:138`, `OpsStatusBar.tsx:319`) so: the panel card keeps
`border-navy-700 bg-navy-800/60` (no coloured card), `IconWarning` keeps
`weight="regular"`, and the icon becomes `IconInfo` in this state. That is the
ISA-18.2 alarm->status demotion made visible. Note `KillSwitchPanel.tsx:153-157`
currently keys the icon on `alarm` while the badge keys on `disarmed`, so a
disarmed card today shows an amber container with a **rose** filled icon -- fix
that inconsistency in the same pass.

**R5 -- fix the fabricated 0.00% (section J).** Change the em-dash guard in
both files to `daily_baseline_missing || daily_baseline_stale`
(`KillSwitchPanel.tsx:189`, `OpsStatusBar.tsx:320`). Independent of the badge
work, this is the more dangerous of the two bugs.

**R6 -- Resume stays DISABLED; only the REASON changes.** Section F: the server
raises **409** on a stale anchor (`paper_trading.py:612-629`), so an enabled
button would produce a click -> 409 -> `window.alert`. Replace the false
"loss baselines unrestorable" text with the truth, mirroring `:618-628` and
avoiding 36.12's banned phrases: state that the baselines are intact, that the
trailing leg is still armed, that the daily leg awaits today's anchor, and that
no operator action is required. Also **add a `title` to
`OpsStatusBar.tsx:363-373`**, which today disables Resume with no reason
exposed at all. (In practice the button is only rendered when `paused === true`
-- `KillSwitchPanel.tsx:213`, `OpsStatusBar.tsx:363` -- and the live book is
`paused:false`, so the badge is the visible harm; fix both anyway.)

**R7 -- accessibility.** Distinct word + distinct icon + distinct colour = three
channels, so WCAG 1.4.1 (Level A) holds. Keep the tooltip as `title=`
(consistent with the file's existing pattern and the operator-only exemption
noted at `OpsStatusBar.tsx:267-270`). Optional, soft: `role="status"` on the
badge span so the transition is announced politely; do NOT use `role="alert"`
(assertive) -- this is by construction not an alert.

**R8 -- tests: extend, never weaken.** Add a `DAILY_STALE` fixture
(`sod_nav: 23838.19, sod_date: "2026-07-25", peak_nav: 24666.57, armed:false,
daily_baseline_stale:true, daily_baseline_missing:false,
trailing_baseline_missing:false`) plus a `STALE_AND_NAV_INVALID` fixture. Keep
all ten existing assertions unchanged -- every current fixture is a genuine
absence / healthy / legacy case and they all stay correct. New assertions
must include: the stale fixture renders neither "ACTIVE" nor "DISARMED";
renders an em-dash and **not** `0.00%` (this one FAILS today -- it is the
mutation-proof pin for R5); the nav-invalid+stale fixture renders **DISARMED**,
not the new state; and the badge/Resume tooltips contain the truthful text and
not "unrestorable" / "could not be restored". Do not assert on a bare colour
substring -- the existing `expect(html).toContain("amber")` at `:150` is a weak
assertion and a sky-token state must not be able to satisfy a DISARMED test by
accident; assert badge TEXT plus the exact pill token.

### Strongest counter-argument to R1-R8 (and why I still recommend them)

**The counter (finding #5, three sources):** ISA-18.2, Hexagon/ALI and Google
SRE all prescribe *deletion*, not *expansion*. Applied here, the "purest"
reading says: during the window the trailing leg is still armed and the book
is still protected against a 10% drawdown, so just render **ACTIVE**, fix the
readout to an em-dash, and add a tooltip. Zero new states, zero new colours,
zero new tests, and it satisfies "do not annunciate a non-actionable
self-clearing condition". A second, weaker counter: `frontend-layout.md` §9
pins pre-attentive status processing (<250 ms), and a fourth colour in a bar
that already carries emerald/amber/rose across Gate, Kill and Cycle segments
measurably costs glanceability.

**Why I reject it:** rendering ACTIVE would assert full coverage while the
daily leg genuinely cannot fire -- a 4% intraday loss would not pause the book
-- which is the phase-36.7 defect in mirror image and violates the rule
`kill_switch.py:809-816` states in its own words: "A leg that cannot measure
cannot fire, so it must not claim it can: unknown is not healthy." ISA-18.2's
reclassify-as-event disposition presupposes the condition has *no* safety
significance; this one has *partial* significance for several hours a day.
The standard's own alarm/message split -- alarms differ from events "by their
urgency requirement" -- is precisely a demotion to a non-urgent status
indication, which is what R1-R8 do. Hexagon's target is *suppression* (hiding
a live condition), and this proposal suppresses nothing. On the glanceability
cost: the Kill segment's state count goes 3 -> 4, but its **alarm** count goes
2 -> 1 (only PAUSED and DISARMED stay alarm-coloured), which is a net
reduction in red/amber pixels on a normal day -- the direction all five sources
endorse.

**What would change my recommendation:** if the contract's real goal is that
Resume work during the window, then the load-bearing change is
`paper_trading.py:612` (a money-path 409), not the badge -- and that belongs in
its own research-gated step, not a P1 frontend step.

### Out of scope but worth queueing (per `feedback_queue_discovered_defects_in_masterplan`)

* `sod_date` (`paper_trading.py:529`) and `baseline_provenance` (`:535`) are
  served and read by **zero** frontend components. `baseline_provenance ==
  "lost_history_anchor"` is the phase-36.12 marker meaning "ARMED, but the
  drawdown it measures starts from a fiction" -- currently invisible in the UI.
  That is a separate, arguably higher-severity display gap.
* `KillSwitchPanel.tsx:153-157` icon/badge colour inconsistency (see R4) if not
  folded into this step.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
      fetched and read (8 high/medium yield, 1 low-yield disclosed); 3 failed
      attempts disclosed separately
- [x] 10+ unique URLs total -- **40**
- [x] Recency scan (2024-2026) performed + reported -- Q4 (2026) + Q6 (2025) +
      2024/2025 PMC hits; result reported even where null
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module -- both components in
      full, the test in full, `evaluate_breach` + both return shapes, the
      endpoint, both fail-open gates, the `paper_trader` consumers, the
      Playwright rig config, and a complete `grep -rn` census over
      `frontend/src` (plus `backend/`, `scripts/`, MCP servers for the
      backend-side census)
- [x] Contradictions / consensus noted -- "eliminate vs reclassify" debate
      surfaced and adjudicated, not smoothed over
- [x] All claims cited per-claim with URL + access date
- [ ] **Tier length**: the `moderate` tier's <=700-word guide is exceeded. The
      overage is entirely the caller-requested internal enumeration (complete
      consumer census + file:line tables + two newly-found defects). External
      analysis itself is moderate-depth. Disclosed rather than truncated.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 28,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Alarm-management, SRE, cloud-vendor and clinical-safety literature converge: a condition requiring no operator response is not an alarm (ISA-18.2), and recurring non-actionable alarms measurably cause real ones to be ignored (AHRQ: 72-99% false, 566 FDA deaths; Google SRE; AWS OPS08-BP04 rates the risk High). Kubernetes readiness-vs-liveness is the exact architectural analogue for stale (self-clearing) vs missing (durable) baselines. Recommend a fourth badge VALUE classified as status not alarm, derived client-side from already-served keys with a mandatory nav_invalid guard, labelled PARTIAL, coloured sky (amber is taken by DISARMED), de-alarmed from the icon/card rollup, with Resume kept DISABLED because paper_trading.py:612 still 409s. Two additional live defects found: the em-dash guard misses the stale case so the cockpit prints a fabricated 0.00% for a leg that cannot fire, and the Resume/badge tooltips assert 'baselines unrestorable' which the backend's own 409 text refutes.",
  "brief_path": "handoff/current/research_brief_36.20.md",
  "gate_passed": true
}
```
