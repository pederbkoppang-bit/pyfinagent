# Contract -- step 86.19

**Step**: `86.19` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-10 | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** `git diff` on the target files is empty at this
moment.

---

## 1. Research gate

**PASSED** -- `wf_7cd63030-ead`, tier `simple`, brief
`handoff/current/research_brief_86.19.md` (27,477 chars). Enforced: **8 sources
read in full** (floor 5), **28 URLs** (floor 10), recency scan performed, all 8
claimed URLs present, `urls_collected_corroborated: 28 <= 28`,
`brief_status_in_brief: COMPLETE`, `rail_dropped: null`.

*(Third consecutive live confirmation of the phase-86.37 born-inert marker.)*

### It corrected my framing on impact -- record this first

**The defect is LATENT AND ARMED, NOT FIRING.** No colliding id currently
declares a `live_check`, so all four ids return `proceed` today. My own working
note called the gate "vacated"; that overstates it. The correct statement is
that the gate would decide about **the wrong subject** the moment a colliding id
acquires a live_check -- which is a real hazard and not a live failure.

### Findings that decide the design

**F1 -- THE ORDERING IS AN ACCIDENT OF KEY-INSERTION ORDER.**
`live_check_gate.py:34-48` is depth-first first-match over `node.values()`, so
whichever key was written first in the JSON wins. `archived_legacy_steps`
precedes `steps`, so the ARCHIVED twin (status `pending`) beats the live `done`
step. Nothing chose that; it fell out of serialisation order.

**F2 -- "FIRST MATCH WINS" IS THE ONE BEHAVIOUR NO STANDARD ENDORSES.** RFC 8259
calls duplicate names "unpredictable" and recognises three behaviours
(last-wins, error, collect-all) -- first-match-wins is not among them. W3C XML
makes id uniqueness document-wide. PEP 20: *"In the face of ambiguity, refuse the
temptation to guess."* JSON Schema SHOULD raise. C# CS0121 refuses ambiguous
overloads rather than picking. **So the remedy is not "pick better" -- it is
"stop picking".**

**F3 -- SALTZER APPLIES DIRECTLY.** An exclusion-shaped mechanism fails by
PERMITTING, unnoticed. A gate that silently resolves to the wrong node emits a
confident `proceed` about a subject nobody asked it about.

**F4 -- SCOPING PRESERVES PROVENANCE; RENUMBERING DESTROYS IT.** Kubernetes
namespacing is the reference: disambiguate by scope, not by mutating the record.
The gate explicitly recommends **do NOT renumber the archive**.

**F5 -- A DRIFT SEAM IS ALREADY OPEN.** `preflight_verify_masterplan.py:90`
defines `ARCHIVE_CONTAINERS` and `LIVE_STEP_CONTAINERS = {"steps","subphases"}`,
and `test_phase_75_19_preflight_calibration.py:151-152` already parametrises over
them. A second, independently-written exclusion in the hook would be a third
copy of the same list -- which is how two consumers drift apart.

**F6 -- A LOAD-TIME UNIQUENESS ASSERTION IS CHEAP AND MEASURABLE NOW.**
Per-type uniqueness is green today (**1230 steps / 114 phases, 0 duplicates
within either type**); cross-type fires exactly once (`phase-6.5`). So the
assertion can be adopted per-type immediately and the single cross-type case
handled explicitly rather than by a blanket rule.

## 2. Measurements I took before the gate returned (criterion 1)

| id | archived twin | live twin |
|---|---|---|
| `5.1` | *Market Expansion Framework* (pending) | **Broker Abstraction Layer (done)** |
| `5.2` | *Market-Specific Research & Considerations* | **Data Provider Abstraction Layer (yfinance + EODHD)** |
| `5.3` | *Cross-Market Intelligence* | **Multi-Asset BQ Schema Extension (FX + futures)** |
| `phase-6.5` | a PHASE at `phases[13]` | **a STEP at `phases[12].steps[4]` (done)** |

`1348` id-bearing nodes, `1344` distinct, **4 duplicated** -- matching the step
text exactly. **My first derivation reported ZERO** because it walked only
`phases[].steps[]` and missed `archived_legacy_steps[]`; I was one step from
declaring the step's premise wrong on a defective method.

**Consumer exposure, measured (criterion 4):**

| consumer | exposed? | evidence |
|---|---|---|
| `live_check_gate.py::find_step` | **YES** -- returns the archived twin for all of 5.1/5.2/5.3 and the phase for `phase-6.5` | the subject |
| `auto-commit-and-push.sh::load_done_ids` | **YES, today** -- builds `{id: name}`, 900 done ids, **1 CLOBBERED: `phase-6.5`**. Archived twins are not `done`, so archives do not pollute it; the cross-type collision does | measured |
| `archive-handoff.sh` `phase-<sid>/` | **YES** -- both twins map to one dir; and `phase-6.5` yields `handoff/archive/phase-phase-6.5/`. **`phase-phase-6.1..6.4` already exist on disk** -- the same raw-`$sid` defect the 86.29 gate found. The two steps share a root cause | measured |
| `preflight_verify_masterplan.py` | **NO** -- already excludes archive containers; it is the reference | read |

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. the duplicate set is RE-DERIVED, not inherited from this write-up: enumerate every duplicated id with all of its locations and names. The four measured on 2026-08-09 (phase-6.5, 5.1, 5.2, 5.3) are the starting point, not the assumed total; if the re-derived count differs, the difference is explained
2. the two classes are handled distinctly with the choice justified in writing -- Class A phase-id-equals-step-id (phase-6.5), Class B live-versus-archived within one phase (5.x). A single blanket renumber that ignores the distinction is out of contract
3. find_step('5.2') resolves to the LIVE step ('Data Provider Abstraction Layer (yfinance + EODHD)'), demonstrated by the verification command's before-and-after output rather than by reasoning about the walk order
4. every id-keyed consumer is enumerated and each is stated as either fixed or deliberately unaffected, with the reason. At minimum: live_check_gate.py::find_step, archive-handoff.sh's handoff/archive/phase-<sid>/ directory naming, and auto-commit-and-push.sh::load_done_ids. scripts/meta/preflight_verify_masterplan.py:90 is the reference exclusion
5. the remedy is MUTATION-TESTED with output recorded: prove it detects a deliberately reintroduced duplicate, and prove that a step declaring a live_check on live 5.2 IS actually gated. Build that declaration in a SCRATCH COPY of the masterplan -- do not edit the real 5.2 to run the test
6. no done step's id is renumbered without first listing what references it on disk. handoff/archive/phase-5.1/ and handoff/archive/phase-6.5/ exist and are keyed by id; if a renumber is chosen, the references are updated in the SAME commit and enumerated in the live_check
7. the masterplan's semantic content is otherwise unchanged: the multiset of (id, name, status) before and after this step's own work is identical apart from the deliberate id changes, printed as a set-difference in both directions
8. if the conclusion is that excluding archive containers in the consumers is sufficient and NO id is changed, the step closes with that stated plainly and the reasoning recorded -- that is a valid and probably preferable outcome

**Verification command** (immutable): the one-liner printing
`find_step('5.2')['name']` -- read-only over `masterplan.json`, touches no live
state, so it is safe to run inside the cycle-observation window.

## 4. Plan -- take criterion 8's outcome, with the gate's three refinements

**NO ID IS RENUMBERED.** Criterion 8 pre-blesses this and F4 supports it:
renumbering mutates a historical record and would require rewriting
`handoff/archive/phase-5.1/` and `phase-6.5/` on disk (criterion 6). Scoping the
resolver preserves provenance.

**P1 -- SCOPE the walk to live containers.** Import `LIVE_STEP_CONTAINERS` /
`ARCHIVE_CONTAINERS` from the existing reference rather than re-declaring them
(F5), so the two consumers cannot drift. This alone fixes **Class B**, and --
because a top-level phase is not inside a live-step container -- **Class A** too.
Verified in advance: the rule resolves all four to the live node.

**P2 -- REFUSE, do not pick (F2/F3).** `find_step` stops returning the first
match. It collects matches within scope; **>1 match returns a distinct AMBIGUOUS
signal, never a node.** `gate_decision` maps ambiguity to the **HOLD** side, the
same as a missing artifact -- an ambiguous lookup must never yield `proceed`.

**P3 -- LOAD-TIME UNIQUENESS ASSERTION (F6)**, per-type, which is green today
(1230/114, 0 duplicates). The single cross-type case is handled by P1's scoping
and stated explicitly rather than suppressed.

**P4 -- state each consumer fixed-or-unaffected** (criterion 4), including that
`archive-handoff.sh` is **NOT fixed here** and why: its `phase-phase-*` defect is
86.29's, the fossil dirs prove it, and splitting one root cause across two steps
mid-flight is worse than naming it.

**P5 -- mutation (criterion 5)**, in a SCRATCH COPY: reintroduce a duplicate and
prove detection; and declare a `live_check` on live `5.2` in the scratch copy and
prove it IS gated. **Never edit the real 5.2.**

**P6 -- criterion 7 set-difference** of the `(id, name, status)` multiset before
and after, printed both directions. Expected: **empty**, since no id changes.

### Explicitly NOT doing

- **Not** renumbering any id, and **not** touching `archived_legacy_steps`.
- **Not** editing the real `5.2` to run a test (criterion 5 forbids it).
- **Not** fixing `archive-handoff.sh`'s `phase-phase-*` naming -- that is 86.29.
- **Not** making the hook fail-closed in general; only the AMBIGUOUS case holds,
  and the hook's existing fail-open-on-error discipline is preserved.

### Risk

`live_check_gate.py` is live auto-commit infrastructure. Every change must keep
the fail-open-on-internal-error behaviour, or a bug here could brick the commit
path. The ambiguity signal is the one deliberate hold, and it is reachable only
when two live nodes share an id -- which, per F6, is **zero** cases today.

## 5. References

- `handoff/current/research_brief_86.19.md` (gate PASSED, `wf_7cd63030-ead`)
- RFC 8259 §4; W3C XML id uniqueness; PEP 20; JSON Schema core; C# CS0121;
  Saltzer & Schroeder; Kubernetes namespaces
- `scripts/meta/preflight_verify_masterplan.py:90` (the reference exclusion)
