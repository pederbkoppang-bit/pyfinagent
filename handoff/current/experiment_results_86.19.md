# Experiment results -- step 86.19

**Step**: `86.19` (phase-86, P2) | **Phase**: GENERATE (PARTIAL) | 2026-08-10

> **INCOMPLETE BY DESIGN, AND SAID UP FRONT.** Criterion 5 (mutation testing)
> and the Q/A are BOTH DEFERRED: the 20:00 CEST book cycle's freeze window began
> at 19:30 and bars running a mutation harness. Criteria 1,2,3,4,6,7,8 are done
> and evidenced below. **This step is NOT claiming completion.**

## 0. Outcome: criterion 8's path -- NO id is renumbered

Criterion 8 pre-blesses "excluding archive containers in the consumers is
sufficient and NO id is changed... a valid and probably preferable outcome". The
evidence points there and the gate agrees ("Do NOT renumber the archive"):
renumbering mutates a historical record and would force rewriting
`handoff/archive/phase-5.1/` and `phase-6.5/` on disk (criterion 6).

## 1. Criterion 3 -- BEFORE and AFTER, from the immutable command

```
BEFORE: find_step(5.2) -> Market-Specific Research & Considerations     [ARCHIVED twin]
AFTER : find_step(5.2) -> Data Provider Abstraction Layer (yfinance + EODHD)   [LIVE]
```

## 2. Criterion 1 -- the duplicate set, RE-DERIVED

**1348 id-bearing nodes, 1344 distinct, 4 duplicated** -- matching the step text.

| id | archived / other twin | live twin (now returned) |
|---|---|---|
| `5.1` | *Market Expansion Framework* (pending) | **Broker Abstraction Layer** (done) |
| `5.2` | *Market-Specific Research & Considerations* | **Data Provider Abstraction Layer** |
| `5.3` | *Cross-Market Intelligence* | **Multi-Asset BQ Schema Extension** |
| `phase-6.5` | a **PHASE** at `phases[13]` | a **STEP** at `phases[12].steps[4]` |

**MY FIRST DERIVATION REPORTED ZERO.** It walked only `phases[].steps[]` and
missed `archived_legacy_steps[]`. I was one step from writing "the step's premise
is wrong" on a defective method -- the recurring failure of the day, caught here
by widening the walk before drawing a conclusion.

## 3. Criterion 2 -- the two classes, handled distinctly

- **Class B (5.x)** -- live-vs-archived inside one phase. Fixed by scoping the
  walk to LIVE step containers.
- **Class A (`phase-6.5`)** -- cross-type: a PHASE and a STEP share an id. **No
  archive exclusion can fix this**, which is exactly why the classes must be
  named separately. It is fixed by the *same* rule for a different reason: a
  top-level phase is reached via the `phases` key, which is not a live-step
  container, so a phase can never match.

One rule, two justifications -- not a blanket renumber.

## 4. Criterion 4 -- every id-keyed consumer

| consumer | status | reason |
|---|---|---|
| `live_check_gate.py::find_step` | **FIXED** | the subject; scoped + refuses ambiguity |
| `auto-commit-and-push.sh::load_done_ids` | **NOT fixed -- disclosed** | it builds `{id: name}` over the whole tree; 900 done ids with **1 CLOBBERED (`phase-6.5`)** today. Archived twins are not `done`, so archives do not pollute it -- only the cross-type collision does. Out of this step's named scope and it is a *reporting* dict, not a gate; queued rather than silently fixed |
| `archive-handoff.sh` `phase-<sid>/` | **NOT fixed -- belongs to 86.29** | both twins map to one dir, and `phase-6.5` yields `handoff/archive/phase-phase-6.5/`. **`phase-phase-6.1..6.4` already exist on disk** -- the same raw-`$sid` defect 86.29's gate found. Splitting one root cause across two in-flight steps is worse than naming it |
| `preflight_verify_masterplan.py` | **UNAFFECTED -- it is the reference** | already excludes archives; defines `LIVE_STEP_CONTAINERS` |

## 5. The design, and the two refinements the gate added

**P1 scope** -- walk only `{"steps","subphases"}`. A positive allowlist, not an
archive denylist: a denylist needs updating whenever a new archive container is
invented.

**P2 REFUSE, do not pick.** "First match wins" is the one behaviour no standard
endorses (RFC 8259 recognises last-wins / error / collect-all; W3C XML makes id
uniqueness document-wide; PEP 20; JSON Schema SHOULD raise; C# CS0121).
`resolve_step` returns a distinct `AMBIGUOUS` token and `gate_decision` maps it
to **HOLD** -- the same side as a missing artifact. A gate that cannot identify
its subject must not wave a commit through.

**F5 drift** -- the container set is **deliberately duplicated** from the
reference rather than imported: this is a hook library on the auto-commit path
whose contract is never to raise, and reaching outside `.claude/` for an import
would add a failure mode to the component whose job is not to have one. The
drift risk is closed by an assertion instead:

```
hook LIVE_STEP_CONTAINERS      = ['steps', 'subphases']
reference LIVE_STEP_CONTAINERS = ['steps', 'subphases']
EQUAL: True
```

## 6. Verbatim

```
all four resolve to the LIVE node:
  5.1 -> done    'Broker Abstraction Layer'
  5.2 -> pending 'Data Provider Abstraction Layer (yfinance + EODHD)'
  5.3 -> pending 'Multi-Asset BQ Schema Extension (FX + futures)'
  phase-6.5 -> done 'Sentiment scorer ladder ...'      <- the STEP, not the phase

live matches per id: {'5.1':1, '5.2':1, '5.3':1, 'phase-6.5':1}

ambiguity refused (synthetic fixture, masterplan untouched):
  resolve_step(two live 9.9) -> 'ambiguous'
  find_step  (two live 9.9) -> None

criterion 7 -- (id,name,status) multiset, before 1348 / after 1348:
  in AFTER not BEFORE : none
  in BEFORE not AFTER : none
```

## 7. Scope honesty

- **Criterion 5 is NOT satisfied.** Mutation testing is deferred past the cycle
  freeze. The step cannot close until it is done.
- **The defect is LATENT, not firing.** The gate corrected my framing: no
  colliding id currently declares a `live_check`, so all four returned `proceed`
  either way today. My working note said the gate was "vacated" -- that
  overstated it, and the accurate statement is that it would have decided about
  the *wrong subject* the moment a colliding id acquired one.
- **The ambiguity HOLD is unreachable today** -- per-type uniqueness is green
  (1230 steps / 114 phases, 0 duplicates within either type), so it holds nothing
  that currently exists.
- `live_check_gate.py` is live auto-commit infrastructure; the fail-open-on-error
  discipline is preserved and only the AMBIGUOUS case deliberately holds.
