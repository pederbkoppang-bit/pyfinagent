# experiment_results — step 86.84

Filed to close cycle-2 Q/A finding **V-9**: the five-file protocol's GENERATE
artifact was absent for this step, and the rolling
`handoff/current/experiment_results.md` belongs to phase-82.6 (2026-08-06).
Written after the fact and labelled as such rather than back-dated.

## What was built or changed

| File | Change |
|---|---|
| `.claude/agents/qa.md` | **`maxTurns: 30` REMOVED.** Body byte-identical (45,398 == 45,398); only the pin line and its rationale comment changed. |
| `.claude/agents/researcher.md` | **`maxTurns: 40` REMOVED.** Only the pin line and its comment changed. |
| `scripts/qa/rail_turn_cap.py` | **NEW.** Measures whether the drop is turn exhaustion; scores each run against the cap in force when it ran; checks the remediation; four controls; cardinality floors with no opt-out. |
| `scripts/qa/mutate_rail_turn_cap.py` | **NEW.** Executable 15-cell mutation matrix for the above (V-1). |
| `scripts/qa/rail_drop_rate.py` | Header superseded; **runtime output** now prints the confound caveat under the by-model table (V-4). |
| `.claude/workflows/qa-verdict.js` | Comment-only. "Mechanism is UNPROVEN" superseded; the `2e-4 … the model split SURVIVES` claim retracted in place (V-4); "REPORTED BELOW" → "REPORTED ABOVE". |
| `.claude/workflows/research-gate.js` | Comment-only. Same supersession, plus the note that two at-40 non-emitters sat inside runs this caller recorded as completed. |
| `handoff/current/live_check_86.84.md` | Regenerated (V-2). |
| `handoff/current/contract_86.84.md` | PLAN, written after the gate passed. |
| `handoff/current/evaluator_critique_86.84.md` | Both verdicts transcribed verbatim. |
| `handoff/harness_log.md` | Cycle 218 + operator review request. |
| `.claude/masterplan.json` | Steps 86.84 and 86.85 filed; 86.84 `audit_basis` corrected (V-3). |

No production code, no threshold, no gate, no `.env`, no flag. Both JS diffs are
comment-only.

## Verification command output, verbatim

```
$ python3 scripts/qa/rail_turn_cap.py --verify   # exit 0

  agentType           cap     n  drop  @cap  >cap  ok p50  ok max  ok@cap
  Explore               -   263     0     0     0       7      56       0
  None                  -   414     0     0     0       9      93       0
  claude-code-guide     -     1     0     0     0       7       7       0
  general-purpose       -   252     0     0     0      12      63       0
  qa                   30   302    39    39     0      18      30       6
  researcher           40    93     9     9     0      24      40       3

Turn counts observed on dropped spawns, per role:
  qa                 cap=30  observed=[30]
  researcher         cap=40  observed=[40]

CONTROLS
  C1 turn counter alive     : 1325 spawns with turns>0; 0 zero-with-assistant-lines (must be 0)
  C2 cap is a real ceiling  : 0 capped spawns exceed their cap (must be 0)
  detector positive control : StructuredOutput emitted by 1257/1267 completed spawns vs 1/48 dropped
  C3 negative control       : 10 spawns in `killed` runs sit at turns [1, 1, 2, 2, 2, 3, 4, 5, 6, 16]; 0 at a cap

CLAIM
  every dropped spawn sits EXACTLY at its cap : True
  drops among UNCAPPED agent types           : 0/930 (uncapped spawns reach 93 turns)
  ... of which AT RISK (>30 turns)        : 0/50  vs a 12.2% drop rate on capped spawns

  spawns sitting AT a cap                    : 57, of which 49 never emitted StructuredOutput
  ... inside runs that COMPLETED anyway      : 2 ['wf_078f4125-57a', 'wf_a6ea31e7-9b9']

REMEDIATION (phase-86.84) -- checked by the same command as the diagnosis
  caps removed at : 2026-08-15T00:00:00Z
  in force before : {'qa': 30, 'researcher': 40}  (phase-59.1 pins)
  live now        : {'qa': None, 'researcher': None}
  all pins removed: True  (must be True)

VERIFY: PASS -- controls green, turn-exhaustion claim holds.
```

```
$ python3 scripts/qa/mutate_rail_turn_cap.py --verify   # exit 0
  control verify_ok=True  live_caps={'qa': None, 'researcher': None}
  ... 22 cells; M4r M5r M9 M8 M7c M7b M7 M15 M16 M17 M18 M19 M20
  ...           M11 M11b M12 M12b M13 M21 KILLED
  ...           M14 (equivalent), M6, M6b (known gap) SURVIVED
BYTE-IDENTICAL RESTORE (md5 before == after, real tree never written): all ok
cells=22  real survivors=0  known/equivalent survivors=3
VERIFY: PASS -- control green, 0 real survivors, tree unchanged.
```

Supporting gates, run and green:

```
$ node --check .claude/workflows/qa-verdict.js         -> 0
$ node --check .claude/workflows/research-gate.js      -> 0
$ node scripts/qa/verify_research_gate_workflow.mjs    -> 0   124 passed, 0 failed
$ node scripts/qa/verify_rail_retry.mjs                -> 0    38 passed, 0 failed
$ uvx ruff check --select F821,F401,F811 scripts/qa/rail_turn_cap.py \
        scripts/qa/rail_drop_rate.py                   -> All checks passed!
```

`verify_rail_retry.mjs` section **[F]** is the executed evidence for criterion 4:
an exhausted retry yields no value, rethrows the original error, `research-gate`
still RECOMPUTES `gate_passed`, and the retry loop assigns no verdict field. It
was green before this step and remains green; it is cited here because the
cycle-2 Q/A found the property proven but unmapped to the criterion (V-10).

## Artifact shape

- `rail_turn_cap.py --verify` → exit 0/1, human table; `--json` → machine summary
  carrying `by_agent_type`, `model_x_agent_type`, `controls`, `claim`,
  `remediation`.
- `mutate_rail_turn_cap.py --verify` → exit 0/1, per-cell matrix plus an md5
  before/after restore proof.

## Honest limits of this GENERATE

- **The change is not in force.** The Agent-tool roster snapshots at session
  start, so the removal takes effect next session. Every number above was
  produced under the OLD caps; none of it is evidence about the uncapped rail.
- **The uncapped qa/researcher turn distribution is unobserved.** "Uncapped
  agents self-terminate" rests on 930 spawns of *other* roles. Re-measuring after
  the next session is the committed follow-up, and it is what would turn this
  from a reasoned fix into a verified one.
- **Ruff was red on this step's own file** (`F401 Counter` unused) from cycle 1
  until it was fixed here; recorded rather than quietly cleaned.
