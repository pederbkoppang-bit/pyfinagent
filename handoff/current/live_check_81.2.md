# Live Check — phase-81.2 — Gate input resolution hardening

Required evidence per `.claude/masterplan.json` 81.2 `verification.live_check`: *"verbatim terminal
output showing the archived-CONDITIONAL fixture returning hold, the resolution-order assertions, and
both mutations going red"*.

## 1. Immutable verification command — exit 0

```
$ source .venv/bin/activate && python -m pytest \
    backend/tests/test_phase_81_2_verdict_resolution.py \
    backend/tests/test_phase_71_3_verdict_gate.py \
    backend/tests/test_phase_38_4_hook_gate.py -q
..........................                                               [100%]
26 passed in 0.06s

$ bash .claude/hooks/lib/harness_log_gate_test.sh
PASS: case 1 -- gate disabled returns proceed
PASS: case 2 -- gate enabled + token present returns passed
PASS: case 3 -- gate enabled + token missing + MODE=block returns skip
PASS: case 3b -- gate enabled + token missing + default mode returns warn
PASS: case 3c -- warn and skip are distinguishable
PASS: case 4 -- missing log file returns proceed (fail-open)
PASS: case 5 -- prefix-match guard (38.6 does not match phase=38.6.1)
ALL PASS

$ python3 scripts/go_live_drills/incident_log_p0_test.py
DRILL PASS: 6/6 incident-log-P0 scenarios verified

FULL COMMAND EXIT: 0
```

## 2. Archived-CONDITIONAL fixture returning HOLD — through the real hook plumbing

A handoff root containing **only** an archived CONDITIONAL, driven through the exact lines
`auto-commit-and-push.sh` runs (`--resolve`, `sed -n 1p/2p`, the `case` dispatch):

```
$ mkdir -p "$S/handoff/archive/phase-80.5" "$S/handoff/current"
$ echo '{"step_id":"80.5","ok":false,"verdict":"CONDITIONAL"}' \
      > "$S/handoff/archive/phase-80.5/evaluator_critique_80.5.json"
$ RAW=$(python3 .claude/hooks/lib/verdict_gate.py --resolve 80.5 "$S/handoff")

=== archived-CONDITIONAL end-to-end (cycle 2) ===
  decision=hold  source=archive:step
  -> archive arm FIRES: NOTE logged + systemMessage emitted
  -> dispatch=HOLD: push held. Disarm-by-move CLOSED.
```

Before 81.2 this same tree produced `no_input` → fail-open → push proceeds. That is the
2026-07-24 failure reproduced and closed.

## 3. Resolution-order assertions

`test_per_step_current_beats_rolling_current_beats_archive` populates all three sources with three
**different** verdicts and walks the chain by unlinking each winner in turn:

| State | source | decision |
|---|---|---|
| all three present | `current:per-step` | `passed` |
| per-step removed | `current:rolling` | `hold` |
| rolling removed too | `archive:step` | `hold` |

Live CLI against the real tree:
```
$ python3 .claude/hooks/lib/verdict_gate.py --resolve 36.7 $PWD/handoff
passed
current:per-step

  step 36.7 -> decision=passed    source=current:per-step
  step 81.2 -> decision=no_input  source=none      (before this verdict was persisted)
  step 99.9 -> decision=no_input  source=none

$ python3 .claude/hooks/lib/verdict_gate.py handoff/current/evaluator_critique_36.7.json 36.7
passed          # legacy 2-arg mode unchanged
```

## 4. Both mutations going red

Run against in-memory copies; the live file is never mutated.

```
BASELINE (unmutated):
  archived-CONDITIONAL -> hold : True
  precedence current-first     : True

MUTATION 1 -- archive branch removed from the candidate chain:
  archived-CONDITIONAL -> hold : False   <-- guard went RED

MUTATION 2 -- resolution order reversed:
  precedence current-first     : False   <-- guard went RED

MUTATION MATRIX: PASS -- both guards can fail

$ git diff --stat .claude/hooks/lib/verdict_gate.py
 1 file changed, 99 insertions(+), 8 deletions(-)      # live file unmutated
```

**Independently reproduced by Q/A cycle 2** with a wider matrix: 14 mutants, 10 killed, run by
patching the module loader so the **shipped** assertions executed against mutants; live-file md5
`23d706819fbc3a84c958088c0beef3f8` identical before and after.

## 5. Dogfood — the repaired gate gating its own step

```
$ python3 .claude/hooks/lib/verdict_gate.py --resolve 81.2 $PWD/handoff
hold
current:per-step
```

With the cycle-1 CONDITIONAL persisted, the gate built by this step correctly **held** this step.
Q/A confirmed this is correct behaviour, not a self-block bug, and that it flips to `passed` once
the cycle-2 PASS verdict is persisted.

## 6. Q/A verdict

Cycle 1 **CONDITIONAL** (two false scope claims in the prose, all 9 criteria met) → remediated →
cycle 2 **PASS**, `violated_criteria: []`, `harness_compliance_ok: true`, `certified_fallback: false`.
Full transcription in `handoff/current/evaluator_critique_81.2.md`.
