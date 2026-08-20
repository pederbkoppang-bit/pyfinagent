# Live check -- step 90.1

Everything below is verbatim terminal output captured 2026-08-20 against the REAL
repository, the REAL ledger and the REAL run records -- not fixtures.

## 1. Backfill over the real ledger, with per-outcome counts

```
$ python3 scripts/harness/attempt_outcomes.py --backfill --dry-run
{
  "attempt_rows": 89,
  "dry_run": true,
  "ledger": "/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/attempt_budget_audit.jsonl",
  "outcome_counts": {
    "CONDITIONAL": 45,
    "FAIL": 10,
    "NO_VERDICT": 18,
    "PASS": 11,
    "UNKNOWN": 5
  },
  "reason_counts": {
    "completed_without_result": 2,
    "graded": 66,
    "no_run_record": 5,
    "not_an_evaluation": 16
  },
  "rows_total": 93,
  "tolerance_s": 30
}

UNKNOWN = 5 (used ONLY where no run record matched; an ambiguous match also resolves UNKNOWN and is never guessed)
```

Re-runnable and idempotent: the line above is a DRY RUN taken AFTER the real backfill
had already been applied, and it reproduces the same counts.

UNKNOWN = 5, and all five are the synthetic `999.2` pipetest rows, which have no run
record. Note `rows_total` is 93 (89 attempt + 4 operator_extension) where the criterion
says 92: the gate is live and kept recording between filing and execution -- one of the
new rows is this step own research gate.

## 2. The sha256 pair -- a non-exhaustion denial leaves a real exhaustion record intact

Claim `86.118.1` (the exact bypass criterion 4 names), driven through the real hook at
the real path. All four pre-existing exhaustion escalations were in the blast zone;
`86.85` is hand-authored by the operator.

```
--- sha256 BEFORE ---
9180bf317cec3aac24c825c5288f2c38514e358c2d29f4bd6f431518508a3e05  handoff/current/escalation_attempt_budget_75.11.4.md
670edd040b83e22603012ea12471469c1d7ac3b6b4e91a58a1e82811f53858bb  handoff/current/escalation_attempt_budget_86.47.md
1d8a53e58131e2a20eff9dcf04f1b816b7c1e053155d0dc967217d052c1779ab  handoff/current/escalation_attempt_budget_86.85.md
6fbbec66810478aaae8bb6d980c72894fb9cc3bf47fcd2cb8bd047aa06879610  handoff/current/escalation_attempt_budget_999.2.md

--- the denial ---
[attempt-gate] DENIED: this launch claims step_id '86.118.1', which is not a step in .claude/masterplan.json. A claimed step must be a real one -- an unrecognised id would otherwise get a fresh, private attempt allowance. Fix the id, or omit step_id entirely if this launch is not a step attempt (that path is still allowed and uncounted). This launch was stopped BEFORE any tokens were spent, and a denial is NOT a verdict. Written to handoff/current/escalation_unknown_step_id_86.118.1.md
EXIT: 2

--- sha256 AFTER ---
9180bf317cec3aac24c825c5288f2c38514e358c2d29f4bd6f431518508a3e05  handoff/current/escalation_attempt_budget_75.11.4.md
670edd040b83e22603012ea12471469c1d7ac3b6b4e91a58a1e82811f53858bb  handoff/current/escalation_attempt_budget_86.47.md
1d8a53e58131e2a20eff9dcf04f1b816b7c1e053155d0dc967217d052c1779ab  handoff/current/escalation_attempt_budget_86.85.md
6fbbec66810478aaae8bb6d980c72894fb9cc3bf47fcd2cb8bd047aa06879610  handoff/current/escalation_attempt_budget_999.2.md
```

**ALL FOUR BYTE-IDENTICAL.** The denial wrote its own reason-named artifact instead:
`handoff/current/escalation_unknown_step_id_86.118.1.md`.

## 3. The four step-id cells, against the real module

```
  86.118       -> ADMITTED  (extract_step_id='86.118')
  86.118.1     -> DENIED    (extract_step_id=None)
  86.1180      -> DENIED    (extract_step_id=None)
  999.99       -> DENIED    (extract_step_id=None)
```

## 4. The immutable verification command

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
(exit code 0)
```

Tail, verbatim:

```
                 -> ONE attempt over DEFAULT_MAX_TOKENS is DENIED on the token ceiling (c3)
  KILLED    M8   criterion 1: the additive-only guard is removed, so the backfill may silently rewrite an existing field
                 -> --backfill leaves every ORIGINAL field byte-identical (append-only enrichment, never a rewrite)
  KILLED    M9   criterion 1: an ambiguous or absent match is GUESSED instead of resolving UNKNOWN
                 -> an at-ceiling launch is DENIED (exit 2) and writes the attempt_budget escalation under its unchanged name; a denied launch is NOT counted as an attempt; a row with no matching run record resolves UNKNOWN, never a guess
  KILLED    M10  the join reverts to `timestamp` semantics by widening the tolerance past the measured ambiguity threshold
                 -> a row 900ms from its run record RESOLVES to the returned verdict -- the join tolerance is exercised, not assumed (c1); and it carries that record's tokens and run_id onto the row, giving the attempt stream a shared key with the verdict ledger

real tree untouched (md5 before == after): True
  scripts/harness/attempt_gate.py: 21f355839d7767f0a1ca047d008fd92b
  scripts/harness/attempt_budget.py: 5511ac7e6f105b6b0716d4b80812a170
  scripts/harness/attempt_outcomes.py: 495fa21f2aff3d25b0b12e0a7c376ac9

KILLED 10 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
```

## 5. Criterion 6 -- the verdict ledger across the whole cell run

```
BEFORE: fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2
AFTER : fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2
```

## 6. Live gate state after the change

```
$ python3 scripts/harness/attempt_gate.py --status 86.118
{
  "step_id": "86.118",
  "attempts_used": 5,
  "max_attempts": 5,
  "tokens_used": 1146295,
  "max_tokens": 1200000,
  "verdicts_seen": 4,
  "dropped": 1,
  "outcome_mix": {
    "NO_VERDICT": 1,
    "CONDITIONAL": 2,
    "FAIL": 2
  },
  "disposition": "ESCALATE",
  "next_launch": "deny",
  "ledger": "/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/attempt_budget_audit.jsonl"
}
```

`tokens_used` is a real number for the first time; before this step every step
reported 0. `verdicts_seen` vs `attempts_used` is the gap the old counter could not see.
