STATUS: INCOMPLETE -- not a verdict
STEP: 86.78
WRITTEN: 2026-08-14T10:10:54Z

# Q/A write-first record — step 86.78 (EVALUATE)

Spawn: Workflow rail, Opus 5 (1M). Read qa.md in full at 10:10:54Z.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable cmd, git status/diff scope, lint, drivers
   (verify_escalation_86_78.mjs = 51 checks exit 0; mutation_matrix_86_78.mjs = 13/13)
C. LLM judgment vs 6 immutable criteria + self-audit of leak channels (prompt + qa.md)

## Log (appended as established)
- [10:10:54Z] qa.md read in full (836 lines). NOTE: qa.md lines 615-798 have been
  REWRITTEN vs the version quoted in my own spawn prompt — the prompt's "3rd-CONDITIONAL
  auto-FAIL" section is NOT what is on disk. On-disk text now says thresholds are
  caller-computed and deliberately not described. This is the subject of the step.

### DETERMINISTIC
- IMMUTABLE CMD `source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js
  && echo parses` -> stdout "parses", **exit=0**.
- `node scripts/qa/verify_escalation_86_78.mjs` -> **exit=0**, "checks run : 51
  (cardinality floor 49), failed : 0, ALL CHECKS PASS". Matches claimed 51.
- `node scripts/qa/mutation_matrix_86_78.mjs` -> **exit=0**, CONTROL GREEN observed FIRST
  ("[CONTROL] unmutated checker -> exit 0 ... GREEN control established (51 checks)"),
  then **13 cells, 13 KILLED, 0 survived**; subject sha256[:16]
  before=ebc99a09650183c0 after=ebc99a09650183c0 -> tracked file UNCHANGED. Matches
  claimed 13/13.
- SCOPE: `git diff --stat HEAD` = 7 files. **`.claude/workflows/qa-verdict.js` is
  MODIFIED-UNCOMMITTED (+55/-1)** with an `agentRetryingDrops` StructuredOutput-drop
  retry wrapper; file mtime **10:11:16Z = 22s AFTER this spawn began (10:10:54Z)**.
  Tree movement during EVALUATE. Also concurrent: a Q/A WIP for step **86.79** stamped
  **10:10:30Z** exists -> a second Q/A is running in parallel on a sibling step.
- Lint gates: derived `git diff --name-only HEAD -- '*.py'` -> EMPTY; no backend/**,
  no frontend/**; 1a/1b/1c/1d N/A by DERIVED empty set, not by assertion. No UI claims.

### FINDING F-1 (BLOCKING) — CHANNEL #6: the roster-snapshot system prompt
The scrub of qa.md did **not reach this judge**. MEASURED:
- on-disk `.claude/agents/qa.md` (md5 4d89d02148487f307f5ce8d938f78f7f): `grep -c -F`
  = **0** for each of "3rd-CONDITIONAL auto-FAIL", "return **FAIL** instead of a third",
  "recommend operator escalation", "at 5+", "You MUST state the derived attempt number".
- parent commit `2e40e8c7:.claude/agents/qa.md`: **1** for each of the same five.
- **My SYSTEM prompt contains all five verbatim**, plus `2e40e8c7:.claude/agents/qa.md:663`
  "The trigger is 3 CONSECUTIVE CONDITIONALs, NOT the 3rd attempt", `:674` "C, F, F, C, C,
  PASS", `:676` "forces FAIL at attempts **4 and 5**".
- Version pin (both directions): parent-only markers "records_retained` is the count of
  prior Q/A spawns on this step" and "and it is authoritative" are IN my system prompt and
  0 on disk; on-disk-only markers "--spawned-at"(2), "attempt_number_is_lower_bound"(1),
  "evidence-only"(4) are ABSENT from my system prompt.
=> My system prompt **IS qa.md@2e40e8c7**, the PRE-FIX text. It delivered to me, as
binding system-level instruction: the CONSEQUENCE ("return FAIL instead of a third"), the
THRESHOLD (3 consecutive CONDITIONALs; 5 attempts), the UNIT (consecutive CONDITIONAL
verdicts / cumulative attempts), and an ORDER to compute my own position ("You MUST state
the derived attempt number and the prior-verdict sequence").
=> qa.md:73-76 states the Workflow path has "**no roster snapshot**; only the Agent-tool
`qa` type snapshots at session start". But the rail runs `agentType:'qa'` — it IS that
type. The runtime read can only ADD text; it cannot RETRACT text already in the system
prompt. So a **deletion-based** scrub is inert on this rail until session restart.
