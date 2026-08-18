# live_check -- step 86.96 (captured 2026-08-17, operator-attended session)

Verbatim command output for every figure in `experiment_results_86.96.md`.

## 1. Reproduction -- the shipped classifyArgs, driven twice per payload

```
$ node <slice classifyArgs from .claude/workflows/qa-verdict.js and drive both stored payloads>
wf_1f6b0398-020 run1: THROWS -> qa-verdict: args ... -- pass a plain object (or vali...
wf_1f6b0398-020 run2: THROWS -> (identical)
wf_88302c2a-d20 run1: THROWS -> (identical)
wf_88302c2a-d20 run2: THROWS -> (identical)
```

Verbatim production error (run-record `error` field, both records):
`Error: qa-verdict: args are PRESENT but not parseable as JSON (typeof=string
isArray=false len=5481 preview="{\"step_id\":\"86.90\",...") -- pass a plain
object (or valid JSON) carrying step_id, or omit args entirely for a dry run.
at fail (workflow.js:78:11) at classifyArgs (workflow.js:87:48)`

## 2. Bisection at the failure offsets

```
wf_1f6b0398-020: raw=FAILS: Expected ',' or ']' after array element | substitute(pos 4939)=PARSES | insert=FAILS: Unexpected non-whitespace character after JSON
   char at pos: "}", context: "n the commands and diff.\"},\"known"
wf_88302c2a-d20: raw=FAILS: (same class)                | substitute(pos 5536)=PARSES | insert=FAILS: (same class)
   char at pos: "}", context: "ed -- cycle 2's did not.\"},\"known"
size control: valid payload of 6088 chars -> PARSES
$ python3 json.loads on both payloads: "Expecting ',' delimiter" at pos 4939 / 5536
```

## 3. Census (population rule in experiment_results section 5)

```
census over 585 run records: {'OBJECT': 95, 'STRING_PARSES': 390, 'STRING_FAILS': 4, 'ABSENT': 96, 'OTHER': 0}
string args share: 394/489 of arg-carrying launches
STRING_FAILS enumerated:
  wf_b098cab6-87b  research-gate  2026-08-06T14:26:44Z  'Unterminated string starting at' pos 123   len=3201
  wf_8375665b-f5a  research-gate  2026-08-09T15:01:31Z  'Expecting ',' delimiter'        pos 4911  len=4911   <- pos == len: TRUNCATION
  wf_1f6b0398-020  qa-verdict     2026-08-16T09:14:58Z  'Expecting ',' delimiter'        pos 4939  len=5481   <- bracket
  wf_88302c2a-d20  qa-verdict     2026-08-16T09:15:30Z  'Expecting ',' delimiter'        pos 5536  len=6090   <- bracket
ESCAPED-QUOTES CONTROL -- parsed production payloads containing \":
  wf_f7d084d8-76c (2) / wf_091e2312-0d8 (8) / wf_ea569c91-52a (8) / wf_4575b02b-eb0 (2)  -> all PARSE
```

## 4. Round-trip + guard (criteria 4 + 6)

```
$ node scripts/qa/verify_prompt_render_86_90.mjs        # section [7] added by this step
[7] BYTE-VERBATIM CRITERIA ROUND-TRIP -- phase-86.96 (criteria 4 + 6)
  ok   [7] object args: script runs and spawns exactly one evaluator
  ok   [7] object args: criterion 1..4 arrive BYTE-VERBATIM (4 checks)
  ok   [7] object args: criteria arrive IN ORDER
  ok   [7] string args: script runs and spawns exactly one evaluator
  ok   [7] string args: criterion 1..4 arrive BYTE-VERBATIM (4 checks)
  ok   [7] object and string paths render IDENTICAL prompt bytes
  ok   [7] fixture sanity: the bracket-defect payload does NOT parse
  ok   [7] a malformed string payload dies LOUD at the boundary, spawning NOTHING
  ok   [7] 7-classifyArgs-repairs-instead-of-refusing: CONTROL is clean
  ok   [7] 7-classifyArgs-repairs-instead-of-refusing: KILLED
  ok   [7] 7-render-mangles-quotes-in-transit: CONTROL is clean
  ok   [7] 7-render-mangles-quotes-in-transit: KILLED
ALL GREEN: 113 passed, 0 failed                                     (exit 0)
```

## 5. Immutable command + checker family

```
$ bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
parses                                                              (exit 0)
$ node scripts/qa/verify_workflow_args_boundary.mjs
ALL GREEN: 96 passed, 0 failed                                      (exit 0)
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 124 passed, 0 failed                                     (exit 0)
```

*(The args-boundary and research-gate-workflow lines above are re-runs of the
existing checkers, quoted to show criterion 7's "family stays green" claim; the
96/0 figure is the 86.92-fixed state the research gate verified.)*
