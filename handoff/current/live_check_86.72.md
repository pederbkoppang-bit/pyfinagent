# live_check -- step 86.72 (2026-08-17; exits taken unpiped)

## 1. Criterion 1 -- absence, before/after, with named controls

```
$ grep -c "research_needed" scripts/harness/run_harness.py
5
$ git show HEAD:.claude/workflows/qa-verdict.js | grep -c "research_needed"
0
$ grep -c "research_needed" .claude/workflows/qa-verdict.js
7
$ grep -c "research_needed" .claude/workflows/research-gate.js
0
$ grep -rc "ZZZ_NO_SUCH_86_72" scripts/harness/run_harness.py
0
```

(HEAD at capture = fff6d8c4, the commit before this GENERATE's edits; "0 ->
7" is the wiring landing. The count moves with future edits; the derivation
is the command.)

## 2. Criterion 2 -- the per-step split, population rule stated

Population: every `agent-*.jsonl` under
`~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/subagents/workflows/wf_*/`,
FIRST user message, step id parsed from the prompt, role classified by
marker ("IMMUTABLE SUCCESS CRITERIA" -> qa; "OBJECTIVE:" -> researcher).

```
top qa-spawn steps: [('86.85', 12), ('86.84', 8), ('86.74', 7), ('86.94', 6), ('86.97', 5), ('86.71', 5)]
  86.85: qa=12 researcher=0
  86.84: qa=8 researcher=0
  86.74: qa=7 researcher=0
  86.94: qa=6 researcher=0
  86.97: qa=5 researcher=0
  86.71: qa=5 researcher=0
```

## 3. Criteria 3/7/8 -- the checker drive and the kills (family after the edit)

```
$ node scripts/qa/verify_prompt_render_86_90.mjs > /tmp/pr8.txt 2>&1; echo EXIT=$?
EXIT=0
$ tail -1 /tmp/pr8.txt
ALL GREEN: 126 passed, 0 failed
$ grep "\[8\]" /tmp/pr8.txt | grep -vE "^\[8\]"
  ok   [8] positional caller text -> judge_was_told_consequence === true
  ok   [8] the matched evidence substring is recorded
  ok   [8] clean caller text -> false
  ok   [8] RULE mention without positional claim -> false (rule vs position distinction)
  ok   [8] absent caller_text -> false, never a throw
  ok   [8] research_needed=true -> surfaced with the spec echoed
  ok   [8] guidance present and carries the Tmax=2 bound
  ok   [8] absent fields -> research_needed null, guidance null
  ok   [8] research_needed=false -> false, guidance null
  ok   [8] 8-recorder-neutered: KILLED
  ok   [8] 8-recorder-hardcoded-false: KILLED
  ok   [8] 8-routing-signal-dropped: KILLED
  ok   [8] 8-tmax-bound-removed: KILLED
$ node scripts/qa/verify_research_gate_workflow.mjs 2>&1 | tail -1
ALL GREEN: 124 passed, 0 failed
$ node scripts/qa/verify_workflow_args_boundary.mjs 2>&1 | tail -1
ALL GREEN: 96 passed, 0 failed
```

Disclosure: the tmax cell's first mutant SURVIVED (too weak -- it removed
middle text while both probe substrings remained) and the section-[8]
slicer's first run threw its own loud anchor assertion (naive first-brace
grab hit `opts = {}`). Both first-run failures are part of the record; the
fixes are in the same commit as the section.
