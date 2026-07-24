# Experiment results — Step 75.20 (make the Q/A live-UI gate enforceable AND the primary Q/A path actually read-only)

Date: 2026-07-24. Execution model: opus-tagged step → Main (Fable 5) GENERATE;
Researcher gate opus/max (wf_0d03eec3-633, gate PASSED, 6 read-in-full); probes on
haiku (trivial self-report). SEPARATION OF DUTIES ACTIVE: this step edits
`.claude/agents/qa.md` + `.claude/settings.json` + `.mcp.json` + qa-verdict.js —
**STATUS FLIP HELD** for operator review + next-session roster confirmation (the
live_check spec itself requires the after-restart confirmation).

## What was built

1. **qa.md tools line** now grants EXACTLY the §1c read-only browser subset:
   `mcp__playwright__browser_navigate, browser_snapshot, browser_take_screenshot,
   browser_console_messages` appended to the existing grant; NO mutation tool.
2. **qa.md §1c amendment**: the capture must be taken BY the evaluator when the path
   grants browser tools; a Main-produced capture is the EXPLICITLY-DEGRADED fallback
   (disclosed in the verdict's notes); browser schemas load ONLY via the deterministic
   `select:` ToolSearch form (keyword search surfaces run_code_unsafe/click in top-5
   while missing navigate/snapshot); dev-server lifecycle stays MAIN's — Q/A observes,
   never starts or kills a server.
3. **settings.json** deny += `mcp__playwright__browser_run_code_unsafe`,
   `mcp__playwright__browser_evaluate` (exact spelling; the typo warning is silent for
   underscore names). Deny rules are deny-first, bypass-proof, and bind Workflow agents —
   and they bound LIVE: the harness stripped both tools from this session's surface on
   write (verbatim notice in live_check §3c).
4. **.mcp.json** playwright: fixed `--user-data-dir` pin → `--isolated`. Two-client
   demonstration: shared profile reproduces the vendor's documented contention error
   verbatim; --isolated runs both clients clean (live_check §4).
5. **qa-verdict.js** `agentType: 'general-purpose'` → `'qa'` with the rationale comment:
   probe-proven removal of Edit/Write-adjacent surface (Artifact/Skill) and the ENTIRE
   MCP surface (7 loaded + hundreds deferred incl. playwright) from the primary path.
   Stall-immunity unchanged (StructuredOutput captured-return, not agent-type-dependent).
6. **New test suite** `backend/tests/test_phase_75_20_qa_browser_grant.py` (13 tests):
   grant presence/absence/superset-envelope, exact deny entries, NON-vacuous isolation
   assert (kills the hazard the immutable command's vacuous assert #3 cannot — R11),
   agentType pin, §1c prose pins (evaluator-capture + degraded fallback, select: form,
   lifecycle).
7. **75.20.1 queued** (research-gated, opus-tagged): the DISCOVERED defect that the
   loader injects Write+Edit into the qa agent past its frontmatter allowlist
   (probe-proven; disallowedTools silently ignored) — written for an executor with no
   memory of this session, per feedback_queue_discovered_defects_in_masterplan.

## Files changed

`.claude/agents/qa.md`, `.claude/settings.json`, `.mcp.json`,
`.claude/workflows/qa-verdict.js`, `backend/tests/test_phase_75_20_qa_browser_grant.py`
(new), `.claude/masterplan.json` (75.20 → in_progress; 75.20.1 inserted),
`handoff/current/{contract.md, research_brief_75.20.md, live_check_75.20.md,
experiment_results.md}`. ZERO product code (`git diff --name-only` shows no
backend/frontend source change beyond the new test).

## Verbatim verification output

Immutable command: `immutable-verification-exit=0` (live_check §1, with the R11 vacuity
of its assert #3 disclosed + proven against the OLD args).

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_20_qa_browser_grant.py -q
.............                                                            [100%]
13 passed in 0.02s
$ uvx ruff check backend/tests/test_phase_75_20_qa_browser_grant.py
All checks passed!
$ python3 -c "import json; json.load(open('.claude/settings.json')); json.load(open('.mcp.json')); print('json OK')"
json OK
```

## Criterion-by-criterion status

- C1 grant: DONE (tests + mutation N1).
- C2 deny: DONE, live-proven binding (tests + mutation N2 + the live strip notice).
- C3 primary path constrained: DONE for the MCP/Artifact/Skill surface (probe-proven,
  before/after in live_check §3); Write/Edit residual DISCLOSED + queued as 75.20.1
  (loader injection, not removable by frontmatter/disallowedTools/session-deny).
- C4 isolation: DONE (edit + vendor-error reproduction + fix demonstration + non-vacuous
  test; immutable assert #3's vacuity disclosed, never used as evidence).
- C5 select: form + lifecycle: DONE (§1c text + tests + mutation N5).
- C6 §1c evaluator-capture + degraded fallback: DONE in text; operator-review request +
  next-session roster verification recorded in harness_log; **roster-live confirmation
  is next-session-owed by construction — status flip HELD.**

## Mutation matrix

7 mutations (6 config + 1 STUB/harness), **7 killed, 0 survivors**, post-restore green
(live_check §5). N3b is the showcase: a hazard restoration that PASSES the immutable
command is killed by the step's own non-vacuous guard.
