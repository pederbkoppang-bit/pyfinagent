export const meta = {
  name: 'qa-verdict',
  description: 'Layer-3 Q/A EVALUATE as a structured-output task -- the verdict IS the captured return value (immune to the Agent-tool end-flush stall). Reads .claude/agents/qa.md from disk at runtime as the single source of truth.',
  whenToUse: 'Every EVALUATE phase of a masterplan step when running unattended / on the Opus Max rail. Pass args={step_id, criteria[], verification_command, evidence, extra}. This is the FIRST-CLASS launch; the Agent-tool `qa` subagent is the documented fallback.',
  phases: [ { title: 'QA', detail: 'read qa.md -> harness-compliance audit -> deterministic checks -> LLM judgment -> return verdict' } ],
}

// ---------------------------------------------------------------------------
// Reusable Q/A launcher (phase-71.1). The captured return value is the verdict;
// Main transcribes it VERBATIM into handoff/current/evaluator_critique.md.
// NEVER auto-PASS on an errored/empty return -- that is NO VERDICT, so Main
// falls back to the Agent-tool `qa` subagent path (never PASS on error).
// The Q/A agent reads .claude/agents/qa.md from disk => any qa.md edit is live
// immediately on THIS path (no roster snapshot; the snapshot caveat binds only
// the Agent-tool `qa` type). Keep the agent on model:'opus' (rider-trap R4:
// the stall is model-agnostic; routing off Opus also violates the effort/model
// policy). Do NOT loop fix->re-grade internally (rider-trap R1): return a
// verdict and STOP; Main owns the fix and spawns a FRESH Q/A on changed
// evidence (CLAUDE.md canonical cycle-2 flow). Do NOT wrap this launch in a
// Monitor/transcript-mtime watchdog (rider-trap R11): the captured-return-value
// path makes polling unnecessary and it contradicts the do-not-poll rule
// (docs/runbooks/per-step-protocol.md Subagent-runtime-semantics).
// ---------------------------------------------------------------------------

// ── phase-86.17: THE ARGS BOUNDARY ─────────────────────────────────────────
// CORRECTED COMMENT (criterion 7). This block previously read: "on any parse
// error, fall back to {} and the prompt tells the agent to self-recover the
// step context from .claude/masterplan.json + handoff/current/." That comment
// DECLARED THE DEFECT DELIBERATE, and a stale comment asserting the opposite of
// the code is how it survived review. The remedy it prescribed -- ask the agent
// to recover its own identity from prose -- is exactly the prompt-level
// self-reflection EviBound measured at 100% false-completion claims. It is
// replaced, not merely amended.
//
// THREE CLASSES, ORDER MATTERS (identical to research-gate.js; the two scripts
// cannot share a module because the Workflow runtime forbids imports, so this
// is a deliberate duplicate and the checker drives BOTH copies):
//
//   A. ABSENT     -- `typeof args === 'undefined'` (UNBOUND on a no-args
//                    launch -- MEASURED at $0, run wf_a1b6c046-b60) or null.
//                    Does NOT throw: the dry run stays legal. But it returns
//                    NO VERDICT AT ALL (see the early return below) rather than
//                    a verdict-shaped object, so Main's transcribe-VERBATIM
//                    rule has nothing it could mistake for an evaluation.
//   B. UNUSABLE   -- present but not a plain object, incl. a DOUBLE-ENCODED
//                    JSON string that parses successfully. THROW.
//   C. INCOMPLETE -- a plain object with no step_id. THROW: a present args
//                    object proves the caller meant to parameterise.
//
// `typeof` is MANDATORY -- a bare `args === undefined` raises ReferenceError
// when the identifier is unbound, which is exactly the no-args case.
function classifyArgs(bound, raw) {
  const describe = (v) => {
    let preview
    try { preview = typeof v === 'string' ? v : JSON.stringify(v) } catch (_e) { preview = '(unstringifiable)' }
    preview = String(preview === undefined ? 'undefined' : preview)
    return 'typeof=' + (typeof v) + ' isArray=' + Array.isArray(v)
      + ' len=' + preview.length + ' preview=' + JSON.stringify(preview.slice(0, 80))
  }
  const fail = (why, v) => {
    throw new Error('qa-verdict: args ' + why + ' (' + describe(v)
      + ') -- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.')
  }

  if (!bound || raw === null) return { status: 'dry_run', blind: true, args: {} }

  let v = raw
  if (typeof v === 'string') {
    if (!v.trim()) fail('are PRESENT but an empty/blank string', raw)
    try { v = JSON.parse(v) } catch (_e) { fail('are PRESENT but not parseable as JSON', raw) }
  }
  if (typeof v !== 'object' || v === null || Array.isArray(v)) fail('did not reduce to a plain object', raw)
  if (!v.step_id && !v.stepId) fail('are a plain object with NO step_id', raw)

  return { status: 'ok', blind: false, args: v }
}

const ARGS_BOUND = typeof args !== 'undefined'
const inputHealth = classifyArgs(ARGS_BOUND, ARGS_BOUND ? args : null)
const a = inputHealth.args
const stepId = a.step_id || a.stepId || 'UNSPECIFIED'
const criteria = Array.isArray(a.criteria) ? a.criteria : []
const verificationCommand = a.verification_command || a.verificationCommand || '(none provided -- read it from .claude/masterplan.json for this step)'
const evidence = a.evidence || 'handoff/current/{contract.md, experiment_results.md, evaluator_critique.md} + the files changed this step (git status --short / git diff)'
const extra = a.extra || ''

const PROMPT = [
  'You are the pyfinagent Layer-3 Q/A evaluator (merged qa-evaluator + harness-verifier) for masterplan step ' + stepId + ', EVALUATE phase.',
  '',
  'STEP 0 (binding): Read .claude/agents/qa.md IN FULL and follow it as your operating instructions -- it is the',
  'single source of truth for the Q/A role (verification order, the deterministic-first discipline, the lint +',
  'runtime-smoke gates, the output schema, the no-auto-PASS clause, the 3rd-CONDITIONAL auto-FAIL rule, and the',
  'no-second-opinion-shopping rule). This runtime read makes any qa.md edit live immediately on the Workflow path.',
  'Also read docs/runbooks/per-step-protocol.md if you need the runbook context.',
  '',
  'STEP 0b (binding, phase-86.31, path revised by phase-86.36): WRITE-FIRST FOR YOUR VERDICT FILE ONLY.',
  'Within your first few tool calls, create',
  '.claude/agent-memory/qa/verdicts/verdict_wip_' + stepId + '__<STAMP>.md (mkdir the verdicts/ dir if absent),',
  'where <STAMP> is the current UTC time from `date -u +%Y%m%dT%H%M%SZ` -- e.g.',
  'verdict_wip_' + stepId + '__20260811T065957Z.md. The RUN STAMP IS MANDATORY: the filename used to be fixed per',
  'step, and because this rule makes you write on your FIRST tool call, a retry destroyed the previous attempt.',
  'MEASURED: verdict_wip_86.34.md went 4,921 -> 796 bytes between two tool calls of one observer. Do NOT omit the',
  'stamp and do NOT reuse another run\'s. Its FIRST LINE is verbatim "STATUS: INCOMPLETE -- not a verdict",',
  'followed by "STEP: ' + stepId + '" and a "WRITTEN: <UTC ISO-8601>" stamp from `date -u +%Y-%m-%dT%H:%M:%SZ`,',
  'which must be the SAME instant as the filename stamp. The WRITTEN header stays load-bearing even now: the',
  'filename keeps attempts from overwriting each other, the header is what lets qa_wip.py decide which record',
  'belongs to the spawn being recovered from. Append findings AS YOU ESTABLISH THEM -- the',
  'immutable command exit code, each deterministic check, each mutation cell, each criterion MET/NOT MET with its',
  'evidence -- never a single end-of-run flush. As your FINAL act before returning, rewrite that first line to',
  '"STATUS: COMPLETE -- write-first record, still NOT a verdict" and append "COMPLETED: <UTC ISO-8601>".',
  'This rail returned NO verdict on 3 of 8 spawns',
  'on 2026-08-10 and one of those had already FOUND A REAL SURVIVING MUTANT before it died; the file is how that',
  'work survives a drop. qa.md section "Write-first for your VERDICT FILE ONLY" governs the details. That path is',
  'the ONLY one qa-write-guard.sh permits you -- no allowlist was added and no deny removed, so production code,',
  'tests, .claude/masterplan.json and every handoff/ artifact stay DENIED. Do not work around a block: if a write',
  'you need is denied, say so in `notes` and return. The WIP file changes NOTHING about what you return or how you',
  'judge -- your structured return is still the deliverable, and a recovered WIP is EVIDENCE for the next spawn,',
  'never a verdict, not even when marked COMPLETE.',
  '',
  'You are INDEPENDENT of the author (Main). Do NOT rubber-stamp. You are READ-ONLY on file contents: you may run',
  'Bash ONLY for non-mutating verification (test -f, ls, grep, jq, git log/status/diff, python -c, pytest,',
  'npx tsc --noEmit) -- NEVER Edit/Write to production files, never rm/mv/sed -i/git commit/git push, no > or >>.',
  '',
  'DO IN ORDER (qa.md governs the details):',
  'A. HARNESS-COMPLIANCE AUDIT FIRST (5 items): research-gate-before-contract (research_brief exists, gate_passed',
  '   true, >=5 sources, recency scan); contract-before-generate (mtime: research < contract < generated artifact);',
  '   experiment_results present; log-last (the step is NOT yet in harness_log with a result / masterplan not yet',
  '   flipped done); no-verdict-shopping (if this is a re-spawn, the evidence CHANGED since the prior verdict).',
  'B. DETERMINISTIC: run the immutable verification command and record its exit code:',
  '     ' + verificationCommand,
  '   Then confirm NO UNINTENDED production code changed (git status --short / git diff) beyond what the contract',
  '   scopes. Run syntax/lint/scoped tests per qa.md when the diff touches code.',
  'C. LLM JUDGMENT against the immutable success criteria below -- contract alignment, scope honesty,',
  '   mutation-resistance, anti-rubber-stamp, research-gate compliance. Name any real weakness; a plausible-sounding',
  '   but wrong finding must be caught. For each criterion, state MET / NOT MET with cited evidence.',
  '',
  'IMMUTABLE SUCCESS CRITERIA for ' + stepId + ':',
  ...(criteria.length ? criteria.map((c, i) => '  ' + (i + 1) + '. ' + c) : ['  (none passed in args -- read them from .claude/masterplan.json for this step and evaluate against them)']),
  '',
  'EVIDENCE / FILES TO READ: ' + evidence,
  extra ? ('\nADDITIONAL CONTEXT: ' + extra) : '',
  '',
  'Return the verdict AS YOUR RETURN VALUE using the schema. This captured object IS the deliverable -- Main',
  'transcribes it VERBATIM into evaluator_critique.md (it is NOT a human-facing message). verdict=PASS only if',
  'EVERY immutable criterion is met AND harness compliance is clean AND no unintended production change. Use',
  'CONDITIONAL for fixable gaps, FAIL for a criterion miss. Before issuing CONDITIONAL, count your own prior',
  'attempts by running `python scripts/qa/qa_wip.py <step_id>` and reading records_retained / prior_records --',
  'NOT by grepping harness_log, which is written in the LOG phase AFTER the verdict and so never contains the',
  'in-flight cycle (measured phase-86.75: qa_wip says 3 for step 86.33 where harness_log says 0). If this would',
  'The TRIGGER is 3 CONSECUTIVE CONDITIONALs, NOT the 3rd attempt (corrected phase-86.21): if this step-id',
  'already has 2 consecutive prior CONDITIONALs with no intervening PASS or FAIL, return FAIL instead of a',
  'third. The consecutive run RESETS on PASS or FAIL. records_retained gives the ATTEMPT number (authoritative);',
  'the verdict SEQUENCE is a different quantity with a weaker source. Take it, in order, from: (1) a',
  '"## Verdict ledger" block in handoff/current/evaluator_critique_<id>.md; (2) header-anchored rows',
  'grep -E "^## Cycle .*phase=<id> .*result=" handoff/harness_log.md, which UNDERCOUNT (4->0 on step 86.62);',
  '(3) Main\'s disclosure, which is ADVISORY ONLY since Main is the constrained party. Do NOT infer verdicts by',
  'scanning prior_records bodies for the words PASS/CONDITIONAL/FAIL -- only 3 of 46 records carry a parseable',
  'verdict line and the bodies DISCUSS verdicts at length (86.21: 15x "CONDITIONAL" in a record whose verdict',
  'was FAIL). If the sequence cannot be established, write sequence: UNKNOWN in notes and do NOT guess -- fall',
  'back to the attempt number against the 5-attempt budget, never to an invented run length.',
  'Measured on step 36.17 (C,F,F,C,C,PASS): an attempt-count trigger forces FAIL at attempts 4 and 5 and would',
  'have denied the PASS it earned at attempt 6, so do NOT use the attempt number as the CONDITIONAL trigger.',
  'Do report the attempt number against F1b\'s 5-attempt cumulative budget: at 5+, recommend operator',
  'escalation in notes rather than inventing a verdict. State the derived attempt number and prior-verdict',
  'sequence in notes. harness_log is a secondary cross-check only; if the two disagree, the ledger governs.',
  'CHECK source_present FIRST (phase-86.21): records_retained==0 is a fact about ATTEMPTS only when',
  'source_present is true. If it is false the sink does not exist, so 0 means the counter has NO INPUT, not',
  '"attempt 1" -- report the attempt number as UNKNOWN in notes and do not let the zero suppress escalation.',
  'NEVER return PASS on a loop-prevention / errored exit.',
].join('\n')

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ok', 'verdict', 'reason', 'violated_criteria', 'violation_details', 'certified_fallback', 'checks_run', 'harness_compliance_ok', 'notes'],
  properties: {
    ok: { type: 'boolean' },
    verdict: { type: 'string', enum: ['PASS', 'CONDITIONAL', 'FAIL'] },
    reason: { type: 'string' },
    violated_criteria: { type: 'array', items: { type: 'string' } },
    violation_details: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['violation_type', 'action', 'state', 'constraint'],
        properties: {
          violation_type: { type: 'string', enum: ['Missing_Assumption', 'Invalid_Precondition', 'Unjustified_Inference', 'Circular_Reasoning', 'Contradiction', 'Overgeneralization', 'Threshold_Not_Met'] },
          action: { type: 'string' },
          state: { type: 'string' },
          constraint: { type: 'string' },
        },
      },
    },
    certified_fallback: { type: 'boolean' },
    checks_run: { type: 'array', items: { type: 'string' } },
    harness_compliance_ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
}

phase('QA')

// phase-86.17 (criterion 4): a BLIND run must not throw.
// PLACED BELOW phase('QA') DELIBERATELY (cycle-2 correction). Checkers
// import this file by slicing at the phase() driver boundary, and a
// top-level `return` is legal only inside the Workflow runtime's async
// wrapper -- above the boundary it would be a SyntaxError in the sliced
// module. Keeping it here also means a checker that slices at phase()
// CANNOT see this block, which is why it needs full-driver coverage
// (verify_workflow_args_boundary.mjs section [5]) rather than the
// module-slice coverage that silently missed it in cycle 1. -- the dry run stays
// legal -- but it must also be incapable of producing a PASS. Returning early
// is stronger than returning a verdict object with verdict:null, because it
// means no max-effort Q/A session is spent evaluating a step that was never
// named. There is nothing here a transcription could mistake for an evaluation.
if (inputHealth.blind) {
  log('qa-verdict: WARNING -- BLIND RUN. args were ABSENT, so there is no step, no criteria and '
      + 'no evidence to evaluate. Returning NO VERDICT (never a PASS) and spawning nothing.')
  return {
    dry_run: true,
    verdict: null,
    ok: false,
    input_health: { status: inputHealth.status, blind: true },
    reason: 'BLIND RUN: args were absent, so no step was identified. This is NOT a verdict -- '
      + 'do not transcribe it into evaluator_critique. Re-launch with args={step_id, criteria, ...}.',
  }
}
// phase-75.20: agentType 'qa' (was 'general-purpose') so the primary path is
// CONSTRAINED BY CONFIGURATION, not prose: the probe (wf_9277ada4-390) showed
// general-purpose carries Edit/Write/Bash + 7 loaded MCP tools + the full
// deferred MCP surface (incl. playwright), while agentType 'qa' collapses the
// MCP surface and drops Artifact/Skill. DISCLOSED RESIDUAL: the loader
// injects Write/Edit into the qa type past its frontmatter allowlist
// (probe-proven; disallowedTools is silently ignored) -- that residual is
// covered by qa.md prose + Main's post-verdict `git status` cleanliness
// check, and queued as its own masterplan step. Stall-immunity is unchanged:
// it comes from the StructuredOutput captured-return, not the agent type.
//
// phase-86.31: VERDICT_SCHEMA above is DELIBERATELY UNCHANGED. The obvious
// addition -- a `wip_path` / `wrote_verdict_file` field so Main could audit
// write-first compliance from the return -- was weighed and REJECTED against
// criterion 7 ("no change to the Q/A's criteria, judgement, effort, or output
// schema beyond adding a completion marker if criterion 3 requires one").
// Criterion 3 puts the marker ON THE ARTIFACT, not in the return, so the field
// is not required by it; and it would buy nothing on the failure it targets,
// because a DROPPED run produces no return object for the field to live in.
// Main verifies compliance from disk instead -- the path is deterministic from
// step_id (`python scripts/qa/qa_wip.py <step_id>`).
const verdict = await agent(PROMPT, {
  label: 'qa-verdict:' + stepId,
  phase: 'QA',
  schema: VERDICT_SCHEMA,
  agentType: 'qa',
  model: 'opus',
  effort: 'max',
})
return verdict
