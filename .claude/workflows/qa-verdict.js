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

// `args` may arrive as a parsed object OR as a JSON string (the Workflow tool
// stringifies scriptPath args on some paths) OR be absent (a dry-run). Handle
// all three so the parameterized launch actually threads its parameters; on any
// parse error, fall back to {} and the prompt tells the agent to self-recover
// the step context from .claude/masterplan.json + handoff/current/.
let a = {}
try {
  if (typeof args === 'string' && args.trim()) a = JSON.parse(args)
  else if (args && typeof args === 'object') a = args
} catch (_e) { a = {} }
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
  'CONDITIONAL for fixable gaps, FAIL for a criterion miss. If this would be the 3rd consecutive CONDITIONAL for',
  'this step-id (grep harness_log), return FAIL instead. NEVER return PASS on a loop-prevention / errored exit.',
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
const verdict = await agent(PROMPT, {
  label: 'qa-verdict:' + stepId,
  phase: 'QA',
  schema: VERDICT_SCHEMA,
  agentType: 'general-purpose',
  model: 'opus',
  effort: 'max',
})
return verdict
