export const meta = {
  name: 'research-gate',
  description: 'Layer-3 RESEARCH gate as a structured-output task -- the envelope IS the captured return value, and the script RECOMPUTES gate_passed rather than trusting the agent. Reads .claude/agents/researcher.md from disk at runtime as the single source of truth.',
  whenToUse: 'Before the PLAN phase of every masterplan step. Pass args={step_id, topic, tier, internal_scope, audit_class, brief_path}. This is the FIRST-CLASS launch; the Agent-tool `researcher` subagent is the documented fallback.',
  phases: [ { title: 'Research', detail: 'read researcher.md -> external + internal research -> write brief incrementally -> return envelope -> script enforces the floors' } ],
}

// ---------------------------------------------------------------------------
// phase-36.27. Companion to qa-verdict.js: the operator instruction of
// 2026-07-27 is that BOTH Layer-3 agents launch on the Workflow rail. Until
// this file existed, every researcher spawn fell back to the Agent tool, so
// the doctrine was implemented for Q/A and unimplementable for the Researcher.
//
// WHY THE SCRIPT ASSERTS THE FLOORS INSTEAD OF THE SCHEMA (the design decision):
//   1. Anthropic structured outputs STRIPS `minimum`/`maximum`/`minLength` from
//      the wire schema and caps `minItems` at 1. `>=5 sources` and `>=10 URLs`
//      are therefore NOT expressible as schema constraints.
//   2. Even where a schema CAN constrain, conformance is STRUCTURAL ONLY. A
//      schema can force `external_sources_read_in_full` to be an integer; it
//      cannot make it TRUE. Measured in the literature: "The Constraint Tax"
//      (2026-05) found wrong-but-schema-valid output rising 49.5% -> 88.9%
//      under constrained decoding, and EviBound measured 100% false completion
//      claims from prompt-level self-reflection alone, falling to 0% ONLY with
//      a post-hoc gate that queries the artifact store.
//   3. `const: true` on a gate field is a TRAP: it makes an honest `false`
//      unrepresentable, so the agent must either lie or fail to return.
//      `gate_passed` is a plain boolean here, deliberately.
//
// => The agent's `gate_passed` is recorded as a SELF-REPORT and the script
//    RECOMPUTES the real one, including an artifact cross-check against the
//    brief actually on disk. A count the artifact does not corroborate fails.
//
// RIDER TRAPS carried over from qa-verdict.js -- do not re-litigate:
//   R1  Do NOT loop research->re-grade internally. Return the envelope and STOP.
//   R4  Keep model:'opus'. The stall is model-agnostic and routing off Opus
//       violates the effort/model policy.
//   R11 Do NOT wrap this launch in a Monitor / transcript-mtime watchdog. The
//       captured-return-value path makes polling unnecessary.
// The researcher reads .claude/agents/researcher.md from disk at runtime, so a
// researcher.md edit is live immediately on THIS path (no roster snapshot).
//
// TOOL SURFACE -- the one place the Q/A precedent deliberately does NOT carry:
// qa-verdict.js pins agentType 'qa' to RESTRICT the surface (Q/A is read-only).
// The researcher legitimately NEEDS Write: write-first is non-negotiable, and a
// session that cannot clear the gate must still leave a partial brief on disk.
// agentType 'researcher' gets Write via its `memory: project` injection, and
// the qa-write-guard PreToolUse hook matches agent_type == 'qa' only, so it
// does not block this path.
// ---------------------------------------------------------------------------

import fs from 'node:fs'
import path from 'node:path'

// `args` may arrive as a parsed object OR a JSON string OR be absent (dry run).
let a = {}
try {
  if (typeof args === 'string' && args.trim()) a = JSON.parse(args)
  else if (args && typeof args === 'object') a = args
} catch (_e) { a = {} }

const stepId = a.step_id || a.stepId || 'UNSPECIFIED'
const topic = a.topic || '(no topic passed -- derive it from the step entry in .claude/masterplan.json)'
const VALID_TIERS = ['simple', 'moderate', 'complex']
const tierRaw = a.tier || 'moderate'
const tier = VALID_TIERS.includes(tierRaw) ? tierRaw : 'moderate'
const tierDefaulted = !a.tier || !VALID_TIERS.includes(tierRaw)
const internalScope = a.internal_scope || a.internalScope || '(none passed -- derive the relevant modules from the step entry)'
const auditClass = a.audit_class === true || a.auditClass === true
// The script tells the agent the EXACT path it will later verify, so write-first
// and the artifact cross-check cannot refer to different files.
const briefPath = a.brief_path || a.briefPath || `handoff/current/research_brief_${stepId}.md`

const FLOOR_SOURCES = 5
const FLOOR_URLS = 10
const K_REQUIRED = 2

const PROMPT = [
  'You are the pyfinagent Layer-3 RESEARCHER (combined external-literature researcher + internal-codebase explorer)',
  'for masterplan step ' + stepId + ', RESEARCH phase. This runs BEFORE the contract is written.',
  '',
  'STEP 0 (binding): Read .claude/agents/researcher.md IN FULL and follow it as your operating instructions --',
  'it is the single source of truth for the role. Also read .claude/rules/research-gate.md IN FULL: it carries the',
  'authoritative floors (>=' + FLOOR_SOURCES + ' sources read IN FULL via WebFetch, >=' + FLOOR_URLS + ' URLs collected,',
  'the mandatory last-2-year recency scan, the three-variant search discipline, the source-quality hierarchy, and the',
  'arXiv html -> ar5iv -> pdfplumber chain -- never WebFetch an arxiv.org/pdf/ URL).',
  'This runtime read makes any researcher.md edit live immediately on the Workflow path.',
  '',
  'OBJECTIVE: ' + topic,
  'TIER: ' + tier + (tierDefaulted ? '  (NOT passed by the caller -- defaulted to moderate; state this assumption in the brief)' : ''),
  'INTERNAL SCOPE: ' + internalScope,
  'AUDIT-CLASS: ' + (auditClass
    ? 'YES. The >=' + FLOOR_SOURCES + ' floor is a FLOOR, not a ceiling. Run the loop-until-dry completeness critic: keep '
      + 'running extra search/fetch rounds until ' + K_REQUIRED + ' CONSECUTIVE rounds surface ZERO new read-in-full findings '
      + 'beyond de-dup, then set coverage.dry = true. gate_passed ADDITIONALLY requires coverage.dry.'
    : 'NO. Report the coverage object for information; coverage.dry is not required for this step.'),
  '',
  'OUTPUT FORMAT -- BOTH are required, and they are checked independently:',
  '  (1) THE BRIEF ON DISK at exactly: ' + briefPath,
  '      WRITE-FIRST IS NON-NEGOTIABLE. Create this file within your first few tool calls and APPEND to it as you',
  '      read each source. Never hold it for a single end-of-session flush. If you cannot clear the gate, still',
  '      leave a partial brief plus an honest gate_passed:false envelope. Padding a brief to mask an under-fetch',
  '      is a protocol breach.',
  '  (2) THE ENVELOPE as your structured return value.',
  '',
  'HONESTY CONTRACT -- read this twice. The calling script does NOT trust your gate_passed. It RECOMPUTES it, and',
  'it CROSS-CHECKS your self-report against the brief actually on disk: every URL you list in sources_read_in_full',
  'must appear in that file, and the list must be at least as long as your external_sources_read_in_full count.',
  'An over-claim is therefore detected, not merely discouraged. Returning gate_passed:false honestly is a NORMAL,',
  'CORRECT outcome and is strictly better than a claim the artifact cannot corroborate. Search snippets do NOT',
  'count as read-in-full; only a full WebFetch of the source does.',
  '',
  'TASK BOUNDARIES: research and report only. Do NOT implement anything, do NOT edit production code, do NOT write',
  'to .claude/workflows/, and do NOT write the contract -- Main owns PLAN. Do not run destructive commands.',
  'Return the envelope and STOP; do not loop internally trying to improve your own gate result.',
].join('\n')

// NOTE: no `minimum` / `minItems` here -- they are stripped or capped on the
// wire, so declaring them would create a false sense of enforcement. Every
// numeric floor is asserted in JS below. `gate_passed` is a PLAIN boolean:
// `const: true` would make honest failure unrepresentable.
const ENVELOPE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: [
    'tier', 'external_sources_read_in_full', 'sources_read_in_full', 'snippet_only_sources',
    'urls_collected', 'recency_scan_performed', 'internal_files_inspected',
    'coverage', 'brief_path', 'summary', 'gate_passed',
  ],
  properties: {
    tier: { type: 'string', enum: VALID_TIERS },
    external_sources_read_in_full: { type: 'integer', description: 'Count of sources fetched and read IN FULL via WebFetch. Snippets do not count.' },
    sources_read_in_full: { type: 'array', items: { type: 'string' }, description: 'The actual URLs read in full. Cross-checked against the brief on disk.' },
    snippet_only_sources: { type: 'integer' },
    urls_collected: { type: 'integer' },
    recency_scan_performed: { type: 'boolean' },
    internal_files_inspected: { type: 'integer' },
    coverage: {
      type: 'object',
      additionalProperties: false,
      required: ['audit_class', 'rounds', 'dry_rounds', 'K_required', 'new_findings_last_round', 'dry'],
      properties: {
        audit_class: { type: 'boolean' },
        rounds: { type: 'integer' },
        dry_rounds: { type: 'integer' },
        K_required: { type: 'integer' },
        new_findings_last_round: { type: 'integer' },
        dry: { type: 'boolean' },
      },
    },
    brief_path: { type: 'string' },
    summary: { type: 'string', description: 'The design-deciding findings, dense. Not a restatement of the objective.' },
    gate_passed: { type: 'boolean', description: 'Your own honest assessment. The caller RECOMPUTES this; an over-claim is detected.' },
  },
}

/**
 * Recompute the gate from the envelope + the artifact on disk.
 * EXPORTED SHAPE: {gate_passed, violations[], checks[], agent_self_reported_gate_passed, self_report_disagreed}
 *
 * Every floor lives HERE, not in the schema -- see the header. This function is
 * the thing a mutation test must be able to break.
 */
function enforceGate(env, opts) {
  const floors = (opts && opts.floors) || { sources: FLOOR_SOURCES, urls: FLOOR_URLS }
  const violations = []
  const checks = []

  // (0) Empty / errored / non-object return => FAILED gate. Never gate_passed.
  if (!env || typeof env !== 'object' || Array.isArray(env)) {
    return {
      gate_passed: false,
      violations: ['empty_or_errored_return'],
      checks: ['empty_or_errored_return: the agent returned ' + JSON.stringify(env === undefined ? null : env)],
      agent_self_reported_gate_passed: null,
      self_report_disagreed: false,
    }
  }

  const selfReported = env.gate_passed === true

  const n = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : -1)
  const sources = n(env.external_sources_read_in_full)
  const urls = n(env.urls_collected)

  if (sources < floors.sources) violations.push('external_sources_read_in_full=' + sources + ' < floor ' + floors.sources)
  else checks.push('sources_floor_ok: ' + sources + ' >= ' + floors.sources)

  if (urls < floors.urls) violations.push('urls_collected=' + urls + ' < floor ' + floors.urls)
  else checks.push('urls_floor_ok: ' + urls + ' >= ' + floors.urls)

  if (env.recency_scan_performed !== true) violations.push('recency_scan_performed is not true')
  else checks.push('recency_scan_ok')

  const cov = env.coverage || {}
  if (cov.audit_class === true) {
    if (cov.dry !== true) violations.push('audit-class step but coverage.dry is not true (loop-until-dry not reached)')
    else checks.push('audit_class_dry_ok')
  } else {
    checks.push('not_audit_class: coverage.dry informational only')
  }

  // (1) ARTIFACT CROSS-CHECK -- the EviBound lesson. A self-report the artifact
  //     cannot corroborate is an over-claim, and must fail rather than pass.
  const listed = Array.isArray(env.sources_read_in_full) ? env.sources_read_in_full.filter(s => typeof s === 'string' && s.trim()) : []
  if (listed.length < sources) {
    violations.push('over-claim: external_sources_read_in_full=' + sources + ' but sources_read_in_full lists only ' + listed.length)
  } else {
    checks.push('listed_sources_consistent: ' + listed.length + ' >= ' + sources)
  }

  const claimedPath = env.brief_path || briefPath
  const abs = path.isAbsolute(claimedPath) ? claimedPath : path.join(process.cwd(), claimedPath)
  let briefText = null
  try {
    briefText = fs.readFileSync(abs, 'utf8')
  } catch (_e) {
    briefText = null
  }
  if (briefText === null) {
    violations.push('brief not found on disk at ' + claimedPath + ' (write-first not honoured)')
  } else if (!briefText.trim()) {
    violations.push('brief at ' + claimedPath + ' is EMPTY')
  } else {
    checks.push('brief_on_disk_ok: ' + claimedPath + ' (' + briefText.length + ' chars)')
    const missing = listed.filter(u => !briefText.includes(u))
    if (missing.length) {
      violations.push('sources claimed but ABSENT from the brief (' + missing.length + '): ' + missing.slice(0, 5).join(', '))
    } else if (listed.length) {
      checks.push('all_' + listed.length + '_claimed_sources_present_in_brief')
    }
  }

  const gate_passed = violations.length === 0
  return {
    gate_passed,
    violations,
    checks,
    agent_self_reported_gate_passed: selfReported,
    self_report_disagreed: selfReported !== gate_passed,
  }
}

// Exported for the re-runnable checker + mutation tests
// (scripts/qa/verify_research_gate_workflow.mjs). The gate logic must be
// drivable WITHOUT spawning an agent, or it cannot be mutation-tested cheaply.
export { enforceGate, ENVELOPE_SCHEMA, FLOOR_SOURCES, FLOOR_URLS }

phase('Research')
const envelope = await agent(PROMPT, {
  label: 'research-gate:' + stepId,
  phase: 'Research',
  schema: ENVELOPE_SCHEMA,
  agentType: 'researcher',
  model: 'opus',
  effort: 'max',
})

const enforcement = enforceGate(envelope)

if (enforcement.violations.length) {
  log('research-gate ' + stepId + ': GATE FAILED -- ' + enforcement.violations.join(' | '))
} else {
  log('research-gate ' + stepId + ': gate passed (' + enforcement.checks.length + ' checks)')
}
if (enforcement.self_report_disagreed) {
  log('research-gate ' + stepId + ': WARNING -- the agent self-reported gate_passed='
      + enforcement.agent_self_reported_gate_passed + ' but the enforced result is '
      + enforcement.gate_passed + '. The ENFORCED value governs.')
}

// The enforced result governs. `envelope` is the raw self-report, kept so Main
// can transcribe it and see any disagreement.
return {
  step_id: stepId,
  gate_passed: enforcement.gate_passed,
  agent_self_reported_gate_passed: enforcement.agent_self_reported_gate_passed,
  self_report_disagreed: enforcement.self_report_disagreed,
  violations: enforcement.violations,
  checks: enforcement.checks,
  brief_path: (envelope && envelope.brief_path) || briefPath,
  envelope: envelope === undefined ? null : envelope,
}
