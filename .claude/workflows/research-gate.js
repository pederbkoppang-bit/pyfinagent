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

// ---------------------------------------------------------------------------
// NO `import fs` / `import path` HERE, AND THAT IS NOT AN OVERSIGHT.
// The Workflow runtime has NO filesystem or Node API access. A static
// `import fs from 'node:fs'` makes the script UNLAUNCHABLE:
//     SyntaxError: Unexpected identifier 'fs'. import call expects one or two arguments.
// MEASURED 2026-08-09 -- and note `node --check` PASSES on that same file,
// because it is valid ESM. So this step's immutable verification command
// (`node --check ... && ls ...`) reports GREEN on a script that cannot run at
// all. That is the second independent way that command is weaker than it looks
// (the first: it reaches criterion 1 only). Criterion 2 -- a LIVE spawn -- is
// the only check that catches this class, which is exactly why it is a
// criterion.
//
// CONSEQUENCE FOR THE DESIGN: the artifact cross-check cannot read the brief
// from disk inside this script. It is therefore delegated to a second, cheap
// agent (stage 2) which CAN read files, and `enforceGate` is kept PURE -- it
// takes the envelope plus that verification object and touches no I/O. Pure
// also means it is cheaply mutation-testable without spawning anything.
// ---------------------------------------------------------------------------

// ── phase-86.17: THE ARGS BOUNDARY ─────────────────────────────────────────
// This block used to be `catch (_e) { a = {} }` followed by `|| 'UNSPECIFIED'`
// fallbacks, so ANY unparseable input produced a gate that ran with no step id,
// no topic and no scope -- writing research_brief_UNSPECIFIED.md and reporting
// nothing. Measured: 9 of 12 input shapes silently defaulted.
//
// THREE CLASSES, AND THE ORDER MATTERS:
//
//   A. ABSENT      -- `typeof args === 'undefined'` (on a no-args launch the
//                     identifier is genuinely UNBOUND -- MEASURED at $0 / 0
//                     agents, run wf_a1b6c046-b60 returned args_is_bound=false)
//                     or an explicit null. This is the DOCUMENTED dry run: it
//                     must NOT throw. But it may NEVER pass -- there is no
//                     step, no topic and no criteria, so a pass would be a
//                     certificate with no referent. Not throwing and being
//                     allowed to pass are INDEPENDENT concerns; the old code
//                     conflated them.
//   B. UNUSABLE    -- present, but does not reduce to a PLAIN OBJECT: malformed
//                     JSON, a raw newline inside a JSON string, an array, a
//                     scalar, '' , or a DOUBLE-ENCODED JSON string. That last
//                     one PARSES SUCCESSFULLY into a string, so catch-hardening
//                     alone cannot see it -- which is why the plain-object
//                     check is re-applied AFTER the parse. THROW.
//   C. INCOMPLETE  -- a plain object carrying no step_id. A present args object
//                     is proof the caller INTENDED to parameterise, so this is
//                     unambiguously a caller bug rather than a dry run. THROW.
//
// `typeof` is MANDATORY and is not a stylistic choice: the re-runnable checker
// imports this slice with `args` UNBOUND, so a bare `args === undefined` raises
// ReferenceError and kills every one of its checks. Do not "simplify" it.
//
// DO NOT REPAIR NEAR-MISSES (the house idiom, and RFC 9413's anti-workaround
// argument): no second JSON.parse on a double-encoded string, no unwrapping an
// array's first element, no synthesising a step id from a brief path. Throwing
// costs the run at this line, before a single token is spent; silently
// defaulting costs a full max-effort session AND deposits a misfiled artifact.
function classifyArgs(bound, raw) {
  const describe = (v) => {
    let preview
    try { preview = typeof v === 'string' ? v : JSON.stringify(v) } catch (_e) { preview = '(unstringifiable)' }
    preview = String(preview === undefined ? 'undefined' : preview)
    return 'typeof=' + (typeof v) + ' isArray=' + Array.isArray(v)
      + ' len=' + preview.length + ' preview=' + JSON.stringify(preview.slice(0, 80))
  }
  const fail = (why, v) => {
    throw new Error('research-gate: args ' + why + ' (' + describe(v)
      + ') -- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.')
  }

  // CLASS A
  if (!bound || raw === null) {
    return { status: 'dry_run', blind: true, args: {} }
  }

  let v = raw
  if (typeof v === 'string') {
    if (!v.trim()) fail('are PRESENT but an empty/blank string', raw)
    try { v = JSON.parse(v) } catch (_e) { fail('are PRESENT but not parseable as JSON', raw) }
  }

  // Re-checked AFTER the parse ON PURPOSE: a double-encoded JSON string parses
  // to a string, so this is the only place that shape is caught.
  if (typeof v !== 'object' || v === null || Array.isArray(v)) fail('did not reduce to a plain object', raw)

  // CLASS C
  if (!v.step_id && !v.stepId) fail('are a plain object with NO step_id', raw)

  return { status: 'ok', blind: false, args: v }
}

const ARGS_BOUND = typeof args !== 'undefined'
const inputHealth = classifyArgs(ARGS_BOUND, ARGS_BOUND ? args : null)
const a = inputHealth.args

const stepId = a.step_id || a.stepId || 'UNSPECIFIED'
const topic = a.topic || '(no topic passed -- derive it from the step entry in .claude/masterplan.json)'
// ── phase-86.28: ABSENT tier vs UNSUPPORTED tier ───────────────────────────
// These are different in KIND and the old single `tierDefaulted` flag
// collapsed them, which had two consequences:
//   (a) the prompt string at TIER: below asserted "NOT passed by the caller"
//       even when the caller HAD passed one (just an unimplemented one), so
//       the agent was told something factually false; and
//   (b) the substitution never reached the RETURN VALUE, only the prompt.
//
// MEASURED: `.claude/agents/researcher.md` documents a FOURTH tier, `deep`
// (grep its "### `deep` tier" heading -- line numbers are omitted on purpose;
// an earlier revision of this comment cited `:204,206-273`, which this very
// cycle's edit to researcher.md staled by ~7 lines). Its gate conditions are
// materially stricter: >=20 sources read in full vs 5, >=1 [ADVERSARIAL]
// source, an explicitly labelled multi-pass structure.
//
// This rail does not implement it. The ORIGINAL wording operationalised that
// as "`grep -c deep` on this file returns 0" -- which was true when written
// and is now FALSE (it returns 8), because this comment block itself contains
// the word. A self-defeating operationalization is worse than none, so the
// durable check is: `VALID_TIERS` below does not contain 'deep', and every
// occurrence of the word in this file is a comment.
//
// So a caller passing tier:'deep' previously got a gate certified at MODERATE
// standards with nothing in the response saying so.
//
// WHY UNSUPPORTED FAILS CLOSED RATHER THAN WARNING. Protocol design allows
// exactly two dispositions for a caller-named capability the implementation
// lacks: fail closed (TLS inappropriate_fallback RFC 7507 s3; HTTP Expect
// /417; LDAP control criticality), or proceed WITH a machine-readable signal
// in the RESPONSE (RFC 7240 s3 -- `Prefer` is ignorable ONLY because
// `Preference-Applied` exists). The deciding variable is whether the caller
// can DETECT the substitution from the response, not the size of the gap.
// Silent substitution is endorsed by no source; RFC 9413 s6 -- hiding
// consequences conceals bugs. Here `tier` is not an ignorable hint: it
// DEFINES what "passed" means. Certifying gate_passed:true at a standard
// that was never applied would be an over-claim BY THE GATE, the exact
// failure class the header above says this rail exists to prevent. So we do
// BOTH -- fail closed AND report the fields.
//
// Fail-closed breaks no existing caller: zero callers pass `deep` today.
// ABSENT keeps today's behaviour EXACTLY -- defaulting is legitimate when
// the caller named nothing, and that path raises no violation.
//
// NOT IN SCOPE, deliberately: adding 'deep' to VALID_TIERS. researcher.md's
// "Multi-subagent fork option" (grep that heading) makes deep's fourth LISTED
// ELEMENT a CONDITIONAL multi-subagent producer fork -- conditional on the
// caller requesting it OR the topic having >=3 separable sub-questions. An
// earlier revision called it deep's "fourth requirement", which overstated it
// (Q/A wf_10c6cbd2-cad, note N2). Conditional or not, it is still
// ("2-3 parallel deep-tier researcher subagents", "~1 Claude Max 5-hour
// rolling window per subagent"). Enabling the tier would ship producer
// fan-out onto this N=1 artifact rail -- one brief path, one stage-2
// verifier, no cross-branch de-dup -- and pre-empt an open operator
// decision. Report the gap; do not close it unilaterally.
const VALID_TIERS = ['simple', 'moderate', 'complex']
const tierRequested = a.tier || null
const tierAbsent = !tierRequested
const tierSupported = !tierAbsent && VALID_TIERS.includes(tierRequested)
const tierUnsupported = !tierAbsent && !tierSupported
const tier = tierSupported ? tierRequested : 'moderate'
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
  'STEP 0b (binding, phase-86.37): WRITE THE ENVELOPE INTO THE BRIEF EARLY, BORN INERT. Within your first few',
  'tool calls, put the JSON envelope in the brief carrying "brief_status": "INCOMPLETE"; update its counts as',
  'sources land; and as your FINAL act flip it to "COMPLETE". This rail DROPS on a measured fraction of runs --',
  'step 86.29 lost a 25,359-byte brief with 15 sources to one, because the envelope was only ever written at the',
  'tail and the run never reached its tail. A brief whose brief_status is INCOMPLETE, or which carries none at',
  'all, does NOT pass the gate whatever its counts say -- that is enforced, not advisory, so an honest',
  'INCOMPLETE costs you nothing and a missing marker costs you the gate.',
  '',
  'OBJECTIVE: ' + topic,
  // phase-86.28: only the ABSENT case can reach a spawn -- an UNSUPPORTED tier
  // early-returns below without spawning, so a branch for it here would be DEAD
  // CODE (the same trap the blind-run duplicate hit in 86.17 cycle 1). The old
  // single string claimed "NOT passed by the caller" even when the caller HAD
  // passed one, which is why this is now conditioned on tierAbsent alone.
  'TIER: ' + tier + (tierAbsent
    ? '  (NOT passed by the caller -- defaulted to moderate; state this assumption in the brief)'
    : ''),
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

// Stage-2 verifier schema: a cheap, independent read of the brief on disk.
// The researcher does NOT get to attest to its own artifact.
// phase-86.28: two fields added so the gate stops trusting two self-reports
// it never corroborated. Both are read off the SAME brief stage 2 already
// opens -- no new agent, no new spawn, no extra round trip.
//
// NAMING DISCIPLINE (EBTE / Proof-or-Stop: "structural is not semantic").
// `recency_section_present` says a SECTION EXISTS. It does NOT say a recency
// scan was substantively performed, and no field here should ever be read as
// saying that. The check exists to catch the cheap lie (claimed a scan, wrote
// no section), not to certify research quality. `coverage.dry` is deliberately
// NOT given a similar proxy: dryness is K consecutive EXECUTED search rounds
// surfacing nothing new, which is a property of executed discovery and not of
// a file, so any file-derived proxy for it would be false assurance -- exactly
// the anti-pattern EBTE names. Leave it uncorroborated and honest.
const BRIEF_VERIFICATION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: [
    'brief_exists', 'brief_non_empty', 'char_count', 'urls_checked', 'urls_present', 'urls_missing',
    'recency_section_present', 'distinct_urls_in_brief', 'brief_status_in_brief',
  ],
  properties: {
    brief_exists: { type: 'boolean' },
    brief_non_empty: { type: 'boolean' },
    char_count: { type: 'integer' },
    urls_checked: { type: 'integer' },
    urls_present: { type: 'integer' },
    urls_missing: { type: 'array', items: { type: 'string' } },
    recency_section_present: { type: 'boolean', description: 'Does the brief carry a dedicated recency-scan section heading? Structural only -- not a judgement that the scan was substantive.' },
    distinct_urls_in_brief: { type: 'integer', description: 'Count of DISTINCT http(s) URLs appearing anywhere in the brief.' },
    // phase-86.37: the born-inert marker, READ FROM THE BRIEF by an agent that
    // is not the author. "COMPLETE" only if the brief's envelope literally says
    // so; "INCOMPLETE" if it says that; "ABSENT" if there is no such field.
    // ABSENT is NOT folded into INCOMPLETE -- a brief with no marker was not
    // written by the write-first path at all, and saying which is more useful
    // than degrading both to the same word.
    brief_status_in_brief: { type: 'string', enum: ['COMPLETE', 'INCOMPLETE', 'ABSENT'], description: 'The brief_status value inside the brief\'s own JSON envelope. ABSENT when the brief carries no such field.' },
  },
}

/**
 * Recompute the gate from the envelope + an independent verification of the
 * artifact. PURE: no I/O, no Node APIs -- both because the Workflow runtime
 * forbids them and because a pure function is cheaply mutation-testable.
 *
 * @param env  the researcher's returned envelope (may be null/garbage)
 * @param verification  stage-2's reading of the brief, or null if it could not run
 * EXPORTED SHAPE: {gate_passed, violations[], checks[], agent_self_reported_gate_passed, self_report_disagreed}
 *
 * Every floor lives HERE, not in the schema -- see the header. This function is
 * the thing a mutation test must be able to break.
 */
function enforceGate(env, verification, opts) {
  const floors = (opts && opts.floors) || { sources: FLOOR_SOURCES, urls: FLOOR_URLS }
  const violations = []
  const checks = []

  // phase-86.28: the tier classification is computed BEFORE the empty-envelope
  // guard, and the ordering is load-bearing rather than cosmetic. The driver's
  // refuse-to-spawn path calls this with env === null ON PURPOSE (no agent was
  // ever asked to run). Computing tier after the guard made that path report
  // `empty_or_errored_return: the agent returned null` -- describing an agent
  // failure that never happened and hiding the real, actionable cause.
  // MEASURED by the live spawn wf_4da39b31-695 before this reordering.
  //
  // ABSENT tier is NOT a violation -- defaulting is legitimate when the caller
  // named nothing. UNSUPPORTED is, because `tier` defines what "passed" MEANS
  // (deep = >=20 sources, not 5), so certifying at the substituted standard
  // would be an over-claim by the gate itself. See the block above VALID_TIERS
  // for the protocol-design basis (RFC 7507 / 7240 / 9413).
  const tierInfo = (opts && opts.tier) || null
  const tierUnsupportedHere = !!(tierInfo && tierInfo.unsupported === true)
  if (tierUnsupportedHere) {
    violations.push('tier_unsupported: the caller requested tier "' + tierInfo.requested
      + '" which this rail does not implement (supported: ' + (tierInfo.valid || []).join(', ')
      + '). Ran at "' + tierInfo.applied + '". Refusing to certify a standard that was never applied'
      + ' -- pass a supported tier, or implement the requested one.')
  } else if (tierInfo && tierInfo.absent === true) {
    checks.push('tier_absent_defaulted_ok: no tier passed, ran at "' + tierInfo.applied + '"')
  } else if (tierInfo) {
    checks.push('tier_supported_ok: "' + tierInfo.applied + '"')
  }

  // (0) Empty / errored / non-object return => FAILED gate. Never gate_passed.
  if (!env || typeof env !== 'object' || Array.isArray(env)) {
    // When the tier was the reason we never spawned, an absent envelope is a
    // CONSEQUENCE, not an independent finding -- reporting both would tell the
    // caller an agent failed when none was asked to run.
    if (!tierUnsupportedHere) {
      violations.push('empty_or_errored_return')
      checks.push('empty_or_errored_return: the agent returned ' + JSON.stringify(env === undefined ? null : env))
    }
    return {
      gate_passed: false,
      violations,
      checks,
      agent_self_reported_gate_passed: null,
      self_report_disagreed: false,
    }
  }

  // phase-86.17: DEFENCE IN DEPTH, and it is deliberate rather than redundant.
  // Classes B and C throw at the boundary, so they never reach here. Class A
  // (the dry run) legitimately does -- and must not be able to pass. Forcing
  // the violation HERE as well means the gate still fails closed on any future
  // path that bypasses the throw, which is Saltzer's complete mediation applied
  // against a regression in this very fix. Absent inputHealth (the checker's
  // existing calls) behaves exactly as before.
  const inputBlind = !!(opts && opts.inputHealth && opts.inputHealth.blind === true)
  if (inputBlind) {
    violations.push('dry_run_no_step_id: args were ABSENT, so there is no step, topic or scope to certify -- a blind run may never pass')
  } else if (opts && opts.inputHealth) {
    checks.push('input_health_ok: ' + opts.inputHealth.status)
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

  const claimedPath = env.brief_path || 'the declared brief_path'
  // `Array.isArray` is not redundant: `typeof [] === 'object'`, so an array
  // would slip this guard and be read as a verification object whose fields are
  // all undefined -- failing closed, but via the wrong branch and with a
  // misleading message. Mirrors the envelope guard above. (Found by the
  // checker's own array case.)
  if (!verification || typeof verification !== 'object' || Array.isArray(verification)) {
    // FAIL CLOSED. If the artifact could not be independently verified, the
    // self-report stands unchecked -- which is precisely the EviBound failure
    // mode this gate exists to prevent. Never pass on an absent verification.
    violations.push('brief verification did not run (stage 2 returned ' + JSON.stringify(verification === undefined ? null : verification) + ') -- failing closed rather than trusting the self-report')
  } else if (verification.brief_exists !== true) {
    violations.push('brief not found on disk at ' + claimedPath + ' (write-first not honoured)')
  } else if (verification.brief_non_empty !== true) {
    violations.push('brief at ' + claimedPath + ' is EMPTY')
  } else {
    checks.push('brief_on_disk_ok: ' + claimedPath + ' (' + verification.char_count + ' chars, independently read)')
    // phase-86.37: THE BORN-INERT MARKER IS A HARD GATE, and it is checked
    // BEFORE any count, deliberately. A brief that dropped mid-run can be long,
    // well-formed and carry plenty of sources -- 86.29's was 25,359 bytes with
    // 15 of them -- and it is still not a completed piece of research. Letting
    // the counts speak first would let exactly that brief pass on volume.
    // ABSENT is treated as INCOMPLETE for gating (it cannot be shown complete)
    // but is reported DISTINCTLY, because "no marker at all" means the brief was
    // not produced by the write-first path and is a different problem from "the
    // run was cut short".
    // FAIL CLOSED on a missing field. The first version of this block tested
    // only the three known values, so an `undefined` (a stage-2 object that
    // omitted the field) matched none of them and the marker gate silently did
    // NOTHING -- green, and blind. The schema requires the field, but a gate
    // must not depend on the schema having been honoured.
    const briefStatus = (verification.brief_status_in_brief === 'COMPLETE'
      || verification.brief_status_in_brief === 'INCOMPLETE')
      ? verification.brief_status_in_brief
      : 'ABSENT'
    if (briefStatus === 'INCOMPLETE') {
      violations.push('brief at ' + claimedPath + ' declares brief_status=INCOMPLETE -- '
        + 'the run did not reach its final act. A partial brief is EVIDENCE for a re-run, '
        + 'never a gate pass (crash-only: partial output is INFORMATION, never RESULT).')
    } else if (briefStatus === 'ABSENT') {
      violations.push('brief at ' + claimedPath + ' carries NO brief_status marker -- it cannot be '
        + 'shown to be complete, so it does not pass. (Distinct from INCOMPLETE: a brief with no '
        + 'marker was not written by the write-first path at all.)')
    } else if (briefStatus === 'COMPLETE') {
      checks.push('brief_status_in_brief: COMPLETE (the brief declares its own final act ran)')
    }
    const missing = Array.isArray(verification.urls_missing) ? verification.urls_missing : []
    if (missing.length) {
      violations.push('sources claimed but ABSENT from the brief (' + missing.length + '): ' + missing.slice(0, 5).join(', '))
    } else if (listed.length) {
      checks.push('all_' + listed.length + '_claimed_sources_present_in_brief')
    }

    // phase-86.28: corroborate the two self-reports that previously answered
    // only to themselves. Both live INSIDE this branch on purpose -- they need
    // an independently-read brief, and when stage 2 did not run the gate has
    // already failed closed above. Adding them here therefore cannot soften
    // the fail-closed path.
    //
    // (a) recency_scan_performed. .claude/rules/research-gate.md makes a
    //     DEDICATED "Recency scan (last 2 years)" section mandatory and
    //     requires it "even when empty". Claiming the scan while the artifact
    //     carries no such section is an over-claim of the same shape the
    //     source cross-check already catches. STRUCTURAL ONLY -- this does not
    //     assert the scan was substantive, and the check name says so.
    if (env.recency_scan_performed === true && verification.recency_section_present !== true) {
      violations.push('over-claim: recency_scan_performed=true but the brief carries NO dedicated recency-scan section '
        + '(structural check -- .claude/rules/research-gate.md requires the section even when it reports no findings)')
    } else if (env.recency_scan_performed === true) {
      checks.push('recency_section_present_in_brief (structural: a section exists; NOT a judgement that the scan was substantive)')
    }

    // (b) urls_collected. The rules require the snippet-only set to be
    //     recorded in its own table, so every collected URL should be
    //     observable in the brief. Claiming more than the artifact carries is
    //     an over-claim; claiming fewer is fine (a brief may cite extra).
    const briefUrls = n(verification.distinct_urls_in_brief)
    if (urls > briefUrls) {
      violations.push('over-claim: urls_collected=' + urls + ' but only ' + briefUrls
        + ' distinct URLs appear in the brief (the snippet-only set must be recorded there too)')
    } else if (urls >= 0 && briefUrls >= 0) {
      checks.push('urls_collected_corroborated: ' + urls + ' <= ' + briefUrls + ' distinct URLs in the brief')
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

// NO `export { ... }` LIST HERE, AND THAT IS ALSO NOT AN OVERSIGHT.
// The Workflow runtime accepts the leading `export const meta` and NOTHING
// else: a trailing export list makes the script unlaunchable with
//     SyntaxError: Unexpected keyword 'export'
// MEASURED 2026-08-09 -- and, again, `node --check` PASSES on that file. That
// is the THIRD independent way this step's immutable command reports green on
// a script that cannot run (the others: it reaches criterion 1 only, and it
// accepts a forbidden `import fs from 'node:fs'`).
//
// The re-runnable checker therefore appends its own export line to the stripped
// source before importing it -- see scripts/qa/verify_research_gate_workflow.mjs.
// The gate logic must stay drivable WITHOUT spawning an agent, or it cannot be
// mutation-tested cheaply.

phase('Research')

// phase-86.17 (cycle-2 correction, criterion 3 second sentence): REFUSE TO
// SPAWN on a blind run. Cycle 1 marked the run blind and forced gate_passed
// false -- correct, and criterion 4 held -- but it still spawned a max-effort
// researcher whose prompt names handoff/current/research_brief_UNSPECIFIED.md
// and tells it "write-first is non-negotiable". That is the EXACT artifact this
// step's own defect narrative calls the harm: a brief written under an identity
// that collides across every step reaching this path. qa-verdict.js already
// refused to spawn for the same reason; the reasoning simply had not been
// applied here, and the asymmetry was not disclosed. Measured by the cycle-1
// Q/A driving the full driver with args unbound.
//
// Returning here keeps BOTH halves of criterion 4: it does not throw (the dry
// run stays legal) and it cannot pass (gate_passed is false, and no brief can
// be written because no researcher exists to write one).
if (inputHealth.blind) {
  log('research-gate: WARNING -- BLIND RUN. args were ABSENT, so there is no step, topic or '
      + 'scope to research. Returning a FAILED gate and spawning nothing -- no brief may be '
      + 'written under an UNSPECIFIED identity.')
  return {
    step_id: null,
    gate_passed: false,
    dry_run: true,
    input_health: { status: inputHealth.status, blind: true },
    violations: ['dry_run_no_step_id: args were ABSENT, so there is no step, topic or scope to certify -- a blind run may never pass'],
    checks: [],
    brief_path: null,
    brief_verification: null,
    envelope: null,
    agent_self_reported_gate_passed: null,
    self_report_disagreed: false,
    reason: 'BLIND RUN: args were absent, so no step was identified. No researcher was spawned and no brief was written.',
  }
}

// phase-86.28: REFUSE TO SPAWN on an unsupported tier, for the same reason the
// args boundary throws rather than defaulting (see :102-106): the outcome is
// already determined, so spawning costs a full max-effort researcher session
// and produces a brief filed under a standard nobody asked for. Rejecting here
// costs zero tokens and tells the caller exactly what to fix.
//
// The enforceGate check is KEPT as well and is not redundant -- it is the same
// complete-mediation pattern 86.17 applied to the blind run: any future path
// that reaches enforceGate with an unsupported tier still fails closed.
if (tierUnsupported) {
  log('research-gate ' + stepId + ': REFUSING TO SPAWN -- caller requested tier "' + tierRequested
      + '" which this rail does not implement (supported: ' + VALID_TIERS.join(', ') + '). '
      + 'Zero agents spawned. Pass a supported tier, or implement the requested one.')
  const refusal = enforceGate(null, null, { inputHealth, tier: {
    requested: tierRequested, applied: tier, supported: false, absent: false, unsupported: true, valid: VALID_TIERS,
  } })
  return {
    step_id: stepId,
    gate_passed: false,
    agent_self_reported_gate_passed: null,
    self_report_disagreed: false,
    violations: refusal.violations,
    checks: refusal.checks,
    input_health: { status: inputHealth.status, blind: inputHealth.blind },
    tier_requested: tierRequested,
    tier_applied: tier,
    tier_supported: false,
    brief_path: null,
    brief_verification: null,
    envelope: null,
    reason: 'UNSUPPORTED TIER: the caller named a tier this rail does not implement. No researcher was spawned.',
  }
}

// ── phase-86.37: THE STAGE-1 DROP IS SURVIVABLE ────────────────────────────
// This call used to be a bare `await`. When the rail drops -- `agent({schema}):
// subagent completed without calling StructuredOutput` -- that throw killed the
// WHOLE workflow: enforceGate never ran, brief_verification was never computed,
// and the caller got an exception instead of a return. MEASURED on step 86.29,
// 2026-08-10: run wf_f23b7949-ea3 dropped after 181,082 subagent tokens and 68
// tool uses, leaving a 25,359-byte brief with 15 sources on disk that NOTHING
// could assess. Write-first had saved the research; the workflow threw it away.
//
// THE ASYMMETRY WAS THE BUG. Stage 2 below has always been wrapped, setting
// `verification = null // fail closed in enforceGate`. Stage 1 was not. This is
// that same pattern, not a new invention.
//
// WHAT THIS DOES **NOT** DO: it does not turn a drop into a pass.
// `.claude/rules/research-gate.md` -- an empty or errored return is a FAILED
// gate, never `gate_passed`. On a drop `envelope` is null, and
// `enforceGate(null, …)` already fails closed by its EXISTING logic. That is
// deliberate: no new special case decides it, because a special case is
// something a future edit can quietly invert. The floors decide it.
//
// What it DOES do is let the run come back with a RECOVERY REPORT -- the brief's
// on-disk verification plus `rail_dropped` -- so a caller can tell "nearly
// complete, cheap re-run" from "nothing usable". A recovered brief is EVIDENCE
// for the re-run, exactly as phase-86.31 made a recovered Q/A record evidence
// and never a verdict.
// ── 2026-08-14: RETRY A STOCHASTIC StructuredOutput DROP ────────────────────
// Full derivation of the measurement lives in the twin comment in
// `.claude/workflows/qa-verdict.js`. The headline, classified from the run
// record's `error` field alone: across 565 recorded runs the drop rate splits by
// MODEL -- opus-4-8[1m] 0/73 = 0.0%, fable-5 4/135 = 3.0%, opus-5[1m]
// 40/351 = 11.4% -- and the mechanism is UNPROVEN: size, wall-clock, effort and
// the documented preamble-suppression trigger were each tested and refuted.
//
// A NOTE ON WHY THE FIGURES IN THIS BLOCK MOVED (phase-86.81). An earlier
// revision claimed this gate was "the worst-hit caller" and justified
// maxAttempts=3 by that gap. THAT WAS A MEASUREMENT ARTEFACT: the first probe
// matched the error string anywhere in the run record -- and the record embeds
// this file's SOURCE, which quotes the string in the 86.37 comment block above.
// research-gate quoted it more often than qa-verdict did, so research-gate
// looked worse. Classified from the `error` field alone the two are
// indistinguishable: research-gate 6/74 = 8.1%, qa-verdict 35/368 = 9.5%.
// There is no per-workflow amplification and nothing here to explain.
//
// The retracted numbers are deliberately NOT restated here. `qa-verdict.js`
// carries the single retraction notice that names them, so a reader who meets a
// stale figure elsewhere can identify it; repeating them in a second file is how
// a correction turns back into a source. Prefer the re-runnable reader
// `python3 scripts/qa/rail_drop_rate.py` to any number pasted in a comment,
// including these.
//
// maxAttempts stays 3 rather than the qa default of 2, but on a DIFFERENT and
// weaker rationale, stated honestly: a dropped research gate is the more
// expensive loss (a full brief, ~190K tokens, and the step cannot proceed
// without it), not because this caller fails more often. 8.2% -> 0.7% at three
// attempts. Revisit if the reader shows the two converging in cost as well as
// rate: `python3 scripts/qa/rail_drop_rate.py`.
//
// Retrying cannot manufacture a pass: `enforceGate` still RECOMPUTES
// gate_passed from the brief on disk, and an exhausted retry still lands in the
// catch below as a rail drop with gate_passed FALSE.
// The retry is a LOOP AROUND the existing try/catch, deliberately NOT a helper
// function. `verify_research_gate_workflow.mjs:840` locates this spawn with
// SPAWN_RE = /(?:const\s+)?envelope\s*=\s*await agent\(PROMPT/ and then
// proximity-pins the wrapper (nearest `try {` before, `catch` after) and the
// tier refusal's position relative to it. Hoisting the call into a helper
// renamed the literal and turned FIVE guards red -- the probe, not the
// behaviour. Weakening SPAWN_RE to make them green would have blunted a guard
// that exists to catch a real unwrap, so the retry is shaped to satisfy the
// existing guards instead. Keep `envelope = await agent(PROMPT` verbatim.
let envelope = null
let railDropped = null
const STAGE1_MAX_ATTEMPTS = 3
for (let attempt = 1; attempt <= STAGE1_MAX_ATTEMPTS; attempt++) {
  railDropped = null
  try {
    envelope = await agent(PROMPT, {
      label: 'research-gate:' + stepId,
      phase: 'Research',
      schema: ENVELOPE_SCHEMA,
      agentType: 'researcher',
      model: 'opus',
      effort: 'max',
    })
    break
  } catch (e) {
    envelope = null
    railDropped = { dropped: true, error: String((e && e.message) || e).slice(0, 400) }
    const isDrop = railDropped.error.includes('without calling StructuredOutput')
    log('research-gate: STAGE-1 RAIL DROPPED (attempt ' + attempt + '/' +
        STAGE1_MAX_ATTEMPTS + ') -- ' + railDropped.error
        + (isDrop && attempt < STAGE1_MAX_ATTEMPTS
            ? ' | stochastic StructuredOutput drop -- RETRYING'
            : ' | continuing to verify the brief on disk so the run returns a recovery '
              + 'report. gate_passed will be FALSE: an errored return is a FAILED gate.'))
    // Retry ONLY the stochastic drop. Any other error -- a real bug, a refusal,
    // an abort -- must surface on the first occurrence, not be re-run 3x.
    if (!isDrop) break
  }
}

// ---------------------------------------------------------------------------
// STAGE 2 -- independent artifact verification. The researcher does NOT attest
// to its own brief. This runs even when stage 1 returned nothing, so an absent
// envelope still fails closed rather than silently skipping the check.
// Cheap by design: a read-only agent, low effort, one file.
// ---------------------------------------------------------------------------
const claimedBriefPath = (envelope && envelope.brief_path) || briefPath
const claimedUrls = (envelope && Array.isArray(envelope.sources_read_in_full))
  ? envelope.sources_read_in_full.filter(s => typeof s === 'string' && s.trim())
  : []

let verification = null
const STAGE2_MAX_ATTEMPTS = 2
for (let attempt = 1; attempt <= STAGE2_MAX_ATTEMPTS; attempt++) {
try {
  verification = await agent([
    'Verify a research brief on disk. Read-only. Do NOT edit, write or create anything.',
    '',
    'FILE: ' + claimedBriefPath,
    '',
    'Report, factually and without interpretation:',
    '  brief_exists     -- does that exact path exist?',
    '  brief_non_empty  -- does it contain any non-whitespace content?',
    '  char_count       -- its length in characters (0 if absent)',
    '  urls_checked     -- how many URLs you were given below',
    '  urls_present     -- how many of them appear as a literal substring of the file',
    '  urls_missing     -- the exact URLs that do NOT appear',
    '  brief_status_in_brief -- look inside the brief for its OWN JSON envelope and report the value',
    '                           of its `brief_status` field EXACTLY: "COMPLETE", "INCOMPLETE", or',
    '                           "ABSENT" if the brief has no such field. Report what is WRITTEN. Do not',
    '                           infer completeness from the brief looking finished, and do not report',
    '                           COMPLETE because the counts look adequate -- a brief that dropped',
    '                           mid-run can be long, well-formed and still INCOMPLETE.',
    '  recency_section_present -- does the brief carry a DEDICATED recency-scan section heading',
    '                            (a heading such as "Recency scan (last 2 years)" or equivalent)?',
    '                            Report ONLY whether such a section EXISTS. Do NOT judge whether the',
    '                            scan was thorough, correct or substantive -- that is not what is asked,',
    '                            and answering the harder question would make this field mean something',
    '                            the caller does not treat it as meaning. A section that exists and says',
    '                            "no relevant new findings" counts as PRESENT.',
    '  distinct_urls_in_brief  -- the count of DISTINCT http(s) URLs appearing anywhere in the file',
    '                            (de-duplicate exact repeats; count every table, footnote and citation).',
    '',
    'You are an INDEPENDENT check on another agent\'s self-report. It claimed to have read these',
    'URLs in full and to have written them into that brief. Do not give it the benefit of the',
    'doubt and do not repair near-misses: a URL either appears verbatim in the file or it does not.',
    'If the file does not exist, every URL is missing.',
    '',
    claimedUrls.length ? ('URLS TO CHECK (' + claimedUrls.length + '):\n' + claimedUrls.map(u => '  ' + u).join('\n'))
                       : 'URLS TO CHECK: (none were listed by the researcher -- report urls_checked: 0)',
  ].join('\n'), {
    label: 'brief-verify:' + stepId,
    phase: 'Research',
    schema: BRIEF_VERIFICATION_SCHEMA,
    agentType: 'Explore',
    model: 'opus',
    effort: 'low',
  })
  break
  // Stage 2 is retried too. It looks harmless because its catch fails closed,
  // but failing closed still FAILS THE GATE: `verification = null` makes
  // enforceGate reject a brief that may be perfectly good. Measured 10.3% on
  // the low-effort Explore path -- HIGHER than the max-effort roles, which is
  // itself why the "effort causes drops" theory died.
} catch (_e) {
  verification = null // fail closed in enforceGate
  const isDrop = String((_e && _e.message) || _e).includes('without calling StructuredOutput')
  log('research-gate: STAGE-2 brief-verify failed (attempt ' + attempt + '/' +
      STAGE2_MAX_ATTEMPTS + ')' + (isDrop && attempt < STAGE2_MAX_ATTEMPTS
        ? ' -- stochastic StructuredOutput drop, RETRYING'
        : ' -- failing closed; enforceGate will reject'))
  if (!isDrop) break
}
}

const tierInfo = {
  requested: tierRequested,
  applied: tier,
  supported: tierSupported,
  absent: tierAbsent,
  unsupported: tierUnsupported,
  valid: VALID_TIERS,
}

const enforcement = enforceGate(envelope, verification, { inputHealth, tier: tierInfo })

if (enforcement.violations.length) {
  log('research-gate ' + stepId + ': GATE FAILED -- ' + enforcement.violations.join(' | '))
} else {
  log('research-gate ' + stepId + ': gate passed (' + enforcement.checks.length + ' checks)')
}
// phase-86.17 note: the blind-run WARNING is emitted at the early return above,
// not here. A second `if (inputHealth.blind)` block at this point would be DEAD
// CODE -- the early return fires before any agent is spawned, so control cannot
// reach this line while blind. Cycle 1 had exactly that duplicate; the checker's
// anchor-uniqueness assertion is what surfaced it.

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
  // phase-86.17: report the degradation as its OWN field rather than folding it
  // into gate_passed. A caller must be able to tell "failed the floors" apart
  // from "never had a subject".
  input_health: { status: inputHealth.status, blind: inputHealth.blind },
  // phase-86.37: the stage-1 drop is its OWN field, for the same reason
  // input_health is: a caller must be able to tell "failed the floors" from
  // "the rail died mid-run" from "never had a subject". Folding a drop into
  // gate_passed or violations makes those three indistinguishable, and the
  // recovery decision differs for each -- re-run vs fix-then-re-run vs
  // fix-the-caller. null when stage 1 returned normally.
  rail_dropped: railDropped,
  // phase-86.28: the substitution is now detectable FROM THE RESPONSE. This is
  // the RFC 7240 `Preference-Applied` pattern -- a preference may be ignored
  // only because the response says so. Previously this reached the agent
  // PROMPT and nothing else, which is payload, not response: no caller could
  // tell that a requested tier had been swapped for a weaker one.
  // `tier_requested` is null when the caller passed none (the ABSENT case).
  tier_requested: tierRequested,
  tier_applied: tier,
  tier_supported: tierSupported,
  brief_path: claimedBriefPath,
  brief_verification: verification === undefined ? null : verification,
  envelope: envelope === undefined ? null : envelope,
}
