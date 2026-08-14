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
// TWO DIFFERENT THINGS SNAPSHOT HERE, AND THIS COMMENT USED TO CONFLATE THEM
// (corrected phase-86.81). It read: "any qa.md edit is live immediately on THIS
// path (no roster snapshot; the snapshot caveat binds only the Agent-tool `qa`
// type)". Both halves need qualifying, and the second was measured false.
//
//  1. THE SCRIPT. `Workflow({name:'qa-verdict'})` dispatches a SESSION SNAPSHOT
//     of this file, not the file on disk. MEASURED 2026-08-14: three named
//     dispatches in one session, launching 07:37:05Z / 08:11:45Z / 09:04:38Z,
//     all carried a byte-identical 18,321-char script matching a commit from
//     00:28:27Z -- up to 8h36m stale, across two intervening commits.
//     `Workflow({scriptPath})` does NOT have this problem: it delivers the
//     on-disk file at dispatch, confirmed three ways -- an 88-second pickup of a
//     fresh commit, a 102-second pickup of another, and a 62-second A/B in which
//     a NAME dispatch took 18,321 chars while a scriptPath dispatch one minute
//     later took 22,961. **So launch by scriptPath whenever the file may have
//     changed this session.** A named launch of a file you just edited runs the
//     old code and reports success.
//  2. THE AGENT DEFINITION. The prompt below tells the agent to read
//     `.claude/agents/qa.md` at runtime, so an ADDITION to qa.md is visible on
//     this path without a restart. A DELETION is not: `agentType:'qa'` also
//     supplies qa.md as the SYSTEM PROMPT, and that IS snapshotted at session
//     start, so a runtime read can only ever ADD text to what the agent already
//     carries -- it can never retract it. Removing a directive still requires a
//     session restart, on this path as much as on the Agent-tool one.
//
// Keep the agent on model:'opus' (rider-trap R4:
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
  'CONDITIONAL for fixable gaps, FAIL for a criterion miss.',
  '',
  '// phase-86.78: THE CONSEQUENCE OF YOUR VERDICT IS DELIBERATELY NOT STATED HERE.',
  'Grade the evidence on its merits. What happens as a RESULT of your verdict -- any',
  'threshold, any escalation, any loop-termination rule -- is computed by the CALLER',
  'AFTER you return, and it is not your concern while grading. This is not an oversight:',
  'arXiv 2604.15224 held content constant, varied a single consequence sentence across',
  '18,240 judgments, and measured judges becoming LENIENT in 58 of 72 cells (p<0.001,',
  'peak -9.8pp) -- and REWARD framing was as lenient as punishment framing, so a',
  'reassuring consequence is no safer than a threatening one. The effect is invisible in',
  'chain-of-thought (ERRJ=0.000), so you would not be able to tell it had happened.',
  'The shape is the sibling research-gate rail\'s, and clinical trials\': the board',
  'RECOMMENDS, the sponsor DECIDES.',
  '',
  'You still need the prior-verdict EVIDENCE to reason about this step, and you should',
  'gather it -- just not as a trigger. `python scripts/qa/verdict_history_86_21.py',
  '--step <step_id>` reads handoff/verdict_ledger.jsonl, prints the sequence, and returns',
  'a STATUS -- ok / no_rows_for_step / ledger_missing / ledger_empty / unparseable --',
  'where the last three report None and FAIL CLOSED rather than printing 0. Do NOT',
  'hand-roll a sequence, and do NOT infer verdicts by scanning prior_records bodies for',
  'the words PASS/CONDITIONAL/FAIL: only 3 of 46 records carry a parseable verdict line',
  'and the bodies DISCUSS verdicts at length (86.21: 15x "CONDITIONAL" in a record whose',
  'verdict was FAIL). If the sequence cannot be established, write sequence: UNKNOWN and',
  'do NOT guess. Main\'s own disclosure is ADVISORY ONLY, since Main is the constrained',
  'party. harness_log is a secondary cross-check only, and it is written in the LOG phase',
  'AFTER the verdict, so it never contains the in-flight cycle.',
  '',
  'CHECK source_present FIRST (phase-86.21): a count of 0 is a fact about ATTEMPTS only',
  'when source_present is true. If it is false the sink does not exist, so 0 means the',
  'counter has NO INPUT, not "attempt 1". `python scripts/qa/qa_wip.py <step_id>',
  '--spawned-at <your-WRITTEN-stamp>` reports attempt_number / prior_attempts',
  '(phase-86.79); PASS --spawned-at, or attempt_number is null by design because no',
  'record can be shown to belong to THIS spawn. null is NEVER 0. records_retained is NOT',
  'the attempt number: it counts retained record FILES, includes your own write-first',
  'record, and pruning can LOWER it -- a gauge, not a counter.',
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
/**
 * phase-86.78 -- THE THRESHOLD IS COMPUTED HERE, AFTER THE VERDICT IS IN HAND.
 *
 * WHY THIS IS NOT INSIDE THE JUDGE
 * --------------------------------
 * Telling a judge what its verdict will TRIGGER shifts the verdict. arXiv 2604.15224
 * held content strictly constant, varied one consequence sentence, and over 18,240
 * judgments measured LENIENCY in 58 of 72 cells (p<0.001, peak -9.8pp). Reward framing
 * ("high scores will be deployed") was as lenient as punishment framing, so the fix is
 * not a gentler consequence -- it is NO consequence at grading time. And the effect is
 * invisible in chain-of-thought (ERRJ = 0.000; corroborated by CAR = 0 in 2509.26072v2
 * and Anthropic's >99%-exploited / <2%-verbalised), so it cannot be audited by reading
 * the judge's own `notes`. Score inside, threshold outside -- the DSMB shape (the board
 * RECOMMENDS, the sponsor DECIDES) and the shape `research-gate.js::enforceGate`
 * already uses on the sibling rail.
 *
 * PURE BY NECESSITY AND BY DESIGN. The Workflow runtime has no filesystem access, so
 * this cannot read handoff/verdict_ledger.jsonl -- the sequence must arrive as data via
 * `args.verdict_sequence`. That is also the honest shape: the caller supplies its input
 * and this function echoes it back, so the input is auditable rather than implicit.
 *
 * IT CANNOT CHANGE A VERDICT. The result is returned ALONGSIDE the verdict, never
 * merged into it. There is no branch here that writes `verdict`, and in particular no
 * path from any input to turning a FAIL into a PASS.
 *
 * FAILS CLOSED. An absent or unusable sequence yields `null`, never `0` -- a spurious
 * zero would silently report "no consecutive run" and suppress termination.
 *
 * NOT `export`ed, deliberately. The shipped workflow exports ONLY `meta` -- the
 * sibling `research-gate.js` keeps `enforceGate` as a plain function for the same
 * reason. The checker drives the REAL function (never a copy) by appending an
 * `export {...}` line to a temp copy of this file and importing that, exactly as
 * `scripts/qa/verify_research_gate_workflow.mjs` does.
 */
function enforceEscalation(verdict, sequence, opts = {}) {
  const maxAttempts = opts.max_attempts ?? 5
  const out = {
    // What the caller was given, echoed back so the input is auditable.
    sequence_supplied: Array.isArray(sequence) ? sequence.slice() : null,
    sequence_status: 'ok',
    consecutive_conditionals: null,
    would_auto_fail: null,
    attempt_number: opts.attempt_number ?? null,
    budget_exhausted: null,
    max_attempts: maxAttempts,
    // Criterion 5, safeguard 1: the burden sits on the party seeking to depart from
    // the computed result, not on the result. The judge's verdict stands by default
    // (law of the case: the prior decision "should continue to govern").
    burden_on: 'the party departing from the computed escalation',
    // Criterion 5, safeguard 2: VERDICT_SCHEMA is additionalProperties:false, so the
    // JUDGE cannot record an override -- correct, it is not the party that should.
    // The CALLER records it here and in the ledger row's free-text `note`.
    override: null,
    override_reason: null,
    judge_was_told_consequence: false,
  }

  if (!Array.isArray(sequence)) {
    out.sequence_status = sequence === undefined || sequence === null
      ? 'not_supplied' : 'unusable'
    return out
  }
  const VALID = new Set(['PASS', 'CONDITIONAL', 'FAIL', 'NO_VERDICT'])
  if (sequence.some(v => !VALID.has(v))) {
    out.sequence_status = 'unparseable'
    return out          // null, NOT 0 -- see FAILS CLOSED above
  }

  // Consecutive CONDITIONALs at the END, RESET on PASS or FAIL. Same rule as
  // verdict_history_86_21.py::consecutive_conditionals. NO_VERDICT is a dropped
  // attempt: it is not a verdict, so it neither extends nor resets the run.
  let n = 0
  for (let i = sequence.length - 1; i >= 0; i--) {
    const v = sequence[i]
    if (v === 'NO_VERDICT') continue
    if (v === 'CONDITIONAL') n++
    else break
  }
  out.consecutive_conditionals = n
  // A THIRD consecutive CONDITIONAL is the terminating one, so two priors arms it.
  out.would_auto_fail = n >= 2 && verdict?.verdict === 'CONDITIONAL'
  if (typeof out.attempt_number === 'number') {
    out.budget_exhausted = out.attempt_number >= maxAttempts
  }
  return out
}

// ── 2026-08-14: RETRY A STOCHASTIC StructuredOutput DROP ────────────────────
// MEASURED over 562 recorded workflow runs on this machine. A drop is the
// runtime throwing `agent({schema}): subagent completed without calling
// StructuredOutput` -- the turn ends, no schema call is emitted, tokens are
// spent, nothing returns.
//
// CORRECTED 2026-08-14, SAME DAY, BEFORE ANY OF THESE FIGURES WERE RELIED ON.
// The first measurement classified a run as dropped if the error string
// appeared ANYWHERE in its record. The record embeds the workflow's own SOURCE,
// and these files quote that string in comments -- including this one. So the
// probe matched itself: 38 of 81 "drops" were comment text. Corrected figures,
// classified ONLY from the record's `error` field:
//
//   by model:     claude-opus-4-8[1m]  0/73   0.0%
//                 claude-fable-5       4/135  3.0%
//                 claude-opus-5[1m]   39/349 11.2%
//   by workflow:  qa-verdict          34/367  9.3%
//                 research-gate        6/73   8.2%
//   overall                           43/563  7.6%
//
// THE "MODEL SPLIT" IS AN ARTEFACT -- RETRACTED phase-86.84 (2026-08-14). This
// line used to read: `P(0 drops in 73 | true rate 11.2%) = 2e-4 -- the model
// split SURVIVES.` The arithmetic was right and the conclusion was wrong,
// because the test asked whether opus-4-8[1m]'s zero could be chance while
// assuming those 73 spawns were exchangeable with the other model's. They were
// not: 223 of opus-4-8[1m]'s 258 spawns were uncapped `general-purpose`, an
// agent type that has dropped 0 times in 930 spawns on EVERY model. The zero is
// explained by WHAT THAT MODEL RAN, so a p-value against a pooled rate is
// testing a hypothesis nobody should have entertained. Hold the model fixed at
// opus-5[1m] and the real split appears: 47/379 on the two agent types that
// carried a `maxTurns` cap, 0/417 on the three that did not. Do not requote the
// 2e-4 or any per-model rate above as evidence about a MODEL.
//
// WHAT DID NOT SURVIVE, and must not be requoted from anywhere: an overall
// "21.8%", a research-gate "53.4%", a "4x amplification" between the two
// workflows (they are 9.3% vs 8.2% -- indistinguishable), and an "August 10
// regression" (the daily rate bounces 0-23% throughout, and 2026-08-13 is
// 0.0%). The re-runnable reader is `scripts/qa/rail_drop_rate.py`, which uses
// the corrected predicate -- prefer it to any number pasted in a comment.
//
// THE MECHANISM IS PROVEN AS OF phase-86.84 (2026-08-14). This block used to
// say it was UNPROVEN; that is SUPERSEDED, and the retry below is now a
// belt-and-braces measure rather than the only defence.
//
// THE CAUSE IS TURN-BUDGET EXHAUSTION. `.claude/agents/qa.md` carried
// `maxTurns: 30` and `researcher.md` carried `maxTurns: 40`. Measured over 572
// run records / 1325 spawns: 39 of 302 `qa` spawns and 9 of 93 `researcher`
// spawns dropped, and EVERY ONE sat at exactly its cap -- the set of turn
// counts on dropped spawns is {30} and {40}, no other value. The three agent
// types with no cap dropped 0 times in 930 spawns and reach 93 turns.
// `maxTurns` counts TOOL-USE turns only and StructuredOutput is itself a tool
// call, so the last permitted turn goes to ordinary work and there is none left
// to emit the schema call. Both pins were REMOVED in phase-86.84.
//   Re-runnable: `python3 scripts/qa/rail_turn_cap.py --verify`
//   Write-up:    handoff/current/live_check_86.84.md
//
// THE FOUR REFUTED HYPOTHESES STAY REFUTED -- prompt/run size, wall clock,
// effort, and the documented preamble-suppression trigger were each correctly
// ruled out, and each is CONSISTENT with turn exhaustion (prompt size does not
// change how many turns an investigation needs, which is why a lean prompt
// still dropped). Turn count is a FIFTH hypothesis those four never tested.
//
// THE MODEL SPLIT REPORTED ABOVE IS CONFOUNDED, and this is the correction that
// matters most for anyone re-reading the old numbers: 223 of
// claude-opus-4-8[1m]'s 258 spawns were uncapped `general-purpose`, a type that
// has never dropped on any model, so its clean 0.0% measured what it RAN.
// Holding the model fixed at claude-opus-5[1m]: 47/379 on the two capped roles
// against 0/417 on the three uncapped ones.
//
// AND THE OLD TOOL-COUNT LINE WAS THE CLUE, MISREAD. It said the rate "rises
// 5%->42% across bands but collapses to 0% above 100 calls, so it is an
// association, not a dose-response." The collapse is not evidence against a
// dose-response -- it is the cap. Nothing capped can REACH 100 tool calls, so
// the only spawns in that band are uncapped types, which never drop.
//
// WHY A RETRY IS NONETHELESS CORRECT, AND NOT A GUESS: the failure is
// STOCHASTIC, and that is measured rather than assumed. Grouping runs by sha1 of
// the embedded `script`, EIGHT distinct byte-identical script versions produced
// BOTH outcomes; the largest dropped 17 times and completed 179. Same bytes,
// same caller, both results -- so a second attempt is asking a question that can
// come back differently. Cost is bounded to the failing subset; a drop otherwise
// costs the whole run AND a manual re-drive.
//
// NO EFFECTIVENESS FIGURE IS STATED HERE, AND THAT IS DELIBERATE (phase-86.81).
// The obvious `p -> p^2` arithmetic assumes attempts are independent, and
// ReliabilityBench (arXiv 2601.06112) refutes that assumption in BOTH directions:
// Gemini 2.0 Flash measures pass^2 = 91.04% where independence predicts 93.86%,
// while GPT-4o lands essentially AT independence. So a squared figure is an UPPER
// BOUND on the benefit, never a forecast. The only honest number is a measured
// conditional rate P(drop on attempt 2 | drop on attempt 1), which needs real
// second attempts on real drops -- of which this repo has, so far, zero.
//
// DO NOT "SIMPLIFY" THIS TO A MODEL PIN. Pinning to claude-opus-4-8 would also
// take the rate to ~0, and was explicitly REJECTED by the operator on
// 2026-08-14: Opus 5's reasoning and self-evaluation are the reason it is here,
// and the gate roles are exactly where that capability matters.
//
// A drop on the FINAL attempt still throws, so the caller's "empty return is NO
// VERDICT, never PASS" rule below is untouched. Retrying a non-answer can never
// manufacture a verdict -- it only asks the same question again.
async function agentRetryingDrops(prompt, opts, maxAttempts = 2) {
  let lastErr = null
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await agent(prompt, opts)
    } catch (e) {
      lastErr = e
      const msg = String((e && e.message) || e)
      // Retry ONLY the structured-output drop. Any other error (a real bug, a
      // refusal, an abort) must surface immediately rather than be re-run.
      if (!msg.includes('without calling StructuredOutput')) throw e
      log('qa-verdict: StructuredOutput DROP on attempt ' + attempt + '/' +
          maxAttempts + (attempt < maxAttempts ? ' -- retrying' : ' -- exhausted, NO VERDICT'))
    }
  }
  throw lastErr
}

const verdict = await agentRetryingDrops(PROMPT, {
  label: 'qa-verdict:' + stepId,
  phase: 'QA',
  schema: VERDICT_SCHEMA,
  agentType: 'qa',
  model: 'opus',
  effort: 'max',
})

// A DROPPED RAIL IS NO VERDICT, NEVER PASS. Wrapping a null/errored return in an
// object that carries an `escalation` key would make a drop LOOK like a result, so the
// null case is returned unmistakably empty-of-verdict and the caller's own
// "empty return is NO VERDICT" rule still applies unchanged.
if (verdict == null || typeof verdict !== 'object') {
  return verdict
}
// The verdict is returned UNCHANGED. `escalation` sits beside it, NEVER inside it.
//
// phase-86.78 cycle-1 Q/A, cell QA-F: flattening this (`...escalation`) SURVIVED the
// whole checker, because "alongside, never merged" was a property asserted in prose and
// nowhere guarded. Merged, the CALLER's fields (would_auto_fail, burden_on, override)
// would surface as top-level siblings of ok/verdict/reason in the object Main
// transcribes VERBATIM -- caller output presented as judge output, the exact doer/judge
// blur this step exists to prevent. So it is now checked at RUNTIME, not just in a test.
const escalation = enforceEscalation(verdict, args?.verdict_sequence, {
  attempt_number: args?.attempt_number,
  max_attempts: args?.max_attempts,
})
const merged = { ...verdict, escalation }
// Exactly ONE returned key may come from the escalation object, and it must be the
// nested one. If a future edit spreads it, this throws rather than silently shipping
// caller fields as judge fields.
const leaked = Object.keys(escalation).filter(k => k !== 'escalation' && k in merged)
if (leaked.length > 0) {
  throw new Error('phase-86.78 invariant violated: escalation fields leaked into the '
    + 'verdict object as top-level siblings: ' + leaked.join(', '))
}
// Computed, not attested. The cycle-1 Q/A noted that a hardcoded `true` would still
// read true if the verdict HAD been modified, which is an attestation, not a check.
const untouched = Object.keys(verdict).every(k => merged[k] === verdict[k])
return { ...merged, verdict_unmodified: untouched }
