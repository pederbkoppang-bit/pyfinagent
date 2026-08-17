#!/usr/bin/env node
/**
 * phase-86.78 -- re-runnable checker for moving the escalation threshold OUT of the
 * judge and into the caller.
 *
 *   node scripts/qa/verify_escalation_86_78.mjs
 *
 * Exit 0 iff every check passes AND the cardinality floor is met. A checker whose
 * loop covers nothing prints no failures and exits 0, which is indistinguishable
 * from success.
 *
 * It drives the REAL `enforceEscalation` from `.claude/workflows/qa-verdict.js`, not
 * a copy: the source is read, an `export {...}` line is appended to a TEMP copy, and
 * that copy is imported. Same mechanism as verify_research_gate_workflow.mjs, and for
 * the same reason -- the shipped workflow exports only `meta`, and `import fs` inside
 * a workflow is a SyntaxError under the Workflow runtime.
 *
 * `sourceOverride` is the mutation seam: mutation_matrix_86_78.mjs passes mutated
 * source so the RED half of criterion 6 is provable without writing to the tracked file.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const WORKFLOW = path.join(REPO, '.claude/workflows/qa-verdict.js')
const QA_MD = path.join(REPO, '.claude/agents/qa.md')

const EXPECTED_CHECKS = 49
const results = []
const check = (label, ok, detail = '') => {
  results.push([label, !!ok, detail])
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}${detail ? ' -- ' + detail : ''}`)
}
const section = (t) => console.log(`\n${'='.repeat(74)}\n${t}\n${'='.repeat(74)}`)

/**
 * The sibling verifier slices a PREFIX at the driver boundary. That cannot work here:
 * the qa-verdict workflow body carries TOP-LEVEL `return` statements (legal under the
 * Workflow runtime, which wraps the body in an async function; a SyntaxError under
 * ESM) both BEFORE and AFTER `enforceEscalation`. So the function's exact source span
 * is extracted by brace matching instead.
 *
 * This is still the REAL function, byte-for-byte from the shipped file -- not a
 * hand-copy that could drift. `enforceEscalation` is deliberately self-contained
 * (it closes over nothing at module scope), which is what makes the extraction sound;
 * the check below asserts that, so the day it stops being true this fails loudly
 * instead of silently testing a different function.
 */
function extractFn (src, name) {
  const start = src.indexOf(`function ${name} (`) >= 0
    ? src.indexOf(`function ${name} (`)
    : src.indexOf(`function ${name}(`)
  if (start < 0) throw new Error(`qa-verdict.js does not define ${name}`)
  // Skip the PARAMETER LIST before looking for the body brace. `opts = {}` is a
  // default parameter, so a naive indexOf('{') lands on it and the scan closes
  // immediately -- which is exactly what happened on the first run of this checker.
  let p = src.indexOf('(', start)
  let pd = 0
  for (; p < src.length; p++) {
    if (src[p] === '(') pd++
    else if (src[p] === ')') { pd--; if (pd === 0) break }
  }
  let depth = 0, i = src.indexOf('{', p)
  const open = i
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++
    else if (src[i] === '}') { depth--; if (depth === 0) break }
  }
  if (depth !== 0) throw new Error(`unbalanced braces extracting ${name}`)
  const body = src.slice(start, i + 1)
  if (open < 0 || body.length < 200) throw new Error(`${name} extraction looks wrong`)
  return body
}

async function load (sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(WORKFLOW, 'utf8')
  const body = extractFn(src, 'enforceEscalation')
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'qa78_')), 'wf.mjs')
  fs.writeFileSync(tmp, body + '\nexport { enforceEscalation }\n')
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.enforceEscalation !== 'function') {
    throw new Error('qa-verdict.js does not define enforceEscalation')
  }
  return mod
}

// Read the SUBJECT UNDER TEST, never unconditionally the tracked file. Getting this
// wrong makes every source-text assertion blind to a mutant -- the first run of this
// matrix scored M9 (consequence restored to the prompt) as SURVIVED for exactly that
// reason, when the mutation had in fact been applied and simply was not being read.
const OVERRIDE = process.env.PYFIN_QA_VERDICT_OVERRIDE
const SRC = fs.readFileSync(OVERRIDE || WORKFLOW, 'utf8')
const { enforceEscalation: E } = await load(OVERRIDE ? SRC : undefined)

const C = (v) => ({ verdict: v, ok: v === 'PASS' })

// ══════════════════════════════════════════════════ CRITERION 1 (exposure)
section('C1 -- the consequence is GONE from the rail prompt; qa.md still carries it')

const GONE = [
  ['return FAIL instead of a', 'the 3rd-CONDITIONAL trigger'],
  ['recommend operator', 'the F1b escalation consequence'],
  ['at 5+', 'the budget threshold'],
  ['State the derived attempt number', 'the self-count demand'],
  // cycle-5 (cycle-4 Q/A, the 420/420 finding): the rail's own STEP-0 line
  // shipped the rule's value, unit and outcome in EVERY prompt, 60 lines
  // above the deliberately-withheld block -- and the census had
  // misattributed it to qa.md-embedding. The enumeration now says
  // 'the loop-termination rule'; this probe pins the leak's exact phrase.
  ['3rd-CONDITIONAL auto-FAIL rule, and the', 'the STEP-0 rule enumeration (the 420/420 leak)'],
]
// cycle-6 (cycle-5 Q/A, MB): the literal probe above kills only the verbatim
// restoration; a REWORDED consequence ('a third straight CONDITIONAL must be
// returned as FAIL') survived. Content-pin the enumeration LINE itself: it
// must exist, carry the neutral name, and be free of the rule's value/unit/
// outcome tokens in ANY spelling this pin can see. Inherently non-exhaustive
// against novel phrasings -- stated, per the cycle-1 precedent -- but it
// kills the measured MB construction and the ordinal/unit/outcome families.
{
  const enumLine = SRC.split('\n').find(l => l.includes('the loop-termination rule')) || ''
  check('STEP-0 enumeration line exists with the neutral rule name', enumLine.length > 0)
  check('STEP-0 enumeration is free of value/unit/outcome tokens (any spelling)',
    enumLine.length > 0
    && !/3rd|third|CONDITIONAL|auto-?FAIL|straight|consecutive/i.test(enumLine),
    'reworded restorations of the consequence must redden here')
}
for (const [needle, what] of GONE) {
  check(`rail prompt no longer states ${what}`, !SRC.includes(needle),
    `"${needle}" occurs ${SRC.split(needle).length - 1}x`)
}
check('positive control: the EVIDENCE pointer is still there (not a blanket deletion)',
  SRC.includes('verdict_history_86_21.py'))
check('the rail says explicitly that the consequence is withheld ON PURPOSE',
  SRC.includes('THE CONSEQUENCE OF YOUR VERDICT IS DELIBERATELY NOT STATED HERE'))

// ── cycle-1 Q/A cell QA-C: the four literal-string scans above are defeated by
// REWORDING. It appended "a further unresolved outcome must close the loop and be
// raised to the operator" -- no banned literal, same consequence -- and every check
// passed. No string scan can enumerate all phrasings, so the block is CONTENT-PINNED
// instead: any edit to the withheld-on-purpose block must be deliberate.
const BLOCK_START = '// phase-86.78: THE CONSEQUENCE OF YOUR VERDICT IS DELIBERATELY NOT STATED HERE.'
const BLOCK_END = 'RECOMMENDS, the sponsor DECIDES.'
const bs = SRC.indexOf(BLOCK_START)
const be = SRC.indexOf(BLOCK_END)
check('the withheld-on-purpose block is present and well-formed', bs >= 0 && be > bs)
const block = bs >= 0 && be > bs ? SRC.slice(bs, be + BLOCK_END.length) : ''
// Length + a normalised digest. A reworded ADDITION anywhere between the criteria line
// and the end of that block changes one or both.
const norm = block.replace(/\s+/g, ' ').trim()
// MEASURED from the shipped block, not guessed -- the first value here was a guess
// (1180) and the check went red against a correct subject. A probe constant must be
// read off the thing it describes.
const EXPECTED_LEN = 886   // re-derive deliberately if the block is edited on purpose
check('the block has not silently grown or shrunk (rewording guard)',
  Math.abs(norm.length - EXPECTED_LEN) <= 40,
  `normalised length ${norm.length}, expected ~${EXPECTED_LEN}+-40`)

// And the prompt REGION between the criteria sentence and the evidence pointer must not
// acquire new imperative sentences about outcomes. Rather than banning words, assert the
// region is exactly the known block: anything inserted lands here.
const CRIT = "'CONDITIONAL for fixable gaps, FAIL for a criterion miss.',"
const cIdx = SRC.indexOf(CRIT)
// Measure from the END of the criteria line, not its start -- the first version of this
// check included the criteria line itself in the gap and went red against a clean tree.
const gap = cIdx >= 0 && bs > cIdx ? SRC.slice(cIdx + CRIT.length, bs) : null
check('nothing sits between the criteria sentence and the withheld-on-purpose block',
  gap !== null && gap.replace(/[\s',]/g, '') === '',
  `gap=${JSON.stringify(gap === null ? 'ANCHOR MISSING' : gap.slice(0, 120))}`)

// ── cycle-1 Q/A cell QA-F: "escalation alongside, never merged" was asserted in prose
// and nowhere guarded; flattening it survived all 37 checks. Now enforced at runtime.
// THE DETECTION, not just the presence of a guard. The first version of this check
// asserted only that the runtime `leaked` throw EXISTS in the source -- and cell M11
// (spread escalation into the verdict) left that throw untouched, so it SURVIVED. A
// check that the guard exists is not a check that the property holds.
// cycle-5 (cycle-4 Q/A QX2/QX6): the cycle-4 property regex was satisfied by
// a COMMENT token -- '// was: const merged = { ...verdict, escalation, ... }'
// or '/* escalation */' inside the merge -- while the returned object carried
// no escalation at all. Strip comments from the merge STATEMENT before
// asserting, so only executable tokens count.
// The statement is located among EXECUTABLE lines only -- a naive regex over
// SRC matched 'const merged = ...' INSIDE the QX2 '// was:' comment and the
// first version of this fix survived exactly the mutant it targeted (caught
// by driving both mutants before shipping; the drive is in the live_check).
// cycle-6 (cycle-5 Q/A, MN): strip /* */ SPANS from the whole source BEFORE
// splitting into lines -- a block comment's interior line is unprefixed, so
// the per-line prefix filter admitted it and the locator found a commented
// decoy merge before the real one. Span-stripping first makes the interior
// lines vanish entirely; the per-line filter then handles // and stragglers.
const SRC_NO_BLOCKS = SRC.replace(/\/\*[\s\S]*?\*\//g, '')
const execLines = SRC_NO_BLOCKS.split('\n').filter(l => {
  const s = l.trim()
  return !s.startsWith('//') && !s.startsWith('*')
})
const mergeStmt = execLines.find(l => l.includes('const merged = ')) || ''
const mergeStripped = mergeStmt.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/, '')
check('escalation is NESTED in the return, not spread into it',
  // cycle-4 (cycle-3 Q/A blocker B1): assert the PROPERTY, not a whole-line
  // literal -- added sibling keys (86.72's research_routing) must not redden
  // this. cycle-5: the token must survive COMMENT-STRIPPING of the executable
  // merge statement (QX2/QX6 killed), and no spread of escalation may exist.
  /\{ \.\.\.verdict, [^}]*\bescalation\b/.test(mergeStripped)
  && !/\{[^}]*\.\.\.escalation/.test(SRC),
  'flatten must die; comment tokens must not count; added sibling keys may pass')
check('...and the comment-stripper is not vacuous (the raw statement was non-empty)',
  mergeStmt.length > 20 && mergeStripped.includes('escalation'),
  'the merge statement was found and carries the executable token')
check('...and the shipped code ALSO throws at runtime (defence in depth)',
  SRC.includes('const leaked = Object.keys(escalation).filter')
  && /if \(leaked\.length > 0\) \{\s*\n\s*throw new Error/.test(SRC))
check('verdict_unmodified is COMPUTED, not a hardcoded attestation',
  SRC.includes('const untouched = Object.keys(verdict).every(')
  && !SRC.includes('verdict_unmodified: true }'))

// ── THE OTHER HALF OF THE EXPOSURE. This check has been INVERTED.
//
// Through cycles 1-2 it asserted the residual STILL EXISTS, because `qa.md` was
// operator-gated and honesty required the checker to fail if anyone edited it
// quietly. The operator has since directed a FRESH EXECUTOR to apply those edits
// (separation of duties: Main authored the code, so Main must not author the agent
// file it is graded against). The residual is gone, so the check now asserts the
// opposite -- and it is a stronger check than the old one, because a regression in
// either direction is now visible.
const QA = fs.readFileSync(QA_MD, 'utf8')
const CONSEQUENCE_IN_QA_MD = [
  'return **FAIL** instead of a third',
  'recommend operator escalation',
  'You MUST state the derived attempt number',
]
const qaHits = CONSEQUENCE_IN_QA_MD.filter(s => QA.includes(s))
check('qa.md no longer states the CONSEQUENCE of the verdict to the judge',
  qaHits.length === 0,
  `${qaHits.length} of ${CONSEQUENCE_IN_QA_MD.length} probes hit` +
  (qaHits.length ? `: ${JSON.stringify(qaHits)}` : ''))
// Positive control: the file must still exist and still be the Q/A's instructions.
// Without this, deleting qa.md entirely would satisfy the check above.
check('positive control: qa.md is still present and still the Q/A operating doc',
  QA.length > 10000 && QA.includes('verdict_history_86_21'),
  `${QA.length} bytes -- a check for ABSENCE is satisfied by an empty file, so the ` +
  'subject has to be shown to exist')
// ── The consequence also reached the judge through the TOOL's stdout, not only prose
// -- found by the fresh executor, twice. First the `auto-FAIL armed` line; then, after
// that was suppressed, the `consecutive : N` AGGREGATE, which hands the judge the
// boundary's unit and its own position in it. Both are now closed, and both are checked
// BEHAVIOURALLY here by running the command, because a prose edit cannot close a
// channel that runs through a tool and a source scan cannot prove a tool's output.
check('the judge-facing verdict_history invocation carries --evidence-only',
  QA.includes('--evidence-only'),
  'qa.md must not tell the judge to run the default (threshold-printing) form')
check('...and no flagless judge-facing invocation survives',
  !/verdict_history_86_21\.py\s+--step\s+<step_id>\s*$/m.test(QA))

const EV = execFileSync('python3',
  [path.join(REPO, 'scripts/qa/verdict_history_86_21.py'), '--step', '86.21',
    '--evidence-only'], { cwd: REPO, encoding: 'utf8' })
check('DRIVEN: evidence-only output states no threshold and no armed flag',
  !/auto-FAIL armed/.test(EV))
check('DRIVEN: evidence-only output states no AGGREGATE over the sequence either',
  !/^consecutive\s*:/m.test(EV),
  'an aggregate reveals the boundary unit and the judge\'s position in it')
check('DRIVEN positive control: the SEQUENCE itself is still delivered',
  /verdicts\s*:\s*\S+\s*->\s*\S+/.test(EV),
  'withholding the aggregate must not withhold the evidence')
const DEF = execFileSync('python3',
  [path.join(REPO, 'scripts/qa/verdict_history_86_21.py'), '--step', '86.21'],
  { cwd: REPO, encoding: 'utf8' })
check('DRIVEN: the DEFAULT output is unchanged for operators (who decide, not grade)',
  /auto-FAIL armed/.test(DEF) && /^consecutive\s*:/m.test(DEF),
  'the flag must scope the suppression to the judge, not remove the capability')
check('the ADR filename cited to the judge does not itself name the bound',
  !/adr\/[^\s`)]*(attempt|consecutive)/i.test(QA),
  'a path is text the judge reads; naming both units in it re-opens the channel')

// ══════════════════════════════════════════════════ CRITERION 3 (relocation)
section('C3 -- the threshold is computed caller-side, from data the judge never sees')

const r0 = E(C('CONDITIONAL'), [])
check('empty history -> 0 consecutive, not armed',
  r0.consecutive_conditionals === 0 && r0.would_auto_fail === false)

const r1 = E(C('CONDITIONAL'), ['CONDITIONAL'])
check('1 prior CONDITIONAL -> not armed', r1.would_auto_fail === false,
  `consecutive=${r1.consecutive_conditionals}`)

const r2 = E(C('CONDITIONAL'), ['CONDITIONAL', 'CONDITIONAL'])
check('2 prior CONDITIONALs + a third -> ARMED (the loop terminates)',
  r2.would_auto_fail === true, `consecutive=${r2.consecutive_conditionals}`)

const rReset = E(C('CONDITIONAL'), ['CONDITIONAL', 'CONDITIONAL', 'PASS'])
check('a PASS RESETS the run', rReset.would_auto_fail === false,
  `consecutive=${rReset.consecutive_conditionals}`)
const rResetF = E(C('CONDITIONAL'), ['CONDITIONAL', 'CONDITIONAL', 'FAIL'])
check('a FAIL RESETS the run', rResetF.would_auto_fail === false)

const rDrop = E(C('CONDITIONAL'), ['CONDITIONAL', 'NO_VERDICT', 'CONDITIONAL'])
check('NO_VERDICT is a dropped ATTEMPT: it neither extends nor resets the run',
  rDrop.consecutive_conditionals === 2 && rDrop.would_auto_fail === true,
  `consecutive=${rDrop.consecutive_conditionals} -- a dropped spawn is not a verdict`)

check('the judge is never given any of this: the function runs AFTER agent() returns',
  SRC.indexOf('const escalation = enforceEscalation(')
    > SRC.indexOf('const verdict = await agent('))
check('...and the flag recording that fact is present and false',
  E(C('PASS'), []).judge_was_told_consequence === false)

// The 36.17 regression the attempt-count trigger would have broken.
const r3617 = E(C('PASS'), ['CONDITIONAL', 'FAIL', 'FAIL', 'CONDITIONAL', 'CONDITIONAL'])
check('step 36.17 (C,F,F,C,C) -> a 6th attempt is NOT force-failed',
  r3617.would_auto_fail === false,
  'the attempt-count trigger would have denied the PASS 36.17 earned at attempt 6')

// ══════════════════════════════════════════════════ CRITERION 4 (semantics)
section('C4 -- verdict semantics UNCHANGED; no path turns a FAIL into a PASS')

for (const v of ['PASS', 'CONDITIONAL', 'FAIL', 'NO_VERDICT']) {
  const seqs = [[], ['CONDITIONAL', 'CONDITIONAL'], ['FAIL'], null, ['bogus']]
  const changed = seqs.some(s => {
    const inp = C(v)
    const before = JSON.stringify(inp)
    E(inp, s)
    return JSON.stringify(inp) !== before
  })
  check(`a ${v} input verdict is never mutated by the threshold, under any sequence`,
    !changed)
}
check('would_auto_fail can only ARM on a CONDITIONAL -- never on a FAIL',
  E(C('FAIL'), ['CONDITIONAL', 'CONDITIONAL']).would_auto_fail === false)
check('...and never on a PASS either (arming a PASS would be a downgrade path)',
  E(C('PASS'), ['CONDITIONAL', 'CONDITIONAL']).would_auto_fail === false)
check('the returned object carries NO writable verdict field of its own',
  !('verdict' in E(C('FAIL'), [])))
check('a dropped rail return is passed through unchanged (NO VERDICT, never PASS)',
  SRC.includes('if (verdict == null || typeof verdict !== \'object\') {'))

// ══════════════════════════════════════════════════ fail-closed
section('C4b -- an uncomputable sequence yields null, NEVER 0')

const rNone = E(C('CONDITIONAL'), undefined)
check('sequence not supplied -> null, not 0',
  rNone.consecutive_conditionals === null && rNone.would_auto_fail === null,
  `status=${rNone.sequence_status}`)
const rBad = E(C('CONDITIONAL'), ['CONDITIONAL', 'WAT'])
check('unparseable sequence -> null, not 0',
  rBad.consecutive_conditionals === null,
  `status=${rBad.sequence_status}`)
const rNotArr = E(C('CONDITIONAL'), 'CONDITIONAL')
check('a non-array -> unusable, null, not 0',
  rNotArr.sequence_status === 'unusable' && rNotArr.consecutive_conditionals === null)
check('a spurious 0 would falsely report "no consecutive run" -- assert it is absent',
  rNone.consecutive_conditionals !== 0 && rBad.consecutive_conditionals !== 0)

// ══════════════════════════════════════════════════ CRITERION 5 (safeguards)
section('C5 -- the two law-of-the-case safeguards')

const s = E(C('CONDITIONAL'), ['CONDITIONAL'])
check('safeguard 1: the BURDEN is named, and sits on the departing party',
  typeof s.burden_on === 'string' && /departing/.test(s.burden_on), s.burden_on)
check('safeguard 2: an override SLOT exists on the caller side',
  'override' in s && 'override_reason' in s)
check('...and it defaults to null -- an override must be recorded, never implied',
  s.override === null && s.override_reason === null)
check('the JUDGE cannot record one: VERDICT_SCHEMA is additionalProperties:false',
  /additionalProperties:\s*false/.test(SRC))
check('the input is echoed back, so what the caller supplied is auditable',
  JSON.stringify(s.sequence_supplied) === JSON.stringify(['CONDITIONAL']))

// ══════════════════════════════════════════════════ budget
section('C5b -- the attempt budget is also caller-side, and also fails closed')

check('no attempt number supplied -> budget_exhausted is null, not false',
  E(C('CONDITIONAL'), []).budget_exhausted === null)
check('attempt 4 of 5 -> not exhausted',
  E(C('CONDITIONAL'), [], { attempt_number: 4 }).budget_exhausted === false)
check('attempt 5 of 5 -> exhausted',
  E(C('CONDITIONAL'), [], { attempt_number: 5 }).budget_exhausted === true)
check('exhaustion does NOT touch the verdict (it escalates, it never passes)',
  E(C('FAIL'), [], { attempt_number: 9 }).would_auto_fail === false)

// ══════════════════════════════════════════════════ result
section('RESULT')
const failed = results.filter(r => !r[1])
console.log(`  checks run : ${results.length}   (cardinality floor ${EXPECTED_CHECKS})`)
console.log(`  failed     : ${failed.length}`)
for (const [l, , d] of failed) console.log(`    FAIL ${l} -- ${d}`)
if (results.length < EXPECTED_CHECKS) {
  console.log(`  *** CARDINALITY FLOOR BREACHED: ${results.length} < ${EXPECTED_CHECKS}`)
}
const ok = failed.length === 0 && results.length >= EXPECTED_CHECKS
console.log(`\n  ${ok ? 'ALL CHECKS PASS' : 'CHECKER RED'}`)
process.exit(ok ? 0 : 1)
