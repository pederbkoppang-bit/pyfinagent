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
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const WORKFLOW = path.join(REPO, '.claude/workflows/qa-verdict.js')
const QA_MD = path.join(REPO, '.claude/agents/qa.md')

const EXPECTED_CHECKS = 30
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
]
for (const [needle, what] of GONE) {
  check(`rail prompt no longer states ${what}`, !SRC.includes(needle),
    `"${needle}" occurs ${SRC.split(needle).length - 1}x`)
}
check('positive control: the EVIDENCE pointer is still there (not a blanket deletion)',
  SRC.includes('verdict_history_86_21.py'))
check('the rail says explicitly that the consequence is withheld ON PURPOSE',
  SRC.includes('THE CONSEQUENCE OF YOUR VERDICT IS DELIBERATELY NOT STATED HERE'))

// The other half of the exposure -- measured, not assumed, and NOT claimed fixed.
const QA = fs.readFileSync(QA_MD, 'utf8')
const qaHits = [
  'return **FAIL** instead of a third',
  'recommend operator escalation',
  'You MUST state the derived attempt number',
].filter(s => QA.includes(s))
check('qa.md STILL carries the consequence (disclosed, operator-gated, NOT fixed here)',
  qaHits.length > 0,
  `${qaHits.length} of 3 probes hit -- this check asserts the residual EXISTS, so it ` +
  'goes red if someone quietly edits qa.md without the operator')

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
