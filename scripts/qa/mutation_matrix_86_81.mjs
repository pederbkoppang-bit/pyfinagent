#!/usr/bin/env node
/**
 * phase-86.81 criterion 4 -- mutation matrix for the StructuredOutput drop RETRY.
 *
 *   node scripts/qa/mutation_matrix_86_81.mjs
 *
 * Each mutant is a COPY of `.claude/workflows/qa-verdict.js` under a temp name, fed to
 * verify_rail_retry.mjs through the `PYFIN_QA_VERDICT_OVERRIDE` seam. The tracked file
 * is never opened for writing, and its sha256 is compared before and after so "I did
 * not touch it" is PROVEN, not asserted.
 *
 * A GREEN CONTROL RUNS FIRST -- a cell that "kills" an already-red checker proves
 * nothing.
 *
 * ANCHOR UNIQUENESS IS CHECKED -- `String.replace` on a non-matching anchor returns a
 * perfectly normal string, so a non-mutation would otherwise be scored as a kill.
 *
 * EACH CELL MUST DISCRIMINATE -- red alone is not a kill. The cell names the assertion
 * it is aimed at, and that assertion must be among the failures. A mutant that goes red
 * only by breaking something else has not been killed by the guard it was testing.
 *
 * OPERATOR CHOICE IS EVIDENCE-LED, NOT ARBITRARY. EMSE 2021 (12,331 mutants) ranks
 * exception-handling mutants by survival: CBR 100%, CBI ~88%, CRE ~85%, CBD ~84%,
 * TSD ~75%, FBD 59% -- the operators that DELETE OR DIVERT A THROW survive most often.
 * M4 and M5 are therefore TSD on the two throws, and M4 is the semantically nastiest
 * mutant in the file: it makes the retry swallow a REAL bug and re-run it at ~185K
 * tokens an attempt.
 *
 * Exit 0 = every cell KILLED by the assertion it was aimed at.
 */
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const SUBJECT = path.join(REPO, '.claude/workflows/qa-verdict.js')
const CHECKER = path.join(REPO, 'scripts/qa/verify_rail_retry.mjs')

const MUTATIONS = [
  {
    id: 'M1-DELETE-DROP-STRING-GUARD',
    desc: 'remove the `if (!msg.includes(...)) throw e` guard entirely, so EVERY error '
        + '-- a refusal, a transport failure, a real bug -- is retried',
    anchor: "      if (!msg.includes('without calling StructuredOutput')) throw e",
    replacement: "      // MUTANT M1: guard deleted",
    expect: 'A3b ...with NO retry',
  },
  {
    id: 'M2-MAXATTEMPTS-ONE',
    desc: 'reduce the default maxAttempts from 2 to 1, so the wrapper is present, parses, '
        + 'and never actually retries anything',
    anchor: 'async function agentRetryingDrops(prompt, opts, maxAttempts = 2)',
    replacement: 'async function agentRetryingDrops(prompt, opts, maxAttempts = 1)',
    expect: 'A1 a drop followed by a success RETURNS the recovered value',
  },
  {
    id: 'M3-BARE-AGENT-CALL-SITE',
    desc: 'restore the pre-fix bare spawn at the call site: the retry function still '
        + 'exists and is still correct, but nothing calls it -- the dead-code fix',
    anchor: 'const verdict = await agentRetryingDrops(PROMPT, {',
    replacement: 'const verdict = await agent(PROMPT, {',
    expect: 'B1 the verdict spawn goes through agentRetryingDrops',
  },
  {
    id: 'M4-TSD-SWALLOW-REAL-BUG',
    desc: 'TSD (EMSE ~75% survival, the highest-risk operator here): delete the THROW in '
        + 'the non-drop branch while keeping the guard, so a genuine bug is silently retried',
    anchor: "if (!msg.includes('without calling StructuredOutput')) throw e",
    replacement: "if (!msg.includes('without calling StructuredOutput')) { /* MUTANT M4: throw deleted */ }",
    expect: 'A3b ...with NO retry',
  },
  {
    id: 'M5-TSD-SILENT-EXHAUSTION',
    desc: 'TSD on the exhaustion throw: on running out of attempts the wrapper returns '
        + 'undefined instead of throwing -- a drop would stop looking like a drop, which '
        + 'is exactly how "NO VERDICT" could decay into a falsy PASS downstream',
    anchor: '  throw lastErr\n}',
    replacement: '  return undefined // MUTANT M5\n}',
    expect: 'A2 a drop on EVERY attempt THROWS',
  },
  {
    id: 'M6-WRONG-DROP-LITERAL',
    desc: 'change the matched drop string to one the runtime never emits, so real drops '
        + 'fall through the guard and are surfaced instead of retried',
    anchor: "const msg = String((e && e.message) || e)",
    replacement: "const msg = 'NEVER MATCHES ANYTHING' // MUTANT M6",
    expect: 'A1 a drop followed by a success RETURNS the recovered value',
  },
]

const sha = (p) => createHash('sha256').update(fs.readFileSync(p)).digest('hex')

function runChecker (overridePath) {
  const env = Object.assign({}, process.env)
  if (overridePath) env.PYFIN_QA_VERDICT_OVERRIDE = overridePath
  else delete env.PYFIN_QA_VERDICT_OVERRIDE
  try {
    const out = execFileSync('node', [CHECKER], { encoding: 'utf8', env, timeout: 400000 })
    return { code: 0, out }
  } catch (e) {
    return { code: e.status == null ? 2 : e.status, out: String((e.stdout || '') + (e.stderr || '')) }
  }
}

const failedNames = (out) => out.split('\n')
  .filter(l => l.trim().startsWith('FAIL '))
  .map(l => l.trim().slice(5).trim())

const before = sha(SUBJECT)
console.log(`subject sha256 BEFORE : ${before}`)

console.log('\n=== CONTROL (unmutated) -- must be GREEN before any cell means anything ===')
const control = runChecker(null)
const controlGreen = control.code === 0 && /ALL GREEN/.test(control.out)
console.log(`  control exit=${control.code} ${controlGreen ? 'GREEN' : 'NOT GREEN'}`)
if (!controlGreen) {
  console.log(control.out.slice(-1800))
  console.log('\nABORT: the control is not green, so no cell below could prove anything.')
  process.exit(1)
}

const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), 'mut8681-'))
const src = fs.readFileSync(SUBJECT, 'utf8')
const results = []

for (const m of MUTATIONS) {
  console.log(`\n=== ${m.id} ===\n  ${m.desc}`)
  const hits = src.split(m.anchor).length - 1
  if (hits !== 1) {
    console.log(`  FAIL anchor is not unique (${hits} occurrences) -- a non-mutation would score as a kill`)
    results.push({ id: m.id, killed: false, why: `anchor occurs ${hits}x` })
    continue
  }
  const mutated = src.replace(m.anchor, m.replacement)
  if (mutated === src) {
    console.log('  FAIL replacement produced an identical file')
    results.push({ id: m.id, killed: false, why: 'no textual change' })
    continue
  }
  const p = path.join(tmpdir, `${m.id}.js`)
  fs.writeFileSync(p, mutated)
  const r = runChecker(p)
  const fails = failedNames(r.out)
  const targeted = fails.some(f => f.startsWith(m.expect))
  const killed = r.code !== 0 && targeted
  console.log(`  exit=${r.code}  failures=${fails.length}`)
  for (const f of fails.slice(0, 6)) console.log(`     - ${f.slice(0, 96)}`)
  console.log(`  aimed at: "${m.expect}"  => ${targeted ? 'HIT' : 'NOT HIT'}`)
  console.log(`  ${killed ? 'KILLED' : 'SURVIVED'}`)
  results.push({ id: m.id, killed, why: killed ? '' : (r.code === 0 ? 'checker stayed green' : 'red for the wrong reason') })
}

const after = sha(SUBJECT)
console.log(`\nsubject sha256 AFTER  : ${after}`)
console.log(`tracked file unchanged: ${before === after ? 'YES' : 'NO -- THE SUBJECT WAS MODIFIED'}`)

const survivors = results.filter(r => !r.killed)
console.log('\n=== MATRIX ===')
for (const r of results) console.log(`  ${r.killed ? 'KILLED  ' : 'SURVIVED'} ${r.id}${r.why ? '  (' + r.why + ')' : ''}`)
console.log(`\n${results.length - survivors.length}/${results.length} killed`)
if (before !== after) { console.log('FAIL: the tracked subject changed during the run.'); process.exit(1) }
if (survivors.length) { console.log(`FAIL: ${survivors.length} mutant(s) survived -- reported, not dropped.`); process.exit(1) }
console.log('ALL CELLS KILLED')
