#!/usr/bin/env node
/**
 * phase-86.81 -- re-runnable proof that the StructuredOutput drop RETRY actually works.
 *
 *   node scripts/qa/verify_rail_retry.mjs
 *
 * Exit 0 iff every check passes AND the cardinality floor is met. A checker whose
 * loop covers nothing prints no failures and exits 0, which is indistinguishable from
 * success -- so the floor has NO opt-out.
 *
 * WHY THIS EXISTS. The retry landed in commit 6b4df8f9 (2026-08-14T10:15:17Z) and had
 * NEVER EXECUTED: measured across all 566 run records, `agentRetryingDrops` is absent
 * from 0/566 dispatched `script` fields, and zero qa-verdict runs had launched since
 * the fix. It parsed, and nothing else was known about it. "Present in source" is not
 * "proven to fire" -- and error-recovery code is precisely the code that is least
 * tested (EMSE 2021, 12,331 mutants: operators that DELETE OR DIVERT A THROW survive
 * far more often -- TSD ~75%, FBD 59% -- than ones that replace a catch body, CBR 100%).
 *
 * WHAT IT DRIVES, AND WHY THAT MATTERS. The REAL functions, byte-for-byte out of the
 * shipped workflow files -- never a hand-copy that can drift. `agentRetryingDrops` is
 * brace-matched out of `.claude/workflows/qa-verdict.js` and imported through a factory
 * that injects its two free variables (`agent`, `log`); the research-gate stage-2 loop
 * is exercised by wrapping the WHOLE workflow body in a drivable async function, the
 * same technique `verify_research_gate_workflow.mjs:78-96` already uses. Fault
 * injection is deterministic (AgentChaos: "All modification functions are deterministic
 * given same configuration and response").
 *
 * WHAT IT DELIBERATELY DOES NOT DO. Stage 1 of research-gate is ALREADY covered by
 * `verify_research_gate_workflow.mjs` (`dropsOnceThenSucceeds`, three named cells).
 * That coverage is ASSERTED here by running it, not duplicated -- rebuilding shipped
 * work is its own defect class on this project.
 *
 * AND IT CANNOT REPORT AN EFFECTIVENESS RATE. Retry math assumes independence, which
 * ReliabilityBench (arXiv 2601.06112) refutes in BOTH directions. Proving the mechanism
 * fires is `L_f` in MAS-FIRE's decomposition; it is not `S_f`, and reporting one as the
 * other is the overreach that paper exists to name.
 *
 * `PYFIN_QA_VERDICT_OVERRIDE` is the mutation seam: mutation_matrix_86_81.mjs points it
 * at a mutated COPY so the RED half is provable without writing the tracked file.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const QA_WF = process.env.PYFIN_QA_VERDICT_OVERRIDE || path.join(REPO, '.claude/workflows/qa-verdict.js')
const RG_WF = path.join(REPO, '.claude/workflows/research-gate.js')
const READER = path.join(REPO, 'scripts/qa/rail_drop_rate.py')
const RG_CHECKER = path.join(REPO, 'scripts/qa/verify_research_gate_workflow.mjs')

// The reader as it shipped BEFORE this step, pinned by commit rather than by HEAD, so
// the contamination demonstration keeps working after this step is itself committed.
const PRE_FIX_READER_COMMIT = 'f88f8190'

const DROP_MSG = 'agent({schema}): subagent completed without calling StructuredOutput'
const EXPECTED_CHECKS = 34

const failures = []
let pass = 0
const check = (name, cond, detail = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { failures.push(`${name}${detail ? ' -- ' + detail : ''}`); console.log(`  FAIL ${name}${detail ? ' -- ' + detail : ''}`) }
}
const section = (t) => console.log(`\n${'='.repeat(74)}\n${t}\n${'='.repeat(74)}`)

/**
 * Extract a named function's exact source span by brace matching.
 *
 * Two traps, both already paid for on this project:
 *  - a `{}` DEFAULT PARAMETER defeats a naive `indexOf('{')`, so the parameter list is
 *    skipped by paren matching first. `agentRetryingDrops` has `maxAttempts = 2`.
 *  - dropping a leading `async` yields a function whose `await` is a SyntaxError, so
 *    the span is walked BACKWARD over whitespace to pick the modifier up.
 */
function extractFn (src, name) {
  let start = src.indexOf(`function ${name} (`)
  if (start < 0) start = src.indexOf(`function ${name}(`)
  if (start < 0) throw new Error(`${path.basename(QA_WF)} does not define ${name}`)
  const before = src.slice(0, start)
  const asyncMatch = /async\s+$/.exec(before)
  if (asyncMatch) start = asyncMatch.index
  let p = src.indexOf('(', src.indexOf(`function ${name}`))
  let pd = 0
  for (; p < src.length; p++) {
    if (src[p] === '(') pd++
    else if (src[p] === ')') { pd--; if (pd === 0) break }
  }
  let depth = 0, i = src.indexOf('{', p)
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++
    else if (src[i] === '}') { depth--; if (depth === 0) break }
  }
  if (depth !== 0) throw new Error(`unbalanced braces extracting ${name}`)
  const body = src.slice(start, i + 1)
  if (body.length < 150) throw new Error(`${name} extraction looks wrong (${body.length} chars)`)
  return body
}

/** Import the REAL agentRetryingDrops, injecting the two globals it closes over. */
async function loadRetry (sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(QA_WF, 'utf8')
  const body = extractFn(src, 'agentRetryingDrops')
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'railretry-')), 'r.mjs')
  fs.writeFileSync(tmp, `export function __make (agent, log) {\n${body}\nreturn agentRetryingDrops\n}\n`)
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.__make !== 'function') throw new Error('could not build a drivable copy')
  return { make: mod.__make, src, body }
}

/** Wrap the WHOLE research-gate body as a drivable async function (house technique). */
async function loadRgDriver () {
  const src = fs.readFileSync(RG_WF, 'utf8')
  const body = src.replace('export const meta', 'const meta')
  if (/^\s*export\s/m.test(body)) throw new Error('unexpected residual export statement in the driver body')
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'rgretry-')), 'd.mjs')
  fs.writeFileSync(tmp, 'export async function __drive(args, phase, log, agent) {\n' + body + '\n}\n')
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.__drive !== 'function') throw new Error('could not build a drivable research-gate copy')
  return mod.__drive
}

const run = async () => {
  // ─────────────────────────────────────────────────────────────────────────
  section('[A] the REAL agentRetryingDrops, driven with deterministic fault injection')
  // ─────────────────────────────────────────────────────────────────────────
  const { make, src: qaSrc, body: retryBody } = await loadRetry()

  // The default is READ OFF the shipped source, never guessed. A guessed constant that
  // disagrees with its subject goes red against correct code.
  const declared = /function\s+agentRetryingDrops\s*\([^)]*maxAttempts\s*=\s*(\d+)/.exec(qaSrc)
  const MAX = declared ? Number(declared[1]) : NaN
  check('maxAttempts default is READ OFF the shipped source, not assumed',
    Number.isInteger(MAX) && MAX >= 2, `parsed maxAttempts=${declared ? declared[1] : 'NOT FOUND'}`)

  // A1 -- recovery.
  {
    let calls = 0
    const logs = []
    const agent = async () => {
      calls++
      if (calls === 1) throw new Error(DROP_MSG)
      return { ok: true, verdict: 'PASS', marker: 'recovered' }
    }
    const fn = make(agent, (m) => logs.push(m))
    let out = null, threw = null
    try { out = await fn('P', {}) } catch (e) { threw = e }
    check('A1 a drop followed by a success RETURNS the recovered value',
      threw === null && out && out.marker === 'recovered',
      threw ? `threw: ${String(threw.message).slice(0, 70)}` : `out=${JSON.stringify(out)}`)
    check('A1b ...and it did so by calling the agent exactly twice (the fault DID fire)',
      calls === 2, `calls=${calls}`)
    check('A1c ...and it LOGGED the drop, so a recovery is observable in the run record',
      logs.length === 1 && /DROP on attempt 1\//.test(logs[0]), `logs=${JSON.stringify(logs)}`)
  }

  // A2 -- exhaustion.
  {
    let calls = 0
    const logs = []
    const agent = async () => { calls++; throw new Error(DROP_MSG) }
    const fn = make(agent, (m) => logs.push(m))
    let out = null, threw = null
    try { out = await fn('P', {}) } catch (e) { threw = e }
    check('A2 a drop on EVERY attempt THROWS -- it never returns a fabricated value',
      threw !== null && out === null,
      threw ? `threw ok` : `RETURNED ${JSON.stringify(out)}`)
    check('A2b ...and the throw carries the drop text, so the caller can classify it',
      threw !== null && /without calling StructuredOutput/.test(String(threw.message)))
    check('A2c ...and it stopped at exactly maxAttempts, so the retry is BOUNDED',
      calls === MAX, `calls=${calls} maxAttempts=${MAX}`)
    check('A2d ...and the final log says EXHAUSTED rather than implying another attempt',
      logs.length === MAX && /exhausted, NO VERDICT/.test(logs[logs.length - 1]),
      `logs=${JSON.stringify(logs)}`)
  }

  // A3 -- a non-drop error must surface immediately. EMSE's highest-survival operator
  // (TSD, ~75%) deletes exactly this throw, producing a mutant that silently retries a
  // real bug, so this cell carries a NAMED assertion on the call count as well.
  {
    let calls = 0
    const logs = []
    const agent = async () => { calls++; throw new Error('Claude refused to answer') }
    const fn = make(agent, (m) => logs.push(m))
    let threw = null
    try { await fn('P', {}) } catch (e) { threw = e }
    check('A3 a NON-drop error surfaces on the FIRST attempt',
      threw !== null && /refused to answer/.test(String(threw.message)),
      threw ? `msg=${String(threw.message).slice(0, 50)}` : 'did not throw')
    check('A3b ...with NO retry -- a real bug must not be re-run at 185K tokens a time',
      calls === 1, `calls=${calls}`)
    check('A3c ...and NOTHING is logged as a drop, so the metric cannot count it',
      logs.length === 0, `logs=${JSON.stringify(logs)}`)
  }

  // A4 -- the class of non-drop failures, not one spelling. A rail can die of a
  // max_tokens cutoff or a transport error; none of those is the stochastic drop.
  {
    const OTHER = [
      'agent({schema}): max_tokens reached before StructuredOutput',
      'fetch failed: ECONNRESET',
      'Error: agent aborted',
    ]
    let allImmediate = true
    const detail = []
    for (const msg of OTHER) {
      let calls = 0
      const fn = make(async () => { calls++; throw new Error(msg) }, () => {})
      try { await fn('P', {}) } catch { /* expected */ }
      if (calls !== 1) { allImmediate = false; detail.push(`${msg.slice(0, 28)}=>${calls}`) }
    }
    check('A4 every NON-drop failure shape surfaces immediately (kills an over-broad catch)',
      allImmediate, detail.join(' '))
  }

  // A5 -- the happy path is not disturbed by the wrapper.
  {
    let calls = 0
    const logs = []
    const fn = make(async () => { calls++; return { ok: true } }, (m) => logs.push(m))
    const out = await fn('P', {})
    check('A5 a clean run calls the agent ONCE and logs nothing',
      calls === 1 && logs.length === 0 && out && out.ok === true,
      `calls=${calls} logs=${logs.length}`)
  }

  // ─────────────────────────────────────────────────────────────────────────
  section('[B] the wrapper is actually USED -- a correct function nothing calls is not a fix')
  // ─────────────────────────────────────────────────────────────────────────
  {
    const usesWrapper = /const\s+verdict\s*=\s*await\s+agentRetryingDrops\s*\(\s*PROMPT/.test(qaSrc)
    check('B1 the verdict spawn goes through agentRetryingDrops, not a bare agent()',
      usesWrapper, usesWrapper ? '' : 'call site does not match the wrapper form')
    // The bare form must be ABSENT: if both exist, the wrapper may be dead code.
    const bareSpawn = /const\s+verdict\s*=\s*await\s+agent\s*\(\s*PROMPT/.test(qaSrc)
    check('B2 no bare `await agent(PROMPT` verdict spawn survives beside it',
      !bareSpawn, bareSpawn ? 'a bare spawn is still present' : '')
    check('B3 the retry body actually contains a loop -- not a renamed passthrough',
      /for\s*\(/.test(retryBody) && /catch/.test(retryBody))
  }

  // ─────────────────────────────────────────────────────────────────────────
  section('[C] research-gate STAGE 2 -- the loop with no coverage before this step')
  // ─────────────────────────────────────────────────────────────────────────
  {
    const drive = await loadRgDriver()
    let explore = 0, researcher = 0
    const logs = []
    // Keyed on agentType, NOT on call ordinal -- the same correction the sibling
    // checker had to make when stage 1 gained a retry and silently defused its test.
    const stage2DropsOnce = async (prompt, opts) => {
      if (opts && opts.agentType === 'researcher') {
        researcher++
        return {
          tier: 'moderate', external_sources_read_in_full: 6, snippet_only_sources: 2,
          urls_collected: 12, recency_scan_performed: true, internal_files_inspected: 3,
          gate_passed: true, sources_read_in_full: [], brief_path: 'X',
        }
      }
      explore++
      if (explore === 1) throw new Error(DROP_MSG)
      return {
        brief_exists: true, brief_non_empty: true, char_count: 40000,
        urls_checked: 0, urls_present: 0, urls_missing: [],
        recency_section_present: true, distinct_urls_in_brief: 25,
        brief_status_in_brief: 'COMPLETE',
      }
    }
    let res = null, threw = null
    try {
      res = await drive({ step_id: 'RETRY-S2', tier: 'moderate' }, () => {}, (m) => logs.push(m), stage2DropsOnce)
    } catch (e) { threw = e }
    check('C1 a stage-2 drop is RETRIED (the Explore agent is spawned twice)',
      threw === null && explore === 2, `threw=${threw ? String(threw.message).slice(0, 60) : 'null'} explore=${explore}`)
    check('C2 ...and the retried stage 2 produces a verification rather than failing closed',
      !!res && res.brief_verification !== null && typeof res.brief_verification === 'object',
      res ? `brief_verification=${JSON.stringify(res.brief_verification).slice(0, 60)}` : 'no result')
    check('C3 ...and the stage-2 retry is LOGGED, so it is countable by the reader',
      logs.some(l => /STAGE-2 brief-verify failed \(attempt/.test(String(l))),
      `logs=${JSON.stringify(logs).slice(0, 160)}`)

    // A non-drop stage-2 error must NOT be retried.
    let explore2 = 0
    const stage2RealBug = async (prompt, opts) => {
      if (opts && opts.agentType === 'researcher') {
        return {
          tier: 'moderate', external_sources_read_in_full: 6, snippet_only_sources: 2,
          urls_collected: 12, recency_scan_performed: true, internal_files_inspected: 3,
          gate_passed: true, sources_read_in_full: [], brief_path: 'X',
        }
      }
      explore2++
      throw new TypeError('x is not a function')
    }
    await drive({ step_id: 'RETRY-S2B', tier: 'moderate' }, () => {}, () => {}, stage2RealBug).catch(() => {})
    check('C4 a NON-drop stage-2 error is not retried (one spawn, then fail closed)',
      explore2 === 1, `explore=${explore2}`)
  }

  // ─────────────────────────────────────────────────────────────────────────
  section('[D] stage 1 -- coverage ASSERTED from the checker that already owns it')
  // ─────────────────────────────────────────────────────────────────────────
  {
    let ok = true, out = ''
    try {
      out = execFileSync('node', [RG_CHECKER], { encoding: 'utf8', timeout: 300000 })
    } catch (e) { ok = false; out = String((e.stdout || '') + (e.stderr || '')) }
    const green = /ALL GREEN: (\d+) passed, 0 failed/.exec(out)
    check('D1 verify_research_gate_workflow.mjs is GREEN (it owns the stage-1 retry cells)',
      ok && !!green, green ? `${green[1]} checks` : out.slice(-140))
    check('D2 ...and it really does carry a stage-1 RETRY cell (not just drop-survival)',
      /a SINGLE stochastic drop is RETRIED/.test(fs.readFileSync(RG_CHECKER, 'utf8')))
  }

  // ─────────────────────────────────────────────────────────────────────────
  section('[E] the metric -- it must count RECOVERIES, never its own source text')
  // ─────────────────────────────────────────────────────────────────────────
  {
    // Every retry SITE's log literal must be one the reader knows about. A renamed log
    // line would otherwise become a silent undercount.
    const readerSrc = fs.readFileSync(READER, 'utf8')
    const rgSrc = fs.readFileSync(RG_WF, 'utf8')
    const SITES = [
      ['qa-verdict', qaSrc, 'StructuredOutput DROP on attempt'],
      ['research-gate stage 1', rgSrc, 'STAGE-1 RAIL DROPPED (attempt'],
      ['research-gate stage 2', rgSrc, 'STAGE-2 brief-verify failed (attempt'],
    ]
    for (const [label, src, literal] of SITES) {
      check(`E0 ${label}'s retry log literal exists in BOTH the workflow and the reader`,
        src.includes(literal) && readerSrc.includes(literal), `literal="${literal}"`)
    }

    // Build a synthetic project dir. The decisive record is R1: its `script` field
    // carries the REAL shipped qa-verdict source (which contains the retry log literal
    // exactly once) while its `logs` are EMPTY. It never retried.
    const proj = fs.mkdtempSync(path.join(os.tmpdir(), 'railmetric-'))
    const wfdir = path.join(proj, 'sessionA', 'workflows')
    fs.mkdirSync(wfdir, { recursive: true })
    const base = (id, extra) => JSON.stringify(Object.assign({
      runId: `wf_${id}`, timestamp: '2026-08-14T09:00:00.000Z',
      startTime: Date.parse('2026-08-14T09:00:00Z'),
      workflowName: 'qa-verdict', status: 'completed', defaultModel: 'claude-opus-5[1m]',
      script: qaSrc, logs: [], totalTokens: 1000,
    }, extra))
    fs.writeFileSync(path.join(wfdir, 'r1.json'), base('R1', {}))
    fs.writeFileSync(path.join(wfdir, 'r2.json'), base('R2', {
      logs: ['qa-verdict: StructuredOutput DROP on attempt 1/2 -- retrying'] }))
    fs.writeFileSync(path.join(wfdir, 'r3.json'), base('R3', {
      workflowName: 'research-gate',
      logs: ['research-gate: STAGE-1 RAIL DROPPED (attempt 1/3) -- x | stochastic StructuredOutput drop -- RETRYING'] }))
    fs.writeFileSync(path.join(wfdir, 'r4.json'), base('R4', {
      workflowName: 'research-gate',
      logs: ['research-gate: STAGE-2 brief-verify failed (attempt 1/2) -- stochastic StructuredOutput drop, RETRYING'] }))
    // R5 -- a run that FAILED for an unrelated reason. Its script quotes the drop
    // string (every shipped copy does), so a blob-scanning predicate misreads it.
    fs.writeFileSync(path.join(wfdir, 'r5.json'), base('R5', {
      status: 'failed', error: 'TypeError: enforceEscalation is not a function' }))
    // R6 -- a genuine exhausted drop.
    fs.writeFileSync(path.join(wfdir, 'r6.json'), base('R6', {
      status: 'failed', error: DROP_MSG }))
    // R7 -- launched BEFORE the fix instant; must land in the "before" bucket.
    fs.writeFileSync(path.join(wfdir, 'r7.json'), base('R7', {
      startTime: Date.parse('2026-08-14T10:10:26Z'), timestamp: '2026-08-14T10:27:30.000Z' }))
    // R8 -- launched AFTER the fix instant, but COMPLETING on the same day. The
    // date-split reader cannot tell R7 and R8 apart; the instant-split one must.
    fs.writeFileSync(path.join(wfdir, 'r8.json'), base('R8', {
      startTime: Date.parse('2026-08-14T10:16:59Z'), timestamp: '2026-08-14T10:28:31.000Z' }))

    const readJson = (script) => JSON.parse(execFileSync('python3',
      [script, '--json', '--project-dir', proj], { encoding: 'utf8' }))
    const now = readJson(READER)

    check('E1 a run whose SCRIPT contains the retry literal but whose LOGS are empty counts 0',
      now.retried === 3, `retried=${now.retried} (expected 3: R2 + R3 + R4 only)`)
    check('E2 all THREE retry log shapes are counted (both workflows, all sites)',
      now.retried === 3, `retried=${now.retried}`)
    check('E3 a run that failed for an UNRELATED reason is not classified as a drop',
      now.exhausted === 1, `exhausted=${now.exhausted} (expected 1: R6 only, not R5)`)
    check('E4 the before/after split uses the LAUNCH instant, not the date',
      now.post_fix_runs === 1 && now.pre_fix_runs === 7,
      `post=${now.post_fix_runs} pre=${now.pre_fix_runs} (expected 1 / 7)`)

    // The contamination demonstration: the PRE-FIX reader, pinned by commit, on the
    // SAME fixture. This is the "show it red first" half of criterion 5.
    let oldOut = null
    try {
      const oldSrc = execFileSync('git', ['show', `${PRE_FIX_READER_COMMIT}:scripts/qa/rail_drop_rate.py`],
        { cwd: REPO, encoding: 'utf8' })
      const oldPath = path.join(proj, 'rail_drop_rate_PREFIX.py')
      fs.writeFileSync(oldPath, oldSrc)
      // The pre-fix reader has no --json, so its stdout is parsed for the summary line.
      const txt = execFileSync('python3', [oldPath, '--project-dir', proj], { encoding: 'utf8' })
      const m = /RETRIED\s+\(recovered\)\s*:\s*(\d+)/.exec(txt)
      const after = /on\/after\s+runs=\s*(\d+)/.exec(txt)
      oldOut = { retried: m ? Number(m[1]) : null, post: after ? Number(after[1]) : null }
    } catch (e) {
      oldOut = { error: String(e.message).slice(0, 120) }
    }
    check('E5 the PRE-FIX reader over-counts retries on the same fixture (contamination shown RED)',
      oldOut && oldOut.retried !== null && oldOut.retried > now.retried,
      `pre-fix retried=${oldOut && oldOut.retried} vs corrected ${now.retried}`)
    check('E6 the PRE-FIX reader also mis-buckets the launch-instant split on the same fixture',
      oldOut && oldOut.post !== null && oldOut.post > now.post_fix_runs,
      `pre-fix on/after=${oldOut && oldOut.post} vs corrected ${now.post_fix_runs}`)

    // The reader must never read the blob again.
    check('E7 the reader computes `retries` from the logs array, never from the record blob',
      /logs\s*$|for line in logs/m.test(readerSrc) && !/blob\.count\(/.test(readerSrc),
      /blob\.count\(/.test(readerSrc) ? 'blob.count( is still present' : '')
    check('E8 the reader classifies `exhausted` from the error field alone',
      !/DROP in blob/.test(readerSrc), /DROP in blob/.test(readerSrc) ? 'blob disjunct survives' : '')
    check('E9 the reader DISCLOSES that retried=0 on a lost run means "not visible"',
      /not visible.*did not happen|RECOVERY COUNT/i.test(readerSrc))
  }

  // ─────────────────────────────────────────────────────────────────────────
  section('[F] no verdict semantics move -- a retry can never manufacture a PASS')
  // ─────────────────────────────────────────────────────────────────────────
  {
    // Exhaustion throws, so the workflow throws, so the caller sees an errored return.
    // That is the "NO VERDICT, never PASS" contract, and it is asserted behaviourally.
    const fn = make(async () => { throw new Error(DROP_MSG) }, () => {})
    let threw = null, out = 'UNSET'
    try { out = await fn('P', {}) } catch (e) { threw = e }
    check('F1 an exhausted retry yields NO value at all -- never an ok/PASS-shaped object',
      threw !== null && out === 'UNSET')
    check('F2 the exhaustion path rethrows the ORIGINAL error rather than a synthesised one',
      threw !== null && String(threw.message).includes('without calling StructuredOutput'))
    const rgSrc = fs.readFileSync(RG_WF, 'utf8')
    check('F3 research-gate still RECOMPUTES gate_passed via enforceGate after the retry loop',
      /const\s+enforcement\s*=\s*enforceGate\(/.test(rgSrc))
    check('F4 the retry loop assigns no verdict/gate field of its own',
      !/gate_passed\s*=\s*true/.test(retryBody) && !/verdict\s*=\s*['"]PASS/.test(retryBody))
  }

  // ─────────────────────────────────────────────────────────────────────────
  section('RESULT')
  // ─────────────────────────────────────────────────────────────────────────
  // A cardinality floor with NO opt-out: a checker whose cells silently stopped
  // running would otherwise print no failures and exit 0.
  if (pass + failures.length < EXPECTED_CHECKS) {
    failures.push(`cardinality floor: ran ${pass + failures.length} checks, expected >= ${EXPECTED_CHECKS}`)
    console.log(`  FAIL cardinality floor -- ran ${pass + failures.length}, expected >= ${EXPECTED_CHECKS}`)
  }
  if (failures.length) {
    console.log(`\n${failures.length} FAILED:`)
    for (const f of failures) console.log(`  - ${f}`)
    console.log(`\n${pass} passed, ${failures.length} failed`)
    return 1
  }
  console.log(`\nALL GREEN: ${pass} passed, 0 failed`)
  return 0
}

run().then(c => process.exit(c)).catch(e => { console.error(e); process.exit(2) })
