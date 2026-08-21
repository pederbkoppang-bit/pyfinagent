#!/usr/bin/env node
/**
 * phase-90.2 -- re-runnable checker for routing the severity the judge already
 * emits, caller-side, without asking it to classify anything new and without
 * moving the verdict.
 *
 *   node scripts/qa/verify_severity_routing_90_2.mjs --self-test   # the immutable command
 *   node scripts/qa/verify_severity_routing_90_2.mjs --replay      # the live-corpus table
 *
 * Exit 0 iff every check passes AND the cardinality floor is met AND the mutation
 * matrix is clean. A checker whose loop covers nothing prints no failures and
 * exits 0, which is indistinguishable from success -- so the floor is not
 * decoration.
 *
 * It drives the REAL `enforceSeverityRouting` from `.claude/workflows/qa-verdict.js`,
 * never a copy: the source is read, the function's exact span (and the two helpers
 * plus two regex constants it closes over) is extracted by brace matching, an
 * `export {...}` line is appended to a TEMP copy, and that copy is imported. Same
 * mechanism and same reason as verify_escalation_86_78.mjs -- the shipped workflow
 * exports only `meta`, and the workflow body carries top-level `return`s that are a
 * SyntaxError under ESM.
 *
 * `sourceOverride` is the mutation seam: cells pass mutated source so the RED half
 * of criterion 6 is provable without ever writing to the tracked file.
 */
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { refuseCloseOnUnfiledResidual } from './residual_close_gate.mjs'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const WORKFLOW = path.join(REPO, '.claude/workflows/qa-verdict.js')
const FIXTURES = path.join(REPO, 'scripts/qa/fixtures/severity_routing_90_2_returns.json')
const VERDICT_LEDGER = path.join(REPO, 'handoff/verdict_ledger.jsonl')

// Bumped deliberately when a check is added. If the loop stops covering things,
// this is what turns a silent pass into a loud failure.
const EXPECTED_CHECKS = 66

const results = []
const check = (label, ok, detail = '') => {
  results.push([label, !!ok, detail])
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}${detail ? ' -- ' + detail : ''}`)
}
const section = (t) => console.log(`\n${'='.repeat(74)}\n${t}\n${'='.repeat(74)}`)

// ---------------------------------------------------------------------------
// Extraction -- the REAL function, byte-for-byte from the shipped file
// ---------------------------------------------------------------------------

function extractSpan (src, header) {
  const start = src.indexOf(header)
  if (start < 0) throw new Error(`qa-verdict.js does not contain: ${header}`)
  // Skip the parameter list before hunting the body brace -- a default parameter
  // `opts = {}` makes a naive indexOf('{') close immediately. (That is not
  // hypothetical: it is exactly what broke the sibling checker's first run.)
  let p = src.indexOf('(', start)
  let pd = 0
  for (; p < src.length; p++) {
    if (src[p] === '(') pd++
    else if (src[p] === ')') { pd--; if (pd === 0) break }
  }
  let depth = 0
  let i = src.indexOf('{', p)
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++
    else if (src[i] === '}') { depth--; if (depth === 0) break }
  }
  if (depth !== 0) throw new Error(`unbalanced braces extracting: ${header}`)
  return src.slice(start, i + 1)
}

function extractConst (src, name) {
  const start = src.indexOf(`const ${name} =`)
  if (start < 0) throw new Error(`qa-verdict.js does not define ${name}`)
  const nl = src.indexOf('\n', src.indexOf('/', start) > start ? start : start)
  // The two constants are regex literals that may wrap one line; take through the
  // end of the statement's last line before a blank/comment line.
  let end = start
  for (;;) {
    const next = src.indexOf('\n', end + 1)
    if (next < 0) { end = src.length; break }
    const line = src.slice(end + 1, next).trim()
    end = next
    if (line === '' || line.startsWith('//') || line.startsWith('function')
        || line.startsWith('const ')) break
  }
  void nl
  return src.slice(start, end)
}

/**
 * The phase-90.2 leak guard is a TOP-LEVEL STATEMENT BLOCK, not a function, so it
 * cannot be extracted the way severityTags is. It is lifted into a callable
 * instead, so the checker DRIVES it rather than grepping for it.
 *
 * WHY THIS EXISTS: the first version of section I was four regexes over the
 * workflow file. The 90.2 cycle-1 Q/A applied two neutering mutants in memory --
 * `if (false && leakedS.length > 0)`, and deleting the if/throw while leaving the
 * message in a comment -- and ALL FOUR checks stayed GREEN. A guard that is only
 * ever matched, never executed, is the illusory-guard shape (sole-coverage
 * vacuity): the literal survives while the behaviour is stripped. Returns null
 * when the guard is absent, so its absence is a NAMED check failure rather than
 * an exception that might be mistaken for a broken harness.
 */
function extractLeakGuard (src) {
  const start = src.indexOf('const leakedS =')
  if (start < 0) return null
  const ifAt = src.indexOf('if (leakedS.length > 0)', start)
  if (ifAt < 0) return null
  let depth = 0
  let i = src.indexOf('{', ifAt)
  if (i < 0) return null
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++
    else if (src[i] === '}') { depth--; if (depth === 0) break }
  }
  if (depth !== 0) return null
  return 'function leakGuard (merged, severity_routing) {\n'
    + src.slice(start, i + 1) + '\n}'
}

async function load (sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(WORKFLOW, 'utf8')
  const parts = [
    extractConst(src, 'SEVERITY_TOKEN'),
    extractConst(src, 'IMMEDIATE_NEGATOR'),
    extractSpan(src, 'function severityTags ('),
    extractSpan(src, 'function deriveSeverity ('),
    extractSpan(src, 'function enforceSeverityRouting ('),
  ]
  const guardSrc = extractLeakGuard(src)
  if (guardSrc) parts.push(guardSrc)
  else parts.push('const leakGuard = null')
  const body = parts.join('\n\n')
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'qa902_'))
  const tmp = path.join(dir, 'wf.mjs')
  fs.writeFileSync(tmp,
    body + '\nexport { enforceSeverityRouting, deriveSeverity, severityTags, leakGuard }\n')
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.enforceSeverityRouting !== 'function') {
    throw new Error('extraction produced no enforceSeverityRouting')
  }
  return mod
}

const sha = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')

// ---------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------

const V = (verdict, entries, details = []) => ({
  ok: verdict === 'PASS',
  verdict,
  reason: 'fixture',
  violated_criteria: entries,
  violation_details: details,
  certified_fallback: false,
  checks_run: [],
  harness_compliance_ok: true,
  notes: '',
})

async function runChecks (mod) {
  const { enforceSeverityRouting: f, deriveSeverity: d } = mod

  section('A. THE VERDICT GUARD IS STRUCTURAL (criterion 2)')
  // No REAL instance of this exists: 0 of the 66 pinned FAILs are all-WARN/NOTE.
  // The fixture is CONSTRUCTED, and that is precisely WHY the guard must be
  // structural -- "never observed" is not "cannot happen".
  const failAllWarn = f(V('FAIL', ['(WARN) a residual note', 'NOTE: cosmetic only']))
  check('a FAIL whose every entry is WARN/NOTE-tagged routes to remediate',
    failAllWarn.route === 'remediate', 'route=' + failAllWarn.route)
  const passAllWarn = f(V('PASS', ['(WARN) a residual note']))
  check('a PASS whose every entry is WARN/NOTE-tagged routes to remediate',
    passAllWarn.route === 'remediate', 'route=' + passAllWarn.route)
  const condAllWarn = f(V('CONDITIONAL', ['(WARN) a residual note', 'NOTE: cosmetic only']))
  check('the SAME entries under CONDITIONAL route to queue_residual -- so the guard '
    + 'DISCRIMINATES rather than always denying',
    condAllWarn.route === 'queue_residual', 'route=' + condAllWarn.route)

  section('B. THE DERIVATION READS A DELIMITED TAG, NOT A TOKEN (criteria 4, 6)')
  check('a parenthesised tag is a tag', d('finding text (WARN)') === 'WARN')
  check('a bracketed tag is a tag', d('finding text [WARN]') === 'WARN')
  check('a colon-suffixed tag is a tag', d('WARN: finding text') === 'WARN')
  check('a dash-suffixed tag is a tag', d('WARN -- finding text') === 'WARN')
  check('an entry-initial tag is a tag', d('WARN the thing is off') === 'WARN')
  check('BLOCK dominates a co-occurring WARN',
    d('[BLOCK] serious (WARN) also') === 'BLOCK')
  check('an untagged finding is UNTAGGED, never silently a NOTE',
    d('the guard does not cover the alias path') === 'UNTAGGED')
  check('an IDENTIFIER carrying a severity substring is NOT a tag',
    d('WARN_provenance_control_x was not asserted') === 'UNTAGGED',
    'got ' + d('WARN_provenance_control_x was not asserted'))
  // NOTE THE SHAPE OF THESE TWO. The obvious fixtures for the negator rule --
  // "no WARN fired on that path" -- pass VACUOUSLY, because the token there is
  // also not in a delimiter position, so the delimiter rule alone already
  // excludes it and the negator check does nothing. Mutants M5 and M8 both
  // SURVIVED the first version of this checker for exactly that reason. Each rule
  // needs a fixture where it is the ONLY thing standing between the entry and a
  // tag.
  check('an IMMEDIATE negator kills a tag that IS in a delimiter position '
    + '("no WARN: nothing fired") -- the negator rule is the only thing excluding it',
    d('no WARN: nothing fired on that path') === 'UNTAGGED',
    'got ' + d('no WARN: nothing fired on that path'))
  const bareInProse = 'the checker never reaches the WARN branch at all'
  check('a BARE token in prose is not a tag -- the delimiter rule is the only thing '
    + 'excluding it', d(bareInProse) === 'UNTAGGED', 'got ' + d(bareInProse))
  check('...and that exclusion changes the ROUTE, not just the label',
    f(V('CONDITIONAL', [bareInProse])).route === 'remediate',
    'route=' + f(V('CONDITIONAL', [bareInProse])).route)
  check('...and so does the negator exclusion',
    f(V('CONDITIONAL', ['no WARN: nothing fired on that path'])).route === 'remediate')
  // The measured correction: a negator belonging to the finding's OWN prose must
  // NOT kill a genuine trailing tag. 6 of 6 real runs were misclassified by a
  // proximity filter before this was fixed.
  check('a negator in the finding\'s own prose does NOT kill a genuine trailing tag '
    + '(the measured 6-of-6 false-positive class)',
    d('relocates file paths but not the HTTP client to :8000 (WARN)') === 'WARN',
    'got ' + d('relocates file paths but not the HTTP client to :8000 (WARN)'))
  check('...and the same for a bracketed one',
    d('the phrase was never contiguous in the pre-fix source [WARN]') === 'WARN')

  section('C. MIXED AND EMPTY RUNS (criterion 4)')
  const mixed = f(V('CONDITIONAL', ['(WARN) residual', 'the alias path is unguarded']))
  check('one WARN + one UNTAGGED routes to remediate',
    mixed.route === 'remediate', 'route=' + mixed.route)
  const withBlock = f(V('CONDITIONAL', ['(WARN) residual', '[BLOCK] real defect']))
  check('one WARN + one BLOCK routes to remediate', withBlock.route === 'remediate')
  const empty = f(V('CONDITIONAL', []))
  check('a CONDITIONAL with NO entries routes to remediate, never queue_residual',
    empty.route === 'remediate', 'route=' + empty.route)
  check('...and names the absence ABSENT rather than scoring it',
    empty.severity_source === 'ABSENT', 'severity_source=' + empty.severity_source)

  section('D. ABSENCE IS NAMED, NEVER VALUED')
  check('disagreed is null (not false) when the judge emitted nothing',
    mixed.disagreed === null, 'disagreed=' + JSON.stringify(mixed.disagreed))
  check('...and the status says why',
    mixed.disagreement_status === 'nothing_emitted_to_compare',
    mixed.disagreement_status)
  const emitted = f(V('CONDITIONAL', ['(WARN) a', '(WARN) b'],
    [{ severity: 'BLOCK' }, { severity: 'WARN' }]))
  check('a judge-emitted severity GOVERNS over the caller derivation (86.98 branch)',
    emitted.severity_source === 'judge_emitted' && emitted.route === 'remediate',
    'source=' + emitted.severity_source + ' route=' + emitted.route)
  check('...and the disagreement with the derivation is reported, not hidden',
    emitted.disagreed === true && emitted.disagreement_status === 'compared')
  check('...and reliability is null on that branch (nothing was derived to qualify)',
    emitted.reliability === null)
  const lenMismatch = f(V('CONDITIONAL', ['(WARN) a'], [{ severity: 'WARN' }, { severity: 'WARN' }]))
  check('non-index-comparable arrays yield disagreed=null with a named status',
    lenMismatch.disagreed === null
      && /not_index_comparable/.test(lenMismatch.disagreement_status),
    String(lenMismatch.disagreement_status).slice(0, 40))

  section('E. NOTHING IS DROPPED, AND THE DERIVATION CARRIES ITS UNRELIABILITY')
  const three = f(V('CONDITIONAL', ['(WARN) a', 'untagged b', '[NOTE] c']))
  check('every reported finding survives into derived_severities',
    three.derived_severities.length === 3, 'n=' + three.derived_severities.length)
  check('...aligned to violated_criteria BY INDEX',
    three.derived_severities.every((x, i) => x.index === i))
  check('...with the per-entry classes intact',
    three.derived_severities.map(x => x.severity).join(',') === 'WARN,UNTAGGED,NOTE',
    three.derived_severities.map(x => x.severity).join(','))
  // GUARD EVERY RETURNED ARRAY, NOT JUST THE ONE THE MATRIX HAPPENS TO REACH.
  // The 90.2 cycle-2 Q/A truncated `governing_severities` in the RETURN LITERAL and
  // it SURVIVED all 66 checks: section E asserted length, alignment and content on
  // `derived_severities` only, and matrix cell M3 mutates the shared source array
  // `derived`, which feeds BOTH fields -- so one kill LOOKED like coverage of two
  // return sites while only one was guarded. `governing_severities` is the array
  // `route` is computed from, and it becomes the authoritative one under 86.98.
  check('every reported finding survives into governing_severities too',
    three.governing_severities.length === 3, 'n=' + three.governing_severities.length)
  check('...with its per-entry classes intact',
    three.governing_severities.join(',') === 'WARN,UNTAGGED,NOTE',
    three.governing_severities.join(','))
  check('...and it agrees with derived_severities index-for-index when nothing was '
    + 'judge-emitted',
    three.governing_severities.every((s, i) => s === three.derived_severities[i].severity))

  section('E2. THE 86.98 BRANCH CANNOT FILE AN UNCLASSIFIED FINDING AWAY')
  // Both of these are unreachable today (additionalProperties: false), and both were
  // reachable in cycle 2's code. "Unreachable today" is a schema, not a property.
  const emittedMismatch = f(V('CONDITIONAL',
    ['an UNTAGGED blocker A', 'an UNTAGGED blocker B'], [{ severity: 'NOTE' }]))
  check('a judge-emitted list that does NOT line up with the findings cannot file '
    + 'two untagged blockers away as residual',
    emittedMismatch.route === 'remediate', 'route=' + emittedMismatch.route)
  check('...and the fallback to the derivation is NAMED, not silent',
    /not_index_comparable/.test(emittedMismatch.severity_source),
    emittedMismatch.severity_source)
  check('...and the judge-emitted list is still REPORTED rather than discarded',
    Array.isArray(emittedMismatch.emitted_severities)
      && emittedMismatch.emitted_severities[0] === 'NOTE')
  const emittedNoFindings = f(V('CONDITIONAL', [], [{ severity: 'NOTE' }]))
  check('an EMPTY findings list cannot reach queue_residual on the emitted branch '
    + 'either -- no findings is never a residual',
    emittedNoFindings.route === 'remediate', 'route=' + emittedNoFindings.route)
  check('a judge-emitted list that DOES line up still governs (86.98 is satisfied, '
    + 'not pre-empted)',
    f(V('CONDITIONAL', ['(WARN) a', '(WARN) b'], [{ severity: 'BLOCK' }, { severity: 'WARN' }]))
      .severity_source === 'judge_emitted')

  section('E3. THE NEGATOR IS NARROW BY MEASUREMENT, AND THE NARROWNESS IS PINNED')
  // The cycle-2 Q/A widened IMMEDIATE_NEGATOR back to the 45-char proximity window
  // this step measured and discarded, and it SURVIVED the whole checker while
  // changing real behaviour: 1 of 906 entries reclassified, and run wf_7fa0e5d6-c50
  // moved out of queue_residual (41 -> 40). Cell M8 kills REMOVING the rule; nothing
  // pinned how far it may reach. This fixture is that pin -- taken verbatim from the
  // run the widening moved.
  const wideCase = "universal 'run_friday_promotion has no caller anywhere' (WARN)"
  check('a negator three words back does NOT kill a genuine trailing tag '
    + '(verbatim from wf_7fa0e5d6-c50, the run a 45-char window moves)',
    d(wideCase) === 'WARN', 'got ' + d(wideCase))
  check('...while an IMMEDIATE negator still does, so the rule is narrow, not absent',
    d('no WARN: nothing fired') === 'UNTAGGED')

  // DISCLOSED, because an unfired clause earns its place only if you say so.
  // `allResidual` carries `entries.length > 0` in ADDITION to `governing.length > 0`.
  // Under the current `comparable` gate that clause is PROVABLY REDUNDANT: when the
  // arrays do not line up, `governing` IS `derivedOnly`, and `comparable` itself
  // requires `derived.length > 0`, so a non-empty `governing` already implies
  // non-empty findings. I MEASURED this rather than assuming it -- a mutation cell
  // removing the clause was an EQUIVALENT mutant and is reported here instead of
  // being padded into the matrix as a kill it cannot make. The clause is kept as the
  // second line of defence if a future edit relaxes `comparable`, and the assertion
  // below pins the behaviour either way.
  check('an empty findings list cannot reach queue_residual under ANY branch -- '
    + 'derived, emitted-comparable, or emitted-mismatched',
    ['remediate'].includes(f(V('CONDITIONAL', [])).route)
      && f(V('CONDITIONAL', [], [{ severity: 'NOTE' }])).route === 'remediate'
      && f(V('CONDITIONAL', [], [])).route === 'remediate')

  section('E4. RELIABILITY TRAVELS WITH THE DERIVATION')
  check('the derivation is labelled NON-authoritative',
    three.reliability && three.reliability.derivation_is_authoritative === false)
  check('...and carries the brief\'s figures attributed to the BRIEF, not to this step',
    !!(three.reliability.measured_by_the_research_brief
      && three.reliability.measured_by_this_step))
  check('queue_residual carries the FILE-don\'t-fix instruction',
    /FILE the residual/.test(condAllWarn.next_action_on_queue_residual || ''))
  check('...and remediate carries null rather than an empty string',
    mixed.next_action_on_queue_residual === null)

  section('F. VERDICT BYTE-IDENTITY OVER >= 20 REAL RETURNS (criterion 3)')
  const fx = JSON.parse(fs.readFileSync(FIXTURES, 'utf8'))
  const rets = fx.returns
  check('the fixture set holds at least 20 REAL returns', rets.length >= 20,
    rets.length + ' returns, pinned ' + fx._provenance.pin_iso)
  const seen = new Set(rets.map(r => r.verdict))
  check('...spanning all three verdict values',
    ['PASS', 'CONDITIONAL', 'FAIL'].every(v => seen.has(v)), [...seen].join(','))
  let identical = 0
  let mutatedInput = 0
  for (const r of rets) {
    const before = r.verdict
    const beforeJson = JSON.stringify(r)
    const out = f(r)
    if (out && before === r.verdict) identical++
    if (JSON.stringify(r) !== beforeJson) mutatedInput++
  }
  check('the judge\'s verdict string is byte-identical after routing, on every return',
    identical === rets.length, identical + '/' + rets.length + ' by string equality')
  check('...and the input object itself is never mutated',
    mutatedInput === 0, mutatedInput + ' mutated')

  section('G. THE FIXTURE BUCKETS REPRODUCE (criterion 4, in miniature)')
  let bucketOk = 0
  const bucketMiss = []
  for (const r of rets) {
    const route = f(r).route
    const want = r.expected_bucket === 'CONDITIONAL_allwn' ? 'queue_residual' : 'remediate'
    if (route === want) bucketOk++
    else bucketMiss.push(`${r.run_id} want=${want} got=${route}`)
  }
  check('every fixture return routes as its recorded bucket predicts',
    bucketOk === rets.length, bucketOk + '/' + rets.length
      + (bucketMiss.length ? ' | ' + bucketMiss.join('; ') : ''))
  const failRets = rets.filter(r => r.verdict === 'FAIL')
  check('no real FAIL in the fixture set routes to queue_residual',
    failRets.every(r => f(r).route === 'remediate'), failRets.length + ' FAILs')

  section('H. THE queue_residual CONSUMER REFUSES A PARENT CLOSE (criterion 5)')
  const planWithResidual = JSON.stringify({
    phases: [{ id: 'phase-90', steps: [
      { id: '90.1', verification: { command: 'x', success_criteria: ['a'] } },
      { id: '90.12', name: 'the residual', audit_basis: 'raised by step 90.1 cycle 5',
        verification: { command: 'python3 x.py --verify', success_criteria: ['a'] } },
    ] }],
  })
  const planNoResidual = JSON.stringify({
    phases: [{ id: 'phase-90', steps: [
      { id: '90.1', verification: { command: 'x', success_criteria: ['a'] } },
    ] }],
  })
  const planToothless = JSON.stringify({
    phases: [{ id: 'phase-90', steps: [
      { id: '90.1', verification: { command: 'x', success_criteria: ['a'] } },
      { id: '90.12', audit_basis: 'raised by step 90.1', verification: { command: '  ', success_criteria: [] } },
    ] }],
  })
  const q = { route: 'queue_residual' }
  const rNo = refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: q, masterplan: planNoResidual })
  check('queue_residual + NO filed residual -> close REFUSED', rNo.allowed === false)
  const rTooth = refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: q, masterplan: planToothless })
  check('queue_residual + a residual that grades NOTHING -> close REFUSED',
    rTooth.allowed === false, rTooth.filed.join(','))
  const rYes = refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: q, masterplan: planWithResidual })
  check('queue_residual + a properly filed residual -> close ALLOWED',
    rYes.allowed === true, 'filed=' + rYes.filed.join(','))
  const rRem = refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: { route: 'remediate' }, masterplan: planNoResidual })
  check('remediate + nothing filed -> close ALLOWED (the gate binds only on queue_residual)',
    rRem.allowed === true)
  const rBad = refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: q, masterplan: '{not json' })
  check('an unparseable plan REFUSES rather than failing open',
    rBad.allowed === false, 'fail-closed')
  const planNeighbour = JSON.stringify({
    phases: [{ id: 'phase-90', steps: [
      { id: '90.100', audit_basis: 'raised by step 90.10', verification: { command: 'y', success_criteria: ['a'] } },
    ] }],
  })
  check('a residual filed for 90.10 does NOT satisfy a debt owed by 90.1',
    refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: q, masterplan: planNeighbour }).allowed === false)
  check('...and the parent cannot be its own residual',
    refuseCloseOnUnfiledResidual({
      parentStepId: '90.1', routing: q,
      masterplan: JSON.stringify({ phases: [{ steps: [{ id: '90.1', notes: 'mentions 90.1',
        verification: { command: 'x', success_criteria: ['a'] } }] }] }),
    }).allowed === false)
  // The gate must work against the REAL plan of record, not only fixtures.
  const realPlan = fs.readFileSync(path.join(REPO, '.claude/masterplan.json'), 'utf8')
  const rReal = refuseCloseOnUnfiledResidual({ parentStepId: '90.1', routing: q, masterplan: realPlan })
  check('driven against the REAL .claude/masterplan.json, 90.1\'s residual IS filed',
    rReal.allowed === true, 'filed=' + rReal.filed.join(',') + ' over ' + rReal.checked + ' steps')

  section('I. THE SIBLING INVARIANT IS EXTENDED, NOT REINVENTED (criterion 1)')
  const wfSrc = fs.readFileSync(WORKFLOW, 'utf8')
  check('severity_routing is spread as a NAMED SIBLING of the judge object',
    /const merged = \{ \.\.\.verdict, escalation, research_routing, severity_routing \}/.test(wfSrc))
  check('...and never flattened into it',
    !/\{\s*\.\.\.verdict,[^}]*\.\.\.severity_routing/.test(wfSrc))
  check('a phase-90.2 leak guard exists at the same throw-site as its two siblings',
    /phase-90\.2 invariant violated: severity_routing fields leaked/.test(wfSrc))
  check('...and it has NO carve-out, unlike the research_routing guard (strictly stronger)',
    /const leakedS = Object\.keys\(severity_routing\)\.filter\(k => k !== 'severity_routing' && k in merged\)/.test(wfSrc))
  // DRIVEN, NOT GREPPED. Each of these fails if the guard is neutered, which the
  // four source scans above provably do not catch.
  const g = mod.leakGuard
  check('the leak guard is EXTRACTABLE and callable -- a deleted if/throw is caught '
    + 'here, not merely missed by a regex', typeof g === 'function', typeof g)
  if (typeof g === 'function') {
    const sr = condAllWarn
    const judge = V('CONDITIONAL', ['(WARN) x'])
    let threw = null
    try { g({ ...judge, escalation: {}, research_routing: {}, severity_routing: sr }, sr) }
    catch (e) { threw = e.message }
    check('...it does NOT throw on the correct sibling shape', threw === null, threw || 'no throw')
    let flat = null
    try { g({ ...judge, ...sr, severity_routing: sr }, sr) } catch (e) { flat = e.message }
    check('...it DOES throw when the routing object is FLATTENED into the verdict',
      flat !== null && /phase-90\.2 invariant violated/.test(flat), (flat || 'NO THROW').slice(0, 70))
    let coll = null
    try { g({ ...judge, route: 'remediate', severity_routing: sr }, sr) } catch (e) { coll = e.message }
    check('...and when a JUDGE field collides with a routing key ("route")',
      coll !== null && /route/.test(coll), (coll || 'NO THROW').slice(0, 70))
    let empty = null
    try { g({ ...judge, severity_routing: {} }, {}) } catch (e) { empty = e.message }
    check('...and it does not throw on an empty routing object (no false positive)',
      empty === null, empty || 'no throw')
  } else {
    check('...it does NOT throw on the correct sibling shape', false, 'guard absent')
    check('...it DOES throw when the routing object is FLATTENED into the verdict', false, 'guard absent')
    check('...and when a JUDGE field collides with a routing key ("route")', false, 'guard absent')
    check('...and it does not throw on an empty routing object (no false positive)', false, 'guard absent')
  }

  const routingKeys = Object.keys(condAllWarn)
  const judgeKeys = Object.keys(V('CONDITIONAL', []))
  const collide = routingKeys.filter(k => judgeKeys.includes(k))
  check('no routing key collides with any judge field name',
    collide.length === 0, collide.join(',') || 'none')

  section('J. VERDICT SEMANTICS FROZEN')
  check('the routing object cannot express a verdict value',
    !('verdict' in condAllWarn) && !('ok' in condAllWarn))
  check('no VERDICT_SCHEMA edit was made (86.98 is not pre-empted)',
    !/severity/.test(wfSrc.slice(wfSrc.indexOf('const VERDICT_SCHEMA'),
      wfSrc.indexOf('const VERDICT_SCHEMA') + 2200)))
}

// ---------------------------------------------------------------------------
// Mutation matrix (criterion 6) -- control GREEN first, typed ERROR probe
// ---------------------------------------------------------------------------

const MUTANTS = [
  { id: 'N0', expect: 'SURVIVED', why: 'NULL MUTANT (comment only). If this scores KILLED the harness is broken and every other kill this run is meaningless.',
    apply: s => s.replace('function enforceSeverityRouting (verdict) {',
      'function enforceSeverityRouting (verdict) { // null mutant') },
  { id: 'M1', expect: 'KILLED', why: 'criterion 2, NAMED: the verdict guard is removed, so a FAIL whose findings are all WARN-tagged can be filed away instead of fixed',
    apply: s => s.replace("const route = (verdict.verdict === 'CONDITIONAL' && allResidual)",
      'const route = (allResidual)') },
  { id: 'M2', expect: 'KILLED', why: 'criterion 6, NAMED: an UNTAGGED finding is treated as a NOTE, so a run mixing a WARN with an unclassified defect is filed as residual',
    apply: s => s.replace("  return 'UNTAGGED'\n}", "  return 'NOTE'\n}") },
  { id: 'M3', expect: 'KILLED', why: 'criterion 6, NAMED: a reported finding is silently dropped from the return',
    apply: s => s.replace('const derived = entries.map((e, i) => ({ index: i, severity: deriveSeverity(e) }))',
      'const derived = entries.slice(1).map((e, i) => ({ index: i, severity: deriveSeverity(e) }))') },
  { id: 'M4', expect: 'KILLED', why: 'BLOCK stops dominating, so a real blocker co-occurring with a WARN is filed away',
    apply: s => s.replace("  if (t.includes('BLOCK')) return 'BLOCK'\n", '') },
  { id: 'M5', expect: 'KILLED', why: 'the delimiter requirement is dropped -- the naive token-anywhere matcher the measurement rejected, which also fires on identifiers',
    apply: s => s.replace('    if (!(delimited || punctuated || initial)) continue\n', '') },
  { id: 'M6', expect: 'KILLED', why: 'absence is recorded as a VALUE: disagreed becomes false rather than null when there is nothing to compare',
    apply: s => s.replace('  const disagreed = comparable\n    ? governing.some((s, i) => derivedOnly[i] !== s)\n    : null',
      '  const disagreed = comparable\n    ? governing.some((s, i) => derivedOnly[i] !== s)\n    : false') },
  { id: 'M7', expect: 'KILLED', why: 'an EMPTY findings list counts as all-residual, so a CONDITIONAL with no findings is filed away',
    apply: s => s.replace('  const allResidual = entries.length > 0 && governing.length > 0\n    && governing.every(s => s === \'WARN\' || s === \'NOTE\')',
      '  const allResidual = governing.every(s => s === \'WARN\' || s === \'NOTE\')') },
  { id: 'M8', expect: 'KILLED', why: 'the immediate-negator check is removed, so "no WARN fired" reads as a WARN tag',
    apply: s => s.replace('    if (IMMEDIATE_NEGATOR.test(before)) continue\n', '') },
  { id: 'M9', expect: 'KILLED', why: 'the judge-emitted branch is ignored, so 86.98\'s "severity comes from the judge" is silently unimplementable',
    apply: s => s.replace('  const anyEmitted = emitted.some(s => s !== null)',
      '  const anyEmitted = false') },
  { id: 'M11', expect: 'KILLED', why: 'criterion 6 clause 2, NAMED: a reported finding is silently dropped from `governing_severities` IN THE RETURN LITERAL -- the cycle-2 survivor. M3 mutates the shared source array and cannot reach this site.',
    apply: s => s.replace('    governing_severities: governing,',
      '    governing_severities: governing.length > 1 ? governing.slice(1) : governing,') },
  { id: 'M12', expect: 'KILLED', why: 'the judge-emitted list governs even when it does not line up with the findings, so untagged blockers and an empty findings list reach queue_residual on the 86.98 branch',
    apply: s => s.replace('  const governing = comparable ? emitted.map(s => s === null ? \'UNTAGGED\' : s) : derivedOnly',
      '  const governing = anyEmitted ? emitted.map(s => s === null ? \'UNTAGGED\' : s) : derivedOnly') },
  { id: 'M14', expect: 'KILLED', why: 'IMMEDIATE_NEGATOR is widened back to the 45-char proximity window this step measured and discarded -- survived cycle 2 while moving one real run out of queue_residual',
    apply: s => s.replace(
      "const IMMEDIATE_NEGATOR = /\\b(no|zero|none|without|neither|nor)\\s+(?:\\w+\\s+){0,1}$/i",
      "const IMMEDIATE_NEGATOR = /\\b(no|zero|none|without|neither|nor)\\b[\\s\\S]{0,45}$/i") },
  { id: 'L1', expect: 'KILLED', why: 'criterion 1, NAMED: the leak guard is made unreachable (`if (false && ...)`) while its literal text survives -- the illusory-guard shape the source scans could not see',
    apply: s => s.replace('if (leakedS.length > 0) {', 'if (false && leakedS.length > 0) {') },
  { id: 'L2', expect: 'KILLED', why: 'criterion 1, NAMED: the if/throw is DELETED and the invariant message left behind in a comment, so every regex still matches',
    apply: s => s.replace(
      "if (leakedS.length > 0) {\n  throw new Error('phase-90.2 invariant violated: severity_routing fields leaked '\n    + 'into the verdict object as top-level siblings: ' + leakedS.join(', '))\n}",
      "// phase-90.2 invariant violated: severity_routing fields leaked into the verdict object as top-level siblings") },
  { id: 'QX', expect: 'ERROR', why: 'ERROR CONTROL: a call site is renamed, so the code cannot RESOLVE A NAME and never runs. It must score ERROR, never a kill.',
    apply: s => s.replace('severity: deriveSeverity(e)', 'severity: deriveSeverity_v2(e)') },
]

// A mutant that cannot run fails to RESOLVE A NAME. A mutant that runs and
// misbehaves raises a DOMAIN exception, or simply fails checks. Typing the
// discriminator is what stops an over-eager probe from silently deleting a
// legitimate cell -- the 90.1 cycle-4 lesson (it flagged a good cell as ERROR)
// AND the 90.1 cycle-5 lesson (a name failure swallowed by a handler emits no
// traceback, so the TYPE must be read from the message too, never the shape).
const UNRESOLVABLE = /\b(ReferenceError|TypeError: .*is not a function|SyntaxError|ERR_MODULE_NOT_FOUND|is not defined)\b/

async function runMatrix () {
  section('K. MUTATION MATRIX (criterion 6) -- control observed GREEN first')
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const rows = []
  for (const m of MUTANTS) {
    const mutated = m.apply(src)
    if (mutated === src) { rows.push([m.id, 'NO-OP', m.expect, m.why]); continue }
    let score
    try {
      const mod = await load(mutated)
      const failures = await scoreAgainstCells(mod)
      score = failures > 0 ? 'KILLED' : 'SURVIVED'
    } catch (e) {
      score = UNRESOLVABLE.test(String(e && (e.message || e))) ? 'ERROR' : 'KILLED'
    }
    rows.push([m.id, score, m.expect, m.why])
  }
  let bad = 0
  for (const [id, got, want, why] of rows) {
    const ok = got === want
    if (!ok) bad++
    console.log(`  ${ok ? 'ok  ' : 'BAD '} ${id.padEnd(4)} ${String(got).padEnd(9)} expected ${want}`)
    console.log(`         ${why}`)
  }
  const nullRow = rows.find(r => r[0] === 'N0')
  const killRow = rows.find(r => r[0] === 'M1')
  const errRow = rows.find(r => r[0] === 'QX')
  check('null mutant SURVIVED (the harness is not scoring everything as a kill)',
    nullRow && nullRow[1] === 'SURVIVED', nullRow && nullRow[1])
  check('a real-kill control was KILLED on the same run (the harness is not scoring '
    + 'everything as a survivor)', killRow && killRow[1] === 'KILLED', killRow && killRow[1])
  check('a mutant that cannot RESOLVE A NAME scored ERROR, never a kill',
    errRow && errRow[1] === 'ERROR', errRow && errRow[1])
  check('every mutation cell scored as expected', bad === 0, bad + ' unexpected')
  check('no cell was a NO-OP (a mutation that changed nothing tests nothing)',
    !rows.some(r => r[1] === 'NO-OP'),
    rows.filter(r => r[1] === 'NO-OP').map(r => r[0]).join(',') || 'none')
  return rows
}

// Cells re-run against a mutant, counting failures. Kept separate from runChecks
// so the mutant is scored on the SAME assertions the control passed.
async function scoreAgainstCells (mod) {
  const saved = results.length
  const savedLog = console.log
  console.log = () => {}
  try { await runChecks(mod) } finally { console.log = savedLog }
  const produced = results.splice(saved)
  return produced.filter(r => !r[1]).length
}

// ---------------------------------------------------------------------------
// Replay over the live corpus (criterion 4) -- separate flag, never in --self-test
// ---------------------------------------------------------------------------

const PIN_MS = 1787056437731 // 2026-08-18T12:33:57.731Z -- the brief's last record

function loadCorpus (pinMs) {
  const base = path.join(os.homedir(), '.claude/projects/-Users-ford--openclaw-workspace-pyfinagent')
  const out = []
  const census = { startsWith: 0, exact: 0, parseable: 0, withVerdict: 0, variantNames: new Set() }
  if (!fs.existsSync(base)) return { out, census }
  for (const d of fs.readdirSync(base)) {
    const wd = path.join(base, d, 'workflows')
    if (!fs.existsSync(wd) || !fs.statSync(wd).isDirectory()) continue
    for (const fn of fs.readdirSync(wd)) {
      if (!fn.endsWith('.json')) continue
      let rec
      try { rec = JSON.parse(fs.readFileSync(path.join(wd, fn), 'utf8')) } catch { continue }
      if (pinMs !== null && (rec.startTime || 0) > pinMs) continue
      const wn = rec.workflowName || ''
      if (!(typeof wn === 'string' && wn.startsWith('qa-verdict'))) continue
      census.startsWith++
      if (wn === 'qa-verdict') census.exact++
      else census.variantNames.add(wn)
      // THE POPULATION IS DERIVED, NOT CHOSEN. masterplan 90.2's audit_basis names
      // "441 `qa-verdict` Workflow run records", and 441 is the startsWith count;
      // the exact-match count is 436. The first version of this replay filtered on
      // `wn !== 'qa-verdict'`, which DROPPED 5 records -- 3 of them non-PASS -- and
      // turned the filing's 247 into 244. I then wrote a paragraph explaining why
      // 247 "does not reproduce". It reproduces exactly; the scope was mine.
      // Caught by the 90.2 cycle-1 Q/A.
      let r = rec.result
      if (typeof r === 'string') { try { r = JSON.parse(r) } catch { r = null } }
      if (!r || typeof r !== 'object') continue
      census.parseable++
      if (typeof r.verdict !== 'string' || !r.verdict) continue
      census.withVerdict++
      out.push({ runId: rec.runId, ret: r })
    }
  }
  return { out, census }
}

// The FILING's matcher, reimplemented verbatim from its description ("token
// anywhere, BLOCK excluded"). Kept so the shipped matcher can be compared against
// a rule authored by someone else, rather than only against itself.
const filingMatcher = (e) => !e.includes('BLOCK') && (e.includes('WARN') || e.includes('NOTE'))

async function replay () {
  const mod = await load()
  for (const [label, pin] of [['PINNED @ 2026-08-18T12:33:57.731Z', PIN_MS], ['LIVE (no pin)', null]]) {
    const { out, census } = loadCorpus(pin)
    const nonPass = out.filter(x => x.ret.verdict !== 'PASS')
    const shipped = { queue_residual: [], remediate: [] }
    const filing = { queue_residual: [], remediate: [] }
    for (const x of nonPass) {
      shipped[mod.enforceSeverityRouting(x.ret).route].push(x.runId)
      const entries = (x.ret.violated_criteria || []).filter(e => typeof e === 'string')
      const fq = x.ret.verdict === 'CONDITIONAL' && entries.length > 0 && entries.every(filingMatcher)
      filing[fq ? 'queue_residual' : 'remediate'].push(x.runId)
    }
    const onlyShipped = shipped.queue_residual.filter(r => !filing.queue_residual.includes(r))
    const onlyFiling = filing.queue_residual.filter(r => !shipped.queue_residual.includes(r))
    console.log(`\n${label}`)
    console.log(`  POPULATION (DERIVED from masterplan 90.2 audit_basis, "441 qa-verdict `
      + `Workflow run records"): workflowName.startsWith('qa-verdict')`)
    console.log(`  DENOMINATORS: startsWith=${census.startsWith} (exact-match=${census.exact}, `
      + `+${census.startsWith - census.exact} under variant names `
      + `${JSON.stringify([...census.variantNames])}) `
      + `parseable=${census.parseable} with_verdict=${census.withVerdict} non-PASS=${nonPass.length}`)
    console.log(`  verdict mix: ` + JSON.stringify(out.reduce((a, x) => {
      a[x.ret.verdict] = (a[x.ret.verdict] || 0) + 1; return a }, {})))
    console.log(`  A. the FILING's matcher (token anywhere)  queue_residual=${filing.queue_residual.length}  remediate=${filing.remediate.length}`)
    console.log(`  B. the SHIPPED matcher (delimited tag)    queue_residual=${shipped.queue_residual.length}  remediate=${shipped.remediate.length}`)
    console.log(`  DISAGREEMENT  A-only=${onlyFiling.length} ${JSON.stringify(onlyFiling)}  B-only=${onlyShipped.length} ${JSON.stringify(onlyShipped)}`)
    console.log(`  FAILs routed to queue_residual by the shipped matcher: `
      + nonPass.filter(x => x.ret.verdict === 'FAIL'
        && mod.enforceSeverityRouting(x.ret).route === 'queue_residual').length
      + '  (structurally impossible, printed to show it)')
    if (pin === PIN_MS) {
      console.log('\n  queue_residual run ids at the pin:')
      for (const r of shipped.queue_residual) console.log('    ' + r)
    }
  }
  console.log(`
  THE FILED COUNTS (criterion 4):
    "41"  REPRODUCES EXACTLY at the pin, under BOTH matchers, with identical run sets.
    "247" REPRODUCES EXACTLY at the pin, on the DERIVED population. 441 records ->
          397 verdicts (PASS 109 / CONDITIONAL 221 / FAIL 67) -> 288 non-PASS ->
          41 queue_residual + 247 remediate.
          CORRECTED: the first version of this replay filtered on an exact
          workflowName match, dropping 5 records under the variant names
          qa-verdict-writefirst-82-5 (x3) and -82-7 (x2). Three of those are
          non-PASS, which is exactly 247 - 3 = 244, and I published 244 together
          with a paragraph asserting that 247 "does not reproduce" and blaming a
          43-of-436 result:null gap -- which explains the 436->393 PARSEABLE gap,
          a different gap entirely. The scope was CHOSEN, not derived; the filing
          names 441, and 441 is the startsWith count.
    "32"  (strict) DOES NOT reproduce under any of four plausible strict definitions:
          token-anywhere 41, bracketed-anywhere 26, starts-with-bare 11,
          starts-with-separator 4. Measured under BOTH populations. The filing's
          strict definition is not recoverable from its text, and that number is
          NOT edited to match either.`)
}

// ---------------------------------------------------------------------------

async function main () {
  const argv = process.argv.slice(2)
  if (argv.includes('--replay')) { await replay(); return 0 }

  const ledgerBefore = fs.existsSync(VERDICT_LEDGER) ? sha(VERDICT_LEDGER) : 'ABSENT'
  const mod = await load()
  await runChecks(mod)
  await runMatrix()

  const ledgerAfter = fs.existsSync(VERDICT_LEDGER) ? sha(VERDICT_LEDGER) : 'ABSENT'
  section('L. THE RUN CHANGED NOTHING IT SHOULD NOT HAVE')
  check('handoff/verdict_ledger.jsonl sha256 byte-identical before and after',
    ledgerBefore === ledgerAfter, ledgerBefore.slice(0, 16) + ' -> ' + ledgerAfter.slice(0, 16))

  section('SUMMARY')
  const failed = results.filter(r => !r[1])
  console.log(`  checks run: ${results.length} (floor ${EXPECTED_CHECKS})`)
  console.log(`  failed:     ${failed.length}`)
  for (const [label, , detail] of failed) console.log(`    FAIL ${label} ${detail}`)
  if (results.length < EXPECTED_CHECKS) {
    console.log(`  CARDINALITY FLOOR NOT MET: ${results.length} < ${EXPECTED_CHECKS}. `
      + 'A checker whose loop covers nothing exits 0 and looks identical to success.')
    return 1
  }
  return failed.length === 0 ? 0 : 1
}

main().then(c => process.exit(c)).catch(e => {
  console.error(e && e.stack ? e.stack : String(e))
  process.exit(1)
})
