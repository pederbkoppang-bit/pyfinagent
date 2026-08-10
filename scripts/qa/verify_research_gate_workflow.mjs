#!/usr/bin/env node
/**
 * phase-36.27 re-runnable checker for `.claude/workflows/research-gate.js`.
 *
 * WHY THIS FILE EXISTS. The step's immutable verification command is:
 *     node --check .claude/workflows/research-gate.js && ls .claude/workflows/research-gate.js
 * That proves the file PARSES and EXISTS -- criterion 1, and nothing else.
 * Criteria 3 (every floor enforced), 4 (an EMPTY return is a failed gate) and 6
 * (MUTATION-TEST: weakening a floor must fail the check enforcing it) are out of
 * its reach. The criteria are immutable, so the command is not amended; this
 * checker carries the rest and is run alongside it.
 *
 * MEASURED, and worth stating because it is the sharper point: `node --check`
 * PASSED on a version of this workflow that could not run at all. A static
 * `import fs from 'node:fs'` is valid ESM but the Workflow runtime has no
 * filesystem/Node API access, so launching it failed with
 *   SyntaxError: Unexpected identifier 'fs'. import call expects one or two arguments.
 * A green `node --check` is therefore not evidence that the script works.
 *
 * This drives the REAL `enforceGate` exported by the workflow -- not a copy. A
 * checker that re-implemented the logic would stay green while production drifted.
 *
 * Spawns nothing, costs nothing, touches no live state.
 *
 *     node scripts/qa/verify_research_gate_workflow.mjs
 *     -> exit 0 all green, exit 1 with the failing case named
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const WORKFLOW = path.join(REPO, '.claude', 'workflows', 'research-gate.js')

let pass = 0
const failures = []
function check(name, cond, detail) {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { failures.push(`${name}${detail ? ' -- ' + detail : ''}`); console.log(`  FAIL ${name}${detail ? ' -- ' + detail : ''}`) }
}

// The workflow's top level calls phase()/agent()/log(), which exist only inside
// the Workflow runtime. Import just the pure exports by stripping everything
// from the `phase('Research')` driver boundary onward.
async function loadModule(sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(WORKFLOW, 'utf8')
  const idx = src.indexOf("phase('Research')")
  if (idx === -1) throw new Error("could not find the phase('Research') driver boundary")
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'rg-')), 'rg.mjs')
  // The workflow carries NO export list: the runtime accepts only the leading
  // `export const meta` and rejects a trailing one outright. So the export is
  // appended HERE, to the stripped copy, keeping the shipped file launchable.
  fs.writeFileSync(tmp, src.slice(0, idx)
    + '\nexport { enforceGate, ENVELOPE_SCHEMA, BRIEF_VERIFICATION_SCHEMA, FLOOR_SOURCES, FLOOR_URLS }\n')
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.enforceGate !== 'function') throw new Error('research-gate.js does not export enforceGate')
  return mod
}

/** phase-86.28 cycle 3 -- load the WHOLE script as a DRIVABLE function.
 *
 *  WHY THIS EXISTS. Two successive Q/A passes defeated a source-scan assertion
 *  of the property "no researcher is spawned on an unsupported tier" -- first
 *  with a `//` comment token, then with a `/* *​/` block comment. Patching the
 *  regex a third time is playing the wrong game: the property is BEHAVIOURAL
 *  (was `agent()` called?), and this step's own research finding F6 says it
 *  outright -- "structural is not semantic". So we drive the real driver with
 *  a RECORDING stub for `agent()` and count the calls. No comment, string
 *  literal, template literal or whitespace trick can survive that, because the
 *  test no longer reads the source at all.
 *
 *  loadModule() slices the file at `phase('Research')` and keeps only the
 *  definitions, so it cannot reach the driver. Here the whole file is wrapped
 *  in an async function instead: that legalises the script's top-level
 *  `return` and `await` (the Workflow runtime wraps it the same way), and
 *  `export const meta` becomes a plain const inside the wrapper.
 *
 *  `args` becomes a PARAMETER, which preserves the `typeof args === 'undefined'`
 *  boundary check exactly: calling __drive() with no argument leaves it
 *  genuinely unbound-equivalent, which is the documented dry-run shape.
 */
async function loadDriver(sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(WORKFLOW, 'utf8')
  const body = src.replace('export const meta', 'const meta')
  // Match a real export STATEMENT (line-start), not the words "export { ... }"
  // that appear inside this file's own explanatory comments.
  if (/^\s*export\s/m.test(body)) throw new Error('unexpected residual export statement in the driver body')
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'rgdrive-')), 'drive.mjs')
  fs.writeFileSync(tmp, 'export async function __drive(args, phase, log, agent) {\n' + body + '\n}\n')
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.__drive !== 'function') throw new Error('could not build a drivable copy')
  return mod.__drive
}

/** Run the driver, recording every agent() spawn. Returns {result, spawns}. */
async function driveRecording(drive, driverArgs) {
  const spawns = []
  const agentStub = async (prompt, opts) => { spawns.push({ prompt, opts }); return null }
  const result = await drive(driverArgs, () => {}, () => {}, agentStub)
  return { result, spawns }
}

// phase-86.28: the fixture brief is now COMPLIANT with what the rules actually
// require of a brief -- a read-in-full table, a snippet-only table, and a
// DEDICATED recency-scan section (.claude/rules/research-gate.md requires the
// section "even when empty"). Before this, the fixture omitted both the
// snippet table and the recency section, so it could not exercise the two
// corroboration checks at all. `withRecency:false` builds the non-compliant
// variant the recency probe needs.
function makeBrief(readUrls, snippetUrls = [], withRecency = true) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rgbrief-'))
  const p = path.join(dir, 'research_brief_TEST.md')
  const parts = [
    '# brief',
    '',
    '## Sources read in full',
    ...readUrls.map(u => `| ${u} | read in full |`),
    '',
    '## Snippet-only sources',
    ...snippetUrls.map(u => `| ${u} | snippet only |`),
    '',
    // phase-86.37: the fixture brief declares its own born-inert marker, so the
    // happy path exercises brief_status_in_brief === 'COMPLETE' rather than
    // leaving it undefined.
    '```json',
    '{ "brief_status": "COMPLETE" }',
    '```',
    '',
  ]
  if (withRecency) parts.push('## Recency scan (last 2 years)', '', 'No relevant new findings in the window.', '')
  fs.writeFileSync(p, parts.join('\n'))
  return p
}

// Stage 2 is told to look for a dedicated recency-scan section HEADING. This
// mirrors that instruction; it deliberately does NOT try to judge whether the
// scan was substantive -- see the naming-discipline note in research-gate.js.
const RECENCY_HEADING_RE = /^#{1,6}\s.*recency\s+scan/im
const URL_RE = /https?:\/\/[^\s|)\]<>"']+/g

/** Stage 2's job, performed here with real fs. The CHECKER may touch disk; only
 *  the Workflow runtime may not. This produces exactly what the verifier agent
 *  returns at runtime, so enforceGate is driven through its real interface.
 *  phase-86.28: must now also produce the two fields the schema newly requires,
 *  or the checker would be driving a stage-2 shape that can no longer occur. */
function verifyBrief(briefPath, urls) {
  let text = null
  try { text = fs.readFileSync(briefPath, 'utf8') } catch (_e) { text = null }
  const missing = text === null ? urls.slice() : urls.filter(u => !text.includes(u))
  return {
    brief_exists: text !== null,
    brief_non_empty: text !== null && text.trim().length > 0,
    char_count: text === null ? 0 : text.length,
    urls_checked: urls.length,
    urls_present: urls.length - missing.length,
    urls_missing: missing,
    recency_section_present: text !== null && RECENCY_HEADING_RE.test(text),
    distinct_urls_in_brief: text === null ? 0 : new Set(text.match(URL_RE) || []).size,
    // phase-86.37: read the born-inert marker out of the brief, the way stage 2
    // is asked to. Derived from the FILE, never assumed.
    brief_status_in_brief: text === null ? 'ABSENT'
      : (/"brief_status"\s*:\s*"COMPLETE"/.test(text) ? 'COMPLETE'
        : (/"brief_status"\s*:\s*"INCOMPLETE"/.test(text) ? 'INCOMPLETE' : 'ABSENT')),
  }
}

const URLS = Array.from({ length: 8 }, (_, i) => `https://example.com/source-${i + 1}`)
// 8 read-in-full + 17 snippet-only = 25 distinct, matching goodEnvelope()'s
// urls_collected: 25 and snippet_only_sources: 17. The fixture is now
// internally consistent, which is what makes the urls corroboration testable.
const SNIPPET_URLS = Array.from({ length: 17 }, (_, i) => `https://example.com/snippet-${i + 1}`)
const briefPath = makeBrief(URLS, SNIPPET_URLS)
const briefPathNoRecency = makeBrief(URLS, SNIPPET_URLS, false)

function goodEnvelope(over = {}) {
  return {
    tier: 'moderate',
    external_sources_read_in_full: 8,
    sources_read_in_full: URLS,
    snippet_only_sources: 17,
    urls_collected: 25,
    recency_scan_performed: true,
    internal_files_inspected: 10,
    coverage: { audit_class: false, rounds: 1, dry_rounds: 0, K_required: 2, new_findings_last_round: 0, dry: false },
    brief_path: briefPath,
    summary: 'x',
    gate_passed: true,
    ...over,
  }
}

/** Sentinel: distinguishes "caller supplied nothing" from "caller supplied
 *  `undefined` on purpose". Using `!== undefined` for that conflates the two,
 *  which silently turned the `verification undefined` case into a real
 *  verification and made it pass -- caught by this checker's own first run. */
const NOT_SUPPLIED = Symbol('not-supplied')

/** Drive enforceGate the way the workflow does: envelope + independent verification. */
function makeGate(mod) {
  // phase-86.28: `opts` passthrough. enforceGate takes (env, verification, opts)
  // and the tier classification arrives via opts, so a probe that cannot pass
  // opts cannot exercise the tier check at all. Omitting it keeps the exact
  // previous behaviour (enforceGate treats an absent opts.tier as "no tier
  // information", which is how every pre-86.28 call site drives it).
  return (env, verificationOverride = NOT_SUPPLIED, opts = undefined) => {
    const urls = Array.isArray(env && env.sources_read_in_full) ? env.sources_read_in_full : []
    const v = verificationOverride === NOT_SUPPLIED
      ? verifyBrief((env && env.brief_path) || '', urls)
      : verificationOverride
    return mod.enforceGate(env, v, opts)
  }
}

// phase-86.28 tier fixtures -- the shape the driver builds at its enforceGate call.
//
// CYCLE 5 CORRECTION. TIER_ABSENT previously carried `supported: true`, which
// the driver NEVER produces: it computes
//     const tierSupported = !tierAbsent && VALID_TIERS.includes(tierRequested)
// so an ABSENT tier is supported:FALSE. The fixture therefore could not
// represent the production state, and a fixture that cannot represent the
// failure keeps the suite green for every possible value of the code
// (qa.md 4c shape #5). Measured consequence: mutant Q1
// (`tierUnsupported = !tierAbsent && !tierSupported` -> `= !tierSupported`)
// left the suite at ALL GREEN 73/0 while an absent tier stopped spawning
// entirely -- a total break of every caller that omits `tier`.
//
// The header comment above was itself the tell: it claimed to be "the shape
// the driver builds" and was false on that one field. Fixture fidelity is now
// ASSERTED against the running driver in [6d] rather than claimed in a comment.
const TIER_VALID = ['simple', 'moderate', 'complex']
const tierOpts = (over) => ({ tier: { requested: null, applied: 'moderate', supported: true, absent: false, unsupported: false, valid: TIER_VALID, ...over } })
const TIER_UNSUPPORTED = tierOpts({ requested: 'deep', supported: false, unsupported: true })
const TIER_ABSENT = tierOpts({ requested: null, absent: true, supported: false })
const TIER_OK = tierOpts({ requested: 'complex', applied: 'complex' })

/** A mutant is KILLED if the weakened build lets a previously-rejected envelope
 *  through, OR if it throws. A throw is a kill: it means the removed guard was
 *  the only thing standing between production and a crash on that input. */
function mutantKilled(probe, g) {
  try { return probe(g) === true ? 'let-a-bad-envelope-through' : false }
  catch (e) { return 'threw: ' + (e && e.message ? e.message.split('\n')[0] : String(e)) }
}

const mod = await loadModule()
const G = makeGate(mod)

console.log('\n[1] control -- a compliant envelope passes')
check('control passes', G(goodEnvelope()).gate_passed === true)

console.log('\n[2] criterion 3 -- every floor REJECTS a short-of-floor return (not rounded up)')
{
  const r = G(goodEnvelope({ external_sources_read_in_full: 4, sources_read_in_full: URLS.slice(0, 4) }))
  check('4 sources (<5) rejected', r.gate_passed === false && r.violations.some(v => v.includes('external_sources_read_in_full')))
}
{
  const r = G(goodEnvelope({ urls_collected: 9 }))
  check('9 URLs (<10) rejected', r.gate_passed === false && r.violations.some(v => v.includes('urls_collected')))
}
{
  const r = G(goodEnvelope({ recency_scan_performed: false }))
  check('recency_scan false rejected', r.gate_passed === false && r.violations.some(v => v.includes('recency_scan')))
}
{
  const cov = { audit_class: true, rounds: 3, dry_rounds: 1, K_required: 2, new_findings_last_round: 2, dry: false }
  const r = G(goodEnvelope({ coverage: cov }))
  check('audit-class without coverage.dry rejected', r.gate_passed === false && r.violations.some(v => v.includes('coverage.dry')))
}
{
  const cov = { audit_class: true, rounds: 4, dry_rounds: 2, K_required: 2, new_findings_last_round: 0, dry: true }
  check('audit-class WITH coverage.dry passes', G(goodEnvelope({ coverage: cov })).gate_passed === true)
}
{
  // The floor is a FLOOR, never a ceiling: audit-class at 4 sources still fails.
  const cov = { audit_class: true, rounds: 4, dry_rounds: 2, K_required: 2, new_findings_last_round: 0, dry: true }
  const r = G(goodEnvelope({ coverage: cov, external_sources_read_in_full: 4, sources_read_in_full: URLS.slice(0, 4) }))
  check('audit-class + dry but 4 sources STILL rejected (floor never lowered)', r.gate_passed === false)
}

console.log('\n[3] criterion 4 -- an EMPTY/errored return is a FAILED gate, never gate_passed')
for (const [label, v] of [['null', null], ['undefined', undefined], ['empty object', {}], ['a string', 'oops'], ['an array', []]]) {
  check(`${label} => gate_passed false`, G(v).gate_passed === false)
}
check('null return is not silently treated as a self-report', G(null).agent_self_reported_gate_passed === null)

console.log('\n[4] the artifact cross-check -- a self-report the brief cannot corroborate FAILS')
{
  const r = G(goodEnvelope({ brief_path: path.join(os.tmpdir(), 'definitely-absent-brief.md') }))
  check('missing brief on disk rejected', r.gate_passed === false && r.violations.some(v => v.includes('brief not found')))
}
{
  const emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rgempty-'))
  const emptyP = path.join(emptyDir, 'b.md'); fs.writeFileSync(emptyP, '   \n')
  const r = G(goodEnvelope({ brief_path: emptyP }))
  check('empty brief rejected', r.gate_passed === false && r.violations.some(v => v.includes('EMPTY')))
}
{
  const r = G(goodEnvelope({ sources_read_in_full: [...URLS.slice(0, 7), 'https://example.com/never-actually-read'] }))
  check('a source claimed but ABSENT from the brief rejected', r.gate_passed === false && r.violations.some(v => v.includes('ABSENT from the brief')))
}
{
  const r = G(goodEnvelope({ external_sources_read_in_full: 8, sources_read_in_full: URLS.slice(0, 3) }))
  check('over-claim (8 claimed, 3 listed) rejected', r.gate_passed === false && r.violations.some(v => v.includes('over-claim')))
}

console.log('\n[5] stage-2 verification is itself load-bearing -- absent verification FAILS CLOSED')
// Called through mod.enforceGate DIRECTLY, not through G. A JS default
// parameter fires on an explicitly-passed `undefined`, so no wrapper default
// can distinguish "omitted" from "explicitly undefined" -- the sentinel I
// reached for first could not either. This checker's own first run caught that.
for (const [label, v] of [['null', null], ['undefined', undefined], ['a string', 'nope'], ['an array', []]]) {
  const r = mod.enforceGate(goodEnvelope(), v)
  check(`verification ${label} => gate_passed false (never trust an unverified self-report)`,
    r.gate_passed === false && r.violations.some(x => x.includes('verification did not run')))
}

console.log('\n[6] the agent does not get to grade itself')
{
  const r = G(goodEnvelope({ gate_passed: true, external_sources_read_in_full: 2, sources_read_in_full: URLS.slice(0, 2) }))
  check('agent gate_passed:true is OVERRIDDEN when the floors fail', r.gate_passed === false)
  check('the disagreement is reported', r.self_report_disagreed === true && r.agent_self_reported_gate_passed === true)
}
{
  const r = G(goodEnvelope({ gate_passed: false }))
  check('an honestly-false self-report still passes when the floors hold (const:true would have made this unrepresentable)',
    r.gate_passed === true && r.self_report_disagreed === true)
}

console.log('\n[6b] phase-86.28 -- an UNSUPPORTED tier fails closed; an ABSENT tier does not')
{
  const r = G(goodEnvelope(), NOT_SUPPLIED, TIER_UNSUPPORTED)
  check('UNSUPPORTED tier => gate_passed false (refuses to certify a standard never applied)',
    r.gate_passed === false && r.violations.some(x => x.includes('tier_unsupported')))
  check('the violation names the requested tier and what actually ran',
    r.violations.some(x => x.includes('deep') && x.includes('moderate')))
}
{
  // REGRESSION, caught by the LIVE spawn wf_4da39b31-695 and not by any check
  // that existed at the time: the refuse-to-spawn path calls enforceGate with
  // env === null on purpose, and the first version computed the tier AFTER the
  // empty-envelope guard, so the refusal reported "the agent returned null" --
  // describing a failure of an agent that was never asked to run, while hiding
  // the actionable cause. The message a caller reads must describe what
  // actually happened.
  const r = mod.enforceGate(null, null, TIER_UNSUPPORTED)
  check('refusal path (env=null, unsupported tier) reports the TIER as the cause',
    r.violations.some(x => x.includes('tier_unsupported')))
  check('refusal path does NOT claim an agent returned null (no agent was asked to run)',
    !r.violations.some(x => x.includes('empty_or_errored_return'))
    && !r.checks.some(x => x.includes('the agent returned')))
  check('refusal path still fails closed', r.gate_passed === false)
}
{
  // The converse must not regress: a genuinely empty return on a SUPPORTED
  // tier must still report empty_or_errored_return.
  const r = mod.enforceGate(null, null, TIER_OK)
  check('empty return on a supported tier still reports empty_or_errored_return',
    r.gate_passed === false && r.violations.some(x => x.includes('empty_or_errored_return')))
}
check('ABSENT tier still PASSES (defaulting is legitimate when the caller named nothing)',
  G(goodEnvelope(), NOT_SUPPLIED, TIER_ABSENT).gate_passed === true)
check('a SUPPORTED tier passes', G(goodEnvelope(), NOT_SUPPLIED, TIER_OK).gate_passed === true)
check('absent opts.tier behaves exactly as before (pre-86.28 call sites unaffected)',
  G(goodEnvelope()).gate_passed === true)

console.log('\n[6c] phase-86.28 -- the two formerly-uncorroborated self-reports are checked against the brief')
{
  // recency_scan_performed: claimed true, but the brief carries no section.
  const r = G(goodEnvelope({ brief_path: briefPathNoRecency }))
  check('recency_scan_performed=true with NO recency section in the brief => gate_passed false',
    r.gate_passed === false && r.violations.some(x => x.includes('recency_scan_performed=true')))
}
check('recency corroboration PASSES when the brief carries the section',
  G(goodEnvelope()).gate_passed === true)
{
  // An honest recency_scan_performed:false is rejected by the ORIGINAL check,
  // not the new one -- confirm the new check did not change that path.
  const r = G(goodEnvelope({ recency_scan_performed: false, brief_path: briefPathNoRecency }))
  check('recency_scan_performed=false still fails via the original check, not the corroboration',
    r.gate_passed === false && r.violations.some(x => x.includes('recency_scan_performed is not true')))
}
{
  // urls_collected over-claim: 99 clears the >=10 floor, so the ONLY thing
  // that can reject it is the corroboration against the brief.
  const r = G(goodEnvelope({ urls_collected: 99 }))
  check('urls_collected over-claim (99 claimed, 25 in the brief) => gate_passed false',
    r.gate_passed === false && r.violations.some(x => x.includes('urls_collected=99')))
}
check('urls_collected within what the brief carries PASSES',
  G(goodEnvelope({ urls_collected: 25 })).gate_passed === true)
{
  // The corroboration must NOT fire when stage 2 did not run -- that path is
  // already fail-closed, and a second violation there would mask which guard
  // actually rejected the run.
  const r = mod.enforceGate(goodEnvelope(), null)
  check('absent verification still fails via fail-closed ONLY (corroboration does not double-fire)',
    r.gate_passed === false
    && r.violations.some(x => x.includes('verification did not run'))
    && !r.violations.some(x => x.includes('recency_scan_performed=true'))
    && !r.violations.some(x => x.includes('urls_collected=')))
}

console.log('\n[6d] phase-86.28 cycle 3 -- BEHAVIOURAL: does the driver actually spawn? (replaces the source scan)')
{
  const drive = await loadDriver()

  // KNOWN-POSITIVE FIRST. If the recorder cannot observe a spawn that DOES
  // happen, then "0 spawns" below proves nothing -- it would be the vacuous
  // pass this whole section exists to eliminate. Establish the instrument
  // before trusting its null reading.
  const supported = await driveRecording(drive, { step_id: 'BEHAVE-supported', tier: 'moderate' })
  check('RECORDER WORKS: a SUPPORTED tier really does spawn (known-positive)',
    supported.spawns.length > 0,
    `expected >0 agent() calls, recorded ${supported.spawns.length}`)
  check('the first spawn is the stage-1 researcher (agentType researcher)',
    supported.spawns.length > 0 && supported.spawns[0].opts && supported.spawns[0].opts.agentType === 'researcher')

  // The ABSENT-tier run, driven once and asserted on below.
  const absentRun = await driveRecording(drive, { step_id: 'BEHAVE-absent' })

  // THE PROPERTY ITSELF, observed rather than pattern-matched.
  const unsupported = await driveRecording(drive, { step_id: 'BEHAVE-unsupported', tier: 'deep' })
  check('UNSUPPORTED tier spawns ZERO agents (measured, not scanned)',
    unsupported.spawns.length === 0,
    `recorded ${unsupported.spawns.length} agent() call(s) -- the refusal did not prevent the spawn`)
  check('UNSUPPORTED tier returns gate_passed:false with the tier reported',
    unsupported.result && unsupported.result.gate_passed === false
    && unsupported.result.tier_supported === false
    && unsupported.result.tier_requested === 'deep')
  check('UNSUPPORTED tier does NOT claim an agent returned null',
    unsupported.result && !unsupported.result.violations.some(v => v.includes('empty_or_errored_return')))

  // A blind run must also spawn nothing (86.17's property, now measured too).
  const blind = await driveRecording(drive, undefined)
  check('BLIND run spawns ZERO agents (86.17 property, measured)',
    blind.spawns.length === 0 && blind.result && blind.result.gate_passed === false)

  // CYCLE 5 -- THE CONVERSE, which cycle 3 never guarded. Every check added for
  // the UNSUPPORTED half asserted that NOTHING happens; none asserted that the
  // ABSENT half still WORKS. So a mutation that made absent-tier callers take
  // the refusal path was invisible (Q/A mutant Q1: ALL GREEN 73/0 while every
  // caller omitting `tier` silently stopped spawning). Guarding one direction
  // of a two-way distinction is half a guard.
  check('ABSENT tier still SPAWNS (the converse of the refusal -- Q/A mutant Q1)',
    supported.spawns.length > 0 && absentRun.spawns.length > 0,
    `absent-tier run recorded ${absentRun.spawns.length} agent() call(s); it must still do the work`)
  check('ABSENT tier raises NO tier_unsupported violation',
    absentRun.result && !(absentRun.result.violations || []).some(v => v.includes('tier_unsupported')),
    JSON.stringify((absentRun.result || {}).violations))
  check('ABSENT tier reports tier_requested null and applied moderate',
    absentRun.result && absentRun.result.tier_requested === null && absentRun.result.tier_applied === 'moderate')

  // CYCLE 6 -- a SUPPORTED tier must be APPLIED, not quietly replaced. The
  // cycle-5 known-positive drove at 'moderate', which is also the fallback, so
  // a mutation forcing `tier = 'moderate'` was invisible: the silent-downgrade
  // defect this whole step exists to fix could have been reintroduced for
  // SUPPORTED tiers without any check noticing. Drive at a non-default value.
  const supportedNonDefault = await driveRecording(drive, { step_id: 'BEHAVE-complex', tier: 'complex' })
  check('a SUPPORTED non-default tier is APPLIED, not silently downgraded',
    supportedNonDefault.result && supportedNonDefault.result.tier_applied === 'complex'
    && supportedNonDefault.result.tier_requested === 'complex'
    && supportedNonDefault.result.tier_supported === true,
    `applied=${supportedNonDefault.result && supportedNonDefault.result.tier_applied}`)
  check('a SUPPORTED non-default tier still spawns',
    supportedNonDefault.spawns.length > 0)

  // FIXTURE FIDELITY, asserted rather than claimed in a comment. The enforceGate
  // fixtures must match what the driver actually computes, or every
  // enforceGate-level probe tests a state production cannot reach.
  //
  // CYCLE 6: pin the BRANCH-STEERING field too. Cycle 5 pinned only `supported`
  // -- which enforceGate never reads -- so a fixture whose `unsupported` field
  // drifted would still have gone unnoticed, and `unsupported` is the field
  // that actually selects the branch. Pinning the one field the consumer
  // ignores is a guard aimed slightly to the left of its subject.
  const driverUnsupported = (r) => !!(r && r.tier_requested !== null && r.tier_supported === false)
  const driverAbsent = (r) => !!(r && r.tier_requested === null)

  check('TIER_ABSENT fixture matches the driver (supported:false for an absent tier)',
    TIER_ABSENT.tier.supported === (absentRun.result && absentRun.result.tier_supported),
    `fixture supported=${TIER_ABSENT.tier.supported}, driver tier_supported=${absentRun.result && absentRun.result.tier_supported}`)
  check('TIER_ABSENT fixture matches the driver on the BRANCH-STEERING fields',
    TIER_ABSENT.tier.unsupported === driverUnsupported(absentRun.result)
    && TIER_ABSENT.tier.absent === driverAbsent(absentRun.result),
    `fixture unsupported=${TIER_ABSENT.tier.unsupported}/absent=${TIER_ABSENT.tier.absent}, driver ${driverUnsupported(absentRun.result)}/${driverAbsent(absentRun.result)}`)
  check('TIER_UNSUPPORTED fixture matches the driver (supported:false)',
    TIER_UNSUPPORTED.tier.supported === unsupported.result.tier_supported)
  check('TIER_UNSUPPORTED fixture matches the driver on the BRANCH-STEERING fields',
    TIER_UNSUPPORTED.tier.unsupported === driverUnsupported(unsupported.result)
    && TIER_UNSUPPORTED.tier.absent === driverAbsent(unsupported.result))

  // The fidelity check must REJECT the exact fixture that caused the cycle-5
  // defect. This is the check proving it would have caught the bug it was
  // written for -- not an argument that it would.
  check('fidelity check REJECTS the cycle-4 fixture shape (supported:true for absent)',
    true !== (absentRun.result && absentRun.result.tier_supported))

  // CYCLE 7 -- the DRIVER-BUILT tierInfo, not just the fixture.
  //
  // Surviving mutant found by Q/A run wf_e03ec2d0-c07 before it dropped:
  // `absent: tierAbsent` -> `absent: false` at the driver's enforceGate call
  // left the suite at ALL GREEN 92/0. The two fidelity checks above compare
  // the FIXTURE against the driver's RETURN VALUE, and `absent` is not a
  // return field -- it is internal to the tierInfo object -- so drift in it
  // was unobservable. Its only visible effect is which branch label
  // enforceGate emits, and the label assertions test enforceGate through the
  // FIXTURE rather than through the driver.
  //
  // The label IS observable end-to-end, because the driver returns
  // `checks`. Assert it there, which closes the gap without a production
  // change: a drifted `absent` emits tier_supported_ok instead.
  const absentChecks = (absentRun.result && absentRun.result.checks) || []
  check('DRIVER-built tierInfo yields the ABSENT branch label end-to-end',
    absentChecks.some(c => c.includes('tier_absent_defaulted_ok')),
    JSON.stringify(absentChecks))
  check('...and NOT the supported-branch label',
    !absentChecks.some(c => c.includes('tier_supported_ok')),
    JSON.stringify(absentChecks))
  const unsupChecks = (unsupported.result && unsupported.result.checks) || []
  check('DRIVER-built tierInfo yields NEITHER branch label for UNSUPPORTED',
    !unsupChecks.some(c => c.includes('tier_absent_defaulted_ok'))
    && !unsupChecks.some(c => c.includes('tier_supported_ok')),
    JSON.stringify(unsupChecks))

  // The enforceGate absent-branch LABEL is what makes ABSENT and UNSUPPORTED
  // distinguishable in the checks[] output; guard it directly.
  {
    const r = mod.enforceGate(goodEnvelope(), verifyBrief(briefPath, URLS), TIER_ABSENT)
    check('enforceGate emits the tier_absent_defaulted_ok label for an ABSENT tier',
      (r.checks || []).some(c => c.includes('tier_absent_defaulted_ok')))
    const u = mod.enforceGate(goodEnvelope(), verifyBrief(briefPath, URLS), TIER_UNSUPPORTED)
    check('...and does NOT emit it for an UNSUPPORTED tier',
      !(u.checks || []).some(c => c.includes('tier_absent_defaulted_ok')))
  }

  // VACUITY: the /* */ block-comment decoy that defeated the cycle-2 source
  // scan (Q/A mutant B1). Against a behavioural test it cannot work, because
  // moving the real refusal after the spawn MAKES THE SPAWN HAPPEN.
  const raw = fs.readFileSync(WORKFLOW, 'utf8')
  const blockRe = /if \(tierUnsupported\) \{[\s\S]*?\n\}\n/
  const m = blockRe.exec(raw)
  if (!m) {
    check('B1 block-comment decoy could be constructed', false, 'refusal block not isolatable')
  } else {
    const b1 = raw.replace(m[0], '/* decoy: if (tierUnsupported) { return { } }\n*/\n') + '\n' + m[0]
    let b1Spawns = -1
    try {
      const b1drive = await loadDriver(b1)
      b1Spawns = (await driveRecording(b1drive, { step_id: 'B1', tier: 'deep' })).spawns.length
    } catch (_e) { b1Spawns = -2 }
    check('B1 (block-comment decoy + relocated refusal) IS CAUGHT behaviourally',
      b1Spawns !== 0,
      `the mutated driver spawned ${b1Spawns} agents; a behavioural test must not read 0 here`)
  }
}

console.log('\n[7] criterion 6 MUTATION-TEST -- weakening a floor in the SOURCE must break the check enforcing it')
{
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const mutants = [
    ['FLOOR_SOURCES 5 -> 1', 'const FLOOR_SOURCES = 5', 'const FLOOR_SOURCES = 1',
      (g) => g(goodEnvelope({ external_sources_read_in_full: 4, sources_read_in_full: URLS.slice(0, 4) })).gate_passed],
    ['FLOOR_URLS 10 -> 1', 'const FLOOR_URLS = 10', 'const FLOOR_URLS = 1',
      (g) => g(goodEnvelope({ urls_collected: 9 })).gate_passed],
    ['recency check removed', 'if (env.recency_scan_performed !== true)', 'if (false)',
      (g) => g(goodEnvelope({ recency_scan_performed: false })).gate_passed],
    ['audit-class dry check removed', 'if (cov.dry !== true)', 'if (false)',
      (g) => g(goodEnvelope({ coverage: { audit_class: true, rounds: 3, dry_rounds: 1, K_required: 2, new_findings_last_round: 2, dry: false } })).gate_passed],
    ['over-claim check removed', 'if (listed.length < sources)', 'if (false)',
      (g) => g(goodEnvelope({ external_sources_read_in_full: 8, sources_read_in_full: URLS.slice(0, 3) })).gate_passed],
    ['fail-closed on absent verification removed', "if (!verification || typeof verification !== 'object' || Array.isArray(verification))", 'if (false)',
      (g) => g(goodEnvelope(), null).gate_passed],
    // phase-86.28: one mutant per NEW check. A check whose mutant survives was
    // never load-bearing, so these are the only evidence the new guards work.
    ['tier_unsupported check removed', 'const tierUnsupportedHere = !!(tierInfo && tierInfo.unsupported === true)',
      'const tierUnsupportedHere = false',
      (g) => g(goodEnvelope(), NOT_SUPPLIED, TIER_UNSUPPORTED).gate_passed],
    ['recency corroboration removed', 'if (env.recency_scan_performed === true && verification.recency_section_present !== true)', 'if (false)',
      (g) => g(goodEnvelope({ brief_path: briefPathNoRecency })).gate_passed],
    ['urls corroboration removed', 'if (urls > briefUrls)', 'if (false)',
      (g) => g(goodEnvelope({ urls_collected: 99 })).gate_passed],
  ]
  for (const [name, from, to, probe] of mutants) {
    if (!src.includes(from)) { check(`mutant "${name}" anchor present`, false, `anchor not found: ${from}`); continue }
    const mutated = src.replace(from, to)
    if (mutated === src) { check(`mutant "${name}" actually applied`, false, 'replace was a no-op'); continue }
    let mutG
    try { mutG = makeGate(await loadModule(mutated)) } catch (e) { check(`mutant "${name}" loads`, false, String(e)); continue }
    // Killed if the weakened build lets a previously-REJECTED envelope through,
    // or if it throws. Not killed => the check was never load-bearing.
    const killed = mutantKilled(probe, mutG)
    check(`mutant "${name}" is KILLED [${killed || 'SURVIVED'}]`, killed !== false,
      'the weakened floor did NOT change the outcome -- that check is not load-bearing')
  }
}

console.log('\n[7b] phase-86.28 cycle 6 -- DRIVER-level mutants (the [7] matrix drives enforceGate ONLY)')
{
  // WHY THIS SECTION EXISTS. The [7] matrix mutates research-gate.js and then
  // probes through `makeGate`, i.e. through enforceGate. Every check added in
  // [6d] is DRIVER-level -- it asserts what the module does end to end -- so
  // the [7] matrix structurally cannot demonstrate any of them. Cycle 5 shipped
  // five new checks with mutants for only two, and criterion 5 says a check
  // whose mutant is not demonstrated is not delivered. These probes close that
  // by mutating the source and re-driving the mutated module.
  //
  // Each probe returns the PREDICATE THE CHECK ASSERTS. Mutant killed <=> the
  // predicate goes false on the mutated build.
  const raw = fs.readFileSync(WORKFLOW, 'utf8')
  const driverMutants = [
    // ANCHOR MUST BE UNIQUE TO THE MAIN RETURN. `tier_requested: tierRequested,`
    // appears TWICE -- once in the refusal path and once in the main return --
    // and a first-match replace lands on the refusal path, which the ABSENT
    // probe never executes. That version of this mutant SURVIVED, and it was
    // the mutant that was wrong, not the check. Anchoring on the trailing
    // `tier_supported: tierSupported` (main return only; the refusal path
    // hard-codes `false`) makes it unambiguous. The uniqueness assertion below
    // is what surfaced this.
    ['driver reports the APPLIED tier as tier_requested (main return)',
      'tier_requested: tierRequested,\n  tier_applied: tier,\n  tier_supported: tierSupported,',
      'tier_requested: tier,\n  tier_applied: tier,\n  tier_supported: tierSupported,',
      async (drive) => {
        const r = (await driveRecording(drive, { step_id: 'DM' })).result
        return r && r.tier_requested === null && r.tier_applied === 'moderate'
      },
      'ABSENT tier reports tier_requested null and applied moderate'],

    ['refusal path claims the tier WAS supported',
      'tier_supported: false,\n    brief_path: null,', 'tier_supported: true,\n    brief_path: null,',
      async (drive) => {
        const r = (await driveRecording(drive, { step_id: 'DM', tier: 'deep' })).result
        return r && r.tier_supported === false
      },
      'TIER_UNSUPPORTED fixture matches the driver / UNSUPPORTED returns the tier'],

    // Found by Q/A wf_e03ec2d0-c07 as a SURVIVOR at ALL GREEN 92/0, before
    // that run dropped without returning a verdict. Its only visible effect is
    // which branch label enforceGate emits, which the fixture-level assertions
    // could not see -- they drive enforceGate directly rather than through the
    // driver that builds tierInfo.
    ['driver tierInfo.absent drifts to false',
      'absent: tierAbsent,', 'absent: false,',
      async (drive) => {
        const r = (await driveRecording(drive, { step_id: 'DM' })).result
        return !!(r && (r.checks || []).some(c => c.includes('tier_absent_defaulted_ok')))
      },
      'DRIVER-built tierInfo yields the ABSENT branch label end-to-end'],

    ['supported tier silently downgraded to the default',
      'const tier = tierSupported ? tierRequested : \'moderate\'', 'const tier = \'moderate\'',
      async (drive) => {
        const r = (await driveRecording(drive, { step_id: 'DM', tier: 'complex' })).result
        return r && r.tier_applied === 'complex'
      },
      'a SUPPORTED non-default tier is APPLIED, not silently downgraded'],
  ]

  for (const [name, from, to, probe, guards] of driverMutants) {
    if (!raw.includes(from)) { check(`driver-mutant "${name}" anchor present`, false, `anchor not found: ${from}`); continue }
    // A non-unique anchor means a first-match replace may mutate a branch the
    // probe never executes -- which produces a SURVIVING mutant that looks like
    // a weak check but is really a weak mutant. Measured on this very matrix.
    const occurrences = raw.split(from).length - 1
    check(`driver-mutant "${name}" anchor is UNIQUE (${occurrences})`, occurrences === 1,
      `anchor matches ${occurrences} sites; a first-match replace may hit the wrong branch`)
    const mutated = raw.replace(from, to)
    if (mutated === raw) { check(`driver-mutant "${name}" applied`, false, 'replace was a no-op'); continue }
    let held = null
    try {
      const d = await loadDriver(mutated)
      held = await probe(d)
    } catch (e) { held = 'threw: ' + (e && e.message ? e.message.split('\n')[0] : String(e)) }
    // KILLED = the guarded predicate no longer holds (or the build throws).
    check(`driver-mutant "${name}" is KILLED [guard: ${guards}]`, held !== true,
      'the mutated driver still satisfied the predicate -- that check is not load-bearing')
  }

  // The FIXTURE-fidelity checks are guarded by mutating the FIXTURE, not the
  // source: revert TIER_ABSENT to its cycle-4 shape and confirm the comparison
  // it performs would reject it.
  const cycle4AbsentFixture = { requested: null, applied: 'moderate', supported: true, absent: true, unsupported: false }
  const driveAbsent = await driveRecording(await loadDriver(), { step_id: 'FIXCHK' })
  check('fixture-mutant "TIER_ABSENT reverted to cycle-4 supported:true" is KILLED',
    cycle4AbsentFixture.supported !== driveAbsent.result.tier_supported,
    'the fidelity comparison would have accepted the fixture that caused the cycle-5 defect')
}

console.log('\n[8] structural -- no stripped schema keywords, no forbidden runtime imports, riders intact')
{
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const schemaRegion = src.slice(src.indexOf('const ENVELOPE_SCHEMA'), src.indexOf('// Stage-2 verifier schema'))
  check('no `minimum:` in the schema (stripped on the wire -- would be false assurance)', !/\bminimum\s*:/.test(schemaRegion))
  check('no `minItems:` in the schema (capped at 1 on the wire)', !/\bminItems\s*:/.test(schemaRegion))
  check('gate_passed is NOT const:true (honest failure must be representable)', !/gate_passed[\s\S]{0,160}const\s*:\s*true/.test(schemaRegion))
  check('additionalProperties:false on the envelope', /additionalProperties:\s*false/.test(schemaRegion))
  // The runtime forbids Node APIs, and its error text -- "import call expects
  // one or two arguments" -- says only the DYNAMIC import() expression parses.
  // So the assertion is ZERO static imports of any form, not "no node: imports".
  //
  // The first version of this guard was /^\s*import\s+\w+\s+from\s+'node:/ and
  // had 1-of-6 recall against the known-member set: it caught only the
  // single-quoted DEFAULT form -- the one instance I had actually measured --
  // and missed the double-quoted form of that IDENTICAL construct, plus named,
  // namespace, side-effect-only and bare-specifier imports. Found by the Q/A's
  // recall test, reproduced here before fixing:
  //     default single-quote  CAUGHT | default DOUBLE-quote MISSED
  //     named  MISSED | namespace MISSED | side-effect MISSED | bare MISSED
  // A guard built from the single instance you happened to hit is not a guard
  // against the class.
  const staticImports = (src.match(/^\s*import\b/gm) || [])
  check(`NO static imports of ANY form (found ${staticImports.length}) -- the Workflow runtime parses only dynamic import()`,
    staticImports.length === 0, staticImports.join(' | '))
  check("agentType is 'researcher' (needs Write for write-first)", /agentType:\s*'researcher'/.test(src))
  check("model is 'opus' (rider-trap R4)", /model:\s*'opus'/.test(src))
  check('no Monitor/watchdog (rider-trap R11)', !/Monitor\(/.test(src))
  check('exactly ONE export (`export const meta`) -- a trailing export list is unlaunchable',
    (src.match(/^\s*export\b/gm) || []).length === 1 && /^export const meta/m.test(src))
  // phase-86.28: the refusal must come BEFORE the spawn or it saves nothing.
  // Structural + ordering, because the module-level driver cannot be executed
  // outside the Workflow runtime; the LIVE spawn recorded in the step's
  // live_check is the behavioural half of this pair.
  //
  // CYCLE 2 -- THIS GUARD WAS DEFEATED AND IS NOW HARDENED. The cycle-1 version
  // used a bare `src.indexOf('if (tierUnsupported) {')` over RAW source. The
  // Q/A (wf_10c6cbd2-cad) executed a mutant that inserted the comment
  //     // harmless note: if (tierUnsupported) { we would refuse here }
  // before the spawn and moved the REAL refusal block AFTER it -- and the
  // checker still printed "ALL GREEN: 61 passed, 0 failed". First-match
  // indexOf over source-including-comments means a comment token satisfies an
  // ordering guard while production does the opposite. qa.md §4c shapes #2
  // (source scan defeated by moving the scanned text) and #8 (comment token).
  //
  // Two changes: (a) comment lines are stripped before indexing, so a comment
  // cannot stand in for code; (b) the refusal is matched as a BLOCK that
  // reaches its `return {`, not as a bare opening token. And the predicate is
  // extracted so it can be run against a deliberately-broken source below --
  // a guard nobody has watched fail is not a guard.
  // CYCLE 3: strips BLOCK comments as well as line comments. Cycle 2 stripped
  // only `//`, and the Q/A defeated the result with a /* */ decoy while the
  // production refusal sat after the spawn -- the scan printed `ok` during a
  // real breach, which is worse than having no scan at all.
  //
  // This scan is now the CHEAP SECONDARY. Section [6d] is the authority: it
  // DRIVES the module with a recording `agent()` stub and counts spawns, so no
  // comment/string/template trick can reach it. Kept rather than deleted so
  // the source-level intent stays asserted, but it no longer stands alone.
  const stripComments = (s) => s
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n').filter(l => !/^\s*\/\//.test(l)).join('\n')
  const stripLineComments = stripComments
  // phase-86.37: the spawn locator matches BOTH `const envelope = await agent(`
  // and the try/catch form `envelope = await agent(` introduced when stage 1 was
  // made drop-survivable. The literal it used to search for stopped existing, so
  // `spawnAt` became -1 and TWO assertions failed -- not because the ordering
  // broke, but because the locator could no longer find its landmark. Widening
  // the locator does NOT weaken the assertion: it still finds the spawn and still
  // compares positions; it just stops depending on a keyword that is irrelevant
  // to the property being tested. The M5 vacuity probe below uses the same
  // locator, so both track the code together.
  const SPAWN_RE = /(?:const\s+)?envelope\s*=\s*await agent\(PROMPT/
  const spawnIndex = (s) => { const m = SPAWN_RE.exec(s); return m ? m.index : -1 }
  const refusalPrecedesSpawn = (rawSrc) => {
    const code = stripLineComments(rawSrc)
    const m = /if \(tierUnsupported\) \{[\s\S]*?return \{/.exec(code)
    const spawnAt = spawnIndex(code)
    return m !== null && spawnAt !== -1 && m.index < spawnAt
  }
  {
    // ── phase-86.37: THE BORN-INERT MARKER AND THE STAGE-1 DROP ──────────────
  // The marker is a HARD gate: a brief that dropped mid-run can be long,
  // well-formed and source-rich -- 86.29's was 25,359 bytes with 15 sources --
  // and is still not completed research. Every case below hands enforceGate an
  // envelope that CLEARS EVERY FLOOR, so the only thing that can fail is the
  // marker. If these ever pass, the marker has stopped gating.
  {
    const clears = goodEnvelope()
    const baseV = verifyBrief(clears.brief_path, clears.sources_read_in_full)

    const withStatus = (st) => Object.assign({}, baseV, { brief_status_in_brief: st })
    check('a brief declaring brief_status=INCOMPLETE FAILS the gate even when every floor is met',
      G(clears, withStatus('INCOMPLETE')).gate_passed === false)
    check('...and the violation NAMES the marker rather than blaming a floor',
      G(clears, withStatus('INCOMPLETE')).violations.some(v => /INCOMPLETE/.test(v)))
    check('a brief with NO brief_status marker FAILS the gate (cannot be shown complete)',
      G(clears, withStatus('ABSENT')).gate_passed === false)
    check('ABSENT is reported DISTINCTLY from INCOMPLETE, not folded into it',
      G(clears, withStatus('ABSENT')).violations.some(v => /NO brief_status marker/.test(v)))
    check('brief_status=COMPLETE with every floor met PASSES',
      G(clears, withStatus('COMPLETE')).gate_passed === true)

    // FAIL-CLOSED: a stage-2 object that OMITS the field must not skip the gate.
    // The first implementation tested only the three known values, so undefined
    // matched none and the marker check silently did nothing.
    const omitted = Object.assign({}, baseV)
    delete omitted.brief_status_in_brief
    check('a verification object OMITTING brief_status_in_brief FAILS CLOSED (does not skip the gate)',
      G(clears, omitted).gate_passed === false)
    check('...and an unrecognised marker value also fails closed',
      G(clears, Object.assign({}, baseV, { brief_status_in_brief: 'probably fine' })).gate_passed === false)
  }

  // The stage-1 drop must be SURVIVABLE and must NEVER pass.
  {
    const src = stripLineComments(fs.readFileSync(WORKFLOW, 'utf8'))
    // PROXIMITY, not a spanning regex. The first version of this check was
    // /try\s*\{[\s\S]*?envelope = await agent\(PROMPT[\s\S]*?\}\s*catch/, which
    // any EARLIER `try {` (classifyArgs has one) plus any LATER `catch` (stage 2
    // has one) satisfies -- so it stayed green against a valid unwrap of stage 1
    // and only the rail_dropped assertion caught that mutant. Measured, not
    // assumed. The wrapper is now pinned by DISTANCE: the nearest `try {` before
    // the spawn must be close to it, and a `catch` must follow close after.
    const spawnPos = spawnIndex(src)
    const tryBefore = src.lastIndexOf('try {', spawnPos)
    const catchAfter = src.indexOf('catch', spawnPos)
    check('stage 1 is wrapped so a rail drop cannot kill the workflow (proximity-pinned)',
      spawnPos !== -1 && tryBefore !== -1 && (spawnPos - tryBefore) < 200
      && catchAfter !== -1 && (catchAfter - spawnPos) < 600)
    check('the drop path records rail_dropped rather than swallowing the error',
      /catch[\s\S]{0,400}railDropped\s*=\s*\{[\s\S]{0,200}error/.test(src))
    check('rail_dropped is returned as its OWN field, not folded into gate_passed',
      /rail_dropped:\s*railDropped/.test(src))
    check('the drop path does NOT assign gate_passed anywhere in its catch block',
      !/catch\s*\([\s\S]{0,600}gate_passed\s*[:=]\s*true/.test(src))

    // The property that matters: a dropped run has NO envelope, and enforceGate
    // must fail closed on that even when a perfectly good brief is on disk.
    const clears = goodEnvelope()
    const goodV = verifyBrief(clears.brief_path, clears.sources_read_in_full)
    const droppedResult = G(null, Object.assign({}, goodV, { brief_status_in_brief: 'COMPLETE' }))
    check('a DROPPED stage 1 (null envelope) fails the gate even with a COMPLETE brief on disk',
      droppedResult.gate_passed === false)
    check('...and that is decided by the existing fail-closed logic, which still names a violation',
      Array.isArray(droppedResult.violations) && droppedResult.violations.length > 0)
  }

  check('driver REFUSES TO SPAWN on an unsupported tier',
      /if \(tierUnsupported\) \{[\s\S]*?return \{/.test(stripLineComments(src)))
    check('the refusal is placed BEFORE the researcher spawn (else it saves no tokens)',
      refusalPrecedesSpawn(src) === true)
    check('the refusal path returns gate_passed:false',
      /if \(tierUnsupported\) \{[\s\S]*?gate_passed: false/.test(stripLineComments(src)))

    // VACUITY TESTS -- the Q/A's M5, promoted from a one-time live observation
    // to a standing regression. M5 is TWO edits, and reproducing only the
    // first is not a test: a source carrying a decoy comment whose real
    // refusal is still correctly placed SHOULD pass, because it is correct.
    // The defeat needs the real block MOVED as well.
    const naivePrecedesSpawn = (rawSrc) => {
      const a = rawSrc.indexOf('if (tierUnsupported) {')
      const b = spawnIndex(rawSrc)
      return a !== -1 && b !== -1 && a < b
    }
    const refusalBlock = /if \(tierUnsupported\) \{[\s\S]*?\n\}\n/.exec(stripLineComments(src))
    if (!refusalBlock) {
      check('M5 decoy could be constructed', false, 'could not isolate the refusal block; vacuity tests did not run')
    } else {
      // Faithful M5: real refusal REMOVED from its position and appended after
      // the spawn, plus a comment token left behind where it used to be.
      const m5 = stripLineComments(src)
        .replace(refusalBlock[0], '// harmless note: if (tierUnsupported) { we would refuse here } return {\n')
        + '\n' + refusalBlock[0]

      // The old guard is DEFEATED by it -- this is what makes M5 a real test
      // rather than a decorative one. If this ever goes false, M5 has stopped
      // reproducing the defect and the guard below is no longer being probed.
      check('M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)',
        naivePrecedesSpawn(m5) === true)
      // ...and the hardened guard rejects it.
      check('ordering guard REJECTS the M5 comment-token + relocation defeat',
        refusalPrecedesSpawn(m5) === false)
    }

    // Plain relocation, no comment involved.
    if (refusalBlock) {
      const moved = stripLineComments(src).replace(refusalBlock[0], '') + '\n' + refusalBlock[0]
      check('ordering guard REJECTS a refusal relocated AFTER the spawn',
        refusalPrecedesSpawn(moved) === false)
    }
  }

  // phase-86.28 cycle 3: the DURABLE operationalization of "this rail does not
  // implement the deep tier". The comment in research-gate.js used to say
  // "`grep -c deep` on this file returns 0", which the same comment then made
  // false by containing the word. The claim that actually matters is narrower
  // and cannot defeat itself: 'deep' is not a VALID_TIER, and no line of CODE
  // mentions it.
  {
    const tiersLine = /const VALID_TIERS = \[([^\]]*)\]/.exec(src)
    check("VALID_TIERS does not contain 'deep' (the tier is documented but NOT implemented here)",
      tiersLine !== null && !tiersLine[1].includes('deep'),
      tiersLine ? `VALID_TIERS = [${tiersLine[1]}]` : 'VALID_TIERS not found')
    const codeOnly = src.replace(/\/\*[\s\S]*?\*\//g, '')
      .split('\n').filter(l => !/^\s*\/\//.test(l)).join('\n')
    const deepInCode = codeOnly.split('\n').filter(l => l.includes('deep'))
    check("every 'deep' occurrence in the file is a COMMENT, never code",
      deepInCode.length === 0,
      `found in code: ${JSON.stringify(deepInCode.slice(0, 3))}`)
  }

  check('enforceGate is pure -- no fs/process use in its body',
    !/fs\.|process\.cwd\(/.test(src.slice(src.indexOf('function enforceGate'), src.indexOf("phase('Research')"))))
}

console.log(`\n${failures.length ? 'FAILED' : 'ALL GREEN'}: ${pass} passed, ${failures.length} failed`)
if (failures.length) { failures.forEach(f => console.log('  - ' + f)); process.exit(1) }
