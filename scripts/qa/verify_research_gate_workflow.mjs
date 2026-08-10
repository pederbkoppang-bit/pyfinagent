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
const TIER_VALID = ['simple', 'moderate', 'complex']
const tierOpts = (over) => ({ tier: { requested: null, applied: 'moderate', supported: true, absent: false, unsupported: false, valid: TIER_VALID, ...over } })
const TIER_UNSUPPORTED = tierOpts({ requested: 'deep', supported: false, unsupported: true })
const TIER_ABSENT = tierOpts({ requested: null, absent: true })
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
  {
    const refusalAt = src.indexOf('if (tierUnsupported) {')
    const spawnAt = src.indexOf('const envelope = await agent(PROMPT')
    check('driver REFUSES TO SPAWN on an unsupported tier', refusalAt !== -1)
    check('the refusal is placed BEFORE the researcher spawn (else it saves no tokens)',
      refusalAt !== -1 && spawnAt !== -1 && refusalAt < spawnAt)
    check('the refusal path returns gate_passed:false', /if \(tierUnsupported\) \{[\s\S]*?gate_passed: false/.test(src))
  }

  check('enforceGate is pure -- no fs/process use in its body',
    !/fs\.|process\.cwd\(/.test(src.slice(src.indexOf('function enforceGate'), src.indexOf("phase('Research')"))))
}

console.log(`\n${failures.length ? 'FAILED' : 'ALL GREEN'}: ${pass} passed, ${failures.length} failed`)
if (failures.length) { failures.forEach(f => console.log('  - ' + f)); process.exit(1) }
