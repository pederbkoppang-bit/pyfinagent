#!/usr/bin/env node
/**
 * phase-36.27 re-runnable checker for `.claude/workflows/research-gate.js`.
 *
 * WHY THIS FILE EXISTS. The step's immutable verification command is:
 *     node --check .claude/workflows/research-gate.js && ls .claude/workflows/research-gate.js
 * which proves the file PARSES and EXISTS -- criterion 1, and nothing else.
 * Criteria 3 (every floor enforced), 4 (an EMPTY return is a failed gate) and 6
 * (MUTATION-TEST: weakening a floor must fail the check enforcing it) are not
 * reachable by it. The criteria are immutable, so the command is not amended;
 * this checker carries the rest and is run alongside it.
 *
 * It drives the REAL `enforceGate` exported by the workflow -- not a copy. A
 * checker that re-implemented the logic would pass while production drifted.
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

// The workflow's top level calls phase()/agent()/log(), which only exist inside
// the Workflow runtime. Import just the pure exports by stripping the trailing
// driver section -- everything from the `phase('Research')` call onward.
async function loadEnforceGate(sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(WORKFLOW, 'utf8')
  const idx = src.indexOf("phase('Research')")
  if (idx === -1) throw new Error("could not find the phase('Research') driver boundary in research-gate.js")
  const pureSrc = src.slice(0, idx)
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'rg-')), 'rg.mjs')
  fs.writeFileSync(tmp, pureSrc)
  const mod = await import(pathToFileURL(tmp).href)
  if (typeof mod.enforceGate !== 'function') throw new Error('research-gate.js does not export enforceGate')
  return mod
}

// A brief on disk that corroborates the envelope, so cross-check passes and the
// FLOORS are what each case is actually testing.
function makeBrief(urls) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rgbrief-'))
  const p = path.join(dir, 'research_brief_TEST.md')
  fs.writeFileSync(p, '# brief\n\n' + urls.map(u => `| ${u} | read in full |`).join('\n') + '\n')
  return p
}

const URLS = Array.from({ length: 8 }, (_, i) => `https://example.com/source-${i + 1}`)

function goodEnvelope(briefPath, over = {}) {
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

const { enforceGate } = await loadEnforceGate()
const briefPath = makeBrief(URLS)

console.log('\n[1] control -- a compliant envelope passes')
check('control passes', enforceGate(goodEnvelope(briefPath)).gate_passed === true)

console.log('\n[2] criterion 3 -- every floor REJECTS a short-of-floor return (not rounded up)')
{
  const r = enforceGate(goodEnvelope(briefPath, { external_sources_read_in_full: 4, sources_read_in_full: URLS.slice(0, 4) }))
  check('4 sources (<5) rejected', r.gate_passed === false && r.violations.some(v => v.includes('external_sources_read_in_full')))
}
{
  const r = enforceGate(goodEnvelope(briefPath, { urls_collected: 9 }))
  check('9 URLs (<10) rejected', r.gate_passed === false && r.violations.some(v => v.includes('urls_collected')))
}
{
  const r = enforceGate(goodEnvelope(briefPath, { recency_scan_performed: false }))
  check('recency_scan false rejected', r.gate_passed === false && r.violations.some(v => v.includes('recency_scan')))
}
{
  const cov = { audit_class: true, rounds: 3, dry_rounds: 1, K_required: 2, new_findings_last_round: 2, dry: false }
  const r = enforceGate(goodEnvelope(briefPath, { coverage: cov }))
  check('audit-class without coverage.dry rejected', r.gate_passed === false && r.violations.some(v => v.includes('coverage.dry')))
}
{
  const cov = { audit_class: true, rounds: 4, dry_rounds: 2, K_required: 2, new_findings_last_round: 0, dry: true }
  const r = enforceGate(goodEnvelope(briefPath, { coverage: cov }))
  check('audit-class WITH coverage.dry passes', r.gate_passed === true)
}
{
  // The floor is a FLOOR, never a ceiling: audit-class at 4 sources still fails.
  const cov = { audit_class: true, rounds: 4, dry_rounds: 2, K_required: 2, new_findings_last_round: 0, dry: true }
  const r = enforceGate(goodEnvelope(briefPath, { coverage: cov, external_sources_read_in_full: 4, sources_read_in_full: URLS.slice(0, 4) }))
  check('audit-class + dry but 4 sources STILL rejected (floor never lowered)', r.gate_passed === false)
}

console.log('\n[3] criterion 4 -- an EMPTY/errored return is a FAILED gate, never gate_passed')
for (const [label, v] of [['null', null], ['undefined', undefined], ['empty object', {}], ['a string', 'oops'], ['an array', []]]) {
  const r = enforceGate(v)
  check(`${label} => gate_passed false`, r.gate_passed === false)
}
check('null return is not silently treated as a self-report', enforceGate(null).agent_self_reported_gate_passed === null)

console.log('\n[4] the artifact cross-check -- a self-report the brief cannot corroborate FAILS')
{
  const r = enforceGate(goodEnvelope(briefPath, { brief_path: path.join(os.tmpdir(), 'definitely-absent-brief.md') }))
  check('missing brief on disk rejected', r.gate_passed === false && r.violations.some(v => v.includes('brief not found')))
}
{
  const emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rgempty-'))
  const emptyP = path.join(emptyDir, 'b.md'); fs.writeFileSync(emptyP, '   \n')
  const r = enforceGate(goodEnvelope(briefPath, { brief_path: emptyP }))
  check('empty brief rejected', r.gate_passed === false && r.violations.some(v => v.includes('EMPTY')))
}
{
  const r = enforceGate(goodEnvelope(briefPath, {
    sources_read_in_full: [...URLS.slice(0, 7), 'https://example.com/never-actually-read'],
  }))
  check('a source claimed but ABSENT from the brief rejected', r.gate_passed === false && r.violations.some(v => v.includes('ABSENT from the brief')))
}
{
  const r = enforceGate(goodEnvelope(briefPath, { external_sources_read_in_full: 8, sources_read_in_full: URLS.slice(0, 3) }))
  check('over-claim (8 claimed, 3 listed) rejected', r.gate_passed === false && r.violations.some(v => v.includes('over-claim')))
}

console.log('\n[5] the agent does not get to grade itself')
{
  const r = enforceGate(goodEnvelope(briefPath, { gate_passed: true, external_sources_read_in_full: 2, sources_read_in_full: URLS.slice(0, 2) }))
  check('agent gate_passed:true is OVERRIDDEN when the floors fail', r.gate_passed === false)
  check('the disagreement is reported', r.self_report_disagreed === true && r.agent_self_reported_gate_passed === true)
}
{
  const r = enforceGate(goodEnvelope(briefPath, { gate_passed: false }))
  check('an honestly-false self-report still passes when the floors hold (const:true would have made this unrepresentable)', r.gate_passed === true && r.self_report_disagreed === true)
}

console.log('\n[6] criterion 6 MUTATION-TEST -- weakening a floor in the SOURCE must break the check enforcing it')
{
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const mutants = [
    ['FLOOR_SOURCES 5 -> 1', 'const FLOOR_SOURCES = 5', 'const FLOOR_SOURCES = 1',
      (m) => m.enforceGate(goodEnvelope(briefPath, { external_sources_read_in_full: 4, sources_read_in_full: URLS.slice(0, 4) })).gate_passed],
    ['FLOOR_URLS 10 -> 1', 'const FLOOR_URLS = 10', 'const FLOOR_URLS = 1',
      (m) => m.enforceGate(goodEnvelope(briefPath, { urls_collected: 9 })).gate_passed],
    ['recency check removed', "if (env.recency_scan_performed !== true)", "if (false)",
      (m) => m.enforceGate(goodEnvelope(briefPath, { recency_scan_performed: false })).gate_passed],
    ['audit-class dry check removed', "if (cov.dry !== true)", "if (false)",
      (m) => m.enforceGate(goodEnvelope(briefPath, { coverage: { audit_class: true, rounds: 3, dry_rounds: 1, K_required: 2, new_findings_last_round: 2, dry: false } })).gate_passed],
  ]
  for (const [name, from, to, probe] of mutants) {
    if (!src.includes(from)) { check(`mutant "${name}" anchor present`, false, `anchor not found: ${from}`); continue }
    const mutated = src.replace(from, to)
    if (mutated === src) { check(`mutant "${name}" actually applied`, false, 'replace was a no-op'); continue }
    const mod = await loadEnforceGate(mutated)
    // The mutant must make a previously-REJECTED envelope pass. If it does not,
    // the check was never load-bearing.
    check(`mutant "${name}" is KILLED (weakening it lets a bad envelope through)`, probe(mod) === true,
      'the weakened floor did NOT change the outcome -- the check is not load-bearing')
  }
}

console.log('\n[7] structural -- the schema does not rely on stripped keywords, and gate_passed is not const')
{
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const schemaRegion = src.slice(src.indexOf('const ENVELOPE_SCHEMA'), src.indexOf('function enforceGate'))
  check('no `minimum:` in the schema (stripped on the wire -- would be false assurance)', !/\bminimum\s*:/.test(schemaRegion))
  check('no `minItems:` in the schema (capped at 1 on the wire)', !/\bminItems\s*:/.test(schemaRegion))
  check('gate_passed is NOT const:true (honest failure must be representable)', !/gate_passed[\s\S]{0,120}const\s*:\s*true/.test(schemaRegion))
  check('additionalProperties:false on the envelope', /additionalProperties:\s*false/.test(schemaRegion))
  check("agentType is 'researcher' (needs Write for write-first)", /agentType:\s*'researcher'/.test(src))
  check("model is 'opus' (rider-trap R4)", /model:\s*'opus'/.test(src))
  check('no Monitor/watchdog (rider-trap R11)', !/Monitor\(/.test(src))
}

console.log(`\n${failures.length ? 'FAILED' : 'ALL GREEN'}: ${pass} passed, ${failures.length} failed`)
if (failures.length) { failures.forEach(f => console.log('  - ' + f)); process.exit(1) }
