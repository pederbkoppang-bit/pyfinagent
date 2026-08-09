#!/usr/bin/env node
/**
 * phase-86.17 -- the args boundary of BOTH Layer-3 Workflow scripts.
 *
 * WHAT THIS CHECKS, AND WHY IT IS A SEPARATE FILE
 * -----------------------------------------------
 * `verify_research_gate_workflow.mjs` has ZERO args coverage: it imports the
 * pure slice with `args` unbound and exercises `enforceGate` only, so every
 * input shape below was invisible to it. That is how a gate that runs with no
 * step id, no topic and no scope -- writing `research_brief_UNSPECIFIED.md` --
 * survived. This file is that missing coverage, for BOTH scripts.
 *
 * Sections:
 *   [1] REPRODUCE. Drive the PRE-FIX blob (from git) across every shape and
 *       record what `stepId` resolved to. This is criterion 1's evidence and it
 *       is regenerated from git rather than transcribed, so it cannot go stale.
 *   [2] FIXED. Same shapes against the CURRENT file: present-but-unusable and
 *       present-but-incomplete must THROW with a message naming what arrived;
 *       absent must NOT throw but must be marked blind.
 *   [3] BLIND CANNOT PASS. `enforceGate` with a blind `inputHealth` must refuse
 *       even a perfect envelope -- the defence-in-depth half.
 *   [4] MUTATION. Revert each new guard individually; the specific new check
 *       must fail. A guard whose mutant survives does not count.
 *
 * ESM CACHES BY URL, so every shape gets a FRESH temp module path. A
 * module-scope free variable cannot otherwise be varied across cases -- that is
 * measured, not theoretical.
 *
 * Run:  node scripts/qa/verify_workflow_args_boundary.mjs
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const SCRIPTS = {
  'research-gate.js': path.join(REPO, '.claude', 'workflows', 'research-gate.js'),
  'qa-verdict.js': path.join(REPO, '.claude', 'workflows', 'qa-verdict.js'),
}
// The commit BEFORE the phase-86.17 args-boundary fix. Used only to regenerate
// the reproduce table; if it ever becomes unreachable the section degrades to a
// reported SKIP rather than a silent pass.
const PREFIX_REF = process.env.ARGS_BOUNDARY_PREFIX_REF || '178a6a59'

let pass = 0
const failures = []
function check(name, cond, detail) {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { failures.push(`${name}${detail ? ' -- ' + detail : ''}`); console.log(`  FAIL ${name}${detail ? ' -- ' + detail : ''}`) }
}

/** The eight shapes criterion 1 names, plus the two the research measured. */
const SHAPES = [
  { id: 'plain-object', literal: '{ step_id: "86.17", topic: "t" }', usable: true },
  { id: 'valid-json-string', literal: JSON.stringify(JSON.stringify({ step_id: '86.17', topic: 't' })), usable: true },
  { id: 'malformed-json-string', literal: JSON.stringify('{"step_id": "86.17"'), usable: false },
  { id: 'json-string-raw-newline', literal: JSON.stringify('{"step_id": "86.17", "topic": "a\nb"}'), usable: false },
  { id: 'array', literal: '["86.17"]', usable: false },
  { id: 'scalar-number', literal: '42', usable: false },
  { id: 'absent', literal: null, usable: 'dry_run' },
  { id: 'double-encoded-json', literal: JSON.stringify(JSON.stringify(JSON.stringify({ step_id: '86.17' }))), usable: false },
  { id: 'empty-string', literal: '""', usable: false },
  { id: 'object-without-step_id', literal: '{ topic: "t" }', usable: false },
]

/** Slice the script at its driver boundary and import it with `args` injected.
 *  Returns {threw, message, stepId, blind}. */
async function drive(source, shape, tag) {
  // Cut before the first runtime-only construct. `phase(` exists in both
  // scripts; the phase-86.17 early return in qa-verdict.js sits ABOVE its
  // phase() call and contains a top-level `return`, which is legal inside the
  // Workflow runtime's async wrapper but a SyntaxError in a plain module -- so
  // it must be cut too.
  const marks = ["if (inputHealth.blind)", "phase('Research')", "phase('QA')"]
    .map(m => source.indexOf(m)).filter(i => i !== -1)
  if (!marks.length) throw new Error('no driver boundary found')
  const body = source.slice(0, Math.min(...marks))

  const prelude = shape.literal === null ? '' : `const args = ${shape.literal};\n`
  const exports = ['stepId']
  if (/\binputHealth\b/.test(body)) exports.push('inputHealth')

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'argsb-'))
  const tmp = path.join(dir, `${tag}-${shape.id}.mjs`)
  fs.writeFileSync(tmp, prelude + body + `\nexport { ${exports.join(', ')} }\n`)
  try {
    const mod = await import(pathToFileURL(tmp).href)
    return { threw: false, stepId: mod.stepId, blind: mod.inputHealth ? mod.inputHealth.blind : null }
  } catch (e) {
    return { threw: true, message: String(e && e.message ? e.message : e) }
  }
}

function gitShow(ref, rel) {
  try {
    return execFileSync('git', ['show', `${ref}:${rel}`], { cwd: REPO, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch (_e) {
    return null
  }
}

console.log('phase-86.17 -- Workflow args-boundary verification\n')

// ── [1] REPRODUCE: what the PRE-FIX code did with each shape ────────────────
console.log(`[1] REPRODUCE -- pre-fix blob at ${PREFIX_REF} (criterion 1)\n`)
let reproduced = 0
for (const [name, abs] of Object.entries(SCRIPTS)) {
  const rel = path.relative(REPO, abs)
  const src = gitShow(PREFIX_REF, rel)
  if (!src) {
    console.log(`  note: ${name} -- pre-fix blob unavailable at ${PREFIX_REF}; section [1] SKIPPED for this script`)
    continue
  }
  console.log(`  ${name}:`)
  for (const shape of SHAPES) {
    const r = await drive(src, shape, 'pre')
    const outcome = r.threw ? `THREW (${r.message.slice(0, 40)}...)` : `stepId=${JSON.stringify(r.stepId)}`
    console.log(`    ${shape.id.padEnd(24)} -> ${outcome}`)
    if (!r.threw && r.stepId === 'UNSPECIFIED') reproduced++
  }
}
check('[1] pre-fix code silently resolved stepId=UNSPECIFIED on multiple shapes',
      reproduced >= 8, `blind resolutions counted: ${reproduced} (expected >= 8 across both scripts)`)

// ── [2] FIXED: the same shapes against the CURRENT file ─────────────────────
console.log('\n[2] FIXED -- current file (criteria 2, 3, 4)\n')
for (const [name, abs] of Object.entries(SCRIPTS)) {
  const src = fs.readFileSync(abs, 'utf8')
  console.log(`  ${name}:`)
  for (const shape of SHAPES) {
    const r = await drive(src, shape, 'cur')
    const outcome = r.threw ? `THREW: ${r.message.slice(0, 72)}` : `stepId=${JSON.stringify(r.stepId)} blind=${r.blind}`
    console.log(`    ${shape.id.padEnd(24)} -> ${outcome}`)

    if (shape.usable === true) {
      check(`[2] ${name} ${shape.id}: usable args resolve the real step id`,
            !r.threw && r.stepId === '86.17', outcome)
    } else if (shape.usable === 'dry_run') {
      check(`[2] ${name} ${shape.id}: absent args do NOT throw and are marked blind`,
            !r.threw && r.blind === true, outcome)
    } else {
      check(`[2] ${name} ${shape.id}: unusable/incomplete args THROW`, r.threw, outcome)
      if (r.threw) {
        check(`[2] ${name} ${shape.id}: the throw NAMES what arrived`,
              /typeof=/.test(r.message) && /isArray=/.test(r.message) && /preview=/.test(r.message),
              r.message.slice(0, 100))
      }
    }
    // No brief and no verdict may EVER be written under an UNSPECIFIED identity.
    check(`[2] ${name} ${shape.id}: never resolves to UNSPECIFIED on a present-args shape`,
          shape.usable === 'dry_run' || r.threw || r.stepId !== 'UNSPECIFIED', outcome)
  }
}

// ── [3] A BLIND RUN CANNOT PASS (criterion 4, second half) ──────────────────
console.log('\n[3] BLIND CANNOT PASS -- enforceGate defence in depth (criterion 4)\n')
{
  const src = fs.readFileSync(SCRIPTS['research-gate.js'], 'utf8')
  const idx = src.indexOf("phase('Research')")
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'argsb-eg-'))
  const tmp = path.join(dir, 'eg.mjs')
  fs.writeFileSync(tmp, src.slice(0, idx) + '\nexport { enforceGate }\n')
  const { enforceGate } = await import(pathToFileURL(tmp).href)

  // A PERFECT envelope -- the gate must still refuse it when the run was blind.
  const env = {
    gate_passed: true, external_sources_read_in_full: 9, urls_collected: 40,
    recency_scan_performed: true, sources_read_in_full: Array.from({ length: 9 }, (_, i) => `https://x/${i}`),
    brief_path: 'handoff/current/research_brief_86.17.md', coverage: { audit_class: false },
  }
  const verification = { brief_exists: true, brief_non_empty: true, char_count: 9000, urls_missing: [] }

  const ok = enforceGate(env, verification, { inputHealth: { status: 'ok', blind: false } })
  check('[3] a healthy run with a perfect envelope PASSES', ok.gate_passed === true, JSON.stringify(ok.violations))

  const blind = enforceGate(env, verification, { inputHealth: { status: 'dry_run', blind: true } })
  check('[3] the SAME perfect envelope CANNOT pass when the run was blind', blind.gate_passed === false)
  check('[3] the blind refusal is NAMED in violations',
        blind.violations.some(v => /dry_run_no_step_id/.test(v)), JSON.stringify(blind.violations))

  const legacy = enforceGate(env, verification)
  check('[3] no regression: enforceGate without inputHealth behaves as before', legacy.gate_passed === true)
}

// ── [4] MUTATION -- every new guard must be able to fail ────────────────────
//
// THE GUARDS ARE LAYERED, AND THAT CHANGED HOW THIS SECTION HAD TO BE WRITTEN.
// My first version asserted "with the guard reverted, the shape no longer
// throws". That was WRONG for three of five cells and I measured it rather than
// assuming it: reverting the JSON catch turns a malformed string into `{}`,
// which the DOWNSTREAM step_id guard then rejects; reverting the post-parse
// plain-object check lets a double-encoded string through to the same step_id
// guard. The door never opens -- which is the layering working.
//
// So the assertion is the one each guard is actually responsible for: its own
// DIAGNOSIS. Every cell compares the unmutated outcome against the mutated one
// for the same shape and requires them to DIFFER -- either it stops throwing,
// or it throws with a different, less accurate message. Shore's rule is the
// basis: an assertion message that does not put the error in context is a
// worse guard, even when the process still stops.
console.log('\n[4] MUTATION -- revert each new guard (criterion 6)\n')
const MUTANTS = [
  {
    id: 'restore-silent-catch',
    script: 'research-gate.js',
    from: "    try { v = JSON.parse(v) } catch (_e) { fail('are PRESENT but not parseable as JSON', raw) }",
    to: "    try { v = JSON.parse(v) } catch (_e) { v = {} }",
    shape: SHAPES.find(s => s.id === 'malformed-json-string'),
    owns: 'not parseable as JSON',
  },
  {
    id: 'drop-post-parse-plain-object-check',
    script: 'research-gate.js',
    from: "  if (typeof v !== 'object' || v === null || Array.isArray(v)) fail('did not reduce to a plain object', raw)",
    to: "  if (false) fail('did not reduce to a plain object', raw)",
    shape: SHAPES.find(s => s.id === 'double-encoded-json'),
    owns: 'did not reduce to a plain object',
  },
  {
    id: 'drop-step_id-requirement',
    script: 'research-gate.js',
    from: "  if (!v.step_id && !v.stepId) fail('are a plain object with NO step_id', raw)",
    to: "  if (false) fail('are a plain object with NO step_id', raw)",
    shape: SHAPES.find(s => s.id === 'object-without-step_id'),
    owns: 'with NO step_id',
  },
  {
    id: 'qa-restore-silent-catch',
    script: 'qa-verdict.js',
    from: "    try { v = JSON.parse(v) } catch (_e) { fail('are PRESENT but not parseable as JSON', raw) }",
    to: "    try { v = JSON.parse(v) } catch (_e) { v = {} }",
    shape: SHAPES.find(s => s.id === 'malformed-json-string'),
    owns: 'not parseable as JSON',
  },
  {
    id: 'qa-drop-step_id-requirement',
    script: 'qa-verdict.js',
    from: "  if (!v.step_id && !v.stepId) fail('are a plain object with NO step_id', raw)",
    to: "  if (false) fail('are a plain object with NO step_id', raw)",
    shape: SHAPES.find(s => s.id === 'object-without-step_id'),
    owns: 'with NO step_id',
  },
]

for (const m of MUTANTS) {
  const src = fs.readFileSync(SCRIPTS[m.script], 'utf8')
  const n = src.split(m.from).length - 1
  if (n !== 1) {
    check(`[4] ${m.id}: anchor is unique`, false, `found ${n} occurrences -- a no-match replace looks identical to success`)
    continue
  }
  // CONTROL first: the unmutated code must diagnose this shape with the phrase
  // the guard owns. Without this, a cell could "pass" because the phrase was
  // never produced in the first place.
  const control = await drive(src, m.shape, `ctl-${m.id}`)
  const controlOwns = control.threw && control.message.includes(m.owns)
  check(`[4] ${m.id}: CONTROL -- the guard owns its diagnosis`, controlOwns,
        control.threw ? control.message.slice(0, 90) : 'control did not throw at all')
  if (!controlOwns) continue

  const mutated = src.replace(m.from, m.to)
  const r = await drive(mutated, m.shape, `mut-${m.id}`)
  const stillOwns = r.threw && r.message.includes(m.owns)
  check(`[4] ${m.id}: KILLED -- reverting it changes the outcome for ${m.shape.id}`, !stillOwns,
        stillOwns ? 'mutant produced the SAME diagnosis -- this guard is not the one doing the work'
                  : (r.threw ? `now diagnosed as: ${r.message.slice(0, 70)}` : 'no longer throws at all'))
}

// Blind-cannot-pass mutant, driven through enforceGate rather than the boundary.
{
  const src = fs.readFileSync(SCRIPTS['research-gate.js'], 'utf8')
  const from = "  if (inputBlind) {"
  const n = src.split(from).length - 1
  if (n !== 1) {
    check('[4] drop-blind-violation: anchor is unique', false, `found ${n}`)
  } else {
    const idx = src.indexOf("phase('Research')")
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'argsb-mut-'))
    const tmp = path.join(dir, 'eg.mjs')
    fs.writeFileSync(tmp, src.slice(0, idx).replace(from, '  if (false) {') + '\nexport { enforceGate }\n')
    const { enforceGate } = await import(pathToFileURL(tmp).href)
    const env = {
      gate_passed: true, external_sources_read_in_full: 9, urls_collected: 40,
      recency_scan_performed: true, sources_read_in_full: Array.from({ length: 9 }, (_, i) => `https://x/${i}`),
      brief_path: 'p', coverage: { audit_class: false },
    }
    const verification = { brief_exists: true, brief_non_empty: true, char_count: 9000, urls_missing: [] }
    const blind = enforceGate(env, verification, { inputHealth: { status: 'dry_run', blind: true } })
    check('[4] drop-blind-violation: KILLED (a blind run would pass without it)', blind.gate_passed === true)
  }
}

console.log('')
if (failures.length) {
  console.log(`FAILED: ${pass} passed, ${failures.length} failed`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log(`ALL GREEN: ${pass} passed, 0 failed`)
