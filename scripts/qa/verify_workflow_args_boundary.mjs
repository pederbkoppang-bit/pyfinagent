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

// ── THE HEALTHY `verification` FIXTURE -- SYNTHETIC AND OWNED BY THIS FILE ──
//
// WHY THIS IS A FACTORY AND NOT AN OBJECT LITERAL (phase-86.92).
//
// It used to be a hand-written literal, cloned in two places. The failure mode
// is worth stating precisely, because it was not "the fixture was wrong" -- it
// was "the fixture rotted SILENTLY, in a direction that produced a MISLEADING
// message, and disabled a mutation cell on the way out".
//
// `enforceGate` grew three `verification.*` fields after this checker was
// written: `recency_section_present` + `distinct_urls_in_brief` (phase-86.6,
// cad38647, 2026-08-10 08:51) and `brief_status_in_brief` (phase-86.37,
// d3bb1dfb). The literal kept supplying the original four. An absent field is
// not an error to `enforceGate` -- it is coerced and FAILS CLOSED, which is
// exactly right for a gate and catastrophic for a fixture: the "healthy run"
// case began failing with violations that talk about A BRIEF ON DISK. For six
// days the red was therefore read as a stale-artifact problem. It never was.
// `enforceGate` is pure; `env.brief_path` is an inert string it interpolates
// into the message. MEASURED: pointing it at a nonexistent file yields the
// byte-identical violations, and one of the three failing cells used
// `brief_path: 'p'`, which was never a file at all.
//
// The fixture is therefore now derived from `BRIEF_VERIFICATION_SCHEMA.required`
// -- the script's OWN declaration of what stage 2 must return. When the next
// field is added there, ONE assertion fails, BY NAME, in ONE place, saying what
// to do. Golden-file practice is explicit that a fixture nobody regenerates
// degrades into an assertion about history rather than about behaviour; the
// repair is to derive it from the contract instead of re-recording it.
const HEALTHY_VERIFICATION_VALUES = {
  brief_exists: true,
  brief_non_empty: true,
  char_count: 9000,
  urls_checked: 9,
  urls_present: 9,
  urls_missing: [],
  recency_section_present: true,
  distinct_urls_in_brief: 40, // >= the envelope's urls_collected, or it over-claims
  brief_status_in_brief: 'COMPLETE',
}
/** A fresh healthy stage-2 return. Never share the object between cases. */
function healthyVerification(overrides) {
  return { ...HEALTHY_VERIFICATION_VALUES, urls_missing: [], ...(overrides || {}) }
}

/** Every `verification.<field>` that `enforceGate` actually READS.
 *
 *  Comments are stripped FIRST. Without that, a field named only in prose would
 *  be demanded of the fixture and this guard would invent a false red -- the
 *  same class of defect it exists to catch. The positive control in section [3]
 *  proves the stripping is live rather than decorative, by injecting a bogus
 *  field into a comment and requiring the un-stripped scan to see it and the
 *  stripped scan to reject it. A control that cannot fail is not a control.
 *
 *  Returns `{fields, anchored}`. `anchored` is false when either boundary
 *  marker has moved, so a silently-mis-sliced region is reported rather than
 *  scanned -- a slicing checker cannot cover what it slices away. */
function verificationFieldsRead(scriptSource) {
  const START = 'function enforceGate'
  const END = '// NO `export { ... }` LIST HERE'
  const start = scriptSource.indexOf(START)
  const end = scriptSource.indexOf(END)
  if (start === -1 || end === -1 || end <= start) return { fields: [], anchored: false }
  const body = scriptSource.slice(start, end)
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n').map(l => l.replace(/(^|[^:])\/\/.*$/, '$1')).join('\n')
  return {
    fields: [...new Set([...body.matchAll(/verification\.([A-Za-z0-9_]+)/g)].map(m => m[1]))].sort(),
    anchored: true,
  }
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
  // Both scripts' blind early-returns now sit BELOW their phase() call, so the
  // phase boundary alone is the cut. That is not a tidy-up: in cycle 1 this
  // list ALSO contained "if (inputHealth.blind)", so deleting that block simply
  // moved the cut and the mutant survived with every assertion green. A slicing
  // checker structurally CANNOT cover code it slices away -- section [5] exists
  // for exactly that reason.
  const marks = ["phase('Research')", "phase('QA')"]
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
  // The schema is exported ALONGSIDE enforceGate. research-gate.js itself is
  // NOT edited -- this file already appends its own export line to a stripped
  // COPY, so the fixture can be derived from the script's own declaration
  // without touching the graded gate (phase-86.92 rail R5 + criterion 3).
  fs.writeFileSync(tmp, src.slice(0, idx) + '\nexport { enforceGate, BRIEF_VERIFICATION_SCHEMA }\n')
  const { enforceGate, BRIEF_VERIFICATION_SCHEMA } = await import(pathToFileURL(tmp).href)

  // ── FIXTURE-ROT CANARY (phase-86.92, criterion 4) ────────────────────────
  // Two directions, because they catch different regressions:
  //   (a) DECLARED -- every field the schema requires has a healthy value. This
  //       is the one that would have fired on 2026-08-10 instead of three
  //       unrelated cells failing with messages about a brief.
  //   (b) CONSUMED -- every field enforceGate actually READS is supplied. A
  //       field read but never declared is itself a finding, so this is not
  //       redundant with (a).
  const declared = [...(BRIEF_VERIFICATION_SCHEMA.required || [])].sort()
  const noValue = declared.filter(f => !(f in HEALTHY_VERIFICATION_VALUES))
  check('[3] fixture canary (declared): every BRIEF_VERIFICATION_SCHEMA.required field has a healthy value',
        declared.length > 0 && noValue.length === 0,
        declared.length === 0 ? 'schema.required is EMPTY or unreachable -- the canary would pass vacuously'
                              : `add a value to HEALTHY_VERIFICATION_VALUES for: ${noValue.join(', ')}`)

  const readInfo = verificationFieldsRead(src)
  check('[3] fixture canary: the enforceGate region was located (boundary markers still present)',
        readInfo.anchored, 'a boundary marker moved -- the scanned region is not enforceGate')
  const unsupplied = readInfo.fields.filter(f => !(f in HEALTHY_VERIFICATION_VALUES))
  check('[3] fixture canary (consumed): every verification.* field enforceGate READS is supplied',
        readInfo.anchored && readInfo.fields.length > 0 && unsupplied.length === 0,
        `enforceGate reads ${readInfo.fields.length} field(s); missing from the fixture: ${unsupplied.join(', ') || '(none)'}`)
  const undeclared = readInfo.fields.filter(f => !declared.includes(f))
  check('[3] fixture canary: enforceGate reads no field the schema does not require',
        undeclared.length === 0, `read but not declared: ${undeclared.join(', ')}`)

  // POSITIVE CONTROL for the comment-stripper. Without this, the scan above
  // could be silently matching nothing and every fixture would look complete.
  {
    const poisoned = src.replace('function enforceGate',
      '// verification.__bogusProseOnlyField__ appears only in prose here\nfunction enforceGate')
    const naive = /verification\.__bogusProseOnlyField__/.test(poisoned)
    const stripped = verificationFieldsRead(poisoned).fields.includes('__bogusProseOnlyField__')
    check('[3] fixture canary CONTROL: a comment-only field IS present in the raw source', naive,
          'the control never injected anything -- it proves nothing')
    check('[3] fixture canary CONTROL: the stripper rejects a comment-only field', naive && !stripped,
          'comment stripping is inert -- a field named in prose would be demanded of the fixture')
  }

  // A PERFECT envelope -- the gate must still refuse it when the run was blind.
  const env = {
    gate_passed: true, external_sources_read_in_full: 9, urls_collected: 40,
    recency_scan_performed: true, sources_read_in_full: Array.from({ length: 9 }, (_, i) => `https://x/${i}`),
    brief_path: 'handoff/current/research_brief_86.17.md', coverage: { audit_class: false },
  }
  const verification = healthyVerification()

  const ok = enforceGate(env, verification, { inputHealth: { status: 'ok', blind: false } })
  check('[3] a healthy run with a perfect envelope PASSES', ok.gate_passed === true, JSON.stringify(ok.violations))

  // MUTATION of the canary itself: a guard that cannot fail does not count.
  // Drop one required field from a COPY of the fixture -- the healthy case must
  // stop passing AND the canary must name the missing field.
  {
    const crippled = healthyVerification()
    delete crippled.brief_status_in_brief
    const r = enforceGate(env, crippled, { inputHealth: { status: 'ok', blind: false } })
    check('[3] fixture canary KILLED: dropping one required field breaks the healthy case',
          r.gate_passed === false, 'the fixture is not load-bearing -- the healthy case passed without the field')
    // DIFFERENTIAL, not absolute. An absolute "names exactly one field" cell is
    // coupled to a baseline that happens to be complete: rot the fixture for an
    // unrelated reason and this cell fails too, reporting a second defect that
    // does not exist. Compare the missing-set BEFORE and AFTER instead, so the
    // cell measures what the mutation did and nothing else.
    const missingBefore = declared.filter(f => !(f in HEALTHY_VERIFICATION_VALUES))
    const missingAfter = declared.filter(f => !(f in crippled))
    const introduced = missingAfter.filter(f => !missingBefore.includes(f))
    check('[3] fixture canary KILLED: the canary names exactly the dropped field',
          introduced.length === 1 && introduced[0] === 'brief_status_in_brief',
          `newly-missing after the mutation: ${introduced.join(', ') || '(none -- the canary did not notice)'}`)
  }

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
    id: 'qa-drop-post-parse-plain-object-check',
    script: 'qa-verdict.js',
    from: "  if (typeof v !== 'object' || v === null || Array.isArray(v)) fail('did not reduce to a plain object', raw)",
    to: "  if (false) fail('did not reduce to a plain object', raw)",
    shape: SHAPES.find(s => s.id === 'double-encoded-json'),
    owns: 'did not reduce to a plain object',
  },
  {
    id: 'drop-empty-string-guard',
    script: 'research-gate.js',
    from: "    if (!v.trim()) fail('are PRESENT but an empty/blank string', raw)",
    to: "    if (false) fail('are PRESENT but an empty/blank string', raw)",
    shape: SHAPES.find(s => s.id === 'empty-string'),
    owns: 'an empty/blank string',
  },
  {
    id: 'qa-drop-empty-string-guard',
    script: 'qa-verdict.js',
    from: "    if (!v.trim()) fail('are PRESENT but an empty/blank string', raw)",
    to: "    if (false) fail('are PRESENT but an empty/blank string', raw)",
    shape: SHAPES.find(s => s.id === 'empty-string'),
    owns: 'an empty/blank string',
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
    // THIS CELL IS WHY FIXTURE ROT IS WORSE THAN A RED CHECKER (phase-86.92).
    // With the stale literal it reported FAIL whether the blind guard was
    // PRESENT or ABSENT -- residual violations from the missing fields held
    // gate_passed at false either way, so the cell could not DISCRIMINATE and
    // had silently stopped being a mutation test. MEASURED both ways:
    //     stale   fixture: guard present -> false, guard absent -> false  (dead)
    //     healthy fixture: guard present -> false, guard absent -> true   (kills)
    // A cell whose control answer and mutant answer agree tests nothing.
    const verification = healthyVerification()
    const blind = enforceGate(env, verification, { inputHealth: { status: 'dry_run', blind: true } })
    check('[4] drop-blind-violation: KILLED (a blind run would pass without it)', blind.gate_passed === true)
  }
}

// ── [5] FULL DRIVER -- the coverage a slicing checker structurally cannot give
//
// WHY THIS SECTION EXISTS. Cycle 1 shipped the blind early-return with ZERO
// re-runnable coverage: the Q/A deleted qa-verdict's entire
// `if (inputHealth.blind) { ... }` block and this checker stayed ALL GREEN,
// because drive() sliced that block away before importing. The behavioural
// differential was real -- 0 spawns became 1 max-effort spawn labelled
// `qa-verdict:UNSPECIFIED` returning a PASS-shaped object. A guard that cannot
// fail when its subject is broken does not count.
//
// So this section runs the WHOLE script, with the Workflow runtime primitives
// stubbed, and observes what a slice cannot: how many agents get spawned and
// what the driver RETURNS. `args` is left genuinely undeclared for class A, so
// `typeof args === 'undefined'` is exercised the way the real runtime does it.
console.log('\n[5] FULL DRIVER -- blind runs must spawn NOTHING (criteria 3, 4, 6)\n')

async function runDriver(source, { argsLiteral = null } = {}) {
  const spawns = []
  const logs = []
  // Strip the `export` keyword: an export is illegal inside the async wrapper.
  let body = source.replace(/^export const meta =/m, 'const meta =')
  const prelude =
    'const __spawns = [];\nconst __logs = [];\n'
    + 'let __call = 0;\n'
    + 'const agent = async (prompt, opts) => { __spawns.push({ label: opts && opts.label, prompt }); '
    + '  __call++; '
    + '  if (__call === 1) return { gate_passed: true, external_sources_read_in_full: 9, urls_collected: 40, '
    + '    recency_scan_performed: true, sources_read_in_full: Array.from({length: 9}, (_, i) => "https://x/" + i), '
    + '    brief_path: "handoff/current/research_brief_UNSPECIFIED.md", coverage: { audit_class: false }, '
    + '    verdict: "PASS", ok: true }; '
    + '  return { brief_exists: true, brief_non_empty: true, char_count: 9000, urls_checked: 9, urls_present: 9, urls_missing: [] }; };\n'
    + 'const phase = () => {};\n'
    + 'const log = (m) => { __logs.push(m); };\n'
    + (argsLiteral === null ? '' : `const args = ${argsLiteral};\n`)

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'argsb-drv-'))
  const tmp = path.join(dir, 'drv.mjs')
  fs.writeFileSync(tmp,
    prelude
    + 'export default async function __run() {\n' + body + '\n}\n'
    + 'export { __spawns, __logs }\n')
  const mod = await import(pathToFileURL(tmp).href)
  let result, threw = null
  try { result = await mod.default() } catch (e) { threw = String(e && e.message ? e.message : e) }
  spawns.push(...mod.__spawns)
  logs.push(...mod.__logs)
  return { result, threw, spawns, logs }
}

const DRIVER_CASES = [
  { name: 'qa-verdict.js', noVerdict: r => r.result && r.result.verdict === null && r.result.dry_run === true },
  { name: 'research-gate.js', noVerdict: r => r.result && r.result.gate_passed === false && r.result.dry_run === true },
]

for (const c of DRIVER_CASES) {
  const src = fs.readFileSync(SCRIPTS[c.name], 'utf8')
  const r = await runDriver(src)
  check(`[5] ${c.name}: a blind run spawns ZERO agents`, r.spawns.length === 0,
        `spawned ${r.spawns.length}: ${r.spawns.map(s => s.label).join(', ')}`)
  check(`[5] ${c.name}: a blind run cannot pass and is marked dry_run`, !r.threw && c.noVerdict(r),
        r.threw ? `threw: ${r.threw}` : JSON.stringify(r.result).slice(0, 140))
  check(`[5] ${c.name}: the blind run is LOGGED`, r.logs.some(l => /BLIND RUN/.test(l)),
        JSON.stringify(r.logs).slice(0, 120))
  // No artifact may be named under an UNSPECIFIED identity (criterion 3, 2nd sentence).
  check(`[5] ${c.name}: no prompt mentioning research_brief_UNSPECIFIED is ever sent`,
        !r.spawns.some(s => /UNSPECIFIED/.test(String(s.prompt))), 'a spawn carried an UNSPECIFIED identity')
}

// Control: a USABLE launch must still spawn, or the checks above would pass
// trivially on a script that never spawns at all.
{
  const src = fs.readFileSync(SCRIPTS['qa-verdict.js'], 'utf8')
  const r = await runDriver(src, { argsLiteral: '{ step_id: "86.17", criteria: ["c"] }' })
  check('[5] CONTROL: a usable launch DOES spawn', r.spawns.length === 1,
        `spawned ${r.spawns.length}`)
}

// MUTATION: delete each blind early-return and require the differential.
for (const c of DRIVER_CASES) {
  const src = fs.readFileSync(SCRIPTS[c.name], 'utf8')
  const marker = 'if (inputHealth.blind) {'
  const n = src.split(marker).length - 1
  if (n !== 1) { check(`[5] ${c.name} mutate-blind-return: anchor unique`, false, `found ${n}`); continue }
  const mutated = src.replace(marker, 'if (false) {')
  const r = await runDriver(mutated)
  check(`[5] ${c.name}: KILLED -- removing the blind early-return makes it spawn`,
        r.spawns.length > 0 || r.threw !== null,
        'mutant still spawned nothing -- this guard is not the one doing the work')
}

console.log('')
if (failures.length) {
  console.log(`FAILED: ${pass} passed, ${failures.length} failed`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log(`ALL GREEN: ${pass} passed, 0 failed`)
