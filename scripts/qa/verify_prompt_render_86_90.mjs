#!/usr/bin/env node
/**
 * phase-86.90 -- the FIELD render boundary of both Layer-3 Workflow scripts.
 *
 * WHY THIS FILE EXISTS, AND WHY IT IS NOT A SECTION OF THE 86.17 CHECKER.
 * `verify_workflow_args_boundary.mjs` exercises the args ENVELOPE: object /
 * JSON string / array / scalar / absent. All ten of its shapes pass every FIELD
 * as a string. The defect lives one level in -- `args.evidence` as an OBJECT
 * inside a perfectly well-formed envelope -- so every existing checker, and the
 * step's own immutable `node --check`, were green while 22 production Q/A
 * spawns received the literal text `[object Object]` instead of their evidence.
 *
 * WHAT IS ASSERTED, AND WHY BEHAVIOURALLY. Sections [1]-[5] DRIVE the real
 * shipped scripts with the Workflow runtime primitives stubbed and capture the
 * prompt actually handed to `agent()`. A source scan for `renderArgField(` would
 * pass on a file that never calls it on the path that matters; two successive
 * Q/A passes have already defeated a source-scan assertion on this rail.
 *
 * THE TWO TRAPS THIS CHECKER EXISTS TO CATCH, both measured during 86.90:
 *   * A TEMPLATE LITERAL IS A NO-OP FIX -- `+` and `${}` both bottom out at
 *     Object.prototype.toString for a plain object. Section [5] mutates the
 *     shipped call BACK to `+` and requires section [2] to go RED; a template
 *     literal would fail identically, which is the point.
 *   * AN ARRAY IS THE QUIET VARIANT -- ['a','b'] coerces to `a,b` with NO
 *     `[object Object]` marker at all. A checker that only greps for that
 *     marker is blind to it, so [2] asserts the array case explicitly.
 *
 * Spawns nothing, costs nothing, touches no live state.
 *
 *     node scripts/qa/verify_prompt_render_86_90.mjs
 *     -> exit 0 all green, exit 1 with the failing case named
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const SCRIPTS = {
  'qa-verdict.js': path.join(REPO, '.claude', 'workflows', 'qa-verdict.js'),
  'research-gate.js': path.join(REPO, '.claude', 'workflows', 'research-gate.js'),
}
// The commit BEFORE the phase-86.90 render boundary landed. Used only to
// regenerate the reproduce table in section [1]; if it becomes unreachable the
// section degrades to a reported SKIP rather than a silent pass.
const PREFIX_REF = process.env.RENDER_PREFIX_REF || '75831f4c'

// The prose-shaped field each script renders, and the prompt header it lands
// under. `probe` is the field name; `header` is what to search the prompt for.
const FIELDS = {
  'qa-verdict.js': [
    { field: 'evidence', header: 'EVIDENCE / FILES TO READ:' },
    { field: 'extra', header: 'ADDITIONAL CONTEXT:' },
  ],
  'research-gate.js': [
    { field: 'topic', header: 'OBJECTIVE:' },
    { field: 'internal_scope', header: 'INTERNAL SCOPE:' },
  ],
}
const BASE_ARGS = {
  'qa-verdict.js': { step_id: '86.90', criteria: ['a criterion'] },
  'research-gate.js': { step_id: '86.90', tier: 'moderate' },
}

let pass = 0
const failures = []
function check(name, cond, detail) {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { failures.push(`${name}${detail ? ' -- ' + detail : ''}`); console.log(`  FAIL ${name}${detail ? ' -- ' + detail : ''}`) }
}

/** Run the WHOLE script with the runtime primitives stubbed, capturing every
 *  prompt handed to agent(). Same technique as verify_workflow_args_boundary.mjs
 *  section [5] -- a slice cannot see code below the driver boundary. */
async function runDriver(source, argsObject) {
  const body = source.replace(/^export const meta =/m, 'const meta =')
  const prelude =
    'const __spawns = [];\nconst __logs = [];\n'
    + 'const agent = async (prompt, opts) => { __spawns.push({ label: opts && opts.label, prompt }); '
    + '  return { verdict: "PASS", ok: true, gate_passed: true, external_sources_read_in_full: 9, '
    + '    urls_collected: 40, recency_scan_performed: true, '
    + '    sources_read_in_full: Array.from({length: 9}, (_, i) => "https://x/" + i), '
    + '    brief_path: "handoff/current/research_brief_86.90.md", coverage: { audit_class: false }, '
    + '    brief_exists: true, brief_non_empty: true, char_count: 9000, urls_checked: 9, '
    + '    urls_present: 9, urls_missing: [] }; };\n'
    + 'const phase = () => {};\n'
    + 'const log = (m) => { __logs.push(m); };\n'
    + `const args = ${JSON.stringify(argsObject)};\n`

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'render-'))
  const tmp = path.join(dir, 'drv.mjs')
  fs.writeFileSync(tmp,
    prelude
    + 'export default async function __run() {\n' + body + '\n}\n'
    + 'export { __spawns, __logs }\n')
  const mod = await import(pathToFileURL(tmp).href)
  let threw = null
  try { await mod.default() } catch (e) { threw = String((e && e.message) || e) }
  return { threw, spawns: [...mod.__spawns], logs: [...mod.__logs] }
}

/** Same, but the value under test cannot survive JSON.stringify of the args
 *  literal (circular / BigInt / function / Map), so it is BUILT in the prelude. */
async function runDriverRaw(source, argsLiteral) {
  const body = source.replace(/^export const meta =/m, 'const meta =')
  const prelude =
    'const __spawns = [];\nconst __logs = [];\n'
    + 'const agent = async (prompt, opts) => { __spawns.push({ label: opts && opts.label, prompt }); return { verdict: "PASS", ok: true }; };\n'
    + 'const phase = () => {};\n'
    + 'const log = (m) => { __logs.push(m); };\n'
    + `const args = ${argsLiteral};\n`
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'render-raw-'))
  const tmp = path.join(dir, 'drv.mjs')
  fs.writeFileSync(tmp,
    prelude + 'export default async function __run() {\n' + body + '\n}\n' + 'export { __spawns, __logs }\n')
  const mod = await import(pathToFileURL(tmp).href)
  let threw = null
  try { await mod.default() } catch (e) { threw = String((e && e.message) || e) }
  return { threw, spawns: [...mod.__spawns] }
}

function gitShow(ref, rel) {
  try {
    return execFileSync('git', ['show', `${ref}:${rel}`], { cwd: REPO, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
  } catch (_e) { return null }
}

const OBJ = { handoff: ['contract.md', 'experiment_results.md'], changed_files: ['x.py'], sha256: 'deadbeef' }
const ARR = ['contract.md', 'experiment_results.md']

console.log('phase-86.90 -- Layer-3 prompt FIELD render boundary\n')

// ── [0] CONTROL FIRST ───────────────────────────────────────────────────────
// Every assertion below is of the form "the prompt does NOT contain X". Those
// pass vacuously on a script that never spawns and therefore has no prompt. The
// control establishes that the driver produces a real prompt before anything
// claims one is clean.
console.log('[0] CONTROL -- the driver produces a REAL prompt (string-shaped, the current caller path)\n')
for (const name of Object.keys(SCRIPTS)) {
  const src = fs.readFileSync(SCRIPTS[name], 'utf8')
  const r = await runDriver(src, { ...BASE_ARGS[name], evidence: 'plain string', extra: 'plain', topic: 'plain', internal_scope: 'plain' })
  // research-gate.js spawns TWO agents (the researcher, then the brief
  // verification pass); qa-verdict.js spawns one. The control asserts "at least
  // one", and every later assertion reads spawns[0] -- the ROLE prompt, which is
  // the only one carrying caller-supplied fields.
  check(`[0] ${name}: a usable launch spawns at least one agent`, !r.threw && r.spawns.length >= 1,
        r.threw ? `threw: ${r.threw}` : `spawned ${r.spawns.length}`)
  check(`[0] ${name}: the prompt carries the step id`,
        r.spawns.length >= 1 && /86\.90/.test(String(r.spawns[0].prompt)), 'step id absent from the prompt')
  check(`[0] ${name}: a STRING field is passed through UNCHANGED (no JSON fence)`,
        r.spawns.length >= 1 && FIELDS[name].every(f =>
          new RegExp(f.header.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&') + ' plain').test(String(r.spawns[0].prompt))),
        'a plain string field was reshaped -- every existing caller is on this path')
}

// ── [1] REPRODUCE from git ──────────────────────────────────────────────────
console.log(`\n[1] REPRODUCE -- pre-fix blob at ${PREFIX_REF} must produce the literal (criterion 1)\n`)
let reproduced = 0
for (const name of Object.keys(SCRIPTS)) {
  const rel = path.relative(REPO, SCRIPTS[name])
  const src = gitShow(PREFIX_REF, rel)
  if (!src) { console.log(`  note: ${name} -- pre-fix blob unavailable at ${PREFIX_REF}; SKIPPED`); continue }
  const r = await runDriver(src, { ...BASE_ARGS[name], evidence: OBJ, extra: OBJ, topic: OBJ, internal_scope: OBJ })
  const prompt = r.spawns.length ? String(r.spawns[0].prompt) : ''
  check(`[1] ${name}: PRE-FIX produced "[object Object]"`, prompt.includes('[object Object]'),
        r.threw ? `threw instead: ${r.threw}` : 'pre-fix blob did not reproduce -- is PREFIX_REF right?')
  if (prompt.includes('[object Object]')) reproduced++
}
check('[1] the reproduction covered BOTH scripts', reproduced === 2, `reproduced on ${reproduced} of 2`)

// ── [2] FIXED ───────────────────────────────────────────────────────────────
console.log('\n[2] FIXED -- object AND array shapes render losslessly (criteria 2, 3, 5)\n')
for (const name of Object.keys(SCRIPTS)) {
  const src = fs.readFileSync(SCRIPTS[name], 'utf8')

  const rObj = await runDriver(src, { ...BASE_ARGS[name], evidence: OBJ, extra: OBJ, topic: OBJ, internal_scope: OBJ })
  const pObj = rObj.spawns.length ? String(rObj.spawns[0].prompt) : ''
  check(`[2] ${name}: OBJECT field no longer yields "[object Object]"`,
        !rObj.threw && pObj.length > 0 && !pObj.includes('[object Object]'),
        rObj.threw ? `threw: ${rObj.threw}` : 'the literal is still present')
  check(`[2] ${name}: every key of the object REACHES the prompt`,
        Object.keys(OBJ).every(k => pObj.includes('"' + k + '"'))
        && pObj.includes('deadbeef') && pObj.includes('experiment_results.md'),
        'a key or value of the structured field did not arrive')

  const rArr = await runDriver(src, { ...BASE_ARGS[name], evidence: ARR, extra: ARR, topic: ARR, internal_scope: ARR })
  const pArr = rArr.spawns.length ? String(rArr.spawns[0].prompt) : ''
  // The quiet variant: Array.prototype.toString would give `a.md,b.md` and
  // leave NO marker for a grep to find.
  check(`[2] ${name}: ARRAY field does not collapse to comma-joined text`,
        !rArr.threw && pArr.length > 0 && !pArr.includes('contract.md,experiment_results.md'),
        rArr.threw ? `threw: ${rArr.threw}` : 'the array was comma-joined -- the marker-free variant')
  check(`[2] ${name}: ARRAY field renders as JSON`,
        pArr.includes('```json') && pArr.includes('"contract.md"'), 'array did not render as JSON')
}

// ── [3] UNRENDERABLE MUST THROW ─────────────────────────────────────────────
// Criterion 5: fail loudly rather than substitute. A bare JSON.stringify would
// SILENTLY drop the first four of these and collapse the fifth -- five new
// silent losses traded for one, which is why each is asserted separately.
console.log('\n[3] UNRENDERABLE -- must THROW, naming the field (criterion 5)\n')
const UNRENDERABLE = [
  { id: 'circular', literal: '(() => { const o = { a: 1 }; o.self = o; return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: o, topic: o } })()' },
  { id: 'bigint', literal: '{ step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: { n: 1n }, topic: { n: 1n } }' },
  { id: 'function-valued key', literal: '{ step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: { f: () => 1 }, topic: { f: () => 1 } }' },
  { id: 'undefined-valued key', literal: '{ step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: { u: undefined }, topic: { u: undefined } }' },
  { id: 'Map', literal: '{ step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: new Map([["a", 1]]), topic: new Map([["a", 1]]) }' },
  { id: 'NaN', literal: '{ step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: { n: NaN }, topic: { n: NaN } }' },
  { id: 'object step_id (would reach a FILENAME)', literal: '{ step_id: { id: "86.90" }, tier: "moderate", criteria: ["c"] }' },
  // ── phase-86.90 cycle-2: the five the cycle-1 Q/A got PAST the walk ────────
  // Each of these rendered LOSSILY WITHOUT THROWING against the cycle-1 code,
  // because that walk used Object.keys -- own ENUMERABLE string keys only. A2 is
  // the sharpest: a non-enumerable toJSON replaced the ENTIRE object with one
  // string, i.e. a placeholder substitution reached THROUGH the guard written to
  // forbid substitutions. None was reachable from a real caller (classifyArgs
  // JSON.parses a string or passes the runtime object through, and a JSON-derived
  // object has no non-enumerables, no getters and no toJSON) -- they are here
  // because a guard whose stated rule is broader than its measured behaviour is
  // the exact failure this step exists to close.
  { id: 'A1 non-enumerable own data property', literal: '(() => { const o = { a: 1 }; Object.defineProperty(o, "hidden", { value: "LOST", enumerable: false }); return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: o, topic: o } })()' },
  { id: 'A2 non-enumerable toJSON (substitutes the WHOLE value)', literal: '(() => { const o = { real_evidence: "x", more: "y" }; Object.defineProperty(o, "toJSON", { value: () => "REPLACED", enumerable: false }); return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: o, topic: o } })()' },
  { id: 'A4 non-deterministic getter (walk reads once, stringify reads AGAIN)', literal: '(() => { let n = 0; const o = {}; Object.defineProperty(o, "v", { get: () => ++n, enumerable: true }); return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: o, topic: o } })()' },
  { id: 'A6 nested non-enumerable', literal: '(() => { const inner = { a: 1 }; Object.defineProperty(inner, "hidden", { value: "LOST", enumerable: false }); return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: { outer: inner }, topic: { outer: inner } } })()' },
  { id: 'A7 array with a non-index own property', literal: '(() => { const a = ["x"]; a.meta = "LOST"; return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: a, topic: a } })()' },
]

// CONTROLS for section [3]: shapes that must still render, so the widened walk
// is not simply refusing everything. A guard that throws on all input would pass
// every assertion above and be useless.
const STILL_RENDERS = [
  { id: 'plain nested object', literal: '{ step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: { a: [1, 2], b: { c: "d" } }, topic: { a: [1, 2] } }' },
  { id: 'null-prototype object', literal: '(() => { const o = Object.create(null); o.a = 1; return { step_id: "86.90", tier: "moderate", criteria: ["c"], evidence: o, topic: o } })()' },
]
for (const name of Object.keys(SCRIPTS)) {
  const src = fs.readFileSync(SCRIPTS[name], 'utf8')
  for (const u of UNRENDERABLE) {
    const r = await runDriverRaw(src, u.literal)
    const named = r.threw && /args\.(evidence|topic|step_id)/.test(r.threw)
    check(`[3] ${name}: ${u.id} THROWS and names the field`, Boolean(r.threw) && named,
          r.threw ? `threw but did not name the field: ${r.threw.slice(0, 120)}` : 'did NOT throw -- a silent substitution survived')
    check(`[3] ${name}: ${u.id} spawns NOTHING`, r.spawns.length === 0,
          `spawned ${r.spawns.length} despite an unrenderable field`)
  }
}

// ── [3b] THE CONTAINER, not just the elements (cycle-4, Q/A finding W1) ────
// `Array.isArray(a.criteria) ? a.criteria : []` sat UPSTREAM of the render
// boundary, so a present-but-wrong-shaped `criteria` was DISCARDED and the
// "(none passed in args)" placeholder substituted -- silently, with the agent
// spawned anyway, on the field the evaluator grades against. Cycle 1 routed the
// ELEMENTS and left the CONTAINER one line above untouched. This is the only
// hole found in this step that is trivially JSON-reachable.
console.log('\n[3b] THE CRITERIA CONTAINER -- a wrong SHAPE must not become "none passed"\n')
{
  const src = fs.readFileSync(SCRIPTS['qa-verdict.js'], 'utf8')
  const PLACEHOLDER = '(none passed in args'

  // CONTROL FIRST: an ARRAY still works, and ABSENT still yields the documented
  // placeholder -- so the assertions below cannot pass by refusing everything.
  const ok = await runDriver(src, { step_id: '86.90', criteria: ['c-one-ALPHA'] })
  check('[3b] CONTROL: an ARRAY of criteria reaches the prompt',
        !ok.threw && ok.spawns.length === 1 && String(ok.spawns[0].prompt).includes('c-one-ALPHA')
        && !String(ok.spawns[0].prompt).includes(PLACEHOLDER),
        ok.threw ? `threw: ${ok.threw}` : 'the criterion text did not arrive')
  const absent = await runDriver(src, { step_id: '86.90' })
  check('[3b] CONTROL: ABSENT criteria still yields the documented placeholder (this stays legal)',
        !absent.threw && absent.spawns.length === 1
        && String(absent.spawns[0].prompt).includes(PLACEHOLDER),
        absent.threw ? `threw: ${absent.threw}` : 'the placeholder is gone -- absent must stay legal')

  for (const c of [
    { id: 'string', literal: '{ step_id: "86.90", criteria: "1. c-one-ALPHA" }' },
    { id: 'object', literal: '{ step_id: "86.90", criteria: { a: "c-one-ALPHA" } }' },
    { id: 'numeric-key object', literal: '{ step_id: "86.90", criteria: { 0: "c-one-ALPHA" } }' },
    { id: 'number', literal: '{ step_id: "86.90", criteria: 7 }' },
  ]) {
    const r = await runDriverRaw(src, c.literal)
    check(`[3b] PRESENT-but-${c.id} criteria THROWS and names the field`,
          Boolean(r.threw) && /args\.criteria/.test(r.threw),
          r.threw ? `threw without naming it: ${r.threw.slice(0, 110)}` : 'did NOT throw -- the placeholder was substituted silently')
    check(`[3b] PRESENT-but-${c.id} criteria spawns NOTHING`, r.spawns.length === 0,
          `spawned ${r.spawns.length} with a discarded rubric`)
  }
}

// CONTROL for [3]: the widened walk must not have become a blanket refusal.
for (const name of Object.keys(SCRIPTS)) {
  const src = fs.readFileSync(SCRIPTS[name], 'utf8')
  for (const c of STILL_RENDERS) {
    const r = await runDriverRaw(src, c.literal)
    // `>= 1`, not `=== 1`: research-gate.js spawns TWO agents (role, then brief
    // verification). Section [0] already records this; asserting `=== 1` here
    // repeated the same fixture bug one section later, which is why it is
    // written out rather than silently corrected.
    check(`[3] CONTROL ${name}: ${c.id} still RENDERS (the walk is not a blanket refusal)`,
          !r.threw && r.spawns.length >= 1
          && String(r.spawns[0].prompt).includes('```json'),
          r.threw ? `threw: ${r.threw.slice(0, 140)}` : `spawned ${r.spawns.length}`)
  }
}

// ── [4] research-gate.js, stated either way (criterion 3) ───────────────────
// The step requires research-gate.js to be CHECKED BY EXECUTION. Section [1]
// showed it was vulnerable pre-fix and [2] that it is not now; this section
// states the historical half that source-reading cannot: no caller ever
// triggered it, which is why it never surfaced there.
console.log('\n[4] research-gate.js -- vulnerable by construction pre-fix, clean now, never triggered in practice\n')
check('[4] research-gate.js was reproduced pre-fix in section [1]', reproduced === 2,
      'the research-gate half of the reproduction did not run')

// ── [5] MUTATION ────────────────────────────────────────────────────────────
// A guard whose mutant survives does not count. Anchor uniqueness is checked
// FIRST: a no-match replace is indistinguishable from a successful one.
console.log('\n[5] MUTATION -- reverting each guard must turn the section above RED (criterion 6)\n')
const MUTANTS = [
  {
    id: 'restore-plus-concat',
    file: 'qa-verdict.js',
    // The leading two spaces and the trailing comma are LOAD-BEARING. Without
    // them the anchor also matches the phase-86.90 comment block, which quotes
    // the defective expression verbatim -- a probe matching its own
    // documentation, which is how the first run of this checker reported
    // "found 2 occurrences" instead of mutating anything.
    from: "  'EVIDENCE / FILES TO READ: ' + evidence,",
    to: "  'EVIDENCE / FILES TO READ: ' + a.evidence,",
    expect: async (src) => {
      const r = await runDriver(src, { ...BASE_ARGS['qa-verdict.js'], evidence: OBJ })
      return String(r.spawns.length ? r.spawns[0].prompt : '').includes('[object Object]')
    },
    why: 'section [2] must go RED when the raw field is concatenated again',
  },
  {
    id: 'restore-plus-concat (research-gate)',
    file: 'research-gate.js',
    from: "'OBJECTIVE: ' + topic",
    to: "'OBJECTIVE: ' + a.topic",
    expect: async (src) => {
      const r = await runDriver(src, { ...BASE_ARGS['research-gate.js'], topic: OBJ })
      return String(r.spawns.length ? r.spawns[0].prompt : '').includes('[object Object]')
    },
    why: 'the research-gate copy must be doing the work too',
  },
  {
    id: 'narrow-the-walk-back-to-Object.keys',
    file: 'qa-verdict.js',
    // Reverting the cycle-2 widening must make A2 -- the substitution case --
    // render lossily again without throwing.
    from: '  const descs = Object.getOwnPropertyDescriptors(value)',
    to: '  const descs = Object.fromEntries(Object.keys(value).map(k => [k, { enumerable: true, value: value[k] }]))',
    expect: async (src) => {
      const r = await runDriverRaw(src, '(() => { const o = { real_evidence: "x" }; Object.defineProperty(o, "toJSON", { value: () => "REPLACED", enumerable: false }); return { step_id: "86.90", criteria: ["c"], evidence: o } })()')
      return !r.threw && String(r.spawns.length ? r.spawns[0].prompt : '').includes('REPLACED')
    },
    why: 'section [3] A2 must go RED when the walk stops seeing non-enumerable properties',
  },
  {
    id: 'container-guard-reverted-to-silent-discard',
    file: 'qa-verdict.js',
    from: "const criteria = requireArgArray('criteria', a.criteria)",
    to: "const criteria = Array.isArray(a.criteria) ? a.criteria : []",
    expect: async (src) => {
      const r = await runDriverRaw(src, '{ step_id: "86.90", criteria: "1. c-one-ALPHA" }')
      return !r.threw && r.spawns.length === 1
        && String(r.spawns[0].prompt).includes('(none passed in args')
    },
    why: 'section [3b] must go RED when a wrong-shaped rubric is silently discarded again',
  },
  {
    id: 'placeholder-instead-of-throw',
    file: 'qa-verdict.js',
    // CYCLE-3 REPAIR. The cycle-2 form ended `void ('` -- an unterminated string,
    // so the mutant was a SyntaxError -- and injected the return AFTER the throw
    // it was meant to replace, i.e. dead code even had it parsed. The harness's
    // catch scored that crash as KILLED, so the cell measured nothing. This form
    // is valid and REACHABLE: the placeholder is returned instead of throwing.
    from: "  if (violation) {\n    throw new Error(RENDER_SCRIPT + ': ' + where + ' cannot be rendered WITHOUT SILENT LOSS -- '",
    to: "  if (violation) {\n    return '(unrenderable)'\n    // eslint-disable-next-line no-unreachable\n    throw new Error(RENDER_SCRIPT + ': ' + where + ' cannot be rendered WITHOUT SILENT LOSS -- '",
    expect: async (src) => {
      const r = await runDriverRaw(src, '{ step_id: "86.90", criteria: ["c"], evidence: new Map([["a", 1]]) }')
      return !r.threw   // KILLED iff swapping the throw for a placeholder stops it throwing
    },
    why: 'section [3] must go RED when the throw becomes a silent placeholder',
  },
  {
    id: 'identity-arg-accepts-objects',
    file: 'qa-verdict.js',
    from: "  if (typeof value === 'string') return value\n  if (typeof value === 'number' && Number.isFinite(value)) return String(value)\n  throw new Error(RENDER_SCRIPT + ': ' + where + ' must be a string",
    to: "  if (typeof value === 'string') return value\n  return String(value)\n  // eslint-disable-next-line\n  throw new Error(RENDER_SCRIPT + ': ' + where + ' must be a string",
    expect: async (src) => {
      const r = await runDriverRaw(src, '{ step_id: { id: "86.90" }, criteria: ["c"] }')
      return !r.threw
    },
    why: 'an object step id must not be allowed to reach a filename',
  },
]
// CYCLE-3: three outcomes, not two. `catch -> KILLED` is how cell M3 scored a
// SyntaxError as a kill for a whole cycle. A mutant that does not BUILD has not
// been tested by anything, so it is UNSCORABLE -- which FAILS the check rather
// than passing it. And each cell is CONTROLLED first: its own expect() must
// return false on the UNMUTATED source, or the cell proves nothing about the
// mutation (arXiv 2408.01760: equivalent mutants are 4-39% and undecidable).
for (const m of MUTANTS) {
  const src = fs.readFileSync(SCRIPTS[m.file], 'utf8')
  const n = src.split(m.from).length - 1
  if (n !== 1) { check(`[5] ${m.id}: anchor is unique`, false, `found ${n} occurrences -- a no-match replace looks identical to success`); continue }

  let control
  try { control = (await m.expect(src)) ? 'DETECTED-ON-CONTROL' : 'clean' }
  catch (e) { control = 'CONTROL-THREW: ' + String((e && e.message) || e).slice(0, 90) }
  check(`[5] ${m.id}: CONTROL is clean (expect() false on the UNMUTATED source)`,
        control === 'clean', control)

  const mutated = src.replace(m.from, m.to)
  let outcome
  try { outcome = (await m.expect(mutated)) ? 'DETECTED' : 'SURVIVED -- this guard is not the one doing the work' }
  catch (e) { outcome = 'UNSCORABLE: the mutant did not build (' + String((e && e.message) || e).slice(0, 90) + ')' }
  check(`[5] ${m.id}: KILLED (${m.why})`, outcome === 'DETECTED', outcome)
}

// ── [6] the two copies have not drifted ─────────────────────────────────────
console.log('\n[6] DUPLICATE INTEGRITY -- the Workflow runtime forbids imports, so the block is duplicated\n')
{
  const START = '// ── phase-86.90: THE FIELD RENDER BOUNDARY'
  const END = '// ── end of the byte-identical phase-86.90 block'
  const blocks = {}
  let ok = true
  for (const name of Object.keys(SCRIPTS)) {
    const src = fs.readFileSync(SCRIPTS[name], 'utf8')
    const i = src.indexOf(START), j = src.indexOf(END)
    if (i === -1 || j === -1 || j < i) { check(`[6] ${name}: the block is present and delimited`, false, `start=${i} end=${j}`); ok = false; continue }
    check(`[6] ${name}: the block is present and delimited`, true)
    blocks[name] = src.slice(i, j)
  }
  if (ok) {
    check('[6] the two copies are BYTE-IDENTICAL',
          blocks['qa-verdict.js'] === blocks['research-gate.js'],
          `lengths ${blocks['qa-verdict.js'].length} vs ${blocks['research-gate.js'].length}`)
  }
}

// ── [7] BYTE-VERBATIM CRITERIA ROUND-TRIP (phase-86.96) ─────────────────────
// The immutable criteria must survive the caller->script->prompt path
// byte-for-byte, on BOTH production args shapes, and a payload that cannot be
// parsed must die LOUD at the boundary before any tokens are spent -- never be
// repaired, defaulted, or have its text altered in transit (the 86.96 step
// name records exactly that workaround happening once). The adversarial corpus
// covers the shapes the 2026-08-16 kills and the parsed cycle-2 payloads
// disagreed about: quotes, newlines, non-ASCII, and the `"},` bracket idiom.
console.log('\n[7] BYTE-VERBATIM CRITERIA ROUND-TRIP -- phase-86.96 (criteria 4 + 6)\n')

const ADVERSARIAL_CRITERIA = [
  'has `backticks`, "double quotes" and a bash -c \'quoted\' command -- bash -c \'source .venv/bin/activate && echo "x"\'',
  'line one\nline two\nline three (raw newlines inside one criterion)',
  'non-ASCII: æ ø å ü ß -- 評価基準 -- naïve café',
  'JSON-shaped: {"k": [1, 2]} plus the "}, close idiom and a stray } bracket',
]
// The MINIMAL failing input class from the 86.96 bisection: a `[` closed by
// `}`. One character wrong (substitution repairs it; insertion does not).
const BRACKET_DEFECT_PAYLOAD =
  '{"step_id":"86.96","criteria":["c"],"extra":{"judge_these_specifically":["a","b"}}'

{
  const qaSrc = fs.readFileSync(SCRIPTS['qa-verdict.js'], 'utf8')
  const argsObj = { step_id: '86.96', criteria: ADVERSARIAL_CRITERIA, verification_command: 'true' }

  const rObj = await runDriver(qaSrc, argsObj)
  check('[7] object args: script runs and spawns exactly one evaluator',
        !rObj.threw && rObj.spawns.length === 1, rObj.threw || `spawns=${rObj.spawns.length}`)
  const pObj = rObj.spawns.length ? String(rObj.spawns[0].prompt) : ''
  ADVERSARIAL_CRITERIA.forEach((c, i) =>
    check(`[7] object args: criterion ${i + 1} arrives BYTE-VERBATIM`, pObj.includes(c)))
  const posO = ADVERSARIAL_CRITERIA.map((c) => pObj.indexOf(c))
  check('[7] object args: criteria arrive IN ORDER',
        posO.every((p, i) => p !== -1 && (i === 0 || p > posO[i - 1])), posO.join(','))

  // The string path is 80.6% of production launches (census 2026-08-17:
  // 390 parsing strings + 4 failing vs 95 objects over 585 records).
  const rStr = await runDriverRaw(qaSrc, JSON.stringify(JSON.stringify(argsObj)))
  check('[7] string args: script runs and spawns exactly one evaluator',
        !rStr.threw && rStr.spawns.length === 1, rStr.threw || `spawns=${rStr.spawns.length}`)
  const pStr = rStr.spawns.length ? String(rStr.spawns[0].prompt) : ''
  ADVERSARIAL_CRITERIA.forEach((c, i) =>
    check(`[7] string args: criterion ${i + 1} arrives BYTE-VERBATIM`, pStr.includes(c)))
  check('[7] object and string paths render IDENTICAL prompt bytes', pObj === pStr,
        `lengths ${pObj.length} vs ${pStr.length}`)

  check('[7] fixture sanity: the bracket-defect payload does NOT parse',
        (() => { try { JSON.parse(BRACKET_DEFECT_PAYLOAD); return false } catch (_e) { return true } })())
  const rKill = await runDriverRaw(qaSrc, JSON.stringify(BRACKET_DEFECT_PAYLOAD))
  check('[7] a malformed string payload dies LOUD at the boundary, spawning NOTHING',
        rKill.threw !== null && /not parseable as JSON/.test(rKill.threw) && rKill.spawns.length === 0,
        rKill.threw ? rKill.threw.slice(0, 100) : `spawns=${rKill.spawns.length}`)
}

// [7] mutation cells -- same three-outcome discipline as [5], control first.
const MUTANTS7 = [
  {
    id: '7-classifyArgs-repairs-instead-of-refusing',
    file: 'qa-verdict.js',
    from: "    try { v = JSON.parse(v) } catch (_e) { fail('are PRESENT but not parseable as JSON', raw) }",
    to: "    try { v = JSON.parse(v) } catch (_e) { v = { step_id: 'RECOVERED' } }",
    expect: async (src) => {
      const r = await runDriverRaw(src, JSON.stringify(BRACKET_DEFECT_PAYLOAD))
      return !r.threw  // DETECTED iff the mutant silently proceeds where the guard refused
    },
    why: '[7] must go RED if a malformed payload is silently repaired/defaulted instead of refused',
  },
  {
    id: '7-render-mangles-quotes-in-transit',
    file: 'qa-verdict.js',
    from: "  if (t === 'string') return value",
    to: "  if (t === 'string') return value.replace(/\"/g, \"'\")",
    expect: async (src) => {
      const r = await runDriver(src, { step_id: '86.96', criteria: ADVERSARIAL_CRITERIA })
      const p = r.spawns.length ? String(r.spawns[0].prompt) : ''
      return !p.includes(ADVERSARIAL_CRITERIA[0])  // DETECTED iff the quotes criterion no longer round-trips
    },
    why: 'the 86.96 step-name harm -- criteria text altered in transit -- must turn [7] RED',
  },
]
for (const m of MUTANTS7) {
  const src = fs.readFileSync(SCRIPTS[m.file], 'utf8')
  const n = src.split(m.from).length - 1
  if (n !== 1) { check(`[7] ${m.id}: anchor is unique`, false, `found ${n} occurrences`); continue }
  let control
  try { control = (await m.expect(src)) ? 'DETECTED-ON-CONTROL' : 'clean' }
  catch (e) { control = 'CONTROL-THREW: ' + String((e && e.message) || e).slice(0, 90) }
  check(`[7] ${m.id}: CONTROL is clean (expect() false on the UNMUTATED source)`,
        control === 'clean', control)
  const mutated = src.replace(m.from, m.to)
  let outcome
  try { outcome = (await m.expect(mutated)) ? 'DETECTED' : 'SURVIVED -- this guard is not the one doing the work' }
  catch (e) { outcome = 'UNSCORABLE: the mutant did not build (' + String((e && e.message) || e).slice(0, 90) + ')' }
  check(`[7] ${m.id}: KILLED (${m.why})`, outcome === 'DETECTED', outcome)
}

// ── [8] CALLER-SIDE ROUTING FUNCTIONS (phase-86.72 + 86.78 cycle-3) ─────────
// Drives the REAL enforceEscalation caller_text computation and the REAL
// enforceResearchRouting, brace-extracted from qa-verdict.js with loud
// anchors (never a copy typed here), then kills one mutant per guard.
console.log('\n[8] CALLER-SIDE ROUTING -- consequence recorder + re-research leg\n')
{
  const qsrc = fs.readFileSync(SCRIPTS['qa-verdict.js'], 'utf8')
  const sliceFn = (src, marker) => {
    const i = src.indexOf(marker)
    if (i === -1) throw new Error('slice anchor missing: ' + marker)
    // Paren-match the SIGNATURE first: the naive first-'{' grab hits the
    // param list's `opts = {}` (the exact trap the 86.85 cycle-10 evaluator
    // documented), and the loud content assertion below caught this slicer
    // doing exactly that on its first run.
    let p0 = src.indexOf('(', i), pd = 0, q = p0
    for (; q < src.length; q++) {
      if (src[q] === '(') pd++
      else if (src[q] === ')') { pd--; if (pd === 0) { q++; break } }
    }
    let j = src.indexOf('{', q), depth = 0, k = j
    for (; k < src.length; k++) {
      if (src[k] === '{') depth++
      else if (src[k] === '}') { depth--; if (depth === 0) { k++; break } }
    }
    return src.slice(i, k)
  }
  const buildMod = async (src) => {
    const reLine = src.split('\n').find(l => l.startsWith('const POSITIONAL_CLAIM_RE'))
    if (!reLine) throw new Error('POSITIONAL_CLAIM_RE line missing')
    const fnE = sliceFn(src, 'function enforceEscalation(')
    const fnR = sliceFn(src, 'function enforceResearchRouting(')
    if (!fnE.includes('judge_was_told_consequence')) throw new Error('enforceEscalation slice lost the recorder')
    if (!fnR.includes('next_action_on_research_needed')) throw new Error('routing slice lost the guidance')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'routing-'))
    const tmp = path.join(dir, 'r.mjs')
    fs.writeFileSync(tmp, reLine + '\n' + fnE + '\n' + fnR
      + '\nexport { enforceEscalation, enforceResearchRouting }\n')
    return await import(pathToFileURL(tmp).href)
  }
  const V = { ok: false, verdict: 'CONDITIONAL' }
  const SEQ = ['CONDITIONAL', 'CONDITIONAL']
  const m = await buildMod(qsrc)

  const posi = m.enforceEscalation(V, SEQ, { caller_text: 'this launch is attempt 5 of 5 at the ceiling' })
  check('[8] positional caller text -> judge_was_told_consequence === true',
        posi.judge_was_told_consequence === true, JSON.stringify(posi.judge_was_told_consequence))
  check('[8] the matched evidence substring is recorded',
        typeof posi.judge_was_told_consequence_evidence === 'string'
        && posi.judge_was_told_consequence_evidence.length > 0,
        String(posi.judge_was_told_consequence_evidence))
  const clean = m.enforceEscalation(V, SEQ, { caller_text: 'grade the step against the criteria; artifacts under handoff/current/' })
  check('[8] clean caller text -> false', clean.judge_was_told_consequence === false)
  const rule = m.enforceEscalation(V, SEQ, { caller_text: 'the 3rd-CONDITIONAL auto-FAIL rule exists and qa.md describes it' })
  check('[8] RULE mention without positional claim -> false (rule vs position distinction)',
        rule.judge_was_told_consequence === false, String(rule.judge_was_told_consequence_evidence))
  const absent = m.enforceEscalation(V, SEQ, {})
  check('[8] absent caller_text -> false, never a throw', absent.judge_was_told_consequence === false)

  const on = m.enforceResearchRouting({ verdict: 'FAIL', research_needed: true,
    research_brief_spec: { objective: 'o', output_format: 'f', tool_scope: 't', task_boundaries: 'b' } })
  check('[8] research_needed=true -> surfaced with the spec echoed',
        on.research_needed === true && on.research_brief_spec && on.research_brief_spec.objective === 'o')
  check('[8] guidance present and carries the Tmax=2 bound',
        typeof on.next_action_on_research_needed === 'string'
        && on.next_action_on_research_needed.includes('at most 2'))
  const off = m.enforceResearchRouting({ verdict: 'PASS' })
  check('[8] absent fields -> research_needed null, guidance null',
        off.research_needed === null && off.next_action_on_research_needed === null)
  const no = m.enforceResearchRouting({ verdict: 'CONDITIONAL', research_needed: false })
  check('[8] research_needed=false -> false, guidance null',
        no.research_needed === false && no.next_action_on_research_needed === null)

  // mutation cells -- control above is the green; each mutant must go red on
  // exactly the guard named. Temp copies only; the repo source is never written.
  const CELLS8 = [
    ['8-recorder-neutered', 'const POSITIONAL_CLAIM_RE = /attempt\\s+\\d+\\s+of\\s+\\d|counted attempt|rail binds|next launch will be denied|if this cycle does not close|PASS-or-FAIL|you are the \\d+(?:st|nd|rd|th)/i',
     'const POSITIONAL_CLAIM_RE = /$^/i',
     async (mod) => (await mod.enforceEscalation(V, SEQ, { caller_text: 'attempt 5 of 5' })).judge_was_told_consequence !== true],
    ['8-recorder-hardcoded-false', 'judge_was_told_consequence: posMatch !== null,',
     'judge_was_told_consequence: false,',
     async (mod) => (await mod.enforceEscalation(V, SEQ, { caller_text: 'attempt 5 of 5' })).judge_was_told_consequence !== true],
    ['8-routing-signal-dropped', "const wanted = verdict.research_needed === true",
     'const wanted = false',
     async (mod) => (await mod.enforceResearchRouting({ research_needed: true,
       research_brief_spec: { objective: 'o', output_format: 'f', tool_scope: 't', task_boundaries: 'b' } })).next_action_on_research_needed === null],
    ['8-tmax-bound-removed', "task_boundaries>, brief_path}}). Bounds: at most 2 '",
     "task_boundaries>, brief_path}}). Bounds: '",
     async (mod) => {
       const g = (await mod.enforceResearchRouting({ research_needed: true,
         research_brief_spec: { objective: 'o', output_format: 'f', tool_scope: 't', task_boundaries: 'b' } })).next_action_on_research_needed
       return !(typeof g === 'string' && g.includes('at most 2') && g.includes('stagnation'))
     }],
  ]
  for (const [id, from, to, red] of CELLS8) {
    const n = qsrc.split(from).length - 1
    if (n !== 1) { check(`[8] ${id}: anchor unique`, false, `found ${n}`); continue }
    let mod2, outcome
    try {
      mod2 = await buildMod(qsrc.replace(from, to))
      outcome = (await red(mod2)) ? 'DETECTED' : 'SURVIVED'
    } catch (e) { outcome = 'DETECTED (mutant broke the build: ' + String((e && e.message) || e).slice(0, 60) + ')' }
    check(`[8] ${id}: KILLED`, outcome.startsWith('DETECTED'), outcome)
  }
}

console.log('')
if (failures.length) {
  console.log(`FAILED: ${pass} passed, ${failures.length} failed`)
  for (const f of failures) console.log(`  - ${f}`)
  process.exit(1)
}
console.log(`ALL GREEN: ${pass} passed, 0 failed`)
