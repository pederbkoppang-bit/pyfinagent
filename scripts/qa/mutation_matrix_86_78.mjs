#!/usr/bin/env node
/**
 * phase-86.78 criterion 6 -- mutation matrix for the RELOCATED escalation counter.
 *
 *   node scripts/qa/mutation_matrix_86_78.mjs
 *
 * Each mutant is a COPY of `.claude/workflows/qa-verdict.js` under a temp name, fed to
 * verify_escalation_86_78.mjs through the `PYFIN_QA_VERDICT_OVERRIDE` seam. The
 * tracked file is never opened for writing, and its sha256 is compared before and
 * after so "I did not touch it" is PROVEN, not asserted.
 *
 * A GREEN CONTROL RUNS FIRST -- a cell that "kills" an already-red checker proves
 * nothing.
 *
 * ANCHOR UNIQUENESS IS CHECKED -- `String.replace` on a non-matching anchor returns a
 * perfectly normal string, so a non-mutation would otherwise be scored as a kill.
 *
 * EACH CELL MUST DISCRIMINATE -- red alone is not a kill; one of the cell's NAMED
 * assertions must be among the failures. On this project three cells in one day went
 * red for the wrong reason (a syntax error, a renamed assertion), and each was refused.
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
const CHECKER = path.join(REPO, 'scripts/qa/verify_escalation_86_78.mjs')

const MUTATIONS = [
  {
    id: 'M1-THRESHOLD-OFF-BY-ONE',
    desc: 'arm at 3 prior CONDITIONALs instead of 2 -- the loop runs one cycle longer '
        + 'than the rule allows, i.e. the harness logs instead of correcting',
    anchor: "out.would_auto_fail = n >= 2 && verdict?.verdict === 'CONDITIONAL'",
    repl: "out.would_auto_fail = n >= 3 && verdict?.verdict === 'CONDITIONAL'  // MUTANT",
    expect: ['2 prior CONDITIONALs + a third -> ARMED (the loop terminates)'],
  },
  {
    id: 'M2-NO-RESET-ON-PASS',
    desc: 'count CONDITIONALs without resetting on PASS/FAIL -- an old run resurrects '
        + 'and force-fails a step that already recovered (the 36.17 regression)',
    anchor: '    if (v === \'NO_VERDICT\') continue\n'
          + '    if (v === \'CONDITIONAL\') n++\n'
          + '    else break',
    repl: '    if (v === \'NO_VERDICT\') continue\n'
        + '    if (v === \'CONDITIONAL\') n++  // MUTANT: no reset on PASS/FAIL',
    expect: ['a PASS RESETS the run', 'a FAIL RESETS the run',
             'step 36.17 (C,F,F,C,C) -> a 6th attempt is NOT force-failed'],
  },
  {
    id: 'M3-FAIL-OPEN-WITH-ZERO',
    desc: 'report 0 instead of null when the sequence is unusable -- a spurious zero '
        + 'reads as "no consecutive run" and SUPPRESSES termination',
    anchor: '    consecutive_conditionals: null,\n    would_auto_fail: null,',
    repl: '    consecutive_conditionals: 0,\n    would_auto_fail: false,  // MUTANT',
    expect: ['sequence not supplied -> null, not 0',
             'unparseable sequence -> null, not 0',
             'a spurious 0 would falsely report "no consecutive run" -- assert it is absent'],
  },
  {
    id: 'M4-ARM-ON-ANY-VERDICT',
    desc: 'drop the CONDITIONAL guard so the threshold arms on a PASS too -- that is a '
        + 'path from the caller side to downgrading a verdict, which must not exist',
    anchor: "out.would_auto_fail = n >= 2 && verdict?.verdict === 'CONDITIONAL'",
    repl: 'out.would_auto_fail = n >= 2  // MUTANT: arms regardless of the verdict',
    expect: ['...and never on a PASS either (arming a PASS would be a downgrade path)',
             'would_auto_fail can only ARM on a CONDITIONAL -- never on a FAIL'],
  },
  {
    id: 'M5-NO_VERDICT-RESETS-THE-RUN',
    desc: 'treat a dropped spawn as if it reset the run -- a rail drop would then erase '
        + 'the consecutive history and the loop could never terminate',
    anchor: "    if (v === 'NO_VERDICT') continue",
    repl: "    if (v === 'NO_VERDICT') break  // MUTANT: a drop resets the run",
    expect: ['NO_VERDICT is a dropped ATTEMPT: it neither extends nor resets the run'],
  },
  {
    id: 'M6-BUDGET-FAILS-OPEN',
    desc: 'report budget_exhausted=false rather than null when no attempt number is '
        + 'known -- claims headroom the caller has no evidence for',
    anchor: '    budget_exhausted: null,',
    repl: '    budget_exhausted: false,  // MUTANT: fail open',
    expect: ['no attempt number supplied -> budget_exhausted is null, not false'],
  },
  {
    id: 'M7-BURDEN-SAFEGUARD-REMOVED',
    desc: 'blank the named burden -- criterion 5 safeguard 1 becomes unrepresentable',
    anchor: "    burden_on: 'the party departing from the computed escalation',",
    repl: "    burden_on: '',  // MUTANT: safeguard 1 removed",
    expect: ['safeguard 1: the BURDEN is named, and sits on the departing party'],
  },
  {
    id: 'M8-OVERRIDE-DEFAULTS-TO-APPLIED',
    desc: 'default the override to applied -- an override would then be IMPLIED rather '
        + 'than recorded, which is exactly what safeguard 2 forbids',
    anchor: '    override: null,\n    override_reason: null,',
    repl: "    override: true,\n    override_reason: 'auto',  // MUTANT: implied override",
    expect: ['...and it defaults to null -- an override must be recorded, never implied'],
  },
  {
    id: 'M9-CONSEQUENCE-RESTORED-TO-THE-PROMPT',
    desc: 'put the consequence sentence back into the judge\'s prompt -- the whole '
        + 'exposure this step removes',
    anchor: "  'CONDITIONAL for fixable gaps, FAIL for a criterion miss.',",
    repl: "  'CONDITIONAL for fixable gaps, FAIL for a criterion miss. If this step-id "
        + "already has 2 consecutive prior CONDITIONALs, return FAIL instead of a third, "
        + "and at 5+ recommend operator escalation.',  // MUTANT",
    expect: ['rail prompt no longer states the 3rd-CONDITIONAL trigger',
             'rail prompt no longer states the F1b escalation consequence'],
  },
  {
    id: 'M10-INPUT-NOT-ECHOED',
    desc: 'stop echoing the supplied sequence -- the caller-supplied input stops being '
        + 'auditable, and the caller is the constrained party',
    anchor: '    sequence_supplied: Array.isArray(sequence) ? sequence.slice() : null,',
    repl: '    sequence_supplied: null,  // MUTANT: input not auditable',
    expect: ['the input is echoed back, so what the caller supplied is auditable'],
  },
]

const sha = (p) => createHash('sha256').update(fs.readFileSync(p)).digest('hex').slice(0, 16)

function runChecker (overridePath) {
  const env = { ...process.env }
  if (overridePath) env.PYFIN_QA_VERDICT_OVERRIDE = overridePath
  else delete env.PYFIN_QA_VERDICT_OVERRIDE
  try {
    const out = execFileSync('node', [CHECKER], { env, cwd: REPO, encoding: 'utf8' })
    return { code: 0, out }
  } catch (e) {
    return { code: e.status ?? 1, out: (e.stdout ?? '') + (e.stderr ?? '') }
  }
}

// Capture the WHOLE remainder of a [FAIL] line. Stripping a ` -- ` suffix here was a
// bug: several assertion LABELS themselves contain ` -- `, so the regex truncated the
// label and M8 was scored RED-WRONG-REASON when it had in fact been killed correctly.
// Matching is by PREFIX below, which tolerates the appended detail without guessing
// where the label ends.
const FAIL_RE = /^\s*\[FAIL\] (.+)$/gm

const before = sha(SUBJECT)
console.log(`subject : .claude/workflows/qa-verdict.js  sha256[:16]=${before}`)
console.log(`checker : scripts/qa/verify_escalation_86_78.mjs`)

const ctl = runChecker(null)
console.log(`\n[CONTROL] unmutated checker -> exit ${ctl.code}`)
if (ctl.code !== 0) {
  console.log('ABORT -- the control is RED, so no kill below would mean anything.')
  console.log(ctl.out.slice(-2000))
  process.exit(1)
}
console.log(`  ok -- GREEN control established (${(ctl.out.match(/checks run : (\d+)/) || [])[1]} checks)`)

const src = fs.readFileSync(SUBJECT, 'utf8')
const rows = []
const survived = []
for (const cell of MUTATIONS) {
  const n = src.split(cell.anchor).length - 1
  if (n !== 1) {
    rows.push([cell.id, 'ANCHOR-STALE', `matched ${n} time(s), expected 1`])
    survived.push(cell.id)
    continue
  }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `qa78m_${cell.id.toLowerCase()}_`))
  const mut = path.join(dir, 'qa-verdict.js')
  fs.writeFileSync(mut, src.replace(cell.anchor, cell.repl))
  const r = runChecker(mut)
  if (r.code === 0) {
    rows.push([cell.id, 'SURVIVED', cell.desc])
    survived.push(cell.id)
  } else {
    const failures = new Set([...r.out.matchAll(FAIL_RE)].map(m => m[1]))
    const aimed = cell.expect.filter(a => [...failures].some(f => f.startsWith(a)))
    if (aimed.length === 0) {
      rows.push([cell.id, 'RED-WRONG-REASON',
        `expected one of ${JSON.stringify(cell.expect)}, got ${JSON.stringify([...failures].slice(0, 4))}`])
      survived.push(cell.id)
    } else {
      rows.push([cell.id, 'KILLED', `by: ${aimed[0]}`])
    }
  }
  fs.rmSync(dir, { recursive: true, force: true })
}

const after = sha(SUBJECT)
console.log(`\n${'='.repeat(74)}\nMUTATION MATRIX\n${'='.repeat(74)}`)
for (const [id, status, detail] of rows) {
  console.log(`  ${status.padEnd(17)} ${id.padEnd(34)} ${detail}`)
}
console.log(`\n  subject sha256[:16] before=${before} after=${after} -> tracked file ${before === after ? 'UNCHANGED' : '*** MODIFIED ***'}`)
console.log(`  cells: ${MUTATIONS.length}   killed: ${MUTATIONS.length - survived.length}   survived/unearned: ${survived.length}`)
const ok = survived.length === 0 && before === after
console.log(`\n  ${ok ? 'ALL CELLS KILLED' : 'MATRIX INCOMPLETE'}`)
process.exit(ok ? 0 : 1)
