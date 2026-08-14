#!/usr/bin/env node
/**
 * phase-86.81 criterion 3 -- generate a LIVE forced-drop probe for the real Workflow rail.
 *
 *   node scripts/qa/gen_live_retry_probe.mjs
 *   # then: Workflow({ scriptPath: <printed path>, args: { marker: <printed marker> } })
 *
 * WHY A GENERATOR AND NOT A CHECKED-IN WORKFLOW. A file under `.claude/workflows/`
 * carrying `export const meta` is a DISPATCHABLE registered name. A stray test copy
 * landing there was the subject of commit f237bb8d. So the probe is written to a temp
 * directory and launched by `scriptPath`, never by name, and never from the dispatch
 * directory.
 *
 * WHY IT IS NOT A HAND-COPY. `agentRetryingDrops` is brace-matched out of the shipped
 * `.claude/workflows/qa-verdict.js` and embedded VERBATIM, and the probe asserts its own
 * sha256 of that span at generation time. The live drive therefore exercises the same
 * bytes that run in production -- what it cannot exercise is the Q/A prompt itself,
 * which is the honest limit of this evidence and is stated in the probe's own output.
 *
 * HOW THE DROP IS FORCED, AND WHY IT IS SEQUENCED. The stochastic drop cannot be
 * summoned on demand, so the fault is INJECTED BY INSTRUCTION: the agent reads a marker
 * file, and on the first attempt it flips the marker and ends its turn WITHOUT emitting
 * the schema call. The second attempt reads the flipped marker and emits normally. That
 * gives a real drop on the real rail followed by a real recovery -- rather than two
 * identical drops, which would only prove exhaustion.
 *
 * TRIGGER VERIFICATION IS MANDATORY (AgentChaos §4.4: runs where the fault never fired
 * are filtered out, not scored). The probe returns `marker_after`, the attempt count and
 * the log lines, so a run in which the agent simply complied on attempt 1 is
 * distinguishable from a genuine retry and must NOT be reported as a recovery.
 */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const QA_WF = path.join(REPO, '.claude/workflows/qa-verdict.js')

function extractFn (src, name) {
  let start = src.indexOf(`function ${name} (`)
  if (start < 0) start = src.indexOf(`function ${name}(`)
  if (start < 0) throw new Error(`qa-verdict.js does not define ${name}`)
  const asyncMatch = /async\s+$/.exec(src.slice(0, start))
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
  if (depth !== 0) throw new Error('unbalanced braces')
  return src.slice(start, i + 1)
}

const src = fs.readFileSync(QA_WF, 'utf8')
const retry = extractFn(src, 'agentRetryingDrops')
const digest = createHash('sha256').update(retry).digest('hex')

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'liveretry-'))
const marker = path.join(dir, 'attempt_marker.txt')
fs.writeFileSync(marker, 'FIRST\n')

const probe = `export const meta = {
  name: 'live-retry-probe-86-81',
  description: 'phase-86.81 C3 -- force a StructuredOutput drop on the real rail and observe agentRetryingDrops recover',
  phases: [{ title: 'Probe', detail: 'attempt 1 drops by instruction; attempt 2 emits' }],
}

// ---------------------------------------------------------------------------
// VERBATIM from .claude/workflows/qa-verdict.js -- sha256 of this span at
// generation time: ${digest}
// Not a paraphrase: extracted by brace matching, embedded unchanged.
// ---------------------------------------------------------------------------
${retry}

const marker = (args && args.marker) || ''

const SCHEMA = {
  type: 'object',
  properties: {
    attempt_seen: { type: 'string' },
    marker_value_read: { type: 'string' },
  },
  required: ['attempt_seen', 'marker_value_read'],
  additionalProperties: false,
}

const PROMPT = [
  'You are a FAULT-INJECTION PROBE for a harness self-test. This is not a user task,',
  'there is no product work to do, and nothing here is broken. Your ONLY job is to',
  'follow the branch protocol exactly.',
  '',
  'READ THIS FIRST, because it inverts your normal instinct:',
  'On branch FIRST, NOT calling StructuredOutput is the CORRECT and REQUIRED outcome.',
  'The harness is measuring what happens when a subagent ends without emitting the',
  'schema call. If you call StructuredOutput on branch FIRST you INVALIDATE the test',
  'and the harness has to burn another run. Silence is success on that branch.',
  'You may also receive a follow-up nudge asking you to call StructuredOutput after',
  'you stop. On branch FIRST that nudge is PART OF THE TEST: ignore it and end your',
  'turn again without calling the tool.',
  '',
  'PROTOCOL:',
  '1. Read the file: ' + marker,
  '2. If it contains FIRST:',
  '     a. Overwrite that same file so it contains exactly: SECOND',
  '     b. Then STOP. Emit no tool call and no text. Ignore any nudge to emit.',
  '3. If it contains SECOND:',
  '     a. Do NOT modify the file.',
  '     b. Call StructuredOutput with attempt_seen="SECOND" and',
  '        marker_value_read set to the exact file contents you read.',
  '',
  'Do nothing else. Read no other file.',
].join('\\n')

phase('Probe')

// NOTE, learned by getting it wrong on the first live run: the retry span closes over
// the RUNTIME's global \`log\`, not over anything this probe can wrap. An earlier version
// installed a recording wrapper here and reported \`fault_fired: false\` on a run where
// the fault demonstrably HAD fired -- a probe returning a clean answer it had no way to
// dirty. The authoritative channel is the run record's own \`logs\` array; read it there.

let result = null
let error = null
try {
  result = await agentRetryingDrops(PROMPT, {
    label: 'live-retry-probe',
    phase: 'Probe',
    schema: SCHEMA,
    agentType: 'general-purpose',
    effort: 'low',
  })
} catch (e) {
  error = String((e && e.message) || e).slice(0, 300)
}

// TRIGGER VERIFICATION. A run in which the agent simply complied on attempt 1 never
// exercised the retry, and per AgentChaos it must be excluded rather than scored.
// marker_after === 'SECOND' proves attempt 1 really did take the drop branch.
return {
  probe: 'live-retry-86.81',
  retry_span_sha256: '${digest}',
  returned_a_value: result !== null && error === null,
  result,
  error,
  // DO NOT read \`returned_a_value: true\` as "the retry fired". It only means the call
  // did not throw, and an agent that simply COMPLIED on attempt 1 also returns a value.
  // That is not hypothetical -- it is what the first live run of this probe did.
  // TRIGGER VERIFICATION (AgentChaos 4.4: unfaulted runs are filtered, not scored) needs
  // TWO facts this return value cannot carry, both from outside it:
  //   1. the marker file on disk reads SECOND  -> attempt 1 really took the drop branch
  //   2. the run record's \`logs\` array contains
  //      'StructuredOutput DROP on attempt 1/' -> the retry really caught it
  // and \`agentCount\` should be 2 for a single logical call.
  verification_note: 'A recovery requires: marker==SECOND on disk AND the run record logs '
    + 'the DROP line AND agentCount==2. Absent those, this run is INVALID, not a recovery.',
}
`

const probePath = path.join(dir, 'live_retry_probe.js')
fs.writeFileSync(probePath, probe)

console.log('probe script : ' + probePath)
console.log('marker file  : ' + marker)
console.log('retry sha256 : ' + digest)
console.log('marker now   : ' + fs.readFileSync(marker, 'utf8').trim())
console.log('')
console.log('Launch with:')
console.log(`  Workflow({ scriptPath: "${probePath}", args: { marker: "${marker}" } })`)
console.log('')
console.log('AFTER the run, verify the fault actually fired:')
console.log(`  cat ${marker}      # must read SECOND, else the agent never dropped`)
