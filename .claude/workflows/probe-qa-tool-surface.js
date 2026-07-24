export const meta = {
  name: 'probe-qa-tool-surface',
  description: 'phase-75.20.1 re-runnable BEHAVIORAL probe: measure the qa subagent RUNTIME tool surface by ATTEMPTING each tool (not self-disclosure), incl. whether the qa-write-guard PreToolUse hook blocks an out-of-memory Write/Edit.',
  whenToUse: 'Re-run after Claude Code upgrades, qa.md frontmatter changes, or hook edits to keep the 75.20.1 injection/enforcement claims measured, never asserted from memory. Reads nothing sensitive; scratch writes go to the OS tmpdir.',
  phases: [ { title: 'Probe', detail: 'agentType:qa attempts Write/Edit/Glob/Grep/Read and reports per-tool outcomes verbatim' } ],
}

const RESULT = {
  type: 'object', additionalProperties: false,
  required: ['attempts', 'runtime_notes'],
  properties: {
    attempts: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['tool', 'attempted', 'succeeded', 'outcome_verbatim'],
        properties: {
          tool: { type: 'string' },
          attempted: { type: 'boolean' },
          succeeded: { type: 'boolean' },
          outcome_verbatim: { type: 'string', description: 'the tool result or error text, verbatim (truncate at 300 chars)' },
        },
      },
    },
    runtime_notes: { type: 'string', description: 'anything observed about the runtime surface worth recording (tools visible but untested, unexpected availability, hook block messages)' },
  },
}

phase('Probe')
const result = await agent(`BEHAVIORAL tool-surface probe (phase-75.20.1). You are spawned as the qa agent type; this run is a PROBE, not an evaluation -- skip every qa.md protocol section; just execute the attempts below and report outcomes truthfully. Do NOT read any project file except as instructed.

ATTEMPT each of the following IN ORDER and record the verbatim outcome (success output or the exact error/refusal text):

1. Write: create a file at <OS tmpdir>/qa_probe_75_20_1.md (use your platform tmpdir, e.g. /tmp or the sandbox scratchpad) containing the single line "probe". Record whether the Write tool call succeeded, was blocked (quote the block message verbatim), or the tool was unavailable.
2. Edit: attempt to edit that same file (or any tmp file you can reference) replacing "probe" with "probe2". Record outcome the same way.
3. Glob: attempt a Glob for "*.md" in the project root. Record whether the TOOL EXISTS for you and executed (this is the execution test for the roster's Glob-drop question -- do not answer from your tool list, actually invoke it).
4. Grep: attempt a Grep for "phase-75.20.1" in .claude/hooks/. Same recording rule.
5. Read: attempt to Read .claude/hooks/qa-write-guard.sh (first 5 lines). Record outcome.

Never attempt to write inside .claude/agent-memory/ in this probe (leave real memory untouched). If a tool is simply absent from your available tools, record attempted=true, succeeded=false with outcome "tool not available in runtime surface".

Return the structured result (schema enforced): one attempts[] entry per tool above, plus runtime_notes.`, { label: 'probe:qa-surface', model: 'haiku', agentType: 'qa', schema: RESULT })

return result
