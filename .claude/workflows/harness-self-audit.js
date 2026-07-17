export const meta = {
  name: 'harness-self-audit',
  description: 'REPORT-ONLY self-audit of the pyfinagent Layer-3 harness + Layer-2/4 MAS vs the latest Claude Code / Anthropic capabilities. Fan-out READ-ONLY auditors -> verify -> return ranked findings. It NEVER edits, commits, pushes, or flips the masterplan -- the return value is a findings report the operator reviews.',
  whenToUse: 'The re-runnable stress-test-doctrine self-audit (phase-71.6). Run manually, or on a weekly REPORT-ONLY cadence the OPERATOR activates (external cron/launchd or a deterministic report writer on the register_meta_evolution_cron APScheduler pattern). Regenerates the register shape of handoff/current/harness_proposals.json. Structurally report-only by TOOL-RESTRICTION (auditors are agentType:Explore -> no Edit/Write/Agent), backstopped by the 62.0 PreToolUse guard (blocks git push / launchctl).',
  phases: [
    { title: 'Audit', detail: 'read-only finders across harness + MAS dimensions' },
    { title: 'Verify', detail: 'adversarially re-check each finding; drop the unsupported' },
  ],
}

// ---------------------------------------------------------------------------
// phase-71.6 -- STRUCTURALLY REPORT-ONLY (enforcement by tool-restriction, not
// by a "report-only" prompt -- the resumption-risk memory: prompts are not
// enforcement). Every auditor is spawned agentType:'Explore' (read-only: no
// Edit/Write/Agent; the workflow SCRIPT itself has no fs/shell/git access). The
// workflow RETURNS ranked findings; it does NOT write files, commit, push, or
// touch .claude/masterplan.json. Main (or a deterministic scheduled writer)
// persists the returned findings to handoff/self_audit/<date>-harness-audit.md.
// Do NOT change the auditors to a write-capable agentType -- that would break
// the structural report-only guarantee.
// ---------------------------------------------------------------------------

const a = (typeof args === 'object' && args) ? args : (typeof args === 'string' && args.trim() ? JSON.parse(args) : {})

// Audit dimensions (each a read-only lens over a different slice of the surface).
const DIMENSIONS = Array.isArray(a.dimensions) && a.dimensions.length ? a.dimensions : [
  { key: 'harness-protocol', focus: 'the Layer-3 harness: docs/runbooks/per-step-protocol.md, CLAUDE.md harness rules, .claude/agents/{researcher,qa}.md, .claude/workflows/qa-verdict.js, the 5-file handoff protocol. Are any encoded assumptions ("the model can\'t do X") now stale (stress-test doctrine)? Any drift between the three cross-linked files?' },
  { key: 'layer2-mas', focus: 'the Layer-2 in-app MAS: backend/agents/multi_agent_orchestrator.py, agent_definitions.py, evaluator_agent.py, skill_optimizer.py + skill_modification_review.py. Silent-failure classes, un-reviewed self-modification, fabricated numbers, clobber bugs, structured-output gaps.' },
  { key: 'layer4-meta', focus: 'the Layer-4 meta-evolution: backend/meta_evolution/* (directive_review, directive_rewriter, cron), meta_coordinator.py. Un-gated writes, dormant code, HITL-vs-auto-apply asymmetries.' },
  { key: 'capabilities-drift', focus: 'the latest Claude Code / Anthropic capabilities (Workflow structured-output, effort tiers, structured outputs GA, hooks, subagent runtime semantics) vs what this harness actually uses. What new capability is NOT yet adopted? What scaffolding is now dead weight?' },
]

const FINDING_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'findings'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['title', 'severity', 'evidence', 'proposal'],
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['P1', 'P2', 'P3'] },
          evidence: { type: 'string', description: 'file:line anchors + why it is a real gap' },
          proposal: { type: 'string', description: 'the smallest safe change; NEVER apply it here -- report-only' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['title', 'verdict', 'reason'],
  properties: {
    title: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'UNCERTAIN'] },
    reason: { type: 'string' },
  },
}

const FINDER = (d) => [
  'You are a READ-ONLY auditor for the pyfinagent harness/MAS self-audit (phase-71.6). You have Read/Grep/Glob/WebFetch',
  'ONLY -- you CANNOT edit, write, commit, push, or flip anything (report-only by construction). Do NOT attempt to.',
  '',
  'AUDIT THIS DIMENSION: ' + d.focus,
  '',
  'Compare what the code/docs ACTUALLY do against the latest Anthropic / Claude Code guidance (fetch the canonical',
  'docs: anthropic.com/engineering/harness-design-long-running-apps, multi-agent-research-system,',
  'building-effective-agents; code.claude.com/docs). Apply the STRESS-TEST DOCTRINE: every harness component encodes',
  'an assumption about what the model can\'t do on its own -- flag any that are now stale.',
  '',
  'Return the ranked findings for this dimension. Each finding: file:line evidence + the SMALLEST safe proposal',
  '(report-only -- you are NOT implementing it). Prefer P1 real bugs / safety gaps over P3 nice-to-haves. If the',
  'dimension is clean, return an empty findings array honestly (do not invent findings).',
].join('\n')

// Fan-out finders (read-only) -> adversarially verify each finding as soon as its
// dimension completes (pipeline: no barrier). REPORT-ONLY throughout.
const results = await pipeline(
  DIMENSIONS,
  (d) => agent(FINDER(d), { label: 'audit:' + d.key, phase: 'Audit', schema: FINDING_SCHEMA, agentType: 'Explore', effort: 'high' }),
  (res) => parallel((res && Array.isArray(res.findings) ? res.findings : []).map((f) => () =>
    agent(
      'READ-ONLY adversarial verification (report-only; you cannot edit/push). Try to REFUTE this audit finding on ' +
      (res.dimension || 'unknown') + ': "' + f.title + '". Evidence claimed: ' + f.evidence +
      '. Read the actual file(s) and decide CONFIRMED (real, reproduces) / REFUTED (wrong or already handled) / UNCERTAIN. Default UNCERTAIN when you cannot confirm.',
      { label: 'verify:' + (f.title || '').slice(0, 24), phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'Explore', effort: 'high' },
    ).then((v) => ({ ...f, dimension: res ? res.dimension : 'unknown', verdict: v }))
  )),
)

// Ranked, verified findings -- the RETURN VALUE is the report (report-only; nothing written/pushed here).
const flat = results.flat().filter(Boolean)
const confirmed = flat.filter((f) => f.verdict && f.verdict.verdict === 'CONFIRMED')
const order = { P1: 0, P2: 1, P3: 2 }
confirmed.sort((x, y) => (order[x.severity] ?? 9) - (order[y.severity] ?? 9))

return {
  report_only: true,
  dimensions_audited: DIMENSIONS.map((d) => d.key),
  total_findings: flat.length,
  confirmed_count: confirmed.length,
  confirmed_findings: confirmed,
  all_findings: flat,
  note: 'REPORT-ONLY. Persist this to handoff/self_audit/<date>-harness-audit.md and review; nothing was applied. Weekly ACTIVATION is operator-gated (schedule_needs_operator=true).',
}
