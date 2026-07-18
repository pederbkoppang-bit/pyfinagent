# Contract — phase-72.5: Rollup + push (closes the phase-72 goal)

**Step id:** 72.5 (phase-72, depends_on 72.4 = done/PASS @037b5580)
**Session role:** Fable 5 + ultracode, AUDIT + RESEARCH ONLY.

## Research-gate summary (gate_passed: true — completeness-critic role)

Researcher via structured-output Workflow `wf_62ba4963-b33` (opus/max, tier=simple; floor held: 7 external sources read in full — SRE postmortem-closure, IIA audit-workpaper standards, Scrum DoD immutability — 15 URLs, recency scan, 9 internal files). Brief: `handoff/current/research_brief_72.5.md`. Returned a structured `dod_gaps` audit: **zero blocker gaps, two cosmetic items** (both fixed in this GENERATE):
1. P2's inline "$137.32 does not reconcile" wording was never back-annotated after P3 resolved it → resolution note appended in place.
2. P3 Recommend-ON table lacked an explicit current→proposed marker → header note added (all rows are at code default per P1).

Verified by the critic (independent, read-only): masterplan immutable criteria **byte-identical** from each step's install commit to HEAD (zero mutation, all 6 top-level + all 9 remediation steps); all 9 remediation steps pending + executor-tagged + live_check; `handoff/archive/phase-72.{0..4}/` each hold the four files + harness_log Cycles 112-116 = the five-file protocol per closed step; every archived critique carries the verbatim-transcribed Workflow verdict (run IDs recorded); origin/main == local main at audit time with 72.0-72.4 commits present; decision sheet ACT-NOW ↔ P3 ↔ P4 internally consistent; the alpha-framing header fix verified on-disk against the fresh file state.

## Hypothesis

With the two cosmetic fixes applied, all four DoD elements of the operator goal are satisfied and the phase can close with the 72.5 flip (whose auto-commit push carries the final working-tree changes — the critic explicitly flagged watching `auto-push.log` for the known stall).

## Immutable success criteria (verbatim from .claude/masterplan.json step 72.5)

- "money_diagnosis_72.md contains all three sub-period sections each with its own verified cause and evidence"
- "operator_decision_sheet_72.md covers P1 tokens + P3 levers + P4 regime policy, every line actionable"
- "Every phase-72 step closed with the five-file protocol and verbatim qa-verdict transcription; remediation steps present as pending executor-tagged masterplan entries; work pushed to origin/main"

verification.command: `bash -c 'test -f handoff/current/money_diagnosis_72.md && test -f handoff/current/operator_decision_sheet_72.md && git log origin/main --oneline -5 | grep -q "phase-72"'`

## Plan

1. GENERATE: apply the two cosmetic fixes (done); `experiment_results.md` with verbatim verification output.
2. EVALUATE via qa-verdict Workflow (final verdict of the phase); transcribe verbatim.
3. LOG (Cycle 117) → flip 72.5 done → auto-commit/push carries the closure diff → verify `auto-push.log` shows the push (manual `git push` fallback per the known-stall memory).
4. Final operator report (goal summary + ACT-NOW block).

## References

- `handoff/current/research_brief_72.5.md` (envelope + DoD-gap audit + closure-standards sources)
- All phase-72 artifacts: `money_diagnosis_72.md`, `operator_decision_sheet_72.md`, `live_check_72.0.md`, archives `handoff/archive/phase-72.*`, harness_log Cycles 112-116
