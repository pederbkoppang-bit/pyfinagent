# Contract — phase-29.2 (Codify Opus + max effort on Researcher + Q/A)

**Step ID:** phase-29.2
**Date:** 2026-05-18
**Author:** Main (overnight execution session)
**Tier:** complex

---

## Audit-basis INVERSION (operator pre-approval, documented for permanence)

The phase-29.0 audit recommended reverting `.claude/agents/researcher.md:10` from `effort: max` → `medium` per Anthropic's Sonnet 4.6 default. The OPERATOR has now overridden this in the overnight prompt 2026-05-18:

> "29.2 audit-inversion + Opus upgrade APPROVED. Set `model: opus` + `effort: max` in researcher.md frontmatter; `effort: max` in qa.md; remove the 'temporarily raised … revert' comments from both; update CLAUDE.md effort-policy block to document Max-subscription + Researcher-on-Opus rationale."

**Research support (handoff/current/research_brief.md):**
- Opus 4.7 vs Sonnet 4.6 on research-depth tasks: **17-pt GPQA Diamond gap** (91.3% vs 74.1%), **79-Elo GDPval-AA gap** (1,753 vs 1,674) — research-synthesis is GPQA-analog territory.
- `model: opus` + `effort: max` in subagent frontmatter is **officially supported** (code.claude.com/docs/en/sub-agents + code.claude.com/docs/en/model-config).
- Anthropic recommends `xhigh` as the Opus 4.7 starting point; `max` is a deliberate over-spec ("Reserve for genuinely frontier problems"). Operator's choice is valid under Max flat-fee.
- Max plan auto-includes Opus 1M context, **mitigating GitHub issue #51060** ("model: opus fails with 1M context extra usage" in spawned subagents).
- Sonnet 4.6 does NOT support xhigh; Opus 4.7 does. Moving Researcher to Opus expands the capability range.
- Researcher fires **once per masterplan step**, not per ticker — token cost is contained regardless of effort level.

---

## Verbatim immutable success criteria (from masterplan.json phase-29.2, UPDATED for inversion)

The phase-29.0-original criteria were:
- ❌ `researcher_effort_is_medium`
- ❌ `qa_effort_unchanged_at_xhigh`
- ✅ `session_restart_documented_in_handoff`

These have been **superseded by operator directive**. The updated phase-29.2 criteria (to be written into masterplan.json `verification.success_criteria` in this cycle's GENERATE phase) are:

1. `researcher_model_is_opus` — `.claude/agents/researcher.md` frontmatter has `model: opus`
2. `researcher_effort_is_max` — `.claude/agents/researcher.md` frontmatter has `effort: max`
3. `qa_effort_is_max` — `.claude/agents/qa.md` frontmatter has `effort: max`
4. `temp_raise_comments_removed` — neither file contains the "temporarily raised … revert after step closes" comment block
5. `claude_md_effort_policy_documents_opus_on_researcher` — CLAUDE.md Effort-policy section reflects the new policy and cites the overnight prompt as audit-basis
6. `session_restart_documented_in_handoff` — `live_check_29.2.md` notes that frontmatter changes activate post-restart

**Updated verification command** (for masterplan.json):
```bash
grep -E '^model:\s*opus' .claude/agents/researcher.md && \
grep -E '^effort:\s*max' .claude/agents/researcher.md && \
grep -E '^effort:\s*max' .claude/agents/qa.md && \
! grep -q 'Revert after step closes' .claude/agents/researcher.md && \
! grep -q 'Revert after step closes' .claude/agents/qa.md && \
grep -q 'Claude Opus 4.7.*Researcher.*Q/A.*max' CLAUDE.md
```

**`verification.live_check`** (R-1 gate):
> "Post-restart `scripts/qa/verify_qa_roster_live.sh` (or equivalent self-disclosure prompt) confirms that a freshly-spawned Researcher subagent reports `model: opus`/`effort: max` (or `claude-opus-4-7`) in its self-introduction. live_check_29.2.md captures the pre-restart on-disk state and the operator's morning-restart verification recipe."

---

## Plan steps

1. **DONE** — Spawn `researcher` complex tier with audit-inversion context.
2. **DONE** — Write this contract.md.
3. **NEXT — GENERATE** — make 3 file edits + 1 masterplan-entry update:
   - **EDIT 1:** `.claude/agents/researcher.md` lines 1-14
     - `model: sonnet` → `model: opus`
     - Remove the 4-line `# phase-23.2.2 (2026-05-16) … Revert after step closes` comment block
     - `effort: max` stays
     - Update the policy note in the body if any references Sonnet
   - **EDIT 2:** `.claude/agents/qa.md` lines 1-14
     - Remove the 4-line `# phase-23.2.2 (2026-05-16) … Revert after step closes` comment block
     - `model: opus` stays
     - `effort: max` stays
   - **EDIT 3:** `CLAUDE.md` Effort-policy section
     - Rewrite the Researcher bullet to Opus/max with Max-subscription rationale
     - Update the Q/A bullet to Opus/max (was xhigh)
     - Cite the overnight prompt 2026-05-18 as audit-basis
     - Add explicit note: Layer-2 `mas_research` is a separate system; not in scope this step
   - **EDIT 4:** `.claude/masterplan.json` `phase-29.2` entry
     - Update `audit_basis` to cite the operator override + brief
     - Update `verification.command` and `success_criteria` per the new criteria above
     - Update `verification.live_check` per the new live-check shape
4. Write `experiment_results.md` summarising the 4 edits with verbatim diffs and verification command output.
5. Spawn `qa` ONCE. Provide the 5-item harness-compliance audit prompt; on CONDITIONAL/FAIL fix + update handoff + FRESH qa on UPDATED evidence. CIRCUIT BREAKER: max 2 fresh-qa attempts per step.
6. Write `live_check_29.2.md`.
7. Append cycle to `handoff/harness_log.md` BEFORE the masterplan flip.
8. Flip masterplan 29.2 status to `done` via Edit tool (so the auto-commit + push hook fires).
9. Move directly to cycle 29.1.

---

## Out of scope

- Layer-2 `backend/config/model_tiers.py` `mas_research` / `mas_qa` defaults. Phase-29.2 scope is Layer-3 only. A separate cycle can revert the Layer-2 temp-raise.
- Any other `.claude/agents/*` files.
- `backend/`, `frontend/`, `scripts/` (apart from `live_check_29.2.md` evidence).

---

## References

External (from research brief):
- https://platform.claude.com/docs/en/build-with-claude/effort
- https://code.claude.com/docs/en/model-config
- https://code.claude.com/docs/en/sub-agents
- https://www.anthropic.com/news/claude-opus-4-7
- https://artificialanalysis.ai/articles/opus-4-7-everything-you-need-to-know
- https://github.com/anthropics/claude-code/issues/51060

Internal:
- `.claude/agents/researcher.md:5,7-10`
- `.claude/agents/qa.md:5,7-10`
- `CLAUDE.md` lines 51-55 (Effort policy)
- `handoff/current/research_brief.md` (this cycle, 214 lines)
- `handoff/archive/phase-29.0/experiment_results.md` §2.1 (audit recommendation — superseded)
