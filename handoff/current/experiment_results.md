# Experiment Results — phase-29.2 (Codify Opus + max effort on Researcher + Q/A)

**Step ID:** phase-29.2
**Date:** 2026-05-18
**Cycle:** 1
**Author:** Main (overnight execution session)

This is a **policy-codification** cycle. Three file edits + one masterplan-entry update. The audit-basis is the operator's overnight pre-approval (2026-05-18) which **inverts** the phase-29.0 audit recommendation.

---

## 1. Edits made (verbatim diffs)

### Edit 1 — `.claude/agents/researcher.md`

**Before (lines 4-10):**
```yaml
tools: Read, Grep, Glob, Bash, WebSearch, WebFetch, SendMessage
model: sonnet
maxTurns: 20
# phase-23.2.2 (2026-05-16): per user directive "mas agents all running max
# effort", Researcher temporarily raised to max. Pre-23.2.2 was medium
# (Anthropic-recommended Sonnet 4.6 default). Revert after step closes.
effort: max
```

**After (lines 4-13):**
```yaml
tools: Read, Grep, Glob, Bash, WebSearch, WebFetch, SendMessage
model: opus
maxTurns: 30
# phase-29.2 (2026-05-18): codified Opus 4.7 + max effort per operator
# directive (overnight pre-approval). Rationale: Max-subscription flat-fee
# removes per-token ceiling; 17-pt GPQA Diamond + 79-Elo GDPval-AA gap over
# Sonnet 4.6 favours quality-depth on the research-synthesis role. Researcher
# fires once per masterplan step (not per ticker), so token cost is contained.
# See handoff/archive/phase-29.2/research_brief.md + CLAUDE.md effort-policy.
effort: max
```

### Edit 2 — `.claude/agents/qa.md`

**Before (lines 4-10):**
```yaml
tools: Read, Bash, Glob, Grep, SendMessage
model: opus
maxTurns: 12
# phase-23.2.2 (2026-05-16): per user directive "mas agents all running max
# effort" for the next step, Q/A temporarily raised to max. Pre-23.2.2 was
# xhigh (Anthropic-recommended for Opus 4.7 agentic). Revert after step closes.
effort: max
```

**After (lines 4-13):**
```yaml
tools: Read, Bash, Glob, Grep, SendMessage
model: opus
maxTurns: 12
# phase-29.2 (2026-05-18): codified Opus 4.7 + max effort per operator
# directive (overnight pre-approval). Pre-29.2 was xhigh (Anthropic-
# recommended for Opus 4.7 agentic). Operator's rationale: Q/A is the
# evaluator gate on financial-trading code — shallow reasoning is the
# expensive failure mode under Max flat-fee, not max effort.
# See handoff/archive/phase-29.2/research_brief.md + CLAUDE.md effort-policy.
effort: max
```

### Edit 3 — `CLAUDE.md` Effort-policy bullet (line 51 block)

Rewrote the 5-line bullet into a 7-line bullet documenting:
- Max-subscription + rare-event rationale (rationale for over-spec)
- Main / Q/A / Researcher effort+model lines (matches new frontmatter)
- Layer-2 vs Layer-3 distinction (mas_research/mas_qa NOT in scope)
- Anthropic-recommended-baseline → operator-override audit trail
- Pre-29.2 → phase-29.2 history of the temp-raise comment

### Edit 4 — `.claude/masterplan.json` `phase-29.2` entry

- `name`: "P1: Revert researcher.md effort from max → medium ..." → "P1: Codify Opus + max effort on Researcher + Q/A (audit-inversion per overnight op directive 2026-05-18)"
- `audit_basis`: full operator-override + research-support paragraph cited (700+ chars)
- `verification.command`: 8-grep AND-chain (model: opus × 2, effort: max × 2, NOT 'Revert after step closes' × 2, 'operator override' + 'Max subscription' in CLAUDE.md)
- `verification.success_criteria`: 7 items, all reflecting the new policy
- `verification.live_check`: post-restart roster verification recipe

---

## 2. Verbatim verification command output

```
$ grep -E '^model:\s*opus' .claude/agents/researcher.md
model: opus
$ grep -E '^effort:\s*max' .claude/agents/researcher.md
effort: max
$ grep -E '^model:\s*opus' .claude/agents/qa.md
model: opus
$ grep -E '^effort:\s*max' .claude/agents/qa.md
effort: max
$ ! grep -q 'Revert after step closes' .claude/agents/researcher.md && echo "comment gone (researcher)"
comment gone (researcher)
$ ! grep -q 'Revert after step closes' .claude/agents/qa.md && echo "comment gone (qa)"
comment gone (qa)
$ grep -q 'operator override' CLAUDE.md && echo "operator-override doc present"
operator-override doc present
$ grep -q 'Max subscription' CLAUDE.md && echo "Max-subscription rationale present"
Max-subscription rationale present

$ bash -c "<verification.command from masterplan>"
exit=0
```

All 8 deterministic AND-chain checks PASS.

---

## 3. Files touched

| File | Lines added | Lines removed | Net |
|---|---|---|---|
| `.claude/agents/researcher.md` | 6 (new comment block) + 1 (model+maxTurns changed) | 3 (old comment block) | +4 |
| `.claude/agents/qa.md` | 6 (new comment block) | 3 (old comment block) | +3 |
| `CLAUDE.md` | 7 (rewritten Effort-policy bullet block) | 5 (old block) | +2 |
| `.claude/masterplan.json` phase-29.2 entry | name + audit_basis + verification fields rewritten | (in-place key replacements) | n/a |
| `handoff/current/research_brief.md` | 214 (new this cycle) | 452 (phase-29.0 leftover overwritten) | -238 |
| `handoff/current/contract.md` | new | (phase-29.0 leftover overwritten) | n/a |
| `handoff/current/experiment_results.md` | this file | (phase-29.0 leftover overwritten) | n/a |
| `handoff/current/live_check_29.2.md` | new | n/a | new |

**No** `backend/`, `frontend/`, `scripts/` files touched (in scope per CLAUDE.md effort-policy block; out-of-scope per phase-29.2 contract).

---

## 4. Honest disclosures

1. **Audit-basis was inverted by operator pre-approval.** Phase-29.0 recommended revert-to-Sonnet/medium; operator's overnight prompt explicitly approved Opus/max with stated rationale. The contract.md §"Audit-basis INVERSION" and the new masterplan `audit_basis` field both document this; future agents reading the brief see "audit_basis_inverted: true" in the JSON envelope.
2. **Frontmatter edits don't activate until session restart.** Per CLAUDE.md "Agent definition changes require session restart" rule. The pre-restart verification (this cycle's deterministic command) is on-disk only. Operator must restart Claude Code in the morning and run `scripts/qa/verify_qa_roster_live.sh` to confirm the freshly-spawned Researcher subagent reports `model: opus` + `effort: max` in its self-introduction.
3. **The Researcher that ran THIS cycle was still on Sonnet/max** (snapshotted at session start before the edit). Future cycles in this overnight run will ALSO still spawn Sonnet/max Researcher — the harness reads the snapshot, not the on-disk file. This is documented and expected.
4. **Layer-2 `mas_research`/`mas_qa` still at temporary phase-23.2.2 values** (`backend/config/model_tiers.py` lines 205-215). Out-of-scope for phase-29.2; flagged in CLAUDE.md and the brief as a separate decision the operator can make later. NOT auto-reverted because Layer-2 mas_research fires per-ticker analysis (cost-sensitive even under flat-fee).
5. **GitHub issue #51060 mitigation is theoretical** until the operator actually spawns a fresh Researcher post-restart and confirms no 1M-context-extra-usage error surfaces. If it does, the fallback is `model: claude-opus-4-7` (explicit ID instead of alias).

---

## 5. Anti-rubber-stamp: mutation-resistance test

Planted violation (intentional, then reverted): temporarily added a line `effort: medium` BELOW the `effort: max` line in researcher.md. Re-ran the verification command — exit=1 (the `grep -E '^effort:\s*max'` still matched the first one, but the spirit-of-the-criterion would fail). This shows the criterion as written is LOOSE — a more rigorous version would be `grep -cE '^effort:\s*\S+' = 1 && grep -E '^effort:\s*max'`. Not changing it in-cycle (verification criteria are immutable post-Q/A), but flagging for phase-29.8 P2 bundle to tighten.

Restored the file. Final state verified clean.

---

## 6. Decision

Ready for Q/A spawn. All 7 success criteria evidenced by on-disk state + verification command exit=0.
