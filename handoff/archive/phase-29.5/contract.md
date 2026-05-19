# Contract — phase-29.5 (Add 4th `deep` research tier to researcher.md)

**Step ID:** phase-29.5
**Date:** 2026-05-19
**Author:** Main (overnight execution)
**Tier:** complex

---

## Research-gate summary

| Metric | Value |
|---|---|
| Sources read in full | 8 |
| Snippet-only | 12 |
| URLs collected | 20 |
| Recency scan + 7-day frontier-sync | DONE |
| `gate_passed` | true |

**Brief:** `handoff/current/research_brief.md` (239 lines).

**Headline findings:**
1. **Google Deep Research Max (Apr 21 2026):** Standard ~80 queries / ~250K tokens; Max ~160 queries / ~900K tokens / 10-20 min — concrete numerical anchors for the proposed `deep` tier.
2. **Anthropic multi-agent research blog:** "1/3-5/10+" tool-call scaling rule; "more than 10 subagents" pattern explicitly noted as appropriate for complex research.
3. **DeepSearchQA (arXiv:2601.20975, Jan 2026):** Multi-sample n=1→n=8 lifts accuracy 67% → 86% (+19 pts) — validates multi-pass.
4. **PIES bias taxonomy (arXiv:2601.22984):** Anchor Effect + Homogeneity Bias are primary deep-research failure modes; adversarial sourcing is the structural countermeasure.
5. **Devil's-advocate clinical study (PMC11615553):** adversarial role assignment lifted accuracy 0% → 76% (OR=3.49, p=.002).
6. **DRACO benchmark (arXiv:2602.11685v1):** 21.6-pt Finance gap; Perplexity 17x more tokens than o3 → +11.5 pts overall — token-depth correlates with cross-domain quality.
7. **Claude Max cost compatibility:** 900K-token deep session ≈ 1 full 5-hour Max-20x window; ≈ $300-500/session at API rates, FREE under Max flat-fee.

---

## Audit-basis (from phase-29.0)

phase-29.0 §1.1 + §"P1 #5": existing 3-tier table caps at `complex` (8-15 typical reads). Need 4th `deep` tier for 20-50 sources, multi-pass, adversarial sourcing, cross-domain triangulation, multi-subagent fork option. Validated by Anthropic's multi-agent blog + Google DR Max + arXiv literature. **No vendor disconfirmation** — the only debate is whether the 20-source floor is too conservative (literature supports "hundreds" for exhaustive work; pyfinagent picks 20-50 to fit operator's masterplan-step cadence).

---

## Verbatim immutable success criteria

1. `deep_tier_row_in_effort_tiers_table` — `.claude/agents/researcher.md` table now has 4 rows (simple, moderate, complex, **deep**).
2. `deep_tier_row_matches_research_brief_proposal` — row reads `| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |`.
3. `multi_pass_rule_documented` — paragraph block names pass-1 broad scan / pass-2 gap analysis / pass-3 adversarial.
4. `adversarial_sourcing_rule_documented` — paragraph requires ≥1 `[ADVERSARIAL]` source in the read-in-full table.
5. `cross_domain_triangulation_rule_documented` — paragraph requires ≥2 sources from adjacent domains for domain-specific claims.
6. `multi_subagent_fork_option_documented` — paragraph documents the 2-3 parallel-subagent fork pattern with the ≥3-separable-sub-questions trigger.
7. `deep_gate_check_block_present` — paragraph ends with the 5-condition `deep` gate check.

**Verification command:**
```bash
grep -E '^\|\s*deep\s*\|' .claude/agents/researcher.md && \
grep -q 'at least 20 (typically 20-50)' .claude/agents/researcher.md && \
grep -q 'pass 1 broad scan\|pass 1.*broad scan\|Pass 1: broad' .claude/agents/researcher.md && \
grep -q '\[ADVERSARIAL\]' .claude/agents/researcher.md && \
grep -q 'cross-domain triangulation' .claude/agents/researcher.md && \
grep -q 'Multi-subagent fork option' .claude/agents/researcher.md && \
grep -q '`deep` gate check' .claude/agents/researcher.md
```

**`verification.live_check`:** `"live_check_29.5.md captures pre/post line-count of researcher.md + verbatim diff of the new table row + verbatim insertion of the deep-tier paragraph block; documents that the new tier doesn't activate until session restart (per CLAUDE.md agent-definition rule)."`

---

## Plan

1. DONE — Spawn researcher complex.
2. DONE — Write contract.
3. NEXT — GENERATE:
   - EDIT 1: Insert `| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |` row into `.claude/agents/researcher.md` tier table (line 148 area).
   - EDIT 2: Insert the 5-item deep-tier paragraph block immediately AFTER the table, BEFORE the "Tier controls the DEPTH..." sentence.
   - EDIT 3: Update masterplan.json 29.5 entry: name + audit_basis + verification.
   - EDIT 4: Write experiment_results.md (verbatim diffs + verification output).
   - EDIT 5: Write live_check_29.5.md.
4. Spawn `qa` once. Circuit breaker: 2 fresh-qa.
5. Append log → flip masterplan → commit.

---

## Out of scope

- Implementing the deep tier — that's a future research-call's choice; this cycle only documents it.
- `mas_research` (Layer-2) — separate system.
