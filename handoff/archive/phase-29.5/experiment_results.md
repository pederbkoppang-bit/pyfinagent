# Experiment Results — phase-29.5 (Add 4th `deep` research tier)

**Step ID:** phase-29.5
**Date:** 2026-05-19
**Cycle:** 1

This is an **agent-definition** cycle. One insertion into `.claude/agents/researcher.md` + masterplan update. No code edits.

---

## 1. Edit made (verbatim)

### `.claude/agents/researcher.md` — tier table + new section

**Before:** 202 lines. Tier table at lines 144-148 had 3 rows: simple / moderate / complex.

**After:** 265 lines (+63). Tier table now has 4 rows; new `### \`deep\` tier — additional requirements (phase-29.5)` section inserted immediately after the table.

**New table row** (between `complex` and the closing paragraph):
```
| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |
```

**New section structure (4 mandatory practices + gate check):**

1. **Multi-pass (scan -> gap -> adversarial)** — Pass 1 broad scan; Pass 2 gap analysis; Pass 3 adversarial pass. Source: arXiv:2601.20975 (DeepSearchQA n=1->n=8: 67%->86%).
2. **Adversarial sourcing (≥1 `[ADVERSARIAL]` source)** — Anchor Effect + Homogeneity Bias countermeasure. Sources: arXiv:2601.22984 (PIES taxonomy) + PMC11615553 (devil's-advocate 0%->76% accuracy, OR=3.49, p=.002).
3. **Cross-domain triangulation (≥2 adjacent-domain sources)** — Source: DRACO benchmark arXiv:2602.11685v1 (21.6-pt Finance gap; +11.5 pts overall from richer context).
4. **Multi-subagent fork option (2-3 parallel subagents, each ≥20 sources independently)** — Source: Anthropic multi-agent research blog "more than 10 subagents".

5-condition gate check at the end.

---

## 2. Verbatim verification command output

```
$ grep -E '^\|\s*deep\s*\|' .claude/agents/researcher.md
| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |

$ grep -q 'at least 20 (typically 20-50)' .claude/agents/researcher.md && \
  grep -q 'Pass 1: broad' .claude/agents/researcher.md && \
  grep -q '\[ADVERSARIAL\]' .claude/agents/researcher.md && \
  grep -q 'Cross-domain triangulation' .claude/agents/researcher.md && \
  grep -q 'Multi-subagent fork option' .claude/agents/researcher.md && \
  grep -q '`deep` gate check' .claude/agents/researcher.md
$ echo exit=$?
exit=0

$ wc -l .claude/agents/researcher.md
265 .claude/agents/researcher.md
```

All 7 criteria PASS. Exit 0.

---

## 3. Files touched

| File | Change |
|---|---|
| `.claude/agents/researcher.md` | +63 lines (1 table row + 1 new section); 202 → 265 |
| `.claude/masterplan.json` phase-29.5 | audit_basis + verification fields rewritten |
| `handoff/current/research_brief.md` | rewritten (8 sources, gate_passed=true) |
| `handoff/current/contract.md` | rewritten |
| `handoff/current/experiment_results.md` | this file |
| `handoff/current/live_check_29.5.md` | new |

**No** `backend/`, `frontend/`, `scripts/` files touched.

---

## 4. Honest disclosures

1. **Researcher subagent stopped mid-flight at first attempt** (same pattern as phase-28.16, phase-28.6/.7/.8) — Main sent a SendMessage continuation directing the agent to write the final brief in a single Write call. Agent completed on continuation (8 sources read in full, gate_passed=true). NOT a fallback to Main authoring — the brief is the researcher's work.
2. **No vendor explicitly documents the adversarial-sourcing requirement** as a discrete procedural step. The proposed rule is a pyfinagent-tier innovation grounded in published bias research (PIES, PMC11615553). The researcher brief §"Key findings" #7 documents this honestly.
3. **The 20-source floor is conservative.** Literature supports "hundreds" for exhaustive academic surveys; 20-50 was chosen to fit the operator's masterplan-step cadence (a `deep` cycle should still complete within a Max 5-hour rolling window).
4. **OpenAI deep-research cookbook was unfetchable** (HTTP 403 on official blog + binary PDF on system card + low-content cookbook); deferred to snippet-only. Brief honestly flags this.
5. **Frontmatter edits don't activate until session restart.** The new tier is on-disk but the Researcher subagent won't recognise `tier: deep` until the operator restarts Claude Code. live_check_29.5.md documents this.
6. **Anti-rubber-stamp:** 7 ANDed grep predicates each anchored on a distinct phrase (table row regex, source-count phrase, pass-1 marker, adversarial tag, cross-domain phrase, fork-option phrase, gate-check backticked phrase). Removing any one of the rule's load-bearing facts fails the verification.

---

## 5. Decision

Ready for Q/A spawn. 7 success criteria all evidenced.
