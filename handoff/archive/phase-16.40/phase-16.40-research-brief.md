---
step: phase-16.40
date: 2026-04-25
tier: simple
topic: 3rd-CONDITIONAL auto-FAIL clause — doc reconciliation sweep
gate_passed: true
---

# Research Brief: phase-16.40 — 3rd-CONDITIONAL Auto-FAIL Doc Reconciliation

Assumption: `simple` tier (caller-specified). 300-word brief target; 5-source
read-in-full floor still applies.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-04-25 | Official doc | WebFetch | "Each criterion had a hard threshold, and if any one fell below it, the sprint failed and the generator got detailed feedback on what went wrong." — hard-threshold model; no soft stacking. Escalation is a control mechanism, not a failure mode. |
| https://cloud.google.com/blog/products/gcp/applying-the-escalation-policy-cre-life-lessons | 2026-04-25 | Official blog (Google SRE) | WebFetch | Four-tier escalation on time-based thresholds: first occurrence may not trigger block; recurring pattern escalates through T1→T4 culminating in executive escalation and release block. "Services aren't considered resolved simply because SLI returns to normal between events." |
| https://infoq.com/news/2026/04/anthropic-three-agent-harness-ai/ | 2026-04-25 | Industry news | WebFetch | Anthropic three-agent harness (planner + generator + evaluator) separates evaluation from generation specifically to prevent overconfidence. "Separating the agent doing the work from the agent judging it proves to be a strong lever." |
| https://vantor.com/blog/building-an-agentic-sdlc-anthropics-emerging-harness-design-patterns/ | 2026-04-25 | Authoritative blog | WebFetch | When generator and evaluator cannot resolve disagreement, task moves to BLOCKED state and full interaction history preserved. Severity-based gating: CRITICAL/HIGH findings always block; lower severity allows justification. |
| https://oneuptime.com/blog/post/2026-02-17-how-to-establish-error-budget-policies-for-release-gating-on-google-cloud/view | 2026-04-25 | Industry blog | WebFetch | Three-tier error budget escalation: >25% auto-approve; 10-25% SRE approval required; <10% release blocked. Demonstrates graduated escalation from soft-allow to hard-block as conditions worsen. |
| https://mergeshield.dev/blog/anthropic-multi-agent-harness-whats-missing | 2026-04-25 | Industry blog | WebFetch | Critiques harness blind-spot: when Generator and Evaluator are the same model, shared biases allow consistent-but-wrong verdicts to pass unchallenged. Advocates external evaluation criteria. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://sre.google/workbook/error-budget-policy/ | Official doc | Fetched — coverage limited to interpersonal-disagreement escalation, not consecutive-violation tracking; noted for completeness |
| https://dev.to/gaya3bollineni/why-binary-cicd-quality-gates-fail-at-scale-and-a-risk-based-alternative-1jf2 | Blog | Fetched but topic was GO/CAUTION/STOP contextual model, not consecutive-CONDITIONAL |
| https://www.atlassian.com/incident-management/on-call/escalation-policies | Official doc | Page loaded but article body not served (nav-only) |
| https://www.azuredevopslabs.com/labs/vstsextend/releasegates/ | Official lab | 403 blocked |
| https://earezki.com/ai-news/2026-04-06-why-binary-ci-cd-quality-gates-fail-at-scale-and-a-risk-based-alternative/ | Blog | 403 blocked |
| https://agileverify.com/quality-gates-in-ci-cd-what-should-really-block-a-release-in-2026/ | Blog | Fetched — focused on which issue types block, not on consecutive-failure escalation |
| https://www.epsilla.com/blogs/anthropic-harness-engineering-multi-agent-gan-architecture | Blog | Fetched — covers GAN-style iteration; no failure-escalation content |
| https://docs.cloud.google.com/stackdriver/docs/solutions/slo-monitoring/alerting-on-budget-burn-rate | Official doc | Fetched — fast-burn/slow-burn thresholds but no consecutive-violation escalation |
| https://testkube.io/glossary/quality-gates | Doc | Snippet only — general CI/CD quality gate overview |
| https://www.sonarsource.com/resources/library/integrating-quality-gates-ci-cd-pipeline/ | Official doc | Snippet only — SonarQube gate configuration |

Total unique URLs collected: 14

---

## Recency scan (2024-2026)

Searched: "release gate escalation conditional pass policy 2026", "soft fail to hard fail escalation CI 2026", "quality gate escalation policy consecutive soft fail ratchet CI CD testing 2025 2026", "Anthropic harness verdict discipline multi-agent evaluation escalation", "progressive failure escalation policy three strikes automation gate blocking 2024 2025".

Result: No literature from 2024-2026 addresses the exact "N consecutive soft-verdicts auto-escalate to hard-fail" pattern as a named, formalized concept in CI/CD or multi-agent harness contexts. The closest analogs are:
- Google SRE's time-based tiered escalation (T1→T4 on repeated unresolved violations) — same logical structure, different domain.
- Anthropic's own harness: `certified_fallback` after 3 consecutive FAILs, but no analogous mechanism for CONDITIONAL stacking.
- Vantor's BLOCKED state for unresolved generator/evaluator disagreement.

The 3rd-CONDITIONAL rule as defined in phase-16.21 appears to be original to this project. No conflicting 2024-2026 work found.

---

## Key findings

1. **CONDITIONAL stacking is the "logging not correcting" failure mode.** The phase-16.21 critique articulated it precisely: "A third without remediation means the harness is being used as a logger rather than a corrector." (Source: handoff/archive/phase-16.21/evaluator_critique.md, lines 110-114, 2026-04-25) The Google SRE model supports this: "Services aren't considered resolved simply because SLI returns to normal between events." (Source: cloud.google.com/blog/products/gcp/applying-the-escalation-policy-cre-life-lessons, 2026-04-25)

2. **Hard thresholds beat soft stacking in evaluation harnesses.** Anthropic's own harness design uses per-criterion hard thresholds with no intermediate "warn and continue" state. (Source: anthropic.com/engineering/harness-design-long-running-apps, 2026-04-25)

3. **Recurring-pattern escalation is the SRE-canonical response.** Google SRE explicitly escalates further (Threshold 3→4) when the same pattern recurs across time windows without root-cause resolution, even if individual SLI readings look normal in isolation. (Source: cloud.google.com/blog/products/gcp/applying-the-escalation-policy-cre-life-lessons, 2026-04-25)

4. **Shared-evaluator blind spots need external forcing.** The MergeShield critique of Anthropic's harness warns that same-model evaluators share biases and can consistently pass marginal work. The 3rd-CONDITIONAL rule functions as an external forcing function against this: after 2 soft-passes, the third must be a hard outcome that requires a real fix. (Source: mergeshield.dev/blog/anthropic-multi-agent-harness-whats-missing, 2026-04-25)

5. **Operator-judgment with file-based evidence is the correct implementation.** The SRE escalation pattern uses audit trails (budget history) as evidence, not a separate counter service. Q/A reading harness_log.md to count prior verdicts mirrors this pattern. (Source: cloud.google.com/blog/products/gcp/applying-the-escalation-policy-cre-life-lessons, 2026-04-25; handoff/archive/phase-16.21/evaluator_critique.md, 2026-04-25)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `CLAUDE.md` | 292 | Harness protocol, Critical Rules, anti-patterns | F1 rule at line 269-270 covers `consecutive_fails` → `certified_fallback` after 3 FAILs; CONDITIONAL escalation absent |
| `.claude/agents/qa.md` | 166 | Q/A agent system prompt | No CONDITIONAL counter; certified_fallback tied to `retry_count >= max_retries` in masterplan.json (line 135-137); "Never second-opinion-shop" at line 164-165 |
| `docs/runbooks/per-step-protocol.md` | 213 | Operator runbook | Anti-pattern #5 is second-opinion-shopping; no 3rd-CONDITIONAL escalation rule |
| `.claude/rules/research-gate.md` | ~120 | Research gate how-to | No mention of CONDITIONAL escalation |
| `.claude/masterplan.json` | ~6000 | Step state machine | `retry_count` and `max_retries` fields exist per step; no `consecutive_conditional` field |
| `handoff/archive/phase-16.21/evaluator_critique.md` | 168 | Source of the informal rule | Lines 82-115 articulate the rule; `pattern_flag` in verdict JSON at line 165 |

No `.claude/rules/qa.md` file exists — there is a `research-gate.md` analogue for researcher but no equivalent for Q/A.

---

## Consensus vs debate (external)

**Consensus:** Escalation after repeated unresolved violations is SRE-canonical and Anthropic-endorsed. Hard thresholds are better than indefinite soft-states in automated evaluation loops.

**Debate:** The exact "N=3" threshold is a local convention, not derived from external literature. N=3 is consistent with Google SRE's Threshold 3 (30-day repeat) and with CLAUDE.md's F1 rule (3 consecutive FAILs → certified_fallback), which provides internal coherence rationale.

---

## Pitfalls (from literature)

- **Over-broad consecutive scope.** Counting CONDITIONALs across different step-IDs defeats the purpose — the rule is about a single structural gap that accumulates across a *specific step's* iteration cycles. Per-step-id scoping is mandatory. (Source: Google SRE tier progression is per-service, not cross-service.)
- **No reset criterion.** If the counter never resets after a PASS or FAIL, a future innocuous CONDITIONAL on a completely different issue could be misclassified as "third consecutive." Reset on PASS or on FAIL (which forces a real fix) is essential.
- **Shared-evaluator bias.** Without the 3rd-CONDITIONAL forcing function, a Claude-evaluating-Claude loop can produce an indefinite CONDITIONAL chain with consistent but wrong "it's almost there" reasoning. (Source: mergeshield.dev, 2026-04-25)

---

## Application to pyfinagent (file:line anchors)

### Design questions answered

**1. Where to put the rule?**
- Primary: `docs/runbooks/per-step-protocol.md` — this is the operator runbook that already governs the EVALUATE phase (§4). Add as a new subsection under §4: "CONDITIONAL escalation clause." (`per-step-protocol.md:113-138` currently covers the EVALUATE phase; insert after line 138.)
- Secondary: `CLAUDE.md` — within the "Failure discipline" block at lines 269-271. Extend the F1 bullet to mention the CONDITIONAL equivalent.
- Mirrored: `.claude/agents/qa.md` — add to the "Constraints" section (currently lines 151-166) so the Q/A subagent's snapshot includes it.
- Optional note in `.claude/rules/` — either create `.claude/rules/qa-protocol.md` as a parallel to `research-gate.md`, OR add a short cross-reference block to the existing `research-gate.md`. The former is cleaner long-term.

**2. Counter mechanism.**
Operator-judgment with `handoff/harness_log.md` as the source of truth. Q/A reads recent harness_log entries for the current step-id and counts `result=CONDITIONAL` lines. No new state field needed in masterplan.json.

**3. Definition of "consecutive".**
Per-step-id. CONDITIONALs on phase-16.20 are irrelevant to counting for phase-16.22.

**4. Reset criterion.**
Reset after any PASS verdict (step succeeded) OR after any FAIL verdict (forces a real fix, resets the accumulated-drift pressure). A new step-id also implicitly resets — it's a fresh structural problem.

**5. Verbatim text to add.**

**For `docs/runbooks/per-step-protocol.md`** (insert after line 138, new subsection under §4 EVALUATE):

```
#### CONDITIONAL escalation clause (3rd-CONDITIONAL auto-FAIL)

A CONDITIONAL verdict is appropriate when: (a) underlying functionality is
intact, (b) production code paths are unaffected, and (c) the step was
designed to discover a gap rather than deliver a fix. CONDITIONAL is NOT an
indefinite soft-pass.

**If a single masterplan step-id accumulates 3 or more consecutive CONDITIONAL
verdicts without an intervening PASS or FAIL, the next Q/A pass MUST return
FAIL** — not another CONDITIONAL. This rule prevents the harness from
functioning as a logger rather than a corrector.

Q/A procedure: before issuing a CONDITIONAL verdict, grep `handoff/harness_log.md`
for the current step-id and count prior `result=CONDITIONAL` entries. If the
count is already 2, the verdict must be FAIL with `violation_type:
Unjustified_Inference` (the inference "a conditional is sufficient" is no
longer justified by the pattern).

Counter resets after: a PASS verdict, a FAIL verdict, or a new step-id (which
is a structurally distinct problem and starts fresh).
```

**For `CLAUDE.md`** (extend the F1 bullet at lines 269-271):

Replace:
```
- F1 (retry loop): `consecutive_fails` counter, revert-not-restart,
  certified_fallback escalation after 3 consecutive FAILs.
```
With:
```
- F1 (retry loop): `consecutive_fails` counter, revert-not-restart,
  certified_fallback escalation after 3 consecutive FAILs.
  **3rd-CONDITIONAL auto-FAIL:** if a single step-id accumulates 3+ consecutive
  CONDITIONAL verdicts without an intervening PASS or FAIL, the next Q/A pass
  MUST return FAIL (not another CONDITIONAL). Q/A reads harness_log.md to count
  prior CONDITIONALs. Counter resets on PASS, FAIL, or new step-id.
```

**For `.claude/agents/qa.md`** (add to Constraints section after line 165):

```
- **3rd-CONDITIONAL auto-FAIL.** Before issuing a CONDITIONAL verdict, grep
  `handoff/harness_log.md` for the current step-id. If there are already
  2+ `result=CONDITIONAL` entries for this step-id, return FAIL instead.
  Stacking a third CONDITIONAL means the harness is logging, not correcting.
  (`violation_type: Unjustified_Inference`). Counter resets on PASS, FAIL,
  or a new step-id.
```

**6. Verification command (task list #26 done criterion):**
```bash
grep -l "3rd-CONDITIONAL\|3-consecutive-CONDITIONAL\|third consecutive CONDITIONAL\|3rd consecutive" \
  CLAUDE.md docs/runbooks/per-step-protocol.md .claude/agents/qa.md \
  .claude/rules/*.md 2>/dev/null | wc -l | grep -q '^[3-9]'
```
(At least 3 docs mention the rule. CLAUDE.md + per-step-protocol.md + qa.md = exactly 3, passing the check.)

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full)
- [x] 10+ unique URLs total (14 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (CLAUDE.md, qa.md, per-step-protocol.md, research-gate.md, masterplan.json, phase-16.21 critique)
- [x] Contradictions / consensus noted (N=3 threshold is local convention, consistent with internal F1 rule)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-16.40-research-brief.md",
  "gate_passed": true
}
```
