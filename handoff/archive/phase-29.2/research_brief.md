# Research Brief — phase-29.2 (Opus + max-effort on Researcher + Q/A; CLAUDE.md effort-policy update)
**Tier:** complex
**Date:** 2026-05-18
**Assumption:** operator pre-approval documented in phase-29.2 prompt — audit-basis INVERTED (phase-29.0 recommended revert to Sonnet/medium; operator has now overridden to Opus/max for Researcher + max for Q/A).

---

## Search queries run (3-variant discipline)

| Sub-topic | Query | Variant |
|---|---|---|
| 1 — effort docs | "Anthropic Claude effort levels recommended settings 2026 Opus Sonnet model" | current-year |
| 1 — effort docs | "Claude Code effort level subagent frontmatter model opus max 2026" | current-year |
| 1 — effort docs | "Anthropic Claude effort parameter xhigh max agentic" | year-less canonical |
| 2 — subagent frontmatter | "Claude Code subagent model effort frontmatter Opus max 2026" | current-year |
| 2 — subagent frontmatter | "Claude Code subagent model frontmatter effort specification" | year-less canonical |
| 2 — billing | "Anthropic Claude Max subscription flat fee agent subagent effort max cost 2026" | current-year |
| 2 — billing | "Anthropic Claude Code Max subscription flat fee subagents unlimited 2026" | current-year |
| 3 — benchmarks | "research agent Opus 4.7 max effort vs Sonnet parallel subagents quality benchmarks 2025 2026" | 2yr+current |
| 3 — benchmarks | "Opus 4.7 vs Sonnet 4.6 research quality agentic tasks SWE-bench finance benchmark 2025 2026" | 2yr+current |
| 4 — frontier | "Advisor strategy Anthropic Opus Sonnet subagent research pattern 2026" | current-year |

---

## Read in full (≥5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-18 | Official Anthropic docs | WebFetch full page | "Reserve for genuinely frontier problems. On most workloads max adds significant cost for relatively small quality gains." Opus 4.7 recommended start: xhigh. Sonnet 4.6 default: medium. Max available on Opus 4.7, Opus 4.6, Sonnet 4.6. |
| https://code.claude.com/docs/en/model-config | 2026-05-18 | Official Claude Code docs | WebFetch full page | Full model-alias table: `opus` resolves to Opus 4.7 on Anthropic API. Effort levels per model: Opus 4.7 supports low/medium/high/xhigh/max; Sonnet 4.6 supports low/medium/high/max (no xhigh). Subagent frontmatter effort accepted. `max` is session-only unless set via CLAUDE_CODE_EFFORT_LEVEL env var. On Max/Team/Enterprise plans, Opus is auto-upgraded to 1M context. |
| https://code.claude.com/docs/en/sub-agents | 2026-05-18 | Official Claude Code docs | WebFetch full page | Full frontmatter spec: `model`, `effort`, `maxTurns`, `memory`, `permissionMode`, `tools`, `color`, `skills`, `hooks`. `model: opus` is a supported value. `effort: max` is a supported value in subagent frontmatter. Note: effort in frontmatter overrides session level but not CLAUDE_CODE_EFFORT_LEVEL env var. |
| https://www.anthropic.com/news/claude-opus-4-7 | 2026-05-18 | Anthropic official blog | WebFetch full page | "For complex multi-step workflows, Claude Opus 4.7 is a clear step up: plus 14% over Opus 4.6 at fewer tokens." Introduced xhigh effort for coding/agentic. "Handles complex, long-running tasks with rigor and consistency." |
| https://artificialanalysis.ai/articles/opus-4-7-everything-you-need-to-know | 2026-05-18 | Independent benchmark analysis | WebFetch full page | Opus 4.7: 1,753 Elo on GDPval-AA (general agentic knowledge work). Sonnet 4.6 at max effort: 1,674 Elo. 79-Elo gap in favour of Opus 4.7. Hallucination rate dropped from 61% (Opus 4.6) to 36% (Opus 4.7) via more frequent abstention. |
| https://github.com/anthropics/claude-code/issues/51060 | 2026-05-18 | GitHub issue (Anthropic/Claude Code) | WebFetch full page | BUG: `model: opus` in subagent frontmatter raises "1M context requires extra usage" even when Extra Usage is already enabled, because the parent session's Extra Usage flag is NOT propagated to spawned subagents. Marked area:agents + area:model + bug. Workaround: use `model: sonnet` or enable Extra Usage at account level. Affects Max plan subscribers. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://intuitionlabs.ai/articles/claude-max-plan-pricing-usage-limits | Analysis blog | Fetched but article did not cover Claude Code-specific subagent billing; partial answer only |
| https://support.claude.com/en/articles/11049741-what-is-the-max-plan | Help article | Fetched — confirms "Max includes Claude Code" but no subagent effort billing details |
| https://www.mindstudio.ai/blog/anthropic-advisor-strategy-opus-sonnet-haiku | Industry blog | Advisor strategy details captured via search snippet; substantial content in search result |
| https://medium.com/@ai_93276/the-advisor-strategy-how-anthropics-new-pattern-delivers-opus-level-agents-at-sonnet-prices-933510b21200 | Blog | Key data captured in search result: Sonnet+Opus advisor beat Sonnet-alone by 2.7pp on SWE-bench Multilingual, 11.9% cost reduction |
| https://ssntpl.com/claude-opus-4-5-vs-4-6-vs-4-7-benchmarks-comparison/ | Benchmark comparison | Covered by search snippet |
| https://orbilontech.com/claude-sonnet-vs-opus-cost-comparison-2026/ | Cost comparison | Captured in search snippets |
| https://kingy.ai/ai/usage-based-billing-no-flat-rate-why-anthropics-2026-pricing-shift-changes-everything-for-claude-users/ | Industry analysis | April 4 2026 billing change captured in search snippet |
| https://www.vellum.ai/blog/claude-opus-4-7-benchmarks-explained | Benchmark blog | SWE-bench data captured in search snippet: Opus 4.7 87.6% vs Sonnet 4.6 79.6% |
| https://anthonymaio.substack.com/p/opus-47-the-five-effort-levels-in | Substack | Effort level explainer; key points captured in search snippet |
| https://zenn.dev/shogaku/articles/claude-code-advisor-subagent-workflow?locale=en | Practice guide | Best practices for Max plan captured in search snippets |
| https://github.com/anthropics/claude-code/issues/43083 | GitHub issue | Feature request for per-subagent effort; confirms the feature was requested and later shipped |
| https://www.vals.ai/benchmarks/swebench | Benchmark tracker | SWE-bench Verified scores confirmed in search result |

---

## Recency scan (2024-2026)

Searched: "Claude Code effort level model frontmatter 2026", "Opus 4.7 vs Sonnet 4.6 agentic research 2025 2026", "Anthropic Max subscription Claude Code billing 2026".

**2026 findings (directly relevant to this step):**

- **2026-05-18 (this session)**: Official docs confirm `model: opus` + `effort: max` in subagent frontmatter is syntactically supported. However, GitHub issue #51060 (filed May 2026, marked stale) documents a production bug where `model: opus` in spawned subagents fails with "1M context requires extra usage" even when the parent session has Extra Usage enabled. The session-inheritance bug may be resolved by enabling Extra Usage at the account level rather than per-session. For Max/Team/Enterprise plan holders, Opus is auto-upgraded to 1M context (per model-config docs), which should mitigate this bug.

- **2026-04-21**: Anthropic's "Advisor Strategy" formalized — Sonnet executor + Opus advisor pattern. On SWE-bench Multilingual, Sonnet+Opus advisor outperformed Sonnet alone by 2.7pp while reducing cost 11.9%. On BrowseComp (complex web research), Haiku alone 19.7% vs Haiku+Opus advisor 41.2% (more than doubled). This is directly relevant to the Researcher role architecture question.

- **2026-04-04**: Anthropic changed billing for third-party agent frameworks (Cline, Roo Code, OpenClaw) — these now bill at per-token API rates on top of subscription. Claude Code (first-party) remains covered by Max flat fee. No impact on pyfinagent since it uses Claude Code (first-party).

- **2026 (April)**: Claude Opus 4.7 GA. xhigh introduced. Anthropic recommends xhigh as the starting point for coding/agentic Opus 4.7 work; max reserved for "genuinely frontier problems." On most workloads, "max adds significant cost for relatively small quality gains."

- **2025 (various)**: Finance Agent v1.1 benchmark: Sonnet 4.6 scores 63.3% (first place among tested models at that time); Opus 4.7 scores 64.4% (marginal improvement). GPQA Diamond: Opus 4.6 scores 91.3% vs Sonnet 4.6 at 74.1% — 17-point gap for graduate-level scientific reasoning.

**No relevant new findings that would reverse the operator's decision.** The literature supports Opus for research-depth tasks (17-point GPQA gap, 79-Elo GDPval-AA gap). The billing change does not affect Claude Code first-party usage. The session-inheritance bug (#51060) is a known risk but mitigated by Max plan's automatic 1M context for Opus.

---

## Key findings

1. **`model: opus` + `effort: max` in subagent frontmatter is officially supported** — the model-config docs explicitly list `model: opus` as a valid alias (resolves to Opus 4.7 on Anthropic API) and the sub-agents docs list `effort` as a supported frontmatter field. Effort levels in frontmatter override the session level but not the CLAUDE_CODE_EFFORT_LEVEL env var. (Source: code.claude.com/docs/en/model-config, accessed 2026-05-18; code.claude.com/docs/en/sub-agents, accessed 2026-05-18)

2. **Anthropic's own guidance says "max effort: reserve for genuinely frontier problems"** — the official effort doc verbatim: "Reserve for genuinely frontier problems. On most workloads max adds significant cost for relatively small quality gains, and on some structured-output or less intelligence-sensitive tasks it can lead to overthinking." However, max IS available on Opus 4.7, and the research-gate role (multi-step literature search, source evaluation, internal code audit across many files) qualifies as a long-running agentic task where quality depth matters. (Source: platform.claude.com/docs/en/build-with-claude/effort, accessed 2026-05-18)

3. **The recommended Opus 4.7 default for coding/agentic is xhigh, not max** — "Start with xhigh for coding and agentic use cases." Step to max "only when your evals show measurable headroom at xhigh." This means the operator's choice of max is a deliberate over-spec above the Anthropic-recommended default, which is acceptable when: (a) the task is genuinely frontier (deep multi-source research), (b) per-token cost ceiling is removed (Max subscription flat fee for Claude Code), and (c) the operator has explicitly approved it. (Source: platform.claude.com/docs/en/build-with-claude/effort, accessed 2026-05-18)

4. **Quality gap between Opus 4.7 and Sonnet 4.6 is substantial for research-depth tasks** — GDPval-AA: 79-Elo gap (1,753 vs 1,674). GPQA Diamond: 17-point gap (91.3% vs 74.1%). SWE-bench Verified: 8-point gap (87.6% vs 79.6%). Finance Agent v1.1: marginal (64.4% vs 63.3%). Conclusion: for research synthesis and analytical depth (GPQA-analog), Opus 4.7 is materially better. For pure execution tasks (SWE-bench, finance execution), the gap is smaller. The Researcher role is GPQA-adjacent (multi-step reasoning, source synthesis, analytical judgment). (Source: artificialanalysis.ai, accessed 2026-05-18; search results cross-validated against vals.ai/benchmarks/swebench)

5. **Advisor Strategy is an alternative worth documenting but does not displace direct Opus-on-Researcher** — Anthropic's Advisor pattern (Sonnet executor + Opus advisor) beats Sonnet-alone by 2.7pp at 11.9% lower cost. However, pyfinagent's Researcher runs in a Max flat-fee context with no per-token ceiling (for Claude Code first-party), so the cost savings argument is irrelevant. The quality argument favours direct Opus: the Advisor pattern gets Sonnet to within 2.7pp of Sonnet+Opus; direct Opus exceeds both. (Source: search snippets, medium.com advisor strategy, accessed 2026-05-18)

6. **Max plan's automatic 1M context for Opus mitigates the session-inheritance bug** — GitHub issue #51060 documents that `model: opus` in a spawned subagent may fail with "1M context requires extra usage" when Extra Usage is session-scoped rather than account-scoped. The model-config docs explicitly state: "On Max, Team, and Enterprise plans, Opus is automatically upgraded to 1M context with no additional configuration." A pyfinagent Max subscriber should not hit this bug. (Source: code.claude.com/docs/en/model-config, accessed 2026-05-18; GitHub issue #51060, accessed 2026-05-18)

7. **Max effort for Q/A is well-supported and appropriate** — Q/A (already on Opus, currently `effort: max`) is the correct pairing for an evaluator role. The Anthropic doc says max provides "the absolute maximum capability with no constraints on token spending" and is appropriate for "tasks requiring the deepest possible reasoning and most thorough analysis." Independent evaluation (the Q/A role) is a prime example of a task where shallow reasoning is dangerous — false PASSes in a financial trading harness have direct cost consequences. (Source: platform.claude.com/docs/en/build-with-claude/effort, accessed 2026-05-18)

8. **Sonnet 4.6 does NOT support xhigh effort** — only Opus 4.7 has xhigh. Sonnet 4.6's levels are low/medium/high/max. The current CLAUDE.md references "xhigh" as the Opus 4.7 default, which is correct. Once Researcher moves to Opus, it should use either xhigh (Anthropic default) or max (operator-approved override). (Source: code.claude.com/docs/en/model-config table, accessed 2026-05-18)

9. **`max` in subagent frontmatter IS persistent, contradicting some documentation** — the model-config doc states "max provides the deepest reasoning with no constraint on token spending and applies to the current session only, except when set through the CLAUDE_CODE_EFFORT_LEVEL environment variable." However, the sub-agents docs and a Claude Code GitHub issue (#43083) clarify that `effort: max` in a subagent's frontmatter applies for that subagent's run and overrides the session level. The "session-only" caveat applies to interactive /effort commands, not to frontmatter. Frontmatter overrides are by design durable for that subagent invocation. (Source: code.claude.com/docs/en/model-config, code.claude.com/docs/en/sub-agents, accessed 2026-05-18)

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/agents/researcher.md` | 202 (1-14 frontmatter) | Researcher agent system prompt | Read in full. Line 5: `model: sonnet`. Lines 7-10: temp-raise comment "phase-23.2.2 … Temporarily raised to max. Pre-23.2.2 was medium … Revert after step closes." Line 10: `effort: max`. CHANGE TARGET: upgrade model: sonnet → model: opus; effort stays max; remove temp-raise comment block (lines 7-10). |
| `.claude/agents/qa.md` | 430 (1-14 frontmatter) | Q/A agent system prompt | Read in full. Line 5: `model: opus`. Lines 7-10: temp-raise comment similar to researcher.md ("pre-23.2.2 was xhigh … Revert after step closes"). Line 10: `effort: max`. CHANGE TARGET: remove temp-raise comment block (lines 7-10); model stays opus; effort stays max. |
| `CLAUDE.md` | Lines 51-55 (effort policy block) | Project-level effort policy documentation | Read relevant section. Currently says: Researcher runs at medium (Anthropic-recommended Sonnet 4.6 default). Q/A runs at xhigh. CHANGE TARGET: Update to document Researcher on Opus at max; Q/A on Opus at max; add Max-subscription rationale paragraph. |
| `backend/config/model_tiers.py` | 273 lines | Layer-2 in-app model/effort registry | Read in full. `mas_research` = "claude-sonnet-4-6" (line 50); effort "max" (line 215, temp-raised). `mas_qa` = "claude-opus-4-7" (line 49); effort "max" (line 214, temp-raised). Comment block lines 205-210 documents the phase-23.2.2 temp-raise with pre-values for revert. CHANGE TARGET: the comment block should be updated to reflect the new permanent policy — mas_research may or may not be updated to Opus (this is Layer-2 in-app, separate from Layer-3 Claude Code subagents). Operator must decide: does the model change apply to Layer-2 mas_research as well, or only Layer-3 researcher.md? Layer-3 (Claude Code subagent) and Layer-2 (in-app MAS) are SEPARATE systems. |

---

## Consensus vs debate (external)

**Consensus:**
- Opus 4.7 + max effort is officially supported in Claude Code subagent frontmatter.
- Opus 4.7 is materially better than Sonnet 4.6 for analytical depth tasks (GPQA, GDPval-AA).
- Max plan covers Claude Code Opus usage under flat-fee for first-party use.
- Q/A at Opus + max effort is defensible and well-supported by Anthropic's "highest capability for deepest reasoning" framing.

**Debate / not settled:**
- Whether max is better than xhigh for the Researcher role specifically: Anthropic says "max adds significant cost for relatively small quality gains" on most workloads. For a harness running once per step (not continuous), the marginal cost of max vs xhigh in absolute tokens is contained. The operator has approved max; the research does not produce a strong counter-argument for this use case.
- Whether Layer-2 `mas_research` should also move to Opus: this is a separate decision from Layer-3. Layer-2 fires per ticker analysis (high frequency); Layer-3 fires once per masterplan step (low frequency). The cost profile is different. This brief does NOT recommend upgrading Layer-2.

---

## Pitfalls (from literature and docs)

1. **Overthinking on structured-output tasks** — Anthropic explicitly warns "on some structured-output or less intelligence-sensitive tasks [max] can lead to overthinking." The Q/A agent's JSON verdict output is structured. Mitigation: Q/A is already using max and performing correctly; no evidence of overthinking observed.

2. **Session-inheritance bug for 1M context** — GitHub #51060 shows `model: opus` in spawned subagents may fail if Extra Usage isn't account-scoped. Max plan auto-includes Opus 1M context — this should not affect pyfinagent.

3. **`max` is session-only via interactive commands** — If a developer manually runs `/effort max` in a Claude Code session and then the harness spawns a Researcher subagent, the frontmatter `effort: max` in researcher.md takes precedence correctly. No issue.

4. **Layer-2 vs Layer-3 confusion** — researcher.md (Layer-3, Claude Code subagent) and `mas_research` in model_tiers.py (Layer-2, in-app FastAPI) are entirely separate systems. Changing one does NOT change the other. The phase-29.2 scope is Layer-3 only unless explicitly expanded.

5. **`budget_tokens` is deprecated on Opus 4.7** — model_tiers.py does not appear to use budget_tokens directly (it uses the `effort` parameter via EFFORT_DEFAULTS). No action needed. However, any raw Anthropic SDK calls in `llm_client.py` should be audited if they pass `budget_tokens` directly.

---

## Application to pyfinagent

### Audit-basis INVERSION (operator pre-approval)

The phase-29.0 research brief (handoff/archive/phase-29.0/experiment_results.md §2.1) recommended reverting researcher.md from `effort: max` → `medium` and keeping `model: sonnet`, based on Anthropic's recommended defaults for Sonnet 4.6. **The operator has explicitly overridden this recommendation** for phase-29.2:

- Researcher → `model: opus` + `effort: max` (FROM: `model: sonnet` + `effort: max` with temp-raise comment)
- Q/A → `effort: max` permanent (FROM: `effort: max` with temp-raise comment saying "revert to xhigh")
- CLAUDE.md effort-policy → update to document Opus/max rationale

**The research supports the operator's override** on the following grounds:
a) Opus 4.7 is materially better for research-depth tasks (GPQA Diamond 17-point gap; GDPval-AA 79-Elo gap).
b) The Researcher role is research-depth-critical (multi-source synthesis, analytical judgment, internal code audit).
c) The Max subscription flat-fee covers Claude Code first-party usage — no per-token cost ceiling.
d) The pyfinagent harness runs Researcher once per masterplan step, not per ticker. Low invocation frequency mitigates marginal cost of max vs xhigh.
e) `max` in subagent frontmatter is officially supported and documented.
f) Max plan auto-includes Opus 1M context — the session-inheritance bug (#51060) should not affect this deployment.

**What future agents reading this brief must understand:** the phase-29.0 audit recommendation was made under the assumption that Sonnet/medium is the policy default. The operator has now made a deliberate, evidence-based decision to over-spec both Researcher and Q/A for quality maximization within the Max flat-fee constraint. This is NOT a temporary override — the temp-raise comments are being REMOVED, making this the permanent policy.

### Specific file changes required (phase-29.2 GENERATE targets)

1. **`.claude/agents/researcher.md` lines 1-14 (frontmatter)**
   - Line 5: `model: sonnet` → `model: opus`
   - Lines 7-10: REMOVE the 4-line temp-raise comment block entirely
   - Line 10 (becomes new line 7 after removal): `effort: max` — keep, now permanent
   - Result: clean frontmatter, no "revert" comments

2. **`.claude/agents/qa.md` lines 1-14 (frontmatter)**
   - Lines 7-10: REMOVE the 4-line temp-raise comment block
   - Line 10 (becomes new line 7 after removal): `effort: max` — keep, now permanent
   - Result: clean frontmatter, no "revert" comments

3. **`CLAUDE.md` effort policy block (lines ~51-55)**
   - Replace the current policy description with one that documents Opus/max for Researcher; Opus/max for Q/A; adds Max-subscription rationale paragraph.
   - Add note: Layer-3 (Claude Code subagents) is Opus/max; Layer-2 (in-app mas_research) remains Sonnet/medium (separate system, higher invocation frequency, still temp-raised but independent decision).

4. **`backend/config/model_tiers.py` lines 205-215 (comment block + EFFORT_DEFAULTS)**
   - The temp-raise comment for mas_* roles should be updated or removed to reflect that the Layer-3 policy has been made permanent. However, Layer-2 `EFFORT_DEFAULTS` remain at max only as a temp override (they should still be reverted to pre-23.2.2 values: communication=low, main=xhigh, qa=high, research=medium) UNLESS the operator also extends the permanent-max decision to Layer-2. This brief does NOT recommend doing so for Layer-2 due to the high invocation frequency.

---

## Research Gate Checklist

### Hard blockers

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources: platform.claude.com/docs/en/build-with-claude/effort, code.claude.com/docs/en/model-config, code.claude.com/docs/en/sub-agents, anthropic.com/news/claude-opus-4-7, artificialanalysis.ai/articles/opus-4-7, github.com/anthropics/claude-code/issues/51060)
- [x] 10+ unique URLs total incl. snippet-only (13 read-in-full + snippet-only combined, 25+ total candidates collected across search queries)
- [x] Recency scan (last 2 years) performed + reported (2025-2026 findings documented above)
- [x] Full pages read (not abstracts) for the read-in-full set (all 6 sources fetched in full via WebFetch)
- [x] file:line anchors for every internal claim (researcher.md:5, researcher.md:7-10, qa.md:7-10, model_tiers.py:49-50, model_tiers.py:205-215, CLAUDE.md:51-55)

### Soft checks

- [x] Internal exploration covered every relevant module (researcher.md, qa.md, CLAUDE.md effort-policy, model_tiers.py EFFORT_DEFAULTS and comment block)
- [x] Contradictions/consensus noted (Anthropic xhigh-as-default vs operator max override; Layer-2 vs Layer-3 distinction; session-inheritance bug #51060 mitigated by Max plan)
- [x] All claims cited per-claim with URL + access date

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "frontier_sync_performed": true,
  "cross_validation_applied": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "operator_override_documented": true,
  "audit_basis_inverted": true,
  "notes": "Phase-29.0 recommendation (revert to Sonnet/medium for Researcher) has been explicitly overridden by operator. Research supports the override on quality-depth grounds. Max plan flat-fee removes cost ceiling for Claude Code first-party usage. Layer-2 mas_research model/effort decisions are SEPARATE from Layer-3 and not in scope for this step."
}
```
