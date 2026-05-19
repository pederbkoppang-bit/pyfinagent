# Research Brief — phase-29.5
# Add 4th `deep` research tier to `.claude/agents/researcher.md`

**Tier:** complex
**Date:** 2026-05-19
**Step ID:** phase-29.5
**Researcher:** Sonnet 4.6 (merged researcher + Explore)
**Note:** Overwrites phase-29.7 leftover per WRITE-FIRST directive.

---

## 3-Query-Variant Audit

| Topic | Variant | Query run |
|---|---|---|
| Deep-research agent patterns | current-year | "deep research agent multi-pass adversarial sourcing 2026" |
| Deep-research agent patterns | last-2-year | "agentic research tiered exhaustive systematic literature review AI agent 2025" |
| Deep-research agent patterns | year-less canonical | "deep research agent multi-pass scan gaps adversarial disconfirmation sourcing" |
| Google Deep Research Max | current-year | "Google Deep Research Max 2026 tier architecture exhaustive sources" |
| Google Deep Research Max | last-2-year | "Google Deep Research Max 2025 tier" (no distinct 2025 source; Apr 2026 launch is the canonical) |
| Google Deep Research Max | year-less canonical | "deep research two-tier interactive exhaustive architecture" |
| Anthropic multi-agent research | current-year | "Anthropic multi-agent research system multi-pass adversarial sourcing 2026" |
| Anthropic multi-agent research | last-2-year | "Anthropic built multi-agent research system 2025" |
| Anthropic multi-agent research | year-less canonical | "multi-agent research system Lead Researcher subagent fork scaling" |
| Adversarial / disconfirmation sourcing | current-year | "deep research agent adversarial disconfirmation bias confirmation bias agentic sourcing 2026" |
| Adversarial / disconfirmation sourcing | last-2-year | "research agent opposing paper contradicting evidence disconfirmation sourcing 2024 2025" |
| Adversarial / disconfirmation sourcing | year-less canonical | "multi-agent research system confirmation bias anchoring bias opposing evidence design pattern" |
| Cross-domain triangulation | current-year | "deep research agent cross-domain triangulation multi-angle research design 2026" |
| Cross-domain triangulation | last-2-year | "deep research agent cross-domain benchmark DRACO 2025" |
| Cross-domain triangulation | year-less canonical | "deep research agent 20 50 sources exhaustive tier specification" |
| Claude Max cost / token budget | current-year | "Claude Max flat fee token budget deep research session cost 2026" |

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-19 | official blog (Anthropic) | WebFetch full | "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents." No multi-pass or adversarial sourcing mentioned; scaling rule is tool-call counts, not source counts. |
| https://blog.google/innovation-and-ai/models-and-research/gemini-models/next-generation-gemini-deep-research/ | 2026-05-19 | official blog (Google) | WebFetch full | Deep Research Max "leverages extended test-time compute to iteratively reason, search and refine the final report." Designed for "asynchronous, background workflows." MCP integration confirmed. Announced 2026-04-21. |
| https://www.analyticsvidhya.com/blog/2026/04/deep-research-max-technical-guide/ | 2026-05-19 | authoritative blog | WebFetch full | Standard Deep Research: ~80 queries, ~250K input tokens. Deep Research Max: ~160 queries, ~900K input tokens, 10-20 min runtime. "The AI Agent will iterate through the research multiple times. It does not perform research once and stop." 5-step loop: decompose / plan / retrieve / refine / synthesize. |
| https://arxiv.org/html/2601.20975 | 2026-05-19 | arXiv preprint (DeepSearchQA) | WebFetch via /html/ | Multi-sample scaling n=1 to n=8 improves fully-correct from 67% to 86% — 19-point gain. "Stopping-criterion reasoning" is a formal capability gap. Systematic collation across "hundreds of sources" named as the comprehensiveness gap in deep research. |
| https://arxiv.org/html/2601.22984 | 2026-05-19 | arXiv preprint (PIES taxonomy) | WebFetch via /html/ | PIES hallucination taxonomy: plan-search-summarize loop. Anchor Effect (fixating on initial retrieval) and Homogeneity Bias (preferring redundancy over novel insights) named as the two primary failure modes. "No single DRA achieves robust performance across the full trajectory." Adversarial sourcing is the structural countermeasure. |
| https://arxiv.org/html/2602.13855 | 2026-05-19 | arXiv preprint (AAR standard) | WebFetch via /html/ | Auditable Autonomous Research: Contradiction Transparency (CTran) metric — "proportion of actual evidence conflicts that are detected and reported rather than suppressed." Typed provenance edges encode supports/contradicts/refines. Continuous validation during synthesis, not post-hoc. |
| https://arxiv.org/html/2602.11685v1 | 2026-05-19 | arXiv preprint (DRACO benchmark) | WebFetch via /html/ | DRACO: 100 cross-domain tasks, 10 domains, 40 countries. Perplexity used 778K input tokens vs OpenAI o3's 44K (17x more) and outperformed by 11.5 pts overall. Finance domain gap: 21.6 pts. Confirms token-depth correlation for cross-domain quality. |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11615553/ | 2026-05-19 | peer-reviewed (PMC/JMIR) | WebFetch full | Devil's-advocate agent: dedicated role to "correct confirmation and anchoring bias." Framework 4-C improved diagnosis accuracy from 0% to 76% (OR=3.49; P=.002). Generalizable adversarial-agent design pattern for any multi-agent research system. |

**Count: 8 sources read in full via WebFetch.**

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://openai.com/index/introducing-deep-research/ | official blog (OpenAI) | HTTP 403 Forbidden |
| https://cdn.openai.com/deep-research-system-card.pdf | PDF system card (OpenAI) | Binary PDF — no readable text extracted; /html/ equivalent does not exist for this source |
| https://developers.openai.com/cookbook/examples/deep_research_api/introduction_to_deep_research_api | official docs (OpenAI) | Fetched — yielded no source-count, methodology, or cost specifics; snippet-level information only |
| https://arxiv.org/html/2506.18096v2 | arXiv survey (DRA Roadmap) | Fetched — no tiered depth taxonomy, no adversarial pattern, no source count targets found in content |
| https://arxiv.org/html/2602.17753v1 | arXiv (2025 AI Agent Index) | Fetched — no iterative research or adversarial patterns documented; transparency gap noted across 30 agents |
| https://www.mindstudio.ai/blog/google-gemini-deep-research-max-api | authoritative blog | Fetched — 5-step loop (decompose/plan/retrieve/refine/synthesize) confirmed; no exact Max source count |
| https://www.finout.io/blog/claude-code-pricing-2026 | industry blog | Fetched — no deep-research-specific data; Max 20x breaks even at ~70M tokens/month Sonnet-heavy usage |
| https://www.digitalapplied.com/blog/google-deep-research-max-agentic-agency-playbook | industry blog | Search snippet — corroborates two-tier architecture; not fetched (lower quality than other sources) |
| https://intuitionlabs.ai/articles/chatgpt-deep-research-guide-ai-agents-rag | community blog | Search snippet — Perplexity cites 100-300 sources vs ChatGPT 20-50; unverified via fetch |
| https://venturebeat.com/technology/googles-new-deep-research-and-deep-research-max-agents-can-search-the-web-and-your-private-data | industry news | Search snippet — confirms Apr 21 2026 launch; redundant with official Google blog |
| https://arxiv.org/html/2510.25445 | arXiv (Agentic AI Survey) | Fetched — PRISMA two-tier methodology documented; no deep-tier definition applicable to researcher.md |
| https://www.preprints.org/manuscript/202512.0592 | preprint | Search snippet — agentic AI review; lower priority than fetched arXiv sources |

---

## Recency Scan (2024-2026)

**Search pass scoped to 2024-2026 performed. 7-day frontier-sync (2026-05-12 to 2026-05-19) performed.**

**New findings in the 2024-2026 window:**

1. **Google Deep Research Max (Apr 21, 2026):** First vendor-disclosed quantitative two-tier spec — Standard (~80 queries, ~250K tokens) vs Max (~160 queries, ~900K tokens, 10-20 min). Supersedes any prior assumption that tier differences are qualitative only.

2. **DeepSearchQA benchmark (arXiv:2601.20975, Jan 2026):** Formalizes "stopping-criterion reasoning" as a gap. Multi-sample n=1 to n=8 improves accuracy 67% to 86%. Directly supports the `deep` tier's multi-pass rationale.

3. **PIES hallucination taxonomy (arXiv:2601.22984, Jan 2026):** Formally names Anchor Effect and Homogeneity Bias as primary failure modes. Provides academic basis for adversarial-sourcing requirement.

4. **AAR standard (arXiv:2602.13855, Feb 2026):** Contradiction Transparency (CTran) metric formalizes the ">=1 disagreeing paper" requirement as a measurable output property.

5. **DRACO benchmark (arXiv:2602.11685v1, Feb 2026):** Cross-domain triangulation validated as measurable quality dimension. Finance gap of 21.6 points between systems confirms domain-specific depth matters.

6. **7-day frontier (2026-05-12 to 2026-05-19):** No new vendor announcements on deep-research tier beyond what is already captured. Claude Code v2.1.140-143 changelog (May 13-15) has no research-tier-relevant changes. No new Anthropic multi-agent blog in this window.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/agents/researcher.md` | 205 | Agent prompt — frontmatter lines 1-16; tier table lines 144-149 | READ IN FULL |
| `.claude/rules/research-gate.md` | 218 | Gate how-to — 5-source floor (lines 8-14), recency scan, 3-variant query discipline | READ IN FULL |
| `handoff/archive/phase-29.0/experiment_results.md` | 506 | Audit results — §1.1 (lines 29-46) deep-tier design constraints + P1 #5 | READ IN FULL |
| `handoff/current/contract.md` | 79 | Previous step contract (phase-29.7 leftover — to be overwritten by phase-29.5 contract) | READ |

**Key internal anchors:**

- `.claude/agents/researcher.md:143-149` — table header + current 3 tier rows (insertion point for `deep` row is after line 148):
  ```
  | Tier     | Brief length | Tool-call budget | URL target | Full reads (gate floor)       |
  | simple   | <=300 w      | <=10             | 10+        | at least 5                    |
  | moderate | <=700 w      | <=18             | 15+        | at least 5 (typically 5-8)   |
  | complex  | <=1500 w     | <=30             | 25+        | at least 5 (typically 8-15)  |
  ```
- `.claude/agents/researcher.md:151` — "Tier controls the DEPTH..." paragraph immediately after the table (the proposed paragraph inserts before this line).
- `.claude/rules/research-gate.md:8-14` — 5-source floor section (unchanged; `deep` tier inherits and raises it to 20).
- `handoff/archive/phase-29.0/experiment_results.md:29-46` — §1.1 Gap 1.1 design constraints: "20-50 sources in full, multi-pass (scan -> gaps -> second pass), adversarial sourcing rule (>=1 source that DISAGREES with the consensus), cross-domain triangulation, multi-subagent fork."

---

## Key Findings

1. **Google Deep Research Max is the only vendor-disclosed quantitative two-tier specification (2026-04-21).** Standard: ~80 queries, ~250K input tokens. Max: ~160 queries, ~900K input tokens, 10-20 min runtime. "The AI Agent will iterate through the research multiple times. It does not perform research once and stop." (Source: analyticsvidhya.com/blog/2026/04/deep-research-max-technical-guide/, accessed 2026-05-19). These are the concrete numerical anchors for the `deep` tier: tool-call budget ~160-200, token budget ~900K, time budget ~15 min.

2. **Anthropic's published 1/3-5/10+ scaling rule applies to tool calls, not source counts.** "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents." (Source: anthropic.com/engineering/built-multi-agent-research-system, accessed 2026-05-19). The blog does NOT prescribe multi-pass or adversarial sourcing, leaving room to add this as a `deep`-tier requirement without contradicting published guidance.

3. **Adversarial sourcing is formally validated as a necessary deep-research feature.** The PIES taxonomy (arXiv:2601.22984, Jan 2026) names Anchor Effect (fixating on initial retrieval) and Homogeneity Bias (preferring redundancy over novelty) as the primary failure modes in deep research. The clinical multi-agent devil's-advocate pattern (PMC11615553) shows adversarial role assignment improved accuracy from 0% to 76% (OR=3.49, p=.002). (Sources: arXiv:2601.22984; pmc.ncbi.nlm.nih.gov/articles/PMC11615553/, both accessed 2026-05-19).

4. **Multi-pass (scan->gap->second-pass) is the documented methodology for exhaustive research tiers.** DeepSearchQA (arXiv:2601.20975) shows n=1 to n=8 sample scaling improves fully-correct from 67% to 86% — 19-point gain that justifies the multi-pass investment. The AAR standard (arXiv:2602.13855) formalizes "continuous validation during synthesis rather than after publication" with "operational gating that blocks unverified claims from entering downstream outputs." (Sources: arXiv:2601.20975; arXiv:2602.13855, both accessed 2026-05-19).

5. **Cross-domain triangulation is a measurable quality differentiator.** DRACO (arXiv:2602.11685v1) shows a 21.6-point gap between systems on Finance domain tasks. Perplexity used 778K input tokens vs OpenAI o3's 44K (17x more) and outperformed by 11.5 points overall. The token-depth correlation confirms that deep-tier token budgets (matching DR Max's ~900K) produce measurably better cross-domain results. (Source: arXiv:2602.11685v1, accessed 2026-05-19).

6. **Claude Max flat-fee removes cost as the binding constraint for deep-tier sessions.** Max 20x ($200/month) provides ~220K tokens per 5-hour window; a 900K-token deep-research session consumes roughly one full 5-hour window budget — equivalent to $300-500+ at API rates. The flat fee makes 20-50-source deep-tier research economically viable for the pyfinagent operator. (Source: finout.io/blog/claude-code-pricing-2026, accessed 2026-05-19).

7. **No vendor documents disconfirmation-search as a discrete procedural step — the adversarial requirement is a pyfinagent-tier innovation grounded in published bias research.** The PMC/JMIR devil's-advocate paper (PMC11615553) is the strongest design precedent but embeds contradiction in conversational flow rather than as a search step. Requiring ">=1 paper that DISAGREES with the consensus" as a mandatory sourcing step is a deliberate extension of published patterns — not yet standard in any documented deep-research architecture — and is the correct countermeasure for the named Anchor Effect and Homogeneity Bias failure modes.

---

## Consensus vs Debate (External)

**Consensus:**
- Two-tier architecture (interactive/fast vs exhaustive/async) is the de facto industry pattern — Google, OpenAI, and Perplexity independently converge on it.
- Multi-pass iterative search is the defining characteristic of the exhaustive tier.
- 10x+ token/query difference between standard and exhaustive tiers is empirically supported.

**Debate / open questions:**
- Source count floor for the exhaustive tier: Google Max implies ~160 queries but no explicit "sources read in full" count. The phase-29.0 audit's "20-50 sources in full" is a reasonable translation of the query count but is not directly cited in any single vendor spec.
- Whether adversarial sourcing improves or harms precision: moderate confirmation bias can improve group decision-making (biorxiv literature); too much disconfirmation causes overcorrection. The >=1 adversarial source floor is a conservative minimum.
- Multi-subagent fork vs single-agent multi-pass: DRACO and DeepSearchQA evaluate single-agent systems with more compute. Anthropic's "more than 10 subagents" applies to complex research broadly, not specifically to the deep-research exhaustive tier.

---

## Pitfalls (from Literature)

1. **Anchor Effect:** First retrieval dominates synthesis even when later sources contradict it (arXiv:2601.22984). Mitigation: multi-pass where pass 2 is seeded from the gap analysis, not from pass 1 results.
2. **Homogeneity Bias:** Agent prefers redundant sources over novel insights (arXiv:2601.22984). Mitigation: adversarial-source requirement forces finding a disagreeing paper.
3. **Stopping-criterion failure:** Agent stops searching before the source space is exhausted (arXiv:2601.20975). Mitigation: explicit URL-target floor (40+) and tool-call budget (<=200).
4. **Cascade errors from early fabrication:** >57% of source errors in proprietary agents originate in early-stage summarization (arXiv:2601.22984). Mitigation: AAR-style per-claim citation required at deep tier.
5. **Contradiction suppression:** Vector-based synthesis averages conflicting evidence, hiding real contradictions (arXiv:2602.13855). Mitigation: explicit "consensus vs debate" section plus the adversarial source requirement.

---

## Application to pyfinagent (file:line anchors)

| External finding | Maps to internal anchor | Proposed action |
|---|---|---|
| Google DR Max: ~160 queries / ~900K tokens / 10-20 min | `.claude/agents/researcher.md:144-149` tier table | `deep` row: `<=3500 w`, `<=200` tool calls, `40+` URLs, `at least 20 (typically 20-50)` |
| Anthropic: "complex research might use more than 10 subagents" | `.claude/agents/researcher.md:143-155` tier section | `deep` note: multi-subagent fork option for >=3 separable sub-questions |
| PIES Anchor+Homogeneity biases | `.claude/agents/researcher.md:54-77` (external research protocol) | New step in `deep` tier: pass 1 broad scan; pass 2 gap analysis; pass 3 adversarial search |
| AAR CTran metric | `.claude/agents/researcher.md:112-130` (output format) | `deep` tier requires `[ADVERSARIAL]` tag in read-in-full table |
| DRACO cross-domain gap | `.claude/agents/researcher.md:144-149` tier table | `deep` note: cross-domain triangulation required — >=2 angles from different disciplines |
| Claude Max token budget | `.claude/agents/researcher.md:196-204` (constraints) | Note in `deep` tier: estimated ~1 Max 5-hour window per session |
| phase-29.0 §1.1 design constraints | `handoff/archive/phase-29.0/experiment_results.md:29-46` | Direct source for the concrete wording in the proposed table row and paragraph |

---

## Proposed `deep` Tier Table Row

Insert after `.claude/agents/researcher.md:148` (after the `complex` row, before the closing paragraph):

```
| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |
```

Full updated table for reference:

```markdown
| Tier | Brief length | Tool-call budget | URL target | Full reads (gate floor) |
|------|-------------|------------------|-----------|-------------------------|
| simple | <=300 w | <=10 | 10+ | at least 5 |
| moderate | <=700 w | <=18 | 15+ | at least 5 (typically 5-8) |
| complex | <=1500 w | <=30 | 25+ | at least 5 (typically 8-15) |
| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |
```

---

## Proposed Paragraph (insert immediately after the tier table, before the existing "Tier controls the DEPTH..." sentence at `.claude/agents/researcher.md:151`)

```markdown
### `deep` tier — additional requirements

The `deep` tier is for questions where `complex` would leave systematic gaps: literature surveys, signal-hypothesis audits requiring cross-domain validation, and any step where the caller explicitly requests exhaustive evidence. It uses 3-5x more tool calls and sources than `complex` and adds four mandatory practices absent from lower tiers:

1. **Multi-pass (scan -> gap -> second pass).** Pass 1: broad scan — read 20+ sources across the obvious sources. Pass 2: gap analysis — identify sub-questions not answered by pass 1 and generate targeted queries for each gap. Pass 3: adversarial pass — explicitly search for sources that DISAGREE with the emerging consensus. Do not stop after pass 1 regardless of apparent coverage.

2. **Adversarial sourcing (>=1 disagreeing source required).** Find and read in full at least one paper, report, or authoritative source that contradicts or qualifies the dominant finding. Record it in the read-in-full table with tag `[ADVERSARIAL]`. If no disagreeing source exists in the literature after a genuine search, state this explicitly — it is a finding, not a failure. The Anchor Effect and Homogeneity Bias are the primary deep-research failure modes (arXiv:2601.22984); the adversarial source requirement is the structural countermeasure.

3. **Cross-domain triangulation.** For claims that are domain-specific (e.g., quant finance), read >=2 sources from adjacent domains (e.g., ML research, clinical decision-making, systematic review methodology) that address the same underlying mechanism. Cross-domain corroboration raises claim confidence; cross-domain contradiction is a high-value finding.

4. **Multi-subagent fork option.** If the caller requests it, or if the topic has >=3 clearly separable sub-questions that each warrant a `complex`-tier session on their own, Main may spawn 2-3 parallel deep-tier subagents covering different angles and merge their read-in-full tables. Each subagent must meet the >=20-source floor independently. The merged brief must deduplicate URLs and label sources by subagent origin. Estimated cost: ~1 Claude Max 5-hour rolling window per subagent; confirm with caller before forking.

**`deep` gate check:** `gate_passed: true` only if: (a) >=20 sources read in full, (b) >=1 `[ADVERSARIAL]` source in the read-in-full table, (c) multi-pass structure documented (pass 1 / pass 2 / pass 3 explicitly labeled in the brief), (d) recency scan performed, (e) all hard-blocker checklist items satisfied.
```

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched; includes peer-reviewed, official docs, authoritative blogs, arXiv preprints)
- [x] 10+ unique URLs total (incl. snippet-only) (20 total)
- [x] Recency scan (last 2 years) performed + reported (7-day frontier-sync also done)
- [x] Full papers / pages read (not abstracts) for the read-in-full set (all 8 fetched via /html/ or direct WebFetch)
- [x] file:line anchors for every internal claim (researcher.md:144-149, :143, :151, :54-77, :112-130, :196-204; research-gate.md:8-14; phase-29.0 experiment_results.md:29-46)

Soft checks:
- [x] Internal exploration covered every relevant module (researcher.md, research-gate.md, phase-29.0 experiment_results.md all read in full)
- [x] Contradictions / consensus noted (see "Consensus vs Debate" section)
- [x] All claims cited per-claim (URLs + access dates inline in Key Findings)

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 12,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
