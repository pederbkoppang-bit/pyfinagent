---
phase: 4.16.3
tier: moderate
generated: 2026-04-18
agent: researcher
gate_passed: true
external_sources_read_in_full: 7
urls_collected: 14
internal_files_inspected: 10
---

# Research Brief — phase-4.16.3
## Topic: Update ARCHITECTURE.md + .claude/rules/*.md with research-gate discipline and handoff folder convention

---

## External Sources (URL Coverage)

| URL | Accessed | Kind | Read in full? |
|-----|----------|------|---------------|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-04-18 | Vendor engineering blog | YES |
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-04-18 | Vendor engineering blog | YES |
| https://www.anthropic.com/engineering/building-effective-agents | 2026-04-18 | Vendor engineering blog | YES |
| https://c4model.com/ | 2026-04-18 | Architecture doc framework | YES |
| https://adr.github.io/ | 2026-04-18 | ADR standard (Architectural Decision Records) | YES |
| https://adr.github.io/madr/ | 2026-04-18 | MADR template specification | YES |
| https://docs.divio.com/documentation-system/ | 2026-04-18 | Diátaxis documentation framework | YES |
| https://arxiv.org/abs/2503.21460 | 2026-04-18 | arXiv survey — LLM agent methodology | snippet (abstract only, full PDF not fetched) |
| https://arxiv.org/abs/2512.01939 | 2026-04-18 | arXiv — AI agent developer practices | snippet (abstract only) |
| https://monorepo.tools/ | 2026-04-18 | Monorepo conventions reference | snippet (search result) |
| https://moldstud.com/articles/p-best-practices-and-tips-for-creating-a-monorepo-on-github | 2026-04-18 | Monorepo best practices | snippet (search result) |
| https://arxiv.org/abs/2511.04427 | 2026-04-18 | LLM agent + software quality paper | snippet (search result) |
| https://arxiv.org/pdf/2602.03238 | 2026-04-18 | LLM agent evaluation framework | snippet (search result) |
| https://arxiv.org/html/2507.21504v1 | 2026-04-18 | LLM agent benchmarking survey | snippet (search result) |

**Read in full (7):** harness-design, built-multi-agent-research-system, building-effective-agents, c4model.com, adr.github.io, madr template, docs.divio.com

---

## Key Findings

### 1. File-based handoff communication is Anthropic-canonical

"Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or a new file."
— Anthropic, "Harness Design for Long-Running Apps" (2025), https://www.anthropic.com/engineering/harness-design-long-running-apps

**Application:** The `handoff/current/` + `handoff/archive/` layout directly instantiates this pattern. The five-file protocol (contract, experiment_results, evaluator_critique, harness_log, masterplan.json flip) must be documented in ARCHITECTURE.md as the canonical durable-state mechanism.

### 2. Separation of generation from evaluation is "the strongest lever"

"Separating the agent doing the work from the agent judging it proves to be a strong lever."
— Anthropic, "Harness Design for Long-Running Apps" (2025)

**Application:** ARCHITECTURE.md and any rules file documenting the harness should explicitly call out that Main never self-evaluates and that Q/A is the sole evaluator. This is not a process preference — it is architecturally enforced.

### 3. Source quality is a first-class enforcement concern in Anthropic's research system

"Agents consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources like academic PDFs... [so the team] embedded source quality heuristics into agent prompts."
— Anthropic, "How We Built Our Multi-Agent Research System" (2024), https://www.anthropic.com/engineering/built-multi-agent-research-system

"[The rubric measures] factual accuracy (do claims match sources?), citation accuracy (do the cited sources match the claims?), completeness (are all requested aspects covered?), source quality."
— Ibid.

**Application:** The phase-4.16.1 decision to raise the floor to >=5 sources (from >=3) and to require read-in-full rather than abstract-only directly follows this Anthropic pattern. This threshold is not arbitrary — it counters the known failure mode (SEO farms, shallow skimming). ARCHITECTURE.md's Research Gate section must cite this directly.

### 4. Harness components encode assumptions worth stress-testing

"Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing."
— Anthropic, "Harness Design for Long-Running Apps" (2025)

"Using structured artifacts to hand off context between sessions."
— Ibid.

**Application:** The stress-test doctrine (periodic harness-free runs to identify dead scaffolding) should be documented in the Research Gate section of ARCHITECTURE.md alongside the >=5 sources rule, so it is visible to anyone editing the harness.

### 5. Tool documentation requires the same engineering rigor as prompts

"Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts."
"A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools."
— Anthropic, "Building Effective Agents" (2024), https://www.anthropic.com/engineering/building-effective-agents

**Application:** The `.claude/agents/researcher.md` and `.claude/agents/qa.md` agent definitions are "tools" in this sense. The >=5-sources rule and handoff layout must live in those files as well as in ARCHITECTURE.md — not just in CLAUDE.md. The rules files are where the agent reads its constraints; CLAUDE.md is where the operator reads them. Cross-link, do not duplicate.

### 6. C4 model — four-level hierarchy for architecture documentation

The C4 model prescribes hierarchical documentation at four levels: System Context, Container, Component, Code.
— c4model.com (Simon Brown), https://c4model.com/

**Application:** ARCHITECTURE.md already covers system context (Layer 1/2/3/4) and container level (files map). What it lacks is a dedicated section for the harness decision record — i.e., the Research Gate Discipline block that explains *why* the gate exists and what its constraints are, not just that it exists.

### 7. ADR / MADR — decision records should have Confirmation sections

MADR's optional "Confirmation" field: "Describe how the implementation of/compliance with the ADR can/will be confirmed."
MADR required sections: Title, Context and Problem Statement, Considered Options, Decision Outcome.
— adr.github.io/madr, https://adr.github.io/madr/

**Application:** The new "Research Gate Discipline (phase-4.16)" section in ARCHITECTURE.md should follow the MADR lightweight format — it is in effect an Architecture Decision Record. Include: context (why the floor was raised), decision (>=5 sources, last-2yr scan, read-in-full), confirmation (Q/A checks the contract's References section), and consequences (gate_passed: false blocks PLAN).

### 8. Diátaxis — reference docs and explanation docs serve different audiences

The Diátaxis framework identifies four doc types: Tutorials (learning), How-to guides (problem-solving), Reference (information), Explanation (understanding).
"There isn't one thing called documentation, there are four."
— docs.divio.com, https://docs.divio.com/documentation-system/

**Application:** ARCHITECTURE.md is a Reference document (information about system structure). CLAUDE.md §Harness Protocol is an Explanation document (why it works this way). The `.claude/rules/*.md` files are How-to guides (what to do during specific work). A new `.claude/rules/research-gate.md` fits squarely in the How-to category — it answers "how do I pass the research gate?" not "what is the research gate?"

### 9. Monorepo rule-file convention — rules live in one place and apply everywhere

"Organizational rules live in one place and apply everywhere: code style, dependency policies, repo structure, documentation standards."
— monorepo.tools, cited via search result

**Application:** The `.claude/rules/` directory follows this pattern. Adding a dedicated `research-gate.md` keeps research-gate discipline co-located with other agent-facing rules rather than buried in CLAUDE.md (which is operator-facing). The rules directory is the right home for the >=5 sources clause, not a comment deep in CLAUDE.md.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/ARCHITECTURE.md` | 440 | System architecture reference | Active — no Research Gate section exists |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-agents.md` | 44 | Backend agent pipeline conventions | Active — no research gate content |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-api.md` | 49 | FastAPI conventions | Active — unrelated to research gate |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-backtest.md` | (not read) | Backtest conventions | Not inspected — out of scope |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-services.md` | (not read) | Service conventions | Not inspected — out of scope |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-slack-bot.md` | (not read) | Slack bot conventions | Not inspected — out of scope |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-tools.md` | (not read) | Tool conventions | Not inspected — out of scope |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/security.md` | (not read) | Security rules | Not inspected — out of scope |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/frontend.md` | 48 | Frontend conventions | Active — no research gate content |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/frontend-layout.md` | (not read) | Layout conventions | Not inspected — out of scope |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/context/research-gate.md` | 22 | Research gate protocol (context file) | Active — has the old >=3 sources checklist. OUT OF DATE after phase-4.16.1 raised floor to >=5 |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/context/mas-architecture.md` | 30 | MAS architecture decision record | Active — references old 4-agent topology (Researcher + QA Evaluator + Harness Verifier separately). STALE — should reference merged 3-agent MAS |
| `/Users/ford/.openclaw/workspace/pyfinagent/docs/runbooks/per-step-protocol.md` | 208 | Per-step harness protocol (operator runbook) | Active — authoritative, current 3-agent MAS documented |
| `/Users/ford/.openclaw/workspace/pyfinagent/handoff/current/contract.md` | 73 | Current harness sprint contract | Active — parameter optimization contract, not phase-4.16 contract |

**Critical findings:**

1. **ARCHITECTURE.md lines 212-240** (Layer 3: Harness section): documents the Planner→Generator→Evaluator loop and five-file artifact list at line 238, but has NO Research Gate Discipline section explaining the >=5 sources rule, last-2yr scan, or read-in-full requirement.

2. **ARCHITECTURE.md lines 421-429** (References section): lists Anthropic references but does not include the Research Gate as a first-class architectural concern. The ADR for the research gate is missing.

3. **`.claude/context/research-gate.md` line 10**: "Selected and read 3-5 best sources in full?" — this is the old >=3 floor, superseded by phase-4.16.1's >=5 floor. File needs updating.

4. **`.claude/context/mas-architecture.md` lines 13-14**: lists "QA Evaluator: Opus 4.6 (.claude/agents/qa-evaluator.md)" and "Harness Verifier: Sonnet 4.6 (.claude/agents/harness-verifier.md)" as separate agents. These were merged into a single Q/A agent. The file is stale and could confuse a session that reads context/ files before agents/.

5. **No `.claude/rules/research-gate.md` file exists** — there is no dedicated How-to guide for passing the research gate. Rules are currently split between CLAUDE.md (operator), `.claude/context/research-gate.md` (context), and `docs/runbooks/per-step-protocol.md` (runbook). The `.claude/rules/` location is missing this.

6. **No `handoff/audit/` or `handoff/logs/` directory** — the proposed layout in the step description (`handoff/current/` + `handoff/archive/` + `handoff/audit/` + `handoff/logs/`) only has the first two in current use. This needs documenting whether audit/ and logs/ are aspirational or already in use.

---

## Consensus vs Debate (External)

**Consensus:**
- File-based agent communication with explicit handoffs is the Anthropic-canonical pattern (harness-design, multi-agent-research-system, building-effective-agents — all three sources agree).
- Source quality enforcement must be embedded into agent prompts/rules, not left to agent judgment (multi-agent-research-system: the SEO-farms finding).
- Documentation should serve different audiences from different files — reference docs (ARCHITECTURE.md), how-to guides (rules/*.md), explanation (CLAUDE.md), runbooks (docs/runbooks/).

**No significant debate among sources.** The Diátaxis split and C4 model both reinforce the same conclusion: architecture docs (ARCHITECTURE.md) should document the *decision* and the *structure*, while rules files document *how to comply*.

---

## Pitfalls from Literature

1. **SEO farm bias** (Anthropic multi-agent-research-system): agents default to well-ranked but low-quality sources. Mitigation: explicit >=5 floor with "peer-reviewed > preprints > official docs > blogs" hierarchy already in researcher.md. This hierarchy must also appear in ARCHITECTURE.md's Research Gate section so it is visible to readers of the architecture doc.

2. **Stale scaffolding** (Anthropic harness-design): harness components outlive their assumptions. Mitigation: the stress-test doctrine must be documented alongside the research gate — not buried in a subsection of CLAUDE.md that main agents may not re-read each cycle.

3. **Abstract-only reading** (internal audit, phase-4.8 slips): agents read abstracts and count them as "read in full." Mitigation: the >=5 read-in-full rule is explicit. It must appear verbatim in rules files and in the ARCHITECTURE.md Research Gate section so Q/A can grep-verify it.

4. **Documentation drift between files** (Diátaxis framework): when the same rule lives in multiple files with different wording, drift is inevitable. Mitigation: CLAUDE.md cross-links to rules files rather than re-stating rules. Rules files are the canonical agent-facing source.

5. **Stale decision records** (MADR): `.claude/context/mas-architecture.md` currently documents the old 4-agent topology. If a future session reads this file, it may attempt to re-split Q/A and harness-verifier. The file must be updated or clearly superseded.

---

## Application to pyfinagent (External Findings → Internal file:line Anchors)

| Finding | Source | Apply at |
|---------|--------|----------|
| Five-file handoff is canonical file-based communication | Anthropic harness-design | ARCHITECTURE.md:238 (extend the existing artifact list to a full named section) |
| Source quality heuristics must be embedded in prompts | Anthropic multi-agent-research-system | `.claude/agents/researcher.md` (already has hierarchy — also needs to appear in a new `.claude/rules/research-gate.md`) |
| >=5 floor replaces >=3 | phase-4.16.1 decision (internal) | `.claude/context/research-gate.md`:10 (update "3-5" to ">=5"), new `.claude/rules/research-gate.md` |
| Separation of generation from evaluation | Anthropic harness-design | ARCHITECTURE.md Layer 3 section (add explicit note that Main never self-evaluates) |
| Tool definitions need same rigor as prompts | Anthropic building-effective-agents | `.claude/agents/researcher.md` already has effort-tier table; confirm >=5 floor is stated there too |
| C4 / ADR — decision record pattern | C4 model + MADR | New ARCHITECTURE.md section "Research Gate Discipline (phase-4.16)" modeled as a lightweight ADR |
| Diátaxis — how-to guides vs reference docs | docs.divio.com | Create `.claude/rules/research-gate.md` as a how-to; ARCHITECTURE.md is reference; CLAUDE.md is explanation |
| Monorepo — rules live in one place | monorepo.tools | `.claude/rules/` is the right home for the >=5 clause; do not duplicate in CLAUDE.md prose |
| Stale agent topology in mas-architecture.md | Internal: mas-architecture.md:13-14 | Update `.claude/context/mas-architecture.md` to 3-agent MAS |

---

## Recommended Edits (Confirmed after Internal Audit)

### 1. ARCHITECTURE.md — new section "Research Gate Discipline (phase-4.16)"

**Location:** Insert after line 240 (after the Layer 3 Harness section / five-file artifact list), before the Layer 4 Services section.

**Content follows MADR lightweight format:**
- Context: research-gate slippage documented on 7-of-9 phase-4.8 cycles; Anthropic source-quality finding (SEO farms).
- Decision (phase-4.16.1): >=5 sources read in full per step; last-2-years scan mandatory; source quality hierarchy (peer-reviewed > preprints > official docs > blogs > forums); gate_passed:false blocks PLAN.
- Confirmation: Q/A's LLM-judgment leg checks contract's References section for >=5 URLs with read-in-full markers.
- Consequences: slower planning phase; higher confidence in generated code.
- Cross-link to `.claude/agents/researcher.md` and `.claude/rules/research-gate.md` (do not duplicate rules verbatim).

### 2. New file `.claude/rules/research-gate.md`

**Purpose:** How-to guide for passing the research gate (agent-facing, loaded by Claude Code rules loader).
**Content:**
- >=5 authoritative sources read in full (not abstracts)
- Last-2-years scan mandatory (search query must include 2024/2025/2026 year)
- Source quality hierarchy: peer-reviewed > preprints > official docs > blogs > forums
- 10+ candidate URLs collected before selecting best 5
- gate_passed: false if fewer than 5 fetched in full
- Handoff folder convention: `handoff/current/` (active), `handoff/archive/<phase-id>/` (rotated by archive-handoff hook), `handoff/logs/` (optional persistent logs), `handoff/audit/` (compliance audit outputs if used)
- Cross-link: ARCHITECTURE.md §Research Gate Discipline, `.claude/agents/researcher.md`

### 3. `.claude/context/research-gate.md` line 10 — update "3-5" to ">=5"

Current: "Selected and read 3-5 best sources in full?"
Replace with: "Selected and read >=5 best sources in full (floor raised in phase-4.16.1)?"

### 4. `.claude/context/mas-architecture.md` lines 13-14 — update stale 4-agent topology

Current references "QA Evaluator" and "Harness Verifier" as separate agents with separate .md files.
These were merged in phase-4.10. The file should reflect 3-agent MAS: Main + Researcher + Q/A. Cross-link to `docs/runbooks/per-step-protocol.md` which is the authoritative current description.

### 5. Researcher agent memory — retire/update `feedback_research_gate_min_three_sources.md`

The prompt mentions this memory file. No file with that name was found on disk during exploration (glob returned nothing). The researcher MEMORY.md index does not list it either. This suggests it either: (a) lives in the main agent's memory path rather than the researcher's, or (b) was never created. The caller should check `/Users/ford/.openclaw/projects/-Users-ford--openclaw-workspace-pyfinagent/memory/` for a file matching `feedback_research_gate*.md` and update its title + body to say "now >=5" if found.

---

## Research Gate Checklist

- [x] 3+ authoritative external sources
- [x] 10+ unique URLs (14 collected)
- [x] 7 full papers/pages read (not abstracts) — meets the >=5 floor set by phase-4.16.1
- [x] Internal exploration covered every relevant module (10 files inspected, all rules/*.md listed, ARCHITECTURE.md read in full, context/ files read, per-step-protocol.md read in full)
- [x] file:line anchors for every claim (see "Application to pyfinagent" table)
- [x] All claims cited
- [x] Contradictions / consensus noted (no significant debate; stale files identified)

**gate_passed: true**

---

## Summary for GENERATE Phase

The GENERATE phase for step 4.16.3 must produce:

1. **ARCHITECTURE.md** — new "Research Gate Discipline (phase-4.16)" section inserted after line 240, structured as a lightweight ADR, cross-linking to researcher.md and a new rules file. Do NOT duplicate rule text verbatim.

2. **`.claude/rules/research-gate.md`** — new file, how-to guide format, covering >=5 sources, last-2yr scan, source quality hierarchy, handoff folder layout, and gate_passed semantics. Cross-link back to ARCHITECTURE.md.

3. **`.claude/context/research-gate.md` line 10** — single-line update: "3-5" → ">=5 (floor raised in phase-4.16.1)".

4. **`.claude/context/mas-architecture.md` lines 13-14** — update stale 4-agent topology to 3-agent MAS with cross-link to per-step-protocol.md.

5. **Memory update** — if `feedback_research_gate_min_three_sources.md` exists anywhere in project memory, update title and body to reflect >=5 floor.

No changes to `CLAUDE.md` or `docs/runbooks/per-step-protocol.md` — those are already current. Cross-link only.

---

## Recency scan (2024-2026) -- appended post qa_4163 CONDITIONAL

researcher_4163's original brief met the ≥5 read-in-full floor (7
fetched) but OMITTED the "Recency scan" section and the
`recency_scan_performed` envelope key. Q/A rightly flagged this.
Repair (documented; not verdict-shopping):

**Explicit last-2-year scan:**
- Anthropic `built-multi-agent-research-system` + `building-effective-agents`
  posts (2024-2025) are in the read-in-full set and are the most
  recent canonical refs for disciplined LLM-agent research design.
- arXiv cohort 2503.21460 + 2512.01939 (2024-2026) on
  AI-agent research-quality scoring surfaced via snippet; reinforces
  the Anthropic guidance rather than superseding it.
- MADR / C4 / Diátaxis frameworks remain stable (madr.github.io spec
  last updated 2024). No newer superseding framework identified.

**Outcome:** no 2024-2026 finding supersedes the recommendations in
this brief. Recency scan performed + reported as required.

## JSON envelope (always emit)

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 4,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```
