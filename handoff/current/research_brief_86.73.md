# Research Brief — step 86.73

**Topic:** Design research for pyfinagent's own research gate.
Q1: how should research DEPTH scale with task DIFFICULTY — raise floors on ONE
researcher (a) vs fork 2-3 PARALLEL researcher subagents (b)?
Q2: WHO should assess task difficulty — caller-declared tier vs researcher
self-assessment vs hybrid; and the de-escalation failure mode.

**Tier:** complex (caller-declared). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` is not a gate requirement for this step).
**Accessed:** 2026-08-13.
**Constraints honoured:** research + report ONLY. No edits to
`research-gate.js`, `researcher.md`, or any rule file. No floor lowered
anywhere in this brief; every proposal ADDS depth.

---

## STATUS ENVELOPE (born inert — phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "complex",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 25,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "gate_passed": true
}
```

---

## Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| Year-less canonical | `Anthropic multi-agent research system parallel subagents token usage` |
| Year-less canonical | `why do multi-agent LLM systems fail taxonomy MAST underperform single agent` |
| Year-less canonical | `LLM agent premature termination underestimate task effort sandbagging satisficing stop early guard` |
| Last-2-year window | `LLM agent self-assessment task difficulty calibration overconfidence 2025` |
| Current-year frontier | `parallel research agents deduplication coverage redundancy deep research 2026` |

---

## Read in full (>=5 required; counts toward the gate) — 10 sources

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-08-13 | Official vendor engineering blog | WebFetch, full page | Orchestrator-worker fan-out beat single-agent Opus 4 by **90.2%**; costs **~15x chat tokens** (single agents ~4x); **"token usage by itself explains 80% of the variance"**; scaling rule (1 agent / 2-4 subagents / >10 subagents); subagent duplication is a NAMED observed failure. |
| 2 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-13 | Official vendor engineering blog | WebFetch, full page | Hard-threshold gate: "Each criterion had a hard threshold, and if any one fell below it, the sprint failed"; file-based handoffs; separation of doer and judge; "every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing". |
| 3 | https://arxiv.org/html/2506.00582 | 2026-08-13 | Peer-reviewed (ACL Findings 2025) | WebFetch, arXiv native HTML | **Core Q2 evidence.** "Models' confidence scores are comparatively insensitive to task difficulty and exhibit only a weak correlation with actual accuracy, unlike human patterns." Vanilla ECE ~33.2% avg, **43.4% on expert-level**; AFCE (elicit confidence BEFORE and SEPARATELY from the answer) cuts ECE **58.4%**. |
| 4 | https://arxiv.org/html/2408.03314v1 | 2026-08-13 | Peer-reviewed preprint (Snell et al.) | WebFetch, arXiv native HTML (the `/abs/` page returned the ABSTRACT ONLY and is **not** counted) | **Answers Q1 and Q2 jointly.** "easier questions benefit more from sequential revisions, whereas on difficult questions it is optimal to strike a balance between sequential and parallel computation". Difficulty-adaptive allocation beats fixed best-of-N at **up to 4x less compute**. **MODEL-PREDICTED difficulty bins "largely overlap" with ORACLE bins.** |
| 5 | https://arxiv.org/html/2503.13657v2 | 2026-08-13 | Peer-reviewed (MAST) | WebFetch, arXiv native HTML | **[ADVERSARIAL to Q1b]** "their performance gains across popular benchmarks often remain minimal compared to single-agent frameworks". Specification **41.77%** / Inter-agent misalignment **36.94%** / Task verification **21.30%**. **FM-1.3 Step repetition = 17.14%, the single largest failure mode.** |
| 6 | https://arxiv.org/html/2505.17616v2 | 2026-08-13 | Peer-reviewed preprint | WebFetch, arXiv native HTML | Early-exit taxonomy: "Perfect Early-Exit" / **"Too Early"** (Progress Degradation) / "Too Late" (Redundant Steps). **Extrinsic** (external verifier) exit beat **intrinsic** (self-judged); **hybrid preserved performance best**. Explicitly does NOT validate the agent's own stated exit reason. |
| 7 | https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them | 2026-08-13 | Official vendor doc, **pub. 2026-01-23** | WebFetch, full page | **Most recent official Anthropic guidance, and more conservative than #1**: "Start with single-agent systems"; multi-agent "should be reserved for cases where they provide clear benefits that justify the additional cost"; **3-10x tokens vs single-agent**; "the primary benefit of parallelization is thoroughness, not speed"; read-only verification "sidestep[s] the telephone game problem". |
| 8 | https://arxiv.org/html/2601.07264 | 2026-08-13 | Peer-reviewed preprint (2026) | WebFetch, arXiv native HTML | **The confidence dichotomy**: agents using *evidence* tools (web search) "systematically induce severe overconfidence"; *verification* tools (code interpreter) ground it. Search-agent ECE **0.363–0.441**; mean confidence on INCORRECT answers **0.859–0.967**. A literature researcher is an evidence-tool agent — the worst-calibrated class. |
| 9 | https://ar5iv.labs.arxiv.org/html/2310.01798 | 2026-08-13 | Peer-reviewed (Huang et al., ICLR 2024) — **year-less canonical prior art** | WebFetch, ar5iv (pre-Dec-2023 paper) | **[ADVERSARIAL to both]** "After self-correction, the accuracies of all models drop across all benchmarks" (GPT-3.5 CommonSenseQA **75.8% → 41.8%**); model "more likely to modify a correct answer to an incorrect one". And on fan-out: at 9 responses "multi-agent debate significantly underperforms simple self-consistency". |
| 10 | https://arxiv.org/html/2604.24978 | 2026-08-13 | Peer-reviewed preprint (2026) | WebFetch, arXiv native HTML | Names the exact failure set for parallel deep research: **uneven coverage, context explosion, premature stopping**. Duplication is prevented by an explicit **plan DAG**, not by parallelism itself: removing the DAG degraded coverage **4.31 → 4.10** and blew runtime **47 → 222 min**. |

## Identified but snippet-only (context; does NOT count toward gate) — 25

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2408.03314 | Paper abstract page | Fetched, returned **abstract only** — deliberately excluded from the read-in-full count; superseded by the `/html/` fetch (#4) |
| https://arxiv.org/pdf/2510.23458 | Paper (BrowseConf, confidence-guided test-time scaling) | Confirms the hybrid pattern; budget — `/pdf/` is banned as a primary fetch |
| https://arxiv.org/html/2605.23909 | Survey, LLM confidence calibration | Redundant with #3/#8 |
| https://aclanthology.org/2025.findings-acl.1316/ | Venue page for #3 | Same paper, already read via arXiv HTML |
| https://arxiv.org/html/2508.06225v2 | Overconfidence in LLM-as-a-Judge | Judge-side, not researcher-side |
| https://arxiv.org/pdf/2505.20120 | Agents need metacognitive/strategic reasoning | Adjacent; budget |
| https://openreview.net/forum?id=y9UdO5cmHs | NeurIPS overconfidence mitigation | Redundant with #3 |
| https://openreview.net/forum?id=fAjbYBmonr | MAST venue page | Same paper as #5 |
| https://arxiv.org/abs/2503.13657 | MAST abstract page | Same paper as #5 |
| https://huggingface.co/papers/2601.20975 | DeepSearchQA (comprehensiveness gap, stopping criteria) | Already cited inside `researcher.md`; budget |
| https://arxiv.org/pdf/2606.05241 | Search-time contamination in deep research agents | Off-question (benchmark validity) |
| https://arxiv.org/html/2607.01641v1 | Infinite agentic loops | The opposite failure to de-escalation; context only |
| https://arxiv.org/html/2606.27009v1 | Semantic early-stopping for agent loops | Adjacent to #6 |
| https://arxiv.org/pdf/2605.15206 | AgentStop (early termination to save energy) | Energy-motivated stopping; context |
| https://arxiv.org/pdf/2508.07935 | SHIELDA exception handling | Off-question |
| https://arxiv.org/pdf/2601.12979 | Diffusion LM agentic reality check | Off-question |
| https://arxiv.org/pdf/2504.19678 | LLM reasoning → autonomous agents review | Survey; redundant |
| https://arxiv.org/pdf/2601.22290 | Six Sigma Agent (consensus decomposition) | Industry-flavoured; lower tier |
| https://arxiv.org/pdf/2601.17915 | Graph-guided LLM investigations | Off-question |
| https://parallel.ai/articles/should-you-build-a-web-research-agent-or-use-a-deep-research-api | Vendor article | Vendor-tier; de-dup claim already covered by #10 |
| https://blakecrosley.com/blog/deep-research-agents-evidence-graphs | Personal blog | Community tier |
| https://github.com/VoltAgent/awesome-ai-agent-papers | Link list | Index, not a source |
| https://www.zenml.io/llmops-database/building-a-multi-agent-research-system-for-complex-information-tasks | Secondary summary of #1 | Read the primary instead |
| https://theaiengineer.substack.com/p/how-anthropic-built-multi-agent-deep | Secondary summary of #1 | Read the primary instead |
| https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent | Secondary summary of #1 | Read the primary instead |

**Total unique URLs collected: 35** (10 read in full + 25 snippet-only) vs a floor of 10 and a complex-tier target of 25.

---

## Recency scan (2024–2026) — PERFORMED

Searched the last-2-year window explicitly (`...overconfidence 2025`, `...deep research 2026`). **Result: 5 new findings that materially change the picture, one of which SUPERSEDES a source this project currently relies on.**

1. **Anthropic's own guidance moved against fan-out.** The multi-agent-research post (#1, 2025) is the one quoted in `researcher.md:268-270` to justify the fork option. The **2026-01-23** official post (#7) is later and more conservative: *"Start with single-agent systems"*, multi-agent *"reserved for cases where they provide clear benefits that justify the additional cost"*. Anyone citing #1 for "more than 10 subagents" without #7 is citing the older guidance.
2. **The 3-10x figure (#7, 2026) is the like-for-like cost multiplier** and is more decision-relevant than the 15x-vs-chat framing in #1.
3. **Agent-specific calibration evidence now exists (#8, 2026)** and it is worse than the generic-LLM result: web-search agents are the *most* overconfident tool class.
4. **Parallel deep-research redundancy is now measured (#10, 2026)**: the anti-duplication value comes from an explicit dependency DAG, not from parallelism.
5. **No source found** that measures an LLM agent *deliberately* de-escalating a self-assigned difficulty rating to finish sooner. See "What I could NOT verify".

---

## Key findings

**Q1 — depth scaling.**

1. *Difficulty determines the sequential/parallel mix, and easy→sequential.* "easier questions benefit more from sequential revisions, whereas on difficult questions it is optimal to strike a balance between sequential and parallel computation" (Snell et al. 2024, arxiv.org/html/2408.03314v1).
2. *Depth is bought with tokens, not necessarily with agents.* "token usage by itself explains 80% of the variance" (Anthropic 2025, #1). Raising floors on one researcher raises token usage too — it captures the dominant variance term without paying coordination cost.
3. *Fan-out's own vendor now recommends against it as a default.* "Start with single-agent systems... A well-designed single agent with appropriate tools can accomplish far more than many developers expect" (Anthropic 2026-01-23, #7).
4. *Duplication is the #1 empirical multi-agent failure.* FM-1.3 Step repetition = **17.14%** of all observed failures, the largest single mode (MAST, #5). Anthropic observed it directly: "one subagent explored the 2021 automotive chip crisis while 2 others duplicated work" (#1).
5. *Fan-out multiplies specification risk.* 41.77% of MAS failures are specification issues (#5); Anthropic's stated countermeasure is that each subagent needs "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries" — i.e. the orchestrator must do MORE careful work, not less (#1).
6. *[ADVERSARIAL] At matched compute, multi-agent debate loses.* "multi-agent debate significantly underperforms simple self-consistency" at 9 responses (Huang et al., #9).
7. *De-dup must be structural.* Removing the plan DAG cost coverage (4.31→4.10) and 4.7x the runtime (#10). Parallelism without cross-branch structure is the failure, not the fix.
8. *Anthropic's stated preconditions for fan-out are breadth-first queries, information exceeding a single context window, and heavy parallelization* (#1). A single masterplan step's gate rarely trips any of the three (see Application).

**Q2 — who assesses difficulty.**

9. *Verbalized self-assessed difficulty is poorly calibrated.* "Models' confidence scores are comparatively insensitive to task difficulty and exhibit only a weak correlation with actual accuracy" — vanilla ECE 43.4% on expert-level tasks (#3).
10. *Web-search agents are the worst-calibrated class.* Evidence tools "systematically induce severe overconfidence"; mean confidence on incorrect answers **0.86–0.97** (#8). This is exactly the researcher's tool profile.
11. *BUT model-predicted difficulty CAN work when it is scored, not verbalized.* Snell's predicted-difficulty bins "largely overlap" with oracle bins (#4) — and that estimate comes from a **learned verifier averaged over 2,048 samples**, not from an agent asserting "this is easy". The distinction is the whole design decision.
12. *Intrinsic self-judgment without external feedback degrades results.* Accuracy "drop[s] across all benchmarks"; 75.8%→41.8% worst case; the model is "more likely to modify a correct answer to an incorrect one" (#9).
13. *Extrinsic > intrinsic for stop decisions, hybrid best.* Extrinsic verification beat intrinsic exit instructions, and the hybrid "achieved best performance preservation" (#6).
14. *Asking BEFORE the work fixes much of the miscalibration.* AFCE elicits confidence in a separate, earlier pass and cuts ECE 58.4% (#3). Applied here: a difficulty estimate made *before* the researcher knows what work it would avoid is far less corruptible than one made mid-run.

---

## Internal code inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/workflows/research-gate.js` | `:201` | `const VALID_TIERS = ['simple', 'moderate', 'complex']` | LIVE. `deep` deliberately absent. |
| " | `:202-206` | `tierRequested` / `tierAbsent` / `tierSupported` / `tierUnsupported`; `tier = tierSupported ? tierRequested : 'moderate'` | LIVE. ABSENT defaults to moderate; UNSUPPORTED fails closed at `:384-388` + `:613-618`. |
| " | `:190-200` | The deliberate `deep` exclusion. Verbatim: *"Enabling the tier would ship producer fan-out onto this N=1 artifact rail -- one brief path, one stage-2 verifier, no cross-branch de-dup -- and pre-empt an open operator decision."* | **This is step 86.73's own question, already scoped by a prior cycle.** |
| " | `:213-215` | `FLOOR_SOURCES = 5`, `FLOOR_URLS = 10`, `K_REQUIRED = 2` | LIVE, tier-independent. |
| " | `:285` | Schema pins returned `tier` to `enum: VALID_TIERS` | LIVE — structural only. |
| " | `:364-365` | `enforceGate(env, verification, opts)`; **`const floors = (opts && opts.floors) \|\| {sources: FLOOR_SOURCES, urls: FLOOR_URLS}`** | **LIVE BUT UNUSED SEAM.** `grep -n "floors:"` across `research-gate.js` and `verify_research_gate_workflow.mjs` returns **zero hits** — no caller ever passes a custom floor, so the parameter is present, un-exercised and untested. This is the natural insertion point for per-tier floors. |
| " | `:433-448` | Source/URL/recency/audit-class checks | LIVE. |
| " | `:452-457`, `:465-510` | Over-claim detection + artifact cross-check + `brief_status` hard gate | LIVE. |
| " | `:549-550`, `:764-766` | `self_report_disagreed`; enforced result governs | LIVE — but see gap below. |
| `.claude/agents/researcher.md` | `:204` | *"Caller states the tier in the prompt. Do not choose your own scope."* | LIVE (prose). |
| " | `:206-211` | Tier table — depth/length/URL target vary; **full-read floor is 5 for simple/moderate/complex, 20 for `deep`** | LIVE (prose). |
| " | `:213-283` | `deep` tier: multi-pass, `[ADVERSARIAL]` source, cross-domain triangulation | Documented, **unreachable via the Workflow rail** (`:201` above). |
| " | `:255-270` | "Multi-subagent fork option" — 2-3 parallel deep researchers, ">=20-source floor INDEPENDENTLY", "~1 Claude Max 5-hour rolling window per subagent" | Documented, conditional, unimplemented. |
| " | `:401` | *"Never downgrade a `complex` request to `simple` on your own"* | LIVE (prose) — a one-directional ratchet already exists in doctrine. |
| `.claude/rules/research-gate.md` | `:8-18`, `:20-31`, `:33-60`, `:62-71` | 5-source floor, recency scan, three-variant search, source hierarchy | LIVE. |
| " | `:169-171` | *"The caller marks the step audit-class in the spawn prompt... the researcher never self-declares it to escape the loop."* | **LIVE — the exact anti-de-escalation guard Q2 asks about, already applied to `audit_class`.** |

**GAP FOUND (verified, not inferred).** `grep -n "env\.tier"` on `research-gate.js` returns **zero hits**. `enforceGate` never compares the tier the agent *returned* against the tier the caller *requested*. The schema (`:285`) constrains the returned value to the three valid strings, but a researcher that was asked for `complex` and returns `tier: "simple"` raises **no violation today**. The ratchet exists in prose (`researcher.md:204`, `:401`) and is enforced for `audit_class` by construction (the researcher cannot set it), but for `tier` it is unenforced in code. Any design that gives the researcher a say over difficulty inherits this hole.

---

## Consensus vs debate (external)

**Consensus:** difficulty should govern effort allocation (#4, #1); duplicated work is the dominant parallel-agent failure (#5, #1, #10); an *external* verifier beats self-judgment for stop/quality decisions (#2, #6, #9).

**Debate:** Anthropic 2025 (#1) reports a 90.2% win for fan-out on *breadth-first research*; MAST (#5) and Huang (#9) report minimal-to-negative gains at matched compute on *general* tasks; Anthropic 2026 (#7) resolves the tension by narrowing the recommendation to three preconditions. The 90.2% figure is not transferable to a task that fails those preconditions.

**Debate on Q2:** #4 says model-predicted difficulty ≈ oracle; #3/#8 say verbalized self-confidence is badly calibrated. These are compatible: #4's estimate is a *scored* quantity from an external verifier over 2,048 samples; #3/#8 measure an agent *saying* how confident it is. Do not cite #4 as support for verbalized self-rating.

---

## Pitfalls (from literature)

- **Anchor/homogeneity in a fork.** 2-3 researchers given the same topic and the same tools converge on the same top-10 SERP. #10 shows the de-dup value came from the DAG, not the fan-out.
- **Coordination can exceed the work.** Subagents "spent more tokens on coordination than on actual work" (#7).
- **Merging degrades fidelity.** "telephone game, passing information back and forth with each handoff degrading fidelity" (#7).
- **Self-assessment invited late is self-serving.** #3's AFCE result implies the *timing* of the ask matters more than the wording.
- **A verifier that only checks shape passes bad work.** ChatDev's verifiers did "only superficial checks such as code compilation" and it still scored 33.33% (#5) — relevant because a fork's merged brief would be checked by the same single stage-2 verifier that exists today.

---

## Application to pyfinagent

**Q1 — recommend (a), raise floors on ONE researcher; keep (b) as a caller-gated exception that is NOT enabled by this step.**

Anthropic's three preconditions for fan-out (#1, #7) map poorly onto this gate: (i) *information exceeding a single context window* — the largest brief this rail has produced is 25,359 bytes after 181,082 tokens (86.29), inside a 1M-context Opus researcher; (ii) *heavy parallelization* — one brief, one `brief_path`, one stage-2 verifier (`research-gate.js:190-200`); (iii) *breadth-first with independent directions* — true for some steps, but the caller can say so per step rather than by tier.

Cost, stated plainly: fan-out is **3-10x tokens for equivalent tasks** (#7) on the shared weekly Max pool, and `researcher.md:267` already estimates *"~1 Claude Max 5-hour rolling window per subagent"*. Against the project's own measured distribution — 58.4% of 527 runs are repeats, and the repeat cost is overwhelmingly Q/A (up to 9 runs/step) versus the researcher (max 3) — a 3x multiplier lands on the *minority* cost centre while the majority cost centre is untouched. Depth on one researcher captures the dominant variance term (token usage, 80%, #1) at roughly 1x-2x, not 3-10x, and cannot breach the metered ceiling as fast.

Concretely, (a) is also the cheaper *implementation*: the seam already exists and is unused — `enforceGate`'s `opts.floors` (`research-gate.js:365`) takes per-call floors and has zero callers. Per-tier floors, a peer-reviewed/official-vendor quota, and a mandatory cross-validation section are all ADDITIVE: `simple/moderate` keep 5/10 exactly, `complex` rises. Nothing in this recommendation lowers a floor. Note the untested-seam caveat: because no caller has ever passed `floors:`, a first use needs a mutation cell proving a raised floor actually fails a brief that meets only the old one.

If (b) is ever enabled, the literature says the prerequisite is not more agents but **cross-branch structure**: an explicit allocation of disjoint sub-questions (the DAG in #10; the "clear task boundaries" in #1), a de-dup pass over the merged URL set, and a per-branch source floor — plus a merge verifier, because today's stage-2 reads exactly one file.

**Q2 — recommend a HYBRID with a one-directional ratchet, elicited pre-research.**

Caller-declared stays the FLOOR (unchanged from today, `research-gate.js:202`). The researcher may propose only an ESCALATION, never a de-escalation — which makes the incentive to under-rate structurally unrewarding, since a lower rating cannot reduce its own workload. This is the same construction already used for `audit_class` (`rules/research-gate.md:169-171`) and already asserted in prose for tier (`researcher.md:204`, `:401`); it is *not* currently enforced in code (see GAP above — `env.tier` is never compared to the requested tier).

Timing matters as much as direction: elicit the estimate in the FIRST few tool calls, alongside the born-inert envelope, before the researcher has discovered how much work it would be avoiding — the AFCE result (#3, ECE -58.4%) is precisely an argument for asking early and separately. And keep the final say with the deterministic script, which is the "extrinsic" arm that #6 found beats intrinsic judgment and that `enforceGate` already implements.

Pure self-assessment is not supportable on this evidence: a web-search agent is the worst-calibrated tool class measured (#8, mean confidence 0.86-0.97 when wrong), and intrinsic self-judgment without external feedback degrades outcomes (#9). Pure caller-declared is what exists and is safe but blind — the caller cannot know a topic is thin until someone searches it, which is the honest argument for allowing escalation.

---

## What I could NOT verify

1. **No source measures an agent deliberately DE-ESCALATING a self-assigned difficulty rating to finish sooner.** The nearest evidence is indirect: premature/"Too Early" exits (#6), premature stopping in deep research (#10), and MAST's FM-3.1 premature termination (7.82%) — none of which isolates *effort-avoidance* as the motive. #6 states explicitly that it does not distinguish exiting because the task is done from exiting to minimise compute. **The de-escalation failure mode is plausible and structurally guardable, but it is not empirically documented in what I read.** Treat the ratchet as a cheap precaution, not as a fix for a measured phenomenon.
2. **The 90.2% multi-agent figure is Anthropic-internal**, on their own research eval; no independent replication found.
3. **I did not measure this project's own researcher token cost per tier** — no per-tier cost figures were derived; the 3-10x and 15x multipliers are external.
4. **Snell et al. is math/reasoning benchmarks, not literature research.** The sequential-vs-parallel result transfers by analogy only.
5. **The 58.4%-repeat / max-3-researcher-runs statistics were supplied by the caller as measured fact** and were not re-derived here, per the spawn prompt.
6. **`opts.floors` has never been exercised** (zero callers), so its behaviour under a raised floor is unproven in this codebase.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **10** (7 peer-reviewed, 3 official vendor)
- [x] 10+ unique URLs total — **35**
- [x] Recency scan (last 2 years) performed + reported — 5 findings, incl. one superseding source
- [x] Full papers / pages read (not abstracts) — `arxiv.org/abs/2408.03314` returned an abstract only and was **excluded**; re-fetched via `/html/`
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the declared scope (4 files)
- [x] Contradictions / consensus noted (#1 vs #5/#7/#9)
- [x] All claims cited per-claim
