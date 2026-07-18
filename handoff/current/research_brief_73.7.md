# Research Brief — phase-73.7 (D4 ROLLUP + PUSH)

**Role:** Layer-3 Researcher — COMPLETENESS CRITIC for the phase-73 goal closure.
**Tier:** simple (5-source floor held).
**Status:** IN PROGRESS (write-first; incremental).

## Objective

Find every gap between the phase-73 goal's Definition-of-Done (DoD) and the
actual artifact state BEFORE the 73.7 rollup claims completion. Also confirm
the baseline defect queue dispositions (purge leak, MAS retry bug at
`multi_agent_orchestrator.py:1238`, PBO-cap doc).

## Phase-73 DoD (verbatim from operator goal)

1. `handoff/current/frontier_map_73.md` — per-dimension adopt/reject verdicts
   with citations + gap grades.
2. `handoff/current/design_pack_73/` — design docs a)-d) + pilot verdicts e).
3. Phase-73 steps installed pending + executor-tagged + live_checks; pushed to
   origin/main.
4. Five-file artifacts per step; qa-verdict verdicts transcribed verbatim;
   zero product-code/.env changes by this session.
   PLUS D4 mandate: queue the newly surfaced defects — purge leak (dispositioned
   via 73.1 re-score + 73.1.1), MAS retry bug (multi_agent_orchestrator.py:1238 —
   check whether queued ANYWHERE; if not, queue in 73.7 GENERATE), PBO-cap
   discrepancy doc (via 73.4.2).

---

## INTERNAL AUDIT (main leg)

### A. Masterplan phase-73 step audit (task item 2)

- phase-73 status `in-progress`; 20 steps total.
- Design/rollup steps `done`: 73.0, 73.1, 73.2, 73.3, 73.4, 73.5, 73.6 (7). 73.7 `in-progress` (this step). MATCHES DoD (73.0-73.6 done).
- 12 build steps `pending`, each with executor tag (in `name` `[executor: ...]`), `depends_on_step`, `verification.command`, 3 `success_criteria`, and `verification.live_check`:
  - 73.1.1/.2/.4 `sonnet-4.6/high`; **73.1.3 `opus-4.8/xhigh; METERED PILOT -- requires operator LLM-cost approval BEFORE first spend`** (matches frontier-map R-A Q3 counterfactual-audit metered flag); 73.2.1/.2/.3, 73.3.1/.2, 73.4.1, 73.5.1 `sonnet-4.6/high`; **73.4.2 `sonnet-4.6/high, DOCS-ONLY`** (PBO-cap doc). All 12 present + tagged + live-checked. NO GAP.
- **73.7.1 = "Defect D1 -- MAS retry bug fix" [executor: sonnet-4.6/high]**, status `pending`, phase-73, 3 criteria + live_check + command. So the MAS retry bug IS already queued (see section B). This makes the pending-build-step count effectively 13 when 73.7.1 is included; the DoD's "12 build steps" refers to the D2 design-derived steps (73.1.1..73.5.1); 73.7.1 is the D4-surfaced defect step.

### B. MAS retry bug — PRECISE CHARACTERIZATION (task item 5; forward-referenced by step 73.7.1)

**Queued?** YES — `phase-73` step **73.7.1** ("Defect D1 -- MAS retry bug fix", executor `sonnet-4.6/high`, pending). Its name says the defect is "in `handoff/current/research_brief_73.7.md`" — i.e. THIS brief is the authoritative characterization. Providing it:

**Line-anchor correction (COMPLETENESS FINDING):** the baseline (`frontier_baseline_2026-07-18.md:16`) and step 73.7.1's name both cite `multi_agent_orchestrator.py:1238`. At the baseline commit `e835464b`, **line 1238 is a comment inside the Fable-5 thinking-config branch** (`# phase-67.6: Fable 5 thinking is always on...`), NOT a retry path. The actual retry logic is ~125 lines below. The `:1238` anchor is stale/imprecise; the real defect is at **`multi_agent_orchestrator.py:1363-1394`** (the `stop_reason == "max_tokens"` branch).

**The defect (verified by reading the control flow in full):**
- The agent tool-loop is `for turn in range(max_turns):` (line 1223). At the TOP of every iteration, an unconditional fresh `response = client.messages.create(..., max_tokens=_max_tokens, ...)` runs (line 1269, normal ceiling).
- On a `max_tokens` stop with an incomplete `tool_use` tail (1363-1372), the branch issues a **doubled-budget retry** `client.messages.create(max_tokens=_retry_max=min(_max_tokens*2, 32768), ...)` at 1381, stores it in `response`, bills its usage (1390-1391), then `continue`s (1394).
- The comment at 1392-1393 says "Re-evaluate the retry response on the next iter" — but `continue` advances to the next `turn`, whose FIRST statement (line 1269) **overwrites `response` with a fresh NORMAL-budget call on the SAME `messages`**. So the doubled-budget retry's result is **never evaluated and its completed tool_use is never executed**.

**Consequences:** (1) wasted tokens/$ on a discarded call every truncation; (2) the retry does not fix the truncation — the next turn re-calls with the SAME `messages` at the SAME `_max_tokens`, so it can truncate again; (3) `_mt_retried_turn` is keyed to the OLD `turn` value (`locals().get("_mt_retried_turn") == turn`, line 1371), so after `continue` the single-retry guard no longer matches the new `turn` → repeated wasteful retries up to `max_turns`. It is a **cost/waste + no-op-retry defect**, not a crash — which is why the baseline tagged it "contested." It IS a genuine, reproducible defect; queuing it is correct.

**Correct semantics (for the executor; not over-prescribed):** the doubled-budget retry's result must be CONSUMED, not discarded. Either (a) do NOT `continue` — fall through and re-process the retried `response` in-place (re-check its `stop_reason` / execute its `tool_use` this same iteration), or (b) append the retried assistant turn to `messages` AND carry `_retry_max` into the next call so the fresh iteration actually benefits. A red→green unit test should assert the doubled-budget completion's tool_use is executed (pre-fix: discarded). Scope strictly the retry path; debate architecture is Grade A- and NOT re-scoped (frontier verdict #8).

### C. Five-file protocol + push state + integrity (task items 3, 4, 6)

- **Five-file per closed step:** `handoff/archive/phase-73.{0..6}/` each contains `contract.md + experiment_results.md + evaluator_critique.md + research_brief.md` (4 rolling files; the 5th is the harness_log append). COMPLETE.
- **harness_log Cycles 118-124:** present, one per phase-73.0..73.6, all `result=PASS`. COMPLETE.
- **Verbatim-transcription markers (wf_* run IDs):** every phase-73 evaluator critique cites its qa-verdict Workflow run ID — 73.0 `wf_691df49d`, 73.1 `wf_2ea57de0`, 73.2 `wf_74a88e7d`, 73.3 `wf_10bcde12`, 73.4 `wf_aa2f203d`, 73.5 `wf_daa65ce6`, 73.6 `wf_65b25f78`. Design-pack headers also cite their gate run IDs (a `wf_5da65207`, b `wf_a195f7b3`, c `wf_95688663`, d `wf_f5b30af7`, e `wf_b9308ff4`); design pack c carries the binding Q/A executor note `wf_10bcde12-835`. COMPLETE.
- **Push state:** local HEAD == origin/main (`da017832`), 0 ahead / 0 behind. Commits for 73.0-73.6 all on origin/main (install `9489d8df`, D1 `b6c95879`, 73.1 `d2485c2e`, 73.2 `1b26ab7e`, 73.3 `1275dff1`, 73.4 `64bdd798`, 73.5 `40629a8e`, 73.6 `7e6cb6cd`). The 12 build steps are COMMITTED to origin/main (pending). **73.7.1 is in the working tree only — masterplan.json is ` M` (modified, uncommitted) — NOT yet pushed.**
- **Immutable-criteria integrity:** install snapshot `9489d8df` held the 8 D-steps (73.0-73.7); the 12 build steps + 73.7.1 were appended later. success_criteria + command byte-identical install-vs-HEAD for all install-present steps, AND byte-identical vs first-appearance commit for every appended build step. NO drift.
- **Zero product-code/.env:** `git diff 9489d8df..HEAD --stat -- backend/ frontend/ scripts/` = EMPTY; no `.env` diff. COMPLETE.

### D. DoD scorecard (completeness critic)

| DoD element | State | Gap? |
|---|---|---|
| 1. frontier_map_73.md verdicts+citations+grades | 10 dims, adopt/reject, per-dim `Sources:`, baseline grades (F/D/C/B/A-) | none (1 cosmetic: :165 AlphaAgent KDD'25 "venue confirmation pending" — disclosed MEDIUM caveat) |
| 2. design_pack_73/ a-e | 5 files, consistent, executor-tagged, gate-cited | none |
| 3. steps pending+executor-tagged+live_checks, pushed | 12 build steps COMMITTED+pushed; 73.7.1 in working tree only | **73.7.1 unpushed** (in-flight; rollup must commit+push) |
| 4. five-file/step + verbatim verdicts + zero product-code | archives+cycles+wf markers complete; zero-diff verified | none (73.7's own 5 files in progress — normal pre-contract state) |
| D4. queue 3 defects | purge-leak→73.1+73.1.1; PBO-cap→73.4.2; MAS retry→73.7.1 | MAS-retry step's `:1238` anchor stale (brief corrects to :1363-1394) |

**Rollup MUST-DO before claiming completion:** (1) commit + push the working-tree masterplan (73.7.1) + this brief + 73.7's five files; (2) 73.7's verification command `git log origin/main | grep phase-73` is already green from 73.0-73.6 and does NOT prove 73.7.1 is pushed — do not treat a green command as proof the defect step shipped.

## EXTERNAL RESEARCH — audit-closure / DoD-verification practices

**Query variants (3-variant discipline):** current-year frontier `definition of done verification audit completeness criteria software 2026`; last-2-year `engineering design document review exit criteria standards 2025`; year-less canonical `definition of done checklist agile`; plus recency-scan `completeness audit gap analysis autonomous AI agent verification 2026`.

### Read in full (>=5 required; 6 read)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://swehb.nasa.gov/display/7150/7.9+-+Entrance+and+Exit+Criteria | 2026-07-18 | official doc (NASA SWE Handbook) | WebFetch full | Exit criteria = "Decisions and actions to be completed before the review is considered complete"; open items allowed only if "a timely closure plan exists"; products must be "approved, baselined and placed under configuration management" |
| https://acqnotes.com/acqnote/acquisitions/critical-design-review | 2026-07-18 | official reference (DoD acquisition) | WebFetch full | CDR success = "detailed design satisfies the CDD"; completeness rule-of-thumb "75-90% ... complete and 100% of all safety-critical ... drawings are complete"; report must carry "issues and actions ... together with their closure plans" |
| https://nkdagility.com/resources/definition-of-done/ | 2026-07-18 | authoritative practitioner (Hinshelwood, Scrum.org PST) | WebFetch full | "Done Means Releasable"; undone work "masquerades as progress"; verify via the visceral "would you be happy to release this and support it on call tonight?" |
| https://nulab.com/learn/software-development/definition-of-done-vs-acceptance-criteria/ | 2026-07-18 | practitioner | WebFetch full | DoD = "universal quality checklist that applies to every user story"; AC = "nitty-gritty details specific to a particular user story"; conflation mixes universal quality with story-specific reqs |
| https://plane.so/blog/definition-of-done-dod-checklist-examples-for-agile-teams | 2026-07-18 | practitioner (2026) | WebFetch full | "Replace vague expectations with measurable criteria ... simple, verifiable statements"; anti-patterns: vague criteria ("properly tested") create "interpretation gaps"; "delivery pressure ... mark work complete before it meets defined standards ... false completion signals"; DoD/AC conflation validates function while "quality checks remain incomplete" |
| https://www.future-processing.com/blog/what-is-the-definition-of-done-dod-in-software-development/ | 2026-07-18 | practitioner | WebFetch full | DoD points must be "clear, concise and very specific ... totally unambiguous"; "reduce the risk of prematurely releasing something that is not quite ready"; work failing DoD "carried over to the next sprint" (no false velocity) |

### Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.atlassian.com/agile/project-management/definition-of-done | practitioner | Fetched but body truncated to navigation — NOT counted as read-in-full |
| https://ieeexplore.ieee.org/document/8984435/ | peer-reviewed (IEEE, MBSE model-maturity exit criteria) | Paywalled; abstract via search only |
| https://swehb.nasa.gov ... NID 7123 / CMS PDR / DAU CDR | official docs | Redundant with NASA 7.9 + AcqNotes CDR read in full |
| https://zylos.ai/research/2026-05-01-ai-agent-governance-compliance-2026/ ; labs.cloudsecurityalliance.org/.../ai-agent-governance-framework-gap-20260403 ; beam.ai/agentic-insights/how-to-audit-ai-agents-before-enterprise-security-review | practitioner/industry (2026) | Recency-scan hits; snippet-level (governance-gap-assessment framing) |
| teachingagile.com/... ; invensislearning.com/... ; medium.com/@anderson.buenogod/... ; wrike.com/... ; staragile.com/... | community/practitioner | Lower-tier duplicates of the DoD canon read in full |

### Recency scan (last 2 years — mandatory)
Performed via the 2025 design-review query and the 2026 AI-agent-governance query. **Findings that COMPLEMENT (not supersede) the canon:** (1) 2025 systems-engineering work (IEEE 8984435) is moving static document-based entrance/exit criteria toward **Model-Based SE "model-maturity" exit criteria** — evaluating readiness by artifact maturity rather than a checklist tick; analogue for us: grade DoD elements by artifact state (as this brief's scorecard does), not by a green command. (2) 2026 **AI-agent governance** frameworks (Zylos, CSA governance-gap note, Beam AI audit guide) converge on: "a complete inventory of deployed agents with named owners," "audit trails reconstructable end-to-end," and periodic "governance gap assessments." This is a NEW angle on audit-closure that directly ratifies the pyfinagent pattern — the executor-tagged step inventory (named owner = executor model), the five-file file-based audit trail with wf_* transcription markers, and the completeness-critic gate itself ARE the "named-owner inventory + reconstructable audit trail + gap assessment." No 2024-2026 source overturns the canonical NASA/Scrum discipline that open items need documented closure plans before closure; the newer work reinforces it and adds the maturity-graded and agent-inventory framings.

### Key findings (cited)
1. **Open items may remain open at closure ONLY if a documented closure plan exists and is baselined under configuration management** — "responses made to all ... RIDs, or a timely closure plan exists for those remaining open"; "Products ... approved, baselined and placed under configuration management" (NASA SWE Handbook 7.9). → Maps 1:1 to the D4 defect queue: each surfaced defect needs a queued step (closure plan) AND that step must be committed/pushed (baselined). Purge-leak (73.1.1) + PBO (73.4.2) are baselined/pushed; **MAS-retry (73.7.1) has a closure plan but is NOT yet baselined (uncommitted)** — the exact NASA gap.
2. **A completed-review report must enumerate issues+actions "together with their closure plans"** (AcqNotes CDR). → The rollup must state, per defect, WHERE it landed — which this brief's scorecard does.
3. **"Delivery pressure ... mark work complete before it meets defined standards ... false completion signals"** and **DoD/AC conflation "validates function while quality checks remain incomplete"** (Plane 2026). → The rollup's own `git log | grep phase-73` verification command is green from 73.0-73.6 and is a **false completion signal** for the 73.7.1 push + defect-queue completeness; do not conflate a green command (function) with the completeness gate (quality).
4. **DoD items must be "simple, verifiable statements," never vague** (Plane, Nulab, Future Processing). → The phase-73 build steps satisfy this (concrete `verification.command` + live_check evidence shapes); the audit confirms measurability holds.
5. **"Done Means Releasable"; undone work masquerades as progress** (Hinshelwood). → The completeness-critic's job is to prevent the rollup from logging "done" over an unpushed defect step.

### Internal code inventory (retry-bug characterization)
| File | Lines | Role | Status |
|---|---|---|---|
| backend/agents/multi_agent_orchestrator.py | 1223 | `for turn in range(max_turns)` tool-loop header | live |
| " | 1269-1276 | unconditional per-iteration `create(max_tokens=_max_tokens)` (top of loop body) | live — **overwrites the retried response after `continue`** |
| " | 1237-1264 | thinking-config branch (Fable/Opus/else); **line 1238 = Fable comment, the baseline's stale `:1238` anchor** | live |
| " | 1363-1394 | `stop_reason=="max_tokens"` retry branch: doubled-budget retry (1381) → bill usage (1390-1391) → `continue` (1394) → **result discarded** | live — **THE DEFECT** |

### Application to pyfinagent (external → this rollup)
- The rollup is a **design-review closure** (NASA/CDR analogue), not a build step: 100% of the design spine (73.0-73.6) is done, 100% of build work is queued-not-built (explicitly design+queue-only), and 100% of surfaced defects have closure plans. That matches the CDR "safety-critical 100% / build-to-packages queued" bar.
- The ONE open exit-criterion (NASA "baselined under configuration management") is the **push of 73.7.1 + this brief + 73.7's five files**. Until pushed, the defect queue is planned-but-not-baselined.
- Anti-pattern guard (Plane/Hinshelwood): the rollup must NOT treat its green verification command as DoD proof — the completeness gate (this brief) is the real exit criterion.

### Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: 2 official/reference + 4 practitioner)
- [x] 10+ unique URLs total (~38 collected across 4 searches)
- [x] Recency scan (last 2 years) performed + reported (2025 MBSE exit-criteria + 2026 AI-agent-governance angles)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (multi_agent_orchestrator.py:1223/1269/1238/1363-1394; masterplan step ids; commit SHAs)

## SUMMARY (completeness-critic verdict)

Phase-73 DoD is **substantially met with ONE must-do open exit-criterion + two minor annotations**. DoD 1 (frontier_map), DoD 2 (design_pack a-e), the 12 build steps (pending+executor-tagged+live_checks, committed+pushed), the five-file protocol per closed step (73.0-73.6 archives + Cycles 118-124 + wf_* transcription markers), immutable-criteria integrity (no drift), and zero-product-code/.env are ALL verified complete. The three surfaced defects are ALL dispositioned: purge-leak → 73.1 re-score + 73.1.1 regression test; PBO-cap → 73.4.2 (DOCS-ONLY, two nested gates 0.5 veto / 0.20 promotion); **MAS retry bug → 73.7.1 (queued, pending) — but 73.7.1 is in the working tree ONLY, not yet committed/pushed**, and its inherited `:1238` anchor is stale (the real defect is a discarded doubled-budget retry at multi_agent_orchestrator.py:1363-1394, characterized above). Rollup MUST commit+push 73.7.1 + this brief + 73.7's five files before claiming closure; do not treat the green `git log | grep phase-73` command as proof (false completion signal).

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 14,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Phase-73 DoD substantially met: frontier_map + design_pack a-e complete; 12 build steps pending+executor-tagged+live_checked, committed+pushed; five-file protocol + Cycles 118-124 + wf_* transcription markers + immutable-criteria integrity (no drift) + zero product-code all verified. All 3 defects dispositioned: purge-leak->73.1+73.1.1, PBO-cap->73.4.2, MAS-retry->73.7.1. Open exit-criterion: 73.7.1 is in the working tree only (uncommitted/unpushed) -- rollup must baseline it. Minor: 73.7.1's :1238 anchor is stale; real defect is a discarded doubled-budget retry at multi_agent_orchestrator.py:1363-1394 (brief corrects it).",
  "brief_path": "handoff/current/research_brief_73.7.md",
  "gate_passed": true
}
```
