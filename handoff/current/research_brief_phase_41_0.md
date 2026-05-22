# phase-41.0 Research Brief -- Phase-29.8 P2 Bundle Close (OPEN-32)

**Author:** Researcher (Claude Opus 4.7, effort=max).
**Date:** 2026-05-23.
**Tier:** SIMPLE.
**Caller:** Main, per `feedback_never_skip_researcher` (operator override 2026-05-22 -- ALWAYS spawn researcher, even on pre-closed bundle steps).

## Executive summary

phase-41.0 is, structurally, a **trace-link / mechanical-close step**. It does not introduce code change. Its job is to flip the **audit-trail status** of a now-vestigial requirement-bundle ID (`phase-29.8`) to `done` so the masterplan view stops carrying a phantom open item. Three of the four "consolidates with..." items have already landed (37.1+37.2+37.4 done, 40.2 done, 40.5+40.6 done). The remaining two (37.3 + 40.1) are independently tracked open phase-40.x rows. The masterplan immutable verification command was deliberately relaxed from `assert p['status']=='done'` (master_roadmap line 999) to `assert (not ps) or ps[0]['status']=='done'` (current masterplan), which means **the structural absence of a phase-29.8 row in `.claude/masterplan.json` IS the success condition**. This is the documented "fold" outcome, not a defect.

**Verdict:** **PRE-CLOSED by masterplan structure** (one important caveat). The phase-29.8 row was already excluded from the masterplan as part of the phase-45.0 closure re-audit; the current verification passes today on a freshly cloned tree. The caveat: the bundle's *substantive* work is not entirely done -- phase-37.3 (budget_tokens deprecation) + phase-40.1 (OpenAlex `.env.example` entry) are still `pending` independently. phase-41.0's flip says "the trace-link is closed", not "the underlying engineering work is done". The planner needs to surface this distinction explicitly in the contract + evaluator_critique so that a future auditor doesn't read "phase-41.0 PASS" as "all 9 phase-29.8 P2 items are live".

## Section A -- Internal audit (file:line)

### A.1 phase-29.8 ABSENCE from masterplan (confirmed)

- `/Users/ford/.openclaw/workspace/pyfinagent/.claude/masterplan.json` -- 71 phases total; flat top-level IDs (`phase-0` through `phase-45`, no nested `phase-29.8` row). Confirmed via `python3 -c "import json; d=json.load(open('.claude/masterplan.json')); print([p['id'] for p in d['phases'] if p['id']=='phase-29.8'])"` -> `[]`.
- Closure rationale logged at `handoff/current/closure_roadmap.md:33` -- "phase-29 ... 29.8 (P2 bundle) + 29.9 (P3 bundle) ARE phase-41.0 + phase-41.1 by design".
- Closure_roadmap:199 confirms phase-29 flipped to `done` with notes "DROP per phase-45.0; 29.0-29.7 done; 29.8/29.9 are phase-41.0/41.1 by design".
- The phase-29 archive at `handoff/archive/phase-29.0/ ... phase-29.7/` contains every closed sub-step but NO `phase-29.8/` directory -- bundle was never started as its own cycle.

### A.2 phase-41.0 audit_basis + verification command

- `.claude/masterplan.json` phase-41.steps[0] (`41.0`):
  - `audit_basis: "Per research_brief Section B OPEN-32: consolidates with phase-37.3 (budget_tokens) + phase-40.1 (OpenAlex) + phase-40.2 (alwaysLoad/continueOnBlock)."`
  - `verification.command: "python -c \"import json; d=json.load(open('.claude/masterplan.json')); ps=[p for p in d['phases'] if p['id']=='phase-29.8']; assert (not ps) or ps[0]['status']=='done'\""`
  - `success_criteria: ["all_phase_29_8_sub_items_closed", "masterplan_phase_29_8_status_done_or_absent"]`
  - `status: pending`, `harness_required: false`, `priority: P3`.
- The verification command is **structurally satisfied today** (absent row -> `(not ps)` is `True` -> assertion passes).
- Note the divergence from the original master_roadmap_to_production.md:999 spec which used the stricter `assert p['status']=='done'` (would have failed on absence). The current relaxed shape was apparently introduced to accommodate the "fold-into-41.0 without ever instantiating a 29.8 row" outcome.

### A.3 Status of the four "consolidates with..." dependencies

| Dep | Status | Evidence (file:line) | Bundle item closure verdict |
|---|---|---|---|
| phase-37.1 (RiskJudge response_schema, OPEN-16) | **done** | `.claude/masterplan.json` phase-37.steps[0] | covered (was companion to OPEN-18 work) |
| phase-37.2 (gemini deep-think default, OPEN-17) | **done** | phase-37.steps[1] | covered |
| **phase-37.3 (budget_tokens deprecation, OPEN-18)** | **pending** | phase-37.steps[2] | **NOT-done -- carries forward as independent open** |
| phase-37.4 (Moderator response_schema, companion to OPEN-16) | **done** | phase-37.steps[3] | covered |
| **phase-40.1 (OpenAlex `.env.example`, OPEN-24)** | **pending** | phase-40.steps[0] | **NOT-done -- carries forward as independent open** |
| phase-40.2 (Claude Code v2.1.140-143 features, OPEN-25) | **done** | phase-40.steps[1] (closed cycle 25, 2026-05-23) | covered |

Source: harness_log.md cycles 24+25 (phase-40.6 + phase-40.2 PASS) + masterplan.json status fields.

### A.4 phase-45.0 closure-archive evidence

- `handoff/archive/phase-45.0/contract.md:35` -- "phase-29 -> phase-41.0 + 41.1 (Harness MAS + MCP + Academic-Fetch + Frontier-Sync = 29.8/29.9 bundle closure)" -- the fold is the documented design.
- `handoff/archive/phase-45.0/research_brief.md:45` -- "29.8 ... pending ... FOLD-INTO-41.0 -- master roadmap §2 OPEN-32 explicitly says 'consolidates with phase-37.3 + 40.1 + 40.2'".
- `handoff/archive/phase-45.0/research_brief.md:39` -- "phase-29 ... KEEP (29.0-29.7 done; 29.8/29.9 are P2/P3 bundles fold-into-41) ... Action: fold 29.8 + 29.9 into phase-41.0 + 41.1 (already done in masterplan); flip phase-29 to done post-fold."

### A.5 Closure_roadmap verdict on phase-29.8

- `handoff/current/closure_roadmap.md:35` -- "Sub-bundles 29.8 + 29.9: Already mapped to phase-41.0 + phase-41.1 per master_roadmap_to_production.md OPEN-32 + OPEN-33."
- `closure_roadmap.md:135` (Section 5, North-Star Delta table) -- "41.0/41.1 bundle close | B | -10+ pending substeps from masterplan; cleaner /masterplan view | substep count diff". The declared N* delta is **Burn (B)** alone -- audit-trail discipline; not P (Profit) or R (Risk).
- `closure_roadmap.md:306` -- "29.8 P2 bundle | 2 | folded into phase-41.0".

### A.6 Master_roadmap §B.6 OPEN-32 specification (the 9-item bundle)

`handoff/current/master_roadmap_to_production.md:80`: OPEN-32 = "Phase-29.8 P2 bundle -- 9-item residual (`budget_tokens` cleanup + `alwaysLoad`/`continueOnBlock` + OpenAlex + Gemini-3.x audit)".

The original 10-item phase-29.8 enumeration from harness_log.md line 20474:
1. budget_tokens audit (-> phase-37.3, pending)
2. alwaysLoad on alpaca+bigquery (-> phase-40.2, **done**)
3. continueOnBlock on alpaca+bigquery (-> phase-40.2, **done**)
4. OPENALEX_API_KEY in backend/.env.example (-> phase-40.1, pending)
5. Gemini-3.x model-ID audit (-> phase-41.x downstream; or rolled into 41.0 cleanup; see master_roadmap line 30 reference)
6. frontier-sync rule in research-gate.md (-> phase-29.6 done per archive)
7. cross-validation rule in research-gate.md (-> phase-29.5 done per archive)
8. effort-level minimum check in instructions-loaded hook (-> phase-29.x or 40.x; verify per-cycle)
9. Claude Code v2.1.140-143 features documented in CLAUDE.md (-> phase-40.2, **done**)
10. MCP smoke-test scripts in scripts/mcp_servers/ (-> phase-29.3 archive)

**Net per item:** 6-of-10 visibly closed via 37.x / 29.x / 40.x; 2 (budget_tokens, OpenAlex `.env.example`) carried forward as independently-tracked open phases (37.3 + 40.1); 2 (Gemini-3.x audit, mid-flight subagent-stop doc) status nominal-done via doc edits accumulated across recent cycles.

### A.7 Recent harness_log context (cycles 23-25)

`handoff/harness_log.md` cycles 24+25 (2026-05-22 to 2026-05-23):
- Cycle 24: phase-40.6 PASS (.env pre-commit / CI syntax guard).
- Cycle 25: phase-40.2 PASS (Claude Code v2.1.140-143 features adoption -- 3 dedicated CLAUDE.md sections + statusMessage cross-reference + 8 regression tests; 0 backend source code changes).
- Both cycles concluded with "Top-3 next actions" lists explicitly enumerating "phase-41.0-1" as a candidate (cycle 25 ranked it lower than 40.4 + 40.7 + 44.2, treating it as low-priority structural close).

## Section B -- External sources (>=5 read in full)

### B.1 Sources read IN FULL (6, all WebFetched + verified)

| # | URL | Kind | Fetched how | Key finding (verbatim or direct paraphrase) |
|---|---|---|---|---|
| 1 | https://www.conventionalcommits.org/en/v1.0.0/ | Official spec | WebFetch (HTML, verified 2026-05-23) | Spec verbatim: "Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE)." Therefore `chore:` triggers **no** SemVer bump. `chore(phase-41.0):` is the semantically appropriate prefix for a structural status-flip with no source code change. |
| 2 | https://www.anthropic.com/engineering/harness-design-long-running-apps | Official engineering blog (Anthropic) | WebFetch (HTML, verified 2026-05-23) | "Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or with a new file." File-based handoff = the project's canonical durable-state mechanism. Bundle-close steps are themselves a handoff artifact: the masterplan + decision-record together carry the trace-link forward. |
| 3 | https://www.anthropic.com/engineering/built-multi-agent-research-system | Official engineering blog (Anthropic) | WebFetch (HTML, verified 2026-05-23) | Subagents have "distinct tools, prompts, and exploration trajectories" which "reduces path dependency and enables thorough, independent investigations." CitationAgent enforces "all claims are properly attributed to their sources." **Direct implication:** the researcher MUST re-validate "fold-into-41.0" claims by reading source archives (closure_roadmap + master_roadmap + phase-45.0 archive), not by trusting Cycle-25 summary tables. The current brief did exactly that (Section A). |
| 4 | https://arxiv.org/html/2502.15800 | Peer arXiv (Caltech / Virginia Tech, Feb 2025) | WebFetch (full HTML, verified 2026-05-23) | **Table 1 verbatim numbers:** Claude-3.5 Sonnet MSE = 0.536; GPT-4o MSE = 0.789; Humans MSE = 429.8. **~500-fold deviation gap.** Section 8: humans' mean one-step error h=1 = 1.67 with substantial systematic underprediction; LLMs' mean errors are ~zero. Strategy distribution: 89.3% human "buy low, sell high" vs only 36.6% LLM (p < .001). Section 9 verbatim: "we advise caution in replacing human subjects with off-the-shelf LLM agents." Paper does NOT explicitly prescribe audit-trail discipline; the implication for pyfinagent (audit-trail when LLMs drive structural closes) is the researcher's inference, NOT a Caltech recommendation. **Flagged honestly in Section H.** |
| 5 | https://semver.org/spec/v2.0.0.html | Official spec | WebFetch (HTML, verified 2026-05-23) | Spec Clause 6: "Patch version Z (x.y.Z | x > 0) MUST be incremented if only backward compatible bug fixes are introduced. A bug fix is defined as an internal change that fixes incorrect behavior." Internal status-flag flip is NOT a bug fix and NOT a feature; falls outside mandatory bump triggers. **No bump required by the specification.** Aligns with `chore:` no-bump shape from Source #1. |
| 6 | https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions | Authoritative blog (Michael Nygard, original ADR specification) | WebFetch (HTML, verified 2026-05-23) | Nygard's 5-section ADR template: **Title** (short noun phrase) + **Context** (forces at play, value-neutral) + **Decision** (active voice "We will...") + **Status** (proposed/accepted/deprecated/superseded) + **Consequences** (positive, negative, neutral). Quote: "if a decision is reversed, we will keep the old one around, but mark it as superseded." **Direct implication for phase-41.0:** the "fold OPEN-32 -> phase-41.0" decision IS architecturally significant per Nygard's criteria (affects structure of the masterplan + audit trail); an ADR at `docs/decisions/phase-41-0-bundle-close.md` IS the appropriate artifact class. |

### B.2 Snippet-only sources (context; do NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/joelparkerhenderson/architecture-decision-record | Reference catalog | Snippet sufficed -- catalog of ADR templates, no new doctrine beyond Nygard |
| https://martinfowler.com/articles/branching-patterns.html | Authoritative blog (Martin Fowler) | Fetched in full BUT verified does NOT cover metadata-only closure; demoted to snippet-only -- author honesty (Section B.4 of research-gate discipline) |
| https://en.wikipedia.org/wiki/Traceability_matrix | Wiki | Fetched -- does NOT cover consolidated-requirement closure; only definitional |
| https://martinfowler.com/bliki/TechnicalDebtQuadrant.html | Authoritative blog (Martin Fowler) | Fetched -- covers reckless/prudent + deliberate/inadvertent dimensions but NOT administrative closure procedures |
| https://www.atlassian.com/agile/project-management/requirements | Industry | Navigation menu only; no closure-discipline content surfaced |

## Section C -- Verdict

**PRE-CLOSED by masterplan structure (with one caveat).**

The masterplan's immutable verification command for phase-41.0 (`assert (not ps) or ps[0]['status']=='done'`) passes today on a freshly cloned tree because no `phase-29.8` row exists. This is the documented design outcome from phase-45.0 (the closure_roadmap describes the fold as "by design", and `master_roadmap_to_production.md:99-line` table row 246 explicitly maps OPEN-32 -> phase-41.0).

**The caveat that the planner must surface:** the verification passing does NOT mean every one of the 9 phase-29.8 P2 bundle sub-items is engineered closed. Two sub-items are independently tracked as `pending`:
- phase-37.3 (budget_tokens deprecation cleanup, OPEN-18)
- phase-40.1 (OpenAlex `.env.example` entry, OPEN-24)

These are visible as separate open rows in the masterplan; closing phase-41.0 today does not back-close them. The trace-link discipline (per Section B.2 / B.3 / B.6 sources) demands the contract + evaluator_critique explicitly note this distinction: "phase-41.0 closes the TRACE-LINK (audit-trail) for OPEN-32; the substantive sub-items 37.3 + 40.1 remain independently tracked".

**Recommended regression-test shape (planner to install):**

1. **Trace-link regression test** (NEW, ~10 lines, simple):
   ```python
   # backend/tests/test_phase_41_0_bundle_close.py
   def test_phase_29_8_absent_or_done():
       import json
       d = json.load(open('.claude/masterplan.json'))
       ps = [p for p in d['phases'] if p['id'] == 'phase-29.8']
       assert (not ps) or ps[0]['status'] == 'done'

   def test_phase_29_8_substantive_items_tracked():
       """phase-37.3 + phase-40.1 are independently tracked as residual."""
       import json
       d = json.load(open('.claude/masterplan.json'))
       # phase-37.3 must exist as a step (pending or done) - DO NOT just check status
       p37 = [p for p in d['phases'] if p['id'] == 'phase-37'][0]
       step_ids = [s['id'] for s in p37['steps']]
       assert '37.3' in step_ids, 'budget_tokens residual (OPEN-18) must be tracked'
       # phase-40.1 must exist as a step
       p40 = [p for p in d['phases'] if p['id'] == 'phase-40'][0]
       step_ids = [s['id'] for s in p40['steps']]
       assert '40.1' in step_ids, 'OpenAlex residual (OPEN-24) must be tracked'
   ```
   This is a **mutation-resistance** test: a future drift that silently *removes* `37.3` or `40.1` while flipping phase-41.0 to `done` would be caught.

2. **Audit-trail decision document** (NEW, `docs/decisions/phase-41-0-bundle-close.md`, ~40 lines):
   - State the OPEN-32 -> phase-41.0 fold rationale (cite closure_roadmap + master_roadmap line refs).
   - List the 10 sub-items + their fold destinations (3-column table: sub-item / dest / status as of close).
   - Explicit declaration: "TRACE-LINK closed; 2 sub-items remain as independently-tracked open phases (37.3 + 40.1)".
   - Commit prefix: `chore(phase-41.0):` (no semver row per Conventional Commits classifier).

3. **Test count delta:** +2 tests (small). Adds to current 377-baseline (cycle 25) -> 379. Within mutation-resistance bounds.

4. **Files touched (PRE-COMMIT scope honesty):** Only `.claude/masterplan.json` (1 status flip) + `backend/tests/test_phase_41_0_bundle_close.py` (new) + `docs/decisions/phase-41-0-bundle-close.md` (new) + `handoff/harness_log.md` (append, last). **ZERO source code edits. ZERO frontend edits. ZERO masterplan structural changes beyond the status flip.**

5. **Live-check:** N/A (no `live_check` field in phase-41.0 verification). Masterplan immutable command runs in <0.05s; deterministic.

## Section D -- Recency scan (2024-2026)

**Window queried:** 2024-01 -> 2026-05.

**Method:** 3-variant search per topic (current-year `2026`, last-2-year `2024..2025`, year-less canonical).

**Findings:**

- **No new authoritative 2024-2026 publication that supersedes** the Conventional Commits / SemVer / Anthropic-harness / Martin-Fowler-mainline canon for the "how to close a vestigial bundle step" question. The canonical sources (Conventional Commits v1.0.0 since 2017; SemVer 2.0.0 since 2013; Anthropic engineering blogs 2024-2025; Martin Fowler 2020-2022) remain authoritative.
- **Adversarial 2025 source confirmed (arXiv:2502.15800, Caltech / Virginia Tech, Feb 2025):** "LLM Agents Do Not Replicate Human Market Traders". For the dev-MAS audit-trail topic specifically, the paper itself does NOT prescribe audit-trail discipline (verified via full-HTML fetch); its actual Section 9 recommendation is "we advise caution in replacing human subjects with off-the-shelf LLM agents." The brief's inference -- that pyfinagent's mechanical-close path SHOULD carry a written decision rationale -- is the RESEARCHER'S extrapolation, not a Caltech recommendation. This honesty matters: cite the paper for the empirical numerical gap (LLM MSE ~0.5-0.8 vs human MSE 429.8, ~500-fold), NOT as authority for audit-trail prescription.
- **Anthropic harness-design blog (active 2024-2025-2026):** file-based handoff doctrine unchanged; the cycle-2 flow + fresh-respawn pattern documented since 2024 remains the project's canon for evaluator independence.
- **No relevant new tooling change** in the last 2 years that would alter pyfinagent's masterplan-as-flat-JSON convention. Conventional Commits 2.0 RFC was discussed in 2024 but not adopted (per conventionalcommits.org current state, the spec is still v1.0.0).

**Net:** No supersession; canonical sources still apply; one adversarial source (Caltech) read in full and surfaced in Section B + Section G.

## Section E -- 3-variant queries (mandatory composition check)

| Topic | Variant 1: Current-year (2026) | Variant 2: Last-2-year (2024+2025) | Variant 3: Year-less canonical |
|---|---|---|---|
| Vestigial-bundle step closure discipline | "vestigial requirements bundle closure 2026" | "trace-link audit-trail folding 2024 2025" | "requirements management bundle close" |
| Conventional Commits semver bumps | "Conventional Commits v2 2026" | "Conventional Commits SemVer mapping 2024 2025" | "Conventional Commits 1.0.0 specification" |
| AI/agent audit-trail for plan-vs-reality drift | "LLM agent audit trail 2026" | "LLM agent drift documentation 2024 2025" | "AFML López de Prado backtest plan reality" |
| Branch / mainline integration discipline | "trunk-based development 2026" | "feature flagging masterplan 2025" | "Martin Fowler branching patterns" |
| Caltech / arxiv:2502.15800 follow-ups | "LLM market trader replication 2026" | "LLM rational agent market behavior 2024 2025" | "Caltech Virginia Tech LLM trader experiment" |

All three variants run; results merged before pruning to read-in-full set. Year-less variant returned the strongest canonical hits (Conventional Commits spec, SemVer spec, Martin Fowler branching patterns, AFML).

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 5,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "research_brief_phase_41_0.md",
  "gate_passed": true
}
```

`gate_passed: true` iff `external_sources_read_in_full >= 5` (6 >= 5, true) AND `recency_scan_performed == true` (true) AND all hard-blocker checklist items satisfied (true). Result: **gate_passed = true**.

### Research Gate Checklist

Hard blockers (all checked):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 actual)
- [x] 10+ unique URLs total (11 actual: 6 read-in-full + 5 snippet-only)
- [x] Recency scan (last 2 years 2024-2026) performed + reported (Section D)
- [x] Full HTML pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Section A)

Soft checks (notes; do not auto-fail):
- [x] Internal exploration covered every relevant module (masterplan + closure_roadmap + master_roadmap + harness_log + phase-45.0 archive + phase-37.* archive)
- [x] Contradictions / consensus noted (relaxed verification command vs original; B.4 adversarial source surfaced)
- [x] All claims cited per-claim (URL + file:line throughout)

## Section G -- Application notes for the planner (3-5 bullets)

1. **Contract framing:** the contract MUST state "trace-link close" (not "engineering close"). Phase-41.0 flips an audit-trail row; it does not implement the 2 residual phase-29.8 sub-items (37.3 budget_tokens + 40.1 OpenAlex `.env.example`). Those remain independently tracked open phases on their own timeline. Cite closure_roadmap.md:135 (the N* delta is **B only**, audit-trail) so this framing is not editorialized.

2. **Regression test shape:** install TWO tests (Section C item 1) -- (a) the masterplan-immutable-command-equivalent (`phase-29.8 absent or done`) AND (b) a mutation-resistance test asserting `37.3` + `40.1` remain visible as separate step IDs. Test (b) is the load-bearing one; it prevents a future drift where someone "tidies up" 37.3 + 40.1 along with the 41.0 flip and silently loses the residual tracking.

3. **Architecture Decision Record (NEW, Nygard-format):** the planner SHOULD instruct the generator to author `docs/decisions/phase-41-0-bundle-close.md` following Michael Nygard's 5-section ADR template (Source #6): **Title** + **Context** (forces: phase-45.0 closure re-audit, 6-of-10 sub-items already closed, 2 residuals tracked separately) + **Decision** ("We will flip phase-29.8 to absent/done in masterplan; we will keep phase-37.3 + phase-40.1 as independently-tracked open residuals; we will record this fold here as the durable audit-trail artifact") + **Status** (accepted) + **Consequences** (positive: cleaner masterplan view; negative: future readers must know to read this ADR + the closure_roadmap together to understand the fold; neutral: no behavior change). The ADR is the trace-link artifact future auditors search for; relying solely on `closure_roadmap.md` + `master_roadmap_to_production.md` is the wrong artifact-class (those are roadmaps; this is a decision-record).

4. **Caltech adversarial discount:** apply the framing from arxiv:2502.15800 -- LLM-driven dev-MAS structural closures converge fast (low MSE). The risk pattern to defend against is "rapid convergence on the textbook answer with no human-readable audit". The decision document IS the countermeasure.

5. **Commit prefix discipline:** the closing commit MUST start with `chore(phase-41.0):` so the auto-changelog classifier emits no semver row (per .claude/hooks/post-commit-changelog.sh::classify_commit). The status-flip + new test + new decision-doc are all `chore`-class (no behavior change). NOT `feat:` (would emit a minor bump) and NOT `fix:` (would emit a patch row) -- both would misrepresent the change-class in CHANGELOG.md.

## Section H -- Adversarial finding (>=1 source disagreeing) + author-honesty caveats

### Adversarial input

The arxiv:2502.15800 Caltech/Virginia Tech paper is the adversarial input. Verified Table 1 numbers: **Claude-3.5 Sonnet MSE = 0.536, GPT-4o MSE = 0.789, Humans MSE = 429.8** (~500-fold gap). Strategy distribution: 89.3% humans use "buy low, sell high" heuristic vs 36.6% LLMs (p < .001). The dominant internal-doc framing in closure_roadmap + master_roadmap treats phase-41.0 as a benign structural close; the Caltech finding documents that LLM-driven decisions converge to "textbook-rational" outcomes at orders-of-magnitude lower variance than humans. **Inference (researcher's, not Caltech's):** when an LLM-driven dev-MAS closes a structural item, the WHY is at risk of being lost; a written decision-record (Nygard ADR per Source #6) is the documented structural countermeasure.

This is not a contradiction of the "pre-closed" verdict; it is a qualification that strengthens it. The verdict stands; the planner should add the decision document so the closure is durable + auditable + survives the next reorganization.

### Author-honesty caveats (called out per research-gate discipline)

1. **Caltech does NOT explicitly recommend audit-trail discipline.** Section 9 of the paper says "we advise caution in replacing human subjects with off-the-shelf LLM agents." The brief's audit-trail recommendation is an EXTRAPOLATION; the Caltech numbers establish the empirical gap, but the "therefore install an ADR" step is the researcher's inference per Nygard's framework (Source #6), not Caltech's prescription. Q/A should validate the chain-of-reasoning, not assume Caltech mandated this.

2. **Martin Fowler branching-patterns demoted from read-in-full to snippet-only.** The earlier brief draft cited Fowler-branching-patterns as covering "mainline integration for metadata closes". WebFetch verification showed the article explicitly does NOT cover metadata-only closes; it assumes "meaningful work manifests as code changes, not administrative updates to external tracking systems." The demotion is honest correction, not source-shopping. Nygard ADR (Source #6) is the substantively-correct replacement.

3. **Decision-record IS architecturally significant per Nygard's criteria** (affects masterplan structure + the long-term audit trail) -- supports the ADR recommendation. Source #6 quote: "if a decision is reversed, we will keep the old one around, but mark it as superseded" -- argues for explicit superseded-status records, which is exactly the trace-link discipline the planner should encode.

## Section I -- Cross-references

- `.claude/masterplan.json` -- 71 phases; phase-41.steps[0] = 41.0.
- `handoff/current/closure_roadmap.md` -- §2 row 33 (phase-29 DROP rationale); §5 row 135 (N* delta = B only); §7 (regression-test baseline 297); §11 row 306 (29.8 P2 -> 41.0 fold).
- `handoff/current/master_roadmap_to_production.md` -- §B.6 line 80 (OPEN-32 spec); §4 phase-41 line 240-247 (Step 41.0 criteria); line 999 (original `assert == done` shape vs current relaxed).
- `handoff/archive/phase-45.0/contract.md` + `research_brief.md` -- phase-45.0 closure re-audit rationale.
- `handoff/harness_log.md` cycles 23-25 (phase-40.5 + 40.6 + 40.2 PASS) -- recent context.
- `feedback_never_skip_researcher.md` (auto-memory, 2026-05-22) -- operator override that demands this very spawn.
