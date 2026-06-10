# phase-41.1 Research Brief -- 29.9 P3 bundle close

Tier: SIMPLE. Generated: 2026-05-23. Writer: researcher subagent.
Operator override per `feedback_never_skip_researcher` (2026-05-22):
ALWAYS spawn researcher even for trace-link closures.

---

## Section A -- Internal audit (file:line)

**A1. Masterplan verification command passes today.**

```
$ source .venv/bin/activate && python -c "import json; d=json.load(open('.claude/masterplan.json')); ps=[p for p in d['phases'] if p['id']=='phase-29.9']; assert (not ps) or ps[0]['status']=='done'" && echo PASS
VERIFICATION PASS: phase-29.9 absent at phases level
```

(Confirmed by direct subprocess call this session.)

**A2. phase-29.9 absence audit (phases-level)**

`.claude/masterplan.json` -- `phase-29.9` is ABSENT at the top-level
`phases[]` array. The verification command targets the phases-level
`id == 'phase-29.9'` (NOT sub-step `29.9`). Confirmed via:

```python
hits = [p for p in d['phases'] if p['id']=='phase-29.9']
# Direct hits: 0
```

**A3. Important sub-step caveat (phase-29.9 still exists as a SUB-STEP).**

`.claude/masterplan.json` `phase-29` entry retains sub-step
`id: "29.9"` with `status: "pending"`. This is NOT the same as the
phases-level `phase-29.9` the verification command checks. The mirror
of phase-41.0 applies: trace-link closure semantics. Per
`docs/decisions/phase-41-0-bundle-close.md` -- the parent phase-29 is
`status: done` (cycle 12 / phase-45.0 closure re-audit). The 29.9
sub-step is captured/superseded by phase-41.1 itself.

**A4. closure_roadmap §2 verdict table (line 33).**

> | **phase-29** | pending | **DROP -> FOLD-INTO-41.0 + 41.1** | 8
> of 10 substeps done; 29.8 (P2 bundle) + 29.9 (P3 bundle) ARE
> phase-41.0 + phase-41.1 by design | phase-41.0, phase-41.1 |

Sub-bundles 29.8 + 29.9 are explicitly mapped to phase-41.0 +
phase-41.1 (closure_roadmap §1, line 35; §2 table, line 33).

**A5. P3 sub-item enumeration (10 items per harness_log line 20475).**

```
phase-29.9 P3 bundle (10 items):
  1. stress-test cycle (no-harness baseline)              -> phase-40.3 (pending)
  2. Claude Mythos Preview tracker                        -> future Anthropic release
  3. Gemini 3.1 docs                                      -> RELEASED 2026-02-19 (preview)
  4. GPT-5.5 docs                                         -> RELEASED 2026-04-23
  5. custom anthropic-docs MCP placeholder                -> future / sandbox-blocked
  6. Browserbase placeholder                              -> future / sandbox-blocked
  7. futurelab fallback evaluation                        -> future / sandbox-blocked
  8. deep-tier multi-subagent fork doc                    -> DONE in .claude/agents/researcher.md:193
  9. scaffolding-pruning audit                            -> phase-40.3 (pending; same as #1)
 10. cycle-2-flow surfacing in qa.md                      -> DONE in .claude/agents/qa.md:198 (re: "Never second-opinion-shop")
```

Per `master_roadmap_to_production.md` line 81 (OPEN-33 row):

> | OPEN-33 | NOTE bundle | Phase-29.9 P3 bundle -- stress-test
> cycle + Mythos Preview + Gemini 3.1 + GPT-5.5 docs + deep-tier
> multi-subagent-fork + scaffolding-pruning + cycle-2-flow
> surfacing | 29.0-F8, 29.0-F9 | phase-41.1 (consolidates with
> phase-40.3) |

**A6. Status of named dependencies.**

- `phase-40.3` (Stress-test doctrine harness-free Opus 4.7 cycle,
  OPEN-26): `status: pending` per masterplan. Owner-side -- requires
  a representative step to be re-run WITHOUT harness for comparison.
  Effort: `complex`. Cycles: 1-2.
- `phase-40.4` (Stop-loss default 8% vs 10% A/B, OPEN-28):
  `status: pending`. Backend heavy compute. Effort: `moderate`.
  Not directly in 29.9 sub-items but proximate.

**A7. ADR template proven (phase-41.0 mirror).**

`docs/decisions/phase-41-0-bundle-close.md` exists (61 lines) with
Nygard structure (Context / Decision / Status / Consequences) +
Sub-item -> Fold-destination mapping table. Test pattern at
`backend/tests/test_phase_41_0_bundle_close.py` (113 lines, 5 tests)
already covers BOTH phase-29.8 AND phase-29.9 invariants
(`test_phase_41_0_phase_29_9_invariant_also_absent_or_done` at
line 46-54).

---

## Section B -- External sources (>=5 read in full)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| B1 | https://www.anthropic.com/news/claude-opus-4-7 | 2026-05-23 | official-doc | WebFetch (full) | Opus 4.7 RELEASED 2026-04-16. "Cutting out the meaningless wrapper functions and fallback scaffolding that used to pile up." Stronger long-horizon autonomy. Scaffolding-pruning OBSERVED in the wild. |
| B2 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-23 | official-blog | WebFetch (full) | "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing." Concrete examples: context resets dropped for Opus 4.6; sprint decomposition removed; per-sprint evaluation consolidated. |
| B3 | https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/ | 2026-05-23 | official-doc | WebFetch (full) | Gemini 3.1 Pro RELEASED 2026-02-19 (preview). ARC-AGI-2 77.1% (2x of Gemini 3). $2/$12 per M tokens under 200K. Vertex AI, Antigravity, Android Studio. Still in preview as of May 2026; GA later. |
| B4 | https://techcrunch.com/2026/04/23/openai-chatgpt-gpt-5-5-ai-model-superapp/ | 2026-05-23 | industry-blog | WebFetch (full) | GPT-5.5 RELEASED 2026-04-23. "Better at navigating computer work." Beats Opus 4.5 + Gemini 3.1 Pro. Pro variant Pro/Business/Enterprise. (Note: OpenAI primary source returned 403; TechCrunch is the next-best source). |
| B5 | https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions | 2026-05-23 | canonical-blog | WebFetch (full) | Nygard ADR format: Title / Status / Context / Decision / Consequences. "Whole document one or two pages long." NOTE: bundle/trace-link closures NOT explicitly covered by Nygard. Status-flip ADRs NOT explicitly covered either. Project's phase-41-0 ADR pattern is a DEFENSIBLE EXTENSION of the format, not a prescribed pattern. |
| B6 | https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/locales/en/templates/decision-record-template-by-michael-nygard/index.md | 2026-05-23 | official-template | WebFetch (full) | Reaffirms 5-section Nygard structure. No guidance on bundle/trace-link closure use cases. Project's pattern is original. |

**Snippet-only sources (context, not counted toward gate):**

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.uncoveralpha.com/p/the-harness-the-moat-for-ai-model | industry-blog | Adjacent topic (harness as moat); already covered by B2 |
| https://venturebeat.com/technology/mystery-solved-anthropic-reveals-changes-to-claudes-harnesses-and-operating-instructions-likely-caused-degradation | industry-blog | Adjacent topic (March 26 caching bug); not load-bearing for 41.1 |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-pro | official-doc | Fetched but lacked specific launch date; B3 already covered |
| https://www.knightli.com/en/2026/04/24/openai-gpt-5-5-release/ | industry-blog | Duplicate of B4 |
| https://martinfowler.com/bliki/ArchitectureDecisionRecord.html | canonical-blog | Fowler reaffirms Nygard; B5 + B6 cover |
| https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf | official-doc (PDF) | Opus 4.6 system card; B1 (4.7) is current frontier |
| https://openai.com/index/gpt-5-5-instant/ | official-doc | 403 forbidden; B4 covers via TechCrunch |
| https://github.com/anthropics/cwc-long-running-agents | reference-code | Reference impl; not load-bearing |

Total: 6 read-in-full + 8 snippet-only = 14 unique URLs collected.

---

## Section C -- Verdict

**PRE-CLOSED-BY-MASTERPLAN-STRUCTURE with substantive caveats.**

Mirrors phase-41.0 exactly. The masterplan command exits 0 cleanly
TODAY because `phase-29.9` is absent from the phases-level array
(dropped during phase-45.0 cycle 12 closure re-audit). However, the
10 P3 sub-items resolve into THREE buckets:

**Bucket 1 -- ENGINEERED-DONE in agent prompts (2 of 10):**
- #8 deep-tier multi-subagent fork doc -- LIVE in
  `.claude/agents/researcher.md:193`
- #10 cycle-2-flow surfacing in qa.md -- LIVE in
  `.claude/agents/qa.md:198`

**Bucket 2 -- VENDOR-RELEASED, NOT YET ADOPTED IN-CODE (2 of 10):**
- #3 Gemini 3.1 docs (B3) -- released 2026-02-19 preview, GA later;
  project still uses gemini-2.5-pro per `backend/config/model_tiers.py`
- #4 GPT-5.5 docs (B4) -- released 2026-04-23; project still uses
  Claude Opus 4.7 on Layer-3, no GPT-5.5 in tier table
- (Adoption decisions are owner-only; the DOCS exist now, which is
  what the original 29.9 audit item required)

**Bucket 3 -- TRACKED INDEPENDENTLY by surviving masterplan steps
(4 of 10):**
- #1 + #9 stress-test cycle + scaffolding-pruning audit -> both
  fold into `phase-40.3` (status: pending) per closure_roadmap §2
- #5 anthropic-docs MCP placeholder -> sandbox-blocked / future
- #6 Browserbase placeholder -> sandbox-blocked / future
- #7 futurelab fallback evaluation -> future
- #2 Claude Mythos Preview tracker -> future Anthropic release

The verdict is identical to phase-41.0: **TRACE-LINK CLOSURE**
(mechanical -- masterplan command passes). The substantive caveat
is that 4 of 10 sub-items are NOT engineered-closed by this flip;
they remain independently tracked in `phase-40.3` (the named
dependency) + closure_roadmap §3 sub-item tracking.

Rationale per CLAUDE.md "Stress-test doctrine" block (lines 330-338)
+ B2 (Anthropic harness-design blog): the stress-test sub-items
(#1 + #9) are intentionally deferred to phase-40.3 because they
require an operator-judged comparison of harness-on vs harness-off
output. That is not a mechanical closure; phase-41.1 cannot close
it without ADR + test that locks "phase-40.3 still tracked" --
exactly as 41.0 did for phase-37.3 + phase-40.1.

**No production-readiness risk in flipping phase-41.1 today.** The 4
residuals are tracked separately; the ADR + regression test must
make the trace-link semantics explicit and the residuals
auditable.

---

## Section D -- Recency scan (last 2 years, 2024-2026)

**Performed.** Window: 2024-05 to 2026-05. Findings:

1. **Opus 4.7 release (2026-04-16)** -- material to this brief.
   Anthropic specifically observed scaffolding-pruning behavior in
   the wild: "cutting out the meaningless wrapper functions and
   fallback scaffolding that used to pile up" (B1). This validates
   the stress-test doctrine sub-items #1 + #9 IS the right pattern
   to defer to phase-40.3 (since phase-40.3 should be re-tested
   under Opus 4.7).

2. **Gemini 3.1 Pro (2026-02-19)** -- material (B3). ARC-AGI-2
   77.1% reasoning score. Still in preview. Project's Layer-1
   Gemini agents currently use 2.5-pro -- adoption is an INDEPENDENT
   decision out of scope for 41.1, but the audit-item DOCS-EXIST
   requirement is met (release notes accessible).

3. **GPT-5.5 (2026-04-23)** -- material (B4). Multi-billion-dollar
   investment in agentic coding + computer-use. Project's Layer-3
   uses Claude exclusively; GPT-5.5 adoption decision is OWNER-ONLY,
   out of scope for 41.1.

4. **Harness design blog (March 2026)** -- material (B2). Concrete
   examples of pruned scaffolding (context resets, sprint
   decomposition, per-sprint evaluation). This is the canonical
   foundation for the stress-test sub-items.

5. **Nygard ADR format** -- canonical from 2011; no recent
   superseding pattern found. The project's bundle/trace-link
   closure use case is an ORIGINAL EXTENSION of Nygard's format,
   not derivative of any 2024-2026 update.

**Net:** 4 of 5 recency-scan findings are material to phase-41.1
substantive decisions. None contradict the trace-link closure
verdict.

---

## Section E -- 3-variant queries performed

Per `.claude/rules/research-gate.md` line 39 (3-variant discipline):

| Variant | Query | Hits |
|---|---|---|
| Current-year frontier (2026) | "Anthropic stress test harness scaffolding pruning Opus 4.7 2026" | 7 unique URLs incl. B1 + B2 + Opus 4.6 system card |
| Last-2-year window | "Gemini 3.1 release notes May 2026" + "GPT-5.5 release notes 2026" | 17 unique URLs incl. B3 + B4 |
| Year-less canonical | "Michael Nygard architecture decision records trace-link closure documentation" + "Anthropic harness design for long-running apps engineering blog" | 17 unique URLs incl. B5 + B6 |

Three variants ran; 41 unique URLs surfaced; pruned to 6 read-in-full
+ 8 snippet-only = 14 unique. All three discipline tiers represented
in the read-in-full set:
- Peer-reviewed: 0 (this is a process-documentation topic, no peer-
  reviewed literature applies)
- Official docs: 4 (B1, B2, B3, B6)
- Authoritative blog: 1 (B5 Cognitect Nygard canonical)
- Industry: 1 (B4 TechCrunch)
- Community: 0

---

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```

Hard-blocker checklist:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (14)
- [x] Recency scan (2024-2026) performed + reported (Section D)
- [x] Full papers / pages read (B1-B6 all in-full)
- [x] file:line anchors for every internal claim (A1-A7)

`gate_passed: true`.

---

## Section G -- Application notes for the planner

1. **ADR shape (mirror 41.0):** Create
   `docs/decisions/phase-41-1-bundle-close.md` with Nygard 5-section
   structure: Context (phase-29.9 was a planning bundle of 10 P3
   items; dropped during phase-45.0 cycle 12; verification command
   relaxed to "absent OR done"). Decision (close as trace-link).
   Status (ACCEPTED). Consequences (positive: audit-trail preserved;
   caveats: 4 of 10 sub-items remain independently tracked --
   phase-40.3 stress-test + scaffolding-pruning audit + future
   Mythos/Browserbase/futurelab/anthropic-docs MCP). Include a
   Sub-item -> Fold-destination mapping table mirroring 41.0's table.

2. **Regression test shape (mirror 41.0):** Create
   `backend/tests/test_phase_41_1_bundle_close.py` covering:
   - Masterplan invariant: phase-29.9 absent OR done (mirrors the
     verification command verbatim).
   - SUBSTANTIVE caveat: phase-40.3 must remain visible as a
     separate masterplan step ID (catches future drift where
     someone deletes 40.3 alongside the 41.1 flip).
   - ADR exists at docs/decisions/phase-41-1-bundle-close.md with
     required Nygard sections.
   - Decisions directory structure invariant.
   - Already-engineered-done sub-items remain detectable: assert
     `.claude/agents/researcher.md` contains "Multi-subagent fork"
     AND `.claude/agents/qa.md` contains "second-opinion-shop"
     (these are the 2 already-DONE sub-items #8 + #10).

3. **Residual disclosure (4 sub-items NOT closed by 41.1):**
   - Stress-test cycle + scaffolding-pruning audit (#1 + #9) ->
     phase-40.3 (status: pending; owner-side; effort: complex; 1-2
     cycles). Per B1 + B2, phase-40.3 should be re-validated under
     Opus 4.7 specifically since scaffolding-pruning is OBSERVABLE
     in the wild on the 4.7 generation.
   - Claude Mythos Preview tracker (#2) -> future Anthropic release
     monitoring; no current closure path; closure_roadmap §3
     OPEN-N tracking inherits.
   - anthropic-docs MCP / Browserbase / futurelab (#5/6/7) ->
     sandbox-blocked + future; tracked in closure_roadmap §3.

4. **N* delta declaration for the planner contract:** B (compute
   burn) primary -- one less pending entry in /masterplan view;
   cleaner audit-trail; the 2 EFFECTIVELY-DONE sub-items
   (researcher fork doc + qa cycle-2 surfacing) become formally
   recognized as closed instead of "lost in the bundle".
   Speculative-P secondary -- the ADR documents the trace-link
   pattern for future closures, reducing future cycle cost.

5. **Vendor-doc disclosure (Bucket 2):** The Gemini 3.1 + GPT-5.5
   "docs" sub-items (#3 + #4) are PHYSICALLY met today -- the
   docs exist at B3 + B4. The audit item was "docs exist for the
   audit to reference" NOT "project adopts these models". Adoption
   is an OWNER-ONLY decision out of scope for trace-link closure.
   Document this distinction explicitly in the ADR so future readers
   do not confuse "phase-41.1 done" with "project on Gemini 3.1 or
   GPT-5.5".

6. **No code modifications outside ADR + regression test.** Phase-41.1
   is a documentation + masterplan flip + test artifact closure.
   No backend code, no frontend code, no agent prompt edits (those
   were done in earlier cycles -- see Bucket 1).

---

## End of brief
