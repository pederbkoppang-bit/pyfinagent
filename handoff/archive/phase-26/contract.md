# phase-45.0 -- CLOSURE Re-Audit + Master Sequencing Plan

**Step id:** `phase-45.0` (new; appended to masterplan in GENERATE step)
**Date:** 2026-05-22
**Mode:** OVERNIGHT planning -- one harness pass; deep-tier research + legacy dedup; NOT execution.
**Author:** Main (Claude Opus 4.7, this Claude Code session)
**Cycle in `handoff/harness_log.md`:** Cycle 12 (after Cycle 11 phase-44.0 frontend master design)

---

## North-star delta (mandated by /goal directive)

**Term:** Burn (compute / cycle / token spend reduction) PRIMARY; Profit (correct sequencing) SECONDARY.

**Quantified estimate:**
- 6 DROP verdicts -> immediate removal of ~30-40 zombie steps from `/masterplan` view, saving ~2-4 cycles per future session in re-orientation cost
- ~40-55 cycles to PRODUCTION_READY (vs ~80+ if legacy phases were walked literally) = ~50% Burn reduction on the closure path
- Profit: correct sequencing means phase-36.1 (the last code BLOCK on profit-protection) lands BEFORE phase-43.0 DoD, which means real n_trades > 0 happens 5-10 cycles sooner. Speculative magnitude: ~2-4 weeks earlier real-trades = ~$0-$500 unrealized profit signal acceleration

**How measured:** Closure cycle count + masterplan pending-step count diff (pre-45.0 vs post-45.0) + days-to-first-non-HOLD-decided-trade.

---

## Research-gate summary

The researcher subagent (id `aeb5b58f03fa94b75`, effort `deep`/max-tier) produced
`handoff/current/research_brief.md` (529 lines) covering:

- **Section A:** 12-row legacy-phase verdict table -- **6 DROP (fold-into-3X.Y)** + **3 DEFER-POST-PROD** + **3 KEEP**. Fold-mappings:
  - phase-4 -> phase-43.0 DoD (Production Readiness is the DoD audit itself)
  - phase-16 -> phase-43.0 (E2E UAT folds into the 26-criterion gate)
  - phase-23.7 -> verify-then-done (Harness plumbing landed via phase-23.7.x)
  - phase-26 -> phase-40.2 + 40.3 + 41.x (Frontier-sync overlaps stress-test doctrine + bundle closure)
  - phase-27 -> phase-37 (Multi-provider is now LLM-route hardening)
  - phase-29 -> phase-41.0 + 41.1 (Harness MAS + MCP + Academic-Fetch + Frontier-Sync = 29.8/29.9 bundle closure)
  - DEFER: phase-5 (Multi-Market Expansion -- post-PRODUCTION_READY), phase-10.7 (Meta-Evolution Engine -- post-PROD), phase-13 (Seatbelt sandboxing -- intentionally blocked by design)
  - KEEP: phase-23.6 (residual phase-23.5 items still open), phase-23.8 partial (Dev-MAS audit -- residual after Cycle 10 closure), phase-28 residual (Candidate Picker)

- **Section B (CRITICAL):** BQ probes confirm phase-35.1 + phase-35.2 are **NOT** closed by c7801712 cycle:
  - `outcome_tracking` table = 0 rows (schema exists but writer missing in code)
  - `agent_memories` table = 0 rows (schema exists but writer missing in code)
  - `llm_call_log` last row = 2026-05-21 05:15 UTC (Risk Judge invocations NOT being telemetered for autonomous-loop path)
  - phase-35.1/35.2 verdicts upgrade to CODE-GAP-CONFIRMED (more focused than "behavior-gap-unknown")

- **Section B silver lining:** phase-32.2 trail discipline production-verified -- 2 trail-stop events today (LITE +9.54% pnl at 16:59 UTC + COHR +17.89% pnl at 18:35 UTC, both 25-day holds, capture_ratio=0.63 on COHR). DoD-3 (phase-32.2 live-verified) effectively LANDED today, ahead of phase-43.0.

- **Section C:** 11 external sources read in full across 7 domains (quant trading + AI research + AI observability + runtime + security + frontend + model-risk governance). One adversarial finding flagged (Caltech arxiv:2502.15800 -- "LLM Agents Do Not Replicate Human Market Traders") -- the planner must surface this honestly in DoD-1 / DoD-2.

- **Section I:** `gate_passed: true` -- 11 of 8-source floor cleared (38% buffer).

**Researcher headline:** Dropping 6 legacy phases shrinks the closure path from ~80+ to ~40-55 cycles. Phase-35.1/35.2 need code (writers), not just observation. Phase-32.2 verified live today.

---

## Hypothesis

> If we produce `handoff/current/closure_roadmap.md` with the
> researcher's 12-row legacy-phase verdict table + the c7801712 BQ-probe
> findings + 11-source 2026 frontier synthesis + integration risk matrix
> + regression-test snapshot + JSON-ready masterplan flips for the 6
> DROP + 3 DEFER verdicts; AND we apply those 9 masterplan flips
> (status `done` for 6 DROPs with fold-note; status `deferred` for 3
> DEFERs); AND we add a new phase-45 entry with step 45.0 representing
> this planning step; THEN the closure execution path from
> phase-35.1 through phase-43.0 PRODUCTION_READY is unambiguous, the
> `/masterplan` view shows only ~14-16 actionable open steps (down
> from ~30+), and the next session can pick up the critical path
> without re-orientation cost.

If true: phase-45.0 closes with one closure_roadmap doc + 9 masterplan
status flips + 1 new phase-45 entry. Q/A verifies per-row coverage,
flip discipline (DROPs land as `done` with fold-note, DEFERs as
`deferred`), 11-source external floor, regression snapshot captured,
plan-only honored.

If false: either coverage misses a legacy phase (Q/A returns
CONDITIONAL), a flip introduces a regression (the DROP-to-done flip
on a phase that secretly has unfinished children), or the c7801712
BQ findings aren't reflected in updated phase-35.1/35.2 audit_basis.
Fix + fresh Q/A. Max 2 retries -> `blocked`.

---

## Immutable success criteria (decomposed from /goal directive)

The /goal directive established the planning-only mandate. Decomposed:

1. **Researcher gate passed:** `research_brief.md` Section I shows
   `gate_passed: true` with `external_sources_read_in_full: >= 8`.
   VERIFIED -- 11 fetched-in-full.

2. **Legacy-phase coverage:** all 12 named legacy phases (4, 5, 10.7,
   13, 16, 23.6-8, 26-29) have a verdict row in `closure_roadmap.md`
   Section A. Zero silent drops.

3. **c7801712 verdict update:** phase-35.1 + 35.2 audit_basis upgraded
   to reflect the schema-empty / writer-missing finding (NOT closed
   organically; code work still needed). phase-32.2 verdict upgrades
   to "live-verified by trail events for LITE + COHR on 2026-05-22".

4. **Masterplan flips applied:** 6 DROP -> `status: done` with
   `closing_phase` note in audit_basis or notes field; 3 DEFER ->
   `status: deferred`. 1 new phase-45 with step 45.0.

5. **Dependency graph acyclic:** updated Mermaid in closure_roadmap.md
   Section ?. Critical path explicit:
   `phase-45.0 -> 35.1+44.1 -> 36.1+44.2 -> 37.1+44.7 -> 35.2+35.3 ->
   sweep -> 43.0`.

6. **N* delta per surviving step:** every step in the new
   closure_roadmap walks-the-graph table has a North-star delta column
   (P / R / B + estimate + how measured).

7. **Regression snapshot captured:** `pytest backend/ --collect-only -q`
   = 297 tests at session start; locked as the baseline.
   closure_roadmap.md Section F records this.

8. **Plan-only:** `git diff --stat backend/` + `git diff --stat
   frontend/src/` = 0 lines. Only on-disk changes: `handoff/current/`
   + `.claude/masterplan.json` (status flips on the 9 legacy phases +
   new phase-45 entry).

9. **JSON valid:** `python -c "import json; json.load(open('.claude/
   masterplan.json'))"` exits 0 after all flips.

10. **Q/A 5-item compliance audit:** researcher gate / contract-pre-
    generate / harness_log-last / status-flip-last / no-second-opinion-
    shopping -- all 5 PASS by the time Q/A spawns.

---

## Plan steps (within this phase-45.0 work)

| # | Step | Tool / Artifact | Status |
|---|---|---|---|
| 1 | Pre-cycle health check (cycle_history tail + kill-switch + regression snapshot 297) | bash | DONE |
| 2 | Researcher deep tier (12-row legacy verdict + 11 external sources + BQ probe) | `handoff/current/research_brief.md` 529 lines | DONE |
| 3 | Write this contract | `handoff/current/contract.md` | IN FLIGHT |
| 4 | GENERATE: closure_roadmap.md + apply 9 legacy flips + add phase-45 entry | edits to `.claude/masterplan.json` + `closure_roadmap.md` | NEXT |
| 5 | Coverage-check (12 legacy phases addressed) | grep + cross-list | NEXT |
| 6 | Spawn Q/A ONCE (max 2 retries) | qa subagent | NEXT |
| 7 | Create `live_check_45.0.md` (so live_check_gate doesn't block push) | edit | NEXT |
| 8 | Append cycle 12 to `handoff/harness_log.md` FIRST | edit | NEXT |
| 9 | Flip phase-45.0 + parent phase-45 status to done LAST (auto-commit fires, prefix `phase-45.0:`) | masterplan.json | NEXT |

---

## Hard guardrails (verbatim from /goal directive)

- **Plan-only -- ZERO code changes** outside `.claude/masterplan.json`
  + `handoff/current/closure_roadmap.md` + `handoff/current/contract.md`
  + `handoff/current/live_check_45.0.md` + `handoff/current/
  evaluator_critique.md` + `handoff/harness_log.md`.
- **No emojis** anywhere (`feedback_no_emojis`).
- **`feedback_masterplan_status_flip_order`:** new phase-45 lands
  `in-progress` initially; the 9 legacy DROP/DEFER flips don't go to
  `done` until the closure_roadmap rationale is recorded; only THEN
  apply the flips. Single masterplan write at the end.
- **`feedback_log_last`:** harness_log Cycle 12 append BEFORE status
  flip.
- **`feedback_qa_harness_compliance_first`:** Q/A starts with the
  5-item compliance audit.
- **Regression count >= 297** at end of cycle (no code edits, so
  trivially satisfied -- but documented).

---

## References

- `handoff/current/research_brief.md` (529 lines, this cycle)
- `handoff/current/master_roadmap_to_production.md` (1182 lines, Cycle 10)
- `handoff/current/frontend_ux_master_design.md` (922 lines, Cycle 11)
- `.claude/masterplan.json` (to be edited in GENERATE)
- `handoff/cycle_history.jsonl` (c7801712 row at tail)
- /goal directive (verbatim above; sets the closure mandate)
- 11 external 2026 sources cited in research_brief.md Section C
