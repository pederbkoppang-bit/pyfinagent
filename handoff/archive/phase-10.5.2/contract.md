---
step: phase-10.5-batch (covers 10.5.0, 10.5.1, 10.5.2, 10.5.3, 10.5.4, 10.5.5, 10.5.6, 10.5.8)
title: Batch retrospective closure of phase-10.5 sovereign UI
cycle_date: 2026-04-24
harness_required: true (per-step in masterplan) -- batched Q/A is the operator's deliberate departure
retrospective: true
batch: true
---

# Sprint Contract -- phase-10.5 batch closure

## Honest framing (read first)

This is a **batched retrospective closure** of 8 masterplan steps in
the phase-10.5 sovereign UI series. All 8 deliverables shipped together
in commit `1122a021` ("feat: phase-10.5 sovereign UI + phase-15
observability tiles + phase-4.17 pre-go-live smoke test") and have
since been running in production (local paper deployment). What was
missing is masterplan bookkeeping: the 8 steps still show
`status: pending`.

Protocol breaches acknowledged up front (Q/A must audit, not hide):

1. **Contract-before-GENERATE violated** for all 8 steps -- deliverables
   predate this contract by days. Same pattern as phase-17.1 retrospective
   closure (see `handoff/archive/phase-17.1/` for the template).
2. **Batched Q/A instead of 8 separate Q/As** -- operator decision to
   avoid 8x ceremony for 8 retrospective closures. Tradeoff: less
   granular audit; Q/A is asked to explicitly accept or reject the
   batching as acceptable given (a) all deliverables shipped in ONE
   commit and (b) evidence per step is independently verifiable.
3. **Two immutable verification commands are broken** (pre-existing
   codebase defects, orthogonal to 10.5.x deliverables). Details in the
   evidence table below. Main is NOT editing the commands; Main runs
   the test logic correctly and documents the command-level breakage.

## Research-gate summary

Per-step research briefs for 10.5.0 through 10.5.8 (except 10.5.7) are
already present in `handoff/current/phase-10.5.N-research-brief.md`,
shipped alongside the code in commit `1122a021`. No fresh research was
spawned this cycle; the pre-existing briefs are current and
tied to the actual deliverables.

Brief inventory (verified to exist):
- phase-10.5.0-research-brief.md (379 lines)
- phase-10.5.1-research-brief.md (290 lines)
- phase-10.5.2-research-brief.md (282 lines)
- phase-10.5.3-research-brief.md (292 lines)
- phase-10.5.4-research-brief.md (255 lines)
- phase-10.5.5-research-brief.md (188 lines)
- phase-10.5.6-research-brief.md (225 lines)
- phase-10.5.8-research-brief.md (331 lines)

Note: phase-10.5.7 has no research brief in this set -- that's the
net-new step (homepage hero embed) which gets its own forward cycle
next.

## Hypothesis

The phase-10.5 sovereign UI shipped in commit `1122a021` is
substantively complete and the deliverables (three backend endpoints,
one BQ view, one Next.js route shell, four React components + one
strategy detail route, one accessibility audit pass) are running green
in the live environment. Masterplan bookkeeping can be closed for all
8 steps in one batch provided Q/A can independently verify each step's
success criteria against the current state.

## Success Criteria -- per step (verbatim from .claude/masterplan.json)

### 10.5.0 Backend read endpoints

Cmd: `cd backend && pytest tests/api/test_sovereign.py -q && python -c "... red-line?window=30d ... assert len(r['series'])>=25"`
- three_endpoints_landed
- p95_latency_under_800ms
- cron_slots_zero_declared

### 10.5.1 BQ view strategy_deployments

Cmd: `python scripts/migrations/create_strategy_deployments_view.py --verify`
- view_exists
- at_least_one_champion_row

### 10.5.2 /sovereign route shell

Cmd: `cd frontend && npm run build && node scripts/audit/sovereign_route.js`
- route_reachable
- sidebar_entry_added
- page_shell_conforms_to_frontend_layout

### 10.5.3 RedLineMonitor component

Cmd: `cd frontend && npm run test -- --filter=RedLineMonitor`
- window_selector_7_30_90
- reference_line_zero
- kill_switch_and_flip_markers_rendered
- recharts_composed_chart

### 10.5.4 ComputeCostBreakdown

Cmd: `cd frontend && npm run test -- --filter=ComputeCostBreakdown`
- deterministic_color_map_present
- providers_cover_anthropic_vertex_openai_bq_altdata
- tooltip_shows_usd_and_percent

### 10.5.5 AlphaLeaderboard

Cmd: `cd frontend && npm run test -- --filter=AlphaLeaderboard`
- columns_match_spec
- status_pill_phosphor_only
- sort_persists_client_side
- filter_by_status_pill_row

### 10.5.6 Strategy detail route

Cmd: `cd frontend && npm run test -- --filter=StrategyDetail`
- equity_curve_scoped_by_strategy
- param_override_timeline_rendered
- kill_switch_events_scoped

### 10.5.8 Accessibility + consistency pass

Cmd: `cd frontend && npm run axe && npm run lint && node scripts/audit/sovereign_consistency.js`
- wcag_2_1_aa_pass
- phosphor_icons_only
- no_emoji_in_ui
- dark_theme_token_0f172a

## Plan steps

No code changes this cycle. The plan is purely bookkeeping + evidence
collection + Q/A audit. Sequence:

1. Verify each step's deliverable exists (git ls-tree, ls, grep)
2. Run each step's verification logic correctly (repo-root pytest;
   frontend tests; BQ --verify; sovereign_consistency.js)
3. Document live-endpoint evidence where applicable
4. Openly disclose the two broken verification-command bugs
5. Spawn a single Q/A; ask it to explicitly rule on batching
6. If PASS / CONDITIONAL: append log, flip 8 statuses in one
   masterplan write, let the (now-fixed) archive hook auto-archive

## What Q/A must audit

1. **Batching legitimacy**: is it defensible to Q/A 8 retrospective
   closures in one spawn, or should this be split?
2. **Per-step evidence**: for each of the 8 steps, does the evidence
   table in experiment_results.md demonstrate the success criteria are
   met in the current state (not just historically)?
3. **Verification-command honesty**: for 10.5.0 (pytest stdlib shadow)
   and 10.5.2 (missing audit script), does Main's documented workaround
   (run tests from repo root; confirm route via npm build) constitute
   honest evidence of deliverable, or should those two steps be split
   out for a separate cycle to fix the commands first?
4. **Research-gate dependency**: pre-existing research briefs exist for
   10.5.0-10.5.8 (minus 10.5.7). Is that sufficient, or must the
   researcher be respawned for the batch?
5. **Archive hook interaction**: this cycle will flip 8 steps in one
   masterplan write, which will trigger the fixed archive-handoff hook
   8 times. Is the hook's new state-file semantics (see
   `.claude/.archive-baseline.json`) correctly handling batches?

## References

- commit `1122a021` -- the phase-10.5 mega-commit that shipped the deliverables
- `handoff/current/phase-10.5.N-research-brief.md` for N in {0,1,2,3,4,5,6,8}
- `handoff/archive/phase-17.1/contract.md` -- previous retrospective-closure template
- `.claude/hooks/archive-handoff.sh` -- newly-fixed archive hook
- `.claude/rules/research-gate.md` -- research-gate discipline
- `CLAUDE.md` -- harness protocol (retrospective closure + 5-file protocol)
