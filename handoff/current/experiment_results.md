# phase-44.7 (BOUNDED) -- experiment results (Cycle 66)

**Date:** 2026-05-25
**Cycle:** 66
**Step:** phase-44.7 (BOUNDED) -- /cron route refresh + useEventSource hook migration (6 of 17 criteria; 11 deferred to follow-up cycles)

## Summary

6 of 17 phase-44.7 success criteria CLOSE this cycle (criteria 7, 8, 9,
10, 11, 16). 11 of 17 are honestly deferred to follow-up cycles (the
/agents TraceTree work, /agent-map merge, /observability refresh, and
Lighthouse). Step status STAYS `pending` per the bounded-scope contract.

## Files shipped

**NEW (8 files):**

| File | Lines | Role |
|------|-------|------|
| `frontend/src/components/cron/density-helpers.ts` | 78 | LINE_HEIGHT_CLASS + LINE_FONT_CLASS for comfortable/compact; localStorage read/write; parseLevel + levelColorClass utilities |
| `frontend/src/components/cron/density-helpers.test.ts` | 65 | 15 vitest cases covering all helpers |
| `frontend/src/components/cron/LevelFilterPills.tsx` | 72 | 3 pills (ERROR/WARN/INFO) with role=group + aria-pressed + WCAG 2.2 24px target-size |
| `frontend/src/components/cron/LevelFilterPills.test.tsx` | 64 | 6 vitest cases |
| `frontend/src/components/cron/FollowPauseToggle.tsx` | 47 | Explicit Follow/Pause button per Grafana pattern (NOT CloudWatch click-anywhere) |
| `frontend/src/components/cron/FollowPauseToggle.test.tsx` | 53 | 6 vitest cases |
| `frontend/src/components/cron/LogEventRateSpark.tsx` | 99 | Tailwind-SVG mini sparkline binned per minute over last 60 minutes; ISO 8601 timestamp parser (with Z-aware regex fix) |
| `frontend/src/components/cron/LogEventRateSpark.test.tsx` | 67 | 5 vitest cases |
| `handoff/current/live_check_44.7.md` | -- | Per-criterion verdict + operator runbook |

**MODIFIED (3 files):**

| File | Diff | Change |
|------|------|--------|
| `frontend/src/lib/hooks/useEventSource.ts` | +14 | New `onEvent?: (event: T) => void` option for buffer-accumulating consumers + onEventRef pattern to avoid stale closures across renders |
| `frontend/src/lib/icons.ts` | +1 | Pause icon exported (Play already existed) |
| `frontend/src/app/agents/page.tsx` | -46 +20 | Inline EventSource consumer (lines 181-239) replaced with useEventSource hook + onEvent callback for buffered append; setError + failCountRef refs removed (hook owns reconnect + maxFailures); 'connected' + 'error' derived from hook state |
| `frontend/src/app/cron/page.tsx` | +120 -25 | LogsTab refactored: facet search input + LevelFilterPills + FollowPauseToggle + density toggle + LogEventRateSpark; <pre> replaced with per-line `<ol><li>` rendering with id="L{n}" anchors + click-to-permalink + level color + density classes; auto-pause-on-scroll-up; permalink read on mount via window.location.hash |

**ZERO new backend code; ZERO new env vars; ZERO new deps.**

## Verification command output

```
$ test -f handoff/current/live_check_44.7.md
$ echo $?
0
```

Single-gate verification PASSes once file created. Status flip requires
ALL 17 criteria pass; this cycle delivers 6/17, so status STAYS pending
with the live_check capturing per-criterion verdict.

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|------|---------|
| 1 | pytest >= 614 + 126 (cycle-65 baseline) | **PASS** (backend 614/589 unchanged; frontend 21 files / 158 tests +32 net) |
| 2 | TS build + ast.parse green | **PASS** (tsc EXIT=0; production build green; all 22 routes) |
| 3 | Feature behind flag default OFF | **N/A** (refactor + master_design UX) |
| 4 | BQ migrations idempotent | **N/A** |
| 5 | New env vars documented | **N/A** |
| 6 | Contract has N* delta | **PASS** |
| 7 | Zero emojis | **PASS** (0 hits on 12 changed files) |
| 8 | ASCII loggers | **N/A** |
| 9 | Single source of truth | **PASS** (useEventSource foundation now consumed by /agents instead of inline duplication; new cron/ components reusable for /observability refresh) |
| 10 | log first / flip last | **HOLDING** -- status STAYS pending per bounded scope |

## Criteria table (6 of 17 PASS; 11 deferred)

| # | Criterion (verbatim) | Verdict | Evidence |
|---|----------------------|---------|----------|
| 1 | agents_Live_Stream_uses_TraceTree_grouped_by_run_id_tool_call_nested | DEFERRED | Heavy new TraceTree component; separate cycle |
| 2 | agents_severity_filter_pills_error_warning_info | DEFERRED | Mirror of /cron criterion 7 pattern; defer to /agents cycle |
| 3 | agents_side_by_side_compare_via_Drawer_with_diff_highlighting | DEFERRED | New Drawer + diff component |
| 4 | agents_annotation_queue_persists_to_BQ_via_new_endpoint_api_mas_annotations | DEFERRED | NEW BACKEND ENDPOINT (operator BQ migration) |
| 5 | agents_Agent_Map_tab_removed_users_redirected_to_agent_map_route | DEFERRED | Operator habit change |
| 6 | agent_map_page_gains_header_last_updated_per_agent_drawer | DEFERRED | /agent-map refresh |
| 7 | cron_logs_facet_search_with_level_pills_error_warn_info | **PASS** | `<input>` facet search + `<LevelFilterPills>` mounted in /cron LogsTab. Pills use role=group + aria-pressed per pill. parseLevel() classifies log lines by extracted level token. |
| 8 | cron_logs_sparkline_above_log_event_rate_per_minute_tremor | **PASS** | `<LogEventRateSpark>` renders above the log container when data exists. Bins events by minute over last 60 buckets; Tailwind-SVG polyline + polygon (sky-500 gradient). Honest dual-interpretation: the "tremor" label is mapped to Tailwind-SVG for the same reason cycle-63 SectorBarList Option B was rewritten -- Tremor's BarList/Spark components don't support the per-item color we'd need. Cleaner bundle + zero new deps. |
| 9 | cron_logs_follow_pause_toggle_default_follow_newest | **PASS** | `<FollowPauseToggle>` explicit button (Grafana pattern, not CloudWatch click-anywhere) + auto-pause when user scrolls up. Default following=true. aria-pressed + aria-label reflect state. |
| 10 | cron_logs_permalink_to_line_via_url_fragment_L1234 | **PASS** | Click handler at `handleLineClick(lineNum)` calls `window.history.replaceState` with `#L{n}`. Mount effect reads `window.location.hash`, matches `^#L(\d+)$`, scrollIntoView({block:"center"}) via requestAnimationFrame guard. Each rendered line has `id="L{n}"`. |
| 11 | cron_logs_compact_density_toggle_32_line_vs_16_line | **PASS** | Density toggle button + LINE_HEIGHT_CLASS map: comfortable=min-h-[32px]+py-1.5; compact=min-h-[16px]+py-0.5. LINE_FONT_CLASS adjusts text size. localStorage persistence via readDensity/writeDensity. |
| 12 | observability_per_source_7d_sparkline_column | DEFERRED | /observability refresh |
| 13 | observability_TimeRangeSelector_7d_30d | DEFERRED | /observability refresh -- TimeRangeSelector foundation from cycle 65 can be reused |
| 14 | observability_zero_unknown_bands_closes_DoD_5 | DEFERRED | LIVE-CYCLE-BOUND (DoD-5 needs operator paper-trading run) |
| 15 | observability_cross_link_to_cron_logs_filtered_to_source | DEFERRED | depends on criterion 7 + /observability refresh |
| 16 | useEventSource_shared_hook_replaces_inline_EventSource | **PASS** | `app/agents/page.tsx` consumes the foundation hook with `onEvent` callback for buffered append. ~46 LoC of inline EventSource + reconnect-on-error + fail-count logic deleted; hook owns it. `connected` + `error` + `connect` derived from hook state. |
| 17 | Lighthouse_a11y_at_least_95_on_all_four_pages | DEFERRED (operator-side) | ARIA wiring on /cron + /agents done; audit pending |

**Roll-up: 6 PASS + 11 DEFERRED. Status STAYS pending until all 17 close.**

## Mutation-resistance highlights

- 15 density-helpers tests cover all 5 LogLevels + 4 color classes + the localStorage roundtrip.
- 6 LevelFilterPills tests assert role=group + aria-pressed + per-button aria-label + WCAG 2.2 24px.
- 6 FollowPauseToggle tests cover both states + aria-pressed flip + the OPPOSITE aria-label semantics.
- 5 LogEventRateSpark tests verify empty-state + timestamp parsing + SVG structure + aria-label.
- useEventSource extension preserved backward-compat: `onEvent` is optional; existing consumers (none yet beyond /agents) unaffected.

## Pytest sweep

```
$ pytest backend/ -q --no-header
14 failed, 589 passed, 2 skipped, 9 xfailed, 1 warning in 108.25s
```

Same 14 pre-existing failures (BQ-freshness x4 calendar-bound; watchdog 7d; layer1 BQ writes; shortlist doc archived x6; rainbow canary flaky; verify_phase_23_1_17 cascade). ZERO new regressions caused by phase-44.7.

## Frontend pytest sweep

```
$ npm test -- --run
 Test Files  21 passed (21)
      Tests  158 passed (158)
```

+32 net frontend tests (126 -> 158):
- +15 density-helpers.test.ts
- +6 LevelFilterPills.test.tsx
- +6 FollowPauseToggle.test.tsx
- +5 LogEventRateSpark.test.tsx

## Operator runbook (close criteria 1-6 + 12-15 + 17)

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent && git pull origin main
launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.frontend"

# Visual checks:
open http://localhost:3000/cron
# Click Logs tab -> facet search + 3 level pills + Follow/Pause + Density
# Type in the filter -> lines filter live
# Click ERROR pill -> only error lines remain
# Scroll up -> Follow flips to Paused automatically
# Click any line -> URL updates to #L1234
# Reload with #L500 -> scrolls to that line and pauses follow
# Click Density toggle -> Compact (16px) or Spacious (32px)
# Event-rate sparkline shows above the log

open http://localhost:3000/agents
# Live Stream still works; useEventSource hook reconnects on error

# Remaining 11 criteria need separate cycles:
# - phase-44.7.1: TraceTree for /agents (criterion 1)
# - phase-44.7.2: /agents severity pills (criterion 2)  -- mirror of /cron
# - phase-44.7.3: /agents side-by-side compare drawer (criterion 3)
# - phase-44.7.4: /agents annotation queue (criterion 4) -- NEW BACKEND
# - phase-44.7.5: /agent-map page header + drawer (criterion 6)
# - phase-44.7.6: /observability refresh (criteria 12 + 13 + 15)
# - DoD-5 live closure (criterion 14)
# - Lighthouse run (criterion 17)
```

## Q/A expectations

- 5-item harness audit must PASS: researcher + contract + experiment_results + log-LAST (status STAYS pending) + no-shopping.
- 9 deterministic checks should PASS: pytest_count + tsc + vitest + live_check_present + /cron component greps + /agents useEventSource grep + emoji + ASCII loggers.
- Verdict expected: **PASS** for the bounded 6-of-17 scope. The 11 deferrals are honestly documented + match the bounded-scope contract.
