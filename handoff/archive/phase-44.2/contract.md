# Contract -- phase-44.7 (BOUNDED) /cron + useEventSource

**Step id:** 44.7
**Cycle:** 66 (2026-05-25)
**Hypothesis:** phase-44.7 has 17 criteria across 4 routes (/agents, /agent-map, /cron, /observability). This cycle bounds scope to /cron route (criteria 7-11) + useEventSource hook migration (criterion 16) = 6 of 17 criteria. The remaining 11 criteria land in follow-up cycles to avoid the >3-cycles-per-step circuit breaker. Step status STAYS `pending` until all 17 close.

## Research gate

- Researcher subagent `a3e5ab01ef4a068c0`, tier=simple-moderate, executed 2026-05-25.
- External sources read in full: **6** (>= 5 floor). All tier-2 (AWS CloudWatch Logs Live Tail + Tremor SparkArea + MDN scrollIntoView + W3C WCAG 2.2 + AWS Cloudscape + Grafana Explore Logs).
- Snippet-only: 16. URLs: 22.
- Recency scan (2024-2026): performed; 5 findings noted (Grafana Logs Drilldown rename, Loki 3.0, useEventSource Jan 2026, CloudWatch SDK Apr 2026, WCAG 2.2 EU AAA Jun 2025).
- 3-variant query discipline across 5 topics.
- Internal codebase audit: 9 file:line entries.
- **gate_passed: true.**
- Brief: `handoff/current/research_brief_phase_44_7.md`.

## North-star (N*) delta

- **B (Burn) primary:** -46 LoC inline EventSource boilerplate (agents/page.tsx) consolidated into the cycle-16 useEventSource foundation. Faceted-log-search reduces operator time-to-issue per incident.
- **R (Risk) speculative:** Follow/pause toggle + permalinks improve incident-investigation speed (operator can pause auto-scroll, share a specific line).
- **P (Profit) speculative:** marginal -- /cron is a debugging surface, not a signal-generation surface.

## Scope (code work) -- 6 of 17 criteria

| # | Criterion | This cycle? | Approach |
|---|-----------|-------------|----------|
| 7 | cron_logs_facet_search_with_level_pills_error_warn_info | YES | Existing facet input + 3 new level pills (ERROR / WARN / INFO) with `role="group"` + `aria-pressed`. Grafana-style queryless filter. |
| 8 | cron_logs_sparkline_above_log_event_rate_per_minute_tremor | YES | Tremor SparkAreaChart binned by minute (last 60 buckets) above the log container. |
| 9 | cron_logs_follow_pause_toggle_default_follow_newest | YES | Explicit `<button>` toggle (Grafana-pattern) NOT click-anywhere (CloudWatch). Default = follow (auto-scroll to newest). Auto-pause when user scrolls up. |
| 10 | cron_logs_permalink_to_line_via_url_fragment_L1234 | YES | Click a log line -> `replaceState` with `#L1234` + `scrollIntoView({block: "center"})`. Initial-load `useEffect` reads hash + scrolls. |
| 11 | cron_logs_compact_density_toggle_32_line_vs_16_line | YES | Spacious (32px / Comfortable per Cloudscape) by default; Compact (16px) toggle. WCAG 2.2 24px target-size satisfied via gutter `min-h-[24px]`. |
| 16 | useEventSource_shared_hook_replaces_inline_EventSource | YES | Migrate `frontend/src/app/agents/page.tsx:181-239` (~46 LoC) to use the foundation hook. `maxFailures: 5` preserved. |

## Out of scope this cycle (11 criteria honest-deferred to follow-up)

| # | Criterion | Why deferred |
|---|-----------|--------------|
| 1 | agents_Live_Stream_uses_TraceTree_grouped_by_run_id_tool_call_nested | Heavy new component (TraceTree); separate cycle |
| 2 | agents_severity_filter_pills_error_warning_info | Mirror of /cron criterion 7 pattern; defer to /agents cycle |
| 3 | agents_side_by_side_compare_via_Drawer_with_diff_highlighting | New Drawer + diff component |
| 4 | agents_annotation_queue_persists_to_BQ_via_new_endpoint | NEW BACKEND ENDPOINT (operator-side BQ migration) |
| 5 | agents_Agent_Map_tab_removed_users_redirected_to_agent_map_route | Operator habit change; needs approval row |
| 6 | agent_map_page_gains_header_last_updated_per_agent_drawer | /agent-map page refresh; separate cycle |
| 12 | observability_per_source_7d_sparkline_column | /observability refresh; separate cycle |
| 13 | observability_TimeRangeSelector_7d_30d | /observability refresh |
| 14 | observability_zero_unknown_bands_closes_DoD_5 | LIVE-CYCLE-BOUND (DoD-5 needs operator paper-trading run) |
| 15 | observability_cross_link_to_cron_logs_filtered_to_source | depends on criterion 7 + new /observability refresh |
| 17 | Lighthouse_a11y_at_least_95_on_all_four_pages | Operator-side Lighthouse |

## Plan steps

1. Extend `useEventSource` if needed for agents-specific shape; otherwise consume as-is.
2. Migrate `agents/page.tsx` inline EventSource -> useEventSource hook.
3. New `<LevelFilterPills>` component (3 pills, role=group + aria-pressed) or inline in /cron.
4. New `<LogEventRateSpark>` component using Tremor SparkAreaChart on /cron.
5. New `<FollowPauseToggle>` button on /cron.
6. /cron permalink: click handler + initial-load hash-read effect.
7. /cron density toggle.
8. Vitest coverage for new components + migration smoke.
9. Verify all gates.

## Files planned

NEW:
- `frontend/src/components/cron/LevelFilterPills.tsx` + `.test.tsx`
- `frontend/src/components/cron/LogEventRateSpark.tsx` + `.test.tsx`
- `frontend/src/components/cron/FollowPauseToggle.tsx` + `.test.tsx`
- `frontend/src/components/cron/density-helpers.ts` (compact vs spacious class tokens)
- `handoff/current/live_check_44.7.md`

MODIFIED:
- `frontend/src/app/cron/page.tsx` (mount new components + level filter + permalink effect + density toggle + follow-pause wiring)
- `frontend/src/app/agents/page.tsx` (useEventSource migration)

ZERO backend changes.

## Verification command (immutable per masterplan)

```
test -f handoff/current/live_check_44.7.md
```

Single-gate -- file created this cycle. BUT status flip requires ALL 17 criteria pass; this cycle delivers 6 of 17, so status STAYS `pending` with the live_check capturing per-criterion verdict. Follow-up cycles close the remaining 11.

## /goal integration-gate plan

| # | Gate | Plan |
|---|------|------|
| 1 | pytest >= 614 + 126 frontend | Run both; no backend. |
| 2 | TS build + ast.parse green | tsc + build. |
| 3 | Feature behind flag default OFF | N/A (refactor + per-master_design UX). |
| 4-5 | N/A | |
| 6 | Contract has N* delta | DONE. |
| 7 | Zero emojis | Grep. |
| 8 | ASCII loggers | N/A. |
| 9 | Single source of truth | useEventSource reused; Tremor SparkArea reused. |
| 10 | log first / flip last | YES — status STAYS pending this cycle. |

## Sign-off

Authored AFTER researcher gate_passed=true. Bounded scope explicitly documented. 6/17 criteria close this cycle; 11/17 honest-deferred to follow-up cycles. Status remains `pending`.
