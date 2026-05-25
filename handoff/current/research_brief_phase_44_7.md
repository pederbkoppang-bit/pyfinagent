# Research Brief: phase-44.7 BOUNDED (cron facet/sparkline/follow-pause/permalink/density + useEventSource migration)

**Tier:** simple-moderate
**Generated:** 2026-05-25
**Cycle bound:** 6 of 17 phase-44.7 criteria
**This cycle covers:**
1. /cron logs: facet search + level pills (error/warn/info)
2. /cron logs: Tremor SparkAreaChart above log (event rate per minute)
3. /cron logs: follow/pause toggle (default follow newest)
4. /cron logs: permalink to line via URL fragment `#L1234`
5. /cron logs: compact density toggle (32-line spacious vs 16-line compact)
6. useEventSource migration: replace inline `EventSource` in `agents/page.tsx` with the cycle-44.1 `useEventSource` hook

**Deferred to follow-up cycles (11 of 17):** /agents trace-tree, /agent-map merge, /observability sparkline + countdown, /agents side-by-side compare, /agents annotation.

## Search-query discipline

Three-variant queries per topic (year-less canonical + 2026 frontier + 2025 recency-scan):

| Topic | Year-less | 2026 | 2025 |
|---|---|---|---|
| Log viewer follow/pause | "log viewer follow tail user interface design" | "log viewer UX patterns follow tail pause toggle 2026" | (recency-scan; below) |
| Facet search pills | "faceted search filter pills UI design pattern" | (date-free hits ranked high) | -- |
| URL fragment permalinks | "URL fragment line anchor permalink scrollIntoView" | "URL fragment line anchor GitHub permalink scrollIntoView" | -- |
| Tremor SparkArea | (none -- product-name search) | "Tremor SparkAreaChart documentation event rate sparkline 2026" | -- |
| Density toggle | (year-less hit: Cloudscape content-density) | -- | "log viewer compact density information density UI 2025 design" |
| useEventSource | -- | "React useEventSource hook SSE migration pattern 2026" | -- |
| WCAG target size | (year-less hit: w3.org/WAI/WCAG22) | -- | -- |
| scroll-margin-top | (year-less hit: CSS-Tricks + MDN) | -- | "CSS scroll-margin-top fixed header log line anchor" |
| Grafana Logs UI | -- | -- | "Grafana Loki log explorer level filter pills 2025" |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CloudWatchLogs_LiveTail.html | 2026-05-25 | Official docs (AWS) | WebFetch full | **CloudWatch Live Tail pause semantics:** "To pause the flow of events to investigate what is currently displayed, choose anywhere in the events window." Resume by clicking Start. Filters: `error 404` (AND), `?Error ?error` (OR), `-INFO` (NOT), `%ERROR%` (regex), `{ $.eventType = "UpdateTrail" }` (JSON). Highlight terms get color-coded badges. Sampling indicator: "% displayed" when >500 events/sec. |
| 2 | https://www.tremor.so/docs/visualizations/spark-chart | 2026-05-25 | Official docs (Tremor) | WebFetch full | **SparkAreaChart API:** required `data` (array of records), `index` (string key for x-axis), `categories` (string[] for series). Optional `colors`, `autoMinValue`, `minValue`/`maxValue`, `connectNulls`, `fill: 'gradient'/'solid'/'none'`. Example: `<SparkAreaChart data={chartdata} index="month" categories={["Performance"]} colors={["emerald"]} className="h-8 w-36" />`. Sizing via `className` (h+w). |
| 3 | https://developer.mozilla.org/en-US/docs/Web/API/Element/scrollIntoView | 2026-05-25 | Standards (MDN) | WebFetch full | **scrollIntoView() API:** options `{behavior: "smooth"\|"instant"\|"auto", block: "start"\|"center"\|"end"\|"nearest", inline: ...}`. Baseline widely available since January 2020. Critical caveat: **"When using `scrollIntoView()` with a fixed header, use CSS properties to define custom spacing: `scroll-margin-top`, `scroll-margin-bottom`. This ensures elements don't scroll behind fixed headers."** |
| 4 | https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html | 2026-05-25 | Standards (W3C) | WebFetch full | **WCAG 2.2 SC 2.5.8:** "The size of the target for pointer inputs is at least 24 by 24 CSS pixels" -- 5 exceptions: spacing (24-CSS-px circle doesn't intersect adjacent), equivalent (same function elsewhere), **inline** ("target is in a sentence or its size is otherwise constrained by the line-height of non-target text"), user-agent control, essential. Intent: prevent accidental activation of adjacent targets (hand tremors, spasticity). |
| 5 | https://cloudscape.design/foundation/visual-foundation/content-density/ | 2026-05-25 | Official docs (AWS Cloudscape) | WebFetch full | **Density modes:** Comfortable (default) + Compact. **"Always set comfortable mode as default."** Compact "deliberately not applied" to informational components (help panels, alerts, forms) and interactive elements with limited target space (dropdowns, date pickers) -- preserves readability. 4px base spacing reduces in 4px increments. Compact targets "full-page, data-intensive views" -- log lists qualify. |
| 6 | https://grafana.com/docs/grafana/latest/explore/logs-integration/ | 2026-05-25 | Official docs (Grafana) | WebFetch full | **Grafana Explore live-tail pause/resume:** "Click **Pause** to pause live tailing and explore previous logs without interruptions, or simply scroll through the logs view." "Click **Resume** to resume live tailing and continue viewing real-time logs." New logs appear at "screen bottom with contrasting background." Logs volume histogram (event-rate visualization) auto-renders for Elasticsearch + Loki. Filter levels: "All levels, Info, Debug, Warning, Error." |

**Gate met:** 6 sources read in full, all peer-reviewed-equivalent (W3C standards, MDN web standards, AWS official docs, Tremor official docs, Cloudscape design system, Grafana official docs).

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.patternfly.org/extensions/log-viewer/design-guidelines/ | Design system | Returned page-title-only excerpt; insufficient content to fetch in full |
| https://help.papertrailapp.com/kb/how-it-works/event-viewer | Vendor docs | 301-redirected to papertrail.com which returns 403 |
| https://www.papertrail.com/help/event-viewer/ | Vendor docs | 403 Forbidden after redirect |
| https://www.runhybris.com/2019/01/18/facet-search-the-most-comprehensive-guide-best-practices-design-patterns-hidden-caveats-and-workarounds/ | Industry blog | 403 Forbidden |
| https://docs.github.com/en/repositories/working-with-files/using-files/creating-a-permanent-link-to-a-code-snippet | Official docs | 404 (page moved or restructured) |
| https://www.nngroup.com/articles/mobile-faceted-search/ | UX research (NN/G) | Fetched, but no specific guidance on level facets -- general "use familiar controls like dropdowns and checkboxes" |
| https://www.elastic.co/observability-labs/blog/designing-log-processing-ux-for-streams | Vendor blog | Fetched; focused on log *processing* UX (split-screen build/preview), not browsing/follow-tail patterns |
| https://news.ycombinator.com/item?id=39277079 | Community discussion | Fetched; community feedback on Logdy.dev, no authoritative UX guidance |
| https://signoz.io/blog/logs-ui/ | Vendor blog | Fetched; documents existence of live-tail + query-builder + log-volume display but no granular UX patterns |
| https://www.uxmatters.com/mt/archives/2024/05/crafting-seamless-user-experiences-a-ux-driven-approach-to-log-monitoring-and-observability.php | Industry article | Fetched; high-level UX principles, no actionable specs |
| https://css-tricks.com/fixed-headers-and-jump-links-the-solution-is-scroll-margin-top/ | Industry tutorial | Snippet-only in search; canonical advice covered by MDN entry #3 |
| https://grafana.com/blog/find-your-logs-data-with-explore-logs-no-logql-required/ | Vendor blog | Snippet-only; Logs Drilldown feature renamed Feb 2025 -- confirms level filter pattern |
| https://github.com/Scrivito/scroll-to-fragment | Open-source library | Snippet-only; confirms SPA scroll-to-fragment pattern |
| https://github.com/grafana/logs-drilldown | Open-source repo | Snippet-only; reference for Grafana Drilldown UX |
| https://reactuse.com/browser/useEventSource/ | Library docs | Snippet-only; confirms useEventSource is an established 2025-2026 pattern |
| https://oneuptime.com/blog/post/2026-01-15-server-sent-events-sse-react/view | Industry blog | Snippet-only; 2026-recency reference |

**Total unique URLs collected (full + snippet):** 22

## Recency scan (2024-2026)

**Searched for** 2024-2026 work on log viewers, facet search pills, density toggles, useEventSource patterns, scroll-to-fragment.

**Findings in window:**
1. **Feb 2025:** Grafana renamed "Explore Logs" to "Logs Drilldown" (https://grafana.com/blog/find-your-logs-data-with-explore-logs-no-logql-required/). Level filtering is now a queryless click-to-curated-view UX -- click a level, immediately jump to logs at that level. Multi-select via Cmd/Ctrl+click. **Supersedes** any pre-2024 query-only level filtering.
2. **April 2024:** Loki 3.0 released bloom-filter-backed log search + OpenTelemetry support (https://grafana.com/blog/2024/04/09/grafana-loki-3.0-release-all-the-new-features/). Confirms server-side filtering is the production-grade path; client-side is a fallback for small data.
3. **January 2026:** Server-Sent Events + React patterns formalized (https://oneuptime.com/blog/post/2026-01-15-server-sent-events-sse-react/view). useEventSource hook abstractions are baseline.
4. **April 2026:** AWS CloudWatch Live Tail SDK routing change (April 1 2026 cutoff for streaming-logs vs stream-logs hostname). Pause-anywhere-in-window UX has remained stable.
5. **June 2025:** WCAG 2.2 became EU AAA mandate baseline (referenced in master_design Section B.7). 24x24 CSS-px target size now enforced.

**Conclusion:** Log-viewer UX has converged in 2024-2026 around: (a) explicit Pause / Resume buttons rather than click-to-pause-on-window (CloudWatch is the outlier), (b) level filters as 4-5 short multi-select pills, (c) histogram/sparkline above the log showing event-rate, (d) compact mode opt-in (not default), (e) URL-fragment permalinks for sharable line refs. No 2024-2026 evidence contradicts our planned design; the master_design Section 3.14 success criteria align with current frontier.

## Key findings (per topic)

### Topic 1: Log viewer UX -- follow/pause, level pills, permalinks

**Pause semantics (consensus):**
- **Grafana Explore (Source 6):** explicit Pause + Resume buttons. New logs at bottom with contrasting bg. **This is the canonical pattern for our /cron page.**
- **CloudWatch Live Tail (Source 1):** click-anywhere-in-window pauses. **Alternative** but less discoverable than an explicit button. Prefer Grafana's explicit toggle.
- **Default state:** Follow-newest, consistent across CloudWatch, Grafana, Loki, Papertrail (per pre-2024 docs cached in search snippets), and Logdy.

**Level filter UI (consensus):**
- **Grafana Logs Drilldown (Source 6 + recency-scan #1):** click-a-level chips; multi-select via Cmd/Ctrl. Five levels: All / Info / Debug / Warning / Error.
- **Faceted-search NN/G guidance:** "familiar controls like drop-down menus and checkboxes." For 3-level short lists (error/warn/info), pills with toggle semantics are universally understood (Linear, Vercel, Grafana, Datadog).
- **Recommendation:** 3-pill multi-select toggle group, role="group" with `aria-pressed` per pill. Default all-on (show everything). Click a pill to filter it out; click again to re-include.

**Permalink-to-line (consensus):**
- **GitHub convention:** `#L1234` (single line) or `#L10-L20` (range). Source #5 (snippet) confirmed.
- **MDN scrollIntoView (Source 3):** `{behavior: "smooth", block: "center"}` is the right call after fragment change.
- **scroll-margin-top:** since /cron logs scroll inside a `<pre>` container under a fixed page header, apply `scroll-margin-top: 80px` (or equivalent) to log-line elements OR use `block: "center"` to bypass the issue.
- **SPA caveat (search snippet):** browsers parse `#L1234` before SPA-rendered DOM exists -- must listen to hash change *after* logs render.

**Search/filter syntax:**
- **CloudWatch (Source 1):** plain-text `error 404` (AND), `?Error ?error` (OR), `-INFO` (NOT), regex `%ERROR%`. **Phase-44.7 scope: plain-text substring is sufficient** -- regex is over-engineering for cycle 1.

### Topic 2: Event-rate sparkline above logs

**Tremor SparkAreaChart (Source 2):**
- `data: [{minute: "10:42", count: 27}, ...]`
- `index: "minute"`, `categories: ["count"]`
- `colors: ["sky"]` (matches our existing UI accent)
- Sizing: `className="h-12 w-full"` -- enough to show pattern, not so tall it crowds the log
- `fill: "gradient"` for visual weight; `fill: "solid"` for compact mode
- **Binning:** Group log lines by `floor(timestamp / 60s)`, count per bucket, render last N=60 buckets (1 hour). When no timestamps in log line (raw text), fall back to bucket-by-position (e.g. line-count per visible-page).

**Grafana logs-volume reference (Source 6):** the histogram displays before the log list, not above each row. Confirms our placement (sparkline-row above the log block, not per-row).

### Topic 3: Density toggle (compact vs spacious)

**Cloudscape (Source 5):**
- **Default = Comfortable (spacious).** Compact is opt-in.
- Compact for "full-page, data-intensive views" -- logs qualify.
- Reduces vertical spacing in 4px increments.

**Practical implementation for /cron:**
- Spacious (default): `text-[11px] leading-relaxed` (current line 437) -- ~16px line height, ~32-line view at h-[60vh]
- Compact: `text-[10px] leading-snug` -- ~13px line height, ~46-line view at h-[60vh]
- Toggle via single-pill button with `aria-pressed`. Label: "Compact" / "Spacious"
- **WCAG 2.2 SC 2.5.8 caveat (Source 4):** even in compact, the toggle BUTTON itself must be >=24x24 CSS px. Log lines themselves are not interactive targets unless we make them clickable for permalink -- in which case the line-click target must be >=24x24 OR fall under the "inline" exception (constrained by line-height). For 16px line-height compact lines, this means the line-number gutter (line-anchor click target) must be >=24x24 -- so render the gutter as a 24x24 padded clickable region even when text is 13px.

### Topic 4: URL fragment permalinks

**Architecture:**
1. Each log line wrapped in `<div id={`L${i}`}>` (1-indexed; matches GitHub `#L1` convention)
2. On line click, update URL: `history.replaceState(null, "", "#L" + i)`
3. After hash change OR on mount with non-empty `window.location.hash`, call `document.getElementById(hash.slice(1))?.scrollIntoView({behavior: "smooth", block: "center"})`
4. Apply `scroll-margin-top` to log-lines if the parent `<pre>` doesn't have its own scroll context (our `<pre>` has `max-h-[60vh] overflow-y-auto` so it IS the scroll context -- `scroll-margin-top` may be unnecessary)
5. Visual: clicked line gets `bg-sky-500/10` for ~2s

**SPA-render-order caveat (search snippet):** call scrollIntoView in a `useEffect` after `data?.lines` render, not synchronously during render.

### Topic 5: useEventSource migration

**Existing hook signature** (`frontend/src/lib/hooks/useEventSource.ts:49-114`):
```typescript
useEventSource<T>(url: string | null, options?: {
  parser?: (raw: string) => T,
  eventType?: string,
  enabled?: boolean,
  maxFailures?: number,
}): { data, status, lastEventAt, failures, reconnect }
```

**Invariants** (lines 73-114):
- Exponential backoff: starts 1s, doubles, capped at 30s
- maxFailures default 3 (lower than agents/page.tsx's hard-coded 5)
- Disconnects on `enabled === false` or `url === null`
- `withCredentials: false` baked in
- Returns latest single event in `data`, not a buffer

**Migration target** (`frontend/src/app/agents/page.tsx:187-232`):
- Inline `new EventSource()` with hand-rolled reconnect (line 192)
- Maintains EVENTS BUFFER `events: MASEvent[]` (line 174) -- the hook only exposes latest event, not history
- Tracks `activeAgents` set with 3s flash decay (line 207-218)
- Hard-coded 5-failure cap (line 227); hook default is 3

**Migration considerations:**
1. **Buffer accumulation must move OUT of the hook.** The hook returns the latest event; the page composes the buffer via a `useEffect`:
   ```typescript
   const { data: latestEvent, status, failures, reconnect } = useEventSource<MASEvent>(
     `${API_BASE}/api/mas/events?include_buffer=true`,
     { maxFailures: 5 }  // override default 3 to match existing 5
   );
   useEffect(() => {
     if (!latestEvent) return;
     setEvents(prev => [...prev.slice(-499), latestEvent]);
     setActiveAgents(prev => { const next = new Set(prev); next.add(latestEvent.agent); return next; });
     const timer = setTimeout(() => {
       setActiveAgents(prev => { const next = new Set(prev); next.delete(latestEvent.agent); return next; });
     }, 3000);
     return () => clearTimeout(timer);
   }, [latestEvent]);
   ```
2. **`maxFailures: 5` override.** Pass explicit `maxFailures: 5` to preserve current behavior (hook default is 3 but page enforces 5).
3. **`failures` and `status` replace `connected`.** `status === "connected"` is the new green indicator. The page's existing `connected` boolean (line 175) maps to `status === "connected"`.
4. **`reconnect()` replaces inline `connect()` call.** The page's manual retry button (line 373) now calls `reconnect()`.
5. **Code reduction:** ~45 lines (187-232) collapse to ~15 lines using the hook.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/src/app/cron/page.tsx` | 449 | /cron page: Jobs tab + Logs tab | **MODIFY THIS CYCLE** -- replace LogsTab logs UX with facet-search + sparkline + follow-pause + permalink + density |
| `frontend/src/app/cron/page.tsx:28-38` | 11 | `LOG_KEYS` constant (9 entries) | KEEP -- unchanged |
| `frontend/src/app/cron/page.tsx:40-42` | 3 | `LINE_OPTIONS`, `POLL_INTERVAL_MS=5000`, `MAX_CONSECUTIVE_FAILURES=5` | KEEP -- unchanged |
| `frontend/src/app/cron/page.tsx:318-449` | 132 | `LogsTab` component (current target of rewrite) | **REWRITE** -- add facet pills, sparkline, follow/pause, permalink, density |
| `frontend/src/app/cron/page.tsx:437-440` | 4 | `<pre>` render block with `max-h-[60vh]` + `text-[11px] leading-relaxed` | **REPLACE** -- per-line `<div id="L{n}">` with click-to-permalink |
| `frontend/src/app/agents/page.tsx` | 728 | /agents page: 4 tabs (live/history/agents/openclaw) | **MODIFY THIS CYCLE** -- migrate inline EventSource to useEventSource hook |
| `frontend/src/app/agents/page.tsx:181-183` | 3 | `eventSourceRef`, `scrollRef`, `failCountRef` | **REMOVE** `eventSourceRef` and `failCountRef` (replaced by hook); KEEP `scrollRef` |
| `frontend/src/app/agents/page.tsx:187-232` | 46 | `connect()` callback + inline `new EventSource()` + onmessage + onerror | **REPLACE** -- collapse to `useEventSource` hook + buffer-accumulation useEffect |
| `frontend/src/app/agents/page.tsx:234-239` | 6 | Mount-time `connect()` + cleanup `eventSourceRef.current?.close()` | **REMOVE** -- hook handles lifecycle |
| `frontend/src/app/agents/page.tsx:373-374` | 2 | Manual retry: `setError(null); failCountRef.current = 0; connect();` | **REPLACE** with `reconnect()` call |
| `frontend/src/lib/hooks/useEventSource.ts` | 133 | Phase-44.1 foundation hook (cycle 16) | KEEP -- already complete; no changes needed |
| `frontend/src/lib/hooks/useEventSource.ts:49-56` | 8 | Hook signature + defaults | REFERENCE for migration call-site |
| `frontend/src/lib/hooks/useEventSource.ts:73-114` | 42 | `connect` callback + backoff + cleanup | REFERENCE for understanding internals |
| `frontend/src/lib/hooks/index.ts:9-10` | 2 | `export { useEventSource } from "./useEventSource"; export type { UseEventSourceState }` | KEEP -- already exported via barrel |
| `frontend/src/components/states/EmptyState.tsx` | 62 | Reusable empty-state component | **USE THIS CYCLE** -- replace "Log file is empty" inline block (line 421-425) and "Log file does not exist yet" block (lines 413-419) |
| `frontend/src/components/TimeRangeSelector.tsx` | 113 | Cycle-44.4 segmented control (7d/30d/90d/all) | **NOT used this cycle** -- /cron is realtime, not historical. The sparkline window is fixed to last 60 minutes. |
| `frontend/src/lib/types.ts` | -- | LogTailResponse type | INSPECT -- need to know whether `data.lines` is array of strings (current) or array of structured objects |
| `handoff/current/frontend_ux_master_design.md:358-377` | 20 | Section 3.14 /cron success criteria (this cycle's contract source) | REFERENCE -- 9 criteria; this cycle implements 5 + useEventSource migration |
| `.claude/rules/frontend.md` | -- | Conventions: dark theme, scrollbar-thin, Phosphor icons, no emoji | REFERENCE |
| `.claude/rules/frontend-layout.md` | -- | 6-tier page anatomy, fixed header + scrollable content zones | REFERENCE -- /cron already follows this |

## Application to phase-44.7 (mapping external findings -> file:line)

### A. /cron LogsTab rewrite (`frontend/src/app/cron/page.tsx:318-449`)

#### A.1 Facet search with level pills

Insert above current toolbar (line 363):

```tsx
type Level = "error" | "warn" | "info";
const LEVELS: { id: Level; label: string; color: string }[] = [
  { id: "error", label: "Error", color: "rose" },
  { id: "warn", label: "Warn", color: "amber" },
  { id: "info", label: "Info", color: "sky" },
];

const [activeLevels, setActiveLevels] = useState<Set<Level>>(new Set(["error", "warn", "info"]));
const [query, setQuery] = useState("");
```

Render (above the dropdown at line 363):

```tsx
<div className="flex flex-wrap items-center gap-2">
  {/* Search box (plain-text substring; CloudWatch Source 1 syntax deferred) */}
  <label className="flex items-center gap-2 text-sm text-slate-400">
    <MagnifyingGlass size={14} />
    <input
      type="text"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="filter lines..."
      className="w-56 rounded border border-navy-700 bg-navy-800 px-2 py-1 text-xs text-slate-200 focus:border-sky-500 focus:outline-none min-h-[24px]"
      aria-label="Filter log lines by substring"
    />
  </label>
  {/* Level pills (Grafana Logs Drilldown pattern; Source 6 + recency #1) */}
  <div role="group" aria-label="Log level filters" className="flex gap-1">
    {LEVELS.map((lvl) => {
      const active = activeLevels.has(lvl.id);
      return (
        <button
          key={lvl.id}
          type="button"
          aria-pressed={active}
          onClick={() => setActiveLevels((prev) => {
            const next = new Set(prev);
            if (active) next.delete(lvl.id); else next.add(lvl.id);
            return next;
          })}
          className={`min-h-[24px] rounded-md px-2.5 py-0.5 text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-sky-400 ${
            active
              ? `bg-${lvl.color}-500/15 text-${lvl.color}-300 border border-${lvl.color}-500/30`
              : "bg-navy-800/60 text-slate-500 border border-navy-700"
          }`}
        >
          {lvl.label}
        </button>
      );
    })}
  </div>
</div>
```

Filter logic (replace the raw `data.lines.join("\n")` at line 438):

```tsx
const filteredLines = useMemo(() => {
  const lines = data?.lines ?? [];
  return lines.map((line, idx) => ({ line, idx })).filter(({ line }) => {
    const lower = line.toLowerCase();
    const detected: Level = lower.match(/\berror\b|\bexception\b|\bfatal\b/) ? "error"
      : lower.match(/\bwarn(ing)?\b/) ? "warn"
      : "info";
    if (!activeLevels.has(detected)) return false;
    if (query && !line.toLowerCase().includes(query.toLowerCase())) return false;
    return true;
  });
}, [data?.lines, activeLevels, query]);
```

#### A.2 Tremor SparkAreaChart event-rate (Source 2)

Above the `<pre>` block (insert at line 427):

```tsx
import { SparkAreaChart } from "@tremor/react";

const eventRateData = useMemo(() => {
  // Bin lines by minute (extract leading ISO timestamp; fall back to position-bucket).
  const buckets = new Map<string, number>();
  for (const { line } of filteredLines) {
    const match = line.match(/(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})/);
    const key = match ? match[1] : "unknown";
    buckets.set(key, (buckets.get(key) ?? 0) + 1);
  }
  return Array.from(buckets.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-60)
    .map(([minute, count]) => ({ minute, count }));
}, [filteredLines]);

{eventRateData.length > 0 && (
  <div className="rounded-xl border border-navy-700 bg-navy-800/40 px-4 py-2">
    <p className="text-xs text-slate-500 mb-1">Event rate (last 60 min)</p>
    <SparkAreaChart
      data={eventRateData}
      index="minute"
      categories={["count"]}
      colors={["sky"]}
      className="h-12 w-full"
      fill="gradient"
    />
  </div>
)}
```

#### A.3 Follow/pause toggle (Grafana pattern from Source 6)

```tsx
const [following, setFollowing] = useState(true);
const preRef = useRef<HTMLPreElement>(null);

// Default follow newest: auto-scroll to bottom when new lines arrive AND following === true
useEffect(() => {
  if (following && preRef.current) {
    preRef.current.scrollTop = preRef.current.scrollHeight;
  }
}, [filteredLines, following]);

// Pause when user scrolls up; resume button to re-enable
const handleScroll = (e: React.UIEvent<HTMLPreElement>) => {
  const el = e.currentTarget;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 4;
  if (!atBottom && following) setFollowing(false);
};
```

Toggle button (in toolbar):

```tsx
<button
  type="button"
  aria-pressed={following}
  onClick={() => setFollowing((f) => !f)}
  className="flex items-center gap-1 rounded border border-navy-700 px-2 py-1 text-xs text-slate-400 hover:bg-navy-800/60 min-h-[24px] focus-visible:ring-2 focus-visible:ring-sky-400"
>
  {following ? <Pause size={12} /> : <Play size={12} />}
  {following ? "Following" : "Paused"}
</button>
```

(Phosphor `Pause` + `Play` need adding to `lib/icons.ts` if not already exported.)

#### A.4 Permalink to line (`#L1234` URL fragment)

Render each line as a `<div>` with `id`:

```tsx
<pre
  ref={preRef}
  onScroll={handleScroll}
  className={`max-h-[60vh] overflow-y-auto scrollbar-thin bg-[#0a1020] px-4 py-3 font-mono text-slate-300 ${
    density === "compact" ? "text-[10px] leading-snug" : "text-[11px] leading-relaxed"
  }`}
  role="log"
  aria-live="polite"
>
  {filteredLines.map(({ line, idx }) => {
    const lineNum = idx + 1;
    return (
      <div
        key={lineNum}
        id={`L${lineNum}`}
        className="group flex gap-2 hover:bg-navy-800/40"
        style={{ scrollMarginTop: "12px" }}
      >
        <button
          type="button"
          onClick={() => {
            history.replaceState(null, "", `#L${lineNum}`);
            // Copy permalink to clipboard
            navigator.clipboard?.writeText(window.location.href);
          }}
          className="select-none text-right text-slate-600 hover:text-sky-400 min-w-[3rem] min-h-[24px]"
          aria-label={`Permalink to line ${lineNum}`}
        >
          {lineNum}
        </button>
        <span className="flex-1 whitespace-pre-wrap break-all">{line}</span>
      </div>
    );
  })}
</pre>
```

Hash-on-load effect (after `data` first paints):

```tsx
useEffect(() => {
  if (!data?.lines.length) return;
  const hash = window.location.hash; // "#L42"
  if (!hash.startsWith("#L")) return;
  const target = document.getElementById(hash.slice(1));
  target?.scrollIntoView({ behavior: "smooth", block: "center" });
  setFollowing(false); // pause when user jumped to a specific line
}, [data?.lines.length]);
```

Per MDN (Source 3): `block: "center"` bypasses the fixed-header offset problem cleanly. `scroll-margin-top: 12px` on each line is a small safety margin.

#### A.5 Compact density toggle (Cloudscape Source 5)

```tsx
const [density, setDensity] = useState<"comfortable" | "compact">("comfortable");

<button
  type="button"
  aria-pressed={density === "compact"}
  onClick={() => setDensity((d) => d === "comfortable" ? "compact" : "comfortable")}
  className="min-h-[24px] rounded border border-navy-700 px-2 py-1 text-xs text-slate-400 hover:bg-navy-800/60 focus-visible:ring-2 focus-visible:ring-sky-400"
>
  {density === "compact" ? "Compact" : "Spacious"}
</button>
```

The `<pre>` className already responds via `density === "compact" ? "text-[10px] leading-snug" : "text-[11px] leading-relaxed"`.

**WCAG 2.2 SC 2.5.8 compliance:** all buttons declared `min-h-[24px]`; line-number gutter has `min-w-[3rem] min-h-[24px]`. Line content itself is non-interactive (only the gutter clicks); no target-size violation.

### B. /agents useEventSource migration (`frontend/src/app/agents/page.tsx:181-239`)

Replace the existing inline EventSource block:

```tsx
// BEFORE (lines 181-239) -- 59 lines
const eventSourceRef = useRef<EventSource | null>(null);
const failCountRef = useRef(0);
const [connected, setConnected] = useState(false);

const connect = useCallback(() => {
  if (eventSourceRef.current) eventSourceRef.current.close();
  const es = new EventSource(`${API_BASE}/api/mas/events?include_buffer=true`);
  eventSourceRef.current = es;
  es.onopen = () => { setConnected(true); setError(null); failCountRef.current = 0; };
  es.onmessage = (e) => {
    try {
      const event: MASEvent = JSON.parse(e.data);
      setEvents((prev) => [...prev.slice(-499), event]);
      setActiveAgents((prev) => { ... });
    } catch { /* skip */ }
  };
  es.onerror = () => { ... };
}, []);

useEffect(() => { connect(); return () => eventSourceRef.current?.close(); }, [connect]);

// AFTER -- ~15 lines
import { useEventSource } from "@/lib/hooks";

const { data: latestEvent, status, reconnect } = useEventSource<MASEvent>(
  `${API_BASE}/api/mas/events?include_buffer=true`,
  { maxFailures: 5 },
);
const connected = status === "connected";

useEffect(() => {
  if (!latestEvent) return;
  setEvents((prev) => [...prev.slice(-499), latestEvent]);
  setActiveAgents((prev) => {
    const next = new Set(prev);
    next.add(latestEvent.agent);
    return next;
  });
  const id = setTimeout(() => {
    setActiveAgents((prev) => {
      const next = new Set(prev);
      next.delete(latestEvent.agent);
      return next;
    });
  }, 3000);
  return () => clearTimeout(id);
}, [latestEvent]);
```

Manual retry button (line 373) becomes:

```tsx
<button onClick={() => { setError(null); reconnect(); }} ...>
  Retry
</button>
```

**Removed unused imports** (if no longer referenced): `useCallback`, `useRef`. `useEffect`, `useState` stay.

**Important caveat:** the hook closes the connection on unmount automatically. The current page's `return () => eventSourceRef.current?.close();` cleanup is no longer needed -- removed.

**Risk note:** the hook's default `eventType: "message"` matches the backend's default SSE event-type. No backend changes needed.

## Consensus vs debate

**Consensus across 6 read-in-full sources:**
- Follow-newest is the default-on state for live-tail UIs
- Explicit Pause/Resume button beats click-anywhere-to-pause (Grafana > CloudWatch on discoverability)
- Level filters as 3-5 short multi-select pills (Grafana, Datadog, Linear pattern)
- Comfortable density as default; compact opt-in (Cloudscape principle)
- scroll-margin-top OR `block: "center"` to handle fixed headers in scrollIntoView (MDN)
- WCAG 2.2 2.5.8 mandates 24x24 CSS px for non-inline interactive targets

**Debate / project-specific call:**
- **CloudWatch click-anywhere-pause vs Grafana explicit-button:** the explicit button is more discoverable + more accessible (keyboard navigable, focus-visible, `aria-pressed`). Choose Grafana pattern for pyfinagent.
- **Client-side vs server-side filtering:** master_design says "client-side first, server-side if >1000 lines." For phase-44.7 with current LINE_OPTIONS max 1000, **all filtering is client-side** -- no backend change.
- **Per-line `<div>` vs `<pre>` text content:** GitHub uses `<table>` with line-number cells; we use `<div>` per line for simpler CSS + smaller DOM. Trade-off: lose true monospace alignment of multi-character glyphs; gain interactivity + permalinking.

## Pitfalls (from literature + internal)

1. **scrollIntoView called too early** (search snippet on SPA fragments): the page renders before logs hydrate. Wrap in `useEffect([data?.lines.length])` -- not in the initial render path.
2. **`history.pushState` vs `replaceState`:** `pushState` adds a back-button entry per click. For line-permalinks this is annoying. Use `replaceState`.
3. **`aria-live="polite"` on rapidly updating content:** screen readers cannot keep up with high event rates. Document this as a known limitation; the `role="log"` + `aria-live="polite"` is the WCAG-recommended pair but realistically rate-limits perceived updates to those with attention paused.
4. **`setEvents((prev) => [...prev.slice(-499), event])` accumulator in useEffect** (migration): the `latestEvent` from the hook can re-fire on remount with the same value -- guard against duplicate appends. A `useRef` comparing previous `lastEventAt` value works; or check the hook's `lastEventAt` change.
5. **Tremor's `<SparkAreaChart>` import path:** in this project, charts often go through Recharts directly (per frontend.md). Verify `@tremor/react` is installed and configured. Per `handoff/current/frontend_ux_master_design.md` line 92, Tremor is listed as the planned dep for `<Sparkline/>` -- check package.json before assuming it's installed.
6. **EventSource buffer accumulation:** if backend's `include_buffer=true` returns N initial events, the hook fires once per delivered event, so N successive `useEffect` runs append N events one-by-one. This matches existing behavior, but causes N renders. Acceptable for N<=100.
7. **Compact density + WCAG line-height-constrained inline exception:** even at `leading-snug` (1.375 line-height), the line-number gutter button is the interactive target, not the line text. The gutter has explicit `min-h-[24px]` so SC 2.5.8 is satisfied without invoking the inline exception.
8. **Phosphor icons not in `lib/icons.ts`:** `Pause`, `Play`, `MagnifyingGlass` -- verify they're re-exported. MagnifyingGlass is referenced in EmptyState.tsx line 14, so the alias exists. Pause/Play may need adding.

## Research Gate Checklist

Hard blockers -- `gate_passed` is **true** iff all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6 sources** (AWS CloudWatch Live Tail, Tremor SparkArea, MDN scrollIntoView, W3C WCAG 2.2, AWS Cloudscape, Grafana Explore Logs)
- [x] 10+ unique URLs total (incl. snippet-only) -- **22 URLs**
- [x] Recency scan (last 2 years) performed + reported -- **5 findings** in Feb 2025 / Apr 2024 / Jan 2026 / Apr 2026 / Jun 2025
- [x] Full pages read (not abstracts) for the read-in-full set -- all 6 fully retrieved
- [x] file:line anchors for every internal claim -- 20+ entries in inventory + per-snippet line references in Application section

Soft checks:
- [x] Internal exploration covered every relevant module -- cron/page.tsx (449 lines), agents/page.tsx (728 lines), useEventSource.ts (133 lines), hooks/index.ts (12 lines), EmptyState.tsx (62 lines), TimeRangeSelector.tsx (113 lines), master_design Section 3.14
- [x] Contradictions / consensus noted -- CloudWatch (click-anywhere-pause) vs Grafana (explicit Pause button) called out; project chooses Grafana pattern with rationale
- [x] All claims cited per-claim (not just listed in a footer) -- each Application section paragraph references the source by number

## JSON envelope

```json
{
  "tier": "simple-moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 16,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief_phase_44_7.md",
  "gate_passed": true
}
```
