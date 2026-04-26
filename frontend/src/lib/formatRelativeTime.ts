/**
 * phase-16.42: relative-time formatter for the home Recent Reports
 * UPDATED column. Stdlib only -- uses `Intl.RelativeTimeFormat` (Baseline
 * Widely Available since 2020).
 *
 * Renders strings like "12 min. ago", "2 hr. ago", "1 day ago", "2 days ago"
 * matching the target screenshot.
 *
 * Pure function. Caller MUST be a "use client" component to avoid
 * server/client hydration mismatch (the wall-clock differs between
 * SSR and CSR by milliseconds-to-seconds).
 */
export function formatRelativeTime(input: string | number | Date | null | undefined): string {
  if (input == null) return "—";
  const ts = input instanceof Date ? input.getTime() : new Date(input).getTime();
  if (!Number.isFinite(ts)) return "—";

  const diffMs = Date.now() - ts;
  const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto", style: "short" });

  const SEC = 1000;
  const MIN = 60 * SEC;
  const HR = 60 * MIN;
  const DAY = 24 * HR;
  const WEEK = 7 * DAY;

  const abs = Math.abs(diffMs);
  if (abs < MIN) return rtf.format(-Math.round(diffMs / SEC), "second");
  if (abs < HR) return rtf.format(-Math.round(diffMs / MIN), "minute");
  if (abs < 2 * DAY) return rtf.format(-Math.round(diffMs / HR), "hour");
  if (abs < WEEK) return rtf.format(-Math.round(diffMs / DAY), "day");
  return rtf.format(-Math.round(diffMs / WEEK), "week");
}
