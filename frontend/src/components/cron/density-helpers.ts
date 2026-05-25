// phase-44.7 -- log line density helpers.
//
// Per researcher topic 3 + AWS Cloudscape: Comfortable density default;
// Compact toggle for data-intensive views. WCAG 2.2 24px target-size
// is satisfied via the gutter rendering (not the line itself) so we
// can render 16px-tall log lines without invoking the SC 2.5.8 inline
// exception.

export type LogDensity = "comfortable" | "compact";

export const LINE_HEIGHT_CLASS: Record<LogDensity, string> = {
  comfortable: "min-h-[32px] py-1.5",
  compact: "min-h-[16px] py-0.5",
};

export const LINE_FONT_CLASS: Record<LogDensity, string> = {
  comfortable: "text-[11px] leading-snug",
  compact: "text-[10px] leading-tight",
};

// localStorage key matches the existing convention
// (`pyfinagent.sidebar.collapsedSections` etc.).
export const DENSITY_STORAGE_KEY = "pyfinagent.cron.logDensity";

export function readDensity(): LogDensity {
  if (typeof window === "undefined") return "comfortable";
  try {
    const raw = window.localStorage.getItem(DENSITY_STORAGE_KEY);
    return raw === "compact" ? "compact" : "comfortable";
  } catch {
    return "comfortable";
  }
}

export function writeDensity(d: LogDensity): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(DENSITY_STORAGE_KEY, d);
  } catch {
    /* ignore quota errors */
  }
}

// Lightweight log-line parser: extract the level token (ERROR / WARN /
// WARNING / INFO / DEBUG) if present near the start of a line. Returns
// null when no level is recognized. Used by the level-filter pills.
export type LogLevel = "ERROR" | "WARN" | "INFO" | "DEBUG" | null;

const LEVEL_PATTERN = /\b(ERROR|WARNING|WARN|INFO|DEBUG)\b/i;

export function parseLevel(line: string): LogLevel {
  const m = line.match(LEVEL_PATTERN);
  if (!m) return null;
  const t = m[1].toUpperCase();
  if (t === "ERROR") return "ERROR";
  if (t === "WARN" || t === "WARNING") return "WARN";
  if (t === "INFO") return "INFO";
  if (t === "DEBUG") return "DEBUG";
  return null;
}

export function levelColorClass(level: LogLevel): string {
  switch (level) {
    case "ERROR":
      return "text-rose-300";
    case "WARN":
      return "text-amber-300";
    case "INFO":
      return "text-sky-300";
    case "DEBUG":
      return "text-slate-500";
    default:
      return "text-slate-300";
  }
}
