"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import {
  Clock,
  ArrowsClockwise,
  Warning,
  Check,
  CalendarBlank,
  FileText,
} from "@/lib/icons";
import { getAllJobs, getLogTail } from "@/lib/api";
import type { JobInfo, LogTailResponse } from "@/lib/types";

// ── Constants ───────────────────────────────────────────────────

type TabId = "jobs" | "logs";

const TABS: { id: TabId; label: string; icon: typeof Clock }[] = [
  { id: "jobs", label: "Jobs", icon: CalendarBlank },
  { id: "logs", label: "Logs", icon: FileText },
];

const LOG_KEYS: { key: string; label: string }[] = [
  { key: "backend",             label: "backend.log" },
  { key: "watchdog",            label: "backend-watchdog.log" },
  { key: "restart",             label: "backend-restart.log" },
  { key: "harness",             label: "mas-harness.log" },
  { key: "autoresearch",        label: "autoresearch.log" },
  { key: "mas_harness_launchd", label: "mas-harness.launchd.log" },
];

const LINE_OPTIONS = [50, 100, 200, 500, 1000];
const POLL_INTERVAL_MS = 5000;
const MAX_CONSECUTIVE_FAILURES = 5;

// ── Helpers ─────────────────────────────────────────────────────

function formatRelative(iso: string | null): string {
  if (!iso) return "--";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const delta = (d.getTime() - Date.now()) / 1000;
    const abs = Math.abs(delta);
    const sign = delta >= 0 ? "in " : "";
    const suffix = delta >= 0 ? "" : " ago";
    if (abs < 60) return `${sign}${Math.round(abs)}s${suffix}`;
    if (abs < 3600) return `${sign}${Math.round(abs / 60)}m${suffix}`;
    if (abs < 86400) return `${sign}${Math.round(abs / 3600)}h${suffix}`;
    return `${sign}${Math.round(abs / 86400)}d${suffix}`;
  } catch {
    return iso;
  }
}

function statusClasses(status: string): string {
  switch (status) {
    case "scheduled":
      return "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";
    case "paused":
      return "bg-amber-500/15 text-amber-300 border border-amber-500/30";
    case "manifest":
      return "bg-slate-500/15 text-slate-400 border border-slate-500/30";
    default:
      return "bg-slate-700/30 text-slate-400 border border-slate-600";
  }
}

function sourceLabel(source: string): string {
  switch (source) {
    case "main_apscheduler": return "main";
    case "slack_bot":         return "slack-bot";
    case "launchd":           return "launchd";
    default:                  return source;
  }
}

// ── Page ────────────────────────────────────────────────────────

export default function CronPage() {
  const [tab, setTab] = useState<TabId>("jobs");

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Fixed header zone */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-100">Cron / Logs</h2>
              <p className="text-sm text-slate-500">
                All scheduled jobs across processes + recent log output. Read-only.
              </p>
            </div>
          </div>

          <div className="mb-4 flex gap-1 rounded-lg bg-navy-800/60 p-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                  tab === t.id
                    ? "bg-sky-500/10 text-sky-400"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <t.icon size={16} weight={tab === t.id ? "fill" : "regular"} />
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {tab === "jobs" && <JobsTab />}
          {tab === "logs" && <LogsTab />}
        </div>
      </main>
    </div>
  );
}

// ── Jobs tab ────────────────────────────────────────────────────

function JobsTab() {
  const [jobs, setJobs] = useState<JobInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const failuresRef = useRef(0);
  const stoppedRef = useRef(false);

  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      const body = await getAllJobs();
      setJobs(body.jobs);
      setGeneratedAt(body.generated_at);
      setError(null);
      failuresRef.current = 0;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      failuresRef.current += 1;
      if (failuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        stoppedRef.current = true;
        setError(
          `Jobs polling stopped after ${MAX_CONSECUTIVE_FAILURES} consecutive failures. Last error: ${msg}`,
        );
      } else {
        setError(msg);
      }
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(() => {
      if (!stoppedRef.current) load();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [load]);

  // phase-23.2.24: Rules-of-Hooks fix. useMemo MUST be called on every
  // render -- it cannot live below conditional early returns. Tolerate
  // jobs === null (return {}); the early-return branches below will then
  // skip the rendering of `grouped` entirely.
  const grouped = useMemo(() => {
    const out: Record<string, JobInfo[]> = {};
    for (const j of jobs ?? []) {
      (out[j.source] ??= []).push(j);
    }
    return out;
  }, [jobs]);

  if (jobs === null && error === null) {
    return (
      <div className="flex items-center gap-3 py-12 text-slate-400">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
        Loading jobs...
      </div>
    );
  }

  if (error && jobs === null) {
    return (
      <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-4">
        <div className="flex items-start gap-2">
          <Warning size={18} className="mt-0.5 flex-shrink-0 text-rose-400" />
          <div>
            <p className="text-sm font-medium text-rose-300">Failed to load jobs</p>
            <p className="mt-1 text-xs text-rose-400/80">{error}</p>
            <button
              type="button"
              onClick={() => {
                stoppedRef.current = false;
                failuresRef.current = 0;
                load();
              }}
              className="mt-3 rounded border border-rose-500/40 px-3 py-1 text-xs text-rose-300 hover:bg-rose-900/40"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (jobs && jobs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <CalendarBlank size={48} weight="duotone" className="text-slate-600" />
        <p className="mt-4 text-lg text-slate-400">No jobs reported</p>
        <p className="mt-1 text-sm text-slate-600">
          Backend may be starting; refresh in a few seconds.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>
          {jobs?.length ?? 0} jobs reported
          {generatedAt ? ` -- ${formatRelative(generatedAt)}` : ""}
        </span>
        <button
          type="button"
          onClick={load}
          disabled={refreshing}
          className="flex items-center gap-1 rounded border border-navy-700 px-2 py-1 text-slate-400 hover:bg-navy-800/60 disabled:opacity-40"
        >
          <ArrowsClockwise size={12} weight={refreshing ? "fill" : "regular"} />
          {refreshing ? "Refreshing" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-950/30 p-3">
          <p className="text-xs text-amber-300">
            Background refresh failed (showing last successful snapshot): {error}
          </p>
        </div>
      )}

      {Object.entries(grouped).map(([source, sourceJobs]) => (
        <section
          key={source}
          className="overflow-hidden rounded-xl border border-navy-700"
        >
          <header className="border-b border-navy-700 bg-navy-800/80 px-4 py-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              {sourceLabel(source)} -- {sourceJobs.length} job{sourceJobs.length === 1 ? "" : "s"}
            </h3>
          </header>
          <table className="w-full text-left text-sm">
            <thead className="border-b border-navy-700 bg-navy-800/40 text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-2 font-medium">Job</th>
                <th className="px-4 py-2 font-medium">Schedule</th>
                <th className="px-4 py-2 font-medium">Next run</th>
                <th className="px-4 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-700/50">
              {sourceJobs.map((j) => (
                <tr key={`${j.source}-${j.id}`} className="transition-colors hover:bg-navy-700/40">
                  <td className="px-4 py-3">
                    <div className="font-mono text-slate-200">{j.id}</div>
                    {j.description && j.description !== j.id && (
                      <div className="mt-0.5 text-xs text-slate-500">{j.description}</div>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-400">
                    {j.schedule}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-400">
                    {j.next_run ? (
                      <span title={j.next_run}>{formatRelative(j.next_run)}</span>
                    ) : (
                      <span className="text-slate-600">--</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${statusClasses(j.status)}`}
                    >
                      {j.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  );
}

// ── Logs tab ────────────────────────────────────────────────────

function LogsTab() {
  const [logKey, setLogKey] = useState<string>(LOG_KEYS[0].key);
  const [lines, setLines] = useState<number>(200);
  const [data, setData] = useState<LogTailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const failuresRef = useRef(0);
  const stoppedRef = useRef(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const body = await getLogTail(logKey, lines);
      setData(body);
      setError(null);
      failuresRef.current = 0;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      failuresRef.current += 1;
      if (failuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        stoppedRef.current = true;
        setError(
          `Log polling stopped after ${MAX_CONSECUTIVE_FAILURES} consecutive failures. Last error: ${msg}`,
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }, [logKey, lines]);

  useEffect(() => {
    setData(null);
    failuresRef.current = 0;
    stoppedRef.current = false;
    load();
    const id = setInterval(() => {
      if (!stoppedRef.current) load();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [logKey, lines, load]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Log:
          <select
            value={logKey}
            onChange={(e) => setLogKey(e.target.value)}
            className="rounded border border-navy-700 bg-navy-800 px-2 py-1 font-mono text-xs text-slate-200 focus:border-sky-500 focus:outline-none"
          >
            {LOG_KEYS.map((opt) => (
              <option key={opt.key} value={opt.key}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Lines:
          <select
            value={lines}
            onChange={(e) => setLines(parseInt(e.target.value, 10))}
            className="rounded border border-navy-700 bg-navy-800 px-2 py-1 text-xs text-slate-200 focus:border-sky-500 focus:outline-none"
          >
            {LINE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => {
            stoppedRef.current = false;
            failuresRef.current = 0;
            load();
          }}
          disabled={loading}
          className="ml-auto flex items-center gap-1 rounded border border-navy-700 px-2 py-1 text-xs text-slate-400 hover:bg-navy-800/60 disabled:opacity-40"
        >
          <ArrowsClockwise size={12} weight={loading ? "fill" : "regular"} />
          {loading ? "Refreshing" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
          <p className="text-xs text-rose-300">{error}</p>
        </div>
      )}

      {data && !data.exists && (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <Warning size={36} weight="duotone" className="text-slate-600" />
          <p className="mt-3 text-sm text-slate-400">Log file does not exist yet.</p>
          <p className="mt-1 text-xs text-slate-600">It will appear here once the writer creates it.</p>
        </div>
      )}

      {data && data.exists && data.n_returned === 0 && (
        <div className="py-8 text-center text-sm text-slate-500">
          Log file is empty.
        </div>
      )}

      {data && data.exists && data.n_returned > 0 && (
        <div className="overflow-hidden rounded-xl border border-navy-700">
          <header className="flex items-center justify-between border-b border-navy-700 bg-navy-800/80 px-4 py-2 text-xs text-slate-500">
            <span className="font-mono">
              {data.log} -- last {data.n_returned} lines
            </span>
            <span className="font-mono">
              {(data.total_size_bytes / 1024).toFixed(1)} KiB total
            </span>
          </header>
          <pre className="max-h-[60vh] overflow-y-auto scrollbar-thin bg-[#0a1020] px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-300">
            {data.lines.join("\n")}
          </pre>
        </div>
      )}

      <p className="text-[11px] text-slate-600">
        <Check size={10} className="mb-0.5 mr-1 inline" />
        Auto-refresh every {POLL_INTERVAL_MS / 1000}s. Polling stops after {MAX_CONSECUTIVE_FAILURES} consecutive failures.
      </p>
    </div>
  );
}
