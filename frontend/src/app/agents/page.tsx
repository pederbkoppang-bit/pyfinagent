"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Sidebar } from "@/components/Sidebar";
import {
  Robot,
  TreeStructure,
  Lightning,
  ChatCircle,
  Brain,
  MagnifyingGlass,
  ShieldCheck,
  ClipboardText,
  Timer,
  Broadcast,
  ArrowsClockwise,
  Warning,
  Check,
  X,
  Database,
  Sparkle,
} from "@phosphor-icons/react";

// ── Types ────────────────────────────────────────────────────────

interface MASEvent {
  event_type: string;
  agent: string;
  timestamp: string;
  run_id: string;
  iteration: number;
  data: Record<string, unknown>;
  duration_ms: number;
  tokens: { input?: number; output?: number };
}

interface EventBusStats {
  total_events: number;
  buffer_size: number;
  subscribers: number;
}

type TabId = "live" | "history" | "agents";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const TABS: { id: TabId; label: string; icon: any }[] = [
  { id: "live", label: "Live Stream", icon: Broadcast },
  { id: "history", label: "Run History", icon: Timer },
  { id: "agents", label: "Agent Map", icon: TreeStructure },
];

// ── Agent node config ────────────────────────────────────────────

const AGENT_NODES = [
  { id: "Communication", label: "Communication Agent", model: "Sonnet 4.6", icon: ChatCircle, col: 1, row: 0 },
  { id: "Ford", label: "Ford (Main)", model: "Opus 4.6", icon: Brain, col: 1, row: 1 },
  { id: "qa", label: "Q&A Analyst", model: "Opus 4.6", icon: MagnifyingGlass, col: 0, row: 2 },
  { id: "research", label: "Researcher", model: "Sonnet 4.6", icon: MagnifyingGlass, col: 2, row: 2 },
  { id: "Quality Gate", label: "Quality Gate", model: "Sonnet 4.6", icon: ShieldCheck, col: 1, row: 3 },
  { id: "CitationAgent", label: "Citation Agent", model: "Sonnet 4.6", icon: ClipboardText, col: 1, row: 4 },
];

// ── Event type styling ───────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const EVENT_STYLES: Record<string, { color: string; icon: any }> = {
  classify: { color: "text-sky-400", icon: ChatCircle },
  plan: { color: "text-amber-400", icon: Brain },
  delegate: { color: "text-purple-400", icon: TreeStructure },
  tool_call: { color: "text-cyan-400", icon: Lightning },
  synthesize: { color: "text-emerald-400", icon: ArrowsClockwise },
  loop_check: { color: "text-amber-300", icon: ArrowsClockwise },
  quality_gate: { color: "text-sky-300", icon: ShieldCheck },
  citation: { color: "text-slate-400", icon: ClipboardText },
  complete: { color: "text-emerald-500", icon: Check },
  memory_save: { color: "text-slate-500", icon: Database },
  mask: { color: "text-slate-600", icon: Sparkle },
  error: { color: "text-rose-400", icon: Warning },
};

// ── Helpers ──────────────────────────────────────────────────────

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "??:??:??";
  }
}

function tokenStr(tokens: { input?: number; output?: number }): string {
  const i = tokens.input ?? 0;
  const o = tokens.output ?? 0;
  if (i === 0 && o === 0) return "";
  return `${i}+${o} tok`;
}

// ── Event Detail Component ────────────────────────────────────────

function EventDetail({ event }: { event: MASEvent }) {
  const d = event.data;
  if (!d) return null;

  switch (event.event_type) {
    case "classify":
      return (
        <p className="mt-0.5 text-xs text-slate-500 truncate">
          → {String(d.primary)} ({String(d.complexity)})
          {d.query_preview ? `: "${String(d.query_preview).slice(0, 80)}"` : ""}
        </p>
      );
    case "plan":
      return d.plan_preview ? (
        <p className="mt-0.5 text-xs text-slate-500 truncate">{String(d.plan_preview).slice(0, 120)}</p>
      ) : null;
    case "delegate":
      return (
        <p className="mt-0.5 text-xs text-slate-500">
          → {String(d.target_name)} ({String(d.model)})
        </p>
      );
    case "quality_gate":
      return (
        <p className={`mt-0.5 text-xs ${d.passed ? "text-emerald-500" : "text-rose-400"}`}>
          {d.passed ? "PASS" : "FAIL → improved"}
        </p>
      );
    case "tool_call":
      return (
        <p className="mt-0.5 text-xs text-slate-500 font-mono">
          {String(d.tool)} (turn {String(d.turn)})
        </p>
      );
    case "loop_check":
      return (
        <p className="mt-0.5 text-xs text-slate-500">
          {d.needs_more ? "NEEDS_MORE → continuing" : "COMPLETE"} ({String(d.findings_count)} findings)
        </p>
      );
    case "complete":
      return (
        <p className="mt-0.5 text-xs text-emerald-500">
          {String(d.agent_type)} | {String(d.complexity)} | {String(d.response_length)} chars
        </p>
      );
    case "error":
      return d.error ? (
        <p className="mt-0.5 text-xs text-rose-400 truncate">{String(d.error)}</p>
      ) : null;
    default:
      return null;
  }
}

// ── Main Page ────────────────────────────────────────────────────

export default function AgentsPage() {
  const [tab, setTab] = useState<TabId>("live");
  const [events, setEvents] = useState<MASEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState<EventBusStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set());
  const eventSourceRef = useRef<EventSource | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const failCountRef = useRef(0);

  // ── SSE Connection ──────────────────────────────────────────

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource("/api/mas/events?include_buffer=true");
    eventSourceRef.current = es;

    es.onopen = () => {
      setConnected(true);
      setError(null);
      failCountRef.current = 0;
    };

    es.onmessage = (e) => {
      try {
        const event: MASEvent = JSON.parse(e.data);
        setEvents((prev) => [...prev.slice(-499), event]);

        // Track active agents (flash for 3s)
        setActiveAgents((prev) => {
          const next = new Set(prev);
          next.add(event.agent);
          setTimeout(() => {
            setActiveAgents((p) => {
              const n = new Set(p);
              n.delete(event.agent);
              return n;
            });
          }, 3000);
          return next;
        });
      } catch {
        // skip unparseable
      }
    };

    es.onerror = () => {
      failCountRef.current += 1;
      setConnected(false);
      if (failCountRef.current >= 5) {
        es.close();
        setError("Lost connection to MAS event stream after 5 failures. Backend may be down.");
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      eventSourceRef.current?.close();
    };
  }, [connect]);

  // ── Fetch stats ─────────────────────────────────────────────

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch("/api/mas/events/stats");
        if (res.ok) setStats(await res.json());
      } catch {
        // silent
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  // ── Auto-scroll ─────────────────────────────────────────────

  useEffect(() => {
    if (scrollRef.current && tab === "live") {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events, tab]);

  // ── Derived data ────────────────────────────────────────────

  const runs = Array.from(new Set(events.filter((e) => e.run_id).map((e) => e.run_id)));
  const latestRun = runs[runs.length - 1] ?? null;
  const latestRunEvents = latestRun ? events.filter((e) => e.run_id === latestRun) : [];
  const completeEvent = latestRunEvents.find((e) => e.event_type === "complete");

  const totalTokensIn = events.reduce((s, e) => s + (e.tokens.input ?? 0), 0);
  const totalTokensOut = events.reduce((s, e) => s + (e.tokens.output ?? 0), 0);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* ── Fixed header zone ── */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          {/* Tier 1: Header */}
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
                <Robot size={28} weight="duotone" className="text-sky-400" />
                Multi-Agent System
              </h2>
              <p className="text-sm text-slate-500">
                Real-time observability — Communication → Ford → Subagents → Quality Gate → Citation
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className={`flex items-center gap-1.5 text-xs font-medium ${connected ? "text-emerald-400" : "text-rose-400"}`}>
                <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`} />
                {connected ? "Connected" : "Disconnected"}
              </span>
              {stats && (
                <span className="text-xs text-slate-500 font-mono">
                  {stats.total_events} events | {stats.subscribers} sub
                </span>
              )}
            </div>
          </div>

          {/* Hero metrics */}
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Events</p>
              <p className="mt-1 text-xl font-bold text-slate-100">{events.length}</p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Runs</p>
              <p className="mt-1 text-xl font-bold text-slate-100">{runs.length}</p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Tokens In</p>
              <p className="mt-1 text-xl font-bold text-sky-400 font-mono">{totalTokensIn.toLocaleString()}</p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Tokens Out</p>
              <p className="mt-1 text-xl font-bold text-amber-400 font-mono">{totalTokensOut.toLocaleString()}</p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Latest Run</p>
              <p className="mt-1 text-xl font-bold text-slate-100 font-mono">
                {completeEvent ? `${(completeEvent.duration_ms / 1000).toFixed(1)}s` : latestRun ? "..." : "—"}
              </p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Active Agents</p>
              <p className="mt-1 text-xl font-bold text-emerald-400">{activeAgents.size}</p>
            </div>
          </div>

          {/* Tier 5: Tab bar */}
          <div className="flex gap-1 rounded-lg bg-navy-800/60 p-1">
            {TABS.map((t) => (
              <button
                key={t.id}
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

        {/* ── Scrollable content zone ── */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8" ref={scrollRef}>
          {/* Error banner */}
          {error && (
            <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 flex items-center justify-between">
              <p className="text-sm text-rose-300">{error}</p>
              <button onClick={() => { setError(null); failCountRef.current = 0; connect(); }}
                className="text-xs text-rose-300 hover:text-rose-100 underline">
                Retry
              </button>
            </div>
          )}

          {/* ── Tab: Live Stream ── */}
          {tab === "live" && (
            <div className="space-y-1">
              {events.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-24 text-center">
                  <Broadcast size={48} weight="duotone" className="text-slate-600" />
                  <p className="mt-4 text-lg text-slate-400">No events yet</p>
                  <p className="mt-1 text-sm text-slate-600">
                    Events will appear here when the MAS processes a message via Slack or iMessage
                  </p>
                </div>
              ) : (
                events.map((event, i) => {
                  const style = EVENT_STYLES[event.event_type] ?? { color: "text-slate-400", icon: Lightning };
                  const Icon = style.icon;
                  return (
                    <div
                      key={`${event.timestamp}-${i}`}
                      className="flex items-start gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-navy-800/40"
                    >
                      <Icon size={16} className={`mt-0.5 flex-shrink-0 ${style.color}`} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-medium ${style.color}`}>
                            {event.event_type}
                          </span>
                          <span className="text-xs text-slate-500">{event.agent}</span>
                          {event.duration_ms > 0 && (
                            <span className="text-xs text-slate-600 font-mono">
                              {event.duration_ms > 1000
                                ? `${(event.duration_ms / 1000).toFixed(1)}s`
                                : `${Math.round(event.duration_ms)}ms`}
                            </span>
                          )}
                          {tokenStr(event.tokens) && (
                            <span className="text-xs text-slate-600 font-mono">{tokenStr(event.tokens)}</span>
                          )}
                          <span className="ml-auto text-xs text-slate-600 font-mono">{formatTime(event.timestamp)}</span>
                        </div>
                        {/* Event-specific details */}
                        <EventDetail event={event} />
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {/* ── Tab: Run History ── */}
          {tab === "history" && (
            <div>
              {runs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-24 text-center">
                  <Timer size={48} weight="duotone" className="text-slate-600" />
                  <p className="mt-4 text-lg text-slate-400">No runs recorded</p>
                  <p className="mt-1 text-sm text-slate-600">Run history will appear after MAS processes queries</p>
                </div>
              ) : (
                <div className="overflow-hidden rounded-xl border border-navy-700">
                  <table className="w-full text-left text-sm">
                    <thead className="border-b border-navy-700 bg-navy-800/80">
                      <tr>
                        <th className="px-4 py-3 font-medium text-slate-400">Run ID</th>
                        <th className="px-4 py-3 font-medium text-slate-400">Agent</th>
                        <th className="px-4 py-3 font-medium text-slate-400">Complexity</th>
                        <th className="px-4 py-3 font-medium text-slate-400">Duration</th>
                        <th className="px-4 py-3 font-medium text-slate-400">Tokens</th>
                        <th className="px-4 py-3 font-medium text-slate-400">Steps</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-navy-700/50">
                      {runs.map((runId) => {
                        const runEvents = events.filter((e) => e.run_id === runId);
                        const complete = runEvents.find((e) => e.event_type === "complete");
                        const classify = runEvents.find((e) => e.event_type === "classify");
                        const totalIn = runEvents.reduce((s, e) => s + (e.tokens.input ?? 0), 0);
                        const totalOut = runEvents.reduce((s, e) => s + (e.tokens.output ?? 0), 0);
                        return (
                          <tr key={runId} className="transition-colors hover:bg-navy-700/40">
                            <td className="px-4 py-3 font-mono text-xs text-slate-300">{runId.slice(0, 8)}</td>
                            <td className="px-4 py-3 text-slate-200">{String(complete?.data?.agent_type ?? classify?.data?.primary ?? "—")}</td>
                            <td className="px-4 py-3 text-slate-400">{String(complete?.data?.complexity ?? classify?.data?.complexity ?? "—")}</td>
                            <td className="px-4 py-3 font-mono text-slate-200">
                              {complete ? `${(complete.duration_ms / 1000).toFixed(1)}s` : "..."}
                            </td>
                            <td className="px-4 py-3 font-mono text-xs text-slate-400">{totalIn}+{totalOut}</td>
                            <td className="px-4 py-3 text-slate-400">{runEvents.length}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* ── Tab: Agent Map ── */}
          {tab === "agents" && (
            <div className="space-y-6">
              {/* Node graph visualization */}
              <div className="rounded-xl border border-navy-700 bg-navy-800/30 p-8">
                <div className="relative mx-auto" style={{ maxWidth: 600, height: 480 }}>
                  {/* Connection lines (SVG) */}
                  <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 0 }}>
                    {/* Communication → Ford */}
                    <line x1="50%" y1="60" x2="50%" y2="140" stroke="#334155" strokeWidth="2" strokeDasharray="6 4" />
                    {/* Ford → Q&A */}
                    <line x1="50%" y1="180" x2="16%" y2="260" stroke="#334155" strokeWidth="2" strokeDasharray="6 4" />
                    {/* Ford → Researcher */}
                    <line x1="50%" y1="180" x2="84%" y2="260" stroke="#334155" strokeWidth="2" strokeDasharray="6 4" />
                    {/* Ford → Quality Gate */}
                    <line x1="50%" y1="180" x2="50%" y2="340" stroke="#334155" strokeWidth="2" strokeDasharray="6 4" />
                    {/* Quality Gate → Citation */}
                    <line x1="50%" y1="380" x2="50%" y2="420" stroke="#334155" strokeWidth="2" strokeDasharray="6 4" />
                  </svg>

                  {/* Agent nodes */}
                  {AGENT_NODES.map((node) => {
                    const isActive = activeAgents.has(node.id);
                    const xPositions = ["8%", "35%", "62%"];
                    const yPositions = [20, 120, 240, 320, 400];
                    const left = node.col === 1 ? "50%" : node.col === 0 ? "16%" : "84%";
                    const top = yPositions[node.row];
                    const Icon = node.icon;

                    return (
                      <div
                        key={node.id}
                        className={`absolute flex items-center gap-2 rounded-xl border px-4 py-2.5 transition-all duration-300
                          ${isActive
                            ? "border-sky-500/60 bg-sky-950/60 shadow-lg shadow-sky-500/10"
                            : "border-navy-700 bg-navy-800/80"
                          }`}
                        style={{
                          left,
                          top,
                          transform: "translate(-50%, 0)",
                          zIndex: 1,
                          minWidth: 160,
                        }}
                      >
                        <div className={`flex-shrink-0 rounded-lg p-1.5 ${isActive ? "bg-sky-500/20" : "bg-navy-700/60"}`}>
                          <Icon size={18} weight="duotone" className={isActive ? "text-sky-400" : "text-slate-400"} />
                        </div>
                        <div className="min-w-0">
                          <p className={`text-xs font-medium ${isActive ? "text-sky-300" : "text-slate-300"}`}>
                            {node.label}
                          </p>
                          <p className="text-[10px] text-slate-500 font-mono">{node.model}</p>
                        </div>
                        {isActive && (
                          <span className="ml-auto h-2 w-2 rounded-full bg-sky-400 animate-pulse" />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Agent details table */}
              <div className="overflow-hidden rounded-xl border border-navy-700">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-navy-700 bg-navy-800/80">
                    <tr>
                      <th className="px-4 py-3 font-medium text-slate-400">Agent</th>
                      <th className="px-4 py-3 font-medium text-slate-400">Model</th>
                      <th className="px-4 py-3 font-medium text-slate-400">Role</th>
                      <th className="px-4 py-3 font-medium text-slate-400">Events</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-navy-700/50">
                    {AGENT_NODES.map((node) => {
                      const agentEvents = events.filter((e) => e.agent === node.id || e.agent === node.label);
                      const roles: Record<string, string> = {
                        Communication: "Router + Quality Gate",
                        Ford: "Orchestrator + Synthesizer",
                        qa: "Quantitative Analysis",
                        research: "Literature + Evidence",
                        "Quality Gate": "Skeptical Reviewer (0.0-1.0)",
                        CitationAgent: "Source Attribution",
                      };
                      return (
                        <tr key={node.id} className="transition-colors hover:bg-navy-700/40">
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <node.icon size={16} className={activeAgents.has(node.id) ? "text-sky-400" : "text-slate-400"} />
                              <span className="text-slate-200">{node.label}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3 font-mono text-xs text-slate-400">{node.model}</td>
                          <td className="px-4 py-3 text-xs text-slate-500">{roles[node.id] ?? "—"}</td>
                          <td className="px-4 py-3 font-mono text-slate-300">{agentEvents.length}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
