/**
 * phase-18.2 + 18.3 AgentMap component.
 *
 * Visualizes the pyfinagent agent topology using React Flow + dagre.
 * Phase 18.3 wires real data from `GET /api/agent-map`, adds Layer-1
 * expand/collapse, and adds provider/layer filter dropdowns.
 *
 * Design (per phase-18.0 brief):
 * - Custom node component shows icon + name + model badge
 * - Dark theme via React Flow `colorMode="dark"`
 * - Dagre TB (top-to-bottom) hierarchical layout, memoized
 * - Solid borders for in-app agents; dashed for harness/external
 * - Color by provider (sky = Anthropic/Claude, emerald = Google/Gemini)
 * - Layer-1 (28 Gemini skills) collapsed by default behind one group node
 */
"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Position,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import dagre from "dagre";
import { Brain, MagnifyingGlass, ShieldCheck } from "@/lib/icons";
import { getAgentMap, type AgentMapNode, type AgentMapResponse } from "@/lib/api";

import "@xyflow/react/dist/style.css";

export interface AgentNodeData extends Record<string, unknown> {
  name: string;
  model: string | null;
  provider: AgentMapNode["provider"];
  kind: AgentMapNode["kind"];
  layer: number;
  role?: string;
  isGroup?: boolean;
  collapsed?: boolean;
  childrenCount?: number;
  /** phase-20.2: true when this node participates in the production workflow_edges
   * AND workflow mode is enabled. Adds a glow/ring style. */
  inWorkflow?: boolean;
  /** phase-22.1: operator's actual runtime model (honors Settings override).
   * Displayed in preference to static `model` if present. */
  liveModel?: string | null;
  /** phase-22.1: cannot run on Claude (Vertex AI Search dependency). Renders a lock badge. */
  geminiLocked?: boolean;
  /** phase-22.1: loses live web-search citations on Claude but still produces text.
   * Renders a smaller "search" indicator. */
  groundingDependent?: boolean;
  /** phase-22.1: human-readable explanation of why a node is locked. Tooltip text. */
  lockReason?: string;
}

const PROVIDER_COLORS: Record<AgentNodeData["provider"], string> = {
  anthropic: "border-sky-500/60 bg-sky-500/10 text-sky-300",
  google: "border-emerald-500/60 bg-emerald-500/10 text-emerald-300",
  openai: "border-violet-500/60 bg-violet-500/10 text-violet-300",
  github_models: "border-amber-500/60 bg-amber-500/10 text-amber-300",
  none: "border-slate-600/60 bg-slate-700/40 text-slate-300",
};

const KIND_ICON: Record<AgentNodeData["kind"], typeof Brain> = {
  harness: Brain,
  in_app: Brain,
  skill: MagnifyingGlass,
  service: ShieldCheck,
  meta_evolution: ShieldCheck,
};

const LAYER_LABEL: Record<number, string> = {
  1: "L1 -- Analysis",
  2: "L2 -- MAS",
  3: "L3 -- Harness",
  4: "L4 -- Services",
};

function AgentNode({ data }: NodeProps) {
  const d = data as AgentNodeData;
  const Icon = KIND_ICON[d.kind] ?? Brain;
  const colorCls = PROVIDER_COLORS[d.provider] ?? PROVIDER_COLORS.none;
  const borderStyle = d.kind === "harness" ? "border-dashed" : "border-solid";

  const workflowRing = d.inWorkflow ? "ring-2 ring-cyan-400/60 ring-offset-2 ring-offset-navy-900" : "";
  // phase-22.1: prefer live_model (operator's actual runtime model) over static `model`
  const displayModel = d.liveModel ?? d.model;
  // Show lock badge for Vertex-locked nodes; show search-icon hint for grounding-dependent
  const titleText = d.lockReason
    ? `${d.role ?? d.name}\n\nLocked: ${d.lockReason}`
    : d.groundingDependent
    ? `${d.role ?? d.name}\n\nUses Google Search Grounding -- loses live citations on Claude but still works.`
    : d.role ?? d.name;
  return (
    <div
      className={`min-w-[200px] rounded-xl border-2 ${borderStyle} ${colorCls} ${workflowRing} px-3 py-2 shadow-md`}
      data-testid="agent-node"
      data-kind={d.kind}
      data-in-workflow={d.inWorkflow ? "true" : "false"}
      data-locked={d.geminiLocked ? "true" : "false"}
      title={titleText}
    >
      <div className="flex items-center gap-2">
        <Icon size={16} weight="duotone" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="truncate text-xs font-semibold text-slate-100">{d.name}</span>
            {d.geminiLocked && (
              <span
                className="rounded bg-amber-500/20 px-1 py-0.5 font-mono text-[9px] font-bold text-amber-300"
                aria-label="Locked to Gemini (Vertex AI Search)"
              >
                LOCKED
              </span>
            )}
            {d.groundingDependent && !d.geminiLocked && (
              <span
                className="rounded bg-sky-500/20 px-1 py-0.5 font-mono text-[9px] font-bold text-sky-300"
                aria-label="Grounding-dependent: loses live citations on Claude"
                title="Loses live Google Search citations on Claude but still produces analysis text"
              >
                SEARCH
              </span>
            )}
          </div>
          {displayModel && (
            <div className="truncate font-mono text-[10px] text-slate-400">{displayModel}</div>
          )}
          {d.isGroup && (
            <div className="mt-1 text-[10px] text-slate-500">
              {d.collapsed ? `+ ${d.childrenCount} (click to expand)` : `- ${d.childrenCount} (click to collapse)`}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const NODE_TYPES = { agent: AgentNode };

const NODE_W = 220;
const NODE_H = 70;

function layoutWithDagre<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge[],
  direction: "TB" | "LR" = "TB",
): { nodes: Node<T>[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, nodesep: 50, ranksep: 70 });

  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) g.setEdge(e.source, e.target);
  dagre.layout(g);

  const out: Node<T>[] = nodes.map((n) => {
    const pos = g.node(n.id);
    return {
      ...n,
      targetPosition: direction === "TB" ? Position.Top : Position.Left,
      sourcePosition: direction === "TB" ? Position.Bottom : Position.Right,
      position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 },
    };
  });
  return { nodes: out, edges };
}

const PROVIDERS: AgentMapNode["provider"][] = [
  "anthropic",
  "google",
  "openai",
  "github_models",
  "none",
];

interface BuildArgs {
  data: AgentMapResponse | null;
  layer1Expanded: boolean;
  providerFilter: AgentMapNode["provider"] | "all";
  layerFilter: number | "all";
  /** phase-20.2: when true, render the directional production-cycle workflow_edges
   * (with step labels + animated style + curved loop-back) INSTEAD of the static
   * topology edges. Nodes referenced by workflow_edges are highlighted. */
  workflowMode: boolean;
}

function buildGraph({ data, layer1Expanded, providerFilter, layerFilter, workflowMode }: BuildArgs): {
  nodes: Node<AgentNodeData>[];
  edges: Edge[];
} {
  if (!data) return { nodes: [], edges: [] };

  // Hide individual layer-1 skills when collapsed (the layer1_pipeline group node represents them).
  const hiddenIds = new Set<string>();
  const layer1Group = data.nodes.find((n) => n.id === "layer1_pipeline");
  if (!layer1Expanded && layer1Group) {
    for (const childId of layer1Group.children) hiddenIds.add(childId);
  }

  // Apply provider/layer filters
  const visible = data.nodes.filter((n) => {
    if (hiddenIds.has(n.id)) return false;
    if (providerFilter !== "all" && n.provider !== providerFilter) return false;
    if (layerFilter !== "all" && n.layer !== layerFilter) return false;
    return true;
  });
  const visibleIds = new Set(visible.map((n) => n.id));

  // In workflow mode, mark nodes that participate in the production cycle.
  const wfEdges = data.workflow_edges ?? [];
  const workflowNodeIds = new Set<string>();
  if (workflowMode) {
    for (const e of wfEdges) {
      workflowNodeIds.add(e.from);
      workflowNodeIds.add(e.to);
    }
  }

  const nodes: Node<AgentNodeData>[] = visible.map((n) => ({
    id: n.id,
    type: "agent",
    position: { x: 0, y: 0 },
    data: {
      name: n.name,
      model: n.model,
      provider: n.provider,
      kind: n.kind,
      layer: n.layer,
      role: n.role,
      isGroup: n.id === "layer1_pipeline",
      collapsed: n.id === "layer1_pipeline" && !layer1Expanded,
      childrenCount: n.id === "layer1_pipeline" ? n.children.length : undefined,
      inWorkflow: workflowMode && workflowNodeIds.has(n.id),
      liveModel: n.live_model ?? undefined,
      geminiLocked: n.gemini_locked ?? false,
      groundingDependent: n.grounding_dependent ?? false,
      lockReason: n.lock_reason ?? undefined,
    },
  }));

  let edges: Edge[];
  if (workflowMode) {
    // Render production workflow_edges with step labels + animated style + dashed loop-back.
    edges = wfEdges
      .filter((e) => visibleIds.has(e.from) && visibleIds.has(e.to))
      .map((e) => ({
        id: `wf-${e.from}-${e.to}`,
        source: e.from,
        target: e.to,
        label: e.step !== undefined ? `${e.step}` + (e.label ? ` ${e.label}` : "") : e.label,
        labelStyle: { fill: "#a5f3fc", fontSize: 10, fontFamily: "monospace" },
        labelBgStyle: { fill: "#0c1424", opacity: 0.9 },
        labelBgPadding: [4, 2] as [number, number],
        labelBgBorderRadius: 4,
        animated: !e.loop, // animate forward edges; loop edge is static
        style: e.loop
          ? { stroke: "#fb923c", strokeWidth: 2, strokeDasharray: "6 4" }
          : { stroke: "#22d3ee", strokeWidth: 2 },
        type: e.loop ? "step" : "smoothstep",
      }));
  } else {
    edges = data.edges
      .filter((e) => visibleIds.has(e.from) && visibleIds.has(e.to))
      .map((e) => ({
        id: `${e.from}-${e.to}`,
        source: e.from,
        target: e.to,
        animated: false,
        style: { stroke: "#475569", strokeWidth: 1.5 },
      }));
  }

  return layoutWithDagre(nodes, edges, "TB");
}

export interface AgentMapProps {
  /** Optional override (used by tests). Default fetches from /api/agent-map. */
  data?: AgentMapResponse | null;
}

export function AgentMap({ data: dataOverride }: AgentMapProps = {}) {
  const [data, setData] = useState<AgentMapResponse | null>(dataOverride ?? null);
  const [loading, setLoading] = useState(!dataOverride);
  const [error, setError] = useState<string | null>(null);
  const [layer1Expanded, setLayer1Expanded] = useState(false);
  const [providerFilter, setProviderFilter] = useState<AgentMapNode["provider"] | "all">("all");
  const [layerFilter, setLayerFilter] = useState<number | "all">("all");
  // phase-20.2: when true, render directional production-cycle workflow_edges with step labels.
  const [workflowMode, setWorkflowMode] = useState(false);

  useEffect(() => {
    if (dataOverride !== undefined) return;
    let mounted = true;
    setLoading(true);
    setError(null);
    getAgentMap()
      .then((res) => {
        if (mounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err: Error) => {
        if (mounted) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, [dataOverride]);

  const { nodes, edges } = useMemo(
    () => buildGraph({ data, layer1Expanded, providerFilter, layerFilter, workflowMode }),
    [data, layer1Expanded, providerFilter, layerFilter, workflowMode],
  );

  function handleNodeClick(_e: React.MouseEvent, node: Node) {
    if (node.id === "layer1_pipeline") {
      setLayer1Expanded((prev) => !prev);
    }
  }

  return (
    <div data-testid="agent-map" className="flex h-full w-full flex-col" style={{ minHeight: 500 }}>
      <div className="mb-3 flex flex-wrap items-center gap-3 border-b border-navy-700 pb-3">
        <span className="text-xs font-semibold text-slate-400">Filters:</span>
        <label className="flex items-center gap-1.5 text-xs text-slate-400">
          Provider
          <select
            value={providerFilter}
            onChange={(e) => setProviderFilter(e.target.value as AgentMapNode["provider"] | "all")}
            className="rounded-md border border-navy-700 bg-navy-800/60 px-2 py-1 text-xs text-slate-200"
          >
            <option value="all">all</option>
            {PROVIDERS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-400">
          Layer
          <select
            value={String(layerFilter)}
            onChange={(e) => {
              const v = e.target.value;
              setLayerFilter(v === "all" ? "all" : Number(v));
            }}
            className="rounded-md border border-navy-700 bg-navy-800/60 px-2 py-1 text-xs text-slate-200"
          >
            <option value="all">all</option>
            {[1, 2, 3, 4].map((l) => (
              <option key={l} value={l}>{LAYER_LABEL[l]}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setLayer1Expanded((p) => !p)}
          className="rounded-md border border-navy-700 bg-navy-800/60 px-3 py-1 text-xs text-slate-200 hover:bg-navy-700/60"
        >
          Layer-1 skills: {layer1Expanded ? "expanded (click to collapse)" : "collapsed (click to expand)"}
        </button>
        <button
          type="button"
          onClick={() => setWorkflowMode((p) => !p)}
          className={`rounded-md border px-3 py-1 text-xs transition-colors ${
            workflowMode
              ? "border-cyan-500/60 bg-cyan-500/10 text-cyan-300"
              : "border-navy-700 bg-navy-800/60 text-slate-200 hover:bg-navy-700/60"
          }`}
          title="Toggle between static topology and the autonomous_loop production daily-cycle"
        >
          View: {workflowMode ? "production flow (click for topology)" : "static topology (click for flow)"}
        </button>
        {data && (
          <span className="ml-auto font-mono text-xs text-slate-500">
            {nodes.length} of {data.nodes.length} agents
          </span>
        )}
      </div>

      <div className="flex-1 min-h-[400px] rounded-xl border border-navy-700 bg-navy-900/40">
        {loading && (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">
            Loading agent topology...
          </div>
        )}
        {error && (
          <div className="flex h-full items-center justify-center text-sm text-rose-300">
            {error}
          </div>
        )}
        {!loading && !error && data && (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            colorMode="dark"
            fitView
            proOptions={{ hideAttribution: true }}
            onNodeClick={handleNodeClick}
          >
            <Background color="#1e293b" gap={24} />
            <Controls position="bottom-right" showInteractive={false} />
          </ReactFlow>
        )}
      </div>
    </div>
  );
}

export default AgentMap;
