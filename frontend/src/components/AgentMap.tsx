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

  return (
    <div
      className={`min-w-[200px] rounded-xl border-2 ${borderStyle} ${colorCls} px-3 py-2 shadow-md`}
      data-testid="agent-node"
      data-kind={d.kind}
      title={d.role ?? d.name}
    >
      <div className="flex items-center gap-2">
        <Icon size={16} weight="duotone" />
        <div className="flex-1 min-w-0">
          <div className="truncate text-xs font-semibold text-slate-100">{d.name}</div>
          {d.model && (
            <div className="truncate font-mono text-[10px] text-slate-400">{d.model}</div>
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
}

function buildGraph({ data, layer1Expanded, providerFilter, layerFilter }: BuildArgs): {
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
    },
  }));

  const edges: Edge[] = data.edges
    .filter((e) => visibleIds.has(e.from) && visibleIds.has(e.to))
    .map((e) => ({
      id: `${e.from}-${e.to}`,
      source: e.from,
      target: e.to,
      animated: false,
      style: { stroke: "#475569", strokeWidth: 1.5 },
    }));

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
    () => buildGraph({ data, layer1Expanded, providerFilter, layerFilter }),
    [data, layer1Expanded, providerFilter, layerFilter],
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
