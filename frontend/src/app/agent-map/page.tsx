/**
 * phase-18.4 /agent-map page route.
 *
 * Hosts the AgentMap component (phase-18.2 + 18.3) which visualizes the
 * pyfinagent agent topology via React Flow. Two-zone shell pattern.
 */
"use client";

import { Sidebar } from "@/components/Sidebar";
import { AgentMap } from "@/components/AgentMap";

export default function AgentMapPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-slate-100">Agent Topology</h2>
            <p className="text-sm text-slate-500">
              Live map of all pyfinagent agents (Layer 1 analysis pipeline, Layer 2 in-app MAS,
              Layer 3 harness MAS, services + meta-evolution). Click the Layer-1 group to expand
              the 28 Gemini skill agents.
            </p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          <AgentMap />
        </div>
      </main>
    </div>
  );
}
