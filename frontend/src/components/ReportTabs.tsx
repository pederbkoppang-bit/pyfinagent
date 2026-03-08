"use client";

import { clsx } from "clsx";
import { useState, type ReactNode } from "react";

export interface TabDef {
  id: string;
  label: string;
  icon: string;
  badge?: string | number | null;
}

interface ReportTabsProps {
  tabs: TabDef[];
  children: (activeTab: string) => ReactNode;
}

export function ReportTabs({ tabs, children }: ReportTabsProps) {
  const [active, setActive] = useState(tabs[0]?.id ?? "overview");

  return (
    <div>
      {/* Tab bar */}
      <div className="mb-6 flex items-center gap-1 overflow-x-auto rounded-xl border border-navy-700 bg-navy-800/70 p-1 backdrop-blur-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            className={clsx(
              "flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition-all whitespace-nowrap",
              active === tab.id
                ? "bg-sky-500/15 text-sky-400 shadow-sm"
                : "text-slate-500 hover:text-slate-300 hover:bg-navy-700/50"
            )}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
            {tab.badge != null && (
              <span
                className={clsx(
                  "ml-1 inline-flex h-5 min-w-[20px] items-center justify-center rounded-full px-1.5 text-[10px] font-bold",
                  active === tab.id ? "bg-sky-500/20 text-sky-300" : "bg-slate-700 text-slate-400"
                )}
              >
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>{children(active)}</div>
    </div>
  );
}
