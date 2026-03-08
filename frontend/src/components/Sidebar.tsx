"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: "🏠" },
  { href: "/signals", label: "Signals", icon: "📡" },
  { href: "/reports", label: "Past Reports", icon: "📜" },
  { href: "/compare", label: "Compare", icon: "🔀" },
  { href: "/performance", label: "Performance", icon: "📈" },
  { href: "/portfolio", label: "Portfolio", icon: "💼" },
  { href: "/settings", label: "Settings", icon: "⚙️" },
];

export function Sidebar() {
  const pathname = usePathname();
  const version = process.env.NEXT_PUBLIC_APP_VERSION || "local";

  return (
    <aside className="flex h-screen w-64 flex-col border-r border-navy-700 bg-navy-800/50 px-4 py-6">
      <div className="mb-8 flex items-center gap-3 px-2">
        <span className="text-2xl">🤖</span>
        <div>
          <h1 className="text-lg font-bold text-slate-100">PyFinAgent</h1>
          <p className="text-xs text-slate-500">AI Financial Analyst</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={clsx(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
              pathname === item.href
                ? "bg-sky-500/10 font-medium text-sky-400"
                : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            )}
          >
            <span>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>

      {/* Version badge */}
      <div className="border-t border-navy-700 pt-4 text-center">
        <span className="inline-block rounded-md bg-slate-800 px-3 py-1 text-xs text-slate-500">
          Version: {version}
        </span>
      </div>
    </aside>
  );
}
