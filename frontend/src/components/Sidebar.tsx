"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { signIn as webAuthnSignIn } from "next-auth/webauthn";
import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  NavHome, NavAnalyze, NavSignals, NavReports,
  NavPerformance, NavPortfolio, NavPaperTrading, NavBacktest, NavSettings,
  LogoIcon, IconKey, IconSignOut,
} from "@/lib/icons";
import type { Icon } from "@phosphor-icons/react";
import { CaretDown, X, Robot } from "@phosphor-icons/react";
import { healthCheck } from "@/lib/api";

interface NavItem { href: string; label: string; icon: Icon }
interface NavSection { label: string; collapsible: boolean; items: NavItem[] }

const NAV_SECTIONS: NavSection[] = [
  {
    label: "Analyze",
    collapsible: true,
    items: [
      { href: "/", label: "Home", icon: NavHome },
      { href: "/analyze", label: "Deep Analysis", icon: NavAnalyze },
      { href: "/signals", label: "Signals", icon: NavSignals },
    ],
  },
  {
    label: "Reports",
    collapsible: true,
    items: [
      { href: "/reports", label: "Reports", icon: NavReports },
      { href: "/performance", label: "Performance", icon: NavPerformance },
    ],
  },
  {
    label: "Trading",
    collapsible: true,
    items: [
      { href: "/portfolio", label: "Portfolio", icon: NavPortfolio },
      { href: "/paper-trading", label: "Paper Trading", icon: NavPaperTrading },
      { href: "/backtest", label: "Backtest", icon: NavBacktest },
    ],
  },
  {
    label: "System",
    collapsible: true,
    items: [
      { href: "/agents", label: "MAS Dashboard", icon: Robot },
    ],
  },
];

// ── Changelog types ───────────────────────────────────────────

interface ChangelogEntry {
  version: string;
  title: string;
  date: string;
  changes: string[];
}

// ── Section Group ─────────────────────────────────────────────

function SectionGroup({
  section,
  pathname,
  collapsed,
  onToggle,
}: {
  section: NavSection;
  pathname: string;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <div>
      {section.collapsible ? (
        <button
          onClick={onToggle}
          className="mb-1.5 flex w-full items-center justify-between px-3 group"
        >
          <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-600 group-hover:text-slate-400 transition-colors">
            {section.label}
          </span>
          <CaretDown
            size={12}
            weight="bold"
            className={clsx(
              "text-slate-600 transition-transform duration-200 group-hover:text-slate-400",
              collapsed && "-rotate-90"
            )}
          />
        </button>
      ) : (
        <p className="mb-1.5 px-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
          {section.label}
        </p>
      )}
      {(!collapsed || !section.collapsible) && (
        <div className="space-y-0.5">
          {section.items.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                pathname === item.href
                  ? "bg-sky-500/10 font-medium text-sky-400"
                  : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              )}
            >
              <item.icon size={18} weight={pathname === item.href ? "fill" : "regular"} />
              {item.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Changelog Modal ───────────────────────────────────────────

function ChangelogModal({ onClose }: { onClose: () => void }) {
  const [entries, setEntries] = useState<ChangelogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/changelog`)
      .then((r) => r.json())
      .then((data) => setEntries(data.entries || []))
      .catch(() => setError("Could not load changelog"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative mx-4 max-h-[80vh] w-full max-w-lg overflow-hidden rounded-xl border border-slate-700 bg-navy-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-700 px-5 py-4">
          <h2 className="text-lg font-semibold text-slate-100">What&apos;s New</h2>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
          >
            <X size={20} />
          </button>
        </div>

        {/* Changelog entries */}
        <div className="overflow-y-auto scrollbar-thin px-5 py-4" style={{ maxHeight: "calc(80vh - 64px)" }}>
          {loading && (
            <div className="flex items-center gap-2 py-8 text-sm text-slate-500">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
              Loading changelog...
            </div>
          )}
          {error && (
            <p className="py-8 text-center text-sm text-slate-500">{error}</p>
          )}
          {!loading && !error && (
            <div className="space-y-6">
              {entries.map((entry, i) => (
                <div key={entry.version}>
                  <div className="flex items-baseline gap-2">
                    <span className="rounded bg-sky-500/10 px-2 py-0.5 font-mono text-xs font-semibold text-sky-400">
                      v{entry.version}
                    </span>
                    {entry.date && (
                      <span className="text-xs text-slate-500">{entry.date}</span>
                    )}
                  </div>
                  <h3 className="mt-1.5 text-sm font-medium text-slate-200">{entry.title}</h3>
                  {entry.changes.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {entry.changes.map((change, j) => (
                        <li key={j} className="flex gap-2 text-xs text-slate-400">
                          <span className="mt-1.5 h-1 w-1 flex-shrink-0 rounded-full bg-slate-600" />
                          {change}
                        </li>
                      ))}
                    </ul>
                  )}
                  {i < entries.length - 1 && (
                    <div className="mt-4 border-b border-slate-800" />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────

export function Sidebar() {
  const pathname = usePathname();
  const { data: session } = useSession();
  const [version, setVersion] = useState(process.env.NEXT_PUBLIC_APP_VERSION || "local");
  const [passkeyStatus, setPasskeyStatus] = useState<string | null>(null);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [showChangelog, setShowChangelog] = useState(false);

  // Track collapsed state per section — all expanded by default
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});

  const toggleSection = (label: string) => {
    setCollapsedSections((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  // Poll backend health every 30s
  useEffect(() => {
    let mounted = true;
    const check = () => {
      healthCheck()
        .then((res) => { if (mounted) { setBackendUp(true); if (res.version) setVersion(res.version); } })
        .catch(() => { if (mounted) setBackendUp(false); });
    };
    check();
    const interval = setInterval(check, 30000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  const registerPasskey = async () => {
    setPasskeyStatus("Registering...");
    try {
      await webAuthnSignIn("passkey", { action: "register" });
      setPasskeyStatus("Passkey registered!");
    } catch {
      setPasskeyStatus("Registration failed");
    }
    setTimeout(() => setPasskeyStatus(null), 3000);
  };

  return (
    <>
      <aside className="flex h-screen w-64 flex-shrink-0 flex-col border-r border-navy-700 bg-navy-800/50">
        {/* ── Fixed header ─────────────────────────────────── */}
        <div className="flex items-center gap-3 px-6 py-6">
          <LogoIcon size={28} weight="bold" className="text-sky-400" />
          <div>
            <h1 className="text-lg font-bold text-slate-100">PyFinAgent</h1>
            <p className="text-xs text-slate-500">AI Financial Analyst</p>
          </div>
        </div>

        {/* ── Scrollable nav ───────────────────────────────── */}
        <nav className="flex-1 space-y-5 overflow-y-auto px-4 scrollbar-thin">
          {NAV_SECTIONS.map((section) => (
            <SectionGroup
              key={section.label}
              section={section}
              pathname={pathname}
              collapsed={!!collapsedSections[section.label]}
              onToggle={() => toggleSection(section.label)}
            />
          ))}
        </nav>

        {/* ── Fixed bottom ─────────────────────────────────── */}
        <div className="border-t border-navy-700 px-4 py-4 space-y-3">
          {/* Settings link */}
          <Link
            href="/settings"
            className={clsx(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
              pathname === "/settings"
                ? "bg-sky-500/10 font-medium text-sky-400"
                : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            )}
          >
            <NavSettings size={18} weight={pathname === "/settings" ? "fill" : "regular"} />
            Settings
          </Link>

          {/* User info + auth */}
          {session?.user && (
            <div className="flex items-center gap-2 px-2">
              {session.user.image ? (
                <img
                  src={session.user.image}
                  alt=""
                  className="h-7 w-7 rounded-full"
                />
              ) : (
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-sky-500/20 text-xs text-sky-400">
                  {session.user.name?.[0] || session.user.email?.[0] || "?"}
                </div>
              )}
              <span className="truncate text-xs text-slate-400">
                {session.user.email}
              </span>
            </div>
          )}

          <div className="flex gap-2 px-2">
            <button
              onClick={registerPasskey}
              className="flex-1 rounded-md bg-slate-800 px-2 py-1.5 text-xs text-slate-400 hover:bg-slate-700 hover:text-slate-200 transition-colors"
              title="Register a passkey for quick login"
            >
              <IconKey size={14} className="inline mr-1" /> Passkey
            </button>
            <button
              onClick={() => signOut({ callbackUrl: "/login" })}
              className="flex-1 rounded-md bg-slate-800 px-2 py-1.5 text-xs text-slate-400 hover:bg-red-500/20 hover:text-red-400 transition-colors"
            >
              <IconSignOut size={14} className="inline mr-1" /> Logout
            </button>
          </div>

          {passkeyStatus && (
            <p className="px-2 text-xs text-sky-400 text-center">{passkeyStatus}</p>
          )}

          {/* Version + health dot — clickable to open changelog */}
          <button
            onClick={() => setShowChangelog(true)}
            className="flex w-full items-center justify-center gap-2 rounded-md bg-slate-800 px-3 py-1.5 text-xs text-slate-500 transition-colors hover:bg-slate-700 hover:text-slate-300"
            title="View changelog"
          >
            <span
              className={clsx(
                "h-2 w-2 rounded-full",
                backendUp === true && "bg-emerald-500",
                backendUp === false && "bg-red-500",
                backendUp === null && "bg-slate-600 animate-pulse"
              )}
            />
            v{version}
          </button>
        </div>
      </aside>

      {/* Changelog modal */}
      {showChangelog && <ChangelogModal onClose={() => setShowChangelog(false)} />}
    </>
  );
}
