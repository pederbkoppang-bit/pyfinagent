"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { signIn as webAuthnSignIn } from "next-auth/webauthn";
import { clsx } from "clsx";
import { useState } from "react";

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
  const { data: session } = useSession();
  const version = process.env.NEXT_PUBLIC_APP_VERSION || "local";
  const [passkeyStatus, setPasskeyStatus] = useState<string | null>(null);

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

      {/* User info + auth */}
      <div className="border-t border-navy-700 pt-4 space-y-3">
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
            🔑 Passkey
          </button>
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="flex-1 rounded-md bg-slate-800 px-2 py-1.5 text-xs text-slate-400 hover:bg-red-500/20 hover:text-red-400 transition-colors"
          >
            Logout
          </button>
        </div>

        {passkeyStatus && (
          <p className="px-2 text-xs text-sky-400 text-center">{passkeyStatus}</p>
        )}

        <div className="text-center">
          <span className="inline-block rounded-md bg-slate-800 px-3 py-1 text-xs text-slate-500">
            Version: {version}
          </span>
        </div>
      </div>
    </aside>
  );
}
