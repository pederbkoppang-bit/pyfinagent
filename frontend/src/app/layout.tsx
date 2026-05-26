import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { AuthProvider } from "@/components/AuthProvider";
import { CommandPalette } from "@/components/CommandPalette";
import "./globals.css";

export const metadata: Metadata = {
  title: "PyFinAgent — AI Financial Analyst",
  description: "Evidence-based stock analysis powered by agentic AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="font-sans antialiased">
        {/* phase-44.1: WCAG 2.2 SC 2.4.1 Skip Link -- visible only on focus.
            EU AAA mandate 2026 per research_brief Section B.7. */}
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:px-3 focus:py-2 focus:rounded-lg focus:bg-sky-600 focus:text-white focus:outline-none focus:ring-2 focus:ring-sky-400"
        >
          Skip to main content
        </a>
        <AuthProvider>
          {/* phase-44.1: Cmd-K command palette (cmdk by Vercel/Pacos).
              Operator-approved 2026-05-22 per handoff/current/operator_approval_44.1.md.
              Mounted at root so Cmd+K opens from any of the 15 routes. */}
          <CommandPalette />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
