/**
 * phase-47.5: canonical Button. In later migration cycles this replaces the
 * ~30 hand-composed inline button class strings (page.tsx, OpsStatusBar.tsx,
 * GoLiveGateWidget.tsx, Sidebar.tsx). Variants: primary | secondary | ghost |
 * danger (danger is the standard for Pause / Flatten / Stop). Bakes in the
 * uniform focus ring + >=24px tap target (frontend.md / WCAG) and a CSS
 * active:scale-95 press -- a single button does not justify the ~34kb Framer
 * Motion bundle; motion/react is reserved for page-level choreography (W6).
 * No emoji; pass a Phosphor icon via children.
 */
"use client";

import { clsx } from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { tokens } from "@/lib/design-tokens";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

const VARIANT: Record<Variant, string> = {
  primary: "bg-sky-500/15 text-sky-300 border border-sky-500/30 hover:bg-sky-500/25",
  secondary: "bg-navy-800/60 text-slate-200 border border-navy-700 hover:bg-navy-700/60",
  ghost: "text-slate-400 hover:text-slate-200 hover:bg-navy-700/40",
  danger: "bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25",
};

const SIZE: Record<Size, string> = {
  sm: "px-2 py-1 text-[11px] gap-1",
  md: "px-3 py-1.5 text-xs gap-1.5",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
}

export function Button({
  variant = "secondary",
  size = "md",
  className,
  type = "button",
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={clsx(
        "inline-flex items-center justify-center rounded-md font-medium",
        "min-h-[24px] min-w-[24px] active:scale-95",
        "disabled:cursor-not-allowed disabled:opacity-40",
        tokens.transition.base,
        tokens.focusRing,
        VARIANT[variant],
        SIZE[size],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}
