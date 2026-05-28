/**
 * phase-47.5: Semantic design-token maps -- the ENFORCEMENT layer over the
 * tailwind.config navy/slate primitives (frontend.md "Dark-mode + readability"
 * + section 6 contrast targets). Components import named roles instead of
 * hand-composing class strings, so the ~120 ad-hoc text/hover/focus/button
 * sites can migrate to ONE consistent vocabulary (ux_roadmap.md W2).
 *
 * Every value is a COMPLETE literal class string -- Tailwind v3 JIT only
 * matches literals, never template concatenation (frontend.md rule 3). Navy +
 * slate only; never zinc (rule 1). Dark-mode-only project, so NO light-mode
 * bg-white / text-zinc base defaults (rule 2).
 */

export const tokens = {
  // Text contrast tiers (frontend.md section 6 -- WCAG 2.2 AAA on bg-navy-800/70)
  text: {
    primary: "text-slate-100", //   >= 13:1  headings, risk-relevant numbers
    default: "text-slate-200", //   >= 12:1  body, table cells
    secondary: "text-slate-300", // >= 10:1  secondary copy
    tertiary: "text-slate-400", //  >= 7:1   chrome only, NEVER P&L / stop-loss
    dim: "text-slate-500", //                labels, captions
  },
  // Surfaces
  surface: {
    card: "bg-navy-800/70 border border-navy-700",
    subtle: "bg-navy-800/60 border border-navy-700",
    inset: "bg-navy-800/40",
  },
  border: {
    default: "border-navy-700",
    subtle: "border-navy-700/50",
  },
  // Interaction
  hover: {
    row: "hover:bg-navy-700/40",
    active: "bg-navy-700/60",
  },
  // ONE uniform keyboard-only focus ring (matches the proven OpsStatusBar idiom;
  // :focus-visible = keyboard-only, no mouse-click ring -- Sara Soueidan).
  focusRing: "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400",
  transition: {
    base: "transition-colors duration-200",
    state: "transition-all duration-300",
    icon: "transition-transform duration-150",
  },
  // Pre-attentive status (frontend.md: green=ok, amber=warn, rose=error,
  // slate=neutral/unavailable, sky=info/actionable).
  status: {
    success: "bg-emerald-500/15 text-emerald-300",
    warning: "bg-amber-500/15 text-amber-300",
    error: "bg-rose-500/15 text-rose-300",
    neutral: "bg-slate-500/15 text-slate-400",
    info: "bg-sky-500/15 text-sky-300",
  },
} as const;

export type StatusVariant = keyof typeof tokens.status;
