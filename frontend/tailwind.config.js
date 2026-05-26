/** @type {import('tailwindcss').Config} */
module.exports = {
  // phase-44.2 cycle 68 UX-audit root-cause fix:
  // default is "media" which only activates dark: variants when the OS
  // itself reports prefers-color-scheme: dark. This project is dark-only
  // and we apply the `dark` class to <html> in app/layout.tsx, so the
  // "selector" strategy gives us reliable dark: activation regardless of
  // OS preference. Documented in handoff/current/research_brief_phase_44_2_uxaudit.md.
  darkMode: "selector",
  // phase-44.2 cycle-69: include @tremor/** so Tailwind JIT compiles
  // every utility class Tremor references internally (fill-{color}-500,
  // text-{color}-500, etc.). Without this path, Tremor DonutChart slices
  // render uncolored (operator-reported 2026-05-26). This is the
  // canonical Tremor v3 install step that was missed when @tremor/react
  // was added in phase-44.0. See:
  // https://www.tremor.so/docs/getting-started/installation
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./node_modules/@tremor/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      colors: {
        navy: {
          900: "#020617",
          800: "#0f172a",
          700: "#1e293b",
          600: "#1a2744",
          500: "#243352",
        },
      },
      borderRadius: {
        card: "12px",
        button: "8px",
        badge: "6px",
      },
      boxShadow: {
        card: "0 1px 3px 0 rgba(9, 9, 11, 0.4), 0 1px 2px -1px rgba(9, 9, 11, 0.4)",
        "card-hover": "0 4px 12px 0 rgba(9, 9, 11, 0.5), 0 2px 4px -2px rgba(9, 9, 11, 0.4)",
      },
      // phase-74 (2026-05-26) Google-Finance-style price-tick flash.
      // 500ms total (200ms hold + 100ms fade approx, ease-in-out) per
      // researcher a3f10c3c35c087f50 Section 1; rgba(16,185,129,0.15) =
      // emerald-500/15, rgba(244,63,94,0.15) = rose-500/15. Animation
      // override for prefers-reduced-motion lives in globals.css.
      keyframes: {
        "flash-up": {
          "0%": { backgroundColor: "rgba(16, 185, 129, 0.15)" },
          "100%": { backgroundColor: "transparent" },
        },
        "flash-down": {
          "0%": { backgroundColor: "rgba(244, 63, 94, 0.15)" },
          "100%": { backgroundColor: "transparent" },
        },
      },
      animation: {
        "flash-up": "flash-up 500ms ease-in-out",
        "flash-down": "flash-down 500ms ease-in-out",
      },
    },
  },
  plugins: [],
};
