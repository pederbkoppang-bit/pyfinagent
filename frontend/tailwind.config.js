/** @type {import('tailwindcss').Config} */
module.exports = {
  // phase-44.2 cycle 68 UX-audit root-cause fix:
  // default is "media" which only activates dark: variants when the OS
  // itself reports prefers-color-scheme: dark. This project is dark-only
  // and we apply the `dark` class to <html> in app/layout.tsx, so the
  // "selector" strategy gives us reliable dark: activation regardless of
  // OS preference. Documented in handoff/current/research_brief_phase_44_2_uxaudit.md.
  darkMode: "selector",
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
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
    },
  },
  plugins: [],
};
