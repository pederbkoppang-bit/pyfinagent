/** @type {import('tailwindcss').Config} */
module.exports = {
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
