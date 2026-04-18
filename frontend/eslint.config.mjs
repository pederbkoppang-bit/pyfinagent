// Flat ESLint config for Next.js 15 + eslint-config-next v16.
// We spread the flat array exported by eslint-config-next/core-web-vitals
// directly (its dist/core-web-vitals.js already returns a flat-array
// config) rather than going through FlatCompat, which tried to
// JSON.stringify plugin objects and triggered a circular-ref error.
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";

export default [
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      "chrome/**",
      "handoff/**",
      "scripts/run-test.mjs",
      "prisma/**",
      "next-env.d.ts",
    ],
  },
  ...nextCoreWebVitals,
  {
    rules: {
      "react-hooks/exhaustive-deps": "warn",
      "@next/next/no-html-link-for-pages": "off",
      // 2026 React Compiler rules (shipped in eslint-plugin-react-hooks
      // as part of eslint-config-next v16). Warning-only during the
      // phase-4.7.5 consistency pass; a dedicated refactor cycle will
      // rewrite the flagged fetch-in-effect patterns to react-query
      // hooks and promote these to errors. Tracked in the step's
      // "Known limitations" section.
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/purity": "warn",
      "react-hooks/immutability": "warn",
      "react-hooks/rules-of-hooks": "error",
    },
  },
];
