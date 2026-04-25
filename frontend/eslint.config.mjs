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
      // phase-16.32 (closes #42): force icons through the centralized
      // @/lib/icons barrel so semantic re-export names stay consistent
      // across the app and the bundle stays tree-shaken.
      // phase-16.39 (closes #50): all 22 prior violators rewritten;
      // rule promoted from "warn" to "error" alongside the sweep.
      "no-restricted-imports": ["error", {
        paths: [{
          name: "@phosphor-icons/react",
          message: "Import icons from @/lib/icons instead of @phosphor-icons/react directly.",
        }],
        patterns: [{
          group: ["@phosphor-icons/react/*"],
          message: "Import icons from @/lib/icons instead of @phosphor-icons/react directly.",
        }],
      }],
    },
  },
  {
    // Centralized icon barrel is the ONE legitimate place to import
    // @phosphor-icons/react directly. Exempt it from the rule above.
    files: ["**/lib/icons.ts", "**/lib/icons.tsx"],
    rules: {
      "no-restricted-imports": "off",
    },
  },
];
