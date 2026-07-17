/** @type {import('next').NextConfig} */
const nextConfig = {
  // phase-64.1: let the functional-E2E :3100 server compile into an ISOLATED
  // build dir so it never shares `.next` with the operator's live :3000 dev
  // server (two `next dev` on the same `.next` cross-invalidate each other's
  // route manifests). Set ONLY by the :3100 webServer env in
  // playwright.config.ts; UNSET everywhere else (:3000 dev, `next build`, CI)
  // -> falls back to the default `.next` (byte-identical behavior).
  ...(process.env.PLAYWRIGHT_DIST_DIR
    ? { distDir: process.env.PLAYWRIGHT_DIST_DIR }
    : {}),
  output: "standalone",
  logging: {
    fetches: { fullUrl: false },
  },
  experimental: {
    optimizePackageImports: ["@phosphor-icons/react"],
  },
  // phase-4.7.1: route consolidation. Three legacy routes were merged
  // into their stronger siblings. 308 (permanent: true) preserves
  // method and cache-friendliness. See handoff/current/phase-4.7.1-contract.md.
  async redirects() {
    return [
      { source: "/compare",   destination: "/reports",        permanent: true },
      { source: "/analyze",   destination: "/signals",        permanent: true },
      { source: "/portfolio", destination: "/paper-trading",  permanent: true },
    ];
  },
};

module.exports = nextConfig;
