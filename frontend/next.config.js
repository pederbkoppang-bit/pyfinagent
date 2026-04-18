/** @type {import('next').NextConfig} */
const nextConfig = {
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
