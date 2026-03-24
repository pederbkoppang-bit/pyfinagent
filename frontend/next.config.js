/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  logging: {
    fetches: { fullUrl: false },
  },
  experimental: {
    optimizePackageImports: ["@phosphor-icons/react"],
  },
};

module.exports = nextConfig;
