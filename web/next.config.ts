import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // deck.gl and maplibre ship ESM that Next.js needs to transpile
  transpilePackages: [
    "deck.gl",
    "@deck.gl/core",
    "@deck.gl/layers",
    "@deck.gl/react",
    "@deck.gl/geo-layers",
    "@deck.gl/mapbox",
    "@luma.gl/core",
    "@luma.gl/engine",
    "@luma.gl/webgl",
    "@luma.gl/shadertools",
    "@probe.gl/log",
    "@probe.gl/stats",
    "@probe.gl/env",
  ],
};

export default nextConfig;
