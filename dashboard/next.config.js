/** @type {import('next').NextConfig} */
const nextConfig = {
  // Dev proxy: the browser calls /api/... (same origin as the dashboard), and
  // Next forwards to the axum API. Keeps CORS out of the picture in dev.
  async rewrites() {
    const api = process.env.DRIFTGUARD_API_URL || "http://localhost:3000";
    return [{ source: "/api/:path*", destination: `${api}/:path*` }];
  },
};

module.exports = nextConfig;
