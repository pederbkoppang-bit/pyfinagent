import NextAuth from "next-auth";
import authConfig from "@/lib/auth.config";

const { auth } = NextAuth(authConfig);

// phase-75.6 (gap2-01): route protection must NOT be gated on which providers are
// configured. `req.auth` validates the JWE session cookie and needs no provider present
// at request time -- providers matter only at login. The prior gate keyed enforcement on
// Google-credential presence (AUTH_GOOGLE_ID && AUTH_GOOGLE_SECRET), so a passkey-only
// (or otherwise Google-less) deployment silently disabled auth for EVERY route. The only
// sanctioned bypasses are the explicit, opt-in, default-off flags below -- never inferred
// from provider absence.

export default auth((req) => {
  const { pathname } = req.nextUrl;

  // Allow auth routes, login page, and static assets
  if (
    pathname.startsWith("/api/auth") ||
    pathname === "/login" ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon")
  ) {
    return;
  }

  // Explicit, opt-in bypasses ONLY (phase-75.6 gap2-01 -- never inferred):
  //  - LIGHTHOUSE_SKIP_AUTH=1 : perf measurement + the :3100 skip-auth Playwright rig
  //    (frontend.md live-UI protocol).
  //  - DEV_DISABLE_AUTH=1 : named dev-open mode, default-off, must be set deliberately.
  // Reintroducing an inferred bypass (e.g. keying on provider presence) is the gap2-01
  // hole and is forbidden.
  if (
    process.env.LIGHTHOUSE_SKIP_AUTH === "1" ||
    process.env.DEV_DISABLE_AUTH === "1"
  ) {
    return;
  }

  // Redirect unauthenticated users to login
  if (!req.auth) {
    const loginUrl = new URL("/login", req.url);
    return Response.redirect(loginUrl);
  }
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
