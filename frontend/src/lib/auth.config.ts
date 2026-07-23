import type { NextAuthConfig } from "next-auth";
import Google from "next-auth/providers/google";
import Passkey from "next-auth/providers/passkey";

/**
 * Edge-compatible auth config (no Prisma adapter).
 * Used by middleware.ts — must not import Node.js-only modules.
 */

const allowedEmails = (process.env.ALLOWED_EMAILS || "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

// phase-75.6 (gap2-02): mirror the backend's AUTH_ENFORCE_ALLOWLIST flag
// (backend/config/settings.py:574 `auth_enforce_allowlist`, shipped phase-75.1,
// default off). When set, an EMPTY allowlist DENIES all sign-ins (fail-closed); when
// unset, an empty allowlist admits (today's behavior) but logs a loud warning. The two
// layers only fail closed together when the operator flips both. Accept the same truthy
// forms pydantic does (`true/1/yes/on/t/y`, case-insensitive) so the frontend flag and
// the backend pydantic bool cannot desync.
const _TRUTHY = new Set(["true", "1", "yes", "on", "t", "y"]);
const enforceAllowlist = _TRUTHY.has(
  (process.env.AUTH_ENFORCE_ALLOWLIST || "").trim().toLowerCase()
);

if (allowedEmails.length === 0) {
  // Edge runtime supports console; this fires at module load. LOUD by design.
  console.warn(
    enforceAllowlist
      ? "[auth] AUTH_ENFORCE_ALLOWLIST is on with an EMPTY ALLOWED_EMAILS -- ALL sign-ins will be DENIED (fail-closed)."
      : "[auth] ALLOWED_EMAILS is empty -- any authenticated account is admitted. Set ALLOWED_EMAILS and AUTH_ENFORCE_ALLOWLIST=true to restrict logins."
  );
}

// Build providers list — only include Google if credentials are configured
const providers: NextAuthConfig["providers"] = [];
if (process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET) {
  providers.push(
    Google({
      clientId: process.env.AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_GOOGLE_SECRET,
    })
  );
}
// Passkey provider handled in auth.ts with experimental flag enabled
// providers.push(Passkey);

export default {
  providers,
  session: {
    strategy: "jwt",
    // phase-75.6 (gap2-05): shortened from thirty days to seven on a kill-switch-capable
    // cockpit, with an explicit updateAge (the JWT is re-issued at most once per day).
    // LIMITATION: JWT-strategy sessions cannot be revoked before expiry -- a leaked token
    // stays valid up to maxAge with no server-side blocklist. The Prisma adapter is
    // already wired (auth.ts), so migrating to `strategy: "database"` for true
    // server-side revocation is the tracked follow-up (see security.md). OWASP recommends
    // a 4-8h absolute timeout for a privileged cockpit; seven days is a real improvement
    // over the prior window, not the endpoint.
    maxAge: 7 * 24 * 60 * 60, // 7 days
    updateAge: 24 * 60 * 60, // re-issue the JWT at most once per day
  },
  pages: {
    signIn: "/login",
    error: "/login",
  },
  callbacks: {
    async signIn({ user, account, profile }) {
      // phase-75.6 (gap2-04): `email_verified` is an OIDC PROFILE claim, not on
      // `account`. The prior code aliased `account as profile`, so this check was dead
      // and unverified Google emails were admitted. Read it off the real `profile` param.
      // Cast: the Auth.js `Profile` type may not declare `email_verified`. Reject ONLY an
      // explicitly-unverified email -- a provider that OMITS the claim (`undefined`) is
      // permitted per OIDC and must not be rejected.
      if (account?.provider === "google") {
        const emailVerified = (
          profile as { email_verified?: boolean | string } | undefined
        )?.email_verified;
        if (
          emailVerified !== undefined &&
          emailVerified !== true &&
          emailVerified !== "true"
        ) {
          return false;
        }
      }

      // phase-75.6 (gap2-02): fail-closed when the flag is on and the allowlist is empty.
      if (enforceAllowlist && allowedEmails.length === 0) {
        return false;
      }

      // phase-75.6 (gap2-06): with an ACTIVE allowlist, a principal lacking an email is
      // REJECTED -- the prior `&& user.email` short-circuit waved it through to
      // `return true`.
      if (allowedEmails.length > 0) {
        if (!user.email) {
          return false;
        }
        if (!allowedEmails.includes(user.email.toLowerCase())) {
          return false;
        }
      }
      return true;
    },
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.email = user.email;
      }
      return token;
    },
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.id as string;
      }
      return session;
    },
    authorized({ auth }) {
      return !!auth?.user;
    },
  },
} satisfies NextAuthConfig;
