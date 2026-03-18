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
providers.push(Passkey);

export default {
  providers,
  session: {
    strategy: "jwt",
    maxAge: 8 * 60 * 60, // 8 hours (OWASP)
  },
  pages: {
    signIn: "/login",
    error: "/login",
  },
  callbacks: {
    async signIn({ user, account }) {
      if (account?.provider === "google") {
        const profile = account as Record<string, unknown>;
        if (!profile.email_verified && profile.email_verified !== undefined) {
          return false;
        }
      }
      if (allowedEmails.length > 0 && user.email) {
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
