import NextAuth from "next-auth";
import Passkey from "next-auth/providers/passkey";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "./prisma";
import authConfig from "./auth.config";

/**
 * Full auth config with Prisma adapter (Node.js only).
 * Used by route handler and server components — NOT middleware.
 */
export const { handlers, auth, signIn, signOut } = NextAuth({
  ...authConfig,
  providers: [...(authConfig.providers || []), Passkey],
  adapter: PrismaAdapter(prisma),
  experimental: { enableWebAuthn: true },
  logger: {
    error: (code, ...message) => console.error(`[auth][error][${code}]`, ...message),
    warn: (code) => {
      if (code === "experimental-webauthn") return; // suppress known noise
      console.warn(`[auth][warn][${code}]`);
    },
  },
});
