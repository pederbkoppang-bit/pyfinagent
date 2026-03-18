import NextAuth from "next-auth";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "./prisma";
import authConfig from "./auth.config";

/**
 * Full auth config with Prisma adapter (Node.js only).
 * Used by route handler and server components — NOT middleware.
 */
export const { handlers, auth, signIn, signOut } = NextAuth({
  ...authConfig,
  adapter: PrismaAdapter(prisma),
  experimental: { enableWebAuthn: true },
});
