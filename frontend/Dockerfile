FROM node:20-alpine AS deps
WORKDIR /app
# phase-75.16 (deps-03/04): `npm install` ignored the committed
# package-lock.json (507KB, tracked in git) -- non-reproducible builds could
# silently resolve different transitive versions than local dev. `npm ci`
# installs exactly what the lockfile pins and fails closed if it drifts.
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000

CMD ["node", "server.js"]
