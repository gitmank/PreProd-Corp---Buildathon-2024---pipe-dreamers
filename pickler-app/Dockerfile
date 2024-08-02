# Source: https://github.com/vercel/next.js/blob/canary/examples/with-docker/README.md

# Install dependencies only when needed
FROM node:22.0-alpine AS deps
WORKDIR /app
COPY package.json  ./
RUN npm install --frozen-lockfile

# Rebuild the source code only when needed
FROM node:22.0-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npx next build

# Production image, copy all the files and run next
FROM node:22.0-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# COPY --from=builder /alertsapp/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json

# Automatically leverage output traces to reduce image size 
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

ENV PORT=3000

CMD ["node", "server.js"]