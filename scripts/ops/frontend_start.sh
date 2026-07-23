#!/usr/bin/env bash
# scripts/ops/frontend_start.sh -- phase-75.11 (sre-ops-09).
#
# Pre-start build wrapper for the frontend launchd agent. The LIVE
# com.pyfinagent.frontend.plist runs `next dev --port 3000` today (verified
# 2026-07-24) while start_services.sh separately runs `next start -p 3000`
# -- two authorities, two different modes. This wrapper is the single
# surviving authority's entry point: it ensures a production build exists
# before `next start` (which refuses to run without one), then execs into
# it so launchd supervises the eventual `next start` process directly (no
# orphaned child, no double-authority race -- honors the
# feedback_second_next_dev_breaks_operator_3000 memory).
#
# "Stale" = any file under frontend/src, frontend/package.json, or a
# frontend/next.config.* newer than the build marker .next/BUILD_ID. A
# missing .next also triggers a build.
set -euo pipefail

REPO="${SRE_OPS_REPO:-/Users/ford/.openclaw/workspace/pyfinagent}"
FRONTEND="$REPO/frontend"
cd "$FRONTEND"

needs_build=0
if [ ! -f ".next/BUILD_ID" ]; then
    needs_build=1
else
    stale=$(find src package.json next.config.js next.config.ts next.config.mjs \
        -newer .next/BUILD_ID -type f 2>/dev/null | head -1)
    [ -n "$stale" ] && needs_build=1
fi

if [ "$needs_build" = "1" ]; then
    echo "[frontend_start] building (.next missing or stale)"
    npm run build
fi

echo "[frontend_start] exec next start -p 3000"
exec npx next start -p 3000
