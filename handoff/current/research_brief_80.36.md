# Research Brief — phase-80.36: Risk Monitor fabricates SAFE/OK with zero data

Tier: T2 (moderate-complex). Started 2026-07-26. Status: IN PROGRESS (write-first).

Caller question: with the backend unreachable, several widgets on
`/paper-trading/positions` assert facts they cannot know (`SAFE`, `OK`, `0% / -15%`,
`+0,00 %` in positive-green, `Positions 0` when 2 are held). Need (A) UX/HCI guidance
on unknown-vs-zero-vs-nominal, (B) prior art on stale/unknown rendering in dashboards,
(C) a React/TS tri-state that makes the bad state unrepresentable, (D) internal
inventory + per-surface minimal fix, (E) the highest-risk way the fix changes the
healthy path.

---

## Search queries run (3-variant discipline)

(filled in as the session proceeds)

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|

## Identified but snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|

## Recency scan (2024-2026)

(pending)

## Internal code inventory

(pending)

