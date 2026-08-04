# live_check -- step 82.4

Captured 2026-08-04.

The masterplan's `verification.live_check` for this step asks for **the FigJam
diagram URLs, plus a note recording whether the Figma MCP was reachable at
capture time**. Both are discharged below. The Q/A confirmed the CONTENT already
satisfied the requirement inside the deliverable, but the gate helper
(`.claude/hooks/lib/live_check_gate.py`) keys on THIS FILENAME, so the pointer
has to exist here.

## 1. FigJam diagram URLs

**None. No board was created.** This is a deliberate substitution recorded in
the deliverable, not an omission and not a failed attempt.

## 2. Was the Figma MCP reachable at capture time?

**YES.** `mcp__claude_ai_Figma__whoami` returned successfully on 2026-08-03:

```
handle: Peder "Norwegian Entertainment" Koppang
plans:  [{name: "...'s team", seat: "View", tier: "starter"}]
```

The connector was live. The blocker was the ENTITLEMENT it revealed, quoted
from Figma's own rate-limit document (read via
`file://figma/docs/rate-limits-access.md`):

> Seat `View, Collab` -- Starter/Professional/Organization/Enterprise:
> **Up to 6/month**

Six MCP tool calls per MONTH. Building and verifying a four-column board would
have consumed most of a month's quota in one sitting, and a write attempt on a
read-only seat risks spending calls for nothing.

## 3. What shipped instead

Mermaid, render-verified with `npx @mermaid-js/mermaid-cli`: 4 subgraphs, 5
nodes each, `direction TB` preserved on all four, **zero cross-subgraph edges**,
38 KB SVG with all four column titles present. Zero Figma quota consumed; the
diagram lives in the repo and diffs in git.

Operator decision recorded 2026-08-03 via AskUserQuestion: mermaid now, FigJam
later on request. The four columns port directly if a board is wanted.

## 4. Verbatim section from the deliverable

### Why these are mermaid and not FigJam

The step asked for FigJam boards via the Figma MCP. They are mermaid instead,
and this is the record of why.

**Figma MCP reachability, checked at capture time (2026-08-03):** the connector
WAS reachable — `whoami` returned the account and plan. What it returned is the
constraint: a **View seat on a Starter plan**, which Figma's own rate-limit doc
caps at **6 MCP tool calls per MONTH** (`View, Collab` seat row; only `Dev`/`Full`
seats get 200+/day). Building and verifying a four-column board would have
consumed most of a month's quota in one sitting, and a failed write on a
read-only seat would have spent calls for nothing.

**Operator decision, 2026-08-03 via AskUserQuestion:** mermaid now, FigJam later
on request. Mermaid costs zero quota, lives in the repo, and diffs in git.

**FigJam URLs: none — no board was created.** This is a deliberate substitution,
not an omission or a failure. The diagram above is render-verified
(`@mermaid-js/mermaid-cli`: 4 subgraphs, 5 nodes each, `direction TB` preserved
on all four, zero cross-subgraph edges, 38 KB SVG). If a board is wanted later,
these four columns port directly.

This is also why no verification criterion for this step names Figma: the MCP is
a claude.ai session connector, absent in headless runs, so a criterion depending
on it would make the step uncloseable for reasons unrelated to the work.
