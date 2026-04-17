# AGPL-3.0 Isolation Policy (pyfinagent)

Status: active (phase-3.5 step 3.5.4)
Owners: Peder (governance) / harness (enforcement)

## Context

Two MCP servers we have chosen to adopt are licensed AGPL-3.0:

- **sec-edgar-mcp** (stefanoamorelli/sec-edgar-mcp) -- SEC EDGAR filings
- **openbb-mcp**    (OpenBB-finance/OpenBBTerminal)  -- OpenBB research tools

AGPL-3.0 is a copyleft license. If we link an AGPL server into a
derivative work and distribute that work (including over a network
service), the AGPL clause can require us to release our entire linked
codebase under AGPL as well. Since pyfinagent is a private, profit-
motivated trading system, we do not want accidental AGPL contamination.

## Isolation rules

1. **Subprocess boundary only.** AGPL servers run as separate OS
   subprocesses (stdio MCP) invoked by uvx / npx. They are NEVER
   imported into the Python backend process. The subprocess is a hard
   license boundary per standard copyleft interpretation (see:
   https://www.gnu.org/licenses/gpl-faq.html#MereAggregation).

2. **Read-only data flow.** pyfinagent consumes the JSON/text the
   AGPL server emits and applies its own derivative computations
   (signals, aggregations, agents). The RAW AGPL-originated data is
   public-domain government data (SEC filings, FRED-like public
   series). Our derivations are independent works.

3. **No bundled distribution.** pyfinagent source tree does not copy
   AGPL source code. We depend on the servers via package managers
   (uvx, pip, npx), which pull upstream versions at install time.

4. **Attribution preserved.** Any rendered output that exposes
   AGPL-sourced raw content in UI / Slack / reports MUST carry a
   short attribution line with the upstream repo URL. This is
   enforced in the Slack formatter + the frontend reports renderer.

5. **No network re-serving of the AGPL server.** We do NOT expose the
   AGPL server's HTTP/MCP interface to any external caller. All
   interactions are internal IPC between the pyfinagent backend and
   the MCP subprocess on the same machine / container.

## Enforcement

- `.mcp.json` lists AGPL servers with their licenses as comments (out
  of scope for MCP spec; we track in the license field of the 3.5.1
  candidate CSV).
- `handoff/mcp_risk_scores.json` tags license=AGPL-3.0 entries with
  `notes="AGPL isolation required"` and `risk_band` weighted higher.
- Phase-3.7 architecture (MAS Paper Trading & MCP Infrastructure)
  will add a boot-time assertion that the backend process does NOT
  import from any AGPL-licensed Python package.

## References

- GNU: When is a program and its plug-ins considered a single program? https://www.gnu.org/licenses/gpl-faq.html#GPLPlugins
- SFLC: Legal Issues Primer for Open Source / Free Software Projects -- Chapter on AGPL. https://www.softwarefreedom.org/resources/2008/foss-primer.html
- Google internal policy note: "Linking against AGPL libraries is prohibited in production services." (Public summary: https://opensource.google/documentation/reference/using/agpl-policy)
