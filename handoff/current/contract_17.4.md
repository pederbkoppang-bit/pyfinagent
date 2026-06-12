# Contract -- phase-17.4 (stale-step closure): Researcher subagent calls Alpaca MCP during a dry-run harness cycle

Date: 2026-06-12. Trigger: Stop-hook flag on the stale in-progress step (verified real
against masterplan -- the 62.0-era inventory missed it). Durable contract file (the
rolling contract.md is OVERWRITTEN by this step's own verification command,
run_harness.py:355 -- research-identified sequencing hazard).

## Research-gate summary

Brief: handoff/current/research_brief_17.4.md (tier simple, gate_passed: true, 5 sources in
full incl. Claude Code sub-agents docs, issue #13898, alpaca-mcp-server repo/docs; recency
scan: Alpaca MCP V2 2026-04-09, #13898 closure + new mcpServers frontmatter). Findings:
- Verbatim criteria confirmed from masterplan (hook message was accurate, one nuance: the
  command's `grep -c ... || true` always exits 0, so criterion 1 is evidenced by the
  harness's own completion line in the log, not shell status).
- Dry-run is NOT rotted post-phase-60: py_compile + import surface PASS; pure-Python, no
  subagents, no LLM calls, ~6s, never touches Alpaca or paper state.
- Historical blocker RESOLVED: ALPACA keys present in shell env (PK prefix = paper);
  pinned alpaca-mcp-server==2.0.1 attaches (smoke exit 0, 61 tools, ALPACA_PAPER_TRADE=true
  hardcoded in .mcp.json).
- Structural fact: researcher.md's tools allowlist omits mcp__ tools, so the LITERAL
  "researcher calls mcp__alpaca__* mid-dry-run" plumbing is closed by configuration; the
  criteria anticipate this with the "recorded in the research brief OR dryrun log" arm.
- DECISIVE EVIDENCE captured by the researcher session itself: read-only get_account_info
  + get_clock invoked on the pinned server over MCP stdio -- paper account PA3VQZZLAKE2
  ACTIVE (same account as 17.3); deny-list clean; $0.
- no_regressions (phase-17 vocabulary per 17.1 critique): handoff-artifacts-only git diff,
  zero source mutations.

## Immutable success criteria (verbatim from masterplan 17.4)

1. "harness dry-run exits 0"
2. "at least one mcp__alpaca* tool call recorded in the research brief or dryrun log"
3. "handoff/current/alpaca-researcher-dryrun.log committed"
4. "no_regressions"

verification.command (verbatim): source .venv/bin/activate && python3
scripts/harness/run_harness.py --cycles 1 --iterations-per-cycle 1 --dry-run 2>&1 | tee
handoff/current/alpaca-researcher-dryrun.log | grep -c 'mcp__alpaca' || true

## Plan

1. Preserve in-flight rolling artifacts (62.1 + 62.2 copies -- DONE).
2. Run the verification command verbatim (accepting its documented side effects: rolling
   contract/brief overwrite + harness_log DRY_RUN append).
3. Append an evidence section to the dryrun log: the researcher's MCP-call record quoted
   from research_brief_17.4.md (puts the mcp__alpaca string in BOTH arms of criterion 2)
   with an honest note that the calls came from the researcher SESSION via MCP stdio, not
   from inside the dry-run process (which by design spawns no subagents -- 2026-04 notes
   field corroborates).
4. git diff audit: handoff-only (criterion 4). Commit the log (criterion 3).
5. ONE fresh Q/A -> harness_log cycle entry -> flip 17.4 done.

## Out of scope

researcher.md tools/mcpServers frontmatter changes (agent-file edits require operator
review per CLAUDE.md separation-of-duties; noted as a return-day candidate instead);
any Alpaca trading tools (deny-listed); 62.x work (resumes after this closure).
