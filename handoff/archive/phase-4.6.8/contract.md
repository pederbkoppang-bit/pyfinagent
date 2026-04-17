# Sprint Contract -- Cycle 43 / phase-4.6 step 4.6.9

Research gate: done (Explore agent).

Findings:
- harness_log.md uses multi-line `## Cycle N ...` Markdown blocks, not
  TSV rows. "phase=4.6" appears in header, "result" is the first token
  of the **Decision:** line.
- MCP servers are IN-PROCESS (importable modules, no separate PIDs).
  The "no stray mcp_server PIDs" criterion is trivially satisfied.
- finalize.py does not yet exist. boot_backend.py's cleanup already
  guarantees port 8765 is free after each step run.

Success criteria (immutable):
- handoff/harness_log.md gained exactly one row
- new row contains phase=4.6 and result field
- no uvicorn process bound to 8765 after exit
- no stray mcp_server PIDs vs pre-test snapshot

Plan: write scripts/smoketest/steps/finalize.py that (a) snapshots line
count of harness_log.md + an mcp_server PID set before-and-after, (b)
appends exactly one `## Cycle N -- phase=4.6 ...` block with
`**Decision:** PASS`, (c) verifies :8765 is unbound via
socket.connect_ex, (d) asserts mcp_server PID delta = 0, (e) emits JSON
verdict.
