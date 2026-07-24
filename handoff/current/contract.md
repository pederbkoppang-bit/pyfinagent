# Contract — phase-75.20: make the Q/A live-UI gate enforceable AND make the primary Q/A path actually read-only

- **Step id:** 75.20 (phase-75, P2, harness_required; executor: opus-tagged → Main-on-Fable GENERATE; Researcher + Q/A gates opus/max via Workflow)
- **Date:** 2026-07-24
- **Boundary (from step text):** harness config only, no product code. SEPARATION OF DUTIES — edits `.claude/agents/qa.md` + `.mcp.json` + `.claude/settings.json`; per CLAUDE.md the authoring session must not self-evaluate work that depends on these edits, and agent-file changes bind the Agent-tool roster only after a session restart. Consequence embraced up front: **the status flip is HELD** (75.18 precedent) — the live_check spec itself demands "the roster-live confirmation after restart", which this session structurally cannot produce. Next session: operator review of the qa.md diff, `scripts/qa/verify_qa_roster_live.sh`, append roster confirmation to live_check, then flip.

## Research-gate summary (gate PASSED — wf_0d03eec3-633)

Envelope: `tier=moderate, external_sources_read_in_full=6, snippet_only_sources=18, urls_collected=24, recency_scan_performed=true, internal_files_inspected=8, gate_passed=true`. Brief: `handoff/current/research_brief_75.20.md`. Every load-bearing claim of the 2026-07-20 prior assessment (`handoff/archive/misc/research_brief_playwright_qa.md`) revalidated against current disk state.

Load-bearing findings:

1. **Deny rules are deny-FIRST and NON-bypassable** (official permissions doc): a matching deny blocks the call even under `defaultMode: bypassPermissions`, and binds subagents AND Workflow-spawned agents. This is the mechanism for C2. Exact-name syntax `mcp__playwright__browser_run_code_unsafe` matches the existing `mcp__bigquery__execute-query` style. CAVEAT: the deny-rule typo warning exempts names containing `_` — every browser_* rule must be spelled exactly; a typo is silent.
2. **R11 — the immutable verification command's assert #3 is VACUOUS** (empirically confirmed): `any('user-data-dir' in str(a) and 'profile' not in str(a) for a in pw['args'])` is already True today because the flag TOKEN `--user-data-dir` matches. The command is immutable and stays; it must NOT be treated as C4 evidence. C4 evidence = the real `--isolated` addition + vendor citation + a two-client concurrency demonstration in live_check, plus a NON-vacuous replacement assert in the step's test file.
3. **R2 correction:** on-disk qa.md:4 is exactly `tools: Read, Bash, Glob, Grep, SendMessage` (no Write/Edit — this session's Agent-registry blurb listing Write/Edit for qa is a registry-side discrepancy, not disk truth; qa.md:464 independently forbids Edit/Write).
4. **C3 cannot use a session-wide Edit/Write deny** (breaks Main). The configuration lever is the qa-verdict.js `agentType` switch `'general-purpose'` → `'qa'` (binds the qa.md tools allowlist), which MUST be settled by a GENERATE-time empirical probe — not assumable from docs. Sub-agents doc: `tools:` accepts exact MCP names; ≥2.1.208 hard-errors on unresolved tool names (a bad grant fails loudly); built-ins in the list keep resolution non-empty so a cold playwright server degrades safely instead of false-PASSing.
5. **Recency (2025-2026):** arXiv:2606.20023 — agents over-select privileged tools and prompt-level controls give only limited mitigation → mechanical deny-rule enforcement over prose. arXiv:2511.19477 — read-only assistant agents lacking click/type give high utility with minimal risk; enforce safety in the execution layer, not the LLM. Anthropic harness-design (canonical ref) gave its evaluator the Playwright MCP to interact with the live page before scoring — the step's direction verbatim.
6. Dev-server lifecycle (frontend.md steps 1/3/5) is mutating and stays with Main (auto-memory `feedback_second_next_dev_breaks_operator_3000`).

## Hypothesis

Granting qa.md a NAMED read-only browser subset, denying the RCE-class playwright tools in settings.json (deny-first, bypass-proof, Workflow-binding), switching the primary qa-verdict launch to the restricted `qa` agent type (probe-verified), and adding `--isolated` to the playwright server makes qa.md §1c enforceable-by-configuration on both launch paths while REDUCING the primary path's surface (Edit/Write/full-MCP → the qa allowlist) — with zero product-code change and Main's own capture workflow intact.

**Declared deviation (with justification, subject to Q/A):** settings.json gains exactly TWO deny entries (`browser_run_code_unsafe`, `browser_evaluate`) — the RCE-class with no legitimate use by any session role, and exactly what the immutable command asserts. `browser_click`/`browser_type`/`browser_fill_form` are kept OUT of qa's grant (C1's test asserts their absence from the grant) but NOT session-denied: settings.json is session-wide and persistent, Main legitimately needs interaction tools for live-UI investigation (phase-70 S1 reproduction; pending phase-44 frontend work), and C3's evaluator-side enforcement comes from the agentType switch binding the grant. If the GENERATE probe shows the agentType switch does NOT bind, fall back to denying all five and accept Main's capability loss.

## Plan

1. **Probe FIRST (before any edit), recorded verbatim for live_check:** (a) Workflow `agentType:'general-purpose'` self-report of tool surface (the BEFORE state); (b) Workflow `agentType:'qa'` self-report (does the qa.md tools allowlist bind on the Workflow path? does it read the session-start snapshot or disk?). The probe decides the C3 mechanism and the deny-set branch above.
2. **Edits:**
   - `qa.md` tools line → `tools: Read, Bash, Glob, Grep, SendMessage, mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot, mcp__playwright__browser_take_screenshot, mcp__playwright__browser_console_messages` (read-only subset; no mutation tool).
   - `qa.md` §1c → capture taken BY the evaluator when the path allows it; Main-produced capture = explicitly-DEGRADED fallback; deterministic `select:` ToolSearch form for loading the browser schemas (keyword search surfaces run_code_unsafe/click in top-5 and misses navigate/snapshot); dev-server lifecycle stays MAIN's (observe-only, never start/kill).
   - `settings.json` permissions.deny += `mcp__playwright__browser_run_code_unsafe`, `mcp__playwright__browser_evaluate` (exact spelling; typo warning is silent for these names).
   - `.mcp.json` playwright args += `--isolated`, drop the fixed `--user-data-dir` pin (vendor: persistent profile serves one instance at a time; capture flow is auth-bypassed on :3100 so no login state is needed). Disclose: a connected stdio server does not respawn on edit — reconnect via /mcp or next session.
   - `qa-verdict.js` → `agentType: 'qa'` (if probe binds) + comment explaining the configuration-enforcement rationale.
3. **Test file** `backend/tests/test_phase_75_20_qa_browser_grant.py`: C1 grant assertions (subset present in the tools LINE; five mutation tools absent from the grant); C2 deny entries exact-match; C4 NON-vacuous isolation assert (`'--isolated' in args AND '--user-data-dir' not in args` — the honest replacement for the vacuous immutable assert); §1c degraded-fallback + select:-form + lifecycle prose pins. Mutation-tested per qa.md §4c incl. a fixture/stub mutation.
4. **Concurrency demonstration:** two concurrent `@playwright/mcp@0.0.76` clients — baseline shared-profile contention vs `--isolated` both-succeed; verbatim in live_check.
5. Run the immutable verification command (expect exit 0) — with the R11 vacuity of its assert #3 explicitly disclosed next to the real C4 evidence.
6. harness_log: operator-review request for the qa.md/settings/mcp edits + the next-session checklist (review → restart → verify_qa_roster_live.sh → append roster confirmation → flip). **Status flip HELD.**
7. Q/A via qa-verdict Workflow with a self-reference disclosure (the evaluator reads the edited qa.md as its rubric on this path) — same disclosure shape as 75.18.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.20)

> command: `python3 -c "import json; qa=open('.claude/agents/qa.md').read(); assert 'mcp__playwright__browser_navigate' in qa and 'mcp__playwright__browser_snapshot' in qa, 'read-only browser subset not granted'; assert 'browser_run_code_unsafe' not in qa.split('tools:')[1].split(chr(10))[0], 'RCE tool granted in tools line'; m=json.load(open('.mcp.json')); pw=m['mcpServers']['playwright']; assert '--isolated' in pw['args'] or any('user-data-dir' in str(a) and 'profile' not in str(a) for a in pw['args']), 'shared-profile concurrency hazard unfixed'; s=json.load(open('.claude/settings.json')); deny=' '.join(str(d) for d in s['permissions'].get('deny',[])); assert 'browser_run_code_unsafe' in deny and 'browser_evaluate' in deny, 'mutation browser tools not denied'"`

1. "qa.md's tools line grants EXACTLY the read-only browser subset needed by §1c (browser_navigate, browser_snapshot, browser_take_screenshot, plus at most browser_console_messages/browser_network_requests/browser_resize) and grants NO mutation tool; a test asserts browser_run_code_unsafe, browser_evaluate, browser_click, browser_type and browser_fill_form are absent from the grant"
2. ".claude/settings.json carries explicit deny rules for the playwright mutation/RCE tools so the restriction survives the primary Workflow path too, which inherits tools rather than declaring them -- defense in depth under defaultMode bypassPermissions"
3. "The primary path is CONSTRAINED, not expanded: qa-verdict.js (or its agent definition) no longer leaves the evaluator with unrestricted Edit/Write/Bash plus the full MCP surface; the read-only property Q/A claims is enforced by configuration, and a probe recorded in the live_check demonstrates the restriction actually binds rather than being asserted"
4. ".mcp.json's playwright server no longer shares a fixed persistent profile across concurrent clients (--isolated or a per-client user-data-dir), and the live_check records the vendor citation plus a demonstration that two concurrent clients no longer contend"
5. "qa.md instructs the deterministic select: ToolSearch form for fetching browser tools, and states that dev-server lifecycle (start :3100, kill it, verify :3000) remains MAIN's responsibility -- Q/A observes an already-running instance and never starts or kills a server"
6. "§1c is updated to say the capture must be taken BY the evaluator when the path allows it, with reading a Main-produced capture named as the explicitly-degraded fallback; harness_log carries the operator-review request for the qa.md edit and the next session verifies the new roster is live per scripts/qa/verify_qa_roster_live.sh"

live_check spec (verbatim): "handoff/current/live_check_75.20.md: verbatim verification command output (exit 0) + git diff --stat + a recorded probe showing the Q/A path's tool surface BEFORE and AFTER (proving the restriction binds) + a concurrency demonstration for the profile fix + the roster-live confirmation after restart."

## References

- `handoff/current/research_brief_75.20.md` (6 read-in-full: Claude Code permissions/settings/sub-agents docs, @playwright/mcp vendor docs, Anthropic harness-design, arXiv 2606.20023 + 2511.19477)
- `handoff/archive/misc/research_brief_playwright_qa.md` (2026-07-20 assessment, revalidated)
- qa.md §1c (:177-195), qa-verdict.js (:111-118), .mcp.json playwright block, settings.json deny (23 entries, zero playwright)
- CLAUDE.md separation-of-duties; 75.18 hold precedent; `feedback_second_next_dev_breaks_operator_3000`
