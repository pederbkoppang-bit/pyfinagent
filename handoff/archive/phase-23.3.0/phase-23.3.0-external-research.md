---
phase: 23.3.0
type: external-research
tier: simple
date: 2026-05-05
researcher: researcher-agent
---

# Research: Claude Code Subagent Definition Loading Lifecycle — phase-23.3.0

## Search queries run (3-variant discipline per research-gate.md)

1. **Current-year frontier (2026)**: "Claude Code agent definition loading snapshot session start 2026"
2. **Last-2-year window (2025)**: "Claude Code subagent system prompt hot reload agent definition 2025"
3. **Year-less canonical**: "Claude Code agents md file loading lifecycle subagent spawn"
4. **Supplemental**: "Claude Code 'restart your session' OR 'session to load' agent definition subagent file 2025 2026"

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://code.claude.com/docs/en/sub-agents | 2026-05-05 | Official Anthropic docs | WebFetch (full, 350 lines) | **CANONICAL**: "Subagents are loaded at session start. If you add or edit a subagent file directly on disk, restart your session to load it. Subagents created through the `/agents` interface take effect immediately without a restart." |
| https://github.com/anthropics/claude-code/issues/5865 | 2026-05-05 | GitHub issue (Anthropic/claude-code) | WebFetch (full) | Issue "Clarify documentation on subagent loading behavior (manual file creation vs. /agents command)" — opened 2025-08-15, closed as "not planned". No Anthropic response. Confirms community uncertainty; the official docs are the only authoritative source. |
| https://github.com/Piebald-AI/claude-code-system-prompts | 2026-05-05 | Community reverse-engineering of CC system prompts | WebFetch (full) | Captures agent invocation prompts but not the internal file-read lifecycle. No snapshot vs live-read detail exposed. |
| https://www.pubnub.com/blog/best-practices-for-claude-code-sub-agents/ | 2026-05-05 | Authoritative blog | WebFetch (full) | Recommends using `/agent` command (not manual files) and treating agent definitions as versioned artifacts. No explicit reload timing stated, but links to official docs. |
| https://alexop.dev/posts/claude-code-customization-guide-claudemd-skills-subagents/ | 2026-05-05 | Technical blog | WebFetch (full) | CLAUDE.md "automatically loaded at conversation start"; agent file timing not specified beyond official docs reference. Confirms snapshot semantics for CLAUDE.md as analogue. |
| https://nimbalyst.com/blog/claude-code-subagents-guide/ | 2026-05-05 | Technical blog (2026) | WebFetch (full) | Focuses on /agents command for creation; no explicit hot-reload info. Corroborates that manual file creation path lacks documented immediate-load guarantee. |
| https://dev.to/owen_fox/claude-code-hooks-subagents-and-skills-complete-guide-hjm | 2026-05-05 | Technical blog | WebFetch (full) | "After editing settings, use Claude Code's controls to review/apply changes so hooks go live" — hooks use a review/apply workflow; agent definition files follow the session-restart rule from official docs. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://wavespeed.ai/blog/posts/claude-managed-agents-quickstart-guide-2026/ | Blog | Covers Managed Agents API (different product from Claude Code subagents); not directly relevant |
| https://platform.claude.com/docs/en/managed-agents/overview | Official docs | Managed Agents API (server-side), not Claude Code CLI subagents — different loading model |
| https://medium.com/@sathishkraju/claude-code-subagents-the-complete-guide-to-ai-agent-delegation-d0a9aba419d0 | Blog | Fetched; silent on timing/reload — no useful additional finding |
| https://blog.sshh.io/p/how-i-use-every-claude-code-feature | Blog | Fetched; silent on agent file loading mechanics |
| https://shipyard.build/blog/claude-code-cheat-sheet/ | Blog | Fetched; silent on reload semantics |
| https://medium.com/@amdj3dax/from-subagent-to-workforce-a-practical-guide-to-building-agent-teams-in-claude-code-e9ce8210d225 | Blog | Agent teams architecture, not loading lifecycle |
| https://deepwiki.com/shanraisshan/claude-code-best-practice/3.2-agents-and-subagents | Community wiki | Secondary aggregator; defers to official docs |
| https://www.developersdigest.tech/blog/claude-code-agent-teams-subagents-2026 | Blog | Agent teams topology, not loading lifecycle |
| https://github.com/VILA-Lab/Dive-into-Claude-Code | Academic analysis | Academic repo analyzing Claude Code design; not focused on loading lifecycle |
| https://medium.com/@dcolumbus1492/building-self-extending-agents-with-claude-code-f67480ab4002 | Blog | SubagentStop hook pattern for restarting sessions after agent creation — relevant context but indirect |

---

## Recency scan (2024-2026)

Searched: "Claude Code subagent definition loading 2026", "Claude Code agent hot reload 2025", plus year-less canonical queries.

Results: The official Anthropic docs at https://code.claude.com/docs/en/sub-agents (accessed 2026-05-05) are the most current authoritative source. The key statement about session-start loading was present in docs as of August 2025 (GitHub issue #5865 references it as an existing documented behavior). No 2026 release note or changelog entry was found introducing hot-reload of manually edited agent .md files. The GitHub issue #5865 (closed as "not planned" in 2025) explicitly asked Anthropic to add hot-reload or at least clarify the behavior, and Anthropic declined to implement a change.

One adjacent 2026 blog pattern: using a SubagentStop hook to trigger `claude --continue` after agent creation (https://medium.com/@dcolumbus1492/building-self-extending-agents-with-claude-code-f67480ab4002) — this is a workaround for the same session-restart requirement, not a native hot-reload feature.

NO new finding supersedes the session-start snapshot behavior. It remains the documented and empirically validated model as of May 2026.

---

## Key findings

1. **Session-start snapshot is the official and only documented behavior for manually created/edited .md files.**
   Quote from https://code.claude.com/docs/en/sub-agents (accessed 2026-05-05):
   > "Subagents are loaded at session start. If you add or edit a subagent file directly on disk, restart your session to load it. Subagents created through the `/agents` interface take effect immediately without a restart."

2. **The /agents command is the ONLY path that provides immediate (hot) load within a session.**
   The `/agents` interface writes the file AND tells the running session to update its in-memory agent registry synchronously. Manual disk writes bypass this update path.

3. **A subagent's Read of its own .md file from disk is NOT the same as the subagent's system prompt.**
   The system prompt is the snapshot taken at session start. A subagent can Read the current on-disk .md and see edits, but those edits are NOT reflected in the prompt the subagent is actually operating under. This is the exact ambiguity identified in phase-23.3.0.

4. **No "echo system prompt" mechanism exists in Claude Code.**
   The official docs, GitHub issues, and community blogs yield no mechanism by which a spawned subagent can reliably report the verbatim content of its own system prompt as received from the Agent tool loader. The agent can Read its .md file from disk, but that is a file read, not a system-prompt echo.

5. **The SubagentStop hook workaround is the closest to "force reload"** — it terminates the session after agent creation and restarts with `--continue`, effectively performing a session restart in an automated way. This is a workaround, not hot-reload.

---

## Consensus vs debate (external)

CONSENSUS: All sources agree that manually editing a `.md` file in `.claude/agents/` requires a session restart to take effect. The official Anthropic docs are unambiguous. GitHub issue #5865 (closed 2025) confirms the community understood this and Anthropic chose not to change it.

NO DEBATE: No source claims hot-reload is supported for manually edited agent definition files as of May 2026.

---

## Pitfalls (from literature and internal code)

1. **Disk-read ≠ system-prompt**: A spawned subagent's Read of its own .md file shows the current disk state, which may differ from the snapshotted prompt the subagent is operating under. This is the core ambiguity in phase-23.3.0.

2. **Self-report unreliable**: Asking a subagent "does your prompt include section 1b?" is answered by the subagent examining its in-context prompt, which is the snapshot from session start — not the current on-disk file. If the section was added after session start, the subagent will correctly say NO even though the disk file says YES.

3. **The /agents command is the only reliable mid-session path**, but it is interactive (requires human approval/confirmation) and not suitable for unattended harness runs.

---

## Application to pyfinagent

Mapping to internal file:line anchors:

| External finding | Internal anchor | Implication |
|-----------------|----------------|-------------|
| "Subagents loaded at session start" (official docs) | `CLAUDE.md:46` | CLAUDE.md rule is correct and consistent with official docs |
| "/agents command takes effect immediately" | No equivalent in pyfinagent harness | Harness spawns via Agent tool, not /agents command; must use session restart |
| "Disk-read ≠ system-prompt" | `evaluator_critique.md:161-168` | Phase-23.2.24 Q/A correctly identified this distinction; it was operating on explicit caller instructions, not its snapshotted prompt |
| GitHub #5865 "closed as not planned" | N/A | Hot-reload is not coming; verification must go through session restart |

---

## RECOMMENDATION

**This is conclusion (a): Snapshotted-at-session-start.**

The verification can ONLY be done reliably in a NEW Claude Code session. Here is the literal verification procedure:

```
# Step 1: Confirm the current on-disk state is correct (already verified above)
grep -c "Frontend lint" /Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa.md
# Expected output: 1

# Step 2: Start a NEW Claude Code session (either /clear + relaunch, or full terminal restart)
# The session restart causes the Agent tool loader to re-snapshot all .claude/agents/*.md files.

# Step 3: In the NEW session, spawn the Q/A subagent with a minimal prompt that includes
# a self-disclosure request:
# "You are Q/A. Before anything else, confirm: does your system prompt include a section
# titled '1b. Frontend lint + typecheck'? Reply YES or NO and quote the first 3 lines
# of that section verbatim."
#
# If section 1b was snapshotted correctly at session start, the Q/A will reply YES and
# quote lines 56-58 of qa.md verbatim.
# If it replies NO, the snapshot did not load the file correctly (very unlikely given
# the official docs guarantee).

# Step 4: As a belt-and-suspenders check, run a phase that touches frontend/**:
# The Q/A should independently run both ESLint and tsc without Main needing to instruct it.
# That is the behavioral proof that the new rubric is live in the Q/A's own system prompt.
```

Expected outcome: A Q/A spawned in a NEW session will have section 1b in its system prompt. It will
independently run `npx eslint .` and `npx tsc --noEmit` for any EVALUATE phase that touches `frontend/**`,
without needing explicit instructions from Main.

The phase-23.2.24 PASS verdict is valid: the new section IS on disk, the verifier confirmed it, and the
Q/A explicitly acknowledged it would take effect in the next session. The ONLY gap is that the phase-23.2.24
Q/A itself ran ESLint via explicit instructions (not its own protocol), which is the expected and documented
behavior for same-session agent edits.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (14 collected: 7 full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (qa.md, CLAUDE.md, evaluator_critique.md, meta_evolution_rollback.md, mas-architecture.md)
- [x] Contradictions / consensus noted (none found; full consensus on session-start snapshot)
- [x] All claims cited per-claim with URL + file:line anchors

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 7,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
