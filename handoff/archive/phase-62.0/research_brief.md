# Research Brief — phase-62.0 (goal-away-ops: hard-rules file + away goal + deferred flips + PreToolUse away-patterns)

Tier: moderate (caller-stated). Date: 2026-06-12. Agent: researcher (Layer-3 MAS, merged Explore).
Disclosed overrun: ~21 tool calls vs the 18 budget (4-item internal audit + one sandbox denial on `backend/.env` — researcher is DENIED that file; Main must grep the PAPER_* flag names itself). All floors met.

## Sources read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://code.claude.com/docs/en/hooks | 2026-06-12 | Official docs (Claude Code) | WebFetch, full page | PreToolUse stdin JSON: `{session_id, transcript_path, cwd, permission_mode, hook_event_name, effort.level, tool_name, tool_input}`. Exit 2 = block, stderr fed to Claude, stdout ignored. Exit 0 + JSON `hookSpecificOutput.permissionDecision: allow|deny|ask|defer` (+`permissionDecisionReason`, `updatedInput`) is the richer alternative; exit-2 takes highest precedence. **Any OTHER non-zero exit = non-blocking error, tool proceeds** (crash = fail-open by construction). Matcher: omitted/`*` = all tools; exact; `Edit\|Write` lists; regex; `if: "Bash(git *)"` evaluated pre-spawn and checks each `&&` subcommand and `$()` substitution. Parallel PreToolUse hooks: any block wins. `CLAUDE_PROJECT_DIR` + `CLAUDE_EFFORT` env available. |
| 2 | https://git-scm.com/docs/git-push | 2026-06-12 | Official docs (git) | WebFetch, full page | The complete force surface: `--force`, `-f`, `--force-with-lease[=<ref>[:<expect>]]`, `--force-if-includes` (no-op standalone), AND the **`+<refspec>` form (`git push origin +main`) which forces with NO flag at all**. Flags are position-free: `git push origin main --force` is valid — flag-after-refspec ordering. Server-side `receive.denyNonFastForwards` rejects all forced updates regardless of client flags. |
| 3 | https://ss64.com/mac/launchctl.html | 2026-06-12 | Official man-page mirror | WebFetch, full page | Removal surface is FOUR commands, not one: `bootout` (modern; "equivalent to unload in the legacy syntax"; takes domain target or plist path), `unload` (legacy; `-w` persists), `remove` (by label, returns immediately), `disable` ("cannot be loaded… persists across boots", label-based). `kill` stops a running instance without unloading; `kickstart -k` is the restart-only verb. A deny list covering only `bootout`+`unload` misses `remove` and `disable`. |
| 4 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-06-12 | Official engineering blog (Anthropic) | WebFetch, full page | Canonical harness article has **minimal explicit destructive-action guardrail content** — safety is implicit (file-based handoffs as checkpoints, evaluator-as-gate, "every component in a harness encodes an assumption about what the model can't do on its own"). Implication: the away-ops 3-layer enforcement (prompt rails + hook + sentinel reconciliation) fills a gap the canonical pattern leaves open; layering is the correct defense-in-depth shape since no single layer is documented as sufficient. |
| 5 | https://mywiki.wooledge.org/BashFAQ/050 | 2026-06-12 | Community-canonical (Greg's Wiki; lowest-tier slot) | WebFetch, full page | "Directly manipulating raw code strings is among the least robust of metaprogramming techniques." Only the shell can parse shell: quoting ambiguity, word splitting, variable expansion mean regex inspection of a command string **cannot** reliably determine which files it writes (sed -i / tee -a / heredoc / `F=…; echo >> $F` all evade). Implication: the .env pattern is a TRIPWIRE for high-confidence write shapes, not a parser — the daily sentinel flag-vs-token reconciliation (approved plan, enforcement layer 3) is the ground-truth backstop. |

## Snippet-only table (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/anthropics/claude-code/issues/24327 | Bug tracker | Exit-2 block can make Claude stop awaiting user input instead of acting on stderr (intermittent, 2026) — captured in Risks |
| https://github.com/anthropics/claude-code/issues/40580 | Bug tracker | "PreToolUse hook exit code ignored for subagent tool calls" — captured in Risks as coverage gap |
| https://www.atlassian.com/blog/it-teams/force-with-lease | Industry blog | Lease nullified by background fetch; Git 2.30 `--force-if-includes` — covered by source 2 |
| https://gist.github.com/pixelhandler/5718585 | Community gist | pre-push git-hook precedent; bypassable with `--no-verify` (our hook sits above git, unaffected) |
| https://sonim1.com/en/blog/git-force-push-safely | Blog | Redundant with source 2 |
| https://www.graphapp.ai/engineering-glossary/git/git-push---force-with-lease | Glossary | Redundant with source 2 |
| https://claudefa.st/blog/tools/hooks/hooks-guide | Blog | Redundant with official doc (source 1) |
| https://stevekinney.com/courses/ai-development/claude-code-hook-control-flow | Course notes | Confirms continue:false > decision:block > exit-2 hierarchy; redundant with source 1 |
| https://www.morphllm.com/claude-code-hooks | Reference site | Redundant with source 1 |
| https://linuxize.com/post/bash-heredoc/ | Tutorial | Heredoc mechanics; subsumed by source 5's stronger claim |

## Search queries (three-variant discipline)

- T1 hooks: `Claude Code PreToolUse hook exit code 2 block tool call JSON decision 2026` (current-year). Year-less canonical = direct fetch of the official doc (source 1); last-2-year window covered by the 2026 GH issues surfaced. Topic is too new (2025-2026 feature) for older prior art — stated explicitly per protocol.
- T3 force-push: `git push force protection pre-push hook --force-with-lease detect` (year-less canonical); recency via Atlassian/`--force-if-includes` hits; git-scm doc is the versionless canonical.
- T5 .env detection: `parsing shell command string pitfalls heredoc sed tee detect file modification` (year-less canonical); BashFAQ/050 is the founding prior art.
- T2 (Anthropic guardrails) + T4 (launchctl): direct canonical fetches of versionless references (engineering article, man page) — no year-variant search run; disclosed; recency addressed below.

## Recency scan (last 2 years)

Searched 2024-2026 window across topics. Findings: (a) `hookSpecificOutput.permissionDecision` + the `if` field + `updatedInput` are 2025-2026 additions to Claude Code — the JSON-decision path now exists alongside the exit-2 path our hook uses (both current; exit-2 remains valid and highest-precedence); (b) two OPEN 2026 issues (#24327 exit-2 stall, #40580 subagent exit-code ignored) qualify hook reliability — fed into Risks; (c) git force semantics unchanged since `--force-if-includes` (Git 2.30, 2021) — no new finding; (d) launchctl semantics stable since macOS 10.11 split — no new finding; (e) no 2024-2026 work supersedes BashFAQ/050's parse-impossibility argument.

## Internal code audit

| File | Lines | Finding |
|---|---|---|
| `.claude/settings.json` | 4-13 | PreToolUse has ONE entry, NO matcher (fires on every tool call), command-type → `pre-tool-use-danger.sh`. Extend that script; do not add a parallel entry (keeps the audit JSONL single-writer). |
| `.claude/settings.json` | 154-171 | Permissions deny layer already holds `Bash(git push --force *)`, `Bash(git push -f *)`, `Bash(git reset --hard *)` — second defense layer; same ordering gap as the hook (see Risks). |
| `.claude/hooks/pre-tool-use-danger.sh` | 19-43 | Input: `CLAUDE_TOOL_NAME`/`CLAUDE_TOOL_INPUT` env first, stdin-JSON fallback parsing `tool_name`/`tool_input` — matches the documented schema (source 1). New patterns slot into the existing `TOOL = "Bash"` branch (:78-159). |
| `.claude/hooks/pre-tool-use-danger.sh` | 10-17, 60-69, 72-75 | Exit 2 = block via `block_with_msg` (stderr + audit line); designed FAIL-OPEN on internal error; `CLAUDE_ALLOW_DANGER=1` escape hatch reads the HOOK process env (a `VAR=1 cmd` prefix in the Bash command string cannot set it — no trivial self-bypass). |
| `.claude/hooks/pre-tool-use-danger.sh` | 46-55, 140-145, 161-168 | Audit JSONL `handoff/audit/pre_tool_use_audit.jsonl`, single writer (repo-wide grep: only this script). Force-push case-glob :141 misses flag-after-refspec + `+refspec` (fix in same edit). `MCP_MIGRATE_TOKEN=granted` gate :163-167 is the in-file precedent for the tokens_cursor gate shape. |
| `.claude/hooks/archive-handoff.sh` | 66-67, 87-111 | Archives ONLY `status == "done"` ids newly seen vs baseline `.claude/.archive-baseline.json`. pending→deferred is invisible to it. |
| `.claude/hooks/auto-commit-and-push.sh` | 82, 106, 129-133 | `load_done_ids` collects only `status=="done"`; newly_done = worktree-vs-HEAD diff; empty → silent exit ("metadata only"). Deferred flips alone trigger NO commit/push. |
| `.claude/masterplan.json` | (walk) | All 10 ids exist in BARE form (`36.2` … `40.1`, no `phase-` prefix), all `status: "pending"`. Status census: done 734, pending 84, in-progress 12, **deferred 8**, dropped 5, superseded 3, blocked 1 → `deferred` matches existing vocabulary exactly. |
| `scripts/mas_harness/cycle_prompt.md` | 8-29, 86-95 | Precedent: numbered "Hard rules you MUST follow:" inline at top + "Context you have access to:" reading list naming rule files (CLAUDE.md, `.claude/rules/`). Away kickoff prompts should mirror: inline rails + `docs/runbooks/away-ops-rules.md` FIRST in the reading list (matches approved plan read order, `handoff/away_ops/approved_plan_2026-06-12.md:124`). |
| `handoff/away_ops/` | (ls) | Contains only `approved_plan_2026-06-12.md` → `tokens_cursor` is a NEW file; absent cursor must read as "no token". `handoff/current/active_goal.md` EXISTS (3371 B, 2026-06-11) → 62.0 refreshes, not creates. |

## Risks & gotchas

1. **.env false-positive (caller-required).** Operator keystrokes (`echo 'PAPER_X=true' >> backend/.env` in their own terminal) never pass through PreToolUse — hooks intercept only Claude tool calls, so the operator path is structurally safe; no carve-out needed. Token-authorized agent flips: gate on `handoff/away_ops/tokens_cursor` freshness (mtime within a short window, or content naming the flag/step), mirroring the `MCP_MIGRATE_TOKEN` precedent (:163). Read-only mentions (`grep PAPER_ backend/.env`, `cat`, `source`) must NOT block — match write shapes only (`>>`/`>` redirection into `backend/.env`, `sed -i`, `tee [-a]`, `perl -i`, `python … open(...'w')` with the path). Per BashFAQ/050 the regex can never be complete (heredocs, `$F` indirection evade) — tripwire only; sentinel reconciliation is the backstop.
2. **Fail-open vs fail-closed.** Project precedent is fail-open on internal error (`danger.sh:13-17`, both masterplan hooks `trap 'exit 0'`). Keep it: a crashed guard exiting non-2 must not brick 3 weeks of unattended sessions (doc source 1: non-2 nonzero = non-blocking). But on a MATCHED away-pattern with missing/stale cursor, the decision is deny (exit 2) — fail-SAFE per pattern, exactly the caller's framing.
3. **Hook coverage gaps.** (a) GH #40580: exit code possibly ignored for SUBAGENT tool calls — don't rely on the hook alone; add the launchctl/push patterns to the settings.json deny list too (layer 2). (b) GH #24327: exit-2 can stall Claude awaiting input — away wrapper already exits 0 on anything (approved plan:119-120); make `block_with_msg` stderr prescribe the next action ("write a token ask; move on" — rail 10). (c) The hook inspects only `TOOL=Bash` command strings — an **Edit/Write tool call on backend/.env bypasses it entirely**; extend matching to `tool_name in {Edit,Write}` with `tool_input.file_path` ending `backend/.env` (caller's spec says "Bash edits", but this hole is one tool-choice away). (d) Self-modification: an away session could edit the hook/settings themselves — cover via rules-file prohibition + sentinel checksum, not more regex.
4. **Force-push pattern gap (fix in same edit).** Current globs (`danger.sh:141`, settings deny:168-169) miss `git push origin main --force` (flags are position-free per git docs) and `git push origin +main`. New pattern needs: any `git push` containing `--force*`/`-f` anywhere, plus refspec tokens starting `+`.
5. **launchctl surface.** Deny must cover `bootout`, `unload`, `remove`, `disable` (and consider `kill`) against `com.pyfinagent.*` labels AND plist paths; `kickstart` stays allowed (rail 9). Blocking only `bootout`/`unload` leaves `remove`/`disable` open.
6. **Deferred-flip push hazard.** The flip edit itself is push-silent (auto-commit sees no new `done`), BUT the diff is HEAD-vs-worktree: if an uncommitted done-flip is sitting in masterplan.json when 62.0 edits it, the chain will commit+push EVERYTHING (`git add -A`). Do the 10 flips on a clean masterplan (done-set identical to HEAD). Flips land via Edit → the Edit-matcher chain (settings.json:76-100) fires identically.

## Recommendations

1. Extend `pre-tool-use-danger.sh` in place (new "away patterns" section after :159), keeping `block_with_msg` + audit JSONL; mirror the three new patterns into the permissions deny list where expressible (defense-in-depth, covers the subagent gap).
2. Fix the existing force-push ordering gap in the same edit; include `+refspec` and `--force-with-lease`/`--force-if-includes` variants.
3. Pattern the .env gate as: (Bash write-shape on `backend/.env` AND line content matching `PAPER_[A-Z_]*=`) OR (Edit/Write with file_path `backend/.env`) → require fresh tokens_cursor, else exit 2 with a "write a token ask" stderr.
4. For the 10 flips: use the literal status string `deferred` (8 existing uses), bare ids, add an `audit_note` field, leave `verification` untouched; sequence on a clean tree.
5. away-ops-rules.md referenced like the mas-harness precedent: numbered binding rails inline in each kickoff prompt + first entry in the session reading list.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
