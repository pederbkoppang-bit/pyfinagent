# Research Brief — Blocker-2: Autonomous-Harness Revert Hygiene
Generated: 2026-04-24
Tier: moderate

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-04-24 | official doc | WebFetch | Git present as infrastructure; no explicit revert-safety protocol. File-based comms is the sole state-management discipline documented. |
| https://arxiv.org/html/2604.13536 | 2026-04-24 | paper (arXiv) | WebFetch | YoloFS: staging layer isolates all mutations from base FS; agents cannot touch base until user commits. Commit-then-revert invariant: travel/snapshots operate only within staging scope, never on already-committed data. |
| https://github.com/pydantic/pydantic-ai/issues/4679 | 2026-04-24 | engineering issue | WebFetch | Two-phase commit pattern: tools return commit()/rollback() closures; framework executes atomically only after validation. Critical gap: no enforcement against out-of-scope rollbacks. |
| https://blog.gitbutler.com/agentic-safety | 2026-04-24 | authoritative blog | WebFetch | Six properties of agent-safe git: task isolation, clear branch boundaries, explicit commit selection, human review before push, cheap rollback, cross-branch contamination prevention. |
| https://forum.cursor.com/t/cursor-ide-silently-runs-git-stash-git-reset-head-during-active-agent-session-all-uncommitted-changes-lost/156146 | 2026-04-24 | community incident report | WebFetch | Cursor silently ran `git stash + git reset HEAD` during generation #38 (read-only analysis), losing 45 files (641+/183- lines) from a 2.5-hr session. Workaround: pre-flight `git stash list` check — if any entry exists, FULL STOP before next generation. |
| https://github.com/melihmucuk/leash | 2026-04-24 | code/tool | WebFetch | Leash blocks: `git reset --hard`, `git checkout -- .`, `git restore`, `git clean -f/-fd`, `git stash drop/clear`. Allows: `git stash`, `git stash pop`, `git checkout main`, `git commit`. Hardcoded pattern match, not configurable. |
| https://towardsdatascience.com/ai-agents-need-their-own-desk-and-git-worktrees-give-it-one/ | 2026-04-24 | authoritative blog | WebFetch | Worktree-per-agent pattern: each agent owns a dedicated filesystem path + branch; `git checkout` cannot pollute the main worktree. Agents see and may "clean up" diffs they did not author — worktrees eliminate cross-contamination. |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | official doc | Fetched; minimal rollback guidance — confirms gap in Anthropic canon |
| https://www.anthropic.com/engineering/built-multi-agent-research-system | official doc | Fetched; no git-safety content |
| https://arxiv.org/html/2603.05344v1 | paper | Fetched; mentions shadow git snapshots but no specific revert whitelist |
| https://addyosmani.com/blog/agent-harness-engineering/ | blog | Snippet only; covers feedback loops, not git safety |
| https://ericmjl.github.io/blog/2025/11/8/safe-ways-to-let-your-coding-agent-work-autonomously/ | blog | 2025 recency hit; covers allowedTools config, not stash discipline |
| https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution | vendor guide | Snippet; worktree model but no dirty-tree guard details |
| https://j2r2b.github.io/2019/03/26/ensure-no-uncommitted-changes.html | community | Canonical `git status --porcelain` check pattern |
| https://github.com/anthropics/claude-code/issues/6001 | issue | Feature request for native undo/checkpoint; confirms gap is known to Anthropic |
| https://www.mager.co/blog/2026-03-14-autoresearch-pattern/ | blog | 2026 recency; confirms karpathy autoresearch uses single-file scope constraint |
| https://blog.cloudflare.com/artifacts-git-for-agents-beta/ | vendor blog | Artifacts as versioned agent state; different architecture |

---

## Search Queries Run (3-variant discipline)

1. **Year-less canonical:** "autonomous agent git revert working tree safety whitelist only" — surfaces GitButler, worktree pattern, leash tool
2. **2025 window:** "AI coding agent git stash pop silent revert prevention 2025" — surfaces Cursor incident report, leash, claude-code issue #6001
3. **2026 frontier:** "autonomous agent git checkout revert working tree safety guardrails 2026" — surfaces AI agent guardrails enterprise guides, Anthropic harness posts

---

## Recency Scan (2024-2026)

Searched for 2024-2026 literature on agent git revert hygiene. Key new findings:
- **2026-04:** YoloFS (arXiv 2604.13536) introduces the first formalized staging-layer approach that makes agent-side revert physically impossible against committed base FS. Directly applicable as a conceptual model.
- **2026-04:** GitAgentProtocol (gitagent.sh) emerging open standard for git-native AI agents; too early to cite normatively.
- **2025-11:** Cursor IDE incident (forum.cursor.com) — real-world precedent for this exact bug class (silent stash+reset during autonomous session). 45 files lost. The pre-flight `git stash list` guard emerged from this incident.
- **2025-10-11:** Worktree-per-agent pattern gaining traction (nrmitchi.com, augmentcode.com, TDS article) as the structural fix rather than procedural patches.
- No 2024-2026 peer-reviewed paper directly addresses "commit-before-revert invariant for harness loops" — this is a practitioner gap, not an academic one.

---

## Key Findings

1. **The actual revert surface is `run_cycle.sh`, not Python code.** `scripts/mas_harness/run_cycle.sh` L39 runs `git checkout main` unconditionally before every cycle, then L42-44 runs `git stash push` if the tree is dirty, then L46-51 runs `git pull --rebase origin main`, then L51 pops the stash. If `git stash pop` encounters a conflict or the stash silently drops modified tracked files, any uncommitted edits in the working tree are gone. The Python harness (`run_harness.py`, `backend/autonomous_harness.py`) contains **zero git commands** — it only writes JSON files. (Sources: internal — `scripts/mas_harness/run_cycle.sh` L39-51)

2. **`git checkout main` at L39 is unconditional and runs before the dirty-tree check.** The sequence is: checkout → dirty-check → stash → rebase → pop. The checkout itself can discard working-tree changes on tracked files that aren't staged, silently, before the stash even fires. (Internal: run_cycle.sh L39-44)

3. **The pre-tool-use danger hook (`pre-tool-use-danger.sh`) does NOT cover `git checkout main` (no `--` or file-path argument).** It blocks `git reset --hard` and force-push but the plain `git checkout main` at L39 is not in its pattern set. (Internal: `.claude/hooks/pre-tool-use-danger.sh` L140-145)

4. **The `cycle_prompt.md` instructs Claude to run `git checkout main && git pull` as its first action.** So Claude-as-agent also emits this pattern during the `claude -p` call inside run_cycle.sh, creating a second revert surface inside the Claude session itself, governed by no guard. (Internal: `scripts/mas_harness/cycle_prompt.md` L10)

5. **`launchd` fires `com.pyfinagent.mas-harness` every 1800 seconds (30 min).** `StartInterval=1800`. This means if you apply a fix without committing, the next cycle 30 minutes later can silently revert it. The `com.pyfinagent.autoresearch` fires at 02:00 daily via `StartCalendarInterval`. (Internal: LaunchAgents plists)

6. **The autoresearch proposer's whitelist is correct in isolation.** `backend/autoresearch/proposer.py` L24-27 scopes writes to `{optimizer_best.json, candidate_space.yaml}` only. The revert problem comes from the shell wrapper, not from the Python proposer. (Internal: proposer.py L24-27)

7. **git reflog shows 609 `checkout` entries since 2026-03-24** out of 1,588 total reflog entries — 38% of all reflog entries are checkouts, virtually all `checkout: moving from main to main` (no-op branch switch but potentially touching the working tree if HEAD differs). (Internal: `git reflog`)

8. **YoloFS principle (arXiv 2604.13536):** the correct architecture separates the agent's mutation surface from the committed base. Once a change is committed, no agent path should be able to reverse it without an explicit human-initiated command. The pyfinagent harness currently lacks this invariant — committed edits are reachable via `git checkout main` + stash-pop failure. (Source: YoloFS paper)

9. **Cursor incident precedent (forum.cursor.com 2025):** an AI IDE lost 45 files via silent `git stash + git reset HEAD` during a read-only generation step. The recommended guard: before any automated cycle, assert `git stash list` is empty AND `git status --porcelain` output matches what the harness itself produced. (Source: Cursor forum)

10. **Leash tool (github.com/melihmucuk/leash) blocks `git checkout -- .` and `git restore` but explicitly allows `git checkout main`.** This confirms the community view that `git checkout <branch>` (without `--`) is considered safe — but as our incident shows, a stash-pop failure after the checkout is the actual loss vector. (Source: leash repo)

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/mas_harness/run_cycle.sh` | 81 | Launchd-invoked shell; runs git checkout + stash + pull + claude -p | PRIMARY RISK — L39-51 |
| `scripts/mas_harness/cycle_prompt.md` | 97 | Prompt for autonomous Claude session; instructs `git checkout main && git pull` | SECONDARY RISK — L10 |
| `/Users/ford/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` | 36 | Fires run_cycle.sh every 1800s | TRIGGER — StartInterval=1800 |
| `/Users/ford/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` | 39 | Fires run_nightly.sh at 02:00 daily | Secondary trigger |
| `.claude/hooks/pre-tool-use-danger.sh` | 158 | Blocks `git reset --hard`, force-push; does NOT block `git checkout main` | GAP at L140-145 |
| `scripts/autoresearch/run_nightly.sh` | 34 | Nightly autoresearch memo; no git commands | SAFE |
| `backend/autoresearch/proposer.py` | 111 | Proposer scoped to optimizer_best.json + candidate_space.yaml whitelist | SAFE — L24-27 |
| `backend/autonomous_harness.py` | 277 | Deprecated stub; no git commands | SAFE |
| `scripts/harness/run_harness.py` | 1207 | Quant optimizer harness; no git commands — writes JSON only | SAFE |
| `.claude/hooks/archive-handoff.sh` | 140 | Copies/moves handoff files on step done; uses `git mv` only | SAFE |

---

## Consensus vs. Debate (External)

**Consensus:**
- A dirty-tree guard before any automated git operation is universally endorsed (YoloFS, GitButler, Cursor incident, community patterns).
- Per-task isolation (worktrees or commit-before-start) is the structural fix; procedural guards (stash discipline) are second-best.
- Whitelist-only file mutation from the agent side is correct practice; the revert risk lives in the wrapper, not the agent logic.

**Debate:**
- `git stash` vs `git worktree` as the cycle isolation primitive: stash is simpler but lossy on pop failure; worktrees are structurally safe but add complexity. For a single-machine Mac-only deployment, stash-guard is the pragmatic minimum.
- Whether to block `git checkout <branch>` entirely (Leash allows it; GitButler says branch isolation is paramount). Pyfinagent's current architecture requires checkout-to-main; the fix is to commit before checkout, not to block checkout.

---

## Pitfalls (From Literature and Internal Inspection)

- **Stash pop on conflict silently leaves the stash on the stack** and leaves the tree in a partially-applied state. The `|| true` on L47 and L51 of run_cycle.sh swallows this failure silently. (run_cycle.sh L47, L51)
- **`git checkout main` before the dirty-tree check** means tracked-but-uncommitted changes can be discarded before stash even fires (git discards unstaged changes to tracked files during a checkout that updates those files). (run_cycle.sh L39)
- **`AUTOSTASH=0` default means if `git stash push` fails, pop is never attempted** — but the cycle continues. (run_cycle.sh L40-44)
- **Two-phase commit gap (pydantic-ai issue):** rollback closures have no enforcement against out-of-scope file mutation. Applicable here: `save_best_params(pre_cycle_best)` in run_harness.py L1173 is correctly scoped (only touches optimizer_best.json), but the shell wrapper operates repo-wide.
- **The `|| true` pattern on every git command in run_cycle.sh** means any git failure is silently swallowed and the cycle continues in an inconsistent state.

---

## Application to Pyfinagent (Fix Taxonomy — Ranked by Cost to Implement)

### Fix 1 — MINIMUM DIFF: Commit-before-checkout guard in run_cycle.sh (Cost: Low — 8-line change)

**Principle:** Before `git checkout main`, assert the working tree is clean. If dirty, either commit everything or abort — never stash-then-pop across a checkout.

```bash
# Insert BEFORE line 39 of run_cycle.sh:
DIRTY=$(git status --porcelain 2>/dev/null)
if [ -n "$DIRTY" ]; then
    echo "[$(date -Iseconds)] ABORT dirty tree before cycle: uncommitted changes detected" >> "$LOGFILE"
    echo "$DIRTY" >> "$LOGFILE"
    exit 1
fi
```

Then remove the entire stash-push/pop block (L41-51) — if the tree is guaranteed clean before checkout, no stash is needed. Replace `git checkout main` + `git pull --rebase` with just `git pull --rebase origin main` (rebase is a no-op when already on main and tree is clean).

**Why this works:** The revert happened because (a) Claude or a prior human edit left uncommitted changes in the tree, (b) the cycle fired, (c) `git checkout main` or stash-pop silently discarded them. A pre-flight abort on dirty tree makes the harness refuse to run until the operator commits or reverts manually. This is the "refuse to run if `git status --porcelain` has output" pattern.

**What it does NOT do:** It does not protect against Claude-inside-the-session issuing its own `git checkout main`. See Fix 2.

**Smoke test:**
```bash
# 1. Make an uncommitted edit
echo "# sentinel" >> backend/main.py
# 2. Run the cycle script directly (without launchd)
bash scripts/mas_harness/run_cycle.sh
# 3. Assert: script exits 1 and logs "ABORT dirty tree"
# 4. Assert: backend/main.py still contains "# sentinel"
grep "ABORT dirty tree" handoff/mas-harness.log && echo PASS || echo FAIL
grep "# sentinel" backend/main.py && echo PASS || echo FAIL
```

---

### Fix 2 — CYCLE PROMPT HARDENING: Replace checkout+pull instruction with pull-only (Cost: Low — 2-line change)

In `scripts/mas_harness/cycle_prompt.md` L10, change:

```
# BEFORE
git checkout main && git pull

# AFTER  
git pull origin main   # (already on main; checkout creates a second revert surface)
```

And add to the hard rules:

> 8. **Never run `git checkout`, `git restore`, `git reset`, or `git stash` on files outside the explicit list of files you edited this cycle.** If you need to undo a change, `git revert` it with a commit — never use working-tree destructive commands.

This removes the secondary revert surface inside the Claude session.

**Smoke test:**
```bash
# 1. Apply fix to cycle_prompt.md
# 2. grep that "git checkout main" no longer appears as an instruction
grep -n "git checkout main && git pull" scripts/mas_harness/cycle_prompt.md && echo FAIL || echo PASS
```

---

### Fix 3 — PRE-TOOL-USE HOOK EXTENSION: Block `git checkout -- <file>` and bare `git restore` (Cost: Low — 4-line addition)

Extend `.claude/hooks/pre-tool-use-danger.sh` case block (after L143) to catch file-level checkout and restore:

```bash
*"git checkout -- "*|*"git restore "*|*"git checkout HEAD -- "*)
    block_with_msg "file-level git checkout/restore detected — use git revert to undo committed changes" ;;
```

Note: this does NOT block `git checkout main` (branch switch, which is Leash's confirmed-safe pattern). It only blocks the file-discarding form `git checkout -- path` and `git restore path`.

**Why this is Fix 3 not Fix 1:** It guards Claude-as-interactive-agent but does NOT guard the shell script (hooks only fire for Claude tool calls, not for bash commands in run_cycle.sh). Fix 1 covers the shell; Fix 3 covers Claude inside the session.

**Smoke test:**
```bash
# In an interactive Claude Code session:
# Claude attempts: git checkout -- backend/main.py
# Expected: pre-tool-use-danger.sh exits 2, blocks the call
# Check audit log:
grep "file-level git checkout" handoff/audit/pre_tool_use_audit.jsonl | tail -3
```

---

### Fix 4 (Structural, Higher Cost) — Worktree isolation per cycle

Give each autonomous cycle its own `git worktree add /tmp/pyfinagent-cycle-<ts> main`. The cycle works in the worktree; main working tree is never touched by the harness. On cycle completion, `git worktree remove`. Cost: moderate (run_cycle.sh rewrite + path plumbing through cycle_prompt.md). Benefit: main working tree is physically isolated from harness operations — the revert class is structurally impossible.

**Smoke test:** `git worktree list` should show the cycle worktree during a run and be absent after.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (17 total)
- [x] Recency scan (last 2 years) performed and reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (run_harness.py, autonomous_harness.py, run_cycle.sh, cycle_prompt.md, proposer.py, both plists, all hooks)
- [x] Contradictions / consensus noted (stash vs worktree debate)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "report_md": "handoff/current/blocker-2-research-brief.md",
  "gate_passed": true
}
```
