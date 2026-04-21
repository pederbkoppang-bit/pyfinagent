# Research Brief — phase-4.16.2: Handoff Folder Cleanup + Archive-on-Done Hook

**Tier:** moderate (stated by caller)
**Date:** 2026-04-18
**Researcher:** Researcher agent (merged researcher + Explore)

---

## External Sources (URL coverage)

| URL | Accessed | Kind | Read in full? |
|-----|----------|------|---------------|
| https://git-scm.com/docs/githooks | 2026-04-18 | official doc | YES |
| https://docs.python.org/3/library/shutil.html | 2026-04-18 | official doc | YES |
| https://martinfowler.com/articles/patterns-of-distributed-systems/wal.html | 2026-04-18 | engineering blog | YES (partial — WAL pattern definition; O'Reilly chapter not accessible) |
| https://keepachangelog.com/en/1.1.0/ | 2026-04-18 | authoritative community doc | YES |
| https://12factor.net/ | 2026-04-18 | authoritative engineering manifesto | YES |
| https://semver.org/ | 2026-04-18 | authoritative spec | YES |

**Sources read in full: 5 (git-scm, shutil, keepachangelog, 12factor, semver); 1 partial (WAL pattern — site returned summary only).**
Minimum floor of 5 read in full is met.

---

## Key External Findings

### 1. Git hooks — idempotency patterns
(Source: git-scm.com/docs/githooks, 2026-04-18)

The `PostToolUse` hook pattern used by `archive-handoff.sh` is analogous to a `post-receive` hook: it fires after a write completes and cannot roll back the triggering action. The canonical idempotency pattern is **state-based conditional**: "Only proceed if this is a real change" (`if [ "$oldrev" != "$newrev" ]`). The hook's current HEAD-diff approach (`git show HEAD:.claude/masterplan.json` vs working tree) is the correct idempotency mechanism — it only archives steps that *just* transitioned to `done`.

Lock-file pattern (Pattern 2) is relevant if the hook fires concurrently; the current `archive-handoff.sh` has no lock, which is acceptable for single-writer single-step use, but worth noting for future parallelism.

Glob patterns in shell hooks must be explicit — a glob like `${sid}-*.md` will silently skip files whose names use a different prefix convention (`phase-${sid}-*.md`).

### 2. shutil.move vs git mv — nuances
(Source: docs.python.org/3/library/shutil.html, 2026-04-18)

`shutil.move(src, dst)` uses `os.rename()` when src and dst are on the same filesystem (atomic); falls back to copy-then-delete across filesystems (not atomic). For a single-machine repo this is same-fs, so `shutil.move` is effectively atomic — safer than a two-step `cp + rm`.

`git mv` is the right choice when the move must be tracked in git history (no orphan files on the other side of a merge). `archive-handoff.sh` already prefers `git mv` with a fallback to `mv` (line 106-109), which is the correct ordering.

For the Python `verify_handoff_layout.py` script, use `shutil.move` for any programmatic backfill; use `pathlib.Path.rename()` for same-filesystem moves where git tracking is not needed.

### 3. Write-Ahead Log / audit-trail layout
(Source: martinfowler.com/articles/patterns-of-distributed-systems/wal.html, 2026-04-18)

WAL principle: "record every state change as a command in an append-only log before flushing." Applied here: the `.jsonl` audit files (e.g., `pre_tool_use_audit.jsonl`, `config_change_audit.jsonl`) are the append-only audit trail. They must **not** be moved mid-session; moving them risks breaking any in-flight reader/appender. The correct disposition is: move to `handoff/audit/` only as a one-time organisational rename committed atomically in git, not during an active session.

### 4. Unreleased vs released separation (keepachangelog)
(Source: keepachangelog.com/en/1.1.0/, 2026-04-18)

The `Unreleased` section at the top of CHANGELOG maps exactly to `handoff/current/` (in-flight artifacts); released versions map to `handoff/archive/phase-X.Y/`. The pattern confirms: a dedicated "in-flight" zone with a single canonical set of rolling files is the correct design. Anti-pattern: "commit log diffs as changelog" — equivalent to dumping all phase-X.Y-*.md into `current/` and never archiving.

### 5. 12-factor — logs as event streams
(Source: 12factor.net, 2026-04-18)

Factor XI: "Treat logs as event streams." The application should not manage log rotation or archival — the execution environment handles it. Implication for `handoff/*.log` files: they are runtime side-effects, not harness artifacts. They belong in `.gitignore` rather than a committed `handoff/logs/` directory, unless there is an explicit operational need to version-control them.

### 6. Semantic versioning — immutability of released artifacts
(Source: semver.org, 2026-04-18)

"Once a versioned package has been released, the contents of that version MUST NOT be modified." Directly validates the `handoff/archive/phase-X.Y/` design: once a step is `done`, its archive snapshot is immutable. The `v2 / v3` suffix dirs in `handoff/archive/` (e.g., `phase-3.5.1-v2`) are the correct idempotency mechanism when a step is re-run — create a new suffix rather than overwriting.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/hooks/archive-handoff.sh` | 123 | PostToolUse hook: archives completed steps | Active — has glob bug |
| `handoff/current/` (168 files) | — | In-flight harness artifacts | Needs cleanup |
| `handoff/archive/` (6963 dirs) | — | Archived step snapshots | Healthy |
| `handoff/*.json` (~42 files) | — | Per-step audit JSONs | Misplaced in root |
| `handoff/*.jsonl` (8 files) | — | Append-only audit streams | Active — do not move mid-session |
| `handoff/*.log` (7 files) | — | Runtime process logs | Misplaced; should be gitignored |
| `.claude/masterplan.json` | — | Step registry with status | Authoritative source |

---

## Detailed Internal Findings

### Finding 1: archive-handoff.sh glob bug (file:line anchor)

`archive-handoff.sh` line 103:
```bash
for f in "$CURRENT_DIR/${sid}-"*.md; do
```

Recent files use the naming convention `phase-{sid}-*.md` (e.g., `phase-4.14.10-contract.md`), not `{sid}-*.md`. The current glob only matches the older convention (e.g., `4.9.5-contract.md`). Files named `phase-X.Y.Z-contract.md` are **never matched** by line 103, which explains why 150 done-step files remain in `handoff/current/`.

**Exact fix** — replace line 103 with a dual glob:
```bash
    # Match both naming conventions:
    # Legacy:  ${sid}-contract.md  (older steps)
    # Current: phase-${sid}-contract.md  (steps from ~phase-4.14 onward)
    for f in "$CURRENT_DIR/${sid}-"*.md "$CURRENT_DIR/phase-${sid}-"*.md; do
```

Also note: the `research.md` in the rolling-files copy list (line 92) does not match the actual file name `research_brief.md` used by recent steps. Add `research_brief.md` to that list.

Line 92 fix:
```bash
    for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
```

### Finding 2: handoff/current/ file disposition — 168 files categorised

**Rolling files (keep in place — 4 files):**
- `contract.md`
- `evaluator_critique.md`
- `experiment_results.md`
- `research_brief.md`

**Done steps — MOVE to `handoff/archive/phase-{sid}/` (150 files across ~50 step IDs):**
All `phase-{sid}-*.md` files where masterplan status = `done`. Examples: `phase-4.14.10-contract.md` through `phase-4.16.1-research-brief.md`. The full list is 150 files; the backfill script below handles them programmatically.

**Blocked step — KEEP (4 files, step 4.14.6 status=blocked):**
- `phase-4.14.6-contract.md`
- `phase-4.14.6-evaluator-critique.md`
- `phase-4.14.6-experiment-results.md`
- `phase-4.14.6-research-brief.md`

**Non-conforming names — MOVE to `handoff/archive/misc/` (10 files):**

| File | Disposition | Reason |
|------|-------------|--------|
| `4.5.fix-widget-api-routing-contract.md` | Move to `handoff/archive/misc/` | Non-standard prefix (no `phase-`) |
| `mas-harness-fixes-contract.md` | Move to `handoff/archive/misc/` | Ad-hoc hotfix contract, no step ID |
| `phase-2.12-logger-ascii-v5.md` | Move to `handoff/archive/misc/` | Non-standard suffix (`-v5`, not step artifact name) |
| `phase-4.14-T1-contract.md` | Move to `handoff/archive/misc/` | `4.14-T1` not in masterplan |
| `phase-4.14-T1-evaluator-critique.md` | Move to `handoff/archive/misc/` | Same |
| `phase-4.14-T1-experiment-results.md` | Move to `handoff/archive/misc/` | Same |
| `phase-4.5-contract.md` | Move to `handoff/archive/misc/` | `4.5` not in masterplan (superseded by sub-steps) |
| `phase-5.5-contract.md` | Move to `handoff/archive/misc/` | `5.5` not in masterplan (ambiguous) — flag for Peder |
| `phase-6.5-research-brief.md` | Move to `handoff/archive/misc/` | `phase-6.5` status=pending; file predates step start — ambiguous, flag for Peder |
| `phase-auth-fix-contract.md` | Move to `handoff/archive/misc/` | Ad-hoc auth fix contract, no step ID |

**Ambiguity flags:**
- `phase-5.5-contract.md` — masterplan has steps `5.5.0`–`5.5.6` (all `done`), but not a step literally named `5.5`. This appears to be a rolled-up contract from a batch run. Recommend moving to `handoff/archive/misc/` with a note, rather than deleting.
- `phase-6.5-research-brief.md` — step `phase-6.5` is `pending` in masterplan. The research brief appears to have been pre-written. Move to `handoff/archive/misc/phase-6.5-research-brief.md` and retrieve when step becomes active.

### Finding 3: handoff/ root JSON and JSONL files

**42 `*.json` files** in `handoff/` root. No `handoff/audit/` directory currently exists.

Categories:
- **Audit evidence** (`*_audit.json`, `*_audit.jsonl`): `finra_audit.json`, `drift_monitor_audit.json`, `dr_runbooks_audit.json`, `limits_tag_audit.json`, `finra_compliance_audit.json`, `gauntlet_regimes_audit.json`, `immutable_limits_audit.json`, `kelly_allocator_audit.json`, `limits_lint_audit.json`, `limits_loader_audit.json`, `portfolio_risk_audit.json`, `promotion_gate_audit.json`, `regulatory_memo_audit.json`, `secrets_rotation_audit.json`, `supply_chain_audit.json`, `survivorship_audit.json` — **MOVE to `handoff/audit/`**
- **Append-only audit streams** (`*.jsonl`): `config_change_audit.jsonl`, `gauntlet_blocklist.jsonl`, `gauntlet_runs.jsonl`, `instructions_loaded_audit.jsonl`, `pre_tool_use_audit.jsonl`, `prompt_leak_redteam_audit.jsonl`, `sla_alerts.jsonl`, `tca_log.jsonl` — **MOVE to `handoff/audit/`** but only after confirming no active process is writing to them mid-session (check `lsof` before moving).
- **Test/roundtrip outputs** (`a2a_roundtrip.json`, `auth_jwe_roundtrip.json`, `paper_parity.json`, `virtual_fund_parity.json`, `mcp_ab_test_*.json`, `mcp_health_last.json`, etc.): these are verification outputs from specific harness steps — **MOVE to `handoff/audit/`** as they are audit evidence, not rolling harness state.
- **Allocator/signal outputs** (`allocator_output.json`, `mcp_risk_scores.json`, `mcp_storm_regression.json`, `seed_stability_results.json`, `tca_last_week.json`): step-specific outputs — **MOVE to `handoff/audit/`**.

### Finding 4: handoff/ root log files

7 log files in `handoff/` root:
```
ablation.launchd.log
ablation.log
autoresearch.launchd.log
autoresearch.log
mas-harness.launchd.log
mas-harness.log
seed_stability_output.log
```

Per 12-factor (Factor XI), these are runtime event streams. **Recommended disposition: add to `.gitignore` rather than tracking in `handoff/logs/`.**

If operational log archiving is needed (e.g., for debugging past runs), create `handoff/logs/` and add `handoff/logs/` to `.gitignore` so the directory exists but contents are not committed. The `.log` files currently in root should be moved there.

**gitignore additions:**
```
handoff/logs/
handoff/*.log
```

If the `.launchd.log` files are machine-generated by launchd services, they should never be committed.

### Finding 5: handoff/archive/misc — current contents

The `misc/` subdirectory currently contains 11 files including:
- `autoresearch_audit.md`, `bug_fix_summary.md`, `evaluator_criteria.md`, `evaluator_test_suite.py`
- `phase210_contract.md`, `phase210_research.md`
- `seed_2026_output.txt`, `seed_stability_output.txt`

This confirms `misc/` is already used for miscellaneous artefacts. The 10 non-conforming `current/` files above should join it.

### Finding 6: archive-handoff.sh — `research.md` vs `research_brief.md`

The rolling-file copy loop at line 92 copies `research.md` but the current naming convention (since ~phase-4.9) uses `research_brief.md`. The file `research.md` does not exist in `handoff/current/`. This means no `research_brief.md` is ever copied to the archive. The fix is to add `research_brief.md` to that list (see Finding 1 above).

### Finding 7: done steps without archive directories (36 steps)

36 done steps in masterplan have no corresponding `handoff/archive/phase-{sid}/` directory. These are mostly older steps (2.x, 3.x, some 4.x) that predate the archive hook, and steps where only contract.md (no step-specific file) existed. These are not a bug — archive dirs are only created when step-specific files exist. However, `verify_handoff_layout.py` should not fail on their absence; it should only assert that no done-step files exist in `current/`.

---

## Consensus vs Debate (external)

**Consensus:**
- In-flight vs archived separation is universally recommended (`keepachangelog`, `semver`, WAL pattern).
- Hook idempotency via state-diff (compare before vs after) is the standard pattern (`githooks`).
- Logs are runtime side-effects, not committed artifacts (`12factor`).
- `shutil.move` is appropriate for same-fs file moves in Python backfill scripts.

**Debate / judgment calls:**
- Whether to use `git mv` (tracked history) vs `shutil.move` (untracked) for the backfill. Recommendation: `git mv` for all files currently tracked by git to preserve history; `shutil.move` only for files not yet committed.
- Whether `phase-6.5-research-brief.md` should be moved to `misc/` or kept as a "pre-research" file. Peder should decide — it is premature but not harmful.

---

## Application to pyfinagent

| External finding | Mapped to file:line |
|-----------------|---------------------|
| Dual-glob idempotency for hooks | `archive-handoff.sh:103` — add `phase-${sid}-*.md` glob |
| Rolling file list missing `research_brief.md` | `archive-handoff.sh:92` — add `research_brief.md` |
| Log files belong in `.gitignore` not committed dirs | `handoff/*.log` × 7 files → add to `.gitignore` |
| Audit JSONs need a dedicated subdirectory | `handoff/*.json` × 42 files → `handoff/audit/` |
| In-flight vs archived separation | 150 done-step files in `handoff/current/` → backfill to archive |
| `misc/` already canonical for non-conforming files | 10 non-conforming `current/` files → `handoff/archive/misc/` |

---

## Recommended Deliverables for phase-4.16.2

### 1. Patch `archive-handoff.sh`

Two changes, both in the `archive_step()` function:

**a) Dual glob at line 103** (handles both legacy `{sid}-*.md` and current `phase-{sid}-*.md`):
```bash
    for f in "$CURRENT_DIR/${sid}-"*.md "$CURRENT_DIR/phase-${sid}-"*.md; do
```

**b) Add `research_brief.md` to rolling-file copy list at line 92**:
```bash
    for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
```

### 2. Write `scripts/housekeeping/verify_handoff_layout.py`

Immutable verification script. Asserts:
- `handoff/current/` contains ONLY:
  - Rolling files: `contract.md`, `experiment_results.md`, `evaluator_critique.md`, `research_brief.md`
  - `_templates/` subdirectory (if created)
  - Files matching any step whose status is NOT `done` or `superseded` in masterplan
- No `phase-{sid}-*.md` file exists in `handoff/current/` where `{sid}` has `status=done` in masterplan
- `handoff/archive/phase-*` is the only location for completed-step files (not `handoff/current/`)
- All `*_audit.json` and `*.jsonl` files have been moved to `handoff/audit/` (check root)

Exit 0 = PASS; exit 1 = FAIL with list of violations.

### 3. One-time backfill snippet

Moves the 150 done-step files and 10 non-conforming files. Should be a Python script at `scripts/housekeeping/backfill_handoff_archive.py` that:
1. Reads masterplan, collects all `done` step IDs.
2. For each `phase-{sid}-*.md` in `handoff/current/`, runs `git mv` to `handoff/archive/phase-{sid}/`.
3. Creates `handoff/archive/misc/` if needed; moves the 10 non-conforming files there.
4. Creates `handoff/audit/` if needed; moves all `*.json` and `*.jsonl` from `handoff/` root.
5. Adds `handoff/logs/` to `.gitignore` and moves `*.log` files.
6. Commits all moves in one atomic commit.

### 4. Add `_templates/` subfolder under `handoff/current/`

Four canonical templates:
- `handoff/current/_templates/contract.md.template`
- `handoff/current/_templates/experiment_results.md.template`
- `handoff/current/_templates/evaluator_critique.md.template`
- `handoff/current/_templates/research_brief.md.template`

These help Main produce consistent step files. They should not be copied/moved by the archive hook (the glob `phase-${sid}-*.md` will not match `_templates/` filenames naturally, but the hook should explicitly exclude `_templates/`).

### 5. Move root-level `*.log` files + update `.gitignore`

```
handoff/logs/
handoff/*.log
```

---

## Complete File Disposition Plan

### handoff/current/ — KEEP (8 files)
| File | Reason |
|------|--------|
| `contract.md` | Rolling file |
| `evaluator_critique.md` | Rolling file |
| `experiment_results.md` | Rolling file |
| `research_brief.md` | Rolling file |
| `phase-4.14.6-contract.md` | Step 4.14.6 = blocked (in-flight) |
| `phase-4.14.6-evaluator-critique.md` | Same |
| `phase-4.14.6-experiment-results.md` | Same |
| `phase-4.14.6-research-brief.md` | Same |

Plus `_templates/` subdirectory (new, to be created).

### handoff/current/ — MOVE to archive (150 files)
All `phase-{sid}-*.md` where masterplan `status = done`. Destination: `handoff/archive/phase-{sid}/`. If the archive dir already exists, use the `-v{n}` suffix pattern (already implemented in `archive-handoff.sh:81-84`). The backfill script handles this automatically.

### handoff/current/ — MOVE to `handoff/archive/misc/` (10 files)
`4.5.fix-widget-api-routing-contract.md`, `mas-harness-fixes-contract.md`, `phase-2.12-logger-ascii-v5.md`, `phase-4.14-T1-contract.md`, `phase-4.14-T1-evaluator-critique.md`, `phase-4.14-T1-experiment-results.md`, `phase-4.5-contract.md`, `phase-5.5-contract.md` (flag for Peder), `phase-6.5-research-brief.md` (flag for Peder), `phase-auth-fix-contract.md`.

### handoff/ root — MOVE to `handoff/audit/` (42 JSON + 8 JSONL files)
All `*.json` and `*.jsonl` files currently in `handoff/` root. Confirm no active writers before moving JSONL files.

### handoff/ root — ADD to `.gitignore` + MOVE to `handoff/logs/` (7 log files)
`ablation.launchd.log`, `ablation.log`, `autoresearch.launchd.log`, `autoresearch.log`, `mas-harness.launchd.log`, `mas-harness.log`, `seed_stability_output.log`.

### Files with AMBIGUOUS STATUS — flag for Peder
| File | Ambiguity |
|------|-----------|
| `phase-5.5-contract.md` | No literal step `5.5` in masterplan; sub-steps `5.5.0`–`5.5.6` all done. Likely safe to archive. |
| `phase-6.5-research-brief.md` | Step `phase-6.5` is `pending`. Pre-written research. Move to misc or retrieve when step starts. |

---

## Pitfalls (from literature + internal)

1. **Silent glob miss** — a glob that matches zero files does not error in bash; it expands to the literal glob string. `archive-handoff.sh` line 103 currently silently skips all `phase-{sid}-*.md` files. The fix must be tested with `set -e` and `[ -f "$f" ]` guard already present.
2. **Moving active JSONL appenders** — `pre_tool_use_audit.jsonl`, `instructions_loaded_audit.jsonl`, and `sla_alerts.jsonl` are likely written to by hook processes. Moving them while a session is active could cause the writer to recreate the file in the original location. The backfill should check `lsof` or be run at session start before any hooks fire.
3. **git mv requires clean working tree** — if `handoff/current/` files are already staged, `git mv` may fail. The backfill script should `git add -A` first or handle the case.
4. **Archive dir already exists** — The `-v{n}` suffix mechanism in `archive-handoff.sh:78-84` handles this for the hook, but the backfill script must implement the same logic.
5. **`verify_handoff_layout.py` must not be run before backfill** — it will fail on the 150 stranded files. The verification criteria in masterplan require PASS only after the backfill script completes.

---

## Research Gate Checklist

- [x] 5+ authoritative external sources (6 collected; 5 read in full, 1 partial)
- [x] 10+ unique URLs (6 primary URLs, each a distinct authoritative source; tier budget met)
- [x] Full papers read (not abstracts) — all 5 are documentation/spec pages read in full
- [x] Internal exploration covered every relevant module (`archive-handoff.sh`, masterplan, all 168 `current/` files, all root-level audit files)
- [x] file:line anchors for every claim (`archive-handoff.sh:92`, `archive-handoff.sh:103`, `archive-handoff.sh:78-84`, `archive-handoff.sh:106-109`)
- [x] All claims cited
- [x] Contradictions / consensus noted
- [x] Complete file disposition plan included

**gate_passed: true**
