# Research Brief -- step 86.105

**Topic:** Best-practice handoff/artifact directory hygiene for long-running agent
harnesses: keeping an append-only audit tree, a rolling `current/` workspace and an
`archive/` partition layout-invariant under concurrent sessions.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).

**Boundaries (caller-set):** Research and plan only. Move NO files. The layout
invariant's fix is 86.105's GENERATE, not this gate. Do not absorb 86.29 (archive
snapshots wrong step) beyond noting the overlap.

**Status:** COMPLETE -- written incrementally (write-first), envelope flipped as
the final act. The identical envelope is repeated at the tail of this file; both
read COMPLETE, so there is no ambiguous marker.

---

## Envelope (born inert -- phase-86.37; flipped COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "The step's audit_basis does not reproduce: the checker returns 667 violations, not 6. 664 are one class caused by STEP_ID_RE matching 0/664 -- a known defect owned by pending step 75.11.4. All three root-level move destinations already exist and are stale, so a bare move clobbers 2,035 bytes of append-only audit history. The backfill script would sweep 666/666 files including this brief.",
  "brief_path": "handoff/current/research_brief_86.105.md",
  "gate_passed": true
}
```

---

## Section 1 -- Internal: the measured failure

### 1.1 HEADLINE -- the step's own audit_basis does not reproduce: 6 vs 667

`.claude/masterplan.json` step 86.105 `audit_basis` states, verbatim:

> MEASURED 2026-08-17: python3 scripts/housekeeping/verify_handoff_layout.py ->
> exit 1 with exactly the six findings named in notes.

**Re-run 2026-08-17 in this session (same command, same repo root):**

```
handoff layout FAIL -- 667 invariant violation(s):
  - current/research_brief_82.4.md has no step-id prefix; move to handoff/archive/misc/
  - current/research_brief_4000.1.md has no step-id prefix; move to handoff/archive/misc/
  ... (664 of this class)
  - handoff/autoresearch.launchd.log is a log; move to handoff/logs/
  - handoff/autoresearch.log is a log; move to handoff/logs/
  - handoff/prompt_leak_redteam_audit.jsonl is audit output; move to handoff/audit/
```
exit 1.

Class breakdown (`grep | sed | sort | uniq -c` over the captured 670-line output):

| Count | Message class |
|---|---|
| 664 | `has no step-id prefix; move to handoff/archive/misc/` |
| 2 | `is a log; move to handoff/logs/` |
| 1 | `is audit output; move to handoff/audit/` |

The **three root-level findings reproduce exactly**. The **"three stray current/
files" do not** -- they are 3 members of a 664-member class, all carrying the
*identical* message. The audit presented a **sample as a census**.

### 1.2 Root cause: `STEP_ID_RE` matches 0 of 664 -- a KNOWN, still-OPEN defect

`scripts/housekeeping/verify_handoff_layout.py:51`:

```python
STEP_ID_RE = re.compile(r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$")
```

The pattern is anchored at `^` to a **leading** step id (`82.4-name.md` /
`phase-82.4-name.md`). Every artifact this harness has written since ~phase-75
uses the **inverse** form `name_<sid>.md` (`research_brief_82.4.md`,
`contract_86.97.md`) -- and `.claude/rules/research-gate.md:224` + the
`research-gate.js` `brief_path` arg both *mandate* that inverse form. Measured
against the live tree:

| Metric | Value |
|---|---|
| files in `handoff/current/` (top level) | 723 |
| exempt via `_is_rolling_keep` (`:45-49`) | 59 |
| candidates reaching `STEP_ID_RE` | 664 |
| **matched by `STEP_ID_RE`** | **0** |
| unmatched but of documented `kind_<sid>.md` form | 570 |

Because the `status == "done"` branch (`:120-125`) is reachable **only** when the
regex matches, the verifier's *primary* documented invariant -- "`current/`
contains NO files belonging to `done` steps" -- **has zero reachable cells**. All
664 files fall into the `if not m` arm at `:111-116` instead. The checker is not
measuring what its docstring (`:3-9`) says it measures.

This is **not a new discovery**. Commit `c3286524` (2026-07-31, `fix(81.2)`),
which is the last commit to touch this file, says verbatim in its message:

> ROOT CAUSE, measured end to end: STEP_ID_RE in both housekeeping scripts matches
> 0 of 127 .md files in handoff/current/ -- it expects the legacy `<sid>-name.md`
> form while every artifact since ~phase-75 uses the inverse `name_<sid>.md`.
> [...] **It does NOT fix the regex -- that belongs to pending step 75.11.4,
> deliberately.**

So the defect is filed, deliberately deferred, and owned by **pending step
75.11.4**. 86.105 cannot make this checker exit 0 without either colliding with
75.11.4 or moving 664 files. Denominator has grown 127 -> 664 in 17 days.

### 1.3 The live blast radius: `backfill_handoff_archive.py` is NOT safe to re-run

`.claude/rules/research-gate.md:326-327` tells the reader:

> Backfill script: `scripts/housekeeping/backfill_handoff_archive.py`
> (idempotent; safe to re-run).

That claim is **false under the current regex**, and the harm is on the record.
The same `c3286524` message continues:

> So backfill_handoff_archive.py:154-158 sets sid=None for every step artifact and
> calls _move(p, MISC). Commit fa9aaf8e (2026-07-24) executed that sweep, took
> handoff/current/evaluator_critique.json with it, and the verdict gate -- which
> resolved exactly one literal path, and fails open on a miss -- ran dark for **13
> consecutive step closes with no signal**.

**Design consequence for 86.105's GENERATE:** the obvious "just run the backfill"
remedy is the exact action that caused a 13-close gate blackout. It would now
sweep 570+ files into `archive/misc/`. Any plan must treat the backfill as
quarantined until 75.11.4 lands.

### 1.4 Corpus is NOT a gitignore artifact (the 86.94 confound, excluded)

Checked because step 86.94's tripwire measured a 89.5%-gitignored corpus and so
measured the laptop rather than the product. Not the case here:

| Bucket | Count |
|---|---|
| `git ls-files handoff/current/` (recursive, incl. `_templates/`) | 904 |
| ignored under `handoff/current/` | 1 |
| untracked-not-ignored under `handoff/current/` | 2 |

The 664-file corpus is ~99.6% **git-tracked**, so the finding is about the
product, not local dirt. `.gitignore:77` does ignore `handoff/*.log`, which is why
the two root logs are untracked -- see 1.5.



### 1.5 The three root-level findings: writers, ownership, and a CLOBBER hazard

| Root file | Tracked? | Writer | Owner |
|---|---|---|---|
| `handoff/autoresearch.log` | UNtracked (`.gitignore:77` `handoff/*.log`) | `scripts/autoresearch/run_nightly.sh:12` `LOG="$REPO/handoff/autoresearch.log"` | **repo** -- editable |
| `handoff/autoresearch.launchd.log` | UNtracked | `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` `StandardOutPath` **and** `StandardErrorPath` | **OPERATOR** -- launchd plist |
| `handoff/prompt_leak_redteam_audit.jsonl` | **tracked** | `scripts/audit/prompt_leak_redteam.py:39` `AUDIT_LOG = REPO / "handoff" / "prompt_leak_redteam_audit.jsonl"`, scheduled at `backend/slack_bot/scheduler.py:282` | **repo** -- editable, but LIVE |

Two consequences the contract must carry:

1. **`autoresearch.launchd.log` is operator-owned.** Its path is set in a launchd
   plist, so the repoint is a **numbered operator ask** under criterion 3, not a
   code edit. It also cannot be repointed by `launchctl kickstart -k` -- only
   `bootout`+`bootstrap` re-reads a plist, and away-ops rail 9 reserves that verb
   for the operator.
2. **Criterion 2's `git mv` applies to exactly ONE of the three.** The two `.log`
   files are untracked (gitignored), so `git mv` will refuse them; only
   `prompt_leak_redteam_audit.jsonl` is tracked. Criterion 2 is worded "every
   misplaced TRACKED file", so this is consistent -- but a plan that reaches for
   `git mv` on all three will fail.

### 1.6 CLOBBER HAZARD -- all three destinations ALREADY EXIST and are stale

Measured 2026-08-17 (`stat -f%z` / `stat -f%Sm`; note `stat -f%Sm` prints LOCAL
time, not Z):

| Destination | Exists? | Size | mtime | Source size / mtime |
|---|---|---|---|---|
| `handoff/logs/autoresearch.log` | **YES** | 4,780 B | Apr 19 2026 | 265,305 B / Aug 17 2026 |
| `handoff/logs/autoresearch.launchd.log` | **YES** | 0 B | May 7 2026 | 0 B / Jul 25 2026 |
| `handoff/audit/prompt_leak_redteam_audit.jsonl` | **YES** | 2,035 B | **Jun 11 2026** | 48,840 B / **Aug 17 09:15** |

**A bare `mv`/`git mv` overwrites a pre-existing file at every one of the three
destinations.** The audit-stream case is the dangerous one: `handoff/audit/
prompt_leak_redteam_audit.jsonl` holds 2,035 bytes of append-only history that
the root copy does not contain. Criterion 2 demands the moved stream stay
"byte-identical across the move" -- satisfiable for the *source* bytes, but a
naive move **destroys the destination's 2,035 bytes**, which is a silent
append-only-history deletion and would breach the spirit of criterion 2 while
passing its letter.

This is a **split-brain append-only stream**: one logical stream living at two
paths with divergent content. The repo already has a codified precedent for the
correct handling -- `verify_handoff_layout.py:69-85` (phase-36.8) protects the
kill-switch audit archives from pruning precisely because `_load_from_audit`
**merges** them, and "the TRUE peak lives in the OLDEST file". Merge, don't
overwrite.

### 1.7 Why 664 accumulated: the archive hook COPIES, and its baseline is seeded

`.claude/hooks/archive-handoff.sh:2-3` describes itself as
"Copy/move handoff/current/* into handoff/archive/phase-<id>/". Archiving is
working in the *destination* direction -- 835 `handoff/archive/phase-*` dirs vs
843 `done` steps in the masterplan -- yet `handoff/current/` still holds 723
files. So the archive is a **copy**, and `current/` never drains. That is the
accumulation mechanism, and it is why the 664-member class exists at all.

`.claude/hooks/archive-handoff.sh:42-52` additionally **seeds** its baseline
(`.claude/.archive-baseline.json`, a single `seen_done` key) with every
currently-done id on first run "so we never retro-archive 100+" -- a deliberate
one-way ratchet. Nothing in the hook removes a file from `current/`.

### 1.8 Configuration drift: the ablation sibling already does it right

`scripts/ops/run_ablation.sh:14` -- `LOG="$REPO/handoff/logs/ablation.log"`.
The ablation nightly already writes into `handoff/logs/`; the autoresearch
nightly does not. `handoff/logs/` already contains `ablation.log`,
`ablation-v2/-v3/-v4.log`, `autoresearch-v2.log`, `autoresearch-v3.log`. So
86.105's repoint has a **proven in-repo template**: make
`run_nightly.sh:12` mirror `run_ablation.sh:14`. This is a one-line change on
the repo-owned writer.

### 1.9 MEASURED: the documented "safe to re-run" backfill would destroy this brief

`scripts/housekeeping/backfill_handoff_archive.py:64` carries a **byte-identical**
copy of the dead regex, and `:154-157` routes every non-match to `_move(p, MISC)`.
Measured today, no writes performed (regex applied in a read-only replica):

```
backfill regex literal: r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$"
.md files in current/: 666; unmatched by regex -> swept to archive/misc/: 666
  experiment_results_86.29.md        exists=True  regex_match=False
  contract_86.1.md                   exists=True  regex_match=False
  research_brief_80.5.md             exists=True  regex_match=False
  contract_86.97.md                  exists=True  regex_match=False
  research_brief_86.105.md           exists=True  regex_match=False
```

**666 of 666 -- a 100% sweep.** The list includes `contract_86.97.md` (live,
shows as `M` in this session's `git status`) and **this brief itself**. The
"idempotent; safe to re-run" line at `.claude/rules/research-gate.md:326-327` is
false and should be corrected by whichever step owns the regex (75.11.4).

### 1.10 Concurrency: step 86.43 is the open, filed exposure

Step **86.43** (`status: pending`) is the concurrency half of this objective and
must not be collided with. Verbatim from its name field:

> Two Claude sessions run concurrently on this repo as a matter of routine. [...]
> the researcher's brief path is STEP-SCOPED [...] the new researcher's FIRST tool
> call truncates the owner's brief. MEASURED: research_brief_86.21.md went 24,904
> bytes -> 1,164 bytes [...] It was recoverable ONLY because it had been committed.

It also names the not-yet-established scope: "whether any OTHER step-scoped
artifact has the same exposure -- `contract_<sid>.md`,
`experiment_results_<sid>.md`, `live_check_<sid>.md` are all written by one role
at a shared path". **Those are exactly the files 86.105 would be moving.** And
86.43 carries an explicit prohibition: "DO NOT 'FIX' THIS BY WEAKENING
WRITE-FIRST".

The only lock in the hook tree is `auto-commit-and-push.sh:299-301`
(`.git/pyfinagent-auto-commit.lock.d`, 20s wait, 120s stale, **fail-open** by
design). Nothing guards `handoff/current/` writes. So a bulk move of
`handoff/current/` is unsynchronised against a peer session.

## Section 2 -- External literature

### 2.1 Read in full (7; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2605.14271 | 2026-08-17 | paper (arXiv, *Auditing Agent Harness Safety*) | WebFetch, arXiv native HTML | "the action sink resequences framework-native events into a unified chronological trace written as _append-only_ JSONL records (rather than mutable state)"; "Artifacts are stored under harness/model/**run-scoped** directories"; "This layout retains the observable evidence required for offline rejudging, so the original agent does not need to be rerun." |
| 2 | https://arxiv.org/html/2606.14249 | 2026-08-17 | paper (arXiv, *HarnessX*) | WebFetch, arXiv native HTML | **Near-miss, reported honestly**: has a "13.1 Per-Run Directory Layout" appendix heading but no substantive body text on layout, concurrency or tree invariants. No usable finding. |
| 3 | https://martinfowler.com/bliki/ParallelChange.html | 2026-08-17 | authoritative blog (year-less canonical) | WebFetch | expand -> migrate -> contract. "Making a change to an interface that impacts all its consumers requires two thinking modes"; **"If the contract phase is not executed you might end up in a worse state than you started, therefore you need discipline to finish the transition successfully."** |
| 4 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-17 | official doc (project's canonical ref) | WebFetch | Confirms only the file-handoff mechanism: "Communication was handled via files: one agent would write a file, another agent would read it and respond..." **Contains NO guidance on directory structure, naming, append-only logs, stale-artifact management, or tree invariants.** |
| 5 | https://agentpatterns.ai/patterns/agent-design/long-running-agents/ | 2026-08-17 | authoritative blog | WebFetch | "Session state lives outside the harness process, as an append-only log of every thought, tool call, and observation"; "Write intermediate state every few units of work -- not every step, which wastes effort, and not only at the end, which is catastrophic on failure"; "Keep it on disk so the agent cannot quietly redefine 'done' mid-run." Explicitly: **"No guidance on cleanup strategies or partition schemes appears."** |
| 6 | https://fast.io/resources/agentic-workflow-storage/ | 2026-08-17 | industry practitioner | WebFetch | "Create folders for different stages of your workflow (inputs, processing, outputs)"; "the agent creates new versions rather than overwriting [...] **A manifest file tracks which version is current**"; "Agents acquire a lock before writing and release it when done." Also states **no** retention/cleanup guidance. |
| 7 | https://nikiforovall.blog/ai/2026/06/08/scratch.html | 2026-08-17 | authoritative blog (2026) | WebFetch | Scratchpad = "a folder containing a `scratchpad.json` manifest. **The folder path _is_ its identity**"; "one concern per file"; disposable `scratch-<label>.md` vs `artifact`-typed permanent deliverables; "the CLI never authors, copies, or moves your content." |

### 2.2 Identified but snippet-only (18; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://addyosmani.com/blog/long-running-agents/ | blog | duplicate of #5's thesis |
| https://addyo.substack.com/p/long-running-agents | blog | mirror of the above |
| https://github.com/NousResearch/hermes-agent/issues/487 | community | hash-chained audit log; out of scope (no crypto requirement here) |
| https://disarray.ai/featured/design-principles-for-long-running-research-agents | blog | lower authority, overlaps #5 |
| https://zylos.ai/research/2026-04-25-agent-identity-provenance-signed-audit-trails/ | industry | signed provenance; out of scope |
| https://github.com/giacomo/agents-lint | community tool | stale-path linting; adjacent, noted in 3.1 |
| https://dev.to/jackm-singularity/ai-agent-workspace-architecture-give-agents-files-tools-and-limits-1g87 | community | superseded by #6 |
| https://dev.to/jackm-singularity/ai-agent-scratchpad-keep-coding-agents-fast-without-polluting-git-329c | community | superseded by #7 |
| https://www.mindstudio.ai/blog/context-rot-ai-coding-agents-explained | community | context rot, not tree layout |
| https://docs.nvidia.com/nemoclaw/0.0.16/workspace/workspace-files.html | vendor doc | vendor-specific workspace model |
| https://mcgarrah.org/ai-agent-context-files-in-practice/ | blog | AGENTS.md focus |
| https://windowsforum.com/threads/agents-md-in-2026-turning-agent-prompts-into-reviewable-repo-policy.430224/ | forum | lowest tier |
| https://fast.io/resources/ai-agent-shared-workspace/ | industry | same vendor as #6 |
| https://fast.io/resources/openclaw-multi-agent-workspaces/ | industry | same vendor as #6 |
| https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution | industry | worktree isolation; noted in 4.4 |
| https://alexlavaee.me/blog/parallel-agent-sessions-infrastructure-gap/ | blog | parallel-session infra gap |
| https://github.com/max-sixty/worktrunk/issues/3552 | community | real TOCTOU rename-under-concurrency bug; corroborates 4.4 |
| https://dev.to/aws/lambda-just-got-a-file-system-i-put-ai-agents-on-it-1ej8/comments | community | cloud FS, not applicable (local-only deployment) |

**URLs collected: 25** (7 read in full + 18 snippet-only).

### 2.3 Search queries run (three-variant discipline)

1. **Year-less canonical** -- `append-only audit log directory layout invariant long-running agent harness artifacts`
2. **Current-year frontier (2026)** -- `agent workspace file hygiene stale artifacts context rot 2026 coding agent scratch directory`
3. **Last-2-year window (2025)** -- `2025 concurrent agent sessions shared filesystem workspace lock atomic rename append-only journal`

## Section 3 -- Recency scan (2024-2026)

Searched the 2024-2026 window (queries 2 and 3 above). **Result: 3 new findings
that complement -- none that supersede -- the canonical sources.**

1. **Manifest-over-inference (2026).** Both #6 and #7 converge on a *manifest*
   as the authority for "what is current", rather than inferring it from
   directory state. #7: "a folder containing a `scratchpad.json` manifest. The
   folder path _is_ its identity." This is new relative to the canonical
   file-handoff literature and directly relevant: pyfinagent currently infers
   step membership from a **filename regex**, which is the failure in 1.2.
2. **Disposable-vs-artifact typing (2026, #7).** Scratch entries are typed at
   creation: `scratch-<label>.md` is "Disposable -- can be deleted after use",
   `artifact` is permanent. pyfinagent has no such type marker; every file in
   `handoff/current/` is indistinguishable to the verifier.
3. **Stale-context linting is now a recognised category (2026).** `agents-lint`
   (snippet-only) exists to "detect stale paths [...] before they make your
   coding agents expensive and wrong", and ETH Zurich / ICSE 2026 work reported
   in the same result set found stale LLM-generated context files "reduced task
   success by 2-3% while increasing cost by over 20%". This corroborates that a
   667-violation checker nobody can act on is a real cost, not just untidiness.

**No 2024-2026 source contradicts the canonical guidance**, and nothing found
supersedes Fowler's ParallelChange for the move-with-live-writer problem.

## Section 4 -- Consensus, debate, and application to pyfinagent

### 4.1 Consensus across sources

- **Append-only streams are immutable and must never be treated as mutable
  state** (#1 "rather than mutable state"; #5 "an append-only log"). -> Directly
  governs `handoff/prompt_leak_redteam_audit.jsonl` (1.6): the destination's
  2,035 bytes are history, and overwriting them is a deletion.
- **Partition by stage/scope, with one writer per path** (#1 run-scoped dirs;
  #6 "folders for different stages"). -> pyfinagent's four-way partition
  (`current/`, `archive/`, `audit/`, `logs/`) already matches best practice.
  **The partition is not the problem; its enforcement is.**
- **Authority should be an explicit manifest, not inferred state** (#6, #7).

### 4.2 Debate / gap

There is a genuine **gap in the literature**: #4 (the project's own canonical
reference) and #5 both explicitly contain *no* guidance on artifact cleanup or
partition schemes, and #6 also declines retention guidance. So pyfinagent's
`handoff/` invariant is **home-grown and un-anchored** -- no external authority
prescribes it. That argues for keeping the invariant *simple and enforceable*
rather than elaborating it.

### 4.3 The load-bearing external finding: ParallelChange explains 1.6 exactly

Every one of the three root files already has a populated destination (1.6).
That is the signature of an **expand** phase that was executed and a
**migrate**/**contract** that never were: the new locations were created, the
writers were never repointed, and both copies then diverged for months (Apr 19 /
Jun 11 destinations vs Aug 17 sources). Fowler's warning is the precise
description of the current state:

> "If the contract phase is not executed you might end up in a worse state than
> you started."

**86.105's criterion 3 ("the WRITER of each root-level log is found and
repointed, or shown already dead -- a move without repointing regresses on the
writer's next fire") is therefore the *migrate* phase, and it is correctly
specified.** The step should be planned as expand -> migrate -> contract, and the
contract phase (deleting the root path) must come *after* the writer repoint, not
before.

### 4.4 Concurrency

#6 prescribes locking ("Agents acquire a lock before writing"); the snippet-only
worktrunk issue #3552 shows a real TOCTOU rename-under-concurrency defect in a
comparable tool. pyfinagent has **no lock on `handoff/current/`** (1.10) and a
filed, still-open collision record (86.43). Combined with `git status` showing
live modified artifacts in `handoff/current/` right now, this makes a bulk move
of 664 files during a concurrent session a genuine data-loss risk, not a
theoretical one.

### 4.5 Internal code inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `scripts/housekeeping/verify_handoff_layout.py` | `:51` regex, `:111-116` fail arm, `:120-125` dead arm | the checker | **RED, 667 findings; primary invariant unreachable** |
| `scripts/housekeeping/backfill_handoff_archive.py` | `:64` regex, `:154-157` MISC sweep | the "fix" script | **QUARANTINE -- would sweep 666/666** |
| `.claude/hooks/archive-handoff.sh` | `:2-3` copy semantics, `:42-52` baseline seed | archiver | works, but **copies**; `current/` never drains |
| `.claude/rules/research-gate.md` | `:316-327` | doc of the invariant | **"safe to re-run" claim is false** |
| `scripts/autoresearch/run_nightly.sh` | `:12` `LOG="$REPO/handoff/autoresearch.log"` | writer #1 | repo-owned; **one-line repoint** |
| `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` | `StandardOutPath` + `StandardErrorPath` | writer #2 | **OPERATOR-OWNED -> numbered ask** |
| `scripts/audit/prompt_leak_redteam.py` | `:39` `AUDIT_LOG` | writer #3 | repo-owned; live (mtime today 09:15) |
| `backend/slack_bot/scheduler.py` | `:282` | schedules writer #3 | consumer/scheduler |
| `scripts/ops/run_ablation.sh` | `:14` `LOG="$REPO/handoff/logs/ablation.log"` | **correct sibling** | template for the repoint |
| `.gitignore` | `:75-77` | `handoff/*.log`, `handoff/logs/` | both logs untracked -> `git mv` N/A |
| `.claude/hooks/auto-commit-and-push.sh` | `:299-301` | only lock in the tree | fail-open; does not guard `handoff/` |
| `.claude/.archive-baseline.json` | `seen_done` (1 key) | archive ratchet | seeded; never retro-archives |
| `.claude/masterplan.json` | step `86.105`, step `86.43`, step `75.11.4` | plan | see 4.6 |

### 4.6 Criteria feasibility -- FLAG FOR MAIN BEFORE THE CONTRACT IS FROZEN

Criteria are immutable once written, so these must be resolved *now*:

1. **Criterion 1 is not satisfiable as worded.** It requires the checker to
   `exit 0` after the fix **and** the before-run to be quoted as "the 2026-08-17
   red run (exit 1, **six** findings)". The 2026-08-17 red run has **667**
   findings. Quoting a six-finding run is impossible because no such run exists
   at this tree state. And reaching `exit 0` requires clearing all 664
   regex-class findings -- which means fixing the regex, and **the regex is
   explicitly assigned to pending step 75.11.4** by commit `c3286524`.
2. **Scope collision.** 86.105 as scoped (three root files) cannot turn the
   checker green. Either 86.105 absorbs 75.11.4's regex fix (and says so), or its
   criterion 1 must be scoped to the three root findings only.
3. **86.43 overlap.** Any mass move of `handoff/current/` intersects the open,
   unestablished exposure in 86.43. Recommend 86.105 does **not** touch
   `handoff/current/` at all.
4. The step's `notes` also say "Step 76.9.5 touches the autoresearch side
   elsewhere -- coordinate, do not duplicate."

### 4.7 Recommended shape for GENERATE (Main owns PLAN; this is input, not a plan)

- Scope to the **three root-level findings only**; leave the 664-class to
  75.11.4 and say so explicitly in the contract.
- Order each move as expand -> **migrate (repoint the writer first)** -> contract.
- For `prompt_leak_redteam_audit.jsonl`: **merge, never overwrite** -- the
  destination holds 2,035 bytes the source lacks; precedent is the phase-36.8
  kill-switch archive-merge rule at `verify_handoff_layout.py:69-85`.
- `git mv` applies to **one** file (the `.jsonl`); the two logs are gitignored.
- The launchd `StandardOutPath`/`StandardErrorPath` repoint is a **numbered
  operator ask** (requires `bootout`+`bootstrap`, reserved for the operator).
- Do **not** run `backfill_handoff_archive.py` (1.9: 666/666 sweep, would take
  this brief and `contract_86.97.md`).
- Correct the false "idempotent; safe to re-run" line at
  `.claude/rules/research-gate.md:326-327` -- but note criterion 5 forbids
  touching `.claude/hooks`, and this file is not a hook, so it is permitted.

## Section 5 -- Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total -- **25** (7 full + 18 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- Section 3, 3 findings
- [x] Full pages read (not abstracts) for the read-in-full set -- arXiv native
      HTML used for both papers; no `arxiv.org/pdf/` fetch attempted
- [x] file:line anchors for every internal claim -- Section 4.5 + inline

Soft checks:
- [x] Internal exploration covered every named module in INTERNAL SCOPE
- [x] Contradictions / consensus noted (4.1, 4.2; #2 honest near-miss)
- [x] Claims cited per-claim
- **Gap disclosed:** source #2 (HarnessX) was fetched in full but yielded no
  usable content; the gate still clears on the other 6.
- **Boundary honoured:** no files moved, no production code edited, no contract
  written. 86.29 noted only as overlap, not absorbed.

---

## Final envelope

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "The step's audit_basis does not reproduce: the checker returns 667 violations, not 6. 664 are one class caused by STEP_ID_RE matching 0/664 -- a known defect owned by pending step 75.11.4. All three root-level move destinations already exist and are stale, so a bare move clobbers 2,035 bytes of append-only audit history. The backfill script would sweep 666/666 files including this brief.",
  "brief_path": "handoff/current/research_brief_86.105.md",
  "gate_passed": true
}
```
