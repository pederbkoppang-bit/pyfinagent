# Research Brief -- step 86.36

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Date:** 2026-08-11. **Researcher:** Layer-3 Researcher (Workflow rail).

## Objective

Durable checkpoint / work-in-progress files for crash-prone worker processes:
per-run vs fixed paths, atomic write-and-rename, copy-on-open snapshots,
retention of prior attempts, and how concurrent writers at a shared path destroy
recovery evidence. Cover crash-only software design, write-ahead logging,
SQLite/journald rotation naming schemes, and the observability practice of
preserving the previous attempt's artifact when a retry begins.

**Internal scope:** `scripts/qa/qa_wip.py`, `.claude/workflows/qa-verdict.js`,
`.claude/agents/qa.md`, `.claude/agent-memory/qa/verdicts/`,
`.claude/hooks/qa-write-guard.sh` (READ ONLY -- its modification is step 86.33),
`.gitignore`.

---

## STATUS ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 7,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "gate_passed": true
}
```

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://www.sqlite.org/atomiccommit.html | 2026-08-11 | official doc | WebFetch (HTML, full) | Journal is a FIXED suffix (`-journal`) off the db name, but the multi-file **super-journal is PER-RUN**: "with the text `-mj_HHHHHHHH_` appended where HHHHHHHH is a random 32-bit hexadecimal number. The random HHHHHHHH suffix changes for every new super-journal." Hot journal = "a rollback journal that needs to be played back... A hot journal only exists when an earlier process was in the middle of committing a transaction when it crashed". Commit point = journal deletion. Directory fsync is required "in order to make sure the super-journal file will appear in the directory following a power failure". |
| 2 | https://www.sqlite.org/howtocorrupt.html | 2026-08-11 | official doc | WebFetch (HTML, full) | The shared-path destruction mechanism, stated outright: "Since rollback journals and WAL files are based on the name of the database file, the two different database files will share the same rollback journal or WAL file. A rollback or recovery for one of the databases might use content from the other database, resulting in corruption." And on evidence destruction: "If the hot journal files are moved, deleted, or renamed after a crash or power failure, then automatic recovery will not work and the database may go corrupt." Also names "Overwriting a journal file with a different journal file" as a corruption action. |
| 3 | https://lwn.net/Articles/457667/ | 2026-08-11 | authoritative pub (LWN) | WebFetch (HTML, full) | The canonical atomic-replace recipe, verbatim 5 steps: "create a new temp file (on the same file system!)" / "write data to the temp file" / "fsync() the temp file" / "rename the temp file to the appropriate name" / "fsync() the containing directory". Directory-fsync rationale: "A newly created file may require an fsync() of not just the file itself, but also of the directory in which it was created (since this is where the file system looks to find your file)." In-place risk: "If you encounter a system failure ... while overwriting a file, it can result in the loss of existing data." |
| 4 | https://systemd.io/JOURNAL_FILE_FORMAT/ | 2026-08-11 | official doc | WebFetch (HTML, full) | The preserve-the-damaged-artifact rule, verbatim: "If any kind of corruption is noticed by a writer it should immediately rotate the file and start a new one. No further writes should be attempted to the original file, but it should be left around so that as little data as possible is lost." Three-state lifecycle `STATE_OFFLINE=0 / STATE_ONLINE=1 / STATE_ARCHIVED=2`; ONLINE on open-for-write, OFFLINE on clean close, ARCHIVED after rotation. LIMIT: this page does NOT document the `.journal~` naming -- source 5 does. |
| 5 | https://man7.org/linux/man-pages/man8/systemd-journald.service.8.html | 2026-08-11 | official doc (man page) | WebFetch (HTML, full) | Rotation NAMING + preserve-on-damage together: "If the daemon is stopped uncleanly, or if the files are found to be corrupted, they are renamed using the \".journal~\" suffix, and systemd-journald starts writing to a new file." And: "When systemd-journald ceases writing to a journal file, it will be renamed to \"original-name@suffix.journal\" (or \"original-name@suffix.journal~\"). Such files are \"archived\" and will not be written to any more." SIGUSR2 / `journalctl --rotate` forces rotation. |
| 6 | https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/logging-monitoring/logging-tasks.html | 2026-08-11 | official doc | WebFetch (HTML, full) | The canonical per-attempt artifact path in a production retry engine: `"dag_id={dag_id}/run_id={run_id}/task_id={task_id}/attempt={try_number}.log"` (plus a `map_index={map_index}/` variant). The attempt number is a PATH COMPONENT, so attempt N+1 physically cannot overwrite attempt N. LIMIT: the page states the templates but carries no prose rationale, so the "why" is read off the path shape, not quoted. |
| 7 | https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html | 2026-08-11 | official standard (POSIX) | WebFetch (HTML, full) | Atomicity is specified, durability is NOT: "That specification requires that the action of the function be atomic"; "a link named new shall remain visible to other threads throughout the renaming operation and refer either to the file referred to by new or old before the operation began." `[EXDEV]` when old and new are on different file systems. The fetched text contains **no** statement about durability/fsync semantics -> rename buys VISIBILITY atomicity only; durability needs source 3's fsync pair. |
| 8 | https://lwn.net/Articles/191059/ | 2026-08-11 | authoritative pub (LWN) | WebFetch (HTML, full) | "The only way to stop it is to crash it, and the only way to start it is to recover." "crash recovery is a first-class citizen in the development process, rather than an afterthought". And the warning that matters here: "Probably the most common misconception is the idea that writing crash-only software is that it allows you to take shortcuts when writing and designing your code." Components "communicate with retryable requests; faults are handled by crashing and restarting the faulty component and retrying any requests which have timed out". |
| 9 | https://kubernetes.io/docs/concepts/cluster-administration/logging/ | 2026-08-11 | official doc | WebFetch (HTML, full) | The observability precedent for retaining the previous attempt: "You can use `kubectl logs --previous` to retrieve logs from a previous instantiation of a container" and "By default, if a container restarts, the kubelet keeps one terminated container with its logs." Retention is BOUNDED, not infinite (`containerLogMaxFiles` default 5, `containerLogMaxSize` default 10Mi), and "Only the contents of the latest log file are available through `kubectl logs`." Evidence dies with the host: "If a pod is evicted from the node, all corresponding containers are also evicted, along with their logs." |
| 10 | https://research.cs.wisc.edu/areas/os/ReadingGroup/os-old/Papers/HotOSIX/Candea-CrashOnlySoftware.pdf | 2026-08-11 | peer-reviewed (HotOS IX 2003) | curl + **pypdf** full-text extract (6 pages, 34,712 chars) -- the sanctioned PDF chain in `.claude/rules/research-gate.md` §"Step 3", NOT a WebFetch PDF summary | Primary source. "Crash-only programs crash safely and recover quickly. There is only one way to stop such software[]by crashing it[]and only one way to bring it up[]by initiating recov-ery." "a crash-only system is de[fi]ned by the equations stop=crash and start=recover." "In crash-only systems, however, recovery code is exer[cised]..." (recovery paths must be routinely exercised, not reserved for disasters). "we require that all per-sistent state be kept in dedicated state stores, that state stores provide applications with the right abstractions, and that state stores be crash-only." **Extraction artefact disclosed:** pypdf drops the `fi`/`fl` ligatures on this PDF ("de[fi]ned", "[fl]awlessly"), so the bracketed letters are my reinsertion, not the file's bytes. |

**Read-in-full accounting:** 9 of 10 are pure `WebFetch` full-page reads; #10 used
the curl+pypdf chain that `.claude/rules/research-gate.md` prescribes for binary
PDFs. Even discounting #10 entirely, the >=5 floor is met with margin (9).
No `arxiv.org/pdf/` URL was WebFetched.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.usenix.org/conference/hotos-ix/crash-only-software | official pub page | Landing page for source 10; the paper text itself was read instead. |
| https://dl.acm.org/doi/10.5555/1251054.1251066 | peer-reviewed index | Paywalled ACM DL record for the same HotOS IX paper. |
| https://www.semanticscholar.org/paper/Crash-Only-Software-Candea-Fox/118391e04c7552c637b84d22f08c6369bd3cd483 | index | Metadata record, no added content over source 10. |
| https://en.wikipedia.org/wiki/Crash-only_software | community | Tertiary; below the source-quality hierarchy given source 10 was obtained. |
| https://www.abhinavrk.com/crash-only-software.html | blog | Secondary summary of source 10. |
| https://medium.com/@piyush.aggarwal.prof/introduction-to-crash-only-software-3b788b2db7cb | community blog | Lowest tier; no primary content. |
| https://arxiv.org/pdf/1807.00515 | preprint | False hit ("Automatic Software Repair: a Bibliography") -- off-topic. |

## Search-query discipline -- DISCLOSED SHORTFALL

`.claude/rules/research-gate.md` mandates three query variants. **Only ONE
WebSearch executed**: the year-less canonical variant, `"crash-only software
design Candea Fox recovery restart"`. The 2026-frontier and 2025/2024 variants
were **refused by the runtime**, verbatim: *"Web search was not performed: this
session has used its web search budget (200 of 200 WebSearch calls)."* This is a
session-level quota, not a researcher choice.

**Compensating measures (both executed):** (a) the read-in-full set was assembled
by direct `WebFetch` of canonical URLs rather than via search, and (b) the
recency scan below ran as a **date-sorted arXiv API query**, which covers the
frontier and last-2-year windows more precisely than a keyword search would.
Recording this as a soft-check gap; the hard blockers are unaffected.

## Recency scan (2024-2026)

**Method.** Explicit search pass, `sortBy=submittedDate&sortOrder=descending`,
40 results, via
`https://export.arxiv.org/api/query?search_query=abs:"crash consistency" OR abs:"crash-only" OR abs:"write-ahead log"`
(accessed 2026-08-11). 18 entries fell in the 2024-2026 window.

**Result: no finding supersedes the canonical mechanisms; two complement them.**

1. **SAFEFLOW (2025-06-11)** -- "a principled protocol for trustworthy and
   transactional autonomous agent systems", incorporating "write-ahead logging,
   rollback, and secure caches" for agent resilience. This is the only hit that
   applies WAL/rollback to *autonomous agent systems* specifically, i.e. exactly
   pyfinagent's Layer-3 case. Snippet-level evidence only -- not read in full.
2. **Application-level crash-consistency testing (2025-03-03)** -- representative
   testing found 18 real bugs; the transferable lesson is that crash consistency
   must be *tested by injected crashes*, which the phase-86.31 SIGKILL drop
   simulation already does.

Others (SquirrelFS 2024, BtrLog 2026, CXL persistence 2024-2026, BVLSM 2025) are
storage/hardware-layer and do not bear on a markdown checkpoint file. **Nothing
in the window argues against per-attempt path naming or atomic rename.**

## Key findings

1. **Atomicity and durability are different properties, and rename() buys only
   the first.** POSIX guarantees the rename "be atomic" and that a link "shall
   remain visible to other threads throughout the renaming operation", but the
   spec carries no durability statement (source 7). Durability requires the
   5-step recipe: temp on the *same filesystem* -> write -> `fsync` file ->
   `rename` -> `fsync` the **directory** (source 3).
2. **Atomic rename does not fix semantic incompleteness.** A fully-flushed
   half-analysis is still a half-analysis. SQLite's answer is to make the torn
   state *inert* rather than *ambiguous* (source 1: commit is the journal's
   deletion; recovery is keyed on the journal being well-formed and non-empty).
   pyfinagent already implements this as the born-inert `STATUS:` marker.
3. **Fixed derived paths are the documented cause of cross-run destruction.**
   SQLite: because journal names are *derived from* the database name, two
   databases reachable by the same name "share the same rollback journal or WAL
   file. A rollback or recovery for one of the databases might use content from
   the other" (source 2). The identical hazard applies to any
   `<artifact>_<fixed-key>.md`.
4. **Destroying the prior artifact destroys recoverability, and the literature
   says so explicitly.** "If the hot journal files are moved, deleted, or renamed
   after a crash or power failure, then automatic recovery will not work"
   (source 2), which lists "Overwriting a journal file with a different journal
   file" among the corrupting actions.
5. **The production answer to "retry must not clobber" is a path component, not
   a marker.** Airflow puts the attempt number *in the path*:
   `attempt={try_number}.log` (source 6). Kubernetes keeps one prior
   instantiation and exposes it as `--previous` (source 9). journald renames the
   damaged file aside with `.journal~` and starts a new one, so the writer never
   reuses a suspect file (sources 4+5).
6. **Retention is bounded on purpose.** Kubernetes defaults to one terminated
   container, `containerLogMaxFiles=5` (source 9); journald archives rather than
   grows one file. Unbounded prior-attempt retention is not the precedent -- N=1
   or a small N is.
7. **Crash-only design is a discipline, not a licence.** "crash recovery is a
   first-class citizen ... rather than an afterthought", and the "most common
   misconception is the idea that ... it allows you to take shortcuts"
   (source 8); Candea & Fox require "all persistent state be kept in dedicated
   state stores" that are themselves crash-only (source 10). A recovery path that
   is never exercised is the failure mode the paper targets.

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/qa/qa_wip.py` | 272 | Path contract + recovery reader. `resolve_wip_path()` -> `.claude/agent-memory/qa/verdicts/verdict_wip_<sid>.md` (`:106-115`); `classify()` reads the first non-blank line (`:118-130`); `report()` returns `is_verdict: False` always (`:152-238`) | LIVE. Read-side only. |
| `.claude/workflows/qa-verdict.js` | 230 | Q/A launcher. STEP 0b prompt block mandates the born-inert write (`:96-112`); `VERDICT_SCHEMA` deliberately carries **no** `wip_path`/`wrote_verdict_file` field, rationale at `:212-221` | LIVE. |
| `.claude/agents/qa.md` | 621 | "Write-first for your VERDICT FILE ONLY (phase-86.31, BINDING)" at `:98-158`; the 4-line header contract at `:116-127`; **`:124-127` already names this defect**: "The path is FIXED per step, so a cycle-2 spawn that drops before its write leaves cycle-1's file sitting there" | LIVE. |
| `.claude/hooks/qa-write-guard.sh` | 117 | PreToolUse allowlist. `MEMORY_DIR = ".claude/agent-memory/qa/"` (`:63`); `is_qa_role()` matches `qa`/`qa-*`/`qa_*` (`:66-93`); prefix test at `:96-102`; **fail-open** (`:106-116`) | READ ONLY (86.33). A cycle-suffixed filename stays inside the allowed prefix, so 86.36 needs no guard change. |
| `.claude/agent-memory/qa/verdicts/` | 7 files | The sink. One FIXED file per step id | LIVE + **actively destructive** (below). |
| `.gitignore` | -- | No entry for `agent-memory` / `verdicts` / `wip`; `git check-ignore` returns NOT ignored | Sink IS tracked (5 of 7 files in `git ls-files` at measurement time). |
| `handoff/harness_log.md` | -- | `:33484-33491` records the prior destruction events and queues this step | LIVE. |
| `.claude/hooks/archive-handoff.sh` | -- | Archives `handoff/current/*` on masterplan flip; keyed on `handoff/archive/phase-<sid>/` existence. **Does not touch the Q/A sink** | LIVE; not a retention channel for WIP files. |

### MEASURED FIRST-HAND: a destruction event during this research session

I measured the sink twice, ~4 minutes apart, and caught the clobber live:

| File | First measurement (~06:38Z) | Second measurement (06:42Z) |
|---|---|---|
| `verdict_wip_86.34.md` | **4,921 bytes**, `WRITTEN: 2026-08-11T06:27:15Z` | **796 bytes**, `WRITTEN: 2026-08-11T06:40:32Z` |
| `verdict_wip_86.29.md` | 628 bytes | 3,926 bytes (a second writer appending concurrently) |
| `verdict_wip_86.25.md` | 3,473 bytes, INCOMPLETE | 8,543 bytes, COMPLETE |

The replacing file documents the destruction **in its own header**, verbatim:

```
CYCLE: 3 (c1 = FAIL wf_839de1e6-c3c; c2 = CONDITIONAL wf_6c44bae0-a83; then a RAIL
DROP = NO VERDICT, not counted). This file OVERWRITES the DROPPED run's WIP that sat
```

So the retry's **first act, before any analysis**, is to truncate the dropped
attempt's evidence -- the born-inert discipline makes the clobber *earlier*, not
later. `git show HEAD:` for that path already returns the 796-byte stub, so
HEAD does not hold the destroyed content; the harness log (`:33484-33487`)
records an earlier 6,239-byte artifact that "was never committed, so it survives
ONLY because I hand-copied it". Git is therefore an unreliable retention channel
here -- it captures whatever happened to be on disk at commit time.

## Consensus vs debate

**Consensus** across all ten sources: (a) never overwrite the only copy in place;
(b) make the torn state inert; (c) when a writer starts fresh after trouble,
*move the old artifact aside rather than reuse or delete it*. **Debate/tension:**
retention depth. journald keeps every archived file until vacuum limits bite;
Kubernetes keeps exactly one prior instantiation; Airflow keeps every attempt.
None of them keeps zero. There is no source advocating a single fixed path
shared across attempts -- SQLite explicitly names it as a corruption mode.

## Pitfalls (from literature)

- **Same-filesystem requirement.** `rename()` fails `[EXDEV]` across filesystems
  (source 7); a temp file must be created beside the target, not in `/tmp`.
- **Forgetting the directory fsync** -- the file exists but the *name* does not
  survive a power cut (sources 1, 3).
- **Deleting the damaged artifact to "clean up"** is SQLite's documented
  human-error mode ("a well-meaning user or system administrator ... deletes the
  hot journal, thinking that they are helping"), for which they know "no way to
  prevent this other than user education" (source 1).
- **Two names for one file.** SQLite's hard-link case: open via a different link
  and "the hot journal will not be located and no rollback will occur"
  (source 1). The analogue is any second key (cycle, run id) that a reader
  derives differently from the writer.
- **Unbounded retention** is not the precedent; every system here bounds it
  (source 9's `containerLogMaxFiles`, journald vacuuming).
- **Crash-only is not a shortcut licence** (source 8) -- and recovery code must
  be *exercised*, not merely written (source 10).

## Application to pyfinagent

1. **The defect is real, live, and already self-documented.** `qa.md:124-127`
   states the fixed-path hazard; `qa_wip.py:81-87` comments it; the harness log
   queues it as 86.36. What exists today is a **read-side** mitigation only
   (`qa_wip.py:191-217` reports `STALE` when `WRITTEN < spawned_at`), and STALE
   detection only helps in the narrow case where the new spawn dropped *before*
   its first write. Once it writes one byte, the prior attempt is gone -- as
   measured above at 06:40:32Z.
2. **The precedent-aligned fix is a path component, not another marker**
   (source 6's `attempt={try_number}`; sources 4+5's rename-aside). Either
   (a) a cycle/run-suffixed name, e.g.
   `verdict_wip_<sid>__c<N>.md` or `...__<run_id>.md`, with `resolve_wip_path()`
   growing a resolver that returns the newest by `WRITTEN` for reads; or
   (b) journald's shape -- on open, if a file already exists at the fixed path,
   **rename it aside** (`verdict_wip_<sid>.md~<WRITTEN>` ) before writing the new
   born-inert stub. (b) is the smaller change and keeps every existing reader
   working on the unsuffixed path.
3. **No guard change is required, which keeps 86.36 disjoint from 86.33.**
   `qa-write-guard.sh:96-102` tests only that the normalised path contains
   `.claude/agent-memory/qa/`; any suffixed sibling stays inside it. But note
   `qa-write-guard.sh` matches `Write|Edit` only and its own comment (`:18-20`)
   discloses that **Bash subprocess writes are not intercepted** -- so a
   `mv`-based rotation performed by the agent via Bash is outside the hook
   entirely. Design accordingly, and prefer having the *launcher or a helper*
   own the rotation over trusting agent prose.
4. **Concurrency is not hypothetical here.** Three files changed during a
   ~4-minute window from at least two concurrent Q/A sessions (see the measured
   table; cf. the standing two-sessions hazard in project memory). A
   rename-aside that is not atomic can itself race. `os.replace()` /
   `os.rename()` on the same directory is atomic (source 7) and is the right
   primitive; an exclusive-create (`O_EXCL`) on a per-attempt name is even
   stronger because it *cannot* clobber by construction.
5. **Durability beyond visibility is probably out of scope.** The failure mode
   observed is *logical clobber by a peer process*, not power loss. Sources 3+7
   matter for choosing `rename` over truncate-and-rewrite; the full
   `fsync`-file + `fsync`-dir pair is defensible but is not what 86.36's
   evidence demands. Say so explicitly rather than shipping ceremony.
6. **Bound the retention.** Follow source 9: keep the previous attempt (N=1) or a
   small N, not every attempt forever -- the sink is git-tracked, so unbounded
   growth lands in repo history.
7. **Exercise the recovery path** (source 10). The phase-86.31 SIGKILL drop
   simulation is the existing idiom; a 86.36 checker should mutate by *deleting
   the rotation* and prove a prior attempt is destroyed, not merely assert the
   new filename exists.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **10** (9 via WebFetch,
      1 via the sanctioned curl+pypdf PDF chain); >=5 met even discounting #10.
- [x] 10+ unique URLs total -- **18** (10 read-in-full + 7 snippet-only + the
      arXiv API query URL).
- [x] Recency scan (last 2 years) performed + reported -- date-sorted arXiv API
      pass; 18 entries in window; result reported above.
- [x] Full pages/papers read, not abstracts -- source 10 is a 6-page, 34,712-char
      full-text extract; the rest are full HTML pages.
- [x] file:line anchors for every internal claim.

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
      (all 6 named paths inspected; `qa-write-guard.sh` read-only).
- [x] Contradictions / consensus noted (retention-depth tension).
- [x] All claims cited per-claim.
- [ ] **Three-variant search discipline: NOT met** -- the session's WebSearch
      quota (200/200) refused variants 1 and 2. Disclosed above with the two
      compensating measures. This is a soft check, not a hard blocker.
- [ ] Tool-call budget: moderate is <=18; this session used ~30. Over budget,
      disclosed rather than smoothed.

**Envelope status: COMPLETE.** `gate_passed: true` -- all hard blockers met, step
is not audit-class, so `coverage.dry` does not gate.
