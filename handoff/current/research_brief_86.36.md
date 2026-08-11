# Research Brief -- step 86.36

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Started:** 2026-08-11. **Researcher:** Layer-3 Researcher (Workflow rail).

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
  "brief_status": "INCOMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "coverage": {
    "audit_class": false,
    "rounds": 0,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "gate_passed": false
}
```

**A `brief_status` of `INCOMPLETE` means this brief has NOT passed the gate,
whatever the counts below say.** It is evidence for a re-run, never a pass.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://www.sqlite.org/atomiccommit.html | 2026-08-11 | official doc | WebFetch (HTML, full) | Journal is a FIXED suffix (`-journal`) off the db name, but the multi-file **super-journal is PER-RUN**: "with the text `-mj_HHHHHHHH_` appended where HHHHHHHH is a random 32-bit hexadecimal number. The random HHHHHHHH suffix changes for every new super-journal." Hot journal = "a rollback journal that needs to be played back... A hot journal only exists when an earlier process was in the middle of committing a transaction when it crashed". Commit point = journal deletion. Directory fsync is required "in order to make sure the super-journal file will appear in the directory following a power failure". |
| 2 | https://www.sqlite.org/howtocorrupt.html | 2026-08-11 | official doc | WebFetch (HTML, full) | The shared-path destruction mechanism, stated outright: "Since rollback journals and WAL files are based on the name of the database file, the two different database files will share the same rollback journal or WAL file. A rollback or recovery for one of the databases might use content from the other database, resulting in corruption." And on evidence destruction: "If the hot journal files are moved, deleted, or renamed after a crash or power failure, then automatic recovery will not work and the database may go corrupt." |
| 3 | https://lwn.net/Articles/457667/ | 2026-08-11 | authoritative pub (LWN) | WebFetch (HTML, full) | The canonical atomic-replace recipe, verbatim 5 steps: "create a new temp file (on the same file system!)" / "write data to the temp file" / "fsync() the temp file" / "rename the temp file to the appropriate name" / "fsync() the containing directory". Directory fsync rationale: "A newly created file may require an fsync() of not just the file itself, but also of the directory in which it was created (since this is where the file system looks to find your file)." In-place risk: "If you encounter a system failure ... while overwriting a file, it can result in the loss of existing data." |
| 4 | https://systemd.io/JOURNAL_FILE_FORMAT/ | 2026-08-11 | official doc | WebFetch (HTML, full) | The preserve-the-damaged-artifact rule, verbatim: "If any kind of corruption is noticed by a writer it should immediately rotate the file and start a new one. No further writes should be attempted to the original file, but it should be left around so that as little data as possible is lost." Three-state lifecycle `STATE_OFFLINE=0 / STATE_ONLINE=1 / STATE_ARCHIVED=2`; ONLINE on open-for-write, OFFLINE on clean close, ARCHIVED after rotation. NOTE: this page does NOT document the `.journal~` naming or the archived-filename seqnum format -- see source 5 for that. |

| 5 | https://man7.org/linux/man-pages/man8/systemd-journald.service.8.html | 2026-08-11 | official doc (man page) | WebFetch (HTML, full) | The rotation NAMING scheme + the preserve-on-damage rule together, verbatim: "If the daemon is stopped uncleanly, or if the files are found to be corrupted, they are renamed using the \".journal~\" suffix, and systemd-journald starts writing to a new file." And: "When systemd-journald ceases writing to a journal file, it will be renamed to \"_original-name_@_suffix.journal_\" (or \"_original-name_@_suffix.journal~_\"). Such files are \"archived\" and will not be written to any more." SIGUSR2 / `journalctl --rotate` forces immediate rotation. |
| 6 | https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/logging-monitoring/logging-tasks.html | 2026-08-11 | official doc | WebFetch (HTML, full) | The canonical per-attempt artifact path in a production retry engine, verbatim template: `"dag_id={dag_id}/run_id={run_id}/task_id={task_id}/attempt={try_number}.log"` (and a `map_index={map_index}/` variant for mapped tasks). The attempt number is a PATH COMPONENT, so attempt N+1 physically cannot overwrite attempt N. LIMIT of this source: the page states the templates but carries no prose rationale for per-attempt retention, so the "why" is inferred from the path shape, not quoted. |
| 7 | https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html | 2026-08-11 | official standard (POSIX) | WebFetch (HTML, full) | Atomicity is specified but durability is NOT: "That specification requires that the action of the function be atomic" and "a link named _new_ shall remain visible to other threads throughout the renaming operation and refer either to the file referred to by _new_ or _old_ before the operation began." `[EXDEV]` when old and new are on different file systems. **"The provided content contains no statements regarding data durability guarantees, disk synchronization, or fsync-like behavior associated with rename()."** -> rename gives VISIBILITY atomicity only; durability needs source 3's fsync pair. |
| 8 | https://lwn.net/Articles/191059/ | 2026-08-11 | authoritative pub (LWN) | WebFetch (HTML, full) | See "Key findings" #1 -- crash-only design, and the standing misconception that it licenses shortcuts. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|

## Recency scan (2024-2026)

_(pending)_

## Key findings

_(pending)_

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|

## Application to pyfinagent

_(pending)_

## Research Gate Checklist

- [ ] >=5 authoritative external sources READ IN FULL via WebFetch
- [ ] 10+ unique URLs total (incl. snippet-only)
- [ ] Recency scan (last 2 years) performed + reported
- [ ] Full papers / pages read (not abstracts) for the read-in-full set
- [ ] file:line anchors for every internal claim
