# Research Brief -- phase-91.18

**Tier:** simple (caller-stated). **Audit-class:** NO (coverage reported for information only).
**Objective:** launchd `StandardOutPath`/`StandardErrorPath` log-file resolution conventions; safe
path-construction patterns for a log-tailing API that must match where a background process actually
writes its logs.
**Internal scope:** `backend/api/cron_dashboard_api.py` (`_log_paths()`), `scripts/ops/run_ablation.sh`,
the `com.pyfinagent.ablation` launchd plist, `handoff/logs/` vs `handoff/` path conventions.
All internal facts measured on disk 2026-08-20.

<!-- ENVELOPE: born inert per phase-86.37. Flipped to COMPLETE as the final act. -->

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 16,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "gate_passed": true
}
```

## Status log (write-first, incremental)

- [t0] Brief created; envelope born INCOMPLETE. Internal exploration starting.
- [t1] Internal half DONE. Two live path mismatches found.
- [t2] Search round 1 done (2 variants). Fetching sources in full.
- [t3] 6 sources read in full; recency-scan variant run.
- [t4] `rotate_logs.sh` REFUTED my own `-vN` inference -- corrected below. Envelope -> COMPLETE.

---

## Read in full (6; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://www.manpagez.com/man/5/launchd.plist/osx-10.12.3.php | 2026-08-20 | Official man page (tier 2) | WebFetch | `StandardOutPath`: "any writes to the job's stdout will go to the given file. **If the file does not exist, it will be created** with writable permissions ... reflecting the umask specified by the Umask key". `WorkingDirectory`: "a directory to chdir to **before running the job**." Man page is **silent** on relative-vs-absolute and on fd reopen. |
| https://www.launchd.info/ | 2026-08-20 | Authoritative community tutorial (tier 3/5) | WebFetch | The load-bearing rule: "**Provide absolute paths when possible. Relative paths are interpreted relative to `RootDirectory` or `/`.**" -- i.e. NOT relative to `WorkingDirectory`. Separately: "Every relative path **the executable** accesses will be relative to its working directory." Also "Shell globbing and variable expansion do not work". |
| https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html | 2026-08-20 | Apple official docs (tier 2) | WebFetch | "Do not set the working directory. Include the `WorkingDirectory` key in your daemon's configuration property list instead." Shows `/var/log/myjob.log` only as a debugging **example**; **prescribes no log-directory convention**. |
| https://github.com/apple-opensource/launchd/issues/1 | 2026-08-20 | Source-repo issue (tier 2/5) | WebFetch | "launchd fails to redirect stdout to the file after log rotation". Last line written is `newsyslog[2662]: logfile turned over due to -F request`, then nothing. launchd does **not** reopen the path. No workaround offered. |
| https://patelhiren.com/blog/macos-newsyslog-openclaw-logs/ | 2026-08-20 | Practitioner blog, dated **2026-02-25** (tier 3/4) | WebFetch | launchd `StandardOutPath` files "just append forever" (author's grew to 38 MB). Rotation tools "default to `root:root` ownership" -> a user-level agent then "fails to start with 'Permission denied'". Fix: put `owner:group` in the newsyslog line. |
| https://portswigger.net/web-security/file-path-traversal | 2026-08-20 | Authoritative security reference (tier 3) | WebFetch | "The most effective way to prevent path traversal vulnerabilities is to **avoid passing user-supplied input to filesystem APIs altogether**." Otherwise: whitelist, then "canonicalize the path. Verify that the canonicalized path starts with the expected base directory." |

## Identified but snippet-only (16; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://forums.developer.apple.com/thread/120784 | Apple forum | Confirms no `$HOME`/`~` expansion; point already covered by launchd.info |
| https://community.jamf.com/general-discussions-2/using-launchagents-to-move-files-23044 | Forum | MDM-specific, not path-resolution |
| https://meshmini.com/en/blog/articles/openclaw-2026-daemon-autostart-launchd-systemd-windows-service-crash-recovery-log-rotation-faq/openclaw-2026-daemon-autostart-launchd-systemd-windows-service-crash-recovery-log-rotation-faq.html | Vendor blog 2026 | Low tier; recency datapoint only |
| https://discussions.apple.com/thread/4475316 | Forum | Old, syslog-specific |
| https://en.wikipedia.org/wiki/Log_rotation | Encyclopedia | Generic background |
| https://www.apisec.ai/blog/path-traversal-in-apis-detection-and-prevention | Vendor blog | Superseded by PortSwigger |
| https://www.securecodinghub.com/guides/path-traversal | Vendor guide 2026 | Superseded by PortSwigger |
| https://nodejsdesignpatterns.com/blog/nodejs-path-traversal-security/ | Blog | Node-specific |
| https://learn.securecodewarrior.com/secure-coding-guidelines/injection-path-traversal | Training | Paywalled/low signal |
| https://www.vulnsy.com/cheat-sheets/lfi | Cheat sheet | Offence-oriented |
| https://www.devsecopsnow.com/path-traversal/ | Blog | Duplicate of above |
| https://www.decryptiondigest.com/blog/path-traversal-directory-traversal-detection-remediation | Blog 2026 | Duplicate |
| https://ismaildawoodjee.medium.com/file-path-traversal-6782ac650f2c | Medium | Community tier |
| https://tonygo.tech/blog/2023/build-your-first-macos-agent-with-launchd | Blog 2023 | Year-less canonical hit; covered by launchd.info |
| https://veerpalbrar.github.io/blog/How-to-Run-A-Program-or-Script-Hourly-on-MacOS/ | Blog | Beginner-level |
| https://surfer.nmr.mgh.harvard.edu/fswiki/MacOsLaunchd | Academic wiki | Documents the `~/Library/Logs/LaunchAgents/$Label/{out,err}.log` convention; snippet sufficed |

### Search-query composition (three-variant discipline)

1. **Year-less canonical** -- `launchd StandardOutPath StandardErrorPath relative path WorkingDirectory resolution`; `path traversal prevention allowlist canonical path resolution log file API endpoint`.
2. **Current-year frontier (2026)** -- `launchd log rotation StandardOutPath file descriptor held open newsyslog 2026`.
3. **Last-2-year window (2025)** -- `launchd plist StandardOutPath best practices 2025 log file path absolute macOS LaunchAgent`.

### Recency scan (2024-2026)

Performed. Result: **one new finding in the window, zero supersessions.** The 2026-02-25 practitioner
article (patelhiren.com) is the only recent source adding anything: it documents the post-rotation
`root:root` ownership failure, which is *newer operational knowledge* than the 2016-era man page but does
**not** supersede it. The `launchd.plist(5)` semantics quoted above are unchanged across every version
surfaced (the 10.12.3 page and the current unversioned manpagez page carry identical text for these
keys). The 2025-scoped query returned only tutorial-tier restatements plus the
`~/Library/Logs/LaunchAgents/$Label/out.log` convention. **launchd's stdio keys are a frozen API**; no
2024-2026 change affects this step.

---

## Key findings (external)

1. **Relative `StandardOutPath` does NOT resolve against `WorkingDirectory`** -- it resolves against
   `RootDirectory` or `/`. "Provide absolute paths when possible. Relative paths are interpreted relative
   to `RootDirectory` or `/`" (launchd.info, https://www.launchd.info/, accessed 2026-08-20). The
   *executable's* relative paths do follow `WorkingDirectory` -- two different resolution bases in one
   plist. This is the single highest-value trap for this step: a plist and its wrapper script can write
   the same relative string to two different places.
2. **No `~`, `$HOME`, or variable expansion in plist paths.** "Shell globbing and variable expansion do
   not work" (ibid.). Absolute paths are the only safe form.
3. **launchd creates the file but never re-opens it.** The path "will be created" on first write
   (`launchd.plist(5)`, https://www.manpagez.com/man/5/launchd.plist/osx-10.12.3.php) -- so a *missing*
   file means the job never wrote to that path, not that the path is wrong per se. After a
   rename-style rotation launchd keeps the old fd and output stops
   (https://github.com/apple-opensource/launchd/issues/1).
4. **Apple prescribes no log-location convention.** `/var/log/myjob.log` appears only as a debugging
   example (https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html).
   The de-facto user-agent convention is `~/Library/Logs/LaunchAgents/$Label/`. **Therefore the
   authoritative answer to "where does this job log?" is the plist itself, never a convention.**
5. **Rotation changes ownership, not just the inode.** Rotators default new files to `root:root`, after
   which a user-level agent gets "Permission denied"
   (https://patelhiren.com/blog/macos-newsyslog-openclaw-logs/, 2026-02-25).
6. **The safest log-tailing API takes a KEY, not a path.** "The most effective way to prevent path
   traversal vulnerabilities is to avoid passing user-supplied input to filesystem APIs altogether"
   (https://portswigger.net/web-security/file-path-traversal). Where a path must be built, canonicalize
   then assert the prefix.

## Consensus vs debate

Consensus is total on findings 1-4 and 6; no source disagrees. The only genuine debate is *where* logs
belong (`~/Library/Logs/...` per the de-facto convention vs an app-owned directory) -- and it is moot
here, because every source agrees the plist is authoritative for its own job.

## Pitfalls (from literature, mapped to this step)

- Assuming `WorkingDirectory` anchors `StandardOutPath`. It does not (finding 1).
- Reading a *convention* instead of the *plist* to decide where a job logs (finding 4).
- Treating an empty/missing launchd log as proof the job failed -- it can equally mean the wrapper
  redirected its own stdout elsewhere, or a rotation stole the fd (findings 3, 5).
- Building the tail path from client input rather than a server-side key map (finding 6).

---

## Internal code inventory (the Explore half)

| File / artifact | Anchor | Role | Status |
|---|---|---|---|
| `backend/api/cron_dashboard_api.py` | `:136` `_REPO_ROOT = Path(__file__).resolve().parents[2]` | Anchors the allowlist | **OK** -- `.resolve()` precedes `.parents[]`; depth correct for `backend/api/x.py` |
| `backend/api/cron_dashboard_api.py` | `:139-165` `_log_paths()` | KEY -> Path allowlist | **2 entries WRONG, 2 DANGLING** (table below) |
| `backend/api/cron_dashboard_api.py` | `:130-134` | Comment: "the client passes a KEY, the server resolves it to a fixed Path. Unknown keys -> 400. The server NEVER echoes a raw path back" | **OK and matches finding 6** -- the defect is mapping *correctness*, not traversal |
| `backend/api/cron_dashboard_api.py` | `:140-144` | phase-23.3.5 comment: "live launchd-managed logs write to `handoff/<x>.log` at repo root, NOT `handoff/logs/<x>.log`" | **OVER-GENERALISED** -- true for `autoresearch`, false for `ablation` |
| `backend/api/cron_dashboard_api.py` | `:564-605` `get_log_tail` | Tail endpoint | **Fail-quiet**: unknown key -> 400, but a *missing file* returns HTTP 200 with `exists:false, lines:[]`. A wrong path is therefore indistinguishable from an idle job at the API layer |
| `scripts/ops/run_ablation.sh` | `:14` `LOG="$REPO/handoff/logs/ablation.log"` | Ablation app log | Writes `handoff/logs/`, not `handoff/` |
| `scripts/ops/run_ablation.sh` | `:44-46` `python ... >> "$LOG" 2>&1` | Redirects child stdout+stderr into `$LOG` | **Explains the 0-byte launchd log legitimately** -- almost nothing is left for launchd to capture |
| `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` | `StandardOutPath`/`StandardErrorPath` | launchd capture | Both = `<repo>/handoff/logs/ablation.launchd.log` (absolute -- correct per finding 2) |
| `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` | `StandardOutPath`/`StandardErrorPath` | launchd capture | Both = `<repo>/handoff/autoresearch.launchd.log` -- the repo-root case 23.3.5 generalised from |
| `~/Library/LaunchAgents/com.pyfinagent.logrotate.plist` | `StandardOutPath` | Rotation job | `<repo>/handoff/logs/logrotate.log` -- `handoff/logs/` is the newer convention |
| `scripts/ops/rotate_logs.sh` | `:16-22`, `:54-55` | Rotation | **Already implements the external fix**: "cp+truncate (never mv/rename) is REQUIRED ... leaves the SAME open FD writing fresh from offset 0 with no restart required -- lsof-verified on this machine" |
| `backend/tests/test_phase_40_5_launchd_descriptions.py` | whole file | Only test near this surface | Asserts launchd *descriptions*; **no test asserts `_log_paths()` targets exist** |

### The mismatches (measured on disk 2026-08-20)

| `_log_paths()` key | Points at | Writer actually writes to | On-disk truth |
|---|---|---|---|
| `ablation` `:163` | `handoff/ablation.log` | `run_ablation.sh:14` -> `handoff/logs/ablation.log` | target **MISSING**; real file 12,227 B, mtime **2026-08-20 03:00** (ran last night) |
| `ablation_launchd` `:164` | `handoff/ablation.launchd.log` | plist -> `handoff/logs/ablation.launchd.log` | target **MISSING**; real file exists, 0 B, mtime 2026-05-07 |
| `harness` `:154` | `handoff/mas-harness.log` | no live plist (`.bak-harness-ABCD` only) | target **MISSING**; `handoff/logs/mas-harness.log` = 2.9 MB but mtime 2026-04-19 (stale) |
| `mas_harness_launchd` `:158` | `handoff/mas-harness.launchd.log` | no live plist | target **MISSING** |
| `autoresearch` `:156` | `handoff/autoresearch.log` | live | **CORRECT** -- 285,075 B, mtime 2026-08-20 02:06 |
| `autoresearch_launchd` `:162` | `handoff/autoresearch.launchd.log` | plist `StandardErrorPath` | **CORRECT** -- exists, 0 B |

**4 of 10 allowlist entries point at files that do not exist.** For `ablation` the fresh, non-empty
truth is one directory away.

### Self-correction (recorded, per honesty contract)

An earlier revision of this brief called the `handoff/logs/ablation.launchd-v2..v4.log` family "the exact
signature of the launchd held-fd rotation trap". **That inference is refuted by `rotate_logs.sh:16-22`,
which deliberately never renames.** The `-vN` files are far better explained by successive plist edits
pointing `StandardOutPath` at new suffixed names (the current plist points back at the unsuffixed name;
`-v4` stops at 2026-07-24 03:00, and the plist's own mtime is 2026-07-24 08:52). The held-fd trap is real
in the literature but is **not** what produced these files here, and this repo already defends against it.

---

## Application to pyfinagent

1. **The plist is the source of truth, not the directory convention** (finding 4). `_log_paths()` at
   `cron_dashboard_api.py:139-165` currently encodes a *convention* ("launchd logs live at repo-root
   `handoff/`") that was only ever true of `autoresearch`. Any fix should derive from, or at minimum be
   validated against, the actual `StandardOutPath` strings.
2. **Both ablation paths need `/ "logs"` inserted** -- `:163` and `:164` -- to match
   `run_ablation.sh:14` and the plist respectively. This is the whole product defect; it is one
   directory component in two dict values.
3. **Absolute-path discipline already holds.** Every plist inspected uses absolute paths, so finding 1's
   relative-resolution trap is latent, not live -- but it is the reason a future plist must never be
   written with a relative `StandardOutPath` even though `WorkingDirectory` is set.
4. **Keep the key-not-path API shape** (`:130-134`, `:564-575`); it already implements the strongest
   PortSwigger recommendation. No traversal hardening is needed.
5. **The real gap is the absence of a liveness assertion.** Because `get_log_tail` returns HTTP 200 with
   `exists:false` for a wrong path (`:583-590`), and no test asserts the targets exist
   (`test_phase_40_5_launchd_descriptions.py` covers descriptions only), this class of drift is silent by
   construction and recurred once already (phase-23.3.5 fixed 3 keys and introduced 2). A test that walks
   `_log_paths()` and asserts each target either exists or is explicitly annotated as a known-dead job
   would make the next drift red instead of blank. Note the two `mas-harness` keys are dangling for a
   *different* reason (job not installed), so a naive "every path must exist" test would fail on them --
   the annotation is load-bearing.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6**
- [x] 10+ unique URLs total (incl. snippet-only) -- **22**
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set (no arXiv PDFs in scope)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope, plus `rotate_logs.sh` and the two
      sibling plists needed to test the convention claim
- [x] Contradictions noted -- Apple prescribes no convention while the community does; and my own `-vN`
      inference was refuted by in-repo evidence and is recorded as a correction rather than deleted
- [x] Claims cited per-claim with URL + access date / file:line
