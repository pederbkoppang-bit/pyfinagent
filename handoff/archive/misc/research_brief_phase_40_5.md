# phase-40.5 Research Brief -- _LAUNCHD_JOBS stale description (OPEN-30)

Tier: SIMPLE. Researcher run 2026-05-23. Audit basis: closure_roadmap
OPEN-30 + masterplan 40.5 audit_basis "stale description". Operator
override 2026-05-22 (`feedback_never_skip_researcher`): ALWAYS spawn
researcher even on cosmetic / harness_required=false steps.

---

## Section A -- Internal audit (file:line precision)

### A.1 -- Live grep verdict

```
$ grep -rn 'FAILING exit 127' /Users/ford/.openclaw/workspace/pyfinagent/backend/ \
                                /Users/ford/.openclaw/workspace/pyfinagent/scripts/
(no matches; exit 1 -- empty)
```

Variant grep `grep -rn 'FAILING'` and `grep -rn 'exit 127'` against
`backend/` + `scripts/` also returned empty. The string is GONE from
source code. Surviving hits in the repo are ALL in non-source paths:

| Path | Role |
|------|------|
| `handoff/harness_log.md:15835, 15876, 15988, 15997, 16030` | Historical log entries describing the original audit + the 23.6.2 cleanup. **Append-only audit stream; correct to leave intact.** |
| `handoff/archive/phase-33.2/research_brief.md:104, 173` | Roadmap snapshot listing OPEN-30 as open. **Frozen; correct to leave intact.** |
| `handoff/archive/phase-23.6.2/*.md` | Snapshot of the cleanup cycle (contract/results/evaluator/research_brief). **Frozen; correct to leave intact.** |
| `handoff/archive/phase-23.5.19/*.md` | Snapshot of the audit cycle that *found* the stale string. **Frozen; correct to leave intact.** |
| `handoff/archive/phase-23.3.4/*.md` | Snapshot of the original launchd audit. **Frozen; correct to leave intact.** |
| `handoff/current/phase-23.6.2-research-brief.md:51, 117, 191` | Stale active-tree copy of the 23.6.2 brief. **Should already have been archived; light housekeeping candidate but not in scope for 40.5.** |
| `handoff/current/master_roadmap_to_production.md:235, 913` | Live roadmap that drove THIS phase. Line 235 narrates OPEN-30 ("FAILING exit 127 stale description"); line 913 is the proposed verifier command `grep -L 'FAILING exit 127' backend/ -r`. **Both correct as-is** -- the roadmap is the spec that delivered the cleanup, not source code. |

### A.2 -- Git history of the cleanup

```
$ git log -S 'FAILING exit 127' --all --oneline
aa421893 phase-44.0: Super-planning: deep research + per-page expanded master design
32ba02e6 phase-33.2: Master roadmap to production (super-planning)
2301b977 phase-23.6: harness MAS cycles 23.6.0-23.6.3 + observability follow-up
a80f4640 phase-23.3.4: launchd audit -- 6-entry manifest + autoresearch finding
```

The string was **introduced** in `a80f4640` (phase-23.3.4, 2026-05-XX
audit) and **removed from source** in `2301b977` (phase-23.6.2,
2026-05-11). The two more-recent SHAs (`32ba02e6` and `aa421893`) only
touch documentation/handoff files, NOT source.

Verbatim from commit `2301b977` body:
> "23.6.2 -- cosmetic schedule labels + autoresearch description
> refresh: ... com.pyfinagent.autoresearch description updated from
> stale 'FAILING exit 127' to current 'exit 1 -- partial .env fix
> applied'."

### A.3 -- `_LAUNCHD_JOBS` location and current shape

`backend/api/cron_dashboard_api.py:108-121` -- 6-tuple of dicts. Current
state of each description string (verified live 2026-05-23 via
`/api/jobs/all`):

| id | schedule (in code) | description (in code) | live status | matches reality? |
|----|---------------------|-----------------------|-------------|------------------|
| `com.pyfinagent.backend-watchdog` | `launchd interval 60s` | `External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)` | `ok` | YES (watchdog plist StartInterval=60) |
| `com.pyfinagent.backend` | `launchd KeepAlive RunAtLoad` | `FastAPI backend daemon (uvicorn :8000); auto-respawns on EXIT` | `running` | YES |
| `com.pyfinagent.frontend` | `launchd KeepAlive RunAtLoad` | `Next.js frontend dev server (:3000)` | `running` | YES |
| `com.pyfinagent.mas-harness` | `launchd interval 1800s` | `MAS harness optimizer cycle (every 30 min)` | `ok` | YES (plist StartInterval) |
| `com.pyfinagent.ablation` | `launchd cron 03:00 daily` | `Nightly feature ablation experiment` | `ok` | YES (plist StartCalendarInterval Hour=3 Minute=0) |
| `com.pyfinagent.autoresearch` | `launchd cron 02:00 daily` | `Nightly autoresearch memo (exit 1 -- partial .env fix applied; python entrypoint still failing -- see phase-23.5.19)` | `failed` (last exit code = 1, runs = 16) | YES -- launchctl print shows `last exit code = 1`; daily ERROR-topic artifacts produced through 2026-05-22 |

**Other potentially stale strings inspected:** the only candidate for
further drift is the autoresearch description's "phase-23.5.19"
reference -- if phase-23.5.19 closes, that reference becomes stale
too. As of 2026-05-23 the autoresearch job is STILL failing with
exit 1 (live launchctl evidence) so the description is still accurate.
**No additional stale strings found in `_LAUNCHD_JOBS`.**

### A.4 -- Existing test coverage for `_LAUNCHD_JOBS`

- `tests/verify_phase_23_6_2.py:118-130` -- Check 4 explicitly asserts
  "no `FAILING exit 127` AND contains `exit 1` AND contains
  `phase-23.5.19`" in the autoresearch description. **This test is
  already a regression guard for the exact issue in OPEN-30.**
- `tests/verify_phase_23_3_4.py:31-33` -- asserts the 6-entry shape
  (each `id` present, `claude-code-proxy` excluded).
- `tests/api/test_cron_dashboard.py:89, 107, 252` -- structural
  assertions on `len(launchd_jobs) == len(cda._LAUNCHD_JOBS)`.

---

## Section B -- External sources (>=5 read in full)

| # | URL | Accessed | Kind | Fetched | Key quote / finding |
|---|-----|----------|------|---------|---------------------|
| 1 | https://keith.github.io/xcode-man-pages/launchd.plist.5.html | 2026-05-23 | Official doc (Apple man page mirror) | WebFetch full | "The last exit code is displayed. A value of 0 indicates that the job finished successfully, a positive number that the job has reported an error, a negative number that the process was terminated because it received a signal." `StartCalendarInterval` "causes the job to be started every calendar interval as specified ... semantics ... much like crontab(5)". |
| 2 | https://launchd.info/ | 2026-05-23 | Authoritative tutorial | WebFetch full | "The second column displays the last **exit code**. A value of 0 ... a positive number that the job has reported an error." Recommends `StandardOutPath` + `StandardErrorPath` for debugging exit-127 specifically. |
| 3 | https://lucaspin.medium.com/where-is-my-path-launchd-fc3fc5449864 | 2026-05-23 | Authoritative blog (named author) | WebFetch full | LaunchD's default daemon PATH = `"/usr/bin:/bin:/usr/sbin:/sbin"` (excludes `/usr/local/bin`). Exit-127 happens because launchD-launched scripts can't find commands in `/usr/local/bin`. Three mitigations: `EnvironmentVariables` in plist, full absolute paths in script, inline `export PATH=...` in script. |
| 4 | https://www.conventionalcommits.org/en/v1.0.0/ | 2026-05-23 | Official spec | WebFetch full | "Types other than `fix:` and `feat:` are allowed" (chore/docs/refactor recommended). Cosmetic cleanup typically `chore:` or `docs:`. Footers like `Closes #N` / `Refs: #N` link cleanup to tickets. |
| 5 | https://tldp.org/LDP/abs/html/exitcodes.html | 2026-05-23 | Authoritative doc (TLDP) | WebFetch full | Reserved exit codes: **127 = "command not found"** (PATH problem or typo). 126 = "command exists but cannot execute" (permissions / not executable). 1 = catchall. The 127 vs 1 distinction matters: "127 immediately points to a PATH or command availability issue rather than a generic failure." |
| 6 | https://snyk.io/blog/the-dangers-of-assert-in-python/ | 2026-05-23 | Authoritative blog (Snyk security) | WebFetch full | "Running Python in optimized mode (PYTHONOPTIMIZE / `-O`) sets `__debug__` to False, thereby disabling assert statements." **Relevance to phase-40.5:** if the regression test uses plain `assert`, it could silently no-op under `-O`. Snyk recommends `if` + `raise` for safety-critical checks. **Caveat:** pytest collects test_* functions and asserts via assertion rewriting; plain `assert` in pytest tests is fine. |
| 7 | https://mobeets.github.io/blog/launchd/ | 2026-05-23 | Authoritative blog | WebFetch full | Debugging launchd: capture stdout/stderr via `StandardOutPath` + `StandardErrorPath`, then `tail -F` + `grep`. Reinforces that "Exited with code: 1" / "code: 127" is the ONLY observable signal from launchd-managed scripts. |
| 8 | https://arxiv.org/html/2510.04952v1 | 2026-05-23 | Peer-reviewed-adjacent arXiv (Safe & Compliant Cross-Market Trade Execution) | WebFetch full | Section 4.3 zkCA: "After each trading session ... Compliance Agent prepares a compliance report ... a log of all trades and a signed statement that no rules were broken. We enhance this process with a zero-knowledge proof of compliance." Section 4.2: shield filtering creates "a gap between what the agent *proposes* versus what actually executes -- a critical accountability mechanism." **Relevance:** the "agent's claimed state vs the actual state" gap is exactly what OPEN-30 surfaces -- a stale string was an agent's *claim* (originally accurate) that drifted from the live actual state. |
| 9 | https://atlan.com/know/context-drift-detection/ | 2026-05-23 | Industry / vendor blog | WebFetch full | Forrester 2025: 'agent drift' is "the silent killer of AI-accelerated development." Four drift signals: schema version staleness / glossary age / lineage completeness / ownership freshness. "A single score per asset reflects the accumulated staleness across the schema, semantic, and lineage dimensions." Active metadata monitoring "recalculates freshness continuously rather than relying on quarterly audits." |

### Identified but snippet-only (context; not counted toward gate)

| URL | Why not fetched in full |
|-----|--------------------------|
| https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html | WebFetch timeout 60s then ECONNREFUSED on retry (gnu.org appears flaky for WebFetch). TLDP exitcodes.html covers the same ground canonically. |
| https://linuxconfig.org/how-to-fix-bash-127-error-return-code | HTTP 403 Forbidden via WebFetch. Snippet content quoted in WebSearch result above. |
| https://docs.pytest.org/en/stable/how-to/assert.html | Page covers pytest assertion mechanics but lacks anti-regression / forbidden-string content. Confirmed in 3 separate searches (year-less, 2026, anti-regression queries). |
| https://www.securityscientist.net/blog/12-questions-and-answers-about-building-an-audit-trail-from-jira-and-git-complete-guide-for-2026/ | Article focuses on Jira/Git automation rather than the cleanup-commit-linking pattern I targeted. |
| https://community.jamf.com/t5/jamf-pro/launchd-plist-erring-with-quot-127-quot/m-p/168533 | Forum thread (community tier, lowest weight in source hierarchy). |
| https://discussions.apple.com/thread/5668076 | Forum (community tier). |
| https://discussions.apple.com/thread/978989 | Forum (community tier). |
| https://manpagez.com/man/5/launchd.plist/ | Same content as source #1 (keith.github.io mirror). |
| https://leancrew.com/all-this/man/man5/launchd.plist.html | Same content as source #1 (mirror). |
| https://tiffanybbrown.com/2023/04/launchd-error-status-29-78-255/ | Pre-Dec-2023; older. |
| https://github.com/fallow-rs/fallow | Tool exists (TypeScript/JS dead-code detection, "stale suppressions") -- not Python so not directly applicable. |
| https://www.codeant.ai/blogs/static-code-analysis-tools | Industry survey; recency hit. |

**Total: 9 in-full + 12 snippet-only = 21 unique URLs evaluated.**

---

## Section C -- Verdict on "pre-closed" status

**TRUE -- the underlying defect is closed.** The string `"FAILING exit
127"` no longer exists in `backend/` or `scripts/` source code. The
cleanup landed in commit `2301b977` (phase-23.6.2, 2026-05-11) and was
done as part of a 4-cycle harness MAS batch with mandatory Q/A
verdict. The existing verifier `tests/verify_phase_23_6_2.py` Check 4
already asserts the regression invariant ("no `FAILING exit 127` AND
contains `exit 1` AND contains `phase-23.5.19`").

**Why the roadmap still lists OPEN-30 as open:** the roadmap was
produced in phase-33.2 (commit `32ba02e6`) which post-dates the
cleanup. The phase-33.2 deduplication scan apparently grepped the
`handoff/archive/` snapshots that *contain quoted references* to the
old string, not the actual source code. Per
`feedback_never_skip_researcher`: "the roadmap is a snapshot in
time and researcher revalidates against drift." This phase is the
revalidation, and OPEN-30 is empirically already closed.

**One ambiguity worth noting (but NOT changing the verdict):** the
roadmap's proposed verifier command at line 913 is
`grep -L 'FAILING exit 127' backend/ -r`. That's `grep -L`
(files-WITHOUT-match) -- which would return ALL files since none
contain the string; the masterplan verification command uses
`grep -rn ... | wc -l == 0` which is the correct shape. Both
verifications PASS today; the masterplan command is the canonical
one.

---

## Section D -- Recency scan (2024-2026)

Last-2-year scan performed. Two findings:

1. **"Agent drift" / context-drift detection (Forrester 2025)** --
   Forrester named context-drift the "silent killer of AI-accelerated
   development" in 2025. Atlan's 2026 guide describes 4 drift signals
   (schema staleness, glossary age, lineage completeness, ownership
   freshness) and recommends "active metadata monitoring [that]
   recalculates freshness continuously rather than relying on
   quarterly audits." **Relevance to phase-40.5:** the OPEN-30 defect
   is a perfect micro-instance of this -- a description string that
   was true in April 2026 (autoresearch failing with exit 127) drifted
   over 1 month to a different failure mode (exit 1) without the
   in-code description tracking. A regression test that grep's for
   the stale token is the cheapest "active metadata monitoring" gate.
2. **Conventional Commits v1.0.0** (the active spec, no v2 in
   2024-2026) remains the canonical commit-message standard. No new
   conventions for cleanup/cosmetic commits have superseded `chore:`
   / `docs:` types.

**No 2024-2026 sources supersede or contradict** the TLDP / launchd
manual / POSIX exit-code documentation -- these are stable canonical
references.

---

## Section E -- 3-variant queries (compliance check)

| Variant | Example query run | Hits surfaced |
|---------|-------------------|---------------|
| Current-year (2026) | `"launchd plist exit code 127 bash command not found 2026"` | linuxconfig + Apple forum + Jamf forum + lucaspin Medium |
| Last-2-year (2024-2025) | `"launchd exit-code documentation 2025 2026"` | launchd.info + manpagez + masklinn-cheatsheet |
| Year-less canonical | `"exit 127" bash POSIX "command not found" specification` (no year) | gnu.org Exit-Status + linuxconfig + tldp exitcodes |
| Year-less canonical (regression-test angle) | `"pytest regression test grep static string assertion forbid pattern"` (no year) | pytest official docs (multiple versions) + pytest-regex pypi |
| Year-less canonical (audit) | `"code cleanup audit trail discipline changelog conventions reproducibility 2026"` | securityscientist + digitalapplied + theaiops |

5+ search variants covering all three categories. **Year-less canonical
sources surfaced TLDP exitcodes + gnu.org Bash Manual + Conventional
Commits v1 spec** -- the founding/canonical docs the year-locked
search would have buried.

---

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 12,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (9)
- [x] 10+ unique URLs total (21)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

---

## Section G -- Application notes for the planner

1. **OPEN-30 is empirically closed.** The verification command
   (`grep -rn 'FAILING exit 127' backend/ scripts/ | wc -l == 0`)
   returns 0 today and has done so since commit `2301b977`
   (phase-23.6.2, 2026-05-11). **Recommended action: ship the
   masterplan entry as a no-source-change closure** with an audit-
   note in `experiment_results.md` documenting the prior cleanup
   SHA and the live grep evidence. Do NOT re-edit
   `cron_dashboard_api.py:120` -- it is already correct.
2. **A regression test ALREADY EXISTS.** `tests/verify_phase_23_6_2.py`
   Check 4 (lines 118-130) already asserts the exact invariant
   "no FAILING exit 127 AND contains exit 1 AND contains
   phase-23.5.19". A new pytest is unnecessary; phase-40.5 can
   simply cite the existing test as the regression guard.
3. **OPTIONAL housekeeping (NOT in 40.5 scope):**
   `handoff/current/phase-23.6.2-research-brief.md` is a stale copy
   in the active tree (the archive copy is the authoritative one).
   `verify_handoff_layout.py` should flag this; don't address it
   here.
4. **False-negative grep risk -- LOW.** `grep -rn 'FAILING exit 127'
   backend/ scripts/` is sufficient because: (a) the string is
   exact-match (no regex variations to worry about), (b) it's
   English (no locale/encoding issues), (c) the original audit only
   ever placed the string in one location (line 120 of
   `cron_dashboard_api.py`), (d) frontend/TypeScript code doesn't
   render `_LAUNCHD_JOBS` descriptions -- those are surfaced via
   `/api/jobs/all` which returns the runtime tuple. Only theoretical
   gap: someone could embed the string in a Python docstring or
   comment elsewhere, but `grep -rn` finds those too.
5. **Forrester "agent drift" frame -- worth mentioning in
   experiment_results.md.** The OPEN-30 cycle is a small but
   illustrative instance of the broader doctrine: descriptions /
   status banners need active drift-detection gates, not quarterly
   audits. The pyfinagent harness already has the file-based audit
   trail (`handoff/harness_log.md` + archive snapshots); the
   single-line regression test is the missing "continuous freshness
   monitor."
6. **Effort tier sanity check:** the masterplan has 40.5 marked
   `simple` / 0.2 effort / SAFE-OVERNIGHT. Researcher gate cleared
   without surprises -- the planner can hold this estimate.

---

## Cross-references

- `CLAUDE.md` -- harness protocol (cycle-2 flow, research gate).
- `.claude/rules/research-gate.md` -- 5-source floor, recency scan,
  3-variant queries.
- `handoff/archive/phase-23.6.2/` -- prior cleanup snapshot (the
  cycle that closed the underlying defect).
- `handoff/archive/phase-23.5.19/` -- prior audit snapshot (the
  cycle that *found* the stale string).
- `tests/verify_phase_23_6_2.py:118-130` -- pre-existing regression
  guard.

---

End of brief.
