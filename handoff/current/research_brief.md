# Research Brief -- phase-62.6 (goal-away-ops): ops hygiene batch (backend.log rotation + autoresearch cron + ablation exit=1 + masterplan 39.1 closure)

Tier: moderate (caller-set). Date: 2026-06-12. Researcher: Layer-3 (merged Explore). STATUS: COMPLETE -- gate PASSED. (~20 tool calls vs 18 budget; overage = WRITE-FIRST preserve of the 62.4 brief + one blocked Write retry.)

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://man.freebsd.org/cgi/man.cgi?query=newsyslog.conf&sektion=5 | 2026-06-12 | official man (macOS newsyslog upstream) | WebFetch full | Fields `logfile owner:group mode count size when flags pid signal`; J=bzip2, Z=gzip, N=no-signal; default signal SIGHUP; rotation is rename-based (no copy-truncate mode exists) |
| https://man.archlinux.org/man/logrotate.8.en | 2026-06-12 | official man | WebFetch full | copytruncate: "Truncate the original log file to zero size in place after creating a copy... very small time slice between copying the file and truncating it, so some logging data might be lost"; for programs that "cannot be told to close [their] logfile" |
| https://pubs.opengroup.org/onlinepubs/9699919799/functions/write.html | 2026-06-12 | standard (POSIX) | WebFetch full | "If the O_APPEND flag... is set, the file offset shall be set to the end of the file prior to each write" -- truncation by another process is safe for O_APPEND writers (next write lands at new EOF; no sparse hole) |
| https://pypi.org/project/langchain-huggingface/ | 2026-06-12 | official package page | WebFetch full | Latest 1.2.2 (2026-04-16); Python >=3.10,<4 (3.14 OK); sentence-transformers is OPTIONAL, required only for local `HuggingFaceEmbeddings` inference (needs sentence-transformers>=5.2.0) |
| https://pip.pypa.io/en/stable/topics/repeatable-installs/ | 2026-06-12 | official docs | WebFetch full | Pin with `==`, `pip freeze`, hash-checking; `--no-deps` protects against unintended installs -- basis for constraints-pinning langchain-core during the install |

## Identified but snippet-only (does NOT count toward gate)
| URL | Kind | Why not fetched |
|---|---|---|
| https://richard-purves.com/2017/11/08/log-rotation-mac-admin-cheats-guide/ | practitioner | snippet sufficed: newsyslog creates rotated files root:root unless owner:group set -> breaks user-agent logs |
| https://patelhiren.com/blog/macos-newsyslog-openclaw-logs/ | practitioner | confirms /etc/newsyslog.d pattern (root-owned dir) |
| https://discussions.apple.com/thread/6752216 | community | "DNS stops logging after rotation" = the rename-vs-held-FD failure in the wild |
| https://access.redhat.com/solutions/3518471 | vendor KB | copytruncate loss case study |
| https://betterstack.com/community/guides/logging/how-to-manage-log-files-with-logrotate-on-ubuntu-20-04/ | guide | logrotate general guide (2024-era) |
| https://community.splunk.com/t5/Getting-Data-In/Why-copytruncate-logrotate-does-not-play-well-with-splunk/m-p/196112 | community | copytruncate + tailing readers caveat |
| + 6 more from the two searches (real-world-systems.com newsyslog man mirror, codedmemes, yandao, usavps, 2x USPTO patent noise) | mixed | low weight |

## Recency scan (2024-2026)
Performed (one search 2025-scoped; PyPI/pip-resolver checks are live 2026 state). Result: no mechanism-level change in rotation tooling (newsyslog/logrotate/POSIX semantics stable); the 2026-current facts are package-level: langchain-huggingface 1.2.2 (Apr 2026) for the langchain 1.x line, and torch 2.12.0 ships cp314 macOS-arm64 wheels (proven by live pip resolve, not literature). Query variants: year-less canonical (newsyslog) + 2025-scoped (copytruncate); no separate 2026-suffixed query -- the dependency frontier was instead established empirically via `pip install --dry-run` against live PyPI, which is stronger evidence than search for this topic.

## Internal code inventory
| File | Lines | Role | Status |
|---|---|---|---|
| ~/Library/LaunchAgents/com.pyfinagent.backend.plist | StandardOut/ErrorPath keys | both point at repo-root backend.log | live (385M / 403,648,199 B at 12:36) |
| ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist | ProgramArguments, Hour=2 | bash scripts/autoresearch/run_nightly.sh | loaded; last exit 0 |
| ~/Library/LaunchAgents/com.pyfinagent.ablation.plist | bash -c, Hour=3 | sources backend/.env, .venv, run_ablation.py --next-untested | loaded; runs=16; last exit 0 |
| scripts/autoresearch/run_nightly.sh | :24,:27 | activates REPO/.venv (THE venv), runs run_memo.py | working |
| scripts/autoresearch/run_memo.py | :131-154,:187-188,:194-197,:37, write_memo :86-91 | preflight skip (51.4) live; NO --dry-run flag; missing ANTHROPIC_API_KEY -> exit 1; memos named `{date}-topic{NN}-{slug}.md` | skip-path nightly since 06-01 |
| scripts/ablation/run_ablation.py | :327-334 | all-tested branch prints `all-features-tested`, `return 0` | exit-0 by design |
| backend/backtest/experiments/feature_ablation_results.tsv | last row 2026-05-24 | 37/37 _NUMERIC_FEATURES tested | complete since 05-24 |
| handoff/ablation.log; ablation.launchd.log | 265 B / 0 B | only last night survives (earlier history truncated by housekeeping) | no traceback recoverable |
| handoff/autoresearch.log; handoff/autoresearch/ | 4 lines; 33 files | last ERROR file 2026-05-31; root_cause.md EXISTS | 11 ERROR-free nights |
| .claude/masterplan.json step 39.1 | verification block | see Sub-item 4 verbatim | pending |
| handoff/current/cycle_block_summary.md | :81 (carried-forward), :20 | prior diagnosis: "pip install + ablation exit=1 triage (operator-gated)" | matches |
| backend/tests/test_phase_23_2_9_ticker_meta_latency.py :72-78; test_phase_23_2_6_sector_cap_emit.py :234 | read backend.log | post-truncate wobble risk (23_2_9 has skip-guard; 23_2_6 guard unverified) | gotcha |
| scripts/away_ops/healthcheck.sh | exists (62.5, 30-min cadence) | candidate home for rotation block; no size-check idiom yet | target |
| .gitignore | :72 | handoff/logs/ ignored -> archive destination | OK |

## Key findings (per sub-item)

### 1. backend.log rotation -- GO, copytruncate, NO restart needed
**Empirical core:** `lsof +fg` shows uvicorn (pid 84680) FD 1/2 on backend.log with flag **`AP` (O_APPEND)** -- launchd opens StandardOutPath in append mode. Per POSIX write(): offset is re-derived at EVERY write, so `cp` + `: > backend.log` is safe -- no restart, no sparse hole, no reopen-by-inode (grep: no code opens backend.log for write; only 2 read-only tests + comments). **newsyslog REJECTED:** rename-based only (no copytruncate mode in the flags table); the launchd-held FD would follow the renamed inode and the new file stays empty until restart (Apple-discussions failure in the wild); /etc/newsyslog.d needs root and creates root:root files that break user LaunchAgents. **Recommended mechanism (ONE):** a size-gated block in `scripts/away_ops/healthcheck.sh` (30-min cadence, observe-role per the 62.5 ownership split; no new plist, no hook-blocked launchctl): `stat -f%z` > 52428800 -> `cp backend.log handoff/logs/backend.log.$(date +%Y%m%d-%H%M%S)` -> `: > backend.log` -> `gzip` the copy (compress AFTER truncate to keep the loss window = cp time, ~1-3 s of INFO lines -- the documented, accepted copytruncate trade-off). First rotation of the 385M backlog: run the same recipe manually in-step. Archives in gitignored `handoff/logs/` (.gitignore:72) -- they contain the FRED key (2,101 plaintext lines; backend/main.py:103 comment), so compressed + local-only + NOT deleted satisfies the forensic-survival criterion. Growth ~6-7 MB/day (397 MB on 06-11 -> 403.6 MB on 06-12): 50 MB cap ≈ weekly rotation cadence.

### 2a. autoresearch cron -- GO with two operator acks
Cron ALREADY exits 0 nightly via the 51.4 preflight skip (run_memo.py:194-197; last night: "skipped... END OK"; zero ERROR files since 05-31). The 62.6 criterion additionally requires `langchain_huggingface` importable from ITS venv = the project `.venv` (run_nightly.sh:24). **Install (bounded, reversible):** `pip install -c <(echo langchain-core==1.2.30) langchain-huggingface sentence-transformers` -> resolver (live dry-run, NO mutation done): langchain-huggingface-1.2.1, sentence-transformers-5.5.1, torch-2.12.0, transformers-5.11.0, safetensors-0.8.0, networkx-3.6.1, sympy-1.14.0, mpmath-1.3.0, setuptools-81.0.0. WITHOUT the constraint, pip UPGRADES langchain-core 1.2.30->1.4.6 (unwanted env drift; pip docs back constraint-pinning). Disk fine (116 Gi free; first nightly run also downloads BAAI/bge-small-en-v1.5 ~130 MB to ~/.cache/huggingface). Record exact versions in the audit note; rollback = `pip uninstall` the nine. **Ack 1:** the install itself (environment mutation, owner-gated per 39.1). **Ack 2 (do not bury):** once importable, the preflight passes and every 02:00 run executes GPTResearcher with Anthropic models (run_memo.py:178-180) -- nightly LLM spend RESUMES (est. $0.10-0.50/night, haiku/sonnet-class) -- requires explicit operator approval per the LLM-cost rule. **Dry invocation trap:** run_memo.py has NO --dry-run flag; any invocation with deps present + ANTHROPIC_API_KEY spends tokens, and WITHOUT the key it exits 1 (:187-188). Recommend adding a 3-line `--preflight-only` flag (exit 0 after `_embedding_preflight`) as the $0 deterministic dry check; else accept one paid memo run as the live check.

### 2b. ablation exit=1 -- NOT REPRODUCIBLE; document, don't disable
Live state: `launchctl print` last exit code = 0, runs = 16; log tail = benign RequestsDependencyWarning + `all-features-tested`; that branch returns 0 BY DESIGN (run_ablation.py:329-331). All 37/37 _NUMERIC_FEATURES have TSV verdicts (complete 2026-05-24 03:20) -> every night since 05-25 is a deterministic no-op exit 0. The historical exit=1 (carried in cycle_block_summary.md:81 from the 06-01 goal) predates the surviving 265-byte log; its traceback is unrecoverable locally. Honest disposition: **fix-not-needed / documented-with-evidence** -- keep the job loaded (--next-untested self-resumes if _NUMERIC_FEATURES grows); NO disable (avoids the hook-blocked `launchctl disable` entirely and is more honest than forcing a skip-exit into a job that already exits 0).

### 3. Masterplan 39.1 -- closable via success_criteria, NOT via the literal grep
Verbatim: command `ls handoff/autoresearch/ | grep -E '2026-05-(2[3-9]|3[01])-PASS' | head -1`; criteria: 3 consecutive launchd-exit-0 nights + root_cause documented in handoff/autoresearch/root_cause.md + operator action in audit trail; live_check_39.1.md = "3 consecutive PASS rows". The command can NEVER match, for three independent reasons: (a) the May 23-31 window produced only `-ERROR-` files; (b) success memos are named `{date}-topic{NN}-{slug}.md` (run_memo.py write_memo :86-91) -- no `-PASS` token exists in any filename the script produces; (c) the pipeline's exit code is head's (always 0) so the command was always evidence-by-output, not exit-code. "Owner-gated" = the FIX needs operator-approved actions (pip install; "Sandbox-blocked from automating" per audit_basis) -- satisfied by operator PRESENCE granting the approval, plus recording it (criterion 3). Closure paths: STRICT -- install today, memo files 06-13/14/15 = 3 exit-0 nights with the dep live -> close Monday 06-15. LENIENT -- 11 ERROR-free skip-path nights since 06-01 (+ launchctl exit 0) already satisfy "exit 0 x3", root_cause.md exists, install recorded today -> close now. Main/Q&A pick; criteria are immutable -- cross-reference, never edit.

## Risks & gotchas
1. cp-to-truncate loss window (~1-3 s of INFO lines) -- documented logrotate trade-off; schedule away from 18:00 UTC cycle / 02:00 / 03:00.
2. Post-truncate, log-grepping tests: test_phase_23_2_9 skips cleanly (:74); test_phase_23_2_6 (:234) guard UNVERIFIED -- check before claiming a green suite (known env-coupled watermelon class).
3. Unconstrained pip would silently bump langchain-core to 1.4.6 -- always pass the constraints file.
4. Archived gz files hold the FRED key until the deferred key rotation -- keep in gitignored handoff/logs/, never commit, never delete.
5. Install flips the nightly job from $0-skip to real spend -- get the explicit cost ack BEFORE installing, not after.
6. ablation/autoresearch logs were already truncated once by housekeeping -- the 62.6 audit note is now the durable record of the exit=1 investigation.

## GO/NO-GO
| Sub-item | Verdict |
|---|---|
| 1 rotation | **GO** (independent of cron items) |
| 2a autoresearch install | **GO** conditional on operator acks 1+2 |
| 2b ablation | **GO** (documentation-only; no code/launchctl change required) |
| 3 39.1 closure | **CONDITIONAL GO** -- lenient path today, strict path Monday 06-15 |

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5)
- [x] 10+ unique URLs total (18)
- [x] Recency scan performed + reported (with query-variant disclosure)
- [x] Full pages read for the read-in-full set
- [x] file:line anchors for every internal claim

```json
{"tier": "moderate", "external_sources_read_in_full": 5, "snippet_only_sources": 13, "urls_collected": 18, "recency_scan_performed": true, "internal_files_inspected": 14, "gate_passed": true}
```
