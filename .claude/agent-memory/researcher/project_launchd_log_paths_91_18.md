---
name: launchd-log-paths-91-18
description: phase-91.18 -- launchd StandardOutPath resolves against RootDirectory NOT WorkingDirectory; the 23.3.5 "repo-root handoff/" rule was a correct observation over-generalised; 4 of 10 _log_paths() entries point at files that do not exist
metadata:
  type: project
---

The /cron log-tail allowlist `backend/api/cron_dashboard_api.py::_log_paths()` encodes a **directory
convention** where it should encode **what the plist actually says**.

**Why:** phase-23.3.5 observed that `autoresearch` and `mas-harness` logged to repo-root `handoff/`
rather than `handoff/logs/`, wrote that as a general rule in the comment at `:140-144`, and applied it
to `ablation` too. But `ablation`'s two writers were both already on `handoff/logs/`:
`scripts/ops/run_ablation.sh:14` sets `LOG="$REPO/handoff/logs/ablation.log"`, and the
`com.pyfinagent.ablation` plist sets both stdio keys to `handoff/logs/ablation.launchd.log`. Measured
2026-08-20: 4 of 10 allowlist targets do not exist on disk, while the real `handoff/logs/ablation.log`
was 12,227 B and fresh from that morning's 03:00 run. Two of the four (`harness`,
`mas_harness_launchd`) are dangling for a *different* reason -- the mas-harness plist exists only as
`.bak-harness-ABCD`, i.e. the job is not installed -- so a naive "every path must exist" test would
fail on them.

**How to apply:** when any question is "where does this background job log?", read the plist's
`StandardOutPath`, never a sibling job's directory. Three durable traps behind this:

1. A relative `StandardOutPath` resolves against `RootDirectory` or `/`, **not** `WorkingDirectory` --
   while the *executable's* own relative paths DO follow `WorkingDirectory`. Two resolution bases in
   one plist file. (https://www.launchd.info/)
2. `get_log_tail` (`:583-590`) returns HTTP **200** with `exists:false` for a missing file, so a wrong
   path is indistinguishable from an idle job at the API layer. This drift class is silent by
   construction and has already recurred once.
3. A 0-byte launchd log is often CORRECT, not broken: `run_ablation.sh:44-46` redirects the child's
   stdout+stderr into its own `$LOG`, leaving launchd almost nothing to capture.

Do NOT read the `handoff/logs/*-vN.log` families as held-fd rotation damage -- `scripts/ops/rotate_logs.sh:16-22`
deliberately uses cp+truncate and never renames ("leaves the SAME open FD writing fresh from offset 0",
lsof-verified). The `-vN` files came from successive plist edits pointing at new suffixed names.
Related: [[reference-launchd-stdio-key-semantics]].
