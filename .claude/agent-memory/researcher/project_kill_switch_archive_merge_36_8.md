---
name: project-kill-switch-archive-merge-36-8
description: Measured 36.8 facts — archives are the SOLE live baseline source, a file cap would delete the true peak, update_peak can't write a lower peak (so the failure door is the swallowed archive-scan exception), and PostgreSQL prefers archive OVER live (refutes live-file-precedence)
metadata:
  type: project
---

Kill-switch archive-merge research (step 36.8, 2026-07-26). Measured, read-only,
live-file md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` unchanged.

1. **The archives are not a hypothetical determinant of the live safety
   threshold — they are the ONLY determinant.** The LIVE
   `handoff/kill_switch_audit.jsonl` held 8 rows, all pause/resume, ZERO
   baselines. `KillSwitchState().snapshot()` → `peak_nav 24666.57` (from the
   OLDEST archive, June 3) + `sod_nav 23838.19` (from `-v4`). Corpus: 5 files,
   897 rows, 20 `peak_update`, **0 `peak_reset` ever written** (reset_peak has
   been DARK since 69.1).
2. **A "keep N newest files" cap is actively harmful, and that is measured:**
   the true peak lives in the OLDEST file. A cap drops it → peak falls to
   24124.77 (−2.2pp trailing-DD headroom); dropping all four disarms the
   trailing leg. All five audit files ARE git-tracked (recoverability backstop).
3. **Boot cost is a non-issue:** 897 rows in 0.88 ms; `KillSwitchState()` mean
   0.95 ms ⇒ ≈1.06 µs/row (linear `json.loads` scan). 1M rows ≈ 1.1 s. The
   criterion-3 risk is SEMANTIC (a pruned file silently moves a threshold), not
   performance.
4. **`update_peak` cannot write a lower `peak_update` while the merged peak is
   higher** — it reads the already-merged in-memory peak. So the step's "book
   re-anchors lower" premise is only reachable via the SWALLOWED archive-scan
   exception (`kill_switch.py:104-105`) / an absent-then-present
   `handoff/audit/` / a later-restored archive file. And once
   `kill_switch_peak_reset_enabled` is flipped, a fresh `peak_reset` already
   wins by ts-order — the DARK flag is what makes the hole reachable.
5. **PostgreSQL refutes "live file wins":** "segments that are available from
   the archive will be used in preference to files in `pg_wal/`" — because the
   live segment is the tearable one. Pair it with Kafka KIP-101 (the high
   watermark could not tell divergent lineages apart; a leader EPOCH could) and
   PostgreSQL timelines: the literature's answer to "a monotonic fold must be
   correctable" is uniformly an in-stream BOUNDARY MARKER, never file recency.
   No 2024-2026 source offers a clever `max()` variant — an honest null.

**Why:** these four measurements each flip a plausible-sounding fix (live-wins,
add a cap, "a withdrawal triggers it") into a wrong one.
**How to apply:** reuse before re-measuring on 36.8/36.15/36.9; re-verify the
corpus numbers first, since any real cycle appends rows.
Related: [[project_kill_switch_36_12_traps]].
