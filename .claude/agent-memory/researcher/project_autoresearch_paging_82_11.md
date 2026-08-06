---
name: autoresearch-paging-82-11
description: phase-82.11 measured facts -- the bash paging seam FIRES but is unobservable (curl >/dev/null + Slack 200-on-error); three failure counters disagree (13/30/62); the Max-rail bridge is UP and only the .env flag is missing; run_memo has 3 silent exit-0 paths so exit-code alerting is structurally blind
metadata:
  type: project
---

Measured 2026-08-06 for masterplan step 82.11 (nightly autoresearch credit-exhaustion +
never-audible failure).

**The seam is not silent -- it is UNOBSERVABLE.** `scripts/autoresearch/run_nightly.sh:49-69`
DID execute last night (`autoresearch_fail_state.json` mtime `Aug 6 02:00:13`,
`consecutive_fails: 13`) and DID pass `new_fails >= PAGE_AFTER_N` (13 >= 3, `:58`). But the
curl at `:63-66` ends `>/dev/null 2>&1 || true`, and Slack `chat.postMessage` returns
**HTTP 200 with `{"ok":false,...}`** for `invalid_auth`/`not_in_channel` -- so `curl -s`
exits 0 either way. "Was the operator paged?" is unanswerable from local state BY
CONSTRUCTION. The Python seam does parse `ok` (`alerting.py:168`). Also: under
`set -euo pipefail`, `BOT_TOKEN=$(grep ... | cut | tr)` at `:59` ABORTS the script when the
key is absent (a failing command substitution in an assignment trips `set -e`).

**Three failure counters, three different numbers.** JSON `consecutive_fails`=13;
consecutive ERROR *dates*=30 (2026-07-08..08-06 unbroken); total ERROR files=62. The JSON is
lower because two MANUAL `run_nightly.sh` runs (07-24, 07-25, max-rail ON) hit the success
branch `:96` and reset it to 0 -- while leaving that night's ERROR file on disk. So
**2026-07-24 and 2026-07-25 each have BOTH an `-ERROR-` file and a success memo**: a naive
"date has an ERROR file => failed" scan miscounts. Nothing in the repo reads the JSON except
`run_nightly.sh:42` and `test_phase_76_9_2_max_bridge.py:333`.

**The Max rail is LIVE; only the operator's `.env` line is missing.** `launchctl list`:
`com.pyfinagent.anthropic-bridge` pid 650, `com.pyfinagent.claude-code-proxy` pid 668;
`curl http://127.0.0.1:18797/health` -> `{"ok":true,"proxy":"claude-code-cli"}`.
`scripts/ops/anthropic_max_bridge.py` present. `AUTORESEARCH_USE_MAX_RAIL=1` is a one-line,
no-restart, revertible $0 fix (`run_nightly.sh:78`; `.claude/masterplan.json:19335` is the
`[OPERATOR ACTION]` step). The two non-ERROR memos on disk (07-24/07-25) ARE the positive
control that the rail produces real memos.

**Exit-code alerting is structurally blind here.** `run_memo.py` has THREE silent exit-0
paths -- WARN/network-weather `:168-182`, `_embedding_preflight` `:306-309`,
`--preflight-only` `:313-316` -- and `handoff/autoresearch/root_cause.md:128-141` records a
real multi-night window where the embedding soft-skip produced NO memo and NO alert. The
2025-2026 dead-man's-switch literature says invert it: assert a **success artifact dated
today exists**, don't wait for a non-zero rc.

**Why:** these are the traps a naive 82.11 fix walks into -- re-implementing a notifier,
counting the wrong N, claiming the rail move as a code deliverable when it is an operator
`.env` line, and gating on rc.

**How to apply:** reuse `raise_cron_alert_sync`
(`backend/services/observability/alerting.py:253`) at severity **P1** (a P2 is logged and
DROPPED while `slack_webhook_url` is empty, `:211-224`), and copy the phase-82.10 module +
test shape wholesale -- see [[freshness-alarm-browser-driven-82-10]] and
[[qa-guards-stop-one-seam-short]]. Derive N from the ERROR DIRECTORY (the criterion's
"fixture directory of prior ERROR files" makes a JSON-only implementation undrivable).
