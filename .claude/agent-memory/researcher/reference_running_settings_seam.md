---
name: unmeasurable-running-flags-86-60
description: backend/.env is permission-denied to agents; GET /api/settings/ is the running-process seam but exposes only 45 keys, so most feature flags are NOT observable live
metadata:
  type: reference
---

**`backend/.env` is permission-denied to this agent** — BOTH the `Read` tool ("File is in a
directory that is denied by your permission settings") and a `grep` via `Bash` are refused. Do not
plan a measurement around reading it; route through the API instead.

**The running-process seam is `curl -s http://127.0.0.1:8000/api/settings/`.** It returns the
`FullSettings` model built by `_settings_to_full` (`backend/api/settings_api.py:350`, served at
`:406-407`) — values from the LIVE process, which is what "committed is NOT in force" requires.

**But it exposes only 45 keys**, and measured 2026-08-17 **none of the eight overlay feature flags
is among them** (`ma_preannounce_enabled`, `peer_leadlag_enabled`, `social_velocity_enabled`,
`call_transcript_gpr_enabled`, `analyst_narrative_enabled`, `insider_signal_screen_enabled`,
`options_flow_screen_enabled`, `analyst_revisions_enabled`). It DOES expose
`news_screen_enabled`, `pead_signal_enabled`, `paper_screen_top_n`, `paper_use_claude_code_route`,
`lite_mode`, `meta_scorer_enabled`, `macro_regime_filter_enabled`, `sector_calendars_enabled`.

**Why this matters:** a step that asks "which flags are ON in the running process" is
**partially unanswerable**, and the honest report is UNMEASURABLE, not an estimate from the file
defaults. The `getattr(settings, "<flag>", False)` idiom means absent ⇒ OFF, which bounds the
answer without settling it.

**How to apply:** pair the curl with `ps -eo pid,lstart,etime,command | grep uvicorn` to pin the
pid and start time, so the reading is attributable to a specific process instance. If a step needs
a flag the endpoint omits, say so and propose adding it to `FullSettings` rather than inferring.
Related: [[slice-vs-entry-path-86-60]].
