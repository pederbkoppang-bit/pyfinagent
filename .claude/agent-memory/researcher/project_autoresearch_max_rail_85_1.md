---
name: autoresearch-max-rail-85-1
description: 85.1's premises REFUTED -- the Max-rail migration already shipped (run_nightly.sh default 1, commit 816378e6 08:36) and the ERROR file it cites is from 02:00 the SAME morning; rail never ran under launchd; `--bare` becoming the -p default is a scheduled kill for the whole shim chain
metadata:
  type: project
---

Researched 2026-08-06 (step spawned with an unfilled `UNSPECIFIED` template;
objective derived from masterplan + harness_log).

**The class this step teaches: a masterplan step's own "MEASURED EVIDENCE"
can be stale by HOURS, not months.** 85.1 cited
`handoff/autoresearch/2026-08-06-ERROR-topic08.md` as proof the job still rides
the metered rail. Measured: ERROR mtime `02:00:12`, fix commit `816378e6`
`08:36:27` the SAME DAY. Always diff artifact mtime against
`git log --date=iso` on the file the step blames before accepting the premise.
Same-day staleness is invisible if you only compare dates.

**What is actually true (re-derive line numbers; they move):**
- `scripts/autoresearch/run_nightly.sh:100` -- `${AUTORESEARCH_USE_MAX_RAIL:-1}`,
  default **ON**. `:101-108` exports `ANTHROPIC_API_URL` + `ANTHROPIC_BASE_URL`
  = `http://127.0.0.1:18797` + a dummy key; `:109-113` pages and `exit 78`
  rather than falling back to metered. Flag born 2026-07-24 (`33d2ca1b`,
  default 0), flipped 2026-08-06 (`816378e6`).
- 85.1's "NOTE FOR THE EXECUTOR" says the 82.11 commit subject overstates what
  shipped ("only the audibility half landed"). **FALSE** -- `816378e6`'s own
  body describes the rail flip. Two refuted premises in one step.
- **The rail has NEVER run under launchd.** `grep max-rail handoff/autoresearch.log`
  -> exactly one hit, `2026-07-25T01:16:47`, a manual 76.9.5 attempt.
  Shipped != exercised.
- `76.9.5` (pending, unmentioned by 85.1) is the real blocker: run_memo WEDGES
  after the research phase at 0% CPU.

**The mechanism (undocumented at the top layer -- fragile to upgrades):**
`langchain_anthropic` 1.4.8 `chat_models.py:949` resolves the base URL from
`["ANTHROPIC_API_URL","ANTHROPIC_BASE_URL"]`;
`gpt_researcher/llm_provider/generic/base.py:107-111` builds
`ChatAnthropic(**kwargs)` with NO `base_url`, so the env var is the only lever.
gpt-researcher's own docs document `OPENAI_BASE_URL` and give Anthropic only an
API key -- there is NO contracted Anthropic base-URL feature. A gptr/langchain
bump can silently kill the rail.

**Chain:** run_nightly -> bridge `scripts/ops/anthropic_max_bridge.py` (:56-58,
:179, pid 650) -> `~/.openclaw/claude-code-proxy.js` (:129, pid 668, UNVERSIONED,
outside the repo) -> `claude -p`. `curl 127.0.0.1:18797/health` ->
`{"ok":true,"proxy":"claude-code-cli"}`.

**`--bare` is a scheduled kill for this AND phase-4000.** code.claude.com/docs/en/headless:
"bare mode doesn't use your subscription login", "never reads OAuth credentials
or the system keychain", and "`--bare` ... will become the default for `-p` in a
future release". The proxy invokes `-p` WITHOUT `--bare` and the job passes a
DUMMY key -- the day the default flips, every nightly call 401s.

**ToS tension, flagged not adjudicated.** anthropic.com/legal/consumer-terms
prohibits "access[ing] the Services through automated or non-human means,
whether through a bot, script, or otherwise" EXCEPT via an API key "or where we
otherwise explicitly permit it" -- and Anthropic explicitly documents scripted
`claude -p` + ships Routines on subscription billing. The repo presents no OAuth
token (the CLI authenticates itself), which is the allowed side; but the proxy's
self-signed cert has **CN=api.anthropic.com**. Operator decision, never an
agent's.

**How to apply:** if asked to "migrate autoresearch off the metered rail", the
migration is DONE -- the step closes by VERIFYING one scheduled 02:00 run
(non-ERROR memo filename + a dated `max-rail ON` log line +
`autoresearch_fail_state.json` back to 0), not by rebuilding it. `consecutive_fails`
was 13 and resets only on the rc=0 branch. Do not conflate
`scripts/autoresearch/run_memo.py` with the `backend/autoresearch/` package.
