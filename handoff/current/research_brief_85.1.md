# Research Brief -- step id UNSPECIFIED (caller template not filled) -> covers **phase-85.1**

Tier: **moderate** (caller-stated). Audit-class: **false**. Write-first: this file
was created before any source was read and grown incrementally.

## Step-derivation note (read this first)

The spawn prompt carried the literal placeholders `UNSPECIFIED`, `(read the step
from .claude/masterplan.json and derive the objective)` and `(derive from the step
description)` -- the workflow launch was not parameterised with a step id. Per the
prompt's own instruction I derived the objective. Evidence for the choice:

- `handoff/harness_log.md` (tail) records 4000.2 closed and states "**Next:**
  4000.3 -- OPERATOR-GATED live window".
- **4000.3 is already claimed by a concurrent researcher**:
  `handoff/current/research_brief_4000.3.md` mtime `2026-08-06 16:15:58`, two
  minutes before this session's first tool call, carrying `## STATUS: IN PROGRESS
  (internal half landed; external reads next)`. Re-running it would duplicate
  spend and risk clobbering a peer's in-flight file.
- The operator chose "Hold for now" on the P2 tail 4000.6-4000.9
  (`handoff/harness_log.md`), so those are out. 4000.4/4000.5 branch on a window
  that has not run.

**Selected: phase-85.1** -- "Take the nightly autoresearch job off the dead metered
rail -- decide between migrating it to the Max-subscription rail and retiring it."
Pending, P1, `harness_required: true`, no research brief anywhere in
`handoff/current/` or `handoff/archive/`, unclaimed by any peer, failing in
production, and directly continuous with phase-4000 (which is proving the same Max
rail carries this project's Claude traffic).

Written to the caller-specified path so no peer file is touched. Main should copy
to `research_brief_85.1.md` if it accepts this derivation.

---

## HEADLINE -- TWO OF 85.1's PREMISES ARE REFUTED BY MEASUREMENT

85.1 instructs: *"Whether gpt-researcher can be driven through that rail is the
central feasibility question this step must answer with a measurement, not an
opinion"* and *"Do not assume the migration was done; measure it."* I measured it.

**R1. The migration is implemented, committed, and defaulted ON as of today.**
`scripts/autoresearch/run_nightly.sh:100` reads
`if [ "${AUTORESEARCH_USE_MAX_RAIL:-1}" = "1" ]; then` -- default **1 (ON)**. On
that path (`:101-108`) it health-probes the bridge and then exports
`ANTHROPIC_API_URL`, `ANTHROPIC_BASE_URL` = `http://127.0.0.1:18797` plus
`ANTHROPIC_API_KEY="max-rail-dummy-key"`; if the bridge is down it pages and
`exit 78` rather than falling back to the metered API (`:109-113`).

**R2. 85.1's "NOTE FOR THE EXECUTOR" is itself stale.** It claims the 82.11 commit
subject *"move autoresearch off the metered rail and make its failures audible"*
**"OVERSTATES what shipped -- only the audibility half landed."** Measured: commit
`816378e6` (2026-08-06 08:36:27 +0200) contains BOTH halves, and says so in its own
body: *"run_nightly.sh's AUTORESEARCH_USE_MAX_RAIL default flips 0 -> 1, routing
every call through the live Max-rail bridge at $0 metered."* The flag itself was
introduced 2026-07-24 in `33d2ca1b` (76.9.2 WIP) defaulting to 0. Working tree is
clean for that file -- the flip is committed, not a local edit.

**R3. 85.1's newest evidence predates the fix by 6h36m.** The step cites
`handoff/autoresearch/2026-08-06-ERROR-topic08.md` ("credit balance is too low").
Measured mtimes: ERROR file `2026-08-06 02:00:12`; flip commit `2026-08-06
08:36:27`. The 02:00 launchd run fired **before** the flip. It is evidence about
the pre-flip world.

**R4. But the rail has never actually run under launchd.** `grep max-rail
handoff/autoresearch.log` returns exactly ONE hit, at `2026-07-25T01:16:47` -- a
manual 76.9.5 attempt. Tonight's 02:00 run is the first scheduled execution on the
Max rail. So the migration is *shipped but unproven in the scheduled environment*.

**R5. The real blocker is an unmentioned sibling step.** `76.9.5` (`pending`):
*"nightly run_memo WEDGES after the research phase: process goes 0% CPU with no
in-flight request and never exits"* -- 5 attempts 2026-07-24/25, one success
(rc=0, 15,809-char memo, 4m40s), the rest wedged or timed out. 85.1 never mentions
76.9.5. An executor taking 85.1 at face value will re-implement an existing
migration and then hit a blocker the step does not name.

## The mechanism, verified end to end

`run_nightly.sh` exports `ANTHROPIC_API_URL`/`ANTHROPIC_BASE_URL` ->
`langchain_anthropic` **1.4.8** resolves the base URL from
`["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"]` (`chat_models.py:949`; `:956`
documents *"`ANTHROPIC_API_URL` and if that is not set, `ANTHROPIC_BASE_URL`"*) ->
`gpt_researcher/llm_provider/generic/base.py:107-111` builds
`ChatAnthropic(**kwargs)` for `provider == "anthropic"` and passes no `base_url`,
so the env var is the only lever -> bridge `127.0.0.1:18797`
(`scripts/ops/anthropic_max_bridge.py:56-58,179`) -> proxy `https://localhost:18796`
(`~/.openclaw/claude-code-proxy.js:9,201`) -> `claude -p --model <m>
--output-format json --max-turns 1` (`claude-code-proxy.js:129`) on the Max plan.

Live-measured this session: `curl -s -m 5 http://127.0.0.1:18797/health` ->
`{"ok":true,"proxy":"claude-code-cli"}` (rc=0). Both services are up:
`650 0 com.pyfinagent.anthropic-bridge`, `668 0 com.pyfinagent.claude-code-proxy`.

## Read in full (>=5 required; counts toward the gate) -- 7

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/headless | 2026-08-06 | official doc (Anthropic) | WebFetch | **RAIL-FATAL RISK:** "Set `ANTHROPIC_API_KEY` before running it, because bare mode doesn't use your subscription login"; "In bare mode, Claude Code never reads OAuth credentials or the system keychain"; and "`--bare` ... **will become the default for `-p` in a future release**." |
| https://code.claude.com/docs/en/routines | 2026-08-06 | official doc (Anthropic) | WebFetch | "Routines draw down subscription usage the same way interactive sessions do"; "API accounts aren't supported for routines"; without usage credits "additional runs are rejected until the window resets". |
| https://www.anthropic.com/legal/consumer-terms | 2026-08-06 | official legal (Anthropic) | WebFetch | **[QUALIFYING/ADVERSARIAL]** Prohibited: "Except when you are accessing our Services via an Anthropic API Key or where we otherwise explicitly permit it, to access the Services through automated or non-human means, whether through a bot, script, or otherwise." |
| https://support.claude.com/en/articles/11145838-using-claude-code-with-your-pro-or-max-plan | 2026-08-06 | official doc (Anthropic) | WebFetch | "all activity in both tools counts against the same usage limits"; "If you have an ANTHROPIC_API_KEY environment variable set ... Claude Code will use this API key ... resulting in API usage charges rather than using your subscription's included usage." |
| https://github.com/assafelovic/gpt-researcher/blob/master/docs/docs/gpt-researcher/llms/llms.md | 2026-08-06 | official project doc | WebFetch | `provider:model` syntax confirmed. Documents `OPENAI_BASE_URL` for custom endpoints but for Anthropic gives **only** `ANTHROPIC_API_KEY` + `FAST_LLM=anthropic:...` -- **no documented base-URL mechanism**. |
| https://docs.langchain.com/oss/python/integrations/chat/anthropic | 2026-08-06 | official doc (LangChain) | WebFetch (after 308 from python.langchain.com) | Public page documents only `ANTHROPIC_API_KEY`; custom base URL is **not surfaced in user-facing docs** -- it exists only in the installed source. |
| https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-06 | canonical book (year-less variant) | WebFetch | "When pages occur too frequently, employees second-guess, skim, or even ignore incoming alerts, sometimes even ignoring a 'real' page that's masked by the noise"; "Will I ever be able to ignore this alert, knowing it's benign?" |

## Identified but snippet-only (context; does NOT count toward the gate) -- 31

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://alternativeto.net/news/2026/2/anthropic-officially-bans-using-subscription-authentication-for-third-party-claude-use | news | superseded by the primary consumer-terms fetch |
| https://winbuzzer.com/2026/02/19/anthropic-bans-claude-subscription-oauth-in-third-party-apps-xcxwbn/ | news | secondary to the primary legal source |
| https://www.mindstudio.ai/blog/anthropic-openclaw-ban-oauth-authentication | blog | secondary |
| https://help.apiyi.com/en/anthropic-claude-subscription-third-party-tools-openclaw-policy-en.html | vendor blog | low tier |
| https://github.com/anthropics/claude-code/issues/37205 | issue tracker | community tier; feature request only |
| https://autonomee.ai/blog/claude-code-terms-of-service-explained/ | blog | secondary to consumer-terms |
| https://lalatenduswain.medium.com/claude-code-on-claude-max-plan-understanding-oauth-token-vs-api-key-authentication-in-2026-96a6213d2cde | blog | community tier |
| https://www.theregister.com/ai-ml/2026/05/14/anthropic-tosses-agents-into-the-api-billing-pool/5240748 | press | the May-2026 change was cancelled; see recency scan |
| https://www.truefoundry.com/blog/claude-code-limits-explained | vendor blog | superseded by official limits doc |
| https://www.finout.io/blog/claude-code-pricing-2026 | vendor blog | pricing recap only |
| https://codersera.com/blog/anthropic-june-2026-billing-change-claude-code/ | blog | secondary |
| https://tech-insider.org/ie/claude-code-agent-pricing-split-2026/ | blog | secondary |
| https://www.developersdigest.tech/blog/claude-code-usage-limits-playbook-2026 | blog | secondary |
| https://pasqualepillitteri.it/en/news/851/claude-code-routines-cloud-automation-guide | blog | superseded by official routines doc |
| https://docs.gptr.dev/docs/gpt-researcher/llms | official doc | **fetch attempted, HTTP 404**; used the GitHub-hosted source of the same doc instead |
| https://docs.litellm.ai/docs/tutorials/claude_non_anthropic_models | official doc (LiteLLM) | adjacent shim pattern; not needed once the repo's own bridge was found |
| https://github.com/Mintplex-Labs/anything-llm/issues/5234 | issue tracker | corroborates that Anthropic base-URL support is a commonly-missing feature |
| https://www.morphllm.com/use-different-llm-claude-code | vendor blog | inverse direction (other models INTO Claude Code) |
| https://imfing.com/til/use-custom-llm-providers-in-claude-code/ | blog | same inverse direction |
| https://open-code.ai/en/docs/providers | competitor doc | out of scope |
| https://glama.ai/mcp/servers/@brendancopley/... | config sample | community tier |
| https://deadmanssnitch.com/ | vendor | SaaS heartbeat -- explicitly barred by the local-only/$0 constraint |
| https://github.com/Kriss-V/deadmancheck | repo | "alerts when jobs run but do nothing" -- pattern noted, not adopted |
| https://onlineornot.com/cron-job-monitoring-guide | vendor guide | pattern already captured by the SRE chapter |
| https://updog.watch/learn/what-is-dead-mans-switch | vendor | ditto |
| https://deadmanping.com/blog/cron-job-failed | vendor | ditto |
| https://oneuptime.com/blog/post/2026-03-02-how-to-monitor-cron-job-execution-and-alerting-on-ubuntu/view | vendor | ditto |
| https://dev-brains-ai.com/blog/cron-job-monitoring-and-alerting-guide | blog | ditto |
| https://nurbak.com/en/blog/dead-mans-switch/ | blog | ditto |
| https://cronradar.com/comparisons/cron-monitoring-best-practices | vendor | ditto |
| https://medium.com/@erwindev/building-a-dead-mans-switch-for-critical-cron-jobs-946898887e98 | blog | community tier |

## Search-query composition (three-variant discipline)

1. **Current-year frontier (2026)** -- "Claude Code subscription rail scheduled
   unattended job Anthropic usage policy 2026".
2. **Last-2-year window (2025)** -- "Anthropic Claude Code OAuth subscription token
   programmatic proxy 2025 terms restrictions".
3. **Year-less canonical** -- "gpt-researcher custom LLM provider base_url
   ANTHROPIC_BASE_URL configuration" and "cron job silent failure monitoring dead
   man's switch retire versus fix chronically failing scheduled job".

## Recency scan (2024-2026)

Performed. **Four findings in the window, all material to this step:**

1. **2026-02-19/20 -- OAuth-in-third-party-tools ban.** Anthropic updated its legal
   terms to prohibit using Free/Pro/Max OAuth tokens in other products or services,
   including the Agent SDK. The same reporting states the Claude Code CLI itself
   remains an intended, official way to use Claude programmatically. **This
   matters to 85.1 because the repo's rail does NOT extract or present an OAuth
   token -- it shells out to the official `claude` CLI, which authenticates
   itself.** That is the allowed side of the line, but see Pitfall P2.
2. **2026-05-14 -> cancelled 2026-06-15.** Anthropic announced moving programmatic
   Claude usage (Agent SDK, `claude -p`, third-party apps) onto separate metered
   credits, then **cancelled it before it took effect**. So the flat-fee `claude -p`
   rail is a *paused policy*, not a permanent guarantee -- consistent with the
   phase-4000.1 finding already in agent memory.
3. **2026-04-14 -- Claude Code Routines** (research preview): a first-party
   scheduler that draws down subscription usage, caps daily runs, and rejects
   over-limit runs when usage credits are off. Phase-85's parent step already
   records the DO-NOT-ADOPT-YET verdict; nothing here changes it.
4. **`--bare` mode** (current headless doc): bare mode "doesn't use your
   subscription login" and "never reads OAuth credentials or the system keychain",
   and is slated to **become the default for `-p`**. This supersedes the assumption
   baked into `claude_code_client.py`'s module docstring and into the proxy.

No 2024-2026 source contradicts the SRE alert-fatigue canon; it complements it.

## Key findings

1. **The migration exists and is ON.** `run_nightly.sh:100` default is `1`
   (`816378e6`, 2026-08-06 08:36:27). 85.1's "measure whether it was done" answer is
   **yes**.
2. **The rail has never run under launchd.** Only one `max-rail ON` line exists in
   `handoff/autoresearch.log`, dated `2026-07-25T01:16:47` (manual). Tonight is the
   first scheduled attempt.
3. **The env-var mechanism is real but undocumented at the gpt-researcher layer.**
   `chat_models.py:949` honours `ANTHROPIC_API_URL`/`ANTHROPIC_BASE_URL`; the
   gpt-researcher docs document `OPENAI_BASE_URL` but nothing equivalent for
   Anthropic. The rail depends on a LangChain implementation detail, not a
   contracted API.
4. **`--bare` is a scheduled time bomb.** "bare mode doesn't use your subscription
   login" + "will become the default for `-p` in a future release"
   (code.claude.com/docs/en/headless). The proxy invokes `-p` WITHOUT `--bare`
   (`claude-code-proxy.js:129`) and the nightly job passes a **dummy** API key -- so
   on the day `--bare` becomes the default, every nightly call fails auth.
5. **Retiring the job is the weaker option.** SRE canon says the failure mode to fix
   is the noise, not the job: "When pages occur too frequently, employees
   second-guess, skim, or even ignore incoming alerts" (sre.google). The job's
   output is non-critical-path, but 13 consecutive fails
   (`autoresearch_fail_state.json`) is already at the page-after-3 threshold
   (`run_nightly.sh` `PAGE_AFTER_N` default 3).
6. **`consecutive_fails: 13` will not self-clear on a partial success.** It resets
   only on the `rc=0` branch of `run_nightly.sh`. If tonight wedges (76.9.5), the
   counter climbs to 14 and keeps paging.

## Internal code inventory

| File | Lines / anchor | Role | Status |
|---|---|---|---|
| `scripts/autoresearch/run_nightly.sh` | `:100` gate, `:101-108` exports, `:109-113` loud-fail, `:57-79` `_record_fail_and_page` | launchd entry; owns the rail decision | **Migration already landed; default ON** |
| `scripts/autoresearch/run_memo.py` | `:308-310` `anthropic:{...}`, `:311` embedding, `:318` retrievers, `:323-324` key guard | the job itself | Unchanged; still provider-tagged `anthropic:` (correct -- the base URL does the routing) |
| `scripts/ops/anthropic_max_bridge.py` | `:56-58`, `:60-62`, `:108-136`, `:179` | SSE->JSON aggregating Anthropic-shaped adapter | Running (pid 650), `/health` OK |
| `~/.openclaw/claude-code-proxy.js` | `:9`, `:129`, `:201` | `claude -p` subprocess shim | Running (pid 668); **outside the repo, unversioned** |
| `backend/agents/claude_code_client.py` | `:1-25` docstring, `:373-374`, `:401-412` key scrub, `:494` health probe | Layer-2 CC rail client | Independent of autoresearch; shares the `--bare` exposure |
| `.venv/.../langchain_anthropic/chat_models.py` | `:949`, `:956` | base-URL resolution | v1.4.8; the load-bearing detail |
| `.venv/.../gpt_researcher/llm_provider/generic/base.py` | `:107-111` | provider dispatch | Passes no `base_url`; env var is the only lever |
| `handoff/away_ops/autoresearch_fail_state.json` | whole file | fail counter | `{"consecutive_fails": 13}` |
| `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` | -- | 02:00 schedule | `launchctl list` last exit **1** |
| `handoff/autoresearch/` | 65 memos, 62 `-ERROR-` | outcome record | newest ERROR predates the flip |

## Consensus vs debate (external)

**Consensus:** the Claude Code CLI is a sanctioned programmatic surface (headless
doc shows `package.json` scripts, CI, GitHub Actions); subscription and CLI usage
share one pool; setting `ANTHROPIC_API_KEY` silently moves you to metered billing.

**Debate / unresolved tension:** the consumer terms prohibit "access[ing] the
Services through automated or non-human means, whether through a bot, script, or
otherwise" *except* via an API key "or where we otherwise explicitly permit it".
Anthropic's own docs explicitly permit scripted `claude -p` and ship a first-party
scheduler (Routines) on subscription billing -- which reads as exactly that
permission. But a local shim that terminates TLS with a **self-signed cert whose CN
is `api.anthropic.com`** (`anthropic_max_bridge.py:20-24`) and relays a third-party
library's traffic sits further from the documented pattern than a direct CLI call.
**I am not rendering a compliance verdict** -- this is flagged for the operator.

## Pitfalls (from literature + measurement)

- **P1 -- `--bare` default flip kills the rail silently.** Mitigation: pin/assert the
  invocation in the proxy, or add a nightly auth-shape assertion. Worth its own step.
- **P2 -- ToS ambiguity (above).** Operator-visible note, not an agent decision.
- **P3 -- Re-implementing shipped work.** 85.1's criteria say "If the outcome was
  MIGRATION ... a real end-to-end run produced at least one memo whose filename does
  NOT contain ERROR". That criterion is **still open** even though the migration
  landed -- so the step is closable by *verification*, not by new implementation.
- **P4 -- Don't conflate the two `autoresearch`es.** `scripts/autoresearch/run_memo.py`
  (nightly memo) is unrelated to the `backend/autoresearch/` rotation package.
- **P5 -- A green launchd exit is not a success.** Cf. the routines doc's own warning:
  "A green status ... does not mean the task in your prompt succeeded." The same
  applies here -- assert on a non-ERROR memo filename, not on `rc`.
- **P6 -- No SaaS heartbeat.** The dead-man's-switch literature is dominated by paid
  SaaS; the local-only/$0 constraint means import the *inversion* (alert on absence
  of a success artifact), not the architecture. That work is already queued as 82.49.

## Application to pyfinagent

1. **Re-scope 85.1 from "decide and migrate" to "verify the shipped migration."**
   The contract should state R1-R4 with the anchors above so the executor does not
   rebuild `run_nightly.sh:100-113`.
2. **The closing evidence is one scheduled run.** Tomorrow's 02:00 launchd run is
   the cheapest possible proof: a non-ERROR filename in `handoff/autoresearch/` plus
   a `max-rail ON` line dated that morning in `handoff/autoresearch.log`, plus
   `autoresearch_fail_state.json` back to `{"consecutive_fails": 0}`.
3. **Declare the 76.9.5 dependency explicitly.** If tonight wedges, 85.1 cannot
   close on outcome (a); the honest fall-back is outcome (c) DISABLE, and 76.9.5
   becomes the blocking step. The contract should pre-state that branch rather than
   discovering it.
4. **The test criterion is satisfiable without a paid call.** Per 85.1's own
   criterion 2, import the config-building code and assert the resolved base URL is
   the loopback bridge -- e.g. exercise `run_nightly.sh`'s export block in a
   subshell with a stub `/health`, then assert `ANTHROPIC_BASE_URL` is not
   `api.anthropic.com`. Guard against the vacuity trap: mutate the default to `0`
   and prove the test goes red.
5. **File the `--bare` exposure as its own step** (P1). It threatens phase-4000's
   rail as well as this job, and per the standing rule a discovered defect gets its
   own research-gated step rather than a prose disclosure.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**)
- [x] 10+ unique URLs total incl. snippet-only (**38**)
- [x] Recency scan (last 2 years) performed + reported (4 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (18 files/artifacts)
- [x] Contradictions / consensus noted (ToS tension recorded, no verdict rendered)
- [x] All claims cited per-claim
- Gap disclosed: I did **not** execute a live `POST /v1/messages` through the
  bridge. `/health` was probed ($0); a real POST would spend Max-rail budget and
  the scheduled 02:00 run is the cheaper, more representative proof.
- Gap disclosed: `~/.openclaw/claude-code-proxy.js` is outside the repo and
  unversioned; its behaviour could change without a git signal.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 31,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_UNSPECIFIED.md",
  "gate_passed": true
}
```
