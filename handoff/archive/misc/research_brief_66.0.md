# Research Brief — phase-66.0 "Recovery re-baseline" (goal-phase66-reactivation)

Tier: SIMPLE (caller-specified). Date: 2026-07-06. Status: COMPLETE.

Question: what must a post-outage recovery re-baseline contain (state reconciliation, alert-backlog
disposition, verify-before-reenable), why did one Claude OAuth credential hard-expire ~5 days into an
unattended window, and what does "clean tree => non-recovery prompt" mean mechanically in
`scripts/away_ops/run_away_session.sh`?

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/authentication | 2026-07-06 | official doc | WebFetch, full page | macOS creds live in "the encrypted macOS Keychain"; for "CI pipelines, scripts, or other environments where interactive browser login isn't available, generate a one-year OAuth token with `claude setup-token`" -> `CLAUDE_CODE_OAUTH_TOKEN` (auth-precedence slot 5, ABOVE subscription `/login` creds at slot 6). NO refresh-token lifetime is published anywhere on the page. |
| https://github.com/anthropics/claude-code/issues/61912 | 2026-07-06 | bug report w/ repro (area:auth) | WebFetch, full page | Refresh fires during transient upstream failure -> unverified/bad token PERSISTED to credential store -> "the next `claude` session reads it and the very first API request returns 401 before the user even finishes typing"; no self-recovery until interactive `/login`. Proposed fixes: don't persist unverified tokens; clear on authentication_error; boot-time credential sanity ping. Workaround: monitor credential mtime staleness. |
| https://sre.google/resources/practices-and-processes/incident-management-guide/ | 2026-07-06 | official (Google SRE) | WebFetch, full page | Post-resolution: "a write-up of the incident is immediately started" covering "detection, mitigation, coordination, or communication", blameless, shared broadly. Honest gap: the guide is thin on mechanical return-to-service steps (practitioner sources below fill that). |
| https://sre.google/sre-book/distributed-periodic-scheduling/ | 2026-07-06 | book chapter (canonical cron) | WebFetch, full page | "Cron job owners can (and should!) monitor their cron jobs... or set up independent monitoring of the effect of cron jobs"; "we favor skipping launches rather than risking double launches". Fail-closed skip is correct — but skip WITHOUT independent paging is fail-silent. |
| https://cronitor.io/docs/heartbeat-monitoring | 2026-07-06 | vendor doc | WebFetch, full page | Dead-man's switch: "When heartbeats stop arriving or report failures an alert is sent" — alerts on ABSENCE of success; explicitly catches scheduler death, host down, and credential/authentication expiration; grace periods + consecutive-miss tolerance. |
| https://web-alert.io/blog/post-incident-monitoring-recovery-checklist | 2026-07-06 | practitioner blog | WebFetch, full page | "Run the exact check that detected the original incident — confirm it passes"; "Check queue backlogs — jobs queued during the incident may cause a processing surge"; "Confirm the service is healthy through an entire cycle before declaring full recovery"; "Every incident should improve your monitoring". |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/anthropics/claude-code/issues/54443 | bug report | Title carries the finding (concurrent sessions + refresh 400 after early 401 -> forced /login); corroborates candidate (b) below; simple-tier budget |
| https://github.com/anthropics/claude-code/issues/33811 | bug report | "401, no recovery path — login/logout all fail" — corroborates the no-self-heal observation |
| https://github.com/anthropics/claude-code/issues/44930 | bug report | same class as 33811 (401 persists 34+ hours, no browser flow) |
| https://github.com/anthropics/claude-code/issues/12447 | bug report | "OAuth token expiration disrupts autonomous workflows" — the generic class |
| https://github.com/anthropics/claude-code/issues/19078, /48079, /65036, /36911, /3591 | bug reports | same failure family; diminishing returns |
| https://platform.claude.com/docs/en/manage-claude/authentication | official doc | Claude-app (not Code) auth; Code doc above is the operative one |
| https://daveswift.com/claude-oauth-update/ | blog | search summary sufficed (access tokens ~short-lived; setup-token workaround) |
| https://rootly.com/sre/2025-sre-incident-management-best-practices-checklist | industry | corroborates web-alert checklist |
| https://arxiv.org/pdf/2502.06994 (SyncMind) | peer-reviewed preprint | agent out-of-sync recovery benchmark — adjacent, noted for recency scan; not load-bearing for a mechanical re-baseline |
| https://incident.io/blog/5-best-ai-powered-incident-management-platforms-2026 | industry | recency-scan datapoint only |
| https://sre.google/sre-book/practical-alerting/, /monitoring-distributed-systems/, /workbook/alerting-on-slos/ | book chapters | cron chapter was the on-point canonical read |
| ~35 further unique hits (watchflow, cronping, onlineornot, drumbeats, betterstack, honeybadger, smartscope, markaicode, itsm-docs, techtarget x2, cyber.gc.ca, oneuptime x2, Azure SRE Agent x2, remoteopenclaw, lobsterfarm, etc.) | vendor/community | unanimous corroboration of the dead-man's-switch pattern and 401-fix folklore; no unique claims |

## Search queries run (three variants per topic)

Topic 1 (OAuth): "Claude Code OAuth token expired refresh token lifetime" (year-less) / "Claude Code OAuth credential expiration 401 unattended 2026" / "Anthropic Claude Code authentication token expiry headless 2025".
Topic 2 (recovery): "return to service checklist after outage state reconciliation incident recovery" (year-less) / "post-incident recovery re-baseline autonomous agents cron fleet 2026" / "SRE incident response recovery verify before re-enable alert backlog 2025".
Topic 3 (fail-safe cron): "cron job silent failure dead man's switch monitoring absence of success" (year-less) / "scheduled job fail-safe alerting design Google SRE cron 2026" / "cron monitoring heartbeat alert when job stops running 2025".

## Recency scan (2024-2026)

Performed (the 2025/2026-suffixed variants above). Findings: (1) the entire Claude Code OAuth
failure corpus IS the 2025-2026 window — #61912 and #54443 are 2026 issues on current builds
(2.1.x), i.e. the transient-error->persistent-401 defect class is CURRENT, not historical; the
`setup-token` one-year path is the current official doc (fetched 2026-07). (2) 2025-2026 adds an
agent-native recovery literature (SyncMind arXiv:2502.06994 on out-of-sync agent recovery; Azure
SRE Agent "alert to verified recovery" posts, 2026) — these complement but do not supersede the SRE
cron/incident canon. (3) The dead-man's-switch pattern is unchanged 2024-2026; vendors have merely
proliferated. No finding invalidates the older canonical sources.

## Key findings

1. **Best-evidenced root-cause candidate (a): transient-failure credential corruption.** Issue
   #61912 documents refresh firing during a transient upstream failure window, persisting an
   unverified token, after which "the very first API request returns 401 before the user even
   finishes typing" in every later session, with NO self-recovery until interactive `/login`
   (Source: github.com/anthropics/claude-code/issues/61912, accessed 2026-07-06). This matches the
   observed signature exactly: network-error prefix (ECONNRESET 06-15..19) -> hard 401 from 06-20 ->
   16 days with zero self-heal across 34 headless sessions.
2. **Candidate (b): concurrent-client refresh race.** #54443 (snippet): refresh returns 400 after an
   early 401; concurrent sessions forced to /login. This machine ran cc_rail + 2 daily `claude -p`
   sessions off one credential store, so a rotation race is plausible as a compounding factor.
3. **Candidate (c) — least evidenced: fixed refresh-token lifetime.** The official authentication
   page publishes NO refresh-token expiry window (Source: code.claude.com/docs/en/authentication,
   accessed 2026-07-06). A hard ~5-day inactivity expiry is therefore conjecture; the 66.0
   root-cause note should say "consistent with the #61912 corruption class", not assert a lifetime.
4. **Official unattended path exists and outranks /login creds:** `claude setup-token` mints a
   one-year, inference-scoped OAuth token consumed via `CLAUDE_CODE_OAUTH_TOKEN` (auth-precedence 5
   of 6, above subscription OAuth) — "for CI pipelines, scripts, or other environments where
   interactive browser login isn't available" (Source: code.claude.com/docs/en/authentication).
   This is the structural fix candidate for 66.4.
5. **Fail-safe canon: alert on absence, independently.** "Cron job owners can (and should!) monitor
   their cron jobs... or set up independent monitoring of the effect of cron jobs" (Source:
   sre.google/sre-book/distributed-periodic-scheduling/); dead-man's-switch monitoring "alerts when
   heartbeats stop arriving" and explicitly catches credential expiration and scheduler death
   (Source: cronitor.io/docs/heartbeat-monitoring). The away stack's paging lived INSIDE the dying
   sessions — the textbook fail-silent anti-pattern; the independent watchdog (healthcheck.sh) ran
   all 21 days but has no "last successful away session" gate.
6. **Return-to-service checklist items that map 1:1 to 66.0:** re-run the exact failed check;
   reconcile state; dispose queued backlog ("jobs queued during the incident may cause a processing
   surge"); verify through one entire healthy cycle before declaring recovery; convert the incident
   into a permanent monitoring improvement (Source: web-alert.io post-incident checklist; postmortem
   discipline per sre.google incident-management guide).
7. **Skip > double-launch:** "we favor skipping launches rather than risking double launches"
   (Source: sre.google/sre-book/distributed-periodic-scheduling/) — do NOT force-fire make-up away
   sessions for the 34 missed ones; let the calendar fire next at 05:30Z and observe.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/away_ops/run_away_session.sh` | 96-102, 119-126 | prompt selection + preflight-test mode | read in full (167 lines) |
| `handoff/away_ops/pending_tokens.json` | 1-100 | canonical open-operator-asks file | read in full |
| `scripts/away_ops/healthcheck.sh` | 1-189 | watchdog observer, frontend-only restart | read in full |
| `scripts/away_ops/sentinel.sh` | 83-123 | budget + flag-vs-token gates | read in full |
| `backend/slack_bot/scheduler.py` | 499-503 | ONLY programmatic parser of pending_tokens.json | grepped w/ context |
| `.claude/hooks/pre-tool-use-danger.sh` | 194, 244 | mentions path in guidance strings only (no parse) | grepped |
| `backend/slack_bot/commands.py` | 137 | mentions path in a message string only | grepped |
| `scripts/autoresearch/run_memo.py` | 166 | mentions path in help text only | grepped |

### A. Prompt-selection condition (run_away_session.sh)

Exact code, lines 96-102 (runs only when `PROMPT_KIND` is still `am`/`pm`, i.e. not already
HALT-degraded or sentinel-downgraded to `digest_only`):

```bash
if [ "$PROMPT_KIND" = "am" ] || [ "$PROMPT_KIND" = "pm" ]; then
    dirty=$(git status --porcelain 2>/dev/null | grep -vE '^.. (handoff/audit/|handoff/away_ops/|handoff/logs/)')
    if [ -n "$dirty" ]; then
        slog "dirty tree detected (non-evidence paths) -- recovery prompt selected"
        PROMPT_KIND="recovery"
    fi
fi
```

- **"Clean" means:** `git status --porcelain` output empty AFTER excluding exactly three path
  prefixes: `handoff/audit/`, `handoff/away_ops/`, `handoff/logs/` (comment at :92-95 explains
  these are perpetually dirty by design). The clean-tree branch is simply *falling through* with
  `PROMPT_KIND` unchanged (`am`/`pm`) to `PROMPT_FILE=".../prompt_${PROMPT_KIND}.md"` (:112).
- **NOT excluded (will trip recovery):** handoff-ROOT files appended by trading cycles and hooks —
  `handoff/.cycle_heartbeat.json`, `handoff/cycle_history.jsonl`, `handoff/kill_switch_audit.jsonl`,
  `handoff/prompt_leak_redteam_audit.jsonl` — all four were `M` at this session's start (git status
  snapshot 2026-07-06) and are re-appended by every trading cycle (weekdays 14:00 UTC) and
  kill-switch/hook event. Also NOT excluded: `handoff/current/` (this brief shows as `??` right now
  and would count as real WIP).
- **Built-in exit evidence:** `AWAY_SESSION_TEST_PREFLIGHT=1 bash scripts/away_ops/run_away_session.sh am`
  exercises HALT-DEV -> sentinel -> dirty-tree -> prompt selection with no git/claude side effects and
  echoes `PREFLIGHT_PROMPT=<kind>` (:122-126). NOTE: `AWAY_SESSION_DRY_RUN=1` is the WRONG tool here —
  it skips the entire preflight block including the dirty check (:80).
- **Ordering caveat:** sentinel runs BEFORE the dirty check (:82-90); a sentinel failure yields
  `PREFLIGHT_PROMPT=digest_only` even on a clean tree. Criterion-1 evidence must show `am` (or `pm`),
  which additionally proves sentinel passed.

### B. pending_tokens.json schema + disposition fields

- Top-level: `updated` (ISO ts), `format_note` (str), `asks` (array). Per-ask keys: `id`,
  `raised_by`, `raised_at`, `due`, `ask`, `reply_options` (exact strings), `recommended`.
- **8 asks confirmed** (matches the known list): `METERED-BREACH-RECURRING` (:6), `TEST-TOKEN-62.2`
  (:19), `FABLE-HEADLESS` (:30), `SDK-CREDIT` (:42), `MAS-PLIST-ZOMBIE` (:54), `WEBHOOK` (:65),
  `AUTORESEARCH-SPEND` (:77), `ENV-LINE-81` (:89). `updated` = 2026-06-19T05:55Z (pre-outage; stale
  by 17 days — re-disposition should bump it).
- **Schema constraints: effectively none for additive fields.** The only programmatic parser is
  `backend/slack_bot/scheduler.py:499-503` (digest builder): `_json.load(f).get("asks", [])` then
  per-key `.get(...)` access — extra per-ask keys (`disposition`, `resolved_at`, `resolution`) and
  extra top-level keys are ignored. The other three references (`pre-tool-use-danger.sh:194,244`,
  `commands.py:137`, `run_memo.py:166`) embed the file PATH in operator-facing strings only; the
  away prompts (`prompt_am/pm/recovery/digest_only.md`) instruct sessions to read it as prose.
  Keep `asks` an array and `raised_at` ISO-parseable (scheduler does `_dt.fromisoformat(...[:10])`).

### C. healthcheck.sh + sentinel.sh vs resolution annotations

- `healthcheck.sh` (read in full, 189 lines): never opens `pending_tokens.json`. Its `ok` verdict
  (:174-181) is service states + HTTP 200s + ADC + disk; P1 logic (:117-157) keys on
  `restart_failed` replay from `health.jsonl`. **Cannot misfire on annotations.**
- `sentinel.sh` `flags_match_tokens` gate (:83-123): compares `PAPER_*=true` lines in
  `backend/.env` against `scripts/away_ops/flag_baseline.json` (grandfathered/exempt) and raw text
  of `handoff/operator_tokens.jsonl`. It never reads `pending_tokens.json`; `flag_baseline` is a
  separate file. **Annotating pending_tokens.json is inert to both gates.** (Converse caution: any
  66.x step that flips a `PAPER_*` flag without a matching operator_tokens.jsonl line WILL trip
  `flags_match_tokens` -> digest-only.)
- Bonus: `pending_tokens.json` lives under `handoff/away_ops/` — one of the three excluded
  prefixes — so editing it does not even dirty the tree for prompt selection.

### D. What "clean git status" can honestly mean (criterion 1 bound)

Hook-appended audit streams re-dirty the tree DURING every session: live proof — after this
session's backlog sweep (commits 899d4a90..68909af1, `git log` confirms 68909af1 = phase-66 goal
install), `git status --porcelain` already shows ` M handoff/audit/pre_tool_use_audit.jsonl`
(appended by the PreToolUse hook on this very session's Bash calls; matches the excluded prefix)
plus `?? handoff/current/research_brief_66.0.md` (NOT excluded). The four handoff-root evidence
files WILL return as `M` at the next trading cycle / kill-switch event. So criterion 1 can only
honestly mean: **at a stated timestamp, no dirty paths outside the three excluded prefixes,
demonstrated by a `AWAY_SESSION_TEST_PREFLIGHT=1` run echoing `PREFLIGHT_PROMPT=am|pm`** — not "git
status is empty forever". PM sessions on trading days (20:00 UTC > 14:00 UTC cycle) will
legitimately re-enter recovery to sweep that day's trading evidence unless a mid-day commit swept
it first; that is the designed churn-sweep behavior, not a residual outage.

## Consensus vs debate

- **Consensus:** (i) monitor scheduled jobs on absence-of-success, via a monitor OUTSIDE the failing
  component (SRE ch.24 + every vendor/community source); (ii) subscription OAuth `/login` creds are
  the wrong credential for multi-day unattended headless use — use `setup-token` (1-year) or an API
  key (official doc + the whole 2026 issue corpus); (iii) fail-closed (skip/degrade) beats
  double-fire for periodic jobs.
- **Debate/nuance:** official docs present OAuth auto-refresh as "a single login persists across
  sessions" under normal use, while the 2026 issue corpus (#61912, #54443, #33811, #44930) shows it
  is NOT reliable across transient-network windows and concurrent clients. Also: Google's official
  incident guide emphasizes postmortem/learning and is thin on return-to-service mechanics —
  practitioner checklists (web-alert, rootly) supply the missing mechanical steps.

## Pitfalls (from literature)

1. **Fail-silent equivalence trap:** "the absence of an error is not the same as the presence of
   success" — 21 silent days happened because the P1 path lived inside the dead sessions.
2. **Root-cause overclaim:** no public refresh-token lifetime exists; write "consistent with the
   #61912 transient-corruption class (candidates a/b/c)", never "the refresh token expired after N
   days".
3. **Backlog processing surge:** disposing 8 asks + 34 dead-session artifacts at once is the
   "queued jobs cause a processing surge" hazard — record dispositions; apply NO bulk .env changes
   (each unmatched `PAPER_*=true` flip trips sentinel `flags_match_tokens` -> digest-only, sentinel.sh:96-119).
4. **Premature all-clear:** declare recovery only after "an entire cycle" — i.e. the next scheduled
   AM session completing normally, not the manual smoke test alone.
5. **Wrong evidence tool:** `AWAY_SESSION_DRY_RUN=1` SKIPS the dirty-tree check entirely
   (run_away_session.sh:80) — preflight evidence produced with it would be false. Use
   `AWAY_SESSION_TEST_PREFLIGHT=1` (:122-126), which exercises the real chain.

## Application to pyfinagent (66.0 mapping)

1. **Root-cause note (criterion feed):** cite #61912's mechanism + the exact timeline match
   (ECONNRESET 06-15..19 = transient window; persistent 401 from 06-20; zero self-heal x34
   sessions); list candidates (a) corruption, (b) concurrent refresh race, (c) unpublished lifetime
   — (c) least evidenced. Feeds 66.4: credential sanity ping (the #61912 proposed fix), Keychain/
   credential mtime staleness watch, and `setup-token` as the structural fix (any plist/env wiring
   is operator-token-gated per pre-tool-use-danger.sh:194,244).
2. **pending_tokens.json re-disposition:** add per-ask `disposition` + `resolved_at` (+ bump
   `updated` from the stale 2026-06-19T05:55Z): safe — sole parser scheduler.py:499-503 is
   `.get()`-tolerant; sentinel/healthcheck never read the file; the file sits under
   `handoff/away_ops/` (an excluded prefix, run_away_session.sh:97) so edits don't dirty the tree.
   Keep `asks` an array; keep `raised_at` ISO-parseable.
3. **Recovery-loop exit evidence (criterion 1):** timestamped
   `AWAY_SESSION_TEST_PREFLIGHT=1 bash scripts/away_ops/run_away_session.sh am` output showing
   `PREFLIGHT_PROMPT=am` — proves clean-per-exclusions tree AND sentinel pass in one shot. State
   the honest bound from §D: handoff-ROOT evidence files (.cycle_heartbeat.json,
   cycle_history.jsonl, kill_switch_audit.jsonl, prompt_leak_redteam_audit.jsonl) are NOT excluded
   and re-dirty at the next trading cycle; PM-after-trading-day recovery churn-sweeps are designed
   behavior, not residual outage.
4. **Verify-before-reenable:** no make-up sessions for the 34 missed (skip > double-launch); full
   recovery is declared on the next scheduled AM session completing end-to-end (web-alert "entire
   cycle" rule) — 66.0 closes on mechanical re-baseline evidence, not on that observation.
5. **Permanent improvement handoff to 66.4:** the missing control is a dead-man's switch on "last
   successful away session/cc_rail call" OUTSIDE the session process — healthcheck.sh already runs
   every 30 min independently and already computes `cycle_age_h` without paging on it (:61-79);
   that is the natural mount point.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: official Anthropic doc, GitHub repro'd bug report, Google SRE guide, SRE book cron chapter, Cronitor doc, practitioner checklist)
- [x] 10+ unique URLs total (~58 unique across 9 searches)
- [x] Recency scan (last 2 years) performed + reported (dedicated section above)
- [x] Full pages read (not abstracts/snippets) for the read-in-full set
- [x] file:line anchors for every internal claim (run_away_session.sh:80,96-102,112,122-126; pending_tokens.json:6-98; sentinel.sh:83-123,96-119; healthcheck.sh:61-79,117-157,174-181; scheduler.py:499-503; pre-tool-use-danger.sh:194,244; commands.py:137; run_memo.py:166)

Soft checks:
- [x] Internal exploration covered every module the caller named (4 files in full + 4 consumers grepped w/ context)
- [x] Contradictions/consensus noted (official "refresh persists" vs 2026 issue corpus)
- [x] All claims cited per-claim
- Note: three-variant query discipline satisfied per topic (queries listed above). Tool-call count
  exceeded the simple-tier soft budget (~22 vs <=10) because the caller mandated 9 search variants +
  6 full reads + a 4-file internal audit; brief length exceeds the 300-word simple guideline for the
  same reason (caller-mandated sections dominate; analysis depth kept to simple tier).

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 52,
  "urls_collected": 58,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief_66.0.md",
  "gate_passed": true
}
```
