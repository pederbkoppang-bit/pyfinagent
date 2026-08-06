# Research Brief -- phase-82.11: autoresearch nightly failure, metered rail exit + audible paging

**Tier:** moderate (caller-specified). **Audit class:** false.
**Researcher:** Layer-3 Researcher (Workflow rail). **Date:** 2026-08-06.
**Status:** COMPLETE. `gate_passed: true` (8 read in full / 33 URLs / recency scan /
12 internal files).

Objective: (a) implement the operator's already-made decision -- move the nightly
autoresearch loop OFF the metered Anthropic direct API, or disable it, without
buying credits; (b) fix the durable defect that 62 consecutive nightly failures
were never audible to the operator, with a Python-drivable notification path a
pytest can capture.

---

## Search queries run (three-variant discipline)

1. **Year-less canonical:** `Google SRE Workbook alerting on SLOs alert fatigue symptom-based paging`
2. **Year-less canonical:** `pytest mock notification path testing pitfalls patch where it is used assert payload`
3. **Last-2-year window:** `dead man's switch cron job heartbeat monitoring failed scheduled job alerting 2025 2026`
4. **Current-year frontier:** `alert escalation policy repeated failures notification deduplication incident 2026 best practice`

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-06 | official doc (Google SRE Book ch.6) | WebFetch full | "Every time the pager goes off, I should be able to react with a sense of urgency. I can only react with a sense of urgency a few times a day before I become fatigued." + "When pages occur too frequently, employees second-guess, skim, or even ignore incoming alerts." |
| 2 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-06 | official doc (SRE Workbook ch.5) | WebFetch full | Four evaluation axes: **precision, recall, detection time, reset time**. "Reset time: How long alerts fire after an issue is resolved. Long reset times can lead to confusion or to issues being ignored." |
| 3 | https://sre.google/workbook/on-call/ | 2026-08-06 | official doc (SRE Workbook ch.8) | WebFetch full | "All alerts should be immediately actionable. There should be an action we expect a human to take immediately after they receive the page that the system is unable to take itself." + "We target a maximum of two incidents per on-call shift." + repeated failures must be root-caused, not explained away. |
| 4 | https://prometheus.io/docs/alerting/latest/configuration/ | 2026-08-06 | official doc (Alertmanager) | WebFetch full | Defaults: `group_wait=30s`, `group_interval=5m`, **`repeat_interval=4h`**. "Notifications are not repeated if any new alerts have fired or any firing alerts have resolved since the last group_interval." Inhibition = higher-severity source suppresses target. |
| 5 | https://docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-08-06 | official doc (pytest) | WebFetch full | "Prefer patching the reference that your code uses instead of patching the original object..." and "For code that you control, a safer long-term pattern is to make dependencies explicit so they can be passed into the code under test instead of patched globally." |
| 6 | https://onlineornot.com/cron-job-monitoring-guide | 2026-08-06 | industry/practitioner | WebFetch full | "The real killer is when a job *doesn't run at all*." Heartbeat alerts on **absence**, catching reboots/deleted crontabs that produce no error to alert on. Daily jobs: 30-60 min grace period. |
| 7 | https://pytest-with-eric.com/mocking/pytest-common-mocking-problems/ | 2026-08-06 | community (focused) | WebFetch full | Five named problems incl. "Confusing or Incorrect Patch Targets Lead To False Positives or Brittle Tests" -> "Always patch the exact location where the method or object is being used"; and `autospec=True` to stop mocks accepting calls the real function would reject. |
| 8 | https://rootly.com/alert-management/alert-escalation-policies | 2026-08-06 | industry/practitioner (2026) | WebFetch full | "An alert escalation policy is a predefined set of rules that determines what happens when an alert is **not acknowledged or resolved** within a specified timeframe." Ladder shapes: time-based / severity-based / dynamic; channel escalates Slack -> SMS/voice. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://sre.google/sre-book/being-on-call/ | official doc | superseded for this question by Workbook ch.8 (#3) |
| https://developers.soundcloud.com/blog/alerting-on-slos/ | industry blog | restates SRE Workbook ch.5 (#2) |
| https://pingfatigue.com/slo-vs-threshold | vendor (2026) | vendor-marketing tier; SLO-burn framing already covered by #2 |
| https://techvzero.com/best-practices-slo-alerts/ | blog | derivative of #2 |
| https://github.com/Kriss-V/deadmancheck | code/OSS | "alerts when jobs run but do nothing" -- relevant prior art, but a tool not guidance |
| https://updog.watch/learn/what-is-dead-mans-switch | vendor | duplicate of #6's concept |
| https://nurbak.com/en/blog/dead-mans-switch/ | vendor (2026) | duplicate of #6 |
| https://crontap.com/blog/dead-man-switch-explained-for-developers | vendor | duplicate of #6 |
| https://cronradar.com/comparisons/cron-monitoring-best-practices | vendor (2026) | duplicate of #6 |
| https://appstatus.io/docs/heartbeats | vendor doc | late->missing state machine is interesting but SaaS-coupled (violates $0/local-only) |
| https://drumbeats.io/heartbeat-monitoring | vendor | duplicate |
| https://apistatuscheck.com/blog/best-cron-job-monitoring-tools-2026 | listicle | tool comparison, no design guidance |
| https://dev-brains-ai.com/blog/cron-job-monitoring-and-alerting-guide | blog | low authority |
| https://rootly.com/alert-management/alert-deduplication-and-correlation | industry (2026) | overlaps #8 + #4 |
| https://sreschool.com/blog/escalation-policy/ | blog (2026) | derivative of #8 |
| https://sreschool.com/blog/alert-deduplication/ | blog (2026) | derivative |
| https://oneuptime.com/blog/post/2026-01-30-alert-deduplication/view | vendor blog (2026) | dedup-key design; covered by #4 defaults |
| https://oneuptime.com/blog/post/2026-02-02-pytest-mocking/view | vendor blog (2026) | derivative of #5 |
| https://recca0120.github.io/en/2026/04/03/pytest-mock/ | blog (2026) | `mocker` fixture ergonomics; repo uses `unittest.mock.patch` |
| https://runebook.dev/en/docs/python/library/unittest.mock/the-patchers | doc mirror | mirror of CPython docs |
| https://incident.io/changelog/reassign-escalations | vendor changelog | product note, not guidance |
| https://www.itoc360.com/what-is-it-alerting/ | vendor | generic |
| https://www.mindfulchase.com/.../troubleshooting-alert-routing-and-escalation-failures-in-pagerduty | blog | PagerDuty-specific |
| https://pypi.org/project/pytest-patch/ | package | not needed |
| https://docs.pytest.org/en/7.1.x/how-to/monkeypatch.html | doc (old ver) | superseded by stable (#5) |

**URLs collected: 33** (8 read in full + 25 snippet-only).

## Recency scan (2024-2026)

Performed -- query variants 3 and 4 above were explicitly scoped to 2025/2026, and
sources #6 and #8 plus 12 of the snippet-only rows are 2025-2026 material.

**Result: 2 new findings that COMPLEMENT (do not supersede) the canonical Google SRE
guidance.**

1. **Dead-man's-switch / heartbeat monitoring is now the mainstream 2025-2026 answer to
   exactly this failure class** -- an entire vendor category (Healthchecks-style ping
   endpoints, AppStatus `late -> missing` state machines, `deadmancheck`'s "runs but does
   nothing" assertions). The load-bearing insight is inversion: alert on the **absence of
   a success signal**, not the presence of an error. Directly relevant here because
   pyfinagent's current design alerts only when `run_memo.py` *returns 1* -- and
   `run_memo.py` has **two silent exit-0 paths** (`_embedding_preflight` at
   `run_memo.py:306-309`, and `--preflight-only` at `:313-316`) plus a WARN path
   (`:168-182`) that all produce zero output and zero alert. root_cause.md:128-141 records
   that the embedding soft-skip already caused a real "many nights produced no memo at
   all" window. An error-triggered alarm cannot see any of those.
2. **Escalation-ladder vocabulary has converged** (Rootly 2026, #8): escalation is keyed
   on *unacknowledged/unresolved duration*, and each tier changes **channel and/or
   audience**, not just message text. "Adaptive escalation that increases priority or
   expands notification scope based on error rate or burn rate" is the recommended shape
   for SLO-driven environments. No 2024-2026 source contradicts the SRE Book's
   symptom-based/actionable-page doctrine; the newer material is operational packaging of
   the same rules.

No source found in the window that argues *against* suppressing steady-state repeats,
and none that recommends re-paging an unchanged known-bad condition nightly.

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/autoresearch/run_nightly.sh` | 106 | launchd wrapper; env sourcing, max-rail preflight, fail-state + Slack page | LIVE; paging seam :49-69 |
| `scripts/autoresearch/run_memo.py` | 322 | gpt-researcher runner; writes memo or ERROR/WARN file | LIVE; failing nightly |
| `backend/services/observability/alerting.py` | 301 | **canonical** Python alert seam (`raise_cron_alert` / `_sync`, AlertDeduper, bot-token fallback) | LIVE, 20+ callers |
| `backend/services/freshness_cron.py` | 267 | phase-82.10 sibling: cron shim + transition gate + injectable `notify=` | LIVE (shipped 2026-08-05, `b7c69bb9`) |
| `backend/tests/test_phase_82_10_freshness_paging.py` | 493 | **the fixture idiom 82.11 must follow** (13 tests, `ALERT_TARGET` patch, precondition asserts) | LIVE |
| `handoff/away_ops/autoresearch_fail_state.json` | 1 | `{"consecutive_fails": 13}` | LIVE |
| `handoff/autoresearch/` | 65 files | 62 `*-ERROR-*`, 0 `*-WARN-*`; newest `2026-08-06-ERROR-topic08.md` | measured 2026-08-06 |

### The paging seam, verbatim (`scripts/autoresearch/run_nightly.sh:58-67`)

```bash
    if [ "$new_fails" -ge "$PAGE_AFTER_N" ]; then
        BOT_TOKEN=$(grep -m1 '^SLACK_BOT_TOKEN=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        CHANNEL=$(grep -m1 '^SLACK_CHANNEL_ID=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$CHANNEL" ] && CHANNEL="C0ANTGNNK8D"
        if [ -n "$BOT_TOKEN" ]; then
            curl -s -m 10 -X POST https://slack.com/api/chat.postMessage \
                -H "Authorization: Bearer $BOT_TOKEN" \
                -H 'Content-type: application/json; charset=utf-8' \
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"P1 AUTORESEARCH: $new_fails consecutive nightly autoresearch failures ($_ctx rc=$_rc). See $LOG.\"}" >/dev/null 2>&1 || true
        fi
    fi
```


## Answers to the contract's questions (internal half -- all measured 2026-08-06)

### Q1. Does the bash paging seam deliver? What are its failure modes?

**It is reachable and it almost certainly fires -- but nothing can prove it did, and it
has no escalation.** Measured evidence:

- `launchctl list` shows `-  1  com.pyfinagent.autoresearch` -- last exit **1**, i.e. the
  `run_memo` failure branch (`run_nightly.sh:97-104`), not the rc=78 preflight branch.
- `handoff/autoresearch.log` tail ends `[2026-08-06T02:00:13+02:00] END nightly
  autoresearch FAIL rc=1` (`run_nightly.sh:99`). `handoff/away_ops/autoresearch_fail_state.json`
  mtime is `Aug 6 02:00:13 2026` and reads `{"consecutive_fails": 13}` -- so
  `_record_fail_and_page` (`:49`) DID execute last night and DID pass the
  `new_fails >= PAGE_AFTER_N` test (13 >= 3) at `:58`.

Concrete failure modes of that seam:

1. **No receipt of any kind.** The curl at `:63-66` ends `>/dev/null 2>&1 || true`.
   stdout, stderr AND exit status are all discarded. Nothing is appended to `$LOG`
   about the page. So "was the operator told?" is **unanswerable from local state** --
   the seam is unobservable by construction. `handoff/autoresearch.launchd.log` is empty.
2. **Slack API errors are HTTP 200.** `chat.postMessage` returns `200 {"ok":false,
   "error":"not_in_channel"|"invalid_auth"|"channel_not_found"}`. `curl -s` exits 0.
   Even without the `|| true`, a bad token or an un-joined channel is indistinguishable
   from success. (Contrast the Python path, which parses `.get("ok")` --
   `alerting.py:168`.)
3. **Token-absent silently skips.** `if [ -n "$BOT_TOKEN" ]` (`:62`) has no else. If
   `SLACK_BOT_TOKEN` is missing/renamed in `backend/.env`, the page is skipped with no
   log line. Worse, under `set -euo pipefail` (`:6`) a `grep` that matches nothing makes
   the `BOT_TOKEN=$(grep ... | cut | tr | tr)` assignment at `:59` exit non-zero, which
   under `set -e` aborts the function/script before `exit "$rc"` at `:104` -- turning a
   missing token into a *different* exit code, silently.
4. **No escalation ladder.** The message body at `:66` is the same single-line P1 every
   night; only the integer `$new_fails` changes. Same channel, same severity, same
   cadence (once per night) forever. It fired on night 3 and has fired on ~11 nights
   since with no change in urgency, no summary, no "this is now 2 weeks old", no
   auto-disable. That is textbook alert fatigue (see Q5).
5. **Credential duplication.** It re-greps `backend/.env` for the token instead of using
   the settings object, so it drifts from `backend/config/settings.py` independently.
6. **Not drivable from Python.** It is a bash function inside a launchd wrapper. The only
   existing way a pytest reaches it is the 76.9.2 idiom (`subprocess.run(["bash",
   NIGHTLY], env={"AUTORESEARCH_REPO": tmp})`, `test_phase_76_9_2_max_bridge.py:302-306`),
   which drives the shell, not the notification payload -- you cannot CAPTURE the emitted
   alert, only observe that curl was (or was not) invoked. The step's criterion 1
   ("capture the emitted alert") therefore cannot be met by the bash seam as written.

### Q2. The canonical Python notification seam

**`backend/services/observability/alerting.py::raise_cron_alert_sync` (`:253-287`)** --
sync wrapper over `raise_cron_alert` (`:179-250`). This is unambiguously the canonical
one: **20+ production call sites** across `kill_switch.py:476,:1029`,
`paper_trader.py:1360`, `cycle_health.py:119,:241`, `autonomous_loop.py:414,:980,:1001`,
`drawdown_alarm.py:154`, `slack_bot/scheduler.py:58`, `meta_evolution/cron.py:159`,
`agents/claude_code_client.py:194`, `news/sources/{alpaca,benzinga,finnhub}.py`,
`econ_calendar/sources/finnhub_earnings.py`, and `freshness_cron.py:132`.

Properties that matter here:
- Signature `(source, error_type, severity, title, details)` -- `severity` **must be
  `"P0"`/`"P1"`** to reach the bot-token fallback: with `slack_webhook_url` empty (the
  measured state on this machine per the 62.7 comment at `alerting.py:211-216`), a P2 is
  logged and dropped (`:219-224`).
- `_bot_token_fallback` (`:136-176`) posts via urllib and **parses `ok`**
  (`:168`) -- unlike the bash curl.
- Fail-open: never raises out (`:243-250`, `:283-287`).
- `AlertDeduper` (`:63-107`): P0/P1 bypass the *consecutive threshold* but NOT the
  *repeat window* (`:83-93`), so a P1 re-fires every `alert_repeat_hours` (default 1h)
  forever. **AlertDeduper does not suppress steady state** -- the exact trap 82.10
  documented (`freshness_cron.py:1-40`). For a once-nightly job the repeat window is
  irrelevant (24h >> 1h), so 82.11 must own its own transition/escalation gate exactly
  as `freshness_cron.run_freshness_check` does.

**Did 82.10 create a reusable seam? Yes -- a PATTERN, not a new emitter.** Commit
`b7c69bb9` (2026-08-05) added `backend/services/freshness_cron.py`: a cron-shim module
with (a) an injectable `notify=` parameter defaulting to a *function-local* import of
`raise_cron_alert_sync` (`:127-132`), (b) module-level transition state
`_last_red_sources` + `reset_transition_state()` test seam (`:76-93`), (c) a fail-open
top-level wrapper returning a summary dict incl. `alerts_emitted` (`:207-226`), and (d)
`register_*_cron(scheduler, ...)` with `replace_existing=True` (`:229-258`).
**82.11 should mirror this module shape rather than duplicate the emitter.**

### Q3. How to derive "N consecutive prior failures"

Measured: the two candidate sources **disagree today**.

| Source | Value today | Why |
|---|---|---|
| `handoff/away_ops/autoresearch_fail_state.json` | **13** | reset to 0 by two *manual* `run_nightly.sh` runs on 07-24 and 07-25 (`:96`) |
| `ls handoff/autoresearch/*-ERROR-*` consecutive-by-date | **30** (2026-07-08..2026-08-06, unbroken) | manual successes wrote a memo but did not delete that night's ERROR file |
| `ls handoff/autoresearch/*-ERROR-*` total | **62** (62 distinct dates: 7 in Apr, 25 in May, 24 in Jul, 6 in Aug) | includes the 2026-04/05 provider-prefix era |

Two dates -- **2026-07-24 and 2026-07-25** -- carry BOTH an `-ERROR-` file and a success
memo. Any naive "this date has an ERROR file => that night failed" scan counts those as
failures. A correct scan must treat a date as a failure only if it has an ERROR file
**and no non-ERROR memo**, and must walk backwards from today stopping at the first
non-failing date.

**Recommendation: derive from the ERROR directory, and keep the JSON as a cross-check.**
Reasons: (a) the criterion says "asserted against a fixture directory of prior ERROR
files", which is only satisfiable by an implementation that *reads a directory* -- a
JSON-only implementation cannot be driven by that fixture at all, so the criterion
forces directory-derivation; (b) the JSON is single-writer bash and is silently reset by
any manual run, and **nothing else in the repo reads it** (grep: only
`run_nightly.sh:42` and `test_phase_76_9_2_max_bridge.py:333`); (c) the directory is the
durable audit trail. Design the function as
`count_consecutive_failures(memo_dir: Path, today: date) -> int` with the dir injected
(that IS the fixture seam), and pass any bash-supplied counter in as an optional
override rather than as the source of truth.

Traps to pin in the test: total-vs-consecutive (62 vs 30), the BOTH-files date, and an
empty directory (must return 0, not crash).

### Q4. Max-rail bridge -- state, and is the flag flip viable?

**The code exists, the processes are UP, and the flag is the only thing missing.**
Measured 2026-08-06:

```
$ launchctl list | grep -iE 'bridge|proxy'
650   0   com.pyfinagent.anthropic-bridge
668   0   com.pyfinagent.claude-code-proxy
$ curl -sf -m 5 http://127.0.0.1:18797/health
{"ok":true,"proxy":"claude-code-cli"}
$ curl -sk -m 5 https://127.0.0.1:18796/health   -> UP
```

- `scripts/ops/anthropic_max_bridge.py` present (8510 bytes, mtime 2026-07-25).
- `run_nightly.sh:78-92` is the guard; default OFF via `${AUTORESEARCH_USE_MAX_RAIL:-0}`.
  ON => `curl -sf /health`, then export `ANTHROPIC_API_URL`/`ANTHROPIC_BASE_URL` to the
  bridge and force `ANTHROPIC_API_KEY="max-rail-dummy-key"` (`:85`) so any leak to
  `api.anthropic.com` 401s = provable $0.
- 12 tests in `backend/tests/test_phase_76_9_2_max_bridge.py` (flag-off inert `:309`,
  bridge-down rc=78 `:319`, healthy-bridge routing + dummy-key override `:337`).
- **Positive control already exists in the record**: the two non-ERROR memos in
  `handoff/autoresearch/` are dated 2026-07-24 and 2026-07-25 -- the manual max-rail runs.
  `.claude/masterplan.json:19335` (an `[OPERATOR ACTION]` step) records the measurement:
  the 01:16 manual run logged `max-rail ON -- routing via http://127.0.0.1:18797` and
  ended OK, while the 02:00 launchd run logged neither `max-rail ON` nor `rc=78` and died
  on the metered 400 -- i.e. **the flag is absent from `backend/.env`**, and the bridge
  was UP across that window.

**Verdict: flipping `AUTORESEARCH_USE_MAX_RAIL=1` is a viable $0 implementation** of
"move autoresearch off the metered rail" -- it is one operator-owned `.env` line, no
restart (`run_nightly.sh:22-30` re-sources `.env` each invocation), and reverting is
deleting the line. Two caveats for the contract: (i) it is **operator-gated** -- the
executor cannot write `backend/.env` (sandbox-denied; also `.env:81` is the known
malformed line), so 82.11's code deliverable cannot *be* the flip; (ii) if the bridge is
down at 02:00 (post-reboot before `OPS-BRIDGE-BOOTSTRAP`), the night exits 78 and pages
-- which is now correct-and-audible behaviour rather than silent metered spend.

**Honest disable path** if the operator prefers off over rerouted:
`launchctl bootout gui/$(id -u)/com.pyfinagent.autoresearch` (or
`launchctl unload ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist`) -- but note
that *silently* disabling it recreates the same invisibility defect in mirror image (a
job that never runs also never reports). Prefer an explicit disabled-state that the
notification path still reports on (a dead-man's-switch, see Q5), or a `--preflight-only`
$0 mode which already exists (`run_memo.py:249-255,:313-316`) and exits 0 without any LLM
call.

### Q5. External literature -- alert fatigue + escalation (what justifies the shape)

The literature gives four rules that between them determine the escalation shape for a
nightly job that has failed 13 (or 30) times in a row:

1. **A page must be actionable, and repeats destroy actionability.** "All alerts should be
   immediately actionable. There should be an action we expect a human to take
   immediately after they receive the page that the system is unable to take itself."
   (SRE Workbook ch.8, #3). "When pages occur too frequently, employees second-guess,
   skim, or even ignore incoming alerts." (SRE Book ch.6, #1). The current seam sends
   the *same* P1 every night for a condition the operator has already decided not to fix
   by buying credits -- by SRE's own definition it is no longer a page.
2. **Reset time is a first-class metric.** "Reset time: How long alerts fire after an
   issue is resolved. Long reset times can lead to confusion or to issues being ignored."
   (SRE Workbook ch.5, #2). The corollary for a *persisting* fault is symmetric: an alert
   that keeps firing while nothing changes has zero incremental information.
3. **Repeat suppression is edge-triggered, not level-triggered.** Alertmanager's
   `repeat_interval` defaults to **4h**, and crucially: "Notifications are not repeated
   if any new alerts have fired or any firing alerts have resolved since the last
   group_interval" (#4). i.e. the industry default is *notify on state change*, and the
   repeat is a slow safety net -- not a per-occurrence broadcast. pyfinagent's
   `AlertDeduper` implements a 1h `repeat_hours` (`alerting.py:124-130`), which for a
   once-nightly job never suppresses anything, so **the caller must own the transition
   gate** -- the same conclusion 82.10 reached (`freshness_cron.py:143-145`).
4. **Escalation changes audience/channel/severity, not the sentence.** "An alert
   escalation policy is a predefined set of rules that determines what happens when an
   alert is not acknowledged or resolved within a specified timeframe" (#8), progressing
   "Slack, Microsoft Teams, mobile push" -> "SMS or voice calls".

**Recommended shape for 82.11** (each element traceable to a source above):

| Condition | Emit | Basis |
|---|---|---|
| run FAILS, `N == 1` | nothing (or log only) | #1 -- a single overnight failure is not yet actionable; also matches the existing `PAGE_AFTER_N=3` intent |
| run FAILS, `N` crosses the threshold (default 3) for the FIRST time | **P1**, once | #1/#3 -- edge-triggered on a state change |
| run FAILS, `N` still above threshold, no state change | **nothing** | #2/#3 -- steady state re-page is the alert-fatigue mode; 82.10 precedent |
| `N` crosses a SECOND, higher bound (e.g. 7) | **escalated** alert -- distinct `error_type`, escalation-tier field in `details`, P0 if warranted | #8 -- ladders key on unresolved *duration*; a new `error_type` also gives `AlertDeduper` a fresh key so it is not swallowed |
| run SUCCEEDS after failures | **recovery** notice (or at minimum reset state) | #2 -- reset time; also proves the channel is alive |
| run produced **no result at all** (silent exit 0) | treat as failure | #6 -- "The real killer is when a job *doesn't run at all*" |

The last row is the highest-value new idea from the recency scan and is currently
un-covered: `run_memo.py` can exit 0 with no memo via `_embedding_preflight` (`:306-309`),
`--preflight-only` (`:313-316`) and the WARN path (`:168-182`). A **success-signal**
predicate ("a non-ERROR memo file dated today exists") is strictly stronger than an
exit-code predicate and costs nothing extra to implement, since the ERROR-directory scan
from Q3 already walks that directory.

### Q6. Pytest notification-path pitfalls -- what makes such a test a fake guard

From #5, #7 and the repo's own 82.10 post-mortem, in descending order of how likely each
is to bite here:

1. **Patching the wrong namespace.** pytest's own docs: "Prefer patching the reference
   that your code uses instead of patching the original object" (#5). This is a *live*
   trap in this exact repo: `cycle_health._fire_freshness_alarm` imports
   `raise_cron_alert_sync` **function-locally**, so
   `patch("backend.services.cycle_health.raise_cron_alert_sync")` patches nothing and the
   guard passes vacuously. 82.10 pinned this with a dedicated test
   (`test_phase_82_10_freshness_paging.py:449-459`, `test_wrong_patch_target_does_not_exist`)
   and used `ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"`
   (`:40`). **82.11 must use the same target string and copy that anti-vacuity pin.**
2. **Monkeypatching the function under test.** If the new code exposes `notify=` and every
   test injects a fake, no test ever proves production resolves the real
   `raise_cron_alert_sync`. 82.10 solved this by having its *primary* criterion-2 test use
   **no injection at all** and patch the real module attribute instead
   (`:323-325`: "No `notify=` injection: the evaluator resolves `raise_cron_alert_sync`
   itself, so this drives the operator channel the production job uses"). Keep `notify=`
   for convenience, but at least one test must NOT use it.
3. **Asserting on a log line instead of the emitted payload.** A `caplog` assertion passes
   if the code logs and then drops the alert -- which is precisely the pre-fix behaviour
   for a P2 (`alerting.py:219-224`). Assert on `mock_alert.call_args.kwargs`:
   `severity == "P1"` (a P2 with an empty webhook is logged and dropped), `source`,
   `error_type`, and the specific `details` keys.
4. **The always-fires guard.** The step's third criterion (healthy run => no alert) is the
   structural counter. But it is only meaningful with a **precondition assertion** that
   the healthy fixture really was healthy -- 82.10 does this at `:423-425` ("so a zero
   cannot come from a broken fixture"). Symmetrically the failure test needs a
   precondition assertion that the fixture really breached (`:334-336`).
5. **Module-level state leaking between tests.** Any transition/last-alerted state must
   have a `reset_*()` test seam plus an `autouse` fixture, or the "no alert" test passes
   because the gate suppressed it rather than because the run was healthy -- 82.10's
   `_clean_transition_state` docstring says exactly this (`:91-98`).
6. **Mocks that accept calls the real function would reject.** Use `autospec=True`
   (#7, problem 3) so a `raise_cron_alert_sync(...)` called with a wrong/renamed kwarg
   fails the test instead of silently recording.
7. **Over-mocking / implementation coupling** (#7, problem 1): do not assert that
   `_bot_token_fallback` or `urlopen` was called -- assert the alert the *operator-facing*
   seam received. That keeps the test valid if the delivery backend changes.
8. **A test that only proves a function exists.** 82.10's criterion-1 guard executes the
   callable the stub scheduler received (`:182-209`) rather than asserting a symbol is
   importable. The analogue here: drive the whole failure->notify path from the
   entry point production uses, not the helper.

## Key findings (cited per claim)

1. **The alarm is not silent -- it is unobservable and un-escalating.** The seam executed
   last night (fail_state mtime `Aug 6 02:00:13`, `consecutive_fails: 13`,
   `run_nightly.sh:53-56`) and passed its `>= PAGE_AFTER_N` test (`:58`), but
   `>/dev/null 2>&1 || true` (`:66`) discards the result, and Slack returns HTTP 200 on
   `{"ok":false}` -- so "did the operator get paged?" is unanswerable. Contrast
   `alerting.py:168` which parses `ok`.
2. **Re-paging an unchanged failure nightly is the documented anti-pattern.** "I can only
   react with a sense of urgency a few times a day before I become fatigued" (SRE Book
   ch.6, #1); Alertmanager's default is to *not* repeat absent a state change (#4).
3. **The canonical Python seam is `raise_cron_alert_sync`** (`alerting.py:253`), 20+
   production callers, and it must be called at severity `P0`/`P1` to survive the empty
   `slack_webhook_url` on this machine (`alerting.py:211-224`).
4. **82.10 (commit `b7c69bb9`, 2026-08-05) is the template, not a competing seam.** Same
   defect class ("correct emitter, no trigger / no audience"), same module shape
   (`freshness_cron.py`), same test idiom (`test_phase_82_10_freshness_paging.py`).
   Reuse; do not build a second notifier.
5. **The two failure counters disagree: 13 (JSON) vs 30 (consecutive ERROR dates) vs 62
   (total ERROR files).** Two dates carry BOTH an ERROR file and a success memo
   (2026-07-24, 2026-07-25). The criterion's "fixture directory of prior ERROR files"
   forces directory-derivation.
6. **The Max rail is live and the flag is the only gap.** `anthropic-bridge` pid 650,
   `claude-code-proxy` pid 668, `GET 127.0.0.1:18797/health -> {"ok":true,...}` measured
   2026-08-06. `AUTORESEARCH_USE_MAX_RAIL=1` in `backend/.env` is a one-line, no-restart,
   operator-owned $0 fix (`run_nightly.sh:78`, masterplan `[OPERATOR ACTION]` step at
   `.claude/masterplan.json:19335`).
7. **Exit-code-based failure detection is structurally blind here.** `run_memo.py` has
   three silent exit-0 paths (`:168-182` WARN, `:306-309` embedding skip, `:313-316`
   preflight-only) and root_cause.md:128-141 records a real multi-night window where they
   produced no memo and no alert. The 2025-2026 dead-man's-switch literature (#6) says
   alert on the **absence of success**.

## Consensus vs debate (external)

**Consensus:** pages must be actionable and rare (#1, #3); dedup/suppression should be
edge-triggered on state change (#4); escalation ladders key on unresolved duration and
change audience/channel (#8); heartbeat/absence monitoring beats error monitoring for
scheduled jobs (#6).

**Debate / tension:** SRE Workbook ch.8 (#3) explicitly warns against *silencing* a
recurring page ("Explaining away a page as 'transient' ... invites the bug to happen
again"), which pulls against suppression. The resolution the literature supports is:
suppress the **repeat**, never the **state change**, and add an escalation tier so a
long-lived unresolved condition gets *louder*, not quieter. That is the shape recommended
in Q5 -- and it is why "just stop paging after N" would be the wrong fix.

Second tension: #6's dead-man's-switch pattern is SaaS-shaped (external ping endpoint).
pyfinagent is local-only + $0 (auto-memory `project_local_only_deployment`), so import the
*inversion* (assert a success artifact exists) not the *architecture* (a third-party
monitor).

## Pitfalls (from literature + repo history)

- Patching a function-locally-imported symbol (patches nothing) -- pinned by 82.10 at
  `test_phase_82_10_freshness_paging.py:449-459`.
- Injecting `notify=` in every test so nothing proves production resolves the real one.
- `caplog` assertions standing in for payload assertions (a dropped P2 still logs).
- A guard with no precondition assertion: a "no alert" pass caused by a broken fixture.
- Module-level state with no reset seam.
- Counting ALL ERROR files (62) instead of the consecutive run (30) -- or trusting the
  JSON counter (13) that any manual run silently resets.
- Building a second notification emitter instead of reusing `raise_cron_alert_sync`.
- Fixing the rail without fixing the audibility (or vice versa) -- the step has two
  independent deliverables.

## Application to pyfinagent

- **New module, 82.10 shape:** e.g. `backend/services/autoresearch_health.py` exposing
  `count_consecutive_failures(memo_dir, today)` and
  `check_autoresearch_health(*, memo_dir=None, fail_state_path=None, notify=None,
  page_after=3, escalate_after=7) -> dict`, fail-open, returning
  `{"ok", "consecutive_failures", "alerts_emitted", "escalated", "last_success_date"}`.
  Mirror `freshness_cron.py:99-226` incl. `reset_*_state()` (`:76-93`) and
  `register_*_cron` (`:229-258`).
- **Emitter:** `raise_cron_alert_sync` from
  `backend/services/observability/alerting.py:253`, `severity="P1"` (escalated tier may
  use `"P0"`), `source="autoresearch"`, `error_type` distinct per tier so the deduper
  keys separately.
- **Test file:** `backend/tests/test_phase_82_11_autoresearch_failure_paging.py`, patching
  `backend.services.observability.alerting.raise_cron_alert_sync`, with `tmp_path`
  fixture directories of `YYYY-MM-DD-ERROR-topicNN.md` files (criterion 2) and the 82.10
  anti-vacuity pins.
- **Rail exit:** the code deliverable is the *loud-fail* path that already exists
  (`run_nightly.sh:78-92`); the flip itself is the operator's `.env` line. Recommend the
  contract records this explicitly rather than pretending the executor can do it.
- **Do NOT delete the bash seam in the same step** -- keeping both while the Python path
  is proven avoids a window with no paging at all. Queue the bash-seam retirement as a
  follow-up if desired.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total (incl. snippet-only) -- **33**
- [x] Recency scan (last 2 years) performed + reported -- 2 complementary findings
- [x] Full pages read (not abstracts) for the read-in-full set -- all 8 via WebFetch
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (12 files/artifacts inspected)
- [x] Contradictions / consensus noted (SRE "don't silence" vs dedup; SaaS vs local-only)
- [x] All claims cited per-claim
- [ ] **Brief length exceeds the `moderate` <=700-word guideline.** Disclosed honestly:
  the caller posed six specific questions with a large internal-code scope. Tier guidance
  is on depth/length, not on the gate; no hard blocker is affected.

Known gaps (not gate-blocking, but the contract should not overclaim):
- `backend/.env` is sandbox-denied to this researcher, so I could NOT verify first-hand
  that `SLACK_BOT_TOKEN` / `SLACK_CHANNEL_ID` are present, nor that
  `AUTORESEARCH_USE_MAX_RAIL` is absent. The absence claim is inherited from the measured
  record at `.claude/masterplan.json:19335` (2026-07-25) plus the fact that
  `handoff/autoresearch.log` shows no `max-rail ON` line for any 02:00 run. **Main should
  re-grep `backend/.env` before writing the contract.**
- Whether any of the ~11 nightly curls actually reached Slack is **unknowable from local
  state** by design (see Q1.1). Only the operator's Slack history can settle it.

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 25,
  "urls_collected": 33,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_82.11.md",
  "gate_passed": true
}
```
