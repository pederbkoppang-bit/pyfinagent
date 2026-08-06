# Contract -- phase-82.11

**Step:** 82.11 (P1) -- the nightly autoresearch loop has failed every night on
metered-rail credit exhaustion, and the failure is not audible to the operator.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.11.md`,
envelope `gate_passed: true`, 8 external sources read in full, 33 URLs
collected, recency scan performed (2 complementary 2025-2026 findings),
12 internal files inspected. Launched on the Workflow structured-output rail.

---

## 1. Research-gate summary

Full brief: `handoff/current/research_brief_82.11.md`.

Load-bearing findings, each with the anchor the contract relies on:

1. **The step's premise is half wrong and the contract must say so.** The step
   text says "the loop writes a failure file and exits silently, so 59 failures
   produced zero operator signal". A paging seam DOES exist and DID execute:
   `scripts/autoresearch/run_nightly.sh:49-69` (`_record_fail_and_page`,
   phase-75.11 / 76.9.2). `handoff/away_ops/autoresearch_fail_state.json` reads
   `{"consecutive_fails": 13}` with mtime `Aug 6 02:00:13 2026`, which only
   `:53-56` writes, and `13 >= PAGE_AFTER_N=3` at `:58`. So the seam ran and
   took its paging branch last night.
2. **What is actually wrong with that seam** (`run_nightly.sh:63-66`): the curl
   ends `>/dev/null 2>&1 || true`, discarding stdout, stderr AND exit status,
   and Slack's `chat.postMessage` returns **HTTP 200** with `{"ok":false}` for
   `invalid_auth` / `not_in_channel`. So delivery is unobservable by
   construction. It also has no escalation ladder (same channel, same severity,
   same sentence nightly; only the integer changes) and is **not drivable from
   pytest** -- it is a bash function inside a launchd wrapper, so the step's
   criterion 1 ("capture the emitted alert") cannot be satisfied by it.
3. **Canonical Python emitter:**
   `backend/services/observability/alerting.py:253 raise_cron_alert_sync`
   (20+ production call sites). Severity **must** be `P0`/`P1`: with
   `slack_webhook_url` empty on this machine, a `P2` is logged and dropped
   (`alerting.py:219-224`), while `P0`/`P1` reach `_bot_token_fallback`
   (`:136-176`), which -- unlike the bash curl -- parses `ok` at `:168`.
4. **phase-82.10 (`b7c69bb9`) is the template, not a competing emitter.**
   `backend/services/freshness_cron.py` supplies the module shape (injectable
   `notify=`, reset seam, fail-open summary dict) and
   `backend/tests/test_phase_82_10_freshness_paging.py` supplies the test idiom
   (`ALERT_TARGET` module-path patch, precondition assertions,
   `test_wrong_patch_target_does_not_exist`).
5. **`AlertDeduper` does not suppress steady state** (`alerting.py:63-107`):
   P0/P1 bypass the consecutive threshold but the repeat window (default 1h)
   is irrelevant to a once-nightly job. The **caller must own the transition
   gate** -- exactly 82.10's conclusion.
6. **External doctrine that fixes the escalation shape.** SRE Book ch.6: "When
   pages occur too frequently, employees second-guess, skim, or even ignore
   incoming alerts." SRE Workbook ch.8: "All alerts should be immediately
   actionable." Alertmanager: "Notifications are not repeated if any new alerts
   have fired or any firing alerts have resolved since the last
   group_interval" (`repeat_interval` default 4h) -- i.e. **notify on state
   change, repeat only as a slow safety net.** Rootly 2026: an escalation tier
   changes severity/audience/channel, not the sentence. SRE Workbook ch.8 also
   warns against *silencing* a recurring page, so the resolution is: suppress
   the **repeat**, never the **state change**, and get **louder** with age.
7. **The Max rail is live.** Measured by Main 2026-08-06:
   `curl http://127.0.0.1:18797/health` -> `{"ok":true,"proxy":"claude-code-cli"}`,
   and a real `POST /v1/messages` returned `BRIDGE_OK`.
   `launchctl list` -> `com.pyfinagent.anthropic-bridge` pid 650,
   `com.pyfinagent.claude-code-proxy` pid 668. The only two non-ERROR memos on
   disk (2026-07-24, 2026-07-25) are the manual max-rail runs -- a positive
   control that the rail produces real memos.
8. **pytest vacuity traps** (brief Q6): patching a function-locally imported
   symbol patches nothing; injecting `notify=` in every test proves nothing
   about production; `caplog` assertions pass on a dropped alert; a guard with
   no precondition assertion can pass on a broken fixture.

### Corrections Main made to the brief before relying on it

- **The brief's "30 consecutive ERROR dates" is wrong by its own rule.** The
  brief itself recommends counting a date as failing only when it has an ERROR
  file **and no non-ERROR memo**. Main re-derived structurally over
  `handoff/autoresearch/*.md`, walking back from 2026-08-06 and stopping at the
  first non-failing date: **12** consecutive failing dates (2026-07-26 ..
  2026-08-06), stopping at 2026-07-25 which carries BOTH an ERROR file and a
  success memo. Totals confirmed: 62 ERROR dates / 62 ERROR files, 2 success
  dates, both-files dates `['2026-07-24', '2026-07-25']`. The brief's other two
  numbers (13 in the JSON, 62 total) reproduce exactly.
- **The brief's disclosed `.env` gap is closed.** Main re-read `backend/.env`
  key names directly: `SLACK_BOT_TOKEN` (non-empty), `SLACK_CHANNEL_ID`
  (non-empty), `ANTHROPIC_API_KEY` (non-empty) are all present, and there is
  **no `AUTORESEARCH_*` key at all** -- confirming the flag is absent and the
  nightly run defaults to the metered direct API.
- **The brief says the rail flip "cannot be 82.11's code deliverable" because
  it is an operator-owned `.env` line. Main rejects that conclusion.**
  `backend/.env` is gitignored (`.gitignore:5`), so an `.env` flip is
  **unauditable** and invisible to review. The default itself lives in tracked
  code at `run_nightly.sh:78` (`${AUTORESEARCH_USE_MAX_RAIL:-0}`). Flipping the
  **default** is version-controlled, review-able, needs no restart
  (`run_nightly.sh:22-30` re-sources `.env` every invocation), and is still
  overridable by the operator with a single `AUTORESEARCH_USE_MAX_RAIL=0` line.
  That is the honest implementation of the operator's decision.

---

## 2. The operator's decision (verbatim), and its derivation

Recorded verbatim from the operator's session directive of 2026-08-06. This is
the operator's own text; nothing here is paraphrased or invented:

> "APPROVAL. I approve every gated step. That unblocks proceeding, but two
> criteria need a recorded DECISION, not just consent -- my standing
> constraints decide them. Record my words plus the derivation; do NOT invent a
> quote from me:
> - 82.11 metered rail: $0-metered stands -> move autoresearch OFF it or
>   disable it. Do NOT buy credits."

The metered-rail sentence again, unquoted and unindented so it is byte-exact
(the blockquote above adds `> ` markers; this block is what the criterion-4
guard matches against):

```
82.11 metered rail: $0-metered stands -> move autoresearch OFF it or
  disable it. Do NOT buy credits.
```

And from the same directive's constraints block:

> "CONSTRAINTS. $0 metered. No credential rotation. Don't touch live positions.
> Leave paper trading running."

**Derivation to an implementation.** The instruction offers exactly two
admissible branches and forbids the third:

| Branch | Admissible? | Evidence |
|---|---|---|
| Buy Anthropic direct-API credits | **FORBIDDEN** -- "Do NOT buy credits" | operator, verbatim above |
| Move autoresearch off the metered rail | **CHOSEN** | the Max-rail bridge is live and measured working today (finding 7); `run_nightly.sh:78-92` already implements the routing + a loud-fail preflight that can never silently fall back to metered |
| Disable autoresearch | admissible fallback | `launchctl bootout`, or the existing `--preflight-only` $0 mode (`run_memo.py`'s `--preflight-only` branch; line numbers omitted because this step's own edit shifts them -- re-derive with `grep -n 'if args.preflight_only'`) |

Stated in the criterion's own vocabulary, so nothing is left implicit -- the
three options were *buy credits*, *move off the metered rail*, and *disable*.
**DECISION: move off the metered rail.**

Main chooses **move it off the metered rail**, because it is the only branch
that both satisfies `$0 metered` and keeps the nightly research memos -- and
because the disable branch would recreate the very defect this step exists to
fix in mirror image (a job that never runs also never reports). The route is
`AUTORESEARCH_USE_MAX_RAIL` defaulting **on** in tracked code, with the
existing `exit 78` + page preflight as the fail-safe.

**Disclosed cost of the chosen branch (not hidden):** the Max rail is `$0`
*metered* but it is not free -- it draws the same weekly Max plan pool as this
Claude Code session and the harness subagents. A nightly `detailed_report` run
consumes real weekly budget. The operator's constraint is specifically
`$0 metered`, which this satisfies; the weekly-pool draw is a trade-off the
operator should see, and it is recorded here rather than assumed away.

---

## 3. Hypothesis

Two independent defects share this step:

- **H1 (rail).** The nightly run reaches `api.anthropic.com` with a real
  metered key because `AUTORESEARCH_USE_MAX_RAIL` defaults to `0`
  (`run_nightly.sh:78`) and is absent from `backend/.env`. Defaulting it to `1`
  routes every LLM call through the live bridge at `$0` metered and forces the
  dummy key (`:85`), so any leak to `api.anthropic.com` 401s -- provable `$0`.
  The existing preflight (`:80-91`) means a down bridge exits 78 and pages
  rather than falling back to metered.
- **H2 (audibility).** The only failure signal is a bash curl whose result is
  discarded and which no test can capture. Emitting through
  `raise_cron_alert_sync` from inside `run_memo.py`'s own failure path, gated
  by a **directory-derived** consecutive-failure count with an **edge-triggered
  two-tier ladder**, makes the failure audible, escalating, testable, and
  fatigue-free.

Falsifiable predictions:
- Flipping the default and running `run_nightly.sh` against a healthy stub
  bridge exports the bridge URL and the dummy key (already pinned by
  `test_phase_76_9_2_max_bridge.py:337-356`); with the bridge down it exits 78
  and writes the fail-state (`:319-335`).
- A credit-exhaustion exception raised inside `run_research` produces exactly
  one `raise_cron_alert_sync` call at `severity="P1"`; the same fixture with 6
  prior ERROR dates produces a **different, escalated** call; a successful run
  produces **zero** calls.

---

## 4. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "a fixture in which the autoresearch run raises a credit-exhaustion error
   emits an operator-visible alert through the notification path, asserted by a
   test capturing the emitted alert"
2. "a fixture in which N consecutive prior runs already failed escalates rather
   than emitting an identical low-priority notice each day, asserted against a
   fixture directory of prior ERROR files"
3. "a successful run emits NO alert, so the guard cannot pass by always firing"
4. "the decision recorded for the metered-rail question (buy credits, move off
   the metered rail, or disable) is written into the step artifact with the
   operator's verbatim instruction"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_11_autoresearch_failure_paging.py -q`

### How each criterion maps to a seam, and what mutation kills it

Per the standing GUARDS rule -- name the seam the criterion points at and drive
THAT; mutate the PRODUCTION call site.

| # | Seam the criterion names | Guard drives | Production mutant that must kill it |
|---|---|---|---|
| 1 | "the autoresearch run" raising -- i.e. `run_memo._main_async`'s except branch | `_main_async` itself, with only `run_research` patched to raise and `MEMO_DIR` redirected; **no `notify=` injection** | delete the `report_run_outcome(...)` call from the except branch; or downgrade `severity` to `"P2"` (which `alerting.py:219-224` drops) |
| 2 | the tier ladder in `report_run_outcome` | a `tmp_path` directory of `YYYY-MM-DD-ERROR-topicNN.md` files | make the ladder emit the same `error_type`/`severity` regardless of N |
| 3 | the success branch of `_main_async` | `_main_async` with `run_research` returning a body | make the emitter unconditional |
| 4 | this artifact + `experiment_results.md` | a test asserting the verbatim operator sentence is present in the artifact on disk | delete or paraphrase the recorded instruction |

---

## 5. Plan

**D1 -- rail exit (tracked, auditable).**
`scripts/autoresearch/run_nightly.sh:78`: `${AUTORESEARCH_USE_MAX_RAIL:-0}` ->
`${AUTORESEARCH_USE_MAX_RAIL:-1}`, with an inline comment recording the
operator instruction, the date, and the revert (`AUTORESEARCH_USE_MAX_RAIL=0`
in `backend/.env`). Nothing else in that file changes -- the preflight, the
loud-fail `exit 78`, and `_record_fail_and_page` are all untouched.

**D1a -- honour the prior step's pin instead of silently breaking it.**
`test_phase_76_9_2_max_bridge.py` deliberately pinned the default OFF. Two of
its tests observe the default:
- `test_nightly_flag_off_is_inert` builds a fixture with no flag and relies on
  the default. Fix: set `AUTORESEARCH_USE_MAX_RAIL=0` explicitly in that
  fixture, which preserves the guard's actual meaning ("flag OFF is inert")
  and stops it doubling as an accidental default pin.
- `test_nightly_default_documented_off` asserts the literal
  `'AUTORESEARCH_USE_MAX_RAIL:-0'`. Fix: repin to `:-1` and rename to match,
  keeping the guard's protective value (the default is pinned and cannot drift
  silently) while recording *why* the value changed. Its two other assertions
  (the loud-fail echo lives in executed code, `exit 78` present) are untouched.

This is a deliberate supersession of a phase-76.9.2 decision, disclosed here
and repeated in `experiment_results.md`. It is not a silent edit.

**D2 -- the audible, escalating, Python-drivable path.**
New module `backend/services/autoresearch_health.py`, mirroring
`freshness_cron.py`'s shape:
- `count_consecutive_failures(memo_dir, today) -> int` -- walks back from
  `today`; a date is failing iff it has `*-ERROR-*` **and** no non-ERROR,
  non-WARN memo; stops at the first non-failing date. (Directory-derived
  because criterion 2's "fixture directory of prior ERROR files" makes a
  JSON-only implementation undrivable, and because the JSON counter is silently
  reset by any manual run.)
- `classify_failure(exc) -> str` -- narrow, in the style of
  `run_memo._is_network_weather`: returns `"credit_exhausted"` for the measured
  message ("credit balance is too low"), `"auth"` for authentication failures,
  else `"generic"`.
- `report_run_outcome(*, failed, exc, memo_dir, today, notify=None, ...) -> dict`
  -- the tier ladder + edge trigger.
- Fail-open throughout: a notification failure must never change the exit code
  of the nightly job.

**Ladder (each row traceable to a source in the brief):**

| Class | Tier 0 (silent) | Tier 1 (P1) | Tier 2 (P0, distinct `error_type`) |
|---|---|---|---|
| `credit_exhausted` / `auth` (config-class, never self-heals) | -- | `n >= 1` | `n >= escalate_after` (7) |
| `generic` | `n < 3` | `3 <= n < 7` | `n >= 7` |

Config-class errors page on night 1 because SRE Workbook ch.8 requires a page
to be *immediately actionable*, and a dead credit balance is actionable
immediately and will never self-heal -- which is also what criterion 1 demands
(a single credit-exhaustion fixture must alert).

**Edge trigger:** emit iff `tier(n) > tier(n-1)` under the ladder selected by
today's class -- the Alertmanager "notify on state change" semantics.
*(Corrected during GENERATE: an earlier draft of this contract claimed the
trigger was "gap-safe, so a skipped night that jumps 2 -> 4 still fires". That
was FALSE and is retracted. Because the count walks backwards over failing
DATES and stops at the first non-failing one, a night the job did not run is
not a failing date and RESETS the count -- so n never jumps, it advances by 1
or resets. The consequence is that a missed night silently rewinds the ladder;
that is the dead-man's-switch hole, already in non-scope below, and it is now
pinned by a test rather than papered over.)*
**Slow safety net:** additionally emit once every `REMIND_EVERY_DAYS = 7`
while at tier 2 (`(n - escalate_after) % 7 == 0`), which is Alertmanager's
`repeat_interval` role and answers SRE Workbook ch.8's warning against
silencing a recurring page. It is emphatically **not** "an identical
low-priority notice each day" (criterion 2).

**Success emits nothing at all** -- not even a recovery notice. Criterion 3 is
unconditional ("a successful run emits NO alert"), so a recovery notice would
violate it. Recorded here as a deliberate choice, not an oversight; the brief
recommended a recovery notice and the criterion overrules it.

**D3 -- wire it into the production entry point.**
`scripts/autoresearch/run_memo.py::_main_async`: call `report_run_outcome` in
the ERROR branch (after the ERROR file is written, so the count includes today)
and in the success branch. Import is function-local and wrapped so it is
fail-open. The WARN/network-weather branch (`:168-182`) is deliberately left
alone -- it is a tolerated outcome by phase-76.9's design.

**D4 -- the guard file.**
`backend/tests/test_phase_82_11_autoresearch_failure_paging.py`, following the
82.10 idiom:
- `ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"`,
  patched with `autospec=True`.
- A `test_wrong_patch_target_does_not_exist`-shaped anti-vacuity pin.
- Criterion-1 and criterion-3 tests drive `_main_async` with **no `notify=`
  injection**, so production resolves the real emitter.
- Precondition assertions on **both** the failing and the healthy fixture.
- Counting traps pinned: total (62) vs consecutive (12), the both-files date,
  the empty directory.

---

## 6. Non-scope (explicit)

- **The bash seam is NOT removed.** Keeping both while the Python path is
  proven avoids a window with no paging at all. Retiring it is a follow-up.
- **The dead-man's-switch / silent-exit-0 hole is NOT fixed here.**
  `run_memo.py` has three paths that exit 0 with no memo and no alert (the
  phase-76.9 WARN branch, `_embedding_preflight`'s soft-skip, and
  `--preflight-only`; line numbers deliberately NOT cited here because this
  step's own edit shifts them -- `experiment_results_82.11.md` §8 carries the
  re-derived positions), and the 2025-2026 literature says the highest-value
  monitor is "a success artifact dated today exists". That is outside all four
  criteria and was **queued as masterplan step 82.49** (P2, `harness_required`,
  5 immutable criteria) at the end of this cycle -- not a prose mention.
  *(Written during PLAN as an intention; the step id was filled in once 82.49
  actually existed, so this sentence is a record and not a promise.)*
- **No `backend/.env` edit**, no credential rotation, no live positions
  touched, paper trading untouched.
- **No change to `_record_fail_and_page`, the preflight, or `exit 78`.**

## 7. References

- `handoff/current/research_brief_82.11.md` (the gate; 8 sources read in full)
- Google SRE Book ch.6 -- https://sre.google/sre-book/monitoring-distributed-systems/
- Google SRE Workbook ch.5 -- https://sre.google/workbook/alerting-on-slos/
- Google SRE Workbook ch.8 -- https://sre.google/workbook/on-call/
- Prometheus Alertmanager configuration -- https://prometheus.io/docs/alerting/latest/configuration/
- pytest monkeypatch how-to -- https://docs.pytest.org/en/stable/how-to/monkeypatch.html
- Rootly, alert escalation policies (2026) -- https://rootly.com/alert-management/alert-escalation-policies
- OnlineOrNot, cron job monitoring guide -- https://onlineornot.com/cron-job-monitoring-guide
- Internal: `backend/services/freshness_cron.py`,
  `backend/tests/test_phase_82_10_freshness_paging.py` (commit `b7c69bb9`),
  `backend/services/observability/alerting.py:63-107,136-176,179-287`,
  `scripts/autoresearch/run_nightly.sh:49-92`,
  `scripts/autoresearch/run_memo.py` (`_main_async` + the preflight guards --
  PRE-EDIT positions :154-194 / :306-318; this step inserts 35 lines above
  `_main_async`, so re-derive rather than reuse these),
  `backend/tests/test_phase_76_9_2_max_bridge.py:302-370`
