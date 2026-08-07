# Research Brief — Step 85.3 (tier=moderate)

**Topic:** The away-ops auth alarm has reported a credential failure every 30
minutes for ~28 days about a credential that works. Root-cause the latch, design
the exit, preserve real paging.

**Status:** COMPLETE. Tier stated by caller: `moderate`. Not audit-class.
**Date:** 2026-08-07. All internal claims re-derived live; all numbers measured.

---

## 0. HEADLINE — the step's root cause is CORRECT but INCOMPLETE, one premise is REFUTED, and one measured claim is wrong

The step names ONE latch. There are **THREE stacked defects**, two of which are
independent latches-with-no-exit that mutually sustain each other.

### D1 — healthcheck.sh stale-401 latch (the step's stated root cause; CONFIRMED)
`scripts/away_ops/healthcheck.sh:110-120` sorts `session_*.json` by mtime, takes
`sessions[-1]`, greps for `"api_error_status": 401`, and at `:119` sets
`auth_ok=false` when `cleared_at is None or mt > cleared_at`. There is **no upper
bound on the age of `newest`**.

Re-derived live by replaying the heredoc verbatim against the real state dir
(read-only, no writes, no paging):
```
newest=session_am_20260710T053005Z.json  mtime_utc=2026-07-10T05:30:10.372439+00:00  AGE=28d 8h
cleared_at=None  -> condition (cleared_at is None or mt>cleared_at) = True
DERIVED: false 401_in_session_am_20260710T053005Z.json
```
Exactly matches the live `auth_detail` in `health.jsonl`. **Measured evidence age
= 28.35 days** (step said 27; it has aged since).

### D2 — REFUTED PREMISE: `cleared_at` has a SECOND writer, inside healthcheck.sh itself
The step asserts *"The only writer of `cleared_at` is
`scripts/away_ops/run_away_session.sh:151`."* **This is wrong.** A repo-wide grep
finds exactly two live writers: `run_away_session.sh:151` AND
**`healthcheck.sh:164`**, which writes
`{"incident_open": false, "cleared_at": <now>, "cleared_by": "healthcheck_healthy"}`.
(Everything else matching `cleared_at` is archive prose or the unrelated
phase-25.D6 optimizer plateau lock.)

**Why this matters enormously:** the "close path that does not require an away
session to run" demanded by **criterion 5 ALREADY EXISTS** at
`healthcheck.sh:154-166`. It is not missing — it is **unreachable**:

```
if  [...] || [ "$auth_ok" = "false" ];  then   # :125  page + write incident_open:true, NO cleared_at (:151)
elif [ -f "$AUTH_STATE" ];              then   # :154  clear: write cleared_at (:164)
fi
```
Stale 401 → `auth_ok=false` → the `if` arm always wins → `:164` never runs →
`cleared_at` stays absent → `cleared_at is None` at `:119` keeps `auth_ok=false`.
**A closed self-sustaining cycle inside one file.** Corollary: adding the
freshness bound alone flips `auth_ok` true, letting the `elif` fire on the very
next 30-min tick and close the latch through existing code. Criterion 5 is
satisfiable by making an existing path *reachable* and proving it.

### D3 — NEW DEFECT: the wrapper recovery probe is a FALSE-NEGATIVE auth test (this is the answer to criterion 9)
Criterion 9 asks why no `session_*.json` since 2026-07-10 "even though the jobs
show last-exit-status 0". **Measured answer: the jobs run fine, twice a day, and
deliberately `exit 0` on a skip path.** `run_away_session.sh:143-158`: if the
latch is open, spend ONE 20s-capped probe instead of a full launch; clear and
proceed if it passes; else `slog "END session result=auth-dead-skip"; exit 0`.

The gate at `run_away_session.sh:150`:
```bash
if [ "$probe_rc" -eq 0 ] && ! grep -q '"api_error_status": *401' "$OPS/auth_probe_last.json" 2>/dev/null; then
```

**The 401 leg is clean. The `probe_rc` leg is impossible to satisfy.** Verbatim
from `handoff/away_ops/auth_probe_last.json`, written **today 2026-08-07 07:30**:
```json
{"is_error":true,"duration_api_ms":3228,"num_turns":2,"stop_reason":"tool_use",
 "total_cost_usd":0.50494,
 "usage":{"input_tokens":2,"cache_creation_input_tokens":50373,"output_tokens":48,...},
 "modelUsage":{"claude-opus-4-8":{...,"costUSD":0.50494,"canonicalModel":"claude-opus-4-8","provider":"firstParty"}},
 "terminal_reason":"max_turns","subtype":"error_max_turns",
 "errors":["Reached maximum number of turns (1)"],"type":"result"}
```
**No `api_error_status` field. No 401.** The credential AUTHENTICATED — it billed
50,373 cache-creation tokens on `provider: firstParty`. The probe fails because
`printf 'ping'` with `--max-turns 1` makes the model reach for a tool
(`stop_reason:"tool_use"`, `num_turns:2`) → CLI exits **rc=1**,
`subtype: error_max_turns`. The PM slot instead hits **rc=124** — `gtimeout -k 5
20` kills it, because rebuilding a 50K-token prompt cache exceeds the 20s cap.

`handoff/away_ops/session.log:2003-2207` — every slot for the last 6 days:
`probe still failing (rc=1)` or `(rc=124)` → `END session result=auth-dead-skip`.
**Never rc=0, not once in 28 days.**

> **The probe conflates "the CLI exited nonzero" with "the credential is dead."**

### D4 — REFUTED MEASURED CLAIM: "every record in health.jsonl carries ok:false"
Measured: **509 records, 34 ok:true, 475 ok:false.** The trailing ok:false streak
is **473 consecutive records spanning 9.85 days** (2026-07-28T17:09:48Z →
2026-08-07T13:35:14Z) — not 28 days of records. Reason: the away-watchdog was
itself **dead 2026-07-06 → 2026-07-28** (the 17-day outage `scripts/ops/rotate_logs.sh:10`
documents), and `logrotate_alarm_state.json` shows its liveness alarm cleared at
`2026-07-28T17:09:33` — **15 seconds before** the first record of the streak.
Across all 473 streak records the non-auth failure reason set is `AUTH_ONLY`
(computed by re-deriving `ok` without the auth leg): backend/frontend/slack_bot
running, api_health 200, frontend_http 200, adc_ok true, disk 127GB. **Auth is
the sole cause, as the step says.** `auth_p1` is `false` on all 473 — the latch
DID correctly suppress re-paging.

### The coupling (why this is one bug, not three)
```
stale 401 is newest  ──►  healthcheck auth_ok=false  ──►  clear arm (:164) unreachable
        ▲                                                            │
        │                                                            ▼
 no new session file  ◄──  wrapper skips session  ◄──  latch stays open + probe rc!=0
```
Both cycles closed. **Clearing the latch alone also restores the away sessions**,
because `run_away_session.sh:143` only enters the probe block `if incident_open`.
But D3 stays armed: the next real 401 opens the latch and the probe can never
clear it again → a verbatim 28-day repeat. **D3 must be filed.**

---

## 1. Internal code inventory

| File | Lines cited | Role | Status |
|---|---|---|---|
| `scripts/away_ops/healthcheck.sh` | 97-123 heredoc; 110-112 newest-by-mtime; 117 dual 401 spelling; 119 latch cond; 125 page arm; 138-142 Slack POST; 151 open-write; 154-166 clear arm; 164 `cleared_at` write; 263 ok gate; 266-270 jsonl; 272 exit | The alarm; run every 30 min | **D1 + D2** |
| `scripts/away_ops/run_away_session.sh` | 142-158 latch probe; 150 rc gate; 151 `cleared_at` write; 186-204 401 detect/page/open | AM/PM wrapper | **D3** |
| `handoff/away_ops/auth_page_state.json` | whole | Latch state | `{"incident_open": true, "opened_at": "2026-07-10T05:30:11.289372+00:00", "detail": "401 in session_am_20260710T053005Z.json", "paged": true}` — **no `cleared_at` key**. `detail` uses spaces → opened by the **wrapper** (`:201`), not healthcheck (which writes `401_in_`) |
| `handoff/away_ops/session_*.json` | 63 files | 401 evidence corpus | newest `session_am_20260710T053005Z.json` (28.35d); oldest `session_am_20260612T092025Z.json` |
| `handoff/away_ops/auth_probe_last.json` | whole | Wrapper probe output | **mtime today 07:30** — proves probe runs, credential ALIVE |
| `handoff/away_ops/health.jsonl` | 509 lines | Alarm log | 473-record ok:false streak; **line 37 is INVALID JSON** (see DD-1) |
| `handoff/away_ops/session.log` | 2003-2207 | Wrapper log | `auth-dead-skip` every slot |
| `~/Library/LaunchAgents/com.pyfinagent.away-watchdog.plist` | whole | Runs healthcheck | `StartInterval 1800`; `launchctl list` → **last-exit-status 1** (confirmed) |
| `com.pyfinagent.away-session-{am,pm}.plist` | StartCalendarInterval | Session cadence | **AM 07:30, PM 22:00 local, daily, no weekday filter**; both last-exit-status 0 |
| `scripts/ops/rotate_logs.sh` | 83-135 | 2nd watchdog: pages if health.jsonl >2h stale (`SRE_OPS_STALE_THRESHOLD_S:-7200`), latch in `logrotate_alarm_state.json` | Healthy; **must keep receiving fresh health.jsonl** |
| `backend/slack_bot/scheduler.py` | 513 | Reads `lines[-1]` of health.jsonl into the digest | **Operator-visible alarm surface** |
| `backend/slack_bot/formatters.py` | 117-127 | Renders `System health: DEGRADED at <ts>` | The 28-day banner; **DD-2 field-name bug** |
| `docs/runbooks/credential-expiry-monitoring.md` | 11-36, 51, 73 | Governing contract | `:73` "the next session slot's probe clears the latch automatically" is **now FALSE** (D3) |
| `docs/runbooks/away-ops-rules.md` | 25 | Rails | "launchctl authority limited to backend/frontend/slack-bot kickstart; **never touch watchdog/session plists**" |
| `backend/tests/test_phase_75_sre_ops.py` | 50-73 | Only test referencing healthcheck.sh | **Source-text assertions only — no test drives the auth derivation today** |

### 1.4 The 401 evidence (verbatim, `session_am_20260710T053005Z.json`)
```json
{"type":"result","subtype":"success","is_error":true,"api_error_status":401,
 "result":"Failed to authenticate. API Error: 401 Invalid authentication credentials",
 "total_cost_usd":0,"num_turns":1,...}
```
The 2026-07-10 incident was **REAL** (`total_cost_usd:0`, zero tokens — never
reached inference), paged once by the wrapper, and was resolved later by
re-login. Nothing wrote `cleared_at` because both writers were blocked (D2
unreachable, D3 rc-gated). Note `subtype:"success"` on a 401 — phase-66.4's
"never key on subtype" rule remains correct.

### 1.5 Live measurement decisive for criterion 8
```
claude auth status  ->  rc=0   (auth_status_ok=true)
```
**The `auth_status_rc_nonzero` leg passes today.** The stale 401 is the *only*
thing holding `auth_ok=false`, so a freshness bound alone makes the live
watchdog emit `ok:true`. Criterion 8 is achievable.

---

## 2. External research

### 2.1 Read in full (7; gate floor is 5)
| # | URL | Accessed | Kind | Tier | Key finding |
|---|---|---|---|---|---|
| 1 | https://assets.nagios.com/downloads/nagioscore/docs/nagioscore/4/en/freshness.html | 2026-08-07 | Official docs | 2 | The canonical freshness pattern, verbatim: *"If the age of the last check result is greater than the freshness threshold, the check result is considered 'stale'"*; on staleness *"Nagios Core will force an active check … even if active checks are disabled"*; *"It is recommended that you explicitly specify a freshness threshold, rather than let Nagios Core pick one for you."* Numeric example: **93600s = 26h** for a job whose finish time varies. |
| 2 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-07 | Official/canonical | 2 | *"Will I ever be able to ignore this alert, knowing it's benign?"*; *"Every page should be actionable"*; *"Pages with rote, algorithmic responses should be a red flag"*; *"I can only react with a sense of urgency a few times a day before I become fatigued."* |
| 3 | https://prometheus.io/docs/practices/alerting/ | 2026-08-07 | Official docs | 2 | *"avoid having pages where there is nothing to do"*; *"Aim to have as few alerts as possible"*; *"Allow for slack in alerting to accommodate small blips."* |
| 4 | https://www.ncbi.nlm.nih.gov/books/NBK555522/ | 2026-08-07 | Peer-reviewed (AHRQ *Making Healthcare Safer III*) | 1 | *"the percentage of false alarms can range from **72 percent to 99 percent**"*; *"staff doubt the reliability of alarms and as a result turn down the volume, ignore, or deactivate the alarms"*; FDA MAUDE received **566 reports of patient deaths** related to monitoring device alarms 2005-2008. |
| 5 | https://www.yokogawa.com/library/resources/faqs/ns-faq-ut-20013-term/ | 2026-08-07 | Vendor official | 2 | *"the alarm latch function maintains the alarm output continuously … until the alarm latch clear command."* A latch is BY DESIGN non-self-clearing; the explicit clear is the whole point. |
| 6 | https://learn.microsoft.com/en-us/azure/well-architected/reliability/monitoring | 2026-08-07 (doc updated **2026-05-21**) | Official docs | 2 | *"**Equate incorrect or stale values to a reliability issue**, requiring the same level of rigor"*; concrete pattern: *"A fixed-interval comparison of the dashboard count against a **live query** of the order table catches the gap"*; *"triggering on meaningful health state changes rather than isolated blips"*; *"poorly tuned thresholds can create noise and reduce effectiveness."* |
| 7 | https://oneuptime.com/blog/post/2026-01-30-alert-lifecycle-management/view | 2026-08-07 (pub **2026-01-30**) | Vendor blog | 4 | Alert-lifecycle staleness is framed only as *"Alert has not fired in N days → consider deprecation"*. **Negative finding:** it does **not** treat auto-resolve or never-resolving alerts at all — the inverse failure (an alert that never STOPS firing) is under-covered in 2026 lifecycle-management writing. |

### 2.2 Attempted but not readable (do NOT count toward the gate)
| URL | Why not read |
|---|---|
| https://arxiv.org/html/2604.17836 | HTTP 404 on the arXiv native-HTML path (per `.claude/rules/research-gate.md` §PDF chain step 1); not escalated — off the critical path |
| https://techdocs.broadcom.com/.../persistent-and-stale-alarms.html | HTTP 404 |

### 2.3 Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://psnet.ahrq.gov/perspective/reducing-safety-hazards-monitor-alert-and-alarm-fatigue | Gov review | Duplicates source #4's evidence |
| https://array.aami.org/doi/full/10.2345/0899-8205-46.4.268 | Peer-reviewed | Paywall-class; #4 covers it |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11851092/ | Peer-reviewed 2025 | Recency snippet retained (§2.4) |
| https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10001798/ | Peer-reviewed | Redundant |
| https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7270842/ | Peer-reviewed | False-alarm ML detection — out of scope |
| https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3752621/ | Peer-reviewed | Sensor correlation — out of scope |
| https://arxiv.org/pdf/2604.16081 | Preprint 2026 | Multi-agent FP suppression; adjacent |
| https://arxiv.org/pdf/2603.09002 | Preprint 2026 | MAS security; off-topic |
| https://arxiv.org/pdf/2604.17836 | Preprint 2026 | 404 on HTML (above) |
| https://forum.checkmk.com/t/removing-stale-service-alerts/28267 | Community | Tier 5 |
| https://github.com/SignalK/signalk-server/issues/2350 | Community | Tier 5, but a good statement of the dual failure: stale data makes zone alarms never fire |
| https://techdocs.broadcom.com/.../persistent-and-stale-alarms.html | Vendor | 404 |
| https://www.logicmonitor.com/support/alert-lifecycle | Vendor | Lifecycle overlap with #7 |
| https://rootly.com/sre/alert-management-tools-compared-what-to-use-in-2026 | Vendor 2026 | Tooling comparison |
| https://sreschool.com/blog/alert/ | Blog 2026 | Recency snippet |
| https://www.vectra.ai/topics/alert-fatigue | Vendor | Security-domain alert fatigue |
| https://oneuptime.com/blog/post/2026-02-13-aws-cloudwatch-alerting-best-practices/view | Vendor 2026 | CloudWatch-specific |
| https://www.machinecdn.com/blog/alarm-management-software-manufacturing/ | Vendor 2026 | ISA-18.2 adjacent |
| https://www.qcecuring.com/education/clm/certificate-monitoring-and-alerting | Vendor | Cert-expiry analogue |
| https://oneuptime.com/blog/post/2026-03-31-redis-data-expiration-strategy-best-practices/view | Vendor 2026 | TTL mechanics |
| https://www.psqh.com/news/battling-alarm-fatigue-for-improved-patient-care-and-safety/ | Trade press | Tier 5 |
| https://nurse.org/articles/alarm-fatigue-statistics-patient-safety/ | Trade press | Tier 5 |
| https://assets.nagios.com/downloads/nagioscore/docs/nagioscore/3/en/freshness.html | Official docs | v3 duplicate of #1 |
| https://nagios.fm4dd.com/docs/en/freshness.shtm | Mirror | Duplicate |
| http://nagios.manubulon.com/traduction/docs14en/freshness.html | Mirror | Duplicate |
| http://sentry2.unina.it/nagios/docs/freshness.html | Mirror | Duplicate |
| https://support.nagios.com/forum/viewtopic.php?f=7&t=46909 | Community | Tier 5 |
| https://support.nagios.com/forum/viewtopic.php?f=7&t=35658 | Community | Tier 5 |

**Total unique URLs collected: 34** (7 read in full + 2 failed + 25 snippet-only).

### 2.4 Search-query composition (3-variant discipline, per `.claude/rules/research-gate.md`)
| Variant | Query run |
|---|---|
| Year-less canonical | `alert latch auto-clear stale evidence expiry monitoring design` |
| Year-less canonical | `alarm fatigue desensitization false alarm rate percentage clinical alarms study` |
| Year-less canonical | `Nagios check freshness threshold stale passive check monitoring` |
| Current-year frontier / last-2-year | `stale alarm auto-resolve monitoring 2026 alert lifecycle resolve condition no longer observed` |

### 2.5 Recency scan (2024-2026) — PERFORMED
**Result: 2 new findings that COMPLEMENT (do not supersede) the canonical sources.**
1. **Azure WAF `reliability/monitoring`, doc-dated 2026-05-21** (read in full, #6) —
   the most current authoritative statement of the exact principle at issue:
   *"Equate incorrect or stale values to a reliability issue."* It also supplies
   the remediation shape this step needs: compare the cached/derived value
   against a **live query** on a fixed interval. This is 2026 guidance and it
   post-dates and reinforces the 2000s-era Nagios freshness design.
2. **2026 alert-lifecycle writing is silent on the never-resolving alert**
   (read in full, #7; corroborated by the Rootly/LogicMonitor/SRE-School
   snippets). Lifecycle frameworks published in 2026 handle "alert never fires →
   deprecate" but not "alert never stops firing → the instrument is destroyed."
   **This is a genuine gap in the recent literature**, which is why the
   load-bearing guidance for this fix still comes from Nagios freshness (canonical),
   Google SRE (canonical), and process-industry latch semantics (Yokogawa).
Nothing found in the 2024-2026 window overturns the canonical sources; the
alarm-fatigue evidence base (#4, AHRQ 2020) remains the standard citation and
2025 studies (e.g. PMC11851092) reproduce rather than revise it.

---

## 3. Key findings

1. **A "stale evidence" bound is the canonical, named pattern — and staleness
   must trigger an ACTIVE re-check, not a silent pass.** *"If the age of the last
   check result is greater than the freshness threshold, the check result is
   considered 'stale' … Nagios Core will force an active check of the host or
   service."* (Nagios Core 4 docs, accessed 2026-08-07). pyfinagent's bug is the
   textbook case: an unbounded-age passive result gating a health state.
2. **The threshold must be explicit and justified, not inferred.** *"It is
   recommended that you explicitly specify a freshness threshold, rather than let
   Nagios Core pick one for you."* Their own worked example is **26 hours** for a
   job with variable completion time (ibid.). pyfinagent already uses 26h for
   cycle freshness (`healthcheck.sh:76`), so a named constant is idiomatic here.
3. **By Google's own disqualifying test, this rule should not exist in its
   current form.** *"Will I ever be able to ignore this alert, knowing it's
   benign?"* — for 28 days the answer has been yes, every 30 minutes. *"Pages
   with rote, algorithmic responses should be a red flag."* (Google SRE Book,
   accessed 2026-08-07).
4. **The harm is quantified in the safety literature and it is desensitization,
   not annoyance.** *"staff doubt the reliability of alarms and as a result turn
   down the volume, ignore, or deactivate the alarms"*; false-alarm rates of
   **72-99%** are associated with **566 FDA MAUDE death reports** (AHRQ, *Making
   Healthcare Safer III*, accessed 2026-08-07). pyfinagent's instance is the
   degenerate limit: **100% false for 28 consecutive days**. The step's framing
   ("a destroyed instrument") is literature-supported, not rhetorical.
5. **Keeping the latch is CORRECT; the defect is the unreachable clear.** *"the
   alarm latch function maintains the alarm output continuously … until the alarm
   latch clear command"* (Yokogawa, accessed 2026-08-07). Do **not** replace the
   latch with a self-clearing alarm — that would restore re-page spam, which
   `healthcheck.sh:90-93` deliberately prevents. Fix the clear command's
   reachability.
6. **Stale operator-facing state is itself a reliability defect, and the fix is a
   live comparison.** *"Equate incorrect or stale values to a reliability issue"*;
   *"a fixed-interval comparison of the dashboard count against a live query …
   catches the gap"* (Azure WAF, doc-dated 2026-05-21). Direct mandate for
   "stale evidence → fall through to the live `claude auth status` probe."
7. **Recent (2026) alert-lifecycle literature does not cover the
   never-resolving alert** (OneUptime 2026-01-30, read in full) — a documented
   gap, recorded per the recency-scan requirement.

---

## 4. Consensus vs debate (external)

**Consensus (unanimous across tiers 1-2):** age-bound your evidence; make
thresholds explicit; an alarm that cannot be ignored-when-benign is
mis-specified; stale state is a reliability bug.

**Genuine debate — "auto-clear vs explicit clear":** Yokogawa/process-industry
says a latch must require an explicit clear (safety: force acknowledgement).
Modern SRE lifecycle tooling leans on auto-resolve when the condition is no
longer observed. **Resolution for this step: adopt BOTH, at different layers** —
the *evidence* gets a max-age (auto-expiring input), while the *incident latch*
keeps requiring an explicit clear write. That is not a compromise; the two
mechanisms address different failure modes (stale input vs. re-page spam), and
D1/D2 shows pyfinagent broke the first while correctly implementing the second.

---

## 5. Pitfalls (from literature + the measured code)

1. **Do not make staleness mean "healthy."** Nagios forces an *active* check.
   Falling through to an unconditional `auth_ok=true` would create a real blind
   spot: a dead credential plus no new session files would read healthy.
2. **Do not remove the latch** — `healthcheck.sh:90-93` documents why (the tail-1
   dedupe re-pages every other run). Yokogawa confirms latch semantics are the
   right primitive.
3. **Do not let the freshness bound weaken a FRESH 401.** This is the whole point
   of mutation M1 in criterion 7.
4. **Do not auto-calculate the window.** Nagios explicitly recommends against it.
5. **Poorly tuned thresholds create their own noise** (Azure WAF) — a window
   shorter than the real session cadence would flap.
6. **Fail-open must survive** (`healthcheck.sh:97` `|| echo "unknown probe_error"`
   + `:263` comment). A seam that raises must still yield `unknown`, never `false`.

---

## 6. Application to pyfinagent

### 6.1 Recommended design (literature-anchored)
**Adopt BOTH evidence max-age AND the existing explicit clear — plus an active
re-check on staleness.** Three layers, mapped to sources:

| Layer | Mechanism | Basis |
|---|---|---|
| L1 Evidence max-age | `AUTH_EVIDENCE_MAX_AGE_S` named constant; a 401 in a session file older than the window **cannot gate health** | Nagios freshness_threshold ("explicitly specify"); Azure WAF "equate stale values to a reliability issue" |
| L2 Active re-check on staleness | When evidence is stale, fall through to the **already-computed** `auth_status_ok` (`healthcheck.sh:96`, `claude auth status`) rather than to an unconditional pass | Nagios "force an active check"; Azure WAF "comparison … against a live query" |
| L3 Explicit latch clear | Keep the latch; make `healthcheck.sh:164` reachable and let the seam own the transition so it is testable | Yokogawa latch semantics; `healthcheck.sh:90-93` re-page prevention |

**Window recommendation: 36h (129600s).** Derivation (state it in the code
comment, per Nagios): sessions start 07:30 and 22:00 local daily → worst-case
inter-start gap **14.5h**; the session file is written at the END and the cap is
`14400s` (4h) → worst-case normal mtime gap ≈ **18.5h**. 36h ≈ 2x that, tolerating
one fully missed slot without flapping. **26h (93600s) is the defensible
alternative** — it matches Nagios's own worked example *and* the repo's existing
cycle-freshness constant at `healthcheck.sh:76` — but it leaves only ~7.5h of
headroom over the worst-case normal gap. Either is arguable; the contract must
pick one and record the derivation.

### 6.2 What a REAL 401 must still do — the paging path, proved
There are **three independent detection legs**, and the fix touches only the
staleness of one input to leg (a):

| Leg | Trigger | Path | Touched by the fix? |
|---|---|---|---|
| (a) healthcheck 401 scan | FRESH `session_*.json` with 401 → within window → `auth_ok=false` | `healthcheck.sh:125` → latch check `:126` → bot-token `chat.postMessage` to `C0ANTGNNK8D` `:138-142` → `auth_p1=true` `:150` → latch open `:151`; `ok=false` at `:263`; exit 1 at `:272` | **Input age-bounded only.** A fresh 401 is by definition inside the window → unchanged |
| (b) healthcheck local status | `claude auth status` rc≠0 → `detail=auth_status_rc_nonzero` at `:101-102`, **independent of any session file** | same page path | **Untouched.** Measured rc=0 today |
| (c) wrapper in-session 401 | session dies rc≠0 with 401 → `run_away_session.sh:186-204` pages once + opens latch | independent Slack POST | **Untouched** |

So a genuine credential death still pages via (a), (b), and (c). The
**latch-exit guard** and the **real-incident-still-pages guard** named by the
caller map to mutations M1 and M6/M5 in §6.4.

### 6.3 Criterion 5 is satisfiable by an existing mechanism
The close path is `healthcheck.sh:154-166` (writes `cleared_at`, `cleared_by:
"healthcheck_healthy"`), driven by a **successful local auth probe** — which is
exactly "a successful auth probe results in the incident being cleared", and it
requires **no away session**. The work is (i) make it reachable and (ii) move the
transition into the extracted seam so a test can drive it.

### 6.4 Test strategy — named mutations
Seam contract (recommended): `scripts/away_ops/auth_state.py`, invoked by
`healthcheck.sh` via `python3 scripts/away_ops/auth_state.py --ops … --state …
--status-ok … --now … [--apply]`, printing `"<auth_ok> <auth_detail>"` on stdout
(same two-token contract the existing `read -r auth_ok auth_detail` consumes, so
the call site changes minimally and the `|| echo "unknown probe_error"`
fail-open wrapper is preserved verbatim).

| ID | Mutation | Must turn RED | Must stay GREEN |
|---|---|---|---|
| **M1** | Delete the freshness bound (criterion 7's required mutation) | stale-401 case + regression fixture | fresh-401 case |
| **M2** | Invert the age comparison (`<` → `>`) | stale-401 + fresh-401 | — |
| **M3** | Widen window to ~100y | stale-401 + regression | fresh-401 |
| **M4** | Shrink window to 0 | fresh-401 (now wrongly "stale") | — |
| **M5** | Drop the `auth_status_ok` leg (L2 active re-check) | dead-status-with-stale-evidence case | — proves L2 is load-bearing, not decorative |
| **M6** | Make the clear unconditional | **latch-exit guard** (must not clear while a FRESH 401 stands) | — |
| **M7** | Replace the fail-open `except` with a `raise` | fail-open case (criterion 6) | — |

### 6.5 Fixture strategy (the decisive detail)
- **Freeze `now`.** Pass an explicit `--now` ISO argument to the seam. A test
  that computes age from `datetime.now()` against a hardcoded 2026-07-10 mtime
  will drift and eventually pass/fail for the wrong reason. Criterion 4 says
  *"27 days before the evaluation instant"* — that phrasing **requires** a
  relative construction: build the fixture mtime as `now - timedelta(days=27)`
  (or 28 to match today's measured 28.35d) with `now` injected.
- **Set mtime explicitly with `os.utime()`.** mtime is BOTH the newest-file
  selector (`:110`) AND the age input (`:118`) — never inherit it from write order.
- **≥3 session files with distinct mtimes** so the `sorted(...)[-1]` selection is
  actually exercised; a one-file fixture cannot catch a selection bug.
- **Both JSON spellings.** `healthcheck.sh:117` checks `"api_error_status": 401`
  *and* `"api_error_status":401`; `run_away_session.sh:150/:186` uses the regex
  `"api_error_status": *401`. Cover both, and include a fresh CLEAN session file.
- **Real-shape payload.** Copy the verbatim §1.4 body — including
  `"subtype":"success"` — so a future "key on subtype" regression is caught.
- **`tmp_path` only.** Criterion 2 forbids reading real `handoff/away_ops/`
  artifacts; assert that too (the test should not resolve the repo OPS dir).
- **State-file variants:** `incident_open:true` + no `cleared_at` (the regression
  case); `cleared_at` present; missing file; malformed/truncated JSON;
  unreadable/`chmod 000` (criterion 6).

### 6.6 live_check plan (criterion 8)
1. **Quiet proof.** Run `bash scripts/away_ops/healthcheck.sh` directly (this IS a
   real invocation; `away-ops-rules.md:25` forbids touching the watchdog *plist*,
   not running the script). It will NOT page — the Slack POST lives only inside
   the `auth_ok=false` arm. Then quote `tail -1 handoff/away_ops/health.jsonl`
   verbatim showing `"ok": true, "auth_ok": "true", "auth_detail": "ok"`, plus
   `echo $?` = 0. Also quote one **naturally scheduled** record from the next
   30-min tick, so the evidence is not only a hand-run.
2. **Latch-clear proof.** `cat handoff/away_ops/auth_page_state.json` afterwards
   showing `incident_open:false` + a populated `cleared_at` +
   `cleared_by:"healthcheck_healthy"` — written by `:164` on that same run. (This
   mutates a live state file; that is the intended production effect. Non-scope
   forbids editing historical `session_*.json`, not the latch.)
3. **Real-incident drill (the "still pages" proof).**
   `HEALTHCHECK_TEST_AUTH_P1=1 bash scripts/away_ops/healthcheck.sh` → expect
   `AUTH_P1_TEST_DELIVERY=true` on stdout, the `[DRILL 66.4]` message delivered
   to `C0ANTGNNK8D`, **no latch write**, and no `auth_p1` in the JSON line
   (`healthcheck.sh:133-148`). This is the sanctioned existing drill — use it
   rather than inventing a new paging test.
4. **Session-resumption proof.** The next AM/PM slot should now emit a NEW
   `session_*.json` instead of `END session result=auth-dead-skip`, because
   `run_away_session.sh:143` only enters the probe block `if incident_open`.
   Quote the `session.log` line.
5. **`launchctl list | grep away-watchdog`** flipping from last-exit-status **1**
   to **0**.

### 6.7 Discovered defects to DISCLOSE (file separately; do not widen 85.3)
- **DD-1 (file as its own step — criterion 9's "distinct defect" clause):** D3,
  the wrapper probe's `probe_rc -eq 0` gate. Fix shape: gate on **401-absence**,
  not exit code (a healthy credential legitimately exits 1 on
  `error_max_turns` and 124 at the 20s cap given a measured 50,373-token
  cache-creation prompt). Also raise/justify the 20s cap. Until fixed, the next
  real incident repeats this outage verbatim.
- **DD-2:** `health.jsonl` line 37 is **invalid JSON** — `"api_health":000000`
  (curl's `-w '%{http_code}'` prints `000` on failure AND `|| echo 000` fires).
  `scheduler.py:513` does `_json.loads(lines[-1])` inside a bare `except: pass`,
  so if a malformed line lands last the **entire System-health digest section
  silently disappears**.
- **DD-3:** `formatters.py:121` reads `h.get('last_cycle_age_h')` but
  `healthcheck.sh:266` emits `cycle_age_h` → the digest permanently prints
  `last_cycle_age_h=?`.
- **DD-4:** `com.pyfinagent.away-watchdog.plist` embeds a literal
  `CLAUDE_CODE_OAUTH_TOKEN` **value** in `EnvironmentVariables` (secret at rest in
  a LaunchAgent plist). Operator-facing; out of scope here.
- **DD-5:** `healthcheck.sh:154`'s clear arm also fires when `auth_ok == "unknown"`,
  so a probe error clears a real incident. Fails toward re-paging (not silence),
  so low severity — but the seam should clear only on `auth_ok == true`.
- **DD-6:** `docs/runbooks/credential-expiry-monitoring.md:73` instructs the
  operator *"Nothing else to do: the next session slot's probe clears the latch
  automatically"* — **false today** because of D3. Correct it when D3 is fixed.

---

## 7. Research Gate Checklist

**Hard blockers**
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch — **7**
- [x] 10+ unique URLs total — **34**
- [x] Recency scan (2024-2026) performed + reported — §2.5, 2 findings
- [x] Full pages read (not abstracts) for the read-in-full set — 2 failed fetches
      disclosed in §2.2 and excluded from the count
- [x] file:line anchors for every internal claim — §1, §6

**Soft checks**
- [x] Internal exploration covered every relevant module (healthcheck, wrapper,
      both runbooks, both launchd units, rotate_logs, digest consumer, tests)
- [x] Contradictions / consensus noted — §4 (auto-clear vs explicit clear)
- [x] All claims cited per-claim with URL + access date

**Protocol notes:** three step premises were tested against the code and state.
One is REFUTED (`cleared_at` sole writer — D2), one measured claim is REFUTED
("every record carries ok:false" — D4), and criterion 9's open question is
ANSWERED with a measured root cause (D3). No production file was modified; the
heredoc was replayed read-only in a scratch process, and `healthcheck.sh` was
never executed.

---

## 8. Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 25,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The stale-401 latch is confirmed, but two step premises are refuted and a third defect found. cleared_at has a SECOND writer at healthcheck.sh:164 -- the criterion-5 close path already exists and is merely unreachable, because auth_ok=false always wins the if-arm that precedes it. health.jsonl is not all-false: 34 of 509 records are ok:true; the streak is 473 records over 9.85 days because the watchdog was itself dead 07-06..07-28. Criterion 9 is answered: run_away_session.sh:150 gates recovery on probe_rc==0, but a HEALTHY credential exits 1 (error_max_turns) or 124 (20s cap vs a 50,373-token cache build) -- measured in today's auth_probe_last.json, which shows no 401 at all. Literature (Nagios freshness, Google SRE, Yokogawa latch, Azure WAF 2026, AHRQ alarm fatigue) supports BOTH an explicit evidence max-age AND an active re-check on staleness, keeping the latch. claude auth status returns rc=0 today, so a freshness bound alone makes the live watchdog report ok:true.",
  "brief_path": "handoff/current/research_brief_85.3.md",
  "gate_passed": true
}
```
