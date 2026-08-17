# live_check -- step 86.69 (2026-08-17; exits unpiped)

## 1. The 61.2 suite, fresh at write time

```
$ python -m pytest backend/tests/test_phase_61_2_decision_integrity.py -q --no-header > /tmp/t612.txt 2>&1; echo T612_EXIT=$?
T612_EXIT=0
$ tail -1 /tmp/t612.txt
33 passed, 1 warning in 2.34s
$ grep -n "integrity_enabled=False\|integrity_enabled=True\|def test_flag_off" backend/tests/test_phase_61_2_decision_integrity.py | head -6
32:        paper_synthesis_integrity_enabled=False,
119:        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_ok)
129:        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_ok)
134:        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_fail)
140:    def test_flag_off_legacy_fabrication_unchanged(self):
150:        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_fail)
```

Both flag states are exercised by the suite (ON: routes + honest degraded
row; OFF: the pinned legacy fabrication).

## 2. The token protocol, as it actually ran (blocks and all)

1. First `>> backend/.env` attempt: **BLOCKED** by pre-tool-use-danger
   ("backend/.env write without a fresh operator token (away-ops rail 1)").
   The gate fired exactly as designed.
2. Compound record+touch+write call: **BLOCKED again** (the hook scans the
   whole command string) -- the cursor touch must precede in its own call.
3. Token `ARM-SYNTHESIS-INTEGRITY-86.69` appended to
   `handoff/away_ops/pending_tokens.json` with the operator's verbatim
   AskUserQuestion answers; `touch handoff/away_ops/tokens_cursor`
   (mtime Aug 17 15:05:59).
4. `printf '...PAPER_SYNTHESIS_INTEGRITY_ENABLED=true\n' >> backend/.env`
   -> `ENV APPENDED OK`.

## 3. The restart and the in-force chain

```
$ launchctl kickstart -k gui/501/com.pyfinagent.backend && echo KICKSTART OK
KICKSTART OK
old pid=47562 new pid=14280
PID CHANGED -- restart real
$ curl -s -m 10 http://localhost:8000/api/health | head -c 120
{"status":"ok","service":"pyfinagent-backend","version":"6.93.228",...
$ ps -o pid,lstart -p 14280
14280 man. 17 aug. 15.06.17 2026
$ stat -f "env mtime: %Sm" backend/.env
env mtime: Aug 17 15:06:04 2026
$ python3 -c "from backend.config.settings import Settings; print(Settings().paper_synthesis_integrity_enabled)"
True
```

env write 15:06:04 < process start 15:06:17; the same loader on the same
file yields True; `GET /api/settings/` does NOT expose the flag (measured:
the only match for synthesis/integrity is `max_synthesis_iterations`), so
the chain above IS the in-force evidence -- stated as a chain.

## 4. The frozen baselines and their queries (for the post-arm re-run)

Population A: `sunny-might-477607-p8.financial_reports.analysis_results`,
rows by `DATE(analysis_date)`; zero-score = `final_score = 0.0`.

```sql
-- shares by regime (PRE ..2026-06-10 / POST 2026-06-11..)
SELECT CASE WHEN DATE(analysis_date) <= '2026-06-10' THEN 'PRE' ELSE 'POST' END AS regime,
       COUNT(*) AS n,
       COUNTIF(final_score = 0.0) AS zero_score,
       ROUND(100*COUNTIF(final_score = 0.0)/COUNT(*), 1) AS zero_pct
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE DATE(analysis_date) BETWEEN '2026-05-01' AND CURRENT_DATE()
GROUP BY regime;
-- baselines at capture: PRE 95/251 = 37.8%; POST 211/260 = 81.2% (q1 doc,
-- re-verified by the research gate). Post-arm rows accrue from the
-- 2026-08-17 evening cycle; the same query then reports the third bucket
-- by adding: WHEN DATE(analysis_date) >= '2026-08-18' THEN 'POST_ARM'.
```

BUY-conversion halves (criterion 5) use the same population with
`UPPER(recommendation) LIKE '%BUY%'` among `final_score > 0` rows.

## 5. TIMEZONE CORRECTION to sections 2-3 (measured 2026-08-17T18:2xZ)

**The `15:0x` timestamps in sections 2 and 3 above are LOCAL (CEST = UTC+2),
not UTC, because `stat -f %Sm` and `ps -o lstart` both print local time.**
This correction REPLACES any reading of them as Z; it does not accompany it.

```
$ date -u -r $(stat -f %m backend/.env) +"%Y-%m-%dT%H:%M:%SZ"
2026-08-17T13:06:04Z          # section 3 wrote "Aug 17 15:06:04" -- same instant, local
$ ps -o lstart= -p 41635
man. 17 aug. 15.57.16 2026    # local; nb_NO month abbrev defeats `date -jf` parsing
```

Consequences, both stated rather than implied:

1. **The true arm instant is `2026-08-17T13:06:17Z`** (pid 14280's start),
   not 15:06Z. A post-arm query keyed on `>= 15:06:17 UTC` would silently
   drop the 13:06-15:06Z window. **Today that window is empty and this cost
   no evidence** -- measured, not assumed: `analysis_results` holds ZERO rows
   for 2026-08-15, -16 and -17 (last row `2026-08-14 19:32:26Z`), so both
   cutoffs return the same empty set. The correction is recorded because the
   NEXT reader would not know that.
2. **The process holding the flag is no longer pid 14280.** A later restart
   replaced it with **pid 41635, started `13:57:16Z`** (derived from
   `ELAPSED=04:27:03` read at `18:24:10Z` -> `13:57:07Z`, consistent with the
   local `15:57:16` reading; the drift is the read gap, not a discrepancy).
   That start is still AFTER the `13:06:04Z` env write, so the in-force chain
   of section 3 survives the restart. Re-verified against the RUNNING system
   rather than the file:

```
$ grep -n "PAPER_SYNTHESIS_INTEGRITY_ENABLED" backend/.env
88:PAPER_SYNTHESIS_INTEGRITY_ENABLED=true
$ python3 -c "from backend.config.settings import Settings; print(Settings().paper_synthesis_integrity_enabled)"
True
```

## 6. Post-arm cycle status (open at write time)

As of `2026-08-17T18:24Z` the first post-arm cycle **has not yet run**.

```sql
SELECT COUNT(*) FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('2026-08-17 13:06:17 UTC');
-- 0
```

Last three write-days observed: `2026-08-13` (6 rows) and `2026-08-14`
(6 rows), nothing on the 15th/16th (weekend) or the 17th so far. Historical
write window is `18:35Z-19:32Z`. The measurement sections of this step's
GENERATE stay OPEN until rows exist; **an empty post-arm population is not a
result and must not be reported as one.**

### 6a. The first post-arm cycle IS IN FLIGHT (captured 18:25:54Z)

This is ephemeral state -- the lock is released at cycle end -- so it is
recorded here at observation time rather than reconstructed afterwards.

```
$ cat handoff/.autonomous_loop.lock
{"pid": 41635, "cycle_id": "cycle-1786989600", "started_at": "2026-08-17T18:00:00.007019+00:00", "state": "held"}
$ date -u -r $(stat -f %m handoff/.autonomous_loop.lock) +"%Y-%m-%dT%H:%M:%SZ"
2026-08-17T18:00:00Z
$ date -u +"%Y-%m-%dT%H:%M:%SZ"
2026-08-17T18:25:54Z
```

Three things this pins, none of them inferred from `last_result`:

1. **The holder is pid 41635** -- the SAME process proven above to load
   `paper_synthesis_integrity_enabled=True`. The cycle is therefore running
   WITH the flag armed; it is a post-arm cycle by construction, not by
   timestamp comparison.
2. **The start instant `18:00:00Z` matches the scheduled trigger exactly.**
   Read from source rather than assumed: `backend/api/paper_trading.py:1439`
   registers `_scheduled_run` as `cron(hour=settings.paper_trading_hour,
   minute=0, day_of_week='mon-fri', timezone=America/New_York)`, and the
   running loader reports `paper_trading_hour = 14`. 14:00 EDT = 18:00Z, and
   2026-08-17 is a Monday.
3. **`state: held` at 18:25:54Z means the cycle had not finished** -- so the
   zero-row reading in section 6 is a cycle still in progress, NOT a cycle
   that ran and produced nothing. The distinction decides whether an empty
   population is evidence or an artifact of when the query was run.

A poller (`analysis_date >= 2026-08-17T13:06:17Z`, the corrected arm instant)
runs until rows appear; the measurement sections are completed from the rows
themselves, not from this note.

### 6b. THE FIRST POST-ARM ROWS AND WHAT THEY DO **NOT** SHOW (18:47Z)

Three rows had landed when this was written (the cycle was still running):

| ts (Z) | ticker | final_score | recommendation | `_path` | `rail` | `_fallback_reason` | `_degraded` |
|---|---|---|---|---|---|---|---|
| 18:34:42 | SNDK | 6.68 | Hold | **full** | claude_code | NULL | NULL |
| 18:36:00 | 009150.KS | 4.92 | Hold | **full** | claude_code | NULL | NULL |
| 18:36:44 | HPE | 5.68 | Hold | **full** | claude_code | NULL | NULL |

**Zero-score share 0 of 3, against the POST baseline of 81.2%.** That is a real
measurement and it is the right direction.

**But it CANNOT be attributed to the armed flag, and this section exists to say
so before anyone infers otherwise.**

```sql
SELECT JSON_VALUE(full_report_json,'$.final_synthesis.error') ...  -- NULL on all three
$ grep -ac "Failed to parse final report" backend.log   # this cycle
0
```

The full path RAN and PARSED CLEANLY on every ticker. `paper_synthesis_integrity_enabled`
guards the *parse-failure* branch, so with zero parse failures **the guard was
never entered**. Tonight's cycle therefore cannot distinguish "the guard works"
from "the full path happened to succeed". The guard's effect is only observable
on a cycle where synthesis actually fails; **this cycle is evidence that the
population is healthy, not evidence that the fix fires.** Reported as a limit of
the measurement rather than left for a reader to assume the stronger claim.

### 6c. TWO CORRECTIONS to `experiment_results_86.69.md`, measured

Both REPLACE the corresponding claims; they do not sit beside them.

1. **"The lite writer is exonerated (title-case `Hold`)" does not hold as a
   discriminator.** All three rows above are `_path=full` and carry title-case
   `Hold` -- because the MODEL returned `"Hold"` as its action string
   (`$.final_synthesis.recommendation.action` = `Hold`, error NULL). So
   title-case is produced by the full path too and is **not** a reliable marker
   of which writer ran. The reliable markers are `_path` (stamped provenance,
   phase-60.1) plus `final_score = 0.0` co-occurring with a non-null
   `final_synthesis.error`.
2. **The log line `"Lite analysis persisted to analysis_results for <T>"` is
   MISLEADING and fires for full-path rows too.** Its emit site is
   `backend/services/autonomous_loop.py:3656`, inside
   `_persist_analysis` (`:3561`), whose own docstring says it was *"Generalized
   from `_persist_lite_analysis` to handle BOTH lite and full paths"* -- the
   message was never updated. All three rows above were logged with that line
   and every one is `_path=full`. **Anyone diagnosing the path from this log
   line will get the wrong answer**, which is how this section's first draft
   read them as lite rows. Queued as an evidence-class residual (message
   should name the actual `_path`), not fixed here.

Also visible in the same window and NOT this step's: `Cost budget exceeded:
$10.7671 > $5.00 limit` on SNDK -- the full 28-agent path is running at ~2x its
per-ticker budget. Recorded for whoever owns cost.
