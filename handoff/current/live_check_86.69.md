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
