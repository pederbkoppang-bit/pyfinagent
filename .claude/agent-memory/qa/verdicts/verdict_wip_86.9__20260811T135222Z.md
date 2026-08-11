STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.9
WRITTEN: 2026-08-11T13:52:22Z
COMPLETED: 2026-08-11T14:01:30Z

# Q/A WIP record -- step 86.9 (paper_cycle_max_seconds budget raise)

## A. HARNESS COMPLIANCE (all 5 clean)
- research gate: research_brief_86.9.md envelope brief_status=COMPLETE, gate_passed=true,
  external_sources_read_in_full=8, urls_collected=21, recency_scan_performed=true,
  internal_files_inspected=14. Disclosed gap (WebSearch 200/200 exhausted) acknowledged.
- contract-before-generate by mtime (UTC): brief 13:45:17 < contract 13:48:21 <
  experiment_results 13:50:50 < live_check 13:51:08. Commits 26037c1e (PLAN) then
  38ae0f9c (GENERATE), each touching ONLY handoff artifacts.
- experiment_results_86.9.md present (6774 B).
- log-last: `grep -F "phase=86.9" handoff/harness_log.md` -> 0 hits; masterplan status=pending.
  (The bare "86.9" grep false-positives on "edge ratio 86.92" -- escaped/-F grep used.)
- no verdict-shopping: first Q/A spawn for this step; 0 prior CONDITIONALs.

## B. DETERMINISTIC
- IMMUTABLE CMD: `source .venv/bin/activate && python -c "...paper_cycle_max_seconds"`
  -> `10800.0`, EXIT=0.
- Lint gate 1a: `git diff --name-only HEAD -- '*.py'` = EMPTY, and `git ls-files --others
  -- '*.py'` = EMPTY. Step deliberately changed no code, so the gate is N/A BY DERIVATION
  (not "green"). Full step file set (4980728c..HEAD): masterplan.json, CHANGELOG.md,
  and the 4 handoff artifacts. NO production file touched. git status clean of prod.
- masterplan diff: 86.9's criteria UNAMENDED; the only insertion is a NEW step 86.53
  (the config-drift defect, filed per the queue-discovered-defects rule).
- Runtime smoke: /api/health -> 200. /api/settings/ -> 200.

## C. CRITERIA
1. MET (independently reproduced). `curl -s http://127.0.0.1:8000/api/settings/` -> 200,
   paper_cycle_max_seconds=10800.0, paper_analyze_top_n=5, paper_screen_top_n=10.
   `lsof -nP -iTCP:8000 -sTCP:LISTEN` -> Python pid 66306; `ps -o pid,lstart -p 66306` ->
   started man. 10 aug. 21.33.01 2026 (local; 19:33:01Z), elapsed 18:20:16.
   NO-SLASH history CONFIRMED not a fabrication: `/api/settings` (no slash) returns
   http_code=307 with size_download=0 -- exactly the empty response Main says misled it.
   The disclosure is honest and self-incriminating; it counts in Main's favour.
2. MET. Diagnostic reproduced by me: CYCLE started=2026-08-10 20:00:02.593 terminal=completed
   wall=4532.113s. backend.log timestamps are LOCAL (last line 15:54:44 vs `date` 15:54:47),
   so that cycle = 2026-08-10T18:00:02Z. `os.stat(backend/.env).st_mtime` = 2026-08-09T13:50:16Z
   -> the raise PRECEDES the cycle by ~28h. VERIFIED.
   Residual (disclosed by Main in substance): the cycle finished 2708s inside the OLD budget,
   so it does not EXERCISE the raise; and `grep -c 10800 backend.log` = 0 and `grep -c 7200`
   = 0, so no log record of the value in force. In-force is INFERRED from .env mtime +
   `_get_settings_fresh.cache_clear()` at services/autonomous_loop.py:2137 (verified present,
   per ticker) + `settings = settings or get_settings()` at :406.
3. MET. I re-ran `scripts/diagnostics/measure_analysis_phase.py`: EVERY figure reproduces
   exactly -- per-ticker CRWD 961.4/DELL 1705.5/HPE 958.1/HUM 1067.7/NTAP 1672.8/PANW 1525.6,
   mean 1315.2, median 1296.6, serial 7891.1, parallelism 1.85, projected 4492,
   cc_rail started=152 timed_out=1 rate=0.0066. (lines_parsed 73404 mine vs 73231 Main's --
   the log is live-appending; no figure moved.)
4. MET. `grep -n "asyncio.timeout" backend/services/autonomous_loop.py` -> only :514, wrapping
   the whole cycle; `_cycle_timeout` captured at :507. `_run_single_analysis` defined :2088,
   called :1229 with NO timeout wrapper. Conclusion "a longer budget delays a hung ticker by
   3600s, it does not remove it" is correct.
5. MET-with-note. #24 RECOMMENDED on censored-distribution grounds; #25 "DEFERRED, not
   withdrawn". Criterion wording is binary (recommended OR withdrawn); "deferred" is a third
   value, but it is explicit and reasoned (single-factor attribution).
6. MET (independently reproduced). Key-by-key parse of backend/.env vs
   backend/.env.bak.20260809T155016: 49 keys each, added=[], removed=[], changed=
   ['PAPER_CYCLE_MAX_SECONDS'] '7200.0' -> '10800.0'. PAPER_ANALYZE_TOP_N is absent from BOTH
   .env files (code default), live endpoint reports 5 -> not lowered. Backup retained.

## FINDINGS (defects in the CLAIMS, not the code -- nothing was changed)
F1 [NOTE] experiment_results §3: "the live backend.log rotated at 08-11 08:41". WRONG DATE.
   `head -2 backend.log` -> first line 2026-08-10 08:41:30; the archive is
   handoff/logs/backend.log.20260810T064130Z.gz (06:41:30Z = 08:41:30 local 08-10).
   Rotation was 08-10, not 08-11. No conclusion moves.
F2 [WARN] experiment_results §3: "The gate's n=7 spans 6 rotated archives". WRONG. The brief
   (:376) used backend.log PLUS EXACTLY ONE archive, backend.log.20260810T064130Z.gz, which
   held "6 more cycles" (:277). 6 CYCLES was restated as 6 ARCHIVES. Misdescribes the
   provenance of the very evidence being attributed.
F3 [WARN] experiment_results §8 "config drift across FOUR sites" -- the table under that
   heading has FIVE rows, and the derived population is at least SIX: `grep -rn
   paper_cycle_max_seconds backend scripts` also returns backend/services/cycle_lock.py:82
   resolving it with `_CYCLE_BUDGET_FALLBACK_SEC` = 7200.0 (cycle_lock.py:63) -- a sixth
   resolution site with its own literal, absent from the table. The undercount PROPAGATED
   into the new masterplan step 86.53 audit_basis ("all four sites read directly"). Contained
   by 86.53's own criterion 1 (derived enumeration required), but it is a census asserted
   rather than derived.
F4 [WARN] §7 headline "the raise was the WRONG fix" omits its own strongest counter-evidence:
   the two overruns project to 8554s and 8529s (brief :396-397), and BOTH FIT INSIDE 10800s
   with ~2250s to spare. So the raise would have converted both observed failures (one
   un-analysed ticker each) into completions. `grep -n "8554\|8529" experiment_results
   contract` -> 0 hits: the number that most directly rebuts the headline appears nowhere.
   The correct statement is "an effective mitigation for the observed magnitude, but not the
   cause"; "the WRONG fix" overstates and is the one framing that could invite a revert.
F5 [NOTE] §4 "The sole inner cap is a per-call 150s at claude_code_client.py:593" -- there are
   also explicit `timeout_s=120` call sites at services/autonomous_loop.py:2960 and :3044
   (trade-decision + risk-judge rail, not the analysis phase). Does not affect the answer.
F6 [NOTE] Citation path ambiguity: two files named autonomous_loop.py exist
   (backend/autonomous_loop.py and backend/services/autonomous_loop.py). The bare
   "autonomous_loop.py:507/:514" citations resolve only against services/. Reading the
   top-level file at those lines shows unrelated code.
F7 [NOTE] §5/§7 numbers 18.1% / 14.9% / 32x150s=4800s are the GATE's (brief :384/:408/:412),
   not Main's own measurement; §3+§9 disclose n=7 as the gate's but §7 restates the derived
   figures without an inline attribution marker.

F8 [WARN] Criterion 5 demands the asks be re-evaluated "against POST-FIX data". The figures
   that carry the #24 recommendation -- "p90 = 134s and the longest SUCCESS = 145s against a
   150s cap" (experiment_results :89) -- are UNDATED and are NOT post-fix: they trace to
   research_brief_85.4.md:321 via the gate brief (:455-456). They cannot be re-derived from
   the post-fix window: `measure_analysis_phase.py` computes `p90_s` and
   `n_within_5s_of_150s_cap` (script :249/:251) but BOTH Main's run and mine print
   `agent latency : None` for the 08-10 cycle. Worse, the post-fix datum that DOES exist
   points the other way on urgency -- 1 timeout in 152 calls (0.66%) -- and §5 does not
   engage with that tension while recommending #24. The recommendation may well be right
   (rates were 9.9-23.4% on five other cycles); the ARGUMENT as written is pre-fix data
   presented as "the data is the argument".

## VERDICT REASONING (worst-of-N lenses, P1 money-path)
- correctness lens: all six criteria MET; every figure I re-ran reproduced exactly -> PASS
- does-it-reproduce lens: immutable cmd exit=0/10800.0; diagnostic table identical;
  .env diff identical; endpoint 200 from pid 66306 -> PASS
- scope-honesty lens: F2/F3/F4/F8 -> CONDITIONAL
- min(lenses) = CONDITIONAL. 0 prior CONDITIONALs for 86.9 (harness_log grep -F) so the
  3rd-CONDITIONAL auto-FAIL rule does NOT bind. retry_count=0 < max_retries=3 ->
  certified_fallback=false.
- NOTE FOR THE RECORD: nothing was changed in this step, so there is no code mutation matrix
  to run. The subject under test is the CLAIMS, and every one that could be re-derived was.
  The headline conclusion is the one thing that cannot be settled by re-derivation, which is
  why F4 (its missing counter-arithmetic) is the finding that matters most.
