# live_check -- masterplan step 75.5 (cycle 4)

Date: 2026-07-20 | Findings: llmeng-01, -03, -04, -06, -10, -11, arch-04

## HOW THIS FILE IS PRODUCED -- read before auditing

Every block below is emitted by a generator that runs the command in its header and
prints the process output **unmodified** -- stdout and stderr concatenated, nothing
filtered, nothing truncated, nothing reformatted.

Cycle 3 rejected the previous version of this file because its note claimed *'every
fenced block is stdout captured verbatim'* with one named filter, while three blocks
were in fact truncated, reformatted, or concatenated from several runs. No number in
them was wrong, but a document whose entire purpose is to be auditable must not
describe less processing than it performed. Blocks that are a COMPOSITION of several
commands are now labelled as such in their own header rather than shown under a single
`$` prompt.

Consequence: these blocks include the unrelated urllib3 `RequestsDependencyWarning`
env noise and pytest's full failure list. That is the point -- an operator sees what
the command actually printed.

## 1. Verification command, verbatim (immutable)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_llm_rail.py -q
.........................................                                [100%]
=============================== warnings summary ===============================
backend/tests/test_phase_75_llm_rail.py::test_pydantic_class_schema_yields_json_schema_argv_with_additional_properties_false
  /Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/google/genai/types.py:42: DeprecationWarning: '_UnionGenericAlias' is deprecated and slated for removal in Python 3.17
    VersionedUnionType = Union[builtin_types.UnionType, _UnionGenericAlias]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
41 passed, 1 warning in 5.55s
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
EXIT=0
```

## 2. Full backend suite -- COMPLETE, untruncated

Shown in full. Cycle 3 caught the previous version truncating this to 2 FAILED
lines while claiming to be verbatim -- and that truncation is exactly what hid the
lock-roster defect from the operator. The complete failure list is enumerated and
categorised in experiment_results.md section 4.

```
$ .venv/bin/python -m pytest backend/tests/ -q --timeout=300
........................................................................ [  5%]
........................................................................ [ 10%]
........................................................................ [ 16%]
........................................................................ [ 21%]
...................................................F....sssssss..xFsF... [ 27%]
.X........Fxx....................s.....F....s.........Fs................ [ 32%]
...............................................x........................ [ 38%]
........................................................................ [ 43%]
..............x......................................................... [ 49%]
........................................................................ [ 54%]
..............................FF.F.............................F........ [ 60%]
.............F.......................................................... [ 65%]
........................................................................ [ 70%]
........................................................................ [ 76%]
........................................................................ [ 81%]
........................................................................ [ 87%]
........................................................................ [ 92%]
.......F....................................................s........... [ 98%]
........................                                                 [100%]
=================================== FAILURES ===================================
______________ test_phase_23_2_10_watchdog_log_present_and_fresh _______________

    def test_phase_23_2_10_watchdog_log_present_and_fresh():
        """The watchdog log file must exist + be fresh (entries within last 24h).
        Stale log = watchdog process is dead = invisible failure mode."""
        if not WATCHDOG_LOG.exists():
            pytest.skip(f"watchdog log not present: {WATCHDOG_LOG}")
        text = WATCHDOG_LOG.read_text(encoding="utf-8", errors="replace")
        ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
        timestamps = []
        for line in text.splitlines():
            m = ts_pattern.search(line)
            if m:
                ts = _parse_iso_z(m.group(1))
                if ts:
                    timestamps.append(ts)
        assert timestamps, "watchdog log must contain at least 1 ISO-Z timestamped entry"
        latest = max(timestamps)
        age = datetime.now(timezone.utc) - latest
>       assert age <= timedelta(hours=LOG_FRESHNESS_HOURS), (
            f"watchdog log stale: latest entry {latest.isoformat()} is "
            f"{age.total_seconds() / 3600:.1f}h old (max {LOG_FRESHNESS_HOURS}h)"
        )
E       AssertionError: watchdog log stale: latest entry 2026-06-11T18:05:48+00:00 is 933.4h old (max 24h)
E       assert datetime.timedelta(days=38, seconds=77176, microseconds=804364) <= datetime.timedelta(days=1)
E        +  where datetime.timedelta(days=1) = timedelta(hours=24)

backend/tests/test_phase_23_2_10_watchdog_no_fire_7d.py:86: AssertionError
____ test_phase_23_2_12_layer1_pipeline_at_least_one_lite_proxy_in_last_7d _____

    def test_phase_23_2_12_layer1_pipeline_at_least_one_lite_proxy_in_last_7d():
        """Cost-proxy substitute for the uncompilable `_path='lite'` clause:
        lite path uses total_cost_usd <= 0.05 (per researcher cite of
        autonomous_loop.py:1498). At least 1 such row in last 7 days = pipeline
        is firing the lite variant at least sometimes."""
        if not _bq_available():
            pytest.skip("google-cloud-bigquery + ADC credentials not available")
    
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID, location="us-central1")
    
        sql = """
        SELECT COUNT(*) AS n_lite_proxy
        FROM `sunny-might-477607-p8.financial_reports.analysis_results`
        WHERE DATE(analysis_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
          AND total_cost_usd <= 0.05
        """
        rows = list(client.query(sql).result())
        assert rows, "BQ query returned 0 rows"
        n = rows[0].n_lite_proxy
>       assert n >= 1, (
            f"Layer-1 pipeline 'lite' proxy (cost <= 0.05) returned 0 rows "
            f"in last 7 days. Pipeline broken at lite path. Got n={n}."
        )
E       AssertionError: Layer-1 pipeline 'lite' proxy (cost <= 0.05) returned 0 rows in last 7 days. Pipeline broken at lite path. Got n=0.
E       assert 0 >= 1

backend/tests/test_phase_23_2_12_layer1_pipeline_active.py:132: AssertionError
_________ test_phase_23_2_12_layer1_analysis_results_has_recent_writes _________

    def test_phase_23_2_12_layer1_analysis_results_has_recent_writes():
        """Loose freshness invariant: at least 1 row in the last 48 hours
        (catches the worst case where the entire pipeline silently halts)."""
        if not _bq_available():
            pytest.skip("google-cloud-bigquery + ADC credentials not available")
    
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID, location="us-central1")
    
        sql = """
        SELECT COUNT(*) AS n, MAX(analysis_date) AS max_ts
        FROM `sunny-might-477607-p8.financial_reports.analysis_results`
        WHERE TIMESTAMP(analysis_date) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 48 HOUR)
        """
        rows = list(client.query(sql).result())
        n = rows[0].n
        max_ts = rows[0].max_ts
>       assert n >= 1, (
            f"Layer-1 pipeline produced 0 rows in last 48h (max_ts={max_ts}). "
            f"Hard failure -- pipeline silently halted."
        )
E       AssertionError: Layer-1 pipeline produced 0 rows in last 48h (max_ts=None). Hard failure -- pipeline silently halted.
E       assert 0 >= 1

backend/tests/test_phase_23_2_12_layer1_pipeline_active.py:186: AssertionError
_______________ test_phase_23_2_15_known_pass_scripts_still_pass _______________

    def test_phase_23_2_15_known_pass_scripts_still_pass():
        """The 8 scripts that PASSed at researcher live-run (2026-05-23) must
        still PASS now. Catches regression in any of cycles 12, 15, 17, 18,
        19, 21, 22, 23."""
        expected_pass_cycles = {c for c, ok in KNOWN_PASS.items() if ok}
        failures: list[str] = []
        for script in _verify_scripts():
            cycle = _script_cycle(script)
            if cycle not in expected_pass_cycles:
                continue
            try:
                rc, out, err = _run_script(script)
            except subprocess.TimeoutExpired:
                failures.append(f"verify_phase_23_1_{cycle}.py: TIMEOUT (>60s)")
                continue
            if rc != 0:
                failures.append(
                    f"verify_phase_23_1_{cycle}.py: exit={rc} "
                    f"(previously PASSed per researcher 2026-05-23)\n"
                    f"  STDOUT tail: ...{out[-200:]}\n"
                    f"  STDERR tail: ...{err[-200:]}"
                )
>       assert not failures, (
            f"phase-23.2.15 REGRESSION: {len(failures)} previously-passing verify "
            f"scripts now FAIL:\n" + "\n".join(failures[:5])
        )
E       AssertionError: phase-23.2.15 REGRESSION: 6 previously-passing verify scripts now FAIL:
E         verify_phase_23_1_15.py: exit=1 (previously PASSed per researcher 2026-05-23)
E           STDOUT tail: ...
E           STDERR tail: ...ions/3.14/lib/python3.14/subprocess.py", line 1990, in _execute_child
E             raise child_exception_type(errno_num, err_msg, err_filename)
E         FileNotFoundError: [Errno 2] No such file or directory: 'python'
E         
E         verify_phase_23_1_18.py: exit=1 (previously PASSed per researcher 2026-05-23)
E           STDOUT tail: ...
E           STDERR tail: ...ions/3.14/lib/python3.14/subprocess.py", line 1990, in _execute_child
E             raise child_exception_type(errno_num, err_msg, err_filename)
E         FileNotFoundError: [Errno 2] No such file or directory: 'python'
E         
E         verify_phase_23_1_19.py: exit=1 (previously PASSed per researcher 2026-05-23)
E           STDOUT tail: ...
E           STDERR tail: ...ions/3.14/lib/python3.14/subprocess.py", line 1990, in _execute_child
E             raise child_exception_type(errno_num, err_msg, err_filename)
E         FileNotFoundError: [Errno 2] No such file or directory: 'python'
E         
E         verify_phase_23_1_21.py: exit=1 (previously PASSed per researcher 2026-05-23)
E           STDOUT tail: ...
E           STDERR tail: ...ions/3.14/lib/python3.14/subprocess.py", line 1990, in _execute_child
E             raise child_exception_type(errno_num, err_msg, err_filename)
E         FileNotFoundError: [Errno 2] No such file or directory: 'python'
E         
E         verify_phase_23_1_22.py: exit=1 (previously PASSed per researcher 2026-05-23)
E           STDOUT tail: ...
E           STDERR tail: ...ions/3.14/lib/python3.14/subprocess.py", line 1990, in _execute_child
E             raise child_exception_type(errno_num, err_msg, err_filename)
E         FileNotFoundError: [Errno 2] No such file or directory: 'python'
E         
E       assert not ["verify_phase_23_1_15.py: exit=1 (previously PASSed per researcher 2026-05-23)\n  STDOUT tail: ...\n  STDERR tail: ....._exception_type(errno_num, err_msg, err_filename)\nFileNotFoundError: [Errno 2] No such file or directory: 'python'\n"]

backend/tests/test_phase_23_2_15_verify_23_1_smoke.py:89: AssertionError
___________ test_phase_23_2_6_backend_log_has_skipping_buy_evidence ____________

    def test_phase_23_2_6_backend_log_has_skipping_buy_evidence():
        """Read-only verification: backend.log must contain at least one
        'Skipping BUY ... at cap' line if the gate has ever fired in the
        log's retention window. Researcher counted 24 today."""
        backend_log = REPO_ROOT / "backend.log"
        if not backend_log.exists() or backend_log.stat().st_size < 100:
            pytest.skip(f"backend.log not present or too small: {backend_log}")
        text = backend_log.read_text(encoding="utf-8", errors="replace")
        skip_count = text.count("Skipping BUY")
        # Defensive lower bound: researcher counted 24; any future cap-firing
        # cycle adds more. phase-62.6: backend.log is rotated (cp+truncate+gzip
        # into handoff/logs/) once it exceeds 50MB -- per this test's own
        # original comment ("the log was rotated and the test should adapt"),
        # fall back to the newest archive before declaring the gate broken.
        if skip_count == 0:
            import gzip
            archives = sorted((REPO_ROOT / "handoff" / "logs").glob("backend.log.*.gz"))
            if archives:
                with gzip.open(archives[-1], "rt", encoding="utf-8", errors="replace") as f:
                    skip_count = sum(line.count("Skipping BUY") for line in f)
            else:
                pytest.skip("backend.log freshly rotated and no archive found")
>       assert skip_count >= 1, (
            f"no 'Skipping BUY' line in backend.log OR its newest archive "
            f"(researcher counted 24 on 2026-05-23); the cap gate may be "
            f"silently disabled."
        )
E       AssertionError: no 'Skipping BUY' line in backend.log OR its newest archive (researcher counted 24 on 2026-05-23); the cap gate may be silently disabled.
E       assert 0 >= 1

backend/tests/test_phase_23_2_6_sector_cap_emit.py:252: AssertionError
______________ test_phase_23_2_9_backend_log_has_prewarm_evidence ______________

    def test_phase_23_2_9_backend_log_has_prewarm_evidence():
        """backend.log must show the prewarm line has fired at least once
        (researcher counted 54 occurrences). Defensive bound: >= 1."""
        backend_log = REPO_ROOT / "backend.log"
        if not backend_log.exists() or backend_log.stat().st_size < 100:
            pytest.skip(f"backend.log not present or too small: {backend_log}")
        text = backend_log.read_text(encoding="utf-8", errors="replace")
        count = text.count("Prewarming ticker-meta cache")
>       assert count >= 1, (
            f"backend.log must contain at least 1 'Prewarming ticker-meta cache' line; "
            f"got {count}. If 0, the prewarm hook is silently broken OR log rotated."
        )
E       AssertionError: backend.log must contain at least 1 'Prewarming ticker-meta cache' line; got 0. If 0, the prewarm hook is silently broken OR log rotated.
E       assert 0 >= 1

backend/tests/test_phase_23_2_9_ticker_meta_latency.py:77: AssertionError
______________ test_reject_binding_main_path_off_emits_on_blocks _______________

    def test_reject_binding_main_path_off_emits_on_blocks():
        """Flag OFF (default): a REJECT candidate's BUY IS emitted (advisory,
        pre-57.1 behavior). Flag ON: the BUY is ABSENT and blocked_out records it."""
        positions: list[dict] = []
        cand = [_candidate("REJ1", "Technology", 0.9, decision="REJECT")]
        state = _portfolio_state(10_000.0, 10_000.0, positions)
    
        # default flag (OFF) -- advisory: BUY emitted
        s_off = _make_settings()
>       assert s_off.paper_risk_judge_reject_binding is False  # ships default-OFF
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AssertionError: assert True is False
E        +  where True = Settings(app_name='PyFinAgent', debug=False, log_level='INFO', cost_tier='build', gcp_project_id='sunny-might-477607-p...mode_enabled=False, morning_digest_hour=8, evening_digest_hour=17, watchdog_interval_minutes=15, first_week_mode=False).paper_risk_judge_reject_binding

backend/tests/test_phase_57_1_reject_binding.py:100: AssertionError
______________ test_reject_binding_swap_path_off_emits_on_blocks _______________

    def test_reject_binding_swap_path_off_emits_on_blocks():
        positions, cands, holds, state = _swap_scenario("REJECT")
    
        # Flag OFF: TECH_NEW1 (REJECT) still swap-buys -- reproduces the
        # away-week vulnerability (advisory verdict executed via swap_buy).
        orders_off = decide_trades(positions, cands, holds, state, _make_settings())
        swap_buys_off = {o.ticker for o in orders_off if o.reason == "swap_buy"}
>       assert "TECH_NEW1" in swap_buys_off, (
            f"flag-OFF must preserve the (vulnerable) swap BUY; swap_buys={swap_buys_off}"
        )
E       AssertionError: flag-OFF must preserve the (vulnerable) swap BUY; swap_buys={'TECH_NEW2'}
E       assert 'TECH_NEW1' in {'TECH_NEW2'}

backend/tests/test_phase_57_1_reject_binding.py:148: AssertionError
----------------------------- Captured stderr call -----------------------------
[33m17:32:35 W [portfolio_manager][0m BINDING RiskJudge gate: BLOCKED BUY TECH_NEW1 (decision=REJECT, final_score=0.85) -- paper_risk_judge_reject_binding=ON (F-3)
[36m17:32:35 I [portfolio_manager][0m Skipping BUY TECH_NEW2: sector Technology at cap (8/2) -- queued for swap check
[36m17:32:35 I [portfolio_manager][0m Swap fired (1/2): SELL TECH0 (score=0.550) -> BUY TECH_NEW2 (score=0.820) delta=27.0%
[36m17:32:35 I [portfolio_manager][0m Trade decisions: 1 sells, 2 buys
------------------------------ Captured log call -------------------------------
WARNING  backend.services.portfolio_manager:portfolio_manager.py:250 BINDING RiskJudge gate: BLOCKED BUY TECH_NEW1 (decision=REJECT, final_score=0.85) -- paper_risk_judge_reject_binding=ON (F-3)
INFO     backend.services.portfolio_manager:portfolio_manager.py:375 Skipping BUY TECH_NEW2: sector Technology at cap (8/2) -- queued for swap check
INFO     backend.services.portfolio_manager:portfolio_manager.py:812 Swap fired (1/2): SELL TECH0 (score=0.550) -> BUY TECH_NEW2 (score=0.820) delta=27.0%
INFO     backend.services.portfolio_manager:portfolio_manager.py:508 Trade decisions: 1 sells, 2 buys
_______________ test_off_identity_prompts_are_verbatim_constants _______________

    def test_off_identity_prompts_are_verbatim_constants():
        s_off = _make_settings()
>       assert al._build_risk_judge_system(s_off) is al._LITE_RISK_JUDGE_SYSTEM
E       AssertionError: assert 'You are an independent Risk Judge for a paper trading portfolio. Your role is to evaluate position risk -- NOT to val...mmended_position_pct (1-10) from these axes alone. Do not simply agree with the trader.\nRespond ONLY with valid JSON.' is 'You are an independent Risk Judge for a paper trading portfolio. Your role is to evaluate position risk -- NOT to val...mmended_position_pct (1-10) from these axes alone. Do not simply agree with the trader.\nRespond ONLY with valid JSON.'
E        +  where 'You are an independent Risk Judge for a paper trading portfolio. Your role is to evaluate position risk -- NOT to val...mmended_position_pct (1-10) from these axes alone. Do not simply agree with the trader.\nRespond ONLY with valid JSON.' = <function _build_risk_judge_system at 0x11c8c6770>(Settings(app_name='PyFinAgent', debug=False, log_level='INFO', cost_tier='build', gcp_project_id='sunny-might-477607-p...mode_enabled=False, morning_digest_hour=8, evening_digest_hour=17, watchdog_interval_minutes=15, first_week_mode=False))
E        +    where <function _build_risk_judge_system at 0x11c8c6770> = al._build_risk_judge_system
E        +  and   'You are an independent Risk Judge for a paper trading portfolio. Your role is to evaluate position risk -- NOT to val...mmended_position_pct (1-10) from these axes alone. Do not simply agree with the trader.\nRespond ONLY with valid JSON.' = al._LITE_RISK_JUDGE_SYSTEM

backend/tests/test_phase_57_1_reject_binding.py:189: AssertionError
_____________ test_60_1_claude_code_rail_declares_latency_profile ______________

    def test_60_1_claude_code_rail_declares_latency_profile():
        pytest.importorskip("backend.agents.claude_code_client")
        from backend.agents.claude_code_client import ClaudeCodeClient
    
        # The declared step budget must sit ABOVE the rail's own subprocess
        # timeout, or the step gives up while the CLI call is still in flight.
        client = ClaudeCodeClient("claude-sonnet-4-6")
>       assert ClaudeCodeClient.recommended_step_timeout > client._timeout_s
E       AssertionError: assert 150 > 150
E        +  where 150 = <class 'backend.agents.claude_code_client._make_claude_code_client_class.<locals>.ClaudeCodeClient'>.recommended_step_timeout
E        +  and   150 = <backend.agents.claude_code_client._make_claude_code_client_class.<locals>.ClaudeCodeClient object at 0x127a97390>._timeout_s

backend/tests/test_phase_60_1_deep_pipeline.py:333: AssertionError
_________________________ test_60_3_flag_defaults_off __________________________

    def test_60_3_flag_defaults_off():
>       assert Settings().paper_data_integrity_enabled is False
E       AssertionError: assert True is False
E        +  where True = Settings(app_name='PyFinAgent', debug=False, log_level='INFO', cost_tier='build', gcp_project_id='sunny-might-477607-p...mode_enabled=False, morning_digest_hour=8, evening_digest_hour=17, watchdog_interval_minutes=15, first_week_mode=False).paper_data_integrity_enabled
E        +    where Settings(app_name='PyFinAgent', debug=False, log_level='INFO', cost_tier='build', gcp_project_id='sunny-might-477607-p...mode_enabled=False, morning_digest_hour=8, evening_digest_hour=17, watchdog_interval_minutes=15, first_week_mode=False) = Settings()

backend/tests/test_phase_60_3_data_integrity.py:220: AssertionError
____________________ test_swap_framework_fills_zero_buy_gap ____________________

    def test_swap_framework_fills_zero_buy_gap():
        """The 2026-05-26 scenario: 8/9 Tech + sector cap = zero-buy without swap.
    
        With swap enabled, expect 2 swap pairs + 1 standard BUY = 5 orders total
        (2 SELLs + 3 BUYs).
        """
        nav = 10_000.0
        # Cash above the 5% min_cash_reserve ($500) so the buy-loop's
        # `available_cash <= 0` guard doesn't short-circuit before
        # sector_blocked can populate. $2000 supports the Industrials slot-fill
        # AND leaves the swap path to do the Tech rebalance net-zero on cash.
        cash = 2_000.0
        # 8 Tech holdings with scores 0.55-0.75; 1 Industrials with score 0.65.
        tech_scores = [0.55, 0.58, 0.60, 0.65, 0.68, 0.70, 0.73, 0.75]
        positions = [
            _holding(f"TECH{i}", "Technology", 1100.0, s)
            for i, s in enumerate(tech_scores)
        ]
        positions.append(_holding("INDU1", "Industrials", 1000.0, 0.65))
    
        # Re-evaluation: all hold (rec=BUY), no sell signal. So decide_trades
        # should NOT generate any signal-based SELLs.
        holding_analyses = []
        for i, p in enumerate(positions):
            score = tech_scores[i] if i < 8 else 0.65
            holding_analyses.append(_holding_analysis(p["ticker"], float(score)))
    
        # 3 candidates: 2 Tech high-conviction, 1 Industrials.
        candidate_analyses = [
            _candidate_analysis("TECH_NEW1", "Technology", 0.85),
            _candidate_analysis("TECH_NEW2", "Technology", 0.82),
            _candidate_analysis("INDU_NEW", "Industrials", 0.70),
        ]
    
        portfolio_state = {
            "nav": nav,
            "cash": cash,
            "positions_value": nav - cash,
            "position_count": len(positions),
        }
        settings = _make_settings()
    
        orders = decide_trades(
            current_positions=positions,
            candidate_analyses=candidate_analyses,
            holding_analyses=holding_analyses,
            portfolio_state=portfolio_state,
            settings=settings,
        )
    
        reasons = [o.reason for o in orders]
        actions = [o.action for o in orders]
    
        # Assert: 2 swap-SELLs + 2 swap-BUYs + 1 standard BUY = 5 orders total.
        swap_sells = [o for o in orders if o.reason == "swap_for_higher_conviction"]
        swap_buys = [o for o in orders if o.reason == "swap_buy"]
        standard_buys = [o for o in orders if o.reason == "new_buy_signal"]
    
>       assert len(swap_sells) == 2, (
            f"Expected 2 swap SELLs, got {len(swap_sells)}; orders={orders}"
        )
E       AssertionError: Expected 2 swap SELLs, got 1; orders=[TradeOrder(ticker='TECH0', action='SELL', amount_usd=None, quantity=None, reason='swap_for_higher_conviction', analysis_id='', risk_judge_decision='', stop_loss_price=None, risk_judge_position_pct=None, price=110.0, signals=[], sector='', market='US', swap_group_id=None, price_at_analysis=None, factor_loadings=None, analysis_recommendation=''), TradeOrder(ticker='INDU_NEW', action='BUY', amount_usd=1000.0, quantity=None, reason='new_buy_signal', analysis_id='2026-05-26', risk_judge_decision='APPROVE_FULL', stop_loss_price=92.0, risk_judge_position_pct=10.0, price=100.0, signals=[{'agent': 'Trader', 'role': 'decision', 'rationale': 'Recommendation: BUY', 'weight': 0.06999999999999999}, {'agent': 'RiskJudge', 'role': 'gate', 'rationale': 'Decision: APPROVE_FULL', 'weight': 1.0}], sector='Industrials', market='US', swap_group_id=None, price_at_analysis=100.0, factor_loadings=None, analysis_recommendation='BUY'), TradeOrder(ticker='TECH_NEW1', action='BUY', amount_usd=1000.0, quantity=None, reason='swap_buy', analysis_id='2026-05-26', risk_judge_decision='APPROVE_FULL', stop_loss_price=92.0, risk_judge_position_pct=10.0, price=100.0, signals=[{'agent': 'Trader', 'role': 'decision', 'rationale': 'Recommendation: BUY', 'weight': 0.08499999999999999}, {'agent': 'RiskJudge', 'role': 'gate', 'rationale': 'Decision: APPROVE_FULL', 'weight': 1.0}], sector='Technology', market='US', swap_group_id=None, price_at_analysis=100.0, factor_loadings=None, analysis_recommendation='BUY')]
E       assert 1 == 2
E        +  where 1 = len([TradeOrder(ticker='TECH0', action='SELL', amount_usd=None, quantity=None, reason='swap_for_higher_conviction', analys... sector='', market='US', swap_group_id=None, price_at_analysis=None, factor_loadings=None, analysis_recommendation='')])

backend/tests/test_portfolio_swap.py:131: AssertionError
----------------------------- Captured stderr call -----------------------------
[36m17:32:54 I [portfolio_manager][0m Skipping BUY TECH_NEW1: sector Technology at cap (8/2) -- queued for swap check
[36m17:32:54 I [portfolio_manager][0m Skipping BUY TECH_NEW2: sector Technology at cap (8/2) -- queued for swap check
[36m17:32:54 I [portfolio_manager][0m Swap fired (1/2): SELL TECH0 (score=0.550) -> BUY TECH_NEW1 (score=0.850) delta=30.0%
[36m17:32:54 I [portfolio_manager][0m Swap skip TECH1 -> TECH_NEW2: delta=24.0% below threshold 25.0% (cand_score=0.820 holding_score=0.580)
[36m17:32:54 I [portfolio_manager][0m Trade decisions: 1 sells, 2 buys
------------------------------ Captured log call -------------------------------
INFO     backend.services.portfolio_manager:portfolio_manager.py:375 Skipping BUY TECH_NEW1: sector Technology at cap (8/2) -- queued for swap check
INFO     backend.services.portfolio_manager:portfolio_manager.py:375 Skipping BUY TECH_NEW2: sector Technology at cap (8/2) -- queued for swap check
INFO     backend.services.portfolio_manager:portfolio_manager.py:812 Swap fired (1/2): SELL TECH0 (score=0.550) -> BUY TECH_NEW1 (score=0.850) delta=30.0%
INFO     backend.services.portfolio_manager:portfolio_manager.py:712 Swap skip TECH1 -> TECH_NEW2: delta=24.0% below threshold 25.0% (cand_score=0.820 holding_score=0.580)
INFO     backend.services.portfolio_manager:portfolio_manager.py:508 Trade decisions: 1 sells, 2 buys
=============================== warnings summary ===============================
.venv/lib/python3.14/site-packages/google/genai/types.py:42
  /Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/google/genai/types.py:42: DeprecationWarning: '_UnionGenericAlias' is deprecated and slated for removal in Python 3.17
    VersionedUnionType = Union[builtin_types.UnionType, _UnionGenericAlias]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED backend/tests/test_phase_23_2_10_watchdog_no_fire_7d.py::test_phase_23_2_10_watchdog_log_present_and_fresh
FAILED backend/tests/test_phase_23_2_12_layer1_pipeline_active.py::test_phase_23_2_12_layer1_pipeline_at_least_one_lite_proxy_in_last_7d
FAILED backend/tests/test_phase_23_2_12_layer1_pipeline_active.py::test_phase_23_2_12_layer1_analysis_results_has_recent_writes
FAILED backend/tests/test_phase_23_2_15_verify_23_1_smoke.py::test_phase_23_2_15_known_pass_scripts_still_pass
FAILED backend/tests/test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
FAILED backend/tests/test_phase_23_2_9_ticker_meta_latency.py::test_phase_23_2_9_backend_log_has_prewarm_evidence
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
FAILED backend/tests/test_phase_60_1_deep_pipeline.py::test_60_1_claude_code_rail_declares_latency_profile
FAILED backend/tests/test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
FAILED backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
12 failed, 1290 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 72.71s (0:01:12)
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
EXIT=1
```

## 3. Lock roster guard (phase-23.2.14) -- the defect cycle 3 found, now fixed

This step added `_DEGRADED_LOCK` (spend.py:39) without the re-audit + count bump the
guard's docstring requires. Re-audited: 17 real sites (16th = `_RAIL_GUARD_LOCK`,
pre-existing from phase-66.1 2026-07-07; 17th = mine). Both non-re-entrant.

```
$ .venv/bin/python -m pytest backend/tests/test_phase_23_2_14_no_reentrant_locks.py -q
.....                                                                    [100%]
5 passed in 0.08s
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
EXIT=0
```

## 4. THE MONEY FIX (llmeng-10) -- real CostTracker, real pricing table

```
$ .venv/bin/python -c "
from backend.agents.cost_tracker import CostTracker, MODEL_PRICING
from backend.agents.llm_client import UsageMeta
from types import SimpleNamespace
m='claude-opus-4-8'; ip=MODEL_PRICING[m][0]
t=CostTracker()
assert type(t).__module__=='backend.agents.cost_tracker'
e=t.record('a', m, SimpleNamespace(usage_metadata=UsageMeta(prompt_token_count=1000, candidates_token_count=0, total_token_count=6000, cache_read_input_tokens=5000)))
correct=(1000*ip + 5000*ip*0.1)/1_000_000; buggy=(5000*ip*0.1)/1_000_000
print(f'model={m} input=\${ip}/Mtok | uncached=1000 cache_read=5000')
print(f'OLD (double-subtracting, clamped): \${buggy:.6f}')
print(f'NEW (this fix):                    \${e.cost_usd:.6f}')
print(f'expected:                          \${correct:.6f}')
print(f'under-report eliminated:           {(1-buggy/correct)*100:.1f}%')
"
model=claude-opus-4-8 input=$5.0/Mtok | uncached=1000 cache_read=5000
OLD (double-subtracting, clamped): $0.002500
NEW (this fix):                    $0.007500
expected:                          $0.007500
under-report eliminated:           66.7%
EXIT=0
```

## 5. Runtime state of the remaining fixes

```
$ .venv/bin/python -c "
from backend.agents.schemas import CriticVerdict
from backend.agents.llm_client import _ensure_additional_properties_false, LLMResponse
from backend.config.model_tiers import GEMINI_WORKHORSE, GEMINI_DEEP_THINK, gemini_retirement_warning
from backend.services.observability import fetch_spend, spend_guard_status
from datetime import date
s=_ensure_additional_properties_false(CriticVerdict.model_json_schema())
print('(a) additionalProperties:', s['additionalProperties'], '| nested sealed:', all(v.get('additionalProperties') is False for v in s.get('\$defs',{}).values() if v.get('type')=='object'))
print('(c) is_truncated max_tokens:', LLMResponse(text='',stop_reason='max_tokens').is_truncated(), '| MAX_TOKENS:', LLMResponse(text='',stop_reason='MAX_TOKENS').is_truncated(), '| end_turn:', LLMResponse(text='',stop_reason='end_turn').is_truncated())
print('(d) WORKHORSE:', GEMINI_WORKHORSE, '| DEEP_THINK:', GEMINI_DEEP_THINK)
print('(d) warn 09-14:', gemini_retirement_warning('gemini-2.5-pro', date(2026,9,14)), '| fires 09-15:', bool(gemini_retirement_warning('gemini-2.5-pro', date(2026,9,15))))
print('(g) fetch_spend public:', callable(fetch_spend), '| guard:', spend_guard_status())
"
(a) additionalProperties: False | nested sealed: True
(c) is_truncated max_tokens: True | MAX_TOKENS: True | end_turn: False
(d) WORKHORSE: gemini-2.5-flash | DEEP_THINK: gemini-2.5-pro
(d) warn 09-14: None | fires 09-15: True
(g) fetch_spend public: True | guard: {'degraded_count': 0, 'last_error': '', 'alerted': False}
EXIT=0
```

## 6. Criterion-4 strict scan -- zero gemini-2.5 literals in the 5 files

```
$ grep -c 'gemini-2\.5' backend/config/settings.py backend/agents/evaluator_agent.py backend/agents/rag_agent_runtime.py backend/agents/skill_modification_review.py backend/autonomous_loop.py
backend/config/settings.py:0
backend/agents/evaluator_agent.py:0
backend/agents/rag_agent_runtime.py:0
backend/agents/skill_modification_review.py:0
backend/autonomous_loop.py:0
EXIT=1
```

## 7. Lint gate (qa.md 1a) -- full changed set, one command

3 findings, ALL PRE-EXISTING -- proven in the next block.

```
$ uvx ruff check --select F821,F401,F811 backend/agents/llm_client.py backend/agents/claude_code_client.py backend/agents/cost_tracker.py backend/agents/llm_parse.py backend/agents/evaluator_agent.py backend/agents/rag_agent_runtime.py backend/agents/skill_modification_review.py backend/config/model_tiers.py backend/config/settings.py backend/autonomous_loop.py backend/services/observability/spend.py backend/api/cost_budget_api.py backend/slack_bot/jobs/cost_budget_watcher.py backend/tests/test_phase_75_llm_rail.py
F401 [*] `typing.Any` imported but unused
  --> backend/agents/rag_agent_runtime.py:45:20
   |
43 | import logging
44 | import os
45 | from typing import Any
   |                    ^^^
46 | from backend.config.model_tiers import GEMINI_WORKHORSE  # phase-75.5 (llmeng-06)
   |
help: Remove unused import: `typing.Any`

F401 `backend.backtest.backtest_engine.BacktestEngine` imported but unused; consider using `importlib.util.find_spec` to test for availability
   --> backend/autonomous_loop.py:409:58
    |
407 |         # Import backtest harness
408 |         try:
409 |             from backend.backtest.backtest_engine import BacktestEngine
    |                                                          ^^^^^^^^^^^^^^
410 |         except ImportError:
411 |             logger.warning("[WARN]  Could not import BacktestEngine. Using mock results.")
    |
help: Remove unused import: `backend.backtest.backtest_engine.BacktestEngine`

F401 `backend.agents.evaluator_agent.EvaluationVerdict` imported but unused; consider using `importlib.util.find_spec` to test for availability
   --> backend/autonomous_loop.py:436:72
    |
435 |         try:
436 |             from backend.agents.evaluator_agent import EvaluatorAgent, EvaluationVerdict
    |                                                                        ^^^^^^^^^^^^^^^^^
437 |         except ImportError:
438 |             logger.warning("[WARN]  Could not import EvaluatorAgent. Using mock evaluation.")
    |
help: Remove unused import: `backend.agents.evaluator_agent.EvaluationVerdict`

Found 3 errors.
[*] 1 fixable with the `--fix` option.
EXIT=1
```

## 8. Same gate over pristine HEAD copies -- COMPOSITION of 2 runs

This block is NOT single-command output. It is two independent runs concatenated,
labelled as such rather than shown under one `$` prompt (cycle-3 blocker).

```
$ git show HEAD:backend/agents/rag_agent_runtime.py | uvx ruff check --stdin-filename backend/agents/rag_agent_runtime.py --select F821,F401,F811 -
F401 [*] `typing.Any` imported but unused
  --> backend/agents/rag_agent_runtime.py:45:20
   |
43 | import logging
44 | import os
45 | from typing import Any
   |                    ^^^
46 |
47 | logger = logging.getLogger(__name__)
   |
help: Remove unused import: `typing.Any`

Found 1 error.
[*] 1 fixable with the `--fix` option.
EXIT=1
```

```
$ git show HEAD:backend/autonomous_loop.py | uvx ruff check --stdin-filename backend/autonomous_loop.py --select F821,F401,F811 -
F401 `backend.backtest.backtest_engine.BacktestEngine` imported but unused; consider using `importlib.util.find_spec` to test for availability
   --> backend/autonomous_loop.py:408:58
    |
406 |         # Import backtest harness
407 |         try:
408 |             from backend.backtest.backtest_engine import BacktestEngine
    |                                                          ^^^^^^^^^^^^^^
409 |         except ImportError:
410 |             logger.warning("[WARN]  Could not import BacktestEngine. Using mock results.")
    |
help: Remove unused import: `backend.backtest.backtest_engine.BacktestEngine`

F401 `backend.agents.evaluator_agent.EvaluationVerdict` imported but unused; consider using `importlib.util.find_spec` to test for availability
   --> backend/autonomous_loop.py:435:72
    |
434 |         try:
435 |             from backend.agents.evaluator_agent import EvaluatorAgent, EvaluationVerdict
    |                                                                        ^^^^^^^^^^^^^^^^^
436 |         except ImportError:
437 |             logger.warning("[WARN]  Could not import EvaluatorAgent. Using mock evaluation.")
    |
help: Remove unused import: `backend.agents.evaluator_agent.EvaluationVerdict`

Found 2 errors.
EXIT=1
```

Identical finding set to the working tree -> zero lint defects introduced by this step.
(I did introduce one F401 myself -- `import os`, orphaned by the arch-04 move -- and
removed it.) Queued as 75.5.6.

## 9. Out-of-scope test protected by the back-compat alias

This file monkeypatches `cost_budget_watcher._default_fetch_spend` and lives OUTSIDE
`backend/tests/`, i.e. outside this step's verification command -- it would have
regressed silently.

```
$ .venv/bin/python -m pytest tests/slack_bot/test_scheduler_wiring_phase991.py -q
.........                                                                [100%]
9 passed in 0.77s
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
EXIT=0
```

## 10. Flag-gated live-loop behavior / UI

No config flag introduced; no UI surface touched -- no ON/OFF diff, no Playwright.
NO live LLM call was made (metered spend needs owner approval) and no live BQ spend
query ran (`fetch_spend` was exercised only through its failure path). A backend
restart is required for the running process to pick these up.

MONEY NOTE: the (e) fix moves REPORTED cost UPWARD (it was under-reporting by 66.7% on
cached calls). Dashboards will step up and the $25/day guard may trip sooner -- that is
the correction working, not a regression.
