# live_check — step 64.3 (Backend gap tests)

## Green run transcript (immutable command)

```
$ source .venv/bin/activate && python -m pytest backend/tests \
    -k '64_3 or kill_switch_machine or currency_path or screener_market or learnings_reader' -q
..................                                                       [100%]
18 passed, 1125 deselected, 1 warning in 3.92s
# exit 0
```

## Per-area test counts (all 4 gap areas covered — criterion 1)

| Gap area | File | Tests |
|---|---|---|
| Kill-switch state machine (rail-5 stays-paused) | `test_64_3_kill_switch_machine.py` | **5 passed** |
| Currency/money (KR-KRW + EU-EUR add-on avg_entry) | `test_64_3_currency_path.py` | **4 passed** |
| Per-market screener (validate_ohlcv + market_for_symbol) | `test_64_3_screener_market.py` | **5 passed** |
| Learnings reader (error != empty) | `test_64_3_learnings_reader.py` | **4 passed** |
| | **total** | **18 passed** |

## requires_live quarantine does NOT grow (criterion 1)

```
$ grep -l requires_live backend/tests/test_64_3_*.py
(none)
$ python -m pytest backend/tests -m requires_live --co -q | tail -1
11/1143 tests collected (1132 deselected)      # unchanged -- 64.3 adds 0
```

All 4 files are PURE (MagicMock bq, patched fx_rates.get_fx_rate + ExecutionRouter, monkeypatched
kill_switch._AUDIT_PATH/_state, pandas frames; conftest sets PYFINAGENT_TEST_NO_BQ=1). None use
`@pytest.mark.requires_live`. (NOTE: the actual marker count is 11, not the research's estimate of 6; criterion 1 is
"does not grow" and 64.3 contributes +0.)

## Criterion 2 (kill-switch stays-paused / rail 5)

`test_64_3_kill_switch_machine_stays_paused_auto_resume_off`: with `enabled=False`, `check_auto_resume(healthy_nav)`
returns action `no_op` + reason `auto_resume_disabled`, and `is_paused()` STILL True — the invariant rail 5
(away-ops-rules.md:17-18: "Kill-switch stays paused after any breach; auto-resume hysteresis stays OFF") depends on.
`..._active_breach_stays_paused`: even with `enabled=True`, an active breach does not resume.

## Criterion 3 (currency KR-KRW + EU-EUR)

`..._kr_avg_entry_stays_krw`: ON avg_entry ~70000 (KRW-scale); OFF < 1000 (legacy USD/local mix).
`..._eu_avg_entry_stays_eur`: ON avg_entry ~150 (EUR-scale); OFF ~162 (USD-inflated, > ON). Mirrors the phase-70.3 fix
(the code fix is 70.3; 61.3 was display-only) in the SHAPE of the 61.3 criteria.

## Method / boundaries
Pure pytest, no live network/BQ. `uvx ruff check` on the 4 files → All checks passed. NO production code changed
(tests only read the code-under-test with mocked IO). $0; live book untouched; historical_macro FROZEN.
