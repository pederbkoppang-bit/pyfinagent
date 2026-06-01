# Experiment Results — phase-54.1 (Cron audit + fix-or-escalate)

**Date:** 2026-06-01. **Status:** complete (full cross-layer audit + the one
autonomously-closable failure FIXED + tested; operator-gated residue escalated in the
artifact).

## What was done

1. **Cross-layer cron audit** (researcher-led, $0 read-only): every launchd job
   (~/Library/LaunchAgents/com.pyfinagent.*) + every APScheduler job (slack_bot +
   main backend) enumerated with loaded-state / last-exit / trigger / next-fire. Full
   table in `handoff/current/live_check_54.1.md`.
2. **Root-caused + FIXED the only real failure**: `autoresearch` + `ablation` (both
   launchctl last-exit=1) share ONE cause — their wrappers `set -a; . backend/.env`
   bash-source the env; `PAPER_MARKETS=["US","EU","KR"]` (JSON) loses its quotes on
   `source` → `[US,EU,KR]` → pydantic-settings' complex-field JSON decoder raised
   `SettingsError` at `get_settings()`. Fixed in `backend/config/settings.py`:
   `paper_markets: Annotated[list[str], NoDecode]` + a `field_validator(mode="before")`
   accepting JSON / bracket-mangled / comma / list forms. No `.env` edit, no new dep,
   purely additive (DO-NO-HARM).
3. **Documented the false positive**: `mas-harness` "not running" = idle
   `StartInterval 1800` job (exit 0), not a defect.
4. **Logged the away-week gaps for 54.2**: slack_bot has no launchd supervisor (single
   point of failure for the Slack lifeline), no external dead-man's-switch, digest lacks
   a cron-health line.

## Files changed

| File | Change |
|------|--------|
| `backend/config/settings.py` | +`Annotated`/`NoDecode`/`field_validator` imports; `paper_markets` → `Annotated[list[str], NoDecode]` + `_parse_paper_markets` validator (JSON/bracket-mangled/comma/list). |
| `backend/tests/test_phase_54_1_paper_markets_parse.py` | NEW — 11 tests: every input form, DO-NO-HARM JSON byte-identity, exact bash-mangled repro, default_factory, empty-string. |
| `handoff/current/live_check_54.1.md` | The cross-layer cron-health table (the masterplan `live_check` artifact); autoresearch/ablation flipped ESCALATE → FIX APPLIED + verification. |
| `handoff/current/{research_brief,contract,experiment_results}.md` | Cycle artifacts. |

## Verification output (verbatim)

### Reproduce the exact bug + confirm the fix at the crash layer
```
REPR: '[US,EU,KR]'                      # bash-mangled value my parser must accept
.env line 78: PAPER_MARKETS=["US","EU","KR"]
pydantic_settings 2.13.1 ; NoDecode: available
# BEFORE: pydantic_settings.exceptions.SettingsError: error parsing value for field "paper_markets" from source "EnvSettingsSource"
# AFTER (set -a; . backend/.env; set +a; get_settings()):
get_settings().paper_markets = ['US', 'EU', 'KR']
```

### Tests
```
python -m pytest backend/tests/test_phase_54_1_paper_markets_parse.py -q
11 passed in 0.09s

# regression (no existing settings/config test broke):
python -m pytest backend/tests/ -q -k "settings or config"
25 passed, 694 deselected, 1 warning in 2.86s
```

### Syntax
```
python -c "import ast; ast.parse(open('backend/config/settings.py').read())"  -> settings.py parses
```

## Acceptance-criteria mapping (phase-54.1)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Every launchd + every APScheduler job enumerated w/ state/last-exit/trigger/next-fire | PASS — full table in live_check_54.1.md (7 launchd + 13 APScheduler) |
| 2 | Every unhealthy job listed w/ root-cause + FIX-applied OR escalation (op-gated fixes escalated) | PASS — autoresearch/ablation root-caused + FIXED (settings.py, non-op-gated); huggingface/LLM residue escalated; mas-harness false-positive documented |
| 3 | autoresearch + ablation + mas-harness each addressed | PASS — both fixed at the crash layer (re-verify next nightly fire); mas-harness = false positive |
| 4 | live_check_54.1.md has the full cross-layer table | PASS |

## DO-NO-HARM / scope honesty

- The live engine's `paper_markets` value is byte-identical (`['US','EU','KR']` via JSON;
  `['US']` default) — proven by `test_live_json_path_is_byte_identical` + the
  default_factory test. The money-path APScheduler job (`paper_trading_daily`) is healthy
  and untouched.
- NOT run: the full autoresearch/ablation jobs (autoresearch huggingface gap + potential
  LLM spend = operator-gated). Fix verified ONLY at `get_settings()`, the crash point.
- No `.env` write (tool-blocked + unnecessary). No new dependency. No launchd load/unload.
- The slack_bot-supervisor / dead-man's-switch / digest-cron-health-line gaps are
  deferred to **54.2** (the Slack-lifeline step) — explicitly scoped there, not silently
  dropped.
